#!/usr/bin/env python3
# Reconstruction test for ENCA: generate one or more observations from chosen
# parameters, reconstruct them with the trained autoencoder, and plot means
# with 10-90% bands. In Fourier mode, the observation is the Hann-windowed
# log FFT amplitude vector used as encoder input during training.

import os
import glob
import time
import argparse
import datetime
import json
import numpy as np
from typing import Optional

# IMPORTANT: init_julia() must happen before importing tensorflow
from julia_bootstrap import init_julia
init_julia()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

import src.generators


PARAM_NAMES = ["tau", "T", "Nd", "sigma", "Bmax"]


def _as_tuple(x):
    return tuple(x) if isinstance(x, list) else x


def load_hparams(run_dir: str) -> dict:
    hp_path = os.path.join(run_dir, "hyper_parameters.json")
    if not os.path.isfile(hp_path):
        raise FileNotFoundError(f"Missing hyper_parameters.json in: {run_dir}")
    with open(hp_path, "r") as f:
        return json.load(f)


def find_latest_checkpoint(logdir: str, ckpt_prefix: str) -> str:
    pattern = os.path.join(logdir, f"{ckpt_prefix}-*.index")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching {pattern}")

    def step_of(fn):
        return int(os.path.basename(fn).replace(".index", "").split("-")[-1])

    best = max(candidates, key=step_of)
    return best.replace(".index", "")


def build_encoder_decoder(
    len_timeseries: int,
    ndims_latent: int,
    num_noise_channels: int,
    representation_mode: str = "time",
    num_fft_components: Optional[int] = None,
):
    if representation_mode == "fourier_amplitude":
        if num_fft_components is None:
            raise ValueError("num_fft_components is required for fourier_amplitude mode")

        x_input = tf.keras.layers.Input(shape=[num_fft_components], name="fft_amplitudes")
        x = tf.keras.layers.Dense(256, activation="relu", name="enc_dense_1")(x_input)
        x = tf.keras.layers.Dense(256, activation="relu", name="enc_dense_2")(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="enc_dense_3")(x)
        z = tf.keras.layers.Dense(ndims_latent, activation=None, name="latent_space")(x)
        encoder = tf.keras.Model(inputs=x_input, outputs=z)

        latent_mappings = tf.keras.layers.Input(shape=[ndims_latent], name="latent_representations")
        y = tf.keras.layers.Dense(128, activation="relu", name="dec_dense_1")(latent_mappings)
        y = tf.keras.layers.Dense(256, activation="relu", name="dec_dense_2")(y)
        y = tf.keras.layers.Dense(256, activation="relu", name="dec_dense_3")(y)
        y = tf.keras.layers.Dense(num_fft_components, activation=None, name="pred_fft_amplitudes")(y)
        decoder = tf.keras.Model(inputs=latent_mappings, outputs=y)
        return encoder, decoder

    if representation_mode != "time":
        raise ValueError(f"Unknown representation_mode: {representation_mode}")

    conv_fn = lambda filters, act=None, name=None: tf.keras.layers.Conv1D(
        filters=filters, kernel_size=3, activation=act, name=name
    )

    x_input = tf.keras.layers.Input(shape=[len_timeseries, 1], name="x_observation")
    x = x_input

    num_conv_filters = [[16, 16], [32, 32]]
    for i in range(len(num_conv_filters)):
        if i != 0:
            x = tf.keras.layers.MaxPool1D(pool_size=2, name=f"maxpool{i+1}")(x)
        for j, nf in enumerate(num_conv_filters[i]):
            x = conv_fn(filters=nf, act="relu", name=f"conv{i+1}_{j+1}")(x)

    x = conv_fn(filters=ndims_latent, act=None, name="final_conv")(x)
    z = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    encoder = tf.keras.Model(inputs=x_input, outputs=z)

    latent_mappings = tf.keras.layers.Input(shape=[ndims_latent], name="latent_representations")
    noise_vectors = tf.keras.layers.Input(shape=[len_timeseries, num_noise_channels], name="noise_vectors")

    tile_layer = tf.keras.layers.Lambda(
        lambda a: tf.tile(tf.expand_dims(a, axis=1), multiples=[1, len_timeseries, 1]),
        name="tile_latent_space"
    )
    concat = tf.keras.layers.Concatenate(axis=-1, name="concatenate_noise_and_latent_dims")(
        [tile_layer(latent_mappings), noise_vectors]
    )

    num_units = 16
    y = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=num_units, return_sequences=True, dtype=tf.float32, name="lstm_cell_1"),
        name="Bi-cell-1"
    )(concat)
    y = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=num_units, return_sequences=True, dtype=tf.float32, name="lstm_cell_2"),
        name="Bi-cell-2"
    )(y)
    y = tf.keras.layers.Dense(units=1, activation=None, name="pred")(y)
    y = tf.keras.layers.Reshape([len_timeseries, 1], name="output_shape")(y)

    decoder = tf.keras.Model(inputs=(latent_mappings, noise_vectors), outputs=y)
    return encoder, decoder


def timeseries_to_fft_log_amplitudes_np(
    x: np.ndarray,
    num_fft_components: int,
    fft_log_eps: float,
) -> np.ndarray:
    """Map [N, time, 1] signals to log normalized FFT amplitudes [N, n_fft]."""
    x_1d = np.squeeze(x, axis=-1).astype(np.float32)
    n_time = x_1d.shape[1]
    window = np.hanning(n_time).astype(np.float32)
    amplitudes = np.abs(np.fft.fft(x_1d * window[None, :], axis=1)) / float(n_time)
    return np.log(amplitudes[:, :num_fft_components] + fft_log_eps).astype(np.float32)


def validate_params(theta, hp):
    tau, T, Nd, sigma, Bmax = theta
    lims = [
        ("tau", tau, _as_tuple(hp["tau_lims"])),
        ("T", T, _as_tuple(hp["T_lims"])),
        ("Nd", Nd, _as_tuple(hp["Nd_lims"])),
        ("sigma", sigma, _as_tuple(hp["sigma_lims"])),
        ("Bmax", Bmax, _as_tuple(hp["Bmax_lims"])),
    ]
    for name, value, (lo, hi) in lims:
        if not (lo <= value <= hi):
            raise ValueError(f"{name}={value} is outside saved prior [{lo}, {hi}]")

    dt = float(hp["dt"])
    steps = round(T / dt)
    if abs(T - steps * dt) > 1e-9:
        raise ValueError(f"T={T} must be a multiple of dt={dt}")


def relative_chisq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(((y_true - y_pred) ** 2) / np.maximum(y_true ** 2, 1e-6)))


def _fmt_tag_value(value: float, scale: int = 10) -> str:
    if float(value).is_integer():
        return str(int(round(value)))
    scaled = int(round(value * scale))
    return str(scaled)


def _fmt_sigma_tag(value: float) -> str:
    scaled = int(round(value * 100))
    return f"{scaled:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="Run dir containing checkpoints + hyper_parameters.json")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--Nd", type=float, required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--Bmax", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1234, help="Seed for the sampled driving noise")
    ap.add_argument("--nseeds", type=int, default=1, help="How many noise realizations to aggregate")
    ap.add_argument("--outdir", default=None, help="Where to save plots (default: <run>/diagnostics)")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--best", action="store_true", help="Use model_best_ckpt-* (default)")
    g.add_argument("--last", action="store_true", help="Use model_ckpt-*")

    args = ap.parse_args()

    run_dir = os.path.abspath(args.logdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"--logdir not found: {run_dir}")

    ckpt_prefix = "model_ckpt" if args.last else "model_best_ckpt"
    hp = load_hparams(run_dir)

    theta = (args.tau, args.T, args.Nd, args.sigma, args.Bmax)
    validate_params(theta, hp)
    if args.nseeds < 1:
        raise ValueError("--nseeds must be >= 1")

    Tobs = int(hp["Tobs"])
    Twarmup = int(hp["Twarmup"])
    dt = float(hp["dt"])
    saveat = float(hp["saveat"])
    ndims_latent = int(hp["ndims_latent"])
    num_noise_channels = int(hp["num_noise_channels"])
    representation_mode = hp.get("representation_mode", "time")
    num_fft_components = int(hp.get("num_fft_components", 100))
    fft_log_eps = float(hp.get("fft_log_eps", 1e-8))
    len_timeseries = int(round(Tobs / saveat))
    observation_length = num_fft_components if representation_mode == "fourier_amplitude" else len_timeseries

    outdir = args.outdir or os.path.join(run_dir, "diagnostics")
    os.makedirs(outdir, exist_ok=True)

    encoder, decoder = build_encoder_decoder(
        len_timeseries=len_timeseries,
        ndims_latent=ndims_latent,
        num_noise_channels=num_noise_channels,
        representation_mode=representation_mode,
        num_fft_components=num_fft_components,
    )
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    ckpt_path = find_latest_checkpoint(run_dir, ckpt_prefix=ckpt_prefix)

    for attempt in (1, 2):
        try:
            ckpt.restore(ckpt_path).expect_partial()
            break
        except Exception as e:
            if attempt == 1:
                time.sleep(0.5)
            else:
                raise RuntimeError(f"Failed to restore checkpoint {ckpt_path}: {e}")

    x_true_all = np.zeros((args.nseeds, observation_length), dtype=np.float32)
    x_pred_all = np.zeros((args.nseeds, observation_length), dtype=np.float32)
    params_true = None

    for i in range(args.nseeds):
        prng = np.random.RandomState(args.seed + i)
        gen = src.generators.DataGenerator_SolarDynamo_SDDE_ENCA(
            prng=prng,
            Tobs=Tobs,
            saveat=saveat,
            num_noise_channels=num_noise_channels,
            Twarmup=Twarmup,
            dt=dt,
            tau_lims=(args.tau, args.tau),
            T_lims=(args.T, args.T),
            Nd_lims=(args.Nd, args.Nd),
            sigma_lims=(args.sigma, args.sigma),
            Bmax_lims=(args.Bmax, args.Bmax),
        )

        x_true, params_i, noise = next(iter(gen))
        if params_true is None:
            params_true = params_i

        x_raw_batch = x_true[np.newaxis, ...].astype(np.float32)
        if representation_mode == "fourier_amplitude":
            x_batch = timeseries_to_fft_log_amplitudes_np(
                x_raw_batch,
                num_fft_components=num_fft_components,
                fft_log_eps=fft_log_eps,
            )
        elif representation_mode == "time":
            x_batch = x_raw_batch
        else:
            raise ValueError(f"Unknown representation_mode: {representation_mode}")

        z_latent = encoder(x_batch, training=False).numpy()
        if representation_mode == "fourier_amplitude":
            x_pred = decoder(z_latent, training=False).numpy()[0]
            x_true_vec = x_batch[0]
        else:
            noise_batch = noise[np.newaxis, ...].astype(np.float32)
            x_pred = decoder((z_latent, noise_batch), training=False).numpy()[0, :, 0]
            x_true_vec = x_true[:, 0]

        x_true_all[i, :] = x_true_vec
        x_pred_all[i, :] = x_pred

    rmse_per_seed = np.sqrt(np.mean((x_pred_all - x_true_all) ** 2, axis=1))
    chi_per_seed = np.array([relative_chisq(x_true_all[i], x_pred_all[i]) for i in range(args.nseeds)])
    rmse = float(np.mean(rmse_per_seed))
    chi = float(np.mean(chi_per_seed))
    flat_mean_pred = float(np.mean(x_true_all))
    flat_mean_rmse_per_seed = np.sqrt(np.mean((x_true_all - flat_mean_pred) ** 2, axis=1))
    flat_mean_rmse = float(np.mean(flat_mean_rmse_per_seed))
    performance_vs_flat_mean = float("nan") if flat_mean_rmse == 0.0 else 1.0 - rmse / flat_mean_rmse
    step = int(os.path.basename(ckpt_path).split("-")[-1])
    if representation_mode == "fourier_amplitude":
        x_axis = np.arange(observation_length)
        x_label = "FFT component"
        y_label = "log normalized amplitude"
        true_label = "input spectrum"
        pred_label = "reconstructed spectrum"
    else:
        x_axis = np.arange(len_timeseries) * saveat
        x_label = "time"
        y_label = "signal"
        true_label = "true"
        pred_label = "reconstructed"

    x_true_mean = np.mean(x_true_all, axis=0)
    x_true_lo = np.percentile(x_true_all, 10, axis=0)
    x_true_hi = np.percentile(x_true_all, 90, axis=0)
    x_pred_mean = np.mean(x_pred_all, axis=0)
    x_pred_lo = np.percentile(x_pred_all, 10, axis=0)
    x_pred_hi = np.percentile(x_pred_all, 90, axis=0)

    fig = plt.figure(figsize=(12, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.8, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    text_ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(x_axis, x_true_lo, x_true_hi, alpha=0.20, label=f"{true_label} 10-90%")
    ax.fill_between(x_axis, x_pred_lo, x_pred_hi, alpha=0.20, label=f"{pred_label} 10-90%")
    ax.plot(x_axis, x_true_mean, label=f"{true_label} mean", linewidth=1.8)
    ax.plot(x_axis, x_pred_mean, label=f"{pred_label} mean", linewidth=1.8, alpha=0.95)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    ax.set_title(f"Reconstruction test @ step {step} ({ckpt_prefix}) | representation={representation_mode}")

    summary = (
        f"tau={params_true[0]:.4g}, T={params_true[1]:.4g}, Nd={params_true[2]:.4g}, "
        f"sigma={params_true[3]:.4g}, Bmax={params_true[4]:.4g}\n"
        f"mean RMSE={rmse:.4g}, flat baseline RMSE={flat_mean_rmse:.4g}, "
        f"performance={performance_vs_flat_mean:.4g}, mean relative_chisq={chi:.4g}, "
        f"seeds={args.seed}..{args.seed + args.nseeds - 1} (n={args.nseeds})"
    )
    text_ax.axis("off")
    text_ax.text(
        0.5,
        0.95,
        summary,
        ha="center",
        va="top",
        family="monospace",
        fontsize=11,
        transform=text_ax.transAxes,
    )

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_tag = (
        f"tau{_fmt_tag_value(args.tau)}_"
        f"T{_fmt_tag_value(args.T)}_"
        f"Nd{_fmt_tag_value(args.Nd)}_"
        f"sig{_fmt_sigma_tag(args.sigma)}_"
        f"B{_fmt_tag_value(args.Bmax)}"
    )
    out_png = os.path.join(outdir, f"recon_{ckpt_prefix}_step{step}_{param_tag}_{stamp}.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"[OK] Restored: {ckpt_path}")
    print(f"[OK] Saved plot: {out_png}")
    print("Parameters:")
    print(
        f"  tau={params_true[0]:.6g}  T={params_true[1]:.6g}  Nd={params_true[2]:.6g}  "
        f"sigma={params_true[3]:.6g}  Bmax={params_true[4]:.6g}"
    )
    flat_mean_name = "flat_mean_spectrum" if representation_mode == "fourier_amplitude" else "flat_mean_signal"
    print(
        f"Metrics: mean_RMSE={rmse:.6g}  mean_relative_chisq={chi:.6g}  "
        f"{flat_mean_name}_baseline_RMSE={flat_mean_rmse:.6g}  "
        f"performance_vs_{flat_mean_name}_baseline={performance_vs_flat_mean:.6g}  "
        f"nseeds={args.nseeds}"
    )


if __name__ == "__main__":
    main()
