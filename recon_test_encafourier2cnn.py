#!/usr/bin/env python3
"""Reconstruction performance test for original and Jupiter ENCAfft2CNN runs."""

import argparse
import datetime
import glob
import json
import os
import time

import numpy as np

# The canonical simulator must initialize Julia before TensorFlow is imported.
from sdde_model import init_julia

init_julia()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import src.generators
from types import SimpleNamespace
from train_ENCAfft2CNN_model3 import (
    Architecture, timeseries_to_fourier_log_amplitude, validate_window,
    validate_fft2_checkpoint,
)


def build_enca_fourier_cnn_encoder_decoder(**kwargs):
    model = Architecture(**kwargs)
    return model.encoder, model.decoder


def timeseries_to_fourier_log_amplitude_np(x, num_fft_components, window):
    return timeseries_to_fourier_log_amplitude(x, num_fft_components, window).numpy()



def _as_tuple(value):
    return tuple(value) if isinstance(value, list) else value


def load_hparams(run_dir):
    path = os.path.join(run_dir, "hyper_parameters.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing hyper_parameters.json in: {run_dir}")
    with open(path, "r") as stream:
        hp = json.load(stream)
    validate_fft2_checkpoint(SimpleNamespace(**hp))
    if hp.get("infer_phase", False):
        raise ValueError("These FFT2 diagnostics currently support original and phase-marginalized Jupiter runs only.")
    if hp.get("simulation_backend") != src.generators.DataGenerator_SolarDynamo_SDDE_Canonical.simulation_backend:
        raise ValueError("Checkpoint does not use the supported canonical SDDE backend.")
    return hp


def find_latest_checkpoint(logdir, ckpt_prefix):
    pattern = os.path.join(logdir, f"{ckpt_prefix}-*.index")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching {pattern}")
    latest = max(
        candidates,
        key=lambda path: int(os.path.basename(path).replace(".index", "").split("-")[-1]),
    )
    return latest.replace(".index", "")


def validate_params(theta, hp, model, require_discrete_delay):
    names = ["tau", "T", "Nd", "sigma", "Bmax"]
    limits = [
        _as_tuple(hp["tau_lims"]),
        _as_tuple(hp["T_lims"]),
        _as_tuple(hp["Nd_lims"]),
        _as_tuple(hp["sigma_lims"]),
        _as_tuple(hp["Bmax_lims"]),
    ]
    if model == "jupiter":
        names.append("Aj")
        limits.append(_as_tuple(hp.get("Aj_lims", (0.0, 0.1))))
    for name, value, (lo, hi) in zip(names, theta, limits):
        if not lo <= value <= hi:
            raise ValueError(f"{name}={value} is outside saved prior [{lo}, {hi}]")
    if require_discrete_delay:
        dt = float(hp["dt"])
        steps = round(theta[1] / dt)
        if abs(theta[1] - steps * dt) > 1e-9:
            raise ValueError(f"T={theta[1]} must be a multiple of dt={dt}")


def relative_chisq(y_true, y_pred):
    return float(np.mean(((y_true - y_pred) ** 2) / np.maximum(y_true**2, 1e-6)))


def normalized_mse(y_true, y_pred, eps):
    scale = np.sqrt(np.mean(y_true**2, axis=1, keepdims=True))
    return float(np.mean(((y_true - y_pred) / np.maximum(scale, eps)) ** 2))


def _fmt_tag_value(value, scale=10):
    if float(value).is_integer():
        return str(int(round(value)))
    return str(int(round(value * scale)))


def _fmt_sigma_tag(value):
    return f"{int(round(value * 100)):03d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--T", type=float, required=True)
    parser.add_argument("--Nd", type=float, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--Bmax", type=float, required=True)
    parser.add_argument("--Aj", type=float, default=None, help="Jupiter modulation amplitude")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nseeds", type=int, default=1)
    parser.add_argument("--outdir", default=None)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--best", action="store_true")
    checkpoint_group.add_argument("--last", action="store_true")
    args = parser.parse_args()

    if args.nseeds < 1:
        raise ValueError("--nseeds must be >= 1")
    run_dir = os.path.abspath(args.logdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"--logdir not found: {run_dir}")
    hp = load_hparams(run_dir)
    representation_mode = hp.get("representation_mode", "enca_fft2_cnn")
    if representation_mode != "enca_fft2_cnn":
        raise ValueError(
            "recon_test_encafourier2cnn.py expects "
            f"representation_mode='enca_fft2_cnn', got {representation_mode!r}."
        )
    model = str(hp.get("model", "original")).strip().lower()
    if model not in {"original", "jupiter"}:
        raise ValueError(f"Unknown saved SDDE model {model!r}.")
    if model == "jupiter" and args.Aj is None:
        raise ValueError("A Jupiter run requires --Aj.")
    if model == "original" and args.Aj is not None:
        raise ValueError("--Aj may only be used with a Jupiter run.")

    theta = (args.tau, args.T, args.Nd, args.sigma, args.Bmax)
    if model == "jupiter":
        theta += (args.Aj,)
    use_canonical_generator = model == "jupiter" or bool(hp.get("simulation_backend"))
    validate_params(theta, hp, model, require_discrete_delay=not use_canonical_generator)

    Tobs = int(hp["Tobs"])
    Twarmup = int(hp["Twarmup"])
    dt = float(hp["dt"])
    saveat = float(hp["saveat"])
    len_timeseries = int(round(Tobs / saveat))
    ndims_latent = int(hp["ndims_latent"])
    num_noise_channels = int(hp.get("num_noise_channels", 1))
    num_fft_components = int(hp.get("num_fft_components", 100))
    window = validate_window(hp.get("window", ""))

    encoder, decoder = build_enca_fourier_cnn_encoder_decoder(
        ndims_latent=ndims_latent,
        len_timeseries=len_timeseries,
        num_noise_channels=num_noise_channels,
        num_fft_components=num_fft_components,
    )
    ckpt_prefix = "model_ckpt" if args.last else "model_best_ckpt"
    ckpt_path = find_latest_checkpoint(run_dir, ckpt_prefix)
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    for attempt in (1, 2):
        try:
            checkpoint.restore(ckpt_path).assert_existing_objects_matched().expect_partial()
            break
        except Exception as exc:
            if attempt == 1:
                time.sleep(0.5)
            else:
                raise RuntimeError(f"Failed to restore checkpoint {ckpt_path}: {exc}") from exc

    true_all = np.zeros((args.nseeds, num_fft_components), dtype=np.float32)
    pred_all = np.zeros((args.nseeds, num_fft_components), dtype=np.float32)
    params_true = None
    generator_class = (
        src.generators.DataGenerator_SolarDynamo_SDDE_Canonical
        if use_canonical_generator
        else src.generators.DataGenerator_SolarDynamo_SDDE_ENCA
    )
    for index in range(args.nseeds):
        generator = generator_class(
            prng=np.random.RandomState(args.seed + index),
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
            Aj_lims=(args.Aj, args.Aj) if model == "jupiter" else (0.0, 0.1),
            phi_lims=tuple(hp.get("phi_lims", (0.0, 2.0 * np.pi))),
            model=model,
            jupiter_period=float(hp.get("jupiter_period", 11.86)),
        )
        observation, parameters, noise = next(iter(generator))
        if params_true is None:
            params_true = parameters
        spectrum = timeseries_to_fourier_log_amplitude_np(
            observation[None, ...], num_fft_components=num_fft_components, window=window
        )
        latent = encoder(spectrum, training=False).numpy()
        prediction = decoder((latent, noise[None, ...]), training=False).numpy()
        true_all[index] = spectrum[0, :, 0]
        pred_all[index] = prediction[0, :, 0]

    rmse_per_seed = np.sqrt(np.mean((pred_all - true_all) ** 2, axis=1))
    rmse = float(np.mean(rmse_per_seed))
    chi = float(
        np.mean([relative_chisq(true_all[i], pred_all[i]) for i in range(args.nseeds)])
    )
    norm_mse = normalized_mse(
        true_all, pred_all, eps=float(hp.get("recon_scale_eps", 1e-3))
    )
    flat_mean = float(np.mean(true_all))
    baseline_rmse = float(
        np.mean(np.sqrt(np.mean((true_all - flat_mean) ** 2, axis=1)))
    )
    performance = float("nan") if baseline_rmse == 0.0 else 1.0 - rmse / baseline_rmse

    x_axis = np.arange(num_fft_components)
    true_mean, pred_mean = np.mean(true_all, axis=0), np.mean(pred_all, axis=0)
    true_lo, true_hi = np.percentile(true_all, [10, 90], axis=0)
    pred_lo, pred_hi = np.percentile(pred_all, [10, 90], axis=0)
    figure = plt.figure(figsize=(12, 6.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[4.8, 1.0])
    axis = figure.add_subplot(grid[0, 0])
    text_axis = figure.add_subplot(grid[1, 0])
    axis.fill_between(x_axis, true_lo, true_hi, alpha=0.20, label="input 10-90%")
    axis.fill_between(x_axis, pred_lo, pred_hi, alpha=0.20, label="reconstruction 10-90%")
    axis.plot(x_axis, true_mean, linewidth=1.8, label="input mean")
    axis.plot(x_axis, pred_mean, linewidth=1.8, label="reconstruction mean")
    axis.set(xlabel="FFT component", ylabel="log1p rFFT amplitude")
    axis.legend(loc="best")
    step = int(os.path.basename(ckpt_path).split("-")[-1])
    axis.set_title(
        f"ENCAfft2CNN {model} reconstruction test @ step {step} ({ckpt_prefix})"
    )
    parameter_summary = (
        f"tau={params_true[0]:.4g}, T={params_true[1]:.4g}, Nd={params_true[2]:.4g}, "
        f"sigma={params_true[3]:.4g}, Bmax={params_true[4]:.4g}"
        + (f", Aj={params_true[5]:.4g}" if model == "jupiter" else "")
    )
    text_axis.axis("off")
    text_axis.text(
        0.5,
        0.95,
        parameter_summary
        + "\n"
        + f"mean RMSE={rmse:.4g}, normMSE={norm_mse:.4g}, "
        + f"flat baseline RMSE={baseline_rmse:.4g}, performance={performance:.4g}, "
        + f"relative_chisq={chi:.4g}, seeds={args.seed}..{args.seed + args.nseeds - 1} "
        + f"(n={args.nseeds}), window={window or 'none'}",
        ha="center",
        va="top",
        family="monospace",
        fontsize=10,
        transform=text_axis.transAxes,
    )

    tag = (
        f"tau{_fmt_tag_value(args.tau)}_T{_fmt_tag_value(args.T)}_"
        f"Nd{_fmt_tag_value(args.Nd)}_sig{_fmt_sigma_tag(args.sigma)}_"
        f"B{_fmt_tag_value(args.Bmax)}"
        + (f"_Aj{_fmt_tag_value(args.Aj, scale=100)}" if model == "jupiter" else "")
    )
    outdir = args.outdir or os.path.join(run_dir, "diagnostics")
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(
        outdir,
        f"recon_encafourier2cnn_{ckpt_prefix}_step{step}_{tag}_{stamp}.png",
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)

    np.savez_compressed(output[:-4] + ".npz", target=true_all, reconstruction=pred_all)
    with open(output[:-4] + ".json", "w") as stream:
        json.dump(dict(checkpoint=ckpt_path, seed=args.seed, nseeds=args.nseeds,
                       parameters=list(map(float, params_true)), mean_rmse=rmse,
                       normalized_mse=norm_mse), stream, indent=2)

    print(f"[OK] Restored: {ckpt_path}")
    print(f"[OK] Saved plot: {output}")
    print(f"Parameters: {parameter_summary}")
    print(
        f"Metrics: mean_RMSE={rmse:.6g}  normMSE={norm_mse:.6g}  "
        f"mean_relative_chisq={chi:.6g}  flat_mean_spectrum_baseline_RMSE={baseline_rmse:.6g}  "
        f"performance_vs_flat_mean_spectrum_baseline={performance:.6g}  "
        f"nseeds={args.nseeds}"
    )


if __name__ == "__main__":
    main()
