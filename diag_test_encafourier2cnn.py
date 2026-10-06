#!/usr/bin/env python3
"""Diagonal performance test for original and Jupiter ENCAfft2CNN runs."""

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



BASE_PARAM_NAMES = ["tau", "T", "Nd", "sigma", "Bmax"]


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


def compute_metrics(y_true, y_pred, prior_lims, param_names):
    metrics = {}
    for index, name in enumerate(param_names):
        true = y_true[:, index]
        pred = y_pred[:, index]
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        true_centered = true - true.mean()
        pred_centered = pred - pred.mean()
        denominator = np.sqrt(
            (true_centered @ true_centered) * (pred_centered @ pred_centered)
        )
        corr = float((true_centered @ pred_centered) / (denominator + 1e-12))
        lo, hi = prior_lims[index]
        width = float(hi - lo)
        rmse_pct = float(100.0 * rmse / width) if width > 0.0 else float("nan")
        metrics[name] = {"rmse": rmse, "corr": corr, "rmse_pct": rmse_pct}
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--nsamples", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--outdir", default=None)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--best", action="store_true")
    checkpoint_group.add_argument("--last", action="store_true")
    args = parser.parse_args()

    if args.nsamples < 1:
        raise ValueError("--nsamples must be >= 1")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")
    run_dir = os.path.abspath(args.logdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"--logdir not found: {run_dir}")

    hp = load_hparams(run_dir)
    representation_mode = hp.get("representation_mode", "enca_fft2_cnn")
    if representation_mode != "enca_fft2_cnn":
        raise ValueError(
            "diag_test_encafourier2cnn.py expects "
            f"representation_mode='enca_fft2_cnn', got {representation_mode!r}."
        )
    model = str(hp.get("model", "original")).strip().lower()
    if model not in {"original", "jupiter"}:
        raise ValueError(f"Unknown saved SDDE model {model!r}.")
    param_names = BASE_PARAM_NAMES + (["Aj"] if model == "jupiter" else [])
    num_model_parameters = int(hp.get("num_model_parameters", len(param_names)))
    if num_model_parameters != len(param_names):
        raise ValueError(
            f"Saved model={model!r} requires {len(param_names)} regressors, but "
            f"num_model_parameters={num_model_parameters}."
        )

    ndims_latent = int(hp["ndims_latent"])
    if ndims_latent < num_model_parameters:
        raise ValueError(
            f"ndims_latent={ndims_latent} is smaller than the "
            f"{num_model_parameters} supervised parameters."
        )
    Tobs = int(hp["Tobs"])
    Twarmup = int(hp["Twarmup"])
    dt = float(hp["dt"])
    saveat = float(hp["saveat"])
    len_timeseries = int(round(Tobs / saveat))
    num_noise_channels = int(hp.get("num_noise_channels", 1))
    num_fft_components = int(hp.get("num_fft_components", 100))
    window = validate_window(hp.get("window", ""))

    tau_lims = _as_tuple(hp.get("tau_lims", (0.1, 10.0)))
    T_lims = _as_tuple(hp.get("T_lims", (0.1, 10.0)))
    Nd_lims = _as_tuple(hp.get("Nd_lims", (1.0, 15.0)))
    sigma_lims = _as_tuple(hp.get("sigma_lims", (0.01, 0.3)))
    Bmax_lims = _as_tuple(hp.get("Bmax_lims", (1.0, 15.0)))
    Aj_lims = _as_tuple(hp.get("Aj_lims", (0.0, 0.1)))

    encoder, _ = build_enca_fourier_cnn_encoder_decoder(
        ndims_latent=ndims_latent,
        len_timeseries=len_timeseries,
        num_noise_channels=num_noise_channels,
        num_fft_components=num_fft_components,
    )
    ckpt_prefix = "model_ckpt" if args.last else "model_best_ckpt"
    ckpt_path = find_latest_checkpoint(run_dir, ckpt_prefix)
    checkpoint = tf.train.Checkpoint(encoder=encoder)
    for attempt in (1, 2):
        try:
            checkpoint.restore(ckpt_path).assert_existing_objects_matched().expect_partial()
            break
        except Exception as exc:
            if attempt == 1:
                time.sleep(0.5)
            else:
                raise RuntimeError(f"Failed to restore checkpoint {ckpt_path}: {exc}") from exc

    generator_class = (
        src.generators.DataGenerator_SolarDynamo_SDDE_Canonical
        if model == "jupiter" or hp.get("simulation_backend")
        else src.generators.DataGenerator_SolarDynamo_SDDE_ENCA
    )
    generator = generator_class(
        prng=np.random.RandomState(args.seed),
        Tobs=Tobs,
        saveat=saveat,
        num_noise_channels=num_noise_channels,
        Twarmup=Twarmup,
        dt=dt,
        tau_lims=tau_lims,
        T_lims=T_lims,
        Nd_lims=Nd_lims,
        sigma_lims=sigma_lims,
        Bmax_lims=Bmax_lims,
        Aj_lims=Aj_lims,
        phi_lims=tuple(hp.get("phi_lims", (0.0, 2.0 * np.pi))),
        model=model,
        jupiter_period=float(hp.get("jupiter_period", 11.86)),
    )
    iterator = iter(generator)
    raw = np.zeros((args.nsamples, len_timeseries, 1), dtype=np.float32)
    true_params = np.zeros((args.nsamples, num_model_parameters), dtype=np.float32)
    for index in range(args.nsamples):
        observation, parameters, _ = next(iterator)
        raw[index] = observation
        true_params[index] = parameters
    spectra = timeseries_to_fourier_log_amplitude_np(
        raw, num_fft_components=num_fft_components, window=window
    )

    latent = np.zeros((args.nsamples, ndims_latent), dtype=np.float32)
    for start in range(0, args.nsamples, args.batch):
        stop = min(args.nsamples, start + args.batch)
        latent[start:stop] = encoder(spectra[start:stop], training=False).numpy()
    predicted_params = latent[:, :num_model_parameters]
    prior_lims = [tau_lims, T_lims, Nd_lims, sigma_lims, Bmax_lims]
    if model == "jupiter":
        prior_lims.append(Aj_lims)
    metrics = compute_metrics(true_params, predicted_params, prior_lims, param_names)

    n_params = len(param_names)
    figure = plt.figure(figsize=(3.6 * n_params, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(2, n_params, height_ratios=[4.8, 1.1])
    for index, name in enumerate(param_names):
        axis = figure.add_subplot(grid[0, index])
        text_axis = figure.add_subplot(grid[1, index])
        true = true_params[:, index]
        pred = predicted_params[:, index]
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        axis.scatter(true, pred, s=8, alpha=0.6)
        axis.plot([lo, hi], [lo, hi], linewidth=1.0)
        axis.set(xlabel="true", title=name, xlim=(lo, hi), ylim=(lo, hi))
        if index == 0:
            axis.set_ylabel("pred")
        text_axis.axis("off")
        text_axis.text(
            0.5,
            0.9,
            f"corr={metrics[name]['corr']:.4f}\n"
            f"rmse={metrics[name]['rmse']:.4g}\n"
            f"rmse/range={metrics[name]['rmse_pct']:.1f}%",
            ha="center",
            va="top",
            family="monospace",
            fontsize=10,
            transform=text_axis.transAxes,
        )

    step = int(os.path.basename(ckpt_path).split("-")[-1])
    figure.suptitle(
        f"ENCAfft2CNN diagonal test @ step {step} ({ckpt_prefix}) | "
        f"nsamples={args.nsamples} | model={model} | regressors={num_model_parameters} | "
        f"ndims_latent={ndims_latent} | window={window or 'none'}"
    )
    outdir = args.outdir or os.path.join(run_dir, "diagnostics")
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(
        outdir, f"diag_encafourier2cnn_{ckpt_prefix}_step{step}_{stamp}.png"
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)

    np.savez_compressed(output[:-4] + ".npz", true_params=true_params,
                        predicted_params=predicted_params, latent=latent)
    with open(output[:-4] + ".json", "w") as stream:
        json.dump(dict(checkpoint=ckpt_path, seed=args.seed, nsamples=args.nsamples,
                       model=model, metrics=metrics), stream, indent=2)

    print(f"[OK] Restored: {ckpt_path}")
    print(f"[OK] Saved plot: {output}")
    for name in param_names:
        print(
            f"  {name:5s}: rmse={metrics[name]['rmse']:.4g}  "
            f"corr={metrics[name]['corr']:.4f}  "
            f"rmse/range={metrics[name]['rmse_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
