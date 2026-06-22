#!/usr/bin/env python3
# Diagonal test for time-domain FNO ENCA: predicted params (encoder z[0:5])
# vs true params. Saves a 5-panel scatter plot to disk.

import os
import glob
import time
import argparse
import datetime
import json
import numpy as np

# IMPORTANT: init_julia() must happen before importing tensorflow.
from julia_bootstrap import init_julia

init_julia()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import src.generators
from src.fno_architecture import build_fno_encoder_decoder


PARAM_NAMES = ["tau", "T", "Nd", "sigma", "Bmax"]


def _as_tuple(x):
    return tuple(x) if isinstance(x, list) else x


def load_hparams(run_dir):
    hp_path = os.path.join(run_dir, "hyper_parameters.json")
    if not os.path.isfile(hp_path):
        raise FileNotFoundError(f"Missing hyper_parameters.json in: {run_dir}")
    with open(hp_path, "r") as f:
        return json.load(f)


def find_latest_checkpoint(logdir, ckpt_prefix):
    pattern = os.path.join(logdir, f"{ckpt_prefix}-*.index")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching {pattern}")

    def step_of(fn):
        return int(os.path.basename(fn).replace(".index", "").split("-")[-1])

    return max(candidates, key=step_of).replace(".index", "")


def compute_metrics(y_true, y_pred, prior_lims):
    out = {}
    eps = 1e-12
    for i, name in enumerate(PARAM_NAMES):
        t = y_true[:, i]
        p = y_pred[:, i]
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        tc = t - t.mean()
        pc = p - p.mean()
        corr = float((tc @ pc) / (np.sqrt((tc @ tc) * (pc @ pc)) + eps))
        lo, hi = prior_lims[i]
        width = float(hi - lo)
        rmse_pct = float(100.0 * rmse / width) if width > 0 else float("nan")
        out[name] = {"rmse": rmse, "corr": corr, "rmse_pct": rmse_pct}
    return out


def build_model_from_hparams(hp):
    Tobs = int(hp.get("Tobs", 929))
    saveat = float(hp.get("saveat", 1.0))
    len_timeseries = int(round(Tobs / saveat))
    return build_fno_encoder_decoder(
        len_timeseries=len_timeseries,
        ndims_latent=int(hp["ndims_latent"]),
        num_noise_channels=int(hp.get("num_noise_channels", 1)),
        fno_width=int(hp.get("fno_width", 64)),
        fno_modes=int(hp.get("fno_modes", 32)),
        fno_layers=int(hp.get("fno_layers", 4)),
        use_time_coordinate=bool(hp.get("use_time_coordinate", True)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="Run dir containing checkpoints + hyper_parameters.json")
    ap.add_argument("--nsamples", type=int, default=1000, help="How many synthetic samples to test")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for encoder forward pass")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", default=None, help="Where to save plots (default: <run>/diagnostics)")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--best", action="store_true", help="Use model_best_ckpt-* (default)")
    g.add_argument("--last", action="store_true", help="Use model_ckpt-*")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.logdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"--logdir not found: {run_dir}")

    ckpt_prefix = "model_ckpt" if args.last else "model_best_ckpt"
    prng = np.random.RandomState(args.seed)
    hp = load_hparams(run_dir)

    if hp.get("representation_mode", "time") != "time":
        raise ValueError("diag_test_fno.py expects representation_mode='time'.")
    if hp.get("architecture_name") not in (None, "fno_enca"):
        raise ValueError(f"diag_test_fno.py expects architecture_name='fno_enca', got {hp.get('architecture_name')!r}.")

    Twarmup = int(hp.get("Twarmup", 200))
    Tobs = int(hp.get("Tobs", 929))
    dt = float(hp.get("dt", 0.1))
    saveat = float(hp.get("saveat", 1.0))
    len_timeseries = int(round(Tobs / saveat))
    ndims_latent = int(hp["ndims_latent"])
    num_noise_channels = int(hp.get("num_noise_channels", 1))

    tau_lims = _as_tuple(hp.get("tau_lims", (0.1, 10.0)))
    T_lims = _as_tuple(hp.get("T_lims", (0.1, 10.0)))
    Nd_lims = _as_tuple(hp.get("Nd_lims", (1.0, 15.0)))
    sigma_lims = _as_tuple(hp.get("sigma_lims", (0.01, 0.3)))
    Bmax_lims = _as_tuple(hp.get("Bmax_lims", (1.0, 15.0)))

    outdir = args.outdir or os.path.join(run_dir, "diagnostics")
    os.makedirs(outdir, exist_ok=True)

    encoder, decoder = build_model_from_hparams(hp)
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

    step = int(os.path.basename(ckpt_path).split("-")[-1])

    gen = src.generators.DataGenerator_SolarDynamo_SDDE_ENCA(
        prng=prng,
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
    )
    it = iter(gen)

    X = np.zeros((args.nsamples, len_timeseries, 1), dtype=np.float32)
    Ptrue = np.zeros((args.nsamples, 5), dtype=np.float32)
    for i in range(args.nsamples):
        x0, p0, _ = next(it)
        X[i] = x0
        Ptrue[i] = p0

    Z = np.zeros((args.nsamples, ndims_latent), dtype=np.float32)
    for i0 in range(0, args.nsamples, args.batch):
        i1 = min(args.nsamples, i0 + args.batch)
        Z[i0:i1] = encoder(X[i0:i1], training=False).numpy()

    Ppred = Z[:, :5]
    prior_lims = [tau_lims, T_lims, Nd_lims, sigma_lims, Bmax_lims]
    metrics = compute_metrics(Ptrue, Ppred, prior_lims)

    fig = plt.figure(figsize=(18, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[4.8, 1.1])
    axes = [fig.add_subplot(gs[0, j]) for j in range(5)]
    text_axes = [fig.add_subplot(gs[1, j]) for j in range(5)]

    for j, name in enumerate(PARAM_NAMES):
        ax = axes[j]
        text_ax = text_axes[j]
        t = Ptrue[:, j]
        p = Ppred[:, j]
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())

        ax.scatter(t, p, s=8, alpha=0.6)
        ax.plot([lo, hi], [lo, hi], linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("true")
        if j == 0:
            ax.set_ylabel("pred")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        text_ax.axis("off")
        text_ax.text(
            0.5,
            0.9,
            (
                f"corr={metrics[name]['corr']:.4f}\n"
                f"rmse={metrics[name]['rmse']:.4g}\n"
                f"rmse/range={metrics[name]['rmse_pct']:.1f}%"
            ),
            ha="center",
            va="top",
            transform=text_ax.transAxes,
            family="monospace",
            fontsize=10,
        )

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(outdir, f"diag_fno_{ckpt_prefix}_step{step}_{stamp}.png")
    fig.suptitle(
        f"FNO diagonal test @ step {step} ({ckpt_prefix}) | nsamples={args.nsamples} | "
        f"ndims_latent={ndims_latent} | modes={hp.get('fno_modes', 32)}"
    )
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"[OK] Restored: {ckpt_path}")
    print(f"[OK] Saved plot: {out_png}")
    for k in PARAM_NAMES:
        print(
            f"  {k:5s}: rmse={metrics[k]['rmse']:.4g}  "
            f"corr={metrics[k]['corr']:.4f}  "
            f"rmse/range={metrics[k]['rmse_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
