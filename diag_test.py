#!/usr/bin/env python3
# Diagonal test for ENCA: predicted params (encoder z[0:5]) vs true params
# Saves a 5-panel scatter plot to disk (works on cluster/headless via Agg backend).

"""
How to run this test:

# from inside sdde_ENCA_runs/
python ../diag_test.py --logdir 20260305_142921_test

# explicitly pick "best" (default anyway)
python ../diag_test.py --logdir 20260305_142921_test --best

# pick "last"
python ../diag_test.py --logdir 20260305_142921_test --last
"""

import os
import glob
import time
import argparse
import datetime
import json
import numpy as np

# IMPORTANT: init_julia() must happen before importing tensorflow
from julia_bootstrap import init_julia
init_julia()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

import src.generators

# ---------------------------
# Helpers
# ---------------------------
PARAM_NAMES = ["tau", "T", "Nd", "sigma", "Bmax"]


def _as_tuple(x):
    # hyper_parameters.json stores tuples as lists
    return tuple(x) if isinstance(x, list) else x


def load_hparams(run_dir: str) -> dict:
    hp_path = os.path.join(run_dir, "hyper_parameters.json")
    if not os.path.isfile(hp_path):
        raise FileNotFoundError(f"Missing hyper_parameters.json in: {run_dir}")
    with open(hp_path, "r") as f:
        hp = json.load(f)
    return hp


def find_latest_checkpoint(logdir: str, ckpt_prefix: str) -> str:
    """
    Finds latest checkpoint by parsing filenames like:
      <logdir>/<ckpt_prefix>-1234.index
    returns: path without extension, e.g. <logdir>/<ckpt_prefix>-1234
    """
    pattern = os.path.join(logdir, f"{ckpt_prefix}-*.index")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching {pattern}")

    def step_of(fn):
        base = os.path.basename(fn)            # e.g. model_ckpt-32100.index
        mid = base.replace(".index", "")
        return int(mid.split("-")[-1])

    best = max(candidates, key=step_of)
    return best.replace(".index", "")


def build_encoder_decoder(len_timeseries: int, ndims_latent: int, num_noise_channels: int):
    """
    Rebuilds the SAME architecture as training script (encoder+decoder).
    Only encoder is used for diagonal test, but we include decoder in checkpoint
    to match saved objects.
    """

    # ---- Encoder ----
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

    # ---- Decoder (needed for ckpt object structure, even if we don't use it) ----
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prior_lims):
    """
    Simple per-parameter diagnostics: RMSE, corr, normalized RMSE to prior width.
    y_true/y_pred: shape (N, 5)
    """
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


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="Run dir containing checkpoints + hyper_parameters.json")
    ap.add_argument("--nsamples", type=int, default=1000, help="How many synthetic samples to test")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for encoder forward pass")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", default=None, help="Where to save plots (default: <run>/diagnostics)")

    # choose checkpoint set
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--best", action="store_true", help="Use model_best_ckpt-* (default)")
    g.add_argument("--last", action="store_true", help="Use model_ckpt-*")

    args = ap.parse_args()

    run_dir = os.path.abspath(args.logdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"--logdir not found: {run_dir}")

    # Default = best
    ckpt_prefix = "model_ckpt" if args.last else "model_best_ckpt"

    prng = np.random.RandomState(args.seed)

    # -------------------------
    # Read hyperparameters saved by training
    # -------------------------
    hp = load_hparams(run_dir)

    Twarmup = int(hp.get("Twarmup", 200))
    Tobs    = int(hp.get("Tobs", 929))
    dt      = float(hp.get("dt", 0.1))
    saveat  = float(hp.get("saveat", 1.0))

    ndims_latent       = int(hp.get("ndims_latent", 20))
    num_noise_channels = int(hp.get("num_noise_channels", 1))

    tau_lims   = _as_tuple(hp.get("tau_lims",   (0.1, 10.0)))
    T_lims     = _as_tuple(hp.get("T_lims",     (0.1, 10.0)))
    Nd_lims    = _as_tuple(hp.get("Nd_lims",    (1.0, 15.0)))
    sigma_lims = _as_tuple(hp.get("sigma_lims", (0.01, 0.3)))
    Bmax_lims  = _as_tuple(hp.get("Bmax_lims",  (1.0, 15.0)))

    len_timeseries = int(round(Tobs / saveat))

    outdir = args.outdir or os.path.join(run_dir, "diagnostics")
    os.makedirs(outdir, exist_ok=True)

    # -------------------------
    # Build model objects (must match checkpoint shapes!)
    # -------------------------
    encoder, decoder = build_encoder_decoder(
        len_timeseries=len_timeseries,
        ndims_latent=ndims_latent,
        num_noise_channels=num_noise_channels
    )

    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder)

    ckpt_path = find_latest_checkpoint(run_dir, ckpt_prefix=ckpt_prefix)

    # Restore (retry once if it races with a write)
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

    # -------------------------
    # Build generator (must match training)
    # -------------------------
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

    # Collect samples
    X = np.zeros((args.nsamples, len_timeseries, 1), dtype=np.float32)
    Ptrue = np.zeros((args.nsamples, 5), dtype=np.float32)

    for i in range(args.nsamples):
        x0, p0, _ = next(it)
        X[i] = x0
        Ptrue[i] = p0

    # Forward pass in batches
    Z = np.zeros((args.nsamples, ndims_latent), dtype=np.float32)
    for i0 in range(0, args.nsamples, args.batch):
        i1 = min(args.nsamples, i0 + args.batch)
        Z[i0:i1] = encoder(X[i0:i1], training=False).numpy()

    # Core diagonal-test assumption:
    Ppred = Z[:, :5]

    prior_lims = [tau_lims, T_lims, Nd_lims, sigma_lims, Bmax_lims]
    metrics = compute_metrics(Ptrue, Ppred, prior_lims)

    # Plot
    fig = plt.figure(figsize=(18, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[4.8, 1.1])
    axes = [fig.add_subplot(gs[0, j]) for j in range(5)]
    text_axes = [fig.add_subplot(gs[1, j]) for j in range(5)]

    for j, name in enumerate(PARAM_NAMES):
        ax = axes[j]
        text_ax = text_axes[j]
        t = Ptrue[:, j]
        p = Ppred[:, j]

        ax.scatter(t, p, s=8, alpha=0.6)
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
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
    out_png = os.path.join(outdir, f"diag_{ckpt_prefix}_step{step}_{stamp}.png")
    fig.suptitle(f"Diagonal test @ step {step} ({ckpt_prefix}) | nsamples={args.nsamples} | ndims_latent={ndims_latent}")
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
    
