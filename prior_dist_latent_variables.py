#!/usr/bin/env python3
"""Sample the SDDE prior and visualize its encoded MLP latent distribution.

The default plots contain every three-dimensional latent-space combination
(z_i, z_j, z_6), i < j <= 5.  The sampled parameters and latent coordinates
are also saved as a compressed NumPy archive so that plotting experiments do
not require rerunning the SDDE simulator.
"""

import argparse
import glob
import json
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np

# IMPORTANT: Julia must be initialized before TensorFlow is imported.
from julia_bootstrap import init_julia

init_julia()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import src.generators


PARAM_NAMES = ("tau", "T", "Nd", "sigma", "Bmax")
DEFAULT_MODEL = "20260611_mlp_z6_1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw synthetic observations from a saved MLP run's parameter "
            "priors, encode them, and plot all (z_i, z_j, z_6) triplets."
        )
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=(
            "Run name or run-directory path "
            f"(default: {DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--nsamples", type=int, default=1000, help="Number of prior draws (default: 1000)"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument(
        "--batch", type=int, default=128, help="Encoder inference batch size (default: 128)"
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <run>/diagnostics/prior_latent)",
    )
    parser.add_argument(
        "--dpi", type=int, default=180, help="Output image resolution (default: 180)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Also save a self-contained interactive Plotly HTML overview",
    )
    parser.add_argument(
        "--obs-sn-data",
        default=None,
        help=(
            "Full SILSO yearly CSV to encode and mark as obsSN; the SABC "
            "[49:-6] crop must match the model's observation length"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--best", action="store_true", help="Use latest model_best_ckpt-* (default)"
    )
    group.add_argument("--last", action="store_true", help="Use latest model_ckpt-*")
    return parser.parse_args()


def resolve_run_dir(model: str) -> Path:
    """Accept either a run directory or a bare run name."""
    candidates = [Path(model), Path("sdde_MLP_runs") / model]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find model run {model!r}; tried: {tried}")


def load_hparams(run_dir: Path) -> dict:
    path = run_dir / "hyper_parameters.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_checkpoint(run_dir: Path, prefix: str) -> str:
    candidates = glob.glob(str(run_dir / f"{prefix}-*.index"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}-*.index checkpoints in {run_dir}")

    def checkpoint_step(filename: str) -> int:
        return int(Path(filename).stem.rsplit("-", 1)[1])

    return str(Path(max(candidates, key=checkpoint_step)).with_suffix(""))


def build_encoder_decoder(ndims_latent: int, num_fft_components: int):
    """Recreate the architecture used by train_MLP_model3.py."""
    x_input = tf.keras.layers.Input(
        shape=[num_fft_components], name="fft_amplitudes"
    )
    x = tf.keras.layers.Dense(256, activation="relu", name="enc_dense_1")(x_input)
    x = tf.keras.layers.Dense(256, activation="relu", name="enc_dense_2")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="enc_dense_3")(x)
    z = tf.keras.layers.Dense(ndims_latent, activation=None, name="latent_space")(x)
    encoder = tf.keras.Model(inputs=x_input, outputs=z)

    z_input = tf.keras.layers.Input(
        shape=[ndims_latent], name="latent_representations"
    )
    y = tf.keras.layers.Dense(128, activation="relu", name="dec_dense_1")(z_input)
    y = tf.keras.layers.Dense(256, activation="relu", name="dec_dense_2")(y)
    y = tf.keras.layers.Dense(256, activation="relu", name="dec_dense_3")(y)
    y = tf.keras.layers.Dense(
        num_fft_components, activation=None, name="pred_fft_amplitudes"
    )(y)
    decoder = tf.keras.Model(inputs=z_input, outputs=y)
    return encoder, decoder


def timeseries_to_fft_log_amplitudes(
    timeseries: np.ndarray, num_fft_components: int, fft_log_eps: float
) -> np.ndarray:
    """Apply the exact Hann-windowed log-amplitude transform used in training."""
    values = np.squeeze(timeseries, axis=-1).astype(np.float32)
    n_time = values.shape[1]
    window = np.hanning(n_time).astype(np.float32)
    amplitudes = np.abs(np.fft.fft(values * window[None, :], axis=1)) / float(n_time)
    return np.log(amplitudes[:, :num_fft_components] + fft_log_eps).astype(
        np.float32
    )


def load_observed_sn(data_path: Path, expected_length: int):
    """Load the full SILSO yearly CSV using the crop defined by SDDEpy."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Could not find obsSN data: {data_path}")
    data = np.loadtxt(data_path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected a two-column [year, sunspot_number] CSV, got {data.shape}"
        )
    observed = data[49:-6]
    if observed.shape[0] != expected_length:
        raise ValueError(
            "The SDDEpy obsSN crop data[49:-6] produced "
            f"{observed.shape[0]} values, but the model expects {expected_length}."
        )
    years = observed[:, 0].astype(np.float64)
    values = observed[:, 1].astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("obsSN contains non-finite sunspot numbers")
    return years, values


def latent_label(index: int) -> str:
    if index < len(PARAM_NAMES):
        return rf"$z_{index + 1}$ ({PARAM_NAMES[index]})"
    return rf"$z_{index + 1}$ (free)"


def add_prior_prism(ax, x_limits, y_limits, z_limits) -> None:
    """Draw the prior rectangle as a translucent prism over the observed z6 range."""
    x_lo, x_hi = x_limits
    y_lo, y_hi = y_limits
    z_lo, z_hi = z_limits
    corners = [
        (x_lo, y_lo, z_lo),
        (x_hi, y_lo, z_lo),
        (x_hi, y_hi, z_lo),
        (x_lo, y_hi, z_lo),
        (x_lo, y_lo, z_hi),
        (x_hi, y_lo, z_hi),
        (x_hi, y_hi, z_hi),
        (x_lo, y_hi, z_hi),
    ]
    faces = [
        [corners[index] for index in (0, 1, 2, 3)],
        [corners[index] for index in (4, 5, 6, 7)],
        [corners[index] for index in (0, 1, 5, 4)],
        [corners[index] for index in (1, 2, 6, 5)],
        [corners[index] for index in (2, 3, 7, 6)],
        [corners[index] for index in (3, 0, 4, 7)],
    ]
    prism = Poly3DCollection(
        faces,
        facecolors="tab:blue",
        edgecolors="tab:blue",
        linewidths=0.7,
        alpha=0.055,
        zorder=0,
    )
    ax.add_collection3d(prism)


def plot_z6_triplets(
    z: np.ndarray,
    output_path: Path,
    *,
    prior_limits: dict,
    observed_z: np.ndarray | None,
    title: str,
    dpi: int,
) -> None:
    """Plot all ten (z_i, z_j, z_6) combinations in one overview."""
    triplets = list(combinations(range(5), 2))
    ncols = 3
    nrows = int(np.ceil(len(triplets) / ncols))
    fig = plt.figure(figsize=(17, 18), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.045],
    )
    z6_values = z[:, 5] if observed_z is None else np.append(z[:, 5], observed_z[5])
    z6_min = float(z6_values.min())
    z6_max = float(z6_values.max())
    if z6_min == z6_max:
        z6_min -= 0.5
        z6_max += 0.5
    color_norm = plt.Normalize(vmin=z6_min, vmax=z6_max)
    color_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap="viridis")

    for panel, (i, j) in enumerate(triplets, start=1):
        row, column = divmod(panel - 1, ncols)
        ax = fig.add_subplot(
            grid[row, column], projection="3d", computed_zorder=False
        )
        x_limits = prior_limits[PARAM_NAMES[i]]
        y_limits = prior_limits[PARAM_NAMES[j]]
        in_prior = (
            (z[:, i] >= x_limits[0])
            & (z[:, i] <= x_limits[1])
            & (z[:, j] >= y_limits[0])
            & (z[:, j] <= y_limits[1])
        )
        if np.any(in_prior):
            ax.scatter(
                z[in_prior, i],
                z[in_prior, j],
                z[in_prior, 5],
                c=z[in_prior, 5],
                cmap="viridis",
                norm=color_norm,
                s=8,
                alpha=0.32,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
        if np.any(~in_prior):
            ax.scatter(
                z[~in_prior, i],
                z[~in_prior, j],
                z[~in_prior, 5],
                c="red",
                s=13,
                alpha=0.8,
                linewidths=0,
                label="outside shown priors",
                rasterized=True,
                zorder=3,
            )
            ax.legend(loc="upper right", fontsize=6, framealpha=0.8)

        add_prior_prism(ax, x_limits, y_limits, (z6_min, z6_max))
        if observed_z is not None:
            ax.scatter(
                observed_z[i],
                observed_z[j],
                observed_z[5],
                marker="*",
                c="#ff00cc",
                edgecolors="black",
                linewidths=0.9,
                s=190,
                depthshade=False,
                label="obsSN",
                zorder=20,
            )
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        x_values = z[:, i] if observed_z is None else np.append(z[:, i], observed_z[i])
        y_values = z[:, j] if observed_z is None else np.append(z[:, j], observed_z[j])
        ax.set_xlim(min(float(x_values.min()), x_limits[0]), max(float(x_values.max()), x_limits[1]))
        ax.set_ylim(min(float(y_values.min()), y_limits[0]), max(float(y_values.max()), y_limits[1]))
        ax.set_zlim(z6_min, z6_max)
        ax.set_xlabel(latent_label(i), labelpad=7)
        ax.set_ylabel(latent_label(j), labelpad=7)
        ax.set_zlabel(latent_label(5), labelpad=7)
        ax.set_title(rf"$(z_{i + 1}, z_{j + 1}, z_6)$", pad=4)
        ax.view_init(elev=24, azim=-58)
        ax.tick_params(labelsize=7, pad=1)

    colorbar_axis = fig.add_subplot(grid[:, -1])
    colorbar = fig.colorbar(color_mappable, cax=colorbar_axis)
    colorbar.set_label(r"$z_6$ (points inside shown priors)")
    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_z6_triplets_interactive(
    z: np.ndarray,
    parameters: np.ndarray,
    output_path: Path,
    *,
    prior_limits: dict,
    observed_z: np.ndarray | None,
    title: str,
) -> None:
    """Save rotatable Plotly versions of all ten z6 triplets."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError(
            "Interactive output requires Plotly. Install the project environment "
            "again or run `python -m pip install plotly`."
        ) from exc

    triplets = list(combinations(range(5), 2))
    ncols = 3
    nrows = int(np.ceil(len(triplets) / ncols))
    subplot_titles = [
        f"(z{i + 1}, z{j + 1}, z6)" for i, j in triplets
    ] + [""] * (nrows * ncols - len(triplets))
    figure = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "scene"}] * ncols for _ in range(nrows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.035,
        vertical_spacing=0.055,
    )

    z6_values = z[:, 5] if observed_z is None else np.append(z[:, 5], observed_z[5])
    z6_min = float(z6_values.min())
    z6_max = float(z6_values.max())
    if z6_min == z6_max:
        z6_min -= 0.5
        z6_max += 0.5
    hover_template = (
        "z1=%{customdata[0]:.5g}<br>z2=%{customdata[1]:.5g}<br>"
        "z3=%{customdata[2]:.5g}<br>z4=%{customdata[3]:.5g}<br>"
        "z5=%{customdata[4]:.5g}<br>z6=%{customdata[5]:.5g}<br>"
        "tau=%{customdata[6]:.5g}<br>T=%{customdata[7]:.5g}<br>"
        "Nd=%{customdata[8]:.5g}<br>sigma=%{customdata[9]:.5g}<br>"
        "Bmax=%{customdata[10]:.5g}<extra></extra>"
    )
    custom_data = np.column_stack((z, parameters))
    outlier_legend_added = False

    for panel, (i, j) in enumerate(triplets):
        row, column = divmod(panel, ncols)
        row += 1
        column += 1
        x_limits = prior_limits[PARAM_NAMES[i]]
        y_limits = prior_limits[PARAM_NAMES[j]]
        in_prior = (
            (z[:, i] >= x_limits[0])
            & (z[:, i] <= x_limits[1])
            & (z[:, j] >= y_limits[0])
            & (z[:, j] <= y_limits[1])
        )

        if np.any(in_prior):
            figure.add_trace(
                go.Scatter3d(
                    x=z[in_prior, i],
                    y=z[in_prior, j],
                    z=z[in_prior, 5],
                    mode="markers",
                    marker={
                        "size": 2.5,
                        "opacity": 0.32,
                        "color": z[in_prior, 5],
                        "coloraxis": "coloraxis",
                    },
                    customdata=custom_data[in_prior],
                    hovertemplate=hover_template,
                    showlegend=False,
                ),
                row=row,
                col=column,
            )
        if np.any(~in_prior):
            figure.add_trace(
                go.Scatter3d(
                    x=z[~in_prior, i],
                    y=z[~in_prior, j],
                    z=z[~in_prior, 5],
                    mode="markers",
                    marker={"size": 3.5, "opacity": 0.85, "color": "red"},
                    customdata=custom_data[~in_prior],
                    hovertemplate=hover_template,
                    name="outside shown priors",
                    legendgroup="outliers",
                    showlegend=not outlier_legend_added,
                ),
                row=row,
                col=column,
            )
            outlier_legend_added = True

        if observed_z is not None:
            figure.add_trace(
                go.Scatter3d(
                    x=[observed_z[i]],
                    y=[observed_z[j]],
                    z=[observed_z[5]],
                    mode="markers",
                    marker={
                        "size": 9,
                        "symbol": "diamond",
                        "color": "#ff00cc",
                        "line": {"color": "black", "width": 3},
                    },
                    customdata=[observed_z],
                    hovertemplate=(
                        "<b>obsSN</b><br>z1=%{customdata[0]:.5g}<br>"
                        "z2=%{customdata[1]:.5g}<br>z3=%{customdata[2]:.5g}<br>"
                        "z4=%{customdata[3]:.5g}<br>z5=%{customdata[4]:.5g}<br>"
                        "z6=%{customdata[5]:.5g}<extra></extra>"
                    ),
                    name="obsSN",
                    legendgroup="obsSN",
                    showlegend=panel == 0,
                ),
                row=row,
                col=column,
            )

        x_lo, x_hi = x_limits
        y_lo, y_hi = y_limits
        x_values = z[:, i] if observed_z is None else np.append(z[:, i], observed_z[i])
        y_values = z[:, j] if observed_z is None else np.append(z[:, j], observed_z[j])
        scene_name = "scene" if panel == 0 else f"scene{panel + 1}"
        figure.layout[scene_name].update(
            xaxis={
                "title": f"z{i + 1} ({PARAM_NAMES[i]})",
                "range": [min(float(x_values.min()), x_lo), max(float(x_values.max()), x_hi)],
            },
            yaxis={
                "title": f"z{j + 1} ({PARAM_NAMES[j]})",
                "range": [min(float(y_values.min()), y_lo), max(float(y_values.max()), y_hi)],
            },
            zaxis={"title": "z6 (free)", "range": [z6_min, z6_max]},
            aspectmode="cube",
        )

    figure.update_layout(
        title={"text": title, "x": 0.5},
        width=1500,
        height=1800,
        margin={"l": 20, "r": 120, "t": 90, "b": 20},
        coloraxis={
            "colorscale": "Viridis",
            "cmin": z6_min,
            "cmax": z6_max,
            "colorbar": {"title": "z6", "len": 0.55},
        },
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def main():
    args = parse_args()
    if args.nsamples < 1:
        raise ValueError("--nsamples must be >= 1")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")
    if args.dpi < 1:
        raise ValueError("--dpi must be >= 1")

    run_dir = resolve_run_dir(args.model)
    hp = load_hparams(run_dir)
    if hp.get("representation_mode") != "fourier_amplitude":
        raise ValueError(
            "This script requires an MLP run with "
            f"representation_mode='fourier_amplitude'; got {hp.get('representation_mode')!r}"
        )

    ndims_latent = int(hp["ndims_latent"])
    if ndims_latent != 6:
        raise ValueError(
            f"The z6-triplet plots require ndims_latent=6; this run has {ndims_latent}"
        )

    num_fft_components = int(hp["num_fft_components"])
    fft_log_eps = float(hp.get("fft_log_eps", 1e-8))
    to_tuple = lambda value: tuple(float(v) for v in value)
    prior_limits = {
        name: to_tuple(hp[f"{name}_lims"])
        for name in PARAM_NAMES
    }

    output_dir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else run_dir / "diagnostics" / "prior_latent"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_prefix = "model_ckpt" if args.last else "model_best_ckpt"
    checkpoint_path = find_latest_checkpoint(run_dir, checkpoint_prefix)
    checkpoint_step = int(Path(checkpoint_path).name.rsplit("-", 1)[1])

    encoder, decoder = build_encoder_decoder(ndims_latent, num_fft_components)
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    for attempt in (1, 2):
        try:
            checkpoint.restore(checkpoint_path).expect_partial()
            break
        except Exception as exc:
            if attempt == 1:
                time.sleep(0.5)
            else:
                raise RuntimeError(
                    f"Failed to restore checkpoint {checkpoint_path}: {exc}"
                ) from exc

    prng = np.random.RandomState(args.seed)
    generator = src.generators.DataGenerator_SolarDynamo_SDDE_ENCA(
        prng=prng,
        Tobs=int(hp["Tobs"]),
        saveat=float(hp["saveat"]),
        num_noise_channels=int(hp["num_noise_channels"]),
        Twarmup=int(hp["Twarmup"]),
        dt=float(hp["dt"]),
        tau_lims=prior_limits["tau"],
        T_lims=prior_limits["T"],
        Nd_lims=prior_limits["Nd"],
        sigma_lims=prior_limits["sigma"],
        Bmax_lims=prior_limits["Bmax"],
    )

    len_timeseries = int(round(float(hp["Tobs"]) / float(hp["saveat"])))
    raw = np.empty((args.nsamples, len_timeseries, 1), dtype=np.float32)
    parameters = np.empty((args.nsamples, len(PARAM_NAMES)), dtype=np.float32)
    iterator = iter(generator)
    progress_interval = max(1, min(100, args.nsamples // 10))
    print(f"Sampling {args.nsamples} prior observations (seed={args.seed}) ...")
    for sample_index in range(args.nsamples):
        observation, theta, _ = next(iterator)
        raw[sample_index] = observation
        parameters[sample_index] = theta
        completed = sample_index + 1
        if completed % progress_interval == 0 or completed == args.nsamples:
            print(f"  simulated {completed}/{args.nsamples}", flush=True)

    spectra = timeseries_to_fft_log_amplitudes(
        raw, num_fft_components, fft_log_eps
    )
    latent = np.empty((args.nsamples, ndims_latent), dtype=np.float32)
    for start in range(0, args.nsamples, args.batch):
        stop = min(start + args.batch, args.nsamples)
        latent[start:stop] = encoder(spectra[start:stop], training=False).numpy()

    observed_years = None
    observed_values = None
    observed_latent = None
    observed_data_path = None
    if args.obs_sn_data:
        observed_data_path = Path(args.obs_sn_data).expanduser().resolve()
        observed_years, observed_values = load_observed_sn(
            observed_data_path, len_timeseries
        )
        observed_raw = observed_values.reshape(1, len_timeseries, 1)
        observed_spectrum = timeseries_to_fft_log_amplitudes(
            observed_raw, num_fft_components, fft_log_eps
        )
        observed_latent = encoder(observed_spectrum, training=False).numpy()[0]
        print(
            "[OK] Encoded obsSN "
            f"({observed_years[0]:g} to {observed_years[-1]:g}): "
            + ", ".join(
                f"z{index + 1}={value:.6g}"
                for index, value in enumerate(observed_latent)
            )
        )

    file_stem = (
        f"prior_latent_{checkpoint_prefix}_step{checkpoint_step}_"
        f"n{args.nsamples}_seed{args.seed}"
    )
    data_path = output_dir / f"{file_stem}.npz"
    plot_path = output_dir / f"{file_stem}_z6_triplets.png"
    interactive_path = output_dir / f"{file_stem}_z6_triplets_interactive.html"
    saved_arrays = {
        "parameters": parameters,
        "latent": latent,
        "parameter_names": np.asarray(PARAM_NAMES),
        "checkpoint": np.asarray(checkpoint_path),
        "seed": np.asarray(args.seed),
    }
    if observed_latent is not None:
        saved_arrays.update(
            observed_latent=observed_latent,
            observed_years=observed_years,
            observed_values=observed_values,
            observed_data_path=np.asarray(str(observed_data_path)),
        )
    np.savez_compressed(data_path, **saved_arrays)
    plot_z6_triplets(
        latent,
        plot_path,
        prior_limits=prior_limits,
        observed_z=observed_latent,
        title=(
            f"Prior latent distribution | {run_dir.name} | {checkpoint_prefix} "
            f"step {checkpoint_step} | n={args.nsamples} | seed={args.seed}"
        ),
        dpi=args.dpi,
    )
    if args.interactive:
        plot_z6_triplets_interactive(
            latent,
            parameters,
            interactive_path,
            prior_limits=prior_limits,
            observed_z=observed_latent,
            title=(
                f"Prior latent distribution | {run_dir.name} | {checkpoint_prefix} "
                f"step {checkpoint_step} | n={args.nsamples} | seed={args.seed}"
            ),
        )

    print(f"[OK] Restored: {checkpoint_path}")
    print(f"[OK] Saved samples: {data_path}")
    print(f"[OK] Saved z6 triplet plots: {plot_path}")
    if args.interactive:
        print(f"[OK] Saved interactive z6 triplets: {interactive_path}")


if __name__ == "__main__":
    main()
