#!/usr/bin/env python3
"""Sample the SDDE prior and visualize its encoded MLP latent distribution.

The default plots contain every three-dimensional latent-space combination
(z_i, z_j, z_last), where z_last is the final latent dimension and i, j are
earlier dimensions.  The sampled parameters and latent coordinates are also
saved as a compressed NumPy archive so that plotting experiments do not require
rerunning the SDDE simulator.
"""

import argparse
import glob
import json
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np

# IMPORTANT: initialize the canonical SABC SDDE package before TensorFlow.
from sdde_model import init_julia

init_julia()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import src.generators


BASE_PARAM_NAMES = ("tau", "T", "Nd", "sigma", "Bmax")
DEFAULT_MODEL = "20260611_mlp_z6_1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw synthetic observations from a saved MLP run's parameter "
            "priors, encode them, and plot all (z_i, z_j, z_last) triplets."
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


def build_encoder_decoder(
    ndims_latent: int,
    num_fft_components: int,
    len_timeseries: int,
    num_noise_channels: int,
):
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
    noise_vectors = tf.keras.layers.Input(
        shape=[len_timeseries, num_noise_channels], name="noise_vectors"
    )
    noise_zero = tf.keras.layers.Lambda(
        lambda n: tf.zeros_like(tf.reduce_sum(n, axis=[1, 2])),
        name="noise_interface_zero",
    )(noise_vectors)
    decoder_input = tf.keras.layers.Lambda(
        lambda values: values[0] + values[1][:, tf.newaxis],
        name="attach_noise_interface",
    )([z_input, noise_zero])
    y = tf.keras.layers.Dense(128, activation="relu", name="dec_dense_1")(decoder_input)
    y = tf.keras.layers.Dense(256, activation="relu", name="dec_dense_2")(y)
    y = tf.keras.layers.Dense(256, activation="relu", name="dec_dense_3")(y)
    y = tf.keras.layers.Dense(
        num_fft_components, activation=None, name="pred_fft_amplitudes"
    )(y)
    decoder = tf.keras.Model(inputs=[z_input, noise_vectors], outputs=y)
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


def latent_label(index: int, param_names) -> str:
    if index < len(param_names):
        return rf"$z_{index + 1}$ ({param_names[index]})"
    return rf"$z_{index + 1}$ (free)"


def add_prior_prism(ax, x_limits, y_limits, z_limits) -> None:
    """Draw the prior rectangle as a translucent prism over the z-axis range."""
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


def plot_last_latent_triplets(
    z: np.ndarray,
    output_path: Path,
    *,
    prior_limits: dict,
    param_names,
    observed_z: np.ndarray | None,
    title: str,
    dpi: int,
) -> None:
    """Plot all (z_i, z_j, z_last) combinations in one overview."""
    last_index = z.shape[1] - 1
    last_label = f"z{last_index + 1}"
    triplets = list(combinations(range(last_index), 2))
    ncols = 3
    nrows = int(np.ceil(len(triplets) / ncols))
    fig = plt.figure(figsize=(17, 18), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.045],
    )
    z_axis_values = (
        z[:, last_index]
        if observed_z is None
        else np.append(z[:, last_index], observed_z[last_index])
    )
    z_axis_min = float(z_axis_values.min())
    z_axis_max = float(z_axis_values.max())
    if z_axis_min == z_axis_max:
        z_axis_min -= 0.5
        z_axis_max += 0.5
    color_norm = plt.Normalize(vmin=z_axis_min, vmax=z_axis_max)
    color_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap="viridis")

    for panel, (i, j) in enumerate(triplets, start=1):
        row, column = divmod(panel - 1, ncols)
        ax = fig.add_subplot(
            grid[row, column], projection="3d", computed_zorder=False
        )
        x_prior = prior_limits[param_names[i]] if i < len(param_names) else None
        y_prior = prior_limits[param_names[j]] if j < len(param_names) else None
        in_prior = np.ones(z.shape[0], dtype=bool)
        if x_prior is not None:
            in_prior &= (z[:, i] >= x_prior[0]) & (z[:, i] <= x_prior[1])
        if y_prior is not None:
            in_prior &= (z[:, j] >= y_prior[0]) & (z[:, j] <= y_prior[1])
        if np.any(in_prior):
            ax.scatter(
                z[in_prior, i],
                z[in_prior, j],
                z[in_prior, last_index],
                c=z[in_prior, last_index],
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
                z[~in_prior, last_index],
                c="red",
                s=13,
                alpha=0.8,
                linewidths=0,
                label="outside shown priors",
                rasterized=True,
                zorder=3,
            )
            ax.legend(loc="upper right", fontsize=6, framealpha=0.8)

        if x_prior is not None and y_prior is not None:
            add_prior_prism(ax, x_prior, y_prior, (z_axis_min, z_axis_max))
        if observed_z is not None:
            ax.scatter(
                observed_z[i],
                observed_z[j],
                observed_z[last_index],
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
        x_reference = x_prior or (float(x_values.min()), float(x_values.max()))
        y_reference = y_prior or (float(y_values.min()), float(y_values.max()))
        ax.set_xlim(min(float(x_values.min()), x_reference[0]), max(float(x_values.max()), x_reference[1]))
        ax.set_ylim(min(float(y_values.min()), y_reference[0]), max(float(y_values.max()), y_reference[1]))
        ax.set_zlim(z_axis_min, z_axis_max)
        ax.set_xlabel(latent_label(i, param_names), labelpad=7)
        ax.set_ylabel(latent_label(j, param_names), labelpad=7)
        ax.set_zlabel(latent_label(last_index, param_names), labelpad=7)
        ax.set_title(rf"$(z_{i + 1}, z_{j + 1}, z_{last_index + 1})$", pad=4)
        ax.view_init(elev=24, azim=-58)
        ax.tick_params(labelsize=7, pad=1)

    colorbar_axis = fig.add_subplot(grid[:, -1])
    colorbar = fig.colorbar(color_mappable, cax=colorbar_axis)
    colorbar.set_label(rf"${last_label}$ (points inside shown priors)")
    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_last_latent_triplets_interactive(
    z: np.ndarray,
    parameters: np.ndarray,
    output_path: Path,
    *,
    prior_limits: dict,
    param_names,
    observed_z: np.ndarray | None,
    title: str,
) -> None:
    """Save rotatable Plotly versions of all z_last triplets."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError(
            "Interactive output requires Plotly. Install the project environment "
            "again or run `python -m pip install plotly`."
        ) from exc

    last_index = z.shape[1] - 1
    last_label = f"z{last_index + 1}"
    triplets = list(combinations(range(last_index), 2))
    ncols = 3
    nrows = int(np.ceil(len(triplets) / ncols))
    subplot_titles = [
        f"(z{i + 1}, z{j + 1}, {last_label})" for i, j in triplets
    ] + [""] * (nrows * ncols - len(triplets))
    figure = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "scene"}] * ncols for _ in range(nrows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.035,
        vertical_spacing=0.055,
    )

    z_axis_values = (
        z[:, last_index]
        if observed_z is None
        else np.append(z[:, last_index], observed_z[last_index])
    )
    z_axis_min = float(z_axis_values.min())
    z_axis_max = float(z_axis_values.max())
    if z_axis_min == z_axis_max:
        z_axis_min -= 0.5
        z_axis_max += 0.5
    latent_hover = "".join(
        f"z{index + 1}=%{{customdata[{index}]:.5g}}<br>"
        for index in range(z.shape[1])
    )
    parameter_hover = "".join(
        f"{name}=%{{customdata[{z.shape[1] + index}]:.5g}}<br>"
        for index, name in enumerate(param_names)
    )
    hover_template = latent_hover + parameter_hover + "<extra></extra>"
    custom_data = np.column_stack((z, parameters))
    outlier_legend_added = False

    for panel, (i, j) in enumerate(triplets):
        row, column = divmod(panel, ncols)
        row += 1
        column += 1
        x_prior = prior_limits[param_names[i]] if i < len(param_names) else None
        y_prior = prior_limits[param_names[j]] if j < len(param_names) else None
        in_prior = np.ones(z.shape[0], dtype=bool)
        if x_prior is not None:
            in_prior &= (z[:, i] >= x_prior[0]) & (z[:, i] <= x_prior[1])
        if y_prior is not None:
            in_prior &= (z[:, j] >= y_prior[0]) & (z[:, j] <= y_prior[1])

        if np.any(in_prior):
            figure.add_trace(
                go.Scatter3d(
                    x=z[in_prior, i],
                    y=z[in_prior, j],
                    z=z[in_prior, last_index],
                    mode="markers",
                    marker={
                        "size": 2.5,
                        "opacity": 0.32,
                        "color": z[in_prior, last_index],
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
                    z=z[~in_prior, last_index],
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
                    z=[observed_z[last_index]],
                    mode="markers",
                    marker={
                        "size": 9,
                        "symbol": "diamond",
                        "color": "#ff00cc",
                        "line": {"color": "black", "width": 3},
                    },
                    customdata=[observed_z],
                    hovertemplate=(
                        "<b>obsSN</b><br>"
                        + latent_hover
                        + "<extra></extra>"
                    ),
                    name="obsSN",
                    legendgroup="obsSN",
                    showlegend=panel == 0,
                ),
                row=row,
                col=column,
            )

        x_values = z[:, i] if observed_z is None else np.append(z[:, i], observed_z[i])
        y_values = z[:, j] if observed_z is None else np.append(z[:, j], observed_z[j])
        x_reference = x_prior or (float(x_values.min()), float(x_values.max()))
        y_reference = y_prior or (float(y_values.min()), float(y_values.max()))
        x_lo, x_hi = x_reference
        y_lo, y_hi = y_reference
        scene_name = "scene" if panel == 0 else f"scene{panel + 1}"
        figure.layout[scene_name].update(
            xaxis={
                "title": f"z{i + 1} ({param_names[i]})",
                "range": [min(float(x_values.min()), x_lo), max(float(x_values.max()), x_hi)],
            },
            yaxis={
                "title": f"z{j + 1} ({param_names[j]})",
                "range": [min(float(y_values.min()), y_lo), max(float(y_values.max()), y_hi)],
            },
            zaxis={
                "title": latent_label(last_index, param_names).replace("$", ""),
                "range": [z_axis_min, z_axis_max],
            },
            aspectmode="cube",
        )

    figure.update_layout(
        title={"text": title, "x": 0.5},
        width=1500,
        height=1800,
        margin={"l": 20, "r": 120, "t": 90, "b": 20},
        coloraxis={
            "colorscale": "Viridis",
            "cmin": z_axis_min,
            "cmax": z_axis_max,
            "colorbar": {"title": last_label, "len": 0.55},
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
    model = str(hp.get("model", "original")).strip().lower()
    if model not in {"original", "jupiter"}:
        raise ValueError(f"Unknown saved SDDE model {model!r}.")
    param_names = BASE_PARAM_NAMES + (("Aj",) if model == "jupiter" else ())
    num_model_parameters = int(hp.get("num_model_parameters", len(param_names)))
    if num_model_parameters != len(param_names):
        raise ValueError(
            f"Saved model={model!r} requires {len(param_names)} regressors, but "
            f"num_model_parameters={num_model_parameters}."
        )
    if ndims_latent < num_model_parameters:
        raise ValueError(
            "The latent space cannot be smaller than the supervised coordinates; "
            f"model={model!r} has {num_model_parameters} regressors but this "
            f"run has ndims_latent={ndims_latent}."
        )

    num_fft_components = int(hp["num_fft_components"])
    fft_log_eps = float(hp.get("fft_log_eps", 1e-8))
    to_tuple = lambda value: tuple(float(v) for v in value)
    prior_limits = {
        name: to_tuple(hp[f"{name}_lims"])
        for name in param_names
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

    encoder, decoder = build_encoder_decoder(
        ndims_latent,
        num_fft_components,
        int(hp["len_timeseries"]),
        int(hp["num_noise_channels"]),
    )
    # Only latent statistics are used below. Encoder-only restore keeps legacy
    # encoders inspectable despite the corrected decoder's noise input.
    checkpoint = tf.train.Checkpoint(encoder=encoder)
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
    generator = src.generators.DataGenerator_SolarDynamo_SDDE_MLP(
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
        Aj_lims=prior_limits.get("Aj", (0.0, 0.1)),
        model=model,
        jupiter_period=float(hp.get("jupiter_period", 11.86)),
    )

    len_timeseries = int(round(float(hp["Tobs"]) / float(hp["saveat"])))
    raw = np.empty((args.nsamples, len_timeseries, 1), dtype=np.float32)
    parameters = np.empty((args.nsamples, len(param_names)), dtype=np.float32)
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
    last_latent_label = f"z{ndims_latent}"
    plot_path = output_dir / f"{file_stem}_{last_latent_label}_triplets.png"
    interactive_path = (
        output_dir / f"{file_stem}_{last_latent_label}_triplets_interactive.html"
    )
    saved_arrays = {
        "parameters": parameters,
        "latent": latent,
        "parameter_names": np.asarray(param_names),
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
    plot_last_latent_triplets(
        latent,
        plot_path,
        prior_limits=prior_limits,
        param_names=param_names,
        observed_z=observed_latent,
        title=(
            f"Prior latent distribution | {run_dir.name} | {checkpoint_prefix} "
            f"step {checkpoint_step} | n={args.nsamples} | seed={args.seed}"
        ),
        dpi=args.dpi,
    )
    if args.interactive:
        plot_last_latent_triplets_interactive(
            latent,
            parameters,
            interactive_path,
            prior_limits=prior_limits,
            param_names=param_names,
            observed_z=observed_latent,
            title=(
                f"Prior latent distribution | {run_dir.name} | {checkpoint_prefix} "
                f"step {checkpoint_step} | n={args.nsamples} | seed={args.seed}"
            ),
        )

    print(f"[OK] Restored: {checkpoint_path}")
    print(f"[OK] Saved samples: {data_path}")
    print(f"[OK] Saved {last_latent_label} triplet plots: {plot_path}")
    if args.interactive:
        print(f"[OK] Saved interactive {last_latent_label} triplets: {interactive_path}")


if __name__ == "__main__":
    main()
