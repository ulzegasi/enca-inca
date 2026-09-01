# ENCA-INCA

[![launch - renku](https://img.shields.io/badge/launch-renku-2ea44f?logo=python)](https://renkulab.io/projects/bistom/enca-inca/sessions/new?autostart=1)

## Learning Summary Statistics for Bayesian Inference with Autoencoders 

The code here implements the published work by Carlo Albert, Simone Ulzega, Fernando Perez-Cruz, Firat Ozdemir, and Antonietta Mira: *Learning summary statistics for Bayesian inference with Autoencoders*, SciPost Phys. Core 5, 043 (2022).

## Contents 
The repository contains the code needed to train the proposed models _explicit noise conditional autoencoder_ (ENCA) and _implicit noise conditional autoencoder_ (INCA) for several simulator settings:

- `train_ENCA_model1.py`, `train_INCA_model1.py`: original model 1 experiments
- `train_ENCA_model2.py`, `train_INCA_model2.py`: original model 2 experiments
- `train_ENCA_model3.py`: solar-dynamo / SDDE ENCA experiments using the Julia-backed simulator
- `train_ENCAFourierCNN_model3.py`: Fourier-space SDDE ENCA with a noise-conditioned CNN decoder
- `train_MLP_model3.py`: solar-dynamo / SDDE MLP experiments on Fourier-amplitude representations
- `train_FNO_model3.py`: solar-dynamo / SDDE FNO experiments with configurable time- or Fourier-domain reconstruction loss

Supporting code lives in `src/`. Local solar-dynamo training outputs are written to `sdde_ENCA_runs/` for time-domain ENCA, `sdde_ENCAFourierCNN_runs/` for Fourier-CNN ENCA, and `sdde_MLP_runs/` for MLP.

## Demo   
The repository can be set up on a clean environment by creating a conda environment:

```bash
conda env create -f environment.yml
conda activate encainca
pip install -e /path/to/SDDE-model
```

This environment currently targets Python 3.10 and TensorFlow 2.x (`tensorflow>=2.14` in `environment.yml`).
`SDDE-model` is the canonical StochasticDelayDiffEq/EM implementation shared
with SABC; install the same checkout in both environments so training and
inference cannot drift between simulator implementations.

For the original experiments, one can run:

```bash
python train_ENCA_model1.py
python train_INCA_model1.py
python train_ENCA_model2.py
python train_INCA_model2.py
``` 

For the solar-dynamo / SDDE experiments, use:

```bash
python train_ENCA_model3.py
python train_MLP_model3.py
python train_FNO_model3.py
```

Before launching one of these training scripts, manually review the `ExpSetup` class in the script you are running and set the key run parameters there. In particular:

```python
self.ndims_latent = 10

self.Tobs = 271  # C14 dataset: 929, obsSN dataset: 271

self.batch_size = 64
self.max_training_steps = int(2500)  # full run example: int(3e6)
self.freq_log = 100
```

## SDDE Model Architectures

The solar-dynamo / SDDE workflows are split by script so the architecture and data representation are explicit.

### ENCA

[train_ENCA_model3.py](/Users/ulzg/switchdrive/ZHAW_BISTOM/RENKU/enca-inca/train_ENCA_model3.py) is the time-domain ENCA model. It keeps `representation_mode = "time"` in `ExpSetup` for checkpoint metadata compatibility.

- input representation: time series with shape `[batch, len_timeseries, 1]`
- encoder: Conv1D blocks with ReLU activations and max pooling, followed by a final Conv1D and global average pooling into the latent summary vector
- decoder: latent vector tiled across time, concatenated with the sampled noise vector, then reconstructed with two bidirectional LSTM layers and a final dense output layer
- default output directory: `sdde_ENCA_runs/`
- logdir override: `ENCA_LOGDIR=/path/to/run`
- cluster launchers: `runtraining_cpu.sh`, `runtraining_gpu.sh`

### Fourier-CNN ENCA

[train_ENCAFourierCNN_model3.py](/Users/ulzg/switchdrive/ZHAW_BISTOM/RENKU/enca-inca/train_ENCAFourierCNN_model3.py) is the TensorFlow/Keras counterpart of the Fourier ENCA used in `AdvancedTopicsInPhysicsOfData-Project`, adapted to the online SDDE workflow in this repository.

- input representation: first 100 components of `log1p(abs(rFFT(x)))`, with shape `[batch, 100, 1]`; pass `--window Hann` to apply a symmetric Hann window to `x` before the rFFT (the default empty value preserves the legacy unwindowed transform)
- encoder: `Conv1D(16) -> Conv1D(16) -> MaxPool1D -> Conv1D(32) -> Conv1D(32) -> Conv1D(ndims_latent) -> GlobalAveragePooling1D`
- decoder: project the latent vector to 100 positions, resize the simulator noise from 271 to 100 positions, concatenate both channels, then apply `Conv1D(32) -> Conv1D(32) -> Conv1D(16) -> Conv1D(1)`
- default settings: `Tobs = 271`, `ndims_latent = 6`, batch size 300, and 1.2 million online-training steps
- loss: balanced per-sample reconstruction MSE plus prior-width-normalized parameter regression MSE
- Jupiter support: set `export MODEL="jupiter"` in the cluster launcher; the first six hard-coded latent
  coordinates regress `(tau, T, Nd, sigma, Bmax, Aj)`, phase is randomized and
  marginalized, and the decoder receives the same bare noise that drives the
  canonical `SDDE-model` NoiseGrid simulation
- the latent width remains configured directly as `self.ndims_latent` in
  `ExpSetup`; if it is set above six for Jupiter, every additional coordinate
  is retained as a free SABC statistic
- original-model MLP and ENCAfftCNN training also use the canonical
  `SDDEProblem + EM() + NoiseGrid` path; legacy original neural checkpoints
  must be retrained in fresh directories before repeating neural-statistics
  inference
- default output directory: `sdde_ENCAFourierCNN_runs/`
- logdir override: `ENCA_FOURIER_CNN_LOGDIR=/path/to/run`
- GPU cluster launcher: `runtraining_gpu_encafouriercnn.sh`

The launcher defines `WINDOW=""`. Set it to `WINDOW="Hann"` for a fresh
Hann-windowed run; it passes the selection to the training script as
`--window "$WINDOW"`. The choice is stored as `window` in
`hyper_parameters.json`, and a checkpoint cannot be resumed with a different
choice. Runs created before this option existed are interpreted as `window=""`.

The reference PyTorch project standardizes its Fourier data using mean and standard deviation computed from a finite training set. This online variant does not use dataset-wide standardization; `log1p` compresses the spectral dynamic range and the balanced reconstruction loss normalizes each sample's error by its RMS spectral amplitude.

### MLP

[train_MLP_model3.py](/Users/ulzg/switchdrive/ZHAW_BISTOM/RENKU/enca-inca/train_MLP_model3.py) is the Fourier-amplitude MLP autoencoder. It keeps `representation_mode = "fourier_amplitude"` in `ExpSetup` for checkpoint metadata compatibility.

- input representation: Hann-windowed log normalized FFT amplitudes with shape `[batch, num_fft_components]`
- encoder: `Dense(256, relu) -> Dense(256, relu) -> Dense(128, relu) -> Dense(ndims_latent)`
- decoder: `Dense(128, relu) -> Dense(256, relu) -> Dense(256, relu) -> Dense(num_fft_components)`
- simulator: canonical `sdde_model` StochasticDelayDiffEq solver with `EM()`,
  also used by SABC
- Jupiter contract: `z1` through `z6` regress `(tau, T, Nd, sigma, Bmax, Aj)`;
  phase is freshly sampled per realization and is not a regressed parameter
- latent width: set `NDIMS_LATENT=6` for only the six Jupiter regressors;
  every additional coordinate is retained as a free statistic (`7` gives one,
  `8` gives two, and so on)
- noise contract: bare Gaussian increments are generated at `dt=0.1`, drive
  the canonical solver, and are also supplied to the decoder at `saveat=1`;
  the current MLP keeps an explicit, zero-weighted noise connection so this
  interface can support noise-aware decoding later
- default output directory: `sdde_MLP_runs/`
- logdir override: `MLP_LOGDIR=/path/to/run`
- cluster launchers: `runtraining_cpu_mlp.sh`, `runtraining_gpu_mlp.sh`

Corrected Jupiter MLP training must use a fresh `MLP_LOGDIR`. The trainer
refuses to resume checkpoints that predate the canonical explicit-noise solver
metadata. Encoder-only diagnostics can still inspect a legacy checkpoint, but
its decoder is incompatible with the corrected two-input decoder and must not
be used for continued training or reconstruction.

### FNO

`train_FNO_model3.py` always feeds the original time-domain signal to the encoder
and always makes the decoder produce a time-domain signal. The reconstruction
objective is selected independently with `FNO_RECON_DOMAIN`:

- `time` (default): compare the true and reconstructed time series
- `fourier_log_amplitude`: apply the same Hann-windowed, normalized log-FFT
  transform as the MLP and compare the first 100 amplitude components

Start a fresh spectral-loss run with an explicit run directory:

```bash
FNO_RECON_DOMAIN=fourier_log_amplitude \
FNO_LOGDIR=sdde_FNO_runs/<new_spectral_run> \
python train_FNO_model3.py
```

The selected domain, `num_fft_components = 100`, and `fft_log_eps = 1e-8` are
saved in `hyper_parameters.json`. Older FNO runs without a saved domain are
treated as time-domain runs. The default FNO learning-rate schedule uses
`lr_decay_steps = 12000`.

## SDDE Loss Configuration

The SDDE training scripts [train_ENCA_model3.py](/Users/ulzg/switchdrive/ZHAW_BISTOM/RENKU/enca-inca/train_ENCA_model3.py) and [train_MLP_model3.py](/Users/ulzg/switchdrive/ZHAW_BISTOM/RENKU/enca-inca/train_MLP_model3.py) support two loss configurations. The active one is selected in `ExpSetup` via:

```python
self.loss_mode = "balanced_mse"
```

The available modes are:

- `legacy_chisq`: reproduces the older chi-square-style implementation
- `balanced_mse`: uses normalized mean-squared errors with consistent reductions for reconstruction and parameter regression

The loss-related settings currently live in `ExpSetup`:

```python
self.loss_mode = "balanced_mse"
self.lambda_recon = 1.0
self.lambda_reg = 1.0
self.recon_scale_eps = 1e-3
```

### Legacy Mode

`legacy_chisq` keeps the previous behavior:

- reconstruction uses a chi-square-like relative error along the time axis
- parameter regression uses a chi-square-like relative error across the batch
- the total loss is the unweighted sum of reconstruction and regression losses

This mode is kept mainly for backward compatibility and comparison with older runs.

### Balanced Mode

`balanced_mse` is now the default. It is designed so the two losses are reduced in a more comparable way, while still keeping their meanings separate.

Reconstruction loss:

- compute one RMS amplitude scale per sample from the true observation `x`
- clamp that scale from below with `recon_scale_eps`
- normalize reconstruction error by that per-sample scale
- compute plain MSE on the normalized signal

In formula form:

```python
scale_x = max(sqrt(mean(x^2 over observation dimensions)), recon_scale_eps)
loss_reconstruction = mean(((x - x_pred) / scale_x)^2)
```

For ENCA, the observation dimensions are the time-domain trajectory samples. For MLP, they are the Hann-windowed log FFT amplitude components.

Parameter regression loss:

- supervise the first five latent coordinates for the original model and the
  first six for Jupiter; in the Jupiter model the sixth target is `Aj`
- leave any coordinates after those regressors free (for example `z7` in a
  seven-dimensional Jupiter run)
- normalize each parameter error by its prior width
- compute plain MSE on those normalized parameter errors

In formula form:

```python
n_params = 6 if model == "jupiter" else 5
loss_regress_params = mean(((params - params_pred[:, :n_params]) / prior_widths)^2)
```

The final training objective is:

```python
loss_total = lambda_recon * loss_reconstruction + lambda_reg * loss_regress_params
```

with defaults:

```python
lambda_recon = 1.0
lambda_reg = 1.0
```

These defaults are only a starting point. They remove the old reduction mismatch, but they do not guarantee that the two losses will have identical magnitudes during training. If one objective is still clearly under-emphasized, increase its corresponding lambda.

### Why Two Modes Exist

The older implementation could make parameter regression numerically dominate reconstruction because the two losses were reduced differently. The new `balanced_mse` mode addresses that by making both losses true means after normalization.

The legacy mode is still available so that:

- older experiments remain reproducible
- past checkpoints remain easier to interpret
- side-by-side comparisons between old and new training objectives remain possible

### How To Switch Modes

Change the following lines in `ExpSetup`:

```python
self.loss_mode = "balanced_mse"
self.lambda_recon = 1.0
self.lambda_reg = 1.0
```

Examples:

```python
# new default
self.loss_mode = "balanced_mse"
self.lambda_recon = 1.0
self.lambda_reg = 1.0
```

```python
# old behavior
self.loss_mode = "legacy_chisq"
self.lambda_recon = 1.0
self.lambda_reg = 1.0
```

At startup, the script prints the active loss mode and the two lambda values.

### TensorBoard Scalars

The following top-level TensorBoard scalars are still written in both modes:

- `loss_total`
- `loss_reconstruction`
- `loss_regress_params`
- ENCA: `RMSE_x_ch_1`
- MLP: `RMSE_observation`
- `RMSE_z_ch_1` to `RMSE_z_ch_5`
- `theta/...` parameter min/max ranges
- `lr_schedule`

In addition, the lambda weights are logged as:

- `loss_weight/lambda_recon`
- `loss_weight/lambda_reg`

The per-component loss names depend on the selected mode.

If `loss_mode = "legacy_chisq"`, TensorBoard shows reconstruction and latent-parameter chi-square tags. For ENCA, the reconstruction tag is:

- `ChiSquare_x_ch_1`

For MLP, the legacy reconstruction tags follow the observation-vector indexing used by the script:

- `ChiSquare_x_ch_1` to `ChiSquare_x_ch_<num_fft_components>`

Both scripts also show:

- `ChiSquare_z_1`
- `ChiSquare_z_2`
- `ChiSquare_z_3`
- `ChiSquare_z_4`
- `ChiSquare_z_5`

If `loss_mode = "balanced_mse"`, TensorBoard shows the normalized latent-parameter tags in both scripts:

- `NormMSE_z_1`
- `NormMSE_z_2`
- `NormMSE_z_3`
- `NormMSE_z_4`
- `NormMSE_z_5`

The reconstruction tag differs by script:

ENCA uses `NormMSE_x_ch_1`; MLP uses `NormMSE_observation`.

So the high-level dashboard remains similar, but the detailed per-component loss tags change when you switch from the legacy chi-square losses to the normalized MSE losses.

## Diagnostics And Reconstruction Tests

After training an SDDE model, run the matching helper scripts against a saved run directory containing checkpoints and `hyper_parameters.json`.

For ENCA runs:

```bash
python diag_test_enca.py --logdir sdde_ENCA_runs/<run_name>
python recon_test_enca.py \
  --logdir sdde_ENCA_runs/<run_name> \
  --tau 2.0 \
  --T 3.0 \
  --Nd 8.0 \
  --sigma 0.12 \
  --Bmax 10.0
```

For MLP runs:

```bash
python diag_test_mlp.py --logdir sdde_MLP_runs/<run_name>
python recon_test_mlp.py \
  --logdir sdde_MLP_runs/<run_name> \
  --tau 2.0 \
  --T 3.0 \
  --Nd 8.0 \
  --sigma 0.12 \
  --Bmax 10.0 \
  --Aj 0.05  # required only for a Jupiter run
```

For FNO runs:

```bash
python diag_test_fno.py --logdir sdde_FNO_runs/<run_name>
python recon_test_fno.py \
  --logdir sdde_FNO_runs/<run_name> \
  --tau 2.0 \
  --T 3.0 \
  --Nd 8.0 \
  --sigma 0.02 \
  --Bmax 10.0
```

`recon_test_fno.py` reads the saved reconstruction domain automatically. It
plots and scores time-domain trajectories for legacy/time runs and log-FFT
amplitudes for spectral-loss runs.

The diagonal tests evaluate how well the encoder recovers the physical parameters from synthetic samples and write a plot into `<run>/diagnostics/`.

CLI options:

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `--logdir <path>` | yes | - | Run directory containing checkpoints and `hyper_parameters.json`. |
| `--nsamples <int>` | no | `1000` | Number of synthetic samples to generate for the diagonal test. |
| `--batch <int>` | no | `64` | Batch size used for encoder forward passes. |
| `--seed <int>` | no | `1234` | Random seed used to generate the synthetic test samples. |
| `--outdir <path>` | no | `<run>/diagnostics` | Directory where the diagnostic plot is written. |
| `--best` | no | enabled by default | Use the latest `model_best_ckpt-*` checkpoint. Mutually exclusive with `--last`. |
| `--last` | no | disabled | Use the latest regular `model_ckpt-*` checkpoint. Mutually exclusive with `--best`. |

Examples:

```bash
# use the latest "best" checkpoint (default)
python diag_test_enca.py --logdir sdde_ENCA_runs/<run_name> --best

# use the latest regular checkpoint instead
python diag_test_mlp.py --logdir sdde_MLP_runs/<run_name> --last

# change the number of synthetic test samples
python diag_test_enca.py --logdir sdde_ENCA_runs/<run_name> --nsamples 2000

# write outputs somewhere else and change the encoder batch size
python diag_test_mlp.py --logdir sdde_MLP_runs/<run_name> --outdir diagnostics/manual --batch 128
```

The reconstruction tests fix one parameter setting, sample one or more noise realizations, reconstruct the relevant representation, and save a comparison plot into `<run>/diagnostics/`. ENCA reconstruction compares time-domain trajectories. MLP reconstruction compares Hann-windowed log FFT amplitude vectors.

CLI options:

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `--logdir <path>` | yes | - | Run directory containing checkpoints and `hyper_parameters.json`. |
| `--tau <float>` | yes | - | Fixed `tau` value for the generated reconstruction sample. |
| `--T <float>` | yes | - | Fixed `T` value for the generated reconstruction sample. |
| `--Nd <float>` | yes | - | Fixed `Nd` value for the generated reconstruction sample. |
| `--sigma <float>` | yes | - | Fixed `sigma` value for the generated reconstruction sample. |
| `--Bmax <float>` | yes | - | Fixed `Bmax` value for the generated reconstruction sample. |
| `--Aj <float>` | Jupiter only | - | Fixed Jupiter modulation amplitude. Rejected for original-model runs. |
| `--seed <int>` | no | `1234` | First random seed used for sampled driving noise. With `--nseeds`, seeds are used consecutively from `seed` to `seed + nseeds - 1`. |
| `--nseeds <int>` | no | `1` | Number of noise realizations to aggregate for the same fixed parameters. Must be at least `1`. |
| `--outdir <path>` | no | `<run>/diagnostics` | Directory where the reconstruction comparison plot is written. |
| `--best` | no | enabled by default | Use the latest `model_best_ckpt-*` checkpoint. Mutually exclusive with `--last`. |
| `--last` | no | disabled | Use the latest regular `model_ckpt-*` checkpoint. Mutually exclusive with `--best`. |

Examples:

```bash
# aggregate several noise realizations for the same parameters
python recon_test_enca.py \
  --logdir sdde_ENCA_runs/<run_name> \
  --tau 2.0 --T 3.0 --Nd 8.0 --sigma 0.12 --Bmax 10.0 \
  --nseeds 8

# evaluate the last checkpoint instead of the best one
python recon_test_mlp.py \
  --logdir sdde_MLP_runs/<run_name> \
  --tau 2.0 --T 3.0 --Nd 8.0 --sigma 0.12 --Bmax 10.0 \
  --last

# choose the starting random seed and output directory
python recon_test_enca.py \
  --logdir sdde_ENCA_runs/<run_name> \
  --tau 2.0 --T 3.0 --Nd 8.0 --sigma 0.12 --Bmax 10.0 \
  --seed 2024 \
  --outdir diagnostics/manual
```

Notes:

- The ENCA scripts expect `representation_mode = "time"` and the MLP scripts expect `representation_mode = "fourier_amplitude"`. Missing `representation_mode` is treated as `"time"` by the ENCA tools for backward compatibility with older ENCA runs.
- `recon_test_enca.py` and `recon_test_mlp.py` check that the supplied parameter values stay within the saved prior limits.
- As with SDDE training, Julia must be available because both scripts initialize `juliacall` before importing TensorFlow.

## MLP Prior Distribution In Latent Space

[`prior_dist_latent_variables.py`](prior_dist_latent_variables.py) visualizes how simulated observations drawn from the parameter prior are distributed in the latent space of a trained Fourier-amplitude MLP. It supports original runs with five regressors and Jupiter runs with six regressors, followed by any number of free statistics. For every prior draw, the script:

1. samples the model's physical parameters using the limits in the run's `hyper_parameters.json` (including `Aj` for Jupiter);
2. simulates an SDDE observation;
3. applies the same Hann-windowed log FFT-amplitude transformation used during training; and
4. passes the transformed observation through the restored encoder without dropping any latent coordinate.

For any Jupiter run, `z1` through `z6` are the supervised estimates
`(tau, T, Nd, sigma, Bmax, Aj)` and `z7` onward are free statistics. The
generated overview shows every `(z_i, z_j, z_last)` combination. Parameters,
including the continuous delay `T`, are sampled uniformly from their saved
priors and simulated by the same canonical solver used in SABC.

The default model is `20260611_mlp_z6_1`, with 1000 prior samples and seed 1234:

```bash
python prior_dist_latent_variables.py
```

A bare model name is looked up under `sdde_MLP_runs/`. A full or relative run-directory path can also be supplied:

```bash
# 2000 samples from the default run, with a reproducible custom seed
python prior_dist_latent_variables.py 20260611_mlp_z6_1 \
  --nsamples 2000 \
  --seed 42

# equivalently, pass the run directory
python prior_dist_latent_variables.py \
  sdde_MLP_runs/20260611_mlp_z6_1 \
  --nsamples 2000

# use the latest regular checkpoint instead of the latest best checkpoint
python prior_dist_latent_variables.py 20260611_mlp_z6_1 --last

# also save a rotatable, zoomable HTML version of all ten plots
python prior_dist_latent_variables.py 20260611_mlp_z6_1 --interactive

# encode the observed SILSO sunspot record and mark it in both plot types
python prior_dist_latent_variables.py 20260611_mlp_z6_1 \
  --obs-sn-data /path/to/silso_SN_y_202601.csv \
  --interactive
```

CLI options:

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `model` | no | `20260611_mlp_z6_1` | Bare run name or path to an MLP run directory. |
| `--nsamples <int>` | no | `1000` | Number of prior observations to simulate and encode. |
| `--seed <int>` | no | `1234` | Seed controlling the parameter draws and driving noise. |
| `--batch <int>` | no | `128` | Batch size for encoder inference. |
| `--outdir <path>` | no | `<run>/diagnostics/prior_latent` | Directory for the plot and numerical output. |
| `--dpi <int>` | no | `180` | Resolution of the saved PNG. |
| `--interactive` | no | disabled | Also save a self-contained interactive Plotly HTML overview. |
| `--obs-sn-data <path>` | no | disabled | Encode the full two-column SILSO yearly CSV using the SDDEpy `[49:-6]` crop and mark obsSN in the plots. |
| `--best` | no | enabled by default | Use the latest `model_best_ckpt-*` checkpoint. Mutually exclusive with `--last`. |
| `--last` | no | disabled | Use the latest regular `model_ckpt-*` checkpoint. Mutually exclusive with `--best`. |

Each run writes a PNG and an NPZ file whose names record the checkpoint step, number of samples, and seed:

```text
prior_latent_model_best_ckpt_step550000_n1000_seed1234_z6_triplets.png
prior_latent_model_best_ckpt_step550000_n1000_seed1234.npz
```

The PNG contains the ten 3D latent-space panels. The compressed NumPy file preserves the underlying values, allowing further analysis without rerunning the simulator. Load it with:

With `--interactive`, the script additionally writes:

```text
prior_latent_model_best_ckpt_step550000_n1000_seed1234_z6_triplets_interactive.html
```

Open this self-contained HTML file in a web browser to rotate, pan, and zoom each 3D panel. The interactive plots omit the prior prism for a cleaner view and render the prior cloud with low opacity, while outliers remain red. Hovering over a point shows every latent coordinate and every sampled physical parameter. The Plotly toolbar can also reset the camera or save the current view as an image.

When `--obs-sn-data` is supplied, the script follows the obsSN loader in SDDEpy and selects `data[49:-6]` from the full SILSO file. For `silso_SN_y_202601.csv`, this produces the 271 yearly values from 1749.5 through 2019.5 expected by the model. The encoded observation is shown above the translucent prior-sample cloud as a large magenta star in the PNG and as a magenta diamond labelled `obsSN` in the interactive HTML. Its numerical inputs and latent coordinates are also added to the NPZ as `observed_years`, `observed_values`, `observed_latent`, and `observed_data_path`.

```python
from pathlib import Path

import numpy as np

output_dir = Path(
    "sdde_MLP_runs/20260611_mlp_z6_1/diagnostics/prior_latent"
)
npz_path = sorted(output_dir.glob("prior_latent_*.npz"))[-1]

with np.load(npz_path, allow_pickle=False) as data:
    parameters = data["parameters"]       # shape: (nsamples, 5)
    latent = data["latent"]               # shape: (nsamples, 6)
    parameter_names = data["parameter_names"]
    checkpoint = str(data["checkpoint"])
    seed = int(data["seed"])
    observed_latent = data.get("observed_latent")  # present with --obs-sn-data

tau, T, Nd, sigma, Bmax = parameters.T
z1, z2, z3, z4, z5, z6 = latent.T

print(parameter_names)
print(checkpoint, seed)
```

For example, a new 3D projection can be made directly from the saved arrays:

```python
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
points = ax.scatter(z1, z3, z6, c=z6, s=8, alpha=0.6)
ax.set(xlabel="z1 (tau)", ylabel="z3 (Nd)", zlabel="z6 (free)")
fig.colorbar(points, ax=ax, label="z6")
fig.savefig("z1_z3_z6.png", dpi=180, bbox_inches="tight")
```

## Julia Requirement

The SDDE training scripts (`train_ENCA_model3.py` and `train_MLP_model3.py`) initialize Julia via `juliacall` before importing TensorFlow. To run these scripts successfully, make sure Julia is installed and available on your system. On first use, `juliacall` may also download or initialize Julia-related components.

If you only need the original model 1 and model 2 experiments, the Julia dependency is not required.

On cluster systems, it can also be necessary to point Julia to a writable depot path before launching the training:

```bash
export JULIA_DEPOT_PATH=/cfs/earth/scratch/ulzg/.julia
```

## Prerequisites   

The current checked-in environment uses:

- Python 3.10
- TensorFlow 2.x (`tensorflow>=2.14`)
- NumPy, SciPy, pandas, matplotlib, and h5py

Please report any issues if you come across bugs or platform-specific dependency problems.

## Note

For the solar-dynamo / SDDE setting, the currently supported workflows in this repository are the time-domain ENCA model and the Fourier-amplitude MLP autoencoder. An INCA model for the SDDE case has not been implemented as part of the maintained workflow at this stage. That may be added in the future, but there is no firm commitment yet.

## Citation  

If you use any content of this repository, please use the following bibtex: 
```
@article{albert2022learning,
  title={Learning Summary Statistics for Bayesian Inference with Autoencoders},
  author={Albert, Carlo and Ulzega, Simone and Perez-Cruz, Fernando and Ozdemir, Firat and Mira, Antonietta},
  journal={SciPost Physics Core},
  volume={5},
  pages={043},
  year={2022}
}
```
