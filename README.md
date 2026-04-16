# ENCA-INCA

[![launch - renku](https://img.shields.io/badge/launch-renku-2ea44f?logo=python)](https://renkulab.io/projects/bistom/enca-inca/sessions/new?autostart=1)

## Learning Summary Statistics for Bayesian Inference with Autoencoders 

The code here implements the published work by Carlo Albert, Simone Ulzega, Fernando Perez-Cruz, Firat Ozdemir, and Antonietta Mira: *Learning summary statistics for Bayesian inference with Autoencoders*, SciPost Phys. Core 5, 043 (2022).

## Contents 
The repository contains the code needed to train the proposed models _explicit noise conditional autoencoder_ (ENCA) and _implicit noise conditional autoencoder_ (INCA) for several simulator settings:

- `train_ENCA_model1.py`, `train_INCA_model1.py`: original model 1 experiments
- `train_ENCA_model2.py`, `train_INCA_model2.py`: original model 2 experiments
- `train_ENCA_model3.py`, `train_INCA_model3.py`: solar-dynamo / SDDE experiments using the Julia-backed simulator

Supporting code lives in `src/`, and local training outputs are written to `sdde_ENCA_runs/` when running the newer SDDE scripts.

## Demo   
The repository can be set up on a clean environment by creating a conda environment:

```bash
conda env create -f environment.yml
conda activate encainca
```

This environment currently targets Python 3.10 and TensorFlow 2.x (`tensorflow>=2.14` in `environment.yml`).

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
python train_INCA_model3.py
```

Before launching one of these training scripts, manually review the `ExpSetup` class in [train_ENCA_model3.py](/Users/ulzg/switchdrive/ZHAW_BISTOM/RENKU/enca-inca/train_ENCA_model3.py) and set the key run parameters there. In particular:

```python
self.ndims_latent = 10

self.Tobs = 271  # C14 dataset: 929, obsSN dataset: 271

self.batch_size = 64
self.max_training_steps = int(2500)  # full run example: int(3e6)
self.freq_log = 100
```

## SDDE ENCA Loss Configuration

The SDDE ENCA training script [train_ENCA_model3.py](/Users/ulzg/switchdrive/ZHAW_BISTOM/RENKU/enca-inca/train_ENCA_model3.py) now supports two loss configurations. The active one is selected in `ExpSetup` via:

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

- compute one RMS amplitude scale per sample from the true signal `x`
- clamp that scale from below with `recon_scale_eps`
- normalize reconstruction error by that per-sample scale
- compute plain MSE on the normalized signal

In formula form:

```python
scale_x = max(sqrt(mean(x^2 over time)), recon_scale_eps)
loss_reconstruction = mean(((x - x_pred) / scale_x)^2)
```

Parameter regression loss:

- use only the first 5 latent coordinates for parameter supervision
- normalize each parameter error by its prior width
- compute plain MSE on those normalized parameter errors

In formula form:

```python
loss_regress_params = mean(((params - params_pred[:, :5]) / prior_widths)^2)
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
- `RMSE_x_ch_1`
- `RMSE_z_ch_1` to `RMSE_z_ch_5`
- `theta/...` parameter min/max ranges
- `lr_schedule`

In addition, the lambda weights are logged as:

- `loss_weight/lambda_recon`
- `loss_weight/lambda_reg`

The per-component loss names depend on the selected mode.

If `loss_mode = "legacy_chisq"`, TensorBoard shows:

- `ChiSquare_x_ch_1`
- `ChiSquare_z_1`
- `ChiSquare_z_2`
- `ChiSquare_z_3`
- `ChiSquare_z_4`
- `ChiSquare_z_5`

If `loss_mode = "balanced_mse"`, TensorBoard shows:

- `NormMSE_x_ch_1`
- `NormMSE_z_1`
- `NormMSE_z_2`
- `NormMSE_z_3`
- `NormMSE_z_4`
- `NormMSE_z_5`

So the high-level dashboard remains similar, but the detailed per-component loss tags change when you switch from the legacy chi-square losses to the normalized MSE losses.

## Diagnostics And Reconstruction Tests

After training an SDDE ENCA model, you can run the helper scripts `diag_test.py` and `recon_test.py` against a saved run directory. These scripts expect a run folder containing checkpoints and `hyper_parameters.json`, for example under `sdde_ENCA_runs/<run_name>/`.

The diagonal test evaluates how well the encoder recovers the physical parameters from synthetic samples and writes a plot into `<run>/diagnostics/`:

```bash
python diag_test.py --logdir sdde_ENCA_runs/<run_name>
```

Useful options:

```bash
# use the latest "best" checkpoint (default)
python diag_test.py --logdir sdde_ENCA_runs/<run_name> --best

# use the latest regular checkpoint instead
python diag_test.py --logdir sdde_ENCA_runs/<run_name> --last

# change the number of synthetic test samples
python diag_test.py --logdir sdde_ENCA_runs/<run_name> --nsamples 2000
```

The reconstruction test fixes one parameter setting, samples one or more noise realizations, reconstructs the resulting trajectories, and saves a comparison plot into `<run>/diagnostics/`:

```bash
python recon_test.py \
  --logdir sdde_ENCA_runs/<run_name> \
  --tau 2.0 \
  --T 3.0 \
  --Nd 8.0 \
  --sigma 0.12 \
  --Bmax 10.0
```

Useful options:

```bash
# aggregate several noise realizations for the same parameters
python recon_test.py \
  --logdir sdde_ENCA_runs/<run_name> \
  --tau 2.0 --T 3.0 --Nd 8.0 --sigma 0.12 --Bmax 10.0 \
  --nseeds 8

# evaluate the last checkpoint instead of the best one
python recon_test.py \
  --logdir sdde_ENCA_runs/<run_name> \
  --tau 2.0 --T 3.0 --Nd 8.0 --sigma 0.12 --Bmax 10.0 \
  --last
```

Notes:

- Both scripts are intended for SDDE / solar-dynamo ENCA runs and rely on the saved hyperparameters from training.
- `recon_test.py` checks that the supplied parameter values stay within the saved prior limits.
- As with SDDE training, Julia must be available because both scripts initialize `juliacall` before importing TensorFlow.

## Julia Requirement

The SDDE training scripts (`train_ENCA_model3.py` and `train_INCA_model3.py`) initialize Julia via `juliacall` before importing TensorFlow. To run these scripts successfully, make sure Julia is installed and available on your system. On first use, `juliacall` may also download or initialize Julia-related components.

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

For the solar-dynamo / SDDE setting, the currently supported workflow in this repository is the ENCA model. An INCA model for the SDDE case has not been implemented as part of the maintained workflow at this stage. That may be added in the future, but there is no firm commitment yet.

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
