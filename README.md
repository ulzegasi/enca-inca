# ENCA-INCA

[![launch - renku](https://img.shields.io/badge/launch-renku-2ea44f?logo=python)](https://renkulab.io/projects/bistom/enca-inca/sessions/new?autostart=1)

## Learning Summary Statistics for Bayesian Inference with Autoencoders 

The code here implements the [work](https://arxiv.org/abs/2201.12059) proposed by Albert, C., Ulzega, S., Ozdemir, F., Perez-Cruz, F. and Mira, A. (2022). Learning Summary Statistics for Bayesian Inference with Autoencoders. arXiv preprint arXiv:2201.12059.

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

## Julia Requirement

The SDDE training scripts (`train_ENCA_model3.py` and `train_INCA_model3.py`) initialize Julia via `juliacall` before importing TensorFlow. To run these scripts successfully, make sure Julia is installed and available on your system. On first use, `juliacall` may also download or initialize Julia-related components.

If you only need the original model 1 and model 2 experiments, the Julia dependency is not required.

## Interactive Environment  

The project can also be run on [renku environment](https://renkulab.io/projects/bistom/enca-inca/sessions/new?autostart=1). 
This allows skipping most of the local environment setup needed to test the repository.

## Prerequisites   

The current checked-in environment uses:

- Python 3.10
- TensorFlow 2.x (`tensorflow>=2.14`)
- NumPy, SciPy, pandas, matplotlib, and h5py

Please report any issues if you come across bugs or platform-specific dependency problems.

## Citation  

If you use any content of this repository, please use the following bibtex: 
```
@article{albert2022learning,
  title={Learning Summary Statistics for Bayesian Inference with Autoencoders},
  author={Albert, Carlo and Ulzega, Simone and Ozdemir, Firat and Perez-Cruz, Fernando and Mira, Antonietta},
  journal={arXiv preprint arXiv:2201.12059},
  year={2022}
}
```
