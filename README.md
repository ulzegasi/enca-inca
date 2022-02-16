# ENCA-INCA

[![launch - renku](https://img.shields.io/badge/launch-renku-2ea44f?logo=python)](https://renkulab.io/projects/bistom/enca-inca/sessions/new?autostart=1)

## Learning Summary Statistics for Bayesian Inference with Autoencoders 

The code here implements the [work](https://arxiv.org/abs/2201.12059) proposed by Albert, C., Ulzega, S., Ozdemir, F., Perez-Cruz, F. and Mira, A. (2022). Learning Summary Statistics for Bayesian Inference with Autoencoders. arXiv preprint arXiv:2201.12059.

## Contents 
The repository contains all the code needed to reproduce the proposed models _explicit noise conditional autoencoder_ (ENCA) and _implicit noise conditional autoencoder_ (INCA) for encoding near-sufficient and highly concentrated summary statistics for the two statistical models experimented in this work. 

## Demo   
The repository can be set up on a clean environment by creating a conda environment by  
  
```bash
conda env create -f environment.yml
conda activate encainca
```

In order to train models ENCA and INCA for statistical model 1 and 2, one can use the provided scripts:
  
```bash
python train_ENCA_model1.py
python train_INCA_model1.py
python train_ENCA_model2.py
python train_INCA_model2.py
``` 

## Interactive Environment  

The project can also be run on [renku environment](https://renkulab.io/projects/bistom/enca-inca/sessions/new?autostart=1). 
This allows to skip the step requiring to set up an environment to test the repo. 

## Prerequisites   

Code is tested for Tensorflow v2.4.1. However preliminary experiments suggest it should also work on TF v2.2.  
Please report any issues if you come across bugs.

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
