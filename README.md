# ENCA-INCA
## Autoencoder-based Summary Statistics for Approximate Bayesian Computation  

The code here implements the work proposed by Anonymous et al. "Autoencoder-based Summary Statistics for Approximate Bayesian Computation", 2021.

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

The project can also be run on [renku environment](https://renkulab.io/projects/bistom/enca-inca). 
This allows to skip the step requiring to set up an environment to test the repo. 

## Prerequisites   

Code is tested for Tensorflow v2.4.1. However preliminary experiments suggest it should also work on TF v2.2.  
Please report any issues if you come across bugs.

## Citation  

If you use any content of this repository, please refer to the following paper: 

#TBD
