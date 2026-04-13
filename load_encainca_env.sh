#!/bin/bash
#
# run this with:
# . /cfs/earth/scratch/ulzg/enca-inca/load_encainca_env.sh
#
# IMPORTANT: use this script in a shell where the conda env is NOT already activated.
# Let this script handle conda activation.

module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load lsfm-init-miniconda/1.0.0
module load openmpi/4.1.4

conda activate encainca

export TMPDIR=/cfs/earth/scratch/ulzg/.tmp
export JULIA_NUM_THREADS=1

