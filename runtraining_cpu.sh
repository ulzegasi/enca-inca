#!/bin/bash
#
#SBATCH --job-name=encaCPU300
#SBATCH --output=/cfs/earth/scratch/ulzg/enca-inca/txtout/info.%x.%j.%N.info
#SBATCH --error=/cfs/earth/scratch/ulzg/enca-inca/txtout/info.%x.%j.%N.info
#SBATCH --chdir=/cfs/earth/scratch/ulzg/enca-inca
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=earth-3
#SBATCH --no-requeue
#SBATCH --constraint=rhel8
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=ulzg@zhaw.ch
#SBATCH --mem=64G

# ==============================
# Environment setup
# ==============================
# IMPORTANT: submit a job using this script from a shell where encainca environment is NOT already activated.
# Let this script handle conda activation.

. /cfs/earth/scratch/ulzg/enca-inca/load_encainca_env.sh

export JULIA_DEPOT_PATH=/cfs/earth/scratch/ulzg/.julia
mkdir -p "$JULIA_DEPOT_PATH"

mkdir -p "$TMPDIR"
mkdir -p /cfs/earth/scratch/ulzg/enca-inca/txtout
mkdir -p /cfs/earth/scratch/ulzg/enca-inca/sdde_ENCA_runs

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

export MPLCONFIGDIR=/cfs/earth/scratch/ulzg/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"

# ==============================
# Diagnostics
# ==============================
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python used: $(command -v python)"
python --version
echo "Julia depot: $JULIA_DEPOT_PATH"
echo "Julia used: $(command -v julia)"
julia -v || true
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

# ==============================
# Run
# ==============================
srun --cpu-bind=cores python train_ENCA_model3.py

