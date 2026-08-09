#!/bin/bash
#
#SBATCH --job-name=encafcnn_z6
#SBATCH --output=/cfs/earth/scratch/ulzg/enca-inca/txtout/info.%x.%j.%N.info
#SBATCH --error=/cfs/earth/scratch/ulzg/enca-inca/txtout/info.%x.%j.%N.info
#SBATCH --chdir=/cfs/earth/scratch/ulzg/enca-inca
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --partition=earth-5
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

module load cuda/11.6.2

export JULIA_DEPOT_PATH=/cfs/earth/scratch/ulzg/.julia
mkdir -p "$JULIA_DEPOT_PATH"

mkdir -p "$TMPDIR"
mkdir -p /cfs/earth/scratch/ulzg/enca-inca/txtout
mkdir -p /cfs/earth/scratch/ulzg/enca-inca/sdde_ENCAFourierCNN_runs

# Continue the existing 2026-08-07 run. The trainer restores the latest
# checkpoint in this directory and advances its absolute target to 1.6M steps.
# RUNSTAMP=$(date +%Y%m%d)  # Fresh run
RUNSTAMP=20260807
export ENCA_FOURIER_CNN_LOGDIR=/cfs/earth/scratch/ulzg/enca-inca/sdde_ENCAFourierCNN_runs/${RUNSTAMP}_encafouriercnn_z6
mkdir -p "$ENCA_FOURIER_CNN_LOGDIR"

# This launcher is intentionally a continuation job. Refuse to start a new
# model silently if the run directory or checkpoint prefix is mistyped.
if ! compgen -G "$ENCA_FOURIER_CNN_LOGDIR/model_ckpt-*.index" > /dev/null; then
    echo "ERROR: no model_ckpt-*.index found in $ENCA_FOURIER_CNN_LOGDIR" >&2
    exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

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
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ENCA_FOURIER_CNN_LOGDIR=$ENCA_FOURIER_CNN_LOGDIR"
nvidia-smi || true

# ==============================
# Run
# ==============================
srun --export=ALL,ENCA_FOURIER_CNN_LOGDIR="$ENCA_FOURIER_CNN_LOGDIR" --cpu-bind=cores python train_ENCAFourierCNN_model3.py
