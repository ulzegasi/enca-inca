#!/bin/bash
#
#SBATCH --job-name=FNOz6m64fftSlow
#SBATCH --output=/cfs/earth/scratch/ulzg/enca-inca/txtout/info.%x.%j.%N.info
#SBATCH --error=/cfs/earth/scratch/ulzg/enca-inca/txtout/info.%x.%j.%N.info
#SBATCH --chdir=/cfs/earth/scratch/ulzg/enca-inca
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=04-00:00:00
#SBATCH --partition=earth-4
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
mkdir -p /cfs/earth/scratch/ulzg/enca-inca/sdde_FNO_runs

# For the first launch, keep the automatic date stamp.
# For a continuation run, replace this with the original run date.
RUNSTAMP=$(date +%Y%m%d)
export FNO_LOGDIR=/cfs/earth/scratch/ulzg/enca-inca/sdde_FNO_runs/${RUNSTAMP}_fno_z6_m64_fourier_slow
export FNO_RECON_DOMAIN=fourier_log_amplitude
mkdir -p "$FNO_LOGDIR"

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
echo "FNO_LOGDIR=$FNO_LOGDIR"
echo "FNO_RECON_DOMAIN=$FNO_RECON_DOMAIN"
nvidia-smi || true

# ==============================
# Run
# ==============================
srun --export=ALL,FNO_LOGDIR="$FNO_LOGDIR",FNO_RECON_DOMAIN="$FNO_RECON_DOMAIN" --cpu-bind=cores python train_FNO_model3.py
