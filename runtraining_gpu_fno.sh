#!/bin/bash
#
#SBATCH --job-name=fno3z10m32
#SBATCH --output=/cfs/earth/scratch/ulzg/enca-inca/txtout/info.%x.%j.%N.info
#SBATCH --error=/cfs/earth/scratch/ulzg/enca-inca/txtout/info.%x.%j.%N.info
#SBATCH --chdir=/cfs/earth/scratch/ulzg/enca-inca
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
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

# ==============================
# Editable variables
# ==============================
FNO_WIDTH="${FNO_WIDTH:-64}"
FNO_MODES="${FNO_MODES:-32}"
FNO_LAYERS="${FNO_LAYERS:-4}"

# For the first launch, keep the automatic date stamp.
# For a continuation run, replace this with the original run date, e.g. RUNSTAMP=20260615.
RUNSTAMP=$(date +%Y%m%d)
export FNO_LOGDIR="${FNO_LOGDIR:-/cfs/earth/scratch/ulzg/enca-inca/sdde_FNO_runs/${RUNSTAMP}_fno_z10_m${FNO_MODES}_1}"
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
echo "FNO_WIDTH=$FNO_WIDTH"
echo "FNO_MODES=$FNO_MODES"
echo "FNO_LAYERS=$FNO_LAYERS"
nvidia-smi || true

# ==============================
# Run
# ==============================
srun --export=ALL,FNO_LOGDIR="$FNO_LOGDIR",FNO_WIDTH="$FNO_WIDTH",FNO_MODES="$FNO_MODES",FNO_LAYERS="$FNO_LAYERS" --cpu-bind=cores python train_FNO_model3.py
