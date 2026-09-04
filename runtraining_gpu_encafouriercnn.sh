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

# SDDE variant. The latent width remains hard-coded in
# train_ENCAFourierCNN_model3.py; keep LATENT_TAG synchronized with it for the
# run-directory name only.
export MODEL="jupiter"
LATENT_TAG=6

export JULIA_DEPOT_PATH=/cfs/earth/scratch/ulzg/.julia
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Required by JuliaCall when Julia worker threads execute inside Python.
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
mkdir -p "$JULIA_DEPOT_PATH"

mkdir -p "$TMPDIR"
mkdir -p /cfs/earth/scratch/ulzg/enca-inca/txtout
mkdir -p /cfs/earth/scratch/ulzg/enca-inca/sdde_ENCAFourierCNN_runs

# Jupiter training must use a fresh run directory. Replace the automatic stamp
# only when continuing a checkpoint created with the same model and backend.
RUNSTAMP=$(date +%Y%m%d)
export ENCA_FOURIER_CNN_LOGDIR=/cfs/earth/scratch/ulzg/enca-inca/sdde_ENCAFourierCNN_runs/${RUNSTAMP}_encafouriercnn_${MODEL}_z${LATENT_TAG}
mkdir -p "$ENCA_FOURIER_CNN_LOGDIR"

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
echo "Julia threads: $JULIA_NUM_THREADS"
echo "Julia used: $(command -v julia)"
julia -v || true
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL=$MODEL"
echo "ENCA_FOURIER_CNN_LOGDIR=$ENCA_FOURIER_CNN_LOGDIR"
python -c "import sdde_model; print('Canonical SDDE model:', sdde_model.__file__)"
nvidia-smi || true

# ==============================
# Run
# ==============================
srun --export=ALL,MODEL="$MODEL",ENCA_FOURIER_CNN_LOGDIR="$ENCA_FOURIER_CNN_LOGDIR" --cpu-bind=cores \
    python train_ENCAFourierCNN_model3.py
