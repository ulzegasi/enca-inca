#!/bin/bash
#
#SBATCH --job-name=encaf2cnn
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

# Training settings: edit these here, then submit with plain sbatch.
export MODEL="jupiter"       # original or jupiter
export INFER_PHASE="true"    # true requires jupiter; false marginalizes phase
export NDIMS_LATENT=""       # empty = automatic (5/6/8); set a number for extra free coordinates

case "$INFER_PHASE" in
  [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]) export INFER_PHASE=true ;;
  [Ff][Aa][Ll][Ss][Ee]|0|[Nn][Oo]) export INFER_PHASE=false ;;
  *) echo "INFER_PHASE must be true or false." >&2; exit 2 ;;
esac
case "$MODEL" in
  original) minimum_latent=5; model_label="" ;;
  jupiter) minimum_latent=6; model_label="_jupiter" ;;
  *) echo "MODEL must be original or jupiter." >&2; exit 2 ;;
esac
phase_label=""
if [[ "$INFER_PHASE" == "true" ]]; then
  if [[ "$MODEL" != "jupiter" ]]; then
    echo "INFER_PHASE=true requires MODEL=jupiter." >&2
    exit 2
  fi
  minimum_latent=8
  phase_label="_phase"
fi
export NDIMS_LATENT="${NDIMS_LATENT:-$minimum_latent}"
if [[ ! "$NDIMS_LATENT" =~ ^[0-9]+$ ]] || [[ "$NDIMS_LATENT" -lt "$minimum_latent" ]]; then
  echo "NDIMS_LATENT must be an integer >= $minimum_latent." >&2
  exit 2
fi
LATENT_TAG="$NDIMS_LATENT"

export JULIA_DEPOT_PATH=/cfs/earth/scratch/ulzg/.julia
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Required by JuliaCall when Julia worker threads execute inside Python.
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
mkdir -p "$JULIA_DEPOT_PATH"

mkdir -p "$TMPDIR"
mkdir -p /cfs/earth/scratch/ulzg/enca-inca/txtout
mkdir -p /cfs/earth/scratch/ulzg/enca-inca/sdde_ENCAFourier2CNN_runs

# New experiments must use a fresh run directory. Replace the automatic stamp
# only when continuing a checkpoint created with the same model and backend.
RUNSTAMP=$(date +%Y%m%d)
export ENCA_FOURIER2_CNN_LOGDIR="${ENCA_FOURIER2_CNN_LOGDIR:-/cfs/earth/scratch/ulzg/enca-inca/sdde_ENCAFourier2CNN_runs/${RUNSTAMP}_encafourier2cnn${model_label}${phase_label}_z${LATENT_TAG}}"
mkdir -p "$ENCA_FOURIER2_CNN_LOGDIR"

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
echo "INFER_PHASE=$INFER_PHASE (NDIMS_LATENT=$NDIMS_LATENT)"
echo "ENCA_FOURIER2_CNN_LOGDIR=$ENCA_FOURIER2_CNN_LOGDIR"
python -c "import sdde_model; print('Canonical SDDE model:', sdde_model.__file__)"
nvidia-smi || true

# ==============================
# Run
# ==============================
srun --export=ALL,MODEL="$MODEL",INFER_PHASE="$INFER_PHASE",NDIMS_LATENT="$NDIMS_LATENT",ENCA_FOURIER2_CNN_LOGDIR="$ENCA_FOURIER2_CNN_LOGDIR" --cpu-bind=cores \
    python train_ENCAfft2CNN_model3.py
