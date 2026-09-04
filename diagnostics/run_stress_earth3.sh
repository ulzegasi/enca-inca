#!/bin/bash
#SBATCH --job-name=sdde_stress
#SBATCH --output=/cfs/earth/scratch/ulzg/enca-inca/txtout/stress.%j.info
#SBATCH --error=/cfs/earth/scratch/ulzg/enca-inca/txtout/stress.%j.info
#SBATCH --chdir=/cfs/earth/scratch/ulzg/enca-inca
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --constraint=rhel8
#SBATCH --no-requeue

# No GPU allocation. Submit from a shell without encainca already activated.
set -e
. /cfs/earth/scratch/ulzg/enca-inca/load_encainca_env.sh
set -u
set -o pipefail
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
export JULIA_DEPOT_PATH=/cfs/earth/scratch/ulzg/.julia
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# Unique output directory: never reuse or write a training run directory.
DIAG_DIR=$(mktemp -d "/cfs/earth/scratch/ulzg/enca-inca/txtout/sdde_stress.${SLURM_JOB_ID}.XXXXXX")
printf 'Diagnostic directory: %s\nHost: %s\n' "$DIAG_DIR" "$(hostname)"
cp diagnostics/stress_sdde.py "$DIAG_DIR/stress_sdde.snapshot.py"
git rev-parse HEAD > "$DIAG_DIR/enca-inca.commit"

# Each srun starts a fresh process. Continue to the control if 16 threads crash.
# A hung case is terminated after 65 minutes; this bounds the combined job time.
DIAG_RESULT=0
for DIAG_THREADS in 16 1; do
    export JULIA_NUM_THREADS="$DIAG_THREADS"
    export PYTHON_JULIACALL_THREADS="$DIAG_THREADS"
    DIAG_OPTIONS=()
    if [[ "${DIAG_WITH_TENSORFLOW:-0}" == "1" ]]; then
        DIAG_OPTIONS+=(--with-tensorflow)
    fi
    printf 'Starting case: %s threads\n' "$DIAG_THREADS"
    if srun --export=ALL --cpu-bind=cores --time=01:05:00 \
        python -u diagnostics/stress_sdde.py --threads "$DIAG_THREADS" \
        --seconds "${DIAG_SECONDS:-3600}" --batches "${DIAG_BATCHES:-25000}" \
        "${DIAG_OPTIONS[@]}" > "$DIAG_DIR/threads_${DIAG_THREADS}.log" 2>&1; then
        DIAG_STATUS=0
    else
        DIAG_STATUS=$?
        DIAG_RESULT=1
    fi
    printf 'threads=%s exit_status=%s\n' "$DIAG_THREADS" "$DIAG_STATUS" | tee -a "$DIAG_DIR/status.txt"
done
printf 'Finished diagnostic. Logs: %s\n' "$DIAG_DIR"
exit "$DIAG_RESULT"
