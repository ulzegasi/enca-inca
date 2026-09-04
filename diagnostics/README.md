# Isolated SDDE crash diagnostic

This CPU-only test targets the intermittent Julia segmentation fault in job
899038. It does not train a network, load/save checkpoints, edit a model, or
change priors, solver settings, or training launchers. All files in this update
are additions under `diagnostics/`.

## Cluster use

From `/cfs/earth/scratch/ulzg/enca-inca`, on `solar_dynamo`:

```bash
git pull --ff-only
mkdir -p txtout
sbatch diagnostics/run_stress_earth3.sh
```

The pull is intended to add only diagnostic files on top of commit `08524f4`.
If Git reports conflicts or unexpected existing-file changes, stop: do not
restore/stash/reset the live training configuration to force the pull.
No SDDE-model pull or package update is required. Submit from a shell without
`encainca` already activated, following the existing cluster launcher convention.

The job requests `earth-3`, 16 CPUs, 64 GB RAM, no GPU, and a 2.5-hour wall limit.
It runs separate 16-thread and one-thread processes sequentially. Each case stops
after 25000 batches or one hour of simulation, whichever comes first. Initialization
is additional; Slurm caps each case at 65 minutes, including a hang. The control
still runs if the first case crashes. Expect around two hours plus queue time,
possibly less. The slower control may complete fewer batches.

Default workload matches the MLP Jupiter generator: seed 1822, batch 300,
warmup 200, observation length 271, dt 0.1, six physical parameters and a random
Jupiter phase. Both cases regenerate the same seeded input sequence. TensorFlow
is not loaded by default. Input generation, shape/finiteness checks, hashing,
and frequent logging add overhead; timings are diagnostic, not training benchmarks.

## Isolation and interpretation

The driver initializes JuliaCall directly against the installed SDDE-model Julia
project and supplies the initialized runtime to the existing lazy simulator bridge.
It skips the normal bootstrap's `Pkg.instantiate()` so it will fail on missing
dependencies rather than repair a shared environment. Simulation functions remain
unchanged. Julia may write normal compilation caches; no package install/update
is requested. The import-only TensorFlow case, if requested later, is not a
reproduction of GPU training or its allocation patterns.

The parent log is `txtout/stress.JOBID.info`. It prints a unique diagnostic folder
containing a log per case, exit statuses, a driver snapshot and repository commit.
Each batch has an unbuffered start marker; the last unmatched marker identifies
the in-flight batch on a crash. Progress includes process peak RSS in MiB and a
cumulative hash of returned observations, parameters and noise. Compare hashes
only at equal completed batch counts. Equal hashes support deterministic output
agreement for that prefix, not general scientific validity. A differing hash
warrants investigation, not an automatic claim of a scientific discrepancy.

An exit status of zero means no failure was observed before the limit. A crash
only in the 16-thread case strengthens, but does not prove, a threading hypothesis.
A pass in both cases does not rule out an intermittent bug. CPU-node hardware
differs from the GPU node where training failed.

Optional follow-up: same experiment with TensorFlow imported after Julia, still
CPU-only and without training:

```bash
sbatch --export=ALL,DIAG_WITH_TENSORFLOW=1 diagnostics/run_stress_earth3.sh
```

For a short launcher smoke test, set `DIAG_SECONDS=30` and `DIAG_BATCHES=2` with
`sbatch --export=ALL,DIAG_SECONDS=30,DIAG_BATCHES=2 ...`. Each batch is allowed to
finish before the Python time limit is checked; the Slurm case limit is the hard cap.
