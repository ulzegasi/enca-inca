#!/usr/bin/env python3
"""Exercise the installed canonical batch simulator, without training/checkpoints."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=positive_int, required=True)
    parser.add_argument("--batches", type=positive_int, default=25000)
    parser.add_argument("--seconds", type=positive_int, default=3600,
                        help="time cap for batch loop (initialization is additional)")
    parser.add_argument("--batch-size", type=positive_int, default=300)
    parser.add_argument("--seed", type=int, default=1822)
    parser.add_argument("--report-every", type=positive_int, default=100)
    parser.add_argument("--with-tensorflow", action="store_true",
                        help="import TensorFlow after Julia; no training or GPU work")
    return parser.parse_args(argv)


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def peak_rss_mib():
    # Linux reports KiB; macOS reports bytes. This is the process high-water mark.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def initialize_runtime(args):
    # The existing bootstrap calls Pkg.instantiate(). For this diagnostic, select
    # the SAME installed project explicitly but never instantiate/update packages.
    # An incomplete environment must fail, not repair a shared cluster environment.
    if "juliacall" in sys.modules or "tensorflow" in sys.modules:
        raise RuntimeError("Run each diagnostic case in a fresh Python process")
    import sdde_model

    project = Path(sdde_model.__file__).resolve().parent / "julia_env"
    if not (project / "Manifest.toml").is_file():
        raise RuntimeError(f"Installed pinned Julia Manifest missing: {project}")
    executable = os.environ.get("PYTHON_JULIACALL_EXE") or shutil.which("julia")
    if not executable:
        raise RuntimeError("No Julia executable: use the same environment as training")
    os.environ.update({
        "PYTHON_JULIACALL_EXE": executable,
        "PYTHON_JULIACALL_PROJECT": str(project),
        "JULIA_PROJECT": str(project),
        "JULIA_NUM_THREADS": str(args.threads),
        "PYTHON_JULIACALL_THREADS": str(args.threads),
        "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
        "JULIA_PKG_OFFLINE": "true",
        "CUDA_VISIBLE_DEVICES": "",
    })
    from juliacall import Main as jl
    from sdde_model import solar_dynamo

    # Process-local bridge initialization only. Its normal lazy loader accepts
    # an already initialized `jl`; simulation definitions and solvers are unchanged.
    solar_dynamo.jl = jl
    # Leave solver loading lazy, as in training: optional TensorFlow is imported
    # after JuliaCall startup but before the first generator call loads the solver.
    actual_threads = int(jl.seval("Threads.nthreads(:default)"))
    if actual_threads != args.threads:
        raise RuntimeError(f"Requested {args.threads} threads, got {actual_threads}")
    emit("runtime", julia=str(jl.seval("VERSION")), juliacall=version("juliacall"),
         python=sys.version, threads=actual_threads, signal_handling="yes",
         sdde_model=sdde_model.__file__, project=str(jl.seval("Base.active_project()")),
         manifest_sha256=hashlib.sha256((project / "Manifest.toml").read_bytes()).hexdigest())
    jl.seval('import Pkg; Pkg.status(["PythonCall", "SciMLBase", "StochasticDelayDiffEq", "DiffEqNoiseProcess"])')
    if args.with_tensorflow:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        emit("tensorflow_loaded", version=tf.__version__, note="import-only, no training")
    elif "tensorflow" in sys.modules:
        raise RuntimeError("Simulator-only control unexpectedly imported TensorFlow")


def validate_batch(batch, batch_size, np):
    shapes = ((batch_size, 271, 1), (batch_size, 6), (batch_size, 271, 1))
    if len(batch) != len(shapes):
        raise ValueError("Expected observations, parameters, noise")
    for name, array, shape in zip(("x", "params", "noise"), batch, shapes):
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name}: shape={array.shape}, expected={shape}; check finiteness")


def run_batches(args, generator, np, *, clock=time.monotonic, log=emit):
    started = clock()
    completed = 0
    digest = hashlib.sha256()
    while completed < args.batches and clock() - started < args.seconds:
        # This unbuffered marker identifies the in-flight batch after a native crash.
        # The seed and batch index allow replay. Do NOT catch/retry bad simulations.
        log("batch_start", batch=completed + 1)
        batch = generator.sample_batch(args.batch_size)
        validate_batch(batch, args.batch_size, np)
        for array in batch:
            digest.update(array.tobytes(order="C"))
        completed += 1
        if completed == 1 or completed % args.report_every == 0:
            elapsed = clock() - started
            log("progress", completed=completed, elapsed_s=round(elapsed, 3),
                seconds_per_batch=round(elapsed / completed, 5),
                peak_rss_mib=round(peak_rss_mib(), 1), prefix_sha256=digest.hexdigest())
    log("finished", completed=completed, elapsed_s=round(clock() - started, 3),
        reason="batch_limit" if completed == args.batches else "time_limit",
        peak_rss_mib=round(peak_rss_mib(), 1), prefix_sha256=digest.hexdigest(),
        conclusion="No failure observed in this case; not a long-run stability guarantee")
    return completed


def main(argv=None):
    args = parse_args(argv)
    # Resolve from this file, not from the caller's current directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    emit("configuration", **vars(args), pid=os.getpid())
    initialize_runtime(args)
    import numpy as np
    from src.generators import DataGenerator_SolarDynamo_SDDE_MLP

    generator = DataGenerator_SolarDynamo_SDDE_MLP(
        prng=np.random.RandomState(args.seed), model="jupiter", Twarmup=200,
        Tobs=271, dt=0.1, saveat=1.0, num_noise_channels=1,
        tau_lims=(0.1, 10.0), T_lims=(0.1, 10.0), Nd_lims=(1.0, 15.0),
        sigma_lims=(0.005, 0.05), Bmax_lims=(1.0, 15.0), Aj_lims=(0.0, 0.1),
    )
    emit("backend", name=generator.simulation_backend)
    run_batches(args, generator, np)


if __name__ == "__main__":
    main()
