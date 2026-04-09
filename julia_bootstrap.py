"""TensorFlow-safe Julia bootstrap using the repo's pinned Julia project."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_INITIALIZED = False
_JL = None


def _project_dir() -> Path:
    return Path(__file__).resolve().parent / "julia_env"


def _configure_juliacall_env(project_dir: Path) -> None:
    import juliapkg

    project_dir = project_dir.resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Julia project directory does not exist: {project_dir}")

    julia_exe = juliapkg.executable()
    project = str(project_dir)

    os.environ["PYTHON_JULIACALL_PROJECT"] = project
    os.environ.setdefault("PYTHON_JULIACALL_EXE", julia_exe)
    os.environ["JULIA_PROJECT"] = project

    subprocess.run(
        [
            julia_exe,
            f"--project={project_dir}",
            "--startup-file=no",
            "-e",
            "import Pkg; Pkg.instantiate()",
        ],
        check=True,
    )


def init_julia():
    """Initialize Julia once and return `juliacall.Main`."""
    global _INITIALIZED, _JL
    if _INITIALIZED:
        return _JL

    _configure_juliacall_env(_project_dir())

    from juliacall import Main as jl

    version = jl.seval("VERSION")
    active_project = jl.seval("Base.active_project()")
    print(f"--- Julia engine: ON (Julia {version}, project={active_project}) ---")

    _JL = jl
    _INITIALIZED = True
    return jl
