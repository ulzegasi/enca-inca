"""
src.sdde_solar_dynamo_julia provides a Python interface to the Julia SDDE solver:
- load packages and defines Julia model functions
- Julia functions are initialized when _init_julia() is called
- _init_julia() will be called lazily inside sn() and summary_statistics() when they are first called.
- _init_julia is governed by a global _INITIALIZED flag to ensure it only runs once.
- sn() runs the SDDE solver and returns the time series of the magnetic field strength
- summary_statistics() computes summary statistics from the time series using FFT
- We do the Julia setup lazily, only when sn() or summary_statistics() is called
- For stability, call julia_bootstrap.init_julia() at the very top of your main script
"""
  
from __future__ import annotations

from typing import Iterable, Optional, Sequence

# IMPORTANT:
# - Do NOT import juliacall.Main at module import time (can crash depending on import order).
# - We only grab it inside _init_julia(), ideally after julia_bootstrap.init_julia() was called.

jl = None
_INITIALIZED = False

def _init_julia():
    """
    One-time Julia setup: imports packages and defines Julia functions.

    For stability:
    - Call julia_bootstrap.init_julia() at the very top of your main script
      BEFORE importing tensorflow or src.* modules that may pull TF in.
    """
    global _INITIALIZED, jl
    if _INITIALIZED:
        return

    # Import Main lazily (already bootstrapped in julia_bootstrap)
    if jl is None:
        from juliacall import Main as _jl
        jl = _jl

    # Julia imports (one-time)
    jl.seval("using StochasticDelayDiffEq")
    jl.seval("using SpecialFunctions: erf")
    jl.seval("using StaticArrays")
    jl.seval("using FFTW")
    jl.seval("using Random")

    # Define Julia functions (one-time)
    jl.seval(
        r"""
        ftilde(x, Bmin, Bmax) = x/4 * (1 + erf(x^2-Bmin^2)) * (1 - erf(x^2-Bmax^2))

        function f(u,h,p,t)
            τ, T, Nd, sigma, Bmax = p
            hist = h(p, t - T, idxs = 1)
            du1 = u[2]
            du2 = -u[1]/τ^2 - 2*u[2]/τ - Nd/τ^2*ftilde(hist, 1, Bmax)
            SA[du1, du2]
        end

        function g(u,h,p,t)
            τ, T, Nd, sigma, Bmax = p
            du1 = 0.0
            du2 = Bmax*sigma / (τ^(3/2))
            SA[du1, du2]
        end

        function bfield(θ, Tsim; dt=0.1, saveat=1.0, seed=nothing)
            τ, T, Nd, sigma, Bmax = θ
            u0 = SA[Bmax, 0.0]
            h(p, t; idxs = nothing) = idxs == 1 ? Bmax : (Bmax, 0.0)
            lags = (T,)
            tspan = (0.0, Tsim)

            prob = SDDEProblem(f, g, u0, h, tspan, θ; constant_lags = lags)

            if seed !== nothing
                Random.seed!(seed)
            end

            solve(prob, EM(); dt=dt, saveat=saveat)
        end

        function sn(θ; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seed=nothing)
            Tsim = Twarmup + Tobs
            sol = bfield(θ, Tsim; dt=dt, saveat=saveat, seed=seed)
            y = map(abs2, sol[1, (Twarmup + 2):end])
            return y
        end

        hann_window(Tmax) = [0.5*(1 - cos(2.0*π*(t-1)/(Tmax-1))) for t in 1:Tmax]

        function summary_statistics(data, window=hann_window(length(data)); fourier_range=1:6:120)
            fs = FFTW.ifft(window .* data)
            ss = abs.(fs[fourier_range])
            return ss
        end

        function summary_statistics_ii(data, window=hann_window(length(data)); fourier_range=1:6:120)
            fs = FFTW.ifft(window .* data)[1:120]
            ss = [real.(fs); imag.(fs)][fourier_range]
            return ss
        end
        """
    )

    _INITIALIZED = True


def sn(
    theta: Sequence[float],
    Twarmup: int = 200,
    Tobs: int = 929,
    dt: float = 0.1,
    saveat: float = 1.0,
    seed: Optional[int] = None,
):
    _init_julia()
    # juliacall likes tuples for small fixed-size vectors
    return jl.sn(tuple(theta), Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed)


def hann_window(Tmax: int):
    _init_julia()
    return jl.hann_window(Tmax)


def summary_statistics(
    data,
    window=None,
    fourier_range=None,
):
    _init_julia()
    if window is None and fourier_range is None:
        return jl.summary_statistics(data)
    if fourier_range is None:
        return jl.summary_statistics(data, window)
    return jl.summary_statistics(data, window, fourier_range=fourier_range)


def summary_statistics_ii(
    data,
    window=None,
    fourier_range=None,
):
    _init_julia()
    if window is None and fourier_range is None:
        return jl.summary_statistics_ii(data)
    if fourier_range is None:
        return jl.summary_statistics_ii(data, window)
    return jl.summary_statistics_ii(data, window, fourier_range=fourier_range)