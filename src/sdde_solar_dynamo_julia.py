"""
Module name: sdde_solar_dynamo_julia.py
Author: Simone Ulzega, March 2026, simone.ulzega@zhaw.ch 
- src.sdde_solar_dynamo_julia provides a Python interface to the Julia SDDE solver:
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
import numpy as np

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
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            Tsim = Twarmup + Tobs
            sol = bfield(θ, Tsim; dt=dt, saveat=saveat, seed=seed)
            y = map(abs2, sol[1, (Twarmup + 2):end])
            return y
        end
        
        # ------------------------------------------------------------
        # Deterministic EM given "bare noise" eps ~ N(0,1)
        # This is the ENCA-friendly model: x = M(theta, eps)
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # ENCA-friendly deterministic EM given dt-level bare noise eps_dt ~ N(0,1)
        # eps_dt is per dt-increment (length = Ndt = Tsim/dt)
        # Output is sampled every 1.0 time unit (saveat==1.0) and then warmup-cropped
        # ------------------------------------------------------------
        function sn_from_noise(theta, eps_dt; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0)
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            @assert dt > 0

            τ, T, Nd, sigma, Bmax = theta

            Tsim = Twarmup + Tobs

            # dt-grid increments
            Ndt = Int(round(Tsim / dt))
            @assert abs(Ndt*dt - Tsim) < 1e-9 "Tsim must be multiple of dt"
            @assert length(eps_dt) >= Ndt "eps_dt too short: need Ndt = Tsim/dt"

            # delay in dt steps
            lag_steps = Int(round(T / dt))
            @assert lag_steps >= 1 "T/dt too small or dt too large"
            @assert abs(lag_steps*dt - T) < 1e-6 "T must be (approximately) a multiple of dt for this discretization"

            # EM noise scale
            coeff = Bmax * sigma / (τ^(3/2))
            sdt = sqrt(dt)

            # save every 1.0 time unit => k substeps per saved point
            k = Int(round(1.0 / dt))
            @assert abs(k*dt - 1.0) < 1e-12 "dt must divide 1.0 when saveat==1.0"

            # total saved points over [0, Tsim]: 0,1,2,...,Tsim
            Nsave = Int(round(Tsim)) + 1
            @assert abs(Tsim - (Nsave - 1)) < 1e-9 "Tsim must be integer when saveat==1.0"

            # state
            B  = Bmax
            dB = 0.0

            # ---- Ring buffer for delayed B (O(1), no popfirst!) ----
            # We store past B values at dt-grid points. The "delayed" value used at a step
            # is Bhist[hidx], where hidx points to the value from T units ago.
            Bhist = fill(Bmax, lag_steps)   # length = lag_steps
            hidx  = 1                       # next "delayed" slot to read/overwrite

            # output on integer-time grid (since saveat==1.0)
            y_save = Vector{Float64}(undef, Nsave)
            y_save[1] = B^2

            i = 0  # index into eps_dt
            @inbounds for j in 1:(Nsave-1)
                # advance by 1.0 time unit = k EM substeps
                for _sub in 1:k
                    i += 1

                    # delayed value (T in the past, discretized)
                    B_delay = Bhist[hidx]

                    # drift (same form as your SDDE definition)
                    du1 = dB
                    du2 = -B/τ^2 - 2*dB/τ - (Nd/τ^2) * ftilde(B_delay, 1, Bmax)

                    # EM update
                    dB_new = dB + du2*dt + coeff*sdt*eps_dt[i]
                    B_new  = B  + du1*dt

                    # update ring buffer with the NEW B (at the new time)
                    Bhist[hidx] = B_new
                    hidx += 1
                    if hidx > lag_steps
                        hidx = 1
                    end

                    B, dB = B_new, dB_new
                end

                y_save[j+1] = B^2
            end

            # crop warmup (integer-time indexing, consistent with your original sn)
            start = Twarmup + 2
            stop  = start + Tobs - 1
            @assert stop <= length(y_save)

            return y_save[start:stop]
        end

        function sn_for_enca(theta; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seed=nothing)
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            Tsim = Twarmup + Tobs

            Ndt = Int(round(Tsim / dt))
            @assert abs(Ndt*dt - Tsim) < 1e-9 "Tsim must be multiple of dt"

            if seed !== nothing
                Random.seed!(seed)
            end
            eps_dt = randn(Ndt)
            return sn_from_noise(theta, eps_dt; Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)
        end

        function test_consistency(theta; seed=123, Twarmup=200, Tobs=50, dt=0.1, saveat=1.0)
            @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
            Tsim = Twarmup + Tobs

            Ndt = Int(round(Tsim / dt))
            @assert abs(Ndt*dt - Tsim) < 1e-9 "Tsim must be multiple of dt"

            Random.seed!(seed)
            eps_dt = randn(Ndt)

            y1 = sn_from_noise(theta, eps_dt; Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)
            y2 = sn_from_noise(theta, eps_dt; Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)

            return maximum(abs.(y1 .- y2))
        end
        
        # ------------------------------------------------------------
        # This is the INCA-friendly model
        # ------------------------------------------------------------
        function sn_nrep(theta; nrep=8, seeds=nothing, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0)

            if seeds === nothing
                seeds = rand(1:2^31-1, nrep)
            end

            L = Int(round(Tobs / saveat))

            X = Array{Float32}(undef, nrep, L)

            for j in 1:nrep
                y = sn(theta;
                    Twarmup=Twarmup,
                    Tobs=Tobs,
                    dt=dt,
                    saveat=saveat,
                    seed=seeds[j])

                X[j, :] .= Float32.(y)
            end

            return X
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

def sn_for_enca(theta, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0, seed=None):
    _init_julia()
    return jl.sn_for_enca(tuple(theta), Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed)

def sn_from_noise(theta, eps, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0):
    _init_julia()
    return jl.sn_from_noise(tuple(theta), eps, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat)

def sn_nrep(theta, seeds, Twarmup=200, Tobs=929, dt=0.1, saveat=1.0):
    _init_julia()
    # juliacall likes tuples for small fixed-size vectors
    X = jl.sn_nrep(
        tuple(theta),
        nrep=len(seeds),
        seeds=seeds,
        Twarmup=Twarmup,
        Tobs=Tobs,
        dt=dt,
        saveat=saveat,
    )
    return np.asarray(X, dtype=np.float32)

def test_consistency(theta, seed=123, Twarmup=200, Tobs=50, dt=0.1, saveat=1.0):
    _init_julia()
    return float(jl.test_consistency(tuple(theta), seed=seed, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat))
