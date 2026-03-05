"""
test_sdde.py

Visual sanity check:
- Compare the original solver path sn(theta) with the ENCA-friendly
  deterministic EM path sn_for_enca(theta), for saveat=1.0.

This script intentionally does not test saveat != 1.0 because
sn_for_enca/sn_from_noise assert saveat==1.0.
"""

import numpy as np
import matplotlib.pyplot as plt

import time
t0 = time.time()
print("About to start the test...")

# Must be first: avoid TF / native runtime conflicts
from julia_bootstrap import init_julia
init_julia() # julia engine: ON

from src.sdde_solar_dynamo_julia import sn, sn_for_enca

theta = (3.0, 3.0, 10.0, 0.01, 10.0)  # (tau, T, Nd, sigma, Bmax)

Twarmup = 200
Tobs = 929
dt = 0.1
saveat = 1.0
seed = 123

y_sn = np.asarray(
    sn(theta, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed),
    dtype=float,
)
y_enca = np.asarray(
    sn_for_enca(theta, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed),
    dtype=float,
)

y_sn_prev = y_sn.copy()
y_enca_prev = y_enca.copy()

print(f"Model run in {time.time() - t0:.1f} s")

def timed_runs(theta, n=50):
    t0 = time.perf_counter()
    ys = []
    for k in range(n):
        y = np.asarray(sn_for_enca(theta, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed+k), dtype=float)
        ys.append(y)
    dt_tot = time.perf_counter() - t0
    print(f"{n} runs total: {dt_tot:.4f} s  ->  {1e3*dt_tot/n:.3f} ms/run")
    return ys

ys = timed_runs((3.0, 3.0, 10.0, 0.01, 10.0), n=50)
print("Sanity: outputs differ across seeds?",
      np.max(np.abs(ys[0]-ys[-1])))

t1 = time.time()
theta = (3.5, 3.2, 9.5, 0.02, 7.0)  # (tau, T, Nd, sigma, Bmax)
y_sn = np.asarray(
    sn(theta, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed),
    dtype=float,
)
y_enca = np.asarray(
    sn_for_enca(theta, Twarmup=Twarmup, Tobs=Tobs, dt=dt, saveat=saveat, seed=seed),
    dtype=float,
)
print(f"Second model run in {time.time() - t1:.5f} s")

print("diff(sn)  :", np.max(np.abs(y_sn - y_sn_prev)))
print("diff(enca):", np.max(np.abs(y_enca - y_enca_prev)))

if y_sn.shape != y_enca.shape:
    raise RuntimeError(f"Shape mismatch: sn={y_sn.shape}, sn_for_enca={y_enca.shape}")

t = np.arange(len(y_sn)) * saveat
diff = y_sn - y_enca

fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax[0].plot(t, y_sn)
ax[0].set_title("sn(theta) — solver path (StochasticDelayDiffEq.solve, EM)")
ax[0].set_ylabel("B²")

ax[1].plot(t, y_enca)
ax[1].set_title("sn_for_enca(theta) — deterministic EM (dt-level eps)")
ax[1].set_ylabel("B²")

ax[2].plot(t, diff)
ax[2].set_title(f"Difference: sn - sn_for_enca   (max|diff| = {np.max(np.abs(diff)):.3e})")
ax[2].set_xlabel("time")
ax[2].set_ylabel("ΔB²")

plt.tight_layout()
plt.show()