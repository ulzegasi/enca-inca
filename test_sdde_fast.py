"""
- Loads the Julia shared runtime into the current Python process -> embeds Julia inside Python.
- We do this at the very top of this test file, before importing tensorflow or src.* modules that may pull TF in.
- If TF initializes first → Julia aborts.
- If Julia initializes first → TF loads fine.
- We prevent low-level runtime conflicts and keep the julia SDDE solver embedded and fast.
"""
from julia_bootstrap import init_julia
init_julia() # Julia engine: on
# --- Julia is now initialized and ready to use ---

# --- Import the Python wrappers --- 
from src.sdde_solar_dynamo_julia import sn, summary_statistics
# Julia functions are not yet defined (because _init_julia() hasn’t been called)
# _init_julia() will be called lazily inside sn() and summary_statistics() when they are first called.

theta = (15.0, 10.0, 20.0, 0.2, 1.0)

# realistic length
y = sn(theta, Twarmup=200, Tobs=929, seed=123)

print("len(y) =", len(y))
print("first 3 =", list(y[:3]))

ss = summary_statistics(y)

print("len(ss) =", len(ss))
print("first 3 ss =", list(ss[:3]))