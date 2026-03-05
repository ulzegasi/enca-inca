"""
julia_bootstrap.py forces the Julia runtime (via juliacall) to initialize 
at the very start of the Python process, 
before TensorFlow or other heavy native libraries are imported, 
thereby preventing low-level runtime conflicts 
while keeping the SDDE solver embedded and fast.
"""
_INITIALIZED = False

def init_julia():
    global _INITIALIZED
    if _INITIALIZED:
        return

    from juliacall import Main as jl
    version = jl.seval("VERSION")
    print(f"--- Julia engine: ON (Julia {version}) ---")
    _INITIALIZED = True
    return jl