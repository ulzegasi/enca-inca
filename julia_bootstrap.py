"""
julia_bootstrap.py forces the Julia runtime (via juliacall) to initialize 
at the very start of the Python process, 
before TensorFlow or other heavy native libraries are imported, 
thereby preventing low-level runtime conflicts 
while keeping the SDDE solver embedded and fast.
"""
def init_julia():
    from juliacall import Main as jl
    jl.seval("VERSION")  # force init
    return jl