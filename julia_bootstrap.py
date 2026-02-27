def init_julia():
    from juliacall import Main as jl
    jl.seval("VERSION")  # force init
    return jl