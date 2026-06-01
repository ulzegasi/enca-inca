import numpy as np
from scipy.integrate import solve_ivp

from ..model.haemodynamics import equations, stop_if_bad

def make_interp_funcs(seg, *, t_col, abp_col="ABP_lp", dp_col="dpA_dt_lp"):
    """
    Create p_A_func(t) and dp_A_dt_func(t) from a prepared segment.

    Works for minutes or seconds depending on t_col units.
    """
    t_grid = seg[t_col].to_numpy(float)
    pA_grid = seg[abp_col].to_numpy(float)
    dpA_grid = seg[dp_col].to_numpy(float)

    t0 = float(t_grid[0])
    t1 = float(t_grid[-1])

    def p_A_func(t):
        tt = float(np.clip(t, t0, t1))
        return float(np.interp(tt, t_grid, pA_grid))

    def dp_A_dt_func(t):
        tt = float(np.clip(t, t0, t1))
        return float(np.interp(tt, t_grid, dpA_grid))

    return p_A_func, dp_A_dt_func


def run_simulation(
    seg,
    params,
    *,
    t_col,
    initial_state,
    q_I=0.0,
    q_E=0.0,
    V_E_target=0.0,
    fitted=None,             # dict of fitted values to override params (optional)
    method="BDF",
    rtol=1e-5,
    atol=1e-7,
    use_events=True,
):
    """
    Run solve_ivp on one segment using hemodynamics.equations.

    fitted: dict like {"alpha_R":..., "alpha_L":..., "tau_R":..., ...}
            applied via params.with_updates(...)
    """
    # override params if fitted values provided
    p = params.with_updates(**fitted) if fitted else params

    t_eval = seg[t_col].to_numpy(float)
    t_span = (float(t_eval[0]), float(t_eval[-1]))

    p_A_func, dp_A_dt_func = make_interp_funcs(seg, t_col=t_col)

    rhs = lambda tt, yy: equations(
        tt, yy, p,
        p_A_func=p_A_func,
        dp_A_dt_func=dp_A_dt_func,
        q_I=q_I,
        q_E=q_E,
        V_E_target=V_E_target,
    )

    sol = solve_ivp(
        rhs,
        t_span,
        initial_state,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        events=stop_if_bad if use_events else None,
    )

    return sol # sol vector = [t, y]

    # y with pBn as y[3]