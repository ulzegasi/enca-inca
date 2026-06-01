import numpy as np
from scipy.optimize import least_squares
from .simulate import run_simulation
def fit_control_segment(
    seg,
    params,
    *,
    t_col,
    initial_state,
    # which params to fit
    fit_k_B=True,
    fit_R_FSn=True,
    fit_q_ACn=False,
    # initial guess overrides (optional)
    theta0=None,
    # bounds (optional)
    bounds=None,
    # solver settings
    method="BDF",
    rtol=1e-5,
    atol=1e-7,
    max_nfev=200,
    loss="soft_l1",
    f_scale=5.0,
    verbose=2,
    # swelling/infusion
    q_I=0.0,
    q_E=0.0,
    V_E_target=0.0,
    normalize=True,
):
    """
    Fit control parameters by minimizing (model ICP - observed ICP) over the segment.

    Observations:
      seg must contain: ICP_lp
    """

    icp_obs = seg["ICP_lp"].to_numpy(float)

    # Parameter vector layout:
    # [alpha, tau, dC_AB1, dC_AB2, (q_ACn?), (k_B?), (R_FSn?)]
    base = [
        params.alpha_R,
        params.tau_R,
        params.dC_AB1_R,
        params.dC_AB2_R,
    ]

    if fit_q_ACn:
        # adjust this name if your params object uses a different field name
        base.append(params.q_ACn_R)

    if fit_k_B:
        base.append(params.k_B)

    if fit_R_FSn:
        base.append(params.R_FSn)

    if theta0 is None:
        theta0 = np.array(base, dtype=float)
    else:
        theta0 = np.array(theta0, dtype=float)

    # default bounds
    if bounds is None:
        eps = 1e-12
        lb = [0.0, eps, 0.0, 0.0]
        ub = [10.0, 60, 5.0, 5.0]

        if fit_q_ACn:
            lb.append(200)
            ub.append(600.0)

        if fit_k_B:
            lb.append(eps)
            ub.append(50.0)

        if fit_R_FSn:
            lb.append(eps)
            ub.append(1e9)

        bounds = (np.array(lb, float), np.array(ub, float))

    def theta_to_updates(theta):
        theta = list(map(float, theta))
        alpha, tau, dC1, dC2 = theta[:4]
        idx = 4

        updates = {
            "alpha_R": alpha,
            "alpha_L": alpha,
            "tau_R": tau,
            "tau_L": tau,
            "dC_AB1_R": dC1,
            "dC_AB1_L": dC1,
            "dC_AB2_R": dC2,
            "dC_AB2_L": dC2,
        }

        if fit_q_ACn:
            q_ACn = theta[idx]
            idx += 1

            updates["q_ACn_R"] = q_ACn
            updates["q_ACn_L"] = q_ACn

        if fit_k_B:
            updates["k_B"] = theta[idx]
            idx += 1

        if fit_R_FSn:
            updates["R_FSn"] = theta[idx]
            idx += 1

        return updates

    def simulate_pB(theta):
        fitted = theta_to_updates(theta)
        sol = run_simulation(
            seg,
            params,
            t_col=t_col,
            initial_state=initial_state,
            fitted=fitted,
            q_I=q_I,
            q_E=q_E,
            V_E_target=V_E_target,
            method=method,
            rtol=rtol,
            atol=atol,
            use_events=False,
        )
        if not sol.success:
            return None
        return sol.y[2, :]

    def residuals(theta):
        pB = simulate_pB(theta)
        if pB is None or np.any(~np.isfinite(pB)) or (len(pB) != len(icp_obs)):
            return 1e6 * np.ones_like(icp_obs)

        res = pB - icp_obs
        if normalize:
            n = max(len(icp_obs), 1)
            res = res / np.sqrt(n)
        return res

    res = least_squares(
        residuals,
        theta0,
        bounds=bounds,
        method="trf",
        verbose=verbose,
        max_nfev=max_nfev,
        loss=loss,
        f_scale=f_scale,
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        diff_step=1e-2,
    )

    fitted_dict = theta_to_updates(res.x)
    return res, fitted_dict