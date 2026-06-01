import numpy as np

def oxygen_transport(q_AC, SaO2_in, params, debug=False):
    """
    Mean tissue pO2 ⟨p̄_ti⟩ via Jung/Böhm Krogh-cylinder model (Böhm approximation),
    using grids + geometry precomputes provided by `params`.

    Mathematical pipeline (notation consistent with our derivation)
    --------------------------------------------------------------
    1) Velocity–flow coupling:
         v = v_n * (q_AC / q_An)

    2) Capillary saturation profile (Böhm approximation):
         b(r_t) = ((r_t^2 - r_c^2)/r_c^2) * (M / (beta * v)) 
         S_bO2(z, r_t) = SaO2 - b(r_t) * z

    3) Hill inversion (saturation → partial pressure):
         p_c(z, r_t) = p50 * (S_bO2 / (1 - S_bO2))^(1/n)

    4) Mean capillary pO2 along capillary length:
         p̄_c(r_t) = (1/L) ∫_0^L p_c(z, r_t) dz

    5) Mean tissue pO2 for that radius:
         p̄_ti(r_t) = p̄_c(r_t) - Δ(r_t)     (Δ is geometry-only diffusion/consumption term)

    6) Average over radius distribution (Gaussian + volume weighting ∝ r^2):
         ⟨p̄_ti⟩ = ∫ p̄_ti(r) w(r) dr, with ∫ w(r) dr = 1

    Parameters
    ----------
    q_AC : float
        Arterial→capillary inflow (same units as params.q_An; e.g. mL/min).

    SaO2_in : float
        Arterial hemoglobin saturation at capillary entrance (z=0), fraction in (0,1).
        (This is SaO2 = S_bO2(0).)

    params : object
        Must provide:
          - scalars: v_n, q_An, L, M, beta, p50, n
          - precomputes: z_grid, r_grid, weights, delta_r, geom_b

    debug : bool
        If True, returns a dict of intermediate arrays.

    Returns
    -------
    p_mean : float
        Mean tissue pO2 ⟨p̄_ti⟩ (mmHg)
    """

    # ---- Pull precomputed grids + geometry (computed once from params) ----
    z_grid  = params.z_grid     # shape: (n_z,)
    r_grid  = params.r_grid     # shape: (n_r,)
    weights = params.weights    # normalized so ∫ weights dr = 1
    delta_r = params.delta_r    # Δ(r), geometry-only correction, shape: (n_r,)
    geom_b  = params.geom_b     # (r^2 - r_c^2)/r_c^2, shape: (n_r,)

    # ---- 1) Velocity from flow (hemodynamic coupling) ----
    # v scales linearly with q_AC / q_An
    v = params.v_n * (q_AC / params.q_An)

    # ---- 2) Boundary condition: SaO2 = S_bO2(z=0) ----
    SaO2 = np.clip(float(SaO2_in), 1e-6, 1.0 - 1e-6)

    # ---- 3) Radius-dependent saturation slope b(r) ----
    # b(r) = geom_b(r) * (M / (beta * v))
    b_vec = geom_b * (params.M / (params.beta * v))   # shape: (n_r,)

    # ---- 4) Linear saturation profile along capillary ----
    # S_bO2(z,r) = SaO2 - b(r) * z
    SbO2 = SaO2 - b_vec[:, None] * z_grid[None, :]     # shape: (n_r, n_z)
    SbO2 = np.clip(SbO2, 1e-6, 1.0 - 1e-6)

    # ---- 5) Hill inversion: saturation → capillary pO2 ----
    inv_n = 1.0 / params.n
    p_c = params.p50 * (SbO2 / (1.0 - SbO2)) ** inv_n  # shape: (n_r, n_z)

    # ---- 6) Mean capillary pO2 along z for each radius ----
    # p̄_c(r) = (1/L) ∫_0^L p_c(z,r) dz
    p_c_mean_vec = np.trapezoid(p_c, z_grid, axis=1) / params.L  # shape: (n_r,)

    # ---- 7) Mean tissue pO2 per radius: p̄_ti(r) = p̄_c(r) - Δ(r) ----
    p_ti_vec = p_c_mean_vec - delta_r

    # ---- 8) Radius-distribution average: ⟨p̄_ti⟩ = ∫ p̄_ti(r) w(r) dr ----
    p_mean = np.trapezoid(p_ti_vec * weights, r_grid)

    if debug:
        # Optional: arterial pO2 corresponding to SaO2 (for inspection only)
        pbO2_arterial = params.p50 * (SaO2 / (1.0 - SaO2)) ** inv_n

        return {
            "SaO2": SaO2,
            "pbO2_arterial": pbO2_arterial,   # debug-only
            "v": v,
            "b_vec": b_vec,
            "SbO2": SbO2,
            "p_c": p_c,
            "p_c_mean_vec": p_c_mean_vec,
            "delta_r": delta_r,
            "p_ti_vec": p_ti_vec,
            "weights": weights,
            "z_grid": z_grid,
            "r_grid": r_grid,
            "p_mean": float(p_mean),
        }

    return float(p_mean)