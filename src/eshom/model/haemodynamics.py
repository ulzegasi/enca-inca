import numpy as np

# ----------------------------------------------------------------------
# Ramp resistance function
# ----------------------------------------------------------------------
def ramp_resistance(x, R_nom, R_max=1e6, k=400.0, x0=0.0):
    """
    Numerically safe ramp resistance function.
    Smoothly transitions from R_max (closed) to R_nom (open)
    around threshold x0 with steepness k.

    - x: control variable (e.g. p_B - p_S)
    - R_nom: open resistance (low)
    - R_max: closed resistance (high)
    - k: steepness of transition
    - x0: midpoint (where resistance = (R_nom + R_max)/2)
    """
    # Limit the exponential argument to avoid overflow
    z = np.clip(-k * (x - x0), -50, 50)  
    s = 1.0 / (1.0 + np.exp(z))          # smooth sigmoid between 0 and 1
    return R_nom + (R_max - R_nom) * (1.0 - s)

# ----------------------------------------------------------------------
# ODE system (two hemispheres)
# ----------------------------------------------------------------------
def equations(
    t,
    state,
    params,
    p_A_func,
    dp_A_dt_func,
    q_I=0.0,
    q_E=0.0,
    V_E_target=0.0,
):
    """
    ODE system for the cerebral hemodynamics model.

    Parameters
    ----------
    t : float
        time (your chosen unit: seconds OR minutes; must match params)
    state : array-like length 8
        [p_V_R, p_V_L, p_B, C_AB_R, C_AB_L, V_V_R, V_V_L, V_F]
    params : Params
        parameter object (no globals)
    p_A_func : callable
        arterial pressure p_A(t)
    dp_A_dt_func : callable
        derivative dp_A/dt at time t
    q_I : float
        infusion rate (same flow unit as model)
    q_E : float
        swelling rate (same flow unit as model)
    V_E_target : float
        swelling target volume (mL)
    """

    # Unpack state variables
    #p_V, p_B, C_AB, V_V, V_F = state
    p_V_R, p_V_L, p_B, C_AB_R, C_AB_L, V_V_R, V_V_L, V_F = state

    # -----------------------
    # Swelling
    # p_A_R = p_A_L if COW is functional
    V_E = q_E * t

    # Accumulate fluid under constant pA assumption
    if V_E < V_E_target:
        p_A_R = params.p_An
        p_A_L = params.p_An
        dp_A_R_dt = 0.0
        dp_A_L_dt = 0.0
    # Once target volume is rached, switch off swelling flow, let pA change
    # If volume target = 0 (no swell), let pA change, proceed as normal
    else:
        V_E = V_E_target
        q_E = 0.0
        p_A_R = float(p_A_func(t))
        p_A_L = float(p_A_func(t))
        dp_A_R_dt = float(dp_A_dt_func(t))
        dp_A_L_dt = float(dp_A_dt_func(t))
    
    # -----------------------
    # Constants
    R_CV_R = params.R_CVn
    R_CV_L = params.R_CVn
    p_S = params.p_S  # = params.p_Sn

    # -----------------------
    # Compliances -> CVB is WIP
    C_VB_R = 1.0 / (params.k_V * (abs(p_V_R - p_B) + params.p_V0))  # Eq 9 (Jung)
    C_VB_L = 1.0 / (params.k_V * (abs(p_V_L - p_B) + params.p_V0))  # Eq 9 (Jung)
    C_B    = 1.0 / (params.k_B * (abs(p_B) + params.p_B0))          # Eq 13 (Jung)

    # -----------------------
    # Arterial resistance 
    denom_R = (C_AB_R * (p_A_R - p_B))
    denom_L = (C_AB_L * (p_A_L - p_B))
    R_AC_R = params.k_RAC_R / (denom_R**2) if denom_R != 0 else 1e6
    R_AC_L = params.k_RAC_L / (denom_L**2) if denom_L != 0 else 1e6

    # -----------------------
    # Venous Starling resistor resistance
    R_VS_R = (1.0 / params.g_VSn) * ((p_V_R - p_S) / (p_V_R - p_B)) if (p_V_R - p_B) else 1e6
    R_VS_L = (1.0 / params.g_VSn) * ((p_V_L - p_S) / (p_V_L - p_B)) if (p_V_L - p_B) else 1e6
    
    # -----------------------
    # CSF production resistance
    # Logic:
    # 1) Compute provisional capillary pressure assuming CSF branch closed.
    # 2) If p_C ≤ p_B → CSF formation closed (high resistance).
    # 3) If p_C > p_B → open CSF branch and recompute node pressure.
    # 4) Safety re-check to prevent non-physical backflow.
        
    # ---- RIGHT ----
    # Provisional capillary pressure (2-branch network: p_A ↔ p_C ↔ p_V)
    if (1/(R_AC_R) + 1/(R_CV_R)) != 0:
        p_C_temp_R = (p_A_R / R_AC_R + p_V_R / R_CV_R) / (1/R_AC_R + 1/R_CV_R)
    else:
        p_C_temp_R = (p_A_R + p_V_R) / 2  # fallback (numerical safety)

    # If capillary pressure ≤ brain pressure → CSF formation closed
    if p_C_temp_R <= p_B:
        R_CF_R = ramp_resistance(p_C_temp_R - p_B, params.R_CFn, R_max=1e6, k=400.0)
        p_C_R = p_C_temp_R
    else:
        # Capillary pressure exceeds brain pressure → CSF branch open
        R_CF_R = params.R_CFn

        # Recompute capillary pressure with 3 branches (p_A, p_V, p_B)
        if (1/(R_AC_R) + 1/(R_CV_R) + 1/(R_CF_R)) != 0:
            p_C_R = (
                p_A_R / R_AC_R +
                p_V_R / R_CV_R +
                p_B   / R_CF_R
            ) / (1/R_AC_R + 1/R_CV_R + 1/R_CF_R)
        else:
            p_C_R = (p_A_R + p_V_R + p_B) / 3  # fallback

    # Safety: if recomputed p_C drops below p_B → close CSF branch again
    if p_C_R < p_B:
        R_CF_R = ramp_resistance(p_C_R - p_B, params.R_CFn, R_max=1e6, k=400.0)
        p_C_R = p_C_temp_R

    # ---- LEFT ----
    p_C_temp_L = (p_A_L / (R_AC_L) + p_V_L / (R_CV_L))/(1/(R_AC_L) + 1/(R_CV_L)) if (1/(R_AC_L) + 1/(R_CV_L)) != 0 else (p_A_L + p_V_L)/2
    if p_C_temp_L <= p_B:
        #R_CF = 1e6
        R_CF_L = ramp_resistance(p_C_temp_L - p_B, params.R_CFn, R_max=1e6, k=400.0)
        p_C_L = p_C_temp_L
    else:
        R_CF_L = params.R_CFn
        p_C_L = (p_A_L / (R_AC_L) + p_V_L / (R_CV_L) + p_B / (R_CF_L))/(1/(R_AC_L) + 1/(R_CV_L) + 1/(R_CF_L)) if \
            (1/(R_AC_L) + 1/(R_CV_L) + 1/(R_CF_L)) != 0 else (p_A_L + p_V_L + p_B)/3
    if p_C_L < p_B:
        #R_CF = 1e10
        R_CF_L = ramp_resistance(p_C_L - p_B, params.R_CFn, R_max=1e6, k=400.0)
        p_C_L = p_C_temp_L

    # ---- LEFT ----
    # Provisional capillary pressure (2-branch network: p_A ↔ p_C ↔ p_V)
    if (1/(R_AC_L) + 1/(R_CV_L)) != 0:
        p_C_temp_L = (p_A_L / R_AC_L + p_V_L / R_CV_L) / (1/R_AC_L + 1/R_CV_L)
    else:
        p_C_temp_L = (p_A_L + p_V_L) / 2  # fallback (numerical safety)

    # If capillary pressure ≤ brain pressure → CSF formation closed
    if p_C_temp_L <= p_B:
        R_CF_L = ramp_resistance(p_C_temp_L - p_B, params.R_CFn, R_max=1e6, k=400.0)
        p_C_L = p_C_temp_L
    else:
        # Capillary pressure exceeds brain pressure → CSF branch open
        R_CF_L = params.R_CFn

        # Recompute capillary pressure with 3 branches (p_A, p_V, p_B)
        if (1/(R_AC_L) + 1/(R_CV_L) + 1/(R_CF_L)) != 0:
            p_C_L = (
                p_A_L / R_AC_L +
                p_V_L / R_CV_L +
                p_B   / R_CF_L
            ) / (1/R_AC_L + 1/R_CV_L + 1/R_CF_L)
        else:
            p_C_L = (p_A_L + p_V_L + p_B) / 3  # fallback

    # Safety: if recomputed p_C drops below p_B → close CSF branch again
    if p_C_L < p_B:
        R_CF_L = ramp_resistance(p_C_L - p_B, params.R_CFn, R_max=1e6, k=400.0)
        p_C_L = p_C_temp_L

    # -----------------------
    # CSF absorption resistance
    R_FS = ramp_resistance(p_B - p_S, params.R_FSn, R_max=1e6, k=400.0)
    R_FS *= ramp_resistance(V_F - 0.1, 1.0, R_max=1e6, k=400.0)

    # -----------------------
    # Flows
    q_AC_R = (p_A_R - p_C_R) / (R_AC_R)
    q_AC_L = (p_A_L - p_C_L) / (R_AC_L)

    q_CV_R = (p_C_R - p_V_R) / (R_CV_R)
    q_CV_L = (p_C_L - p_V_L) / (R_CV_L)

    q_CF_R = (p_C_R - p_B) / (R_CF_R) if R_CF_R < 1e9 else 0
    q_CF_L = (p_C_L - p_B) / (R_CF_L) if R_CF_L < 1e9 else 0 

    q_VS_R = (p_V_R - p_S) / (R_VS_R)
    q_VS_L = (p_V_L - p_S) / (R_VS_L)

    q_FS = (p_B - p_S) / (R_FS) if R_FS < 1e9 else 0 

    # -----------------------
    # Venous pressure evolution term
    term_4_R = (1.0 / C_VB_R) * (q_CV_R - q_VS_R)
    term_4_L = (1.0 / C_VB_L) * (q_CV_L - q_VS_L)

    # -----------------------
    # Autoregulation 
    x_R = (q_AC_R - params.q_ACn_R) / params.q_ACn_R
    x_L = (q_AC_L - params.q_ACn_L) / params.q_ACn_L

    dC_AB_R = params.dC_AB1_R if x_R < 0 else params.dC_AB2_R
    dC_AB_L = params.dC_AB1_L if x_L < 0 else params.dC_AB2_L

    C_ABreg_R = params.C_ABn_R - dC_AB_R * np.tanh((x_R * params.alpha_R) / dC_AB_R)
    C_ABreg_L = params.C_ABn_L - dC_AB_L * np.tanh((x_L * params.alpha_L) / dC_AB_L)

    term_5_R = -(1.0 / params.tau_R) * (C_AB_R - C_ABreg_R)
    term_5_L = -(1.0 / params.tau_L) * (C_AB_L - C_ABreg_L)


    # -----------------------
    # ODEs

    # Fluid volume as a sum of inflows/outflows
    dV_F_dt = q_I + q_CF_R + q_CF_L - q_FS 

    # Venous volumes as a sum of inflows/outflows per hemisphere
    dV_V_R_dt = q_CV_R - q_VS_R
    dV_V_L_dt = q_CV_L - q_VS_L

    # Arterial compliances 
    dC_AB_R_dt = term_5_R 
    dC_AB_L_dt = term_5_L

    # Intracranial pressure as function of global volume balance
    # divided by effective intracranial compliance
    term_1 = 1/(C_B + C_AB_R + C_AB_L) # effective intracranial compliance
    # Right hemisphere volume contributions for veins and arteries
    term_2_R = (term_5_R * (p_A_R - p_B) # dC_AB(pA - PB)
               + (q_CV_R - q_VS_R)       # dV_V_dt 
               + C_AB_R * (dp_A_R_dt))   # C_AB(dpA_dt)
    # Left hemisphere volume contributions for veins and arteries
    term_2_L = (term_5_L * (p_A_L - p_B) # dC_AB(pA - PB)
               + (q_CV_L - q_VS_L)       # dV_V_dt 
               + C_AB_L * (dp_A_L_dt))   # C_AB(dpA_dt)
    term_2_sum = term_2_R + term_2_L 
    # Fluid and swelling volume contibutions
    term_3 = q_CF_R + q_CF_L + q_I - q_FS # dV_F_dt 
    # total volume derivate/effective compliance
    dp_B_dt = term_1 * (term_2_sum + term_3 + q_E) 

    # Venous pressure as a function of derivative of intracranial pressure
    # and local venous pressure (using volume change over compliance)
    dp_V_R_dt = ((term_1*(term_2_sum + term_3 + q_E)) # dp_B_dt
                + term_4_R) # dV_Vdt/CVB
    dp_V_L_dt = ((term_1*(term_2_sum + term_3 + q_E)) # dp_B_dt
                + term_4_L) # dV_Vdt/CVB

    return [dp_V_R_dt, dp_V_L_dt, dp_B_dt, dC_AB_R_dt, dC_AB_L_dt, dV_V_R_dt, dV_V_L_dt, dV_F_dt]


# ----------------------------------------------------------------------
# Intermediates/derived values
# ----------------------------------------------------------------------

def compute_intermediates(t, state, params, p_A_func):

    # unpack state
    p_V_R, p_V_L, p_B, C_AB_R, C_AB_L, V_V_R, V_V_L, V_F = state


    # -----------------------
    # arterial pressure -> p_A_R = p_A_L under COW assumption
    p_A_R = float(p_A_func(t))
    p_A_L = float(p_A_func(t))
    # sinus pressure
    p_S = params.p_S

    # -----------------------
    # Compliances
    C_VB_R = 1.0 / (params.k_V * (abs(p_V_R - p_B) + params.p_V0))
    C_VB_L = 1.0 / (params.k_V * (abs(p_V_L - p_B) + params.p_V0))
    C_B    = 1.0 / (params.k_B * (abs(p_B) + params.p_B0))


    # -----------------------
    # Resistances 

    # arterial
    R_AC_R = params.k_RAC_R  / ((C_AB_R * (p_A_R - p_B))**2) if C_AB_R*(p_A_R-p_B) != 0 else 1e6
    R_AC_L = params.k_RAC_L / ((C_AB_L * (p_A_L - p_B))**2) if C_AB_L*(p_A_L-p_B) != 0 else 1e6

    # venous Starling resistor 
    R_VS_R = (1/params.g_VSn) * ((p_V_R - p_S)/(p_V_R - p_B)) if (p_V_R - p_B) else 1e6
    R_VS_L = (1/params.g_VSn) * ((p_V_L - p_S)/(p_V_L - p_B)) if (p_V_L - p_B) else 1e6

    # capillary outflow resistance
    R_CV_R = params.R_CVn
    R_CV_L = params.R_CVn

    # Capillary pressure + CSF formation (same logic)
    denom_tmp_R = (1.0 / R_AC_R + 1.0 / R_CV_R)
    p_C_temp_R = (p_A_R / R_AC_R + p_V_R / R_CV_R) / denom_tmp_R if denom_tmp_R != 0 else 0.5 * (p_A_R + p_V_R)

    if p_C_temp_R <= p_B:
        R_CF_R = ramp_resistance(p_C_temp_R - p_B, params.R_CFn, R_max=1e6, k=400.0)
        p_C_R = p_C_temp_R
    else:
        R_CF_R = params.R_CFn
        denom_R2 = (1.0 / R_AC_R + 1.0 / R_CV_R + 1.0 / R_CF_R)
        p_C_R = (p_A_R / R_AC_R + p_V_R / R_CV_R + p_B / R_CF_R) / denom_R2 if denom_R2 != 0 else (p_A_R + p_V_R + p_B) / 3.0

    if p_C_R < p_B:
        R_CF_R = ramp_resistance(p_C_R - p_B, params.R_CFn, R_max=1e6, k=400.0)
        p_C_R = p_C_temp_R

    denom_tmp_L = (1.0 / R_AC_L + 1.0 / R_CV_L)
    p_C_temp_L = (p_A_L / R_AC_L + p_V_L / R_CV_L) / denom_tmp_L if denom_tmp_L != 0 else 0.5 * (p_A_L + p_V_L)

    if p_C_temp_L <= p_B:
        R_CF_L = ramp_resistance(p_C_temp_L - p_B, params.R_CFn, R_max=1e6, k=400.0)
        p_C_L = p_C_temp_L
    else:
        R_CF_L = params.R_CFn
        denom_L2 = (1.0 / R_AC_L + 1.0 / R_CV_L + 1.0 / R_CF_L)
        p_C_L = (p_A_L / R_AC_L + p_V_L / R_CV_L + p_B / R_CF_L) / denom_L2 if denom_L2 != 0 else (p_A_L + p_V_L + p_B) / 3.0

    if p_C_L < p_B:
        R_CF_L = ramp_resistance(p_C_L - p_B, params.R_CFn, R_max=1e6, k=400.0)
        p_C_L = p_C_temp_L

    # CSF absorption
    R_FS = ramp_resistance(p_B - p_S, params.R_FSn, R_max=1e6, k=400.0)
    R_FS *= ramp_resistance(V_F - 0.1, 1.0, R_max=1e6, k=400.0)

    # Flows
    q_AC_R = (p_A_R - p_C_R) / R_AC_R
    q_AC_L = (p_A_L - p_C_L) / R_AC_L

    q_CV_R = (p_C_R - p_V_R) / R_CV_R
    q_CV_L = (p_C_L - p_V_L) / R_CV_L

    q_CF_R = (p_C_R - p_B) / R_CF_R if R_CF_R < 1e9 else 0.0
    q_CF_L = (p_C_L - p_B) / R_CF_L if R_CF_L < 1e9 else 0.0

    q_VS_R = (p_V_R - p_S) / R_VS_R
    q_VS_L = (p_V_L - p_S) / R_VS_L

    q_FS = (p_B - p_S) / R_FS if R_FS < 1e9 else 0.0

    # q_A 
    R_R = R_AC_R + R_CV_R + R_VS_R
    R_L = R_AC_L + R_CV_L + R_VS_L
    q_A = ((p_A_R * R_L) + (p_A_L * R_R)) / (R_R * R_L)
    q_A_R = p_A_R / R_R
    q_A_L = p_A_L / R_L

    # arterial volumes 
    V_A0_R = params.V_A_Rn - params.C_ABn_R * (params.p_An - params.p_Bn)
    V_A0_L = params.V_A_Ln - params.C_ABn_L * (params.p_An - params.p_Bn)

    V_A_R = C_AB_R * (p_A_R - p_B) + V_A0_R
    V_A_L = C_AB_L * (p_A_L - p_B) + V_A0_L

    # Monro–Kellie brain tissue volume
    V_B_total = (params.V_An + params.V_Vn + params.V_Bn + params.V_Cn + params.V_Sn + params.V_Fn) \
                - (V_A_R + V_A_L) - (V_V_R + V_V_L) - params.V_Cn - params.V_Sn - V_F

    return {
        "p_A_R": p_A_R, "p_A_L": p_A_L,
        "p_C_R": p_C_R, "p_C_L": p_C_L,

        "R_AC_R": R_AC_R, "R_AC_L": R_AC_L,
        "R_CV_R": R_CV_R, "R_CV_L": R_CV_L,
        "R_VS_R": R_VS_R, "R_VS_L": R_VS_L,
        "R_CF_R": R_CF_R, "R_CF_L": R_CF_L,
        "R_FS": R_FS,

        "C_VB_R": C_VB_R, "C_VB_L": C_VB_L,
        "C_B": C_B,

        "q_AC_R": q_AC_R, "q_AC_L": q_AC_L,
        "q_CV_R": q_CV_R, "q_CV_L": q_CV_L,
        "q_CF_R": q_CF_R, "q_CF_L": q_CF_L,
        "q_VS_R": q_VS_R, "q_VS_L": q_VS_L,
        "q_FS": q_FS,

        "q_A": q_A, "q_A_R": q_A_R, "q_A_L": q_A_L,

        "V_A_R": V_A_R, "V_A_L": V_A_L,
        "V_B_total": V_B_total,
    }

# ----------------------------------------------------------------------
# Stop event
# ----------------------------------------------------------------------
stop_reason = {"var": None, "value": None}

def stop_if_bad(t, state):
    p_V_R, p_V_L, p_B, C_AB_R, C_AB_L, V_V_R, V_V_L, V_F = state

    vals = {
        "p_V_R": p_V_R,
        "p_V_L": p_V_L,
        "p_B": p_B,
        "V_V_R": V_V_R,
        "V_V_L": V_V_L,
        "C_AB_R": C_AB_R,
        "C_AB_L": C_AB_L,
        # "V_F": V_F,
    }

    var, val = min(vals.items(), key=lambda kv: kv[1])
    stop_reason["var"] = var
    stop_reason["value"] = val
    return val

stop_if_bad.terminal = True
stop_if_bad.direction = -1