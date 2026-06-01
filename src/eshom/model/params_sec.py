import numpy as np
from .params_base import BaseParams

sqrt2 = np.sqrt(2)

params_seconds = BaseParams(

    # ======================================================
    # PRESSURES  [mmHg]
    # ======================================================
    p_An  = 100.0,   # Arterial pressure
    p_V_Rn = 15.0,   # Venous pressure right
    p_V_Ln = 15.0,   # Venous pressure left
    p_Bn  = 10.0,    # Brain tissue pressure
    p_Fn  = 10.0,    # CSF pressure
    p_Cn  = 25.0,    # Capillary pressure
    p_Sn  = 5.0,     # Sagittal sinus pressure


    # ======================================================
    # VOLUMES  [mL]
    # ======================================================
    V_A_Rn = 15.0 / sqrt2,   # Arterial volume right
    V_A_Ln = 15.0 / sqrt2,   # Arterial volume left
    V_V_Rn = 40.0 / sqrt2,   # Venous volume right
    V_V_Ln = 40.0 / sqrt2,   # Venous volume left
    V_Bn   = 1000.0,         # Brain tissue volume
    V_C_Rn = 10.0 / sqrt2,   # Capillary volume right
    V_C_Ln = 10.0 / sqrt2,   # Capillary volume left
    V_Sn   = 80.0,           # Sagittal sinus volume
    V_Fn   = 30.0,           # CSF volume


    # ======================================================
    # FLOWS  [mL/s]
    # ======================================================
    q_ACn_R  = (600.0 / 2) / 60,   # Baseline arterial inflow per hemisphere
    q_ACn_L  = (600.0 / 2) / 60, 
    q_CFn = (0.4 / 2) / 60,     # Baseline CSF production per hemisphere


    # ======================================================
    # RESISTANCES  [mmHg / (mL/s)]
    # ======================================================
    R_ACn = 0.125 * 2 * 60,
    R_VSn = 0.01668 * 2 * 60,
    R_CFn = 37.5 * 2 * 60, 
    R_CVn = 0.01668 * 2 * 60,
    R_FSn = 12.5 * 60,


    # ======================================================
    # CONDUCTANCE  [(mL/s)/mmHg]
    # ======================================================
    g_VSn = ((119.9041 / 2) / 60),


    # ======================================================
    # AUTOREGULATION
    # ======================================================
    alpha_R = 1.4,     # Dimensionless
    tau_R   = 2.5,     # Time constant [seconds]

    alpha_L = 1.4,
    tau_L   = 2.5,


    # ======================================================
    # ARTERIAL COMPLIANCE  [mL/mmHg]
    # ======================================================
    C_ABn_R = 0.15 / sqrt2,
    C_ABn_L = 0.15 / sqrt2,

    dC_AB1_R = 0.119,
    dC_AB2_R = 0.046,
    dC_AB1_L = 0.119,
    dC_AB2_L = 0.046,


    # ======================================================
    # COMPLIANCE MODEL PARAMETERS
    # ======================================================
    k_V  = 0.3,    # 1/mL
    k_B  = 0.26,   # 1/mL
    p_V0 = 2.5,    # mmHg
    p_B0 = 2.5,    # mmHg


    # ======================================================
    # OXYGEN MODEL  (second-based)
    # ======================================================
    n     = 2.6,        # Hill coefficient
    p50   = 26.0,       # mmHg
    beta  = 0.201,      # mL O2 / mL blood
    gamma = 3e-5,       # mL O2 / (mL blood·mmHg)

    r_c = 3.5e-4,       # cm
    r_t = 31e-4,        # cm
    L   = 770e-4,       # cm

    D   = 1.8e3,       # cm²/s
    c   = 2.6e-5,       # mL O2 / (mL tissue·mmHg)
    M   = 4.5e-4,       # mL O2 / (mL tissue·s)
    v_n = 400e-4,       # cm/s
)
