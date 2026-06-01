import numpy as np
from .params_base import BaseParams

sqrt2 = np.sqrt(2)

params_minutes = BaseParams(

    # ===============================
    # Pressures  [mmHg]
    # ===============================
    p_An  = 100.0,   # Arterial pressure (mmHg)
    p_V_Rn = 15.0,   # Venous pressure right (mmHg)
    p_V_Ln = 15.0,   # Venous pressure left (mmHg)
    p_Bn  = 10.0,    # Brain tissue pressure (mmHg)
    p_Fn  = 10.0,    # CSF pressure (mmHg)
    p_Cn  = 25.0,    # Capillary pressure (mmHg)
    p_Sn  = 5.0,     # Sagittal sinus pressure (mmHg)

    # ===============================
    # Volumes  [mL]
    # ===============================
    V_A_Rn = 15.0 / sqrt2,   # Arterial volume right (mL)
    V_A_Ln = 15.0 / sqrt2,   # Arterial volume left (mL)
    V_V_Rn = 40.0 / sqrt2,   # Venous volume right (mL)
    V_V_Ln = 40.0 / sqrt2,   # Venous volume left (mL)
    V_Bn   = 1000.0,         # Brain tissue volume (mL)
    V_C_Rn = 10.0 / sqrt2,   # Capillary volume right (mL)
    V_C_Ln = 10.0 / sqrt2,   # Capillary volume left (mL)
    V_Sn   = 80.0,           # Sagittal sinus volume (mL)
    V_Fn   = 30.0,           # CSF volume (mL)

    # ===============================
    # Flows  [mL/min]
    # ===============================
    q_ACn_R  = 600.0 / 2,       # Baseline arterial inflow per hemisphere (mL/min)
    q_ACn_L  = 600.0 / 2,       # Baseline arterial inflow per hemisphere (mL/min)
    q_CFn = 0.4 / 2,         # Baseline CSF production per hemisphere (mL/min)

    # ===============================
    # Resistances  [mmHg / (mL/min)]
    # ===============================
    R_ACn = 0.125 * 2,      # Arterial–capillary resistance
    R_VSn = 0.01668 * 2,    # Venous–sinus resistance
    R_CFn = 37.5 * 2,       # Capillary–CSF formation resistance
    R_CVn = 0.01668 * 2,    # Capillary–venous resistance
    R_FSn = 12.5,           # CSF–sinus absorption resistance

    # ===============================
    # Conductance  [(mL/min)/mmHg]
    # ===============================
    g_VSn = (119.9041 / 2),  # Venous Starling resistor conductance

    # ===============================
    # Autoregulation
    # ===============================
    alpha_R = 1.4,           # Autoregulation strength (dimensionless)
    tau_R   = 2.5 / 60,      # Time constant (minutes)

    alpha_L = 1.4,           # Autoregulation strength (dimensionless)
    tau_L   = 2.5 / 60,      # Time constant (minutes)

    # ===============================
    # Arterial compliance  [mL/mmHg]
    # ===============================
    C_ABn_R = 0.15 / sqrt2,  # Baseline arterial compliance right
    C_ABn_L = 0.15 / sqrt2,  # Baseline arterial compliance left

    dC_AB1_R = 0.119,        # Max dilation change (mL/mmHg)
    dC_AB2_R = 0.046,        # Max constriction change (mL/mmHg)

    dC_AB1_L = 0.119,
    dC_AB2_L = 0.046,

    # ===============================
    # Compliance model parameters
    # ===============================
    k_V  = 0.3,      # Venous compliance nonlinearity (1/mL)
    k_B  = 0.26,     # Brain compliance nonlinearity (1/mL)
    p_V0 = 2.5,      # Venous offset pressure (mmHg)
    p_B0 = 2.5,      # Brain offset pressure (mmHg)

    # ===============================
    # Oxygen model (minute-based)
    # ===============================
    n     = 2.6,          # Hill coefficient (dimensionless)
    p50   = 26.0,         # O2 half-saturation pressure (mmHg)
    beta  = 0.201,        # Hb-bound O2 capacity (mL O2 / mL blood)
    gamma = 3e-5,         # Dissolved O2 (mL O2 / (mL blood·mmHg))

    r_c = 3.5e-4,         # Capillary radius (cm)
    r_t = 31e-4,          # Mean tissue radius (cm)
    L   = 770e-4,         # Capillary length (cm)

    D = 1.8e-5 * 60,      # Diffusion coefficient (cm²/min)
    c = 2.6e-5,           # Tissue solubility (mL O2 / (mL tissue·mmHg))
    M = 4.5e-4 * 60,      # Tissue O2 consumption (mL O2 / (mL tissue·min))
    v_n = (400e-4) * 60,  # Baseline blood velocity (cm/min)
)