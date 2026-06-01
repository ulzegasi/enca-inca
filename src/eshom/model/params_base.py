from dataclasses import dataclass, replace
import numpy as np

@dataclass(frozen=True)
class BaseParams:
    # Pressures (mmHg)
    p_An: float
    p_V_Rn: float
    p_V_Ln: float
    p_Bn: float
    p_Fn: float
    p_Cn: float
    p_Sn: float

    # Volumes (mL)
    V_A_Rn: float
    V_A_Ln: float
    V_V_Rn: float
    V_V_Ln: float
    V_Bn: float
    V_C_Rn: float
    V_C_Ln: float
    V_Sn: float
    V_Fn: float

    # Flows
    q_ACn_R: float
    q_ACn_L: float
    q_CFn: float

    # Resistances
    R_ACn: float
    R_VSn: float
    R_CFn: float
    R_CVn: float
    R_FSn: float

    # Starling conductance
    g_VSn: float

    # Autoregulation
    alpha_R: float
    tau_R: float
    alpha_L: float
    tau_L: float

    # Compliance parameters
    C_ABn_R: float
    C_ABn_L: float
    dC_AB1_R: float
    dC_AB2_R: float
    dC_AB1_L: float
    dC_AB2_L: float
    k_V: float
    k_B: float
    p_V0: float
    p_B0: float

    # Oxygen parameters
    n: float
    p50: float
    beta: float
    gamma: float
    r_c: float
    r_t: float
    L: float
    D: float
    c: float
    M: float
    v_n: float

    # Update without mutating globals
    def with_updates(self, **kwargs) -> "BaseParams":
        return replace(self, **kwargs)
    
    # Derived scalar totals / constants
    @property
    def p_S(self) -> float:
        return self.p_Sn

    @property
    def V_An(self) -> float:
        return self.V_A_Rn + self.V_A_Ln

    @property
    def V_Vn(self) -> float:
        return self.V_V_Rn + self.V_V_Ln

    @property
    def V_Cn(self) -> float:
        return self.V_C_Rn + self.V_C_Ln

    @property
    def k_RAC_R(self) -> float:
        return self.R_ACn * (self.C_ABn_R ** 2) * (self.p_An - self.p_Bn) ** 2

    @property
    def k_RAC_L(self) -> float:
        return self.R_ACn * (self.C_ABn_L ** 2) * (self.p_An - self.p_Bn) ** 2

    @property
    def k_RVS_R(self) -> float:
        return self.R_VSn * (self.V_V_Rn ** 2)

    @property
    def k_RVS_L(self) -> float:
        return self.R_VSn * (self.V_V_Ln ** 2)


    # Oxygen grids + precomputed geometry
    @property
    def z_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.L, int(self.z_points))

    @property
    def r_grid(self) -> np.ndarray:
        r_mean = self.r_t
        r_min = self.r_c * self.r_min_factor
        r_max = r_mean * self.r_max_factor
        return np.linspace(r_min, r_max, int(self.r_points))

    @property
    def weights(self) -> np.ndarray:
        r = self.r_grid
        r_mean = self.r_t
        mu_r = r_mean
        sigma_r = r_mean

        phi_raw = np.exp(-(r - mu_r) ** 2 / (2.0 * sigma_r ** 2))
        vol_factor = r ** 2 * self.L

        w = phi_raw * vol_factor
        denom = np.trapezoid(w, r)
        if denom == 0 or not np.isfinite(denom):
            raise ValueError("Weights normalisation failed (denom=0 or non-finite).")
        return w / denom

    @property
    def geom_b(self) -> np.ndarray:
        r = self.r_grid
        rc2 = self.r_c ** 2
        rt2 = r ** 2
        return (rt2 - rc2) / rc2

    @property
    def delta_r(self) -> np.ndarray:
        r = self.r_grid
        rc2 = self.r_c ** 2
        rc4 = rc2 ** 2
        rt2 = r ** 2
        rt4 = rt2 ** 2

        num = (
            -rc4 + 4.0 * rc2 * rt2 - 3.0 * rt4
            + 4.0 * rt4 * np.log(r / self.r_c)
        )
        den = (rt2 - rc2)
        den = np.where(den == 0, np.nan, den)

        delta = (self.M / (8.0 * self.D * self.c)) * (num / den)
        if np.any(~np.isfinite(delta)):
            raise ValueError("delta_r contains non-finite values. Check r_grid bounds.")
        return delta