# project/main.py
from pathlib import Path
import matplotlib.pyplot as plt

from eshom.model.params_min import params_minutes
from eshom.model.params_sec import params_seconds

from eshom.pipeline.io import load_patient_csv
from eshom.pipeline.preprocess import prepare_segment
from eshom.pipeline.windows import find_stationary_windows
from eshom.pipeline.fit import fit_control_segment
from eshom.pipeline.simulate import run_simulation
from eshom.pipeline.plots import plot_abp_cpp_icp_residual

def main():
    # -----------------------
    # User settings
    # -----------------------

    # Set time resolution to minutes or seconds
    res = "min"           # "min" or "sec"ß
    time_col = "time_min" if res == "min" else "time_s"
    time_unit = res       # "min" or "sec"

    # Set path to data folder
    BASE_DATA_DIR = Path("/cfs/earth/scratch/durreayl/ZHAW_esHOM/Data")

    # Set directory to selected resolution
    if res == "min":
        OUT_DIR = BASE_DATA_DIR / "SAB_minutes"
    else:
        OUT_DIR = BASE_DATA_DIR / "SAB_seconds"


    pid = "272f2cc" # choose which patient 

    # choose params that match your solver time base
    params = params_minutes if res == "min" else params_seconds 

    # pick segment strategy
    use_stationary_window = False   # set True if you want auto window selection

    # window selection (native units)
    win = 120 if res == "min" else 7200     # example: 120 min OR 9000 s 
    step = 5 if res == "min" else 300       # example: 5 min OR 300 s

    # filtering
    # (cutoff is in Hz; adjust depending on res)
    cutoff_hz = 0.003 if res == "min" else 0.05  # defaults
    order = 4

    # manual segment bounds (native units)
    x1 = 16010 
    x2 = x1 + 30

    if res == "sec":
        x1 = x1 * 60
        x2 = x2 * 60 

    # -----------------------
    # Load patient data
    # -----------------------
    df = load_patient_csv(OUT_DIR, pid, res=res)
    #df = load_patient_csv(BASE_DATA_DIR / "SAB_seconds", pid, res="sec")
    print(df.head())
    print("rows:", len(df))

    # -----------------------
    # Select a window
    # -----------------------
    if use_stationary_window:
        stationary = find_stationary_windows(
            df,
            time_col=time_col,
            time_unit=time_unit,
            win=win,
            step=step,
            slope_thresh_mmhg_per_min=0.02,
        )
        if stationary.empty:
            raise RuntimeError("No stationary windows found.")
        pick = stationary.iloc[0]
        x1 = float(pick["t0"])
        x2 = float(pick["t1"])
        print(f"Using stationary window: [{x1}, {x2}] ({res})")

    # -----------------------
    # Prepare (filter + dpA/dt)
    # -----------------------
    seg = prepare_segment(
        df,
        x1=x1,
        x2=x2,
        time_col=time_col,
        time_unit=time_unit,
        cutoff_hz=cutoff_hz,
        order=order,
    )

    # -----------------------
    # Initial state from params
    # -----------------------
    initial_state = [
        params.p_V_Rn, params.p_V_Ln, params.p_Bn,
        params.C_ABn_R, params.C_ABn_L,
        params.V_V_Rn, params.V_V_Ln,
        params.V_Fn,
    ]

    # -----------------------
    # Fit
    # -----------------------
    
    res_fit, fitted = fit_control_segment(
        seg,
        params,
        t_col=time_col,
        initial_state=initial_state,
        fit_k_B=True,
        fit_R_FSn=True,
        max_nfev=200,
        verbose=2,
    )
    print("Fitted params:")
    print(fitted)
    
    # -----------------------
    # Use stored fitted params (no fitting)
    # -----------------------
    '''
    if res == "min":
        fitted = {
            'alpha_R': 0.4032583402796699,
            'alpha_L': 0.4032583402796699,
            'tau_R': 0.013727077608837217,
            'tau_L': 0.013727077608837217,
            'dC_AB1_R': 0.1244359438975132,
            'dC_AB1_L': 0.1244359438975132,
            'dC_AB2_R': 0.014290304167147449,
            'dC_AB2_L': 0.014290304167147449,
            'k_B': 0.033784046385415246,
            'R_FSn': 30.816398699605305,
        }
    else:
        fitted = {
            'alpha_R': 0.5292671639240702,
            'alpha_L': 0.5292671639240702,
            'tau_R': 1.119300545574891,
            'tau_L': 1.119300545574891,
            'dC_AB1_R': 0.2723824576800598,
            'dC_AB1_L': 0.2723824576800598,
            'dC_AB2_R': 0.06400740076772375,
            'dC_AB2_L': 0.06400740076772375,
            'k_B': 0.0837479496503939,
            'R_FSn': 1464.7427723805783,
        }

    print("Using stored fitted params:")
    print(fitted)
    '''
    # -----------------------
    # Simulate with fitted
    # -----------------------
    sol = run_simulation(
        seg,
        params,
        t_col=time_col,
        initial_state=initial_state,
        fitted=fitted,
        method="BDF",
        rtol=1e-5,
        atol=1e-7,
        use_events=True,
    )

    print("sol.success:", sol.success)
    print("sol.status:", sol.status)
    print("sol.message:", sol.message)
    print("sol.t[0]:", sol.t[0])
    print("sol.t[-1]:", sol.t[-1])
    print("seg start:", seg[time_col].iloc[0])
    print("seg end:", seg[time_col].iloc[-1])

    if hasattr(sol, "t_events"):
        print("t_events:", sol.t_events)

    # -----------------------
    # Plot
    # -----------------------
    fig = plot_abp_cpp_icp_residual(
        seg,
        sol,
        t_col=time_col,
        xlabel="Time (min)" if res == "min" else "Time (s)",
    )
    out_path= Path("test_plot.png")
    fig.savefig(out_path)
    print(f"Plot saved to {out_path}")



if __name__ == "__main__":
    main()