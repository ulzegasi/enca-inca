import numpy as np
import matplotlib.pyplot as plt


def plot_abp_cpp_icp_residual(
    seg,
    sol,
    *,
    t_col,
    title=None,
    abp_col="ABP_lp",
    icp_col="ICP_lp",
    show_abp=True,          # toggle ABP line on the top panel
    shift_to_one=True,      # shift time axis to start at 1
    xlabel=None,            # e.g. "Time (min)" or "Time (s)"
):
    """
    3-panel plot:
      1) ABP (optional) + CPP observed/predicted (twin y-axis)
      2) ICP observed vs predicted
      3) Residual (pred - obs)

    Assumes ICP model is sol.y[2,:] == p_B.
    """
    if not sol.success:
        raise RuntimeError(f"Simulation failed: {sol.message}")

    t = seg[t_col].to_numpy(float)
    pA = seg[abp_col].to_numpy(float)
    icp_obs = seg[icp_col].to_numpy(float)

    t_model = sol.t
    pB_raw = sol.y[2, :]
    pB_model = np.interp(t, t_model, pB_raw)

    # errors
    err = pB_model - icp_obs
    rmse = float(np.sqrt(np.mean(err**2)))

    # CPP
    cpp_obs = pA - icp_obs
    cpp_model = pA - pB_model

    # time shift
    if shift_to_one:
        t_plot = t - t[0] + 1
    else:
        t_plot = t

    if xlabel is None:
        xlabel = f"Time ({t_col})" + (" shifted" if shift_to_one else "")

    fig, (ax_abp, ax_icp, ax_res) = plt.subplots(
        3, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2, 0.8]},
    )

    if title is None:
        title = f"Model–Data Agreement: RMSE = {rmse:.2f} mmHg"
    ax_abp.set_title(title)

    # -------------------------
    # 1 — ABP (optional) + CPP
    # -------------------------
    if show_abp:
        ax_abp.plot(t_plot, pA, color = "red", linewidth=2, label="ABP")
        ax_abp.set_ylabel("ABP (mmHg)")
        ax_abp.grid(alpha=0.3)

    ax_cpp = ax_abp.twinx()
    ax_cpp.plot(t_plot, cpp_obs, linewidth=1.8, label="CPP observed")
    ax_cpp.plot(t_plot, cpp_model, linestyle="--", linewidth=1.8, label="CPP predicted")
    ax_cpp.set_ylabel("CPP (mmHg)")
    ax_cpp.grid(alpha=0.3)

    # combine legends from both axes
    lines1, labels1 = ax_abp.get_legend_handles_labels()
    lines2, labels2 = ax_cpp.get_legend_handles_labels()
    if lines1 or lines2:
        ax_abp.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    # nice x-limits (like your notebook)
    ax_abp.set_xlim(t_plot[0], t_plot[-1])

    # -------------------------
    # 2 — ICP
    # -------------------------
    ax_icp.plot(t_plot, icp_obs, linewidth=2, label="ICP observed")
    ax_icp.plot(t_plot, pB_model, linestyle="--", linewidth=2, label="ICP predicted")
    ax_icp.set_ylabel("ICP (mmHg)")
    ax_icp.legend(loc="upper left")
    ax_icp.grid(alpha=0.3)

    # -------------------------
    # 3 — Residual
    # -------------------------
    ax_res.plot(t_plot, err, linewidth=1.5)
    ax_res.axhline(0, linestyle="--")
    ax_res.set_xlabel(xlabel)
    ax_res.set_ylabel("Residual (mmHg)")
    ax_res.set_xlim(t_plot[0], t_plot[-1])
    ax_res.grid(alpha=0.3)

    plt.tight_layout()
    return fig