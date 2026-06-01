import numpy as np
import pandas as pd


def split_valid_segments(
    df,
    *,
    time_col,
    abp_col="ABP",
    icp_col="ICP",
    gap_col="gap",
    dt_break=None,          # in SAME units as time_col
    dt_break_min=1.1,       # convenience (minutes) if dt_break is None
    time_unit="min",        # tells us how to convert dt_break_min if needed
):
    """
    Keep only rows with no gaps and non-zero ABP/ICP, then split into continuous segments.

    Segment breaks when dt > dt_break (native units). If dt_break is None, uses dt_break_min.
    """
    x = df.copy().sort_values(time_col).reset_index(drop=True)

    if gap_col in x.columns:
        x = x[~x[gap_col].astype(bool)]

    x = x[(x[abp_col] > 0) & (x[icp_col] > 0)]
    if x.empty:
        return []

    if dt_break is None:
        if time_unit == "min":
            dt_break = float(dt_break_min)
        elif time_unit == "sec":
            dt_break = float(dt_break_min) * 60.0
        else:
            raise ValueError(f"time_unit must be 'min' or 'sec', got {time_unit!r}")
    else:
        dt_break = float(dt_break)

    dt = x[time_col].diff()
    seg_id = (dt > dt_break).cumsum()
    return [g.reset_index(drop=True) for _, g in x.groupby(seg_id)]


def icp_slope_ls(window, *, time_col, icp_col="ICP"):
    """
    Least squares slope of ICP vs time.

    Returns slope in native units:
      - mmHg per time_unit_of_time_col (e.g. mmHg/min or mmHg/sec)
    """
    t = window[time_col].to_numpy(float)
    y = window[icp_col].to_numpy(float)

    t0 = t - t[0]
    A = np.vstack([t0, np.ones_like(t0)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def find_stationary_windows(
    df,
    *,
    time_col,
    time_unit="min",                 # "min" or "sec"
    abp_col="ABP",
    icp_col="ICP",
    gap_col="gap",

    # --- window sizes (native-first) ---
    win=None,                        # native units of time_col (e.g. 30 if minutes, 9000 if seconds)
    step=None,                       # native units of time_col

    # --- optional convenience inputs in minutes ---
    win_min=30.0,
    step_min=5.0,

    # slope threshold specified in mmHg/min by default (common interpretation)
    slope_thresh_mmhg_per_min=0.02,

    min_coverage=0.9,
    abp_std_min=2.0,

    dt_break=None,                   # native units
    dt_break_min=1.1,                # minutes convenience
):
    """
    Return windows where |ICP slope| is small (stationary).

    You control window sizing exactly:
      - If win/step are provided: interpreted in native units (same as time_col)
      - Otherwise: uses win_min/step_min converted using time_unit

    slope_thresh is provided in mmHg/min (human friendly) and converted to native:
      - time_unit="min" -> threshold in mmHg/min (no change)
      - time_unit="sec" -> threshold in mmHg/sec (divide by 60)
    """
    # resolve window sizes
    if win is None:
        win = float(win_min) if time_unit == "min" else float(win_min) * 60.0
    else:
        win = float(win)

    if step is None:
        step = float(step_min) if time_unit == "min" else float(step_min) * 60.0
    else:
        step = float(step)

    # convert slope threshold to native
    if time_unit == "min":
        slope_thresh_native = float(slope_thresh_mmhg_per_min)
    elif time_unit == "sec":
        slope_thresh_native = float(slope_thresh_mmhg_per_min) / 60.0
    else:
        raise ValueError(f"time_unit must be 'min' or 'sec', got {time_unit!r}")

    segments = split_valid_segments(
        df,
        time_col=time_col,
        abp_col=abp_col,
        icp_col=icp_col,
        gap_col=gap_col,
        dt_break=dt_break,
        dt_break_min=dt_break_min,
        time_unit=time_unit,
    )

    out = []

    for seg_i, seg in enumerate(segments):
        seg = seg.sort_values(time_col).reset_index(drop=True)
        t_start = float(seg[time_col].iloc[0])
        t_end   = float(seg[time_col].iloc[-1])

        # estimate expected sample count from median dt (native)
        dt_arr = seg[time_col].diff().dropna().to_numpy(float)
        if dt_arr.size == 0:
            continue
        dt_med = float(np.median(dt_arr))
        if not np.isfinite(dt_med) or dt_med <= 0:
            continue

        expected = int(round(win / dt_med)) + 1
        min_n = int(min_coverage * expected)

        t0 = t_start
        while t0 + win <= t_end:
            t1 = t0 + win
            w = seg[(seg[time_col] >= t0) & (seg[time_col] <= t1)]

            if len(w) >= min_n:
                slope_native = icp_slope_ls(w, time_col=time_col, icp_col=icp_col)
                if abs(slope_native) < slope_thresh_native:
                    abp_std = float(np.std(w[abp_col].to_numpy(float), ddof=1))
                    icp_std = float(np.std(w[icp_col].to_numpy(float), ddof=1))
                    if abp_std >= abp_std_min:
                        # report slope also in mmHg/min for easy comparison
                        slope_mmhg_per_min = slope_native if time_unit == "min" else slope_native * 60.0
                        out.append({
                            "seg_i": seg_i,
                            "t0": float(t0),
                            "t1": float(t1),
                            "win_native": float(win),
                            "step_native": float(step),
                            "icp_slope_native": float(slope_native),
                            "icp_slope_mmhg_per_min": float(slope_mmhg_per_min),
                            "abp_std": abp_std,
                            "icp_std": icp_std,
                            "n": int(len(w)),
                            "datetime_start": w["datetime"].iloc[0] if "datetime" in w.columns else None,
                            "datetime_end": w["datetime"].iloc[-1] if "datetime" in w.columns else None,
                        })

            t0 += step

    res = pd.DataFrame(out)
    if not res.empty:
        res["rank_key"] = res["icp_slope_mmhg_per_min"].abs() + 0.01 * res["icp_std"]
        res = res.sort_values("rank_key").drop(columns=["rank_key"]).reset_index(drop=True)

    return res