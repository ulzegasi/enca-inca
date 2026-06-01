from __future__ import annotations

from typing import Any

import pandas as pd

from eshom.pipeline.preprocess import prepare_segment
from eshom.pipeline.fit import fit_control_segment
from eshom.pipeline.simulate import run_simulation


def datetime_to_time_min(
    df: pd.DataFrame,
    dt_str: str,
    datetime_col: str = "datetime",
    time_col: str = "time_min",
    local_tz: str = "Europe/Zurich",
) -> float:
    """
    Convert a datetime string to the nearest value in df[time_col].

    Parameters
    ----------
    df : pd.DataFrame
        Must contain datetime_col and time_col.
    dt_str : str
        Target datetime string, optionally timezone-aware.
    datetime_col : str
        Name of datetime column in df.
    time_col : str
        Name of time column in df.
    local_tz : str
        Time zone used to convert timezone-aware timestamps before matching.

    Returns
    -------
    float
        Nearest time value in minutes.
    """
    if datetime_col not in df.columns:
        raise KeyError(f"Missing required column: '{datetime_col}'")
    if time_col not in df.columns:
        raise KeyError(f"Missing required column: '{time_col}'")

    target_dt = pd.to_datetime(dt_str)

    # Convert tz-aware timestamps to naive local time
    if getattr(target_dt, "tzinfo", None) is not None:
        target_dt = target_dt.tz_convert(local_tz).tz_localize(None)

    idx = (df[datetime_col] - target_dt).abs().idxmin()
    return float(df.loc[idx, time_col])


def _get_regime_window(
    regime_name: str,
    t: float,
) -> tuple[float, float]:
    """
    Return (x1, x2) in minutes for the requested regime.
    """
    if regime_name == "hydro":
        return t, t + 45.0

    if regime_name == "no_hydro":
        return t, t + 45.0

    if regime_name == "vaso_onset":
        return t, t + 30.0

    if regime_name == "vaso_treatment":
        return t - 15.0, t + 15.0

    if regime_name == "vaso_resolve":
        return t, t + 45.0

    raise ValueError(f"Unknown regime: {regime_name}")


def _check_segment_quality(
    seg: pd.DataFrame,
    *,
    time_col: str = "time_min",
    min_coverage_frac: float = 0.9,
    min_nonzero_frac: float = 0.3,
    min_unique_vals: int = 5,
    min_icp_range: float = 3.0,
) -> dict[str, Any] | None:
    """
    Return quality info dict if segment is valid, otherwise None.
    """
    if seg.empty or len(seg) < 2:
        return None

    if "ICP_lp" not in seg.columns:
        return None

    dt_est = seg[time_col].diff().dropna().median()
    if pd.isna(dt_est) or dt_est <= 0:
        return None

    window_len_min = float(seg[time_col].iloc[-1] - seg[time_col].iloc[0])
    expected_points = int(round(window_len_min / dt_est)) + 1
    coverage_frac = len(seg) / expected_points if expected_points > 0 else 0.0

    if coverage_frac < min_coverage_frac:
        return None

    icp = seg["ICP_lp"].astype(float).dropna()
    if icp.empty:
        return None

    nonzero_frac = float((icp.abs() > 1e-3).mean())
    unique_vals = int(icp.round(3).nunique())
    icp_range = float(icp.max() - icp.min())

    if nonzero_frac < min_nonzero_frac:
        return None
    if unique_vals < min_unique_vals:
        return None
    if icp_range < min_icp_range:
        return None

    return {
        "expected_points": expected_points,
        "coverage_frac": coverage_frac,
        "icp_nonzero_frac": nonzero_frac,
        "icp_unique_vals": unique_vals,
        "icp_range": icp_range,
    }


def fit_and_simulate_regime(
    df: pd.DataFrame,
    params: Any,
    *,
    regime_name: str,
    event_dt: str,
    time_col: str = "time_min",
    time_unit: str = "min",
    datetime_col: str = "datetime",
    cutoff_hz: float = 0.003,
    order: int = 4,
    search_radius_min: int = 720,
    step_min: int = 15,
    manual_offset_min: float | None = None,
    local_tz: str = "Europe/Zurich",
) -> tuple[dict[str, Any], pd.DataFrame, Any, Any]:
    """
    Find the first valid segment near an event, fit model parameters, and simulate.

    Returns
    -------
    row : dict
        Metadata + fitted parameter summary.
    seg : pd.DataFrame
        Segment used for simulation.
    sol : Any
        Output of run_simulation().
    res_fit : Any
        Output of fit_control_segment().
    """
    event_t = datetime_to_time_min(
        df,
        event_dt,
        datetime_col=datetime_col,
        time_col=time_col,
        local_tz=local_tz,
    )

    if manual_offset_min is not None:
        offsets = [manual_offset_min]
    else:
        offsets = [0]
        k = step_min
        while k <= search_radius_min:
            offsets.extend([-k, k])
            k += step_min

    seg = None
    x1 = x2 = offset_min = None
    quality_info = None

    for offset in offsets:
        t = event_t + offset
        cand_x1, cand_x2 = _get_regime_window(regime_name, t)

        try:
            cand_seg = prepare_segment(
                df,
                x1=cand_x1,
                x2=cand_x2,
                time_col=time_col,
                time_unit=time_unit,
                cutoff_hz=cutoff_hz,
                order=order,
            )
        except Exception:
            continue

        quality = _check_segment_quality(cand_seg, time_col=time_col)
        if quality is None:
            continue

        seg = cand_seg
        x1, x2 = cand_x1, cand_x2
        offset_min = offset
        quality_info = quality
        break

    if seg is None:
        raise RuntimeError(
            f"No valid segment found within ±{search_radius_min} min "
            f"for regime '{regime_name}' around event {event_dt}"
        )

    initial_state = [
        params.p_V_Rn,
        params.p_V_Ln,
        params.p_Bn,
        params.C_ABn_R,
        params.C_ABn_L,
        params.V_V_Rn,
        params.V_V_Ln,
        params.V_Fn,
    ]

    seg_fit = seg.copy()

    if regime_name == "hydro":
        seg_fit = seg_fit[seg_fit[time_col] > seg_fit[time_col].iloc[0] + 5].copy()
        if len(seg_fit) < 5:
            raise RuntimeError("Hydro segment too short after trimming")

    res_fit, fitted = fit_control_segment(
        seg_fit,  # important: use seg_fit, not seg
        params,
        t_col=time_col,
        initial_state=initial_state,
        fit_k_B=False,
        fit_q_ACn=True,
        fit_R_FSn=False,
        max_nfev=200,
        verbose=0,
    )

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

    row = {
        "regime": regime_name,
        "event_dt": event_dt,
        "window_start": seg[datetime_col].iloc[0],
        "window_end": seg[datetime_col].iloc[-1],
        "x1_min": x1,
        "x2_min": x2,
        "window_len_min": x2 - x1,
        "offset_from_event_min": offset_min,
        "n_points": len(seg),
        "expected_points": quality_info["expected_points"],
        "coverage_frac": quality_info["coverage_frac"],
        "icp_nonzero_frac": quality_info["icp_nonzero_frac"],
        "icp_unique_vals": quality_info["icp_unique_vals"],
        "icp_range": quality_info["icp_range"],
    }
    row.update(fitted)

    return row, seg, sol, res_fit