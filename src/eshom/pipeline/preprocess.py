import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def lowpass_butter_zero_phase(x, fs_hz, cutoff_hz, order=4):
    """
    Zero-phase low-pass Butterworth filter (filtfilt), NaN-safe.

    Parameters
    ----------
    x : array-like
        Signal
    fs_hz : float
        Sampling frequency (Hz)
    cutoff_hz : float
        Cutoff frequency (Hz), must be < Nyquist
    order : int
        Filter order

    Returns
    -------
    y : np.ndarray
        Filtered signal
    """
    x = np.asarray(x, dtype=float)
    s = pd.Series(x)

    if s.isna().any():
        s = s.interpolate(limit_direction="both")
    x = s.to_numpy(float)

    nyq = 0.5 * fs_hz
    wn = cutoff_hz / nyq
    if not (0.0 < wn < 1.0):
        raise ValueError(f"cutoff_hz must be in (0, {nyq}). Got cutoff_hz={cutoff_hz}, fs_hz={fs_hz}")

    b, a = butter(order, wn, btype="low")

    padlen = min(3 * (max(len(a), len(b)) - 1), len(x) - 1)
    if padlen < 1:
        return x.copy()

    return filtfilt(b, a, x, padlen=padlen)


def _infer_fs_hz_from_time_numeric(t, time_unit):
    """
    Infer sampling frequency in Hz from a numeric time axis.

    t is numeric and expressed in `time_unit` ("sec" or "min").
    """
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot infer dt from time axis (all NaN/empty).")

    dt_med = float(np.median(dt))
    if dt_med <= 0 or not np.isfinite(dt_med):
        raise ValueError("Non-positive or non-finite dt inferred from time axis.")

    # convert dt to seconds, then to Hz
    if time_unit == "sec":
        dt_seconds = dt_med
    elif time_unit == "min":
        dt_seconds = dt_med * 60.0
    else:
        raise ValueError(f"time_unit must be 'sec' or 'min', got {time_unit!r}")

    fs_hz = 1.0 / dt_seconds
    return fs_hz, dt_med


def prepare_segment(
    df,
    x1,
    x2,
    *,
    time_col,
    time_unit,           # "min" or "sec" (explicit)
    abp_col="ABP",
    icp_col="ICP",
    cutoff_hz=None,     # None = No filtering
    order=4,
    fs_hz=None,          # if None -> inferred from time_col + time_unit
):
    """
    Slice a time window and create filtered ABP/ICP plus dpA/dt.

    Returns a cleaned segment with:
      - ABP_lp
      - ICP_lp
      - dpA_dt_lp   (units: mmHg/min if time_unit="min", else mmHg/s)

    Parameters
    ----------
    df : pd.DataFrame
    x1, x2 : float
        Window bounds in units of `time_col`.

    time_col : str
        Column to slice on ("time_min" or "time_s").

    time_unit : {"min","sec"}
        Declares the unit of `time_col`, used to:
          - infer fs_hz in Hz
          - compute dpA_dt_lp in mmHg/min or mmHg/s

    cutoff_hz : float
        Low-pass cutoff in Hz.

    fs_hz : float | None
        Sampling frequency in Hz. If None, inferred from `time_col` + `time_unit`.

    Returns
    -------
    seg : pd.DataFrame
    """
    if time_unit not in {"min", "sec"}:
        raise ValueError(f"time_unit must be 'min' or 'sec', got {time_unit!r}")
    if time_col not in df.columns:
        raise ValueError(f"time_col='{time_col}' not found in df columns.")

    seg = df[(df[time_col] >= x1) & (df[time_col] <= x2)].copy()
    if seg.empty:
        raise ValueError("Selected segment is empty. Check x1/x2 and time_col units.")

    seg = seg.sort_values(time_col).drop_duplicates(time_col).reset_index(drop=True)

    if cutoff_hz is None:
        seg["ABP_lp"] = seg[abp_col].to_numpy(float)
        seg["ICP_lp"] = seg[icp_col].to_numpy(float)
    else:
        if fs_hz is None:
            fs_hz, _dt_native = _infer_fs_hz_from_time_numeric(
                seg[time_col].to_numpy(), time_unit=time_unit
            )

        seg["ABP_lp"] = lowpass_butter_zero_phase(
            seg[abp_col].to_numpy(float), fs_hz=fs_hz, cutoff_hz=cutoff_hz, order=order
        )
        seg["ICP_lp"] = lowpass_butter_zero_phase(
            seg[icp_col].to_numpy(float), fs_hz=fs_hz, cutoff_hz=cutoff_hz, order=order
        )

    t_native = seg[time_col].to_numpy(float)
    seg["dpA_dt_lp"] = np.gradient(seg["ABP_lp"].to_numpy(float), t_native)

    return seg