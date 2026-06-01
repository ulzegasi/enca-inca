from pathlib import Path
import pandas as pd


def load_patient_csv(out_dir, pid, res="min"):
    """
    Load synced ABP/ICP CSV for one patient at minute or second resolution.

    Parameters
    ----------
    out_dir : str | Path
        Directory containing processed CSV files.

    pid : str
        Patient ID (e.g., "272f2cc")

    res : {"min","sec"}
        Which resolution file to load.

        - "min" -> expects: {pid}__ABP_ICP_minute_synced.csv
                 requires column: time_min
        - "sec" -> expects: {pid}__ABP_ICP_second_synced.csv  (adjust if your naming differs)
                 requires column: time_s  (or time_sec; handled below)

    Returns
    -------
    df : pd.DataFrame
        Sorted by time column. `datetime` parsed if available.
    """
    out_dir = Path(out_dir)

    if res not in {"min", "sec"}:
        raise ValueError(f"res must be 'min' or 'sec', got {res!r}")

    # --- filename pattern (edit here if your second-file name differs) ---
    if res == "min":
        csv_path = out_dir / f"{pid}__ABP_ICP_minute_synced.csv"
        time_col_expected = "time_min"
    else:
        csv_path = out_dir / f"{pid}__ABP_ICP_second_synced.csv"
        time_col_expected = "time_s"  # we will also accept time_sec

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Parse datetime if present
    df = pd.read_csv(csv_path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Basic checks
    required = {"ABP", "ICP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path}")

    # Normalise time column for seconds case
    if res == "sec":
        if "time_s" not in df.columns:
            if "time_sec" in df.columns:
                df = df.rename(columns={"time_sec": "time_s"})
            elif "time_seconds" in df.columns:
                df = df.rename(columns={"time_seconds": "time_s"})

    if time_col_expected not in df.columns:
        raise ValueError(f"Expected time column '{time_col_expected}' in {csv_path.name}")

    # Sort + de-duplicate time
    df = df.sort_values(time_col_expected).drop_duplicates(time_col_expected).reset_index(drop=True)

    return df