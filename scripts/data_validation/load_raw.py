"""
Raw dataset loading: CSV to a schema-validated parquet.

The provided CSV is read once, validated against the pandera schema (dtype,
nullability and legal values per column, all failures collected together), enriched
with a real datetime and category dtypes, and written to parquet. Later calls read
the parquet if it is newer than both the CSV and the schema module, so the
validation cost is paid once and a schema edit forces a rebuild.

`match_date` arrives either as an Excel serial number (days since 1899-12-30, the
data dictionary's description) or as an ISO date string; both are handled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cricket_schema import schema

# Fixed-label columns stored as `category` (smaller file, faster groupby).
CATEGORY_COLS = ["team_bat", "team_bowl", "ground", "country", "bat_hand", "bowl_kind", "outcome"]
EXCEL_EPOCH = "1899-12-30"


def parquet_is_fresh(parquet: Path, *sources: Path) -> bool:
    """True if the parquet exists and is newer than every source file."""
    if not parquet.exists():
        return False
    mtime = parquet.stat().st_mtime
    return all(s.exists() and mtime >= s.stat().st_mtime for s in sources)


def match_date_to_datetime(match_date: pd.Series) -> pd.Series:
    """Excel serial numbers or ISO strings to datetime."""
    numeric = pd.to_numeric(match_date, errors="coerce")
    if numeric.notna().all():
        return pd.to_datetime(numeric, unit="D", origin=EXCEL_EPOCH)
    return pd.to_datetime(match_date, errors="coerce")


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    """Post-validation: real datetime from match_date plus category dtypes."""
    out = frame.copy()
    out["match_date_dt"] = match_date_to_datetime(out["match_date"])
    for col in CATEGORY_COLS:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def load_raw_dataset(csv_path: Path, parquet_path: Path, rebuild: bool = False) -> pd.DataFrame:
    """Read the validated parquet if fresh; else validate the CSV and rewrite it."""
    schema_path = Path(__file__).with_name("cricket_schema.py")
    if not rebuild and parquet_is_fresh(parquet_path, csv_path, schema_path):
        return pd.read_parquet(parquet_path, engine="pyarrow")
    raw = pd.read_csv(csv_path)
    validated = enrich(schema.validate(raw, lazy=True))   # coerce + lazy come from the schema
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(parquet_path, engine="pyarrow", index=False)
    return validated
