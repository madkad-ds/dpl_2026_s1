"""
Result writing with provenance.

Every table goes out as CSV plus a `.meta.json` sidecar carrying the seed, the
bootstrap settings, the run timestamp, the producing script, and the versions of the
numerical libraries. The sidecar is the "seed in the footer" of the reproducibility
package: a table read on its own can be traced back to the exact run that made it.
Figures are saved as PNG with the same sidecar.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Settings

VERSIONED_PACKAGES = ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "pymc", "arviz",
                      "ruptures", "matplotlib")


def library_versions() -> dict:
    """Versions of the numerical stack actually importable in this environment."""
    versions = {"python": platform.python_version()}
    for name in VERSIONED_PACKAGES:
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:  # not installed or no __version__; record the absence
            versions[name] = None
    return versions


def _sidecar(settings: Settings, analysis: str, seed: int, producer: str,
             extra: dict | None) -> dict:
    return {
        "analysis": analysis,
        "seed": int(seed),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": producer,
        "bootstrap": settings.section("bootstrap"),
        "seasons": settings.section("seasons"),
        "versions": library_versions(),
        "command": " ".join(sys.argv) if sys.argv else "",
        **(extra or {}),
    }


def write_table(table: pd.DataFrame, settings: Settings, analysis: str, name: str,
                seed: int, producer: str, extra: dict | None = None) -> Path:
    """Write `name.csv` and `name.meta.json` under outputs/<analysis>/."""
    folder = settings.output_dir(analysis)
    csv_path = folder / f"{name}.csv"
    table.to_csv(csv_path, index=False)
    with open(folder / f"{name}.meta.json", "w") as handle:
        json.dump(_sidecar(settings, analysis, seed, producer, extra), handle, indent=2)
    return csv_path


def write_json(payload: dict, settings: Settings, analysis: str, name: str,
               seed: int, producer: str, extra: dict | None = None) -> Path:
    """Write a nested result (not tabular) as `name.json` with the same sidecar."""
    folder = settings.output_dir(analysis)
    path = folder / f"{name}.json"
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=_jsonable)
    with open(folder / f"{name}.meta.json", "w") as handle:
        json.dump(_sidecar(settings, analysis, seed, producer, extra), handle, indent=2)
    return path


def write_figure(fig, settings: Settings, analysis: str, name: str, seed: int,
                 producer: str, dpi: int = 200) -> Path:
    folder = settings.output_dir(analysis)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    with open(folder / f"{name}.meta.json", "w") as handle:
        json.dump(_sidecar(settings, analysis, seed, producer, None), handle, indent=2)
    return path


def _jsonable(value):
    """json.dump default: numpy scalars and arrays, pandas objects."""
    import numpy as np
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    raise TypeError(f"not JSON serialisable: {type(value)}")
