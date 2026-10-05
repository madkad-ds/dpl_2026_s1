"""
The cleaning step: the column contract every analysis reads.

Each transformation was established by a check in the validation notebook; this
module applies them in one place, in order, and writes the cleaned parquet. Nothing
here removes rows. Raw columns are left untouched; every fix is a new column or an
in-place repair of a documented sentinel.

Columns added, and the check that motivates each:
  bowler_charge    runs charged to the bowler under the standard scoring law
                   (bat runs, wides, the no-ball penalty; never byes or leg byes).
                   The feed's `bowlruns` nets byes off the bowler on no-balls.
  max_balls        repaired in place where it was 0, a missing-value sentinel, from
                   inns_balls + inns_balls_rem within the innings.
  p_out_clean      the dismissed batter's id only on genuine dismissals; the feed's
                   `p_out` carries the striker on every ball.
  team names       one spelling per team across team_bat, team_bowl, winner, toss.
  control_clean    shot control only on faced balls; the feed leaks values onto wides.
  wagonzone_clean  fielding zone with missing zones filled from real coordinates
                   where the ball was hit; (0, 0) coordinates are a sentinel.
  has_placement    True where wagon-wheel coordinates are genuine.
  bowl_type_group  the 31 bowl_type labels collapsed to nine families plus
                   `unclassified` (untagged) and `other_rare`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dismissal_checks import add_p_out_clean
from .entity_checks import canonicalise_team_names
from .ledger_checks import add_bowler_charge, repair_max_balls
from .missingness_checks import add_control_clean
from .tagging_checks import collapse_bowl_type
from .wagon_wheel_checks import add_has_placement, backfill_wagonzone

ADDED_COLUMNS = ["bowler_charge", "p_out_clean", "control_clean", "wagonzone_clean",
                 "has_placement", "bowl_type_group"]


def apply_cleaning(validated: pd.DataFrame) -> pd.DataFrame:
    """Every repair and derived column, in the order the checks established them."""
    frame = add_bowler_charge(validated)
    frame, _, _ = repair_max_balls(frame)
    frame = add_p_out_clean(frame)
    frame = canonicalise_team_names(frame)
    frame = add_control_clean(frame)
    frame = backfill_wagonzone(frame)
    frame = add_has_placement(frame)
    frame = collapse_bowl_type(frame)
    return frame


def write_clean_dataset(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    return path


def added_columns_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per added column: dtype, non-null count, and distinct values."""
    rows = []
    for col in ADDED_COLUMNS:
        if col in frame.columns:
            rows.append({"column": col, "dtype": str(frame[col].dtype),
                         "non_null": int(frame[col].notna().sum()),
                         "distinct": int(frame[col].nunique(dropna=True))})
    return pd.DataFrame(rows)
