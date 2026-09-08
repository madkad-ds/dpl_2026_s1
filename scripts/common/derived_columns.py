"""
Derived-column contract for the ball-by-ball frame (real-schema conventions).

One place, one definition, for every flag the phase-profile rates are built from.
Conventions match the validated season-contrast harness and the confirmed data
dictionary, so a rate built here lines up with the register's Thread-1 findings.

Denominator split (this is the load-bearing decision, from the data dictionary):
  - run rate  : numerator `score` (total incl. extras) over LEGAL balls (wide==0 &
                noball==0), x6. `score` because team scoring includes extras and the
                validated run-rate findings used it; legal balls because a wide or
                no-ball adds runs without advancing the over.
  - pace share: over LEGAL balls ("share of legal balls bowled by pace").
  - boundary / six / dot / aggressive : striker-facing rates over FACED balls
                (`ballfaced == 1`). A no-ball is faced (the batter can score off it);
                a wide is not. `ballfaced` encodes that distinction and
                `wide==0 & noball==0` does not, so faced rates key off `ballfaced`.

Value-source conventions:
  - boundary/six/dot from `batruns` (runs off the bat), NOT `score`: `score` would
    miscount a bye or leg-bye (both populated in the real feed) as batting activity.
  - wicket from `out` (never `p_out`).
  - control from `control_clean` (raw `control` leaks onto wides).
  - pace from `bowl_kind` by substring "pace" (real set: "pace bowler",
    "spin bowler", "mixture/unknown"; substring is robust to the trailing word and
    correctly leaves mixture/unknown out of the pace count).

Every function returns a mask/Series aligned to the input index and mutates nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Aggressive shot vocabulary, identical to the powerplay-stratification definition
# so the tagged layer here is comparable with that finding. Drives and orthodox
# sweeps are excluded on purpose: a drive is often controlled and blurs intent.
AGGRESSIVE_SHOTS = frozenset(
    {"PULL", "SLOG_SHOT", "SLOG_SWEEP", "HOOK", "UPPER_CUT", "RAMP", "REVERSE_SCOOP", "REVERSE_PULL"}
)


def is_legal_delivery(frame: pd.DataFrame) -> pd.Series:
    """Counts toward the over: not a wide, not a no-ball. Run-rate/pace denominator."""
    wide = pd.to_numeric(frame["wide"], errors="coerce").fillna(0)
    noball = pd.to_numeric(frame["noball"], errors="coerce").fillna(0)
    return (wide == 0) & (noball == 0)


def is_ball_faced(frame: pd.DataFrame) -> pd.Series:
    """Striker faced this delivery (excludes wides, includes faced no-balls).

    Reads `ballfaced` directly rather than reconstructing from wide/noball, because
    a no-ball is faced while a wide is not, and only `ballfaced` encodes that.
    """
    return pd.to_numeric(frame["ballfaced"], errors="coerce").fillna(0) == 1


def is_six(frame: pd.DataFrame) -> pd.Series:
    """Six off the bat: outcome=='six' or batruns==6 (matches harness)."""
    by_runs = pd.to_numeric(frame["batruns"], errors="coerce") == 6
    if "outcome" in frame.columns:
        by_outcome = frame["outcome"].astype("string").str.lower().eq("six")
        return (by_outcome | by_runs).fillna(False)
    return by_runs.fillna(False)


def is_four(frame: pd.DataFrame) -> pd.Series:
    """Four off the bat, from batruns."""
    return (pd.to_numeric(frame["batruns"], errors="coerce") == 4).fillna(False)


def is_boundary(frame: pd.DataFrame) -> pd.Series:
    """Four or six off the bat."""
    return is_four(frame) | is_six(frame)


def is_dot_off_bat(frame: pd.DataFrame) -> pd.Series:
    """Dot off the bat: a faced legal ball the batter scored nothing off (batruns==0).

    batruns, not score: a ball that ran a bye or leg-bye has batruns==0 but score>0,
    and it is still a dot from the batter's perspective. Restricted to legal faced
    balls so a no-ball (never a dot) is excluded from the numerator.
    """
    faced = is_ball_faced(frame)
    legal = is_legal_delivery(frame)
    zero_bat = pd.to_numeric(frame["batruns"], errors="coerce").fillna(-1) == 0
    return faced & legal & zero_bat


def is_wicket(frame: pd.DataFrame) -> pd.Series:
    """Batter dismissal on this ball, from `out` (never `p_out`)."""
    if "out" in frame.columns:
        return pd.to_numeric(frame["out"], errors="coerce").fillna(0).astype(bool)
    if "dismissal" in frame.columns:
        return frame["dismissal"].astype("string").fillna("").str.strip().ne("")
    raise KeyError("neither `out` nor `dismissal` present: cannot derive wickets")


def is_pace(frame: pd.DataFrame) -> pd.Series:
    """Pace delivery: `bowl_kind` contains 'pace'. mixture/unknown -> neither pace nor spin."""
    return frame["bowl_kind"].astype("string").str.contains("pace", case=False, na=False)


def is_spin(frame: pd.DataFrame) -> pd.Series:
    """Spin delivery: `bowl_kind` contains 'spin'."""
    return frame["bowl_kind"].astype("string").str.contains("spin", case=False, na=False)


def is_aggressive_shot(frame: pd.DataFrame) -> pd.Series:
    """Aggressive shot from the tagged `shot` column. Carries the 7-10% tagging floor."""
    shot = frame["shot"].astype("string").str.upper().str.strip()
    return shot.isin(AGGRESSIVE_SHOTS).fillna(False)


def control_lost(frame: pd.DataFrame) -> pd.Series:
    """Loss of batter control from `control_clean` (raw `control` leaks on wides).

    Nullable float: NaN where control was not recorded, so a caller excludes those
    balls from a control rate rather than scoring them as controlled.
    """
    if "control_clean" not in frame.columns:
        raise KeyError("`control_clean` absent; refuse to fall back to raw `control` (wide leak)")
    control = pd.to_numeric(frame["control_clean"], errors="coerce")
    return 1.0 - control
