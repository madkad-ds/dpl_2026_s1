"""
Delivery classes for the bowling-side analysis.

The tagged line and length columns have five and six values; at 170 balls a cell a
line-by-length grid cannot carry a season contrast, so both are coarsened to three
levels. Speed and release-point bands are tracked (blank on wides and no-balls and on
about 3% of legal balls, the same in both seasons) and are bowled almost entirely by
pace, so those two dimensions are read on pace deliveries only. The type family is
the nine-group collapse of the bowl-type tag made in the validation step.

Classes, per dimension:
  length3   full (full, yorker, full toss) / good / short (short, short of a good length)
  line3     off (outside off, wide outside off) / stumps / leg (down leg, wide down leg)
  line_length  the nine combinations, as "full|off" etc.
  speed3    over 140 / 130 to 140 / under 130 (pace only)
  release2  over 2.0 m / 2.0 m or lower (pace only)
  type      bowl_type_group (nine families plus unclassified and other_rare)

A class value of "untagged" marks a delivery whose tag is blank; it is kept as its
own class so the shares still sum to one and a tagging gap cannot leak into a mix
change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LENGTH3 = {"FULL": "full", "YORKER": "full", "FULL_TOSS": "full", "GOOD_LENGTH": "good",
           "SHORT_OF_A_GOOD_LENGTH": "short", "SHORT": "short"}
LINE3 = {"OUTSIDE_OFFSTUMP": "off", "WIDE_OUTSIDE_OFFSTUMP": "off", "ON_THE_STUMPS": "stumps",
         "DOWN_LEG": "leg", "WIDE_DOWN_LEG": "leg"}
SPEED3 = {1: "over_140", 2: "130_to_140", 3: "under_130", 4: "under_130", 5: "under_130",
          6: "under_130", 7: "under_130", 8: "under_130"}
RELEASE2 = {1: "over_2.0m", 2: "2.0m_or_lower", 3: "2.0m_or_lower", 4: "2.0m_or_lower"}

DIMENSIONS = {
    "line_length": dict(pace_only=False, order=None),
    "length3": dict(pace_only=False, order=["full", "good", "short"]),
    "line3": dict(pace_only=False, order=["off", "stumps", "leg"]),
    "type": dict(pace_only=False, order=None),
    "speed3": dict(pace_only=True, order=["over_140", "130_to_140", "under_130"]),
    "release2": dict(pace_only=True, order=["over_2.0m", "2.0m_or_lower"]),
}


def assign_classes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add one column per dimension to a copy of the frame."""
    out = frame.copy()
    length = out["length"].astype("string").str.upper().map(LENGTH3)
    line = out["line"].astype("string").str.upper().map(LINE3)
    out["length3"] = length.fillna("untagged")
    out["line3"] = line.fillna("untagged")
    out["line_length"] = np.where((length.notna() & line.notna()).to_numpy(),
                                  out["length3"].astype(str) + "|" + out["line3"].astype(str), "untagged")
    speed = pd.to_numeric(out["bowl_speed_category"], errors="coerce")
    release = pd.to_numeric(out["bowl_release_point"], errors="coerce")
    out["speed3"] = speed.map(SPEED3).fillna("untagged")
    out["release2"] = release.map(RELEASE2).fillna("untagged")
    out["type"] = out["bowl_type_group"].astype(str) if "bowl_type_group" in out.columns else "untagged"
    return out


def coverage_by_season(frame: pd.DataFrame, phase_mask: np.ndarray) -> pd.DataFrame:
    """Blank rates of the five raw tag columns in the phase, by season and bowler kind.
    A season difference here would make a mix change out of a tagging change."""
    cols = ["line", "length", "bowl_type", "bowl_speed_category", "bowl_release_point"]
    sub = frame.loc[phase_mask]
    by_season = sub.groupby("year")[cols].apply(lambda g: g.isna().mean())
    by_kind = sub.groupby(["year", "bowl_kind"], observed=True)[cols].apply(lambda g: g.isna().mean())
    by_season.index = pd.MultiIndex.from_product([by_season.index, ["all"]], names=["year", "bowl_kind"])
    return pd.concat([by_season, by_kind]).round(4).reset_index()
