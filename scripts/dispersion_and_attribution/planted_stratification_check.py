"""
Planted-effect check for the powerplay stratification models on the synthetic fixture.

The synthetic fixture has no between-team heterogeneity, so on it every dispersion
model must read null. This module plants a known team-level effect and asks the
models to find it:

  - in the later season only, each team's powerplay `score` is multiplied by a
    team factor spread evenly over [1 - spread, 1 + spread] (the middle overs are
    untouched), so the later season carries genuine between-team spread in the
    powerplay run-rate level and in the gear-up from baseline;
  - the aggressive-shot tags are untouched, so the attempt outcome stays null.

Expected reading: P(widen) high for the run-rate level and the gear-up outcomes,
low for attempt log-odds; the shared factor's tau widens. Mechanics only; nothing
from this frame is a finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mean_level_and_tail import season_contrast_bootstrap as scb


def plant_powerplay_stratification(frame: pd.DataFrame, spread: float = 0.25,
                                   later_season: int = scb.LATER_SEASON) -> pd.DataFrame:
    """Return a copy of the frame with the later season's powerplay scores scaled by
    a per-team factor. Scores are rounded back to integers so the frame stays
    schema-faithful; batruns is left alone (the level and gear-up outcomes use score)."""
    work = frame.copy()
    teams = np.sort(work["team_bat"].unique())
    factors = dict(zip(teams, np.linspace(1 - spread, 1 + spread, len(teams))))
    in_pp = scb.in_phase(work, scb.POWERPLAY_OVERS)
    later = work["year"].to_numpy() == later_season
    mask = in_pp & later
    scale = work.loc[mask, "team_bat"].map(factors).to_numpy()
    work.loc[mask, "score"] = np.rint(work.loc[mask, "score"].to_numpy() * scale).astype(int)
    return work
