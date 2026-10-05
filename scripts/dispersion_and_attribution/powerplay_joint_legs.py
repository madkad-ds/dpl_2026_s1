"""
Powerplay batting stratification: three legs on one set of draws.

The three operationalizations of powerplay spread (run-rate level, aggressive
attempt share, gear-up from the team's own middle-overs baseline) share matches,
and two of them share the `score` column, so they are correlated. Their agreement
is weaker evidence than "three independent legs" would be. This module makes the
dependence visible by computing all three between-team SD differences on the same
match resamples and reporting:

  1. the marginal P(widen) per leg;
  2. the resample correlation matrix of the three SD differences;
  3. P(all three widened) across resamples, the joint evidence.

The hierarchical model is the de-attenuated reading of the level and gear-up legs;
here the bootstrap versions of those legs stand in so all three read off the same
draws. The multivariate hierarchical model (powerplay_multivariate_model.py) is the
confirmatory rung.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mean_level_and_tail import season_contrast_bootstrap as scb
from .team_spread_bootstrap import (PhaseRunRateSpread, attempt_share_spread_estimand,
                                    gear_up_spread_estimand)

LEGS = ("run_rate_level", "attempt_share", "gear_up_from_baseline")


def run_joint_legs(frame, settings, seed: int, n_iter: int | None = None) -> dict:
    """Three powerplay legs on shared draws: marginal rows, correlation matrix, and
    P(all widened)."""
    n_iter = n_iter or settings.n_resamples
    alpha = 1.0 - settings.interval_level
    powerplay, middle = settings.phase("powerplay"), settings.phase("middle")
    teams = sorted(frame["team_bat"].unique())

    estimands = {
        "run_rate_level": PhaseRunRateSpread(frame, powerplay, teams),
        "attempt_share": attempt_share_spread_estimand(frame, powerplay),
        "gear_up_from_baseline": gear_up_spread_estimand(frame, powerplay, middle),
    }
    n_e, n_l = estimands["run_rate_level"].n_matches()
    for est in estimands.values():
        assert est.n_matches() == (n_e, n_l), "legs must resample the same match lists"
    counts_e, counts_l = scb.shared_resample_counts(n_e, n_l, n_iter, seed)

    replicates = {}
    rows = []
    for leg in LEGS:
        est = estimands[leg]
        point = float(est(np.ones((1, n_e)), np.ones((1, n_l)))[0])
        reps = est(counts_e, counts_l)
        replicates[leg] = reps
        finite = reps[np.isfinite(reps)]
        rows.append({"leg": leg, "sd_diff": point,
                     "ci_low": float(np.percentile(finite, 100 * alpha / 2)),
                     "ci_high": float(np.percentile(finite, 100 * (1 - alpha / 2))),
                     "p_widen": float((finite > 0).mean()),
                     "column_exposure": "tagged (shot)" if leg == "attempt_share" else "clean (score, over_num)"})

    stacked = np.column_stack([replicates[leg] for leg in LEGS])
    keep = np.all(np.isfinite(stacked), axis=1)
    stacked = stacked[keep]
    correlation = pd.DataFrame(np.corrcoef(stacked, rowvar=False), index=LEGS, columns=LEGS)
    p_all_widened = float(np.all(stacked > 0, axis=1).mean())
    p_clean_pair_widened = float(np.all(stacked[:, [0, 2]] > 0, axis=1).mean())

    marginals = pd.DataFrame(rows)
    marginals["p_all_three_widened"] = p_all_widened
    marginals["p_both_clean_legs_widened"] = p_clean_pair_widened
    marginals["n_resamples"] = int(keep.sum())
    return {"marginals": marginals, "correlation": correlation,
            "p_all_three_widened": p_all_widened,
            "p_both_clean_legs_widened": p_clean_pair_widened}
