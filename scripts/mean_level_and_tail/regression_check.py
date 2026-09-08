"""
Regression check against the previous run's stored results.

The interval machinery changed (season-stratified jackknife, bootstrap-t, log-scale
ratios) but the estimands did not. This check proves that on the real data:

  1. Every point estimate matches the stored value exactly (to the stored rounding).
  2. With the previous seeds and the pooled jackknife (`stratified_jackknife=False`),
     the BCa interval reproduces the stored interval exactly: same draws, same
     acceleration, same quantiles. For ratio themes the comparison is against the
     raw-scale BCa, which is what the previous run reported.
  3. The size of the interval change under the stratified jackknife is reported for
     information (points unchanged, intervals expected to move slightly).

`reference_previous_run.csv` holds the stored numbers with the seed each table was
produced under (the previous notebook used seed 36 for the theme table and the
quantile shifts, and 20240601 for the shape contrasts). Run this once on the full
cleaned dataset; a mismatch in (1) or (2) means a logic change slipped in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import season_contrast_bootstrap as scb
from .phase_shape_contrasts import build_shape_themes

REFERENCE_PATH = Path(__file__).with_name("reference_previous_run.csv")
ROUNDING = 4


def _quantile_themes(frame: pd.DataFrame, earlier_season: int, later_season: int):
    """The previous quantile-shift cell: all innings, quantiles 0.90/0.95/0.97."""
    earlier = frame[frame["year"] == earlier_season]
    later = frame[frame["year"] == later_season]
    innings_e = earlier.groupby(["p_match", "inns"])["score"].sum().reset_index()
    innings_l = later.groupby(["p_match", "inns"])["score"].sum().reset_index()
    matches_e = np.sort(earlier["p_match"].unique())
    matches_l = np.sort(later["p_match"].unique())

    def by_match(innings_frame, all_matches):
        lookup = {m: g["score"].to_numpy() for m, g in innings_frame.groupby("p_match")}
        return [lookup.get(m, np.array([])) for m in all_matches]

    themes = [scb.Theme(f"innings_total_q{int(q * 100)}_shift",
                        scb.InningsQuantileShift(by_match(innings_e, matches_e),
                                                 by_match(innings_l, matches_l), quantile=q),
                        0.0, "fragile_tail")
              for q in (0.90, 0.95, 0.97)]
    return themes, len(matches_e), len(matches_l)


def _run_pooled_and_stratified(themes, n_e, n_l, seed, n_iter):
    counts_e, counts_l = scb.shared_resample_counts(n_e, n_l, n_iter, seed)
    pooled = [scb.run_theme(t, n_e, n_l, counts_e, counts_l, stratified_jackknife=False)
              for t in themes]
    revised = [scb.run_theme(t, n_e, n_l, counts_e, counts_l, stratified_jackknife=True)
               for t in themes]
    return pooled, revised


def run_regression_check(frame: pd.DataFrame, settings, n_iter: int = 10000) -> pd.DataFrame:
    """One row per stored theme: stored vs reproduced point and interval, plus the
    interval under the stratified jackknife for the same draws."""
    reference = pd.read_csv(REFERENCE_PATH)
    earlier_season, later_season = settings.earlier_season, settings.later_season
    earlier = frame[frame["year"] == earlier_season]
    later = frame[frame["year"] == later_season]

    produced = {}
    themes, n_e, n_l, _ = scb.build_themes(frame)
    for pooled, revised in zip(*_run_pooled_and_stratified(themes, n_e, n_l, 36, n_iter)):
        produced[("run_all", pooled.name)] = (pooled, revised)

    q_themes, n_e, n_l = _quantile_themes(frame, earlier_season, later_season)
    for pooled, revised in zip(*_run_pooled_and_stratified(q_themes, n_e, n_l, 36, n_iter)):
        produced[("quantile_shift", pooled.name)] = (pooled, revised)

    shape_themes = build_shape_themes(earlier, later, settings.phase("powerplay"),
                                      settings.phase("middle"), settings.phase("death"))
    n_e, n_l = earlier["p_match"].nunique(), later["p_match"].nunique()
    for pooled, revised in zip(*_run_pooled_and_stratified(shape_themes, n_e, n_l, 20240601, n_iter)):
        produced[("shape_contrasts", pooled.name)] = (pooled, revised)

    rows = []
    for _, ref in reference.iterrows():
        pooled, revised = produced[(ref["source"], ref["theme"])]
        # The previous run reported BCa on the raw scale for every theme.
        if pooled.scale == "log":
            legacy_lo, legacy_hi = pooled.raw_scale_bca_low, pooled.raw_scale_bca_high
            revised_lo, revised_hi = revised.raw_scale_bca_low, revised.raw_scale_bca_high
        else:
            legacy_lo, legacy_hi = pooled.bca_low, pooled.bca_high
            revised_lo, revised_hi = revised.bca_low, revised.bca_high
        rows.append({
            "source": ref["source"], "theme": ref["theme"],
            "stored_estimate": ref["estimate"], "reproduced_estimate": round(pooled.point, ROUNDING),
            "estimate_matches": bool(np.isclose(ref["estimate"], round(pooled.point, ROUNDING), atol=1e-4)),
            "stored_ci_low": ref["ci_low"], "stored_ci_high": ref["ci_high"],
            "pooled_jackknife_bca_low": round(legacy_lo, ROUNDING), "pooled_jackknife_bca_high": round(legacy_hi, ROUNDING),
            "interval_matches": bool(np.isclose(ref["ci_low"], round(legacy_lo, ROUNDING), atol=1e-4)
                                     and np.isclose(ref["ci_high"], round(legacy_hi, ROUNDING), atol=1e-4)),
            "stratified_bca_low": round(revised_lo, ROUNDING), "stratified_bca_high": round(revised_hi, ROUNDING),
            "stratified_shift_low": round(revised_lo - legacy_lo, ROUNDING),
            "stratified_shift_high": round(revised_hi - legacy_hi, ROUNDING),
        })
    return pd.DataFrame(rows)
