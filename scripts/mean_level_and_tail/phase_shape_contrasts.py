"""
Bookended acceleration, tested as composite shape contrasts with equivalence verdicts.

The claim is a shape: the ends of the innings (powerplay, death) accelerating while
the middle lagged. Testing it as a difference of two phase deltas spends power badly
on the noisy run-rate channel, so it is recast as:

  1. Curvature contrast: one statistic weighting the three phase season-deltas
     +1 (powerplay), -2 (middle), +1 (death). Positive means the ends moved
     differently from the middle (a U shape). Run in three channels: boundary share
     and six share (proportions over thousands of balls per phase, the densest
     channels) and run rate.
  2. Middle against league: the middle-overs run-rate delta minus the whole-innings
     delta, testing the middle's lag directly against the pooled baseline.

All four are read off the same match resamples (shared draws), so the interval of a
composite carries the correlation between its phase terms.

A null here is only informative against a pre-stated smallest effect of interest
(SESOI) per channel, so each contrast gets an equivalence verdict: rejects null,
equivalent to zero, or inconclusive, with the minimum detectable effect reported.
The hypothesis is retired only if the two dense channels (boundary, six) return
"equivalent to zero"; if either is inconclusive the shape is unresolved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.equivalence import equivalence_from_theme_result
from . import season_contrast_bootstrap as scb


def is_four(f):
    by_runs = f["batruns"].to_numpy() == 4
    if "outcome" in f.columns:
        return by_runs | f["outcome"].astype(str).str.lower().eq("four").to_numpy()
    return by_runs


def is_boundary(f):
    return is_four(f) | scb.is_six(f)


# Per-channel numerator/denominator and the base scope each channel measures over.
# rpo is runs per legal ball (times six) using bbb data; boundary%/six% are events per ball faced,
# where ballfaced already zeroes wides so the phase scope needs no legal filter.
def channel_weights(season, channel):
    if channel == "rpo":
        return (season["score"].to_numpy().astype(float),
                scb.is_legal_delivery(season).astype(float),
                True, scb.is_legal_delivery(season))
    if channel == "boundary":
        return (is_boundary(season).astype(float),
                scb.is_ball_faced(season).astype(float),
                False, np.ones(len(season), bool))
    if channel == "six":
        return (scb.is_six(season).astype(float),
                scb.is_ball_faced(season).astype(float),
                False, np.ones(len(season), bool))
    raise ValueError(channel)


def phase_diff(earlier, later, channel, phase):
    ne, de, per_over, base_e = channel_weights(earlier, channel)
    nl, dl, _, base_l = channel_weights(later, channel)
    sums_e = scb.per_match_sums(earlier, scb.in_phase(earlier, phase) & base_e, ne, de)
    sums_l = scb.per_match_sums(later, scb.in_phase(later, phase) & base_l, nl, dl)
    return scb.RateDifference(sums_e, sums_l, per_over=per_over)


def league_diff(earlier, later, channel):
    # Reference is the whole innings, all overs, so the middle is compared to the
    # overall acceleration it is part of, not to the other two phases alone.
    ne, de, per_over, base_e = channel_weights(earlier, channel)
    nl, dl, _, base_l = channel_weights(later, channel)
    return scb.RateDifference(
        scb.per_match_sums(earlier, base_e, ne, de),
        scb.per_match_sums(later, base_l, nl, dl),
        per_over=per_over,
    )


# Channel -> (SESOI config key, unit label, multiplier from stored fraction to the
# unit the SESOI is stated in). Shares are stored as fractions; SESOI in points.
CHANNEL_SESOI = {
    "rpo": ("run_rate_rpo", "runs per over", 1.0),
    "boundary": ("boundary_share_points", "percentage points", 100.0),
    "six": ("six_share_points", "percentage points", 100.0),
}


def build_shape_themes(earlier: pd.DataFrame, later: pd.DataFrame,
                       powerplay, middle, death) -> list[scb.Theme]:
    """The four shape contrasts as harness themes."""
    # Middle lag tested directly against the pooled league delta (negative = lagged).
    middle_vs_league = scb.Theme(
        "middle_rpo_minus_league_rpo",
        scb.WeightedSum([(1.0, phase_diff(earlier, later, "rpo", middle)),
                         (-1.0, league_diff(earlier, later, "rpo"))]),
        null_value=0.0, power_tier="composite",
        uncertainty_note="Negative = middle lagged the overall acceleration.",
    )

    # U-shape curvature: ends up, middle down. Positive = bookended.
    curvature_themes = [
        scb.Theme(
            f"bookending_curvature_{channel}",
            scb.WeightedSum([(1.0, phase_diff(earlier, later, channel, powerplay)),
                             (-2.0, phase_diff(earlier, later, channel, middle)),
                             (1.0, phase_diff(earlier, later, channel, death))]),
            null_value=0.0,
            power_tier="composite" if channel == "rpo" else "well_powered",
            uncertainty_note="Positive = ends rose relative to the middle (U-shape).",
        )
        for channel in ("boundary", "six", "rpo")
    ]
    return [middle_vs_league, *curvature_themes]


def _channel_of(theme_name: str) -> str:
    return "rpo" if theme_name.endswith("rpo") else theme_name.rsplit("_", 1)[-1]


def run_shape_contrasts(frame: pd.DataFrame, settings, seed: int,
                        n_iter: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap the four contrasts on shared draws and attach equivalence verdicts.

    Returns (results table, verdict table). The verdict table states, per channel,
    the point, the 90% and 95% intervals, the SE, the SESOI, the MDE, and the verdict.
    SESOI for share channels is given in percentage points and converted to the
    fraction scale the estimand lives on before the comparison.
    """
    earlier = frame[frame["year"] == settings.earlier_season]
    later = frame[frame["year"] == settings.later_season]
    themes = build_shape_themes(earlier, later, settings.phase("powerplay"),
                                settings.phase("middle"), settings.phase("death"))

    n_e = earlier["p_match"].nunique()
    n_l = later["p_match"].nunique()
    n_iter = n_iter or settings.n_resamples
    counts_e, counts_l = scb.shared_resample_counts(n_e, n_l, n_iter, seed)
    bootstrap_cfg = settings.section("bootstrap")

    results = [scb.run_theme(t, n_e, n_l, counts_e, counts_l,
                             alpha=1.0 - settings.interval_level,
                             bias_flag_se_multiple=bootstrap_cfg["ratio_bias_flag_se_multiple"])
               for t in themes]
    table = pd.DataFrame([r.as_row() for r in results])

    sesoi_cfg = settings.section("sesoi")
    verdict_rows = []
    for r in results:
        key, unit, to_unit = CHANNEL_SESOI[_channel_of(r.name)]
        sesoi_fraction = float(sesoi_cfg[key]) / to_unit
        row = equivalence_from_theme_result(
            r, sesoi=sesoi_fraction, mde_se_multiple=bootstrap_cfg["mde_se_multiple"],
            unit=unit, equivalence_level=settings.equivalence_level,
            interval_level=settings.interval_level)
        # Report shares in points so the table reads in the SESOI's own unit.
        for col in ("estimate", "ci90_low", "ci90_high", "ci95_low", "ci95_high",
                    "se", "sesoi", "mde_80pct_power"):
            row[col] = row[col] * to_unit
        verdict_rows.append(row)
    verdicts = pd.DataFrame(verdict_rows)

    dense = verdicts[verdicts["channel"].isin(["bookending_curvature_boundary",
                                               "bookending_curvature_six"])]
    retire = bool((dense["verdict"] == "equivalent_to_zero").all())
    verdicts.attrs["hypothesis_status"] = "retired" if retire else "unresolved"
    return table, verdicts
