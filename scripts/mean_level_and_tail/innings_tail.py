"""
Innings-total tail: 220+ share and quantile shifts on an uncensored population.

A successful chase is right-censored at the target: its total reflects the
opposition's score, not the chasing side's capacity. Early chase finishes rose
across seasons, which alone moves the upper quantiles and the 220+ share of a pooled
innings population. So:

  Primary population   : first innings with the full 120-ball allowance.
  Sensitivity population: primary plus failed chases (terminal totals, not censored
                         by the target). Both are reported.

Computations, per population:
  1. 220+ share: a season x (total >= 220) 2x2 table; the headline is the risk
     difference (2024 minus 2023) with a Newcombe score interval and Fisher's exact
     p-value. The match bootstrap log-ratio interval is a secondary check (first
     innings are one per match, so the cluster is trivial there).
  2. Shift function: the season difference in the innings-total quantile at a fixed
     grid (0.50 to 0.90), match-bootstrapped on shared draws with BCa intervals.
     No 95th or 97th percentile: with about 70 innings per season those are set by
     three or four innings.
  3. Exceedance at 180, 200, 220: the same 2x2 machinery as the 220+ share.
  4. Generalized Pareto above 180 per season, only if both seasons have at least the
     configured minimum of exceedances: reports the fitted P(total > 220) per season
     and the season difference with a match-bootstrap percentile interval. The GPD
     shape from 15 to 30 excesses is loosely determined; the interval carries that.
  5. The 90th/95th/97th percentile shifts, kept as an appendix table on the same
     population so they can be compared with the all-innings versions in the theme
     table. Sparse order statistics; not for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from common import innings_population as ip
from . import season_contrast_bootstrap as scb


# ----------------------------------------------------------------------------
# 2x2 machinery: risk difference with a Newcombe score interval, Fisher's exact
# ----------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - level) / 2)
    p = successes / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return (centre - half, centre + half)


def newcombe_risk_difference(k_later: int, n_later: int, k_earlier: int, n_earlier: int,
                             level: float = 0.95) -> dict:
    """Risk difference (later minus earlier) with the Newcombe hybrid score interval
    (method 10 of Newcombe 1998): each proportion's Wilson limits are combined in
    quadrature around the observed difference. Behaves well at small counts, where a
    Wald interval overshoots or collapses."""
    p_l = k_later / n_later if n_later else np.nan
    p_e = k_earlier / n_earlier if n_earlier else np.nan
    l_l, u_l = wilson_interval(k_later, n_later, level)
    l_e, u_e = wilson_interval(k_earlier, n_earlier, level)
    rd = p_l - p_e
    lower = rd - np.sqrt((p_l - l_l) ** 2 + (u_e - p_e) ** 2)
    upper = rd + np.sqrt((u_l - p_l) ** 2 + (p_e - l_e) ** 2)
    return {"share_later": p_l, "share_earlier": p_e, "risk_difference": rd,
            "rd_low": lower, "rd_high": upper}


def two_by_two_contrast(innings: pd.DataFrame, threshold: int, earlier: int, later: int,
                        level: float = 0.95) -> dict:
    """Season x (total >= threshold) table with the risk difference and Fisher's exact test."""
    counts = {}
    for year in (earlier, later):
        season = innings[innings["year"] == year]
        counts[year] = (int((season["total"] >= threshold).sum()), int(len(season)))
    k_e, n_e = counts[earlier]
    k_l, n_l = counts[later]
    table = np.array([[k_l, n_l - k_l], [k_e, n_e - k_e]])
    _, fisher_p = stats.fisher_exact(table)
    out = {"threshold": threshold,
           "n_earlier": n_e, "over_earlier": k_e, "n_later": n_l, "over_later": k_l}
    out.update(newcombe_risk_difference(k_l, n_l, k_e, n_e, level))
    out["fisher_exact_p"] = float(fisher_p)
    return out


# ----------------------------------------------------------------------------
# Match bootstrap on an innings table (shared draws for every tail quantity)
# ----------------------------------------------------------------------------

@dataclass
class InningsBootstrapMaterial:
    """Per-season match lists and per-match innings totals for one population, so
    every tail estimand (share ratio, quantile shift, GPD) reads the same draws."""
    matches_earlier: np.ndarray
    matches_later: np.ndarray
    totals_by_match_earlier: list[np.ndarray]
    totals_by_match_later: list[np.ndarray]
    counts_earlier: np.ndarray
    counts_later: np.ndarray

    @property
    def n_matches(self) -> tuple[int, int]:
        return len(self.matches_earlier), len(self.matches_later)


def _totals_by_match(innings: pd.DataFrame, matches: np.ndarray) -> list[np.ndarray]:
    by_match = {m: g["total"].to_numpy(dtype=float) for m, g in innings.groupby("p_match")}
    return [by_match.get(m, np.array([])) for m in matches]


def bootstrap_material(innings: pd.DataFrame, earlier: int, later: int,
                       n_iter: int, seed: int) -> InningsBootstrapMaterial:
    inn_e = innings[innings["year"] == earlier]
    inn_l = innings[innings["year"] == later]
    matches_e = np.sort(inn_e["p_match"].unique())
    matches_l = np.sort(inn_l["p_match"].unique())
    counts_e, counts_l = scb.shared_resample_counts(len(matches_e), len(matches_l), n_iter, seed)
    return InningsBootstrapMaterial(matches_e, matches_l,
                                    _totals_by_match(inn_e, matches_e),
                                    _totals_by_match(inn_l, matches_l),
                                    counts_e, counts_l)


def _threshold_sums(totals_by_match: list[np.ndarray], matches: np.ndarray,
                    threshold: int) -> scb.SeasonMatchSums:
    numer = np.array([float((t >= threshold).sum()) for t in totals_by_match])
    denom = np.array([float(t.size) for t in totals_by_match])
    return scb.SeasonMatchSums(matches, numer, denom)


def share_ratio_bootstrap(material: InningsBootstrapMaterial, threshold: int,
                          alpha: float, bias_flag_se_multiple: float) -> dict:
    """Secondary check on the 2x2 headline: the match-bootstrap ratio of the
    threshold share, estimated on the log scale with a BCa interval."""
    theme = scb.Theme(
        f"innings_{threshold}_plus_share_ratio",
        scb.InningsFractionRatio(
            _threshold_sums(material.totals_by_match_earlier, material.matches_earlier, threshold),
            _threshold_sums(material.totals_by_match_later, material.matches_later, threshold)),
        1.0, "fragile_tail",
        "Log-scale BCa; effective n is the count of qualifying innings.")
    n_e, n_l = material.n_matches
    return scb.run_theme(theme, n_e, n_l, material.counts_earlier, material.counts_later,
                         alpha=alpha, bias_flag_se_multiple=bias_flag_se_multiple).as_row()


def quantile_shift_function(material: InningsBootstrapMaterial, grid, alpha: float,
                            label_prefix: str = "innings_total_shift_q") -> pd.DataFrame:
    """Season difference in the innings-total quantile at each grid point, BCa on
    shared draws. A quantile is not a ratio of sums, so no closed-form SE: BCa is
    the primary interval here."""
    n_e, n_l = material.n_matches
    rows = []
    for q in grid:
        theme = scb.Theme(f"{label_prefix}{int(round(q * 100))}",
                          scb.InningsQuantileShift(material.totals_by_match_earlier,
                                                   material.totals_by_match_later, quantile=q),
                          0.0, "fragile_tail", "")
        row = scb.run_theme(theme, n_e, n_l, material.counts_earlier, material.counts_later,
                            alpha=alpha).as_row()
        row["quantile"] = q
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Generalized Pareto tail above a threshold
# ----------------------------------------------------------------------------

def _gpd_exceedance_probability(totals: np.ndarray, threshold: int, target: int) -> dict:
    """P(total > target) from a GPD fitted to excesses over `threshold`:
    exceedance rate times the fitted survival at (target - threshold)."""
    excess = totals[totals > threshold] - threshold
    n_exceed = excess.size
    if n_exceed < 2 or totals.size == 0:
        return {"n_exceed": int(n_exceed), "shape": np.nan, "scale": np.nan,
                "exceed_rate": np.nan, "p_over_target": np.nan}
    shape, _, scale = stats.genpareto.fit(excess, floc=0)
    exceed_rate = n_exceed / totals.size
    p_over = exceed_rate * stats.genpareto.sf(target - threshold, shape, loc=0, scale=scale)
    return {"n_exceed": int(n_exceed), "shape": float(shape), "scale": float(scale),
            "exceed_rate": float(exceed_rate), "p_over_target": float(p_over)}


def gpd_tail_contrast(material: InningsBootstrapMaterial, threshold: int, target: int,
                      min_exceedances: int, n_resamples: int, seed: int, alpha: float) -> dict:
    """GPD-implied P(total > target) per season and its difference, with a match
    bootstrap percentile interval, gated on the exceedance count in both seasons."""
    totals_e = np.concatenate(material.totals_by_match_earlier)
    totals_l = np.concatenate(material.totals_by_match_later)
    n_exceed_e = int((totals_e > threshold).sum())
    n_exceed_l = int((totals_l > threshold).sum())
    out = {"threshold": threshold, "target": target,
           "n_exceed_earlier": n_exceed_e, "n_exceed_later": n_exceed_l,
           "gate_min_exceedances": min_exceedances,
           "fitted": bool(min(n_exceed_e, n_exceed_l) >= min_exceedances)}
    if not out["fitted"]:
        out["note"] = "GPD not fitted: exceedances below the gate in at least one season"
        return out
    fit_e = _gpd_exceedance_probability(totals_e, threshold, target)
    fit_l = _gpd_exceedance_probability(totals_l, threshold, target)
    out["earlier"] = fit_e
    out["later"] = fit_l
    out["p_over_target_difference"] = fit_l["p_over_target"] - fit_e["p_over_target"]

    # Match bootstrap: refit the GPD on each resample's innings. Its own draw count
    # because each resample is a numerical fit, not a matrix product.
    n_e, n_l = material.n_matches
    counts_e, counts_l = scb.shared_resample_counts(n_e, n_l, n_resamples, seed)

    def resampled_totals(counts_row, totals_by_match):
        return np.concatenate([np.repeat(t, int(c)) for t, c in zip(totals_by_match, counts_row)
                               if c > 0 and t.size] or [np.array([])])

    diffs = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        pe = _gpd_exceedance_probability(resampled_totals(counts_e[i], material.totals_by_match_earlier),
                                         threshold, target)["p_over_target"]
        pl = _gpd_exceedance_probability(resampled_totals(counts_l[i], material.totals_by_match_later),
                                         threshold, target)["p_over_target"]
        diffs[i] = pl - pe
    finite = diffs[np.isfinite(diffs)]
    out["difference_low"] = float(np.quantile(finite, alpha / 2)) if finite.size else np.nan
    out["difference_high"] = float(np.quantile(finite, 1 - alpha / 2)) if finite.size else np.nan
    out["n_bootstrap_fits"] = int(finite.size)
    return out


# ----------------------------------------------------------------------------
# Driver: both populations, every quantity
# ----------------------------------------------------------------------------

def run_tail_analysis(frame: pd.DataFrame, settings, seed: int,
                      n_iter: int | None = None) -> dict:
    """All tail quantities on the primary and sensitivity populations.

    Returns a dict of tables: population_counts, threshold_contrasts (2x2 rows for
    every threshold and population), share_ratio_bootstrap, shift_function,
    gpd (nested), and sparse_upper_quantile_shifts (the 90th/95th/97th
    percentile shifts, set by a handful of innings; appendix only).
    """
    tail = settings.section("tail")
    earlier, later = settings.earlier_season, settings.later_season
    alpha = 1.0 - settings.interval_level
    n_iter = n_iter or settings.n_resamples
    bias_multiple = settings.section("bootstrap")["ratio_bias_flag_se_multiple"]

    innings = ip.build_innings_table(frame, full_innings_balls=tail["full_innings_balls"],
                                     death_first_over=settings.phase("death")[0])
    populations = {
        "first_innings_full_allowance": ip.uncensored_first_innings(innings),
        "first_innings_plus_failed_chases": ip.terminal_innings(innings),
    }

    contrasts, ratio_rows, shift_rows, sparse_rows, gpd = [], [], [], [], {}
    for pop_name, pop in populations.items():
        material = bootstrap_material(pop, earlier, later, n_iter, seed)
        for threshold in tail["exceedance_thresholds"]:
            row = two_by_two_contrast(pop, threshold, earlier, later, settings.interval_level)
            row["population"] = pop_name
            row["headline"] = threshold == tail["threshold_runs"]
            contrasts.append(row)

        ratio = share_ratio_bootstrap(material, tail["threshold_runs"], alpha, bias_multiple)
        ratio["population"] = pop_name
        ratio_rows.append(ratio)

        shift = quantile_shift_function(material, tail["quantile_grid"], alpha)
        shift["population"] = pop_name
        shift_rows.append(shift)

        sparse = quantile_shift_function(material, (0.90, 0.95, 0.97), alpha,
                                         label_prefix="sparse_upper_quantile_shift_q")
        sparse["population"] = pop_name
        sparse_rows.append(sparse)

        gpd[pop_name] = gpd_tail_contrast(material, tail["gpd_threshold"], tail["threshold_runs"],
                                          tail["gpd_min_exceedances"], tail["gpd_resamples"],
                                          seed, alpha)

    return {
        "innings_table": innings,
        "population_counts": ip.population_counts(innings),
        "threshold_contrasts": pd.DataFrame(contrasts),
        "share_ratio_bootstrap": pd.DataFrame(ratio_rows),
        "shift_function": pd.concat(shift_rows, ignore_index=True),
        "sparse_upper_quantile_shifts": pd.concat(sparse_rows, ignore_index=True),
        "gpd": gpd,
    }


def plot_shift_function(shift: pd.DataFrame, interval_level: float):
    """Quantile shift (later minus earlier) across the grid, one line per population,
    with the bootstrap interval as a band."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (pop, rows) in enumerate(shift.groupby("population", sort=False)):
        rows = rows.sort_values("quantile")
        colour = f"C{i}"
        ax.plot(rows["quantile"], rows["estimate"], "o-", color=colour, label=pop.replace("_", " "))
        ax.fill_between(rows["quantile"], rows["ci_low"], rows["ci_high"], color=colour, alpha=0.18)
    ax.axhline(0, color="#C0392B", ls="--", lw=1)
    ax.set_xlabel("innings-total quantile")
    ax.set_ylabel("shift in runs (later minus earlier)")
    ax.set_title("Innings-total shift function", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.01, -0.03, f"Match cluster bootstrap, {int(interval_level * 100)}% BCa intervals, "
             "season-stratified jackknife. Resampling stability, not replication.",
             fontsize=7.5, color="#6A6A6A", style="italic")
    fig.tight_layout()
    return fig
