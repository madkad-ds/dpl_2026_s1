"""
Empirical coverage of the cluster bootstrap intervals at the real cluster counts.

Every mean-level interval in the season contrasts is built from about 70 match
clusters per season (14 per team for a team-level rate). Whether a nominal 95%
BCa or bootstrap-t interval actually covers 95% of the time at those counts is not
something theory settles, so it is checked by simulation:

  1. Generative model. Per-match (numerator, denominator) pairs are taken from the
     real data for a rate (powerplay run rate: score over legal balls; powerplay six
     rate: sixes over faced balls). A simulated season is n pairs drawn with
     replacement from the pooled two-season set, so both seasons come from one
     population and the only season difference is the planted effect: the later
     season's numerators are shifted by effect x denominator (rate difference) or
     multiplied by a factor (rate ratio). The true value of the estimand is then the
     effect itself (zero or one under the null).
  2. For each simulated dataset, the harness machinery runs unchanged: shared match
     resamples, point estimate, season-stratified BCa, bootstrap-t from the closed
     form cluster SE, and a plain percentile interval for reference. Ratio estimands
     run on the log scale as in the analyses.
  3. Coverage is the share of datasets whose interval contains the true value;
     mean interval width is reported alongside, since a wide interval covers cheaply.

Cluster counts: the league count (about 70 per season) and the team count (14 per
season, drawn from per team-innings pairs, since a team's match contributes only its
own innings). If coverage at 14 clusters falls below about 0.90 the team-level
intervals are labelled approximate.

The Monte Carlo standard error of a coverage estimate near 0.95 is about
sqrt(0.95 x 0.05 / n_datasets), which is 0.002 at 10,000 datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mean_level_and_tail import season_contrast_bootstrap as scb

# Rate definitions (numerator column or flag, denominator flag, per-over scaling).
RATES = {
    "powerplay_run_rate": dict(per_over=True, label="runs per over, legal balls"),
    "powerplay_six_rate": dict(per_over=False, label="sixes per ball faced"),
}


@dataclass
class ClusterPairPool:
    """Observed per-cluster (numerator, denominator) pairs for one rate, pooled over
    both seasons, at one cluster definition (match or team-innings)."""
    rate: str
    cluster_kind: str
    numerator: np.ndarray
    denominator: np.ndarray

    @property
    def rate_value(self) -> float:
        return float(self.numerator.sum() / self.denominator.sum())


def _powerplay_weights(frame, rate: str, phase):
    scope = scb.in_phase(frame, phase)
    if rate == "powerplay_run_rate":
        legal = scb.is_legal_delivery(frame)
        return scope & legal, frame["score"].to_numpy(dtype=float), legal.astype(float)
    if rate == "powerplay_six_rate":
        faced = scb.is_ball_faced(frame)
        return scope & faced, scb.is_six(frame).astype(float), faced.astype(float)
    raise ValueError(rate)


def observed_pair_pools(frame, phase) -> dict:
    """Pools keyed by (rate, cluster_kind): match pairs and team-innings pairs, both
    seasons pooled. Clusters with a zero denominator are dropped (they carry no rate)."""
    pools = {}
    for rate in RATES:
        scope, numer, denom = _powerplay_weights(frame, rate, phase)
        by_match = scb.per_match_sums(frame, scope, numer, denom)
        keep = by_match.denominator > 0
        pools[(rate, "match")] = ClusterPairPool(rate, "match", by_match.numerator[keep],
                                                 by_match.denominator[keep])
        work = pd.DataFrame({"p_match": frame["p_match"].to_numpy(), "team": frame["team_bat"].to_numpy(),
                             "numer": numer * scope, "denom": denom * scope})
        g = work.groupby(["p_match", "team"])[["numer", "denom"]].sum()
        g = g[g["denom"] > 0]
        pools[(rate, "team_innings")] = ClusterPairPool(rate, "team_innings", g["numer"].to_numpy(),
                                                        g["denom"].to_numpy())
    return pools


def simulate_dataset(pool: ClusterPairPool, n_clusters: int, effect: float, estimand: str,
                     rng: np.random.Generator) -> tuple[scb.SeasonMatchSums, scb.SeasonMatchSums]:
    """Two seasons of n_clusters pairs drawn with replacement from the pool; the later
    season's numerators carry the planted effect."""
    idx_e = rng.integers(0, len(pool.numerator), size=n_clusters)
    idx_l = rng.integers(0, len(pool.numerator), size=n_clusters)
    numer_l = pool.numerator[idx_l].copy()
    denom_l = pool.denominator[idx_l]
    if estimand == "difference":
        scale = 6.0 if RATES[pool.rate]["per_over"] else 1.0
        numer_l = numer_l + (effect / scale) * denom_l
    elif estimand == "log_ratio":
        numer_l = numer_l * effect
    else:
        raise ValueError(estimand)
    ids = np.arange(n_clusters)
    return (scb.SeasonMatchSums(ids, pool.numerator[idx_e], pool.denominator[idx_e]),
            scb.SeasonMatchSums(ids, numer_l, denom_l))


def _percentile_interval(result: scb.ThemeResult) -> tuple[float, float]:
    reps = result.replicates[np.isfinite(result.replicates)]
    alpha = result.alpha
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if result.scale == "log":
        return float(np.exp(lo)), float(np.exp(hi))
    return float(lo), float(hi)


def one_dataset(pool: ClusterPairPool, n_clusters: int, effect: float, estimand: str,
                n_resamples: int, alpha: float, seed: int) -> dict:
    """Intervals for one simulated dataset and whether each covers the truth."""
    rng = np.random.default_rng(seed)
    sums_e, sums_l = simulate_dataset(pool, n_clusters, effect, estimand, rng)
    if estimand == "difference":
        est = scb.RateDifference(sums_e, sums_l, per_over=RATES[pool.rate]["per_over"])
        theme = scb.Theme("sim", est, 0.0, "well_powered")
        truth = effect
    else:
        est = scb.RateRatio(sums_e, sums_l)
        theme = scb.Theme("sim", est, 1.0, "well_powered")
        truth = effect
    counts_e, counts_l = scb.shared_resample_counts(n_clusters, n_clusters, n_resamples,
                                                    int(rng.integers(0, 2 ** 31)))
    result = scb.run_theme(theme, n_clusters, n_clusters, counts_e, counts_l, alpha=alpha)
    pct = _percentile_interval(result)
    return {
        "point": result.point,
        "covered_bca": bool(result.bca_low <= truth <= result.bca_high),
        "covered_boot_t": bool(np.isfinite(result.boot_t_low) and result.boot_t_low <= truth <= result.boot_t_high),
        "covered_percentile": bool(pct[0] <= truth <= pct[1]),
        "width_bca": result.bca_high - result.bca_low,
        "width_boot_t": result.boot_t_high - result.boot_t_low,
        "width_percentile": pct[1] - pct[0],
        "boot_t_defined": bool(np.isfinite(result.boot_t_low)),
    }


def _run_batch(pool: ClusterPairPool, scenario: dict, seeds: list[int],
               n_resamples: int, alpha: float) -> list[dict]:
    """A chunk of datasets for one scenario, run in one worker."""
    return [one_dataset(pool, scenario["n_clusters"], scenario["effect"], scenario["estimand"],
                        n_resamples, alpha, s) for s in seeds]


def coverage_table(frame, settings, seed: int, n_datasets: int | None = None,
                   n_resamples: int | None = None, n_jobs: int | None = None,
                   scenarios: list[dict] | None = None, batch_size: int = 250) -> pd.DataFrame:
    """Run every scenario (rate x estimand x cluster count x effect) and return one
    row each with empirical coverage per interval method, mean widths, the Monte
    Carlo standard error, and the approximate flag.

    Datasets are run in chunks of `batch_size` per task on the threading backend:
    the work is NumPy matrix products and quantiles that release the GIL, so threads
    parallelise it without the per-task pickling and worker restarts that a process
    pool incurs on tens of thousands of small tasks. `n_jobs` defaults to the config
    value; leave a couple of cores free on a laptop.
    """
    from joblib import Parallel, delayed

    cfg = settings.section("interval_coverage")
    n_datasets = n_datasets or int(cfg["n_datasets"])
    n_resamples = n_resamples or int(cfg["n_resamples_per_dataset"])
    n_jobs = int(n_jobs if n_jobs is not None else cfg.get("n_jobs", 1))
    alpha = 1.0 - settings.interval_level
    pools = observed_pair_pools(frame, settings.phase("powerplay"))
    scenarios = scenarios or default_scenarios(cfg)

    rows = []
    seed_seq = np.random.SeedSequence(seed)
    for scenario in scenarios:
        pool = pools[(scenario["rate"], scenario["cluster_kind"])]
        seeds = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(n_datasets)]
        chunks = [seeds[i:i + batch_size] for i in range(0, n_datasets, batch_size)]
        batches = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_batch)(pool, scenario, chunk, n_resamples, alpha) for chunk in chunks)
        res = pd.DataFrame([r for batch in batches for r in batch])
        row = {**scenario, "n_datasets": n_datasets, "n_resamples": n_resamples,
               "pool_size": int(len(pool.numerator)), "pool_rate": pool.rate_value,
               "nominal_level": settings.interval_level,
               "coverage_bca": float(res["covered_bca"].mean()),
               "coverage_boot_t": float(res["covered_boot_t"].mean()),
               "coverage_percentile": float(res["covered_percentile"].mean()),
               "mean_width_bca": float(res["width_bca"].mean()),
               "mean_width_boot_t": float(res["width_boot_t"].mean()),
               "mean_width_percentile": float(res["width_percentile"].mean()),
               "boot_t_defined_share": float(res["boot_t_defined"].mean()),
               "mean_point": float(res["point"].mean()),
               "monte_carlo_se": float(np.sqrt(settings.interval_level * alpha / n_datasets))}
        row["approximate_flag"] = bool(min(row["coverage_bca"], row["coverage_boot_t"]) < 0.90)
        rows.append(row)
    return pd.DataFrame(rows)


def default_scenarios(cfg: dict) -> list[dict]:
    """Rate x estimand x cluster count x effect, from the config lists."""
    league, team = int(cfg["league_clusters_per_season"]), int(cfg["team_clusters_per_season"])
    scenarios = []
    for effect in cfg["run_rate_difference_effects_rpo"]:
        for kind, n in (("match", league), ("team_innings", team)):
            scenarios.append(dict(rate="powerplay_run_rate", estimand="difference", cluster_kind=kind,
                                  n_clusters=n, effect=float(effect)))
    for effect in cfg["six_rate_difference_effects"]:
        for kind, n in (("match", league), ("team_innings", team)):
            scenarios.append(dict(rate="powerplay_six_rate", estimand="difference", cluster_kind=kind,
                                  n_clusters=n, effect=float(effect)))
    for factor in cfg["six_rate_ratio_effects"]:
        for kind, n in (("match", league), ("team_innings", team)):
            scenarios.append(dict(rate="powerplay_six_rate", estimand="log_ratio", cluster_kind=kind,
                                  n_clusters=n, effect=float(factor)))
    return scenarios


def plot_coverage(table: pd.DataFrame, nominal: float):
    """Coverage per scenario for the three interval methods, one panel per estimand."""
    import matplotlib.pyplot as plt

    methods = [("coverage_boot_t", "bootstrap-t"), ("coverage_bca", "BCa, stratified"),
               ("coverage_percentile", "percentile")]
    groups = list(table.groupby(["rate", "estimand"], sort=False))
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 2.4 * len(groups) + 0.8), squeeze=False)
    for ax, ((rate, estimand), rows) in zip(axes[:, 0], groups):
        rows = rows.sort_values(["cluster_kind", "effect"])
        labels = [f"{r.cluster_kind} n={r.n_clusters}, effect={r.effect:g}" for r in rows.itertuples()]
        x = np.arange(len(rows))
        for k, (col, name) in enumerate(methods):
            ax.plot(x + (k - 1) * 0.15, rows[col], "o", ms=6, label=name, color=f"C{k}")
        ax.axhline(nominal, color="#C0392B", ls="--", lw=1)
        ax.axhline(0.90, color="#9AA0A6", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylim(min(0.80, float(rows[[m[0] for m in methods]].min().min()) - 0.02), 1.0)
        ax.set_ylabel("coverage")
        ax.set_title(f"{rate}, {estimand}", loc="left", fontsize=10, fontweight="bold")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    return fig
