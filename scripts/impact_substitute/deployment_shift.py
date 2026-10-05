"""
Impact-substitute deployment: innings reallocation, entry timing, and why the
pooled death-entry share moved.

Source: the Cricsheet replacements table restricted to impact-player events (the one
concussion substitute is an injury replacement, not a tactical choice, and is
excluded by the reason field). Independent of the ball-by-ball threads and free of
tagging exposure. The resampling unit is the match, not the deployment: the two subs
in a match respond to the same game state.

Structure of the analysis, in the order the notebook runs it:

  1. Pooled death-entry share. The share of all deployments that entered at the
     death, later minus earlier: the statistic as it was first read, before any
     conditioning on innings.
  2. Exact decomposition of that pooled change on the same resamples. With w_i the
     share of deployments in innings i and d_i the death-entry share within innings
     i, the pooled death share is sum_i w_i d_i, so its change splits exactly into
       mix component    = sum_i (w_i^later - w_i^earlier) * mean(d_i)
       timing component = sum_i mean(w_i) * (d_i^later - d_i^earlier)
     (means over the two seasons). The mix component is what the innings
     reallocation alone would have done to the pooled share with within-innings
     timing held fixed; the timing component is what changed inside innings.
  3. Innings reallocation: the second-innings share of deployments, later minus
     earlier, as a harness theme (bootstrap-t primary, BCa alongside).
  4. Within-innings entry timing: the death-entry share per innings, later minus
     earlier, one theme per innings.
  5. Equivalence verdicts for 3 and 4 against the pre-stated SESOI in config.toml:
     direction is a finding only under "rejects null"; otherwise the row is
     directional with its minimum detectable effect stated, or equivalent to zero.

Every rate is a ratio of per-match sums, so the harness machinery applies unchanged
and every quantity reads the same draws. A match with no deployment in an innings is
still a match: it keeps its resampling slot with zero totals for that innings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.equivalence import equivalence_from_theme_result
from mean_level_and_tail import season_contrast_bootstrap as scb


def load_impact_events(path) -> pd.DataFrame:
    """Impact-player substitutions only, one row per deployment. The concussion
    substitute is a different mechanism and is excluded via the reason field."""
    events = pd.read_parquet(path)
    events = events[events["reason"] == "impact_player"].copy()
    events["season"] = pd.to_numeric(events["season"], errors="coerce").astype(int)
    events["innings"] = pd.to_numeric(events["innings"], errors="coerce").astype(int)
    events["deploy_phase"] = events["deploy_phase"].astype(str)
    return events


def deployment_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Events and matches per season and innings, plus the phase mix within each
    innings (the descriptive shape behind the timing themes)."""
    counts = (events.groupby(["season", "innings"])
              .agg(events=("match_id", "size"), matches=("match_id", "nunique")).reset_index())
    mix = events.groupby(["innings", "season", "deploy_phase"]).size().rename("events").reset_index()
    mix["share"] = mix["events"] / mix.groupby(["innings", "season"])["events"].transform("sum")
    return counts, mix


# ----------------------------------------------------------------------------
# Per-match sums for every rate, on one match list per season
# ----------------------------------------------------------------------------

def _season_sums(season_events: pd.DataFrame, matches: np.ndarray, numer: pd.Series,
                 denom: pd.Series) -> scb.SeasonMatchSums:
    work = pd.DataFrame({"match_id": season_events["match_id"].to_numpy(),
                         "numer": numer.to_numpy(dtype=float), "denom": denom.to_numpy(dtype=float)})
    g = work.groupby("match_id").sum().reindex(matches, fill_value=0.0)
    return scb.SeasonMatchSums(matches, g["numer"].to_numpy(), g["denom"].to_numpy())


def build_deployment_themes(events: pd.DataFrame, earlier: int, later: int) -> dict:
    """Harness themes for the pooled death share, the second-innings share, and the
    per-innings death-entry shares, all over the same season match lists."""
    themes = {}
    sums = {}
    for season in (earlier, later):
        s = events[events["season"] == season]
        matches = np.sort(s["match_id"].unique())
        death = s["deploy_phase"].eq("death").astype(float)
        second = s["innings"].eq(2).astype(float)
        ones = pd.Series(1.0, index=s.index)
        sums[(season, "pooled_death")] = _season_sums(s, matches, death, ones)
        sums[(season, "second_innings")] = _season_sums(s, matches, second, ones)
        for inns in (1, 2):
            in_inns = s["innings"].eq(inns).astype(float)
            sums[(season, f"death_innings_{inns}")] = _season_sums(s, matches, death * in_inns, in_inns)
            sums[(season, f"share_innings_{inns}")] = _season_sums(s, matches, in_inns, ones)

    def diff(name):
        return scb.RateDifference(sums[(earlier, name)], sums[(later, name)])

    themes["pooled_death_entry_share_diff"] = scb.Theme(
        "pooled_death_entry_share_diff", diff("pooled_death"), 0.0, "well_powered",
        "Death-entry share of ALL deployments; pools the innings mix with timing.")
    themes["second_innings_share_diff"] = scb.Theme(
        "second_innings_share_diff", diff("second_innings"), 0.0, "well_powered",
        "Share of deployments made in the second innings. Negative = moved to the first innings.")
    for inns in (1, 2):
        themes[f"death_entry_share_diff_innings_{inns}"] = scb.Theme(
            f"death_entry_share_diff_innings_{inns}", diff(f"death_innings_{inns}"), 0.0, "well_powered",
            f"Death-entry share within innings {inns}; conditioning on innings removes the mix.")
    return themes, sums


# ----------------------------------------------------------------------------
# Decomposition of the pooled change on the shared draws
# ----------------------------------------------------------------------------

def _rate(counts, s: scb.SeasonMatchSums):
    num, den = counts @ s.numerator, counts @ s.denominator
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def decompose_pooled_change(sums: dict, earlier: int, later: int, counts_e, counts_l,
                            alpha: float) -> pd.DataFrame:
    """Pooled death-share change = mix component + timing component, with percentile
    intervals from the shared draws. The identity holds exactly on every resample."""
    def parts(counts_earlier, counts_later):
        w = {(season, i): _rate(c, sums[(season, f"share_innings_{i}")])
             for season, c in ((earlier, counts_earlier), (later, counts_later)) for i in (1, 2)}
        d = {(season, i): _rate(c, sums[(season, f"death_innings_{i}")])
             for season, c in ((earlier, counts_earlier), (later, counts_later)) for i in (1, 2)}
        mix = sum((w[(later, i)] - w[(earlier, i)]) * 0.5 * (d[(later, i)] + d[(earlier, i)]) for i in (1, 2))
        timing = sum(0.5 * (w[(later, i)] + w[(earlier, i)]) * (d[(later, i)] - d[(earlier, i)]) for i in (1, 2))
        pooled = (_rate(counts_later, sums[(later, "pooled_death")])
                  - _rate(counts_earlier, sums[(earlier, "pooled_death")]))
        return pooled, mix, timing

    n_e, n_l = counts_e.shape[1], counts_l.shape[1]
    point = parts(np.ones((1, n_e)), np.ones((1, n_l)))
    reps = parts(counts_e, counts_l)
    rows = []
    for name, p, r in zip(("pooled_death_entry_share_change", "innings_mix_component",
                           "within_innings_timing_component"), point, reps):
        r = r[np.isfinite(r)]
        rows.append({"component": name, "estimate": float(p[0]),
                     "ci_low": float(np.percentile(r, 100 * alpha / 2)),
                     "ci_high": float(np.percentile(r, 100 * (1 - alpha / 2))),
                     "p_positive": float((r > 0).mean()), "interval_method": "percentile"})
    table = pd.DataFrame(rows)
    table.attrs["identity_max_abs_error"] = float(np.nanmax(np.abs(reps[0] - reps[1] - reps[2])))
    return table


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run_deployment_analysis(events: pd.DataFrame, settings, seed: int,
                            n_iter: int | None = None) -> dict:
    """Counts and phase mix, the four themes on shared draws, the decomposition, and
    the equivalence verdicts for the reallocation and timing themes."""
    earlier, later = settings.earlier_season, settings.later_season
    n_iter = n_iter or settings.n_resamples
    alpha = 1.0 - settings.interval_level
    bootstrap_cfg = settings.section("bootstrap")
    sesoi_fraction = float(settings.section("sesoi")["impact_share_points"]) / 100.0

    counts, mix = deployment_counts(events)
    themes, sums = build_deployment_themes(events, earlier, later)
    n_e = sums[(earlier, "pooled_death")].n_matches
    n_l = sums[(later, "pooled_death")].n_matches
    counts_e, counts_l = scb.shared_resample_counts(n_e, n_l, n_iter, seed)

    results = {name: scb.run_theme(t, n_e, n_l, counts_e, counts_l, alpha=alpha,
                                   bias_flag_se_multiple=bootstrap_cfg["ratio_bias_flag_se_multiple"])
               for name, t in themes.items()}
    table = pd.DataFrame([r.as_row() for r in results.values()])
    for season, col in ((earlier, "share_earlier"), (later, "share_later")):
        table[col] = [float(_rate(np.ones((1, sums[(season, k)].n_matches)), sums[(season, k)])[0])
                      for k in ("pooled_death", "second_innings", "death_innings_1", "death_innings_2")]
    table["n_matches_earlier"], table["n_matches_later"] = n_e, n_l

    decomposition = decompose_pooled_change(sums, earlier, later, counts_e, counts_l, alpha)

    verdict_rows = []
    for name in ("second_innings_share_diff", "death_entry_share_diff_innings_1",
                 "death_entry_share_diff_innings_2"):
        row = equivalence_from_theme_result(
            results[name], sesoi=sesoi_fraction, mde_se_multiple=bootstrap_cfg["mde_se_multiple"],
            unit="percentage points", equivalence_level=settings.equivalence_level,
            interval_level=settings.interval_level)
        for col in ("estimate", "ci90_low", "ci90_high", "ci95_low", "ci95_high", "se", "sesoi", "mde_80pct_power"):
            row[col] = row[col] * 100.0
        row["direction_status"] = {
            "rejects_null": "established direction",
            "equivalent_to_zero": "equivalent to zero",
            "inconclusive": "directional; MDE stated",
        }[row["verdict"]]
        row["direction"] = "later lower" if row["estimate"] < 0 else "later higher"
        verdict_rows.append(row)
    verdicts = pd.DataFrame(verdict_rows)

    return {"counts": counts, "phase_mix": mix, "themes": table, "decomposition": decomposition,
            "verdicts": verdicts, "n_matches": (n_e, n_l)}


def plot_decomposition(decomposition: pd.DataFrame, interval_level: float):
    """The pooled death-share change beside its two exact components."""
    import matplotlib.pyplot as plt

    labels = {"pooled_death_entry_share_change": "pooled death-entry share change",
              "innings_mix_component": "innings-mix component",
              "within_innings_timing_component": "within-innings timing component"}
    rows = decomposition.set_index("component").loc[list(labels)]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.hlines(y, rows["ci_low"] * 100, rows["ci_high"] * 100, color="#2F5C8A", lw=2)
    ax.plot(rows["estimate"] * 100, y, "o", color="#2F5C8A", ms=7)
    ax.axvline(0, color="#C0392B", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[c] for c in rows.index], fontsize=9)
    ax.set_xlabel("percentage points, later minus earlier")
    ax.set_title("Where the pooled death-entry change comes from", loc="left", fontsize=10, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.01, -0.04, f"Match cluster bootstrap, shared draws, {int(interval_level * 100)}% percentile "
             "intervals. The two components sum to the pooled change on every resample.",
             fontsize=7.5, color="#6A6A6A", style="italic")
    fig.tight_layout()
    return fig
