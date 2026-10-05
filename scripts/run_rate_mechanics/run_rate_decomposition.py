"""
Where did the extra runs come from? An exact decomposition of the run-rate change.

The run rate in a phase is six times the runs per legal ball. Runs per legal ball is
an identity in a few mechanical rates, all per legal ball in the phase:

    runs/ball = 4 f + 6 s + (1 - f - s) (1 - q) rho + e

    f    fours per legal ball
    s    sixes per legal ball
    q    dot share among non-boundary legal balls (no runs off the bat)
    rho  runs off the bat per non-boundary scoring ball (mostly 1, sometimes 2 or 3)
    e    byes and leg byes per legal ball

A second arrangement of the same identity separates how often the ball was hit to
the fence from how big the hit was:

    runs/ball = B (4 + 2 sigma) + (1 - B) (1 - q) rho + e

    B      boundaries per legal ball
    sigma  six share among boundaries (a six adds two runs to a four)

The season change in the run rate is attributed to the components exactly by
Shapley values: for every ordering of the components, each is moved from its
earlier-season to its later-season value in turn and the increment recorded; the
average over all orderings is the component's contribution, and the contributions
sum to the observed change. Every quantity is built from per-match sums, so the
match cluster bootstrap redoes the decomposition on each draw and the identity holds
on every resample.

Population: all innings, the same basis as the phase run-rate contrasts. Wides and
no-balls are not legal balls and do not enter (the run-rate convention of the
project). A no-ball's bat runs are therefore excluded here as they are everywhere.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from common import derived_columns as dc
from mean_level_and_tail import season_contrast_bootstrap as scb

COMPONENTS_A = ["fours", "sixes", "dots", "running", "extras"]
COMPONENTS_B = ["boundary_frequency", "six_share_of_boundaries", "dots", "running", "extras"]
SUM_COLUMNS = ["legal", "fours", "sixes", "nonboundary", "nonboundary_dots", "nonboundary_batruns", "extras"]


def per_match_phase_sums(frame: pd.DataFrame, phase, earlier: int, later: int) -> dict:
    """Per season: matches x SUM_COLUMNS on legal balls in the phase."""
    legal = dc.is_legal_delivery(frame).to_numpy()
    in_phase = scb.in_phase(frame, phase) if phase is not None else np.ones(len(frame), bool)
    scope = legal & in_phase
    batruns = pd.to_numeric(frame["batruns"], errors="coerce").fillna(0).to_numpy(dtype=float)
    byes = pd.to_numeric(frame["byes"], errors="coerce").fillna(0).to_numpy(dtype=float)
    legbyes = pd.to_numeric(frame["legbyes"], errors="coerce").fillna(0).to_numpy(dtype=float)
    four = dc.is_four(frame).to_numpy()
    six = dc.is_six(frame).to_numpy()
    nonboundary = ~(four | six)
    work = pd.DataFrame({
        "year": frame["year"].to_numpy(), "p_match": frame["p_match"].to_numpy(),
        "legal": scope.astype(float), "fours": (scope & four).astype(float), "sixes": (scope & six).astype(float),
        "nonboundary": (scope & nonboundary).astype(float),
        "nonboundary_dots": (scope & nonboundary & (batruns == 0)).astype(float),
        "nonboundary_batruns": np.where(scope & nonboundary, batruns, 0.0),
        "extras": np.where(scope, byes + legbyes, 0.0),
    })
    out = {}
    for season in (earlier, later):
        s = work[work["year"] == season]
        matches = np.sort(frame.loc[frame["year"].to_numpy() == season, "p_match"].unique())
        g = s.groupby("p_match")[SUM_COLUMNS].sum().reindex(matches, fill_value=0.0)
        out[season] = g.to_numpy(dtype=float)
    return out


def component_rates(totals: np.ndarray) -> dict:
    """Component rates from summed columns (rows = resamples)."""
    legal, fours, sixes, nonb, nonb_dots, nonb_runs, extras = totals.T
    with np.errstate(invalid="ignore", divide="ignore"):
        f = fours / legal
        s = sixes / legal
        q = nonb_dots / nonb
        scoring = nonb - nonb_dots
        rho = np.where(scoring > 0, nonb_runs / scoring, 0.0)
        e = extras / legal
        boundary = f + s
        sigma = np.where(boundary > 0, s / boundary, 0.0)
    return {"fours": f, "sixes": s, "dots": q, "running": rho, "extras": e,
            "boundary_frequency": boundary, "six_share_of_boundaries": sigma,
            "run_rate": 6.0 * (4 * f + 6 * s + (1 - f - s) * (1 - q) * rho + e)}


def _run_rate_a(c):
    return 6.0 * (4 * c["fours"] + 6 * c["sixes"] + (1 - c["fours"] - c["sixes"]) * (1 - c["dots"]) * c["running"] + c["extras"])


def _run_rate_b(c):
    return 6.0 * (c["boundary_frequency"] * (4 + 2 * c["six_share_of_boundaries"])
                  + (1 - c["boundary_frequency"]) * (1 - c["dots"]) * c["running"] + c["extras"])


def shapley_contributions(early: dict, late: dict, components: list, run_rate_fn) -> dict:
    """Exact Shapley attribution of run_rate_fn(late) - run_rate_fn(early) over the
    named components, averaging the sequential increments over every ordering."""
    contributions = {c: np.zeros_like(early[components[0]]) for c in components}
    orders = list(permutations(components))
    for order in orders:
        current = {c: early[c] for c in components}
        previous = run_rate_fn(current)
        for c in order:
            current[c] = late[c]
            now = run_rate_fn(current)
            contributions[c] += now - previous
            previous = now
    return {c: v / len(orders) for c, v in contributions.items()}


def decompose_phase(frame: pd.DataFrame, settings, seed: int, phase_name: str | None,
                    n_iter: int, counts: tuple | None = None) -> dict:
    """Both arrangements of the decomposition for one phase (None = whole innings),
    with component rates per season and intervals on every contribution."""
    earlier, later = settings.earlier_season, settings.later_season
    alpha = 1.0 - settings.interval_level
    phase = settings.phase(phase_name) if phase_name else None
    sums = per_match_phase_sums(frame, phase, earlier, later)
    n_e, n_l = sums[earlier].shape[0], sums[later].shape[0]
    counts = counts or scb.shared_resample_counts(n_e, n_l, n_iter, seed)
    counts_e, counts_l = counts

    def rates_for(c_e, c_l):
        return component_rates(c_e @ sums[earlier]), component_rates(c_l @ sums[later])

    pe, pl = rates_for(np.ones((1, n_e)), np.ones((1, n_l)))
    re, rl = rates_for(counts_e, counts_l)

    rows = []
    for name, comps, fn in (("fours_sixes_dots_running_extras", COMPONENTS_A, _run_rate_a),
                            ("boundary_frequency_and_size", COMPONENTS_B, _run_rate_b)):
        point = shapley_contributions(pe, pl, comps, fn)
        reps = shapley_contributions(re, rl, comps, fn)
        change_point = float(pl["run_rate"][0] - pe["run_rate"][0])
        change_reps = rl["run_rate"] - re["run_rate"]
        for c in comps:
            a = reps[c][np.isfinite(reps[c])]
            rows.append({"phase": phase_name or "whole_innings", "arrangement": name, "component": c,
                         f"rate_{earlier}": float(pe[c][0]), f"rate_{later}": float(pl[c][0]),
                         "contribution_rpo": float(point[c][0]),
                         "ci_low": float(np.percentile(a, 100 * alpha / 2)), "ci_high": float(np.percentile(a, 100 * (1 - alpha / 2))),
                         "p_positive": float((a > 0).mean()),
                         "share_of_change": float(point[c][0] / change_point) if change_point else np.nan})
        a = change_reps[np.isfinite(change_reps)]
        rows.append({"phase": phase_name or "whole_innings", "arrangement": name, "component": "run_rate_change",
                     f"rate_{earlier}": float(pe["run_rate"][0]), f"rate_{later}": float(pl["run_rate"][0]),
                     "contribution_rpo": change_point,
                     "ci_low": float(np.percentile(a, 100 * alpha / 2)), "ci_high": float(np.percentile(a, 100 * (1 - alpha / 2))),
                     "p_positive": float((a > 0).mean()), "share_of_change": 1.0})
        identity = np.nanmax(np.abs(sum(reps[c] for c in comps) - change_reps))
        rows[-1]["identity_max_abs_error"] = float(identity)
    return {"table": pd.DataFrame(rows), "counts": counts}


def run_all_phases(frame: pd.DataFrame, settings, seed: int, n_iter: int | None = None) -> pd.DataFrame:
    n_iter = n_iter or settings.n_resamples
    tables, counts = [], None
    for phase_name in ("powerplay", "middle", "death", None):
        out = decompose_phase(frame, settings, seed, phase_name, n_iter, counts)
        counts = out["counts"]
        tables.append(out["table"])
    return pd.concat(tables, ignore_index=True)


def plot_contributions(table: pd.DataFrame, arrangement: str, title: str):
    """Stacked contribution per phase, with the total change marked."""
    import matplotlib.pyplot as plt

    sub = table[(table["arrangement"] == arrangement) & (table["component"] != "run_rate_change")]
    phases = ["powerplay", "middle", "death", "whole_innings"]
    comps = list(dict.fromkeys(sub["component"]))
    colours = dict(zip(comps, ["#2F5C8A", "#d95f0e", "#6c8ebf", "#8c564b", "#9AA0A6"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(phases))
    pos = np.zeros(len(phases)); neg = np.zeros(len(phases))
    for c in comps:
        vals = np.array([float(sub[(sub["phase"] == p) & (sub["component"] == c)]["contribution_rpo"].iloc[0]) for p in phases])
        bottoms = np.where(vals >= 0, pos, neg)
        ax.bar(x, vals, bottom=bottoms, color=colours[c], label=c.replace("_", " "), width=0.6)
        pos += np.where(vals >= 0, vals, 0); neg += np.where(vals < 0, vals, 0)
    totals = [float(table[(table["arrangement"] == arrangement) & (table["phase"] == p) & (table["component"] == "run_rate_change")]["contribution_rpo"].iloc[0]) for p in phases]
    ax.plot(x, totals, "k_", ms=22, mew=2, label="run-rate change")
    ax.axhline(0, color="#C0392B", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([p.replace("_", " ") for p in phases])
    ax.set_ylabel("runs per over, later minus earlier")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
