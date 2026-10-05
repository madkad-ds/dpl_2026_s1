"""
Why pace displaced spin: selection, deployment, or workload?

Pace share rose in every phase in 2024. A league's spin share of legal balls is an
identity in three things a team decides and one it does not:

    spin share = (spinners named per innings)          selection
               x (share of named spinners who bowled)  deployment
               x (spin balls per spinner who bowled)   workload
               / (legal balls per innings)             innings length

with every factor a season-level ratio of sums over bowling innings, so the
identity holds exactly. The season change in the spin share is attributed to the
four factors by Shapley values (every ordering, increments averaged), per phase and
for the whole innings, under the match cluster bootstrap. Phase versions use the
phase's spin balls and legal balls with the same selection and deployment counts,
so they say whether the phase change was a change in how spinners were used within
the innings or in how many were picked.

Who is a spinner: a bowler whose legal balls across the two seasons are mostly
spin (`bowl_kind`), with at least a minimum count of balls bowled. Named spinners
per bowling innings are counted from the rosters (the starting eleven plus the
impact substitute, the twelve who could take the field); spinners who bowled are
counted from the ball-by-ball.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from common import derived_columns as dc
from mean_level_and_tail import season_contrast_bootstrap as scb

FACTORS = ["spinners_named_per_innings", "share_of_named_spinners_used", "spin_balls_per_spinner_used",
           "legal_balls_per_innings"]


def classify_spinners(frame: pd.DataFrame, min_balls: int) -> pd.DataFrame:
    """One row per bowler: legal balls, spin share of them, and whether he counts as a
    spinner (mostly spin, at or above the balls floor)."""
    legal = dc.is_legal_delivery(frame).to_numpy()
    spin = dc.is_spin(frame).to_numpy()
    work = pd.DataFrame({"p_bowl": frame["p_bowl"].to_numpy(), "legal": legal.astype(int), "spin": (legal & spin).astype(int)})
    g = work.groupby("p_bowl").sum()
    g["spin_share"] = g["spin"] / g["legal"].replace(0, np.nan)
    g["is_spinner"] = (g["spin_share"] >= 0.5) & (g["legal"] >= min_balls)
    names = frame.drop_duplicates("p_bowl").set_index("p_bowl")["bowl"]
    g["name"] = names.reindex(g.index)
    return g.reset_index()


def bowling_innings_table(frame: pd.DataFrame, rosters: pd.DataFrame, spinners: pd.DataFrame, settings) -> pd.DataFrame:
    """One row per bowling innings (match, bowling team): named spinners, spinners who
    bowled, spin and legal balls in each phase and overall."""
    spinner_ids = set(spinners.loc[spinners["is_spinner"], "p_bowl"])
    r = rosters.copy()
    r["match_id"] = r["match_id"].astype(int)
    named = r[r["role"].isin(["starting_xi", "impact_sub"])]
    named_spinners = (named.assign(is_spinner=named["dataset_id"].isin(spinner_ids))
                      .groupby(["match_id", "team"])["is_spinner"].sum().rename("spinners_named"))

    legal = dc.is_legal_delivery(frame).to_numpy()
    spin = dc.is_spin(frame).to_numpy()
    cols = {"legal_all": legal.astype(float), "spin_all": (legal & spin).astype(float)}
    for name in ("powerplay", "middle", "death"):
        m = np.asarray(scb.in_phase(frame, settings.phase(name)))
        cols[f"legal_{name}"] = (legal & m).astype(float)
        cols[f"spin_{name}"] = (legal & spin & m).astype(float)
    work = pd.DataFrame(cols)
    work["season"] = frame["year"].to_numpy(); work["p_match"] = frame["p_match"].to_numpy()
    work["team"] = frame["team_bowl"].astype(str).to_numpy(); work["p_bowl"] = frame["p_bowl"].to_numpy()
    per = work.groupby(["season", "p_match", "team"]).sum(numeric_only=True).reset_index()
    used = (work[work["p_bowl"].isin(spinner_ids) & (work["legal_all"] > 0)]
            .groupby(["p_match", "team"])["p_bowl"].nunique().rename("spinners_used"))
    per = per.merge(used.reset_index(), on=["p_match", "team"], how="left").fillna({"spinners_used": 0})
    per = per.merge(named_spinners.reset_index().rename(columns={"match_id": "p_match"}), on=["p_match", "team"], how="left")
    per["spinners_named"] = per["spinners_named"].fillna(per["spinners_used"]).clip(lower=per["spinners_used"])
    return per


def _factors(tot: np.ndarray, phase_prefix: str) -> dict:
    """tot rows: [innings, named, used, spin_all, legal_all, spin_p, legal_p]."""
    innings, named, used, spin, legal = tot[:, 0], tot[:, 1], tot[:, 2], tot[:, 5], tot[:, 6]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"spinners_named_per_innings": named / innings,
                "share_of_named_spinners_used": used / named,
                "spin_balls_per_spinner_used": spin / used,
                "legal_balls_per_innings": legal / innings,
                "spin_share": spin / legal}


def _spin_share(f):
    return (f["spinners_named_per_innings"] * f["share_of_named_spinners_used"] * f["spin_balls_per_spinner_used"]
            / f["legal_balls_per_innings"])


def _shapley(early, late):
    contributions = {k: np.zeros_like(early[FACTORS[0]]) for k in FACTORS}
    orders = list(permutations(FACTORS))
    for order in orders:
        current = {k: early[k] for k in FACTORS}
        previous = _spin_share(current)
        for k in order:
            current[k] = late[k]
            now = _spin_share(current)
            contributions[k] += now - previous
            previous = now
    return {k: v / len(orders) for k, v in contributions.items()}


def run_decomposition(innings: pd.DataFrame, settings, seed: int, n_iter: int) -> pd.DataFrame:
    """Per phase (and whole innings): the spin share per season, its change, and the
    Shapley contribution of each factor with a match bootstrap interval."""
    earlier, later = settings.earlier_season, settings.later_season
    alpha = 1.0 - settings.interval_level
    mats, matches = {}, {}
    phases = ["all", "powerplay", "middle", "death"]
    for season in (earlier, later):
        s = innings[innings["season"] == season]
        matches[season] = np.sort(s["p_match"].unique())
        m_ix = {m: i for i, m in enumerate(matches[season])}
        arr = {p: np.zeros((len(matches[season]), 7)) for p in phases}
        for _, row in s.iterrows():
            for p in phases:
                arr[p][m_ix[row["p_match"]]] += [1, row["spinners_named"], row["spinners_used"], row["spin_all"], row["legal_all"],
                                                  row[f"spin_{p}"], row[f"legal_{p}"]]
        mats[season] = arr
    n_e, n_l = len(matches[earlier]), len(matches[later])
    counts_e, counts_l = scb.shared_resample_counts(n_e, n_l, n_iter, seed)
    rows = []
    for p in phases:
        fe0, fl0 = _factors(np.ones((1, n_e)) @ mats[earlier][p], p), _factors(np.ones((1, n_l)) @ mats[later][p], p)
        fe, fl = _factors(counts_e @ mats[earlier][p], p), _factors(counts_l @ mats[later][p], p)
        point, reps = _shapley(fe0, fl0), _shapley(fe, fl)
        change_pt, change_rp = fl0["spin_share"][0] - fe0["spin_share"][0], fl["spin_share"] - fe["spin_share"]
        identity = float(np.nanmax(np.abs(sum(reps[k] for k in FACTORS) - change_rp)))
        for k in FACTORS:
            a = reps[k][np.isfinite(reps[k])]
            rows.append({"phase": p, "factor": k, f"value_{earlier}": float(fe0[k][0]), f"value_{later}": float(fl0[k][0]),
                         "contribution_points": 100 * float(point[k][0]),
                         "ci_low": 100 * float(np.percentile(a, 100 * alpha / 2)), "ci_high": 100 * float(np.percentile(a, 100 * (1 - alpha / 2))),
                         "p_positive": float((a > 0).mean()), "share_of_change": float(point[k][0] / change_pt) if change_pt else np.nan})
        a = change_rp[np.isfinite(change_rp)]
        rows.append({"phase": p, "factor": "spin_share_change", f"value_{earlier}": float(fe0["spin_share"][0]), f"value_{later}": float(fl0["spin_share"][0]),
                     "contribution_points": 100 * float(change_pt),
                     "ci_low": 100 * float(np.percentile(a, 100 * alpha / 2)), "ci_high": 100 * float(np.percentile(a, 100 * (1 - alpha / 2))),
                     "p_positive": float((a > 0).mean()), "share_of_change": 1.0, "identity_max_abs_error": identity})
    return pd.DataFrame(rows)


def selection_by_team(innings: pd.DataFrame) -> pd.DataFrame:
    """Mean named and used spinners per bowling innings by team and season."""
    g = innings.groupby(["team", "season"]).agg(innings=("p_match", "size"), spinners_named=("spinners_named", "mean"),
                                                spinners_used=("spinners_used", "mean"),
                                                spin_share=("spin_all", "sum"), legal=("legal_all", "sum"))
    g["spin_share"] = g["spin_share"] / g["legal"]
    return g.drop(columns="legal").reset_index().pivot(index="team", columns="season",
                                                        values=["spinners_named", "spinners_used", "spin_share"]).round(3)


def plot_contributions(table: pd.DataFrame, title: str):
    import matplotlib.pyplot as plt

    phases = ["all", "powerplay", "middle", "death"]
    labels = {"spinners_named_per_innings": "selection: spinners named", "share_of_named_spinners_used": "deployment: share used",
              "spin_balls_per_spinner_used": "workload: balls per spinner", "legal_balls_per_innings": "innings length"}
    colours = ["#2F5C8A", "#d95f0e", "#6c8ebf", "#9AA0A6"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(phases)); pos = np.zeros(len(phases)); neg = np.zeros(len(phases))
    for k, c in zip(FACTORS, colours):
        vals = np.array([float(table[(table["phase"] == p) & (table["factor"] == k)]["contribution_points"].iloc[0]) for p in phases])
        ax.bar(x, vals, bottom=np.where(vals >= 0, pos, neg), color=c, label=labels[k], width=0.6)
        pos += np.where(vals >= 0, vals, 0); neg += np.where(vals < 0, vals, 0)
    totals = [float(table[(table["phase"] == p) & (table["factor"] == "spin_share_change")]["contribution_points"].iloc[0]) for p in phases]
    ax.plot(x, totals, "k_", ms=22, mew=2, label="spin share change")
    ax.axhline(0, color="#C0392B", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(["whole innings", "powerplay", "middle", "death"])
    ax.set_ylabel("change in spin share, percentage points")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
