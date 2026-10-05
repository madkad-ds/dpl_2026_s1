"""
Does the target explain how hard chasing sides go at the powerplay?

Chasing sides attacked the powerplay harder than sides batting first in both
seasons, and in 2024 their gear-up jumped by about a run an over. First-innings
totals were higher in 2024, so the obvious candidate is the target: a side with 210
to get may have to attack early in a way a side with 160 to get does not. This
module asks whether the target accounts for the 2024 change.

Unit: one chasing innings (one per match). Per innings: the target, the powerplay
attempt share, the powerplay run rate, and the gear-up (powerplay run rate minus the
same innings' middle-overs run rate, which in a chase can be cut short; innings
with too few middle-overs balls carry no gear-up).

Two reads, both under the match cluster bootstrap on shared draws:
  1. Binned: each metric by target band and season, with counts.
  2. A linear model per metric: metric = a + b * (target - 180) + c * later
     + d * (target - 180) * later. b is the response to the target per run in the
     earlier season; c is the season difference at a target of 180; d is the change
     in the response. The season difference adjusted to a common target (c) is
     compared with the raw season difference: if the raw jump shrinks to nothing
     once the target is held fixed, the target explains it.
Descriptive: a chase innings is also shaped by the pitch, the bowling and the
match situation, none of which is in the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import derived_columns as dc
from mean_level_and_tail import season_contrast_bootstrap as scb

TARGET_CENTRE = 180.0
TARGET_BANDS = [0, 160, 190, 400]
TARGET_LABELS = ["under 160", "160 to 189", "190 and over"]
MIN_MIDDLE_BALLS = 24


def chase_innings_table(frame: pd.DataFrame, settings) -> pd.DataFrame:
    """One row per chasing innings: target, powerplay attempt share, powerplay run
    rate, gear-up, and the balls behind each."""
    chase = frame[frame["inns"] == 2]
    pp = scb.in_phase(chase, settings.phase("powerplay"))
    mid = scb.in_phase(chase, settings.phase("middle"))
    faced = dc.is_ball_faced(chase).to_numpy()
    legal = dc.is_legal_delivery(chase).to_numpy()
    aggressive = dc.is_aggressive_shot(chase).to_numpy()
    score = pd.to_numeric(chase["score"], errors="coerce").fillna(0).to_numpy(dtype=float)
    work = pd.DataFrame({
        "season": chase["year"].to_numpy(), "p_match": chase["p_match"].to_numpy(),
        "team": chase["team_bat"].astype(str).to_numpy(),
        "target": pd.to_numeric(chase["target"], errors="coerce").to_numpy(),
        "pp_attempts": (aggressive & faced & pp).astype(float), "pp_faced": (faced & pp).astype(float),
        "pp_runs": np.where(legal & pp, score, 0.0), "pp_legal": (legal & pp).astype(float),
        "mid_runs": np.where(legal & mid, score, 0.0), "mid_legal": (legal & mid).astype(float),
    })
    agg = work.groupby(["season", "p_match", "team"]).agg(
        target=("target", "max"), pp_attempts=("pp_attempts", "sum"), pp_faced=("pp_faced", "sum"),
        pp_runs=("pp_runs", "sum"), pp_legal=("pp_legal", "sum"),
        mid_runs=("mid_runs", "sum"), mid_legal=("mid_legal", "sum")).reset_index()
    agg = agg[(agg["pp_faced"] > 0) & agg["target"].notna()].copy()
    agg["attempt_share"] = agg["pp_attempts"] / agg["pp_faced"]
    agg["pp_run_rate"] = agg["pp_runs"] / agg["pp_legal"] * 6.0
    with np.errstate(invalid="ignore", divide="ignore"):
        agg["gear_up"] = np.where(agg["mid_legal"] >= MIN_MIDDLE_BALLS,
                                  agg["pp_run_rate"] - agg["mid_runs"] / agg["mid_legal"] * 6.0, np.nan)
    agg["target_band"] = pd.cut(agg["target"], TARGET_BANDS, labels=TARGET_LABELS, right=False)
    return agg


def chase_phase_table(frame: pd.DataFrame, settings, phase_name: str, min_balls: int = 12) -> pd.DataFrame:
    """One row per chasing innings that had at least `min_balls` legal balls in the
    phase: target, attempt share, run rate in the phase, and for the death the
    gear-up over the same innings' middle overs. Chases that ended before the phase
    are absent, which is selection on match state; the share of chases reaching the
    phase per season is reported beside the metrics."""
    chase = frame[frame["inns"] == 2]
    in_phase = np.asarray(scb.in_phase(chase, settings.phase(phase_name)))
    mid = np.asarray(scb.in_phase(chase, settings.phase("middle")))
    faced = dc.is_ball_faced(chase).to_numpy()
    legal = dc.is_legal_delivery(chase).to_numpy()
    aggressive = dc.is_aggressive_shot(chase).to_numpy()
    score = pd.to_numeric(chase["score"], errors="coerce").fillna(0).to_numpy(dtype=float)
    work = pd.DataFrame({
        "season": chase["year"].to_numpy(), "p_match": chase["p_match"].to_numpy(),
        "team": chase["team_bat"].astype(str).to_numpy(),
        "target": pd.to_numeric(chase["target"], errors="coerce").to_numpy(),
        "attempts": (aggressive & faced & in_phase).astype(float), "faced": (faced & in_phase).astype(float),
        "runs": np.where(legal & in_phase, score, 0.0), "legal": (legal & in_phase).astype(float),
        "mid_runs": np.where(legal & mid, score, 0.0), "mid_legal": (legal & mid).astype(float),
    })
    agg = work.groupby(["season", "p_match", "team"]).agg(
        target=("target", "max"), attempts=("attempts", "sum"), faced=("faced", "sum"),
        runs=("runs", "sum"), legal=("legal", "sum"), mid_runs=("mid_runs", "sum"), mid_legal=("mid_legal", "sum")).reset_index()
    agg["reached_phase"] = agg["legal"] >= min_balls
    agg = agg[agg["target"].notna()].copy()
    agg["attempt_share"] = np.where(agg["faced"] > 0, agg["attempts"] / agg["faced"], np.nan)
    agg["run_rate"] = np.where(agg["legal"] > 0, agg["runs"] / agg["legal"] * 6.0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        agg["gear_up"] = np.where((phase_name == "death") & (agg["mid_legal"] >= MIN_MIDDLE_BALLS) & (agg["legal"] > 0),
                                  agg["run_rate"] - agg["mid_runs"] / agg["mid_legal"] * 6.0, np.nan)
    agg["target_band"] = pd.cut(agg["target"], TARGET_BANDS, labels=TARGET_LABELS, right=False)
    agg["phase"] = phase_name
    kept = agg[agg["reached_phase"]].copy()
    kept.attrs["reach_by_season"] = agg.groupby("season")["reached_phase"].mean().to_dict()
    kept.attrs["chases_by_season"] = agg.groupby("season").size().to_dict()
    return kept


def phase_target_response(innings: pd.DataFrame, settings, seed: int, n_iter: int, metrics=("attempt_share", "run_rate", "gear_up")) -> dict:
    """target_response for a phase table: same model, metrics named generically."""
    earlier, later = settings.earlier_season, settings.later_season
    alpha = 1.0 - settings.interval_level
    e = innings[innings["season"] == earlier].reset_index(drop=True)
    l = innings[innings["season"] == later].reset_index(drop=True)
    counts_e, counts_l = scb.shared_resample_counts(len(e), len(l), n_iter, seed)
    rows = []
    for metric in metrics:
        if metric not in innings.columns or innings[metric].notna().sum() < 20:
            continue
        keep_e, keep_l = e[metric].notna().to_numpy(), l[metric].notna().to_numpy()
        both = pd.concat([e[keep_e], l[keep_l]], ignore_index=True)
        y = both[metric].to_numpy(dtype=float)
        t = both["target"].to_numpy(dtype=float) - TARGET_CENTRE
        sn = (both["season"].to_numpy() == later).astype(float)
        design = np.column_stack([np.ones(len(y)), t, sn, t * sn])
        counts = np.hstack([counts_e[:, keep_e], counts_l[:, keep_l]])
        point = _weighted_ols(design, y, np.ones((1, len(y))))[0]
        reps = _weighted_ols(design, y, counts)
        raw_reps = (counts_l[:, keep_l] @ l.loc[keep_l, metric].to_numpy()) / counts_l[:, keep_l].sum(axis=1) \
            - (counts_e[:, keep_e] @ e.loc[keep_e, metric].to_numpy()) / counts_e[:, keep_e].sum(axis=1)
        raw_point = float(l.loc[keep_l, metric].mean() - e.loc[keep_e, metric].mean())
        names = ["intercept_at_180", "per_run_of_target", "season_difference_at_180", "change_in_target_response"]
        for j, name in enumerate(names):
            a = reps[:, j]
            rows.append({"phase": innings["phase"].iloc[0], "metric": metric, "term": name, "estimate": float(point[j]),
                         "ci_low": float(np.percentile(a, 100 * alpha / 2)), "ci_high": float(np.percentile(a, 100 * (1 - alpha / 2))),
                         "p_positive": float((a > 0).mean()), "n_innings": int(len(y))})
        rows.append({"phase": innings["phase"].iloc[0], "metric": metric, "term": "raw_season_difference", "estimate": raw_point,
                     "ci_low": float(np.percentile(raw_reps, 100 * alpha / 2)), "ci_high": float(np.percentile(raw_reps, 100 * (1 - alpha / 2))),
                     "p_positive": float((raw_reps > 0).mean()), "n_innings": int(len(y))})
    table = pd.DataFrame(rows)
    table["target_centre"] = TARGET_CENTRE
    return {"coefficients": table, "reach_by_season": innings.attrs.get("reach_by_season", {}),
            "chases_by_season": innings.attrs.get("chases_by_season", {})}


def binned_by_target(innings: pd.DataFrame) -> pd.DataFrame:
    rate_col = "pp_run_rate" if "pp_run_rate" in innings.columns else "run_rate"
    return (innings.groupby(["season", "target_band"], observed=True)
            .agg(innings=("p_match", "size"), mean_target=("target", "mean"),
                 attempt_share=("attempt_share", "mean"), run_rate=(rate_col, "mean"),
                 gear_up=("gear_up", "mean")).reset_index())


def _weighted_ols(design: np.ndarray, y: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Coefficients for every resample row of counts (n_iter x n) by weighted least
    squares on per-innings sufficient statistics."""
    xtx = np.einsum("ni,nj->nij", design, design).reshape(len(y), -1)
    xty = design * y[:, None]
    out = np.empty((counts.shape[0], design.shape[1]))
    for r in range(counts.shape[0]):
        a = (counts[r] @ xtx).reshape(design.shape[1], design.shape[1])
        b = counts[r] @ xty
        out[r] = np.linalg.lstsq(a, b, rcond=None)[0]
    return out


def target_response(innings: pd.DataFrame, settings, seed: int, n_iter: int) -> dict:
    """Per metric: the model coefficients with intervals, the raw season difference,
    and the season difference adjusted to a common target of 180."""
    earlier, later = settings.earlier_season, settings.later_season
    alpha = 1.0 - settings.interval_level
    e = innings[innings["season"] == earlier].reset_index(drop=True)
    l = innings[innings["season"] == later].reset_index(drop=True)
    counts_e, counts_l = scb.shared_resample_counts(len(e), len(l), n_iter, seed)
    rows = []
    for metric in ("attempt_share", "pp_run_rate", "gear_up"):
        keep_e, keep_l = e[metric].notna().to_numpy(), l[metric].notna().to_numpy()
        both = pd.concat([e[keep_e], l[keep_l]], ignore_index=True)
        y = both[metric].to_numpy(dtype=float)
        t = both["target"].to_numpy(dtype=float) - TARGET_CENTRE
        s = (both["season"].to_numpy() == later).astype(float)
        design = np.column_stack([np.ones(len(y)), t, s, t * s])
        # Resample multiplicities per innings, in the same order as `both`.
        counts = np.hstack([counts_e[:, keep_e], counts_l[:, keep_l]])
        point = _weighted_ols(design, y, np.ones((1, len(y))))[0]
        reps = _weighted_ols(design, y, counts)
        raw_reps = (counts_l[:, keep_l] @ l.loc[keep_l, metric].to_numpy()) / counts_l[:, keep_l].sum(axis=1) \
            - (counts_e[:, keep_e] @ e.loc[keep_e, metric].to_numpy()) / counts_e[:, keep_e].sum(axis=1)
        raw_point = float(l.loc[keep_l, metric].mean() - e.loc[keep_e, metric].mean())
        names = ["intercept_at_180", "per_run_of_target", "season_difference_at_180", "change_in_target_response"]
        for j, name in enumerate(names):
            a = reps[:, j]
            rows.append({"metric": metric, "term": name, "estimate": float(point[j]),
                         "ci_low": float(np.percentile(a, 100 * alpha / 2)), "ci_high": float(np.percentile(a, 100 * (1 - alpha / 2))),
                         "p_positive": float((a > 0).mean()), "n_innings": int(len(y))})
        rows.append({"metric": metric, "term": "raw_season_difference", "estimate": raw_point,
                     "ci_low": float(np.percentile(raw_reps, 100 * alpha / 2)), "ci_high": float(np.percentile(raw_reps, 100 * (1 - alpha / 2))),
                     "p_positive": float((raw_reps > 0).mean()), "n_innings": int(len(y))})
    table = pd.DataFrame(rows)
    table["target_centre"] = TARGET_CENTRE
    return {"coefficients": table, "mean_target": {earlier: float(e["target"].mean()), later: float(l["target"].mean())}}


def plot_target_response(innings: pd.DataFrame, metric: str, unit: str, title: str):
    """Chase innings: metric against target, by season, with a straight-line fit per season."""
    import matplotlib.pyplot as plt

    colours = {2023: "#2c7fb8", 2024: "#d95f0e"}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for season, rows in innings.dropna(subset=[metric]).groupby("season"):
        c = colours.get(int(season), "grey")
        ax.scatter(rows["target"], rows[metric], s=22, color=c, alpha=0.55, label=str(season), edgecolor="white", linewidth=0.5)
        coef = np.polyfit(rows["target"], rows[metric], 1)
        xs = np.linspace(rows["target"].min(), rows["target"].max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color=c, lw=1.6)
    ax.set_xlabel("target (runs)")
    ax.set_ylabel(unit)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, title="season")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
