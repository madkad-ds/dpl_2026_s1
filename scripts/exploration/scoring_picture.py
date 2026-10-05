"""
The season-versus-season scoring picture at innings level: totals, run rate,
boundary and dot shares, wickets; the shape of each distribution and the tail
mass above 180, 200 and 220; and whether the rise in very high totals is spread
across grounds or concentrated where the fixture list changed.

Full 120-ball innings only, so rates are like for like. Dot balls here are legal
balls with no runs at all (`score == 0`), the innings-level definition; the
analyses use the batter-level definition (`batruns == 0` on a faced ball).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEASON_COLOURS = {2023: "#4C72B0", 2024: "#DD8452"}


def macro_scoring_picture(df, exclude_reduced=True):
    """Season-vs-season macro batting picture, per completed innings: totals, run rate,
    boundary %, dot %, wickets. Reports distribution shape (quantiles, tail mass), not
    just means, because a shift can live in the tail (more very-high scores) while the
    average barely moves. Restricts to full 120-ball innings by default so rate
    comparisons aren't distorted by rain-reduced games.
    """
    d = df.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    for c in ["batruns","score","max_balls","inns_runs","inns_balls"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Legal-ball flag and boundary/dot flags at delivery level.
    d["is_legal"] = (pd.to_numeric(d["noball"],errors="coerce").fillna(0)==0) & \
                    (pd.to_numeric(d["wide"],errors="coerce").fillna(0)==0)
    d["is_boundary"] = d["batruns"].isin([4,6])
    d["is_dot"] = d["is_legal"] & (d["score"]==0)          # legal ball, no runs at all
    d["is_wicket"] = d["out"].astype(str).str.strip().str.lower()=="true"

    # Optionally drop rain-reduced innings so run-rate/total comparisons are like-for-like.
    if exclude_reduced:
        d = d[d["max_balls"]==120]

    # Aggregate to one row per (match, innings).
    g = d.groupby(["p_match","inns","year"])
    innings = g.agg(
        total=("score","sum"),
        legal_balls=("is_legal","sum"),
        boundaries=("is_boundary","sum"),
        dots=("is_dot","sum"),
        wickets=("is_wicket","sum"),
        deliveries=("is_legal","size"),
    ).reset_index()
    innings["run_rate"] = innings["total"] / (innings["legal_balls"]/6)
    innings["boundary_pct"] = innings["boundaries"] / innings["legal_balls"]
    innings["dot_pct"] = innings["dots"] / innings["legal_balls"]

    metrics = ["total","run_rate","boundary_pct","dot_pct","wickets"]

    # --- Distribution summary per season: mean, spread, quantiles, and tail mass ---
    print("=== Distribution summary by season (shape, not just mean) ===\n")
    for m in metrics:
        summ = innings.groupby("year")[m].agg(
            mean="mean", std="std",
            p10=lambda s: s.quantile(0.10),
            p50="median",
            p90=lambda s: s.quantile(0.90),
        ).round(3)
        print(f"{m}:")
        print(summ.to_string())
        # tail mass for totals: fraction of innings above a high threshold
        if m == "total":
            for thresh in (180, 200, 220):
                frac = innings.groupby("year")["total"].apply(lambda s: (s>=thresh).mean()).round(3)
                print(f"  fraction of innings >= {thresh}: {dict(frac)}")
        print()

    # --- n per season, so the reader knows the sample behind each distribution ---
    print("Innings per season (post-filter):")
    print(innings.groupby("year").size().to_string())
    return innings


def overlaid_distributions(innings, metrics=None):
    """Overlaid per-season KDEs for each macro metric, so a shift in distribution shape
    (e.g. a fatter high-scoring tail) is visible, not just a shift in the mean. A filled
    density per season plus a dashed median line lets central-tendency and shape shifts
    be read at once."""
    if metrics is None:
        metrics = ["total", "run_rate", "boundary_pct", "dot_pct", "wickets"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for season, colour in SEASON_COLOURS.items():
            vals = innings.loc[innings["year"] == season, metric].dropna()
            if vals.nunique() < 2:                     # KDE needs spread; skip degenerate series
                continue
            sns.kdeplot(vals, ax=ax, color=colour, fill=True, alpha=0.3,
                        linewidth=2, label=str(season))
            ax.axvline(vals.median(), color=colour, linestyle="--", linewidth=1)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.legend(title="season", fontsize=8)
    fig.tight_layout()
    return fig


def venue_tail_check(innings, df, high_score=220):
    """Test whether the rise in very-high totals is league-wide or driven by a few grounds
    hosting more matches in 2024. Attaches each innings to its ground, then compares, per
    season: how many high-scoring innings each ground produced, and — the real confound —
    whether the ground's SHARE of innings shifted between seasons. A tail rise spread
    across many grounds is tactical; one concentrated in grounds that also hosted more
    2024 matches is fixture-mix.
    """
    ground = df.drop_duplicates("p_match")[["p_match", "ground"]]
    inns = innings.merge(ground, on="p_match", how="left")
    inns["is_high"] = inns["total"] >= high_score

    # High-scoring innings by ground and season.
    high_by_ground = (inns[inns["is_high"]]
                      .groupby(["ground", "year"]).size()
                      .unstack(fill_value=0))
    # Total innings by ground and season (the exposure — did the mix shift?).
    innings_by_ground = inns.groupby(["ground", "year"]).size().unstack(fill_value=0)
    # High-score RATE per ground per season (controls for how many matches each hosted).
    high_rate = (high_by_ground / innings_by_ground).fillna(0).round(3)

    table = pd.concat({
        "innings": innings_by_ground,
        f"high(>={high_score})": high_by_ground,
        "high_rate": high_rate,
    }, axis=1)
    print(f"High-scoring innings (>= {high_score}) by ground and season:\n")
    print(table.sort_values(("high_rate", 2024), ascending=False).to_string())

    # How many DISTINCT grounds produced a high score each season? (spread vs concentration)
    grounds_with_high = (high_by_ground > 0).sum()
    print(f"\nDistinct grounds producing a >= {high_score} innings:")
    print(grounds_with_high.to_string())

    return inns, high_by_ground, innings_by_ground, high_rate



def plot_venue_tail(high_by_ground, innings_by_ground, high_score=220):
    """Two-panel read: (left) count of high scores per ground per season — is the rise
    on many grounds or a few; (right) each ground's share of total innings per season —
    did the fixture mix tilt toward grounds that happen to be high-scoring. If left shows
    broad rises and right shows a stable mix, the tail rise is tactical, not fixture-driven.
    """
    grounds = innings_by_ground.index.tolist()
    y = np.arange(len(grounds))
    h = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.5 * len(grounds) + 2), sharey=True)

    for i, season in enumerate((2023, 2024)):
        offset = (i - 0.5) * h
        hb = high_by_ground.get(season, pd.Series(0, index=grounds)).reindex(grounds, fill_value=0)
        ax1.barh(y + offset, hb.values, height=h, color=SEASON_COLOURS[season], label=str(season))
        # right panel: share of that season's innings hosted at each ground
        shares = (innings_by_ground.get(season, pd.Series(0, index=grounds))
                  .reindex(grounds, fill_value=0))
        shares = shares / shares.sum()
        ax2.barh(y + offset, shares.values, height=h, color=SEASON_COLOURS[season], label=str(season))

    ax1.set_yticks(y); ax1.set_yticklabels(grounds, fontsize=8)
    ax1.set_xlabel(f"# innings >= {high_score}")
    ax1.set_title("High scores per ground")
    ax1.legend(title="season", fontsize=8)
    ax2.set_xlabel("share of season's innings")
    ax2.set_title("Fixture mix (share of innings by ground)")
    ax2.legend(title="season", fontsize=8)
    fig.tight_layout()
    return fig
