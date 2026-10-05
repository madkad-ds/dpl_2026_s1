"""
Where the runs went and which shots were played, then which teams moved with
the league: scoring and boundary geography on the batter-relative wagon zones,
zonal entropy, the shot repertoire, and per-team changes in run rate, wicket
rate, boundary share, pace share and middle-overs run rate. Team-season cells
hold about fourteen matches, so team results are read as tiers, not fine ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .phase_and_bowling import FALL, GRID, RISE


def spatial_shot_evolution(df, exclude_reduced=True):
    """Season-vs-season scoring geography and shot repertoire. Geography uses the
    handedness-normalised wagon zones (batter-relative, so LHB/RHB pool directly) on
    balls with real placement only. Shot repertoire uses the shot column, excluding the
    known-mislabelled leaves (shot_suspect). Reports where runs/boundaries are hit and
    which shots are played, per season, to locate the spatial signature of the scoring rise.
    """
    d = df.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    for c in ["batruns","max_balls","wagonzone_clean"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if exclude_reduced:
        d = d[d["max_balls"] == 120]

    d["is_boundary"] = d["batruns"].isin([4,6])
    d["is_six"] = d["batruns"] == 6

    # --- Scoring geography: zone distribution of scoring shots, per season ---
    # Only balls with real placement and actual runs (a zone is meaningful when the ball
    # went somewhere). Zones are batter-relative (already handedness-normalised).
    placed = d[d["wagonzone_clean"].notna() & (d["batruns"] > 0)].copy()

    def zone_share(g):
        vc = g["wagonzone_clean"].value_counts(normalize=True)
        return vc.reindex(range(1,9), fill_value=0)
    zone_dist = placed.groupby("year").apply(zone_share).round(3)
    print("=== Scoring-shot zone distribution by season (share of scoring shots) ===")
    print("(zones are batter-relative; 1-8 around the ground)\n")
    print(zone_dist.to_string())

    # --- Boundary geography: where boundaries specifically come from ---
    bnd = d[d["wagonzone_clean"].notna() & d["is_boundary"]].copy()
    bnd_dist = bnd.groupby("year").apply(
        lambda g: g["wagonzone_clean"].value_counts(normalize=True).reindex(range(1,9), fill_value=0)
    ).round(3)
    print("\n=== Boundary zone distribution by season (share of boundaries) ===\n")
    print(bnd_dist.to_string())

    # --- Zonal diversity: how many zones does the average scoring shot spread across? ---
    # Entropy of the zone distribution per season (higher = runs spread more evenly =
    # harder to set a field to). Uses natural log; max entropy for 8 zones is ln(8)=2.08.
    def entropy(shares):
        p = shares[shares > 0]
        return -(p * np.log(p)).sum()
    print("\n=== Zonal entropy of scoring shots (higher = more spread) ===")
    for yr in sorted(placed["year"].dropna().unique()):
        e = entropy(zone_dist.loc[yr])
        print(f"  {int(yr)}: {e:.3f}  (max ln(8) = {np.log(8):.3f})")

    # --- Shot repertoire: share of each shot type, per season ---
    shots = d[d["shot"].notna()].copy()   # the left-alone contradictions are few and stay in
    shots["shot"] = shots["shot"].astype(str).str.strip().str.upper()
    shot_share = (shots.groupby(["year","shot"]).size()
                  .groupby(level=0).transform(lambda s: s/s.sum()))
    shot_tab = shot_share.unstack("year").round(3)
    shot_tab["delta"] = (shot_tab.get(2024,0) - shot_tab.get(2023,0)).round(3)
    print("\n=== Shot repertoire share by season (top movers) ===\n")
    print(shot_tab.reindex(shot_tab["delta"].abs().sort_values(ascending=False).index).head(15).to_string())

    return zone_dist, bnd_dist, shot_tab


def team_theme_attribution_full(df, exclude_reduced=True, min_balls=150):
    """Attribute the remaining league themes to teams: efficient aggression (scoring up
    vs wickets), the pace/spin bowling shift, boundary-vs-rotation scoring mechanism, and
    whether any team bucked the middle-overs lag. Team-season cells (~14 matches) are
    noisy, so results are read as tiers (clear movers / middle / laggards), not fine ranks.
    """
    d = df.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["over_num"] = pd.to_numeric(d["over_num"], errors="coerce")
    for c in ["batruns","score","bowler_charge","max_balls"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if exclude_reduced:
        d = d[d["max_balls"] == 120]
    d["is_legal"] = (pd.to_numeric(d["noball"],errors="coerce").fillna(0)==0) & \
                    (pd.to_numeric(d["wide"],errors="coerce").fillna(0)==0)
    d["is_boundary"] = d["batruns"].isin([4,6])
    d["is_wicket"] = d["out"].astype(str).str.strip().str.lower()=="true"
    d["phase"] = pd.cut(d["over_num"], bins=[0,6,15,20], labels=["powerplay","middle","death"])
    d["kind3"] = d["bowl_kind"].astype(str).str.strip().str.lower().map(
        lambda k: "pace" if "pace" in k else "spin" if "spin" in k else "other")

    # --- Efficient aggression: batting run rate delta vs wicket rate delta, per team ---
    bat = d.groupby(["team_bat","year"]).agg(
        legal=("is_legal","sum"), runs=("score","sum"), wkts=("is_wicket","sum")).reset_index()
    bat = bat[bat["legal"] >= min_balls*4]   # full-innings floor (all phases)
    bat["run_rate"] = bat["runs"]/(bat["legal"]/6)
    bat["wkt_rate"] = bat["wkts"]/bat["legal"]
    rr = bat.pivot(index="team_bat", columns="year", values="run_rate")
    wr = bat.pivot(index="team_bat", columns="year", values="wkt_rate")
    eff = pd.DataFrame({"run_rate_delta": (rr.get(2024)-rr.get(2023)).round(3),
                        "wkt_rate_delta": (wr.get(2024)-wr.get(2023)).round(4)})
    # "Efficient" = scoring up AND wickets flat/down (positive rr delta, <=0 wkt delta)
    eff["efficient"] = (eff["run_rate_delta"] > 0) & (eff["wkt_rate_delta"] <= 0)
    print("=== Efficient aggression by team (scoring up, wickets flat/down = efficient) ===\n")
    print(eff.sort_values("run_rate_delta", ascending=False).to_string())

    # --- Bowling pace shift: change in pace share of a team's bowling, per team ---
    bowl = d[d["is_legal"] & d["kind3"].isin(["pace","spin"])].copy()
    pace_share = (bowl.groupby(["team_bowl","year"])["kind3"]
                  .apply(lambda s: (s=="pace").mean()).unstack("year"))
    pace_share["delta"] = (pace_share.get(2024)-pace_share.get(2023)).round(3)
    print("\n=== Pace-share shift by team (who pivoted to pace) ===\n")
    print(pace_share.round(3).sort_values("delta", ascending=False).to_string())

    # --- Boundary mechanism: boundary% delta per team (rise = boundary-driven) ---
    bnd = d[d["is_legal"]].groupby(["team_bat","year"]).agg(
        legal=("is_legal","sum"), bnd=("is_boundary","sum")).reset_index()
    bnd = bnd[bnd["legal"] >= min_balls*4]
    bnd["boundary_pct"] = bnd["bnd"]/bnd["legal"]
    bp = bnd.pivot(index="team_bat", columns="year", values="boundary_pct")
    bp["delta"] = (bp.get(2024)-bp.get(2023)).round(3)
    print("\n=== Boundary% shift by team ===\n")
    print(bp.round(3).sort_values("delta", ascending=False).to_string())

    # --- Middle-overs exception: which teams (if any) accelerated the middle ---
    mid = d[d["phase"]=="middle"].groupby(["team_bat","year"]).agg(
        legal=("is_legal","sum"), runs=("score","sum")).reset_index()
    mid = mid[mid["legal"] >= min_balls]
    mid["run_rate"] = mid["runs"]/(mid["legal"]/6)
    mp = mid.pivot(index="team_bat", columns="year", values="run_rate")
    mp["delta"] = (mp.get(2024)-mp.get(2023)).round(3)
    print("\n=== Middle-overs run rate shift by team (league lagged here — any exceptions?) ===\n")
    print(mp.round(3).sort_values("delta", ascending=False).to_string())

    return eff, pace_share, bp, mp


def team_theme_heatmap(frame, title, value_fmt=None):
    """Render one team x theme heatmap, ordered by the frame's own mean alignment so the
    drivers sit at the top for THIS theme (not inherited from another figure's order).
    Colour is z-scored across teams (shared RdBu_r meaning: red = moved with the league's
    2024 evolution, blue = against); cells annotated with raw deltas so magnitude is kept.
    Sign conventions are set by the caller so 'red = aligned with the league' holds for
    every theme (e.g. wickets are passed as 'wickets saved', already flipped).
    """
    frame = frame.dropna(how="all")
    z = (frame - frame.mean()) / frame.std(ddof=0)
    order = z.mean(axis=1).sort_values(ascending=False).index      # self-ordered
    frame, z = frame.loc[order], z.loc[order]

    fig, ax = plt.subplots(figsize=(1.7 + 1.4 * frame.shape[1], 0.5 * len(frame) + 1.5))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")

    ax.set_xticks(range(frame.shape[1]))
    ax.set_xticklabels(frame.columns, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels(frame.index, fontsize=9)

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            raw, zval = frame.iloc[i, j], z.iloc[i, j]
            fmt = value_fmt.get(frame.columns[j], "{:+.3f}") if value_fmt else "{:+.3f}"
            ax.annotate(fmt.format(raw), (j, i), ha="center", va="center",
                        fontsize=8, color="white" if abs(zval) > 1.3 else "#222222")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("standardised change across teams\n(red = with the league's 2024 evolution)",
                   fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.text(0.5, -0.08,
             "Colour z-scored across teams; numbers are raw change (2024 − 2023). Teams ordered "
             "by alignment on this theme group. ~14 matches/team/season — read the column pattern, "
             "not single cells.",
             ha="center", fontsize=8, color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def team_metric_bars(series, title, unit, value_fmt="{:+.3f}"):
    """Single-metric team ranking as a horizontal bar chart — the right format for one
    theme, avoiding the thin-sliver look and colorbar overlap of a one-column heatmap.
    Bars sorted by value; colour carries sign (red = moved with the league's 2024
    evolution, blue = against). Sign conventions set by the caller so red is always
    'aligned' (e.g. pass wickets as 'wickets saved', already flipped).
    """
    s = series.dropna().sort_values()          # ascending so largest sits at the top bar
    colours = [RISE if v >= 0 else FALL for v in s.values]

    fig, ax = plt.subplots(figsize=(7, 0.42 * len(s) + 1.2))
    bars = ax.barh(range(len(s)), s.values, color=colours, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=9)

    # Value labels just outside each bar end, so no axis-reading needed.
    xpad = (s.max() - s.min()) * 0.02 or 0.01
    for i, v in enumerate(s.values):
        ax.annotate(value_fmt.format(v), (v, i),
                    xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=8, color="#222222")

    ax.set_xlabel(f"change 2024 − 2023 ({unit})", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    # headroom so labels don't clip at the extremes
    pad = (s.max() - s.min()) * 0.18 or 0.02
    ax.set_xlim(min(0, s.min()) - pad, max(0, s.max()) + pad)
    fig.tight_layout()
    return fig
