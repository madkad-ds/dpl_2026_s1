"""
Presentation figures, one function per figure, each reading the written outputs of
the analysis notebooks (never rerunning an analysis) and drawing at slide size in
the presentation style. Every function returns (fig, numbers): the numbers are the
values the figure used, so the text registry can be filled and checked without
reading the image.

Reads from outputs/<analysis>/ as written by the notebooks:
  mean_level_and_tail/theme_contrasts.csv
  run_rate_mechanics/run_rate_decomposition.csv
  expected_wickets/powerplay_contrast.csv, fixed_vs_refit_intervals.csv, shot_level_rung.csv
  win_probability_ledger/powerplay_ledger.csv, powerplay_aggression.csv
  powerplay_risk_ledger/per_team_season_ledger.csv
  bowling_response/delivery_menu.csv, shift_share_components.csv
  spin_displacement/spin_share_decomposition.csv
  batter_continuity/powerplay_six_rate_components.csv, composition_by_auction_route.csv
  impact_substitute_mechanics/chase_target_response.csv, chase_target_response_middle_death.csv,
      deployments_by_side_innings.csv, substitute_contribution_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

from . import style as st

TEAM_SHORT = {"Sunrisers Hyderabad": "Sunrisers", "Kolkata Knight Riders": "Kolkata", "Delhi Capitals": "Delhi",
              "Royal Challengers Bengaluru": "Bengaluru", "Mumbai Indians": "Mumbai", "Chennai Super Kings": "Chennai",
              "Rajasthan Royals": "Rajasthan", "Punjab Kings": "Punjab", "Lucknow Super Giants": "Lucknow", "Gujarat Titans": "Gujarat"}


def _read(outputs: Path, analysis: str, name: str) -> pd.DataFrame:
    path = Path(outputs) / analysis / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run {analysis}.ipynb first")
    return pd.read_csv(path)


def _interval_row(ax, y, est, lo, hi, colour, clears: bool, ms: float = 8, lw: float = 3):
    ax.plot([lo, hi], [y, y], color=colour, lw=lw, solid_capstyle="round")
    ax.plot(est, y, "o", color=colour, ms=ms, mfc=colour if clears else "white", mew=2)


# ---------------------------------------------------------------------------
# 1. Season contrasts forest (three panels: ratio, runs per over, share points)
# ---------------------------------------------------------------------------

def season_contrasts_forest(outputs: Path, earlier: int, later: int, title: str | None = None):
    t = _read(outputs, "mean_level_and_tail", "theme_contrasts").set_index("theme")
    st.apply_style()
    fig, axes = plt.subplots(3, 1, figsize=st.SIZES["half_tall"], gridspec_kw={"height_ratios": [1, 1.4, 1.4], "hspace": 0.95})
    ax = axes[0]; r = t.loc["powerplay_six_rate_ratio"]
    _interval_row(ax, 0, r["estimate"], r["ci_low"], r["ci_high"], st.SEASON["later"], bool(r["resampling_stable"]), ms=9)
    ax.axvline(1, color=st.GREY, ls="--", lw=1); ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks([1, 1.2, 1.4, 1.6, 1.8]); ax.set_xticklabels(["1.0", "1.2", "1.4", "1.6", "1.8"])
    ax.set_yticks([0]); ax.set_yticklabels(["Powerplay\nsix rate"]); ax.set_ylim(-0.7, 0.7); ax.set_xlim(0.95, 1.95); st.tidy(ax)
    ax.text(1.74, 0, f"{r['estimate']:.2f}×  [{r['ci_low']:.2f}, {r['ci_high']:.2f}]", va="center", fontsize=10, color=st.INK)
    ax.set_xlabel(f"ratio, {later} over {earlier} (log scale)")
    ax = axes[1]
    for i, theme in enumerate(("powerplay_run_rate_diff", "middle_run_rate_diff", "death_run_rate_diff")):
        r = t.loc[theme]; clears = bool(r["resampling_stable"]); c = st.SEASON["later"] if clears else st.GREY
        _interval_row(ax, 2 - i, r["estimate"], r["ci_low"], r["ci_high"], c, clears, ms=9 if clears else 7)
        ax.text(1.55, 2 - i, f"{r['estimate']:+.2f}  [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]", va="center", fontsize=10, color=st.INK if clears else st.GREY_TEXT)
    ax.axvline(0, color=st.GREY, ls="--", lw=1); ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["Powerplay", "Middle", "Death"]); ax.set_xlim(-0.4, 2.4); ax.set_ylim(-0.6, 2.6); st.tidy(ax)
    ax.set_xlabel(f"change in runs per over, {later} minus {earlier}")
    ax = axes[2]
    for i, theme in enumerate(("pace_share_diff_powerplay", "pace_share_diff_middle", "pace_share_diff_death")):
        r = t.loc[theme]; clears = bool(r["resampling_stable"]); c = st.SEASON["later"] if clears else st.GREY
        _interval_row(ax, 2 - i, 100 * r["estimate"], 100 * r["ci_low"], 100 * r["ci_high"], c, clears, ms=9 if clears else 7)
        ax.text(16.5, 2 - i, f"{100 * r['estimate']:+.1f} pts  [{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}]", va="center", fontsize=10, color=st.INK)
    ax.axvline(0, color=st.GREY, ls="--", lw=1); ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["Powerplay", "Middle", "Death"]); ax.set_xlim(-2.5, 26); ax.set_ylim(-0.6, 2.6); st.tidy(ax)
    ax.set_xlabel("change in pace share of legal balls, percentage points")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    keys = ("powerplay_six_rate_ratio", "powerplay_run_rate_diff", "middle_run_rate_diff", "death_run_rate_diff",
            "pace_share_diff_powerplay", "pace_share_diff_middle", "pace_share_diff_death")
    numbers = {k: {"estimate": float(t.loc[k, "estimate"]), "ci_low": float(t.loc[k, "ci_low"]), "ci_high": float(t.loc[k, "ci_high"])} for k in keys}
    return fig, numbers


# ---------------------------------------------------------------------------
# 2. Run-rate mechanism, one panel, sixes in the accent
# ---------------------------------------------------------------------------

def run_rate_mechanism(outputs: Path, earlier: int, later: int, title: str | None = None):
    t = _read(outputs, "run_rate_mechanics", "run_rate_decomposition")
    t = t[t["arrangement"] == "fours_sixes_dots_running_extras"]
    phases = ["powerplay", "middle", "death", "whole_innings"]
    comps = ["sixes", "fours", "dots", "running", "extras"]
    cols = {"sixes": st.SEASON["later"], "fours": st.GREYS[0], "dots": st.GREYS[1], "running": st.GREYS[2], "extras": st.GREYS[3]}
    fig, ax = st.new_figure("half")
    x = np.arange(len(phases)); pos = np.zeros(len(phases)); neg = np.zeros(len(phases))
    values = {c: np.array([float(t[(t["phase"] == p) & (t["component"] == c)]["contribution_rpo"].iloc[0]) for p in phases]) for c in comps}
    for c in comps:
        v = values[c]; b = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=b, color=cols[c], width=0.55, label=c, edgecolor="white", linewidth=0.6)
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    totals = [float(t[(t["phase"] == p) & (t["component"] == "run_rate_change")]["contribution_rpo"].iloc[0]) for p in phases]
    for i, tot in enumerate(totals):
        ax.plot([i - 0.33, i + 0.33], [tot, tot], color=st.INK, lw=2); ax.text(i + 0.37, tot, f"{tot:+.2f}", va="center", fontsize=10, color=st.INK)
        six, four = values["sixes"][i], values["fours"][i]
        ax.text(i, max(four, 0) + six / 2, f"sixes\n{six:+.2f}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    ax.axhline(0, color="#9aa0a6", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([p.replace("_", " ") for p in phases]); st.tidy(ax)
    ax.set_ylabel(f"change in runs per over, {later} minus {earlier}"); ax.set_ylim(min(neg.min(), 0) - 0.06, max(totals) * 1.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), handlelength=1.2)
    if title: ax.set_title(title)
    return fig, {"totals": dict(zip(phases, totals)), "sixes": dict(zip(phases, values["sixes"].tolist()))}


# ---------------------------------------------------------------------------
# 3. No wicket cost at three levels
# ---------------------------------------------------------------------------

def wicket_cost_three_levels(outputs: Path, earlier: int, later: int, sesoi_per_ball: float, balls_per_powerplay: int = 36, title: str | None = None):
    contrast = _read(outputs, "expected_wickets", "powerplay_contrast")
    refit = _read(outputs, "expected_wickets", "fixed_vs_refit_intervals")
    rung = _read(outputs, "expected_wickets", "shot_level_rung")
    carrying = contrast[(contrast["variant"] == "situation") & (contrast["role"] == "carrying")].set_index("quantity")
    actual = carrying.loc["actual_change"]
    xw_ref = refit[(refit["variant"] == "situation") & (refit["quantity"] == "xw_change")].iloc[0]
    inter = rung[(rung["variant"] == "situation") & (rung["coefficient"] == "aggressive_x_later_season")].iloc[0]
    bp = balls_per_powerplay
    rows = [("Actual wicket rate", bp * actual["point"], bp * actual["lo"], bp * actual["hi"], "wickets per powerplay"),
            ("Expected, given the\nsituations faced", bp * xw_ref["point"], bp * xw_ref["refit_lo"], bp * xw_ref["refit_hi"], "wickets per powerplay, model refit interval"),
            ("Cost per aggressive shot", 100 * inter["estimate"], 100 * inter["ci_low"], 100 * inter["ci_high"], "extra wickets per 100 aggressive shots")]
    fig, ax = st.new_figure("half")
    y = np.arange(3)[::-1]
    band = bp * sesoi_per_ball
    ax.axvspan(-band, band, color=st.LIGHT, zorder=0, ymin=0.33, ymax=0.63)
    xmax = max(r[3] for r in rows) * 1.05
    for yi, (lab, e, lo, hi, unit) in zip(y, rows):
        _interval_row(ax, yi, e, lo, hi, st.GREY, True, ms=8); ax.plot(e, yi, "o", color=st.INK, ms=8)
        ax.text(xmax * 1.05, yi, unit, va="center", fontsize=9, color=st.GREY_TEXT)
    ax.axvline(0, color=st.SEASON["later"], ls="--", lw=1.2)
    ax.text(band + 0.03, 1.42, "SESOI", fontsize=9, color=st.GREY_TEXT, va="center")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(min(r[2] for r in rows) * 1.15, xmax * 1.02); ax.set_ylim(-0.6, 2.75); st.tidy(ax)
    ax.set_xlabel(f"change, {later} minus {earlier} (zero is the null in every row)")
    if title: ax.set_title(title)
    return fig, {r[0].replace("\n", " "): {"estimate": float(r[1]), "ci_low": float(r[2]), "ci_high": float(r[3])} for r in rows}


# ---------------------------------------------------------------------------
# 4. Powerplay ledger at earlier-season prices
# ---------------------------------------------------------------------------

def powerplay_ledger(outputs: Path, earlier: int, later: int, title: str | None = None):
    t = _read(outputs, "win_probability_ledger", "powerplay_ledger")
    t = t[(t["scope"] == "both_innings") & (t["prices"] == f"prices_{earlier}")].set_index("outcome")
    order = [("six", "six"), ("four", "four"), ("no_run", "fewer dots"), ("one_to_three", "one to three"), ("wicket", "wicket"),
             ("wide", "wide"), ("no_ball", "no ball"), ("bye_or_leg_bye", "bye or leg bye"), ("net", "net")]
    order = [o for o in order if o[0] in t.index]
    fig, ax = st.new_figure("half")
    y = np.arange(len(order))[::-1]
    for yi, (key, lab) in zip(y, order):
        r = t.loc[key]; accent = key in ("six", "net"); c = st.SEASON["later"] if accent else st.GREY
        _interval_row(ax, yi, r["points_per_powerplay"], r["ci_low"], r["ci_high"], c, True, ms=9 if key == "net" else 7, lw=4 if key == "net" else 3)
        if key == "net": ax.plot(r["points_per_powerplay"], yi, "o", color=st.INK, ms=9)
        ax.text(t["ci_high"].max() * 1.1, yi, f"{r['points_per_powerplay']:+.1f}", va="center", fontsize=10, color=st.INK, fontweight="bold" if key == "net" else "normal")
    ax.axvline(0, color=st.GREY, ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels([o[1] for o in order]); st.tidy(ax)
    ax.set_xlim(t["ci_low"].min() * 1.2, t["ci_high"].max() * 1.3); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(f"win-probability points per 36-ball powerplay, {later} mix at {earlier} prices")
    if title: ax.set_title(title)
    net = t.loc["net"]
    return fig, {"net": float(net["points_per_powerplay"]), "net_low": float(net["ci_low"]), "net_high": float(net["ci_high"]), "six": float(t.loc["six", "points_per_powerplay"])}


# ---------------------------------------------------------------------------
# 5. Aggression premium dumbbell
# ---------------------------------------------------------------------------

def aggression_premium(outputs: Path, earlier: int, later: int, title: str | None = None):
    t = _read(outputs, "win_probability_ledger", "powerplay_aggression")
    t = t[t["scope"] == "both_innings"].set_index("season")
    fig, ax = st.new_figure("small")
    for yi, season in ((1, earlier), (0, later)):
        r = t.loc[season]; c = st.season_colour(season, earlier)
        _interval_row(ax, yi, r["premium"], r["premium_low"], r["premium_high"], c, True, ms=9)
        ax.text(t["premium_high"].max() * 1.08, yi, f"{r['premium']:+.2f}  [{r['premium_low']:.2f}, {r['premium_high']:.2f}]", va="center", fontsize=10, color=st.INK)
    ax.set_yticks([1, 0]); ax.set_yticklabels([str(earlier), str(later)]); ax.set_xlim(0, t["premium_high"].max() * 1.6); ax.set_ylim(-0.7, 1.7); st.tidy(ax)
    ax.set_xlabel("win-probability premium of an aggressive-shot ball over other balls, points per ball")
    if title: ax.set_title(title)
    return fig, {str(s): float(t.loc[s, "premium"]) for s in (earlier, later)}


# ---------------------------------------------------------------------------
# 6. Team value slope chart with tiers
# ---------------------------------------------------------------------------

def team_value_slope(outputs: Path, earlier: int, later: int, tier_cuts=(15.0, 9.0), title: str | None = None):
    t = _read(outputs, "powerplay_risk_ledger", "per_team_season_ledger")
    wide = t.pivot(index="team", columns="season", values="value_points")
    fig, ax = st.new_figure("half_tall")
    top, mid = tier_cuts
    ymax = float(wide.to_numpy().max()) * 1.15
    ax.axhspan(top, ymax, color=st.ACCENT_PALE, zorder=0); ax.axhspan(mid, top, color=st.PALE, zorder=0)
    order = wide[later].sort_values(ascending=False)
    label_y = {}; last = None
    for team, v in order.items():
        y = v if last is None else min(v, last - ymax * 0.045)
        label_y[team] = y; last = y
    for team, row in wide.iterrows():
        a, b = row[earlier], row[later]
        c = st.SEASON["later"] if b >= top else (st.GREY_DARK if b >= mid else st.GREY)
        ax.plot([0, 1], [a, b], color=c, lw=2 if b >= top else 1.2); ax.plot([0, 1], [a, b], "o", color=c, ms=5)
        ax.text(1.05, label_y[team], TEAM_SHORT.get(team, team), va="center", fontsize=10, color=c, fontweight="bold" if b >= top else "normal")
    ax.set_xticks([0, 1]); ax.set_xticklabels([str(earlier), str(later)]); ax.set_xlim(-0.08, 1.5); ax.set_ylim(0, ymax); st.tidy(ax)
    ax.set_ylabel("value of the powerplay outcome mix, points per powerplay")
    if title: ax.set_title(title)
    return fig, {TEAM_SHORT.get(team, team): {str(earlier): float(r[earlier]), str(later): float(r[later])} for team, r in wide.iterrows()}


# ---------------------------------------------------------------------------
# 7. Line-by-length grid, two season panels
# ---------------------------------------------------------------------------

def line_length_grid(outputs: Path, earlier: int, later: int, rate: str = "six_rate", thin_floor: int = 300, title: str | None = None):
    menu = _read(outputs, "bowling_response", "delivery_menu")
    sub = menu[(menu["dimension"] == "line_length") & (menu["kind"] == "all") & (menu["class"] != "untagged")].copy()
    sub[["length3", "line3"]] = sub["class"].str.split("|", expand=True)
    lengths, lines = ["full", "good", "short"], ["off", "stumps", "leg"]
    vmax = max(float(sub[[f"{rate}_{earlier}", f"{rate}_{later}"]].max().max()), 1e-6)
    ramps = {earlier: mcolors.LinearSegmentedColormap.from_list("g", [st.PALE, st.SEASON["earlier"]]),
             later: mcolors.LinearSegmentedColormap.from_list("m", [st.PALE, st.SEASON["later"]])}
    st.apply_style()
    fig, ax = plt.subplots(figsize=st.SIZES["wide"])
    max_balls = float(sub[[f"balls_{earlier}", f"balls_{later}"]].max().max())
    for col, season in enumerate((earlier, later)):
        x0 = col * 3.6
        ax.text(x0 + 1.5, 3.2, str(season), ha="center", fontsize=13, fontweight="bold", color=st.season_colour(season, earlier))
        for _, r in sub.iterrows():
            x, y = lines.index(r["line3"]), lengths.index(r["length3"])
            balls = float(r[f"balls_{season}"]); v = float(r[f"{rate}_{season}"])
            size = 0.45 + 0.5 * np.sqrt(balls / max_balls); thin = balls < thin_floor
            ax.add_patch(Rectangle((x0 + x + 0.5 - size / 2, y + 0.5 - size / 2), size, size, color=ramps[season](v / vmax), hatch="///" if thin else None, ec="white", lw=1))
            ax.text(x0 + x + 0.5, y + 0.5, f"{100 * v:.1f}", ha="center", va="center", fontsize=10 if not thin else 8.5, color=st.INK if v < 0.62 * vmax else "white", fontweight="bold")
        for x, li in enumerate(lines):
            ax.text(x0 + x + 0.5, -0.18, li, ha="center", va="top", fontsize=10)
    for y, ln in enumerate(lengths):
        ax.text(-0.15, y + 0.5, ln, ha="right", va="center", fontsize=10)
    ax.text(-0.15, 3.2, "sixes per\n100 balls" if rate == "six_rate" else rate, fontsize=9, color=st.GREY_TEXT, ha="right", va="center")
    ax.set_xlim(-0.9, 7.0); ax.set_ylim(-0.55, 3.55); ax.axis("off")
    if title: ax.set_title(title)
    comps = _read(outputs, "bowling_response", "shift_share_components")
    row = comps[(comps["dimension"] == "line_length") & (comps["kind"] == "all") & (comps["rate"] == rate)].set_index("component")
    numbers = {"within_share": float(row.loc["within", "share_of_change"]), "mix_share": float(row.loc["mix", "share_of_change"]),
               "within": float(row.loc["within", "estimate"]), "mix": float(row.loc["mix", "estimate"])}
    return fig, numbers


# ---------------------------------------------------------------------------
# 8. Speed bands dumbbell
# ---------------------------------------------------------------------------

def speed_bands(outputs: Path, earlier: int, later: int, title: str | None = None):
    menu = _read(outputs, "bowling_response", "delivery_menu")
    sub = menu[(menu["dimension"] == "speed3") & (menu["kind"] == "pace") & (menu["class"] != "untagged")].set_index("class")
    labels = {"over_140": "over 140 kph", "130_to_140": "130 to 140", "under_130": "under 130"}
    order = [c for c in ("over_140", "130_to_140", "under_130") if c in sub.index]
    slower = menu[(menu["dimension"] == "type") & (menu["kind"] == "all") & (menu["class"] == "slower_ball_deception")]
    fig, ax = st.new_figure("small")
    y = np.arange(len(order))[::-1]
    xmax = 100 * float(sub[[f"six_rate_{earlier}", f"six_rate_{later}"]].max().max())
    for yi, c in zip(y, order):
        r = sub.loc[c]; a, b = 100 * r[f"six_rate_{earlier}"], 100 * r[f"six_rate_{later}"]
        lo, hi = 100 * r["six_rate_change_low"], 100 * r["six_rate_change_high"]
        clears = lo > 0 or hi < 0
        ax.plot([a, b], [yi, yi], color=st.GREY, lw=3); ax.plot(a, yi, "o", color=st.SEASON["earlier"], ms=9); ax.plot(b, yi, "o", color=st.SEASON["later"], ms=9)
        note = f"{b - a:+.1f}  [{lo:+.1f}, {hi:+.1f}]" if clears else f"{b - a:+.1f}, held  [{lo:+.1f}, {hi:+.1f}]"
        ax.text(xmax * 1.08, yi, note, va="center", fontsize=9.5, color=st.INK)
    ax.set_yticks(y); ax.set_yticklabels([labels[c] for c in order]); ax.set_xlim(0, xmax * 1.9); ax.set_ylim(-0.6, len(order) - 0.4); st.tidy(ax)
    ax.set_xlabel("sixes per 100 balls, pace bowling in the powerplay")
    ax.plot([], [], "o", color=st.SEASON["earlier"], label=str(earlier)); ax.plot([], [], "o", color=st.SEASON["later"], label=str(later)); ax.legend(loc="upper right")
    if title: ax.set_title(title)
    numbers = {labels[c]: {str(earlier): 100 * float(sub.loc[c, f"six_rate_{earlier}"]), str(later): 100 * float(sub.loc[c, f"six_rate_{later}"])} for c in order}
    if len(slower):
        numbers["slower_ball_share"] = {str(earlier): float(slower[f"share_{earlier}"].iloc[0]), str(later): float(slower[f"share_{later}"].iloc[0])}
    return fig, numbers


# ---------------------------------------------------------------------------
# 9. Spin decomposition, de-inked
# ---------------------------------------------------------------------------

def spin_decomposition(outputs: Path, title: str | None = None):
    t = _read(outputs, "spin_displacement", "spin_share_decomposition")
    phases = ["all", "powerplay", "middle", "death"]; labels = ["whole innings", "powerplay", "middle", "death"]
    factors = [("spinners_named_per_innings", "#2f5c8a", "selection: fewer spinners named"), ("share_of_named_spinners_used", "#8fb3d9", "deployment: named, not bowled"),
               ("spin_balls_per_spinner_used", st.GREYS[0], "workload per spinner"), ("legal_balls_per_innings", st.GREYS[3], "innings length")]
    fig, ax = st.new_figure("half")
    x = np.arange(len(phases)); pos = np.zeros(len(phases)); neg = np.zeros(len(phases))
    for key, c, lab in factors:
        v = np.array([float(t[(t["phase"] == p) & (t["factor"] == key)]["contribution_points"].iloc[0]) for p in phases])
        b = np.where(v >= 0, pos, neg); ax.bar(x, v, bottom=b, color=c, width=0.55, label=lab, edgecolor="white", linewidth=0.6)
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    totals = [float(t[(t["phase"] == p) & (t["factor"] == "spin_share_change")]["contribution_points"].iloc[0]) for p in phases]
    for i, tot in enumerate(totals):
        ax.plot([i - 0.33, i + 0.33], [tot, tot], color=st.INK, lw=2); ax.text(i + 0.37, tot, f"{tot:+.1f}", va="center", fontsize=10)
    ax.axhline(0, color="#9aa0a6", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels); st.tidy(ax)
    ax.set_ylabel("change in spin share of legal balls, points"); ax.set_ylim(neg.min() * 1.15, max(pos.max(), 0) + 2)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    if title: ax.set_title(title)
    return fig, {"totals": dict(zip(labels, totals))}


# ---------------------------------------------------------------------------
# 10. Batter decomposition with the entrants split by auction route
# ---------------------------------------------------------------------------

def batter_decomposition(outputs: Path, earlier: int, later: int, title: str | None = None):
    comp = _read(outputs, "batter_continuity", "powerplay_six_rate_components").set_index("quantity")
    routes = _read(outputs, "batter_continuity", "composition_by_auction_route")
    rows = [("league change", "league_change", st.INK), ("same batters, changed rate", "within", st.SEASON["later"]), ("different batters", "composition", st.SEASON["earlier"]),
            ("   continuing, reweighted", "composition_continuing_reweight", st.GREY), ("   entrants", "composition_entrants", st.SEASON["earlier"]), ("   exits", "composition_exits", st.GREY)]
    fig, ax = st.new_figure("half_tall")
    y = np.arange(len(rows))[::-1] + 1.7
    plotted = comp.loc[[key for _, key, _ in rows]]
    xmax = 100 * float(plotted["ci_high"].max()); xmin = 100 * float(plotted["ci_low"].min())
    #xmax = 100 * float(comp["ci_high"].max()); xmin = 100 * float(comp["ci_low"].min())
    for yi, (lab, key, c) in zip(y, rows):
        r = comp.loc[key]; clears = r["ci_low"] > 0 or r["ci_high"] < 0
        _interval_row(ax, yi, 100 * r["estimate"], 100 * r["ci_low"], 100 * r["ci_high"], c, clears, ms=8)
        ax.text(xmax * 1.08, yi, f"{100 * r['estimate']:+.2f}", va="center", fontsize=10, color=st.INK)
    entrants = routes[routes["group"] == "entrant"].sort_values("composition_contribution", ascending=False)
    route_colour = {"bought": st.SEASON["later"], "not_in_auction_table": "#e98bb0", "promoted_from_squad": st.GREY, "traded_in": "#f2b8cf"}
    route_label = {"bought": "bought at auction", "not_in_auction_table": "replacement signings", "promoted_from_squad": "promoted from squad", "traded_in": "traded in"}
    left = 0.0
    for _, r in entrants.iterrows():
        v = 100 * r["composition_contribution"]; c = route_colour.get(r["route"], st.GREY)
        ax.barh(0.35, v, left=left, height=0.45, color=c, edgecolor="white")
        left += v
    ax.text(xmin, 0.35, "entrants\nby route", fontsize=9.5, color=st.GREY_DARK, va="center")
    # A legend column to the right of the bar, each entry carrying its value, so nothing
    # sits inside a thin segment.
    x_leg = left + 0.12 * (xmax - xmin)
    for k, (_, r) in enumerate(entrants.iterrows()):
        v = 100 * r["composition_contribution"]; c = route_colour.get(r["route"], st.GREY)
        y_leg = 0.35 + 0.36 * (len(entrants) - 1) / 2 - 0.36 * k
        ax.plot(x_leg, y_leg, "s", color=c, ms=8, clip_on=False)
        ax.text(x_leg + 0.04 * (xmax - xmin), y_leg, f"{route_label.get(r['route'], r['route'])} {v:+.2f}", va="center", fontsize=9, color=st.GREY_DARK)
    ax.axvline(0, color=st.GREY, ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(xmin * 1.15, xmax * 1.25); ax.set_ylim(-0.3, len(rows) + 1.5); st.tidy(ax)
    ax.set_xlabel("change in powerplay six rate, sixes per 100 balls faced")
    if title: ax.set_title(title)
    numbers = {key: float(100 * comp.loc[key, "estimate"]) for _, key, _ in rows}
    numbers["entrants_by_route"] = {r["route"]: float(100 * r["composition_contribution"]) for _, r in entrants.iterrows()}
    return fig, numbers


# ---------------------------------------------------------------------------
# 11. Chase target response, powerplay raw and adjusted, later phases adjusted
# ---------------------------------------------------------------------------

def chase_target(outputs: Path, earlier: int, later: int, title: str | None = None):
    pp = _read(outputs, "impact_substitute_mechanics", "chase_target_response")
    later_phases = _read(outputs, "impact_substitute_mechanics", "chase_target_response_middle_death")
    pp = pp[pp["metric"] == "pp_run_rate"].set_index("term")
    rows = [("Powerplay, raw", pp.loc["raw_season_difference"]), ("Powerplay, at a common target", pp.loc["season_difference_at_180"])]
    for phase, label in (("middle", "Middle overs, at a common target"), ("death", "Death, at a common target")):
        r = later_phases[(later_phases["phase"] == phase) & (later_phases["metric"] == "run_rate") & (later_phases["term"] == "season_difference_at_180")]
        if len(r):
            rows.append((label, r.iloc[0]))
    fig, ax = st.new_figure("half")
    y = np.arange(len(rows))[::-1]
    xmax = max(float(r["ci_high"]) for _, r in rows)
    for yi, (lab, r) in zip(y, rows):
        clears = r["ci_low"] > 0 or r["ci_high"] < 0; c = st.SEASON["later"] if clears else st.GREY
        _interval_row(ax, yi, r["estimate"], r["ci_low"], r["ci_high"], c, clears, ms=8)
        ax.text(xmax * 1.08, yi, f"{r['estimate']:+.2f}", va="center", fontsize=10, color=st.INK)
    ax.axvline(0, color=st.GREY, ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(min(float(r["ci_low"]) for _, r in rows) * 1.2, xmax * 1.3); ax.set_ylim(-0.6, len(rows) - 0.4); st.tidy(ax)
    ax.set_xlabel(f"chasing run rate, {later} minus {earlier}, runs per over")
    if title: ax.set_title(title)
    return fig, {lab: {"estimate": float(r["estimate"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"])} for lab, r in rows}


# ---------------------------------------------------------------------------
# 12. Substitute numbers (text only, for a slide element)
# ---------------------------------------------------------------------------

def substitute_numbers(outputs: Path, earlier: int, later: int) -> dict:
    side = _read(outputs, "impact_substitute_mechanics", "deployments_by_side_innings")
    contrib = _read(outputs, "impact_substitute_mechanics", "substitute_contribution_summary")
    first = {(int(r["season"]), r["side"]): int(r["1"]) for _, r in side.iterrows()}
    out = {"bowling_first_first_innings": {str(earlier): first.get((earlier, "bowled_first"), 0), str(later): first.get((later, "bowled_first"), 0)},
           "batting_first_first_innings": {str(earlier): first.get((earlier, "batted_first"), 0), str(later): first.get((later, "batted_first"), 0)}}
    for _, r in contrib.iterrows():
        out.setdefault("substitute_contribution", {})[str(int(r["season"]))] = {
            "swaps": int(r["swaps"]), "sub_batted_share": float(r["sub_batted_share"]),
            "sub_strike_rate": float(r["sub_strike_rate_pooled"]), "others_strike_rate": float(r["others_strike_rate_pooled"])}
    return out
