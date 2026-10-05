"""
Where in the innings the scoring moved, and what was bowled: per-phase batting
rates (powerplay overs 1 to 6, middle 7 to 15, death 16 to 20) with the
season-over-season change, and the bowling picture split into mix (pace versus
spin share of legal balls) and effectiveness (economy from `bowler_charge`,
boundary share, wicket rate). Full innings only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .scoring_picture import SEASON_COLOURS

PHASE_KEYS = ["powerplay", "middle", "death"]
PHASE_LABELS = ["Powerplay\n(ov 1\u20136)", "Middle\n(ov 7\u201315)", "Death\n(ov 16\u201320)"]
RISE = "#C44E52"      # muted red for increases (with the league's 2024 direction)
FALL = "#4C72B0"      # muted blue for decreases
GRID = "#E6E6E6"
PACE_COLOUR = "#C44E52"
SPIN_COLOUR = "#4C72B0"
# Death-overs spin in 2024 rests on ~560 balls with a selection effect (teams stopped
# bowling it), so its economy delta is flagged as low-confidence in the figure.
LOW_SAMPLE = {("death", "spin")}


def phase_structure(df, exclude_reduced=True):
    """Per-phase batting rates by season, so the macro scoring shift can be located.
    Phases: powerplay (overs 1-6, field restricted), middle (7-15), death (16-20).
    For each phase and season computes run rate, boundary %, dot %, and wicket rate on a
    per-ball basis (pooling all deliveries in the phase), which is well-powered since each
    phase-season pools thousands of balls. Restricts to full innings so phase lengths are
    comparable across seasons.
    """
    d = df.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["over_num"] = pd.to_numeric(d["over_num"], errors="coerce")
    for c in ["batruns","score","max_balls"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if exclude_reduced:
        d = d[d["max_balls"] == 120]

    d["is_legal"] = (pd.to_numeric(d["noball"],errors="coerce").fillna(0)==0) & \
                    (pd.to_numeric(d["wide"],errors="coerce").fillna(0)==0)
    d["is_boundary"] = d["batruns"].isin([4,6])
    d["is_six"] = d["batruns"] == 6
    d["is_dot"] = d["is_legal"] & (d["score"] == 0)
    d["is_wicket"] = d["out"].astype(str).str.strip().str.lower() == "true"

    # Phase from over number (over_num is 1-based here).
    d["phase"] = pd.cut(d["over_num"], bins=[0,6,15,20],
                        labels=["powerplay","middle","death"])

    # Per-ball rates, pooled within phase x season. Rates use LEGAL balls as the
    # denominator (run rate per 6 legal balls; boundary/dot as share of legal balls).
    def phase_rates(g):
        legal = g["is_legal"].sum()
        return pd.Series({
            "legal_balls": legal,
            "run_rate": g["score"].sum() / (legal/6) if legal else np.nan,
            "boundary_pct": g["is_boundary"].sum() / legal if legal else np.nan,
            "six_pct": g["is_six"].sum() / legal if legal else np.nan,
            "dot_pct": g["is_dot"].sum() / legal if legal else np.nan,
            "wicket_rate": g["is_wicket"].sum() / legal if legal else np.nan,
        })

    rates = d.groupby(["phase","year"], observed=True).apply(phase_rates).round(3)
    print("=== Per-phase batting rates by season ===\n")
    print(rates.to_string())

    # Season-over-season DELTA per phase, to see which phase moved most.
    print("\n=== Change (2024 - 2023) by phase ===\n")
    for metric in ["run_rate","boundary_pct","six_pct","dot_pct","wicket_rate"]:
        pivot = rates[metric].unstack("year")
        pivot["delta"] = (pivot[2024] - pivot[2023]).round(3)
        print(f"{metric}:")
        print(pivot.to_string())
        print()
    return rates, d


def plot_phase_structure(phase_rates):
    """Two-panel phase view. Left: 2023 vs 2024 bars per phase for run rate, boundary %
    and six %, so the 'barbell' shape (ends move, middle lags) is visible. Right: the
    season-over-season delta per phase for the same metrics, which is the actual finding —
    a positive delta at both ends and a smaller one in the middle reads as the barbell.
    """
    metrics = ["run_rate", "boundary_pct", "six_pct"]
    # phase_rates has a (phase, year) index; pull each metric into a phase x year frame.
    def metric_frame(m):
        return (phase_rates[m].unstack("year").reindex(PHASE_KEYS))

    fig, (ax_level, ax_delta) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: levels, seasons side by side, one metric group per phase ---
    # Boundary/six % are fractions (~0-0.25); run rate is ~8-11. Plot run rate on a
    # secondary axis so both are legible in one panel.
    x = np.arange(len(PHASE_KEYS))
    n_metrics = len(metrics)
    width = 0.8 / (n_metrics * 2)   # two seasons per metric

    rr = metric_frame("run_rate")
    ax_rr = ax_level.twinx()        # run rate on its own scale

    pos = 0
    handles_labels = {}
    for mi, m in enumerate(metrics):
        frame = metric_frame(m)
        target_ax = ax_rr if m == "run_rate" else ax_level
        for si, season in enumerate((2023, 2024)):
            offset = (pos - (n_metrics * 2 - 1) / 2) * width
            bars = target_ax.bar(x + offset, frame[season].values, width,
                                 color=SEASON_COLOURS[season],
                                 alpha=0.55 if m != "six_pct" else 0.9,
                                 hatch="" if m == "run_rate" else ("//" if m == "boundary_pct" else ".."),
                                 edgecolor="white", linewidth=0.5)
            handles_labels[f"{m} {season}"] = bars
            pos += 1

    ax_level.set_xticks(x)
    ax_level.set_xticklabels(PHASE_KEYS)
    ax_level.set_ylabel("share of legal balls (boundary %, six %)")
    ax_rr.set_ylabel("run rate (runs / over)")
    ax_level.set_title("Phase batting rates: 2023 vs 2024\n(hatch: none=run rate, //=boundary%, ..=six%)")
    ax_level.legend(handles_labels.values(), handles_labels.keys(), fontsize=7, ncol=3, loc="upper center")

    # --- Right panel: deltas (2024 - 2023) per phase, one bar per metric ---
    # This is the finding: tall bars at powerplay & death, short in the middle = barbell.
    delta_width = 0.25
    for mi, m in enumerate(metrics):
        frame = metric_frame(m)
        delta = (frame[2024] - frame[2023]).reindex(PHASE_KEYS)
        # Scale run_rate delta down to sit alongside the fraction deltas visually, and label it.
        scaled = delta / 10 if m == "run_rate" else delta
        label = "run rate (÷10)" if m == "run_rate" else m
        ax_delta.bar(x + (mi - 1) * delta_width, scaled.values, delta_width, label=label)
    ax_delta.axhline(0, color="black", linewidth=0.8)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(PHASE_KEYS)
    ax_delta.set_ylabel("change 2024 − 2023")
    ax_delta.set_title("Season-over-season change by phase\n(taller at both ends = barbell)")
    ax_delta.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_phase_deltas_report(phase_rates):
    """Report figure: season-over-season change (2024 − 2023) in run rate, boundary %,
    six % and wicket rate, one panel per metric, phases on the x-axis. Small multiples on
    independent scales avoid the dual-axis / rescaling compromises of an exploratory combo
    chart. The intended read is the 'barbell': acceleration concentrated at the innings
    ends (powerplay, death) with the middle overs lagging.
    """
    metrics = [
        ("run_rate",     "Run rate",      "runs/over",          "{:+.2f}"),
        ("boundary_pct", "Boundary rate", "share of balls",     "{:+.3f}"),
        ("six_pct",      "Six rate",      "share of balls",     "{:+.3f}"),
        ("wicket_rate",  "Wicket rate",   "share of balls",     "{:+.3f}"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.2))
    x = np.arange(len(PHASE_KEYS))

    for ax, (key, title, unit, fmt) in zip(axes, metrics):
        frame = phase_rates[key].unstack("year").reindex(PHASE_KEYS)
        delta = (frame[2024] - frame[2023]).values
        colours = [RISE if v >= 0 else FALL for v in delta]

        bars = ax.bar(x, delta, width=0.62, color=colours, edgecolor="white", linewidth=1)
        ax.axhline(0, color="#333333", linewidth=1)

        # Direct value labels — reader gets the number without reading the axis.
        for xi, v in zip(x, delta):
            ax.annotate(fmt.format(v), (xi, v),
                        textcoords="offset points",
                        xytext=(0, 4 if v >= 0 else -12),
                        ha="center", fontsize=9, color="#222222")

        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylabel(f"change 2024 − 2023\n({unit})", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(PHASE_KEYS, fontsize=8)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=0)
        # Symmetric-ish headroom so labels don't clip.
        pad = max(abs(delta)) * 0.35 if max(abs(delta)) else 0.01
        ax.set_ylim(min(0, delta.min()) - pad, max(0, delta.max()) + pad)

    fig.suptitle("Where the innings accelerated: 2024 vs 2023",
                 fontsize=14, fontweight="bold", y=1.02)
    # fig.text(0.5, -0.06,
    #          "Scoring rose fastest at the innings' ends. The powerplay and death overs each added "
    #          "~0.75 runs/over, the middle overs about half that. The powerplay six rate rose most in "
    #          "relative terms (+39%) at almost no added wicket cost — a shift toward aerial aggression "
    #          "up top. Wicket rates barely moved in any phase.",
    #          ha="center", fontsize=9, color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def bowling_texture(df, exclude_reduced=True):
    """Season-vs-season bowling: what was bowled (pace/spin mix, by phase) and how it
    fared (economy, boundary %, wicket rate). Separates MIX (deployment share) from
    EFFECTIVENESS (rates conceded), so a change reads as either 'bowled differently' or
    'same bowling, different outcome'. Economy uses bowler_charge (laws-correct runs
    charged to the bowler: bat runs + wides + no-ball penalty, excluding byes/leg-byes).
    """
    d = df.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["over_num"] = pd.to_numeric(d["over_num"], errors="coerce")
    for c in ["batruns","bowler_charge","max_balls"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if exclude_reduced:
        d = d[d["max_balls"] == 120]

    d["is_legal"] = (pd.to_numeric(d["noball"],errors="coerce").fillna(0)==0) & \
                    (pd.to_numeric(d["wide"],errors="coerce").fillna(0)==0)
    d["is_boundary"] = d["batruns"].isin([4,6])
    d["is_wicket"] = d["out"].astype(str).str.strip().str.lower()=="true"
    d["phase"] = pd.cut(d["over_num"], bins=[0,6,15,20],
                        labels=["powerplay","middle","death"])

    # --- MIX: share of legal balls bowled by pace vs spin, per phase per season ---
    kind = d[d["is_legal"]].copy()
    kind["bowl_kind"] = kind["bowl_kind"].astype(str).str.strip().str.lower()
    kind["kind3"] = kind["bowl_kind"].map(
        lambda k: "pace" if "pace" in k else "spin" if "spin" in k else "other")

    # Count balls per (phase, year, kind), then divide by the (phase, year) total.
    # Building counts first and dividing by a transform avoids the grouped-apply
    # reset_index collision (where 'year' ends up both an index level and a column).
    counts = (kind.groupby(["phase", "year", "kind3"], observed=True)
              .size().rename("balls").reset_index())
    counts["phase_year_total"] = counts.groupby(["phase", "year"], observed=True)["balls"].transform("sum")
    counts["share"] = (counts["balls"] / counts["phase_year_total"]).round(3)
    mix = counts   # the per-(phase, year, kind) share table this function returns

    print("=== Bowling mix: pace vs spin share of legal balls, by phase ===\n")
    print(mix.pivot_table(index=["phase", "kind3"], columns="year", values="share").to_string())

    # --- EFFECTIVENESS: economy, boundary %, wicket rate by pace/spin x phase ---
    def rates(g):
        legal = g["is_legal"].sum()
        return pd.Series({
            "legal_balls": legal,
            "economy": g["bowler_charge"].sum() / (legal/6) if legal else np.nan,
            "boundary_pct": g["is_boundary"].sum() / legal if legal else np.nan,
            "wicket_rate": g["is_wicket"].sum() / legal if legal else np.nan,
        })
    d["kind3"] = d["bowl_kind"].astype(str).str.strip().str.lower().map(
        lambda k: "pace" if "pace" in k else "spin" if "spin" in k else "other")
    eff = (d[d["kind3"].isin(["pace","spin"])]
           .groupby(["phase","kind3","year"], observed=True).apply(rates).round(3))
    print("\n=== Bowling effectiveness by pace/spin and phase ===\n")
    print(eff.to_string())

    # --- Season delta in economy, the headline effectiveness signal ---
    print("\n=== Economy change (2024 - 2023) by phase and pace/spin ===\n")
    econ = eff["economy"].unstack("year")
    econ["delta"] = (econ[2024] - econ[2023]).round(3)
    print(econ.to_string())

    return mix, eff


def plot_bowling_texture(mix, eff):
    """Two panels telling one story. Left: pace vs spin share of legal balls per phase,
    2023 vs 2024 — teams shifted toward pace everywhere. Right: economy change (2024−2023)
    by pace/spin per phase — spin got more expensive than pace in every phase, extremely
    so at the death. Read left-to-right: teams bowled less spin BECAUSE spin leaked runs.
    """
    fig, (ax_mix, ax_econ) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(len(PHASE_KEYS))
    w = 0.38

    # ---- LEFT: pace/spin share, seasons as adjacent stacked bars per phase ----
    def share(phase, kind, year):
        row = mix[(mix["phase"] == phase) & (mix["kind3"] == kind) & (mix["year"] == year)]
        return float(row["share"].iloc[0]) if len(row) else 0.0

    for si, (year, off) in enumerate([(2023, -w/2), (2024, w/2)]):
        pace = [share(p, "pace", year) for p in PHASE_KEYS]
        spin = [share(p, "spin", year) for p in PHASE_KEYS]
        ax_mix.bar(x + off, pace, w, color=PACE_COLOUR, edgecolor="white",
                   label="pace" if si == 0 else None)
        ax_mix.bar(x + off, spin, w, bottom=pace, color=SPIN_COLOUR, edgecolor="white",
                   label="spin" if si == 0 else None)
        # year tick under each bar
        for xi in x:
            ax_mix.annotate(str(year), (xi + off, -0.04), ha="center", va="top",
                            fontsize=7, color="#666666")
        # pace-share label on the pace segment
        for xi, pv in zip(x, pace):
            ax_mix.annotate(f"{pv:.0%}", (xi + off, pv/2), ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")

    ax_mix.set_xticks(x); ax_mix.set_xticklabels(PHASE_LABELS, fontsize=9)
    ax_mix.set_ylim(0, 1.02); ax_mix.set_ylabel("share of legal balls")
    ax_mix.set_title("Bowling mix shifted toward pace in every phase", fontsize=11, pad=10)
    ax_mix.legend(loc="lower left", fontsize=9, framealpha=0.9)
    for s in ("top","right"): ax_mix.spines[s].set_visible(False)
    ax_mix.tick_params(length=0)

    # ---- RIGHT: economy delta by pace/spin per phase ----
    econ = eff["economy"].unstack("year")
    econ["delta"] = econ[2024] - econ[2023]

    def econ_delta(phase, kind):
        try:
            return float(econ.loc[(phase, kind), "delta"])
        except KeyError:
            return np.nan

    pace_d = [econ_delta(p, "pace") for p in PHASE_KEYS]
    spin_d = [econ_delta(p, "spin") for p in PHASE_KEYS]

    b_pace = ax_econ.bar(x - w/2, pace_d, w, color=PACE_COLOUR, edgecolor="white", label="pace")
    b_spin = ax_econ.bar(x + w/2, spin_d, w, color=SPIN_COLOUR, edgecolor="white", label="spin")
    ax_econ.axhline(0, color="#333333", linewidth=1)

    for xi, (pv, sv, phase) in enumerate(zip(pace_d, spin_d, PHASE_KEYS)):
        ax_econ.annotate(f"+{pv:.2f}", (xi - w/2, pv), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=8)
        spin_label = f"+{sv:.2f}"
        # mark the low-sample death-spin bar
        if (phase, "spin") in LOW_SAMPLE:
            spin_label += " *"
        ax_econ.annotate(spin_label, (xi + w/2, sv), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=8,
                         fontweight="bold" if (phase,"spin") in LOW_SAMPLE else "normal")

    ax_econ.set_xticks(x); ax_econ.set_xticklabels(PHASE_LABELS, fontsize=9)
    ax_econ.set_ylabel("economy change 2024 − 2023 (runs/over)")
    ax_econ.set_title("Spin got more expensive than pace in every phase", fontsize=11, pad=10)
    ax_econ.legend(fontsize=9)
    ax_econ.yaxis.grid(True, color=GRID, linewidth=0.8); ax_econ.set_axisbelow(True)
    for s in ("top","right","left"): ax_econ.spines[s].set_visible(False)
    ax_econ.tick_params(length=0)

    fig.suptitle("Bowling texture, 2023 → 2024: teams turned to pace as spin leaked runs",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.08,
             "Pace displaced spin in every phase (left), the largest swing in the middle overs. "
             "Spin economy rose more than pace in every phase (right), dramatically at the death "
             "(+2.24 rpo), where spin lost its traditional economy advantage. * Death-overs spin in "
             "2024 rests on ~560 balls with a selection effect, so its change is directional, not precise.",
             ha="center", fontsize=9, color="#444444", wrap=True)
    fig.tight_layout()
    return fig
