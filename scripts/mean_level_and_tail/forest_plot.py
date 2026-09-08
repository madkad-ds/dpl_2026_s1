"""
Forest plot of the season-over-season effects that survive the bootstrap.

Reads the harness results table. The stability flag does the filtering, so a
finding that stops clearing its null on a rerun drops out of the figure on its own
rather than lingering because it was named here. Unit and tier are analyst
judgements the results table does not carry: unit picks the panel and axis, tier
sets emphasis so a confounded result never reads as equal to a clean one.

The innings tail is not in this figure: the 220+ share lives on the first-innings
population with a 2x2 interval (see innings_tail.py), and can be appended as a
`points` row by the caller if a combined figure is wanted.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

SPEC = {
    "powerplay_six_rate_ratio":         ("ratio",  "well_powered", "no added wicket cost"),
    "powerplay_run_rate_diff":          ("rpo",    "well_powered", ""),
    "pace_share_diff_powerplay":        ("points", "well_powered", ""),
    "pace_share_diff_middle":           ("points", "well_powered", ""),
    "pace_share_diff_death":            ("points", "well_powered", ""),
    "shot_controlled_minus_slog_shift": ("points", "caveated", "tagging floor, true interval wider"),
    "death_spin_economy_diff":          ("rpo",    "caveated", "selection confounded, not causal"),
    "innings_220_plus_risk_difference": ("points", "caveated", "first innings, Newcombe interval"),
}

LABEL = {
    "powerplay_six_rate_ratio": "Powerplay six-rate",
    "powerplay_run_rate_diff": "Powerplay run rate",
    "pace_share_diff_powerplay": "Pace share, powerplay",
    "pace_share_diff_middle": "Pace share, middle",
    "pace_share_diff_death": "Pace share, death",
    "shot_controlled_minus_slog_shift": "Controlled-power vs slog",
    "death_spin_economy_diff": "Death spin economy",
    "innings_220_plus_risk_difference": "220+ share, first innings",
}

# Each unit has its own null and axis. Share differences are stored as fractions
# in the table, so points are scaled to percentage points at draw time.
PANELS = {
    "ratio":  dict(null=1.0, log=True,  xlabel="ratio  (2024 / 2023),  log scale", scale=1.0, tick="x"),
    "rpo":    dict(null=0.0, log=False, xlabel="change in runs per over  (2024 minus 2023)", scale=1.0, tick="rpo"),
    "points": dict(null=0.0, log=False, xlabel="change in share  (percentage points)", scale=100.0, tick="pts"),
}
PANEL_ORDER = ["ratio", "rpo", "points"]

STYLE = {
    "well_powered": dict(color="#2F5C8A", mfc="#2F5C8A", alpha=1.0,  lw=1.9, ms=7),
    "caveated":     dict(color="#9AA0A6", mfc="white",   alpha=0.95, lw=1.5, ms=7),
}


def rows_from_results(results: pd.DataFrame) -> list[dict]:
    """Positive, resampling-stable findings pulled straight from the table."""
    picked = []
    for name, (unit, tier, caveat) in SPEC.items():
        row = results.loc[results["theme"] == name]
        if row.empty or not bool(row["resampling_stable"].iloc[0]):
            continue
        r = row.iloc[0]
        picked.append(dict(
            label=LABEL[name], estimate=float(r["estimate"]),
            ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]),
            unit=unit, tier=tier, caveat=caveat,
            method=str(r.get("interval_method", "")),
        ))
    return picked


def draw_forest(findings: list[dict], title: str, n_resamples: int, interval_level: float):
    by_panel = {u: [f for f in findings if f["unit"] == u] for u in PANEL_ORDER}
    by_panel = {u: fs for u, fs in by_panel.items() if fs}
    order = [u for u in PANEL_ORDER if u in by_panel]
    for u in order:                                    # clean results sit above caveated ones
        by_panel[u].sort(key=lambda f: f["tier"] != "well_powered")

    heights = [len(by_panel[u]) for u in order]
    fig, axes = plt.subplots(
        len(order), 1, figsize=(9.2, 0.62 * sum(heights) + 1.4),
        gridspec_kw={"height_ratios": heights}, squeeze=False,
    )
    axes = axes[:, 0]

    for ax, unit in zip(axes, order):
        spec = PANELS[unit]
        panel_rows = by_panel[unit]
        y = np.arange(len(panel_rows))[::-1]           # first finding at the top

        for yi, f in zip(y, panel_rows):
            s = STYLE[f["tier"]]
            est, lo, hi = (f[k] * spec["scale"] for k in ("estimate", "ci_low", "ci_high"))
            ax.plot([lo, hi], [yi, yi], color=s["color"], lw=s["lw"],
                    alpha=s["alpha"], solid_capstyle="round")
            ax.plot(est, yi, "o", color=s["color"], mfc=s["mfc"], ms=s["ms"],
                    mew=1.6, alpha=s["alpha"], zorder=3)
            unit_txt = spec["tick"]
            value = f"{est:.2f} [{lo:.2f}, {hi:.2f}] {unit_txt}"
            if f["caveat"]:
                value += f"   ·  {f['caveat']}"
            ax.text(1.02, yi, value, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8.5,
                    color="#3A3A3A" if f["tier"] == "well_powered" else "#8A8A8A")

        ax.axvline(spec["null"], color="#C0392B", ls="--", lw=1.1, alpha=0.7, zorder=1)
        if spec["log"]:
            ax.set_xscale("log")
            # Plain ratio labels (1.2, 1.5, 2) instead of scientific log ticks.
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.xaxis.get_major_formatter().set_scientific(False)
            ax.xaxis.get_minor_formatter().set_scientific(False)
        ax.set_yticks(y)
        ax.set_yticklabels([f["label"] for f in panel_rows], fontsize=9.5)
        ax.set_xlabel(spec["xlabel"], fontsize=9)
        ax.set_ylim(-0.6, len(panel_rows) - 0.4)
        ax.margins(x=0.08)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=0)

    axes[0].set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)
    fig.text(0.01, -0.02 / len(order),
             f"Match-cluster bootstrap, {n_resamples:,} resamples, {int(interval_level * 100)}% intervals: "
             "bootstrap-t for rates and shares, log-scale BCa for ratios, BCa elsewhere. "
             "Red dashed line is the null. Resampling stability, not out-of-sample replication. "
             "Grey findings clear the null but carry the noted caveat.",
             fontsize=7.6, color="#6A6A6A", style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    return fig, axes


def risk_difference_row(contrast: dict) -> dict:
    """Turn the 220+ 2x2 headline into a results-table row so the forest plot can
    show it beside the bootstrap findings (stored as a fraction, like the shares)."""
    return {"theme": "innings_220_plus_risk_difference",
            "estimate": contrast["risk_difference"],
            "ci_low": contrast["rd_low"], "ci_high": contrast["rd_high"],
            "interval_method": "newcombe_score", "null": 0.0,
            "resampling_stable": bool(contrast["rd_low"] > 0 or contrast["rd_high"] < 0)}
