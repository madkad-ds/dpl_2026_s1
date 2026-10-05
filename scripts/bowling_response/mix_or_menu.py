"""
Mix or menu: did the powerplay surge come from what was bowled, or from everything
being hit harder?

A league rate against deliveries (sixes per legal ball, runs per legal ball) is a
share-weighted average over delivery classes. Its season change splits exactly into

    within  sum_c wbar[c] * (r[c, later] - r[c, earlier])   every class got hit harder
    mix     sum_c (w[c, later] - w[c, earlier]) * rbar[c]   bowlers bowled a different mix

with w the class's share of the phase's legal deliveries and r its rate, bars the
two-season averages. A class absent from a season carries the other season's rate,
so its contribution is all mix. The identity holds on every match resample.

Rates per legal delivery in the phase: six rate, runs per ball (score), dot rate
(no runs off the bat), wicket rate. The menu table gives each class's share, its
rates in both seasons, the season change in each rate with a bootstrap-t interval,
and a thin flag below a balls floor. Pace and spin are split where the dimension
is not pace-only, because a good length from a spinner is a different delivery.

Selection caveat: the class a bowler chooses depends on the batter and the state,
so a class can look cheap because of when it was used. The menu is descriptive; the
decomposition is the inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import derived_columns as dc
from mean_level_and_tail import season_contrast_bootstrap as scb
from .delivery_classes import DIMENSIONS, assign_classes

RATES = {
    "six_rate": "sixes per legal ball",
    "runs_per_ball": "runs (including extras) per legal ball",
    "dot_rate": "dots off the bat per legal ball",
    "wicket_rate": "wickets per legal ball",
}


def _ball_columns(frame: pd.DataFrame) -> pd.DataFrame:
    legal = dc.is_legal_delivery(frame).to_numpy()
    batruns = pd.to_numeric(frame["batruns"], errors="coerce").fillna(0).to_numpy(dtype=float)
    return pd.DataFrame({
        "legal": legal.astype(float),
        "six": (dc.is_six(frame).to_numpy() & legal).astype(float),
        "runs": np.where(legal, pd.to_numeric(frame["score"], errors="coerce").fillna(0).to_numpy(dtype=float), 0.0),
        "dot": (legal & (batruns == 0)).astype(float),
        "wicket": (dc.is_wicket(frame).to_numpy() & legal).astype(float),
    })


class ClassSeasonSums:
    """Per season: matches x classes matrices of legal balls, sixes, runs, dots and
    wickets for one dimension, classes aligned across seasons."""

    def __init__(self, frame: pd.DataFrame, dimension: str, scope: np.ndarray, earlier: int, later: int):
        self.dimension = dimension
        self.earlier, self.later = earlier, later
        cols = _ball_columns(frame)
        cols = cols.mul(scope.astype(float), axis=0)
        cls = frame[dimension].astype(str).to_numpy()
        order = DIMENSIONS[dimension]["order"]
        present = sorted(set(cls[scope & (cols["legal"].to_numpy() > 0)]))
        self.classes = ([c for c in order if c in present] + [c for c in present if order is None or c not in order]) if order else present
        c_ix = {c: j for j, c in enumerate(self.classes)}
        self.mats = {}
        for season in (earlier, later):
            rows = frame["year"].to_numpy() == season
            matches = np.sort(frame.loc[rows, "p_match"].unique())
            m_ix = {m: i for i, m in enumerate(matches)}
            mat = np.zeros((len(matches), len(self.classes), 5))
            sub = cols[rows & scope].assign(p_match=frame.loc[rows & scope, "p_match"].to_numpy(), cls=cls[rows & scope])
            for (m, c), g in sub.groupby(["p_match", "cls"]):
                if c in c_ix:
                    mat[m_ix[m], c_ix[c], :] += g[["legal", "six", "runs", "dot", "wicket"]].sum().to_numpy()
            self.mats[season] = (matches, mat)

    def n_matches(self):
        return len(self.mats[self.earlier][0]), len(self.mats[self.later][0])


def _rates(tot: np.ndarray) -> dict:
    """tot: (n_iter, classes, 5). Returns share and each rate per class, plus league rates."""
    legal = tot[:, :, 0]
    total = legal.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"share": legal / total,
               "six_rate": np.where(legal > 0, tot[:, :, 1] / legal, np.nan),
               "runs_per_ball": np.where(legal > 0, tot[:, :, 2] / legal, np.nan),
               "dot_rate": np.where(legal > 0, tot[:, :, 3] / legal, np.nan),
               "wicket_rate": np.where(legal > 0, tot[:, :, 4] / legal, np.nan)}
    for k, j in (("six_rate", 1), ("runs_per_ball", 2), ("dot_rate", 3), ("wicket_rate", 4)):
        out[f"league_{k}"] = tot[:, :, j].sum(axis=1) / total[:, 0]
    return out


def shift_share(sums: ClassSeasonSums, counts_e: np.ndarray, counts_l: np.ndarray, rate: str) -> dict:
    """Within and mix components for one rate across all resample rows, plus the
    per-class within and mix contributions."""
    e = _rates(np.einsum("im,mck->ick", counts_e, sums.mats[sums.earlier][1]))
    l = _rates(np.einsum("im,mck->ick", counts_l, sums.mats[sums.later][1]))
    pe, pl = e[rate], l[rate]
    present_e, present_l = np.isfinite(pe), np.isfinite(pl)
    pe_f, pl_f = np.where(present_e, pe, pl), np.where(present_l, pl, pe)
    w_bar, r_bar = 0.5 * (e["share"] + l["share"]), 0.5 * (pe_f + pl_f)
    within_c = np.where(present_e & present_l, w_bar * (pl_f - pe_f), 0.0)
    mix_c = np.where(present_e | present_l, (l["share"] - e["share"]) * r_bar, 0.0)
    return {"league_earlier": e[f"league_{rate}"], "league_later": l[f"league_{rate}"],
            "change": l[f"league_{rate}"] - e[f"league_{rate}"],
            "within": within_c.sum(axis=1), "mix": mix_c.sum(axis=1),
            "within_by_class": within_c, "mix_by_class": mix_c,
            "share_e": e["share"], "share_l": l["share"], "rate_e": pe, "rate_l": pl,
            "all_e": e, "all_l": l}


def _band(a, alpha):
    """Percentile interval; blank when nothing finite (a class bowled in one season only)."""
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    return float(np.percentile(a, 100 * alpha / 2)), float(np.percentile(a, 100 * (1 - alpha / 2)))


def run_dimension(frame: pd.DataFrame, settings, seed: int, dimension: str, phase_name: str,
                  kind: str, n_iter: int, counts: tuple | None = None, thin_floor: int = 300) -> dict:
    """Shift-share for every rate plus the menu table, for one dimension, phase and
    bowler kind ("all", "pace", "spin"). Returns tables and the shared counts."""
    earlier, later = settings.earlier_season, settings.later_season
    alpha = 1.0 - settings.interval_level
    work = assign_classes(frame)
    scope = scb.in_phase(work, settings.phase(phase_name)).to_numpy() if hasattr(scb.in_phase(work, settings.phase(phase_name)), "to_numpy") else np.asarray(scb.in_phase(work, settings.phase(phase_name)))
    if kind == "pace" or DIMENSIONS[dimension]["pace_only"]:
        scope = scope & dc.is_pace(work).to_numpy()
    elif kind == "spin":
        scope = scope & dc.is_spin(work).to_numpy()
    sums = ClassSeasonSums(work, dimension, scope, earlier, later)
    n_e, n_l = sums.n_matches()
    counts = counts or scb.shared_resample_counts(n_e, n_l, n_iter, seed)
    counts_e, counts_l = counts
    ones = (np.ones((1, n_e)), np.ones((1, n_l)))

    component_rows, menu_rows = [], []
    for rate in RATES:
        point = shift_share(sums, *ones, rate)
        reps = shift_share(sums, counts_e, counts_l, rate)
        for comp in ("change", "within", "mix"):
            lo, hi = _band(reps[comp], alpha)
            component_rows.append({"dimension": dimension, "phase": phase_name, "kind": kind, "rate": rate,
                                   "component": comp, "estimate": float(point[comp][0]), "ci_low": lo, "ci_high": hi,
                                   "p_positive": float(np.nanmean(reps[comp] > 0)),
                                   "share_of_change": float(point[comp][0] / point["change"][0]) if point["change"][0] else np.nan,
                                   "identity_max_abs_error": float(np.nanmax(np.abs(reps["change"] - reps["within"] - reps["mix"])))})
        if rate == "six_rate":
            base_point, base_reps = point, reps
    for j, c in enumerate(sums.classes):
        row = {"dimension": dimension, "phase": phase_name, "kind": kind, "class": c,
               f"balls_{earlier}": int(sums.mats[earlier][1][:, j, 0].sum()), f"balls_{later}": int(sums.mats[later][1][:, j, 0].sum()),
               f"share_{earlier}": float(base_point["share_e"][0, j]), f"share_{later}": float(base_point["share_l"][0, j]),
               "share_change_points": 100 * float(base_point["share_l"][0, j] - base_point["share_e"][0, j])}
        lo, hi = _band(100 * (base_reps["share_l"][:, j] - base_reps["share_e"][:, j]), alpha)
        row["share_change_low"], row["share_change_high"] = lo, hi
        for rate in RATES:
            pe, pl = base_point["all_e"][rate][0, j], base_point["all_l"][rate][0, j]
            re_, rl = base_reps["all_e"][rate][:, j], base_reps["all_l"][rate][:, j]
            lo, hi = _band(rl - re_, alpha)
            row.update({f"{rate}_{earlier}": float(pe), f"{rate}_{later}": float(pl), f"{rate}_change": float(pl - pe),
                        f"{rate}_change_low": lo, f"{rate}_change_high": hi})
        row["six_within_contribution"] = float(base_point["within_by_class"][0, j])
        row["six_mix_contribution"] = float(base_point["mix_by_class"][0, j])
        row["thin"] = bool(min(row[f"balls_{earlier}"], row[f"balls_{later}"]) < thin_floor)
        menu_rows.append(row)
    return {"components": pd.DataFrame(component_rows), "menu": pd.DataFrame(menu_rows), "counts": counts,
            "classes": sums.classes}


def run_all_dimensions(frame: pd.DataFrame, settings, seed: int, phase_name: str = "powerplay",
                       n_iter: int | None = None, thin_floor: int = 300) -> dict:
    """Every dimension, for all bowlers and split by pace and spin where the
    dimension is not pace-only, on one shared set of draws."""
    n_iter = n_iter or settings.n_resamples
    components, menus, counts = [], [], None
    for dimension, spec in DIMENSIONS.items():
        kinds = ["pace"] if spec["pace_only"] else ["all", "pace", "spin"]
        for kind in kinds:
            out = run_dimension(frame, settings, seed, dimension, phase_name, kind, n_iter, counts, thin_floor)
            counts = out["counts"]
            components.append(out["components"])
            menus.append(out["menu"])
    return {"components": pd.concat(components, ignore_index=True), "menu": pd.concat(menus, ignore_index=True), "counts": counts}


def plot_line_length_grid(menu: pd.DataFrame, earlier: int, later: int, kind: str = "all",
                          rate: str = "six_rate", title: str = ""):
    """Line-by-length grid, one panel per season: cell area is the class's share of
    deliveries, cell colour its rate; a third panel colours the change."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sub = menu[(menu["dimension"] == "line_length") & (menu["kind"] == kind) & (menu["class"] != "untagged")].copy()
    sub[["length3", "line3"]] = sub["class"].str.split("|", expand=True)
    lengths, lines = ["full", "good", "short"], ["off", "stumps", "leg"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    panels = [(f"{earlier}", f"share_{earlier}", f"{rate}_{earlier}", "Blues", None),
              (f"{later}", f"share_{later}", f"{rate}_{later}", "Blues", None),
              (f"change {later} minus {earlier}", f"share_{later}", f"{rate}_change", "RdBu_r", "diverging")]
    vmax = max(float(np.nanmax(sub[[f"{rate}_{earlier}", f"{rate}_{later}"]].to_numpy())), 1e-9)
    cmax = max(float(np.nanmax(np.abs(sub[f"{rate}_change"]))), 1e-9)
    for ax, (label, share_col, val_col, cmap, mode) in zip(axes, panels):
        cm = plt.get_cmap(cmap)
        for _, r in sub.iterrows():
            x, y = lines.index(r["line3"]), lengths.index(r["length3"])
            size = 0.35 + 0.6 * np.sqrt(r[share_col] / sub[share_col].max())
            v = r[val_col]
            colour = cm(0.5 + 0.5 * v / cmax) if mode == "diverging" else cm(v / vmax)
            ax.add_patch(Rectangle((x + 0.5 - size / 2, y + 0.5 - size / 2), size, size, color=colour,
                                   hatch="//" if r["thin"] else None, ec="white"))
            ax.text(x + 0.5, y + 0.5, f"{100 * v:+.1f}" if mode == "diverging" else f"{100 * v:.1f}",
                    ha="center", va="center", fontsize=8, color="#1F2937")
        ax.set_xlim(0, 3); ax.set_ylim(0, 3)
        ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(lines)
        ax.set_yticks([0.5, 1.5, 2.5]); ax.set_yticklabels(lengths)
        ax.set_title(label, fontsize=10, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(title or f"{rate.replace('_', ' ')} per 100 balls by line and length (cell size = share of deliveries; hatched = thin)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_components(components: pd.DataFrame, phase_name: str, rate: str, title: str):
    """Within and mix components per dimension and kind, with intervals."""
    import matplotlib.pyplot as plt

    sub = components[(components["phase"] == phase_name) & (components["rate"] == rate) & (components["component"] != "change")]
    sub = sub.assign(label=sub["dimension"] + " (" + sub["kind"] + ")")
    labels = list(dict.fromkeys(sub["label"]))
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 1.5))
    for comp, colour, offset in (("within", "#2F5C8A", 0.15), ("mix", "#d95f0e", -0.15)):
        s = sub[sub["component"] == comp].set_index("label").loc[labels]
        ax.hlines(y + offset, 100 * s["ci_low"], 100 * s["ci_high"], color=colour, lw=2)
        ax.plot(100 * s["estimate"], y + offset, "o", color=colour, ms=5, label=comp)
    ax.axvline(0, color="#C0392B", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"contribution to the change in {rate.replace('_', ' ')}, per 100 balls")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
