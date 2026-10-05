"""
Same batters, or different batters? Decomposing a league rate change at batter level.

A league rate in a phase (the powerplay six rate: sixes per ball faced) is a
ball-weighted average of the batters who faced those balls. It can rise because
the same batters changed how they bat, or because the balls were faced by
different batters. The two are separated exactly:

  R_s = sum_i w[i, s] * r[i, s]        w = share of the phase's faced balls, r = batter's rate

  R_later - R_earlier
    = sum_i wbar[i] * (r[i, later] - r[i, earlier])        WITHIN: continuing batters changed
    + sum_i (w[i, later] - w[i, earlier]) * rbar[i]        COMPOSITION: who faced the balls changed

with wbar and rbar the two-season averages. A batter absent from one season has
w = 0 there and, by convention, the same rate in both (so the within term is zero
and the whole contribution is composition). The composition term is further split
into entrants (present in the later season only), exits (earlier only), and
reweighting among continuing batters. The identity holds exactly on every resample.

Everything is computed from per-(match, batter) sufficient statistics, so the match
cluster bootstrap reweights matches and the decomposition is redone on each draw.
"Continuing" for the decomposition means facing at least one ball in both seasons
(the identity needs no threshold); the per-batter display and the continuing-only
league contrast use a minimum-balls floor from config so a batter's own rate means
something.

Rates supported: six rate and boundary rate (per ball faced), run rate (score per
legal ball, times six). Phase from the fixed grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import derived_columns as dc
from mean_level_and_tail import season_contrast_bootstrap as scb

RATES = {
    "six_rate": dict(label="sixes per ball faced", per_over=False),
    "boundary_rate": dict(label="fours and sixes per ball faced", per_over=False),
    "run_rate": dict(label="runs per over, legal balls", per_over=True),
}


def _numerator_denominator(frame: pd.DataFrame, rate: str) -> tuple[np.ndarray, np.ndarray]:
    faced = dc.is_ball_faced(frame).to_numpy()
    legal = dc.is_legal_delivery(frame).to_numpy()
    if rate == "six_rate":
        return (dc.is_six(frame).to_numpy() & faced).astype(float), faced.astype(float)
    if rate == "boundary_rate":
        return (dc.is_boundary(frame).to_numpy() & faced).astype(float), faced.astype(float)
    if rate == "run_rate":
        score = pd.to_numeric(frame["score"], errors="coerce").fillna(0).to_numpy(dtype=float)
        return np.where(legal, score, 0.0), legal.astype(float)
    raise ValueError(rate)


class BatterSeasonSums:
    """Per season: matches x batters matrices of numerator and denominator sums in
    the phase, with batter ids aligned across seasons (a batter absent from a season
    has an all-zero column there)."""

    def __init__(self, frame: pd.DataFrame, rate: str, phase, earlier: int, later: int,
                 player_column: str = "p_bat"):
        # player_column="p_bat" decomposes over batters; "p_bowl" over bowlers, in which
        # case the rate is what the bowler conceded per ball bowled.
        self.rate = rate
        self.player_column = player_column
        self.earlier, self.later = earlier, later
        numer, denom = _numerator_denominator(frame, rate)
        in_phase = scb.in_phase(frame, phase)
        work = pd.DataFrame({"year": frame["year"].to_numpy(), "p_match": frame["p_match"].to_numpy(),
                             "p_bat": frame[player_column].to_numpy(),
                             "numer": numer * in_phase, "denom": denom * in_phase})
        self.batters = np.sort(work.loc[work["denom"] > 0, "p_bat"].unique())
        b_ix = {b: j for j, b in enumerate(self.batters)}
        self.matches, self.numer, self.denom = {}, {}, {}
        for season in (earlier, later):
            season_rows = frame["year"].to_numpy() == season
            matches = np.sort(frame.loc[season_rows, "p_match"].unique())
            m_ix = {m: i for i, m in enumerate(matches)}
            # Only rows with a phase ball: a batter whose season had none (or only
            # wides in the phase) has no column in the batter matrix.
            g = work[season_rows & (work["denom"] > 0).to_numpy()].groupby(["p_match", "p_bat"])[["numer", "denom"]].sum()
            n_mat = np.zeros((len(matches), len(self.batters)))
            d_mat = np.zeros((len(matches), len(self.batters)))
            for (m, b), row in g.iterrows():
                n_mat[m_ix[m], b_ix[b]] += row["numer"]
                d_mat[m_ix[m], b_ix[b]] += row["denom"]
            self.matches[season], self.numer[season], self.denom[season] = matches, n_mat, d_mat

    def n_matches(self) -> tuple[int, int]:
        return len(self.matches[self.earlier]), len(self.matches[self.later])

    def batter_names(self, frame: pd.DataFrame) -> dict:
        name_column = "bat" if self.player_column == "p_bat" else "bowl"
        names = frame.drop_duplicates(self.player_column).set_index(self.player_column)[name_column]
        return {b: str(names.get(b, b)) for b in self.batters}


def decompose(sums: BatterSeasonSums, counts_earlier: np.ndarray, counts_later: np.ndarray) -> dict:
    """Decomposition for every resample row of the multiplicity matrices. Returns
    arrays of shape (n_iter,) per component, plus per-batter arrays for the
    within-batter change (n_iter x batters) used for the per-batter intervals."""
    scale = 6.0 if RATES[sums.rate]["per_over"] else 1.0
    n_e = counts_earlier @ sums.numer[sums.earlier]
    d_e = counts_earlier @ sums.denom[sums.earlier]
    n_l = counts_later @ sums.numer[sums.later]
    d_l = counts_later @ sums.denom[sums.later]
    tot_e, tot_l = d_e.sum(axis=1, keepdims=True), d_l.sum(axis=1, keepdims=True)
    w_e, w_l = d_e / tot_e, d_l / tot_l
    with np.errstate(invalid="ignore", divide="ignore"):
        r_e = np.where(d_e > 0, n_e / d_e, np.nan) * scale
        r_l = np.where(d_l > 0, n_l / d_l, np.nan) * scale
    present_e, present_l = d_e > 0, d_l > 0
    continuing = present_e & present_l
    # A batter absent from a season carries the other season's rate there, so the
    # within term is zero and the whole contribution lands in composition.
    r_e_filled = np.where(present_e, r_e, r_l)
    r_l_filled = np.where(present_l, r_l, r_e)
    w_bar = 0.5 * (w_e + w_l)
    r_bar = 0.5 * (r_e_filled + r_l_filled)
    present_either = present_e | present_l
    # A batter absent from both seasons of a resample (a thin batter whose only
    # match was not drawn) has no rate on either side; the contribution is zero, not
    # 0 x NaN, or one such batter would poison the whole resample's sum.
    within_by_batter = np.where(continuing, w_bar * (r_l_filled - r_e_filled), 0.0)
    composition_by_batter = np.where(present_either, (w_l - w_e) * r_bar, 0.0)
    entrants = present_l & ~present_e
    exits = present_e & ~present_l
    league_e = (n_e.sum(axis=1) / tot_e[:, 0]) * scale
    league_l = (n_l.sum(axis=1) / tot_l[:, 0]) * scale
    return {
        "league_earlier": league_e, "league_later": league_l, "league_change": league_l - league_e,
        "within": within_by_batter.sum(axis=1),
        "composition": composition_by_batter.sum(axis=1),
        "composition_continuing_reweight": np.where(continuing, composition_by_batter, 0.0).sum(axis=1),
        "composition_entrants": np.where(entrants, composition_by_batter, 0.0).sum(axis=1),
        "composition_exits": np.where(exits, composition_by_batter, 0.0).sum(axis=1),
        "share_later_faced_by_entrants": np.where(entrants, w_l, 0.0).sum(axis=1),
        "share_earlier_faced_by_exits": np.where(exits, w_e, 0.0).sum(axis=1),
        "rate_entrants": np.where(entrants, n_l, 0.0).sum(axis=1) / np.maximum(np.where(entrants, d_l, 0.0).sum(axis=1), 1) * scale,
        "rate_exits": np.where(exits, n_e, 0.0).sum(axis=1) / np.maximum(np.where(exits, d_e, 0.0).sum(axis=1), 1) * scale,
        "rate_continuing_earlier": np.where(continuing, n_e, 0.0).sum(axis=1) / np.maximum(np.where(continuing, d_e, 0.0).sum(axis=1), 1) * scale,
        "rate_continuing_later": np.where(continuing, n_l, 0.0).sum(axis=1) / np.maximum(np.where(continuing, d_l, 0.0).sum(axis=1), 1) * scale,
        "_within_by_batter": within_by_batter, "_r_e": r_e, "_r_l": r_l, "_d_e": d_e, "_d_l": d_l,
    }


def _summarise(point: dict, reps: dict, alpha: float) -> pd.DataFrame:
    rows = []
    for name in ("league_earlier", "league_later", "league_change", "within", "composition",
                 "composition_continuing_reweight", "composition_entrants", "composition_exits",
                 "share_later_faced_by_entrants", "share_earlier_faced_by_exits",
                 "rate_entrants", "rate_exits", "rate_continuing_earlier", "rate_continuing_later"):
        a = reps[name]
        a = a[np.isfinite(a)]
        rows.append({"quantity": name, "estimate": float(point[name][0]),
                     "ci_low": float(np.percentile(a, 100 * alpha / 2)),
                     "ci_high": float(np.percentile(a, 100 * (1 - alpha / 2))),
                     "p_positive": float((a > 0).mean()), "interval_method": "percentile"})
    table = pd.DataFrame(rows)
    identity = reps["league_change"] - reps["within"] - reps["composition"]
    table.attrs["identity_max_abs_error"] = float(np.nanmax(np.abs(identity)))
    return table


def run_decomposition(frame: pd.DataFrame, settings, seed: int, rate: str = "six_rate",
                      phase_name: str = "powerplay", n_iter: int | None = None,
                      counts: tuple | None = None, player_column: str = "p_bat") -> dict:
    """Point decomposition, bootstrap intervals, and the per-player table for one
    rate in one phase. `counts` lets several rates share the same draws;
    `player_column` chooses batters (p_bat) or bowlers (p_bowl)."""
    earlier, later = settings.earlier_season, settings.later_season
    n_iter = n_iter or settings.n_resamples
    alpha = 1.0 - settings.interval_level
    min_balls = int(settings.section("batter_continuity")["min_balls_per_season"])
    sums = BatterSeasonSums(frame, rate, settings.phase(phase_name), earlier, later, player_column)
    n_e, n_l = sums.n_matches()
    counts = counts or scb.shared_resample_counts(n_e, n_l, n_iter, seed)
    point = decompose(sums, np.ones((1, n_e)), np.ones((1, n_l)))
    reps = decompose(sums, *counts)
    table = _summarise(point, reps, alpha)

    scale = 6.0 if RATES[rate]["per_over"] else 1.0
    names = sums.batter_names(frame)
    within_reps = reps["_within_by_batter"]
    rows = []
    for j, b in enumerate(sums.batters):
        d_e, d_l = point["_d_e"][0, j], point["_d_l"][0, j]
        status = "continuing" if d_e > 0 and d_l > 0 else ("entrant" if d_l > 0 else "exit")
        change = point["_r_l"][0, j] - point["_r_e"][0, j] if status == "continuing" else np.nan
        # Per-batter change interval: the batter's own rate change across resamples,
        # taken where the batter is present in both seasons of the resample.
        rr = reps["_r_l"][:, j] - reps["_r_e"][:, j]
        rr = rr[np.isfinite(rr)]
        rows.append({"p_bat": b, "bat": names[b], "status": status,
                     f"balls_{earlier}": int(d_e), f"balls_{later}": int(d_l),
                     f"rate_{earlier}": point["_r_e"][0, j], f"rate_{later}": point["_r_l"][0, j],
                     "rate_change": change,
                     "change_low": float(np.percentile(rr, 100 * alpha / 2)) if rr.size and status == "continuing" else np.nan,
                     "change_high": float(np.percentile(rr, 100 * (1 - alpha / 2))) if rr.size and status == "continuing" else np.nan,
                     "within_contribution": float(point["_within_by_batter"][0, j]),
                     "meets_floor_both_seasons": bool(d_e >= min_balls and d_l >= min_balls)})
    batters = pd.DataFrame(rows).sort_values("within_contribution", key=np.abs, ascending=False).reset_index(drop=True)

    return {"rate": rate, "phase": phase_name, "unit": RATES[rate]["label"], "components": table,
            "batters": batters, "n_matches": (n_e, n_l), "counts": counts, "sums": sums,
            "n_batters": {"continuing": int((batters["status"] == "continuing").sum()),
                          "entrants": int((batters["status"] == "entrant").sum()),
                          "exits": int((batters["status"] == "exit").sum()),
                          "continuing_meeting_floor": int(batters["meets_floor_both_seasons"].sum())}}


def continuing_batter_contrast(frame: pd.DataFrame, settings, seed: int, rate: str, phase_name: str,
                               counts: tuple, min_balls: int, player_column: str = "p_bat") -> pd.DataFrame:
    """Harness contrast of the league rate computed on continuing batters only (those
    at or above the balls floor in both seasons) beside the all-batter contrast, on
    the same draws. If the two agree, the league change is a change in how the same
    batters bat."""
    earlier, later = settings.earlier_season, settings.later_season
    numer, denom = _numerator_denominator(frame, rate)
    in_phase = scb.in_phase(frame, settings.phase(phase_name))
    balls = (pd.Series(denom * in_phase).groupby([frame[player_column].to_numpy(), frame["year"].to_numpy()]).sum()
             .unstack(fill_value=0.0))
    continuing = set(balls.index[(balls.get(earlier, 0) >= min_balls) & (balls.get(later, 0) >= min_balls)])
    is_continuing = frame[player_column].isin(continuing).to_numpy()
    per_over = RATES[rate]["per_over"]
    themes = []
    for label, mask in (("all_batters", in_phase), ("continuing_batters_only", in_phase & is_continuing)):
        e = frame["year"].to_numpy() == earlier
        l = frame["year"].to_numpy() == later
        est = scb.RateDifference(scb.per_match_sums(frame[e], mask[e], numer[e], denom[e]),
                                 scb.per_match_sums(frame[l], mask[l], numer[l], denom[l]), per_over=per_over)
        themes.append(scb.Theme(f"{phase_name}_{rate}_diff_{label}", est, 0.0, "well_powered", ""))
        if rate != "run_rate":
            ratio = scb.RateRatio(scb.per_match_sums(frame[e], mask[e], numer[e], denom[e]),
                                  scb.per_match_sums(frame[l], mask[l], numer[l], denom[l]))
            themes.append(scb.Theme(f"{phase_name}_{rate}_ratio_{label}", ratio, 1.0, "well_powered", ""))
    n_e, n_l = counts[0].shape[1], counts[1].shape[1]
    rows = [scb.run_theme(t, n_e, n_l, counts[0], counts[1], alpha=1.0 - settings.interval_level).as_row() for t in themes]
    out = pd.DataFrame(rows)
    out["n_continuing_batters"] = len(continuing)
    return out


def plot_batter_scatter(batters: pd.DataFrame, earlier: int, later: int, unit: str, title: str,
                        label_top: int = 8):
    """Continuing batters at the floor: earlier rate against later rate, point size by
    balls faced, the diagonal as no change. Entrants and exits shown on the axes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    cont = batters[(batters["status"] == "continuing") & batters["meets_floor_both_seasons"]]
    size = (cont[f"balls_{earlier}"] + cont[f"balls_{later}"]) / 2
    ax.scatter(cont[f"rate_{earlier}"], cont[f"rate_{later}"], s=size / size.max() * 260 + 10,
               color="#2F5C8A", alpha=0.6, edgecolor="white", linewidth=0.6, label="continuing (at floor)")
    lim = max(float(cont[[f"rate_{earlier}", f"rate_{later}"]].max().max()) * 1.1, 1e-6)
    ax.plot([0, lim], [0, lim], "--", color="#C0392B", lw=1, label="no change")
    top = cont.reindex(cont["within_contribution"].abs().sort_values(ascending=False).index).head(label_top)
    for _, r in top.iterrows():
        ax.annotate(r["bat"], (r[f"rate_{earlier}"], r[f"rate_{later}"]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points", color="#1F2937")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"{earlier} ({unit})"); ax.set_ylabel(f"{later} ({unit})")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_components(components: pd.DataFrame, unit: str, title: str):
    """League change beside its within and composition parts, with intervals."""
    import matplotlib.pyplot as plt

    order = ["league_change", "within", "composition", "composition_continuing_reweight",
             "composition_entrants", "composition_exits"]
    labels = {"league_change": "league change", "within": "same batters, changed rate",
              "composition": "different batters (all)", "composition_continuing_reweight": "  continuing batters reweighted",
              "composition_entrants": "  entrants", "composition_exits": "  exits"}
    rows = components.set_index("quantity").loc[order]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colours = ["#1F2937", "#2F5C8A", "#d95f0e", "#d95f0e", "#d95f0e", "#d95f0e"]
    ax.hlines(y, rows["ci_low"], rows["ci_high"], color=colours, lw=2)
    ax.plot(rows["estimate"], y, "o", color="#1F2937", ms=6)
    ax.axvline(0, color="#C0392B", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([labels[q] for q in order], fontsize=9)
    ax.set_xlabel(f"change, later minus earlier ({unit})")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
