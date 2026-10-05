"""
What the 2024 powerplay approach was worth in win-probability points.

The provided win probability (`wprob`, the batting side's chance of winning in
percent) is stated after each ball, so the ball-to-ball change is a price for what
happened on that ball: a six is worth a few points, a wicket costs a few. Because a
calibrated probability is a martingale, the average change over all balls is close
to zero in every season, so "did win probability rise more in 2024" is not the
question. The question is an accounting one: value the 2024 outcome mix at fixed
prices and compare with the 2023 mix.

Three ledgers, all on powerplay deliveries, all under the match cluster bootstrap on
shared draws:

  1. Outcome-mix ledger. Price each outcome category (six, four, one to three runs,
     no run, wicket, wide, no ball, bye or leg bye) as the mean change in wprob on
     balls with that outcome; take the change in each category's share of powerplay
     deliveries between seasons; the value of the approach per 36-ball powerplay is
     36 x sum over categories of (share change x price). Prices are held at their
     2023 level (primary), and at the pooled and 2024 levels for comparison. Prices
     are also computed by innings, since first-innings win probability is about the
     eventual total and second-innings about a known target, and they differ.
  2. Situational against realised wicket cost. The situational expected-wickets model
     found the expected powerplay wicket rate flat; the actual rate rose a little
     without clearing zero. Priced at the wicket cost, the two give the expected and
     the realised wicket side of the ledger; the difference is the part of the cost
     that was luck rather than approach.
  3. Direct pricing of aggression. Mean change in wprob on powerplay balls played
     with an aggressive shot against other balls, by season: what an aggressive
     shot was worth, and whether that changed.

Caveats that bind: win probability is team-agnostic (it under-credits strong
sides), and its first-innings calibration is weaker than its second-innings one, both
established in the validation notebook; prices are averages over states, so a six at
0 for 2 is priced the same as one at 60 for 0. The ledgers are descriptive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import derived_columns as dc
from mean_level_and_tail import season_contrast_bootstrap as scb

OUTCOMES = ["six", "four", "one_to_three", "no_run", "wicket", "wide", "no_ball", "bye_or_leg_bye"]


def ball_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Per delivery: change in wprob from the previous delivery of the same innings
    (the first delivery of an innings has no previous value and is dropped), the
    outcome category, phase membership and the aggressive-shot flag."""
    order = [c for c in ["p_match", "inns", "over_num", "ball", "ball_in_over"] if c in frame.columns]
    work = frame.sort_values(order).reset_index(drop=True)
    wprob = pd.to_numeric(work["wprob"], errors="coerce")
    prev = wprob.groupby([work["p_match"], work["inns"]]).shift(1)
    work["delta_wprob"] = wprob - prev
    out = dc.is_wicket(work)
    six = dc.is_six(work)
    four = dc.is_four(work)
    wide = pd.to_numeric(work["wide"], errors="coerce").fillna(0) > 0
    noball = pd.to_numeric(work["noball"], errors="coerce").fillna(0) > 0
    extras = (pd.to_numeric(work["byes"], errors="coerce").fillna(0) + pd.to_numeric(work["legbyes"], errors="coerce").fillna(0)) > 0
    batruns = pd.to_numeric(work["batruns"], errors="coerce").fillna(0)
    category = np.select(
        [out, wide, noball, six, four, batruns > 0, extras],
        ["wicket", "wide", "no_ball", "six", "four", "one_to_three", "bye_or_leg_bye"], default="no_run")
    work["outcome_category"] = category
    work["aggressive"] = dc.is_aggressive_shot(work).astype(int) if "shot" in work.columns else 0
    return work[work["delta_wprob"].notna()].reset_index(drop=True)


def _per_match_matrices(balls: pd.DataFrame, season: int, matches: np.ndarray, inns_filter=None) -> dict:
    """Per match: count and delta sum by outcome category, plus aggressive and other
    ball counts and delta sums."""
    s = balls[balls["year"] == season]
    if inns_filter is not None:
        s = s[s["inns"] == inns_filter]
    m_ix = {m: i for i, m in enumerate(matches)}
    counts = np.zeros((len(matches), len(OUTCOMES)))
    sums = np.zeros((len(matches), len(OUTCOMES)))
    agg = np.zeros((len(matches), 4))   # aggressive n, aggressive delta sum, other n, other delta sum
    for (m, cat), g in s.groupby(["p_match", "outcome_category"]):
        counts[m_ix[m], OUTCOMES.index(cat)] += len(g)
        sums[m_ix[m], OUTCOMES.index(cat)] += g["delta_wprob"].sum()
    for m, g in s.groupby("p_match"):
        a = g["aggressive"].to_numpy() == 1
        agg[m_ix[m]] = [a.sum(), g.loc[a, "delta_wprob"].sum(), (~a).sum(), g.loc[~a, "delta_wprob"].sum()]
    return {"counts": counts, "sums": sums, "aggressive": agg}


def powerplay_ledger(frame: pd.DataFrame, settings, seed: int, n_iter: int, balls_per_powerplay: int = 36) -> dict:
    """The three ledgers on powerplay deliveries, pooled over innings and per innings."""
    earlier, later = settings.earlier_season, settings.later_season
    alpha = 1.0 - settings.interval_level
    balls = ball_deltas(frame)
    balls = balls[scb.in_phase(balls, settings.phase("powerplay"))].reset_index(drop=True)
    matches = {s: np.sort(frame.loc[frame["year"] == s, "p_match"].unique()) for s in (earlier, later)}
    counts_e, counts_l = scb.shared_resample_counts(len(matches[earlier]), len(matches[later]), n_iter, seed)
    resample = {earlier: counts_e, later: counts_l}

    def band(a):
        a = a[np.isfinite(a)]
        return float(np.percentile(a, 100 * alpha / 2)), float(np.percentile(a, 100 * (1 - alpha / 2)))

    price_rows, ledger_rows, agg_rows = [], [], []
    for scope, inns_filter in (("both_innings", None), ("first_innings", 1), ("second_innings", 2)):
        mats = {s: _per_match_matrices(balls, s, matches[s], inns_filter) for s in (earlier, later)}

        def season_stats(s, c):
            n = c @ mats[s]["counts"]; d = c @ mats[s]["sums"]
            total = n.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                share = n / total
                price = np.where(n > 0, d / n, np.nan)
            mean_delta = d.sum(axis=1) / total[:, 0]
            return share, price, mean_delta

        point = {s: season_stats(s, np.ones((1, mats[s]["counts"].shape[0]))) for s in (earlier, later)}
        reps = {s: season_stats(s, resample[s]) for s in (earlier, later)}
        pooled_price_point = np.nansum([point[s][1] * point[s][0] for s in (earlier, later)], axis=0) / np.nansum([point[s][0] for s in (earlier, later)], axis=0)
        pooled_price_reps = np.nansum([reps[s][1] * reps[s][0] for s in (earlier, later)], axis=0) / np.nansum([reps[s][0] for s in (earlier, later)], axis=0)

        for j, cat in enumerate(OUTCOMES):
            price_rows.append({"scope": scope, "outcome": cat,
                               f"share_{earlier}": float(point[earlier][0][0, j]), f"share_{later}": float(point[later][0][0, j]),
                               "share_change_per_100_balls": 100 * float(point[later][0][0, j] - point[earlier][0][0, j]),
                               f"price_{earlier}": float(point[earlier][1][0, j]), f"price_{later}": float(point[later][1][0, j]),
                               "price_pooled": float(pooled_price_point[0, j]),
                               "share_change_low": band(100 * (reps[later][0][:, j] - reps[earlier][0][:, j]))[0],
                               "share_change_high": band(100 * (reps[later][0][:, j] - reps[earlier][0][:, j]))[1],
                               "price_pooled_low": band(pooled_price_reps[:, j])[0], "price_pooled_high": band(pooled_price_reps[:, j])[1]})

        for price_name, price_point, price_reps in ((f"prices_{earlier}", point[earlier][1], reps[earlier][1]),
                                                    ("prices_pooled", pooled_price_point, pooled_price_reps),
                                                    (f"prices_{later}", point[later][1], reps[later][1])):
            share_change_pt = point[later][0] - point[earlier][0]
            share_change_rp = reps[later][0] - reps[earlier][0]
            contrib_pt = balls_per_powerplay * np.nan_to_num(share_change_pt * price_point)
            contrib_rp = balls_per_powerplay * np.nan_to_num(share_change_rp * price_reps)
            for j, cat in enumerate(OUTCOMES):
                lo, hi = band(contrib_rp[:, j])
                ledger_rows.append({"scope": scope, "prices": price_name, "outcome": cat,
                                    "points_per_powerplay": float(contrib_pt[0, j]), "ci_low": lo, "ci_high": hi})
            total = contrib_rp.sum(axis=1)
            lo, hi = band(total)
            ledger_rows.append({"scope": scope, "prices": price_name, "outcome": "net",
                                "points_per_powerplay": float(contrib_pt.sum()), "ci_low": lo, "ci_high": hi,
                                "p_positive": float((total > 0).mean())})
        for s in (earlier, later):
            ledger_rows.append({"scope": scope, "prices": "martingale_check", "outcome": f"mean_delta_per_ball_{s}",
                                "points_per_powerplay": balls_per_powerplay * float(point[s][2][0]),
                                "ci_low": band(balls_per_powerplay * reps[s][2])[0], "ci_high": band(balls_per_powerplay * reps[s][2])[1]})

        for s in (earlier, later):
            a = resample[s] @ mats[s]["aggressive"]
            a0 = np.ones((1, mats[s]["aggressive"].shape[0])) @ mats[s]["aggressive"]
            with np.errstate(invalid="ignore", divide="ignore"):
                agg_price, oth_price = a[:, 1] / a[:, 0], a[:, 3] / a[:, 2]
                agg0, oth0 = a0[0, 1] / a0[0, 0], a0[0, 3] / a0[0, 2]
            agg_rows.append({"scope": scope, "season": s, "aggressive_balls": int(a0[0, 0]), "other_balls": int(a0[0, 2]),
                             "price_aggressive": float(agg0), "price_other": float(oth0), "premium": float(agg0 - oth0),
                             "premium_low": band(agg_price - oth_price)[0], "premium_high": band(agg_price - oth_price)[1]})
    prices = pd.DataFrame(price_rows)
    ledger = pd.DataFrame(ledger_rows)
    aggression = pd.DataFrame(agg_rows)
    return {"prices": prices, "ledger": ledger, "aggression": aggression, "n_balls": int(len(balls)),
            "balls_per_powerplay": balls_per_powerplay}


def situational_vs_realised(ledger_prices: pd.DataFrame, xw_change: float, xw_low: float, xw_high: float,
                            actual_change: float, actual_low: float, actual_high: float,
                            scope: str = "both_innings", balls_per_powerplay: int = 36) -> pd.DataFrame:
    """Wicket side of the ledger at the pooled wicket price: the change in expected
    wickets per ball from the situational model, and the change in the actual rate,
    each times the price and the balls per powerplay."""
    price = float(ledger_prices.loc[(ledger_prices["scope"] == scope) & (ledger_prices["outcome"] == "wicket"), "price_pooled"].iloc[0])
    rows = []
    for name, c, lo, hi in (("situational_expected_wickets", xw_change, xw_low, xw_high),
                            ("realised_wicket_rate", actual_change, actual_low, actual_high)):
        rows.append({"side": name, "wicket_rate_change_per_ball": c, "wicket_price_points": price,
                     "points_per_powerplay": balls_per_powerplay * c * price,
                     "ci_low": balls_per_powerplay * min(lo * price, hi * price),
                     "ci_high": balls_per_powerplay * max(lo * price, hi * price)})
    rows.append({"side": "luck_component_realised_minus_expected", "wicket_rate_change_per_ball": actual_change - xw_change,
                 "wicket_price_points": price, "points_per_powerplay": balls_per_powerplay * (actual_change - xw_change) * price,
                 "ci_low": np.nan, "ci_high": np.nan})
    return pd.DataFrame(rows)


def plot_ledger(ledger: pd.DataFrame, scope: str, prices: str, title: str):
    """Per-outcome contribution to the net value with intervals, and the net."""
    import matplotlib.pyplot as plt

    sub = ledger[(ledger["scope"] == scope) & (ledger["prices"] == prices)]
    order = OUTCOMES + ["net"]
    sub = sub.set_index("outcome").loc[order]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colours = ["#2F5C8A"] * len(OUTCOMES) + ["#1F2937"]
    ax.hlines(y, sub["ci_low"], sub["ci_high"], color=colours, lw=2)
    ax.plot(sub["points_per_powerplay"], y, "o", color="#1F2937", ms=6)
    ax.axvline(0, color="#C0392B", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([o.replace("_", " ") for o in order], fontsize=9)
    ax.set_xlabel("win-probability points per 36-ball powerplay, 2024 mix minus 2023 mix")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
