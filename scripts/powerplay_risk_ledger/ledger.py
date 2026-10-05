"""
The powerplay risk ledger: what each powerplay took on, what it was expected to
cost, what it cost, and what it bought, in win-probability points.

Three levels, each built from the one below:
  per ball          every powerplay delivery with its situational expected-wicket
                    value (out-of-fold, from the situation-only model), its outcome
                    category, the price of that outcome (mean change in win
                    probability on such balls, both seasons pooled), and the actual
                    change in win probability.
  per innings       one row per powerplay: legal balls, aggressive-shot share,
                    sixes, fours, wickets, expected wickets (sum of per-ball
                    values), the wicket residual (actual minus expected, the luck
                    component), and the value of the outcome mix at fixed prices
                    split into its parts.
  per team-season   the innings rows summed and averaged per powerplay, with a
                    match cluster bootstrap interval, for the strategic comparison.

Prices are pooled over both seasons so a season's innings are not valued at its own
prices; the ledger notebook showed the season difference in the mix is what
matters and the price level is a scaling. Expected wickets come from the logistic
situation-only model, the carrying model in the expected-wickets analysis; it is
refitted here out-of-fold so the ledger is self-contained (a minute or so).

Caveats carried from the earlier notebooks: win probability is team-agnostic and
weaker in first innings; prices are averages over game states; a team-season is
fourteen powerplays, so team rows are read as tiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import derived_columns as dc
from expected_wickets import xw_model as xm
from expected_wickets.xw_features import FEATURE_VARIANTS, build_xw_features
from mean_level_and_tail import season_contrast_bootstrap as scb
from win_probability_ledger.powerplay_ledger import OUTCOMES, ball_deltas

WICKET = "wicket"


def per_ball_ledger(frame: pd.DataFrame, settings, seed: int, variant: str = "situation",
                    kind: str = "logistic") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Powerplay deliveries with expected wickets (legal balls), outcome category,
    price and actual win-probability change. Returns (balls, prices)."""
    data = build_xw_features(frame)
    X = data.X[FEATURE_VARIANTS[variant]]
    xw, _ = xm.oof_xw(X, data.y, data.groups, kind=kind, seed=seed)
    # Align xW back to the frame rows: the feature builder orders by innings and ball
    # and keeps legal deliveries; rebuild the same ordering here.
    order = [c for c in ["p_match", "inns", "over_num", "ball", "ball_in_over"] if c in frame.columns]
    ordered = frame.sort_values(order).reset_index(drop=True)
    legal = dc.is_legal_delivery(ordered).to_numpy()
    ordered["xw"] = np.nan
    ordered.loc[legal, "xw"] = xw
    balls = ball_deltas(ordered)
    pp = np.asarray(scb.in_phase(balls, settings.phase("powerplay")))
    balls = balls[pp].reset_index(drop=True)
    prices = (balls.groupby("outcome_category")["delta_wprob"].agg(["mean", "size"])
              .reindex(OUTCOMES).rename(columns={"mean": "price_points", "size": "balls"}).reset_index())
    price_of = prices.set_index("outcome_category")["price_points"]
    balls["price_points"] = balls["outcome_category"].map(price_of)
    balls["legal"] = dc.is_legal_delivery(balls).astype(int).to_numpy()
    balls["faced"] = dc.is_ball_faced(balls).astype(int).to_numpy()
    balls["six"] = dc.is_six(balls).astype(int).to_numpy()
    balls["four"] = dc.is_four(balls).astype(int).to_numpy()
    balls["wicket"] = dc.is_wicket(balls).astype(int).to_numpy()
    keep = ["p_match", "year", "inns", "team_bat", "team_bowl", "over_num", "ball", "outcome_category", "price_points",
            "delta_wprob", "xw", "legal", "faced", "aggressive", "six", "four", "wicket", "batruns", "score"]
    keep = [c for c in keep if c in balls.columns]
    return balls[keep].copy(), prices


def per_innings_ledger(balls: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per powerplay (match, innings)."""
    wicket_price = float(prices.set_index("outcome_category").loc[WICKET, "price_points"])
    g = balls.groupby(["year", "p_match", "inns", "team_bat", "team_bowl"])
    out = g.agg(deliveries=("legal", "size"), legal_balls=("legal", "sum"), faced=("faced", "sum"),
                aggressive_shots=("aggressive", "sum"), sixes=("six", "sum"), fours=("four", "sum"),
                wickets=("wicket", "sum"), runs=("score", "sum"), expected_wickets=("xw", "sum"),
                value_points=("price_points", "sum"), actual_wprob_change=("delta_wprob", "sum")).reset_index()
    out["attempt_share"] = out["aggressive_shots"] / out["faced"].replace(0, np.nan)
    out["expected_wicket_cost_points"] = out["expected_wickets"] * wicket_price
    out["realised_wicket_cost_points"] = out["wickets"] * wicket_price
    out["wicket_residual"] = out["wickets"] - out["expected_wickets"]
    out["luck_points"] = out["wicket_residual"] * wicket_price
    six_price = float(prices.set_index("outcome_category").loc["six", "price_points"])
    four_price = float(prices.set_index("outcome_category").loc["four", "price_points"])
    out["boundary_value_points"] = out["sixes"] * six_price + out["fours"] * four_price
    out["value_excluding_wickets_points"] = out["value_points"] - out["realised_wicket_cost_points"]
    out["expected_net_points"] = out["value_excluding_wickets_points"] + out["expected_wicket_cost_points"]
    return out


def per_team_season_ledger(innings: pd.DataFrame, settings, seed: int, n_iter: int) -> pd.DataFrame:
    """Team-season means per powerplay with match bootstrap intervals (each team's
    matches resampled within season)."""
    alpha = 1.0 - settings.interval_level
    metrics = ["attempt_share", "sixes", "wickets", "expected_wickets", "wicket_residual", "expected_wicket_cost_points",
               "realised_wicket_cost_points", "luck_points", "boundary_value_points", "value_points", "expected_net_points"]
    rows = []
    rng = np.random.default_rng(seed)
    for (team, season), g in innings.groupby(["team_bat", "year"]):
        n = len(g)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_iter).astype(float)
        row = {"team": team, "season": int(season), "powerplays": n}
        for m in metrics:
            v = g[m].to_numpy(dtype=float)
            row[m] = float(v.mean())
            reps = (counts @ v) / n
            row[f"{m}_low"], row[f"{m}_high"] = float(np.percentile(reps, 100 * alpha / 2)), float(np.percentile(reps, 100 * (1 - alpha / 2)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["season", "value_points"], ascending=[True, False]).reset_index(drop=True)


def plot_team_ledger(team: pd.DataFrame, earlier: int, later: int):
    """Two panels: net value per powerplay by team and season with intervals; and
    expected against realised wicket cost per powerplay, one point per team-season."""
    import matplotlib.pyplot as plt

    colours = {earlier: "#2c7fb8", later: "#d95f0e"}
    order = team[team["season"] == later].sort_values("value_points")["team"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(order) + 1.8))
    ax = axes[0]
    y = np.arange(len(order))
    for season in (earlier, later):
        s = team[team["season"] == season].set_index("team").reindex(order)
        off = 0.15 if season == later else -0.15
        ax.hlines(y + off, s["value_points_low"], s["value_points_high"], color=colours[season], lw=2)
        ax.plot(s["value_points"], y + off, "o", color=colours[season], ms=6, label=str(season))
    ax.axvline(0, color="#C0392B", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("value of the powerplay outcome mix, win-probability points per powerplay")
    ax.set_title("What the powerplay bought, by team", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, title="season")
    ax = axes[1]
    for season in (earlier, later):
        s = team[team["season"] == season]
        ax.scatter(-s["expected_wicket_cost_points"], -s["realised_wicket_cost_points"], color=colours[season], s=40, label=str(season), alpha=0.85, edgecolor="white")
        for _, r in s.iterrows():
            ax.annotate(r["team"].split()[0], (-r["expected_wicket_cost_points"], -r["realised_wicket_cost_points"]), fontsize=7, xytext=(3, 2), textcoords="offset points", color=colours[season])
    lim = max(float((-team[["expected_wicket_cost_points", "realised_wicket_cost_points"]]).max().max()) * 1.1, 1e-6)
    ax.plot([0, lim], [0, lim], "--", color="#C0392B", lw=1)
    ax.set_xlabel("expected wicket cost per powerplay (points, given the situations faced)")
    ax.set_ylabel("realised wicket cost per powerplay (points)")
    ax.set_title("Expected against realised wicket cost", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, title="season")
    for a in axes:
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
