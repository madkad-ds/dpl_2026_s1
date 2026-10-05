"""
Who moved the powerplay spread: rank probabilities and variance contributions.

Naming teams as "drivers" when no individual interval clears zero is post-hoc
labelling, and the change in a team's absolute deviation has a skewed posterior,
so point rankings mislead. This module reads the gear-up model's posterior draws
(no refit) and reports probabilities instead of labels.

Per draw d, team t, season s, with theta the shrunk team-season gear-up:
  1. deviation      e[t,s] = theta[t,s] - mean over teams of theta[.,s]
                    (relative to each season's own league mean, so a league-wide
                    level shift does not register as anyone pulling away);
  2. pull           p[t]   = |e[t,2024]| - |e[t,2023]|, positive = moved away from
                    the pack;
  3. rank of p[t] within the draw; reported per team as P(rank <= top_k), P(p > 0),
     posterior median pull with an interval;
  4. variance contribution c[t] = (e[t,2024]^2 - e[t,2023]^2) / n_teams, in
     run-rate-squared units. Contributions sum to the posterior change in the
     between-team variance of the shrunk effects, so they decompose the widening
     exactly;
  5. shares c[t] / sum(c) only on draws where sum(c) > 0, and only if
     P(sum(c) > 0) is at least the configured gate; otherwise omitted because the
     denominator is unstable.

Kept from before: the check that every team's raw (unshrunk) 2024 deviation has the
same sign as its shrunk deviation, so no ordering is a pooling artefact.

Figures: a team x rank heatmap of rank probabilities, and a dot-and-interval plot of
variance contributions sorted by median.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mean_level_and_tail import season_contrast_bootstrap as scb


def team_gear_draws(idata, teams) -> tuple[np.ndarray, np.ndarray]:
    """Shrunk team gear-up draws split by season: (earlier, later), each
    (draws, n_teams) in the team order the fit returned. team_gear is flattened
    season-major: first block earlier, second block later."""
    post = idata.posterior
    n_teams = len(teams)
    gear = post["team_gear"].values.reshape(-1, 2 * n_teams)
    return gear[:, :n_teams], gear[:, n_teams:]


def attribution_from_draws(earlier: np.ndarray, later: np.ndarray, teams,
                           top_k: int = 3, credible_level: float = 0.90,
                           share_gate: float = 0.90) -> dict:
    """Rank probabilities, pull, and variance contributions per team from
    posterior draws of the team-season gear-up."""
    n_teams = len(teams)
    lo_q, hi_q = 100 * (1 - credible_level) / 2, 100 * (1 + credible_level) / 2

    earlier_dev = earlier - earlier.mean(axis=1, keepdims=True)
    later_dev = later - later.mean(axis=1, keepdims=True)
    pull = np.abs(later_dev) - np.abs(earlier_dev)                     # (draws, teams)
    contribution = (later_dev ** 2 - earlier_dev ** 2) / n_teams       # (draws, teams)

    # Rank 1 = largest pull within the draw.
    order = np.argsort(-pull, axis=1)
    ranks = np.empty_like(order)
    rows_idx = np.arange(pull.shape[0])[:, None]
    ranks[rows_idx, order] = np.arange(1, n_teams + 1)[None, :]
    rank_prob = np.stack([(ranks == r).mean(axis=0) for r in range(1, n_teams + 1)], axis=1)

    total = contribution.sum(axis=1)
    p_total_positive = float((total > 0).mean())
    shares_reported = p_total_positive >= share_gate
    if shares_reported:
        positive = total > 0
        shares = contribution[positive] / total[positive, None]
    else:
        shares = None

    rows = []
    for j, team in enumerate(teams):
        row = {
            "team": team,
            f"gear_dev_{scb.EARLIER_SEASON}": float(np.median(earlier_dev[:, j])),
            f"gear_dev_{scb.LATER_SEASON}": float(np.median(later_dev[:, j])),
            "pull_median": float(np.median(pull[:, j])),
            "pull_low": float(np.percentile(pull[:, j], lo_q)),
            "pull_high": float(np.percentile(pull[:, j], hi_q)),
            "p_pull_positive": float((pull[:, j] > 0).mean()),
            f"p_rank_top_{top_k}": float((ranks[:, j] <= top_k).mean()),
            "contribution_median": float(np.median(contribution[:, j])),
            "contribution_low": float(np.percentile(contribution[:, j], lo_q)),
            "contribution_high": float(np.percentile(contribution[:, j], hi_q)),
        }
        if shares_reported:
            row["share_median"] = float(np.median(shares[:, j]))
            row["share_low"] = float(np.percentile(shares[:, j], lo_q))
            row["share_high"] = float(np.percentile(shares[:, j], hi_q))
        rows.append(row)
    table = (pd.DataFrame(rows).sort_values(f"p_rank_top_{top_k}", ascending=False)
             .reset_index(drop=True))
    rank_table = pd.DataFrame(rank_prob, index=list(teams),
                              columns=[f"rank_{r}" for r in range(1, n_teams + 1)])
    return {
        "table": table,
        "rank_probabilities": rank_table,
        "variance_change_median": float(np.median(total)),
        "variance_change_low": float(np.percentile(total, lo_q)),
        "variance_change_high": float(np.percentile(total, hi_q)),
        "p_variance_increased": p_total_positive,
        "shares_reported": bool(shares_reported),
        "credible_level": credible_level,
        "top_k": top_k,
    }


def raw_gear_up_deviation_by_team(frame, powerplay, middle) -> pd.DataFrame:
    """Unshrunk per-team powerplay gear-up delta (powerplay run rate minus that
    team's middle-overs run rate) each season, as deviations from the season's
    league mean, and the change in distance from the pack.

    This is the raw-data counterpart to the model's shrunk team effects. The
    model pulls magnitudes toward the league but must not flip the direction of a
    team's movement; comparing raw against shrunk direction is the check that an
    ordering is driven by the team's own data, not by pooling.
    """
    legal = scb.is_legal_delivery(frame)

    def phase_rate(phase):
        m = legal & scb.in_phase(frame, phase)
        g = (frame.loc[m].groupby(["year", "team_bat"])
                  .agg(runs=("score", "sum"), balls=("score", "size")))
        return (g["runs"] / g["balls"] * 6.0).rename("rate")

    pp = phase_rate(powerplay)
    mid = phase_rate(middle)
    gear = (pp - mid).rename("gear_up").reset_index()

    # Relative to each season's league mean gear-up, matching the attribution.
    gear["league_mean"] = gear.groupby("year")["gear_up"].transform("mean")
    gear["gear_dev"] = gear["gear_up"] - gear["league_mean"]

    wide = gear.pivot(index="team_bat", columns="year", values="gear_dev")
    wide.columns = [f"raw_dev_{c}" for c in wide.columns]
    wide["raw_pull"] = wide.iloc[:, 1].abs() - wide.iloc[:, 0].abs()   # later minus earlier distance from pack
    return wide.sort_values("raw_pull", ascending=False)


def shrunk_vs_raw_sign_check(attribution_table: pd.DataFrame, raw_table: pd.DataFrame) -> pd.DataFrame:
    """Every team: sign of the shrunk later-season deviation against the raw one.
    A disagreement means pooling flipped a direction and that team's ordering is
    not its own data."""
    later_col = f"gear_dev_{scb.LATER_SEASON}"
    raw_col = [c for c in raw_table.columns if c.startswith("raw_dev_") and c.endswith(str(scb.LATER_SEASON))][0]
    rows = []
    for _, r in attribution_table.iterrows():
        raw = float(raw_table.loc[r["team"], raw_col]) if r["team"] in raw_table.index else np.nan
        rows.append({"team": r["team"], "shrunk_dev_later": r[later_col], "raw_dev_later": raw,
                     "signs_agree": bool(np.sign(r[later_col]) == np.sign(raw)) if np.isfinite(raw) else None})
    return pd.DataFrame(rows)


def plot_rank_probabilities(rank_table: pd.DataFrame, top_k: int):
    import matplotlib.pyplot as plt

    n = rank_table.shape[0]
    fig, ax = plt.subplots(figsize=(0.62 * n + 3.0, 0.42 * n + 1.6))
    ordered = rank_table.loc[rank_table.iloc[:, :top_k].sum(axis=1).sort_values(ascending=False).index]
    image = ax.imshow(ordered.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels([c.replace("rank_", "") for c in ordered.columns])
    ax.set_yticks(range(n))
    ax.set_yticklabels(ordered.index, fontsize=9)
    ax.set_xlabel("rank by pull from the pack (1 = largest)")
    ax.axvline(top_k - 0.5, color="#C0392B", lw=1, ls="--")
    for i in range(n):
        for j in range(n):
            v = ordered.iat[i, j]
            if v >= 0.10:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "#1F2937")
    ax.set_title("Posterior probability of each rank in powerplay pull from the pack",
                 loc="left", fontsize=10, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def plot_variance_contributions(table: pd.DataFrame, credible_level: float):
    import matplotlib.pyplot as plt

    ordered = table.sort_values("contribution_median", ascending=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(ordered) + 1.5))
    ax.hlines(y, ordered["contribution_low"], ordered["contribution_high"], color="#2F5C8A", lw=1.8)
    ax.plot(ordered["contribution_median"], y, "o", color="#2F5C8A", ms=6)
    ax.axvline(0, color="#C0392B", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["team"], fontsize=9)
    ax.set_xlabel("contribution to the change in between-team gear-up variance (rpo squared)")
    ax.set_title("Variance contributions by team", loc="left", fontsize=10, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.01, -0.02, f"Posterior median and {int(credible_level * 100)}% interval from the "
             "gear-up model draws; contributions sum to the variance change.",
             fontsize=7.5, color="#6A6A6A", style="italic")
    fig.tight_layout()
    return fig
