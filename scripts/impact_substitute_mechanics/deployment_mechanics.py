"""
Impact-substitute mechanics: what the first-innings deployments were, how teams
built their elevens around the substitute, and whether batting first with a
batting-heavy eleven shows in powerplay behaviour.

Three questions, each with its own section:

  1. Who deployed in the first innings, and whom. A team bowling first is expected
     to name its strongest bowling eleven and bring the extra batter in during its
     fielding innings (a "warm-up" for the chase); a team batting first is expected
     to bring a bowler in for a dismissed batter. Each deployment is classified by
     whether the substituting team batted first, the innings of deployment, the
     over of entry, and the incoming and outgoing players' roles.
  2. Eleven composition by batting order. With player roles classified from each
     player's two-season record (batter, bowler, all-rounder), the initial eleven of
     every team-match (the named eleven before the substitute entered) is counted by
     role and compared for teams batting first against teams bowling first, per
     season, under the match cluster bootstrap.
  3. Batting first against chasing, within team. Powerplay aggressive-attempt share
     and gear-up (powerplay run rate minus the team's own middle-overs run rate) for
     each team's batting-first innings against its chasing innings in the same
     season; the mean team gap per season and its change from 2023 to 2024, under the
     match cluster bootstrap on shared draws. The chase carries its own target
     pressure, so this is a descriptive contrast with intervals, not a test of the
     depth mechanism.

Player roles: a player is a bowler if he bowled at least `role_min_balls` legal
balls across the two seasons and faced fewer than that; a batter if the reverse;
an all-rounder if both; thin otherwise. The threshold lives in config.toml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import derived_columns as dc
from mean_level_and_tail import season_contrast_bootstrap as scb


# ----------------------------------------------------------------------------
# Player roles from the two-season record
# ----------------------------------------------------------------------------

def classify_player_roles(frame: pd.DataFrame, min_balls: int) -> pd.DataFrame:
    """One row per player id: balls faced, legal balls bowled, and role."""
    faced = (pd.Series(dc.is_ball_faced(frame).astype(int).to_numpy()).groupby(frame["p_bat"].to_numpy()).sum()
             .rename("balls_faced"))
    bowled = (pd.Series(dc.is_legal_delivery(frame).astype(int).to_numpy()).groupby(frame["p_bowl"].to_numpy()).sum()
              .rename("balls_bowled"))
    roles = pd.concat([faced, bowled], axis=1).fillna(0).astype(int)
    bats = roles["balls_faced"] >= min_balls
    bowls = roles["balls_bowled"] >= min_balls
    roles["role"] = np.select([bats & bowls, bowls, bats], ["all_rounder", "bowler", "batter"], default="thin")
    names = pd.concat([frame.drop_duplicates("p_bat").set_index("p_bat")["bat"],
                       frame.drop_duplicates("p_bowl").set_index("p_bowl")["bowl"]])
    roles["name"] = names[~names.index.duplicated()].reindex(roles.index)
    return roles.rename_axis("player_id").reset_index()


# ----------------------------------------------------------------------------
# 1. First-innings deployments: who, whom, when
# ----------------------------------------------------------------------------

def classify_deployments(impact: pd.DataFrame, frame: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Impact-player events with the substituting side's batting order, the
    incoming and outgoing players' two-season roles, and whether the incoming
    player batted or bowled in that match."""
    events = impact[impact["reason"] == "impact_player"].copy()
    events["match_id"] = events["match_id"].astype(int)
    events["season"] = events["season"].astype(int)
    role_of = roles.set_index("player_id")["role"]
    events["in_role"] = events["in_dataset_id"].map(role_of).fillna("unknown")
    events["out_role"] = events["out_dataset_id"].map(role_of).fillna("unknown")
    batted = set(zip(frame["p_match"].to_numpy(), frame["p_bat"].to_numpy()))
    bowled = set(zip(frame["p_match"].to_numpy(), frame["p_bowl"].to_numpy()))
    events["in_batted_in_match"] = [((m, p) in batted) for m, p in zip(events["match_id"], events["in_dataset_id"])]
    events["in_bowled_in_match"] = [((m, p) in bowled) for m, p in zip(events["match_id"], events["in_dataset_id"])]
    events["side"] = np.where(events["subbing_team_batted_first"], "batted_first", "bowled_first")
    # In the first innings the batting-first side is batting and the bowling-first side
    # fielding; in the second innings the reverse. Name the discipline the substituting
    # side was engaged in when the substitute entered.
    events["discipline_at_entry"] = np.where(
        (events["innings"] == 1) == events["subbing_team_batted_first"], "batting", "fielding")
    return events


def dismissal_linked_swaps(events: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Batting-first sides' first-innings substitutions: had the outgoing player
    already been dismissed in that innings before the substitute entered, how many
    balls after the dismissal did the entry come, and did the incoming player bowl in
    the second innings. If most swaps follow a dismissal within an over and the
    incoming player then bowls, the side batted with depth and converted a spent
    batter into a bowler."""
    swaps = events[(events["innings"] == 1) & (events["side"] == "batted_first")].copy()
    first = frame[frame["inns"] == 1]
    dismissed = first[first["p_out_clean"].notna()][["p_match", "p_out_clean", "ball_id"]]
    dismissed = dismissed.rename(columns={"p_out_clean": "player", "ball_id": "dismissal_ball_id"})
    dismissed["player"] = dismissed["player"].astype("Int64")
    second_bowlers = set(zip(frame.loc[frame["inns"] == 2, "p_match"].to_numpy(),
                             frame.loc[frame["inns"] == 2, "p_bowl"].to_numpy()))
    rows = []
    for _, e in swaps.iterrows():
        d = dismissed[(dismissed["p_match"] == e["match_id"]) & (dismissed["player"] == e["out_dataset_id"])]
        dismissal_ball = float(d["dismissal_ball_id"].min()) if len(d) else np.nan
        out_dismissed_before = bool(len(d)) and dismissal_ball <= e["deploy_ball_id"]
        # ball_id is over.ball; balls between as (whole overs x 6 + balls) on the raw count
        if out_dismissed_before:
            lag = (int(e["deploy_ball_id"]) - int(dismissal_ball)) * 6 + round((e["deploy_ball_id"] % 1) * 100) - round((dismissal_ball % 1) * 100)
        else:
            lag = np.nan
        rows.append({"season": e["season"], "match_id": e["match_id"], "team": e["team"],
                     "player_out": e["player_out"], "out_role": e["out_role"],
                     "player_in": e["player_in"], "in_role": e["in_role"],
                     "deploy_over": int(e["dataset_over"]), "deploy_ball_id": e["deploy_ball_id"],
                     "out_dismissed_before_entry": out_dismissed_before,
                     "dismissal_ball_id": dismissal_ball, "balls_after_dismissal": lag,
                     "in_bowled_second_innings": (e["match_id"], e["in_dataset_id"]) in second_bowlers})
    return pd.DataFrame(rows)


def dismissal_linked_summary(swaps: pd.DataFrame) -> pd.DataFrame:
    """Per season: swaps, share following the outgoing player's dismissal, median
    balls after it, share of incoming players who bowled in the second innings, and
    the incoming role mix."""
    if swaps.empty:
        return pd.DataFrame()
    g = swaps.groupby("season")
    out = g.agg(swaps=("match_id", "size"),
                out_dismissed_before_entry=("out_dismissed_before_entry", "mean"),
                median_balls_after_dismissal=("balls_after_dismissal", "median"),
                within_one_over=("balls_after_dismissal", lambda x: float((x <= 6).mean()) if x.notna().any() else np.nan),
                in_bowled_second_innings=("in_bowled_second_innings", "mean")).reset_index()
    roles = swaps.groupby(["season", "in_role"]).size().unstack("in_role", fill_value=0).add_prefix("in_").reset_index()
    return out.merge(roles, on="season")


def swaps_by_team(events: pd.DataFrame) -> pd.DataFrame:
    """Batting-first in-innings swaps per team and season: a league-wide habit or a
    few teams."""
    swaps = events[(events["innings"] == 1) & (events["side"] == "batted_first")]
    return (swaps.groupby(["team", "season"]).size().rename("swaps").unstack("season", fill_value=0)
            .reset_index().sort_values(by=list(sorted(events["season"].unique()))[-1], ascending=False))


def deployment_summary(events: pd.DataFrame) -> dict:
    """Counts by season, side and innings; first-innings deployments by side and the
    incoming player's role; over of entry for first-innings deployments by side."""
    by_side_innings = (events.groupby(["season", "side", "innings"]).size().rename("events")
                       .unstack("innings", fill_value=0).reset_index())
    first = events[events["innings"] == 1]
    first_roles = (first.groupby(["season", "side", "in_role"]).size().rename("events")
                   .unstack("in_role", fill_value=0).reset_index())
    first_out_roles = (first.groupby(["season", "side", "out_role"]).size().rename("events")
                       .unstack("out_role", fill_value=0).reset_index())
    entry = (first.groupby(["season", "side"])["dataset_over"]
             .agg(events="size", median_over="median", mean_over="mean",
                  share_in_death=lambda s: float((s >= 16).mean()))
             .reset_index())
    warm_up = first[first["side"] == "bowled_first"]
    warm_up_batted_later = (warm_up.groupby("season")
                            .agg(events=("in_batted_in_match", "size"),
                                 in_player_batted=("in_batted_in_match", "mean"),
                                 in_player_bowled=("in_bowled_in_match", "mean")).reset_index())
    return {"by_side_innings": by_side_innings, "first_innings_in_roles": first_roles,
            "first_innings_out_roles": first_out_roles, "first_innings_entry_over": entry,
            "bowled_first_first_innings_subs": warm_up_batted_later}


# ----------------------------------------------------------------------------
# 2. Eleven composition by batting order
# ----------------------------------------------------------------------------

def initial_elevens(rosters: pd.DataFrame, frame: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team): role counts in the initial eleven (starting eleven
    plus the player later replaced), and whether the team batted first."""
    r = rosters.copy()
    r["match_id"] = r["match_id"].astype(int)
    r["season"] = r["season"].astype(int)
    initial = r[r["role"].isin(["starting_xi", "replaced"])].copy()
    role_of = roles.set_index("player_id")["role"]
    initial["player_role"] = initial["dataset_id"].map(role_of).fillna("unknown")
    counts = (initial.groupby(["match_id", "season", "team", "player_role"]).size()
              .unstack("player_role", fill_value=0).reset_index())
    for col in ("batter", "bowler", "all_rounder", "thin"):
        if col not in counts:
            counts[col] = 0
    counts["eleven_size"] = initial.groupby(["match_id", "season", "team"]).size().to_numpy()
    counts["recognised_bowlers"] = counts["bowler"] + counts["all_rounder"]
    counts["recognised_batters"] = counts["batter"] + counts["all_rounder"]
    first_bat = (frame[frame["inns"] == 1].drop_duplicates("p_match")[["p_match", "team_bat"]]
                 .rename(columns={"p_match": "match_id", "team_bat": "team_batted_first"}))
    first_bat["team_batted_first"] = first_bat["team_batted_first"].astype(str)
    # Inner merge: a roster match absent from the ball-by-ball frame has no batting
    # order and is dropped rather than counted as bowling first.
    counts = counts.merge(first_bat, on="match_id", how="inner")
    counts["batted_first"] = counts["team"].astype(str) == counts["team_batted_first"]
    return counts


def eleven_composition_contrast(elevens: pd.DataFrame, settings, seed: int, n_iter: int) -> pd.DataFrame:
    """Mean recognised bowlers (and batters) in the initial eleven, batting-first
    teams against bowling-first teams, per season, and the gap with a match cluster
    bootstrap interval; then the 2023 to 2024 change in the gap."""
    earlier, later = settings.earlier_season, settings.later_season
    alpha = 1.0 - settings.interval_level
    rows = []
    per_match = {}
    for season in (earlier, later):
        s = elevens[elevens["season"] == season]
        matches = np.sort(s["match_id"].unique())
        # Each match contributes one batting-first and one bowling-first eleven.
        bf = s[s["batted_first"]].set_index("match_id").reindex(matches)
        bo = s[~s["batted_first"]].set_index("match_id").reindex(matches)
        per_match[season] = {"matches": matches, "bf": bf, "bo": bo}
    n_e, n_l = len(per_match[earlier]["matches"]), len(per_match[later]["matches"])
    counts = dict(zip((earlier, later), scb.shared_resample_counts(n_e, n_l, n_iter, seed)))
    out = {}
    for metric in ("recognised_bowlers", "recognised_batters", "bowler", "batter", "all_rounder"):
        gaps = {}
        for season in (earlier, later):
            pm = per_match[season]
            bf = pm["bf"][metric].to_numpy(dtype=float)
            bo = pm["bo"][metric].to_numpy(dtype=float)
            c = counts[season]
            mean_bf = (c @ np.nan_to_num(bf)) / (c @ (~np.isnan(bf)).astype(float))
            mean_bo = (c @ np.nan_to_num(bo)) / (c @ (~np.isnan(bo)).astype(float))
            gap = mean_bf - mean_bo
            gaps[season] = gap
            rows.append({"metric": metric, "season": season,
                         "mean_batting_first": float(np.nanmean(bf)), "mean_bowling_first": float(np.nanmean(bo)),
                         "gap_batting_minus_bowling_first": float(np.nanmean(bf) - np.nanmean(bo)),
                         "gap_low": float(np.percentile(gap, 100 * alpha / 2)),
                         "gap_high": float(np.percentile(gap, 100 * (1 - alpha / 2))),
                         "p_gap_positive": float((gap > 0).mean())})
        change = gaps[later] - gaps[earlier]
        rows.append({"metric": metric, "season": f"{later} minus {earlier}",
                     "mean_batting_first": np.nan, "mean_bowling_first": np.nan,
                     "gap_batting_minus_bowling_first": float(rows[-1]["gap_batting_minus_bowling_first"] - rows[-2]["gap_batting_minus_bowling_first"]),
                     "gap_low": float(np.percentile(change, 100 * alpha / 2)),
                     "gap_high": float(np.percentile(change, 100 * (1 - alpha / 2))),
                     "p_gap_positive": float((change > 0).mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3a. Batting first against chasing, within team
# ----------------------------------------------------------------------------

def _team_innings_sums(frame: pd.DataFrame, settings) -> pd.DataFrame:
    """Per (season, match, team, batting order): powerplay aggressive attempts and
    faced balls; powerplay and middle-overs runs and legal balls."""
    pp = scb.in_phase(frame, settings.phase("powerplay"))
    mid = scb.in_phase(frame, settings.phase("middle"))
    faced = dc.is_ball_faced(frame).to_numpy()
    legal = dc.is_legal_delivery(frame).to_numpy()
    aggressive = dc.is_aggressive_shot(frame).to_numpy()
    score = pd.to_numeric(frame["score"], errors="coerce").fillna(0).to_numpy(dtype=float)
    work = pd.DataFrame({
        "season": frame["year"].to_numpy(), "p_match": frame["p_match"].to_numpy(),
        "team": frame["team_bat"].astype(str).to_numpy(),
        "batted_first": (frame["inns"].to_numpy() == 1),
        "pp_attempts": (aggressive & faced & pp).astype(float), "pp_faced": (faced & pp).astype(float),
        "pp_runs": np.where(legal & pp, score, 0.0), "pp_legal": (legal & pp).astype(float),
        "mid_runs": np.where(legal & mid, score, 0.0), "mid_legal": (legal & mid).astype(float),
    })
    return work.groupby(["season", "p_match", "team", "batted_first"]).sum().reset_index()


def bat_first_vs_chase_contrast(frame: pd.DataFrame, settings, seed: int, n_iter: int) -> dict:
    """Per season: each team's powerplay attempt share and gear-up when batting first
    and when chasing, the team gap (batting first minus chasing), the mean gap across
    teams with a match cluster bootstrap interval, and the change in the mean gap
    from the earlier to the later season."""
    earlier, later = settings.earlier_season, settings.later_season
    alpha = 1.0 - settings.interval_level
    sums = _team_innings_sums(frame, settings)
    teams = sorted(sums["team"].unique())
    t_ix = {t: j for j, t in enumerate(teams)}
    cols = ["pp_attempts", "pp_faced", "pp_runs", "pp_legal", "mid_runs", "mid_legal"]

    mats, matches = {}, {}
    for season in (earlier, later):
        s = sums[sums["season"] == season]
        matches[season] = np.sort(s["p_match"].unique())
        m_ix = {m: i for i, m in enumerate(matches[season])}
        arr = {order: np.zeros((len(matches[season]), len(teams), len(cols))) for order in (True, False)}
        for _, row in s.iterrows():
            arr[bool(row["batted_first"])][m_ix[row["p_match"]], t_ix[row["team"]], :] += row[cols].to_numpy(dtype=float)
        mats[season] = arr
    n_e, n_l = len(matches[earlier]), len(matches[later])
    counts = dict(zip((earlier, later), scb.shared_resample_counts(n_e, n_l, n_iter, seed)))

    def team_metrics(c, arr):
        tot = np.einsum("im,mtk->itk", c, arr)              # (n_iter, teams, cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            attempt = tot[:, :, 0] / tot[:, :, 1]
            gear = (tot[:, :, 2] / tot[:, :, 3] - tot[:, :, 4] / tot[:, :, 5]) * 6.0
        return attempt, gear

    results, per_team_rows, gap_draws = [], [], {}
    for season in (earlier, later):
        c = counts[season]
        att_bf, gear_bf = team_metrics(c, mats[season][True])
        att_ch, gear_ch = team_metrics(c, mats[season][False])
        att_bf0, gear_bf0 = team_metrics(np.ones((1, c.shape[1])), mats[season][True])
        att_ch0, gear_ch0 = team_metrics(np.ones((1, c.shape[1])), mats[season][False])
        gap_draws[season] = {"attempt_share": np.nanmean(att_bf - att_ch, axis=1),
                             "gear_up": np.nanmean(gear_bf - gear_ch, axis=1)}
        for metric, bf0, ch0, draws in (("attempt_share", att_bf0, att_ch0, gap_draws[season]["attempt_share"]),
                                        ("gear_up", gear_bf0, gear_ch0, gap_draws[season]["gear_up"])):
            results.append({"metric": metric, "season": season,
                            "batting_first_mean_over_teams": float(np.nanmean(bf0)),
                            "chasing_mean_over_teams": float(np.nanmean(ch0)),
                            "mean_team_gap": float(np.nanmean(bf0 - ch0)),
                            "gap_low": float(np.nanpercentile(draws, 100 * alpha / 2)),
                            "gap_high": float(np.nanpercentile(draws, 100 * (1 - alpha / 2))),
                            "p_gap_positive": float(np.nanmean(draws > 0))})
        for j, t in enumerate(teams):
            per_team_rows.append({"season": season, "team": t,
                                  "attempt_share_batting_first": float(att_bf0[0, j]), "attempt_share_chasing": float(att_ch0[0, j]),
                                  "gear_up_batting_first": float(gear_bf0[0, j]), "gear_up_chasing": float(gear_ch0[0, j])})
    for metric in ("attempt_share", "gear_up"):
        change = gap_draws[later][metric] - gap_draws[earlier][metric]
        e = next(r for r in results if r["metric"] == metric and r["season"] == earlier)
        l = next(r for r in results if r["metric"] == metric and r["season"] == later)
        results.append({"metric": metric, "season": f"{later} minus {earlier}",
                        "batting_first_mean_over_teams": np.nan, "chasing_mean_over_teams": np.nan,
                        "mean_team_gap": l["mean_team_gap"] - e["mean_team_gap"],
                        "gap_low": float(np.nanpercentile(change, 100 * alpha / 2)),
                        "gap_high": float(np.nanpercentile(change, 100 * (1 - alpha / 2))),
                        "p_gap_positive": float(np.nanmean(change > 0))})
    per_team = pd.DataFrame(per_team_rows)
    per_team["attempt_share_gap"] = per_team["attempt_share_batting_first"] - per_team["attempt_share_chasing"]
    per_team["gear_up_gap"] = per_team["gear_up_batting_first"] - per_team["gear_up_chasing"]
    return {"summary": pd.DataFrame(results), "per_team": per_team}


def death_scoring_by_swap(frame: pd.DataFrame, events: pd.DataFrame, settings) -> pd.DataFrame:
    """Batting-first innings only, per season: death-overs run rate for innings where
    the side made a batter-to-bowler swap during the innings against innings where it
    did not, side by side with the wickets down entering the death, since the swap
    happens because a batter fell and swap innings are selected on wickets. A
    descriptive table with counts; no interval, because the matched comparison
    rests on two or three dozen innings."""
    death = settings.phase("death")
    first = frame[frame["inns"] == 1]
    swapped = set(zip(events.loc[(events["innings"] == 1) & (events["side"] == "batted_first"), "match_id"],
                      events.loc[(events["innings"] == 1) & (events["side"] == "batted_first"), "team"].astype(str)))
    legal = dc.is_legal_delivery(first).to_numpy()
    in_death = scb.in_phase(first, death)
    score = pd.to_numeric(first["score"], errors="coerce").fillna(0).to_numpy(dtype=float)
    wickets = dc.is_wicket(first).astype(int).to_numpy()
    before_death = pd.to_numeric(first["over_num"], errors="coerce").to_numpy() < death[0]
    work = pd.DataFrame({"season": first["year"].to_numpy(), "p_match": first["p_match"].to_numpy(),
                         "team": first["team_bat"].astype(str).to_numpy(),
                         "death_runs": np.where(legal & in_death, score, 0.0), "death_legal": (legal & in_death).astype(float),
                         "wkts_before_death": (wickets * before_death).astype(float)})
    per = work.groupby(["season", "p_match", "team"]).sum().reset_index()
    per = per[per["death_legal"] > 0]
    per["swap_in_innings"] = [((m, t) in swapped) for m, t in zip(per["p_match"], per["team"])]
    per["death_run_rate"] = per["death_runs"] / per["death_legal"] * 6.0
    per["wkts_band"] = pd.cut(per["wkts_before_death"], [-1, 2, 4, 10], labels=["0-2", "3-4", "5+"])
    out = (per.groupby(["season", "wkts_band", "swap_in_innings"], observed=True)
           .agg(innings=("p_match", "size"), death_run_rate=("death_run_rate", "mean"),
                mean_wkts_before_death=("wkts_before_death", "mean")).reset_index())
    overall = (per.groupby(["season", "swap_in_innings"])
               .agg(innings=("p_match", "size"), death_run_rate=("death_run_rate", "mean"),
                    mean_wkts_before_death=("wkts_before_death", "mean")).reset_index())
    overall.insert(1, "wkts_band", "all")
    return pd.concat([overall, out], ignore_index=True)


def substitute_contribution(swaps: pd.DataFrame, frame: pd.DataFrame, events: pd.DataFrame, settings) -> pd.DataFrame:
    """For each batting-first in-innings swap: the incoming player's own batting in
    that innings (balls faced, runs, strike rate, balls faced at the death) beside
    the rest of the innings' batters after the entry ball. Says whether the added
    batter did the scoring or was a passenger; with a few dozen cases this is a
    table to read, not a test."""
    death_start = settings.phase("death")[0]
    first = frame[frame["inns"] == 1]
    faced = dc.is_ball_faced(first).to_numpy()
    batruns = pd.to_numeric(first["batruns"], errors="coerce").fillna(0).to_numpy(dtype=float)
    in_death = pd.to_numeric(first["over_num"], errors="coerce").to_numpy() >= death_start
    ball_id = pd.to_numeric(first["ball_id"], errors="coerce").to_numpy()
    p_bat = first["p_bat"].to_numpy()
    p_match = first["p_match"].to_numpy()
    in_ids = events.set_index(["match_id", "team", "innings", "deploy_ball_id"])["in_dataset_id"]
    rows = []
    for _, sw in swaps.iterrows():
        m = p_match == sw["match_id"]
        after = m & (ball_id >= sw["deploy_ball_id"])
        sub_id = in_ids.get((sw["match_id"], sw["team"], 1, sw["deploy_ball_id"]), np.nan)
        is_sub = p_bat == sub_id
        sub_faced, sub_runs = int((faced & is_sub & m).sum()), float(batruns[faced & is_sub & m].sum())
        sub_death = int((faced & is_sub & m & in_death).sum())
        others = after & faced & ~is_sub
        oth_faced, oth_runs = int(others.sum()), float(batruns[others].sum())
        rows.append({"season": sw["season"], "match_id": sw["match_id"], "team": sw["team"],
                     "player_in": sw["player_in"], "in_role": sw["in_role"], "deploy_over": sw["deploy_over"],
                     "sub_batted": sub_faced > 0, "sub_balls": sub_faced, "sub_runs": sub_runs,
                     "sub_strike_rate": 100 * sub_runs / sub_faced if sub_faced else np.nan,
                     "sub_death_balls": sub_death,
                     "others_balls_after_entry": oth_faced, "others_runs_after_entry": oth_runs,
                     "others_strike_rate_after_entry": 100 * oth_runs / oth_faced if oth_faced else np.nan})
    table = pd.DataFrame(rows)
    return table


def substitute_contribution_summary(contrib: pd.DataFrame) -> pd.DataFrame:
    g = contrib.groupby("season")
    return g.agg(swaps=("match_id", "size"), sub_batted_share=("sub_batted", "mean"),
                 sub_balls_total=("sub_balls", "sum"), sub_runs_total=("sub_runs", "sum"),
                 sub_death_balls_total=("sub_death_balls", "sum"),
                 others_balls_after_entry_total=("others_balls_after_entry", "sum"),
                 others_runs_after_entry_total=("others_runs_after_entry", "sum")).reset_index().assign(
        sub_strike_rate_pooled=lambda t: 100 * t["sub_runs_total"] / t["sub_balls_total"].replace(0, np.nan),
        others_strike_rate_pooled=lambda t: 100 * t["others_runs_after_entry_total"] / t["others_balls_after_entry_total"].replace(0, np.nan),
        sub_share_of_balls_after_entry=lambda t: t["sub_balls_total"] / (t["sub_balls_total"] + t["others_balls_after_entry_total"]))


def plot_team_gaps(per_team: pd.DataFrame, metric: str, unit: str, title: str):
    """Per team, the batting-first minus chasing gap in each season."""
    import matplotlib.pyplot as plt

    wide = per_team.pivot(index="team", columns="season", values=f"{metric}_gap")
    wide = wide.loc[wide.mean(axis=1).sort_values().index]
    y = np.arange(len(wide))
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(wide) + 1.5))
    colours = {wide.columns[0]: "#2c7fb8", wide.columns[-1]: "#d95f0e"}
    for season in wide.columns:
        ax.plot(wide[season], y, "o", color=colours.get(season, "grey"), label=str(season), ms=6)
    ax.axvline(0, color="#C0392B", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(wide.index, fontsize=9)
    ax.set_xlabel(f"batting first minus chasing ({unit})")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, title="season")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
