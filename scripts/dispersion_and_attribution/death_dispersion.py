"""
Death-overs dispersion on a population that a chase cannot truncate.

Why a population restriction. A chase ends when the target is reached, so the
second-innings death overs are censored by the opposition's total; early finishes
rose across seasons, which by itself thins and reshapes the death sample.
Conditioning on "played a complete death" does not fix this: it selects on the
outcome (close chases and first innings), which is a different selected sample,
not an unselected one.

Population used here:
  - first innings, full 120-ball allowance (no DLS reduction at any point);
  - not all out before the death began (all ten wickets down before over 16).
    All-out is also a selection, but rare and season-stable; the count per season
    is reported so that claim can be checked.
  - second innings: descriptive only. The raw between-team gear-up SD is reported
    with a "censored, not tested" label and no interval.

Computation on the retained first innings:
  1. gear-up delta per team-innings = death run rate minus that innings' middle
     overs run rate, formed inside the nested hierarchical model (both phases'
     runs and balls observed; the delta is a derived quantity);
  2. season-specific between-team SD of the gear-up; estimand log(sd_2024/sd_2023)
     with P(widen) and P(narrow), the SD difference alongside;
  3. refit under the Exponential prior.

The death-ball shortfall diagnosis is kept as the evidence that the missing death
balls are early chase finishes and weather, with an "unexplained" class that must be
near zero (a recording gap would bias the death estimate itself).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import innings_population as ip
from mean_level_and_tail import season_contrast_bootstrap as scb
from . import hierarchical_dispersion_models as hdm
from .team_spread_bootstrap import per_team_gear_up_delta


def diagnose_death_ball_shortfall(frame):
    """Attribute per-innings death-over ball counts to causes, to tell whether
    2024's thinner death sample is early finishes and weather (selection or
    structural) or an unexplained recording gap.

    For each (match, innings) it compares death balls present against death balls
    expected given max_balls, and classifies any shortfall:
    1. weather/DLS reduced: max_balls < 120, so the innings never had a full
       death to begin with (structural, not selection).
    2. chase ended early: second innings that reached the target before overs ran
       out (selection, the surviving-to-death teams are a filtered set).
    3. first innings all out before the death: innings terminated by wickets
       (selection on batting collapse).
    4. unexplained: balls missing with none of the above, a recording gap that
       would bias the death phase estimate itself.
    """
    death_lo, death_hi = scb.DEATH_OVERS
    rows = []
    for (match, inns), g in frame.groupby(["p_match", "inns"]):
        year = g["year"].iloc[0]
        max_balls = g["max_balls"].iloc[0]
        legal = scb.is_legal_delivery(g)
        total_legal = int(legal.sum())

        death_mask = scb.in_phase(g, scb.DEATH_OVERS) & scb.is_legal_delivery(g)
        death_present = int(death_mask.sum())

        # Death balls that should exist given the innings' own ball allowance:
        # everything past the end of the middle overs up to max_balls.
        middle_end_balls = (death_lo - 1) * 6           # legal balls before the death starts
        death_expected = max(0, min(max_balls, 120) - middle_end_balls)

        shortfall = death_expected - death_present
        target = g["target"].iloc[0] if "target" in g.columns else np.nan
        is_chase = (inns == 2) or (not np.isnan(target) and target > 0)

        # Did the innings reach its allowance, or stop short?
        reached_allowance = total_legal >= min(max_balls, 120) - 1  # 1-ball tolerance

        if max_balls < 119:
            cause = "weather_dls_reduced"
        elif shortfall <= 1:
            cause = "full_death"
        elif is_chase and not reached_allowance:
            cause = "chase_ended_early"          # target hit before overs ran out
        elif inns == 1 and not reached_allowance:
            cause = "first_innings_all_out"      # collapse before the death
        else:
            cause = "unexplained"

        rows.append(dict(year=year, p_match=match, inns=inns,
                         max_balls=int(max_balls), total_legal=total_legal,
                         death_present=death_present, death_expected=int(death_expected),
                         shortfall=int(max(0, shortfall)), cause=cause))

    return pd.DataFrame(rows)


def summarise_death_shortfall(shortfall_frame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Season breakdown of death-ball shortfall causes (innings and balls lost),
    plus the unexplained rows. The unexplained count is the one that must be near
    zero: selection and weather are interpretable, a recording gap is not."""
    by_cause = (shortfall_frame.groupby(["year", "cause"])
                .agg(innings=("p_match", "size"),
                     balls_lost=("shortfall", "sum"))
                .reset_index())
    unexplained = shortfall_frame[shortfall_frame["cause"] == "unexplained"]
    return by_cause, unexplained


def death_population(frame, settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retained first innings (full allowance, not all out before the death), the
    exclusion table by season, and the second-innings ball rows for the
    descriptive read."""
    tail = settings.section("tail")
    table = ip.build_innings_table(frame, full_innings_balls=tail["full_innings_balls"],
                                   death_first_over=settings.phase("death")[0])
    first_full = table[table["census"] == "first_innings"]
    retained = first_full[~first_full["all_out_before_death"]]

    exclusions = []
    for year, season in table.groupby("year"):
        exclusions.append({
            "year": int(year),
            "dls_reduced": int((season["census"] == "dls_reduced").sum()),
            "second_innings": int(season["census"].isin(["successful_chase", "failed_chase",
                                                          "second_innings_no_target"]).sum()),
            "first_innings_all_out_before_death": int(
                ((season["census"] == "first_innings") & season["all_out_before_death"]).sum()),
            "retained_first_innings": int(
                ((season["census"] == "first_innings") & ~season["all_out_before_death"]).sum()),
        })
    second_innings = frame[frame["inns"].astype(int) == 2]
    return (ip.restrict_frame_to_innings(frame, retained), pd.DataFrame(exclusions), second_innings)


def run_death_dispersion(frame, settings, seed: int) -> dict:
    """Shortfall diagnosis, population table, the nested gear-up model on the
    retained first innings under both priors, and the second-innings descriptive SD."""
    mcmc = settings.section("mcmc")
    death, middle = settings.phase("death"), settings.phase("middle")

    shortfall = diagnose_death_ball_shortfall(frame)
    by_cause, unexplained = summarise_death_shortfall(shortfall)

    retained_frame, exclusions, second_innings = death_population(frame, settings)

    model_rows, posteriors = [], {}
    for prior in ("halfnormal", "exponential"):
        idata, teams, _ = hdm.fit_gear_up_sd(retained_frame, death, middle, mcmc, seed, prior=prior)
        model_rows.append(hdm.sd_posterior_summary(idata, "gear_sd", "death_gear_up_first_innings",
                                                   prior, settings.interval_level))
        posteriors[prior] = idata

    _, raw_sd_first, depth_first = per_team_gear_up_delta(retained_frame, death, middle)
    _, raw_sd_second, depth_second = per_team_gear_up_delta(second_innings, death, middle)
    descriptive = pd.DataFrame([
        {"population": "first_innings_retained", "year": int(yr),
         "raw_gear_up_sd": float(raw_sd_first.get(yr, np.nan)),
         "min_team_death_balls": int(depth_first[("balls_phase", "min")].get(yr, 0)),
         "status": "tested (model rows)"}
        for yr in (settings.earlier_season, settings.later_season)
    ] + [
        {"population": "second_innings", "year": int(yr),
         "raw_gear_up_sd": float(raw_sd_second.get(yr, np.nan)),
         "min_team_death_balls": int(depth_second[("balls_phase", "min")].get(yr, 0)),
         "status": "censored by the target, not tested"}
        for yr in (settings.earlier_season, settings.later_season)
    ])

    return {"shortfall_by_cause": by_cause, "shortfall_unexplained": unexplained,
            "population_exclusions": exclusions, "model_summary": pd.DataFrame(model_rows),
            "descriptive_sd": descriptive, "posteriors": posteriors}
