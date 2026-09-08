"""
Per-innings population table and the censoring classification.

Why this exists. A chasing innings ends when the target is reached (a successful
chase) and so its total, its death overs, and its balls faced are cut short by the
opposition's score, not by the chasing side's scoring capacity. That is right
censoring at the target. A weather or DLS reduced innings is cut short by a different
external event. Any analysis of innings totals, death-overs behaviour, or over
profiles therefore needs a population that is not truncated by a chase, and needs to
report how many innings each exclusion removed. This module builds that table once
so every analysis classifies innings the same way.

Classification per innings, one label each, in this priority order:
  1. dls_reduced      : the innings ball allowance was below the full 120 at any point
                        (max_balls is reduced mid-innings when rain intervenes, so the
                        innings minimum of max_balls is the binding allowance).
  2. successful_chase : second innings, total reached the target. Censored.
  3. failed_chase     : second innings, total short of the target. Terminal, not
                        censored by the target (the innings ran out of balls or wickets).
  4. first_innings    : first innings with a full allowance. The primary population.

`all_out_before_death` is carried as a separate flag (not an exclusion) because the
death-dispersion population additionally excludes first innings that lost all ten
wickets before the death overs began; that count is reported per season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import derived_columns as dc

INNINGS_KEYS = ["p_match", "inns"]


def build_innings_table(frame: pd.DataFrame, full_innings_balls: int = 120,
                        death_first_over: int = 16) -> pd.DataFrame:
    """One row per (p_match, inns) with total, allowance, target, wickets, and the
    censoring class. Totals use `score` (runs including extras), the same numerator
    as the run-rate findings."""
    work = frame.copy()
    work["_score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0)
    work["_legal"] = dc.is_legal_delivery(work).astype(int)
    work["_out"] = dc.is_wicket(work).astype(int)
    work["_max_balls"] = pd.to_numeric(work["max_balls"], errors="coerce")
    work["_target"] = pd.to_numeric(work["target"], errors="coerce")
    # Wickets lost before the death began: count dismissals on balls in earlier overs.
    work["_out_before_death"] = work["_out"] * (pd.to_numeric(work["over_num"], errors="coerce") < death_first_over)

    grouped = work.groupby(INNINGS_KEYS, sort=True)
    table = grouped.agg(
        year=("year", "first"),
        team_bat=("team_bat", "first"),
        team_bowl=("team_bowl", "first"),
        total=("_score", "sum"),
        legal_balls=("_legal", "sum"),
        wickets=("_out", "sum"),
        wickets_before_death=("_out_before_death", "sum"),
        max_balls=("_max_balls", "min"),   # the binding allowance after any reduction
        target=("_target", "max"),
    ).reset_index()

    is_second = table["inns"].astype(int) == 2
    reached = is_second & table["target"].notna() & (table["total"] >= table["target"])
    short = is_second & table["target"].notna() & (table["total"] < table["target"])
    reduced = table["max_balls"] < full_innings_balls

    census = np.select(
        [reduced, reached, short, ~is_second],
        ["dls_reduced", "successful_chase", "failed_chase", "first_innings"],
        default="second_innings_no_target",   # a second innings with no target recorded; inspect
    )
    table["census"] = census
    table["all_out_before_death"] = table["wickets_before_death"] >= 10
    table["year"] = table["year"].astype(int)
    return table


def population_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Innings per season by census class, plus the all-out-before-death count among
    first innings, for the exclusion table every population-restricted analysis reports."""
    counts = (table.groupby(["year", "census"]).size()
              .unstack("census", fill_value=0).reset_index())
    first = table[table["census"] == "first_innings"]
    all_out = first.groupby("year")["all_out_before_death"].sum().rename("first_innings_all_out_before_death")
    return counts.merge(all_out.reset_index(), on="year", how="left").fillna({"first_innings_all_out_before_death": 0})


def uncensored_first_innings(table: pd.DataFrame) -> pd.DataFrame:
    """Primary population: first innings with the full ball allowance."""
    return table[table["census"] == "first_innings"].copy()


def terminal_innings(table: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity population: the primary population plus failed chases. A failed
    chase is terminal (the side was bowled out or ran out of balls), so its total
    reflects scoring capacity rather than the target."""
    return table[table["census"].isin(["first_innings", "failed_chase"])].copy()


def restrict_frame_to_innings(frame: pd.DataFrame, innings: pd.DataFrame) -> pd.DataFrame:
    """Ball-level rows belonging to the given innings."""
    keys = innings[INNINGS_KEYS].drop_duplicates()
    return frame.merge(keys, on=INNINGS_KEYS, how="inner")
