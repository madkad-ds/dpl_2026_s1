"""
Are the two seasons comparable in shape? Match and innings counts, team set and
match load, venue and host-country mix, toss and bat-first split, result types,
and the prevalence of rain-reduced innings. Anything that differed here would
confound a scoring or tactics comparison before any analysis began.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def comparability_groundwork(df):
    """Structural comparison of the two seasons, to establish that later tactical
    differences aren't artefacts of the seasons differing in shape. Checks match/innings
    counts, team participation, venue and host-country mix, toss/bat-first split, match
    result types (incl. rain/no-result), and innings length — anything that, if it
    differed between seasons, would confound a scoring or tactics comparison.
    """
    d = df.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")

    # One row per match, and one row per (match, innings), for counting.
    matches = d.drop_duplicates("p_match")[
        ["p_match","year","ground","country","winner","toss","max_balls"]]
    innings = d.drop_duplicates(["p_match","inns"])[["p_match","year","inns","team_bat"]]

    print("=== Match and innings volume ===")
    vol = pd.DataFrame({
        "matches": matches.groupby("year")["p_match"].nunique(),
        "innings": innings.groupby("year").size(),
        "deliveries": d.groupby("year").size(),
    })
    print(vol.to_string())

    print("\n=== Teams and matches per team ===")
    # A team appears as batting or bowling; count distinct teams and their match load.
    tb = d.groupby("year")["team_bat"].nunique().rename("distinct_teams")
    per_team = (d.drop_duplicates(["p_match","team_bat"])
                .groupby(["year","team_bat"])["p_match"].nunique()
                .groupby("year").agg(["mean","min","max"]).round(1))
    print(pd.concat([tb, per_team], axis=1).to_string())

    print("\n=== Venue and host-country mix ===")
    print("Distinct grounds per season:")
    print(matches.groupby("year")["ground"].nunique().to_string())
    print("\nDistinct host countries per season:")
    print(matches.groupby("year")["country"].nunique().to_string())
    # If venue mix differs a lot, scoring comparisons need venue control.

    print("\n=== Toss and bat-first split ===")
    # Did the team winning the toss choose to bat or bowl? Proxy: did toss-winner bat first?
    # (toss column = toss winner; inns-1 batting team = batted first.)
    first_bat = innings[innings["inns"]==1][["p_match","team_bat"]].rename(columns={"team_bat":"batted_first"})
    toss = matches[["p_match","year","toss"]].merge(first_bat, on="p_match", how="left")
    toss["toss_won_and_batted"] = toss["toss"] == toss["batted_first"]
    print("Fraction where toss-winner batted first:")
    print(toss.groupby("year")["toss_won_and_batted"].mean().round(3).to_string())

    print("\n=== Match result types ===")
    # Completed vs no-result (rain/abandoned). winner == '-' or blank => no result.
    matches["result_type"] = matches["winner"].astype(str).str.strip().apply(
        lambda w: "no_result" if w in {"-","","nan","no result","abandoned"} else "completed")
    print(pd.crosstab(matches["year"], matches["result_type"]).to_string())

    print("\n=== Innings length (reduced-overs prevalence) ===")
    # max_balls < 120 => rain/DLS-reduced innings. If more prevalent in one season, it
    # confounds scoring/rate comparisons.
    innings_full = d.drop_duplicates(["p_match","inns"])[["year","inns","max_balls"]]
    innings_full["max_balls"] = pd.to_numeric(innings_full["max_balls"], errors="coerce")
    innings_full["reduced"] = innings_full["max_balls"] < 120
    print("Fraction of innings reduced (<120 balls):")
    print(innings_full.groupby("year")["reduced"].mean().round(3).to_string())

    return {"volume": vol, "matches": matches}
