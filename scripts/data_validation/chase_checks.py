"""
Chase-column coherence for the second innings: target, runs remaining, required
rate and the winner all agree with the first-innings total and with each other.

Expected exceptions, resolved by inspection rather than treated as errors: a
rain-reduced chase carries a revised target, an abandoned match has no target and a
"-" winner, and a chase can succeed with the winner recorded under a different
spelling of the same team name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_chase_consistency(df, tol=0.01):
    d = df.copy()
    for c in ["inns_runs","inns_runs_rem","inns_balls","inns_balls_rem",
              "inns_rr","inns_rrr","target","max_balls"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    out = {}
    inns2 = d[d["inns"] == 2]
    inns1 = d[d["inns"] == 1]
    # chase columns populated in inns 2, blank in inns 1.
    chase_cols_inns2_missing = inns2[inns2[["target","inns_runs_rem","inns_rrr"]].isna().any(axis=1)]
    chase_cols_inns1_present = inns1[inns1[["target","inns_runs_rem","inns_rrr"]].notna().any(axis=1)]
    out["chase_cols_inns2_missing"] = chase_cols_inns2_missing
    out["chase_cols_inns1_present"] = chase_cols_inns1_present
    # target == (final inns_runs of inns 1) + 1, per match.
    first_tot = inns1.groupby("p_match")["inns_runs"].max().rename("first_innings_total")
    tgt = inns2.groupby("p_match")["target"].first().rename("target")
    link = pd.concat([first_tot, tgt], axis=1)
    link["expected_target"] = link["first_innings_total"] + 1
    target_link_bad = link[link["target"] != link["expected_target"]]
    out["target_link"] = target_link_bad
    # inns_runs_rem == max(target - inns_runs, 0) on every inns-2 ball.
    # Clamped at 0: when the winning runs come off a boundary, inns_runs overshoots target
    expected_rem = (inns2["target"] - inns2["inns_runs"]).clip(lower=0)
    runs_rem_identity_bad = inns2[inns2["inns_runs_rem"] != expected_rem]
    out["runs_rem_identity"] = runs_rem_identity_bad
    # recompute run rates and compare within tolerance, only where the rate is defined
    # RRR is undefined once balls remaining hits 0 (final ball of a completed chase) -> exclude,
    # else division by zero yields inf and floods the mismatch count.
    valid_rr  = inns2["inns_balls"] > 0
    valid_rrr = inns2["inns_balls_rem"] > 0
    rr_calc  = inns2.loc[valid_rr,  "inns_runs"]     / (inns2.loc[valid_rr,  "inns_balls"]     / 6)
    rrr_calc = inns2.loc[valid_rrr, "inns_runs_rem"] / (inns2.loc[valid_rrr, "inns_balls_rem"] / 6)
    rr_mismatch_bad  = inns2.loc[valid_rr][(rr_calc  - inns2.loc[valid_rr,  "inns_rr"]).abs()  > tol]
    rrr_mismatch_bad = (inns2.loc[valid_rrr][(rrr_calc - inns2.loc[valid_rrr, "inns_rrr"]).abs() > tol]
                        .dropna(subset=["inns_rrr"]))
    out["rr_mismatch"]  = rr_mismatch_bad
    out["rrr_mismatch"] = rrr_mismatch_bad
    # winner consistency — did the chase reach the target?
    chase_final = inns2.groupby("p_match").agg(
        chasing_team=("team_bat","first"),
        chase_runs=("inns_runs","max"),
        target=("target","first"),
        winner=("winner","first"))
    chase_final["chase_succeeded"] = chase_final["chase_runs"] >= chase_final["target"]
    chase_final["winner_is_chaser"] = chase_final["winner"] == chase_final["chasing_team"]
    winner_consistency_bad = chase_final[chase_final["chase_succeeded"] != chase_final["winner_is_chaser"]]
    out["winner_consistency"] = winner_consistency_bad
    def st(x): return "PASS" if len(x)==0 else f"FAIL ({len(x)})"
    for k,v in out.items(): print(f"{k:28s}: {st(v)}")
    return out


def diagnose_chase_consistency(df, chase, tol=0.01):
    d = df.copy()
    for c in ["inns_runs","inns_runs_rem","inns_balls","inns_balls_rem",
              "inns_rr","inns_rrr","target","max_balls"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    inns2 = d[d["inns"] == 2]
    # Per-match flags for the usual edge causes.
    max_balls_by_match = d.groupby("p_match")["max_balls"].min()   # <120 => reduced/DLS
    has_super_over = d.groupby("p_match")["inns"].max() > 2
    # ---- target link ----
    print("=== target-link failures ===")
    target_link = chase["target_link"]
    for pm in target_link.index:
        mb = max_balls_by_match.get(pm)
        print(f"  match {pm}: first_inns_total={target_link.loc[pm,'first_innings_total']:.0f}, "
              f"target={target_link.loc[pm,'target']:.0f}, expected={target_link.loc[pm,'expected_target']:.0f}, "
              f"max_balls={mb:.0f} -> {'DLS/reduced (target revised)' if mb<120 else 'INSPECT'}")
    # ---- winner consistency ----
    print("\n=== winner-consistency failures ===")
    winner_consistency = chase["winner_consistency"].copy()
    winner_consistency["super"] = winner_consistency.index.map(has_super_over) if winner_consistency.index.name=="p_match" else None
    print(winner_consistency.to_string())
    print("  (chase_succeeded != winner_is_chaser is EXPECTED for ties / super-over decisions)")


def match_target_and_winner(df: pd.DataFrame, p_match) -> tuple[pd.Series, pd.Series]:
    """Value counts of target and winner for one match (an abandoned match shows no
    target and a "-" winner)."""
    sub = df[df["p_match"] == p_match]
    return sub["target"].value_counts(), sub["winner"].value_counts()
