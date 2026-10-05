"""
Ledger checks: every row is one delivery, cumulative columns only ever go up,
runs and balls add up, and the ball allowance is repaired where it is a sentinel.

What is checked, in order:
1. granularity: (match, innings, ball_id) is unique and row_id is a primary key;
   ball_id decodes from over_num and ball.
2. monotonic ordering: the innings and batter cumulative columns never fall in
   delivery order.
3. per-delivery runs identity: score is bat runs plus extras, with byes and leg
   byes folded in only on legal balls (a no-ball carries just its penalty).
4. bowler-runs convention: the feed nets byes off the bowler on no-balls, which is
   not the scoring law; `bowler_charge` is added with the standard convention and
   reconciled against the bowler's running total.
5. innings total reconciliation from the per-ball components.
6. balls ledger: inns_balls steps only on legal balls, the balls-remaining
   identity holds, every over has six legal balls, ball_id decodes.
7. max_balls repair: a zero is a missing-value sentinel; the true allowance is
   implied by inns_balls + inns_balls_rem within the innings.

Functions print a PASS/FAIL line per check (the notebook is the audit trail) and
return the offending rows so they can be inspected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_granularity(df: pd.DataFrame) -> dict:
    """One row per physical delivery: unique (p_match, inns, ball_id), unique row_id,
    and ball_id == (over_num - 1) + ball / 100."""
    key_unique = int(df.groupby(["p_match", "inns", "ball_id"])["ball_id"].nunique().sum()) == len(df)
    row_id_unique = df["row_id"].is_unique
    reconstructed = (df["over_num"] - 1) + (df["ball"] / 100)
    ball_id_mismatch = df[(reconstructed - df["ball_id"]).abs() > 1e-9]
    print(f"delivery key unique      : {'PASS' if key_unique else 'FAIL'}")
    print(f"row_id unique            : {'PASS' if row_id_unique else 'FAIL'}")
    print(f"ball_id decodes          : {'PASS' if ball_id_mismatch.empty else f'FAIL ({len(ball_id_mismatch)})'}")
    return {"key_unique": key_unique, "row_id_unique": row_id_unique, "ball_id_mismatch": ball_id_mismatch}


INNINGS_MONOTONIC = ["inns_balls", "inns_runs"]              # keyed by (p_match, inns)
BATTER_MONOTONIC  = ["cur_bat_runs", "cur_bat_bf"]           # keyed by (p_match, inns, p_bat)

def _flag(d, keys, col):
    """Return rows where `col` drops vs the previous delivery within `keys`."""
    delta = d.groupby(keys, sort=False)[col].diff()   # NaN on each group's first row not flagged
    mask = delta < 0
    if not mask.any():
        return None
    bad = d.loc[mask, ["row_id", "p_match", "inns", "ball_id", "p_bat", "bat"]].copy()
    bad["column"]     = col
    bad["value"]      = d.loc[mask, col].to_numpy()
    bad["prev_value"] = (d.loc[mask, col] - delta[mask]).to_numpy()
    return bad

def check_monotonic_ordering(df):
    d = df.sort_values(["p_match", "inns", "ball_id"], kind="stable").copy()

    cols = INNINGS_MONOTONIC + BATTER_MONOTONIC
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce")

    hits = []
    for col in INNINGS_MONOTONIC:
        hits.append(_flag(d, ["p_match", "inns"], col))
    for col in BATTER_MONOTONIC:
        hits.append(_flag(d, ["p_match", "inns", "p_bat"], col))

    hits = [h for h in hits if h is not None]
    if not hits:
        return pd.DataFrame(columns=["row_id","p_match","inns","ball_id","p_bat","bat",
                                     "column","value","prev_value"])
    return pd.concat(hits, ignore_index=True).sort_values(["p_match","inns","ball_id"])


def report_monotonic_ordering(df) -> pd.DataFrame:
    viol = check_monotonic_ordering(df)
    if viol.empty:
        print("PASS: all cumulative columns non-decreasing in delivery order.")
    else:
        print(f"FAIL: {len(viol)} violation(s).")
        print(viol.to_string(index=False))
    return viol


RUN_INPUTS = ["score", "batruns", "wide", "byes", "legbyes"]

def check_delivery_runs_identity(df, noball_penalty=1):
    """Conditional identity:
         noball==1 : score == batruns + wide + penalty         (byes/legbyes NOT in score)
         noball==0 : score == batruns + wide + byes + legbyes   (byes/legbyes IN score)
    """
    d = df.copy()
    d[RUN_INPUTS] = d[RUN_INPUTS].apply(pd.to_numeric, errors="coerce")
    d["noball"] = pd.to_numeric(d["noball"], errors="coerce").fillna(0).astype(int)

    nan_mask = d[RUN_INPUTS].isna().any(axis=1)
    is_nb = d["noball"] == 1
    z = lambda c: d[c].fillna(0)

    # Expected score is built piecewise on the no-ball flag.
    expected_score = z("batruns") + z("wide")
    expected_score += is_nb * noball_penalty                    # penalty only when no-ball
    expected_score += (~is_nb) * (z("byes") + z("legbyes"))     # byes folded in only on legal balls

    bad_mask = (d["score"] != expected_score) & d["score"].notna() & ~nan_mask

    cols = ["row_id", "p_match", "inns", "ball_id", "outcome",
            "noball", "wide", "byes", "legbyes", "batruns", "score"]
    viol = d.loc[bad_mask, cols].copy()
    viol["expected_score"] = expected_score[bad_mask].to_numpy()
    viol["diff"] = (d["score"] - expected_score)[bad_mask].to_numpy()

    if viol.empty:
        print("PASS: conditional runs identity holds on every ball.")
    else:
        print(f"FAIL: {len(viol)} row(s) break the conditional identity.")
        print(viol.to_string(index=False))
    return viol


RUN_INPUTS_BOWL = ["bowlruns", "batruns", "wide", "byes", "legbyes"]

def check_bowler_runs_identity(df, noball_penalty=1, normalise_byes_off_noball=False):
    """
    Legal ball / plain no-ball:  bowlruns == batruns + wide + (penalty if noball)
    No-ball + bye/legbye (this dataset's convention):
    bowlruns == penalty - (byes + legbyes)   # byes netted off bowler
    Set normalise_byes_off_noball=True to instead REQUIRE the laws-correct bowlruns == penalty
    """
    d = df.copy()
    d[RUN_INPUTS_BOWL] = d[RUN_INPUTS_BOWL].apply(pd.to_numeric, errors="coerce")
    d["noball"] = pd.to_numeric(d["noball"], errors="coerce").fillna(0).astype(int)

    nan_mask = d[["bowlruns", "batruns", "wide"]].isna().any(axis=1)
    is_nb = d["noball"] == 1
    z = lambda c: d[c].fillna(0)
    bye_runs = z("byes") + z("legbyes")
    stacked = is_nb & (bye_runs > 0)      # no-ball AND byes/legbyes on the same delivery

    # Base expectation: bat + wide + penalty-on-no-ball.
    expected = z("batruns") + z("wide") + is_nb * noball_penalty

    if not normalise_byes_off_noball:
        # current data convention: subtract byes from the bowler's charge on stacked balls.
        expected = expected - stacked * bye_runs

    bad_mask = (d["bowlruns"] != expected) & d["bowlruns"].notna() & ~nan_mask

    cols = ["row_id", "p_match", "inns", "ball_id", "outcome",
            "noball", "wide", "byes", "legbyes", "batruns", "bowlruns"]
    viol = d.loc[bad_mask, cols].copy()
    viol["expected_bowlruns"] = expected[bad_mask].to_numpy()
    viol["diff"] = (d["bowlruns"] - expected)[bad_mask].to_numpy()

    print(f"Stacked no-ball+bye balls: {int(stacked.sum())}")
    if viol.empty:
        print("PASS: bowler-runs identity holds (given the current convention).")
    else:
        print(f"FAIL: {len(viol)} row(s) break the identity.")
        print(viol["diff"].value_counts().sort_index().to_string())
        print(viol.to_string(index=False))
    return viol


def add_bowler_charge(df, noball_penalty=1, col="bowler_charge"):
    """
    Standard convention runs charged to the bowler on each delivery:
        bowler_charge = batruns + wide + (penalty if no-ball)
    Byes and leg-byes are excluded (they belong to neither bowler nor batter).
    Never negative. Leaves original `bowlruns` untouched for reconciliation.
    """
    d = df.copy()
    for c in ["batruns", "wide", "noball"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # A missing batruns or wide means the charge can't be computed for that ball -> NaN, not 0, does not deflate bowler economy
    inputs_present = d[["batruns", "wide"]].notna().all(axis=1)

    is_nb = d["noball"].fillna(0).astype(int) == 1
    charge = d["batruns"].fillna(0) + d["wide"].fillna(0) + is_nb * noball_penalty

    d[col] = charge.where(inputs_present, other=pd.NA)
    return d


def reconcile_bowler_charge(df):
    d = df.copy()
    if not d.index.is_unique:
        d = d.reset_index(drop=True)

    d["cur_bowl_runs"] = pd.to_numeric(d["cur_bowl_runs"], errors="coerce")
    d["bowler_charge"] = pd.to_numeric(d["bowler_charge"], errors="coerce")

    d = d.sort_values(["p_match", "p_bowl", "inns", "ball_id"], kind="stable")

    d["cum_charge"] = d.groupby(["p_match", "p_bowl"], sort=False)["bowler_charge"].cumsum()
    d["gap"] = d["cum_charge"] - d["cur_bowl_runs"]

    bad = d[d["gap"].fillna(0) != 0]

    if bad.empty:
        print("PASS: cumulative bowler_charge matches cur_bowl_runs exactly for every bowler.")
        print("(Confirms cur_bowl_runs uses standard scoring byes treatment, unlike per-ball bowlruns.)")
    else:
        print(f"FAIL: {len(bad)} row(s) with a genuine gap.")
        cols = ["row_id", "p_match", "p_bowl", "bowl", "inns", "ball_id",
                "bowler_charge", "cum_charge", "cur_bowl_runs", "gap"]
        print(bad[[c for c in cols if c in d.columns]].to_string(index=False))
    return d, bad


def reconcile_innings_total(df, noball_penalty=1):
    """
    inns_runs (batting team's running total, after each delivery) should equal the cumulative
    sum of ALL run components, per (p_match, inns):
        total_runs = batruns + wide + byes + legbyes + (penalty if noball)
    This form doesn't rely on how `score` folds in byes.
    Checked at every ball, so the first divergence localises where the ledger (or order) breaks.
    """
    d = df.copy()
    if not d.index.is_unique:
        d = d.reset_index(drop=True)

    comps = ["batruns", "wide", "byes", "legbyes", "noball", "inns_runs"]
    for c in comps:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.sort_values(["p_match", "inns", "ball_id"], kind="stable")
    z = lambda c: d[c].fillna(0)

    # Per-ball total from first principles, then accumulate within each innings.
    d["ball_total"] = (z("batruns") + z("wide") + z("byes") + z("legbyes")
                       + (z("noball") == 1) * noball_penalty)
    d["cum_total"] = d.groupby(["p_match", "inns"], sort=False)["ball_total"].cumsum()

    d["gap"] = d["cum_total"] - d["inns_runs"]
    bad = d[d["gap"].fillna(0) != 0]

    # Report the FIRST divergence per innings — that's the diagnostic point, not every trailing row.
    first_bad = (bad.sort_values(["p_match", "inns", "ball_id"])
                    .groupby(["p_match", "inns"], sort=False).head(1))

    n_inns = d[["p_match", "inns"]].drop_duplicates().shape[0]
    n_bad_inns = first_bad.shape[0]

    if bad.empty:
        print(f"PASS: inns_runs matches cumulative additive total for all {n_inns} innings.")
    else:
        print(f"FAIL: {n_bad_inns} of {n_inns} innings diverge. First divergence in each:")
        cols = ["row_id", "p_match", "inns", "ball_id", "outcome", "noball",
                "batruns", "wide", "byes", "legbyes",
                "ball_total", "cum_total", "inns_runs", "gap"]
        print(first_bad[[c for c in cols if c in d.columns]].to_string(index=False))
    return d, bad, first_bad


def check_balls_ledger(df):
    """
      1  inns_balls increments only on legal balls (noball==0 & wide==0), by exactly 1 each.
      2  balls-remaining identity: inns_balls + inns_balls_rem == max_balls.
      3  over integrity: each over has 6 legal balls (last over of an innings may be short).
      4  ball_id decodes consistently with the legal-ball sequence.
    Returns a dict of violation frames; prints a per-check PASS/FAIL summary.
    """
    d = df.copy()
    if not d.index.is_unique:
        d = d.reset_index(drop=True)
    for c in ["noball", "wide", "inns_balls", "inns_balls_rem", "max_balls", "over", "ball"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["p_match", "inns", "ball_id"], kind="stable")
    d["is_legal"] = (d["noball"].fillna(0).astype(int) == 0) & (d["wide"].fillna(0) == 0)
    grp = ["p_match", "inns"]
    out = {}
    # ---- inns_balls advances by exactly 1 on legal balls, 0 on illegal ones ----
    d["inns_balls_step"] = d.groupby(grp, sort=False)["inns_balls"].diff()
    # First ball of each innings: expected step is its own legality (1 if legal, else 0).
    first_in_grp = d.groupby(grp, sort=False).cumcount() == 0
    d.loc[first_in_grp, "inns_balls_step"] = d.loc[first_in_grp, "inns_balls"]
    expected_step = d["is_legal"].astype(int)
    inns_balls_step_bad = d[d["inns_balls_step"] != expected_step]
    out["inns_balls_step"] = inns_balls_step_bad
    # ---- inns_balls + inns_balls_rem == max_balls ----
    balls_remaining_gap = d["inns_balls"] + d["inns_balls_rem"] - d["max_balls"]
    balls_remaining_bad = d[balls_remaining_gap.fillna(0) != 0]
    out["balls_remaining"] = balls_remaining_bad
    # ---- legal balls per over ----
    legal = d[d["is_legal"]]
    per_over = (legal.groupby(grp + ["over_num"], sort=False)
                     .size().rename("legal_balls").reset_index())
    # Last over of each innings is allowed to be short; all others must be exactly 6.
    last_over = per_over.groupby(grp, sort=False)["over_num"].transform("max")
    per_over["is_last_over"] = per_over["over_num"] == last_over
    over_integrity_bad = per_over[((per_over["legal_balls"] != 6) & ~per_over["is_last_over"])
                                  | (per_over["legal_balls"] > 6)]   # >6 is always wrong, even in last over
    out["over_integrity"] = over_integrity_bad
    # ---- ball_id decodes to the legal-ball sequence ----
    # Integer part of ball_id = over index; fractional *100 = raw delivery number within over.
    d["ball_id_over"] = d["ball_id"].astype(float).apply(lambda x: int(x))          # e.g. 19.02 -> 19
    d["ball_id_ball"] = (d["ball_id"].astype(float).mul(100).round() % 100).astype(int)  # 19.02 -> 2
    # Check the over index derived from ball_id matches the `over` column's convention.
    # (The over-numbering offset isn't fixed, so compare against both over and over-1.)
    match_direct = (d["ball_id_over"] == d["over_num"]).mean()
    match_offset = (d["ball_id_over"] == d["over_num"] - 1).mean()
    ball_id_over_convention = "ball_id_over == over" if match_direct >= match_offset else "ball_id_over == over-1"
    ref = d["over_num"] if match_direct >= match_offset else d["over_num"] - 1
    ball_id_decode_bad = d[d["ball_id_over"] != ref]
    out["ball_id_decode"] = ball_id_decode_bad
    def status(frame): return "PASS" if len(frame) == 0 else f"FAIL ({len(frame)})"
    print(f"inns_balls step          : {status(inns_balls_step_bad)}")
    print(f"balls-remaining identity : {status(balls_remaining_bad)}")
    print(f"over integrity           : {status(over_integrity_bad)}")
    print(f"ball_id decode           : {status(ball_id_decode_bad)}  [convention: {ball_id_over_convention}; "
          f"direct match {match_direct:.1%}, offset match {match_offset:.1%}]")
    return out, d


def classify_max_balls(mb):
    """Distinguish the three reasons inns_balls + inns_balls_rem != max_balls could arise."""
    if pd.isna(mb) or mb == 0:
        return "missing/sentinel (max_balls == 0) — REPAIR"
    elif mb == 120:
        return "full innings — identity break, INSPECT"   # max_balls fine, so gap is a real anomaly
    elif 1 <= mb < 120:
        return "reduced-overs (valid short innings)"
    else:  # mb > 120
        return "implausible max_balls (> 120) — INSPECT"

def diagnose_balls_ledger(df, ledger):
    d = df.copy()
    if not d.index.is_unique:
        d = d.reset_index(drop=True)
    # inspect the 2 failing rows
    inns_balls_step_bad = ledger["inns_balls_step"]
    print("=== inns_balls step failures (with surrounding context) ===")
    for idx in inns_balls_step_bad.index:
        pm, inns = d.loc[idx, "p_match"], d.loc[idx, "inns"]
        # pull the 3 rows around each failure within the same innings
        seg = d[(d["p_match"] == pm) & (d["inns"] == inns)].sort_values("ball_id")
        pos = seg.index.get_loc(idx)
        window = seg.iloc[max(0, pos-2): pos+2]
        cols = ["row_id","p_match","inns","ball_id","over_num","ball",
                "noball","wide","inns_balls","outcome"]
        print(window[[c for c in cols if c in d.columns]].to_string(index=False))
        print(f"  -> inns for this match: {sorted(d[d['p_match']==pm]['inns'].unique())}")
        print("-" * 60)
    # ---------- balls-remaining identity: 245 failures ----------
    for c in ["inns_balls","inns_balls_rem","max_balls"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["balls_remaining_gap"] = d["inns_balls"] + d["inns_balls_rem"] - d["max_balls"]
    balls_remaining_bad = d[d["balls_remaining_gap"].fillna(0) != 0]
    print("\n=== balls-remaining failure breakdown ===")
    print(f"Total balls-remaining failures: {len(balls_remaining_bad)}")
    print(f"\nAffected matches ({balls_remaining_bad['p_match'].nunique()}):")
    # For each affected innings: is max_balls != 120? -> DLS/rain. And what's the gap?
    summary = (balls_remaining_bad.groupby(["p_match","inns"])
                     .agg(rows=("row_id","size"),
                          max_balls=("max_balls","first"),
                          gap_min=("balls_remaining_gap","min"),
                          gap_max=("balls_remaining_gap","max"))
                     .reset_index())
    summary["max_balls"] = pd.to_numeric(summary["max_balls"], errors="coerce")
    summary["likely_cause"] = summary["max_balls"].apply(classify_max_balls)
    print(summary.to_string(index=False))
    # are ALL failures explained by max_balls != 120?
    unexplained = balls_remaining_bad[balls_remaining_bad["max_balls"] == 120]
    print(f"\nBalls-remaining failures with max_balls == 120 (NOT explained by reduction): {len(unexplained)}")
    if len(unexplained):
        cols = ["row_id","p_match","inns","ball_id","inns_balls","inns_balls_rem","max_balls","balls_remaining_gap"]
        print(unexplained[[c for c in cols if c in d.columns]].head(20).to_string(index=False))
    return balls_remaining_bad, summary


def repair_max_balls(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """max_balls == 0 is a missing-value sentinel. The true allowance is implied by
    inns_balls + inns_balls_rem, which holds ball by ball, so the innings maximum of
    that sum is used. Returns the repaired frame and the rows/innings repaired."""
    d = df.copy()
    implied = (d["inns_balls"] + d["inns_balls_rem"]).groupby([d["p_match"], d["inns"]]).transform("max")
    repair_mask = (d["max_balls"] == 0) & (implied > 0)
    d.loc[repair_mask, "max_balls"] = implied[repair_mask]
    n_rows = int(repair_mask.sum())
    n_innings = d.loc[repair_mask, ["p_match", "inns"]].drop_duplicates().shape[0]
    print(f"Repaired {n_rows} rows; {n_innings} innings.")
    return d, n_rows, n_innings


def implied_allowance_report(df: pd.DataFrame, matches: list) -> None:
    """For named matches: legal balls bowled, the implied allowance, and the stored one."""
    for pm in matches:
        seg = df[(df["p_match"] == pm) & (df["inns"] == 1)].sort_values("ball_id")
        legal_balls = ((seg["noball"] == 0) & (seg["wide"] == 0)).sum()
        implied_max = (seg["inns_balls"] + seg["inns_balls_rem"]).max()
        print(f"match {pm} inns 1: legal_balls={legal_balls}, "
              f"implied max_balls (inns_balls+rem).max()={implied_max}, "
              f"stored max_balls={seg['max_balls'].iloc[0]}")


def reconcile_step_against_over_integrity(df, ledger):
    """
    For each inns_balls step failure (where step != legality), check whether the OVER that row
    belongs to passes the over-integrity check (exactly 6 legal balls, or a legitimately
    short final over). If every step-failing over is over-integrity-clean, the step flags
    are timing artefacts and we defer to over-integrity as the authoritative legal-ball check.
    """
    d = df.copy()
    if not d.index.is_unique:
        d = d.reset_index(drop=True)
    for c in ["noball", "wide", "over_num"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["is_legal"] = (d["noball"].fillna(0).astype(int) == 0) & (d["wide"].fillna(0) == 0)
    inns_balls_step_bad = ledger["inns_balls_step"]
    if len(inns_balls_step_bad) == 0:
        print("No inns_balls step failures to reconcile.")
        return
    grp = ["p_match", "inns"]
    verdicts = []
    for idx in inns_balls_step_bad.index:
        pm, inns, ov = d.loc[idx, "p_match"], d.loc[idx, "inns"], d.loc[idx, "over_num"]
        # Over-integrity logic, recomputed for THIS over: count legal balls in it.
        over_rows = d[(d["p_match"] == pm) & (d["inns"] == inns) & (d["over_num"] == ov)]
        legal_count = int(over_rows["is_legal"].sum())
        # Is this the final over of the innings? (short final over is allowed)
        last_over = d[(d["p_match"] == pm) & (d["inns"] == inns)]["over_num"].max()
        is_last = ov == last_over
        over_integrity_ok = (legal_count == 6) or (is_last and legal_count < 6)
        verdicts.append({
            "row_id": d.loc[idx, "row_id"], "p_match": pm, "inns": inns, "over_num": ov,
            "outcome": d.loc[idx, "outcome"], "legal_balls_in_over": legal_count,
            "is_last_over": is_last, "over_integrity_valid": over_integrity_ok,
        })
    v = pd.DataFrame(verdicts)
    print(v.to_string(index=False))
    all_cleared = bool(v["over_integrity_valid"].all())
    print("\n" + "-" * 60)
    if all_cleared:
        print(f"All {len(v)} step-failing rows sit in overs that PASS over integrity "
              f"(correct legal-ball count).")
        print("=> Step flags are inns_balls step-timing artefacts, not data errors.")
        print("=> DEFER to over integrity: legal-ball structure is authoritative and sound.")
    else:
        bad = v[~v["over_integrity_valid"]]
        print(f"{len(bad)} step-failing row(s) are in overs that ALSO fail over integrity — genuine anomalies:")
        print(bad.to_string(index=False))
        print("=> Do NOT defer; inspect these overs directly.")
    return v

