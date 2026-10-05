"""
Dismissal coherence: the out flag, the dismissal method, the dismissed-batter id,
the innings wicket count and the bowler's wicket tally all tell the same story.

Two conventions in this feed matter:
- `retired not out (hurt)` sets out=True but is not a dismissal; wicket counts
  exclude it.
- `p_out` is populated on every ball with the striker's id, not blank on non-wicket
  balls as the data dictionary states. `p_out_clean` restores the dictionary
  definition (populated only on genuine dismissals) and is reconciled three ways.
"""

from __future__ import annotations

import pandas as pd


BOWLER_CREDITED = {"caught", "bowled", "leg before wicket", "stumped", "hit wicket"}
# Retired-hurt sets out=True in this feed but is NOT a dismissal (batter departs, not out).
NON_DISMISSAL = {"retired not out (hurt)"}
KNOWN_DISMISSALS = BOWLER_CREDITED | {"run out", "retired out", "obstructing the field"} | NON_DISMISSAL

def check_wickets_dismissals(df):
    d = df.copy()
    if not d.index.is_unique:
        d = d.reset_index(drop=True)
    def to_bool(s):
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    d["out"] = to_bool(d["out"])
    d["bat_out"] = to_bool(d["bat_out"])
    d["dismissal_norm"] = d["dismissal"].astype(str).str.strip().str.lower()
    blanks = ["", "nan", "none", "-", "na", "<na>", "null"]
    d.loc[d["dismissal_norm"].isin(blanks), "dismissal_norm"] = pd.NA
    d["inns_wkts"] = pd.to_numeric(d["inns_wkts"], errors="coerce")
    d["cur_bowl_wkts"] = pd.to_numeric(d["cur_bowl_wkts"], errors="coerce")
    d = d.sort_values(["p_match", "inns", "ball_id"], kind="stable")
    def is_present(series):
        s = series.astype(str).str.strip().str.lower()
        return series.notna() & ~s.isin(blanks)
    # True wicket for counting = out flag set AND not a retired-hurt.
    d["is_wicket"] = (d["out"] == True) & ~d["dismissal_norm"].isin(NON_DISMISSAL)
    out = {}
    # out<->dismissal coherence per the DATA DICTIONARY —
    #   out=True  -> dismissal method AND p_out both present
    #   out=False -> dismissal method AND p_out both absent
    dis_present = is_present(d["dismissal"])
    pout_present = is_present(d["p_out"])
    out_coherence_bad = d[((d["out"] == True)  & (~dis_present | ~pout_present))
                          | ((d["out"] == False) & ( dis_present |  pout_present))]
    out["out_dismissal_coherence"] = out_coherence_bad
    # striker attribution + legitimate non-striker cases.
    striker_mismatch_bad = d[(d["bat_out"] == True)
                             & (d["p_out"].astype(str).str.strip().str.lower()
                                != d["p_bat"].astype(str).str.strip().str.lower())]
    non_striker_ok = {"run out", "obstructing the field", "retired out"}
    nonstriker_unexpected_bad = d[(d["out"] == True) & (d["bat_out"] == False)
                                  & ~d["dismissal_norm"].isin(non_striker_ok | NON_DISMISSAL)]
    out["striker_pout_mismatch"] = striker_mismatch_bad
    out["nonstriker_unexpected"] = nonstriker_unexpected_bad
    # reconcile using is_wicket (excludes retired-hurt), never > 10.
    d["cum_wkts"] = d.groupby(["p_match", "inns"], sort=False)["is_wicket"].transform(
        lambda s: s.astype(int).cumsum())
    wkts_reconcile_bad = d[d["cum_wkts"] != d["inns_wkts"]]
    wkts_over10_bad = d[d["inns_wkts"] > 10]
    out["wkts_reconcile"] = wkts_reconcile_bad
    out["wkts_over10"] = wkts_over10_bad
    # bowler credit only on credited dismissals.
    d["bowl_wkt_step"] = d.groupby(["p_match", "p_bowl"], sort=False)["cur_bowl_wkts"].diff()
    first_bowl = d.groupby(["p_match", "p_bowl"], sort=False).cumcount() == 0
    d.loc[first_bowl, "bowl_wkt_step"] = d.loc[first_bowl, "cur_bowl_wkts"]
    is_credited = (d["out"] == True) & d["dismissal_norm"].isin(BOWLER_CREDITED)
    bowler_wkt_credit_bad = d[d["bowl_wkt_step"].fillna(0) != is_credited.astype(int)]
    out["bowler_wkt_credit"] = bowler_wkt_credit_bad
    # vocabulary.
    seen = set(d["dismissal_norm"].dropna().unique())
    unknown_dismissals = sorted(seen - KNOWN_DISMISSALS)
    out["unknown_dismissals"] = unknown_dismissals
    def st(x): return "PASS" if len(x) == 0 else f"FAIL ({len(x)})"
    print(f"out<->dismissal coherence        : {st(out_coherence_bad)}")
    print(f"striker p_out mismatch           : {st(striker_mismatch_bad)}")
    print(f"unexpected non-striker dismissal : {st(nonstriker_unexpected_bad)}")
    print(f"inns_wkts reconciliation         : {st(wkts_reconcile_bad)}")
    print(f"inns_wkts > 10                   : {st(wkts_over10_bad)}")
    print(f"bowler-wkt credit                : {st(bowler_wkt_credit_bad)}")
    print(f"unknown dismissal terms          : {'PASS' if not unknown_dismissals else unknown_dismissals}")
    return out, d


def report_p_out_convention(df: pd.DataFrame) -> None:
    """p_out is never blank; off wicket balls it equals the striker id."""
    print(df["p_out"].isna().sum(), "true NaN in p_out;",
          (df["p_out"].astype(str).str.strip() == "").sum(), "empty-string in p_out")
    nonwkt = df[df["out"] != True]
    print(f"Non-wicket balls where p_out != p_bat: {int((nonwkt['p_out'] != nonwkt['p_bat']).sum())}")


def add_p_out_clean(df: pd.DataFrame) -> pd.DataFrame:
    """p_out restricted to genuine dismissals (out flag set and not retired hurt),
    which is the data dictionary's definition of the column."""
    d = df.copy()
    is_dismissal = (
        (d["out"].astype(str).str.strip().str.lower() == "true")
        & (d["dismissal"].astype(str).str.strip().str.lower() != "retired not out (hurt)")
    )
    d["p_out_clean"] = d["p_out"].where(is_dismissal)
    print(f"p_out_clean populated on {int(d['p_out_clean'].notna().sum())} balls "
          f"(should equal total genuine dismissals).")
    return d


def reconcile_p_out_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Per innings, populated p_out_clean values equal the final wicket count; and the
    three independent dismissal counts (p_out_clean, out flag excluding retired hurt,
    sum of final inns_wkts) agree."""
    per_inns = df.groupby(["p_match", "inns"]).agg(
        pout_filled=("p_out_clean", lambda s: s.notna().sum()),
        inns_wkts_final=("inns_wkts", "max"),
    )
    mismatch_pout = per_inns[per_inns["pout_filled"] != per_inns["inns_wkts_final"]]
    if mismatch_pout.empty:
        print(f"PASS: p_out_clean count matches final inns_wkts for all {len(per_inns)} innings.")
    else:
        print(f"FAIL: {len(mismatch_pout)} innings where p_out_clean count != final inns_wkts.")
        print(mismatch_pout.to_string())

    n_pout_clean = int(df["p_out_clean"].notna().sum())
    out_bool = df["out"].astype(str).str.strip().str.lower() == "true"
    retired_hurt = df["dismissal"].astype(str).str.strip().str.lower() == "retired not out (hurt)"
    n_out_flag = int((out_bool & ~retired_hurt).sum())
    n_inns_wkts = int(df.groupby(["p_match", "inns"])["inns_wkts"].max().sum())
    print(f"p_out_clean populated        : {n_pout_clean}")
    print(f"out flag (excl. retired-hurt): {n_out_flag}")
    print(f"sum of final inns_wkts       : {n_inns_wkts}")
    if n_pout_clean == n_out_flag == n_inns_wkts:
        print(f"PASS: all three independent counts agree at {n_pout_clean}.")
    else:
        print("FAIL: dismissal counts disagree; investigate")
    return mismatch_pout
