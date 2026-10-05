"""
Behaviour of the two model columns, win probability (wprob) and projected score
(predscore): where the extreme values sit, whether the last ball converges on the
result, whether the value is stated before or after the ball, how well calibrated
it is, and whether it knows anything about team strength.

Why it matters downstream: a win-probability delta is a valid per-ball impact
measure only if wprob is stated after the ball; and a team-agnostic model
under-credits players on strong teams, so player impact should be read against a
team baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extreme_wprob_report(df: pd.DataFrame) -> pd.DataFrame:
    """The 0 and 100 values should sit late in chases at extreme states."""
    w = pd.to_numeric(df["wprob"], errors="coerce")
    extreme = df[(w == 0) | (w == 100)].copy()
    print("Innings phase of extreme wprob values (over_num):")
    print(extreme["over_num"].describe()[["min", "25%", "50%", "mean", "max"]].round(1).to_string())
    print("\nInnings distribution (should skew to 2nd innings / death overs):")
    print(extreme["inns"].value_counts().to_string())
    print("\nWickets down at these points (should skew high for the 0s):")
    print(pd.to_numeric(extreme["inns_wkts"], errors="coerce").describe()[["min", "50%", "mean", "max"]].round(1).to_string())
    return extreme


def check_terminal_convergence(df):
    d = df.copy()
    for c in ["predscore","wprob","inns_runs","inns_balls","inns_balls_rem","over_num"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["p_match","inns","ball_id"], kind="stable")
    # --- predscore at the LAST ball of an innings should equal final inns_runs ---
    last_ball = d.groupby(["p_match","inns"], sort=False).tail(1)
    pred_gap = (last_ball["predscore"] - last_ball["inns_runs"])
    print("predscore vs actual final inns_runs (at last ball of innings):")
    print(f"    mean abs gap: {pred_gap.abs().mean():.2f}, "
          f"median: {pred_gap.abs().median():.2f}, max: {pred_gap.abs().max():.0f}")
    print(f"    exact matches: {(pred_gap==0).sum()} of {len(last_ball)} innings")
    print(f"    within 5 runs: {(pred_gap.abs()<=5).mean():.1%}")
    # --- wprob at match end should be ~100 for winner's perspective, ~0 for loser ---
    # wprob is the BATTING team's win prob. Last ball of the 2nd innings:
    #   chase succeeded -> batting team won -> wprob should be ~100
    #   chase failed    -> batting team lost -> wprob should be ~0
    inns2_last = d[d["inns"]==2].groupby("p_match", sort=False).tail(1).copy()
    inns2_last["chase_won"] = inns2_last["inns_runs"] >= pd.to_numeric(
        df.loc[inns2_last.index, "target"], errors="coerce")
    won  = inns2_last[inns2_last["chase_won"]]
    lost = inns2_last[~inns2_last["chase_won"]]
    print("\nwprob at final ball of chase:")
    print(f"    chase WON  ({len(won)}): mean wprob {won['wprob'].mean():.1f} (want ~100), "
          f"min {won['wprob'].min():.0f}")
    print(f"    chase LOST ({len(lost)}): mean wprob {lost['wprob'].mean():.1f} (want ~0), "
          f"max {lost['wprob'].max():.0f}")
    return last_ball, inns2_last


def terminal_predscore_gaps(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """Innings whose last-ball projected score is far from the final total, with the
    likely causes (innings ended early; reduced allowance)."""
    d = df.copy()
    for colm in ["predscore", "inns_runs", "inns_balls", "inns_balls_rem", "max_balls", "over_num", "wprob"]:
        d[colm] = pd.to_numeric(d[colm], errors="coerce")
    d = d.sort_values(["p_match", "inns", "ball_id"], kind="stable")
    last = d.groupby(["p_match", "inns"], sort=False).tail(1).copy()
    last["gap"] = (last["predscore"] - last["inns_runs"]).abs()
    bad = last[last["gap"] > threshold].sort_values("gap", ascending=False)
    print(f"Innings with terminal predscore gap > {threshold:g}: {len(bad)}")
    cols = ["p_match", "inns", "over_num", "inns_balls", "inns_balls_rem", "max_balls",
            "inns_runs", "predscore", "gap", "wprob"]
    print(bad[cols].head(20).to_string(index=False))
    print("\nInnings ended EARLY (balls_rem > 0 at last ball -> all out or chase won early):",
          int((bad["inns_balls_rem"] > 0).sum()))
    print("Reduced/DLS innings (max_balls < 120):", int((bad["max_balls"] < 120).sum()))
    return bad


def check_wprob_timing(df):
    d = df.copy()
    for c in ["wprob","batruns","over_num"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["out_bool"] = d["out"].astype(str).str.strip().str.lower() == "true"
    d = d.sort_values(["p_match","inns","ball_id"], kind="stable")
    grp = ["p_match","inns"]
    # Delta = change in wprob from previous ball to this ball, within an innings.
    d["wprob_delta"] = d.groupby(grp, sort=False)["wprob"].diff()
    # Event flags for THIS row.
    d["is_boundary"] = d["batruns"].isin([4,6])
    d["is_wicket"]   = d["out_bool"]
    # The SAME event, shifted to the NEXT row (i.e. "did a boundary/wicket happen on the
    # PREVIOUS ball?"). If wprob is stated BEFORE the ball, this ball's delta reflects the
    # PREVIOUS ball's outcome.
    d["prev_boundary"] = d.groupby(grp, sort=False)["is_boundary"].shift(1)
    d["prev_wicket"]   = d.groupby(grp, sort=False)["is_wicket"].shift(1)
    # --- Batting-team perspective: a boundary should RAISE wprob, a wicket should LOWER it ---
    # Mean wprob_delta conditioned on where the event sits tells us the timing.
    def mean_delta(mask):
        s = d.loc[mask, "wprob_delta"].dropna()
        return (round(s.mean(), 2), len(s))
    print("Mean wprob change (delta) by event position:\n")
    print("If AFTER-the-ball: delta correlates with THIS row's event.")
    print("If BEFORE-the-ball: delta correlates with the PREVIOUS row's event.\n")
    print(f"  boundary on THIS ball   -> delta {mean_delta(d['is_boundary'])}   (want +ve if AFTER)")
    print(f"  boundary on PREV ball   -> delta {mean_delta(d['prev_boundary']==True)}   (want +ve if BEFORE)")
    print(f"  wicket on THIS ball     -> delta {mean_delta(d['is_wicket'])}   (want -ve if AFTER)")
    print(f"  wicket on PREV ball     -> delta {mean_delta(d['prev_wicket']==True)}   (want -ve if BEFORE)")
    # Baseline: a dot ball / no-event should have delta near 0 either way.
    neutral = ~d["is_boundary"] & ~d["is_wicket"] & (d["prev_boundary"]!=True) & (d["prev_wicket"]!=True)
    print(f"\n  neutral balls (no event here or prev) -> delta {mean_delta(neutral)}   (want ~0)")
    return d


def check_wprob_calibration(df, n_bins=10, min_over=None, inns_filter=None):
    d = df.copy()
    d["wprob"] = pd.to_numeric(d["wprob"], errors="coerce")

    # --- Actual outcome: did THIS ball's batting team win the match? ---
    # Requires a decided match. Filter out no-results (winner is '-'/blank).
    d["winner"] = d["winner"].astype(str).str.strip()
    decided = ~d["winner"].isin(["-", "", "nan", "no result", "abandoned"])
    d = d[decided].copy()

    # batting_won = did the team batting on this ball win? (winner == team_bat for this row)
    d["batting_won"] = (d["team_bat"].astype(str).str.strip() == d["winner"]).astype(int)

    # Optional scoping: calibration is cleanest where wprob has real spread (2nd innings, later overs).
    if inns_filter is not None:
        d = d[d["inns"] == inns_filter]
    if min_over is not None:
        d = d[pd.to_numeric(d["over_num"], errors="coerce") >= min_over]

    d = d.dropna(subset=["wprob"])
    # wprob is a percentage; convert to proportion for clarity.
    p = d["wprob"] / 100.0

    # --- Bin into deciles and compare predicted vs actual win rate ---
    bins = np.linspace(0, 1, n_bins + 1)
    d["bin"] = pd.cut(p, bins, include_lowest=True)
    rel = d.groupby("bin", observed=True).agg(
        n=("batting_won", "size"),
        predicted=("wprob", lambda s: s.mean() / 100.0),
        actual=("batting_won", "mean"),
    )
    rel["gap"] = (rel["actual"] - rel["predicted"]).round(3)
    rel["predicted"] = rel["predicted"].round(3)
    rel["actual"] = rel["actual"].round(3)

    print(f"Reliability diagram ({len(d)} balls, decided matches"
          + (f", inns={inns_filter}" if inns_filter else "")
          + (f", over>={min_over}" if min_over else "") + "):\n")
    print(rel.to_string())

    # Expected Calibration Error: average |actual - predicted| weighted by bin size.
    ece = (rel["n"] * (rel["actual"] - rel["predicted"]).abs()).sum() / rel["n"].sum()
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}  (lower = better)")
    print("'actual' should track 'predicted' down the table; 'gap' near 0 in every bin.")
    return rel

# Run on all decided-match balls first, then optionally the 2nd-innings/death subset.


def check_wprob_team_awareness(df, by_year=False, wprob_band=(40, 60), min_balls=30):
    """
    Test whether wprob conditions on team strength or is average-vs-average (agnostic).
    Logic: hold wprob ~constant (mid-range chase states); if predicted is flat across
    strength groups but ACTUAL win rate differs, the model is blind to team strength.

    by_year=True runs the test separately per season, recomputing strength WITHIN each
    year (a team's 2023 form != its 2024 form). Also reports win-rate dispersion per
    subset so a null result can be read as 'no strength effect' vs 'proxy too weak'.
    """
    d = df.copy()
    d["wprob"] = pd.to_numeric(d["wprob"], errors="coerce")
    d["winner"] = d["winner"].astype(str).str.strip()
    d = d[~d["winner"].isin(["-", "", "nan", "no result", "abandoned"])].copy()
    d["batting_won"] = (d["team_bat"].astype(str).str.strip() == d["winner"]).astype(int)
    lo, hi = wprob_band

    def _team_strength(sub):
        teams = pd.unique(sub[["team_bat", "team_bowl"]].values.ravel())
        s = {}
        for t in teams:
            played = sub[(sub["team_bat"] == t) | (sub["team_bowl"] == t)]["p_match"].nunique()
            won = sub[sub["winner"] == t]["p_match"].nunique()
            s[t] = won / played if played else np.nan
        return pd.Series(s)

    def _run(sub, label):
        strength = _team_strength(sub)

        # --- strength dispersion (parity context) ---
        disp_std = strength.std()
        disp_min, disp_max = strength.min(), strength.max()

        s = sub.copy()
        s["bat_strength"] = s["team_bat"].map(strength)
        mid = s[(s["inns"] == 2) & (s["wprob"].between(lo, hi))].copy()

        print(f"\n{'='*64}\n{label}")
        # Full ranked team strength, with the median marked so you can see the split point.
        med = strength.median()
        ranked = strength.sort_values(ascending=False)
        print(f"  team strength (win rate), median={med:.3f}:")
        for t, v in ranked.items():
            side = "strong" if v > med else "weak"
            print(f"    {v:.3f}  {side:6s}  {t}")
        print(f"  dispersion: std={disp_std:.3f}, range=[{disp_min:.2f}, {disp_max:.2f}]  "
              f"({len(strength)} teams)")

        if len(mid) < min_balls:
            print(f"  mid-range chase balls: {len(mid)} (< {min_balls}) — INSUFFICIENT, skipping test")
            return {"label": label, "n": len(mid), "verdict": "insufficient",
                    "disp_std": disp_std}

        mid["strong_bat"] = mid["bat_strength"] > mid["bat_strength"].median()

        grp = mid.groupby("strong_bat").agg(
                n=("batting_won", "size"),
                mean_wprob=("wprob", lambda x: round(x.mean() / 100, 3)),
                actual_win=("batting_won", lambda x: round(x.mean(), 3)),
            )
        grp.index = grp.index.map({False: "weak_bat", True: "strong_bat"})

        diff_pred = abs(grp["mean_wprob"].diff().iloc[-1])
        diff_actual = abs(grp["actual_win"].diff().iloc[-1])

        print(f"  mid-range chase balls (wprob {lo}-{hi}, inns 2): {len(mid)}")
        print(grp.to_string())
        print(f"  predicted differs by strength: {diff_pred:.3f}")
        print(f"  ACTUAL win rate differs by strength: {diff_actual:.3f}")

        agnostic = (diff_pred < 0.02) and (diff_actual > 0.05)
        if agnostic:
            verdict = "TEAM-AGNOSTIC (flat predicted, actual varies)"
        elif diff_pred < 0.02 and diff_actual <= 0.05:
            if disp_std < 0.08:
                verdict = "NO STRENGTH EFFECT (genuine parity — little to detect)"
            else:
                verdict = ("NO EFFECT DETECTED despite strength dispersion "
                           "— likely PROXY/SAMPLING noise, not team-awareness")
        else:
            verdict = "TEAM-AWARE / unclear (predicted tracks strength)"
        print(f"  -> {verdict}")

        return {"label": label, "n": len(mid), "diff_pred": diff_pred,
                "diff_actual": diff_actual, "disp_std": disp_std, "verdict": verdict}

    results = []
    if by_year:
        d["year"] = pd.to_numeric(d["year"], errors="coerce")
        for yr in sorted(d["year"].dropna().unique()):
            results.append(_run(d[d["year"] == yr], f"YEAR {int(yr)}"))
        # cross-year synthesis
        print(f"\n{'='*64}\nSYNTHESIS")
        valid = [r for r in results if r.get("verdict") != "insufficient"]
        if len(valid) > 1:
            preds = [r["diff_pred"] for r in valid]
            acts = [r["diff_actual"] for r in valid]
            disps = [r["disp_std"] for r in valid]
            print(f"  predicted-diff across years: {[round(p,3) for p in preds]} "
                  f"(all <0.02 => agnostic architecture confirmed)"
                  if all(p < 0.02 for p in preds) else
                  f"  predicted-diff across years: {[round(p,3) for p in preds]}")
            print(f"  actual-diff across years   : {[round(a,3) for a in acts]} "
                  f"(varies => bias magnitude is season-dependent / proxy-noisy)")
            print(f"  dispersion across years     : {[round(dd,3) for dd in disps]} "
                  f"(similar => parity comparable; magnitude differences are proxy/sampling)")
    else:
        results.append(_run(d, "ALL YEARS"))

    return results

# Pooled:

