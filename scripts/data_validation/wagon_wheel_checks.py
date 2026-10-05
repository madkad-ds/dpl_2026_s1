"""
Wagon-wheel (shot placement) geometry: are the eight zones distinct angular
sectors, is the origin the striker, are coordinates already batter-relative, and
what does (0, 0) mean.

Findings this module encodes: (0, 0) is a missing-coordinate sentinel, not a
position (it sits on dot balls, wides, extras, wickets, and a handful of scoring
balls); zone 3 straddles the 0/360 degree seam, so angular statistics must be
circular; LHB and RHB zone distributions are aligned, so no reflection is needed
before pooling. `wagonzone_clean` fills a missing zone from real coordinates where
the ball was hit, and `has_placement` flags balls with genuine coordinates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_wagon_geometry(df):
    d = df.copy()
    for c in ["wagonx","wagony","wagonzone","batruns","ballfaced"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    out = {}

    has_coord = d["wagonx"].notna() & d["wagony"].notna() & ~((d["wagonx"]==0) & (d["wagony"]==0))
    has_zone  = d["wagonzone"].fillna(0) > 0

    # ---- placement present where scored; absent on no-contact balls ----
    scored = d["batruns"] > 0
    out["scored_no_zone"] = d[scored & ~has_zone]
    out["nocontact_zone"] = d[(d["shot"].astype(str).str.upper()=="LEFT_ALONE") & has_zone]

    # ---- Origin: range-midpoint ----
    # The wheel centre is a fixed geometric point (the striker). Median-of-hits pulls the
    # "centre" toward wherever batters score most (leg side), skewing every derived angle.
    # The midpoint of the coordinate RANGE recovers the image/geometry centre instead.
    ox = (d.loc[has_coord,"wagonx"].min() + d.loc[has_coord,"wagonx"].max()) / 2
    oy = (d.loc[has_coord,"wagony"].min() + d.loc[has_coord,"wagony"].max()) / 2
    print(f"Origin (range-midpoint): x={ox:.1f}, y={oy:.1f}")

    deg = (np.degrees(np.arctan2(d["wagony"]-oy, d["wagonx"]-ox)) + 360) % 360
    d["_deg"] = deg
    d["_rad"] = np.radians(deg)

    # ---- per-zone LINEAR vs CIRCULAR stats, with a verdict per zone ----
    def zone_stats(g):
        lin_mean, lin_std = g["_deg"].mean(), g["_deg"].std()
        a = g["_rad"].dropna()
        C, S = np.cos(a).mean(), np.sin(a).mean()
        R = np.hypot(C, S)                                   # 0..1: hit concentration
        circ_mean = (np.degrees(np.arctan2(S, C)) + 360) % 360
        circ_std  = np.degrees(np.sqrt(-2*np.log(R))) if R > 0 else np.nan
        # Verdict: if linear std is large but circular std is small, the zone straddles the
        # 0/360 seam and the LINEAR figure is the artefact — the zone is really tight.
        if pd.notna(circ_std) and lin_std > 45 and circ_std < 25:
            verdict = "TIGHT (linear inflated by 0/360 wraparound)"
        elif pd.notna(circ_std) and circ_std < 25:
            verdict = "TIGHT"
        else:
            verdict = "LOOSE (genuine spread — inspect)"
        return pd.Series({"count": len(g),
                          "lin_mean": round(lin_mean,1), "lin_std": round(lin_std,1),
                          "circ_mean": round(circ_mean,1), "circ_std": round(circ_std,1),
                          "R": round(R,3), "verdict": verdict})

    zsummary = (d[has_coord & has_zone].groupby("wagonzone").apply(zone_stats))
    print("\nper-zone linear vs circular angle stats:")
    print(zsummary.to_string())
    out["zone_stats"] = zsummary

    n_wrap = (zsummary["verdict"].str.contains("wraparound")).sum()
    n_loose = (zsummary["verdict"].str.startswith("LOOSE")).sum()
    print(f"\n  -> {n_wrap} zone(s) tight but inflated by the 0/360 seam (use circular stats).")
    print(f"  -> {n_loose} zone(s) with genuine loose spread.")
    if n_loose == 0:
        print("  -> CONCLUSION: all zones are distinct tight sectors; geometry valid. "
              "Always compute angular stats CIRCULARLY.")
    else:
        print("  -> CONCLUSION: some zones genuinely dispersed — check origin or zone definitions.")

    # ---- handedness — raw (mirrored) vs already-normalised ----
    ct = (d[has_zone].groupby(["bat_hand","wagonzone"]).size()
          .groupby(level=0).transform(lambda s: s/s.sum()).round(3)
          .rename("frac").reset_index())
    pivot = ct.pivot(index="wagonzone", columns="bat_hand", values="frac")
    # Correlate LHB vs RHB zone distributions. High positive => aligned (normalised).
    # Mirror image would show as anti-correlation once zones are reversed.
    if {"LHB","RHB"}.issubset(pivot.columns):
        corr_aligned = pivot["LHB"].corr(pivot["RHB"])
        rhb_mirrored = pd.Series(pivot["RHB"].to_numpy()[::-1], index=pivot.index)
        corr_mirror  = pivot["LHB"].corr(rhb_mirrored)
        print(f"\nhandedness: LHB~RHB corr aligned={corr_aligned:.2f}, "
              f"mirrored={corr_mirror:.2f}")
        if corr_aligned > 0.7:
            print("  -> NORMALISED: zones are batter-relative; pool LHB/RHB directly, no reflection.")
        elif corr_mirror > 0.7:
            print("  -> RAW: coordinates are absolute; reflect one handedness before pooling.")
        else:
            print("  -> INCONCLUSIVE: inspect zone distributions directly.")
        print(pivot.to_string())
    out["hand_zone"] = pivot

    def st(x): return "PASS" if len(x)==0 else f"FAIL ({len(x)})"
    print(f"\nscored-but-no-zone  : {st(out['scored_no_zone'])}")
    print(f"no-contact-with-zone: {st(out['nocontact_zone'])}")
    return out


def diagnose_scored_without_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Balls that scored off the bat yet carry zone 0: what kind of balls, what
    shots, and whether coordinates are absent too (they are not; they are (0, 0))."""
    d = df.copy()
    for column in ["batruns", "wagonzone", "byes", "legbyes", "score"]:
        d[column] = pd.to_numeric(d[column], errors="coerce")
    sub = d[(d["batruns"] > 0) & (d["wagonzone"].fillna(0) == 0)]
    print(f"scored-but-no-zone: {len(sub)}")
    print("\nby outcome:");   print(sub["outcome"].value_counts().to_string())
    print("\nby shot:");      print(sub["shot"].value_counts(dropna=False).head(10).to_string())
    print("\nbatruns distribution:"); print(sub["batruns"].value_counts().sort_index().to_string())
    print("\ncoords also missing?:", f"{sub['wagonx'].isna().sum()} of {len(sub)} have no wagonx either")
    return sub


def diagnose_left_alone_with_zone(df: pd.DataFrame) -> pd.DataFrame:
    """A genuine leave scores nothing and has no placement; a leave with a zone and
    runs is a shot-tag contradiction, sized here."""
    d = df.copy()
    d["batruns"] = pd.to_numeric(d["batruns"], errors="coerce")
    d["wagonzone"] = pd.to_numeric(d["wagonzone"], errors="coerce")
    sub2 = d[(d["shot"].astype(str).str.upper() == "LEFT_ALONE") & (d["wagonzone"].fillna(0) > 0)]
    print(f"left-alone but has zone: {len(sub2)}")
    print("\nby outcome:");  print(sub2["outcome"].value_counts().to_string())
    print("\nbatruns on these:"); print(sub2["batruns"].value_counts().sort_index().to_string())
    print("\ncontrol values:");   print(sub2["control"].value_counts(dropna=False).to_string())
    print("\nleft-alone WITH runs (contradiction):", int((sub2["batruns"] > 0).sum()))
    return sub2


# Validated sector centres (circular means), in degrees.
ZONE_CENTRES = {1: 290.9, 2: 328.6, 3: 15.9, 4: 62.8,
                5: 120.6, 6: 165.5, 7: 211.4, 8: 247.7}

def backfill_wagonzone(df, origin=(181.5, 181.5), centres=ZONE_CENTRES,
                       out_col="wagonzone_clean"):
    """
    Fill missing wagonzone (0/blank) from wagonx/wagony, but ONLY where the ball was hit
    (batruns > 0) and coordinates are REAL. (0,0) is a missing-coordinate placeholder/sentinel in this
    dataset, not a real position, so it is excluded like NaN, those balls stay NaN.
    Assigns each fillable ball to the zone whose validated sector-centre is CIRCULARLY
    nearest its computed angle. Leaves raw `wagonzone` untouched.
    """
    d = df.copy()
    ox, oy = origin
    x = pd.to_numeric(d["wagonx"], errors="coerce")
    y = pd.to_numeric(d["wagony"], errors="coerce")
    zone = pd.to_numeric(d["wagonzone"], errors="coerce")
    batruns = pd.to_numeric(d["batruns"], errors="coerce")
    # Coordinates are valid only if present AND not the (0,0) missing sentinel.
    valid_coord = x.notna() & y.notna() & ~((x == 0) & (y == 0))
    # Angle of each hit about the origin, 0..360.
    ang = (np.degrees(np.arctan2(y - oy, x - ox)) + 360) % 360
    # Rows needing a fill: zone absent (0/NaN), ball was hit, coordinates REAL.
    need = (zone.fillna(0) == 0) & (batruns > 0) & valid_coord
    # Start output = existing valid zones only.
    d[out_col] = zone.where(zone.fillna(0) > 0)
    # Vectorised circular nearest-centre assignment for the fillable rows.
    if need.any():
        czones = np.array(list(centres.keys()))
        cangles = np.array(list(centres.values()))
        a = ang[need].to_numpy()[:, None]                 # (n_fill, 1)
        diff = np.abs(a - cangles[None, :])               # (n_fill, 8)
        circ_dist = np.minimum(diff, 360 - diff)          # wrap-aware distance
        d.loc[need, out_col] = czones[circ_dist.argmin(axis=1)]
    # Report both outcomes explicitly.
    n_filled = int(need.sum())
    n_sentinel = int(((zone.fillna(0) == 0) & (batruns > 0) & ~valid_coord).sum())
    print(f"Backfilled {n_filled} zone(s) from real coordinates; "
          f"{n_sentinel} hit-but-no-usable-coord (incl. (0,0) sentinel) left as NaN.")
    return d


def sentinel_coordinate_report(df: pd.DataFrame) -> None:
    """What the (0, 0) coordinate balls are: their raw zone and their outcomes."""
    z = pd.to_numeric(df["wagonzone"], errors="coerce")
    x = pd.to_numeric(df["wagonx"], errors="coerce")
    y = pd.to_numeric(df["wagony"], errors="coerce")
    sentinel = (x == 0) & (y == 0)
    print(f"(0,0) coordinate balls: {int(sentinel.sum())}")
    print("\nTheir raw wagonzone distribution:")
    print(z[sentinel].value_counts(dropna=False).sort_index().to_string())
    print("\nWhat kind of balls are these? (outcome):")
    print(df.loc[sentinel, "outcome"].value_counts().to_string())
    if "wagonzone_clean" in df.columns:
        print("\nwagonzone_clean on (0,0) balls:")
        print(df.loc[sentinel, "wagonzone_clean"].value_counts(dropna=False).to_string())


def add_has_placement(df: pd.DataFrame) -> pd.DataFrame:
    """Flag deliveries with genuine wagon-wheel coordinates (present and not (0, 0))."""
    d = df.copy()
    x = pd.to_numeric(d["wagonx"], errors="coerce")
    y = pd.to_numeric(d["wagony"], errors="coerce")
    d["has_placement"] = ~((x == 0) & (y == 0)) & x.notna() & y.notna()
    print(f"Balls with real wagon placement: {int(d['has_placement'].sum())} of {len(d)}")
    return d
