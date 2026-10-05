"""
Missingness of the tracked and tagged columns, and the control column's leak.

Blank rates per column, whether blanks cluster by match (systematic) or scatter
(random), the control blank rate by season, and the control-versus-faced split:
control is a batter's shot quality, so it is meaningful only on a faced ball. Values
on wides are a leak and are nulled in `control_clean`.
"""

from __future__ import annotations

import pandas as pd


TRACKING_COLS = ["line","length","shot","control","wagonx","wagony","wagonzone",
                 "bowl_speed_category","bowl_release_point","bowl_type"]

def check_missingness(df):
    d = df.copy()
    out = {}
    def blank_mask(s):
        """True where a value is genuinely absent: real NaN, or a blank/sentinel string.
        Works for numeric columns (control is {0,1}+NaN) and string columns alike."""
        st = s.astype(str).str.strip().str.lower()
        return s.isna() | st.isin(["", "-", "nan", "none", "na", "<na>", "null"])
    # per-column blank rate.
    rates = {c: round(blank_mask(d[c]).mean(), 4) for c in TRACKING_COLS if c in d.columns}
    print("blank rates:")
    for c, r in sorted(rates.items(), key=lambda x: -x[1]):
        print(f"    {c:22s}: {r:.1%}")
    out["blank_rates"] = rates
    # is missingness SYSTEMATIC? Per-match blank rate; clusters at 0%/100% => (non)systematic.
    print("\nper-match all-or-nothing test:")
    for c in TRACKING_COLS:
        if c not in d.columns: continue
        per_match = d.assign(_b=blank_mask(d[c])).groupby("p_match")["_b"].mean()
        frac_full_blank = (per_match > 0.99).mean()
        frac_full_present = (per_match < 0.01).mean()
        print(f"    {c:22s}: {frac_full_blank:.1%} matches fully blank, "
              f"{frac_full_present:.1%} fully present")
        out[f"{c}_bymatch"] = per_match
    # control blank rate by year (temporal coverage stability).
    d["_ctrl_blank"] = blank_mask(d["control"])
    print("\ncontrol blank rate by year:")
    print(d.groupby("year")["_ctrl_blank"].mean().round(3).to_string())
    # ---- control validity vs ballfaced ----
    # Control is meaningful ONLY on faced balls. Two error modes to separate:
    #   (a) faced but control blank      -> genuine coding gap (small residual, just drop)
    #   (b) NOT faced but control present -> leakage onto wides (null before use)
    bf = pd.to_numeric(d["ballfaced"], errors="coerce")
    ctrl_present = pd.to_numeric(d["control"], errors="coerce").notna()
    faced_no_ctrl   = d[(bf == 1) & ~ctrl_present]        
    notfaced_ctrl   = d[(bf == 0) &  ctrl_present]        
    out["faced_no_control"]     = faced_no_ctrl
    out["notfaced_has_control"] = notfaced_ctrl
    print("\ncontrol vs ballfaced:")
    print(f"    faced balls with control blank      : {len(faced_no_ctrl)}")
    print(f"    not-faced balls with control set  : {len(notfaced_ctrl)}")
    if len(notfaced_ctrl):
        print("    leakage outcome breakdown:")
        print(notfaced_ctrl["outcome"].value_counts().to_string().replace("\n", "\n      "))
    return out


def add_control_clean(df: pd.DataFrame) -> pd.DataFrame:
    """control restricted to faced balls; values on unfaced deliveries (wides) are nulled."""
    d = df.copy()
    bf = pd.to_numeric(d["ballfaced"], errors="coerce")
    n_before = pd.to_numeric(d["control"], errors="coerce").notna().sum()
    d["control_clean"] = pd.to_numeric(d["control"], errors="coerce").where(bf == 1)
    n_after = d["control_clean"].notna().sum()
    print(f"control non-null: {n_before} -> {n_after} (removed {n_before - n_after} not-faced)")
    return d
