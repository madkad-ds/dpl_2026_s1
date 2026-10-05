"""
Manually tagged delivery columns (bowl_type, length, line, shot): vocabulary
discovery, cross-column leakage of terms, contradictory tag pairs on one ball, the
phase pattern of missing bowl_type tags, and a coarse bowl_type grouping.

Length and bowl_type are orthogonal axes (where the ball pitched versus how it
was delivered), so most combinations are legitimate; only bowl_types whose name
itself encodes a length (the yorker and bouncer families) can contradict the
length column. Missing bowl_type tags are three times as common in the death
overs, so that missingness is informative rather than random.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Logical-consistency audit for manually-tagged delivery columns.
#
# Context: bowl_type, length, line, shot are hand-tagged.
# Two failure modes to catch:
#   (1) VALUE-IN-WRONG-COLUMN: a term that belongs to one column's vocabulary
#       appears in another (e.g. 'yorker'/'bouncer' — really lengths/types —
#       showing up in the wrong field).
#   (2) CONTRADICTORY-PAIR: two tags on the same ball that can't co-occur
#       (e.g. bowl_type='bouncer' with length='FULL' — a bouncer is short).
#
# The goal is to size the problem and locate the systematic patterns, not to chase every single row.
# ---------------------------------------------------------------------------

TAG_COLS = ["bowl_type", "length", "line", "shot"]

def audit_tag_consistency(df):
    d = df.copy()
    for c in TAG_COLS:
        d[c] = d[c].astype(str).str.strip().str.upper().replace(
            {"NAN": pd.NA, "": pd.NA, "-": pd.NA, "NONE": pd.NA})

    # --- Step 1: discover the actual vocabulary of each column ---
    print("=== Value vocabularies (inspect for cross-column leakage) ===")
    vocab = {}
    for c in TAG_COLS:
        vocab[c] = set(d[c].dropna().unique())
        print(f"\n{c} ({len(vocab[c])} distinct):")
        print(d[c].value_counts(dropna=False).to_string())

    # --- Step 2: cross-column value leakage ---
    # A term that appears in more than one column's vocabulary is a leakage suspect:
    # the same word shouldn't legitimately be a valid tag in two different columns.
    print("\n\n=== Cross-column value overlaps (same term valid in 2+ columns) ===")
    all_terms = {}
    for c in TAG_COLS:
        for term in vocab[c]:
            all_terms.setdefault(term, []).append(c)
    overlaps = {t: cols for t, cols in all_terms.items() if len(cols) > 1}
    if overlaps:
        for t, cols in sorted(overlaps.items()):
            print(f"  '{t}' appears in: {cols}")
    else:
        print("  none — vocabularies are disjoint.")

    # --- Step 3: known taxonomy-crossers ---
    # Terms that are conceptually a LENGTH or TYPE but may be mis-filed. Extend this
    # list as inspection reveals more. These are the 'yorker in length' class.
    LENGTH_TERMS = {"YORKER", "FULL_TOSS", "FULL", "GOOD_LENGTH", "SHORT",
                    "SHORT_OF_LENGTH", "BACK_OF_LENGTH", "HALF_VOLLEY"}
    TYPE_TERMS   = {"BOUNCER", "SLOWER_BALL", "GOOGLY", "DOOSRA", "YORKER",
                    "FULL_TOSS", "OFF_CUTTER", "LEG_CUTTER"}  # yorker/full_toss span both

    print("\n\n=== Suspected mis-filed terms ===")
    # length-words showing up in bowl_type, and vice versa
    len_in_type = d[d["bowl_type"].isin(LENGTH_TERMS - TYPE_TERMS)]
    type_in_len = d[d["length"].isin(TYPE_TERMS - LENGTH_TERMS)]
    print(f"  length-like term in bowl_type: {len(len_in_type)} rows")
    if len(len_in_type):
        print(d.loc[len_in_type.index, "bowl_type"].value_counts().to_string())
    print(f"  type-like term in length: {len(type_in_len)} rows")
    if len(type_in_len):
        print(d.loc[type_in_len.index, "length"].value_counts().to_string())

    # --- Step 4: contradictory pairs on the same ball ---
    # Encode a few physically-impossible combinations. A bouncer can't be FULL;
    # a yorker can't be SHORT; a full-toss can't be a GOOD_LENGTH; etc.
    # Extend as domain logic dictates.
    contradictions = [
        ("bowl_type", "BOUNCER", "length", {"FULL", "FULL_TOSS", "HALF_VOLLEY", "YORKER"}),
        ("bowl_type", "YORKER",  "length", {"SHORT", "SHORT_OF_LENGTH", "BACK_OF_LENGTH"}),
        ("bowl_type", "SLOWER_BALL", "length", set()),  # slower ball is any length — no contradiction
        ("length", "YORKER", "line", {"WIDE_DOWN_LEG"}),  # example placeholder; refine per domain
    ]
    print("\n\n=== Contradictory tag pairs on the same delivery ===")
    contra_hits = {}
    for col_a, val_a, col_b, bad_b in contradictions:
        if not bad_b:
            continue
        mask = (d[col_a] == val_a) & (d[col_b].isin(bad_b))
        n = int(mask.sum())
        contra_hits[f"{col_a}={val_a} & {col_b} in {bad_b}"] = n
        print(f"  {col_a}={val_a} with {col_b} in {sorted(bad_b)}: {n} rows")

    # --- Step 5: cross-tab the two length/type columns to see the joint distribution ---
    # The most efficient way to SPOT contradictions I haven't hard-coded: look at
    # which (bowl_type, length) combinations occur and how often. Rare off-diagonal
    # combos are the suspects.
    print("\n\n=== bowl_type x length joint frequency (rare combos = suspects) ===")
    ct = pd.crosstab(d["bowl_type"], d["length"])
    print(ct.to_string())

    return {"vocab": vocab, "overlaps": overlaps,
            "len_in_type": len_in_type, "type_in_len": type_in_len,
            "contradiction_counts": contra_hits, "crosstab": ct}


# ---------------------------------------------------------------------------
# Revised consistency lists, derived from the observed bowl_type x length crosstab.
#
# Key realisation: length (6 vals) and bowl_type (31 vals) are orthogonal axes —
# 'how delivered' vs 'where pitched'. Most combinations are legitimate. Only the
# bowl_types that THEMSELVES encode a length can contradict the length column.
# Those are the YORKER family and the BOUNCER family. Everything else is
# length-agnostic and generates no contradiction.
# ---------------------------------------------------------------------------

# The full observed length vocabulary (all 6 legitimate).
LENGTH_VALUES = {"GOOD_LENGTH", "SHORT_OF_A_GOOD_LENGTH", "FULL",
                 "SHORT", "FULL_TOSS", "YORKER"}

# bowl_types whose NAME encodes 'yorker' -> the ball pitched full/at the feet.
# A yorker cannot be SHORT or short-of-a-length. It should be length FULL or YORKER.
YORKER_TYPES = {"SLOW YORKER", "REVERSESWING YORKER", "INSWINGING YORKER"}

# bowl_types whose NAME encodes 'bouncer' -> the ball pitched short and rose.
# A bouncer cannot be FULL/FULL_TOSS/YORKER. Ambiguous with GOOD_LENGTH (a bouncer
# can climb from back-of-a-good-length), so GOOD_LENGTH is NOT flagged as an error.
BOUNCER_TYPES = {"BOUNCER", "SLOW BOUNCER"}

# Lengths that are physically incompatible with each family.
YORKER_BAD_LENGTHS  = {"SHORT", "SHORT_OF_A_GOOD_LENGTH", "GOOD_LENGTH"}
BOUNCER_BAD_LENGTHS = {"FULL", "FULL_TOSS", "YORKER"}   # NOT good_length (arguable)

# Contradiction rules: (type_set, incompatible_length_set)
CONTRADICTION_RULES = [
    (YORKER_TYPES,  YORKER_BAD_LENGTHS),
    (BOUNCER_TYPES, BOUNCER_BAD_LENGTHS),
]


def flag_type_length_contradictions(df, rules=CONTRADICTION_RULES):
    d = df.copy()
    d["bowl_type"] = d["bowl_type"].astype(str).str.strip().str.upper()
    d["length"]    = d["length"].astype(str).str.strip().str.upper()

    flags = pd.Series(False, index=d.index)
    for type_set, bad_lengths in rules:
        m = d["bowl_type"].isin(type_set) & d["length"].isin(bad_lengths)
        n = int(m.sum())
        flags |= m
        print(f"  types {sorted(type_set)} with length in {sorted(bad_lengths)}: {n} rows")

    print(f"\nTotal contradictory rows: {int(flags.sum())} "
          f"({flags.mean():.2%} of dataset)")
    d["type_length_contradiction"] = flags
    return d


def bowl_type_missing_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """bowl_type NaN rate by phase, and by season x phase."""
    phase = pd.cut(pd.to_numeric(df["over_num"], errors="coerce"), [0, 6, 15, 20], labels=["PP", "Mid", "Death"])
    missing = df["bowl_type"].isna()
    print("bowl_type NaN rate by phase:")
    print(missing.groupby(phase, observed=True).mean().round(3).to_string())
    by_year = missing.groupby([pd.to_numeric(df["year"], errors="coerce"), phase], observed=True).mean().round(3).unstack()
    print(by_year.to_string())
    return by_year


BOWL_TYPE_GROUPS = {
    # Spin — wrist
    "LEG SPIN": "wrist_spin", "GOOGLY": "wrist_spin", "CHINAMAN": "wrist_spin",
    "WRONG ONE": "wrist_spin", "FLIPPER": "wrist_spin", "SLIDER": "wrist_spin",
    "CARROM BALL": "wrist_spin",
    # Spin — finger
    "OFF SPIN": "finger_spin", "ORTHODOX": "finger_spin", "ARM BALL": "finger_spin",
    "OFF CUTTER": "cutter", "LEGCUTTER": "cutter",   # cutters: seam-spin hybrids, own group
    # Seam/swing — stock
    "ANGLED IN": "seam_stock", "ANGLED ACROSS": "seam_stock", "STRAIGHT BALL": "seam_stock",
    # Seam/swing — swing
    "OUTSWINGER": "swing", "INSWINGER": "swing", "REVERSE SWING": "swing",
    "NIPBACKER": "swing", "NIPPED AWAY": "swing",
    # Slower balls / deception
    "SLOWER BALL": "slower_ball_deception", "KNUCKLEBALL": "slower_ball_deception",
    "BACKHAND SLOWER": "slower_ball_deception", "FASTER ONE": "slower_ball_deception",
    # Short balls — own group
    "BOUNCER": "bouncer", "SLOW BOUNCER": "bouncer",
    # Yorkers — own group
    "SLOW YORKER": "yorker", "REVERSESWING YORKER": "yorker", "INSWINGING YORKER": "yorker",
    # Dropped — too thin / ambiguous
    "DRIFTER": None, "TOP SPINNER": None,
}


def collapse_bowl_type(df, mapping=BOWL_TYPE_GROUPS,
                       src="bowl_type", out="bowl_type_group"):
    d = df.copy()
    key = d[src].astype(str).str.strip().str.upper().replace({"NAN": pd.NA, "": pd.NA})
    grp = key.map(mapping)
    # NaN (untagged) -> its own informative category, NOT dropped.
    grp = grp.where(key.notna(), other="unclassified")
    # Dropped types (DRIFTER/TOP SPINNER mapped to None) -> distinguish from untagged.
    grp = grp.where(grp.notna(), other="other_rare")
    d[out] = grp
    print(d[out].value_counts(dropna=False).to_string())
    return d

