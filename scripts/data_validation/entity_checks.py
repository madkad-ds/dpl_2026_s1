"""
Name-to-id integrity: player names map to one id and ids to one name, player
attributes (batting hand, bowling style) are stable within a player, and the
winner and toss fields name one of the match's two teams.

Royal Challengers Bangalore was renamed Bengaluru in 2024; the feed uses the new
name in the team columns and the old one in winner and toss. The canonicalisation
is auditable: counts before and after per column are printed.
"""

from __future__ import annotations

import pandas as pd

TEAM_CANON = {"Royal Challengers Bangalore": "Royal Challengers Bengaluru"}
TEAM_COLUMNS = ["team_bat", "team_bowl", "winner", "toss"]


def check_entity_integrity(df):
    d = df.copy()
    out = {}
    # ID <-> name cardinality. Many names per ID = tolerable 
    for id_col, name_col in [("p_bat","bat"), ("p_bowl","bowl")]:
        ids_per_name = d.groupby(name_col)[id_col].nunique()
        names_per_id = d.groupby(id_col)[name_col].nunique()
        out[f"{name_col}_multi_id"]   = ids_per_name[ids_per_name > 1]   # investigate
        out[f"{id_col}_multi_name"]   = names_per_id[names_per_id > 1]   # usually benign
    # attribute stability per player.
    bat_hand_var = d.groupby("p_bat")["bat_hand"].nunique()
    out["bat_hand_unstable"] = bat_hand_var[bat_hand_var > 1]
    for col in ["bowl_style","bowl_kind"]:
        v = d.groupby("p_bowl")[col].nunique()
        out[f"{col}_unstable"] = v[v > 1]
    # referential integrity: winner/toss must be one of the match's two teams.
    bad_ref = []
    for pm, g in d.groupby("p_match"):
        teams = set(g["team_bat"]) | set(g["team_bowl"])
        w, t = g["winner"].iloc[0], g["toss"].iloc[0]
        if pd.notna(w) and w not in teams and str(w).strip().lower() not in {"tie","no result",""}:
            bad_ref.append({"p_match": pm, "field": "winner", "value": w, "teams": teams})
        if pd.notna(t) and t not in teams:
            bad_ref.append({"p_match": pm, "field": "toss", "value": t, "teams": teams})
    out["referential"] = pd.DataFrame(bad_ref)
    def st(x): return "PASS" if len(x)==0 else f"FAIL ({len(x)})"
    for k,v in out.items():
        failed = len(v) > 0
        tag = "  [investigate]" if failed and ("multi_id" in k or "unstable" in k) else ""
        print(f"{k:34s}: {st(v)}{tag}")
    return out


def canonicalise_team_names(df: pd.DataFrame, mapping: dict = TEAM_CANON,
                            team_cols: list = TEAM_COLUMNS) -> pd.DataFrame:
    """Replace every old spelling with the canonical one across all team-bearing
    columns, reporting counts before and after so the fix is auditable."""
    d = df.copy()
    for old, new in mapping.items():
        before = {c: int((d[c] == old).sum()) for c in team_cols}
        for c in team_cols:
            if isinstance(d[c].dtype, pd.CategoricalDtype):
                d[c] = d[c].astype(str)
            d[c] = d[c].replace({old: new})
        after = {c: int((d[c] == old).sum()) for c in team_cols}
        print(f"Replaced '{old}' -> '{new}' per column:")
        for c in team_cols:
            print(f"  {c:10s}: {before[c]} -> {after[c]} remaining")
        remaining = sum(after.values())
        print(f"Total '{old}' occurrences left across team columns: {remaining}")
    return d
