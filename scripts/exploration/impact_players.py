"""
Impact-player substitutions and match rosters from the Cricsheet archive, joined
to the ball-by-ball data by match id and by a reconciled player name key.

The two sources use different player id namespaces and different name formats
(Cricsheet uses initials, the dataset full first names), so the bridge is a
surname-plus-first-initial key with a short fix list for compound surnames and
preferred names. Deployment points are carried in both conventions: Cricsheet's
0-based over and the dataset's `ball_id`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .cricsheet_archive import SEASONS, season_matches


def extract_replacements(archive, seasons=SEASONS):
    """One row per substitution event across the two seasons. Cricsheet records each
    replacement at the delivery it occurs, under innings->overs->deliveries->replacements.
    Captures the incoming and outgoing player, team, reason (impact_player vs
    concussion_substitute), and the innings/over of deployment. Match filenames are the
    Cricinfo match id, identical to p_match in the ball-by-ball data, so this joins on match_id.
    """
    rows = []
    for match_id, m in season_matches(archive, seasons):
        info = m["info"]
        season = str(info.get("season", ""))[:4]

        for inns_idx, inn in enumerate(m["innings"], start=1):
            batting_team = inn.get("team")
            for over in inn.get("overs", []):
                over_num = over.get("over")  # 0-based in Cricsheet (over 0 = 1st over)
                for ball_idx, delivery in enumerate(over.get("deliveries", [])):
                    for r in delivery.get("replacements", {}).get("match", []):
                        rows.append({
                            "match_id": match_id,
                            "season": season,
                            "innings": inns_idx,
                            "batting_team_this_innings": batting_team,
                            "cricsheet_over": over_num,          # 0-based
                            "ball_in_over": ball_idx + 1,
                            "player_in": r.get("in"),
                            "player_out": r.get("out"),
                            "team": r.get("team"),               # team making the sub
                            "reason": r.get("reason"),
                        })
    reps = pd.DataFrame(rows)
    # A replacement can be logged more than once if it appears on multiple deliveries;
    # keep the FIRST occurrence per (match, team, player_in) as the deployment point.
    reps = (reps.sort_values(["match_id", "innings", "cricsheet_over", "ball_in_over"])
            .drop_duplicates(["match_id", "team", "player_in", "reason"], keep="first")
            .reset_index(drop=True))
    print(f"Substitution events: {len(reps)}")
    print(reps["reason"].value_counts().to_string())
    return reps


def name_key(series):
    """Format-robust key (surname|first-initial). Cricsheet uses initials ('TU Deshpande'),
    the dataset uses full first names ('Tushar Deshpande'); both reduce to 'deshpande|t'."""
    s = (series.astype(str).str.lower().str.replace(r"[^a-z ]", "", regex=True)
         .str.replace(r"\s+", " ", regex=True).str.strip())
    parts = s.str.split(" ")
    return parts.str[-1] + "|" + parts.str[0].str[0]


# Same-player pairs the key can't bridge (compound surnames, preferred/formal names).
# Maps a lowercased DATASET name to the lowercased CRICSHEET name so both resolve alike.
NAME_FIX = {
    "sai sudharsan": "b sai sudharsan", "sai kishore": "r sai kishore",
    "varun chakravarthy": "cv varun", "dinesh karthik": "kd karthik",
    "bhanuka rajapaksa": "pbb rajapaksa", "dushmantha chameera": "pvd chameera",
    "wanindu hasaranga": "pwh de silva", "dasun shanaka": "md shanaka",
}

TEAM_CANON = {"Royal Challengers Bangalore": "Royal Challengers Bengaluru"}

def build_crosswalk(dataset_df, archive, seasons=SEASONS, name_fix=NAME_FIX):
    """Player crosswalk: reconciled name-key with dataset id and Cricsheet id both attached.
    The two sources use different id namespaces, so the name-key is the bridge. Cricsheet
    ids come from each match's registry; dataset ids from p_bat/p_bowl."""
    # dataset side
    b = dataset_df[["p_bat", "bat"]].rename(columns={"p_bat": "dataset_id", "bat": "name"})
    w = dataset_df[["p_bowl", "bowl"]].rename(columns={"p_bowl": "dataset_id", "bowl": "name"})
    ds = pd.concat([b, w]).dropna(subset=["dataset_id", "name"]).drop_duplicates()
    ds["key"] = name_key(ds["name"].str.lower().replace(name_fix))
    ds_ids = ds.groupby("key").agg(dataset_id=("dataset_id", "first"),
                                   dataset_name=("name", "first")).reset_index()

    # cricsheet side: union registries across the two seasons
    reg_rows = []
    for member in archive.namelist():
        if not member.endswith(".json"):
            continue
        m = json.loads(archive.read(member))
        if str(m["info"].get("season", ""))[:4] not in seasons:
            continue
        for nm, pid in m["info"].get("registry", {}).get("people", {}).items():
            reg_rows.append({"cricsheet_name": nm, "cricsheet_id": pid})
    reg = pd.DataFrame(reg_rows).drop_duplicates()
    reg["key"] = name_key(reg["cricsheet_name"].str.lower())
    cs_ids = reg.groupby("key").agg(cricsheet_id=("cricsheet_id", "first"),
                                    cricsheet_name=("cricsheet_name", "first")).reset_index()

    crosswalk = ds_ids.merge(cs_ids, on="key", how="outer")
    crosswalk["matched_both"] = crosswalk["dataset_id"].notna() & crosswalk["cricsheet_id"].notna()
    # collision guard: no key should map to two dataset ids
    collisions = crosswalk.groupby("key")["dataset_id"].nunique()
    print(f"Crosswalk rows: {len(crosswalk)} | matched both: {int(crosswalk['matched_both'].sum())} "
          f"| key collisions: {int((collisions > 1).sum())}")
    return crosswalk


def build_impact_player_table(reps, crosswalk, dataset_df, team_canon=TEAM_CANON):
    """Per impact-player substitution: named sub in/out with dataset ids attached (kept as
    the dataset's integer id type), the team, the innings, and the deployment point in BOTH
    conventions — Cricsheet's 0-based over/ball and the dataset's ball_id (over.ball with the
    integer part = over-1, fractional = delivery number in hundredths). Also flags whether the
    subbing team batted first, to test the 'switch for the second discipline' idea. Concussion
    subs are retained but flagged via `reason`. Saved for later analysis.
    """
    d = reps.copy()
    d["team"] = d["team"].replace(team_canon)
    d["batting_team_this_innings"] = d["batting_team_this_innings"].replace(team_canon)

    # attach dataset ids for in/out via the reconciled name key
    link = crosswalk[["key", "dataset_id", "dataset_name"]]
    d["key_in"] = name_key(d["player_in"].str.lower())
    d["key_out"] = name_key(d["player_out"].str.lower())
    d = (d.merge(link.rename(columns={"key": "key_in", "dataset_id": "in_dataset_id",
                                      "dataset_name": "in_dataset_name"}), on="key_in", how="left")
           .merge(link.rename(columns={"key": "key_out", "dataset_id": "out_dataset_id",
                                       "dataset_name": "out_dataset_name"}), on="key_out", how="left"))

    # Keep player ids as the dataset's integer type. The left-merge introduces NaN for any
    # unmatched name, which forces float; Int64 (nullable integer) preserves the int values
    # while still allowing missing, so ids read as 1234 not 1234.0.
    d["in_dataset_id"] = d["in_dataset_id"].astype("Int64")
    d["out_dataset_id"] = d["out_dataset_id"].astype("Int64")

    # --- deployment point in both conventions ---
    # Cricsheet over is 0-based (over 0 = 1st over); dataset over_num is 1-based.
    d["dataset_over"] = d["cricsheet_over"] + 1
    # dataset ball_id = (over_num - 1) + ball/100 = cricsheet_over + ball_in_over/100,
    # since the Cricsheet 0-based over IS the ball_id integer part.
    d["deploy_ball_id"] = (d["cricsheet_over"] + d["ball_in_over"] / 100).round(2)

    # Did the substituting team bat first?
    first_bat = (dataset_df[dataset_df["inns"] == 1]
                 .drop_duplicates("p_match")[["p_match", "team_bat"]])
    first_bat["p_match"] = first_bat["p_match"].astype(str)
    first_bat["team_bat"] = first_bat["team_bat"].replace(team_canon)
    d = d.merge(first_bat.rename(columns={"team_bat": "team_batted_first"}),
                left_on="match_id", right_on="p_match", how="left")
    d["subbing_team_batted_first"] = d["team"] == d["team_batted_first"]

    d["deploy_phase"] = pd.cut(d["cricsheet_over"], bins=[-1, 5, 14, 19],
                               labels=["powerplay", "middle", "death"])

    cols = ["match_id", "season", "team", "reason", "innings",
            "cricsheet_over", "ball_in_over",          # Cricsheet convention (0-based over)
            "dataset_over", "deploy_ball_id",          # dataset convention (1-based over, ball_id)
            "deploy_phase",
            "player_in", "in_dataset_id",
            "player_out", "out_dataset_id",
            "subbing_team_batted_first"]
    table = d[cols].sort_values(["match_id", "team"]).reset_index(drop=True)

    unmatched_in = int(table["in_dataset_id"].isna().sum())
    print(f"Impact/sub events: {len(table)} | incoming players unmatched to dataset id: {unmatched_in}")
    if unmatched_in:
        print("  unmatched incoming names:",
              table.loc[table["in_dataset_id"].isna(), "player_in"].unique().tolist())
    return table


def impact_utilisation_eda(impact_table, dataset_df, team_canon=TEAM_CANON):
    """Utilisation of the impact player, 2023 vs 2024. Uses only reason == impact_player.
    Reports: deployment timing (which innings/phase subs enter), the batting-vs-bowling
    posture of the subbing team (does the sub arrive for the second discipline), and whether
    the incoming player then featured with bat or ball (deployed vs merely swapped in).
    """
    ip = impact_table[impact_table["reason"] == "impact_player"].copy()

    print("=== Impact-player substitutions per season ===")
    print(ip.groupby("season").size().to_string())

    print("\n=== Deployment innings (1st vs 2nd) by season ===")
    print(ip.groupby(["season", "innings"]).size().unstack(fill_value=0).to_string())

    print("\n=== Deployment phase by season (when in the innings the sub enters) ===")
    print(ip.groupby(["season", "deploy_phase"], observed=True).size()
          .unstack(fill_value=0).to_string())

    print("\n=== Did the sub enter for the team's SECOND discipline? ===")
    # A team batting first that subs in innings 2 is swapping in for their bowling/defending;
    # a team bowling first that subs in innings 2 is swapping in for their chase.
    ip["sub_in_own_second_innings"] = (
        (ip["subbing_team_batted_first"] & (ip["innings"] == 2)) |
        (~ip["subbing_team_batted_first"] & (ip["innings"] == 2)))
    # More precisely: is the substitution in the second innings of the match at all?
    print("Share of impact subs deployed in the 2nd innings of the match:")
    print(ip.groupby("season")["innings"].apply(lambda s: (s == 2).mean()).round(3).to_string())

    # --- Did the incoming player then feature (bat or bowl)? ---
    feat = pd.concat([
        dataset_df[["p_match", "p_bat"]].rename(columns={"p_bat": "pid"}),
        dataset_df[["p_match", "p_bowl"]].rename(columns={"p_bowl": "pid"}),
    ]).dropna(subset=["pid"])
    feat["p_match"] = feat["p_match"].astype(str)
    featured = set(zip(feat["p_match"], feat["pid"]))
    ip["in_featured"] = ip.apply(
        lambda r: (r["match_id"], r["in_dataset_id"]) in featured, axis=1)
    print("\n=== Did the impact player feature with bat or ball once subbed in? ===")
    print(ip.groupby("season")["in_featured"].agg(["mean", "sum", "size"]).round(3).to_string())

    return ip


def build_match_rosters_with_utilisation(archive, replacements, crosswalk, dataset_df,
                                         seasons={"2023", "2024"}, team_canon=None):
    """Per (match, team) roster of all named players with their role and whether they were
    utilised (batted or bowled in the dataset). Cricsheet names 12 per team: the starting XI
    plus the impact sub. The `replacements` field identifies which named player is the impact
    sub (the incoming player) and whom they replaced. Roles assigned:
        impact_sub   — the incoming player from a reason=='impact_player' replacement
        replaced     — the outgoing player that the impact sub came in for
        starting_xi  — the remaining named players
    Utilised = the player's dataset id appears as a batter or bowler in that match.
    """
    canon = team_canon or {}

    # --- who featured (batted or bowled) per match, by dataset id ---
    feat = pd.concat([
        dataset_df[["p_match", "p_bat"]].rename(columns={"p_bat": "pid"}),
        dataset_df[["p_match", "p_bowl"]].rename(columns={"p_bowl": "pid"}),
    ]).dropna(subset=["pid"])
    feat["p_match"] = feat["p_match"].astype(str)
    featured_pairs = set(zip(feat["p_match"], feat["pid"]))

    # --- name-key -> dataset id, from the crosswalk ---
    key_to_id = (crosswalk.dropna(subset=["dataset_id"])
                 .set_index("key")["dataset_id"].astype("Int64").to_dict())

    # --- impact-sub in/out lookup: (match_id, team) -> {in_names, out_names} ---
    ip = replacements[replacements["reason"] == "impact_player"].copy()
    ip["team"] = ip["team"].replace(canon)
    sub_in = ip.groupby(["match_id", "team"])["player_in"].apply(set).to_dict()
    sub_out = ip.groupby(["match_id", "team"])["player_out"].apply(set).to_dict()

    # --- walk every match's 12-man team lists ---
    rows = []
    for member in archive.namelist():
        if not member.endswith(".json"):
            continue
        m = json.loads(archive.read(member))
        if str(m["info"].get("season", ""))[:4] not in seasons:
            continue
        match_id = member.removesuffix(".json")
        season = str(m["info"].get("season", ""))[:4]

        for team, players in m["info"].get("players", {}).items():
            team_c = canon.get(team, team)
            ins = sub_in.get((match_id, team_c), set())
            outs = sub_out.get((match_id, team_c), set())
            for name in players:
                if name in ins:
                    role = "impact_sub"
                elif name in outs:
                    role = "replaced"
                else:
                    role = "starting_xi"
                pid = key_to_id.get(name_key(pd.Series([name.lower()]))[0], pd.NA)
                utilised = (match_id, pid) in featured_pairs if pd.notna(pid) else False
                rows.append({
                    "match_id": match_id, "season": season, "team": team_c,
                    "player": name, "dataset_id": pid, "role": role,
                    "utilised": utilised,
                })

    roster = pd.DataFrame(rows)
    roster["dataset_id"] = roster["dataset_id"].astype("Int64")
    roster = roster.sort_values(["match_id", "team", "role", "player"]).reset_index(drop=True)

    unmatched = int(roster["dataset_id"].isna().sum())
    print(f"Roster rows: {len(roster)} | players unmatched to dataset id: {unmatched}")
    if unmatched:
        print("  unmatched:", roster.loc[roster["dataset_id"].isna(), "player"].unique().tolist())
    return roster


def roster_utilisation_eda(roster):
    """Utilisation by role and season: what share of named players in each role actually
    batted or bowled. Separates the starting XI (where non-utilisation = tail-enders/backup
    bowlers not needed) from the impact sub (where non-utilisation = named but not deployed
    with bat/ball) and the replaced player (who may have featured before being withdrawn).
    """
    print("=== Named players per team-match by role (sanity: XI≈11, impact_sub≈1) ===")
    per = (roster.groupby(["match_id", "team", "role"]).size()
           .groupby("role").agg(["mean", "min", "max"]).round(2))
    print(per.to_string())

    print("\n=== Utilisation rate by role and season (share who batted or bowled) ===")
    util = (roster.groupby(["season", "role"])["utilised"]
            .agg(rate="mean", n="size").round(3))
    print(util.to_string())

    print("\n=== Starting-XI utilisation distribution (how many of the XI feature) ===")
    xi = roster[roster["role"] == "starting_xi"]
    xi_util = xi.groupby(["match_id", "team", "season"])["utilised"].sum()
    print("Players from the starting XI who featured, per team-match, by season:")
    print(xi_util.reset_index().groupby("season")["utilised"].describe().round(2).to_string())

    return util

