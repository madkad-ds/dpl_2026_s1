"""
Where the missing spinners went: squad, eleven, or bench.

The selection-or-usage decomposition found fewer spinners named per eleven and
fewer of the named ones bowled. The auction tables add the level above the eleven:
how many spinners each squad carried each season, and whether the fall in named
spinners came from squads carrying fewer (auction and retention decisions) or from
squads carrying the same and naming fewer (selection from within). Squad spinners
are counted among squad members who can be classified from the ball-by-ball data
(bowled a ball in either season); a squad member who never bowled is unclassified
and counted separately, with the auction's own type as the only guide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def squad_composition(auction_matched: pd.DataFrame, spinners: pd.DataFrame) -> pd.DataFrame:
    """Per team and season: squad size, players classified as spinners, as pace,
    unclassified, and the auction's own bowler and all-rounder counts."""
    a = auction_matched[auction_matched["in_squad"]].copy()
    spin_ids = set(spinners.loc[spinners["is_spinner"], "p_bowl"])
    classified = spinners.dropna(subset=["legal"])
    bowled_ids = set(classified.loc[classified["legal"] > 0, "p_bowl"])
    a["kind"] = np.select([a["player_id"].isin(spin_ids), a["player_id"].isin(bowled_ids)],
                          ["spinner", "pace_or_other"], default="unclassified")
    a["auction_bowls"] = a["type"].isin(["Bowler", "Allrounder"])
    g = a.groupby(["team", "season"]).agg(
        squad=("player_name", "size"), spinners=("kind", lambda s: int((s == "spinner").sum())),
        pace_or_other=("kind", lambda s: int((s == "pace_or_other").sum())),
        unclassified=("kind", lambda s: int((s == "unclassified").sum())),
        auction_bowlers_and_allrounders=("auction_bowls", "sum"),
        spend_cr=("final_price_cr", "sum")).reset_index()
    spend_spin = a[a["kind"] == "spinner"].groupby(["team", "season"])["final_price_cr"].sum().rename("spend_on_spinners_cr")
    g = g.merge(spend_spin.reset_index(), on=["team", "season"], how="left").fillna({"spend_on_spinners_cr": 0.0})
    g["spend_share_spinners"] = g["spend_on_spinners_cr"] / g["spend_cr"]
    return g


def spinner_flow(auction_matched: pd.DataFrame, spinners: pd.DataFrame, earlier: int, later: int) -> pd.DataFrame:
    """Each classified spinner's status in both seasons: retained, sold, traded,
    unsold, or absent, so the league's change in squad spinners can be read as
    releases, unsold at auction, and new purchases."""
    spin_ids = set(spinners.loc[spinners["is_spinner"], "p_bowl"])
    names = spinners.set_index("p_bowl")["name"]
    rows = []
    a = auction_matched.dropna(subset=["player_id"])
    for pid in sorted(spin_ids):
        e = a[(a["season"] == earlier) & (a["player_id"] == pid)]
        l = a[(a["season"] == later) & (a["player_id"] == pid)]
        rows.append({"player_id": pid, "name": names.get(pid, pid),
                     f"status_{earlier}": e["status"].iloc[0] if len(e) else "absent",
                     f"team_{earlier}": e["team"].iloc[0] if len(e) else "",
                     f"status_{later}": l["status"].iloc[0] if len(l) else "absent",
                     f"team_{later}": l["team"].iloc[0] if len(l) else "",
                     f"price_{later}_cr": l["final_price_cr"].iloc[0] if len(l) else np.nan})
    flow = pd.DataFrame(rows)
    flow["movement"] = np.select(
        [flow[f"status_{later}"].str.lower().eq("unsold"),
         flow[f"status_{later}"].eq("absent") & flow[f"status_{earlier}"].ne("absent"),
         flow[f"status_{earlier}"].isin(["absent"]) & flow[f"status_{later}"].str.lower().isin(["sold", "traded", "retained"]),
         flow[f"status_{later}"].str.lower().eq("sold"),
         flow[f"status_{later}"].str.lower().eq("traded")],
        ["unsold_at_later_auction", "left_the_league", "new_to_the_league", "bought_by_a_team", "traded"],
        default="retained")
    return flow


def flow_summary(flow: pd.DataFrame) -> pd.DataFrame:
    return flow["movement"].value_counts().rename_axis("movement").rename("spinners").reset_index()


def squad_to_eleven(squads: pd.DataFrame, innings: pd.DataFrame) -> pd.DataFrame:
    """Per team and season: spinners in the squad, named per eleven, used per innings,
    so the drop can be placed at the squad, the eleven, or the bench."""
    named = innings.groupby(["team", "season"]).agg(spinners_named=("spinners_named", "mean"),
                                                    spinners_used=("spinners_used", "mean")).reset_index()
    out = squads[["team", "season", "squad", "spinners", "unclassified", "spend_share_spinners"]].merge(named, on=["team", "season"], how="left")
    out["named_share_of_squad_spinners"] = out["spinners_named"] / out["spinners"].replace(0, np.nan)
    return out.sort_values(["team", "season"]).reset_index(drop=True)
