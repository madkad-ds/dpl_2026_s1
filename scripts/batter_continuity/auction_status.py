"""
Were the entrants bought or promoted? The composition part of the batter-level
decomposition, split by the entrant's auction status.

The decomposition found about half of the powerplay six-rate rise to be different
batters facing the balls. The auction tables say how each entrant came to be there:
sold at the later season's auction (bought), traded, retained (already in the
squad, so promoted or newly selected), or absent from the auction table (in the
data but not on a squad list). Each entrant's composition contribution is summed
by that status, and the same is done for exits by their later-season status
(unsold, not retained, retained but not selected). Descriptive: a per-status sum
of contributions, with the balls behind each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def entrant_status(batters: pd.DataFrame, auction_matched: pd.DataFrame, earlier: int, later: int) -> pd.DataFrame:
    """Per-batter table (from the decomposition) with the later-season auction
    status for entrants and the later-season status for exits."""
    later_status = auction_matched[auction_matched["season"] == later].dropna(subset=["player_id"]).drop_duplicates("player_id")
    later_status = later_status.set_index("player_id")[["status", "team", "final_price_cr", "type"]]
    earlier_status = auction_matched[auction_matched["season"] == earlier].dropna(subset=["player_id"]).drop_duplicates("player_id")
    earlier_status = earlier_status.set_index("player_id")[["status", "team"]]
    out = batters.copy()
    out["status_later"] = out["p_bat"].map(later_status["status"]).fillna("not_in_auction_table")
    out["team_later"] = out["p_bat"].map(later_status["team"])
    out["price_later_cr"] = out["p_bat"].map(later_status["final_price_cr"])
    out["type_later"] = out["p_bat"].map(later_status["type"])
    out["status_earlier"] = out["p_bat"].map(earlier_status["status"]).fillna("not_in_auction_table")
    out["team_earlier"] = out["p_bat"].map(earlier_status["team"])
    # How an entrant got there: bought at the later auction, traded in, or already
    # in a squad (retained) and newly given powerplay balls.
    out["entry_route"] = np.select(
        [out["status"] != "entrant",
         out["status_later"].str.lower().eq("sold"),
         out["status_later"].str.lower().eq("traded"),
         out["status_later"].str.lower().eq("retained")],
        ["", "bought", "traded_in", "promoted_from_squad"], default="not_in_auction_table")
    # How an exit left: unsold at the later auction, absent from it (released or
    # retired), or retained but no longer given powerplay balls.
    out["exit_route"] = np.select(
        [out["status"] != "exit",
         out["status_later"].str.lower().eq("unsold"),
         out["status_later"].str.lower().eq("retained"),
         out["status_later"].str.lower().isin(["sold", "traded"])],
        ["", "unsold", "retained_not_used", "moved_team"], default="not_in_auction_table")
    return out


def composition_by_route(batters_with_status: pd.DataFrame, earlier: int, later: int) -> pd.DataFrame:
    """Sum of composition contributions by entry route (entrants) and exit route
    (exits), with balls, sixes-rate and counts, from the per-batter table. The
    composition contribution of a one-season batter is share x rate, so it is
    recomputed here from the table's balls and rates."""
    b = batters_with_status
    total_e = b[f"balls_{earlier}"].sum()
    total_l = b[f"balls_{later}"].sum()
    rows = []
    for group, route_col in (("entrant", "entry_route"), ("exit", "exit_route")):
        sub = b[b["status"] == group]
        for route, g in sub.groupby(route_col):
            if group == "entrant":
                share = g[f"balls_{later}"].sum() / total_l
                rate = (g[f"rate_{later}"] * g[f"balls_{later}"]).sum() / g[f"balls_{later}"].sum()
                contribution = share * rate
                balls = int(g[f"balls_{later}"].sum())
            else:
                share = g[f"balls_{earlier}"].sum() / total_e
                rate = (g[f"rate_{earlier}"] * g[f"balls_{earlier}"]).sum() / g[f"balls_{earlier}"].sum()
                contribution = -share * rate
                balls = int(g[f"balls_{earlier}"].sum())
            rows.append({"group": group, "route": route, "batters": int(len(g)), "balls": balls,
                         "share_of_season_balls": float(share), "six_rate": float(rate),
                         "composition_contribution": float(contribution),
                         "median_price_cr": float(g["price_later_cr"].median()) if group == "entrant" else np.nan})
    columns = ["group", "route", "batters", "balls", "share_of_season_balls", "six_rate", "composition_contribution", "median_price_cr"]
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values(["group", "composition_contribution"], ascending=[True, False]).reset_index(drop=True)
