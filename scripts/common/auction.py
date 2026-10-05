"""
The auction tables: one row per player per season with status (retained, sold,
traded, unsold), type, team and price, reconciled to dataset player ids by name.

The workbook has one sheet per season with a title row above the header. Prices
are strings in lakh ("20.00 L") or crore ("6.75 Cr"); "--" is blank. Names are full
names as in the ball-by-ball data, so an exact match on a letters-only key covers
most players; the remainder are tried on surname plus first initial, then on a
unique surname, and a short fix list handles known spellings. Unmatched auction rows
are reported, never guessed.

A player's squad membership for a season is any status other than unsold.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SHEETS = {2023: "IPL 2023", 2024: "IPL 2024"}
TEAM_CODES = {"CSK": "Chennai Super Kings", "DC": "Delhi Capitals", "GT": "Gujarat Titans",
              "KKR": "Kolkata Knight Riders", "LSG": "Lucknow Super Giants", "MI": "Mumbai Indians",
              "PBKS": "Punjab Kings", "RR": "Rajasthan Royals", "RCB": "Royal Challengers Bengaluru",
              "SRH": "Sunrisers Hyderabad"}
# Auction spelling -> dataset spelling, for names the automatic keys do not bridge.
NAME_FIX = {
    "varun chakaravarthy": "varun chakravarthy",
    "joshua little": "josh little",
    "shahrukh khan": "m shahrukh khan",
    "sundar washington": "washington sundar",
    "aman hakim khan": "aman khan",
    "yudhvir singh charak": "yudhvir singh",
    "ravisrinivasan sai kishore": "sai kishore",
    "bhagath varma": "kanumuri bhagath varma",
}


def _letters(name) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def _surname_initial(name) -> str:
    parts = [p for p in re.sub(r"[^a-z ]", "", str(name).lower()).split() if p]
    return f"{parts[-1]}|{parts[0][0]}" if parts else ""


def parse_price(value) -> float:
    """Price in crore; blank for '--'."""
    s = str(value).strip()
    m = re.match(r"([\d.]+)\s*(L|Cr)", s)
    if not m:
        return np.nan
    amount, unit = float(m.group(1)), m.group(2)
    return amount / 100.0 if unit == "L" else amount


def load_auction(path: Path) -> pd.DataFrame:
    """Both seasons stacked: season, player_name, type, country, status, base and
    final price in crore, team (full name; 'Unsold' kept as is), in_squad."""
    frames = []
    for season, sheet in SHEETS.items():
        d = pd.read_excel(path, sheet_name=sheet, header=1)
        d = d.rename(columns={"Player Name": "player_name", "Type": "type", "Country": "country", "Status": "status",
                              "Base Price": "base_price", "Final Price": "final_price", "Team": "team_code"})
        d["season"] = season
        d["base_price_cr"] = d["base_price"].map(parse_price)
        d["final_price_cr"] = d["final_price"].map(parse_price)
        d["team"] = d["team_code"].map(TEAM_CODES).fillna(d["team_code"])
        d["in_squad"] = d["status"].astype(str).str.lower() != "unsold"
        frames.append(d[["season", "player_name", "type", "country", "status", "base_price_cr", "final_price_cr", "team_code", "team", "in_squad"]])
    return pd.concat(frames, ignore_index=True)


def dataset_players(frame: pd.DataFrame) -> pd.DataFrame:
    """Every player id in the ball-by-ball data with the name it carries."""
    bat = frame.drop_duplicates("p_bat")[["p_bat", "bat"]].rename(columns={"p_bat": "player_id", "bat": "name"})
    bowl = frame.drop_duplicates("p_bowl")[["p_bowl", "bowl"]].rename(columns={"p_bowl": "player_id", "bowl": "name"})
    return pd.concat([bat, bowl]).drop_duplicates("player_id").reset_index(drop=True)


def match_auction_to_dataset(auction: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach dataset ids to auction rows. Returns (auction with player_id and
    match_method, unmatched rows). Only players who appear in the ball-by-ball data
    can match; an auction row with no match is a squad member who never played a
    ball in either season, or a spelling to add to NAME_FIX."""
    players = players.copy()
    players["key"] = players["name"].map(_letters)
    players["si"] = players["name"].map(_surname_initial)
    players["surname"] = players["name"].map(lambda n: _surname_initial(n).split("|")[0])
    by_key = players.drop_duplicates("key").set_index("key")["player_id"]
    by_si = players.groupby("si")["player_id"].agg(lambda s: s.iloc[0] if len(s) == 1 else np.nan).dropna()

    out = auction.copy()
    fixed = out["player_name"].astype(str).str.lower().str.strip().map(lambda n: NAME_FIX.get(n, n))
    keys = fixed.map(_letters)
    ids = keys.map(by_key)
    method = np.where(ids.notna(), "exact", None)
    sis = fixed.map(_surname_initial)
    fallback = sis.map(by_si)
    method = np.where(ids.isna() & fallback.notna(), "surname_initial", method)
    ids = ids.fillna(fallback)
    out["player_id"] = ids.astype("Int64")
    out["match_method"] = method
    # Where two auction rows land on one player in a season, keep the exact match.
    out["_priority"] = np.where(out["match_method"] == "exact", 0, 1)
    out = out.sort_values(["season", "_priority"]).drop(columns="_priority")
    unmatched = out[out["player_id"].isna()][["season", "player_name", "type", "status", "team"]]
    return out, unmatched


def match_report(matched: pd.DataFrame) -> pd.DataFrame:
    """Match rate by season and status; squad members who never played are expected
    to be unmatched, so the informative rate is among sold and retained players who
    appear in the data."""
    g = matched.groupby(["season", "status"]).agg(rows=("player_name", "size"),
                                                  matched=("player_id", lambda s: int(s.notna().sum())))
    g["match_rate"] = g["matched"] / g["rows"]
    return g.reset_index()
