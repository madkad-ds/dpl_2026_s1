"""
Cricsheet IPL archive: fetch once, cache under data/, and inspect its format.

Cricsheet publishes every IPL match as a JSON file inside one zip. The archive is
regenerated as seasons are added, so a fresh download a year from now is not the
same file. The archive is therefore cached under data/ with a sidecar recording the
download date, URL and SHA-256, and read from the cache whenever it is present; a
rerun reproduces the same tables until the cache is deliberately refreshed.

Match filenames are the Cricinfo match id, identical to `p_match` in the
ball-by-ball data, so the two sources join on match id. Substitutions are recorded
at the delivery where they occur, under innings -> overs -> deliveries ->
replacements, with a reason of `impact_player` or `concussion_substitute`.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CRICSHEET_IPL_JSON = "https://cricsheet.org/downloads/ipl_json.zip"
SEASONS = {"2023", "2024"}


def fetch_ipl_archive(cache_path: Path, url: str = CRICSHEET_IPL_JSON,
                      refresh: bool = False) -> zipfile.ZipFile:
    """The archive as a ZipFile, from the cache when present, else downloaded and
    cached with a provenance sidecar (URL, UTC timestamp, SHA-256, size)."""
    cache_path = Path(cache_path)
    if refresh or not cache_path.exists():
        import requests
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        sidecar = {"url": url, "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "sha256": hashlib.sha256(resp.content).hexdigest(), "bytes": len(resp.content)}
        cache_path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2))
    return zipfile.ZipFile(io.BytesIO(cache_path.read_bytes()))


def archive_provenance(cache_path: Path) -> dict:
    meta = Path(cache_path).with_suffix(".meta.json")
    return json.loads(meta.read_text()) if meta.exists() else {}


def season_matches(archive: zipfile.ZipFile, seasons=SEASONS):
    """Yield (match_id, parsed json) for matches in the wanted seasons."""
    for member in archive.namelist():
        if not member.endswith(".json"):
            continue
        m = json.loads(archive.read(member))
        if str(m["info"].get("season", ""))[:4] not in seasons:
            continue
        yield member.removesuffix(".json"), m


def inspect_match_format(archive: zipfile.ZipFile, match_id: str) -> None:
    """Top-level structure of one match file: keys, player lists, where the
    replacement records sit."""
    name = f"{match_id}.json"
    if name not in archive.namelist():
        print(f"{name} not in archive; available sample:", archive.namelist()[:5])
        return
    match = json.loads(archive.read(name))
    print("Top-level keys:", list(match.keys()))
    print("\nInfo keys:", list(match["info"].keys()))
    print("\nPlayers listed per team:")
    for team, players in match["info"].get("players", {}).items():
        print(f"  {team}: {len(players)} players")
    print("\nHas registry:", "registry" in match["info"])
    print("Innings count:", len(match["innings"]))
    for i, inn in enumerate(match["innings"]):
        print(f"  innings {i + 1} team: {inn.get('team')}; innings-level keys:",
              [k for k in inn.keys() if k not in ("team", "overs")])
        for over in inn.get("overs", []):
            for delivery in over.get("deliveries", []):
                if "replacements" in delivery:
                    print(f"    over {over['over']}: replacement:", delivery["replacements"])


def replacement_coverage(archive: zipfile.ZipFile, seasons=SEASONS) -> None:
    """Per season: matches with and without replacement records, replacements per
    match, the reasons seen, and the players listed per team."""
    coverage, per_match, reasons, players_per_team = Counter(), Counter(), Counter(), {}
    for _, m in season_matches(archive, seasons):
        season = str(m["info"].get("season", ""))[:4]
        seen = []
        for inn in m["innings"]:
            for ov in inn.get("overs", []):
                for d in ov.get("deliveries", []):
                    for r in d.get("replacements", {}).get("match", []):
                        seen.append(r)
                        reasons[r.get("reason")] += 1
        coverage[(season, bool(seen))] += 1
        per_match[len(seen)] += 1
        for team, pl in m["info"].get("players", {}).items():
            players_per_team.setdefault(season, Counter())[len(pl)] += 1
    print("matches by (season, has replacements):")
    for k in sorted(coverage):
        print(" ", k, coverage[k])
    print("replacements per match:", dict(sorted(per_match.items())))
    print("reasons:", dict(reasons))
    for season, c in sorted(players_per_team.items()):
        print(f"{season}: players listed per team {dict(sorted(c.items()))}")
