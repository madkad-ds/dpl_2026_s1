"""
Extended synthetic two-season ball-by-ball frame, aligned to the REAL schema.

This is a distinct fixture from the harness's `_synthetic_two_seasons` stub, which
is left byte-stable for the existing dispersion tests. This one emits the real
column names and encodings and the richer columns the two NEW analyses need
(phase-boundary profiles and situation-only xW): `batruns`/`score` split,
`ballfaced`, real `bowl_kind` categories, populated `byes`/`legbyes`, tagged
columns, and pre-ball match-state features.

MECHANICS FIXTURE ONLY. No number from this frame is a finding. Its job is to let
the analysis code be checked against a known truth (the answer key below).

ANSWER KEY (the truth the analyses must recover or leave alone)
--------------------------------------------------------------
Phase-boundary analysis:
  * Planted boundary MOVE. Death-style behaviour (six/aggression ramp) is anchored
    to `death_edge_over`, set to 16 in 2023 and 15 in 2024 -- the death starts one
    over earlier in the later season. Break detection must locate the sequence break
    ~one over earlier in 2024 than in 2023, and the bootstrap must show that move is
    stable rather than a wandering artifact.
  * Pace share by over is FIXED across seasons, so pace contributes phase SHAPE but
    no planted cross-season move of its own.

Situation-only xW analysis:
  * Known situation -> wicket hazard. Per faced ball the dismissal probability is a
    smooth function of pre-ball SITUATION only:
        hazard = base_by_phase
                 + 0.020 * fraction_of_innings_bowled     (later balls riskier)
                 + 0.015 * sigmoid(required_rate - 10)     (chase pressure, inns 2)
    with base_by_phase = {powerplay 0.030, middle 0.033, death 0.055}, clipped to
    [0.005, 0.12]. A calibrated situation-only xW must track this; it uses no
    outcome columns, so it must land at modest accuracy with a good calibration
    curve, not the 93-97% that signals leakage.
  * Two validated design fixes are adopted:
      - Wickets are drawn from a SEPARATE rng stream, so dismissal risk is not
        coupled to the six/scoring draws (which would manufacture a spurious
        wicket-vs-aggression link).
      - A per-innings form multiplier ~ Normal(1.0, 0.18) scales six/four prob and
        the run mean for that innings, giving innings totals a realistic spread that
        reaches past 220 for the tail behaviour.
  * Powerplay six probability rises 0.035 -> 0.052 across seasons (a level answer
    key inherited from the stub's style; independent of the boundary move).

Leakage decoys (`wprob`, `predscore`) are emitted post-ball so the xW whitelist has
something to (correctly) exclude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = [f"Team{i:02d}" for i in range(10)]

AGGRESSIVE_SHOTS = ["PULL", "SLOG_SHOT", "SLOG_SWEEP", "HOOK", "UPPER_CUT", "RAMP", "REVERSE_SCOOP", "REVERSE_PULL"]
ORTHODOX_SHOTS = ["DEFENDED", "PUSH", "DRIVE", "SQUARE_DRIVE", "COVER_DRIVE", "SWEEP", "CUT", "GLANCE", "FLICK", "LEAVE"]

POWERPLAY = (1, 6)
MIDDLE = (7, 15)
DEATH = (16, 20)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _phase_of(over_num: int) -> str:
    if over_num <= POWERPLAY[1]:
        return "powerplay"
    if over_num <= MIDDLE[1]:
        return "middle"
    return "death"


def _scoring_profile(over_num: int, season_pp_six: float, death_edge_over: int, form: float) -> dict:
    """Per-faced-ball scoring propensities as smooth functions of the over.

    A genuine death-boundary move shifts ALL death-associated behaviour earlier
    together: the scoring ramp, the aggression ramp, AND the pace-back-to-death
    rise. So every death-anchored channel here keys off `death_edge_over`, and
    moving it between seasons plants a coherent multivariate boundary move (not just
    a level change inside a fixed death window). `form` scales scoring for the tail.
    """
    o = float(over_num)
    # Powerplay hump, fixed location (overs 1-6): this boundary does NOT move.
    powerplay = np.exp(-((o - 3.0) ** 2) / 6.0)
    # Death onset, centred half an over below the edge so the steepest step falls
    # between (edge-1) and edge; this whole onset slides when death_edge_over moves.
    death = 1.0 / (1.0 + np.exp(-1.6 * (o - (death_edge_over - 0.5))))
    # Pace: high with the new ball, dips through the middle, rises back at the death;
    # the death rise is anchored to the SAME moving edge, so a boundary move is coherent.
    early_pace = 1.0 / (1.0 + np.exp(-1.5 * (6.5 - o)))

    six_base = 0.022 + (season_pp_six - 0.022 + 0.010) * powerplay + 0.10 * death
    four_base = 0.050 + 0.030 * powerplay + 0.055 * death
    run_mean = 1.00 + 0.18 * powerplay + 0.85 * death
    aggressive_share = 0.10 + 0.06 * powerplay + 0.36 * death
    pace_share = 0.40 + 0.42 * early_pace + 0.33 * death

    return {
        "p_six": float(np.clip(six_base * form, 0.0, 0.30)),
        "p_four": float(np.clip(four_base * form, 0.0, 0.30)),
        "run_mean": float(run_mean * form),
        "aggressive_share": float(np.clip(aggressive_share, 0.02, 0.75)),
        "pace_share": float(np.clip(pace_share, 0.30, 0.92)),
    }


def _wicket_hazard(over_num: int, balls_bowled_legal: int, max_balls: int,
                   required_rate: float | None) -> float:
    """Situation-only dismissal hazard for a faced ball (the xW answer key)."""
    base = {"powerplay": 0.030, "middle": 0.033, "death": 0.055}[_phase_of(over_num)]
    frac_bowled = balls_bowled_legal / max_balls if max_balls else 0.0
    hazard = base + 0.020 * frac_bowled
    if required_rate is not None:
        hazard += 0.015 * _sigmoid(required_rate - 10.0)
    return float(np.clip(hazard, 0.005, 0.12))


def make_synthetic_ballframe(
    n_matches_per_season: int = 70,
    seasons=(2023, 2024),
    death_edge_by_season=None,
    pp_six_by_season=None,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a real-schema ball-by-ball frame with the answer-key effects injected."""
    rng = np.random.default_rng(seed)
    wicket_rng = np.random.default_rng(seed + 101)  # separate stream (validated fix)

    if death_edge_by_season is None:
        death_edge_by_season = {seasons[0]: 16, seasons[-1]: 15}
    if pp_six_by_season is None:
        pp_six_by_season = {seasons[0]: 0.035, seasons[-1]: 0.052}

    rows = []
    for season in seasons:
        edge = death_edge_by_season.get(season, 16)
        pp_six = pp_six_by_season.get(season, 0.035)
        for n in range(1, n_matches_per_season + 1):
            p_match = season * 1000 + n
            bat_first, bowl_first = rng.choice(TEAMS, size=2, replace=False)
            first_innings_total = None

            for inns in (1, 2):
                team_bat = bat_first if inns == 1 else bowl_first
                team_bowl = bowl_first if inns == 1 else bat_first
                max_balls = 120 if rng.random() > 0.08 else int(rng.choice([90, 96, 108]))
                overs_available = max_balls // 6
                form = float(np.clip(rng.normal(1.0, 0.18), 0.4, 1.9))  # per-innings form (validated fix)
                target = (first_innings_total + 1) if (inns == 2 and first_innings_total is not None) else None

                wkts = 0
                balls_bowled_legal = 0
                runs_so_far = 0

                for over_num in range(1, overs_available + 1):
                    legal_this_over = 0
                    ball_in_over = 0
                    while legal_this_over < 6:
                        ball_in_over += 1
                        is_wide = rng.random() < 0.03
                        is_noball = (not is_wide) and (rng.random() < 0.012)

                        # Pre-ball situation (strictly before this delivery resolves).
                        inns_wkts_preball = wkts
                        inns_balls_rem_preball = max_balls - balls_bowled_legal
                        required_rate = None
                        if target is not None and inns_balls_rem_preball > 0:
                            required_rate = (target - runs_so_far) / inns_balls_rem_preball * 6.0

                        prof = _scoring_profile(over_num, pp_six, edge, form)
                        bowl_kind = _draw_bowl_kind(rng, prof["pace_share"])
                        bat_hand = "LHB" if rng.random() < 0.32 else "RHB"

                        batruns = 0
                        wide_runs = 0
                        noball_flag = 0
                        byes = 0
                        legbyes = 0
                        out = 0
                        dismissal = ""
                        shot = ""
                        control = 1
                        ballfaced = 0

                        if is_wide:
                            wide_runs = int(1 + rng.poisson(0.12))
                            ballfaced = 0
                            outcome = "wide"
                        else:
                            ballfaced = 1  # a no-ball is still faced
                            noball_flag = 1 if is_noball else 0
                            # Wicket from the SEPARATE stream, situation-only hazard.
                            hazard = _wicket_hazard(over_num, balls_bowled_legal, max_balls, required_rate)
                            if (not is_noball) and (wicket_rng.random() < hazard):
                                out = 1
                                dismissal = rng.choice(["caught", "bowled", "lbw", "run out", "stumped"])
                                batruns = 0
                                control = 0
                                shot = rng.choice(AGGRESSIVE_SHOTS + ORTHODOX_SHOTS)
                                outcome = "out"
                            else:
                                draw = rng.random()
                                if draw < prof["p_six"]:
                                    batruns = 6
                                    outcome = "six"
                                elif draw < prof["p_six"] + prof["p_four"]:
                                    batruns = 4
                                    outcome = "four"
                                else:
                                    batruns = int(np.clip(rng.poisson(prof["run_mean"]), 0, 3))
                                    outcome = "no run" if batruns == 0 else "run"
                                # Shot + control: aggressive shots mishit more.
                                play_aggressive = rng.random() < prof["aggressive_share"]
                                if play_aggressive:
                                    shot = rng.choice(AGGRESSIVE_SHOTS)
                                    control = 0 if rng.random() < 0.46 else 1
                                else:
                                    shot = rng.choice(ORTHODOX_SHOTS)
                                    control = 0 if rng.random() < 0.22 else 1
                                # Occasional byes / leg-byes on a no-score ball (populated, unlike the stub).
                                if batruns == 0 and rng.random() < 0.04:
                                    if rng.random() < 0.4:
                                        byes = int(rng.choice([1, 2, 4], p=[0.7, 0.25, 0.05]))
                                        outcome = "bye"
                                    else:
                                        legbyes = int(rng.choice([1, 2], p=[0.85, 0.15]))
                                        outcome = "leg bye"
                            if is_noball:
                                outcome = "no ball"

                        legal = not (is_wide or is_noball)
                        noball_runs = 1 if is_noball else 0
                        score = batruns + wide_runs + noball_runs + byes + legbyes
                        runs_so_far += score
                        bowler_charge = float(batruns + wide_runs + noball_runs)  # not byes/legbyes
                        control_clean = np.nan if is_wide else float(control)

                        rows.append({
                            "p_match": p_match, "year": season, "inns": inns,
                            "over_num": over_num, "ball_in_over": ball_in_over,
                            "team_bat": team_bat, "team_bowl": team_bowl,
                            "batruns": int(batruns), "score": int(score),
                            "ballfaced": int(ballfaced), "outcome": outcome,
                            "out": int(out), "dismissal": dismissal,
                            "wide": int(wide_runs), "noball": int(noball_flag),
                            "byes": int(byes), "legbyes": int(legbyes),
                            "bowler_charge": bowler_charge,
                            "shot": shot, "control": int(control), "control_clean": control_clean,
                            "line": rng.choice(["off", "middle", "leg", "wide_line"]),
                            "length": rng.choice(["full", "good", "short", "yorker"]),
                            "bowl_type": rng.choice(["RF", "RM", "OB", "LB", "SLA"]),
                            "bowl_kind": bowl_kind, "bat_hand": bat_hand,
                            "inns_wkts": int(inns_wkts_preball),
                            "inns_balls_rem": int(inns_balls_rem_preball),
                            "target": (float(target) if target is not None else np.nan),
                            "max_balls": int(max_balls),
                            "wprob": float(np.clip(rng.normal(0.5, 0.2), 0.01, 0.99)),
                            "predscore": float(runs_so_far + rng.normal(60, 20)),
                        })

                        if legal:
                            legal_this_over += 1
                            balls_bowled_legal += 1
                        if out:
                            wkts += 1
                        if wkts >= 10 or (target is not None and runs_so_far >= target):
                            break
                    if wkts >= 10 or (target is not None and runs_so_far >= target):
                        break
                if inns == 1:
                    first_innings_total = runs_so_far

    return pd.DataFrame(rows)


def _draw_bowl_kind(rng: np.random.Generator, pace_share: float) -> str:
    """Draw a real-vocabulary bowl_kind. ~0.3% mixture/unknown, the rest pace/spin."""
    if rng.random() < 0.003:
        return "mixture/unknown"
    return "pace bowler" if rng.random() < pace_share else "spin bowler"


if __name__ == "__main__":
    df = make_synthetic_ballframe()
    print("rows:", len(df), "| matches:", df["p_match"].nunique())
    print("columns:", list(df.columns))
    print("over_num range:", (df["over_num"].min(), df["over_num"].max()))
    print("wicket rate (faced):", round(df.loc[df.ballfaced == 1, "out"].mean(), 4))
    print(df.groupby("year")["bowl_kind"].value_counts())
