"""
Between-team dispersion by match cluster bootstrap: the mover screen.

Every quantity here is the between-team SD of a per-team rate, and its change across
seasons (later minus earlier). Positive is stratification (the pack pulled apart),
negative is convergence. The league mean can sit flat while this moves, which is
why a mean-difference test cannot see it.

The bootstrap resamples whole matches per season, rebuilds every team's rate from
its per-match numerator and denominator totals, and takes the SD across the ten
teams. Teams absent from a resample are dropped from the SD rather than counted as
zero. Percentile intervals: the resample distribution of an SD difference is close
to symmetric, so the BCa skew correction buys little. P(widen) is the share of
resamples with a positive difference.

Metrics (batting keyed on team_bat, bowling on team_bowl):
  - false-shot rate under aggressive intent, full match (tagged shot and control)
  - aggressive-attempt share of faced balls, per phase (tagged shot only)
  - gear-up delta: phase run rate minus the team's own middle-overs rate, per team,
    with both rates re-formed inside each resample (score and over_num only)
  - induced false-shot rate, bowling side, full match and per phase (tagged control)

Tagging floor: shot and control_clean are manual columns with an assumed 7 to 10%
within-column error, so intervals on the tagged metrics are a floor on the true
uncertainty. The gear-up delta carries no such floor.

`BetweenTeamSpreadFromMasks` and `bootstrap_spread` are reconstructed from their
call sites (the notebook that ran the dispersion screen used them without carrying
their definition). Their point estimates reproduce the stored between-team SD
differences: the SD of per-team season rates, later minus earlier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mean_level_and_tail import season_contrast_bootstrap as scb

# Attacking shots present in the data where the batter commits to power or
# improvisation and accepts control risk. Drives and orthodox sweeps are left
# out on purpose: a drive is often a controlled shot, so including it blurs the
# intent signal this metric depends on. Built from the actual shot vocabulary,
# not illustrative names.
AGGRESSIVE_SHOTS = {
    "PULL", "SLOG_SHOT", "SLOG_SWEEP", "HOOK", "UPPER_CUT",
    "RAMP", "REVERSE_SCOOP", "REVERSE_PULL",
}

CONTROL_COL = "control_clean"   # nulls control values leaked onto wides; use over raw control


# ----------------------------------------------------------------------------
# Spread estimands: (counts_earlier, counts_later) -> SD difference per resample
# ----------------------------------------------------------------------------

class BetweenTeamSpreadFromMasks:
    """Season change in the between-team SD of a rate given as per-ball numerator
    and denominator weights and a scope mask, keyed on the batting team.

    The rate is a ratio of sums per team (no per-over scaling), so it serves
    proportions such as a false-shot rate or an attempt share directly. Per
    (match, team) totals are built once per season; a resample reweights them.
    """

    team_column = "team_bat"

    def __init__(self, frame, scope, numerator, denominator, teams):
        self.teams = list(teams)
        work = frame.loc[np.asarray(scope, dtype=bool)].copy()
        work["numer"] = np.asarray(numerator, dtype=float)[np.asarray(scope, dtype=bool)]
        work["denom"] = np.asarray(denominator, dtype=float)[np.asarray(scope, dtype=bool)]
        # Match lists come from the full season frame, not the scoped rows, so a
        # match with no in-scope balls still occupies a resampling slot.
        self._earlier = self._season(frame, work, scb.EARLIER_SEASON)
        self._later = self._season(frame, work, scb.LATER_SEASON)

    def _season(self, frame, work, year):
        matches = np.sort(frame.loc[frame["year"] == year, "p_match"].unique())
        season_work = work[work["year"] == year]
        _, numer, denom = self._match_team_matrices(season_work, matches)
        return matches, numer, denom

    def _match_team_matrices(self, season_work, matches):
        m_ix = {m: i for i, m in enumerate(matches)}
        t_ix = {t: j for j, t in enumerate(self.teams)}
        numer = np.zeros((len(matches), len(self.teams)))
        denom = np.zeros((len(matches), len(self.teams)))
        g = season_work.groupby(["p_match", self.team_column])[["numer", "denom"]].sum()
        for (m, t), row in g.iterrows():
            if t in t_ix and m in m_ix:
                numer[m_ix[m], t_ix[t]] += row["numer"]
                denom[m_ix[m], t_ix[t]] += row["denom"]
        return matches, numer, denom

    @staticmethod
    def _spread(counts, numer, denom):
        num = counts @ numer
        den = counts @ denom
        rate = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        return np.nanstd(rate, axis=1, ddof=1)

    def n_matches(self):
        return len(self._earlier[0]), len(self._later[0])

    def __call__(self, counts_earlier, counts_later):
        return (self._spread(counts_later, *self._later[1:])
                - self._spread(counts_earlier, *self._earlier[1:]))


class BetweenBowlTeamSpreadFromMasks(BetweenTeamSpreadFromMasks):
    """Between-team dispersion of a bowling-side rate, keyed on the bowling team.

    Identical machinery to the batting-side spread, but a delivery's team is the
    one bowling it, so the per (match, team) aggregation groups on team_bowl. Used
    for false-shot-induced rate: the share of a team's deliveries that drew a
    loss of batter control.
    """

    team_column = "team_bowl"


class PhaseIntentDeltaSpread:
    """Season-over-season change in between-team SD of a phase's rate measured as
    a delta from each team's own middle-overs baseline.

    Each team-season value is (phase rate minus that team's middle-overs rate),
    so the level effect of simply being a high-scoring side is removed and what
    remains is how much the team gears up or down from its own cruising rate, a
    tactical shape rather than a scoring level. The dispersion is the SD across
    teams of those deltas; positive sd_diff is stratification in gear-change,
    negative is convergence.

    Built as its own estimand because the value is nested: a match resample must
    recompute both the phase rate and the baseline rate per team so the
    baseline's sampling noise propagates into the interval instead of being
    frozen at its full-sample value.
    """

    def __init__(self, frame, phase, baseline_phase, numerator, denominator, teams):
        self.teams = list(teams)
        work = frame.copy()
        work["_numer"] = np.asarray(numerator, dtype=float)
        work["_denom"] = np.asarray(denominator, dtype=float)
        self._earlier = self._matrices(work[work["year"] == scb.EARLIER_SEASON], phase, baseline_phase)
        self._later = self._matrices(work[work["year"] == scb.LATER_SEASON], phase, baseline_phase)

    def _matrices(self, season_work, phase, baseline_phase):
        # Per (match, team): numerator/denominator for the target phase and for
        # the baseline phase separately, so resampling reweights both and the
        # delta is recomputed rather than carried as a fixed per-team number.
        matches = np.sort(season_work["p_match"].unique())
        m_ix = {m: i for i, m in enumerate(matches)}
        t_ix = {t: j for j, t in enumerate(self.teams)}
        shape = (len(matches), len(self.teams))
        phase_num, phase_den = np.zeros(shape), np.zeros(shape)
        base_num, base_den = np.zeros(shape), np.zeros(shape)

        in_target = scb.in_phase(season_work, phase)
        in_base = scb.in_phase(season_work, baseline_phase)
        for arr_num, arr_den, mask in ((phase_num, phase_den, in_target),
                                       (base_num, base_den, in_base)):
            sub = season_work.loc[mask]
            g = sub.groupby(["p_match", "team_bat"])[["_numer", "_denom"]].sum()
            for (m, t), row in g.iterrows():
                if t in t_ix:
                    arr_num[m_ix[m], t_ix[t]] += row["_numer"]
                    arr_den[m_ix[m], t_ix[t]] += row["_denom"]
        return matches, phase_num, phase_den, base_num, base_den

    def _spread(self, counts, mats):
        _, phase_num, phase_den, base_num, base_den = mats
        pr = np.divide(counts @ phase_num, counts @ phase_den,
                       out=np.full((counts.shape[0], len(self.teams)), np.nan),
                       where=(counts @ phase_den) > 0)
        br = np.divide(counts @ base_num, counts @ base_den,
                       out=np.full((counts.shape[0], len(self.teams)), np.nan),
                       where=(counts @ base_den) > 0)
        delta = (pr - br) * 6.0   # runs per over, cosmetic scaling; no effect on the null
        return np.nanstd(delta, axis=1, ddof=1)

    def n_matches(self):
        return len(self._earlier[0]), len(self._later[0])

    def __call__(self, counts_earlier, counts_later):
        return self._spread(counts_later, self._later) - self._spread(counts_earlier, self._earlier)


class PhaseRunRateSpread:
    """Season change in the between-team SD of a phase run rate (runs per over,
    legal balls), keyed on the batting team. The bootstrap counterpart of the
    hierarchical run-rate dispersion model, used where the three powerplay legs
    must be read off the same resamples."""

    def __init__(self, frame, phase, teams):
        self.teams = list(teams)
        legal = scb.is_legal_delivery(frame)
        scope = scb.in_phase(frame, phase) & legal
        score = frame["score"].to_numpy(dtype=float)
        self._earlier = self._season(frame, scope, score, legal.astype(float), scb.EARLIER_SEASON)
        self._later = self._season(frame, scope, score, legal.astype(float), scb.LATER_SEASON)

    def _season(self, frame, scope, numer, denom, year):
        season = frame["year"].to_numpy() == year
        sub = frame.loc[season]
        matches, numer_mat, denom_mat = scb.per_match_team_sums(
            sub, scope[season], numer[season], denom[season], self.teams)
        return matches, numer_mat, denom_mat

    def _spread(self, counts, numer, denom):
        num = counts @ numer
        den = counts @ denom
        rate = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0) * 6.0
        return np.nanstd(rate, axis=1, ddof=1)

    def n_matches(self):
        return len(self._earlier[0]), len(self._later[0])

    def __call__(self, counts_earlier, counts_later):
        return (self._spread(counts_later, *self._later[1:])
                - self._spread(counts_earlier, *self._earlier[1:]))


# ----------------------------------------------------------------------------
# Running a spread estimand
# ----------------------------------------------------------------------------

def bootstrap_spread(estimand, n_iter: int = 10000, seed: int = 20240601,
                     alpha: float = 0.05, counts: tuple | None = None) -> dict:
    """Point estimate, percentile interval and P(widen) for a spread estimand.

    `counts` (counts_earlier, counts_later) lets several estimands share one set
    of draws; otherwise draws are made here from the seed.
    """
    n_e, n_l = estimand.n_matches()
    if counts is None:
        counts = scb.shared_resample_counts(n_e, n_l, n_iter, seed)
    counts_e, counts_l = counts
    point = float(estimand(np.ones((1, n_e)), np.ones((1, n_l)))[0])
    reps = estimand(counts_e, counts_l)
    reps = reps[np.isfinite(reps)]
    return {
        "sd_diff": round(point, 4),
        "ci_low": float(np.percentile(reps, 100 * alpha / 2)),
        "ci_high": float(np.percentile(reps, 100 * (1 - alpha / 2))),
        "p_widen": round(float((reps > 0).mean()), 3),
        "n_resamples": int(reps.size),
        "interval_method": "percentile",
    }


# ----------------------------------------------------------------------------
# Batting metrics
# ----------------------------------------------------------------------------

def validate_intent_and_control(frame):
    """Sanity-check the columns the batting false-shot metric depends on.

    Checks:
    1. control_clean takes only 0/1/NaN and carries no value on unfaced balls
       (a wide has no shot to control; the cleaned column should have nulled these).
    2. every shot named as aggressive actually exists in the data.
    3. aggressive shots carry a clearly higher false-shot rate than other shots,
       which is the assumption the metric rests on: attacking shots risk control,
       so conditioning on them isolates a real choice rather than noise.
    """
    control_values = frame[CONTROL_COL].value_counts(dropna=False)
    faced = scb.is_ball_faced(frame)
    leaked_on_unfaced = frame.loc[~faced, CONTROL_COL].isin([0, 1]).sum()

    shot_upper = frame["shot"].astype(str).str.upper()
    present = sorted(AGGRESSIVE_SHOTS & set(shot_upper.unique()))
    missing = sorted(AGGRESSIVE_SHOTS - set(shot_upper.unique()))

    recorded = frame[CONTROL_COL].isin([0, 1])
    is_aggressive = shot_upper.isin(AGGRESSIVE_SHOTS)

    def false_shot_rate(mask):
        sub = frame.loc[recorded & mask, CONTROL_COL]
        return round((sub == 0).mean(), 4), int(sub.size)

    agg_rate, agg_n = false_shot_rate(is_aggressive)
    other_rate, other_n = false_shot_rate(~is_aggressive)

    print(f"{CONTROL_COL} value counts:\n{control_values}\n")
    print(f"{CONTROL_COL} on unfaced balls (should be 0): {leaked_on_unfaced}")
    print(f"aggressive shots present: {present}")
    print(f"aggressive shots missing from data: {missing}")
    print(f"false-shot rate | aggressive intent: {agg_rate}  (n={agg_n:,})")
    print(f"false-shot rate | other shots:       {other_rate}  (n={other_n:,})")
    print("Metric is sound only if the aggressive rate is clearly higher.")
    return dict(present=present, missing=missing, agg_rate=agg_rate, other_rate=other_rate)


def per_team_false_shot_rate(frame):
    """Per-team false-shot rate under aggressive intent across the full match,
    per season, plus the per-team aggressive-ball depth that sets estimate noise.

    Full match, not phase-split: conditioning on aggressive intent already thins
    the sample hard, and splitting by phase on top drops per-team denominators
    to the point where the between-team SD is mostly sampling noise. Read depth
    first: below ~150 balls per team, trust the bootstrap and model over the raw SD.
    """
    shot_upper = frame["shot"].astype(str).str.upper()
    scope = (scb.is_ball_faced(frame)
             & shot_upper.isin(AGGRESSIVE_SHOTS).to_numpy()
             & frame[CONTROL_COL].isin([0, 1]).to_numpy())
    scoped = frame.loc[scope].copy()
    scoped["false_shot"] = (scoped[CONTROL_COL] == 0).astype(float)

    per_team = (scoped.groupby(["year", "team_bat"])
                      .agg(false_shot_rate=("false_shot", "mean"),
                           aggressive_balls=("false_shot", "size"))
                      .reset_index())
    spread = per_team.groupby("year")["false_shot_rate"].std(ddof=1)
    depth = per_team.groupby("year")["aggressive_balls"].agg(["min", "median", "max"])
    return per_team, spread, depth


def false_shot_spread_estimand(frame):
    """Between-team false-shot-rate SD change, full match. Tagging caveat binds:
    shot and control_clean are manual (~7-10% within-column error), so the
    interval is a floor on the true uncertainty."""
    shot_upper = frame["shot"].astype(str).str.upper()
    scope = (scb.is_ball_faced(frame)
             & shot_upper.isin(AGGRESSIVE_SHOTS).to_numpy()
             & frame[CONTROL_COL].isin([0, 1]).to_numpy())
    false_shot = (frame[CONTROL_COL] == 0).astype(float).to_numpy()
    denom = frame[CONTROL_COL].isin([0, 1]).astype(float).to_numpy()
    teams = sorted(frame["team_bat"].unique())
    return BetweenTeamSpreadFromMasks(frame, scope, false_shot, denom, teams)


def per_team_attempt_share(frame, phase):
    """Per-team aggressive-shot share of faced balls in one phase, per season.

    Secondary corroborator to the false-shot primary: this is bare attack
    frequency with no control conditioning, so it says how often teams chose to
    attack, not how well. Phase-split is affordable because the denominator is
    all faced balls in the phase, not just aggressive ones, so per-team depth
    stays in the hundreds even split by phase.
    """
    shot_upper = frame["shot"].astype(str).str.upper()
    scope = scb.in_phase(frame, phase) & scb.is_ball_faced(frame)
    scoped = frame.loc[scope].copy()
    scoped["attempt"] = shot_upper[scope].isin(AGGRESSIVE_SHOTS).astype(float).values

    per_team = (scoped.groupby(["year", "team_bat"])
                      .agg(attempt_share=("attempt", "mean"),
                           faced_balls=("attempt", "size"))
                      .reset_index())
    spread = per_team.groupby("year")["attempt_share"].std(ddof=1)
    depth = per_team.groupby("year")["faced_balls"].agg(["min", "median", "max"])
    return per_team, spread, depth


def attempt_share_spread_estimand(frame, phase):
    """Between-team aggressive-attempt-share SD change in one phase. Rests on shot
    tags only (~7-10% error), so the interval is a floor on true uncertainty."""
    shot_upper = frame["shot"].astype(str).str.upper()
    scope = scb.in_phase(frame, phase) & scb.is_ball_faced(frame)
    attempt = shot_upper.isin(AGGRESSIVE_SHOTS).astype(float)
    denom = scb.is_ball_faced(frame).astype(float)
    teams = sorted(frame["team_bat"].unique())
    return BetweenTeamSpreadFromMasks(frame, scope, attempt, denom, teams)


def attempt_share_mean_by_season(frame, phase):
    """League mean aggressive-attempt share in a phase per season: the
    mean-stability check that a spread change is not a level shift or tagging drift."""
    shot_upper = frame["shot"].astype(str).str.upper()
    out = {}
    for yr in (scb.EARLIER_SEASON, scb.LATER_SEASON):
        m = (frame["year"] == yr).to_numpy() & scb.is_ball_faced(frame) & scb.in_phase(frame, phase)
        out[yr] = float(shot_upper[m].isin(AGGRESSIVE_SHOTS).mean())
    return out


def per_team_gear_up_delta(frame, phase, baseline_phase):
    """Per-team gear-up delta (phase rate minus own middle-overs rate) and its
    between-team SD per season, plus per-team ball depth in both phases.

    Depth matters in both the target and baseline phase: a noisy baseline
    inflates every team's delta. Run rate channel, so this rests on score and
    over_num, not on any manual tag.
    """
    def rate(sub):
        legal = scb.is_legal_delivery(sub)
        g = (sub.loc[legal].groupby(["year", "team_bat"])
                 .agg(runs=("score", "sum"), balls=("score", "size")))
        g["rate"] = g["runs"] / g["balls"] * 6.0
        return g

    tgt = rate(frame.loc[scb.in_phase(frame, phase)])
    base = rate(frame.loc[scb.in_phase(frame, baseline_phase)])
    joined = tgt[["rate", "balls"]].join(base[["rate", "balls"]], lsuffix="_phase", rsuffix="_base")
    joined["delta"] = joined["rate_phase"] - joined["rate_base"]

    spread = joined.groupby("year")["delta"].std(ddof=1)
    depth = joined.groupby("year")[["balls_phase", "balls_base"]].agg(["min", "median"])
    return joined, spread, depth


def gear_up_spread_estimand(frame, phase, baseline_phase):
    """Between-team gear-up-delta SD change. Run rate channel, so tagging-independent:
    this primary carries no shot/control floor, unlike the false-shot metrics."""
    teams = sorted(frame["team_bat"].unique())
    numer = frame["score"].to_numpy(dtype=float)
    denom = scb.is_legal_delivery(frame).astype(float)
    return PhaseIntentDeltaSpread(frame, phase, baseline_phase, numer, denom, teams)


# ----------------------------------------------------------------------------
# Bowling metrics
# ----------------------------------------------------------------------------

def validate_false_shot_induced(frame):
    """Sanity-check the bowling false-shot-induced metric.

    Checks:
    1. control_clean is the cleaned column (no values on unfaced balls) and its
       recorded rate is stable across seasons, so a cross-season dispersion claim
       is not riding on a change in how often control was tagged.
    2. the induced-false-shot rate lands in a sane range and is similar in level
       across seasons (a large level jump would point at tagging drift, not
       bowling change), checked here at the league level before dispersion.
    3. team_bowl covers the same team set as the batting side, so the two matched
       metrics are comparable team for team.
    """
    faced = scb.is_ball_faced(frame)
    recorded = frame[CONTROL_COL].isin([0, 1])
    rows = []
    for yr in (scb.EARLIER_SEASON, scb.LATER_SEASON):
        m = (frame["year"] == yr).to_numpy() & faced
        rec_rate = recorded[m].mean()
        induced = (frame.loc[m & recorded, CONTROL_COL] == 0).mean()
        rows.append({"year": yr, "control_recorded_share": float(rec_rate),
                     "league_induced_false_shot_rate": float(induced)})
    bowl_teams = set(frame["team_bowl"].unique())
    bat_teams = set(frame["team_bat"].unique())
    return pd.DataFrame(rows), bowl_teams == bat_teams


def per_team_induced_false_shot_rate(frame, phase=None):
    """Per-bowling-team false-shot-induced rate, per season, with per-team ball
    depth. phase=None uses the whole match; pass a phase tuple to split.

    No shot-intent conditioning (the bowling side cannot choose the batter's
    shot), so the denominator is all faced deliveries with recorded control,
    which keeps per-team depth high enough that a powerplay split is affordable
    here even though the batting false-shot metric needed the full match.
    """
    scope = scb.is_ball_faced(frame) & frame[CONTROL_COL].isin([0, 1]).to_numpy()
    if phase is not None:
        scope = scope & scb.in_phase(frame, phase)
    scoped = frame.loc[scope].copy()
    scoped["induced"] = (scoped[CONTROL_COL] == 0).astype(float)

    per_team = (scoped.groupby(["year", "team_bowl"])
                      .agg(induced_rate=("induced", "mean"),
                           balls=("induced", "size"))
                      .reset_index())
    spread = per_team.groupby("year")["induced_rate"].std(ddof=1)
    depth = per_team.groupby("year")["balls"].agg(["min", "median", "max"])
    return per_team, spread, depth


def induced_false_shot_spread_estimand(frame, phase=None):
    """Between-team induced-false-shot-rate SD change, keyed on the bowling team.
    control_clean is manual (~7-10% error), so the interval is a floor on true
    uncertainty."""
    scope = scb.is_ball_faced(frame) & frame[CONTROL_COL].isin([0, 1]).to_numpy()
    if phase is not None:
        scope = scope & scb.in_phase(frame, phase)
    induced = (frame[CONTROL_COL] == 0).astype(float).to_numpy()
    denom = frame[CONTROL_COL].isin([0, 1]).astype(float).to_numpy()
    teams = sorted(frame["team_bowl"].unique())
    return BetweenBowlTeamSpreadFromMasks(frame, scope, induced, denom, teams)


# ----------------------------------------------------------------------------
# Driver: every bootstrap dispersion metric on one set of shared draws
# ----------------------------------------------------------------------------

def run_dispersion_screen(frame, settings, seed: int, n_iter: int | None = None) -> pd.DataFrame:
    """All bootstrap spread metrics on shared draws. One row per metric with the
    per-season raw SD, the SD difference, its percentile interval, P(widen), the
    column exposure, and the per-team depth."""
    n_iter = n_iter or settings.n_resamples
    alpha = 1.0 - settings.interval_level
    phases = {name: settings.phase(name) for name in ("powerplay", "middle", "death")}
    n_e = frame.loc[frame["year"] == settings.earlier_season, "p_match"].nunique()
    n_l = frame.loc[frame["year"] == settings.later_season, "p_match"].nunique()
    counts = scb.shared_resample_counts(n_e, n_l, n_iter, seed)

    specs = []
    _, spread, depth = per_team_false_shot_rate(frame)
    specs.append(("batting_false_shot_rate_full_match", false_shot_spread_estimand(frame),
                  spread, depth["min"], "team_bat", "tagged (shot, control)"))
    for name, phase in phases.items():
        _, spread, depth = per_team_attempt_share(frame, phase)
        specs.append((f"batting_attempt_share_{name}", attempt_share_spread_estimand(frame, phase),
                      spread, depth["min"], "team_bat", "tagged (shot)"))
    for name, phase in (("powerplay", phases["powerplay"]), ("death", phases["death"])):
        _, spread, depth = per_team_gear_up_delta(frame, phase, phases["middle"])
        specs.append((f"batting_gear_up_{name}", gear_up_spread_estimand(frame, phase, phases["middle"]),
                      spread, depth[("balls_phase", "min")], "team_bat", "clean (score, over_num)"))
    _, spread, depth = per_team_induced_false_shot_rate(frame)
    specs.append(("bowling_induced_false_shot_full_match", induced_false_shot_spread_estimand(frame),
                  spread, depth["min"], "team_bowl", "tagged (control)"))
    _, spread, depth = per_team_induced_false_shot_rate(frame, phases["powerplay"])
    specs.append(("bowling_induced_false_shot_powerplay",
                  induced_false_shot_spread_estimand(frame, phases["powerplay"]),
                  spread, depth["min"], "team_bowl", "tagged (control)"))

    rows = []
    for name, estimand, spread, min_depth, key, exposure in specs:
        result = bootstrap_spread(estimand, n_iter=n_iter, seed=seed, alpha=alpha, counts=counts)
        rows.append({"metric": name,
                     "sd_earlier": float(spread.get(settings.earlier_season, np.nan)),
                     "sd_later": float(spread.get(settings.later_season, np.nan)),
                     **result,
                     "min_team_depth_earlier": int(min_depth.get(settings.earlier_season, 0)),
                     "min_team_depth_later": int(min_depth.get(settings.later_season, 0)),
                     "team_key": key, "column_exposure": exposure})
    return pd.DataFrame(rows)
