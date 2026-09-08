"""
Cluster bootstrap harness for the IPL 2023-vs-2024 tactical-evolution themes.

Every interval this module produces answers one question: if we had drawn a
different set of matches from the same season, how much would the observed
season-to-season effect have moved? That is resampling stability at the
precision the data affords. It is NOT out-of-sample replication: these themes
were found by EDA on exactly these matches, so a stable interval says the
effect survives reshuffling the matches we have, not that it would recur in a
fresh season.

Why the match is the resampling unit
------------------------------------
Deliveries within an innings share pitch, matchup, and game state, so they are
positively correlated. Resampling deliveries independently would treat that
shared variation as if it were independent information, understate the variance,
and hand out significance that one good night from two teams could erase. The
unit of roughly independent variation is the match, so we resample whole matches
with replacement and carry each match's deliveries in as a block. Each season is
resampled from its own match pool so the season's match count is preserved and
the cross-season contrast keeps its real structure.

Why differences are the primary estimand
----------------------------------------
The primary estimand is the difference of a rate between seasons (2024 minus
2023, null zero); a difference of two ratio-of-sums rates is close to unbiased
under the match bootstrap. A ratio (2024 over 2023, null one) is reported only
where the claim is inherently relative. Ratios are estimated on the log scale,
where the resample distribution is closer to symmetric and the bootstrap bias
correction behaves well; intervals are back-transformed for reporting, and the
difference is always reported alongside a ratio.

How intervals are built
-----------------------
Every rate here is a ratio of per-match sums, so its cluster standard error has
a closed form (delta method over match influence values). That gives a
studentized (bootstrap-t) interval, which is the primary interval for rates,
rate differences, and linear contrasts of rates. BCa (bias-corrected and
accelerated) is reported alongside, and is the primary interval where no
closed-form SE exists: quantile shifts and between-team SD differences. The
BCa acceleration uses a jackknife stratified by season, because the bootstrap
itself is stratified by season and the two-sample statistic needs the
leave-one-out values centred within their own season.

What the intervals cannot widen for
-----------------------------------
The bootstrap resamples the observed manual tags (line, length, shot,
bowl_type) and treats them as truth. For any theme resting on those columns the
sampling interval is a floor on the true uncertainty, not the whole of it: the
honest interval is this one widened by tagging error we cannot fully quantify.
Themes carrying that exposure are flagged in their uncertainty_note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Callable, Sequence

import numpy as np
import pandas as pd

# Phase boundaries in over number (1-indexed), matching the season's structure.
# These are the cricket phases (powerplay / middle / death), not analysis steps.
POWERPLAY_OVERS = (1, 6)
MIDDLE_OVERS = (7, 15)
DEATH_OVERS = (16, 20)

EARLIER_SEASON = 2023
LATER_SEASON = 2024

_NORMAL = NormalDist()


# ----------------------------------------------------------------------------
# Scope and per-ball weight helpers
# ----------------------------------------------------------------------------

def in_phase(frame: pd.DataFrame, phase: tuple[int, int]) -> np.ndarray:
    """Boolean mask for deliveries whose over falls inside a phase."""
    lo, hi = phase
    return (frame["over_num"] >= lo).to_numpy() & (frame["over_num"] <= hi).to_numpy()


def is_legal_delivery(frame: pd.DataFrame) -> np.ndarray:
    """A delivery that counts toward the over: not a wide, not a no-ball.

    Run rate and economy denominators are measured in legal balls, because a
    wide or no-ball adds runs without advancing the over.
    """
    return (frame["wide"].to_numpy() == 0) & (frame["noball"].to_numpy() == 0)


def is_ball_faced(frame: pd.DataFrame) -> np.ndarray:
    """A delivery the striker is charged with facing (excludes wides).

    This is the denominator for striker-facing rates such as six-rate, where a
    wide is not a ball the batter had a chance to score off.
    """
    return frame["ballfaced"].to_numpy() == 1


def is_six(frame: pd.DataFrame) -> np.ndarray:
    if "outcome" in frame.columns:
        by_outcome = frame["outcome"].astype(str).str.lower().eq("six").to_numpy()
    else:
        by_outcome = np.zeros(len(frame), dtype=bool)
    by_runs = frame["batruns"].to_numpy() == 6
    return by_outcome | by_runs


def is_pace(frame: pd.DataFrame) -> np.ndarray:
    return frame["bowl_kind"].astype(str).str.contains("pace", case=False, na=False).to_numpy()


def is_spin(frame: pd.DataFrame) -> np.ndarray:
    return frame["bowl_kind"].astype(str).str.contains("spin", case=False, na=False).to_numpy()


def is_batter_dismissed(frame: pd.DataFrame) -> np.ndarray:
    """Wicket credited on this delivery, read from the out flag not p_out.

    p_out is populated every ball in the raw data, so it cannot be used to
    detect wickets. The out flag is the correct source.
    """
    return frame["out"].astype(bool).to_numpy()


# ----------------------------------------------------------------------------
# Per-match aggregation: turn a season's deliveries into per-match sums
# ----------------------------------------------------------------------------

@dataclass
class SeasonMatchSums:
    """Per-match numerator and denominator totals for one rate, one season.

    Once a rate is written as sum(numerator) / sum(denominator) over the
    in-scope deliveries, a match bootstrap becomes a weighted sum of these
    per-match totals: draw a multiplicity for each match, dot it against these
    vectors. That makes the bootstrap a matrix product and makes leave-one-match
    jackknife exact, both for free.
    """

    match_ids: np.ndarray            # one row per match in this season
    numerator: np.ndarray            # summed weighted numerator per match
    denominator: np.ndarray          # summed weighted denominator per match

    @property
    def n_matches(self) -> int:
        return len(self.match_ids)


def per_match_sums(
    frame: pd.DataFrame,
    scope: np.ndarray,
    numerator_weight: np.ndarray,
    denominator_weight: np.ndarray,
) -> SeasonMatchSums:
    """Collapse one season's deliveries to per-match numerator/denominator sums.

    scope selects the deliveries in play. The weights are per-delivery values
    (typically 0/1 masks or run counts). Matches with no in-scope deliveries
    still appear as a row with zero totals, so the match count stays the season
    count and resampling probabilities are correct.
    """
    scoped = frame.loc[scope]
    numer = np.asarray(numerator_weight, dtype=float)[scope]
    denom = np.asarray(denominator_weight, dtype=float)[scope]

    working = pd.DataFrame(
        {"p_match": scoped["p_match"].to_numpy(), "numer": numer, "denom": denom}
    )
    grouped = working.groupby("p_match", sort=True).sum()

    # Reindex onto the full season match list so empty-scope matches are kept.
    all_matches = np.sort(frame["p_match"].unique())
    grouped = grouped.reindex(all_matches, fill_value=0.0)
    return SeasonMatchSums(
        match_ids=all_matches,
        numerator=grouped["numer"].to_numpy(),
        denominator=grouped["denom"].to_numpy(),
    )


def per_match_team_sums(
    frame: pd.DataFrame,
    scope: np.ndarray,
    numerator_weight: np.ndarray,
    denominator_weight: np.ndarray,
    teams: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-match, per-team numerator/denominator totals for between-team spread.

    Returns matrices shaped (n_matches, n_teams). A team's season aggregate is
    the sum of its rows across the matches a resample draws, so resampling
    matches propagates each team's own sampling noise into the between-team
    spread instead of leaving it baked in as if the team rate were exact.
    """
    scoped = frame.loc[scope].copy()
    scoped["numer"] = np.asarray(numerator_weight, dtype=float)[scope]
    scoped["denom"] = np.asarray(denominator_weight, dtype=float)[scope]

    all_matches = np.sort(frame["p_match"].unique())
    match_index = {m: i for i, m in enumerate(all_matches)}
    team_index = {t: j for j, t in enumerate(teams)}

    numer_mat = np.zeros((len(all_matches), len(teams)))
    denom_mat = np.zeros((len(all_matches), len(teams)))

    grouped = scoped.groupby(["p_match", "team_bat"], sort=False)[["numer", "denom"]].sum()
    for (match_id, team), row in grouped.iterrows():
        if team not in team_index:
            continue
        i = match_index[match_id]
        j = team_index[team]
        numer_mat[i, j] += row["numer"]
        denom_mat[i, j] += row["denom"]

    return all_matches, numer_mat, denom_mat


# ----------------------------------------------------------------------------
# Estimands. Each is a callable (counts_earlier, counts_later) -> values.
# counts are (k, n_matches) multiplicity matrices. k is the number of scenarios
# evaluated at once: 1 for the point estimate, n_iter for the bootstrap, and
# n_matches for the jackknife. One code path serves all three.
# ----------------------------------------------------------------------------

def _rate(counts: np.ndarray, sums: SeasonMatchSums) -> np.ndarray:
    numer = counts @ sums.numerator
    denom = counts @ sums.denominator
    return np.divide(numer, denom, out=np.full_like(numer, np.nan, dtype=float), where=denom > 0)


def _rate_influence(counts: np.ndarray, sums: SeasonMatchSums) -> np.ndarray:
    """Per-match influence values of a ratio-of-sums rate, per scenario row.

    For R = sum(N_m) / sum(D_m) over the drawn matches, match m contributes
    u_m = (N_m - R * D_m) / sum(D). The linearized (delta-method) cluster variance
    of R is G/(G-1) * sum over drawn matches of u_m^2, where a match drawn c times
    contributes c copies. Shape (k, n_matches), aligned with `counts`.
    """
    rate = _rate(counts, sums)                       # (k,)
    denom_total = counts @ sums.denominator          # (k,)
    residual = sums.numerator[None, :] - rate[:, None] * sums.denominator[None, :]
    return np.divide(residual, denom_total[:, None],
                     out=np.full_like(residual, np.nan, dtype=float),
                     where=denom_total[:, None] > 0)


@dataclass
class RateDifference:
    """Later-season rate minus earlier-season rate for one scoped rate.

    Primary estimand form. The null is zero.
    """
    earlier: SeasonMatchSums
    later: SeasonMatchSums
    per_over: bool = False   # multiply both rates by 6 to read as runs per over

    def __call__(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> np.ndarray:
        scale = 6.0 if self.per_over else 1.0
        return scale * (_rate(counts_later, self.later) - _rate(counts_earlier, self.earlier))

    def influence(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-match influence values (earlier, later) of the difference, feeding
        the closed-form cluster SE. The earlier season enters with a minus sign."""
        scale = 6.0 if self.per_over else 1.0
        return (-scale * _rate_influence(counts_earlier, self.earlier),
                scale * _rate_influence(counts_later, self.later))


@dataclass
class RateRatio:
    """Later-season rate divided by earlier-season rate. Null is one.

    Reported only where the claim is inherently relative, and always beside the
    matching difference. Themes built on this estimand are run on the log scale
    (see run_theme): the resample distribution of log(ratio) is closer to
    symmetric than that of the ratio, and ratio bias can go either way, so
    nothing about a ratio is a conservative floor.
    """
    earlier: SeasonMatchSums
    later: SeasonMatchSums

    def __call__(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> np.ndarray:
        r_earlier = _rate(counts_earlier, self.earlier)
        r_later = _rate(counts_later, self.later)
        return np.divide(
            r_later, r_earlier,
            out=np.full_like(r_later, np.nan, dtype=float),
            where=r_earlier > 0,
        )

    def influence(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-match influence values of log(ratio), the scale ratio themes are
        estimated on: the influence of log R is the influence of R divided by R."""
        r_earlier = _rate(counts_earlier, self.earlier)
        r_later = _rate(counts_later, self.later)
        u_earlier = _rate_influence(counts_earlier, self.earlier)
        u_later = _rate_influence(counts_later, self.later)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (-u_earlier / r_earlier[:, None], u_later / r_later[:, None])


@dataclass
class DifferenceOfDifferences:
    """Contrast of two season-differences, both read off the same resample.

    This is how a shape claim like bookended acceleration is tested: the
    powerplay season-difference minus the middle-overs season-difference. It is
    less powered than either marginal difference, so a wide interval here means
    the shape is suggestive, not that the marginals failed.
    """
    focus: RateDifference       # the phase whose rise is the claim
    reference: RateDifference   # the phase it is claimed to outpace

    def __call__(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> np.ndarray:
        return self.focus(counts_earlier, counts_later) - self.reference(counts_earlier, counts_later)

    def influence(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Influence values subtract match by match, so the SE carries the
        correlation between the two differences through their shared matches."""
        fe, fl = self.focus.influence(counts_earlier, counts_later)
        re_, rl = self.reference.influence(counts_earlier, counts_later)
        return (fe - re_, fl - rl)


@dataclass
class WeightedSum:
    """A fixed linear combination of estimands, all read off the same resample.

    Sharing the draws keeps the component correlations, so the combination's
    interval reflects that the phase deltas move together, not as if independent.
    This is how a shape claim (curvature across phases, middle against league) is
    tested as one composite statistic rather than several separate tests.
    """
    terms: list                       # list of (weight, estimand)

    def __call__(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> np.ndarray:
        return sum(w * est(counts_earlier, counts_later) for w, est in self.terms)

    def influence(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Influence values combine linearly with the same weights, so the closed-form
        SE of the composite includes the covariance between its terms."""
        earlier_total = None
        later_total = None
        for w, est in self.terms:
            ue, ul = est.influence(counts_earlier, counts_later)
            earlier_total = w * ue if earlier_total is None else earlier_total + w * ue
            later_total = w * ul if later_total is None else later_total + w * ul
        return earlier_total, later_total


@dataclass
class BetweenTeamSpreadDifference:
    """Later minus earlier season in the SD of team rates across the league.

    This is the bifurcation recast as dispersion, and the first rung of the
    variance ladder. A positive value is stratification (the pack pulled apart),
    a negative value is convergence. The league mean can sit flat while this
    moves, which is exactly why the middle-overs bifurcation was invisible to a
    mean-difference test.
    """
    earlier_numer: np.ndarray
    earlier_denom: np.ndarray
    later_numer: np.ndarray
    later_denom: np.ndarray
    per_over: bool = True

    def _spread(self, counts: np.ndarray, numer_mat: np.ndarray, denom_mat: np.ndarray) -> np.ndarray:
        numer = counts @ numer_mat            # (k, n_teams)
        denom = counts @ denom_mat
        rate = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
        if self.per_over:
            rate = rate * 6.0
        # Teams absent from a resample (no matches drawn) are dropped from the
        # spread rather than counted as zero, which would fake convergence.
        return np.nanstd(rate, axis=1, ddof=1)

    def __call__(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> np.ndarray:
        return (
            self._spread(counts_later, self.later_numer, self.later_denom)
            - self._spread(counts_earlier, self.earlier_numer, self.earlier_denom)
        )


@dataclass
class InningsFractionRatio:
    """Ratio of the share of innings clearing a run threshold (e.g. 220+).

    A tail event. The effective sample is the count of qualifying innings, not
    the ball count, so the interval is wide by nature and BCa matters because
    the resample distribution is skew.
    """
    earlier_qualifying: SeasonMatchSums   # numerator = innings over threshold, denom = innings
    later_qualifying: SeasonMatchSums

    def __call__(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> np.ndarray:
        return RateRatio(self.earlier_qualifying, self.later_qualifying)(counts_earlier, counts_later)

    def influence(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return RateRatio(self.earlier_qualifying, self.later_qualifying).influence(counts_earlier, counts_later)


@dataclass
class InningsQuantileShift:
    """Later minus earlier season in an upper quantile of innings totals.

    Better powered than the threshold-crossing rate and reports the same tail
    from the distribution side, giving the tail-mass-and-quantile pairing the
    analysis wants. Non-additive, so it gathers the innings totals of the drawn
    matches per scenario rather than using the matrix shortcut.
    """
    earlier_match_innings: list[np.ndarray]   # innings totals grouped by match
    later_match_innings: list[np.ndarray]
    quantile: float = 0.90

    def _season_quantile(self, counts: np.ndarray, match_innings: list[np.ndarray]) -> np.ndarray:
        # Expand once: every innings total with the index of its match, so a
        # resample is np.repeat of the totals by the match multiplicities (the same
        # pool the per-row concatenation built, without the Python loop over matches).
        totals = np.concatenate([np.asarray(t, dtype=float) for t in match_innings] or [np.array([])])
        match_of_innings = np.concatenate(
            [np.full(len(t), m) for m, t in enumerate(match_innings)] or [np.array([], dtype=int)]
        ).astype(int)
        out = np.empty(counts.shape[0])
        for row_i, multiplicities in enumerate(counts):
            weights = multiplicities[match_of_innings].astype(int)
            pool = np.repeat(totals, weights)
            out[row_i] = np.quantile(pool, self.quantile) if pool.size else np.nan
        return out

    def __call__(self, counts_earlier: np.ndarray, counts_later: np.ndarray) -> np.ndarray:
        return (
            self._season_quantile(counts_later, self.later_match_innings)
            - self._season_quantile(counts_earlier, self.earlier_match_innings)
        )


# ----------------------------------------------------------------------------
# Running an estimand: point, bootstrap, jackknife, BCa
# ----------------------------------------------------------------------------

Estimand = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _ones(n: int) -> np.ndarray:
    return np.ones((1, n))


def point_estimate(estimand: Estimand, n_earlier: int, n_later: int) -> float:
    """The estimand on the full data (every match at multiplicity one)."""
    return float(estimand(_ones(n_earlier), _ones(n_later))[0])


def _multinomial_counts(n_matches: int, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """Sampling matches with replacement is a multinomial over match slots.

    Each row sums to n_matches, so every resampled season keeps the real match
    count while letting matches repeat or drop out.
    """
    return rng.multinomial(n_matches, np.full(n_matches, 1.0 / n_matches), size=n_iter).astype(float)


def bootstrap_replicates(
    estimand: Estimand,
    counts_earlier: np.ndarray,
    counts_later: np.ndarray,
) -> np.ndarray:
    """Evaluate the estimand across shared, pre-drawn resamples.

    Passing the same draws to every theme preserves the joint dependence across
    statistics, which is what lets a joint claim (six-rate up while wickets flat)
    be read off one set of resamples with the right correlation instead of
    pretending the pieces are independent.
    """
    return estimand(counts_earlier, counts_later)


def jackknife_influence(estimand: Estimand, n_earlier: int, n_later: int) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-match-out values, returned per season (earlier, later).

    Feeds the BCa acceleration term, which corrects the interval for how fast
    the estimand's variance changes as the sample shifts (skew). Dropping an
    earlier-season match perturbs only the earlier-season rate and vice versa.
    The values are kept in their own season so the acceleration can centre each
    season's leave-one-out values on that season's mean: the bootstrap is
    stratified by season, and a two-sample statistic needs the matching
    stratified jackknife (a pooled centring mixes the two seasons' level
    difference into the skew estimate).
    """
    eye_earlier = np.ones((n_earlier, n_earlier)) - np.eye(n_earlier)
    leave_out_earlier = estimand(eye_earlier, np.ones((n_earlier, n_later)))

    base_earlier = np.ones((n_later, n_earlier))
    eye_later = np.ones((n_later, n_later)) - np.eye(n_later)
    leave_out_later = estimand(base_earlier, eye_later)

    return leave_out_earlier, leave_out_later


def _acceleration(jackknife_strata: Sequence[np.ndarray], stratified: bool = True) -> float:
    """BCa acceleration from leave-one-out values.

    stratified=True centres each season's values on its own mean (the correct
    form for a season-stratified bootstrap). stratified=False pools all values
    and centres once; kept only so an earlier run can be reproduced exactly.
    """
    strata = [np.asarray(j, dtype=float) for j in jackknife_strata]
    strata = [j[np.isfinite(j)] for j in strata]
    if stratified:
        centred = np.concatenate([j.mean() - j for j in strata if j.size])
    else:
        pooled = np.concatenate(strata)
        centred = pooled.mean() - pooled
    denom = 6.0 * (np.sum(centred ** 2) ** 1.5)
    return float(np.sum(centred ** 3) / denom) if denom != 0 else 0.0


def bca_interval(
    theta_hat: float,
    replicates: np.ndarray,
    jackknife_values,
    alpha: float = 0.05,
    stratified: bool = True,
) -> tuple[float, float]:
    """Bias-corrected and accelerated interval.

    BCa over percentile because the tail-mass and low-count themes have skew
    resample distributions where a plain percentile interval sits in the wrong
    place. Reduces to the percentile interval when bias and acceleration are
    zero. `jackknife_values` is the (earlier, later) pair from jackknife_influence
    or a single pooled array.
    """
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 2:
        return (np.nan, np.nan)

    # Bias correction: how far the median replicate sits from the point estimate.
    share_below = np.mean(finite < theta_hat)
    share_below = min(max(share_below, 1e-6), 1 - 1e-6)
    z0 = _NORMAL.inv_cdf(share_below)

    strata = jackknife_values if isinstance(jackknife_values, (tuple, list)) else [jackknife_values]
    accel = _acceleration(strata, stratified=stratified)

    z_lo = _NORMAL.inv_cdf(alpha / 2)
    z_hi = _NORMAL.inv_cdf(1 - alpha / 2)

    def adjusted(z: float) -> float:
        adj = z0 + (z0 + z) / (1 - accel * (z0 + z))
        return _NORMAL.cdf(adj)

    lo_q = adjusted(z_lo)
    hi_q = adjusted(z_hi)
    return (
        float(np.quantile(finite, lo_q)),
        float(np.quantile(finite, hi_q)),
    )


def linearized_se(estimand, counts_earlier: np.ndarray, counts_later: np.ndarray) -> np.ndarray:
    """Closed-form cluster SE of the estimand for every scenario row.

    Var = G_e/(G_e-1) * sum_m c_e[m] u_e[m]^2 + G_l/(G_l-1) * sum_m c_l[m] u_l[m]^2,
    the delta-method variance of a ratio of per-match sums, with each season's
    matches as independent clusters and a match drawn c times counted c times.
    Returns NaN where the estimand carries no influence values (quantiles, SDs).
    """
    if not hasattr(estimand, "influence"):
        return np.full(counts_earlier.shape[0], np.nan)
    u_earlier, u_later = estimand.influence(counts_earlier, counts_later)
    g_e, g_l = counts_earlier.shape[1], counts_later.shape[1]
    # A resample with no in-scope balls in a season gives infinite or undefined
    # influence values; the SE is NaN there and that resample is dropped downstream.
    with np.errstate(invalid="ignore", over="ignore"):
        var = (g_e / (g_e - 1)) * np.sum(counts_earlier * u_earlier ** 2, axis=1) \
            + (g_l / (g_l - 1)) * np.sum(counts_later * u_later ** 2, axis=1)
    return np.sqrt(var)


def bootstrap_t_interval(theta_hat: float, se_hat: float, replicates: np.ndarray,
                         se_replicates: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Studentized (bootstrap-t) interval.

    Each resample is studentized by its own closed-form SE, t* = (theta* - theta_hat)
    / SE*, and the interval is theta_hat - t*_{1-alpha/2} SE, theta_hat - t*_{alpha/2}
    SE. Second-order accurate for a smooth statistic, and unlike BCa it does not
    depend on a skew estimate from about 70 clusters. Resamples where the SE is
    zero or undefined (a season with no in-scope balls) are dropped.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (replicates - theta_hat) / se_replicates
    t = t[np.isfinite(t)]
    if t.size < 2 or not np.isfinite(se_hat) or se_hat <= 0:
        return (np.nan, np.nan)
    return (
        float(theta_hat - np.quantile(t, 1 - alpha / 2) * se_hat),
        float(theta_hat - np.quantile(t, alpha / 2) * se_hat),
    )


def shared_resample_counts(n_earlier: int, n_later: int, n_iter: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One shared set of match resamples for both seasons, drawn in a fixed order.

    Every theme in an analysis is evaluated on the same draws so joint claims keep
    their correlation. Drawing earlier then later from one generator is the order
    every module uses, so two modules given the same seed see the same matches.
    """
    rng = np.random.default_rng(seed)
    counts_earlier = _multinomial_counts(n_earlier, n_iter, rng)
    counts_later = _multinomial_counts(n_later, n_iter, rng)
    return counts_earlier, counts_later


# ----------------------------------------------------------------------------
# Theme specification and the report it produces
# ----------------------------------------------------------------------------

@dataclass
class Theme:
    name: str
    estimand: Estimand
    null_value: float                 # 0 for differences, 1 for ratios
    power_tier: str                   # well_powered / composite / fragile_tail
    uncertainty_note: str = ""

    @property
    def log_scale(self) -> bool:
        """Ratio themes (null one) are estimated on the log scale."""
        return self.null_value == 1.0


@dataclass
class ThemeResult:
    """One theme's point estimate and intervals, plus the resample material needed
    to rebuild an interval at another level (used by the equivalence verdicts).

    Interval methods:
      bootstrap_t : studentized, from the closed-form cluster SE. Primary for rates,
                    rate differences and linear contrasts of rates.
      bca         : bias-corrected and accelerated, season-stratified jackknife.
                    Primary where no closed-form SE exists (quantiles, SD
                    differences) and for fragile-tail themes, where a resample can
                    hold zero events and the studentized ratio degenerates.
    Ratio themes carry all of these on the log scale, back-transformed for
    reporting; `raw_scale_bca_*` is the BCa interval computed on the ratio itself,
    kept so the two constructions can be compared side by side.
    """
    name: str
    point: float
    ci_low: float
    ci_high: float
    interval_method: str
    bca_low: float
    bca_high: float
    boot_t_low: float
    boot_t_high: float
    raw_scale_bca_low: float
    raw_scale_bca_high: float
    null_value: float
    excludes_null: bool
    se_boot: float                    # SD of the replicates on the analysis scale
    se_linearized: float              # closed-form cluster SE at the full data; NaN if none
    log_ratio_bias: float             # mean(log replicates) - log(point); NaN for differences
    log_ratio_bias_flag: bool
    scale: str                        # "linear" or "log"
    power_tier: str
    uncertainty_note: str
    alpha: float
    replicates: np.ndarray = field(repr=False, default=None)         # analysis scale
    se_replicates: np.ndarray = field(repr=False, default=None)
    jackknife_strata: tuple = field(repr=False, default=None)
    n_matches: tuple = field(repr=False, default=None)

    def interval(self, level: float, method: str | None = None) -> tuple[float, float]:
        """Interval at another confidence level, same method as the primary unless
        given. Back-transforms for ratio themes."""
        method = method or self.interval_method
        alpha = 1.0 - level
        theta = np.log(self.point) if self.scale == "log" else self.point
        if method == "bootstrap_t":
            lo, hi = bootstrap_t_interval(theta, self.se_linearized, self.replicates,
                                          self.se_replicates, alpha)
        else:
            lo, hi = bca_interval(theta, self.replicates, self.jackknife_strata, alpha)
        if self.scale == "log":
            return (float(np.exp(lo)), float(np.exp(hi)))
        return (lo, hi)

    def as_row(self) -> dict:
        return {
            "theme": self.name,
            "estimate": round(self.point, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "interval_method": self.interval_method,
            "bca_low": round(self.bca_low, 4),
            "bca_high": round(self.bca_high, 4),
            "boot_t_low": round(self.boot_t_low, 4) if np.isfinite(self.boot_t_low) else np.nan,
            "boot_t_high": round(self.boot_t_high, 4) if np.isfinite(self.boot_t_high) else np.nan,
            "raw_scale_bca_low": round(self.raw_scale_bca_low, 4) if np.isfinite(self.raw_scale_bca_low) else np.nan,
            "raw_scale_bca_high": round(self.raw_scale_bca_high, 4) if np.isfinite(self.raw_scale_bca_high) else np.nan,
            "null": self.null_value,
            "resampling_stable": self.excludes_null,
            "se_boot": round(self.se_boot, 5),
            "se_linearized": round(self.se_linearized, 5) if np.isfinite(self.se_linearized) else np.nan,
            "log_ratio_bias": round(self.log_ratio_bias, 5) if np.isfinite(self.log_ratio_bias) else np.nan,
            "log_ratio_bias_flag": self.log_ratio_bias_flag,
            "scale": self.scale,
            "interval_level": 1.0 - self.alpha,
            "power_tier": self.power_tier,
            "note": self.uncertainty_note,
        }


def run_theme(
    theme: Theme,
    n_earlier: int,
    n_later: int,
    counts_earlier: np.ndarray,
    counts_later: np.ndarray,
    alpha: float = 0.05,
    stratified_jackknife: bool = True,
    bias_flag_se_multiple: float = 0.25,
) -> ThemeResult:
    """Point estimate, bootstrap-t and BCa intervals, and the primary verdict.

    Primary interval: bootstrap-t when the estimand carries influence values and
    the theme is not a fragile tail; BCa otherwise. Ratio themes run on the log
    scale throughout and are back-transformed. `stratified_jackknife=False`
    reproduces the pooled-jackknife acceleration of earlier runs and exists only
    for that regression comparison.
    """
    point = point_estimate(theme.estimand, n_earlier, n_later)
    raw_replicates = bootstrap_replicates(theme.estimand, counts_earlier, counts_later)
    raw_jack = jackknife_influence(theme.estimand, n_earlier, n_later)

    if theme.log_scale:
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = float(np.log(point)) if point > 0 else np.nan
            replicates = np.where(raw_replicates > 0, np.log(raw_replicates), np.nan)
            jack = tuple(np.where(j > 0, np.log(j), np.nan) for j in raw_jack)
        raw_scale_bca = bca_interval(point, raw_replicates, raw_jack, alpha, stratified_jackknife)
    else:
        theta, replicates, jack = point, raw_replicates, raw_jack
        raw_scale_bca = (np.nan, np.nan)

    bca = bca_interval(theta, replicates, jack, alpha, stratified_jackknife)

    se_replicates = linearized_se(theme.estimand, counts_earlier, counts_later)
    se_hat = float(linearized_se(theme.estimand, _ones(n_earlier), _ones(n_later))[0])
    has_se = np.isfinite(se_hat) and se_hat > 0
    boot_t = bootstrap_t_interval(theta, se_hat, replicates, se_replicates, alpha) if has_se else (np.nan, np.nan)

    use_boot_t = has_se and theme.power_tier != "fragile_tail" and all(np.isfinite(boot_t))
    method = "bootstrap_t" if use_boot_t else "bca"
    lo, hi = boot_t if use_boot_t else bca

    finite = replicates[np.isfinite(replicates)]
    se_boot = float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
    if theme.log_scale and finite.size > 1 and np.isfinite(theta):
        # Bootstrap bias of the log ratio; BCa already corrects for it, the flag
        # only makes a large bias visible in the table.
        bias = float(finite.mean() - theta)
        bias_flag = bool(abs(bias) > bias_flag_se_multiple * se_boot)
    else:
        bias, bias_flag = np.nan, False

    def back(pair):
        if theme.log_scale:
            return (float(np.exp(pair[0])), float(np.exp(pair[1])))
        return (float(pair[0]), float(pair[1]))

    lo, hi = back((lo, hi))
    bca_lo, bca_hi = back(bca)
    bt_lo, bt_hi = back(boot_t)

    excludes = (lo > theme.null_value) or (hi < theme.null_value)

    return ThemeResult(
        name=theme.name,
        point=point,
        ci_low=lo,
        ci_high=hi,
        interval_method=method,
        bca_low=bca_lo,
        bca_high=bca_hi,
        boot_t_low=bt_lo,
        boot_t_high=bt_hi,
        raw_scale_bca_low=raw_scale_bca[0],
        raw_scale_bca_high=raw_scale_bca[1],
        null_value=theme.null_value,
        excludes_null=bool(excludes) if np.isfinite(lo) and np.isfinite(hi) else False,
        se_boot=se_boot,
        se_linearized=se_hat,
        log_ratio_bias=bias,
        log_ratio_bias_flag=bias_flag,
        scale="log" if theme.log_scale else "linear",
        power_tier=theme.power_tier,
        uncertainty_note=theme.uncertainty_note,
        alpha=alpha,
        replicates=replicates,
        se_replicates=se_replicates,
        jackknife_strata=jack,
        n_matches=(n_earlier, n_later),
    )


# ----------------------------------------------------------------------------
# Precision-stability guard for the variance themes
# ----------------------------------------------------------------------------

def balls_per_team_in_phase(frame: pd.DataFrame, phase: tuple[int, int]) -> pd.Series:
    """Legal balls faced by each team in a phase.

    A between-team variance-ratio is more sensitive to a cross-season change in
    estimation precision than a mean-difference is, because noise enters the
    variance directly. If one season has markedly fewer balls per team in a
    phase, its team rates are noisier and the between-team spread rises as an
    artefact. This surfaces that before any widening is called stratification.
    """
    scope = in_phase(frame, phase) & is_legal_delivery(frame)
    scoped = frame.loc[scope]
    return scoped.groupby("team_bat").size().sort_values()


# ----------------------------------------------------------------------------
# Loading the real dataset (used once the cleaned file is available)
# ----------------------------------------------------------------------------

REQUIRED_COLUMNS = (
    "p_match", "inns", "over_num", "team_bat", "team_bowl",
    "score", "batruns", "ballfaced", "out",
    "bowl_kind", "wide", "noball", "year",
)


def load_cleaned_dataset(path: str) -> pd.DataFrame:
    """Read the cleaned ball-by-ball dataset and restrict to the two seasons.

    Result-independent themes use every completed match. The caller supplies the
    dataset that already carries the validated derived columns (bowler_charge,
    bowl_type_group, control_clean, and the rest); this function does not
    recompute them, it only checks the columns the harness reads are present.
    """
    frame = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    two_seasons = frame[frame["year"].isin((EARLIER_SEASON, LATER_SEASON))].copy()
    return two_seasons


# ----------------------------------------------------------------------------
# Theme construction from a two-season frame
# ----------------------------------------------------------------------------

def build_themes(frame: pd.DataFrame) -> tuple[list[Theme], int, int, np.ndarray]:
    """Assemble every theme against one two-season frame.

    Returns the themes plus the two seasons' match counts and the sorted team
    list, so the caller can draw one shared set of resamples and the
    precision-stability guard can run on the same frame.
    """
    earlier = frame[frame["year"] == EARLIER_SEASON]
    later = frame[frame["year"] == LATER_SEASON]

    def sums(season_frame: pd.DataFrame, scope, numer, denom) -> SeasonMatchSums:
        return per_match_sums(season_frame, scope, numer, denom)

    themes: list[Theme] = []

    # Powerplay six-rate: sixes per ball faced in overs 1 to 6. Well powered.
    pp_e = in_phase(earlier, POWERPLAY_OVERS)
    pp_l = in_phase(later, POWERPLAY_OVERS)
    six_ratio = RateRatio(
        sums(earlier, pp_e, is_six(earlier), is_ball_faced(earlier)),
        sums(later, pp_l, is_six(later), is_ball_faced(later)),
    )
    themes.append(Theme(
        "powerplay_six_rate_ratio", six_ratio, 1.0, "well_powered",
        "Aerial aggression up top. Pair with the wicket-cost theme below.",
    ))

    # Powerplay wicket cost: batter dismissals per ball faced, overs 1 to 6.
    # Expected to straddle zero. Read jointly with the six-rate off shared draws.
    wicket_diff = RateDifference(
        sums(earlier, pp_e, is_batter_dismissed(earlier), is_ball_faced(earlier)),
        sums(later, pp_l, is_batter_dismissed(later), is_ball_faced(later)),
    )
    themes.append(Theme(
        "powerplay_wicket_rate_diff", wicket_diff, 0.0, "well_powered",
        "Cost side of the aggression claim. Straddling zero is the finding, not a miss.",
    ))

    # Per-phase run rate differences, and the bookending shape contrast.
    phase_rpo_diff = {}
    for label, phase in (("powerplay", POWERPLAY_OVERS), ("middle", MIDDLE_OVERS), ("death", DEATH_OVERS)):
        se = in_phase(earlier, phase) & is_legal_delivery(earlier)
        sl = in_phase(later, phase) & is_legal_delivery(later)
        rpo = RateDifference(
            sums(earlier, se, earlier["score"].to_numpy(), is_legal_delivery(earlier)),
            sums(later, sl, later["score"].to_numpy(), is_legal_delivery(later)),
            per_over=True,
        )
        phase_rpo_diff[label] = rpo
        themes.append(Theme(
            f"{label}_run_rate_diff", rpo, 0.0, "well_powered",
            "Runs per over, legal balls only.",
        ))

    themes.append(Theme(
        "bookending_powerplay_vs_middle", DifferenceOfDifferences(phase_rpo_diff["powerplay"], phase_rpo_diff["middle"]),
        0.0, "composite",
        "Shape test. Wide interval means bookending is suggestive, not that the phase rises failed.",
    ))
    themes.append(Theme(
        "bookending_death_vs_middle", DifferenceOfDifferences(phase_rpo_diff["death"], phase_rpo_diff["middle"]),
        0.0, "composite",
        "Second leg of the bookended-acceleration shape.",
    ))

    # Pace share by phase: share of legal balls bowled by pace. Well powered.
    for label, phase in (("powerplay", POWERPLAY_OVERS), ("middle", MIDDLE_OVERS), ("death", DEATH_OVERS)):
        se = in_phase(earlier, phase) & is_legal_delivery(earlier)
        sl = in_phase(later, phase) & is_legal_delivery(later)
        share = RateDifference(
            sums(earlier, se, is_pace(earlier), is_legal_delivery(earlier)),
            sums(later, sl, is_pace(later), is_legal_delivery(later)),
        )
        themes.append(Theme(
            f"pace_share_diff_{label}", share, 0.0, "well_powered",
            "League and phase level. Per-team pace shift is texture, not tested here.",
        ))

    # Middle-overs between-team spread: the bifurcation as dispersion.
    teams = sorted(set(earlier["team_bat"].unique()) | set(later["team_bat"].unique()))
    me_e = in_phase(earlier, MIDDLE_OVERS) & is_legal_delivery(earlier)
    me_l = in_phase(later, MIDDLE_OVERS) & is_legal_delivery(later)
    _, e_num, e_den = per_match_team_sums(earlier, me_e, earlier["score"].to_numpy(), is_legal_delivery(earlier), teams)
    _, l_num, l_den = per_match_team_sums(later, me_l, later["score"].to_numpy(), is_legal_delivery(later), teams)
    themes.append(Theme(
        "middle_overs_team_spread_diff",
        BetweenTeamSpreadDifference(e_num, e_den, l_num, l_den, per_over=True),
        0.0, "composite",
        "Positive is stratification (the pack pulled apart). Read with the precision guard.",
    ))

    # Shot repertoire: one aggregate contrast, not six per-shot intervals.
    if "shot" in frame.columns:
        controlled = {"PULL", "SQUARE_DRIVE"}
        slog = {"SLOG"}

        def shot_in(season_frame, names):
            return season_frame["shot"].astype(str).str.upper().isin(names).to_numpy()

        faced_e = is_ball_faced(earlier)
        faced_l = is_ball_faced(later)
        controlled_share = RateDifference(
            sums(earlier, faced_e, shot_in(earlier, controlled), faced_e),
            sums(later, faced_l, shot_in(later, controlled), faced_l),
        )
        slog_share = RateDifference(
            sums(earlier, faced_e, shot_in(earlier, slog), faced_e),
            sums(later, faced_l, shot_in(later, slog), faced_l),
        )
        themes.append(Theme(
            "shot_controlled_minus_slog_shift",
            DifferenceOfDifferences(controlled_share, slog_share),
            0.0, "composite",
            "Tagging-dependent: this interval is a floor, widen for ~7-10% shot-label error.",
        ))

    # 220+ innings share and the upper-quantile shift of innings totals.
    innings_earlier = earlier.groupby(["p_match", "inns"])["score"].sum().reset_index()
    innings_later = later.groupby(["p_match", "inns"])["score"].sum().reset_index()

    def innings_threshold_sums(innings_frame, all_matches, threshold=220):
        over = innings_frame["score"].to_numpy() >= threshold
        work = pd.DataFrame({
            "p_match": innings_frame["p_match"].to_numpy(),
            "numer": over.astype(float),
            "denom": 1.0,
        })
        g = work.groupby("p_match", sort=True).sum().reindex(all_matches, fill_value=0.0)
        return SeasonMatchSums(all_matches, g["numer"].to_numpy(), g["denom"].to_numpy())

    matches_earlier = np.sort(earlier["p_match"].unique())
    matches_later = np.sort(later["p_match"].unique())
    themes.append(Theme(
        "innings_220_plus_ratio",
        InningsFractionRatio(
            innings_threshold_sums(innings_earlier, matches_earlier),
            innings_threshold_sums(innings_later, matches_later),
        ),
        1.0, "fragile_tail",
        "Tail event on ALL innings, so chases are censored at the target; "
        "the first-innings tail analysis (innings_tail.py) is the reported version.",
    ))

    def match_innings_list(innings_frame, all_matches):
        by_match = {m: g["score"].to_numpy() for m, g in innings_frame.groupby("p_match")}
        return [by_match.get(m, np.array([])) for m in all_matches]

    themes.append(Theme(
        "innings_total_q90_shift",
        InningsQuantileShift(
            match_innings_list(innings_earlier, matches_earlier),
            match_innings_list(innings_later, matches_later),
            quantile=0.90,
        ),
        0.0, "fragile_tail",
        "Upper quantile on ALL innings (censored chases included); "
        "the first-innings shift function (innings_tail.py) is the reported version.",
    ))

    # Death-overs spin economy: weakest theme, selection-confounded even if stable.
    ds_e = in_phase(earlier, DEATH_OVERS) & is_spin(earlier) & is_legal_delivery(earlier)
    ds_l = in_phase(later, DEATH_OVERS) & is_spin(later) & is_legal_delivery(later)
    charge_col = "bowler_charge" if "bowler_charge" in frame.columns else "score"
    death_spin_econ = RateDifference(
        sums(earlier, ds_e, earlier[charge_col].to_numpy(), is_legal_delivery(earlier)),
        sums(later, ds_l, later[charge_col].to_numpy(), is_legal_delivery(later)),
        per_over=True,
    )
    themes.append(Theme(
        "death_spin_economy_diff", death_spin_econ, 0.0, "fragile_tail",
        "~560 balls league-wide. Even a clear interval is selection-confounded",
    ))

    return themes, len(matches_earlier), len(matches_later), np.array(teams)


def run_all(
    frame: pd.DataFrame,
    n_iter: int = 10000,
    seed: int = 20240601,
    alpha: float = 0.05,
    stratified_jackknife: bool = True,
    return_results: bool = False,
):
    """Draw one shared set of match resamples and evaluate every theme on it.

    Returns the results table; with return_results=True also the ThemeResult
    objects (they carry the replicates, for equivalence verdicts and diagnostics).
    """
    themes, n_earlier, n_later, _ = build_themes(frame)
    counts_earlier, counts_later = shared_resample_counts(n_earlier, n_later, n_iter, seed)

    results = [
        run_theme(theme, n_earlier, n_later, counts_earlier, counts_later,
                  alpha=alpha, stratified_jackknife=stratified_jackknife)
        for theme in themes
    ]
    table = pd.DataFrame([r.as_row() for r in results])
    if return_results:
        return table, results
    return table


# ----------------------------------------------------------------------------
# Synthetic smoke test: schema-matched, with two effects injected so the engine
# can be checked for recovering what it should and staying quiet where it should.
# ----------------------------------------------------------------------------

def _synthetic_two_seasons(seed: int = 0) -> pd.DataFrame:
    """A schema-matched two-season frame with known injected effects, so the
    harness can be validated without the real dataset. Not for analysis, only
    for checking the engine recovers what it should and stays quiet elsewhere.

    Injected truths, so the smoke test has an answer key:
    1. Powerplay six probability rises 0.035 to 0.052 (six-rate ratio ~1.5).
    2. Middle-overs team means go from tight (all near 7.6 rpo) to stratified
       (SD ~0.85 rpo): the bifurcation-as-widening the spread estimand must catch.
    3. Wickets, pace share, and shot mix are held fixed across seasons, so those
       themes must read as null.
    A per-innings form multiplier gives innings totals a realistic spread that
    reaches past 220, so the tail themes run on a non-degenerate distribution.
    """
    run_rng = np.random.default_rng(seed)
    # Wickets draw from their own stream so the number of scoring draws, which
    # depends on the six rate, cannot couple into the dismissal rate.
    wicket_rng = np.random.default_rng(seed + 1000)
    teams = [f"Team{i:02d}" for i in range(10)]
    rows = []

    def non_boundary_runs(mean_per_ball: float) -> int:
        # A per-ball run mean turned into a plausible 0..4 outcome. Kept off 6 so
        # only the explicit six draw feeds the six-rate theme.
        return int(min(run_rng.poisson(mean_per_ball), 4))

    def simulate_season(year, six_prob_pp, middle_mean_by_team):
        match_id = year * 1000
        pairings = [(a, b) for i, a in enumerate(teams) for b in teams[i + 1:]]
        run_rng.shuffle(pairings)
        n_matches = 10 * 14 // 2
        for bat_team, bowl_team in pairings[:n_matches]:
            match_id += 1
            for inns, (batting, bowling) in enumerate([(bat_team, bowl_team), (bowl_team, bat_team)], start=1):
                form = run_rng.normal(1.0, 0.18)   # innings-level scoring multiplier, drives the tail
                for over in range(1, 21):
                    if POWERPLAY_OVERS[0] <= over <= POWERPLAY_OVERS[1]:
                        six_p, mean_ball, pace_p = six_prob_pp, 1.25, 0.75
                    elif MIDDLE_OVERS[0] <= over <= MIDDLE_OVERS[1]:
                        six_p, mean_ball, pace_p = 0.03, middle_mean_by_team[batting] / 6.0, 0.45
                    else:
                        six_p, mean_ball, pace_p = 0.11, 1.85, 0.70
                    six_p_eff = min(six_p * form, 0.6)
                    mean_eff = max(mean_ball * form, 0.05)
                    for _ in range(6):
                        is_wide = run_rng.random() < 0.03
                        is_nb = run_rng.random() < 0.01
                        six = (not is_wide) and run_rng.random() < six_p_eff
                        if is_wide:
                            batruns = 0
                        elif six:
                            batruns = 6
                        else:
                            batruns = non_boundary_runs(mean_eff)
                        extras = (1 if is_wide else 0) + (1 if is_nb else 0)
                        out = (not is_wide) and wicket_rng.random() < 0.035
                        rows.append({
                            "p_match": match_id, "year": year, "inns": inns,
                            "over": over, "team_bat": batting, "team_bowl": bowling,
                            "batruns": batruns, "score": batruns + extras,
                            "ballfaced": 0 if is_wide else 1,
                            "outcome": "six" if six else ("wide" if is_wide else "run"),
                            "out": out,
                            "bowl_kind": "pace bowler" if run_rng.random() < pace_p else "spin bowler",
                            "wide": 1 if is_wide else 0, "noball": 1 if is_nb else 0,
                            "byes": 0, "legbyes": 0,
                            "bowler_charge": batruns + extras,
                            "shot": "SLOG" if six and run_rng.random() < 0.5 else run_rng.choice(["PULL", "SQUARE_DRIVE", "PUSH", "DEFENDED"]),
                        })

    tight_middle = {t: 7.6 for t in teams}
    stratified_middle = dict(zip(teams, [8.7, 8.6, 8.5, 8.4, 7.6, 7.6, 7.6, 6.7, 6.6, 6.5]))
    simulate_season(EARLIER_SEASON, six_prob_pp=0.035, middle_mean_by_team=tight_middle)
    simulate_season(LATER_SEASON, six_prob_pp=0.052, middle_mean_by_team=stratified_middle)

    return pd.DataFrame(rows)


def _smoke_test() -> None:
    # The stub writes `over`; the harness reads the real column name `over_num`.
    frame = _synthetic_two_seasons(seed=7).rename(columns={"over": "over_num"})
    print(f"synthetic deliveries: {len(frame):,}  matches: {frame['p_match'].nunique()}")
    print(f"seasons: {sorted(frame['year'].unique())}\n")

    print("precision-stability guard (legal balls per team, middle overs):")
    for year in (EARLIER_SEASON, LATER_SEASON):
        counts = balls_per_team_in_phase(frame[frame['year'] == year], MIDDLE_OVERS)
        print(f"  {year}: min={counts.min()}  median={int(counts.median())}  max={counts.max()}")
    print()

    results = run_all(frame, n_iter=3000, seed=1)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(results.to_string(index=False))


if __name__ == "__main__":
    _smoke_test()
