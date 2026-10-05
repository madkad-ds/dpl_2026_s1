"""
Hierarchical models for de-attenuated between-team dispersion.

A between-team SD computed from ten team rates is inflated by the sampling noise of
estimating each rate from about 14 matches. These models put that noise in its own
variance component and read the between-team SD clean of it, with partial pooling
so a thin-sample team cannot masquerade as an outlier.

Two models:
  1. Single-rate phase model (`fit_between_team_sd`): team-innings run rate in one
     phase, Normal around the team-season rate with observation SD scaled by
     1/sqrt(balls); team-season rates Normal around a season mean with a
     season-specific between-team SD. Non-centered.
  2. Nested gear-up model (`fit_gear_up_sd`): each team-innings contributes its
     phase rate and its middle-overs baseline rate as separate observations; the
     team-season gear-up (phase minus baseline) is formed inside the model so both
     rates' sampling noise propagates. Season-specific between-team SD of the
     gear-up is the estimand. Non-centered.

Both report the season SD difference and, because ratio-scale comparisons of an SD
are better behaved near zero, the log SD ratio log(sd_later / sd_earlier) with
P(widen). Every SD result is refit under a second prior (Exponential in place of
HalfNormal) because ten teams inform an SD weakly; the sign and size must survive
the change of prior to be in the data rather than the prior.

Sampling settings (chains, draws, tune, target_accept) come from config.toml [mcmc].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mean_level_and_tail import season_contrast_bootstrap as scb


def _pymc():
    import pymc as pm
    return pm


def _sd_prior(pm, name, prior: str, halfnormal_sigma: float, shape):
    """Between-team SD prior by name. HalfNormal is the primary; Exponential(1)
    (mean 1, still weakly informative) has a different shape near zero and is the
    prior-sensitivity refit."""
    if prior == "halfnormal":
        return pm.HalfNormal(name, sigma=halfnormal_sigma, shape=shape)
    if prior == "exponential":
        return pm.Exponential(name, lam=1.0, shape=shape)
    raise ValueError(f"unknown prior {prior!r}; use 'halfnormal' or 'exponential'")


def _sample(pm, mcmc: dict, seed: int):
    # One worker per chain up to the CPUs this process may use; `cores` in the
    # [mcmc] config section overrides the detection (some containers report zero).
    import os
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    cores = int(mcmc.get("cores") or max(1, min(mcmc["chains"], available)))
    return pm.sample(draws=mcmc["draws"], tune=mcmc["tune"], chains=mcmc["chains"],
                     cores=cores, target_accept=mcmc["target_accept"], random_seed=seed,
                     progressbar=False)


# ----------------------------------------------------------------------------
# Single-rate phase model
# ----------------------------------------------------------------------------

def team_innings_rate(frame, phase):
    """One row per (season, team, innings): phase run rate and its ball count.

    The innings is the observation unit because it is the level teams actually
    vary at match to match, and its ball count sets how noisy that observation
    is, which is what lets the model separate sampling noise from true spread.
    """
    scope = scb.in_phase(frame, phase) & scb.is_legal_delivery(frame)
    scoped = frame.loc[scope]
    grouped = (scoped.groupby(["year", "team_bat", "p_match", "inns"])
                     .agg(runs=("score", "sum"), balls=("score", "size"))
                     .reset_index())
    grouped = grouped[grouped["balls"] >= 6]          # drop slivers that carry no rate signal
    grouped["rate"] = grouped["runs"] / grouped["balls"] * 6.0
    return grouped


def fit_between_team_sd(frame, phase, mcmc: dict, seed: int, prior: str = "halfnormal"):
    """Season-specific between-team SD in one phase, with sampling noise separated.

    Levels:
    1. team-innings run rate is Normal around that team-season's true rate, with
       observation SD scaled by 1/sqrt(balls) so short innings inform less.
    2. team-season true rates are Normal around a season mean, with a
       season-specific between-team SD (the parameter of interest), sampled in
       non-centered form so the team effects are not coupled to the SD during
       sampling.
    The posterior of sd_later minus sd_earlier is the convergence/stratification
    answer: negative is convergence, positive is stratification, mass on zero
    is stable spread.
    """
    pm = _pymc()
    obs = team_innings_rate(frame, phase)
    seasons = np.array([scb.EARLIER_SEASON, scb.LATER_SEASON])
    teams = np.sort(obs["team_bat"].unique())
    team_ix = {t: i for i, t in enumerate(teams)}
    season_ix = {s: i for i, s in enumerate(seasons)}

    obs_team = obs["team_bat"].map(team_ix).to_numpy()
    obs_season = obs["year"].map(season_ix).to_numpy()
    obs_rate = obs["rate"].to_numpy()
    obs_balls = obs["balls"].to_numpy()

    # A team appears in both seasons, so team-season effects index on the pair.
    n_teams, n_seasons = len(teams), len(seasons)
    ts_row = obs_season * n_teams + obs_team
    season_of_ts = np.repeat(np.arange(n_seasons), n_teams)

    with pm.Model():
        season_mean = pm.Normal("season_mean", mu=8.0, sigma=3.0, shape=n_seasons)

        # The estimand. Half-Normal is weakly informative; the prior matters at
        # n=10 teams, so the Exponential refit confirms the sign of the
        # difference is not a prior artefact.
        between_team_sd = _sd_prior(pm, "between_team_sd", prior, 1.5, n_seasons)

        # Non-centered: sample standard-normal offsets and scale them by the
        # season SD, so the team effects are not coupled to the SD while
        # sampling. Same model as the centered form, but geometry the sampler
        # can move through, which is what clears the funnel divergences that a
        # centered hierarchical SD produces.
        team_offset = pm.Normal("team_offset", 0.0, 1.0, shape=n_seasons * n_teams)
        team_rate = pm.Deterministic(
            "team_rate",
            season_mean[season_of_ts] + team_offset * between_team_sd[season_of_ts],
        )

        # Innings-level dispersion of run rate around the team-season rate. The
        # per-innings SD shrinks with balls faced, so a full middle innings pins
        # the rate far harder than a truncated one.
        innings_scale = pm.HalfNormal("innings_scale", sigma=3.0)
        sigma_obs = innings_scale / np.sqrt(obs_balls)

        pm.Normal("likelihood", mu=team_rate[ts_row], sigma=sigma_obs, observed=obs_rate)

        pm.Deterministic(
            "sd_difference",
            between_team_sd[season_ix[scb.LATER_SEASON]]
            - between_team_sd[season_ix[scb.EARLIER_SEASON]],
        )
        pm.Deterministic(
            "log_sd_ratio",
            pm.math.log(between_team_sd[season_ix[scb.LATER_SEASON]])
            - pm.math.log(between_team_sd[season_ix[scb.EARLIER_SEASON]]),
        )
        idata = _sample(pm, mcmc, seed)

    return idata, teams, seasons


# ----------------------------------------------------------------------------
# Nested gear-up model
# ----------------------------------------------------------------------------

def team_innings_phase_pair(frame, phase, baseline_phase):
    """One row per (season, team, match, innings): runs and legal balls in the
    target phase and in the baseline phase separately.

    Both phases are kept as raw runs-and-balls, not a precomputed delta, so the
    model can carry each phase's sampling noise and form the gear-up difference
    internally rather than treating a noisy observed delta as if it were exact.
    """
    def phase_totals(sub, phase):
        legal = scb.is_legal_delivery(sub) & scb.in_phase(sub, phase)
        s = sub.loc[legal]
        return s.groupby(["year", "team_bat", "p_match", "inns"])["score"].agg(["sum", "size"])

    tgt = phase_totals(frame, phase).rename(columns={"sum": "phase_runs", "size": "phase_balls"})
    base = phase_totals(frame, baseline_phase).rename(columns={"sum": "base_runs", "size": "base_balls"})
    paired = tgt.join(base, how="inner").reset_index()
    # Both phases need enough balls for the innings to inform a gear-up delta.
    paired = paired[(paired["phase_balls"] >= 6) & (paired["base_balls"] >= 12)]
    return paired


def fit_gear_up_sd(frame, phase, baseline_phase, mcmc: dict, seed: int,
                   prior: str = "halfnormal"):
    """Season-specific between-team SD of the gear-up delta, with the two-rate
    sampling noise separated from the true between-team spread.

    Structure:
    1. each team-innings' phase run rate and baseline run rate are Normal around
       that team-season's true phase and baseline rates, observation SD scaled by
       1/sqrt(balls) so short phases inform less.
    2. a team-season's true gear-up delta is (true phase rate minus true baseline
       rate); team deltas are Normal around a season mean delta, with a
       season-specific between-team SD (the parameter of interest), sampled
       non-centered to avoid the funnel geometry.
    The posterior of sd_later minus sd_earlier is the de-attenuated answer the
    bootstrap could not resolve: whether gear-change spread genuinely widened once
    the noise of estimating two rates per team is removed.
    `prior` selects HalfNormal (primary) or Exponential (sensitivity refit) on the
    gear-up SD; everything else is identical.
    """
    pm = _pymc()
    obs = team_innings_phase_pair(frame, phase, baseline_phase)
    seasons = np.array([scb.EARLIER_SEASON, scb.LATER_SEASON])
    teams = np.sort(obs["team_bat"].unique())
    team_ix = {t: i for i, t in enumerate(teams)}
    season_ix = {s: i for i, s in enumerate(seasons)}

    n_teams, n_seasons = len(teams), len(seasons)
    row_team = obs["team_bat"].map(team_ix).to_numpy()
    row_season = obs["year"].map(season_ix).to_numpy()
    ts = row_season * n_teams + row_team          # team-season index per observation
    season_of_ts = np.repeat(np.arange(n_seasons), n_teams)

    phase_rate_obs = (obs["phase_runs"] / obs["phase_balls"] * 6.0).to_numpy()
    base_rate_obs = (obs["base_runs"] / obs["base_balls"] * 6.0).to_numpy()
    phase_balls = obs["phase_balls"].to_numpy()
    base_balls = obs["base_balls"].to_numpy()

    with pm.Model():
        # True team-season baseline (middle-overs) rate, weakly informed.
        base_mean = pm.Normal("base_mean", mu=8.0, sigma=3.0)
        base_spread = pm.HalfNormal("base_spread", sigma=2.0)
        base_z = pm.Normal("base_z", 0.0, 1.0, shape=n_seasons * n_teams)
        team_base_rate = pm.Deterministic("team_base_rate", base_mean + base_z * base_spread)

        # Season mean gear-up (how much the league lifts into the phase) and the
        # season-specific between-team SD of that gear-up: the estimand.
        gear_mean = pm.Normal("gear_mean", mu=1.0, sigma=3.0, shape=n_seasons)
        gear_sd = _sd_prior(pm, "gear_sd", prior, 1.5, n_seasons)
        gear_z = pm.Normal("gear_z", 0.0, 1.0, shape=n_seasons * n_teams)
        team_gear = pm.Deterministic(
            "team_gear", gear_mean[season_of_ts] + gear_z * gear_sd[season_of_ts]
        )

        # True phase rate is baseline plus the team's gear-up.
        team_phase_rate = pm.Deterministic("team_phase_rate", team_base_rate + team_gear)

        # Innings-level observation noise on each rate, shrinking with balls.
        phase_scale = pm.HalfNormal("phase_scale", sigma=4.0)
        base_scale = pm.HalfNormal("base_scale", sigma=4.0)
        pm.Normal("phase_like", mu=team_phase_rate[ts],
                  sigma=phase_scale / np.sqrt(phase_balls), observed=phase_rate_obs)
        pm.Normal("base_like", mu=team_base_rate[ts],
                  sigma=base_scale / np.sqrt(base_balls), observed=base_rate_obs)

        pm.Deterministic(
            "sd_difference",
            gear_sd[season_ix[scb.LATER_SEASON]] - gear_sd[season_ix[scb.EARLIER_SEASON]],
        )
        pm.Deterministic(
            "log_sd_ratio",
            pm.math.log(gear_sd[season_ix[scb.LATER_SEASON]])
            - pm.math.log(gear_sd[season_ix[scb.EARLIER_SEASON]]),
        )
        idata = _sample(pm, mcmc, seed)

    return idata, teams, seasons


# ----------------------------------------------------------------------------
# Reporting and diagnostics
# ----------------------------------------------------------------------------

def sd_posterior_summary(idata, sd_var: str, label: str, prior: str,
                         interval_level: float = 0.95) -> dict:
    """Season SD medians and intervals, the SD difference, the log SD ratio, and
    P(widen), as one flat row."""
    post = idata.posterior
    sd = post[sd_var].values.reshape(-1, post[sd_var].shape[-1])
    diff = post["sd_difference"].values.ravel()
    log_ratio = post["log_sd_ratio"].values.ravel()
    lo_q, hi_q = 100 * (1 - interval_level) / 2, 100 * (1 + interval_level) / 2

    def band(x):
        return float(np.median(x)), float(np.percentile(x, lo_q)), float(np.percentile(x, hi_q))

    row = {"model": label, "prior": prior}
    for i, yr in enumerate((scb.EARLIER_SEASON, scb.LATER_SEASON)):
        med, lo, hi = band(sd[:, i])
        row.update({f"sd_{yr}": med, f"sd_{yr}_low": lo, f"sd_{yr}_high": hi})
    med, lo, hi = band(diff)
    row.update({"sd_difference": med, "sd_difference_low": lo, "sd_difference_high": hi})
    med, lo, hi = band(log_ratio)
    row.update({"log_sd_ratio": med, "log_sd_ratio_low": lo, "log_sd_ratio_high": hi,
                "sd_ratio": float(np.exp(med))})
    row["p_widen"] = float((diff > 0).mean())
    row["p_narrow"] = float((diff < 0).mean())
    row["interval_level"] = interval_level
    row.update(convergence_checks(idata, [sd_var, "sd_difference"]))
    return row


def convergence_checks(idata, varnames) -> dict:
    """Divergence count, max r-hat and min effective sample size over the named
    variables (ArviZ). Zero divergences, r-hat at or below about 1.01 and ESS in
    the thousands is the bar every reported model met."""
    import arviz as az
    out = {"divergences": int(idata.sample_stats["diverging"].sum())}
    rhat = az.rhat(idata, var_names=list(varnames))
    ess = az.ess(idata, var_names=list(varnames))
    out["max_rhat"] = float(max(float(np.max(rhat[v].values)) for v in varnames))
    out["min_ess"] = float(min(float(np.min(ess[v].values)) for v in varnames))
    return out
