"""
Powerplay stratification as one joint model instead of three separate legs.

Unit: a team-innings in the powerplay, both seasons. Each unit carries three
outcomes and the team-season sits behind them as a three-dimensional effect
theta[s, t] = (run-rate level, aggressive-attempt log-odds, gear-up from the team's
own middle-overs baseline):

  (a) powerplay run rate     ~ Normal(theta_level, phase_scale / sqrt(balls))
  (b) aggressive attempts    ~ Binomial(faced balls, logit^-1(theta_attempt))
  (c) middle-overs run rate  ~ Normal(theta_level - theta_gear, base_scale / sqrt(balls))

so the gear-up is formed inside the model from the powerplay and baseline rates,
exactly as the nested single-leg model does, and the baseline never appears as a
precomputed delta.

Team-season effects: theta[s, t] ~ MVN(mu_s, Sigma_s) with
Sigma_s = diag(sigma_s) Omega diag(sigma_s). Omega is one correlation matrix shared
across seasons (LKJ(2) prior); sigma_s is season-specific (HalfNormal, refit with
Exponential). Non-centered: theta = mu_s + diag(sigma_s) L z with L the Cholesky
factor of Omega and z standard normal.

Estimands:
  - log(sigma_2024,k / sigma_2023,k) for each of the three outcomes, with P(widen);
  - the posterior correlation Omega (how far the legs are one signal);
  - from a second fit with a shared factor, theta[s, t] = mu_s + lambda * eta[s, t]
    + eps[s, t] with eta ~ Normal(0, tau_s^2), a single P(tau_2024 > tau_2023):
    the one-number answer to "did powerplay batting stratify". The level loading is
    fixed at one to identify the factor's scale and sign; residual scales are
    shared across seasons so a change in tau is a change in the common spread.

Shared-factor robustness (`shared_factor_robustness`): the primary fit is repeated
with the loading fixed on the gear-up instead of the level, with season-specific
residual scales, and with the tagged attempt outcome dropped. P(tau widened) should
sit in the same region across all four for the result to stand as established.

The tagging-clean versus tagged argument still holds as a robustness statement:
the attempt outcome is the only tagged one, and the level and gear-up outcomes
rest on score and over_num alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mean_level_and_tail import season_contrast_bootstrap as scb
from .hierarchical_dispersion_models import _sample, convergence_checks, team_innings_phase_pair
from .team_spread_bootstrap import AGGRESSIVE_SHOTS

OUTCOMES = ("run_rate_level", "attempt_log_odds", "gear_up_from_baseline")
# HalfNormal scale per outcome for the between-team SD prior; the Exponential refit
# uses rate 1/scale so the prior mean matches.
SD_PRIOR_SCALE = np.array([1.5, 0.5, 1.5])


def team_innings_outcomes(frame, powerplay, middle) -> pd.DataFrame:
    """One row per team-innings with powerplay runs and balls, middle-overs runs and
    balls (the pair the nested model uses, same minimum-ball filters), plus
    aggressive attempts and faced balls in the powerplay."""
    paired = team_innings_phase_pair(frame, powerplay, middle)
    faced = scb.is_ball_faced(frame) & scb.in_phase(frame, powerplay)
    shot_upper = frame["shot"].astype(str).str.upper()
    work = frame.loc[faced, ["year", "team_bat", "p_match", "inns"]].copy()
    work["attempt"] = shot_upper[faced].isin(AGGRESSIVE_SHOTS).astype(int).to_numpy()
    attempts = (work.groupby(["year", "team_bat", "p_match", "inns"])
                .agg(attempts=("attempt", "sum"), faced=("attempt", "size")).reset_index())
    out = paired.merge(attempts, on=["year", "team_bat", "p_match", "inns"], how="inner")
    return out[out["faced"] > 0].reset_index(drop=True)


def _index_structure(obs):
    seasons = np.array([scb.EARLIER_SEASON, scb.LATER_SEASON])
    teams = np.sort(obs["team_bat"].unique())
    team_ix = {t: i for i, t in enumerate(teams)}
    season_ix = {s: i for i, s in enumerate(seasons)}
    row_team = obs["team_bat"].map(team_ix).to_numpy()
    row_season = obs["year"].map(season_ix).to_numpy()
    return seasons, teams, season_ix, row_team, row_season


def fit_multivariate_stratification(frame, powerplay, middle, mcmc: dict, seed: int,
                                    prior: str = "halfnormal"):
    """The joint model. Returns (idata, teams, seasons)."""
    import pymc as pm
    import pytensor.tensor as pt

    obs = team_innings_outcomes(frame, powerplay, middle)
    seasons, teams, season_ix, row_team, row_season = _index_structure(obs)
    n_teams, n_seasons, k = len(teams), len(seasons), len(OUTCOMES)

    phase_rate_obs = (obs["phase_runs"] / obs["phase_balls"] * 6.0).to_numpy()
    base_rate_obs = (obs["base_runs"] / obs["base_balls"] * 6.0).to_numpy()
    phase_balls = obs["phase_balls"].to_numpy()
    base_balls = obs["base_balls"].to_numpy()
    attempts = obs["attempts"].to_numpy()
    faced = obs["faced"].to_numpy()

    with pm.Model():
        mu = pm.Normal("mu", mu=np.array([8.0, np.log(0.15 / 0.85), 1.0]),
                       sigma=np.array([3.0, 1.0, 3.0]), shape=(n_seasons, k))
        if prior == "halfnormal":
            sigma = pm.HalfNormal("sigma", sigma=SD_PRIOR_SCALE, shape=(n_seasons, k))
        elif prior == "exponential":
            sigma = pm.Exponential("sigma", lam=1.0 / SD_PRIOR_SCALE, shape=(n_seasons, k))
        else:
            raise ValueError(prior)
        # pm.LKJCorr (PyMC 6) returns the lower Cholesky factor of the correlation
        # matrix (rows of unit norm), so it feeds the non-centered transform
        # directly; the correlation matrix itself is L L^T, kept for reporting.
        chol_omega = pm.LKJCorr("omega_chol", n=k, eta=2.0)
        pm.Deterministic("omega_corr", chol_omega @ chol_omega.T)

        z = pm.Normal("z", 0.0, 1.0, shape=(n_seasons, n_teams, k))
        # theta[s, t, :] = mu[s] + diag(sigma[s]) @ L @ z[s, t]; batched over s and t.
        scaled = sigma[:, None, :] * pt.einsum("kj,stj->stk", chol_omega, z)
        theta = pm.Deterministic("theta", mu[:, None, :] + scaled)

        level = theta[row_season, row_team, 0]
        attempt_logit = theta[row_season, row_team, 1]
        gear = theta[row_season, row_team, 2]

        phase_scale = pm.HalfNormal("phase_scale", sigma=4.0)
        base_scale = pm.HalfNormal("base_scale", sigma=4.0)
        pm.Normal("phase_like", mu=level, sigma=phase_scale / np.sqrt(phase_balls),
                  observed=phase_rate_obs)
        pm.Normal("base_like", mu=level - gear, sigma=base_scale / np.sqrt(base_balls),
                  observed=base_rate_obs)
        pm.Binomial("attempt_like", n=faced, logit_p=attempt_logit, observed=attempts)

        later, earlier = season_ix[scb.LATER_SEASON], season_ix[scb.EARLIER_SEASON]
        pm.Deterministic("log_sd_ratio", pt.log(sigma[later]) - pt.log(sigma[earlier]))
        pm.Deterministic("sd_difference", sigma[later] - sigma[earlier])
        idata = _sample(pm, mcmc, seed)

    return idata, teams, seasons


def fit_shared_factor(frame, powerplay, middle, mcmc: dict, seed: int,
                      prior: str = "halfnormal", reference_outcome: str = "run_rate_level",
                      season_specific_residuals: bool = False, include_attempt: bool = True):
    """Shared-factor refit: theta[s, t] = mu_s + lambda * eta[s, t] + eps[s, t],
    eta ~ Normal(0, tau_s), eps_k ~ Normal(0, psi_k). Returns (idata, teams, seasons).

    Robustness levers, each a structural choice the primary fit makes silently:
      reference_outcome        : which loading is fixed at one to identify the
                                 factor's scale and sign (primary: run-rate level).
      season_specific_residuals: psi_k per season instead of shared, so a widening
                                 cannot be forced into tau by holding residuals equal.
      include_attempt          : False drops the tagged attempt outcome and fits the
                                 factor on the two clean outcomes only.
    """
    import pymc as pm
    import pytensor.tensor as pt

    outcomes = list(OUTCOMES) if include_attempt else [o for o in OUTCOMES if o != "attempt_log_odds"]
    if reference_outcome not in outcomes:
        raise ValueError(f"reference outcome {reference_outcome!r} not among {outcomes}")
    k = len(outcomes)
    ref = outcomes.index(reference_outcome)
    col = {name: j for j, name in enumerate(outcomes)}
    prior_scale = np.array([SD_PRIOR_SCALE[list(OUTCOMES).index(o)] for o in outcomes])
    prior_mu = np.array([{"run_rate_level": 8.0, "attempt_log_odds": np.log(0.15 / 0.85),
                          "gear_up_from_baseline": 1.0}[o] for o in outcomes])
    prior_mu_sd = np.array([{"run_rate_level": 3.0, "attempt_log_odds": 1.0,
                             "gear_up_from_baseline": 3.0}[o] for o in outcomes])

    obs = team_innings_outcomes(frame, powerplay, middle)
    seasons, teams, season_ix, row_team, row_season = _index_structure(obs)
    n_teams, n_seasons = len(teams), len(seasons)

    phase_rate_obs = (obs["phase_runs"] / obs["phase_balls"] * 6.0).to_numpy()
    base_rate_obs = (obs["base_runs"] / obs["base_balls"] * 6.0).to_numpy()
    phase_balls = obs["phase_balls"].to_numpy()
    base_balls = obs["base_balls"].to_numpy()
    attempts = obs["attempts"].to_numpy()
    faced = obs["faced"].to_numpy()

    psi_shape = (n_seasons, k) if season_specific_residuals else (k,)
    with pm.Model():
        mu = pm.Normal("mu", mu=prior_mu, sigma=prior_mu_sd, shape=(n_seasons, k))
        if prior == "halfnormal":
            tau = pm.HalfNormal("tau", sigma=1.0, shape=n_seasons)
            psi = pm.HalfNormal("psi", sigma=np.broadcast_to(prior_scale, psi_shape), shape=psi_shape)
        elif prior == "exponential":
            tau = pm.Exponential("tau", lam=1.0, shape=n_seasons)
            psi = pm.Exponential("psi", lam=np.broadcast_to(1.0 / prior_scale, psi_shape), shape=psi_shape)
        else:
            raise ValueError(prior)
        # One loading fixed at one so the factor has a scale and a sign; the others free.
        free_loadings = pm.Normal("free_loadings", 0.0, 1.0, shape=k - 1)
        loading_parts = []
        free_i = 0
        for j in range(k):
            if j == ref:
                loading_parts.append(pt.ones(1))
            else:
                loading_parts.append(free_loadings[free_i:free_i + 1])
                free_i += 1
        loadings = pm.Deterministic("loadings", pt.concatenate(loading_parts))

        z_eta = pm.Normal("z_eta", 0.0, 1.0, shape=(n_seasons, n_teams))
        eta = pm.Deterministic("eta", tau[:, None] * z_eta)
        z_eps = pm.Normal("z_eps", 0.0, 1.0, shape=(n_seasons, n_teams, k))
        psi_b = psi[:, None, :] if season_specific_residuals else psi
        theta = pm.Deterministic(
            "theta", mu[:, None, :] + eta[:, :, None] * loadings[None, None, :] + psi_b * z_eps)

        level = theta[row_season, row_team, col["run_rate_level"]]
        gear = theta[row_season, row_team, col["gear_up_from_baseline"]]

        phase_scale = pm.HalfNormal("phase_scale", sigma=4.0)
        base_scale = pm.HalfNormal("base_scale", sigma=4.0)
        pm.Normal("phase_like", mu=level, sigma=phase_scale / np.sqrt(phase_balls),
                  observed=phase_rate_obs)
        pm.Normal("base_like", mu=level - gear, sigma=base_scale / np.sqrt(base_balls),
                  observed=base_rate_obs)
        if include_attempt:
            attempt_logit = theta[row_season, row_team, col["attempt_log_odds"]]
            pm.Binomial("attempt_like", n=faced, logit_p=attempt_logit, observed=attempts)

        later, earlier = season_ix[scb.LATER_SEASON], season_ix[scb.EARLIER_SEASON]
        pm.Deterministic("log_tau_ratio", pt.log(tau[later]) - pt.log(tau[earlier]))
        pm.Deterministic("tau_difference", tau[later] - tau[earlier])
        idata = _sample(pm, mcmc, seed)

    idata.attrs["outcomes"] = list(outcomes)
    return idata, teams, seasons


SHARED_FACTOR_VARIANTS = {
    # name: (reference_outcome, season_specific_residuals, include_attempt)
    "primary_level_reference": ("run_rate_level", False, True),
    "gear_up_reference": ("gear_up_from_baseline", False, True),
    "season_specific_residuals": ("run_rate_level", True, True),
    "clean_outcomes_only": ("run_rate_level", False, False),
}


def shared_factor_robustness(frame, powerplay, middle, mcmc: dict, seed: int,
                             interval_level: float = 0.95, prior: str = "halfnormal") -> pd.DataFrame:
    """The primary shared-factor fit and its three structural variants in one table:
    tau per season, log tau ratio with interval, P(tau widened), loadings, checks."""
    rows = []
    for name, (reference, season_psi, with_attempt) in SHARED_FACTOR_VARIANTS.items():
        idata, _, _ = fit_shared_factor(frame, powerplay, middle, mcmc, seed, prior=prior,
                                        reference_outcome=reference,
                                        season_specific_residuals=season_psi,
                                        include_attempt=with_attempt)
        row = summarise_shared_factor(idata, prior, interval_level)
        rows.append({"variant": name, "reference_outcome": reference,
                     "season_specific_residuals": season_psi, "include_attempt": with_attempt, **row})
    return pd.DataFrame(rows)


def summarise_multivariate(idata, prior: str, interval_level: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-outcome log SD ratio rows and the posterior mean correlation matrix."""
    post = idata.posterior
    lo_q, hi_q = 100 * (1 - interval_level) / 2, 100 * (1 + interval_level) / 2
    sigma = post["sigma"].values.reshape(-1, *post["sigma"].shape[-2:])   # (draws, seasons, k)
    log_ratio = post["log_sd_ratio"].values.reshape(-1, len(OUTCOMES))
    diff = post["sd_difference"].values.reshape(-1, len(OUTCOMES))
    rows = []
    for j, name in enumerate(OUTCOMES):
        rows.append({
            "outcome": name, "prior": prior,
            f"sd_{scb.EARLIER_SEASON}": float(np.median(sigma[:, 0, j])),
            f"sd_{scb.LATER_SEASON}": float(np.median(sigma[:, 1, j])),
            "log_sd_ratio": float(np.median(log_ratio[:, j])),
            "log_sd_ratio_low": float(np.percentile(log_ratio[:, j], lo_q)),
            "log_sd_ratio_high": float(np.percentile(log_ratio[:, j], hi_q)),
            "sd_ratio": float(np.exp(np.median(log_ratio[:, j]))),
            "sd_difference": float(np.median(diff[:, j])),
            "p_widen": float((diff[:, j] > 0).mean()),
            "interval_level": interval_level,
        })
    checks = convergence_checks(idata, ["sigma", "log_sd_ratio"])
    table = pd.DataFrame(rows)
    for key, value in checks.items():
        table[key] = value
    omega = post["omega_corr"].values.reshape(-1, len(OUTCOMES), len(OUTCOMES)).mean(axis=0)
    correlation = pd.DataFrame(omega, index=OUTCOMES, columns=OUTCOMES)
    return table, correlation


def summarise_shared_factor(idata, prior: str, interval_level: float = 0.95) -> dict:
    post = idata.posterior
    lo_q, hi_q = 100 * (1 - interval_level) / 2, 100 * (1 + interval_level) / 2
    outcomes = list(idata.attrs.get("outcomes", OUTCOMES))
    tau = post["tau"].values.reshape(-1, 2)
    log_ratio = post["log_tau_ratio"].values.ravel()
    loadings = post["loadings"].values.reshape(-1, len(outcomes))
    out = {
        "prior": prior,
        f"tau_{scb.EARLIER_SEASON}": float(np.median(tau[:, 0])),
        f"tau_{scb.LATER_SEASON}": float(np.median(tau[:, 1])),
        "log_tau_ratio": float(np.median(log_ratio)),
        "log_tau_ratio_low": float(np.percentile(log_ratio, lo_q)),
        "log_tau_ratio_high": float(np.percentile(log_ratio, hi_q)),
        "p_tau_widened": float((tau[:, 1] > tau[:, 0]).mean()),
        "interval_level": interval_level,
    }
    for name in OUTCOMES:
        out[f"loading_{name}"] = (float(np.median(loadings[:, outcomes.index(name)]))
                                  if name in outcomes else np.nan)
    out.update(convergence_checks(idata, ["tau", "log_tau_ratio"]))
    return out
