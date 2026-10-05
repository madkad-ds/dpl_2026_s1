"""
Over-behaviour profiles and the feature transform that feeds break detection.

Unit of analysis: the (year, over_num) cell -- each of the 20 overs, per season,
summarised by league behavioural rates aggregated over every match in the season.
Each cell is built from thousands of deliveries, which is what makes this
unsupervised analysis well powered despite living on 20 points per season.

Feature families kept apart on purpose:
  CLEAN channels drive break detection (no tagging floor):
    run_rate     : 6 * sum(score) / legal balls        (score incl. extras; legal den.)
    boundary_rate: (fours + sixes) / faced balls        (batruns 4/6; faced den.)
    dot_rate     : batruns==0 on faced legal balls / faced
    wicket_rate  : dismissals (out) on faced balls / faced
    pace_share   : pace legal balls / legal balls        (bowl_kind substring 'pace')
  TAGGED channel is carried for the descriptive map only, never in the break matrix:
    aggressive_share : aggressive shots / faced balls    (tagging floor)

Denominators split by feature (legal vs faced) exactly as the register/harness do;
each is carried per cell so the inverse-variance weighting uses the right count.

Transform agreed for this analysis (both refit inside every bootstrap resample):
  1. Pooled robust scaling (median/IQR over BOTH seasons' rows) -- one coordinate
     system so a genuine cross-season shift survives into the geometry, robust so the
     extreme death overs don't distort it on 20 rows.
  2. Inverse-variance feature weighting -- weight each feature by how well its
     per-cell rate is estimated, so a noisy channel (low-base wicket rate) can't
     dominate through a small denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import derived_columns as dc

CLEAN_FEATURES = ["run_rate", "boundary_rate", "dot_rate", "wicket_rate", "pace_share"]
TAGGED_FEATURES = ["aggressive_share"]

# Feature groups analysed separately by design. Batting channels carry the break
# test; pace share is a bowling-deployment choice and is read as a single channel.
BATTING_FEATURES = ["run_rate", "boundary_rate", "dot_rate", "wicket_rate"]
DEPLOYMENT_FEATURES = ["pace_share"]
# Pre-stated weights for the batting matrix. The wicket down-weight is a design
# decision (a low-base rate on 20 cells is the noisiest channel), not a quantity
# estimated from sampling variance, so it cannot hand one channel the geometry.
FIXED_BATTING_WEIGHTS = {"run_rate": 1.0, "boundary_rate": 1.0, "dot_rate": 1.0, "wicket_rate": 0.5}

# Which denominator each feature is a rate over: 'legal' or 'faced'. Drives both the
# rate and its sampling-variance estimate.
FEATURE_DENOMINATOR = {
    "run_rate": "legal",
    "pace_share": "legal",
    "boundary_rate": "faced",
    "dot_rate": "faced",
    "wicket_rate": "faced",
    "aggressive_share": "faced",
}


def build_over_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (year, over_num): league behavioural rates plus the denominators
    and per-cell run variance the inverse-variance weighting needs."""
    work = frame.copy()
    legal = dc.is_legal_delivery(work)
    faced = dc.is_ball_faced(work)
    work["_legal"] = legal.to_numpy()
    work["_faced"] = faced.to_numpy()
    work["_boundary_faced"] = (dc.is_boundary(work) & faced).to_numpy()
    work["_dot_faced"] = dc.is_dot_off_bat(work).to_numpy()  # already faced & legal
    work["_wicket_faced"] = (dc.is_wicket(work) & faced).to_numpy()
    work["_pace_legal"] = (dc.is_pace(work) & legal).to_numpy()
    work["_aggressive_faced"] = (dc.is_aggressive_shot(work) & faced).to_numpy()
    work["_score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0).to_numpy()

    records = []
    for (year, over_num), cell in work.groupby(["year", "over_num"], sort=True):
        n_legal = int(cell["_legal"].sum())
        n_faced = int(cell["_faced"].sum())
        legal_scores = cell.loc[cell["_legal"], "_score"].to_numpy(dtype=float)
        run_ball_var = float(np.var(legal_scores, ddof=1)) if len(legal_scores) > 1 else 0.0
        records.append({
            "year": int(year), "over_num": int(over_num),
            "n_legal": n_legal, "n_faced": n_faced,
            "run_rate": 6.0 * legal_scores.sum() / n_legal if n_legal else np.nan,
            "boundary_rate": cell["_boundary_faced"].sum() / n_faced if n_faced else np.nan,
            "dot_rate": cell["_dot_faced"].sum() / n_faced if n_faced else np.nan,
            "wicket_rate": cell["_wicket_faced"].sum() / n_faced if n_faced else np.nan,
            "pace_share": cell["_pace_legal"].sum() / n_legal if n_legal else np.nan,
            "aggressive_share": cell["_aggressive_faced"].sum() / n_faced if n_faced else np.nan,
            "_run_ball_var": run_ball_var,
        })
    return pd.DataFrame.from_records(records).sort_values(["year", "over_num"]).reset_index(drop=True)


def _cell_sampling_variance(profiles: pd.DataFrame, feature: str) -> np.ndarray:
    """Estimation variance of a feature's per-cell value, in the feature's own units.

    Proportions: p(1-p)/n over the feature's denominator. Run rate (= 6 * mean
    per-legal-ball score): 36 * s^2 / n_legal with s^2 the per-ball score variance.
    """
    den_kind = FEATURE_DENOMINATOR[feature]
    n = profiles["n_legal" if den_kind == "legal" else "n_faced"].to_numpy(dtype=float)
    n = np.where(n > 0, n, np.nan)
    if feature == "run_rate":
        return 36.0 * profiles["_run_ball_var"].to_numpy(dtype=float) / n
    p = profiles[feature].to_numpy(dtype=float)
    return p * (1.0 - p) / n


@dataclass
class PooledRobustScaler:
    """Median/IQR scaler fitted on both seasons' rows pooled. Refit per resample."""

    features: list
    medians_: dict = field(default_factory=dict)
    iqrs_: dict = field(default_factory=dict)

    def fit(self, profiles: pd.DataFrame) -> "PooledRobustScaler":
        for feature in self.features:
            values = profiles[feature].to_numpy(dtype=float)
            q25, q75 = np.percentile(values, [25, 75])
            iqr = q75 - q25
            self.medians_[feature] = float(np.median(values))
            self.iqrs_[feature] = float(iqr) if iqr > 1e-12 else 1e-12  # floor a constant feature
        return self

    def transform(self, profiles: pd.DataFrame) -> pd.DataFrame:
        scaled = {f: (profiles[f].to_numpy(dtype=float) - self.medians_[f]) / self.iqrs_[f]
                  for f in self.features}
        return pd.DataFrame(scaled, index=profiles.index)


def _cap_weights(weights: dict, max_weight: float) -> dict:
    """Cap any single feature weight at max_weight, redistributing the freed budget
    across the uncapped features in proportion to their reliability, so the weights
    still average one. Water-filling: capping one feature can lift others past the
    cap, so repeat until stable. Guards against a single reliable channel (pace share)
    dominating the geometry -- the sensitivity lever for the pace-masking concern.
    """
    w = dict(weights)
    k = len(w)
    for _ in range(k):
        over = [f for f, v in w.items() if v > max_weight + 1e-9]
        if not over:
            break
        budget = k - max_weight * len(over)  # weight mass left for the uncapped features
        uncapped = {f: v for f, v in w.items() if f not in over}
        us = sum(uncapped.values())
        for f in over:
            w[f] = max_weight
        if us > 0 and budget > 0:
            for f in uncapped:
                w[f] = uncapped[f] * budget / us
        elif uncapped:
            for f in uncapped:
                w[f] = budget / len(uncapped)
    return w


def inverse_variance_weights(profiles: pd.DataFrame, scaler: PooledRobustScaler, features: list,
                             max_weight: float | None = None) -> dict:
    """Reliability weight per feature: inverse mean per-cell sampling variance in
    SCALED units, normalised so weights average one.

    Scaling divides a feature by its IQR, so its sampling variance divides by IQR^2;
    reliabilities are only comparable after that rescaling. Applied later as
    sqrt(weight) on the scaled column, so squared distance (PCA covariance) sees the
    weight itself. Averaging to one keeps the matrix's overall scale ~unchanged.

    max_weight (optional): cap any single weight at this value and redistribute, so no
    one channel dominates. None = pure inverse-variance (default).
    """
    mean_scaled_var = {}
    for feature in features:
        scaled_cell_var = _cell_sampling_variance(profiles, feature) / (scaler.iqrs_[feature] ** 2)
        mean_scaled_var[feature] = float(np.nanmean(scaled_cell_var))
    raw = {f: (1.0 / v if v > 0 else 0.0) for f, v in mean_scaled_var.items()}
    total = sum(raw.values())
    k = len(features)
    if total <= 0:
        return {f: 1.0 for f in features}
    weights = {f: w * k / total for f, w in raw.items()}
    if max_weight is not None:
        weights = _cap_weights(weights, max_weight)
    return weights


@dataclass
class BreakFeatureTransform:
    """Raw profile table -> weighted, scaled matrix that break detection runs on.
    Bundles scaler + weights so one fit() can be repeated inside each resample.
    Set weighting=False to compare against unweighted scaling.
    """

    features: list
    weighting: bool = True
    max_weight: float = None  # cap any single feature weight; None = uncapped
    scaler_: PooledRobustScaler = None
    weights_: dict = field(default_factory=dict)

    def fit(self, profiles: pd.DataFrame) -> "BreakFeatureTransform":
        self.scaler_ = PooledRobustScaler(features=self.features).fit(profiles)
        if self.weighting:
            self.weights_ = inverse_variance_weights(
                profiles, self.scaler_, self.features, max_weight=self.max_weight)
        else:
            self.weights_ = {f: 1.0 for f in self.features}
        return self

    def transform(self, profiles: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler_.transform(profiles)
        for feature in self.features:
            scaled[feature] = scaled[feature] * np.sqrt(self.weights_[feature])
        out = scaled.copy()
        out.insert(0, "over_num", profiles["over_num"].to_numpy())
        out.insert(0, "year", profiles["year"].to_numpy())
        return out

    def fit_transform(self, profiles: pd.DataFrame) -> pd.DataFrame:
        return self.fit(profiles).transform(profiles)


@dataclass
class FixedWeightTransform:
    """Raw profile table -> pooled-robust-scaled matrix with pre-stated feature
    weights. Same scaling as BreakFeatureTransform (median/IQR over both seasons'
    rows, refit inside every resample), but the weights are fixed by design rather
    than estimated from sampling variance. Weights are applied as sqrt(weight) on
    the scaled column so a squared-error cost sees the weight itself.
    """

    features: list
    weights: dict
    scaler_: PooledRobustScaler = None

    def fit(self, profiles: pd.DataFrame) -> "FixedWeightTransform":
        self.scaler_ = PooledRobustScaler(features=self.features).fit(profiles)
        return self

    def transform(self, profiles: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler_.transform(profiles)
        for feature in self.features:
            scaled[feature] = scaled[feature] * np.sqrt(self.weights.get(feature, 1.0))
        out = scaled.copy()
        out.insert(0, "over_num", profiles["over_num"].to_numpy())
        out.insert(0, "year", profiles["year"].to_numpy())
        return out

    def fit_transform(self, profiles: pd.DataFrame) -> pd.DataFrame:
        return self.fit(profiles).transform(profiles)


def per_season_matrix(feature_matrix: pd.DataFrame, year: int, features: list) -> np.ndarray:
    """One season's 20 x k array, ordered by over, for PCA / break detection."""
    rows = feature_matrix[feature_matrix["year"] == year].sort_values("over_num")
    return rows[features].to_numpy(dtype=float)
