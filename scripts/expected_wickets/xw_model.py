"""
Situation-only xW: calibrated out-of-fold model, baseline, phase-aware calibration,
model-refit uncertainty, feature-mix decomposition, equivalence framing, and the
shot-level residual rung.

The estimand is descriptive: does the EXPECTED powerplay wicket rate differ across
seasons, set against the ACTUAL change? So the model must be honestly calibrated
inside the phase under test and evaluated out-of-fold with matches never split
across folds, and the xW used for the contrast must be out-of-fold or the contrast
is circular.

Design:
  - Nested match-grouped CV. Outer GroupKFold gives out-of-fold xW for every ball
    from a model trained on OTHER matches. Calibration is fit inside each outer
    training set on inner out-of-fold probabilities, with phase as a stratum
    (separate calibrators for the powerplay, middle and death), the method
    (isotonic or sigmoid) chosen per stratum by grouped log-loss, so no test match
    informs its own xW and the calibration bias is not averaged across phases.
  - Two base models through the identical pipeline: the fixed-hyperparameter
    histogram gradient booster and a logistic regression (over as a categorical,
    other features linear, missing required rate as an indicator). The better
    calibrated one by Brier carries the contrast; on a tie the logistic model does
    and the booster is the robustness check.
  - Two feature variants (situation; situation plus bowler type), both reported.
  - Scoring is calibration-first: reliability, Brier, log-loss; AUC only as the
    leakage sentinel.
  - The finding: mean powerplay xW per season against the actual rate, with the
    match cluster bootstrap on the fixed out-of-fold xW (all draws) and, on a subset
    of the same draws, the full pipeline refit on the resampled matches so the
    interval also carries model uncertainty. The wider is reported.
  - Equivalence framing against a pre-stated SESOI for the change in expected
    powerplay wickets per ball.
  - Feature-mix decomposition: with the final model held fixed, the later season's
    distribution of one feature at a time is substituted into the earlier season's
    powerplay rows, averaged over random feature orders (a sampled Shapley
    attribution of the mean-xW change to features).
  - Shot-level rung: the per-ball residual (actual minus xW) regressed on aggressive
    intent with a season interaction, match cluster bootstrap. The interaction is
    the direct test of whether aggressive shots became costlier; it carries the
    shot-tag floor.
League level only; no team or player xW on two seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from mean_level_and_tail import season_contrast_bootstrap as scb


def _sklearn_seed(seed: int) -> int:
    """scikit-learn accepts random_state up to 2**32 - 1; the project seed is a
    larger timestamp-shaped integer, so it is folded into that range here."""
    return int(seed) % (2 ** 32 - 1)

# Pre-committed, well-regularized booster. Low depth + L2 so a handful of situation
# features can't overfit; fixed so CV is evaluation, not selection.
HGB_PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=300,
                  l2_regularization=1.0, min_samples_leaf=40, early_stopping=False)
LOGIT_C = 1.0
PHASE_BINS = [0, 6, 15, 20]          # overs 1-6, 7-15, 16-20
_EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, _EPS, 1 - _EPS)


# ---------------------------------------------------------------------------
# Base models
# ---------------------------------------------------------------------------

class LogisticBaseline:
    """Logistic regression with over as a categorical (one indicator per over),
    the other features linear and standardised, and a missing-required-rate
    indicator (NaN outside a chase). Same fit/predict_proba surface as the booster
    so the nested pipeline treats the two identically."""

    def __init__(self, C: float = LOGIT_C, seed: int = 0):
        self.C = C
        self.seed = seed

    def _design(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X = X.copy()
        over = pd.to_numeric(X.pop("over_num"), errors="coerce").astype(int).to_numpy()
        over_dummies = (over[:, None] == np.arange(1, 21)[None, :]).astype(float)
        cols = list(X.columns)
        values = X.to_numpy(dtype=float)
        missing = np.isnan(values).astype(float)
        values = np.where(np.isnan(values), 0.0, values)
        if fit:
            self._mean = values.mean(axis=0)
            self._sd = np.where(values.std(axis=0) > 0, values.std(axis=0), 1.0)
            self._missing_cols = [i for i in range(len(cols)) if missing[:, i].any()]
        z = (values - self._mean) / self._sd
        parts = [over_dummies, z]
        if self._missing_cols:
            parts.append(missing[:, self._missing_cols])
        return np.hstack(parts)

    def fit(self, X, y, sample_weight=None):
        design = self._design(X, fit=True)
        self.model_ = LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")
        self.model_.fit(design, np.asarray(y), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._design(X))


def make_base_model(kind: str, seed: int):
    if kind == "hgb":
        return HistGradientBoostingClassifier(**HGB_PARAMS, random_state=_sklearn_seed(seed))
    if kind == "logistic":
        return LogisticBaseline(seed=seed)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Calibration, phase-aware
# ---------------------------------------------------------------------------

def phase_index(over_num) -> np.ndarray:
    """0 powerplay, 1 middle, 2 death from the over number."""
    over = np.asarray(over_num, dtype=float)
    return np.digitize(over, PHASE_BINS[1:-1], right=True)


def _fit_calibrator(p: np.ndarray, y: np.ndarray, method: str):
    """Fit a 1-D calibrator mapping raw probability -> calibrated probability."""
    p = _clip(np.asarray(p, dtype=float))
    y = np.asarray(y, dtype=float)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    else:  # sigmoid (Platt) on the log-odds
        z = np.log(p / (1 - p)).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs").fit(z, y)
    return (method, model)


def _apply_calibrator(cal, p: np.ndarray) -> np.ndarray:
    method, model = cal
    p = _clip(np.asarray(p, dtype=float))
    if method == "isotonic":
        return model.predict(p)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return model.predict_proba(z)[:, 1]


def _select_calibration_method(p_inner, y, groups) -> str:
    """Pick isotonic vs sigmoid by grouped-CV log-loss on the inner-OOF probabilities."""
    y = np.asarray(y, dtype=float)
    p_inner = np.asarray(p_inner, dtype=float)
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 3 or y.sum() < 5:
        return "sigmoid"
    scores = {"isotonic": [], "sigmoid": []}
    for method in scores:
        for a, b in GroupKFold(3).split(p_inner.reshape(-1, 1), y, groups):
            cal = _fit_calibrator(p_inner[a], y[a], method)
            pb = _clip(_apply_calibrator(cal, p_inner[b]))
            scores[method].append(log_loss(y[b], pb, labels=[0, 1]))
    return min(scores, key=lambda m: float(np.mean(scores[m])))


class PhaseCalibrator:
    """One calibrator per phase stratum, each with its own method choice. Fit on
    inner out-of-fold probabilities of the outer training matches; applied by the
    phase of the ball being scored."""

    def __init__(self, phases: np.ndarray):
        self.phases = np.asarray(phases)

    def fit(self, p_inner, y, groups, phases):
        phases = np.asarray(phases)
        self.calibrators_ = {}
        self.methods_ = {}
        for ph in np.unique(self.phases):
            m = phases == ph
            method = _select_calibration_method(p_inner[m], np.asarray(y)[m], np.asarray(groups)[m])
            self.calibrators_[int(ph)] = _fit_calibrator(p_inner[m], np.asarray(y)[m], method)
            self.methods_[int(ph)] = method
        return self

    def transform(self, p, phases):
        phases = np.asarray(phases)
        out = np.full(len(p), np.nan)
        for ph, cal in self.calibrators_.items():
            m = phases == ph
            if m.any():
                out[m] = _apply_calibrator(cal, np.asarray(p)[m])
        return out


# ---------------------------------------------------------------------------
# Nested pipeline
# ---------------------------------------------------------------------------

def _inner_oof_base_probs(X, y, groups, n_inner, seed, kind, sample_weight=None):
    """Grouped out-of-fold uncalibrated probabilities across the training matches,
    so the calibrator is fit on scores no base model saw during its own training."""
    oof = np.full(len(y), np.nan)
    for itr, ite in GroupKFold(n_inner).split(X, y, groups):
        base = make_base_model(kind, seed)
        sw = None if sample_weight is None else np.asarray(sample_weight)[itr]
        base.fit(X.iloc[itr], y.iloc[itr], sample_weight=sw)
        oof[ite] = base.predict_proba(X.iloc[ite])[:, 1]
    return oof


def fit_calibrated_model(X, y, groups, phases, kind: str, n_inner: int, seed: int,
                         sample_weight=None):
    """Base model fit on all given rows plus the phase-aware calibrator fit on inner
    out-of-fold probabilities. Returns (base, calibrator)."""
    p_inner = _inner_oof_base_probs(X, y, groups, n_inner, seed, kind, sample_weight)
    cal = PhaseCalibrator(np.unique(phases)).fit(p_inner, y.to_numpy(), groups, phases)
    base = make_base_model(kind, seed)
    base.fit(X, y, sample_weight=sample_weight)
    return base, cal


def oof_xw(X: pd.DataFrame, y: pd.Series, groups: pd.Series, kind: str = "hgb",
           n_outer: int = 5, n_inner: int = 4, seed: int = 0, sample_weight=None):
    """Out-of-fold, phase-calibrated xW for every ball. Returns (oof_probs, methods
    per outer fold)."""
    phases = phase_index(X["over_num"])
    oof = np.full(len(y), np.nan)
    chosen = []
    for tr, te in GroupKFold(n_outer).split(X, y, groups):
        sw = None if sample_weight is None else np.asarray(sample_weight)[tr]
        base, cal = fit_calibrated_model(X.iloc[tr], y.iloc[tr], groups.iloc[tr], phases[tr],
                                         kind, n_inner, seed, sw)
        chosen.append(cal.methods_)
        oof[te] = cal.transform(base.predict_proba(X.iloc[te])[:, 1], phases[te])
    return oof, chosen


def score_xw(y: pd.Series, xw: np.ndarray) -> dict:
    """Calibration-first scores. AUC is the leakage sentinel, not the objective."""
    y = np.asarray(y, dtype=float)
    xw = _clip(xw)
    return {
        "base_rate": float(y.mean()),
        "mean_xw": float(xw.mean()),
        "brier": float(brier_score_loss(y, xw)),
        "log_loss": float(log_loss(y, xw, labels=[0, 1])),
        "auc_leakage_sentinel": float(roc_auc_score(y, xw)),
    }


def reliability_curve(y: pd.Series, xw: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Binned predicted vs observed wicket rate (equal-count bins)."""
    y = np.asarray(y, dtype=float)
    xw = np.asarray(xw, dtype=float)
    order = np.argsort(xw)
    bins = np.array_split(order, n_bins)
    rows = [{"mean_predicted": float(xw[b].mean()),
             "observed_rate": float(y[b].mean()),
             "n": int(len(b))} for b in bins if len(b)]
    return pd.DataFrame(rows)


def holdout_evaluation(X, y, groups, kind: str = "hgb", test_frac: float = 0.2, seed: int = 0) -> tuple:
    """Fit the pipeline once on a random set of matches, evaluate on the held-out
    ones: the uncontaminated check that the calibrated model generalizes."""
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=_sklearn_seed(seed)).split(X, y, groups))
    phases = phase_index(X["over_num"])
    base, cal = fit_calibrated_model(X.iloc[tr], y.iloc[tr], groups.iloc[tr], phases[tr], kind, 4, seed)
    xw_te = cal.transform(base.predict_proba(X.iloc[te])[:, 1], phases[te])
    out = score_xw(y.iloc[te], xw_te)
    out.update({"calibration_methods": cal.methods_, "n_holdout_matches": int(groups.iloc[te].nunique()),
                "n_holdout_balls": int(len(te))})
    return out, reliability_curve(y.iloc[te], xw_te)


def season_phase_bias(xw: np.ndarray, y: pd.Series, meta: pd.DataFrame) -> pd.DataFrame:
    """Season x phase table of mean(xW) - mean(actual): the calibration bias shown
    where the contrast lives, not asserted."""
    tmp = meta[["year", "phase"]].assign(xw=np.asarray(xw), y=np.asarray(y, dtype=float))
    g = tmp.groupby(["year", "phase"], observed=True).agg(mean_xw=("xw", "mean"), actual=("y", "mean"), n=("y", "size"))
    g["bias_xw_minus_actual"] = g["mean_xw"] - g["actual"]
    return g.reset_index()


def powerplay_reliability_by_season(xw, y, meta, n_bins: int = 6) -> pd.DataFrame:
    rows = []
    pp = meta["phase"].astype(str).eq("powerplay").to_numpy()
    for yr in sorted(meta["year"].unique()):
        m = pp & (meta["year"].to_numpy() == yr)
        rc = reliability_curve(np.asarray(y)[m], np.asarray(xw)[m], n_bins)
        rc.insert(0, "year", int(yr))
        rows.append(rc)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# The finding: powerplay xW vs actual, by season, match cluster bootstrap
# ---------------------------------------------------------------------------

def _season_match_stats(xw, y, meta, groups):
    """Per (season, match) sums of xW, actual and ball count on powerplay legal balls."""
    pp = meta["phase"].astype(str).eq("powerplay").to_numpy()
    df = pd.DataFrame({"year": meta["year"].to_numpy()[pp], "p_match": np.asarray(groups)[pp],
                       "xw": np.asarray(xw)[pp], "y": np.asarray(y, dtype=float)[pp]})
    per = {}
    for yr, s in df.groupby("year"):
        agg = s.groupby("p_match").agg(sum_xw=("xw", "sum"), sum_y=("y", "sum"), n=("xw", "size"))
        per[int(yr)] = agg
    return per


def powerplay_xw_contrast(xw, y, meta, groups, earlier: int, later: int, counts: tuple,
                          alpha: float = 0.05) -> dict:
    """Mean powerplay xW and actual wicket rate per season and the season contrast,
    resampling whole matches with the given multiplicity draws (shared across the
    analysis), recomputing both rates from the fixed out-of-fold per-ball xW."""
    per = _season_match_stats(xw, y, meta, groups)
    counts_e, counts_l = counts
    mats = {earlier: per[earlier].to_numpy(dtype=float), later: per[later].to_numpy(dtype=float)}
    assert mats[earlier].shape[0] == counts_e.shape[1] and mats[later].shape[0] == counts_l.shape[1]

    def rates(mat, c):
        tot = c @ mat                     # (n_iter, 3): sum_xw, sum_y, n
        return tot[:, 0] / tot[:, 2], tot[:, 1] / tot[:, 2]

    xw_e, y_e = rates(mats[earlier], counts_e)
    xw_l, y_l = rates(mats[later], counts_l)
    point_xw = {yr: float(mats[yr][:, 0].sum() / mats[yr][:, 2].sum()) for yr in (earlier, later)}
    point_y = {yr: float(mats[yr][:, 1].sum() / mats[yr][:, 2].sum()) for yr in (earlier, later)}
    draws = {"xw_change": xw_l - xw_e, "actual_change": y_l - y_e}
    draws["gap_xw_minus_actual"] = draws["xw_change"] - draws["actual_change"]

    def band(a, level):
        lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
        return float(np.percentile(a, lo)), float(np.percentile(a, hi))

    out = {"per_season": {yr: {"mean_xw": point_xw[yr], "actual_rate": point_y[yr]} for yr in (earlier, later)},
           "n_resamples": int(counts_e.shape[0])}
    points = {"xw_change": point_xw[later] - point_xw[earlier],
              "actual_change": point_y[later] - point_y[earlier]}
    points["gap_xw_minus_actual"] = points["xw_change"] - points["actual_change"]
    for name, a in draws.items():
        lo, hi = band(a, 1 - alpha)
        lo90, hi90 = band(a, 0.90)
        out[name] = {"point": float(points[name]), "mean": float(a.mean()), "se": float(a.std(ddof=1)),
                     "lo": lo, "hi": hi, "lo90": lo90, "hi90": hi90}
    out["_draws"] = draws
    return out


def model_refit_contrast(X, y, groups, meta, kind: str, earlier: int, later: int,
                         counts: tuple, n_refits: int, seed: int, n_jobs: int = 1) -> dict:
    """Refit the full nested pipeline on the resampled matches for the first
    n_refits shared draws and recompute the powerplay contrast on each, so the
    interval carries model uncertainty as well as resampling. Each resample is
    represented by its unique matches with the multiplicity as sample weight, so a
    match never sits on both sides of a fold split."""
    from joblib import Parallel, delayed

    counts_e, counts_l = counts
    per = _season_match_stats(np.zeros(len(y)), y, meta, groups)
    match_ids = {earlier: per[earlier].index.to_numpy(), later: per[later].index.to_numpy()}
    group_arr = np.asarray(groups)
    year_arr = meta["year"].to_numpy()

    def one(i):
        weight = np.zeros(len(y))
        for yr, c in ((earlier, counts_e[i]), (later, counts_l[i])):
            w = dict(zip(match_ids[yr], c))
            m = year_arr == yr
            weight[m] = pd.Series(group_arr[m]).map(w).fillna(0.0).to_numpy()
        keep = weight > 0
        Xs = X.loc[keep].reset_index(drop=True)
        ys = y.loc[keep].reset_index(drop=True)
        gs = groups.loc[keep].reset_index(drop=True)
        meta_s = meta.loc[keep].reset_index(drop=True)
        xw_s, _ = oof_xw(Xs, ys, gs, kind=kind, seed=seed + i, sample_weight=weight[keep])
        # Contrast on this resample: per-match sums of the refit xW, weighted by the
        # same multiplicities that defined the resample.
        per_s = _season_match_stats(xw_s, ys, meta_s, gs)
        rate = {}
        for yr, c in ((earlier, counts_e[i]), (later, counts_l[i])):
            mat = per_s[yr].reindex(match_ids[yr], fill_value=0.0).to_numpy(dtype=float)
            tot = c @ mat
            rate[yr] = (tot[0] / tot[2], tot[1] / tot[2])
        xw_change = rate[later][0] - rate[earlier][0]
        actual_change = rate[later][1] - rate[earlier][1]
        return {"xw_change": float(xw_change), "actual_change": float(actual_change),
                "gap_xw_minus_actual": float(xw_change - actual_change)}

    rows = Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(n_refits))
    draws = pd.DataFrame(rows)
    out = {"n_refits": int(n_refits)}
    for name in draws.columns:
        a = draws[name].to_numpy()
        out[name] = {"mean": float(a.mean()), "se": float(a.std(ddof=1)),
                     "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5)),
                     "lo90": float(np.percentile(a, 5)), "hi90": float(np.percentile(a, 95))}
    out["_draws"] = draws
    return out


def interval_comparison(fixed: dict, refit: dict, n_refits: int) -> pd.DataFrame:
    """Fixed-model interval on all draws, fixed-model interval on the refit subset
    of draws, and the refit interval, per quantity, with the wider marked."""
    rows = []
    for name in ("xw_change", "actual_change", "gap_xw_minus_actual"):
        full = fixed[name]
        sub = fixed["_draws"][name][:n_refits]
        r = refit[name]
        rows.append({"quantity": name, "point": full["point"],
                     "fixed_lo": full["lo"], "fixed_hi": full["hi"],
                     "fixed_subset_lo": float(np.percentile(sub, 2.5)), "fixed_subset_hi": float(np.percentile(sub, 97.5)),
                     "refit_lo": r["lo"], "refit_hi": r["hi"],
                     "fixed_width": full["hi"] - full["lo"], "refit_width": r["hi"] - r["lo"],
                     "reported": "refit" if (r["hi"] - r["lo"]) > (full["hi"] - full["lo"]) else "fixed"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature-mix decomposition (sampled Shapley by sequential substitution)
# ---------------------------------------------------------------------------

def feature_mix_decomposition(base, cal, X, meta, earlier: int, later: int, n_orders: int, seed: int) -> pd.DataFrame:
    """With the final model fixed, attribute the change in mean powerplay xW between
    seasons to features: for each random feature order, substitute the later
    season's distribution of one feature at a time (random draw with replacement
    from the later powerplay rows) into the earlier season's powerplay rows, and
    record the increment in mean xW. Averaging over orders gives an order-free
    attribution; the increments sum to the full substitution change."""
    rng = np.random.default_rng(seed)
    pp = meta["phase"].astype(str).eq("powerplay").to_numpy()
    X_e = X.loc[pp & (meta["year"].to_numpy() == earlier)].reset_index(drop=True)
    X_l = X.loc[pp & (meta["year"].to_numpy() == later)].reset_index(drop=True)
    phases_e = phase_index(X_e["over_num"])
    features = list(X.columns)

    def mean_xw(frame):
        return float(cal.transform(base.predict_proba(frame)[:, 1], phases_e).mean())

    base_mean = mean_xw(X_e)
    contributions = {f: [] for f in features}
    for _ in range(n_orders):
        order = rng.permutation(features)
        current = X_e.copy()
        previous = base_mean
        for f in order:
            current[f] = rng.choice(X_l[f].to_numpy(), size=len(current), replace=True)
            now = mean_xw(current)
            contributions[f].append(now - previous)
            previous = now
    full_change = previous - base_mean
    rows = [{"feature": f, "contribution": float(np.mean(v)), "contribution_sd_over_orders": float(np.std(v))}
            for f, v in contributions.items()]
    table = pd.DataFrame(rows).sort_values("contribution", key=np.abs, ascending=False).reset_index(drop=True)
    table.attrs["base_mean_xw"] = base_mean
    table.attrs["full_substitution_change"] = float(full_change)
    return table


# ---------------------------------------------------------------------------
# Shot-level rung: residual on aggressive intent with a season interaction
# ---------------------------------------------------------------------------

def shot_level_rung(xw, y, meta, groups, earlier: int, later: int, counts: tuple) -> dict:
    """OLS of the per-ball residual (actual - xW) on aggressive intent, season, and
    their interaction, on faced powerplay legal balls, under the match cluster
    bootstrap (weighted least squares with the resample multiplicities, from
    per-match sufficient statistics). The interaction coefficient is the change in
    the residual wicket cost of an aggressive shot between seasons: null or
    negative means aggressive shots did not become costlier. Shot tags are manual,
    so the interval is a floor on the true uncertainty."""
    pp = meta["phase"].astype(str).eq("powerplay").to_numpy() & (meta["faced"].to_numpy() == 1)
    resid = np.asarray(y, dtype=float)[pp] - np.asarray(xw)[pp]
    aggressive = meta["aggressive_shot"].to_numpy(dtype=float)[pp]
    season = (meta["year"].to_numpy()[pp] == later).astype(float)
    design = np.column_stack([np.ones(pp.sum()), aggressive, season, aggressive * season])
    match = np.asarray(groups)[pp]
    year = meta["year"].to_numpy()[pp]

    counts_e, counts_l = counts
    per = _season_match_stats(np.zeros(len(y)), y, meta, groups)
    match_ids = {earlier: per[earlier].index.to_numpy(), later: per[later].index.to_numpy()}

    def sufficient(yr):
        xtx = np.zeros((len(match_ids[yr]), 4, 4))
        xty = np.zeros((len(match_ids[yr]), 4))
        idx = {m: i for i, m in enumerate(match_ids[yr])}
        rows = np.where(year == yr)[0]
        for r in rows:
            i = idx[match[r]]
            d = design[r]
            xtx[i] += np.outer(d, d)
            xty[i] += d * resid[r]
        return xtx.reshape(len(match_ids[yr]), 16), xty

    xtx_e, xty_e = sufficient(earlier)
    xtx_l, xty_l = sufficient(later)

    def solve(c_e, c_l):
        xtx = (c_e @ xtx_e + c_l @ xtx_l).reshape(-1, 4, 4)
        xty = c_e @ xty_e + c_l @ xty_l
        return np.array([np.linalg.lstsq(a, b, rcond=None)[0] for a, b in zip(xtx, xty)])

    point = solve(np.ones((1, len(match_ids[earlier]))), np.ones((1, len(match_ids[later]))))[0]
    reps = solve(counts_e, counts_l)
    names = ["intercept", "aggressive", "later_season", "aggressive_x_later_season"]
    rows = []
    for j, name in enumerate(names):
        a = reps[:, j]
        rows.append({"coefficient": name, "estimate": float(point[j]), "se": float(a.std(ddof=1)),
                     "ci_low": float(np.percentile(a, 2.5)), "ci_high": float(np.percentile(a, 97.5)),
                     "p_positive": float((a > 0).mean())})
    cells = (pd.DataFrame({"year": year, "aggressive": aggressive.astype(int), "residual": resid})
             .groupby(["year", "aggressive"]).agg(mean_residual=("residual", "mean"), n=("residual", "size")).reset_index())
    return {"coefficients": pd.DataFrame(rows), "cell_means": cells, "n_balls": int(pp.sum())}
