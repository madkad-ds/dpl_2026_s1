"""
Expected wickets (xW) analysis driver: runs the pipeline for both feature variants
and both base models, chooses the model per variant by calibration, and produces
every reported quantity on one shared set of match draws.

Order of operations, per feature variant:
  1. feature audit and leakage tripwire;
  2. out-of-fold, phase-calibrated xW from the booster and from the logistic
     baseline; calibration-first scores for both; the model with the lower Brier
     carries the contrast, the other is the robustness check (a tie within the
     configured tolerance goes to the logistic model);
  3. whole-match holdout for the carrying model;
  4. season x phase calibration-bias table and the powerplay reliability curve per
     season;
  5. the powerplay contrast (xW change, actual change, their gap) on the shared
     draws, with equivalence framing of the xW change against the SESOI;
  6. the same contrast from the full pipeline refit on the first n shared draws,
     and the fixed-versus-refit interval comparison;
  7. feature-mix decomposition from the final model fit on all matches;
  8. the shot-level residual rung.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.equivalence import equivalence_row
from mean_level_and_tail import season_contrast_bootstrap as scb
from . import xw_model as xm
from .xw_features import FEATURE_VARIANTS, assert_whitelist_only, build_xw_features, leakage_tripwire

MODELS = ("hgb", "logistic")


def _shared_counts(groups, meta, earlier, later, n_iter, seed):
    per = xm._season_match_stats(np.zeros(len(groups)), np.zeros(len(groups)), meta, groups)
    return scb.shared_resample_counts(len(per[earlier]), len(per[later]), n_iter, seed)


def run_xw_analysis(frame: pd.DataFrame, settings, seed: int, n_iter: int | None = None,
                    n_refits: int | None = None, refit_n_jobs: int | None = None) -> dict:
    cfg = settings.section("expected_wickets")
    earlier, later = settings.earlier_season, settings.later_season
    n_iter = n_iter or settings.n_resamples
    n_refits = int(cfg["model_refit_resamples"]) if n_refits is None else n_refits
    refit_n_jobs = int(cfg.get("refit_n_jobs", 1)) if refit_n_jobs is None else refit_n_jobs
    alpha = 1.0 - settings.interval_level
    sesoi = float(settings.section("sesoi")["expected_wickets_per_ball"])
    tie_tol = float(cfg["baseline_tie_tolerance_brier"])
    bootstrap_cfg = settings.section("bootstrap")

    data = build_xw_features(frame)
    counts = _shared_counts(data.groups, data.meta, earlier, later, n_iter, seed)

    out = {"variants": {}, "n_balls": int(len(data.y)), "base_rate": float(data.y.mean())}
    score_rows, contrast_rows, verdict_rows, comparison_rows = [], [], [], []
    for variant, features in FEATURE_VARIANTS.items():
        X = data.X[features]
        assert_whitelist_only(X, features)
        tripwire = leakage_tripwire(X, data.y, data.groups, seed=seed)

        xw_by_model, scores = {}, {}
        for kind in MODELS:
            xw, methods = xm.oof_xw(X, data.y, data.groups, kind=kind, seed=seed)
            xw_by_model[kind] = xw
            scores[kind] = xm.score_xw(data.y, xw)
            scores[kind]["calibration_methods"] = str(methods)
            score_rows.append({"variant": variant, "model": kind, **scores[kind]})
        brier = {k: scores[k]["brier"] for k in MODELS}
        if abs(brier["hgb"] - brier["logistic"]) <= tie_tol:
            chosen = "logistic"
        else:
            chosen = min(brier, key=brier.get)
        robustness = "hgb" if chosen == "logistic" else "logistic"

        holdout, holdout_curve = xm.holdout_evaluation(X, data.y, data.groups, kind=chosen, seed=seed)
        bias = xm.season_phase_bias(xw_by_model[chosen], data.y, data.meta)
        pp_reliability = xm.powerplay_reliability_by_season(xw_by_model[chosen], data.y, data.meta)

        contrasts = {}
        for kind in MODELS:
            c = xm.powerplay_xw_contrast(xw_by_model[kind], data.y, data.meta, data.groups, earlier, later, counts, alpha)
            contrasts[kind] = c
            for name in ("xw_change", "actual_change", "gap_xw_minus_actual"):
                contrast_rows.append({"variant": variant, "model": kind, "role": "carrying" if kind == chosen else "robustness",
                                      "quantity": name, **{k: v for k, v in c[name].items()}})
        c = contrasts[chosen]
        verdict = equivalence_row("expected_powerplay_wicket_rate_change", c["xw_change"]["point"], 0.0,
                                  (c["xw_change"]["lo"], c["xw_change"]["hi"]),
                                  (c["xw_change"]["lo90"], c["xw_change"]["hi90"]),
                                  c["xw_change"]["se"], sesoi, bootstrap_cfg["mde_se_multiple"],
                                  unit="wickets per ball", interval_method="percentile")
        verdict.update({"variant": variant, "model": chosen})
        verdict_rows.append(verdict)

        refit = None
        if n_refits > 0:
            refit = xm.model_refit_contrast(X, data.y, data.groups, data.meta, chosen, earlier, later, counts,
                                            n_refits, seed, n_jobs=refit_n_jobs)
            comparison = xm.interval_comparison(c, refit, n_refits)
            comparison.insert(0, "variant", variant)
            comparison.insert(1, "model", chosen)
            comparison_rows.append(comparison)

        phases = xm.phase_index(X["over_num"])
        base, cal = xm.fit_calibrated_model(X, data.y, data.groups, phases, chosen, 4, seed)
        decomposition = xm.feature_mix_decomposition(base, cal, X, data.meta, earlier, later,
                                                     int(cfg["decomposition_orders"]), seed)
        decomposition.insert(0, "variant", variant)

        rung = xm.shot_level_rung(xw_by_model[chosen], data.y, data.meta, data.groups, earlier, later, counts)
        rung["coefficients"].insert(0, "variant", variant)
        rung["cell_means"].insert(0, "variant", variant)

        out["variants"][variant] = {
            "features": features, "tripwire": tripwire, "chosen_model": chosen, "robustness_model": robustness,
            "xw": xw_by_model, "holdout": holdout, "holdout_curve": holdout_curve, "bias": bias,
            "powerplay_reliability": pp_reliability, "contrasts": contrasts, "refit": refit,
            "decomposition": decomposition, "shot_rung": rung,
        }

    out["scores"] = pd.DataFrame(score_rows)
    out["contrasts"] = pd.DataFrame(contrast_rows)
    out["verdicts"] = pd.DataFrame(verdict_rows)
    out["interval_comparison"] = pd.concat(comparison_rows, ignore_index=True) if comparison_rows else pd.DataFrame()
    out["data"] = data
    out["counts"] = counts
    return out
