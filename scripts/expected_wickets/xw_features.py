"""
Situation-only xW: pre-ball feature builder and leakage audit.

xW answers one descriptive question, not "predict the next wicket for its own sake":
did the EXPECTED powerplay wicket rate rise 2023->2024 while the ACTUAL rate stayed
flat? For that the model must be (a) built only from information available BEFORE the
ball, and (b) honestly calibrated. Leakage is the whole game -- a feature that encodes
the ball's own result yields the 93-97% cricket-ML trap and silently turns xW into a
re-statement of the outcome. So this module fixes the whitelist and audits it before any
model is fit.

Modelling unit: one row per LEGAL delivery (wides and no-balls excluded), matching a
per-ball wicket rate. Target y = a wicket fell on this delivery (`out`).

Strictly-pre-ball construction: the situation features are rebuilt inside each innings as
cumulative-sum-MINUS-the-current-ball, so a delivery's own outcome can never enter its
own features. The provided cumulative columns (inns_runs, inns_runs_rem, ...) are NOT
used, because several are documented as "after this delivery" and would leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import derived_columns as dc


def _sklearn_seed(seed: int) -> int:
    """scikit-learn accepts random_state up to 2**32 - 1; the project seed is a
    larger timestamp-shaped integer, so it is folded into that range here."""
    return int(seed) % (2 ** 32 - 1)

# The only columns the model is allowed to see. Every one is known before the ball.
# Two feature variants are run: "situation" (game state and batter hand) and
# "situation_bowler" (the same plus the pace/spin indicators). Bowler type is a
# deployment choice inside the situation, so the two are reported side by side: if
# the season contrast agrees across them the claim stands; if it does not, pace
# deployment is the driver.
WHITELIST = [
    "over_num",         # phase is learnable from this by a tree; kept raw and minimal
    "inns_wkts",        # wickets already down (pre-ball)
    "inns_balls_rem",   # legal balls left in the innings (pre-ball; from DLS-aware max_balls)
    "is_chase",         # 1 in the second innings, else 0
    "required_rate",    # (target - runs_before) / balls_rem * 6; NaN in the first innings
    "bat_is_lhb",       # striker left-handed (clean)
    "bowl_is_pace",     # pace delivery (clean bowl_kind)
    "bowl_is_spin",     # spin delivery (clean bowl_kind); mixture/unknown -> both 0
]

SITUATION_FEATURES = ["over_num", "inns_wkts", "inns_balls_rem", "is_chase", "required_rate", "bat_is_lhb"]
FEATURE_VARIANTS = {
    "situation": SITUATION_FEATURES,
    "situation_bowler": SITUATION_FEATURES + ["bowl_is_pace", "bowl_is_spin"],
}

# Columns that describe or sum the ball's RESULT. Any of these in the feature matrix is
# leakage; the audit refuses to proceed if one appears.
LEAKAGE_COLUMNS = [
    "score", "batruns", "outcome", "out", "dismissal", "bat_out",
    "wprob", "predscore", "bowler_charge",
    "control", "control_clean", "shot", "line", "length",
    "wide", "noball", "byes", "legbyes",
    # provided cumulatives that include the current ball or are stated after it:
    "inns_runs", "inns_runs_rem", "inns_rr", "inns_rrr",
    "cur_bat_runs", "cur_bat_bf", "cur_bowl_runs", "cur_bowl_wkts", "ball_id",
]


@dataclass
class XwData:
    """The model's view (X, y, groups) kept strictly apart from the descriptive keys
    (meta) used for the season contrast, so contrast columns can never become features."""
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series          # p_match -- the cluster unit for grouped CV and bootstrap
    meta: pd.DataFrame         # year, over_num, phase, inns -- for the contrast, NOT features


def _phase(over_num: pd.Series) -> pd.Series:
    return pd.cut(over_num, bins=[0, 6, 15, 20], labels=["powerplay", "middle", "death"])


def build_xw_features(frame: pd.DataFrame) -> XwData:
    """Build the strictly-pre-ball situation matrix on legal deliveries."""
    work = frame.copy()
    work["_legal"] = dc.is_legal_delivery(work).to_numpy()
    work["_out"] = dc.is_wicket(work).astype(int).to_numpy()
    work["_score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0).to_numpy()

    # Order within each innings so the pre-ball cumulatives are correct.
    order_cols = ["p_match", "inns", "over_num", "ball", "ball_in_over"]
    order_cols = [c for c in order_cols if c in work.columns]
    work = work.sort_values(order_cols).reset_index(drop=True)

    g = work.groupby(["p_match", "inns"], sort=False)
    # Pre-ball state = cumulative-including-current MINUS current, so the current
    # delivery contributes nothing to its own features.
    runs_before = g["_score"].cumsum() - work["_score"]
    legal_before = g["_legal"].cumsum() - work["_legal"]
    wkts_before = g["_out"].cumsum() - work["_out"]

    max_balls = pd.to_numeric(work["max_balls"], errors="coerce")
    balls_rem_before = max_balls - legal_before

    is_chase = (pd.to_numeric(work["inns"], errors="coerce") == 2).astype(int)
    target = pd.to_numeric(work["target"], errors="coerce")
    # required rate defined only for a chase with balls left; NaN otherwise (first
    # innings has no target). NaN is handled natively by the gradient booster.
    with np.errstate(invalid="ignore", divide="ignore"):
        rrr = (target - runs_before) / balls_rem_before * 6.0
    required_rate = np.where((is_chase == 1) & (balls_rem_before > 0), rrr, np.nan)

    bowl_kind = work["bowl_kind"].astype("string")
    features = pd.DataFrame({
        "over_num": pd.to_numeric(work["over_num"], errors="coerce"),
        "inns_wkts": wkts_before.astype(float),
        "inns_balls_rem": balls_rem_before.astype(float),
        "is_chase": is_chase.astype(float),
        "required_rate": required_rate,
        "bat_is_lhb": work["bat_hand"].astype("string").str.upper().eq("LHB").astype(float),
        "bowl_is_pace": bowl_kind.str.contains("pace", case=False, na=False).astype(float),
        "bowl_is_spin": bowl_kind.str.contains("spin", case=False, na=False).astype(float),
    })[WHITELIST]

    meta = pd.DataFrame({
        "year": pd.to_numeric(work["year"], errors="coerce").astype(int),
        "over_num": features["over_num"].astype(int),
        "phase": _phase(features["over_num"]),
        "inns": pd.to_numeric(work["inns"], errors="coerce").astype(int),
        # Shot intent rides in meta for the shot-level residual rung only. It is a
        # post-ball tag and must never enter the feature matrix.
        "aggressive_shot": dc.is_aggressive_shot(work).astype(int).to_numpy() if "shot" in work.columns else 0,
        "faced": dc.is_ball_faced(work).astype(int).to_numpy(),
    })

    legal_mask = work["_legal"].astype(bool).to_numpy()
    return XwData(
        X=features.loc[legal_mask].reset_index(drop=True),
        y=work.loc[legal_mask, "_out"].reset_index(drop=True).rename("wicket"),
        groups=work.loc[legal_mask, "p_match"].reset_index(drop=True).rename("p_match"),
        meta=meta.loc[legal_mask].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Leakage audit -- run BEFORE any model is trusted
# ---------------------------------------------------------------------------

def assert_whitelist_only(X: pd.DataFrame, allowed: list | None = None) -> None:
    """Refuse to proceed unless X is exactly the allowed feature list (the whitelist by
    default, or one variant's list) and contains no leakage column."""
    allowed = list(allowed) if allowed is not None else WHITELIST
    cols = list(X.columns)
    extra = [c for c in cols if c not in allowed]
    leaks = [c for c in cols if c in LEAKAGE_COLUMNS]
    missing = [c for c in allowed if c not in cols]
    problems = []
    if leaks:
        problems.append(f"leakage columns present: {leaks}")
    if extra:
        problems.append(f"non-whitelist columns present: {extra}")
    if missing:
        problems.append(f"whitelist columns missing: {missing}")
    if problems:
        raise AssertionError("xW feature audit failed -- " + "; ".join(problems))


def leakage_tripwire(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                     auc_alarm: float = 0.80, seed: int = 0) -> dict:
    """Out-of-fold, match-grouped AUC as a leakage detector.

    A situation-only model at a ~4-5% wicket base rate should land at a MODEST AUC. A
    high AUC is the 93-97% trap: something is encoding the outcome. This is a detector,
    not the model -- it uses a quick fixed booster and reports the number to inspect.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                             max_iter=200, l2_regularization=1.0,
                                             random_state=_sklearn_seed(seed))
        clf.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = clf.predict_proba(X.iloc[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    return {"oof_auc": auc, "base_rate": float(y.mean()),
            "alarm": auc >= auc_alarm,
            "verdict": "SUSPICIOUS - inspect for leakage" if auc >= auc_alarm
                       else "modest AUC, consistent with situation-only"}


def legal_aligned_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a raw column restricted to legal deliveries, in the SAME row order as
    build_xw_features produces. Only for the leakage-audit demo (injecting a known-leaky
    column to confirm the tripwire fires); never used to build real features.
    """
    order = [c for c in ["p_match", "inns", "over_num", "ball", "ball_in_over"] if c in frame.columns]
    ordered = frame.sort_values(order).reset_index(drop=True)
    legal = dc.is_legal_delivery(ordered).to_numpy()
    return pd.to_numeric(ordered.loc[legal, column], errors="coerce").reset_index(drop=True)
