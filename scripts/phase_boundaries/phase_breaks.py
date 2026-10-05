"""
Phase-boundary detection: slope-change breaks, per channel, with a pre-stated move
criterion.

The question is "did a phase boundary move between seasons", never "how many
phases": a phase count on 20 points is a cluster-count decision with no reliable
diagnostic. Design decisions, each stated once here:

  Population. First innings with the full 120-ball allowance. A chase ends when the
  target is reached, so second-innings death overs are censored by the opposition's
  total; pooling them into an over profile mixes that censoring into the death
  shape. Second innings are reported as a separate descriptive profile only.

  Channels, by design not by sensitivity. The batting matrix (run rate, boundary
  rate, dot rate, wicket rate) carries the test, pooled-robust scaled with fixed
  weights 1, 1, 1, 0.5. Pace share is a bowling-deployment choice and is analysed
  alone as a single channel. Aggressive-shot share is tagged and stays on the
  descriptive map.

  Detector. The death profile is a ramp, not a step, so a constant-mean cost reads
  a slope feature as a boundary that moves with the penalty. The cost here is
  piecewise linear: within a segment each channel is fitted as intercept plus slope
  in the over index, and a break is a change in that trend. PELT with contiguity
  (segments are runs of adjacent overs), minimum segment three overs. An RBF cost
  is the robustness check.

  Penalty. Chosen once on the pooled full-data batting matrix by scanning a grid
  and taking the log-midpoint of the widest contiguous run of penalties that yield
  two or three breaks in both seasons; then held fixed across seasons and
  resamples. Per-channel detection uses the same penalty per unit of weighted cost
  (pooled penalty divided by the sum of channel weights), so a single channel is
  not held to a different standard than the matrix. The value is written out.

  Primary result. Per-channel breaks (each scaled channel separately). The pooled
  batting-matrix break is descriptive.

  Stability and the move rule. Match cluster bootstrap per season, scaling refit
  inside each resample, every break location recorded. A named boundary (the
  powerplay-to-middle edge near over 7, the middle-to-death edge near over 16) is
  anchored at the full-data break nearest the grid edge within three overs; in each
  resample the break nearest that anchor within three overs is taken, missing if
  none. On paired resamples (same draw index in both seasons) the boundary has
  MOVED if P(|loc_later - loc_earlier| >= 1 over) >= 0.90; otherwise it is "not
  contradicted", never "validated".

  Resolution. The synthetic fixture plants a one-over death-boundary move; the
  rule must fire on it. If it does not, the detector's resolution is stated and the
  grid claim is downgraded to "no move of at least k overs detected".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import ruptures as rpt
from ruptures.base import BaseCost
from sklearn.decomposition import PCA

from common import derived_columns as dc
from mean_level_and_tail import season_contrast_bootstrap as scb
from .over_profiles import (BATTING_FEATURES, DEPLOYMENT_FEATURES, FIXED_BATTING_WEIGHTS,
                            FixedWeightTransform, PooledRobustScaler, per_season_matrix)

MIN_SEGMENT_OVERS = 3
GRID_EDGES = {"powerplay_to_middle": 7, "middle_to_death": 16}
ANCHOR_WINDOW_OVERS = 3
MOVE_THRESHOLD_OVERS = 1.0
MOVE_PROBABILITY = 0.90
PENALTY_GRID = np.geomspace(0.02, 200.0, 160)
TARGET_BREAK_COUNTS = (2, 3)


# ---------------------------------------------------------------------------
# Sufficient statistics: collapse deliveries to per-(season, match, over) totals
# so a bootstrap resample is a weighted sum, never a frame expansion.
# ---------------------------------------------------------------------------

_STAT_COLUMNS = [
    "n_legal", "n_faced", "sum_score_legal", "sumsq_score_legal",
    "boundary_faced", "dot_faced", "wicket_faced", "pace_legal", "aggressive_faced",
]


def precompute_match_over_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (year, p_match, over_num) sufficient statistics for every profile feature."""
    work = frame.copy()
    legal = dc.is_legal_delivery(work).to_numpy()
    faced = dc.is_ball_faced(work).to_numpy()
    score = pd.to_numeric(work["score"], errors="coerce").fillna(0).to_numpy(dtype=float)

    work["n_legal"] = legal.astype(float)
    work["n_faced"] = faced.astype(float)
    work["sum_score_legal"] = np.where(legal, score, 0.0)
    work["sumsq_score_legal"] = np.where(legal, score ** 2, 0.0)
    work["boundary_faced"] = ((dc.is_boundary(work).to_numpy()) & faced).astype(float)
    work["dot_faced"] = dc.is_dot_off_bat(work).to_numpy().astype(float)
    work["wicket_faced"] = ((dc.is_wicket(work).to_numpy()) & faced).astype(float)
    work["pace_legal"] = ((dc.is_pace(work).to_numpy()) & legal).astype(float)
    work["aggressive_faced"] = ((dc.is_aggressive_shot(work).to_numpy()) & faced).astype(float)

    grouped = work.groupby(["year", "p_match", "over_num"])[_STAT_COLUMNS].sum().reset_index()
    return grouped


def _profiles_from_stats(cell_stats: pd.DataFrame) -> pd.DataFrame:
    """Turn per-(year, over) summed sufficient statistics into a profile table."""
    g = cell_stats
    n_legal = g["n_legal"].to_numpy()
    n_faced = g["n_faced"].to_numpy()
    mean_score = np.where(n_legal > 0, g["sum_score_legal"] / n_legal, np.nan)
    run_ball_var = np.where(
        n_legal > 0, g["sumsq_score_legal"] / n_legal - mean_score ** 2, 0.0
    )  # population var from the summed squares; ddof negligible at ~800 balls/cell
    out = pd.DataFrame({
        "year": g["year"].astype(int),
        "over_num": g["over_num"].astype(int),
        "n_legal": n_legal.astype(int),
        "n_faced": n_faced.astype(int),
        "run_rate": 6.0 * mean_score,
        "boundary_rate": np.where(n_faced > 0, g["boundary_faced"] / n_faced, np.nan),
        "dot_rate": np.where(n_faced > 0, g["dot_faced"] / n_faced, np.nan),
        "wicket_rate": np.where(n_faced > 0, g["wicket_faced"] / n_faced, np.nan),
        "pace_share": np.where(n_legal > 0, g["pace_legal"] / n_legal, np.nan),
        "aggressive_share": np.where(n_faced > 0, g["aggressive_faced"] / n_faced, np.nan),
        "_run_ball_var": np.clip(run_ball_var, 0.0, None),
    })
    return out.sort_values(["year", "over_num"]).reset_index(drop=True)


def profiles_full(stats: pd.DataFrame) -> pd.DataFrame:
    """Full-data profiles: every match at multiplicity one."""
    cell_stats = stats.groupby(["year", "over_num"])[_STAT_COLUMNS].sum().reset_index()
    return _profiles_from_stats(cell_stats)


# ---------------------------------------------------------------------------
# Piecewise-linear cost and the detectors
# ---------------------------------------------------------------------------

class PiecewiseLinearCost(BaseCost):
    """Segment cost = sum over channels of the residual sum of squares of a
    least-squares fit of the channel on (intercept, over index) within the segment.
    A ramp is one segment; a break is a change of slope or level. Prefix sums make
    each segment cost O(1) so PELT stays cheap inside the bootstrap."""

    model = "piecewise_linear"
    min_size = MIN_SEGMENT_OVERS

    def fit(self, signal):
        signal = np.asarray(signal, dtype=float)
        if signal.ndim == 1:
            signal = signal[:, None]
        n = signal.shape[0]
        t = np.arange(n, dtype=float)
        self.signal = signal
        z = np.zeros((1, signal.shape[1]))
        self._sy = np.vstack([z, np.cumsum(signal, axis=0)])
        self._sty = np.vstack([z, np.cumsum(signal * t[:, None], axis=0)])
        self._syy = np.vstack([z, np.cumsum(signal ** 2, axis=0)])
        self._st = np.concatenate([[0.0], np.cumsum(t)])
        self._stt = np.concatenate([[0.0], np.cumsum(t ** 2)])
        return self

    def error(self, start, end):
        m = end - start
        st = self._st[end] - self._st[start]
        stt = self._stt[end] - self._stt[start]
        sy = self._sy[end] - self._sy[start]
        sty = self._sty[end] - self._sty[start]
        syy = self._syy[end] - self._syy[start]
        det = m * stt - st * st
        if det <= 1e-12:
            return float(np.sum(syy - sy ** 2 / m))
        # Normal equations for (intercept, slope); RSS = syy - beta . X'y.
        b0 = (stt * sy - st * sty) / det
        b1 = (m * sty - st * sy) / det
        rss = syy - (b0 * sy + b1 * sty)
        return float(np.sum(np.clip(rss, 0.0, None)))


def detect_break_starts(matrix: np.ndarray, pen: float, cost: str = "piecewise_linear") -> list:
    """Over numbers (1-indexed) at which a new segment starts along the over
    sequence, under PELT with the given penalty. `cost` is "piecewise_linear" (the
    detector) or "rbf" (the robustness check)."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if cost == "piecewise_linear":
        algo = rpt.Pelt(custom_cost=PiecewiseLinearCost(), min_size=MIN_SEGMENT_OVERS, jump=1)
    elif cost == "rbf":
        algo = rpt.Pelt(model="rbf", min_size=MIN_SEGMENT_OVERS, jump=1)
    else:
        raise ValueError(cost)
    endpoints = algo.fit(matrix).predict(pen=pen)
    n = matrix.shape[0]
    return [e + 1 for e in endpoints if e < n]


def choose_penalty(season_matrices: dict, cost: str = "piecewise_linear",
                   grid: np.ndarray = PENALTY_GRID, targets=TARGET_BREAK_COUNTS) -> dict:
    """Scan the penalty grid on the full-data matrices; a penalty qualifies if every
    season yields a break count in `targets`. Returns the log-midpoint of the widest
    contiguous qualifying run, with the scan table."""
    rows = []
    for pen in grid:
        counts = {year: len(detect_break_starts(m, pen, cost)) for year, m in season_matrices.items()}
        rows.append({"penalty": float(pen), **{f"breaks_{y}": c for y, c in counts.items()},
                     "qualifies": all(c in targets for c in counts.values())})
    scan = pd.DataFrame(rows)
    best, run_start, best_len = None, None, 0
    q = scan["qualifies"].to_numpy()
    for i in range(len(q) + 1):
        if i < len(q) and q[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start > best_len:
                best_len, best = i - run_start, (run_start, i - 1)
            run_start = None
    if best is None:
        return {"penalty": np.nan, "scan": scan, "plateau": None,
                "note": "no penalty gives the target break count in every season"}
    lo, hi = scan.loc[best[0], "penalty"], scan.loc[best[1], "penalty"]
    return {"penalty": float(np.exp(0.5 * (np.log(lo) + np.log(hi)))), "scan": scan,
            "plateau": (float(lo), float(hi)), "plateau_grid_points": int(best_len)}


# ---------------------------------------------------------------------------
# Feature matrices per season
# ---------------------------------------------------------------------------

def batting_matrices(profiles: pd.DataFrame) -> tuple[dict, FixedWeightTransform]:
    """Per-season weighted batting matrices (20 x 4) and the fitted transform."""
    transform = FixedWeightTransform(BATTING_FEATURES, FIXED_BATTING_WEIGHTS).fit(profiles)
    fm = transform.transform(profiles)
    return {int(y): per_season_matrix(fm, y, BATTING_FEATURES) for y in sorted(fm["year"].unique())}, transform


def channel_series(profiles: pd.DataFrame, features: list) -> dict:
    """Per-season, per-channel pooled-robust-scaled series (unweighted, IQR = 1), so
    per-channel breaks are comparable across channels."""
    scaler = PooledRobustScaler(features=features).fit(profiles)
    scaled = scaler.transform(profiles)
    scaled["year"] = profiles["year"].to_numpy()
    scaled["over_num"] = profiles["over_num"].to_numpy()
    out = {}
    for year in sorted(scaled["year"].unique()):
        rows = scaled[scaled["year"] == year].sort_values("over_num")
        out[int(year)] = {f: rows[f].to_numpy(dtype=float) for f in features}
    return out


def channel_penalty(pooled_penalty: float) -> float:
    """Per-channel penalty on the same scale per unit of weighted cost as the pooled
    batting penalty: pooled penalty over the sum of the fixed weights."""
    return pooled_penalty / sum(FIXED_BATTING_WEIGHTS.values())


# ---------------------------------------------------------------------------
# Full-data breaks and boundary anchors
# ---------------------------------------------------------------------------

def full_data_breaks(profiles: pd.DataFrame, pooled_penalty: float, cost: str = "piecewise_linear",
                     rbf_pooled_penalty: float | None = None) -> pd.DataFrame:
    """Break-start overs per season for every channel (batting channels and pace
    share, primary) and the pooled batting matrix (descriptive)."""
    pen = pooled_penalty if cost == "piecewise_linear" else rbf_pooled_penalty
    per_ch = channel_penalty(pen)
    series = channel_series(profiles, BATTING_FEATURES + DEPLOYMENT_FEATURES)
    matrices, _ = batting_matrices(profiles)
    rows = []
    for year in sorted(series):
        for f, s in series[year].items():
            rows.append({"year": year, "channel": f, "kind": "per_channel",
                         "break_starts": detect_break_starts(s, per_ch, cost)})
        rows.append({"year": year, "channel": "batting_matrix", "kind": "pooled",
                     "break_starts": detect_break_starts(matrices[year], pen, cost)})
    return pd.DataFrame(rows)


def _nearest_within(breaks: list, anchor: float, window: float) -> float:
    if not breaks or not np.isfinite(anchor):
        return np.nan
    arr = np.asarray(breaks, dtype=float)
    d = np.abs(arr - anchor)
    j = int(np.argmin(d))
    return float(arr[j]) if d[j] <= window else np.nan


def boundary_anchors(full_breaks: pd.DataFrame) -> pd.DataFrame:
    """For each (channel, season, named boundary): the full-data break nearest the
    grid edge within the anchor window. NaN means the channel shows no break near
    that edge on the full data."""
    rows = []
    for _, r in full_breaks.iterrows():
        for name, edge in GRID_EDGES.items():
            rows.append({"year": r["year"], "channel": r["channel"], "kind": r["kind"], "boundary": name,
                         "grid_edge": edge,
                         "anchor_over": _nearest_within(r["break_starts"], edge, ANCHOR_WINDOW_OVERS)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Match-cluster bootstrap: break locations per resample, paired move test
# ---------------------------------------------------------------------------

@dataclass
class SeasonStats:
    year: int
    match_ids: np.ndarray
    match_index: np.ndarray       # per row of stats, the match slot
    over_num: np.ndarray
    values: np.ndarray            # rows x _STAT_COLUMNS


def _season_stats(stats: pd.DataFrame) -> dict:
    out = {}
    for year, s in stats.groupby("year"):
        matches = s["p_match"].drop_duplicates().to_numpy()
        idx = {m: i for i, m in enumerate(matches)}
        out[int(year)] = SeasonStats(int(year), matches, s["p_match"].map(idx).to_numpy(),
                                     s["over_num"].to_numpy(), s[_STAT_COLUMNS].to_numpy(dtype=float))
    return out


def _resampled_profiles(season_stats: dict, mult_by_year: dict) -> pd.DataFrame:
    cells = []
    for year, ss in season_stats.items():
        weighted = ss.values * mult_by_year[year][ss.match_index][:, None]
        tmp = pd.DataFrame(weighted, columns=_STAT_COLUMNS)
        tmp["year"] = year
        tmp["over_num"] = ss.over_num
        cells.append(tmp.groupby(["year", "over_num"])[_STAT_COLUMNS].sum().reset_index())
    return _profiles_from_stats(pd.concat(cells, ignore_index=True))


def break_stability(stats: pd.DataFrame, pooled_penalty: float, n_iter: int, seed: int,
                    cost: str = "piecewise_linear", include_pooled: bool = True) -> dict:
    """Relocate every break under match resampling and return the raw break lists.

    For each resample: draw a match-multiplicity vector per season (shared draw
    index across seasons, so the seasons pair), rebuild the per-over profiles from
    the sufficient statistics, refit the pooled scaling, redetect the breaks per
    channel (and on the pooled batting matrix). Returns {year: {channel: [list of
    break lists]}} plus the draws' seed for provenance.
    """
    season_stats = _season_stats(stats)
    years = sorted(season_stats)
    counts = dict(zip(years, scb.shared_resample_counts(len(season_stats[years[0]].match_ids),
                                                        len(season_stats[years[1]].match_ids),
                                                        n_iter, seed)))
    per_ch = channel_penalty(pooled_penalty)
    channels = BATTING_FEATURES + DEPLOYMENT_FEATURES + (["batting_matrix"] if include_pooled else [])
    breaks = {y: {c: [] for c in channels} for y in years}
    for it in range(n_iter):
        profiles = _resampled_profiles(season_stats, {y: counts[y][it] for y in years})
        series = channel_series(profiles, BATTING_FEATURES + DEPLOYMENT_FEATURES)
        matrices = batting_matrices(profiles)[0] if include_pooled else None
        for y in years:
            for f, s in series[y].items():
                breaks[y][f].append(detect_break_starts(s, per_ch, cost))
            if include_pooled:
                breaks[y]["batting_matrix"].append(detect_break_starts(matrices[y], pooled_penalty, cost))
    return {"breaks": breaks, "years": years, "n_iter": n_iter, "seed": seed, "cost": cost}


def move_test(stability: dict, anchors: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Paired-resample move test per (channel, boundary).

    In each resample the break nearest the channel-season anchor within the window
    is taken (missing if none). Reports per season the location distribution and
    the missing share; then on paired resamples where both seasons have a location,
    P(|later - earlier| >= 1 over), the mean and the distribution of the paired
    difference, and the verdict: "moved" if that probability is at least 0.90,
    "not_contradicted" otherwise, "not_anchored" if a season has no full-data break
    near the edge, "underpowered" if fewer than half the resamples pair up.
    """
    years = stability["years"]
    earlier, later = years[0], years[-1]
    rows, histograms = [], {}
    for (channel, boundary), grp in anchors.groupby(["channel", "boundary"], sort=False):
        anchor = {int(r["year"]): r["anchor_over"] for _, r in grp.iterrows()}
        if channel not in stability["breaks"][earlier]:
            continue
        locs = {}
        for y in years:
            found = [_nearest_within(b, anchor.get(y, np.nan), ANCHOR_WINDOW_OVERS)
                     for b in stability["breaks"][y][channel]]
            locs[y] = np.asarray(found, dtype=float)
        paired = np.isfinite(locs[earlier]) & np.isfinite(locs[later])
        diff = locs[later][paired] - locs[earlier][paired]
        row = {"channel": channel, "boundary": boundary, "kind": grp["kind"].iloc[0],
               "grid_edge": int(grp["grid_edge"].iloc[0]),
               f"anchor_{earlier}": anchor.get(earlier, np.nan), f"anchor_{later}": anchor.get(later, np.nan)}
        for y in years:
            finite = locs[y][np.isfinite(locs[y])]
            vals, cnts = np.unique(finite, return_counts=True)
            row[f"modal_over_{y}"] = int(vals[np.argmax(cnts)]) if len(vals) else np.nan
            row[f"missing_share_{y}"] = float(np.mean(~np.isfinite(locs[y])))
            histograms[(channel, boundary, y)] = dict(zip(vals.astype(int).tolist(), cnts.tolist()))
        row["paired_share"] = float(paired.mean())
        row["mean_paired_difference"] = float(diff.mean()) if diff.size else np.nan
        row["p_abs_difference_ge_1"] = float(np.mean(np.abs(diff) >= MOVE_THRESHOLD_OVERS)) if diff.size else np.nan
        row["p_later_earlier_by_ge_1"] = float(np.mean(diff <= -MOVE_THRESHOLD_OVERS)) if diff.size else np.nan
        row["p_later_later_by_ge_1"] = float(np.mean(diff >= MOVE_THRESHOLD_OVERS)) if diff.size else np.nan
        dvals, dcnts = np.unique(diff, return_counts=True)
        histograms[(channel, boundary, "difference")] = dict(zip(dvals.astype(int).tolist(), dcnts.tolist()))
        if not all(np.isfinite(anchor.get(y, np.nan)) for y in years):
            verdict = "not_anchored"
        elif row["paired_share"] < 0.5:
            verdict = "underpowered"
        elif row["p_abs_difference_ge_1"] >= MOVE_PROBABILITY:
            verdict = "moved"
        else:
            verdict = "not_contradicted"
        row["verdict"] = verdict
        rows.append(row)
    return pd.DataFrame(rows), histograms


def histograms_table(histograms: dict) -> pd.DataFrame:
    rows = []
    for (channel, boundary, which), dist in histograms.items():
        for value, count in dist.items():
            rows.append({"channel": channel, "boundary": boundary, "series": str(which),
                         "value": int(value), "count": int(count)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Population, driver, resolution check
# ---------------------------------------------------------------------------

def first_innings_full_allowance(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ball rows of first innings with a full allowance, and the population table."""
    from common import innings_population as ip
    table = ip.build_innings_table(frame, full_innings_balls=settings.section("tail")["full_innings_balls"],
                                   death_first_over=settings.phase("death")[0])
    kept = ip.restrict_frame_to_innings(frame, ip.uncensored_first_innings(table))
    return kept, ip.population_counts(table)


def run_phase_boundaries(frame: pd.DataFrame, settings, seed: int, n_iter: int | None = None,
                         robustness_iter: int | None = None) -> dict:
    """The full analysis on the first-innings population: penalty choice, full-data
    breaks per channel and pooled, anchors, bootstrap stability, the move test, the
    RBF robustness rerun, and the descriptive pieces (second innings profiles, PCA
    map of the batting matrix)."""
    cfg = settings.section("phase_boundaries")
    n_iter = n_iter or int(cfg["n_resamples"])
    robustness_iter = robustness_iter or int(cfg["robustness_resamples"])

    kept, population = first_innings_full_allowance(frame, settings)
    stats = precompute_match_over_stats(kept)
    profiles = profiles_full(stats)
    matrices, transform = batting_matrices(profiles)

    penalty = choose_penalty(matrices, "piecewise_linear")
    rbf_penalty = choose_penalty(matrices, "rbf")

    breaks = full_data_breaks(profiles, penalty["penalty"])
    anchors = boundary_anchors(breaks)
    stability = break_stability(stats, penalty["penalty"], n_iter, seed)
    move, hist = move_test(stability, anchors)

    rbf_breaks = full_data_breaks(profiles, np.nan, "rbf", rbf_penalty["penalty"])
    rbf_anchors = boundary_anchors(rbf_breaks)
    rbf_stability = break_stability(stats, rbf_penalty["penalty"], robustness_iter, seed, cost="rbf")
    rbf_move, _ = move_test(rbf_stability, rbf_anchors)

    second = frame[frame["inns"].astype(int) == 2]
    second_profiles = profiles_full(precompute_match_over_stats(second))

    return {
        "population": population, "profiles": profiles, "second_innings_profiles": second_profiles,
        "penalty": penalty, "rbf_penalty": rbf_penalty, "transform": transform,
        "full_breaks": breaks, "anchors": anchors, "move": move, "histograms": histograms_table(hist),
        "rbf_full_breaks": rbf_breaks, "rbf_move": rbf_move,
        "trajectory": season_trajectories(transform.transform(profiles), BATTING_FEATURES),
        "scree": pca_scree(transform.transform(profiles), BATTING_FEATURES),
    }


def resolution_check(settings, seed: int, n_iter: int, death_edge_by_season: dict | None = None,
                     n_matches_per_season: int = 70) -> pd.DataFrame:
    """Run the same pipeline on the synthetic fixture with a planted death-boundary
    move (default one over: 16 in the earlier season, 15 in the later) and return
    the move-test table for the middle-to-death boundary. The rule must fire on at
    least one batting channel for a one-over move to count as resolvable."""
    from common.synthetic_ballframe import make_synthetic_ballframe
    fixture = make_synthetic_ballframe(n_matches_per_season=n_matches_per_season,
                                       death_edge_by_season=death_edge_by_season, seed=7)
    kept, _ = first_innings_full_allowance(fixture, settings)
    stats = precompute_match_over_stats(kept)
    profiles = profiles_full(stats)
    matrices, _ = batting_matrices(profiles)
    penalty = choose_penalty(matrices, "piecewise_linear")
    breaks = full_data_breaks(profiles, penalty["penalty"])
    stability = break_stability(stats, penalty["penalty"], n_iter, seed, include_pooled=True)
    move, _ = move_test(stability, boundary_anchors(breaks))
    move["penalty"] = penalty["penalty"]
    move["planted_edges"] = str(death_edge_by_season or {settings.earlier_season: 16, settings.later_season: 15})
    return move[move["boundary"] == "middle_to_death"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Descriptive map and figures
# ---------------------------------------------------------------------------

def season_trajectories(feature_matrix: pd.DataFrame, features) -> dict:
    """PCA (2 components) of the weighted 40-row matrix; per-season 20-point lines."""
    matrix = feature_matrix[features].to_numpy(dtype=float)
    pca = PCA(n_components=2, random_state=0).fit(matrix)
    coords = pca.transform(matrix)
    result = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
    fm = feature_matrix.reset_index(drop=True)
    for year in sorted(fm["year"].unique()):
        mask = (fm["year"] == year).to_numpy()
        order = np.argsort(fm.loc[mask, "over_num"].to_numpy())
        result[int(year)] = coords[mask][order]
    return result


def pca_scree(feature_matrix: pd.DataFrame, features) -> np.ndarray:
    """Explained-variance ratio of every component of the weighted batting matrix:
    how faithful the 2-D map is, not how many components to keep."""
    X = feature_matrix[features].to_numpy(dtype=float)
    return PCA(n_components=min(X.shape)).fit(X).explained_variance_ratio_


def plot_trajectories(traj: dict, title: str | None = None):
    """Both seasons' 20-over paths through the batting PCA plane, every over
    annotated. Descriptive only; no test reads this figure."""
    import matplotlib.pyplot as plt

    season_colors = {2023: "#2c7fb8", 2024: "#d95f0e"}
    years = [y for y in traj if isinstance(y, int)]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for year in years:
        coords = traj[year]
        color = season_colors.get(year)
        ax.plot(coords[:, 0], coords[:, 1], "-", color=color, alpha=0.6, linewidth=1.6, zorder=1)
        ax.scatter(coords[:, 0], coords[:, 1], s=26, color=color, zorder=2, label=str(year))
        for i in range(coords.shape[0]):
            ax.annotate(str(i + 1), (coords[i, 0], coords[i, 1]), fontsize=7, color=color,
                        xytext=(3, 3), textcoords="offset points")
    evr = traj.get("explained_variance_ratio", [np.nan, np.nan])
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%} of variance)")
    ax.set_title(title or "First-innings over profiles in the batting PCA plane, by season", loc="left",
                 fontsize=10, fontweight="bold")
    ax.legend(title="season", frameon=False)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def plot_channel_profiles(profiles: pd.DataFrame, features, title: str):
    """Raw per-over channel values, one panel per channel, both seasons."""
    import matplotlib.pyplot as plt

    season_colors = {2023: "#2c7fb8", 2024: "#d95f0e"}
    fig, axes = plt.subplots(1, len(features), figsize=(3.2 * len(features), 3.0))
    for ax, f in zip(np.atleast_1d(axes), features):
        for year, rows in profiles.groupby("year"):
            rows = rows.sort_values("over_num")
            ax.plot(rows["over_num"], rows[f], "-o", ms=3, color=season_colors.get(int(year)), label=str(year))
        ax.set_title(f, fontsize=9)
        ax.set_xlabel("over")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_break_histograms(histograms: pd.DataFrame, channel: str, boundary: str, years):
    """Break-location histograms per season and the paired-difference histogram for
    one channel and boundary."""
    import matplotlib.pyplot as plt

    sub = histograms[(histograms["channel"] == channel) & (histograms["boundary"] == boundary)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, series in zip(axes, [str(years[0]), str(years[1]), "difference"]):
        rows = sub[sub["series"] == series].sort_values("value")
        ax.bar(rows["value"], rows["count"], color="#2F5C8A" if series != "difference" else "#d95f0e")
        ax.set_title(f"{series}" if series == "difference" else f"break location {series}", fontsize=9)
        ax.set_xlabel("over" if series != "difference" else "later minus earlier (overs)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(f"{channel}: {boundary}", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig
