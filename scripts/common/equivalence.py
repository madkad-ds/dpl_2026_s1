"""
Equivalence verdicts for a null result.

A wide interval containing zero is not evidence of absence. To retire a hypothesis
the interval has to sit inside a pre-stated band of effects too small to matter,
the smallest effect size of interest (SESOI). This module turns a bootstrap result
into a three-way verdict:

  rejects_null      : the 95% interval excludes the null.
  equivalent_to_zero: the 90% interval lies entirely inside [-SESOI, +SESOI]. A 90%
                      interval is the two-one-sided-tests (TOST) procedure at
                      alpha = 0.05.
  inconclusive      : neither. Report the shape as unresolved, not retired.

Alongside the verdict it reports the bootstrap SE and the minimum detectable effect
at 80% power, two-sided 5%: MDE = 2.80 * SE. An "inconclusive" with an MDE well
above the SESOI says the test could not have found the effect it was looking for.

SESOI values live in config.toml under [sesoi] with a one-line justification each;
they are fixed before a rerun, never read off the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equivalence_row(name: str, point: float, null_value: float,
                    ci95: tuple[float, float], ci90: tuple[float, float],
                    se: float, sesoi: float, mde_se_multiple: float = 2.80,
                    unit: str = "", interval_method: str = "") -> dict:
    """One verdict from already-built intervals (both on the scale of `point`)."""
    lo95, hi95 = ci95
    lo90, hi90 = ci90
    mde = mde_se_multiple * se if np.isfinite(se) else np.nan
    if np.isfinite(lo95) and np.isfinite(hi95) and (lo95 > null_value or hi95 < null_value):
        verdict = "rejects_null"
    elif (np.isfinite(lo90) and np.isfinite(hi90)
          and lo90 >= null_value - sesoi and hi90 <= null_value + sesoi):
        verdict = "equivalent_to_zero"
    else:
        verdict = "inconclusive"
    return {
        "channel": name,
        "estimate": point,
        "ci90_low": lo90,
        "ci90_high": hi90,
        "ci95_low": lo95,
        "ci95_high": hi95,
        "se": se,
        "sesoi": sesoi,
        "mde_80pct_power": mde,
        "mde_exceeds_sesoi": bool(np.isfinite(mde) and mde > sesoi),
        "verdict": verdict,
        "unit": unit,
        "interval_method": interval_method,
    }


def equivalence_from_theme_result(result, sesoi: float, mde_se_multiple: float = 2.80,
                                  unit: str = "", equivalence_level: float = 0.90,
                                  interval_level: float = 0.95) -> dict:
    """Verdict for a harness ThemeResult (season_contrast_bootstrap.ThemeResult).

    The SE is the closed-form cluster SE when the estimand has one (rates and their
    linear contrasts), else the SD of the replicates. Intervals use the result's
    primary method. Ratio themes are judged on the log scale: SESOI is then a log
    ratio band and the intervals are compared on that scale.
    """
    on_log = result.scale == "log"
    ci95 = result.interval(interval_level)
    ci90 = result.interval(equivalence_level)
    if on_log:
        ci95 = (np.log(ci95[0]), np.log(ci95[1]))
        ci90 = (np.log(ci90[0]), np.log(ci90[1]))
        point, null = float(np.log(result.point)), 0.0
    else:
        point, null = result.point, result.null_value
    se = result.se_linearized if np.isfinite(result.se_linearized) else result.se_boot
    return equivalence_row(result.name, point, null, ci95, ci90, se, sesoi,
                           mde_se_multiple, unit, result.interval_method)


def verdict_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
