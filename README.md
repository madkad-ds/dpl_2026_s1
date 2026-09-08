# dpl_2026_s1

IPL 2023 vs 2024 tactical evolution, ball by ball. Each analysis is a folder of scripts under `scripts/` and one notebook at the root that runs it and shows the tables and figures. Methodology decisions live in the module docstrings; this file covers layout, environment and conventions.

## Layout

```
config.toml                  seeds, resample counts, interval levels, SESOI values, populations, paths
requirements.txt             pinned versions exported from uv.lock
pyproject.toml, uv.lock      the uv project definition and lockfile
data/                        input datasets (not committed): dpl_prelims_clean.parquet,
                             impact_player_table.parquet, match_rosters.parquet
outputs/<analysis>/          tables (.csv), nested results (.json), figures (.png), each with a .meta.json sidecar
scripts/common/              modules shared by every analysis
scripts/<analysis>/          one folder per analysis
<analysis>.ipynb             the notebook that runs that analysis
```

Analyses (one folder and one notebook each):

| Folder | Notebook | Covers |
|---|---|---|
| `mean_level_and_tail` | `mean_level_and_tail.ipynb` | Threads 1 to 3: phase scoring and pace share, the bookended-acceleration shape with equivalence verdicts, the first-innings tail |
| `dispersion_and_attribution` | | Threads 4 and 5: between-team dispersion (bootstrap screen, hierarchical models), powerplay divergence attribution |
| `interval_coverage` | | Simulation of the coverage of bootstrap-t and BCa intervals at the real cluster counts |
| `impact_substitute` | | Thread 6: impact-substitute deployment, with the innings-mix decomposition |
| `phase_boundaries` | | Thread 7: phase-boundary detection on first-innings over profiles |
| `expected_wickets` | | Thread 8: situation-only expected wickets |

Shared modules in `scripts/common/`:

- `config.py`: reads `config.toml`, resolves paths relative to the project root.
- `results_io.py`: writes tables, JSON and figures with a provenance sidecar.
- `derived_columns.py`: the ball-flag contract (legal, faced, boundary, dot, wicket, pace, aggressive shot, control).
- `innings_population.py`: per-innings table with the censoring classification (DLS reduced, successful chase, failed chase, first innings).
- `equivalence.py`: equivalence verdicts against a pre-stated smallest effect of interest.
- `synthetic_ballframe.py`: the schema-faithful synthetic fixture with its answer key. Mechanics only; nothing from it is a finding.

## Environment

Python 3.12 (`python --version` should print 3.12.x). Either

```
uv sync                                  # uses pyproject.toml and uv.lock
```

or

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Library versions in `requirements.txt` are the ones in `uv.lock`; the versions actually loaded at run time are recorded in every output sidecar.

## Running

Put the cleaned dataset in `data/` (file names in `config.toml` under `[paths]`), open the analysis notebook at the root, run top to bottom. Notebooks add `scripts/` to `sys.path` and import the analysis modules; the modules do the work, the notebook displays and writes. If the cleaned dataset is absent the notebook runs on the synthetic fixture and says so; the outputs it writes are then mechanics checks, not results.

## Provenance conventions

- Seeds are fixed integers in `config.toml`, chosen once when written and never derived from the clock. One seed per analysis (all currently equal to the global seed).
- Every table, JSON and figure in `outputs/` has a `.meta.json` sidecar with the analysis name, seed, run timestamp, producing notebook or script, the bootstrap settings, the season pair, library versions, and any analysis-specific settings passed by the producer.
- All intervals are 95% unless stated. Equivalence verdicts use a 90% interval (two one-sided tests at 5%).
- The resampling unit is the match everywhere; each season is resampled from its own match pool; every theme in an analysis reads the same draws.

## Regression against the stored results

`scripts/mean_level_and_tail/regression_check.py` reproduces the previous notebook's stored table with its original seeds and the pooled jackknife, so that unchanged estimands are shown to be unchanged bit for bit while the interval machinery moves. It runs from the last cell of `mean_level_and_tail.ipynb` on real data only.
