# IPL 2023 to 2024: what changed, how sure we can be, and what a franchise can do with it

A ball-by-ball comparison of two IPL seasons from one provided file, built to answer three questions: what changed in how teams batted, bowled and used the new substitute rule; whether each change is real once the luck of about seventy matches a season is accounted for; and whether it was a change in how teams played or in who played. The work is a set of notebooks, each backed by a folder of scripts, meant to be read in the order below. Every notebook opens with the question it answers, explains its method in plain terms, shows the results, and says what they mean.

Read `prelims_submission.pdf` for a walkthrough of the project and results.  `findings.md` holds every result at the confidence it earned, and `methodology.md` explains each method, why it was chosen, and what it cannot do.

## The short version

- The powerplay six rate rose about 40% at no measurable wicket cost, checked three ways (actual rate, the rate the situations predicted, the cost per aggressive shot). Priced in the provided win probability, the 2024 powerplay mix was worth about four points per powerplay, almost all from sixes.
- The extra runs were boundaries and nothing else; dots, running and extras did not move in any phase.
- Half the powerplay rise was three continuing openers; most of the other half was batters bought or signed to hit sixes.
- There was no line-and-length answer: every class of delivery was hit harder, the fast ball went from safest to most hit, and the slower ball, the one delivery that held, was bowled a third as often. Spin was benched at the team sheet, not released at the auction.
- The aggression that rose was in the chase, at every target, decided in the first six overs. The impact substitute became insurance for the batting innings.
- The ten teams pulled apart in the powerplay into an aggressive wing and a conservative one; the middle overs held; the death did not change on the population a chase cannot truncate.
- Four plausible stories did not survive testing: a bookended innings, routine 220s, converging death overs, moving phase boundaries.

Everything is one step from one season to the next, and nothing is causal.

## Read in this order

| Notebook | What it does |
|---|---|
| `data_validation.ipynb` | Checks the provided file for internal consistency, records its conventions, repairs documented placeholders, and writes the cleaned parquet every other notebook reads |
| `exploration.ipynb` | The first look: are the seasons comparable, how did scoring change and where in the innings, what was bowled, where the runs went, which teams moved; extracts the impact-substitute table from Cricsheet |
| `mean_level_and_tail.ipynb` | Season contrasts with intervals: phase run rates, the powerplay six rate and its wicket cost, pace share, the shape of the acceleration, the tail of innings totals |
| `dispersion_and_attribution.ipynb` | Did the ten teams converge or pull apart, per phase and discipline, and who moved the powerplay spread |
| `interval_coverage.ipynb` | Do the intervals used throughout cover at their stated level with this many matches |
| `impact_substitute.ipynb` | When and where the impact substitute was used, with the innings-mix decomposition |
| `phase_boundaries.ipynb` | Did the phase boundaries themselves move; the detector's resolution |
| `expected_wickets.ipynb` | Was the powerplay situationally safer, or just luckier; the shot-level cost of aggression |
| `run_rate_mechanics.ipynb` | Where the extra runs came from: the run-rate change decomposed exactly into fours, sixes, dots, running and extras |
| `batter_continuity.ipynb` | Same batters or different batters: the six-rate change decomposed into within-batter change and personnel change; entrants by auction route |
| `win_probability_ledger.ipynb` | What the 2024 powerplay approach was worth in win-probability points, with the wicket side read against the situational model |
| `bowling_response.ipynb` | The bowling side of the surge: mix or menu, the delivery menu, the line-by-length grid, same attacks or different bowlers |
| `spin_displacement.ipynb` | Why pace displaced spin: selection, deployment, workload, and the squad level from the auction tables |
| `impact_substitute_mechanics.ipynb` | Who deployed the substitute in the first innings and whom, eleven composition, batting first against chasing, and whether the target explains the chase |
| `powerplay_risk_ledger.ipynb` | Per innings and per team-season: aggression taken, expected and realised wicket cost, value in win-probability points |
| `presentation_figures.ipynb` | Every deck figure drawn from the written outputs at slide size, exported without titles, with all slide text in one JSON |

Each notebook imports from the matching folder under `scripts/` and writes tables and figures under `outputs/<notebook name>/`, every file with a `.meta.json` sidecar recording the seed, settings, timestamp and library versions. Numbers quoted in a notebook's text are from the run under the seed in `config.toml`; if a table ever disagrees with the text, the table is current.

## Running

### One command

```
python run_all.py            # full run at the counts in config.toml (a few hours)
python run_all.py --quick    # lowered counts, about fifteen minutes, every notebook still runs end to end
python run_all.py --from spin_displacement   # resume partway through the order
python run_all.py --only expected_wickets    # one notebook
```

`run_all.py` executes every notebook in reading order, in place, stops at the first failure and names it, and writes `outputs/run_manifest.json` (what ran, how long, the mode, the config used). Quick mode rewrites the counts in `config.toml` for the run and restores the full config afterwards.

### By hand

Put the inputs in `data/` (see below), open a notebook at the root, run top to bottom. Notebooks add `scripts/` to `sys.path`; the modules do the work, the notebook displays and writes. If the cleaned dataset is absent, an analysis notebook runs on a synthetic fixture with a known answer key and says so; those outputs are mechanics checks, not results.

### Environment

Python 3.12. Either

```
uv sync                                  # uses pyproject.toml and uv.lock
```

or

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The versions actually loaded at run time are recorded in every output sidecar. Runtime at the full counts: the dispersion notebook fits about twenty PyMC models (10 to 20 minutes); phase boundaries 10 to 15 minutes; the expected-wickets refit cell 15 to 60 minutes with four workers; the coverage simulation about half an hour. Everything else is seconds to a few minutes.

## Data flow

1. Put the provided ball-by-ball CSV and the auction workbook in `data/` under the names in `config.toml` (`[paths]`).
2. `data_validation.ipynb` validates the CSV against a declared schema, writes `dpl_prelims.parquet` (values unchanged) and `dpl_prelims_clean.parquet` (repairs and derived columns; no rows removed).
3. `exploration.ipynb` downloads the Cricsheet IPL archive once into `data/` with a sidecar recording the date and checksum (later runs read the cache) and writes `impact_player_table.parquet` and `match_rosters.parquet`.
4. The analysis notebooks read the cleaned parquet and, where needed, the impact table, the rosters and the auction workbook.
5. `presentation_figures.ipynb` reads only the written tables under `outputs/`.

`data/` is not committed; `data/README.md` lists every file and which notebook writes it.

## Layout

```
config.toml                  seeds, resample counts, interval levels, smallest effects of interest, populations, paths
run_all.py                   the one-command driver
findings.md                  every result at the confidence it earned, with where it lives
methodology.md               each method, why it was chosen, what it cannot do; strengths and weaknesses; glossary
requirements.txt             pinned versions exported from uv.lock
pyproject.toml, uv.lock      the uv project definition and lockfile
data/                        inputs and intermediate files (not committed)
outputs/<analysis>/          tables (.csv), nested results (.json), figures (.png), each with a .meta.json sidecar
scripts/common/              modules shared by every analysis
scripts/<analysis>/          one folder per notebook
scripts/presentation/        deck figures: style, text registry, one function per figure
```

Shared modules in `scripts/common/`: `config.py` (reads `config.toml`, resolves paths), `results_io.py` (writes outputs with sidecars), `derived_columns.py` (the ball-flag contract: legal, faced, boundary, dot, wicket, pace, aggressive shot, control), `innings_population.py` (per-innings census with the chase-censoring classification), `equivalence.py` (verdicts against a pre-stated smallest effect), `auction.py` (the auction tables reconciled to player ids by name), `synthetic_ballframe.py` (a schema-faithful synthetic fixture with a known answer key; mechanics only).

## Conventions used everywhere

- The resampling unit is the match, never the ball; each season is resampled from its own matches; every contrast in a notebook reads the same draws.
- All intervals are 95% unless stated. Equivalence verdicts use a 90% interval (two one-sided tests at 5%) against the smallest effects of interest in `config.toml`, fixed before the runs.
- Ratios are estimated on the log scale and back-transformed; the difference is always reported alongside.
- Seeds are fixed integers in `config.toml`, chosen once and never derived from the clock.
- Run rate uses `score` over legal balls; striker rates use `batruns` over faced balls; economy uses `bowler_charge`; wickets come from the dismissal flag, never the dismissed-player column; control from `control_clean`; the innings allowance is the innings minimum of the repaired `max_balls`.
- Totals, death overs and phase boundaries are measured on first innings with a full allowance, because a chase stops at its target.
- Hand-tagged columns (shot, control, line, length, bowl type) carry about 8% labelling error, the same in both seasons; findings that rest on them say so.
- In figures: 2023 in grey-blue, 2024 in magenta, everything else in greys; filled markers clear the null, hollow markers lean.

## Status vocabulary

Every finding in `findings.md` carries one label: *established* (the interval excludes the null, or the posterior probability is at or above 0.95, and the result survived its robustness check); *directional* (a clear sign, interval reaches the null); *tested null* (inside a pre-stated band of effects too small to matter); *unresolved* (the interval is too wide, the minimum detectable effect stated); *descriptive* (no test, because a test would not mean what it appears to mean).

## Sources

The provided ball-by-ball file and auction tables; Cricsheet (https://cricsheet.org) for match records. Method references are listed at the end of `methodology.md`.
