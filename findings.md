# Findings: how the 2024 season differed from 2023, and how sure we can be

Each finding names the notebook and output file it comes from. Numbers are from the runs of 9 to 13 September 2026 under the seeds in `config.toml`; a rerun with those seeds reproduces them, and where a table and this text ever disagree, the table is current.

## How to read this document

**Status words.** Every finding carries one of five labels.

- *Established*: the 95% interval excludes the null, or the posterior probability is at or above 0.95, under the primary method, and the result survived whatever robustness check was run.
- *Directional*: the point estimate has a clear sign and most of the resampling or posterior mass agrees, but the interval reaches the null. Worth reporting as a lean, never as a fact.
- *Tested null*: the change is small and the interval is narrow enough to say so, either by sitting inside a pre-stated band of effects too small to matter (an equivalence verdict) or by a posterior centred on zero with useful precision.
- *Unresolved*: the interval is too wide to say anything; the minimum detectable effect is stated so the reader knows what could not have been seen.
- *Descriptive*: a read of the data with no test, because a test would not mean what it appears to mean (a selected population, a confound that cannot be removed, or a table of a few dozen cases).

**Conventions used everywhere.** The unit of resampling is the match, never the ball, because balls within a match are not independent; each season is resampled from its own matches, and every contrast in an analysis reads the same draws. All intervals are 95% unless stated. Equivalence verdicts use a 90% interval against a smallest effect of interest fixed before the run. A *powerplay* is overs 1 to 6, the *middle* 7 to 15, the *death* 16 to 20. A *legal ball* is not a wide or a no-ball. Run rate is runs (including extras) per six legal balls; striker rates (six, four, dot) are per ball faced. *Tagged* columns (shot type, control, line, length, bowl type) are labelled by hand with an error of several percent, checked stable across seasons; intervals on measures that use them are a floor on the true uncertainty.

**Two seasons.** Everything here is a comparison of one season with the next. There is no trend, only a step, and nothing here is causal: the data say what changed, not why.

## The data, and what the validation established

The provided file is one row per delivery, 34,967 rows over 145 matches (74 in 2023, 71 in 2024), ten teams, one host country. Every check in `data_validation.ipynb` passed or was explained. The conventions that matter downstream:

- The file's bowler-runs column nets byes off the bowler on no-balls, which is not the scoring law; `bowler_charge` carries the standard convention and reconciles exactly with every bowler's running total. Economy uses it.
- The dismissed-player id is populated on every ball, not only on wickets; `p_out_clean` restores the documented definition and reconciles three independent ways (1,797 dismissals).
- Two innings carried an allowance of zero, a placeholder; the allowance is repaired from balls bowled plus balls remaining. A rain reduction can arrive mid-innings, so the innings minimum of the repaired allowance is the binding value.
- Control (whether the batter middled the ball) leaks onto 447 wides, which are not faced balls; `control_clean` nulls them. Fifty-five faced balls have no control tag and are excluded from rates, never scored as controlled.
- Shot-placement coordinates of (0, 0) are a missing-value placeholder, not a position, on 5,628 balls; 29,339 balls have real placement, and left and right-handers' zones pool directly.
- One team is spelled two ways across columns (Bangalore in the winner and toss fields, Bengaluru elsewhere); one spelling is used.
- The provided win-probability column is stated after the ball, is well calibrated in chases and over-confident in first innings in the 60 to 90% band, and is team-agnostic (it knows the match state, not the teams). The projected-score column converges on the final total except where an innings ended early.
- The bowl-type tag is missing three times as often in the death overs (11%) as elsewhere (4%) in both seasons; untagged balls are kept as their own group.

The impact-substitute records are not in the provided file. They were extracted from Cricsheet's per-match archive (cached under `data/` with its download date and checksum): 277 substitution events over the 145 matches, 276 impact players and one concussion replacement, every incoming player matched to a dataset id.

A second supplement arrived later: the auction tables for both seasons (every squad plus the unsold list, with status, type, team and price). Players are reconciled to dataset ids by an exact name key, then surname plus first initial, with a short fix list; a surname-only fallback was tried and dropped because it paired different players who share a surname. Match rates are 91% to 93% for retained players and 74% to 78% for sold ones; the unmatched remainder never faced or bowled a ball, and the thirteen players in the data with no auction row are mid-season replacement signings.

## What the exploration showed

`exploration.ipynb` is the descriptive first look, on full 120-ball innings. It is where every question below came from, and it is also where two face-value reads were later corrected.

The seasons are structurally alike: same ten teams, similar match loads, one country, similar toss and bat-first split, one no-result, rain-reduced innings at 0.7% and 1.4%. Season differences in scoring can be read as on-field change.

Scoring rose: mean total 175 to 183, run rate 8.97 to 9.60. The change was not uniform. The 90th percentile of totals moved 11 runs against 2 at the median, and innings of 220 or more went from 4.8% to 14.3% of all innings, while the boundary share rose two points at the mean and nearly four at the 90th percentile. Wickets per innings were flat. The rise was in boundaries, at no visible wicket cost, concentrated in the highest-scoring innings; it was spread across grounds, with Delhi and Kolkata amplifying it.

By phase, the powerplay (+0.77 runs per over) and the death (+0.74) rose almost equally and the middle about half as much (+0.42). The largest relative change anywhere was the powerplay six share, 4.6% to 6.4% of balls, with the wicket rate 4.1% to 4.4%.

Bowling: more pace in every phase, most in the middle (34% to 45% of legal balls); spin got more expensive than pace everywhere, most at the death, where it went from clearly cheaper to equally expensive on a shrinking and self-selected set of balls. The powerplay surge was hit off pace.

Where the runs went did not change: zone shares of scoring shots and boundaries are within a point or two in every zone. The shot repertoire re-weighted slightly toward the pull and away from the slog.

Teams did not move together: Delhi and Kolkata added nearly two runs an over with fewer wickets; Sunrisers and Bengaluru added 1.0 to 1.5 with more; Gujarat scored less with more wickets lost. Every team bowled more pace, by 3 to 14 points.

The impact substitute: deployments were almost identical in number (139 and 137) but moved into the first innings (26 to 43), so the share made in the second innings fell from 81% to 69%; at face value more entered at the death.

## Findings

### 1. The season contrasts with intervals

From `mean_level_and_tail.ipynb`, table `theme_contrasts.csv`. Every contrast on one shared set of 10,000 match resamples; bootstrap-t primary for rates, BCa alongside; the two agree to the second decimal everywhere.

| Contrast | Change | Interval | Status |
|---|---|---|---|
| Powerplay six rate | 1.39 times | 1.14 to 1.70 | Established |
| Powerplay wicket rate | +0.4 per 100 balls | −0.3 to +1.1 | Tested null (see finding 6) |
| Powerplay run rate | +0.73 runs per over | +0.18 to +1.35 | Established |
| Middle run rate | +0.36 | −0.18 to +0.92 | Directional |
| Death run rate | +0.68 | −0.01 to +1.40 | Directional |
| Pace share, powerplay | +6 points | +2 to +10 | Established |
| Pace share, middle | +10 points | +6 to +15 | Established |
| Pace share, death | +8 points | +3 to +12 | Established |
| Controlled-power shots against the slog | +3 points of shot share | +1.7 to +4.0 | Established, tagged |
| Death spin economy | +2.2 runs per over | +0.7 to +3.7 | Established as a description of the spin that was bowled, not of what spin would have cost; the set of balls shrank and was self-selected |

The pairing of the first two rows is the headline of the project: about 40% more powerplay sixes with no measurable wicket cost.

### 2. Did the ends accelerate while the middle lagged? Unresolved

The exploration read the phase changes as a barbell. Tested as one shape statistic (a curvature contrast weighting powerplay +1, middle −2, death +1) in boundary share, six share and run rate, plus the middle overs against the whole-innings change, with equivalence verdicts against smallest effects fixed in advance (0.5 runs per over, 1.5 points of boundary share, 0.75 of six share). Table `shape_contrast_verdicts.csv`.

- Middle against league run rate: −0.17 runs per over, 90% interval −0.41 to +0.06, inside the ±0.5 band. *Tested null*: the middle did not lag the league by a meaningful amount.
- Curvature contrasts: all lean positive (+2.4 points of boundary share, +1.4 of six share, +0.68 runs per over) with intervals including zero, and the minimum detectable effect in each channel is about three times the smallest effect of interest. *Unresolved*.

The acceleration was broad; whether it was larger at the ends could not be determined at this sample size. "Bookended" is retired as a claim.

### 3. The tail of innings totals: directional on the fair population

A chasing side stops when it reaches its target, so its total says nothing about its capacity; early chase finishes rose across the seasons, which alone moves the upper quantiles of a pooled distribution. The question is asked on first innings with a full allowance (74 and 70 innings), with failed chases (terminal, not censored) as a sensitivity population. Tables `tail_threshold_contrasts.csv`, `tail_shift_function.csv`, `tail_gpd.json`.

- 220+ share, first innings: 9.5% (7 of 74) to 18.6% (13 of 70), risk difference +9 points, Newcombe interval −2 to +21, Fisher p 0.15; bootstrap ratio 2.0 (0.8 to 5.3). *Directional*.
- With failed chases added (114 and 104 innings): +11 points (+3 to +20), Fisher p 0.011; ratio 2.8 (1.2 to 7.6). *Established on the sensitivity population*.
- Shift function: median +2.5 runs, 90th percentile +18, every interval including zero on first innings; the 80th and 90th percentiles clear (+12 and +21 runs) only with failed chases included.
- Generalized Pareto fit above 180: 220+ probability 12% to 22%, difference +10 points, interval −2 to +23 on first innings and +1 to +21 with failed chases.

On first innings, the share of 220+ totals roughly doubled, from about one in ten to one in five; the change is large but the interval on 144 innings does not exclude zero, and it does once failed chases are included. The pooled "one in twenty to one in seven" overstated the change by counting chases cut short at their target.

### 4. Where the extra runs came from: sixes, and almost nothing else

An exact decomposition of the run-rate change into fours, sixes, dots, running and extras by Shapley attribution, per phase, in `run_rate_mechanics.ipynb`, table `run_rate_decomposition.csv`. The totals reproduce the phase contrasts to the last digit.

- Powerplay: of +0.73 runs per over, sixes +0.60 (+0.24 to +0.97, the only component that clears), fours +0.12, dots and extras near zero, running −0.04. *Established*.
- Whole innings: sixes +0.38 of +0.54, 72% of the change. *Established*.
- Middle: sixes +0.26 of +0.36 (P 0.91). Death: sixes +0.41 and fours +0.25 of +0.68. *Directional*.
- In the frequency-against-size arrangement, powerplay boundaries per ball contribute +0.56 (clears) and the six share among boundaries +0.15 (clears; 20% to 26% of boundaries). About three quarters of the powerplay rise is more boundaries and a quarter is boundaries becoming sixes. At the death the six share was 46% in both seasons: frequency alone.
- The dot share moved by less than a point in every phase and running is flat: the non-boundary game did not change.

### 5. Same batters or different batters? Half and half in the powerplay, turnover elsewhere

The league powerplay six rate is a ball-weighted average of the batters who faced those balls; its change splits exactly into a within-batter part and a composition part. `batter_continuity.ipynb`, tables `powerplay_six_rate_components.csv`, `powerplay_six_rate_by_batter.csv`.

- League change +1.8 points (4.6% to 6.4%). Same batters +0.8 (−0.04 to +1.6, P 0.97): *directional*. Different batters +1.0 (+0.3 to +1.8): *established*. Batters new to the 2024 powerplay faced 22% of its balls and hit sixes at 7.0%, twice the 3.5% of the batters they replaced; reweighting among continuing batters did nothing.
- The within part is three openers: Abhishek Sharma (1.2% to 17.2% of balls, change interval +10 to +23 points), Virat Kohli (1.8% to 9.1%, +3 to +13) and David Warner (3.4% to 10.3%, +1 to +12), with KL Rahul also clearing. Most of the 30 continuing batters at the 30-ball floor did not change measurably.
- On continuing batters alone the powerplay six rate rose 4.9% to 6.3%, ratio 1.29 (1.03 to 1.61), against 1.39 for all batters. *Established*.
- Outside the powerplay it is turnover: middle overs within −0.2 points (P 0.29) and composition +1.0 (+0.4 to +1.8); death within +0.1 and composition +1.1 (P 0.96). Entrants hit sixes at roughly twice the rate of exits in both phases.
- Who the entrants were, from the auction tables (`composition_by_auction_route.csv`): of the +1.56 points entrants contributed, +0.82 came from fifteen batters bought at the 2024 auction (12% of the powerplay's balls, six rate 6.6%, median price 0.75 crore), +0.41 from two mid-season replacement signings not on any list (almost all Jake Fraser-McGurk at 19.8%), and +0.33 from eleven batters promoted from within squads (4.5%). Recruitment is about three quarters of the entrant effect. Travis Head is the largest bought contribution; Daryl Mitchell at 14 crore hit 2.7%. Exits were mostly batters kept but moved out of the powerplay, not batters released. *Descriptive*.

In the powerplay, about half of the rise is three continuing openers hitting more sixes, and most of the other half is batters teams bought or signed to do exactly that; through the rest of the innings the rise in six hitting is mainly a change in who bats.

### 6. Was the powerplay situationally safer, or just luckier? Neither: the wicket half of the headline holds

A situation-only expected-wickets model (over, wickets down, balls remaining, chase flag, required rate, batter hand; a second variant adds pace and spin) with strictly pre-ball features, a leakage audit, nested match-grouped cross-validation, calibration per phase, a logistic baseline beside a gradient booster, and refits on resampled seasons for model uncertainty. `expected_wickets.ipynb`, tables `powerplay_contrast.csv`, `equivalence_verdict.csv`, `fixed_vs_refit_intervals.csv`, `shot_level_rung.csv`.

- Expected powerplay wicket rate 0.0421 in 2023 and 0.0423 in 2024, change +0.0002 per ball, refit interval −0.0005 to +0.0004, identical across both feature variants and both models. The 90% interval sits inside the ±0.008 band (0.3 wickets per powerplay) by an order of magnitude. *Tested null*.
- Actual rate 0.0401 to 0.0444, +0.0043 (−0.0023 to +0.0113), the season-contrast reading; gap between actual and expected −0.004 (−0.011 to +0.003).
- Adding bowler type changes nothing: pace deployment is not what kept the powerplay situationally safe.
- Calibration where it matters: Brier 0.0502 for every fit; holdout Brier 0.051 on 29 matches; powerplay bias +0.2 points in 2023 and −0.2 in 2024, small but the same size as the expected change and opposite in sign, which is why the gap is read with its own interval.
- Shot-level rung: an aggressive shot carries +0.052 residual wickets per ball (+0.034 to +0.071) in both seasons; the season interaction is +0.007 (−0.021 to +0.035). Aggressive intent was as costly per shot in 2024 as in 2023. *Tested null on the change, tagged*.

The powerplay six rate rose about 40% with no measurable wicket cost; the expected wicket rate given situation was equivalent to zero change under two feature sets and two models, and the cost per aggressive shot did not change. The remaining uncertainty is in the actual wicket rate, not in the situational or shot-level accounting.

### 7. What the approach was worth in win-probability points: about four per powerplay

The provided win probability is stated after each ball, so the ball-to-ball change prices the ball. The 2024 outcome mix valued at fixed prices, per 36-ball powerplay, in `win_probability_ledger.ipynb`, tables `powerplay_ledger.csv`, `wicket_side_situational_vs_realised.csv`, `powerplay_aggression.csv`.

- At 2023 prices: +3.8 points per powerplay (+0.2 to +7.6, P 0.98). Sixes +3.2 (+1.3 to +5.2), fewer dots +0.8 (+0.02 to +1.7), fours +0.4, wickets −0.6 (−1.7 to +0.4). Pooled prices +3.5, 2024 prices +3.3. *Established as a league-average, state-average accounting*.
- By innings: second innings +6.5 (+1.8 to +11.3), first innings +1.2 on a wide interval. The second-innings figure is the better-founded one, because the model's first-innings calibration is weaker; the first innings is unresolved, not small.
- Wicket side: expected cost from the situational model −0.03 points per powerplay; realised −0.7; the difference is luck. The approach was worth about +3.8 realised and +4.5 in expectation.
- The premium of an aggressive shot halved, +0.93 (+0.73 to +1.14) to +0.52 (+0.26 to +0.77) points per ball, because boundaries were priced lower in 2024 (a six +5.0 to +4.3) while the wicket cost per shot was unchanged. Diminishing returns as the baseline rose, or the provided model re-basing to 2024 scoring; the two cannot be separated. *Descriptive*.
- Caveat: win probability drifts up by about 0.25 points per powerplay ball in both seasons (the model under-rates the batting side early, more in the first innings). The ledger cancels a uniform drift exactly.

The same accounting at the innings and team level (`powerplay_risk_ledger.ipynb`, tables `per_innings_ledger.csv`, `per_team_season_ledger.csv`) reproduces the league figure from the innings up (7.9 to 11.5 points per powerplay at pooled prices, boundary value up 3.4, wicket cost up 0.7, expected wickets 1.48 in both seasons) and attaches names. In 2024 Sunrisers (20.5 points per powerplay, 3.7 sixes), Kolkata (20.0, attempt share 24%) and Delhi (16.2) took the most risk and bought two to three times the value of the five teams that attacked least (3.4 to 8.7, attempt share 12% to 13%); no team-season paid a wicket cost far from what its situations predicted (realised 2.7 to 9.3 points against an expected 6.6 to 6.8). Risk in the 2024 powerplay was cheap to carry and expensive to forgo. *Descriptive with intervals; tiers on fourteen powerplays.*

### 8. Was there a line-and-length answer to the surge? No: menu, not mix

The bowling side of the powerplay, `bowling_response.ipynb`, tables `shift_share_components.csv`, `delivery_menu.csv`, `bowler_six_rate_components.csv`. Delivery classes are coarse (three lengths, three lines, nine type families, three speed bands and two release bands for pace); the league six rate against deliveries is split exactly into a within-class part (every class hit harder) and a mix part (a changed mix of what was bowled), under the match bootstrap.

- Of the +1.8 sixes per 100 powerplay balls, the within-class part is +1.75 (interval +0.66 to +2.86, 97% of the change) and the mix part +0.05 (−0.08 to +0.19); the same split holds in every dimension and for runs per ball (95% within). Bowlers changed what they bowled (fuller, faster, stock seam from 41% to 52% of deliveries, the slower ball from 14% to 5.5%, wrist spin halved, higher releases) but the classes traded were priced so alike that the shifts netted to nothing. *Established*: the surge was hit through, not bowled into.
- Pace was the target. Balls over 140 kph went from the safest class in 2023 (3.1% sixes) to the most hit in 2024 (7.0%; +3.9, interval +1.6 to +6.3); 130 to 140 rose 1.7 (clears); balls under 130 held (5.3% to 5.4%) and their wicket rate rose from 2.8% to 4.4%. Spin was hit hardest of all (+4.3 per 100 balls, interval +1.2 to +7.8).
- What held: the slower delivery (under 130 kph; the slower-ball family 4.8% to 5.0%) and the good length on the stumps (5.5% to 5.9%). The slower delivery is the one class with evidence of resisting the surge, and bowlers cut it by more than half. *Descriptive menu, selection caveat.*
- Same attacks, hit harder: continuing bowlers account for 83% of the change (+1.5 per 100, +0.4 to +2.5); entrants conceded more than exits (7.6% against 5.5%) but nearly cancel; continuing bowlers' six rate conceded rose 4.3% to 6.0%, ratio 1.36 (1.03 to 1.84). Only Harshit Rana's individual change clears. *Established.*

There was no line-and-length answer to the 2024 powerplay; batters adapted to pace, the fast ball went from safest to most hit, and the one delivery that held was the one bowlers stopped bowling.

### 9. Why pace displaced spin: selection and benching, not rationing or squad building

`spin_displacement.ipynb`, tables `spin_share_decomposition.csv`, `squad_to_eleven.csv`, `spinner_flow.csv`. The spin share of legal balls is an exact identity in spinners named per eleven, the share of named spinners who bowled, balls per spinner who bowled, and innings length; the change is attributed by Shapley values under the match bootstrap, per phase.

- Spin share fell from 41.6% to 33.5%, −8.0 points (interval −10.8 to −5.3). Selection (0.29 fewer spinners named per innings, −4.2 points, −7.3 to −1.3, 52%) and deployment (the used share of named spinners falling from 89% to 80%, −3.8 points, −6.2 to −1.3, 47%) did it in equal measure; workload per spinner who bowled was unchanged. *Established.*
- The middle overs, where pace rose most, were a headcount story: −10 points, all selection and deployment, with the spinners who bowled bowling slightly more middle overs each. At the death the available spinners were also taken off (workload −4.2, interval clears).
- Squads did not shed spinners: 3.6 to 3.8 classified spinners per squad, 17.5% to 16.4% of auction spend, 32 of 41 retained, three bought, two unsold. The cut was made at the team sheet: the share of a squad's spinners named per innings fell from 0.74 to 0.60, sharpest at Bengaluru, Mumbai, Lucknow and Kolkata; Sunrisers carried six and named 3.6; Punjab named all four and bowled 1.9; Gujarat went the other way. *Descriptive.*

Teams kept and bought spinners as before and chose not to play them, a match-day decision that reads as a response to what spin was conceding (finding 8).

### 10. Did the teams converge or pull apart? The powerplay stratified; the middle held; the death is flat on the fair population

The between-team spread of a measure, and its season change, as a third axis beyond the league mean. Bootstrap screen, hierarchical models under two priors, and a joint treatment of the powerplay. `dispersion_and_attribution.ipynb`, tables `dispersion_screen.csv`, `run_rate_dispersion_models.csv`, `powerplay_gear_up_model.csv`, `powerplay_legs_shared_draws.csv`, `powerplay_multivariate_model.csv`, `powerplay_shared_factor.csv`, `powerplay_shared_factor_robustness.csv`, `death_*.csv`.

- Powerplay aggressive-attempt share: between-team SD up 2.3 points (+0.3 to +3.9, P 0.99) around a league mean that did not move (15.1% to 15.2%). *Established, tagged*.
- Powerplay run-rate level, noise separated: SD 0.2 to 0.9 runs per over, +0.66, P 0.95 (0.94 under the second prior). *Established at the threshold*.
- Powerplay gear-up (a team's powerplay rate above its own middle-overs rate): SD 0.29 to 0.90, +0.58, P 0.92 under both priors. *Directional*.
- The three measures share matches: resample correlation 0.60 between level and gear-up; P(all three widened) 0.85. In the joint model P(widen) is 0.96 for attempts, 0.88 for level, 0.65 for gear-up (part of what the single model called gear-up spread is level spread once they may correlate).
- As one common factor: spread 0.19 to 0.93, log ratio 1.6 (0.3 to 4.6), P 0.99 under both priors, and unchanged when the factor is anchored differently or residual spreads vary by season. Dropping the tagged attempt measure pulls it to P 0.89 with an interval including zero. *Established when shot intent is included; directional on the tagging-clean measures alone*.
- Middle overs: flat (−0.05, P 0.41). *Tested null*. The exploration's apparent bifurcation was rank swapping, not spread.
- Death: on the honest population (first innings, full allowance, not all out before the death; 74 and 70 innings), gear-up SD 0.42 in both seasons, P(widen) 0.50. *Tested null on that population*, on a wide interval. The raw narrowing seen on all innings was early-chase selection. Every missing death ball is accounted for (31 and 42 chases ending early, a few weather reductions, zero unexplained).
- Bowling side: flat everywhere (P 0.4 to 0.6). Teams did not spread out in what they induced from batters. *Tested null, tagged*.

### 11. Who moved the powerplay spread: two poles, each team's place a coin flip or a little better

From the gear-up model's draws, as probabilities rather than a ranked list. Table `attribution_by_team.csv`.

- Between-team variance rose +0.54 (−0.01 to +1.30, P 0.95).
- Probability of ranking in the top three movers: Gujarat 0.58, Sunrisers 0.55, Kolkata 0.49, Delhi 0.40, Lucknow 0.30. Pull from the pack 0.42 to 0.76 runs per over with P(positive) 0.81 to 0.92. No team's variance contribution excludes zero; every team's raw and shrunk directions agree.
- Two poles are visible, an aggressive wing (Sunrisers, Delhi, Kolkata) and a conservative one (Gujarat, Lucknow). *Descriptive with probabilities*; the honest report is the pattern, not a list of five drivers.

### 12. The impact substitute: moved into the first innings, and used as insurance, not as a licence

Deployment (`impact_substitute.ipynb`, tables `deployment_themes.csv`, `death_entry_decomposition.csv`, `deployment_equivalence_verdicts.csv`):

- Second-innings share of deployments 81% to 69%, −12.7 points (−23.7 to −2.0). *Established*.
- The apparent shift toward death-overs entry (+7.6 points, −2 to +17) decomposes exactly into an innings-mix component of +8.1 (+1.2 to +15.5) and a within-innings timing component of −0.5 (−7.8 to +7.0). Within the second innings the death-entry share moved −1.6 points, 90% interval −7 to +4, inside the ten-point band: *tested null*. Within the first innings +2.9 points on 26 and 43 events: *unresolved*, minimum detectable effect 33 points.

Mechanics (`impact_substitute_mechanics.ipynb`, tables `deployments_*.csv`, `batting_first_swaps*.csv`, `eleven_composition_contrast.csv`, `bat_first_vs_chase_summary.csv`, `death_scoring_by_swap.csv`, `substitute_contribution*.csv`, `chase_target_response.csv`):

- The warm-up is real and unchanged: bowling-first sides made 17 first-innings substitutions in each season, mostly a batter for a bowler, at over 18 to 19, and the batter went on to bat about two thirds of the time. *Descriptive*.
- The 2024 rise is batting-first sides adding a batter mid-innings: 9 to 26 swaps, always after the outgoing batter's dismissal but a median of 35 balls later, a batter or all-rounder more often than a bowler, only 35% of them bowling later. Sunrisers ran it seven times from zero; the aggressive wing accounts for 14 of 26. *Descriptive*.
- The added batter was a passenger: six of 26 never batted; the rest faced 18% of the balls after entry at a strike rate of 112 against 134 for the batters already there.
- Elevens went the opposite way from a depth story: batting-first elevens are batting-heavier in both seasons, but the gap narrowed in 2024 (recognised-bowler gap −0.80 to −0.24, change +0.56, +0.07 to +1.06). *Established*.
- Batting first did not license powerplay risk: teams attacked less in the powerplay batting first than chasing in both seasons (about 3 points of attempt share, both clear), and in 2024 chasing sides geared up 0.9 runs per over more than batting-first sides (−1.80 to −0.02). *Descriptive with intervals* (a chase has its own target pressure).
- The target does not explain the 2024 chase powerplay: run rate +1.15 raw and +1.02 at a common target of 180 (+0.3 to +1.8), faster at every target band. A change of approach, not a response to bigger totals. *Descriptive with intervals*.
- Later in the chase the target does the work: in the middle overs the run rate responds to the target at 0.033 runs per over per run (clears) and the raw season difference of +0.43 shrinks to +0.34 at a common target (interval includes zero); at the death, reached by 86% and 82% of chases, the season effect is directional (+0.65 adjusted, −0.41 to +1.66). The chase aggression that rose in 2024 was front-loaded: decided in the powerplay, not forced by the scoreboard. *Descriptive with intervals*.
- Swap innings scored less at the death even at matched wickets (8.7 against 11.0 runs per over in the five-or-more band, 2024). *Descriptive*; insurance for an innings that had lost shape.

### 13. Did the phase boundaries move? No, to the detector's resolution of two overs

Slope-change breaks per channel on first-innings over profiles, with a pre-stated paired-resample move rule and a resolution check on a synthetic season with planted moves. `phase_boundaries.ipynb`, tables `move_test.csv`, `resolution_check.csv`, `dot_rate_settling_contrast.csv`.

- The powerplay edge sits at over 7 in every channel and both seasons: boundary rate at 7 in 99.8% of resamples per season, pace share in every resample, run rate and the pooled matrix at 7. *Established as stable to within one over*.
- The batting channels have no slope-change death boundary at all: the death is one ramp from over 7 to the end. The wicket channel and the pooled matrix put a late break at 16 to 18 that pairs inside the no-move band.
- Resolution: with no move planted the rule returns probabilities of 0.65 to 0.87; a one-over plant does not fire; a two-over plant does. The grid statement is "no batting-channel boundary moved by two or more overs".
- The alternative RBF cost returned "moved" at the death but also returned "moved" on identical seasons; it is set aside.
- The one channel with a diffuse 2023 edge (dot rate) shows no settling change on a direct test (overs 7 to 9: +0.2 points, −3.7 to +4.1).

The fixed grid is a sound basis for every phase-conditioned contrast.

### 14. Do the intervals cover? Yes at the league count; bootstrap-t is the safer construction at the team count

Simulation on real per-cluster pairs with planted effects, `interval_coverage.ipynb`, table `interval_coverage.csv`. At 70 clusters every method covers about 0.94 for differences and ratios, with and without an effect. At 14 clusters BCa slips to 0.91 to 0.92 while bootstrap-t holds at 0.93 to 0.94; nothing crosses 0.90, so team-level intervals keep the nominal label. Bootstrap-t is about 2% wider than BCa at 70 clusters and 10% at 14, which is where BCa's undercoverage comes from. Ratios on the log scale cover as well as differences.

## What holds together

Read as one story rather than twelve, the findings say this.

The 2024 season scored faster because more balls reached the fence, and in the powerplay more of them cleared it. Nothing changed in dots, running or extras. The powerplay six rate rose about 40%, the strongest single change in the data, and it came at no measurable cost: not in the actual wicket rate, not in the wicket rate expected from the situations teams faced, and not in the wicket cost per aggressive shot. Priced in win probability, the changed powerplay mix was worth about four points per powerplay, almost all from sixes, with the small realised wicket cost mostly luck.

Who did it is half and half in the powerplay: the same openers hit more sixes (three of them account for the whole within-batter part) and teams also put higher-hitting batters at the top, three quarters of them bought or signed for the purpose. On the bowling side it was the same attacks conceding more. Through the rest of the innings the rise in six hitting is mainly a change in who bats. Teams did not move together: the powerplay was where the ten teams pulled apart, into an aggressive wing and a conservative one, with each team's place in that ordering only a little better than a coin flip. The middle overs held their spread; the death did not change on the population where it can be measured.

Bowling could not answer it. Every class of delivery was hit harder in 2024; the mix changed (fuller, faster, more stock seam, far fewer slower balls) and the changes netted to nothing; the fast ball went from safest to most hit and the slower delivery, the one class that held, was bowled a third as often. Pace displaced spin, most in the middle overs, by picking fewer spinners and benching more of the ones picked while keeping and buying them as before: a match-day response to what spin was conceding. The impact substitute moved into the first innings, where batting-first sides began using it as a twelfth batter after a dismissal rather than converting a spent batter into a bowler; the added batter mostly did not bat, elevens became more balanced rather than more batting-heavy, and none of it showed as extra powerplay aggression for the batting-first side. The powerplay aggression that rose in 2024 rose in the chase, at every target, which is a change of approach rather than a response to bigger totals; later in the chase the scoreboard took over. At the innings and team level the same accounting says the sides that attacked the powerplay hardest bought two to three times the value of the sides that did not and paid no more for it than their situations predicted.

The phase grid every one of these numbers rests on did not move, and the intervals every one of them carries cover at their stated level with this many matches.

## Corrections to the exploration's face-value reads

Four reads from the first look did not survive testing, and the report should not carry them.

1. *The acceleration was bookended.* Unresolved; the middle did not lag the league by a meaningful amount and the ends-against-middle contrast could not detect an effect of the size that would matter.
2. *220+ innings went from one in twenty to one in seven.* On first innings, where totals are honest, from about one in ten to one in five, directional; the pooled figure counted chases cut short at their target.
3. *Impact substitutes entered more at the death in 2024.* Composition: more first-innings deployments, which have always tended to enter late; within-innings timing did not change.
4. *The middle overs bifurcated and the death converged.* The middle held its spread (the large per-team changes were rank swaps); the death's raw narrowing was early-chase selection and is flat on the fair population.
5. *Bowlers bowled into the surge.* A first read of the line-by-length grid, where the mix moved toward the classes that conceded most, suggested it; the exact decomposition put the mix at 3% of the change and the report should not say it.

## Standing limits

- Two seasons, so every finding is a step, not a trend, and nothing is causal.
- Ten teams and about fourteen matches per team: team-level findings resolve large changes only, and team orderings carry probabilities that are rarely far from a coin flip.
- Tagged columns carry an error of several percent, stable across seasons so it cancels in a difference but still widens the true interval; every finding that uses them says so.
- Intervals are resampling stability on these matches, not out-of-sample replication; the themes were found on the same matches they were tested on.
- The win-probability and expected-wicket ledgers are league-average, state-average accountings; the provided win-probability model is team-agnostic and weaker in first innings.
- Populations restricted to first innings (the tail, the death dispersion, the phase boundaries) are half the innings; that is the honest count and it is why those intervals are wide.

## Where everything lives

| Question | Notebook | Key outputs (`outputs/<notebook name>/`) |
|---|---|---|
| Data conventions | `data_validation.ipynb` | `added_columns.csv`; the cleaned parquet in `data/` |
| Descriptive picture, impact table | `exploration.ipynb` | `innings_summary.csv`, `phase_rates.csv`, `bowling_*.csv`, `team_*.csv`, `impact_player_table.csv` |
| Season contrasts, shape, tail | `mean_level_and_tail.ipynb` | `theme_contrasts.csv`, `shape_contrast_verdicts.csv`, `tail_*.csv`, `tail_gpd.json`, `regression_check.csv` |
| Run-rate mechanics | `run_rate_mechanics.ipynb` | `run_rate_decomposition.csv` |
| Same or different batters | `batter_continuity.ipynb` | `powerplay_six_rate_components.csv`, `powerplay_six_rate_by_batter.csv`, `other_rates_components.csv`, `composition_by_auction_route.csv` |
| Situational wicket cost | `expected_wickets.ipynb` | `powerplay_contrast.csv`, `equivalence_verdict.csv`, `fixed_vs_refit_intervals.csv`, `shot_level_rung.csv` |
| Win-probability value | `win_probability_ledger.ipynb`, `powerplay_risk_ledger.ipynb` | `powerplay_ledger.csv`, `wicket_side_situational_vs_realised.csv`, `powerplay_aggression.csv`; `per_innings_ledger.csv`, `per_team_season_ledger.csv` |
| Bowling response | `bowling_response.ipynb` | `shift_share_components.csv`, `delivery_menu.csv`, `bowler_six_rate_components.csv`, `bowler_continuing_contrast.csv` |
| Pace over spin | `spin_displacement.ipynb` | `spin_share_decomposition.csv`, `selection_by_team.csv`, `squad_to_eleven.csv`, `spinner_flow.csv` |
| Dispersion and attribution | `dispersion_and_attribution.ipynb` | `dispersion_screen.csv`, `*_model.csv`, `powerplay_shared_factor*.csv`, `death_*.csv`, `attribution_by_team.csv` |
| Impact substitute | `impact_substitute.ipynb`, `impact_substitute_mechanics.ipynb` | `deployment_*.csv`, `death_entry_decomposition.csv`, `batting_first_swaps*.csv`, `eleven_composition_contrast.csv`, `bat_first_vs_chase_summary.csv`, `substitute_contribution*.csv`, `chase_target_response*.csv`, `chase_phase_reach.csv` |
| Phase boundaries | `phase_boundaries.ipynb` | `move_test.csv`, `resolution_check.csv`, `rbf_resolution_check.csv`, `dot_rate_settling_contrast.csv` |
| Interval coverage | `interval_coverage.ipynb` | `interval_coverage.csv`, `team_level_interval_label.json` |

Every output has a `.meta.json` sidecar with the seed, settings, timestamp and library versions of the run that produced it.
