# Methodology: how each result was produced, and why that way

A record of the methods behind every finding in this project, written so I can defend each decision later and so a reader meeting these methods for the first time can follow what was done and why. It pairs with `findings.md`, which holds the results; this document holds the reasoning. Every method is explained in plain terms the first time it appears, then referred to by name. Code for each method lives in the scripts folder named in the relevant section; the notebooks only run the scripts and display the results.

## 1. The shape of the problem

The data are two seasons of ball-by-ball records, about 35,000 deliveries over 145 matches, ten teams, and the question is how the second season differed from the first in how teams batted, bowled and used the new substitute rule. Three features of that shape drive every method choice.

**Two seasons make a step, not a trend.** Every comparison is one season against the next. Nothing here can say whether a change is the start of a direction or a one-off, and nothing here is causal: the data say what changed, not why.

**Balls are not independent; matches are.** The deliveries of one match share a pitch, a weather, two line-ups and a scoreboard. Treating each of 35,000 balls as a separate observation would make every uncertainty far too small. The natural independent unit is the match, and there are only about 70 per season. Most of the methodology is about being honest with 70 units.

**The samples are small where it matters.** A team-season is about fourteen matches. A first innings that ran its full course is only about half the innings. Very high totals are a handful of games. Where a question rests on these small counts, the method has to say so, and the vocabulary in `findings.md` (established, directional, tested null, unresolved, descriptive) exists to keep the claims proportionate.

## 2. Shared methods

These are used across the analyses and documented once here.

### 2.1 The match cluster bootstrap

**What it is.** A *bootstrap* estimates the uncertainty of a statistic by rebuilding the dataset many times from itself: draw units at random with replacement, recompute the statistic, and read the spread of the recomputed values. A *cluster* bootstrap draws whole groups of related observations rather than single ones. Here the groups are matches: each season is rebuilt 10,000 times by drawing 70-odd matches with replacement from that season's matches (a match can appear twice or not at all), and the statistic is recomputed on each rebuilt season.

**Why the match.** Balls within a match are correlated, so a ball-level bootstrap understates the noise and makes everything look significant. Drawing matches keeps every within-match correlation intact because a match is drawn whole.

**How it is computed.** Every rate in this project is a ratio of sums (runs over balls, sixes over balls faced), and sums add across matches. So each analysis first collapses the deliveries to per-match totals of the numerator and the denominator, and a rebuilt season is a weighted sum of those totals with the draw multiplicities as weights. That makes 10,000 resamples a matrix product rather than 10,000 passes over the data, and it is why every quantity in this project can carry an interval.

**Shared draws.** Within an analysis, every statistic is evaluated on the same 10,000 draws. Two statistics that share matches then share their noise, and a joint statement about them (a decomposition, a correlation between measures, a difference of differences) is computed consistently rather than as if they were independent. Draws are generated in a fixed order from a fixed seed (section 2.9), so two analyses with the same seed see the same matches.

**What it cannot do.** The bootstrap describes how much a statistic would move if the same season were replayed with a different draw of its own matches. It is *resampling stability*, not out-of-sample replication: the patterns were found on these matches and tested on these matches. It also cannot fix a biased sample; if the population is wrong (section 2.5), resampling it carefully does not help.

Code: `scripts/mean_level_and_tail/season_contrast_bootstrap.py`, reused everywhere.

### 2.2 Estimands: differences, ratios, and the log scale

An *estimand* is the thing being estimated. The primary estimand throughout is a difference, later season minus earlier (null zero), because the difference of two ratio-of-sums rates is close to unbiased under the bootstrap and reads in natural units (runs per over, percentage points). A *ratio* (later over earlier, null one) is reported only where the claim is inherently relative, such as "40% more sixes", and always beside the difference.

Ratios are estimated on the *log scale*: the resampled values of log(ratio) are closer to symmetric than those of the ratio itself, and the bootstrap's bias correction behaves well there. Intervals are computed on the log scale and converted back. A common belief that a ratio estimated with noise is always pulled toward one, so that a ratio clearing one is a conservative floor, is not true in general; ratio bias can go either way. Nothing here treats a ratio as a floor.

One consequence to know: a log ratio of two standard deviations, when the earlier SD sits near zero, has an interval that runs far at the low end because the log of a value near zero is unbounded. In those places the SD difference and the posterior probability are the readable numbers and the log ratio is reported for completeness.

### 2.3 Intervals: bootstrap-t, BCa, percentile

Three ways to turn 10,000 resampled values into an interval are used, and the coverage simulation (section 3.12) says how each behaves at the real sample sizes.

**Percentile.** Take the 2.5th and 97.5th percentiles of the resampled values. Simple, and the interval used for decompositions and spread measures where the resample distribution is close to symmetric.

**BCa (bias-corrected and accelerated).** A percentile interval adjusted for two things: whether the resampled values sit systematically above or below the point estimate (bias), and whether the statistic's spread changes as the sample shifts (skew, estimated from *jackknife* values, the statistic recomputed leaving each match out in turn). Because the bootstrap draws each season separately, the jackknife is *stratified* by season: each season's leave-one-out values are centred on that season's own mean. A pooled jackknife would mix the between-season difference into the skew estimate. BCa is the primary interval where no closed-form standard error exists: quantile shifts and between-team SD differences.

**Bootstrap-t (studentized).** Every rate here has a closed-form standard error from the *delta method*: each match's contribution to a ratio of sums is a known quantity (its numerator minus the rate times its denominator, over the total denominator), and the variance of the rate is a sum of squares of those contributions. Differences and linear combinations of rates inherit it. The bootstrap-t interval divides each resampled deviation by that resample's own standard error, takes percentiles of those standardised deviations, and scales them back by the full-data standard error. It is the primary interval for rates, rate differences and contrasts of rates, because it does not depend on estimating skew from about 70 clusters, and the coverage simulation shows it holds its level better than BCa at 14 clusters at the price of being a little wider.

Every results table reports the primary interval, names the method, and carries the other constructions alongside; the two agree to the second decimal on the real data at 70 matches per season.

### 2.4 Equivalence verdicts and the smallest effect of interest

A wide interval containing zero is not evidence that nothing happened; it is evidence that the test could not tell. To say "no change" as a finding, the interval has to sit inside a band of effects too small to matter, and that band has to be chosen before the result is seen. The band is the *smallest effect size of interest* (SESOI), set in `config.toml` with a one-line justification each (for example 0.5 runs per over for a shape contrast, 0.3 wickets per powerplay for the expected wicket rate, ten percentage points for a substitute deployment share).

The verdict is three-way. *Rejects null*: the 95% interval excludes the null. *Equivalent to zero*: the 90% interval lies entirely inside the band; a 90% interval is the two one-sided tests procedure at the 5% level. *Inconclusive*: neither. Beside the verdict sits the *minimum detectable effect*, 2.80 standard errors, the smallest true effect the test would have found with 80% power; an inconclusive verdict with a minimum detectable effect three times the band says "could not have been seen", which is a different sentence from "was not there".

Code: `scripts/common/equivalence.py`.

### 2.5 Populations and censoring

A chasing side stops when it reaches its target. Its total, its death overs and its balls faced are cut short by the opposition's score, not by its own capacity; a statistician would say the chase is *right-censored at the target*. A rain-reduced innings is cut short by a different external event. And early chase finishes rose across the seasons, so pooling chases with first innings does not just add noise, it adds a season-dependent bias to anything about totals, the death overs, or over-by-over profiles.

Every analysis of those quantities therefore runs on a population a chase cannot truncate: first innings with the full 120-ball allowance, and for the death overs additionally not all out before the death began. Failed chases (the chasing side fell short, so its innings ran its course) are a sensitivity population for the tail. Second innings are shown descriptively where they are shown at all. Each such analysis reports its exclusion counts per season. Conditioning instead on "played a complete death" would not help: that selects on the outcome and is a different biased sample, not an unbiased one.

The classification is done once, in `scripts/common/innings_population.py`, so every analysis uses the same census.

### 2.6 Hierarchical models

**The problem they solve.** Ten teams' rates, each estimated from about fourteen matches, are noisy, and the raw spread across the ten is inflated by that noise: even ten identical teams would show a spread. A *hierarchical model* estimates each team's true rate while assuming the ten true rates come from a common distribution with its own spread, and it estimates that spread directly, separated from the match-to-match noise. Noisy teams are pulled toward the league (*shrinkage*); the estimated spread is what would remain if the noise were removed.

**How they are built.** Each team-innings run rate is modelled as Normal around the team's true rate with a standard deviation that shrinks with the number of balls (a short innings informs less). Team true rates are Normal around a season mean with a season-specific between-team SD, the quantity of interest. The models are written in *non-centred* form (team effects as standard-normal offsets scaled by the SD) because the centred form produces a geometry the sampler cannot explore; this is a computational choice, not a change of model. For the gear-up (a team's phase run rate minus its own middle-overs rate) both rates are observed per innings and the gear-up is formed inside the model, so both rates' noise propagates. The joint powerplay model gives each team-season a three-part effect (run-rate level, attempt log-odds, gear-up) from a multivariate Normal with a correlation shared across seasons and season-specific spreads; the common-factor version loads the three on one factor whose spread is estimated per season.

**Priors and checks.** Ten teams inform a spread weakly, so the prior on the spread matters. Every spread result is fitted under two priors of different shape near zero (HalfNormal, then Exponential) and reported under both; a result that survives the swap is in the data rather than the prior. Every fit reports zero divergences, r-hat near one and effective sample sizes in the thousands, which the runs met. The common-factor result was additionally refitted with the factor anchored on a different measure, with residual spreads free by season, and with the tagged measure dropped; the last is where it weakened, and the finding is stated at that strength.

**Reading the output.** Posterior probability that the spread widened, P(widen), is the main number; the SD difference in runs per over is the interpretable size; the log SD ratio carries the caveat in section 2.2.

Code: `scripts/dispersion_and_attribution/hierarchical_dispersion_models.py`, `powerplay_multivariate_model.py`.

### 2.7 Exact decompositions

Several findings are of the form "the league change is the sum of these parts". Two kinds of identity are used, and both are checked on every resample by computing the largest gap between the parts' sum and the whole (reported as `identity_max_abs_error`; it is at floating-point precision throughout).

**Shift-share (within against composition).** A league rate is a ball-weighted average of the batters (or innings types) who faced the balls. Its change splits exactly into a *within* part (each unit's rate changed, at average weights) and a *composition* part (the weights moved, at average rates), with entries and exits handled by giving an absent unit a zero weight and the same rate in both seasons. Used for same-batters-or-different-batters and for the impact substitute's innings-mix question.

**Shapley attribution.** When a quantity is a known function of several components (runs per ball as a function of fours, sixes, dots, running and extras), the change is attributed to the components by moving each from its earlier to its later value in turn and recording the increment, averaged over every ordering of the components. The average is order-free and the contributions sum exactly to the change. With five components there are 120 orderings, cheap enough to enumerate. Used for the run-rate mechanics and, in sampled form with a fixed model, for the expected-wickets feature-mix decomposition.

### 2.8 Synthetic fixtures with known answers

Model-based methods (the hierarchical models, the expected-wickets pipeline, the break detector) were checked on a synthetic two-season dataset with the real schema and a known answer planted in it: a powerplay six-rate rise, a one-over move of the death boundary, a dismissal hazard that is a known function of the pre-ball situation, and for the joint powerplay model a per-team scaling of powerplay scores. Each method had to recover what was planted and stay quiet on what was held fixed; the break detector's *resolution* (the smallest planted move it reliably finds) was measured this way and turned out to be two overs, which is reported as the limit of the phase-boundary claim.

Nothing from a fixture is a finding. The fixture proves the code does what it claims on data whose truth is known; the real data supply the results. When the real data file is absent, every analysis notebook runs on the fixture and says so.

Code: `scripts/common/synthetic_ballframe.py`.

### 2.9 Reproducibility

All seeds are fixed integers in `config.toml`, chosen once and never taken from the clock; a rerun reproduces every number. One configuration file holds seeds, resample counts, interval levels, smallest effects of interest, population rules and paths. Every table, figure and nested result is written with a sidecar recording the analysis, seed, settings, timestamp and library versions. The cleaned dataset is rebuilt from the raw file by one function whose output was verified against the previously cleaned file column by column. A regression check reproduces an earlier run's stored table exactly under its original seed, so that changes to the interval machinery can be shown not to have changed the estimands. Pinned library versions are exported from the lockfile.

### 2.10 Reconciling a supplement to the ball-by-ball data

Two outside tables were joined to the ball-by-ball file: Cricsheet's match records (for the impact substitute and the rosters) and the auction tables (squad status and price per player per season). Cricsheet shares the match id, so matches join exactly; players in both supplements are joined by name, because the id namespaces differ. The rule is the same in both: an exact match on a letters-only name key, then surname plus first initial where that is unique, then a short fix list of known spellings; nothing else. A surname-only fallback was tried for the auction tables and dropped after it paired different players who share a surname (Tom and Sam Curran, Sushant and Amit Mishra). Every unmatched row is listed rather than guessed, and for the auction tables the unmatched players in the data are all mid-season replacement signings who never went through an auction, which is what they should be.

Code: `scripts/exploration/impact_players.py`, `scripts/common/auction.py`.

### 2.11 The derived-column contract and tagged columns

Every rate is built from one set of ball flags defined once (`scripts/common/derived_columns.py`): legal ball, faced ball, four, six, dot off the bat, wicket, pace, aggressive shot, control lost. The denominators are deliberate: run rate and pace share are over legal balls, because a wide or no-ball adds runs without advancing the over; striker rates (six, four, dot, attempt) are over balls faced, because a no-ball is faced and a wide is not. Wickets come from the dismissal flag, never from the dismissed-player id, which is populated on every ball. Control comes from the cleaned column.

Shot type, control, line, length and bowl type are tagged by hand. The validation notebook sized their error at several percent and checked it stable across seasons, so it cancels in a season difference but still widens the true interval. Every finding that rests on a tagged column says so, and wherever a clean channel could carry the same question (run rate, boundary rate) the clean channel is primary.

## 3. Method by analysis

### 3.1 Data validation

`data_validation.ipynb`, `scripts/data_validation/`. The provided file is checked column by column against a declared schema, then for internal consistency: one row per delivery with unique keys; running totals never falling in delivery order; the runs identity on each ball (conditional on the no-ball flag); the bowler-runs convention and its reconciliation with running totals; the balls ledger and the allowance placeholder; dismissal coherence across five columns; chase-column identities with the two legitimate exceptions (rain-revised targets, an abandoned match); name-to-id integrity and a two-spelled team; missingness by column, by match and by season; the shot-placement geometry (circular statistics for the zone that straddles the 0/360 seam; left and right-handers pool); the two model columns' timing, convergence, calibration and team-awareness; and the tagged columns' vocabularies and contradictions. Every repair and derived column is applied by one function and the result written once. A fail is a lead, and each was either explained or repaired; none was silently dropped.

### 3.2 Exploration

`exploration.ipynb`, `scripts/exploration/`. Descriptive, on full 120-ball innings so rates compare like for like, always as rates because the seasons have different match counts. Distributions are reported with percentiles rather than means alone, because a change can live in a tail. The Cricsheet archive is cached with its date and checksum; substitutions are reconciled to dataset ids by a surname-plus-initial key with a short fix list, and every incoming player matched. The exploration asks questions; it does not test them.

### 3.3 Season contrasts

`mean_level_and_tail.ipynb`, `scripts/mean_level_and_tail/`. Each contrast (phase run rates, powerplay six and wicket rates, pace share by phase, a shot-mix shift, death spin economy) is a difference or ratio of per-match sums under the match bootstrap (section 2.1), bootstrap-t primary (section 2.3), on shared draws. The death spin economy contrast is reported as a description of the spin that was bowled, because teams bowled less spin at the death and the remaining balls are self-selected.

### 3.4 Shape contrasts

Same notebook. "The ends accelerated while the middle lagged" is one claim, so it is tested as one statistic: a *curvature contrast* weighting the three phase changes +1, −2, +1, positive when the ends moved more than the middle. It is run in three channels, boundary share and six share (dense, thousands of balls per phase, so well powered) and run rate, plus the middle overs against the whole-innings change. Each gets an equivalence verdict (section 2.4). The hypothesis is retired only if both dense channels come back equivalent to zero; otherwise it is unresolved, which is what happened.

### 3.5 The tail of innings totals

Same notebook. On the uncensored population (section 2.5): the 220+ share as a two-by-two table with a *risk difference* and a Newcombe score interval (which behaves well at small counts, where a normal-approximation interval overshoots or collapses) and Fisher's exact test; the match-bootstrap ratio as a secondary check; a *shift function* (the change in the innings-total quantile at 0.5, 0.6, 0.7, 0.8 and 0.9, each with a BCa interval) to show where in the distribution the change sits; exceedance at 180, 200 and 220; and a *generalized Pareto* fit to the excesses above 180, a standard model for the tail of a distribution, gated on at least 15 exceedances per season and refit inside the bootstrap. The 95th and 97th percentiles are not reported as findings because with 70 innings they are set by three or four scores; they are kept in an appendix table.

### 3.6 Run-rate mechanics

`run_rate_mechanics.ipynb`, `scripts/run_rate_mechanics/`. Runs per legal ball is an exact identity in fours per ball, sixes per ball, the dot share among non-boundary balls, runs per non-boundary scoring ball, and extras; a second arrangement separates boundary frequency from boundary size. Shapley attribution (section 2.7) under the bootstrap, per phase and whole innings, on all innings to match the season contrasts; the totals reproduce those contrasts exactly.

### 3.7 Same batters or different batters

`batter_continuity.ipynb`, `scripts/batter_continuity/`. The shift-share decomposition (section 2.7) of the league six rate at batter level, from per-match per-batter sums, under the bootstrap. A batter is continuing for the decomposition if he faced a ball in both seasons (the identity needs no floor); a batter's own rate is displayed and tested only at 30 or more balls in both seasons. A batter absent from both seasons of a resample contributes zero by construction. The continuing-batters-only league contrast beside the all-batter contrast, on the same draws, is the plain-language version. With the auction tables joined (section 2.10), each entrant's composition contribution is summed by how he came to face the balls (bought at the later auction, traded in, promoted from a retained squad place, or a replacement signing on no list) and each exit's by how he left; that split is descriptive, a re-labelling of contributions the decomposition already produced.

### 3.8 Situation-only expected wickets

`expected_wickets.ipynb`, `scripts/expected_wickets/`. The question is descriptive: did the expected powerplay wicket rate, given only what was known before each ball, change between seasons? The discipline is about leakage and calibration.

*Features.* Strictly pre-ball: over, wickets down, balls remaining, chase flag, required rate (blank in a first innings), batter hand; a second variant adds pace and spin indicators. The situation is rebuilt inside each innings as cumulative-minus-current, so a ball's own outcome can never enter its features; the provided running totals are not used because several are stated after the ball. A whitelist assertion refuses any other column, and a tripwire (out-of-fold AUC, which should be modest for a situation-only model at a 5% base rate) is checked before any model is trusted, with a deliberate leak injected to show the alarm fires.

*Models and evaluation.* Two models through the identical pipeline: a fixed-hyperparameter gradient booster and a logistic regression with over as a categorical. *Nested match-grouped cross-validation*: an outer split by match gives every ball an expected-wicket value from a model that never saw that ball's match (*out-of-fold*, without which the season contrast would be circular); a calibration step is fitted inside each outer training set on inner out-of-fold scores, with a separate calibrator per phase (so a bias that averaged out across phases could not hide inside the powerplay) and the calibration method chosen per phase. Scoring is calibration-first (Brier score, log-loss, reliability curves); the model with the lower Brier carries the contrast and the other is the robustness check, a tie going to the simpler model. A whole-match holdout is the uncontaminated check. The season-by-phase bias table shows the calibration where the contrast lives rather than asserting it.

*The contrast and its uncertainty.* Mean powerplay expected wickets per season against the actual rate, on the shared draws, with an equivalence verdict against 0.3 wickets per powerplay. The fixed-model interval treats the fitted model as known, so the whole pipeline is refitted on the first 500 shared draws and the wider of the fixed and refit intervals is reported. A sampled Shapley decomposition with the final model held fixed attributes the (small) expected change to features. The *shot-level rung* regresses the residual (actual minus expected) on aggressive intent with a season interaction under the bootstrap; it asks the question the situation model is blind to by design.

### 3.9 The win-probability ledger

`win_probability_ledger.ipynb`, `scripts/win_probability_ledger/`. The validation notebook established that the provided win probability is stated after each ball, so the ball-to-ball change prices what the ball produced. A calibrated probability is a *martingale* (its expected change is zero), so "did win probability rise more in 2024" is not the question; the ledger values the 2024 outcome mix at fixed prices. Each outcome category is priced as its mean change in win probability; the change in each category's share of powerplay deliveries is multiplied by its price; the sum per 36 balls is the value of the approach, at 2023 prices (primary), pooled and 2024 prices, pooled and per innings, under the bootstrap. Because shares sum to one, a constant added to every price cancels, so the ledger is invariant to a uniform drift in the model; it is not invariant to a change in relative prices, which is why fixed 2023 prices are primary. The wicket side is read twice, at the expected change from the situational model and at the realised change, and the difference is the luck component. Caveats carried: prices are averages over game states; the model is team-agnostic and weaker in first innings, so the second-innings figure is the better-founded one.

### 3.10 Dispersion and attribution

`dispersion_and_attribution.ipynb`, `scripts/dispersion_and_attribution/`. Whether the ten teams converged or pulled apart is a third question beyond the league mean, and a mean contrast cannot answer it.

*The screen.* The season change in the between-team SD of each measure (false-shot rate under aggressive intent, aggressive-attempt share by phase, gear-up, the bowling side's induced false-shot rate) under the match bootstrap, on shared draws, with the league mean checked stable so a spread change is not a level shift or a tagging drift, and with a precision guard (balls per team per phase, similar across seasons) read first because a spread is sensitive to estimation precision.

*The models.* Section 2.6, for run rate per phase and for the gear-up, under two priors.

*The powerplay as one question.* Three measures pointed at powerplay spread and share matches, so their agreement is weaker than three independent tests. All three are bootstrapped on the same draws with their correlation and the joint probability that all widened; then the joint hierarchical model and the common factor with its robustness variants.

*The death.* Re-asked on the uncensored population (section 2.5), with a diagnosis first that every missing death ball is explained (early chase finish, weather, all out) and none unexplained.

*Attribution.* From the gear-up model's draws, without a refit: each team's deviation from its season's league mean, its change in distance from the pack, the probability it ranks among the top movers, and its contribution to the change in between-team variance (contributions sum to the change exactly). Probabilities rather than a ranked list, because a team's change in distance from the pack has a skewed posterior and a point ranking on fourteen matches misleads. A check that no team's direction was flipped by shrinkage.

### 3.11 The impact substitute

`impact_substitute.ipynb`, `scripts/impact_substitute/`. The match is the unit (two substitutes in one match respond to the same game state). The pooled death-entry share change is decomposed exactly (section 2.7) into an innings-mix and a within-innings timing component; reallocation and timing shares get equivalence verdicts against ten points, and a direction is reported as a finding only when the interval excludes zero.

`impact_substitute_mechanics.ipynb`, `scripts/impact_substitute_mechanics/`. A watcher's model of why the substitute moved is broken into testable claims. Player roles are classified from each player's two-season record (bowler, batter, all-rounder at a 60-ball floor). Each deployment is classified by who made it, when, and whom it brought in and took out; for batting-first in-innings swaps, whether the outgoing batter had already been dismissed and how long before. Initial elevens are counted by role and compared by batting order under the bootstrap. Each team's powerplay attempt share and gear-up batting first against chasing is a *within-team* contrast, so team quality cancels, reported descriptively because a chase carries its own target pressure. Whether the target explains the chase powerplay is asked with a linear model per metric (target centred at 180, season, interaction) under the bootstrap, reading the season difference at a common target against the raw one; the same model is then run for the middle and death overs of chasing innings, with the share of chases reaching each phase reported first, because a chase that ended before the phase is absent and that is selection on match state. Death scoring by whether the side ran a swap is read at matched wickets and left descriptive, because the swap follows a dismissal and swap innings are selected on wickets; the substitute's own batting contribution is a table of a few dozen cases.

### 3.12 The bowling response

`bowling_response.ipynb`, `scripts/bowling_response/`. The batting findings raised the bowling question: was there a line-and-length answer to the surge? The prior question has to be settled first: did the extra sixes come from a changed mix of what was bowled, or from every class of delivery being hit harder? That is the shift-share decomposition of section 2.7 with delivery classes as the units: the league six rate (and runs per ball, dot rate, wicket rate) against deliveries is a share-weighted average over classes, and its change splits exactly into a within-class and a mix component, under the match bootstrap, for six dimensions (line by length, length, line, type family, speed band, release band) and for pace and spin separately.

Two decisions shape it. *Coarseness*: the tagged line and length have five and six values, and a thirty-cell grid at 170 balls a cell cannot carry a season contrast, so both are cut to three levels (full, good, short; off side, stumps, leg side), speed to three bands and release to two, with a class below 300 balls in either season marked thin. *Coverage first*: a season difference in how often a tag is blank would turn a tagging change into a mix change, so blank rates by season and bowler kind are checked before anything is decomposed (they are the same in both seasons; speed and release are blank on wides and no-balls and about 3% of legal balls, scattered across matches). The per-class menu (share, rates, changes with bootstrap-t intervals, the class's contribution to each component) is descriptive because the class a bowler chooses depends on the batter and the state; the decomposition is the inference. The bowler-level split reuses the batter decomposition (section 3.7) over bowlers.

### 3.13 Why pace displaced spin

`spin_displacement.ipynb`, `scripts/spin_displacement/`. The spin share of legal balls is an exact identity in four factors, three of them decisions a team makes: spinners named per bowling innings (from the twelve on each roster), the share of named spinners who bowled, spin balls per spinner who bowled, and legal balls per innings. The change is attributed by Shapley values (section 2.7) under the match bootstrap, for the whole innings and per phase; the phase versions keep the same selection and deployment counts and use the phase's balls, so they separate "picked fewer spinners" from "used them differently within the innings". A spinner is a bowler whose legal balls across the two seasons are mostly spin, at a 60-ball floor. The auction tables (section 2.10) add the level above the eleven: spinners carried per squad and the spend on them, so the drop can be placed at the squad, the team sheet or the bench.

### 3.14 The powerplay risk ledger

`powerplay_risk_ledger.ipynb`, `scripts/powerplay_risk_ledger/`. The league-level ledger (section 3.9) at the two levels a coach manages. Per ball: the situation-only logistic model (section 3.8) refitted out-of-fold so the ledger is self-contained, giving every powerplay delivery an expected-wicket value, beside its outcome category, the price of that outcome (pooled over both seasons, so a season's innings are not valued at its own prices) and the actual change in win probability. Per innings: sums of those, so each powerplay carries what was taken on (aggressive-shot share, sixes), what it was expected to cost (expected wickets times the wicket price), what it cost, the residual between them (the luck component), and the value of its outcome mix. Per team-season: means per powerplay with a match bootstrap interval. The team level is read as tiers; a team-season is fourteen powerplays.

### 3.15 Phase boundaries

`phase_boundaries.ipynb`, `scripts/phase_boundaries/`. The question is whether a boundary moved, never how many phases there are (a phase count on 20 points is a cluster-count decision with no reliable diagnostic). Over profiles per season on the uncensored population: run rate, boundary rate, dot rate, wicket rate (the batting matrix, pooled-robust scaled with fixed weights 1, 1, 1, 0.5, the wicket down-weight a design decision because a low-base rate on 20 cells is the noisiest channel) and pace share (a deployment choice, analysed as its own channel; an appendix shows that estimated inverse-variance weights would hand it the whole geometry).

*Detector.* The death is a ramp, not a step, so a constant-level break detector places its break wherever the ramp crosses a level, which moves with how high the ramp goes. The cost used is *piecewise linear*: within a segment each channel is a straight line in the over index and a break is a change of slope; PELT with contiguity and a minimum segment of three overs. The penalty is scanned on the pooled full-data batting matrix and set once at the log-midpoint of the widest run of values yielding two or three breaks in both seasons, then held fixed for every channel and resample. Per-channel breaks are primary; the pooled break is descriptive.

*Stability and the move rule.* Break locations are relocated on every match resample with the scaling refit inside each; a named boundary is anchored at the full-data break nearest the grid edge within three overs; on paired resamples, P(the two seasons' locations differ by at least one over) at or above 0.90 is "moved", otherwise "not contradicted", never "validated". The resolution check (section 2.8) showed that this rule's probability is 0.65 to 0.87 with nothing planted, does not fire on a one-over plant and does on a two-over plant, so every real-data probability is read against that band and the claim is stated at two-over resolution. An alternative RBF cost was run as a robustness check and returned "moved" on identical planted seasons, so it is set aside with that reason stated.

### 3.16 Interval coverage

`interval_coverage.ipynb`, `scripts/interval_coverage/`. Whether a nominal 95% cluster-bootstrap interval covers 95% of the time at 70 clusters, or at 14 for a team-level rate, is settled by simulation: real per-cluster pairs are drawn with replacement to build artificial seasons, a known effect is planted (or none), the exact interval machinery is run, and coverage is the share of datasets whose interval contained the truth. Twelve scenarios, 10,000 datasets each with 10,000 resamples. This is what licenses quoting every interval at its stated level, and it is why bootstrap-t is the primary construction for rates.

## 4. Strengths and weaknesses of the methodology

**Strengths.** The unit of inference matches the structure of the data, and the intervals' coverage was checked rather than assumed. Every claim carries a stated status and every null carries a stated smallest effect and minimum detectable effect, so absence of evidence and evidence of absence are kept apart. Populations are chosen to avoid the chase censoring that would otherwise bias totals, death overs and profiles, and the exclusions are counted. Decompositions are exact and verified on every resample. Model-based methods were validated on planted answers before they touched real data, and one method's resolution was measured and reported as a limit. Where a raw table suggested a story (bowlers bowling into the surge), the exact decomposition was allowed to overrule it. Everything reproduces from a raw file, a config and a seed, with provenance on every output.

**Weaknesses.** Two seasons give a step, not a trend, and no causal claim is possible. The team-level work is underpowered by construction (fourteen matches per team), and its established results rest partly on a hand-tagged column. Several results depend on the provided win-probability and expected-wicket ledgers being league-average and state-average accountings; a six at 0 for 2 is priced like one at 60 for 0. Intervals are resampling stability, not replication. The uncensored populations are half the innings, which is honest and expensive. The break detector resolves two overs, not one, so a one-over boundary move would be invisible to it. And the whole edifice rests on one provided file whose conventions were reverse-engineered from consistency checks; where those checks could not settle a convention (the exact role of the projected-score column, the hand-tagging procedure), the affected columns were kept out of the primary results.

## 5. Glossary

- *Bootstrap*: rebuilding a dataset from itself by random draws with replacement to measure how much a statistic would move.
- *Cluster*: a group of observations drawn together because they are related; here, a match.
- *Interval (95%)*: the range containing the middle 95% of the resampled values of a statistic, after any correction; "clears zero" means it does not contain the null value.
- *Bootstrap-t*: an interval built from resampled values standardised by their own standard errors.
- *BCa*: a percentile interval corrected for bias and skew, the skew estimated by leaving each match out in turn (jackknife).
- *Estimand*: the quantity being estimated.
- *Equivalence verdict*: a test of whether a change is inside a pre-stated band of effects too small to matter.
- *Minimum detectable effect*: the smallest true effect a test would have found with 80% power.
- *Censoring*: an observation cut short by an external event (a chase ending at its target) so its value understates what it would have been.
- *Hierarchical model*: a model that estimates each unit's true value while assuming the units share a common distribution, separating real spread from noise.
- *Prior*: a model's starting assumption about a parameter, before the data; two are tried so the result can be shown not to depend on it.
- *Posterior probability*: the model's probability of a statement after seeing the data, such as P(widen).
- *Shapley attribution*: crediting a change to its components by averaging each component's contribution over every order in which the components could be changed.
- *Out-of-fold*: a prediction for an observation from a model that never saw that observation's group.
- *Calibration*: whether predicted probabilities match observed frequencies; Brier score and log-loss measure it.
- *Leakage*: information about a ball's outcome finding its way into the features used to predict it.
- *Martingale*: a probability whose expected change from one step to the next is zero; a calibrated win probability behaves this way.
- *PELT*: an algorithm that finds the set of break points minimising a fit cost plus a penalty per break.
- *Resolution*: the smallest change a detector reliably finds, measured on planted cases.
- *Mix or menu*: whether a league rate against deliveries changed because of what was bowled (mix) or because every class of delivery was hit differently (menu); the within-class and mix components of a shift-share.
- *Delivery class*: a coarse grouping of deliveries by line and length, type, speed or release point, sized so a season contrast per class is possible.

## 6. References

- Rushe, Stephen. Cricsheet. https://cricsheet.org. Accessed September 2026. 
- Efron, Bradley. "Better Bootstrap Confidence Intervals." Journal of the American Statistical Association 82, no. 397 (1987): 171–85.
- Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap. New York: Chapman and Hall, 1993.
- Davison, A. C., and D. V. Hinkley. Bootstrap Methods and Their Application. Cambridge: Cambridge University Press, 1997.
- Cameron, A. Colin, Jonah B. Gelbach, and Douglas L. Miller. "Bootstrap-Based Improvements for Inference with Clustered Errors." Review of Economics and Statistics 90, no. 3 (2008): 414–27
- Lakens, Daniël, Anne M. Scheel, and Peder M. Isager. "Equivalence Testing for Psychological Research: A Tutorial." Advances in Methods and Practices in Psychological Science 1, no. 2 (2018): 259–69.
- Kitagawa, Evelyn M. "Components of a Difference between Two Rates." Journal of the American Statistical Association 50, no. 272 (1955): 1168–94.
- Shapley, Lloyd S. "A Value for n-Person Games." In Contributions to the Theory of Games, vol. 2, edited by Harold W. Kuhn and Albert W. Tucker, 307–17. Princeton, NJ: Princeton University Press, 1953.
Gelman, Andrew, and Jennifer Hill. Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge: Cambridge University Press, 2007.
- Betancourt, Michael, and Mark Girolami. "Hamiltonian Monte Carlo for Hierarchical Models." In Current Trends in Bayesian Methodology with Applications, edited by Satyanshu K. Upadhyay, Umesh Singh, Dipak K. Dey, and Appaia Loganathan, 79–101. Boca Raton, FL: CRC Press, 2015.
- Vehtari, Aki, Andrew Gelman, Daniel Simpson, Bob Carpenter, and Paul-Christian Bürkner. "Rank-Normalization, Folding, and Localization: An Improved R̂ for Assessing Convergence of MCMC." Bayesian Analysis 16, no. 2 (2021): 667–718
- Abril-Pla, Oriol, Virgile Andreani, Colin Carroll, Larry Dong, Christopher J. Fonnesbeck, Maxim Kochurov, Ravin Kumar, et al. "PyMC: A Modern, and Comprehensive Probabilistic Programming Framework in Python." PeerJ Computer Science 9 (2023): e1516.
- Niculescu-Mizil, Alexandru, and Rich Caruana. "Predicting Good Probabilities with Supervised Learning." In Proceedings of the 22nd International Conference on Machine Learning, 625–32. New York: ACM, 2005.
- Killick, Rebecca, Paul Fearnhead, and Idris A. Eckley. "Optimal Detection of Changepoints with a Linear Computational Cost." Journal of the American Statistical Association 107, no. 500 (2012): 1590–98.
- Truong, Charles, Laurent Oudre, and Nicolas Vayatis. "Selective Review of Offline Change Point Detection Methods." Signal Processing 167 (2020): 107299.
- Coles, Stuart. An Introduction to Statistical Modeling of Extreme Values. London: Springer, 2001.
- IPL Impact Player Regulations, 2023 
