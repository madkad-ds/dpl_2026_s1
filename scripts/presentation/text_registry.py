"""
Every piece of text that belongs to a presentation figure, in one place: title,
subtitle, caption (method line), axis units, and the narration line for the video.
Figure functions read from here for notebook viewing; the export writes the entry
as JSON beside the image so slide text is edited in the slide software, never in
the image. Numbers that appear in text are filled at build time from the outputs
(see figures.py: each builder returns the numbers it used).
"""

TEXT = {
    "season_contrasts_forest": {
        "title": "What changed between the seasons, with intervals",
        "caption": "Match cluster bootstrap on the same 10,000 match resamples, 95% intervals (bootstrap-t for rates and shares, log-scale for the ratio). Filled markers clear the null; hollow markers lean. All innings.",
        "narration": "About 40% more powerplay sixes; the powerplay run rate clears, the middle and death lean; pace displaced spin in every phase, most in the middle.",
    },
    "run_rate_mechanism": {
        "title": "Where the extra runs came from: sixes, and almost nothing else",
        "caption": "Exact Shapley attribution of the run-rate change to fours, sixes, dots, running and extras; the parts sum to the total on every match resample. All innings.",
        "narration": "In the powerplay, three quarters of the rise is more boundaries and a quarter is boundaries becoming sixes. Dots, running and extras did not move in any phase.",
    },
    "wicket_cost_three_levels": {
        "title": "No wicket cost, at three levels of accounting",
        "caption": "Change 2024 minus 2023, zero the common null. Actual rate: match bootstrap. Expected rate: situation-only model, out-of-fold, refit interval. Per aggressive shot: residual regression under the match bootstrap; carries the shot-tag floor. Band: the SESOI (smallest effect size of interest) stated in advance for the expected rate, 0.3 wickets per powerplay.",
        "narration": "The extra sixes cost nothing in wickets: not in the actual rate, not in the rate the situations predicted, not per aggressive shot.",
    },
    "powerplay_ledger": {
        "title": "What the 2024 powerplay mix was worth, at 2023 prices",
        "caption": "Each outcome priced as the mean change in the provided win probability on such balls; the change in each outcome's share of powerplay deliveries times its price, per 36-ball powerplay, match bootstrap intervals. Win probability is team-agnostic and weaker in first innings.",
        "narration": "About four win-probability points per powerplay, almost all from sixes; the wicket cost was small and mostly luck.",
    },
    "aggression_premium": {
        "title": "Diminishing returns: each aggressive shot bought half as much in 2024",
        "caption": "Mean change in win probability on aggressive-shot balls minus other balls, powerplay, both innings, match bootstrap intervals.",
        "narration": "Boundaries were priced lower as the league's baseline rose; the wicket cost per shot did not change. The approach paid through volume.",
    },
    "team_value_slope": {
        "title": "Who took the risk: three teams, and the order reversed",
        "caption": "Value of the powerplay outcome mix per team-season, points per powerplay at pooled prices, means over about fourteen powerplays; read as tiers.",
        "narration": "Sunrisers, Kolkata and Delhi attacked most and bought two to three times the value of the five teams that attacked least, paying a wicket cost no different from what their situations predicted.",
    },
    "line_length_grid": {
        "title": "Menu, not mix: every class of delivery was hit harder",
        "caption": "Powerplay sixes per 100 legal balls by line and length; cell area is the class's share of deliveries; hatched cells are thin (under 300 balls). Line and length are hand-tagged, coverage the same in both seasons.",
        "narration": "Bowlers went fuller and faster and it made no difference: 97% of the extra sixes came from every class being hit harder, 3% from the changed mix.",
    },
    "speed_bands": {
        "title": "Pace was the target; the slower ball held, and was cut",
        "caption": "Pace deliveries in the powerplay by tracked speed band, sixes per 100 legal balls, season change with match bootstrap intervals.",
        "narration": "Balls over 140 went from the safest class to the most hit; balls under 130 held their six rate and took more wickets, and bowlers cut the slower ball from 14% to 5.5% of deliveries.",
    },
    "spin_decomposition": {
        "title": "Spin was picked less and benched more; squads and workload unchanged",
        "caption": "Change in the spin share of legal balls, percentage points, attributed exactly to selection (spinners named per eleven), deployment (share of named spinners who bowled), workload (balls per spinner who bowled) and innings length, under the match bootstrap.",
        "narration": "Teams kept and bought spinners as before, named a third of a spinner fewer per eleven and bowled fewer of the ones they named; the middle overs were a headcount story.",
    },
    "batter_decomposition": {
        "title": "Half the rise was three openers; most of the other half was bought",
        "caption": "Exact decomposition of the powerplay six-rate change over batters, match bootstrap intervals; entrants split by how they came to face the balls, from the auction tables.",
        "narration": "Abhishek Sharma, Virat Kohli and David Warner carry more than the whole same-batters part; of the entrant effect, three quarters were bought or signed and a quarter promoted from within squads.",
    },
    "chase_target": {
        "title": "Chasing sides went harder at every target, and decided it early",
        "caption": "Change in the chasing run rate, 2024 minus 2023, raw and at a common target of 180 from a linear model with a season term, match bootstrap intervals; later phases on the chases that reached them.",
        "narration": "The powerplay change survives at a common target, so it is a change of approach; in the middle overs the scoreboard, not the season, drives the rate. Batting first attacked less than chasing in both seasons.",
    },
    "substitute_numbers": {
        "title": "The substitute: warm-up unchanged, twelfth batter new",
        "caption": "Impact-substitute deployments in the first innings by which side made them, from Cricsheet match records; the substitute's own batting from the ball-by-ball.",
        "narration": "Bowling-first sides warmed up the chase batter in both seasons; the 2024 change is batting-first sides adding a batter some 35 balls after a dismissal, who mostly did not bat.",
    },
}
