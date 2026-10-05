"""
Pandera schema for the ball-by-ball cricket dataset.

Pandera lets you declare a *schema* — a contract for each column (its dtype,
whether it may be blank, and which values are legal) — and validate a DataFrame
against it, failing loudly on any violation.

Usage
-----
    import pandas as pd
    from cricket_schema import schema

    df = pd.read_csv("deliveries.csv")
    print(df.dtypes)                 # sanity-check dtypes BEFORE validating
    schema.validate(df, lazy=True)   # lazy=True collects ALL failures at once,
                                     # instead of stopping at the first one

Notes / assumptions (adjust to your actual file)
------------------------------------------------
* Import path: this uses the modern `pandera.pandas` namespace. On pandera
  < 0.20, change the import below to `import pandera as pa`.
* Dtypes: without seeing the CSV, ID columns (p_match, p_bat, p_bowl) are
  assumed integer. If any are strings, switch those `int` -> `str`.
* Nullable-int gotcha: pandas turns any integer column containing a blank into
  a float column (the blank becomes NaN). So every column that can be blank is
  declared `float` here, not `int`. If you read with pandas' nullable dtypes
  (e.g. read_csv(..., dtype_backend="numpy_nullable")), switch these to Int64.
* "Open set" columns (line, length, shot, bowl_style, bowl_kind) are validated
  for type/nullability only — the data dictionary lists examples ("e.g."), not
  a closed list, so an isin() check would raise false failures. Once you've run
  df[col].value_counts() and confirmed the real values, tighten them.
* `outcome` and `control` use isin() with the documented values; verify the
  exact spelling/casing in your CSV and edit the lists if they differ.
"""

import pandera.pandas as pa  # pandera < 0.20: use `import pandera as pa`

OUTCOMES = ['run', 'no run', 'four', 'six', 'out', 'wide', 'leg bye', 'bye', 'no ball']

DISMISSALS = ['caught', 'bowled', 'leg before wicket', 'run out', 'stumped', 'retired not out (hurt)', 'retired out', 'hit wicket', 'obstructing the field']

BOWL_STYLE = ['RF', 'RM', 'LBG', 'SLA', 'OB', 'RFM', 'LFM', 'LMF', 'RMF', 'LB', 'LWS', 'LM', 'LF', 'OB/LB', 'SLA/LWS', 'RM/OB/LB', 'RM/OB']

BOWL_KIND = ['pace bowler', 'spin bowler', 'mixture/unknown']

LINE = ['OUTSIDE_OFFSTUMP', 'ON_THE_STUMPS', 'DOWN_LEG', 'WIDE_OUTSIDE_OFFSTUMP', 'WIDE_DOWN_LEG']

LENGTH = ['GOOD_LENGTH', 'SHORT_OF_A_GOOD_LENGTH', 'FULL', 'SHORT', 'FULL_TOSS',
       'YORKER']

SHOT = ['FLICK', 'ON_DRIVE', 'COVER_DRIVE', 'PULL', 'CUT_SHOT', 'PUSH',
        'DEFENDED', 'SLOG_SHOT', 'LEFT_ALONE', 'SQUARE_DRIVE', 'STEERED',
       'STRAIGHT_DRIVE', 'SWEEP_SHOT', 'SLOG_SWEEP', 'DAB', 'LEG_GLANCE',
       'REVERSE_SWEEP', 'HOOK', 'UPPER_CUT', 'RAMP', 'PADDLE_SWEEP',
       'LATE_CUT', 'REVERSE_SCOOP', 'REVERSE_PULL', 'PADDLE_AWAY']

BOWL_TYPE = ['ANGLED IN', 'STRAIGHT BALL', 'ANGLED ACROSS', 'LEG SPIN',
       'Knuckleball', 'ORTHODOX', 'OFF SPIN', 'OFF CUTTER', 'GOOGLY',
       'OUTSWINGER', 'SLOWER BALL', 'ARM BALL', 'INSWINGER', 'BOUNCER',
       'CARROM BALL', 'CHINAMAN', 'WRONG ONE', 'LEGCUTTER', 'SLOW BOUNCER',
       'BACKHAND SLOWER', 'NIPBACKER', 'FLIPPER', 'NIPPED AWAY', 'SLIDER',
       'FASTER ONE', 'REVERSE SWING', 'TOP SPINNER', 'SLOW YORKER',
       'REVERSESWING YORKER', 'DRIFTER', 'INSWINGING YORKER']


schema = pa.DataFrameSchema(
    columns={
        # ---- Identifiers & delivery outcome ---------------------------------
        "row_id":    pa.Column(int,   nullable=False),
        "p_match":   pa.Column(int,   nullable=False),
        "inns":      pa.Column(int,   pa.Check.isin([1, 2])),
        "bat":       pa.Column(str,   nullable=False),
        "p_bat":     pa.Column(int,   nullable=False),
        "team_bat":  pa.Column(str,   nullable=False),
        "bowl":      pa.Column(str,   nullable=False),
        "p_bowl":    pa.Column(int,   nullable=False),
        "team_bowl": pa.Column(str,   nullable=False),
        "ball":      pa.Column(int,   pa.Check.ge(1)),      # raw count, incl. extras
        "ball_id":   pa.Column(float, pa.Check.ge(0)),      # over.ball, e.g. 0.01
        "outcome":   pa.Column(str,   pa.Check.isin(OUTCOMES), nullable=True),
        "score":     pa.Column(int,   pa.Check.ge(0)),      # bat + extras
        "out":       pa.Column(bool),
        "dismissal": pa.Column(str,   pa.Check.isin(DISMISSALS),   nullable=True),       # blank if no wicket
        "p_out":     pa.Column(int, nullable=True),       # blank if no wicket

        # ---- Running match state --------------------------------------------
        "over_num":           pa.Column(int,   pa.Check.in_range(1, 20)),
        "cur_bat_runs":   pa.Column(int,   pa.Check.ge(0)),
        "cur_bat_bf":     pa.Column(int,   pa.Check.ge(0)),
        "cur_bowl_ovr":   pa.Column(float, pa.Check.ge(0)),  # x.y overs format
        "cur_bowl_wkts":  pa.Column(int,   pa.Check.ge(0)),
        "cur_bowl_runs":  pa.Column(int,   pa.Check.ge(0)),
        "inns_runs":      pa.Column(int,   pa.Check.ge(0)),
        "inns_wkts":      pa.Column(int,   pa.Check.in_range(0, 10)),
        "inns_balls":     pa.Column(int,   pa.Check.ge(0)),
        "inns_runs_rem":  pa.Column(float, nullable=True),   # 2nd innings only
        "inns_balls_rem": pa.Column(int,   pa.Check.ge(0)),
        "inns_rr":        pa.Column(float, pa.Check.ge(0)),
        "inns_rrr":       pa.Column(float, nullable=True),   # 2nd innings only
        "target":         pa.Column(float, nullable=True),   # 2nd innings only
        "max_balls":      pa.Column(int,   pa.Check.in_range(0, 120)),

        # ---- Match meta, players, wagon wheel -------------------------------
        "match_date":       pa.Column(str,   nullable=False),
        "year":       pa.Column(int,   pa.Check.in_range(2003, 2026)),  # widen if needed
        "ground":     pa.Column(str,   nullable=False),
        "competition":     pa.Column(str,   nullable=False),
        "country":    pa.Column(str,   nullable=True),
        "winner":     pa.Column(str,   nullable=True),       # blank on no-result/tie
        "toss":       pa.Column(str,   nullable=True),
        "bat_hand":   pa.Column(str,   pa.Check.isin(["LHB", "RHB"]), nullable=False),
        "bowl_style": pa.Column(str,   pa.Check.isin(BOWL_STYLE), nullable=False),       # open set
        "bowl_kind":  pa.Column(str,   pa.Check.isin(BOWL_KIND), nullable=True),       # pace/spin/mixture/unknown
        "batruns":    pa.Column(int,   pa.Check.ge(0)),      # off the bat, excl. extras
        "ballfaced":  pa.Column(int,   pa.Check.isin([0, 1])),
        "bowlruns":   pa.Column(int),
        "wagonx":     pa.Column(float, pa.Check.ge(0), nullable=True),
        "wagony":     pa.Column(float, pa.Check.ge(0), nullable=True),
        "wagonzone":  pa.Column(float, pa.Check.in_range(0, 8), nullable=True),

        # ---- Delivery detail & model outputs --------------------------------
        "bat_out":   pa.Column(bool),
        "line":      pa.Column(str,   pa.Check.isin(LINE), nullable=True),        # open set
        "length":    pa.Column(str,   pa.Check.isin(LENGTH), nullable=True),        # open set
        "shot":      pa.Column(str,   pa.Check.isin(SHOT), nullable=True),        # open set
        "control":   pa.Column(float, nullable=True),
        "predscore": pa.Column(float, pa.Check.ge(0), nullable=True),
        "wprob":     pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
        "noball":    pa.Column(int,   pa.Check.isin([0, 1])),
        "wide":      pa.Column(int,   pa.Check.ge(0)),       # can exceed 1 (wide + byes)
        "byes":      pa.Column(int,   pa.Check.ge(0)),
        "legbyes":   pa.Column(int,   pa.Check.ge(0)),
        "bowl_speed_category": pa.Column(float, pa.Check.in_range(1, 8), nullable=True),
        "bowl_release_point":  pa.Column(float, pa.Check.in_range(1, 4), nullable=True),
        "bowl_type": pa.Column(str,   pa.Check.isin(BOWL_TYPE),   nullable=True),
    },
    coerce=True,   # cast each column to its declared dtype before checking
    strict=True,   # fail if the CSV has unexpected extra columns; set False to relax
)