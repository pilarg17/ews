# -*- coding: utf-8 -*-
# Independent date-window QA: safe to run in a Dataiku notebook.
# No production datasets are read or written.

import pandas as pd
from pandas.tseries.offsets import BDay

# The last confirmed usable source business date is 2026-10-21.
# Simulate the following run date, without modifying the production date recipes.
QA_REFERENCE_DATE = pd.Timestamp("2026-10-21")
QA_RUN_DATE = (QA_REFERENCE_DATE + BDay(1)).normalize()
QA_PREVIOUS_DATE = (QA_REFERENCE_DATE - BDay(1)).normalize()

# Apply the exact Previous-branch date-shift logic.
QA_PREV_BRANCH_RUN = QA_REFERENCE_DATE
QA_PREV_BRANCH_REFERENCE = QA_PREVIOUS_DATE
QA_PREV_BRANCH_PREVIOUS = (QA_PREV_BRANCH_REFERENCE - BDay(1)).normalize()

assert QA_RUN_DATE == pd.Timestamp("2026-10-22")
assert QA_REFERENCE_DATE == pd.Timestamp("2026-10-21")
assert QA_PREVIOUS_DATE == pd.Timestamp("2026-10-20")
assert QA_PREV_BRANCH_RUN == pd.Timestamp("2026-10-21")
assert QA_PREV_BRANCH_REFERENCE == pd.Timestamp("2026-10-20")
assert QA_PREV_BRANCH_PREVIOUS == pd.Timestamp("2026-10-19")

print("LAST AVAILABLE DATE - DATE WINDOW QA")
print("Current branch : RUN_DATE={}, REFERENCE_DATE={}, PREVIOUS_DATE={}".format(
    QA_RUN_DATE.date(), QA_REFERENCE_DATE.date(), QA_PREVIOUS_DATE.date()))
print("Previous branch: RUN_DATE={}, REFERENCE_DATE={}, PREVIOUS_DATE={}".format(
    QA_PREV_BRANCH_RUN.date(), QA_PREV_BRANCH_REFERENCE.date(), QA_PREV_BRANCH_PREVIOUS.date()))
print("PASS: both date windows are aligned for a 2026-10-21 reference date.")
print("NOTE: this verifies date logic only, not source availability or EWS outputs.")
