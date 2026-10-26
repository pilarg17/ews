# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
from pandas.tseries.offsets import BDay


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration
# Use the same local timezone for every daily run, regardless of server settings.

TIMEZONE = "Europe/Luxembourg"


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Central implementation dates
#
# Daily EWS convention:
#   RUN_DATE       = process / alert date (today)
#   REFERENCE_DATE = latest completed business day (yesterday)
#   PREVIOUS_DATE  = business day immediately before REFERENCE_DATE
# Business-day offsets skip weekends but do not account for bank holidays.
# This recipe calculates dates dynamically whenever it is executed.
TIMEZONE = "Europe/Luxembourg"
RUN_TIMESTAMP = pd.Timestamp.now(tz=TIMEZONE)
RUN_DATE = RUN_TIMESTAMP.tz_localize(None).normalize()
REFERENCE_DATE = (RUN_DATE - BDay(1)).normalize()
PREVIOUS_DATE = (REFERENCE_DATE - BDay(1)).normalize()
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validation
# Stop execution if the three dates are not in chronological order.

if not (
    PREVIOUS_DATE
    < REFERENCE_DATE
    < RUN_DATE
):
    raise ValueError(
        "Expected PREVIOUS_DATE < REFERENCE_DATE < RUN_DATE."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Output
# Publish a single-row date contract for all current-branch recipes.
# The printed dates make it easier to trace which days a scenario processed.

implementation_reference_date_df = pd.DataFrame({
    "RUN_DATE": [
        RUN_DATE
    ],
    "REFERENCE_DATE": [
        REFERENCE_DATE
    ],
    "PREVIOUS_DATE": [
        PREVIOUS_DATE
    ],
})


print()
print("IMPLEMENTATION REFERENCE DATE")
print("-----------------------------")
print(
    "Timezone:",
    TIMEZONE,
)
print(
    "Run timestamp:",
    RUN_TIMESTAMP.strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    ),
)
print(
    "Run date:",
    RUN_DATE.strftime(
        "%Y-%m-%d"
    ),
)
print(
    "Reference date:",
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    ),
)
print(
    "Reference weekday:",
    REFERENCE_DATE.day_name(),
)
print(
    "Previous business date:",
    PREVIOUS_DATE.strftime(
        "%Y-%m-%d"
    ),
)
print(
    "Previous business weekday:",
    PREVIOUS_DATE.day_name(),
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Refresh the shared dates dataset with the values used for this run.

dataiku.Dataset(
    "Implementation_Reference_Date"
).write_with_schema(
    implementation_reference_date_df
)
