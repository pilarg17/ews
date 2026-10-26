# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
from pandas.tseries.offsets import BDay


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Input and output
# Read the current-branch date contract and publish the shifted previous branch.

INPUT_DATASET = "Implementation_Reference_Date"
OUTPUT_DATASET = "Implementation_Reference_Date_previous"


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read current implementation dates
# Require exactly one row and all expected fields before shifting dates.

current_df = dataiku.Dataset(
    INPUT_DATASET
).get_dataframe()


if current_df.empty:
    raise ValueError(
        "{} is empty.".format(
            INPUT_DATASET
        )
    )


if len(current_df) != 1:
    raise ValueError(
        "{} must contain exactly one row.".format(
            INPUT_DATASET
        )
    )


required_columns = [
    "RUN_DATE",
    "REFERENCE_DATE",
    "PREVIOUS_DATE",
]


missing_columns = [
    column
    for column in required_columns
    if column not in current_df.columns
]


if missing_columns:
    raise KeyError(
        "{} is missing required column(s): {}".format(
            INPUT_DATASET,
            ", ".join(
                missing_columns
            ),
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Parse dates
# Convert incoming dates to timezone-naive midnight timestamps.
# Keep the same calendar dates provided by the current-branch dataset.

def parse_date(value):

    return (
        pd.to_datetime(
            value,
            errors="raise",
            utc=True,
        )
        .tz_localize(None)
        .normalize()
    )


CURRENT_RUN_DATE = parse_date(
    current_df.loc[
        current_df.index[0],
        "RUN_DATE",
    ]
)

CURRENT_REFERENCE_DATE = parse_date(
    current_df.loc[
        current_df.index[0],
        "REFERENCE_DATE",
    ]
)

CURRENT_PREVIOUS_DATE = parse_date(
    current_df.loc[
        current_df.index[0],
        "PREVIOUS_DATE",
    ]
)


if not (
    CURRENT_PREVIOUS_DATE
    < CURRENT_REFERENCE_DATE
    < CURRENT_RUN_DATE
):
    raise ValueError(
        "Expected PREVIOUS_DATE < REFERENCE_DATE < RUN_DATE."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Previous branch dates
# Move the previous branch back one business day:
# - the current reference date becomes this branch's run date;
# - the current previous date becomes this branch's reference date;
# - one more business-day offset supplies this branch's previous date.

RUN_DATE = CURRENT_REFERENCE_DATE

REFERENCE_DATE = CURRENT_PREVIOUS_DATE

PREVIOUS_DATE = (
    REFERENCE_DATE
    - BDay(1)
).normalize()


if not (
    PREVIOUS_DATE
    < REFERENCE_DATE
    < RUN_DATE
):
    raise ValueError(
        "Invalid previous-branch date sequence."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Build output
# Preserve the input dataset structure and replace only the three dates.

previous_df = current_df.copy()


previous_df.loc[
    previous_df.index[0],
    "RUN_DATE",
] = RUN_DATE

previous_df.loc[
    previous_df.index[0],
    "REFERENCE_DATE",
] = REFERENCE_DATE

previous_df.loc[
    previous_df.index[0],
    "PREVIOUS_DATE",
] = PREVIOUS_DATE


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Log the exact date window that downstream Previous recipes will read.

print()
print("=" * 90)
print("IMPLEMENTATION REFERENCE DATE - PREVIOUS")
print("=" * 90)

print(
    "RUN_DATE       :",
    RUN_DATE.strftime(
        "%Y-%m-%d"
    ),
)

print(
    "REFERENCE_DATE :",
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    ),
)

print(
    "PREVIOUS_DATE  :",
    PREVIOUS_DATE.strftime(
        "%Y-%m-%d"
    ),
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Publish the shifted dates without changing the current-branch dataset.

dataiku.Dataset(
    OUTPUT_DATASET
).write_with_schema(
    previous_df
)
