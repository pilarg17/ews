# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration
# Set the inputs used to prepare the daily loan dataset.

REFERENCE_DATASET = "Implementation_Reference_Date_previous"
RFO_DATASET = "RFO_PD_loan_Previous"
OUTPUT_DATASET = "vars_loan_Previous"


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers
# Use a consistent format for identifiers and dates from the source datasets.

def normalize_id(series):

    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )


def normalize_date(series):

    return (
        pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )


def parse_mixed_date(series):

    text = (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    compact = pd.to_datetime(
        text,
        format="%Y%m%d",
        errors="coerce",
    )

    generic = normalize_date(
        series
    )

    return (
        compact
        .fillna(
            generic
        )
        .dt.normalize()
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Reference date
# Read the date for which the daily loan records are being prepared.

reference_df = dataiku.Dataset(
    REFERENCE_DATASET
).get_dataframe()


if len(reference_df) != 1:
    raise ValueError(
        "{} must contain exactly one row.".format(
            REFERENCE_DATASET
        )
    )


if "REFERENCE_DATE" not in reference_df.columns:
    raise KeyError(
        "{} is missing REFERENCE_DATE.".format(
            REFERENCE_DATASET
        )
    )


REFERENCE_DATE = normalize_date(
    reference_df[
        "REFERENCE_DATE"
    ]
).iloc[0]


if pd.isna(REFERENCE_DATE):
    raise ValueError(
        "REFERENCE_DATE could not be parsed."
    )


REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read already-filtered RFO snapshot
# Load the RFO loan positions and check that all expected fields are available.

loan = dataiku.Dataset(
    RFO_DATASET
).get_dataframe()


loan.columns = [
    column.upper()
    for column in loan.columns
]


required_rfo_columns = [
    "INSTRUMENT_ID",
    "COUNTERPARTY_RISK",
    "RFO_SOURCE_DATE",
    "NET_EXPOSURE",
    "STAGE",
    "PAST_DUE_DAYS",
    "PERFORMING_STATUS",
    "GROSS_EXPOSURE",
    "COLLATERAL_ASSIGNED_AMOUNT",
    "IND_FORBEARANCE",
]


missing_columns = [
    column
    for column in required_rfo_columns
    if column not in loan.columns
]


if missing_columns:
    raise KeyError(
        "{} is missing required column(s): {}".format(
            RFO_DATASET,
            ", ".join(
                missing_columns
            ),
        )
    )


if loan.empty:
    raise ValueError(
        "{} is empty.".format(
            RFO_DATASET
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Prepare daily loan state
# Link loans to their entities and make sure the RFO snapshot is not later than the reference date.

loan[
    "INSTRUMENT_ID"
] = normalize_id(
    loan[
        "INSTRUMENT_ID"
    ]
)


loan[
    "ENTITY_ID"
] = normalize_id(
    loan[
        "COUNTERPARTY_RISK"
    ]
)


loan[
    "MAP_DTE"
] = REFERENCE_DATE


source_date = parse_mixed_date(
    loan[
        "RFO_SOURCE_DATE"
    ]
)


source_dates = (
    source_date
    .dropna()
    .drop_duplicates()
    .sort_values()
)


if len(source_dates) != 1:
    raise ValueError(
        "{} must contain exactly one RFO snapshot. Found {}.".format(
            RFO_DATASET,
            len(
                source_dates
            ),
        )
    )


if (
    len(source_dates) == 1
    and source_dates.iloc[0] > REFERENCE_DATE
):
    raise ValueError(
        "RFO source snapshot {} is after REFERENCE_DATE {}.".format(
            source_dates.iloc[0].strftime(
                "%Y-%m-%d"
            ),
            REFERENCE_DATE_STR,
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep only fields required by Daily EWS and CSV exposure output
# Keep the loan fields needed by the risk indicators and the exposure export.

output_columns = [
    "INSTRUMENT_ID",
    "ENTITY_ID",
    "MAP_DTE",
    "IND_FORBEARANCE",
    "STAGE",
    "NET_EXPOSURE",
    "PAST_DUE_DAYS",
    "PERFORMING_STATUS",
    "GROSS_EXPOSURE",
    "COLLATERAL_ASSIGNED_AMOUNT",
]


loan = loan[
    output_columns
].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final QA
# Check that each instrument appears once and that valid entity links are available.

duplicate_instruments = int(
    loan[
        "INSTRUMENT_ID"
    ]
    .duplicated()
    .sum()
)


if duplicate_instruments > 0:
    raise ValueError(
        "vars_loan_Previous contains duplicate INSTRUMENT_ID rows."
    )


rows = len(
    loan
)

entities = int(
    loan[
        "ENTITY_ID"
    ]
    .nunique(
        dropna=True
    )
)

instruments = int(
    loan[
        "INSTRUMENT_ID"
    ]
    .nunique(
        dropna=True
    )
)

missing_entities = int(
    loan[
        "ENTITY_ID"
    ]
    .isna()
    .sum()
)

print()
print("=" * 90)
print("VARS_LOAN PREVIOUS - QA")
print("=" * 90)

print(
    "Reference date      :",
    REFERENCE_DATE_STR,
)

print(
    "RFO snapshot        :",
    source_dates.iloc[0].strftime(
        "%Y-%m-%d"
    ),
)

print(
    "Rows                :",
    "{:,}".format(
        rows
    ),
)

print(
    "Instruments         :",
    "{:,}".format(
        instruments
    ),
)

print(
    "Entities            :",
    "{:,}".format(
        entities
    ),
)

print(
    "Rows without entity :",
    "{:,}".format(
        missing_entities
    ),
)

print(
    "Duplicate instruments:",
    "{:,}".format(
        duplicate_instruments
    ),
)


if rows == 0:
    raise ValueError(
        "vars_loan_Previous would be empty."
    )


if entities == 0:
    raise ValueError(
        "vars_loan_Previous has no populated ENTITY_ID values."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the prepared loan positions for the daily indicators and exposure export.

dataiku.Dataset(
    OUTPUT_DATASET
).write_with_schema(
    loan
)
