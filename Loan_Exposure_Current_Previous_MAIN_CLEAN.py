# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import numpy as np
import pandas as pd
import time
from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration
# Keep the existing SQL connection and contract-purpose codes used to classify mortgage and
# Lombard exposure.

DB_CONNECTION = "PRD_BDWH"

MORTGAGE_PURPOSE_CODES = [
    "101",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "120",
    "121",
    "123",
    "126",
    "127",
    "128",
    "139",
    "149",
]

LOMBARD_PURPOSE_CODE = "981"


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers
# Run SQL extracts, normalise identifiers and dates, and combine distinct source values for
# mapping checks.

def execute_query(database, query):

    start_time = time.time()

    result = SQLExecutor2(
        connection=database
    ).query_to_df(
        query
    )

    print(
        "Time to perform query: {:.1f} seconds".format(
            time.time() - start_time
        )
    )

    return result


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


def parse_report_date(series):

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).astype("Int64")

    return pd.to_datetime(
        numeric.astype("string"),
        format="%Y%m%d",
        errors="coerce",
    )


def concat_unique(series):

    values = (
        series
        .dropna()
        .astype("string")
        .str.strip()
    )

    values = values[
        values.ne("")
    ]

    return " | ".join(
        sorted(
            values.unique()
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation reference date
# Anchor the previous branch to its own reference date and derive the start of the current
# reporting month.

reference_df = dataiku.Dataset(
    "Implementation_Reference_Date_previous"
).get_dataframe()

if reference_df.empty:
    raise ValueError(
        "Implementation_Reference_Date_previous is empty."
    )

if "REFERENCE_DATE" not in reference_df.columns:
    raise KeyError(
        "Implementation_Reference_Date_previous is missing REFERENCE_DATE."
    )

REFERENCE_DATE = (
    pd.to_datetime(
        reference_df[
            "REFERENCE_DATE"
        ].iloc[0],
        errors="raise",
        utc=True,
    )
    .tz_localize(None)
    .normalize()
)

REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)

REFERENCE_DATE_KEY = int(
    REFERENCE_DATE.strftime(
        "%Y%m%d"
    )
)

PERIOD_START = REFERENCE_DATE.replace(
    day=1
)

PERIOD_START_KEY = int(
    PERIOD_START.strftime(
        "%Y%m%d"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read loan mapping
# Load the previous-branch instrument-to-entity and contract mapping.

loan_level_df = dataiku.Dataset(
    "vars_loan_level_Previous"
).get_dataframe()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate required loan mapping fields
# Check that contract identifiers, owning entities, purpose codes and snapshot dates are
# available.

required_columns = [
    "INSTRUMENT_ID",
    "ENTITY_ID",
    "MAP_DTE",
    "CONTRACT_ID_SHORT",
    "CONTRACT_INTERNAL_PURPOSE_CODE",
]

missing_columns = [
    column
    for column in required_columns
    if column not in loan_level_df.columns
]

if missing_columns:
    raise KeyError(
        "vars_loan_level_Previous is missing required column(s): {}".format(
            ", ".join(
                missing_columns
            )
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise loan mapping
# Align identifier and purpose-code formats before comparing contracts across datasets.

loan_level_df[
    "INSTRUMENT_ID"
] = normalize_id(
    loan_level_df[
        "INSTRUMENT_ID"
    ]
)

loan_level_df[
    "ENTITY_ID"
] = normalize_id(
    loan_level_df[
        "ENTITY_ID"
    ]
)

loan_level_df[
    "CONTRACT_ID_SHORT"
] = normalize_id(
    loan_level_df[
        "CONTRACT_ID_SHORT"
    ]
)

loan_level_df[
    "MAP_DTE"
] = normalize_date(
    loan_level_df[
        "MAP_DTE"
    ]
)

loan_level_df[
    "CONTRACT_INTERNAL_PURPOSE_CODE"
] = (
    loan_level_df[
        "CONTRACT_INTERNAL_PURPOSE_CODE"
    ]
    .astype("string")
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True,
    )
)

loan_level_df[
    "CONTRACT_INTERNAL_PURPOSE_CODE"
] = (
    loan_level_df[
        "CONTRACT_INTERNAL_PURPOSE_CODE"
    ]
    .mask(
        loan_level_df[
            "CONTRACT_INTERNAL_PURPOSE_CODE"
        ].isin(
            [
                "",
                "<NA>",
                "nan",
                "None",
            ]
        ),
        pd.NA,
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Resolve latest available loan mapping date
# Select the most recent available loan mapping not later than this branch reference date,
# and report its age.

available_mapping_dates = (
    loan_level_df.loc[
        loan_level_df[
            "MAP_DTE"
        ].notna()
        & (
            loan_level_df[
                "MAP_DTE"
            ] <= REFERENCE_DATE
        ),
        "MAP_DTE",
    ]
    .drop_duplicates()
    .sort_values()
)

if available_mapping_dates.empty:
    raise ValueError(
        "vars_loan_level_Previous contains no mapping date "
        "on or before implementation reference date {}.".format(
            REFERENCE_DATE_STR
        )
    )

LOAN_MAPPING_DATE = pd.Timestamp(
    available_mapping_dates.iloc[-1]
).normalize()

LOAN_MAPPING_DATE_STR = (
    LOAN_MAPPING_DATE.strftime(
        "%Y-%m-%d"
    )
)

LOAN_MAPPING_LAG_DAYS = int(
    (
        REFERENCE_DATE
        - LOAN_MAPPING_DATE
    ).days
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep latest available mapping snapshot
# Use only the selected mapping snapshot and discard records with missing contract or
# entity identifiers.

loan_mapping_raw_df = (
    loan_level_df[
        loan_level_df[
            "MAP_DTE"
        ] == LOAN_MAPPING_DATE
    ]
    .copy()
)

loan_mapping_raw_df = (
    loan_mapping_raw_df[
        loan_mapping_raw_df[
            "ENTITY_ID"
        ].notna()
        & loan_mapping_raw_df[
            "CONTRACT_ID_SHORT"
        ].notna()
    ]
    .copy()
)

if loan_mapping_raw_df.empty:
    raise ValueError(
        "vars_loan_level_Previous contains no usable rows "
        "for resolved mapping date {}.".format(
            LOAN_MAPPING_DATE_STR
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate contract ownership
# Stop if the same contract is linked to more than one entity within the chosen mapping
# snapshot.

contract_entity_count = (
    loan_mapping_raw_df
    .groupby(
        "CONTRACT_ID_SHORT"
    )[
        "ENTITY_ID"
    ]
    .nunique(
        dropna=True
    )
)

multi_entity_contracts = int(
    (
        contract_entity_count > 1
    ).sum()
)

if multi_entity_contracts > 0:

    problematic_contracts = (
        contract_entity_count[
            contract_entity_count > 1
        ]
        .index
    )

    print()
    print("MULTI-ENTITY CONTRACT SAMPLE")
    print("----------------------------")

    print(
        loan_mapping_raw_df[
            loan_mapping_raw_df[
                "CONTRACT_ID_SHORT"
            ].isin(
                problematic_contracts
            )
        ][
            [
                "ENTITY_ID",
                "INSTRUMENT_ID",
                "CONTRACT_ID_SHORT",
                "CONTRACT_INTERNAL_PURPOSE_CODE",
            ]
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    raise ValueError(
        "{} CONTRACT_ID_SHORT values are mapped "
        "to more than one ENTITY_ID.".format(
            multi_entity_contracts
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Classify purpose at instrument level
# Classify instruments as Lombard, mortgage or non-mortgage using the existing purpose-code
# rules.

purpose_code = (
    loan_mapping_raw_df[
        "CONTRACT_INTERNAL_PURPOSE_CODE"
    ]
)

loan_mapping_raw_df[
    "_PURPOSE_CLASS"
] = np.select(
    [
        purpose_code.eq(
            LOMBARD_PURPOSE_CODE
        ),
        purpose_code.isin(
            MORTGAGE_PURPOSE_CODES
        ),
    ],
    [
        "LOMBARD",
        "MORTGAGE",
    ],
    default="NON_MORTGAGE",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Consolidate mapping to contract level
# Combine instrument-level purpose information at entity and contract level.

contract_mapping_df = (
    loan_mapping_raw_df
    .groupby(
        [
            "ENTITY_ID",
            "CONTRACT_ID_SHORT",
        ],
        as_index=False,
    )
    .agg(
        PURPOSE_CLASS_COUNT=(
            "_PURPOSE_CLASS",
            "nunique",
        ),
        PURPOSE_CLASSES=(
            "_PURPOSE_CLASS",
            concat_unique,
        ),
        PURPOSE_CODES=(
            "CONTRACT_INTERNAL_PURPOSE_CODE",
            concat_unique,
        ),
        INSTRUMENT_COUNT=(
            "INSTRUMENT_ID",
            "nunique",
        ),
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Resolve contract-level classification
# Mark contracts as ambiguous when their mapped instruments have different purpose classes.

contract_mapping_df[
    "PURPOSE_CLASS"
] = np.where(
    contract_mapping_df[
        "PURPOSE_CLASS_COUNT"
    ].eq(
        1
    ),
    contract_mapping_df[
        "PURPOSE_CLASSES"
    ],
    "AMBIGUOUS",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Mapping conflict QA
# Report mixed-purpose contracts and the difference between the reference date and loan
# mapping date.

ambiguous_mapping_df = (
    contract_mapping_df[
        contract_mapping_df[
            "PURPOSE_CLASS"
        ].eq(
            "AMBIGUOUS"
        )
    ]
    .copy()
)

ambiguous_contract_count = len(
    ambiguous_mapping_df
)

print()
print("LOAN MAPPING DATE QA")
print("--------------------")

print(
    "Implementation reference date:",
    REFERENCE_DATE_STR,
)

print(
    "Loan mapping date:",
    LOAN_MAPPING_DATE_STR,
)

print(
    "Loan mapping lag days:",
    LOAN_MAPPING_LAG_DAYS,
)

print(
    "Raw loan mapping rows:",
    "{:,}".format(
        len(
            loan_mapping_raw_df
        )
    ),
)

print(
    "Contract-level mapping rows:",
    "{:,}".format(
        len(
            contract_mapping_df
        )
    ),
)

print(
    "Ambiguous purpose contracts:",
    "{:,}".format(
        ambiguous_contract_count
    ),
)

if ambiguous_contract_count > 0:

    print()
    print("AMBIGUOUS PURPOSE CONTRACTS")
    print("---------------------------")

    print(
        ambiguous_mapping_df[
            [
                "ENTITY_ID",
                "CONTRACT_ID_SHORT",
                "INSTRUMENT_COUNT",
                "PURPOSE_CODES",
                "PURPOSE_CLASSES",
            ]
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read Loandepo exposure
# Retrieve positive LOANDEPO exposures reported in the implementation month up to the
# reference date.

loandepo_query = """
SELECT
    ld.contract_reference,
    ld.report_datekey,
    ld.currency,
    ld.outstanding AS nominal_amount_source,
    ld.report_versionid
FROM DTM_RFO_PLY.LOANDEPO_VW ld
WHERE ld.report_datekey >= {period_start}
  AND ld.report_datekey <= {reference_date}
  AND (ld.accruals + ld.outstanding) > 0
""".format(
    period_start=PERIOD_START_KEY,
    reference_date=REFERENCE_DATE_KEY,
)

loandepo_df = execute_query(
    DB_CONNECTION,
    loandepo_query,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read account exposure
# Retrieve positive account exposures from the same monthly reporting window.

account_query = """
SELECT
    acc.account_reference AS contract_reference,
    acc.report_datekey,
    acc.currency,
    acc.balance AS nominal_amount_source,
    acc.report_versionid
FROM DTM_RFO_PLY.ACCOUNT_VW acc
WHERE acc.report_datekey >= {period_start}
  AND acc.report_datekey <= {reference_date}
  AND (acc.accruals + acc.balance) > 0
""".format(
    period_start=PERIOD_START_KEY,
    reference_date=REFERENCE_DATE_KEY,
)

account_df = execute_query(
    DB_CONNECTION,
    account_query,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Prepare exposure source
# Standardise each RFO extract and keep the most recent available version for each contract
# reference.

def prepare_exposure_source(
    frame,
    source_type,
):

    if frame.empty:

        return pd.DataFrame(
            columns=[
                "CONTRACT_REFERENCE",
                "RFO_DATE",
                "CURRENCY",
                "NOMINAL_AMOUNT_SOURCE",
                "REPORT_VERSIONID",
                "SOURCE_TYPE",
            ]
        )

    frame = frame.copy()

    frame.columns = [
        column.upper()
        for column in frame.columns
    ]

    required_source_columns = [
        "CONTRACT_REFERENCE",
        "REPORT_DATEKEY",
        "CURRENCY",
        "NOMINAL_AMOUNT_SOURCE",
        "REPORT_VERSIONID",
    ]

    missing_source_columns = [
        column
        for column in required_source_columns
        if column not in frame.columns
    ]

    if missing_source_columns:
        raise KeyError(
            "{} exposure source is missing required column(s): {}".format(
                source_type,
                ", ".join(
                    missing_source_columns
                ),
            )
        )

    frame[
        "CONTRACT_REFERENCE"
    ] = normalize_id(
        frame[
            "CONTRACT_REFERENCE"
        ]
    )

    frame[
        "RFO_DATE"
    ] = parse_report_date(
        frame[
            "REPORT_DATEKEY"
        ]
    )

    frame[
        "CURRENCY"
    ] = (
        frame[
            "CURRENCY"
        ]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    frame[
        "NOMINAL_AMOUNT_SOURCE"
    ] = pd.to_numeric(
        frame[
            "NOMINAL_AMOUNT_SOURCE"
        ],
        errors="coerce",
    )

    frame[
        "REPORT_VERSIONID"
    ] = pd.to_numeric(
        frame[
            "REPORT_VERSIONID"
        ],
        errors="coerce",
    )

    frame[
        "SOURCE_TYPE"
    ] = source_type

    frame = (
        frame[
            frame[
                "CONTRACT_REFERENCE"
            ].notna()
            & frame[
                "RFO_DATE"
            ].notna()
        ]
        .copy()
    )

    frame = (
        frame
        .sort_values(
            [
                "CONTRACT_REFERENCE",
                "RFO_DATE",
                "REPORT_VERSIONID",
            ]
        )
        .drop_duplicates(
            subset=[
                "CONTRACT_REFERENCE",
            ],
            keep="last",
        )
        .copy()
    )

    return frame[
        [
            "CONTRACT_REFERENCE",
            "RFO_DATE",
            "CURRENCY",
            "NOMINAL_AMOUNT_SOURCE",
            "REPORT_VERSIONID",
            "SOURCE_TYPE",
        ]
    ]


loandepo_current_df = prepare_exposure_source(
    loandepo_df,
    "LOANDEPO",
)

account_current_df = prepare_exposure_source(
    account_df,
    "ACCOUNT",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Combine exposure sources
# Bring the loan and account exposure extracts together for contract-level matching.

exposure_source_df = pd.concat(
    [
        loandepo_current_df,
        account_current_df,
    ],
    ignore_index=True,
)

if exposure_source_df.empty:
    raise ValueError(
        "No RFO nominal exposure is available "
        "for the implementation month."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate cross-source contract identifiers
# Stop if a contract reference appears in both RFO exposure sources.

cross_source_count = (
    exposure_source_df
    .groupby(
        "CONTRACT_REFERENCE"
    )[
        "SOURCE_TYPE"
    ]
    .nunique()
)

cross_source_duplicates = int(
    (
        cross_source_count > 1
    ).sum()
)

if cross_source_duplicates > 0:

    problematic_contracts = (
        cross_source_count[
            cross_source_count > 1
        ]
        .index
    )

    print()
    print("RFO CROSS-SOURCE DUPLICATE SAMPLE")
    print("---------------------------------")

    print(
        exposure_source_df[
            exposure_source_df[
                "CONTRACT_REFERENCE"
            ].isin(
                problematic_contracts
            )
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    raise ValueError(
        "{} RFO contract references are present "
        "in both LOANDEPO and ACCOUNT.".format(
            cross_source_duplicates
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read FX rates
# Retrieve available FX observations for the reporting month through the reference date.

fx_query = """
SELECT
    fx.ccy_code,
    fx.value_date,
    fx.report_datekey,
    fx.rate,
    fx.report_versionid
FROM DTM_RFO_PLY.MKT_FX_INPUT_VW fx
WHERE fx.report_datekey >= {period_start}
  AND fx.report_datekey <= {reference_date}
""".format(
    period_start=PERIOD_START_KEY,
    reference_date=REFERENCE_DATE_KEY,
)

fx_df = execute_query(
    DB_CONNECTION,
    fx_query,
)

if fx_df.empty:
    raise ValueError(
        "No FX rates are available for the implementation month."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Prepare FX rates
# Standardise currency codes, dates, rates and source versions before using them in
# conversions.

fx_df.columns = [
    column.upper()
    for column in fx_df.columns
]

required_fx_columns = [
    "CCY_CODE",
    "VALUE_DATE",
    "REPORT_DATEKEY",
    "RATE",
    "REPORT_VERSIONID",
]

missing_fx_columns = [
    column
    for column in required_fx_columns
    if column not in fx_df.columns
]

if missing_fx_columns:
    raise KeyError(
        "FX source is missing required column(s): {}".format(
            ", ".join(
                missing_fx_columns
            )
        )
    )

fx_df[
    "CCY_CODE"
] = (
    fx_df[
        "CCY_CODE"
    ]
    .astype("string")
    .str.strip()
    .str.upper()
)

fx_df[
    "FX_DATE"
] = parse_report_date(
    fx_df[
        "REPORT_DATEKEY"
    ]
)

fx_df[
    "VALUE_DATE"
] = normalize_date(
    fx_df[
        "VALUE_DATE"
    ]
)

fx_df[
    "RATE"
] = pd.to_numeric(
    fx_df[
        "RATE"
    ],
    errors="coerce",
)

fx_df[
    "REPORT_VERSIONID"
] = pd.to_numeric(
    fx_df[
        "REPORT_VERSIONID"
    ],
    errors="coerce",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep valid same-day FX rates
# Retain FX rates whose reporting and value dates match, keeping the latest version per
# currency and date.

fx_df = (
    fx_df[
        fx_df[
            "FX_DATE"
        ].notna()
        & fx_df[
            "FX_DATE"
        ].eq(
            fx_df[
                "VALUE_DATE"
            ]
        )
    ]
    .copy()
)

fx_df = (
    fx_df
    .sort_values(
        [
            "CCY_CODE",
            "FX_DATE",
            "REPORT_VERSIONID",
        ]
    )
    .drop_duplicates(
        subset=[
            "CCY_CODE",
            "FX_DATE",
        ],
        keep="last",
    )
    .copy()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Convert nominal exposure to EUR
# Match each contract exposure with the FX rate for its source reporting date and currency.
# EUR is treated as 1 when needed.

exposure_source_df = (
    exposure_source_df
    .merge(
        fx_df[
            [
                "CCY_CODE",
                "FX_DATE",
                "RATE",
            ]
        ],
        how="left",
        left_on=[
            "CURRENCY",
            "RFO_DATE",
        ],
        right_on=[
            "CCY_CODE",
            "FX_DATE",
        ],
        validate="m:1",
    )
)

exposure_source_df.loc[
    exposure_source_df[
        "CURRENCY"
    ].eq(
        "EUR"
    )
    & exposure_source_df[
        "RATE"
    ].isna(),
    "RATE",
] = 1.0


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate FX coverage
# Reject exposures for which a required currency conversion rate is missing.

missing_fx_df = (
    exposure_source_df[
        exposure_source_df[
            "NOMINAL_AMOUNT_SOURCE"
        ].notna()
        & exposure_source_df[
            "RATE"
        ].isna()
    ]
    .copy()
)

if not missing_fx_df.empty:

    print()
    print("MISSING FX RATE QA")
    print("------------------")

    print(
        missing_fx_df[
            "CURRENCY"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    raise ValueError(
        "Missing FX rate for one or more non-EUR exposures."
    )


exposure_source_df[
    "NOMINAL_AMOUNT_EUR"
] = (
    pd.to_numeric(
        exposure_source_df[
            "NOMINAL_AMOUNT_SOURCE"
        ],
        errors="coerce",
    )
    *
    pd.to_numeric(
        exposure_source_df[
            "RATE"
        ],
        errors="coerce",
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Merge contract classification with exposure
# Connect classified entity contracts to their available RFO nominal exposure and assign
# the branch reference date.

loan_exposure_df = (
    contract_mapping_df
    .merge(
        exposure_source_df[
            [
                "CONTRACT_REFERENCE",
                "RFO_DATE",
                "CURRENCY",
                "SOURCE_TYPE",
                "NOMINAL_AMOUNT_EUR",
            ]
        ],
        how="left",
        left_on="CONTRACT_ID_SHORT",
        right_on="CONTRACT_REFERENCE",
        validate="m:1",
    )
)

loan_exposure_df[
    "MAP_DTE"
] = REFERENCE_DATE


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Exposure coverage
# Measure contract matches and keep strictly positive exposures for downstream portfolio
# calculations.

matched_exposure_rows = int(
    loan_exposure_df[
        "NOMINAL_AMOUNT_EUR"
    ].notna().sum()
)

unmatched_exposure_rows = int(
    loan_exposure_df[
        "NOMINAL_AMOUNT_EUR"
    ].isna().sum()
)

if matched_exposure_rows == 0:
    raise ValueError(
        "No loan contract could be matched "
        "to current-month RFO nominal exposure."
    )

loan_exposure_matched_df = (
    loan_exposure_df[
        loan_exposure_df[
            "NOMINAL_AMOUNT_EUR"
        ].notna()
    ]
    .copy()
)

loan_exposure_matched_df[
    "NOMINAL_AMOUNT_EUR"
] = (
    pd.to_numeric(
        loan_exposure_matched_df[
            "NOMINAL_AMOUNT_EUR"
        ],
        errors="coerce",
    )
    .clip(
        lower=0.0
    )
)

loan_exposure_matched_df = (
    loan_exposure_matched_df[
        loan_exposure_matched_df[
            "NOMINAL_AMOUNT_EUR"
        ].gt(
            0
        )
    ]
    .copy()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Build classified exposure components
# Allocate matched exposure into mortgage, non-mortgage, Lombard and ambiguous components.

is_mortgage = (
    loan_exposure_matched_df[
        "PURPOSE_CLASS"
    ].eq(
        "MORTGAGE"
    )
)

is_non_mortgage = (
    loan_exposure_matched_df[
        "PURPOSE_CLASS"
    ].eq(
        "NON_MORTGAGE"
    )
)

is_lombard = (
    loan_exposure_matched_df[
        "PURPOSE_CLASS"
    ].eq(
        "LOMBARD"
    )
)

is_ambiguous = (
    loan_exposure_matched_df[
        "PURPOSE_CLASS"
    ].eq(
        "AMBIGUOUS"
    )
)

loan_exposure_matched_df[
    "_MORTGAGE_EXPOSURE"
] = np.where(
    is_mortgage,
    loan_exposure_matched_df[
        "NOMINAL_AMOUNT_EUR"
    ],
    0.0,
)

loan_exposure_matched_df[
    "_NON_MORTGAGE_EXPOSURE"
] = np.where(
    is_non_mortgage,
    loan_exposure_matched_df[
        "NOMINAL_AMOUNT_EUR"
    ],
    0.0,
)

loan_exposure_matched_df[
    "_LOMBARD_EXPOSURE"
] = np.where(
    is_lombard,
    loan_exposure_matched_df[
        "NOMINAL_AMOUNT_EUR"
    ],
    0.0,
)

loan_exposure_matched_df[
    "_AMBIGUOUS_EXPOSURE"
] = np.where(
    is_ambiguous,
    loan_exposure_matched_df[
        "NOMINAL_AMOUNT_EUR"
    ],
    0.0,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Legacy non-mortgage treatment for QA
# Reproduce the earlier non-mortgage definition only for comparison with the Lombard-
# excluded result.

loan_exposure_matched_df[
    "_LEGACY_NON_MORTGAGE_EXPOSURE"
] = np.where(
    is_non_mortgage
    | is_lombard,
    loan_exposure_matched_df[
        "NOMINAL_AMOUNT_EUR"
    ],
    0.0,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Aggregate exposure to entity level
# Sum classified contract exposures by entity and date.

exposure_df = (
    loan_exposure_matched_df
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
        ],
        as_index=False,
    )
    .agg(
        nominal_amount=(
            "_MORTGAGE_EXPOSURE",
            "sum",
        ),
        non_mortgage_loan_amt=(
            "_NON_MORTGAGE_EXPOSURE",
            "sum",
        ),
        lombard_loan_amt=(
            "_LOMBARD_EXPOSURE",
            "sum",
        ),
        ambiguous_loan_amt_qa=(
            "_AMBIGUOUS_EXPOSURE",
            "sum",
        ),
        legacy_non_mortgage_loan_amt_qa=(
            "_LEGACY_NON_MORTGAGE_EXPOSURE",
            "sum",
        ),
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Calculate non-mortgage share excluding Lombard
# Calculate the non-mortgage proportion using mortgage plus non-mortgage exposure only,
# excluding Lombard.

exposure_df[
    "total_exposure_excl_lombard"
] = (
    exposure_df[
        "nominal_amount"
    ]
    +
    exposure_df[
        "non_mortgage_loan_amt"
    ]
)

exposure_df[
    "non_mortgage_exposure_share"
] = np.where(
    exposure_df[
        "total_exposure_excl_lombard"
    ].gt(
        0
    ),
    (
        exposure_df[
            "non_mortgage_loan_amt"
        ]
        /
        exposure_df[
            "total_exposure_excl_lombard"
        ]
    ),
    np.nan,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Legacy share for QA
# Compare the previous denominator definition with the Lombard-excluded proportion.

legacy_total_exposure = (
    exposure_df[
        "nominal_amount"
    ]
    +
    exposure_df[
        "legacy_non_mortgage_loan_amt_qa"
    ]
)

legacy_share = pd.Series(
    np.where(
        legacy_total_exposure.gt(
            0
        ),
        (
            exposure_df[
                "legacy_non_mortgage_loan_amt_qa"
            ]
            /
            legacy_total_exposure
        ),
        np.nan,
    ),
    index=exposure_df.index,
)

share_difference = (
    legacy_share
    -
    exposure_df[
        "non_mortgage_exposure_share"
    ]
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final QA
# Print coverage, mapping age, exposure classification and duplicate entity-day counts.

duplicate_entity_days = int(
    exposure_df.duplicated(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
        ]
    ).sum()
)

lombard_rows = int(
    is_lombard.sum()
)

lombard_entities = int(
    loan_exposure_matched_df.loc[
        is_lombard,
        "ENTITY_ID",
    ].nunique()
)

lombard_exposure = float(
    exposure_df[
        "lombard_loan_amt"
    ].sum()
)

ambiguous_matched_contracts = int(
    is_ambiguous.sum()
)

ambiguous_exposure = float(
    exposure_df[
        "ambiguous_loan_amt_qa"
    ].sum()
)

affected_entities = int(
    share_difference
    .abs()
    .gt(
        1e-12
    )
    .sum()
)

rfo_min_date = (
    loan_exposure_matched_df[
        "RFO_DATE"
    ].min()
)

rfo_max_date = (
    loan_exposure_matched_df[
        "RFO_DATE"
    ].max()
)


print()
print("LOAN EXPOSURE QA")
print("----------------")

print(
    "Implementation reference date:",
    REFERENCE_DATE_STR,
)

print(
    "Loan mapping date:",
    LOAN_MAPPING_DATE_STR,
)

print(
    "Loan mapping lag days:",
    LOAN_MAPPING_LAG_DAYS,
)

print(
    "RFO exposure date range:",
    "{} to {}".format(
        rfo_min_date.strftime(
            "%Y-%m-%d"
        )
        if pd.notna(
            rfo_min_date
        )
        else "N/A",
        rfo_max_date.strftime(
            "%Y-%m-%d"
        )
        if pd.notna(
            rfo_max_date
        )
        else "N/A",
    ),
)

print(
    "Contract-level mapping rows:",
    "{:,}".format(
        len(
            contract_mapping_df
        )
    ),
)

print(
    "Rows matched to nominal exposure:",
    "{:,}".format(
        matched_exposure_rows
    ),
)

print(
    "Rows without nominal exposure:",
    "{:,}".format(
        unmatched_exposure_rows
    ),
)

print(
    "Matched positive exposure rows:",
    "{:,}".format(
        len(
            loan_exposure_matched_df
        )
    ),
)

print(
    "Mortgage contracts:",
    "{:,}".format(
        int(
            is_mortgage.sum()
        )
    ),
)

print(
    "Non-mortgage contracts excluding Lombard:",
    "{:,}".format(
        int(
            is_non_mortgage.sum()
        )
    ),
)

print(
    "Lombard contracts:",
    "{:,}".format(
        lombard_rows
    ),
)

print(
    "Entities with Lombard exposure:",
    "{:,}".format(
        lombard_entities
    ),
)

print(
    "Lombard exposure EUR:",
    "{:,.2f}".format(
        lombard_exposure
    ),
)

print(
    "Ambiguous matched contracts:",
    "{:,}".format(
        ambiguous_matched_contracts
    ),
)

print(
    "Ambiguous exposure excluded from share EUR:",
    "{:,.2f}".format(
        ambiguous_exposure
    ),
)

print(
    "Entities whose non-mortgage share changes:",
    "{:,}".format(
        affected_entities
    ),
)

print(
    "Duplicate ENTITY_ID + MAP_DTE:",
    duplicate_entity_days,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate Lombard exclusion
# Confirm no Lombard exposure enters the non-mortgage exposure component.

lombard_leakage = (
    loan_exposure_matched_df.loc[
        is_lombard,
        "_NON_MORTGAGE_EXPOSURE",
    ]
    .fillna(
        0.0
    )
    .abs()
    .gt(
        0
    )
    .any()
)

if lombard_leakage:
    raise ValueError(
        "Lombard exposure is still included "
        "in non-mortgage exposure."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate output
# Require one aggregated exposure row per entity and reference date.

if duplicate_entity_days > 0:
    raise ValueError(
        "Loan_Exposure_Current_Previous contains duplicate "
        "ENTITY_ID + MAP_DTE rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Prepare output
# Keep the entity-level exposure columns consumed by portfolio and indicator calculations.

output_df = (
    exposure_df[
        [
            "ENTITY_ID",
            "MAP_DTE",
            "nominal_amount",
            "non_mortgage_loan_amt",
            "lombard_loan_amt",
            "total_exposure_excl_lombard",
            "non_mortgage_exposure_share",
        ]
    ]
    .copy()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Write the previous-branch loan exposure dataset.

dataiku.Dataset(
    "Loan_Exposure_Current_Previous"
).write_with_schema(
    output_df
)