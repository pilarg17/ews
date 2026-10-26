# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import time

import dataiku
import numpy as np
import pandas as pd
from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration
# Define mortgage purpose codes and the exposure and LTV conditions for the Retail portfolios.

DB_CONNECTION = "PRD_BDWH"

MORTGAGE_PURPOSE_CODES = {
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
}

RRE_MIN_MORTGAGE_EXPOSURE = 50000.0
RRE_MIN_LTV = 0.60
MIN_EXPOSURE = 5000.0


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helper functions
# Use consistent identifiers and dates, and check the expected source columns.


def execute_query(query):
    start_time = time.time()

    result = SQLExecutor2(
        connection=DB_CONNECTION
    ).query_to_df(
        query
    )

    print(
        "Query completed in {:.1f} seconds".format(
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


def parse_rfo_date(series):
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

    parsed_key = pd.to_datetime(
        text,
        format="%Y%m%d",
        errors="coerce",
    )

    parsed_generic = normalize_date(
        series
    )

    return parsed_key.fillna(
        parsed_generic
    )


def resolve_column(frame, dataset_name, candidates):
    for column in candidates:
        if column in frame.columns:
            return column

    raise KeyError(
        "{} is missing required column. Expected one of: {}".format(
            dataset_name,
            ", ".join(
                candidates
            ),
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation reference date
# Build the portfolio using the date selected for this snapshot.

reference_df = dataiku.Dataset(
    "Implementation_Reference_Date_previous"
).get_dataframe()

if reference_df.empty:
    raise ValueError(
        "Implementation_Reference_Date_previous is empty."
    )

if len(reference_df) != 1:
    raise ValueError(
        "Implementation_Reference_Date_previous must contain exactly one row."
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

REFERENCE_DATE_STR = REFERENCE_DATE.strftime(
    "%Y-%m-%d"
)


REFERENCE_DATE_INT = int(
    REFERENCE_DATE.strftime(
        "%Y%m%d"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current Retail population
# Start with entities recorded on the reference date.

entity = dataiku.Dataset(
    "Entity_Information_Previous"
).get_dataframe()

required_entity_columns = [
    "ENTITY_ID",
    "MAP_DTE",
]

missing_entity_columns = [
    column
    for column in required_entity_columns
    if column not in entity.columns
]

if missing_entity_columns:
    raise KeyError(
        "Entity_Information is missing required column(s): {}".format(
            ", ".join(
                missing_entity_columns
            )
        )
    )

entity = entity[
    required_entity_columns
].copy()

entity[
    "ENTITY_ID"
] = normalize_id(
    entity[
        "ENTITY_ID"
    ]
)

entity[
    "MAP_DTE"
] = normalize_date(
    entity[
        "MAP_DTE"
    ]
)

entity = entity.loc[
    entity[
        "MAP_DTE"
    ].eq(
        REFERENCE_DATE
    )
].copy()

entity = (
    entity
    .dropna(
        subset=[
            "ENTITY_ID",
        ]
    )
    .drop_duplicates(
        subset=[
            "ENTITY_ID",
        ]
    )
    .reset_index(
        drop=True
    )
)

if entity.empty:
    raise ValueError(
        "Entity_Information has no rows for reference date {}.".format(
            REFERENCE_DATE_STR
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current loan-level population
# Keep loans linked to those entities on the selected date.

loan_level = dataiku.Dataset(
    "vars_loan_level_Previous"
).get_dataframe()

required_loan_columns = [
    "INSTRUMENT_ID",
    "ENTITY_ID",
    "MAP_DTE",
    "CONTRACT_INTERNAL_PURPOSE_CODE",
]

missing_loan_columns = [
    column
    for column in required_loan_columns
    if column not in loan_level.columns
]

if missing_loan_columns:
    raise KeyError(
        "vars_loan_level is missing required column(s): {}".format(
            ", ".join(
                missing_loan_columns
            )
        )
    )

loan_level = loan_level[
    required_loan_columns
].copy()

loan_level[
    "INSTRUMENT_ID"
] = normalize_id(
    loan_level[
        "INSTRUMENT_ID"
    ]
)

loan_level[
    "ENTITY_ID"
] = normalize_id(
    loan_level[
        "ENTITY_ID"
    ]
)

loan_level[
    "MAP_DTE"
] = normalize_date(
    loan_level[
        "MAP_DTE"
    ]
)

loan_level[
    "CONTRACT_INTERNAL_PURPOSE_CODE"
] = normalize_id(
    loan_level[
        "CONTRACT_INTERNAL_PURPOSE_CODE"
    ]
)

loan_level = loan_level.loc[
    loan_level[
        "MAP_DTE"
    ].eq(
        REFERENCE_DATE
    )
].copy()

loan_level = loan_level.dropna(
    subset=[
        "INSTRUMENT_ID",
        "ENTITY_ID",
    ]
).copy()

if loan_level.empty:
    raise ValueError(
        "vars_loan_level has no rows for reference date {}.".format(
            REFERENCE_DATE_STR
        )
    )

duplicate_loan_rows = int(
    loan_level.duplicated(
        subset=[
            "INSTRUMENT_ID",
            "ENTITY_ID",
            "MAP_DTE",
        ]
    ).sum()
)

if duplicate_loan_rows > 0:
    raise ValueError(
        "vars_loan_level contains duplicate "
        "INSTRUMENT_ID + ENTITY_ID + MAP_DTE rows."
    )

loan_level = loan_level.merge(
    entity[
        [
            "ENTITY_ID",
        ]
    ],
    how="inner",
    on="ENTITY_ID",
    validate="m:1",
)

if loan_level.empty:
    raise ValueError(
        "No loan-level records remain after applying the Retail population."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Exclude Lombard instruments
# Exclude instruments classified as Lombard products.


lombard_query = """
SELECT DISTINCT
    TRIM(
        CAST(
            instrument_id AS VARCHAR2(100)
        )
    ) AS INSTRUMENT_ID
FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_VW
WHERE TRUNC(TEK_STRT_DTE) <= DATE '{reference_date}'
  AND DATE '{reference_date}' < TRUNC(
        NVL(TEK_END_DTE, DATE '9999-12-31')
      )
  AND UPPER(TRIM(INTERNAL_PRODUCT_TYPE)) = 'INVEST ACCOUNT LOMBARD'
""".format(
    reference_date=REFERENCE_DATE_STR
)

lombard = execute_query(
    lombard_query
)

if lombard.empty:
    lombard_instruments = set()
else:
    lombard.columns = [
        column.upper()
        for column in lombard.columns
    ]

    lombard[
        "INSTRUMENT_ID"
    ] = normalize_id(
        lombard[
            "INSTRUMENT_ID"
        ]
    )

    lombard_instruments = set(
        lombard[
            "INSTRUMENT_ID"
        ]
        .dropna()
        .tolist()
    )

lombard_mask = loan_level[
    "INSTRUMENT_ID"
].isin(
    lombard_instruments
)

excluded_lombard_instruments = int(
    lombard_mask.sum()
)

portfolio_instruments = loan_level.loc[
    ~lombard_mask
].copy()

if portfolio_instruments.empty:
    raise ValueError(
        "No RRE or Other Retail instruments remain after Lombard exclusion."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current nominal exposure
# Use the latest common RFO loan and account snapshot available by the reference date.


rfo_date_query = """
SELECT MAX(loan_dates.report_datekey) AS RFO_SOURCE_DATE
FROM (
    SELECT DISTINCT report_datekey
    FROM DTM_RFO_PLY.LOANDEPO_VW
    WHERE report_datekey <= {reference_date}
) loan_dates
INNER JOIN (
    SELECT DISTINCT report_datekey
    FROM DTM_RFO_PLY.ACCOUNT_VW
    WHERE report_datekey <= {reference_date}
) account_dates
    ON loan_dates.report_datekey = account_dates.report_datekey
""".format(
    reference_date=REFERENCE_DATE_INT,
)

rfo_date_df = execute_query(
    rfo_date_query
)

if (
    rfo_date_df.empty
    or pd.isna(
        rfo_date_df.iloc[0][
            "RFO_SOURCE_DATE"
        ]
    )
):
    raise ValueError(
        "No common RFO LOANDEPO / ACCOUNT snapshot is available "
        "on or before {}.".format(
            REFERENCE_DATE_STR
        )
    )

RFO_SOURCE_DATE_INT = int(
    pd.to_numeric(
        rfo_date_df.iloc[0][
            "RFO_SOURCE_DATE"
        ],
        errors="raise",
    )
)

RFO_SOURCE_DATE = pd.to_datetime(
    str(
        RFO_SOURCE_DATE_INT
    ),
    format="%Y%m%d",
    errors="raise",
)

RFO_SOURCE_DATE_STR = RFO_SOURCE_DATE.strftime(
    "%Y-%m-%d"
)

print(
    "RFO nominal exposure source date:",
    RFO_SOURCE_DATE_STR,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### RFO nominal exposure
# Retrieve the outstanding balances of loans and accounts from that snapshot.

exposure_query = """
SELECT
    instrument_id,
    report_datekey,
    currency,
    nominal_amount,
    report_versionid,
    source_type
FROM (
    SELECT
        ld.contract_reference AS instrument_id,
        ld.report_datekey AS report_datekey,
        ld.currency AS currency,
        ld.outstanding AS nominal_amount,
        ld.report_versionid AS report_versionid,
        'LOANDEPO' AS source_type
    FROM DTM_RFO_PLY.LOANDEPO_VW ld
    WHERE ld.report_datekey = {rfo_source_date}
      AND (ld.accruals + ld.outstanding) > 0

    UNION ALL

    SELECT
        acc.account_reference AS instrument_id,
        acc.report_datekey AS report_datekey,
        acc.currency AS currency,
        acc.balance AS nominal_amount,
        acc.report_versionid AS report_versionid,
        'ACCOUNT' AS source_type
    FROM DTM_RFO_PLY.ACCOUNT_VW acc
    WHERE acc.report_datekey = {rfo_source_date}
      AND (acc.accruals + acc.balance) > 0
)
""".format(
    rfo_source_date=RFO_SOURCE_DATE_INT,
)

exposure = execute_query(
    exposure_query
)

if exposure.empty:
    raise ValueError(
        "No RFO nominal exposure was returned for source date {}.".format(
            RFO_SOURCE_DATE_STR
        )
    )

exposure.columns = [
    column.upper()
    for column in exposure.columns
]

exposure[
    "INSTRUMENT_ID"
] = normalize_id(
    exposure[
        "INSTRUMENT_ID"
    ]
)

exposure[
    "RFO_DATE"
] = parse_rfo_date(
    exposure[
        "REPORT_DATEKEY"
    ]
)

exposure[
    "NOMINAL_AMOUNT"
] = pd.to_numeric(
    exposure[
        "NOMINAL_AMOUNT"
    ],
    errors="coerce",
)

exposure[
    "REPORT_VERSIONID"
] = pd.to_numeric(
    exposure[
        "REPORT_VERSIONID"
    ],
    errors="coerce",
)

exposure[
    "CURRENCY"
] = (
    exposure[
        "CURRENCY"
    ]
    .astype("string")
    .str.strip()
    .str.upper()
)

exposure = exposure.dropna(
    subset=[
        "INSTRUMENT_ID",
        "RFO_DATE",
    ]
).copy()

exposure = (
    exposure
    .sort_values(
        [
            "INSTRUMENT_ID",
            "RFO_DATE",
            "SOURCE_TYPE",
            "REPORT_VERSIONID",
        ]
    )
    .drop_duplicates(
        subset=[
            "INSTRUMENT_ID",
            "RFO_DATE",
            "SOURCE_TYPE",
        ],
        keep="last",
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### FX conversion
# Convert amounts to EUR using the matching FX rate and select one exposure per instrument.

fx_query = """
SELECT
    ccy_code,
    value_date,
    report_datekey,
    rate,
    report_versionid
FROM DTM_RFO_PLY.MKT_FX_INPUT_VW
WHERE report_datekey = {rfo_source_date}
""".format(
    rfo_source_date=RFO_SOURCE_DATE_INT,
)

fx = execute_query(
    fx_query
)

if fx.empty:
    exposure[
        "RATE"
    ] = np.nan
else:
    fx.columns = [
        column.upper()
        for column in fx.columns
    ]

    fx[
        "FX_DATE"
    ] = parse_rfo_date(
        fx[
            "REPORT_DATEKEY"
        ]
    )

    fx[
        "VALUE_DATE"
    ] = normalize_date(
        fx[
            "VALUE_DATE"
        ]
    )

    fx[
        "RATE"
    ] = pd.to_numeric(
        fx[
            "RATE"
        ],
        errors="coerce",
    )

    fx[
        "REPORT_VERSIONID"
    ] = pd.to_numeric(
        fx[
            "REPORT_VERSIONID"
        ],
        errors="coerce",
    )

    fx[
        "CCY_CODE"
    ] = (
        fx[
            "CCY_CODE"
        ]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    fx = fx.loc[
        fx[
            "FX_DATE"
        ].eq(
            fx[
                "VALUE_DATE"
            ]
        )
    ].copy()

    fx = (
        fx
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
    )

    exposure = exposure.merge(
        fx[
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

exposure.loc[
    exposure[
        "CURRENCY"
    ].eq(
        "EUR"
    )
    & exposure[
        "RATE"
    ].isna(),
    "RATE",
] = 1.0

exposure[
    "NOMINAL_AMOUNT_EUR"
] = (
    exposure[
        "NOMINAL_AMOUNT"
    ]
    * pd.to_numeric(
        exposure[
            "RATE"
        ],
        errors="coerce",
    )
)

exposure[
    "SOURCE_PRIORITY"
] = np.where(
    exposure[
        "SOURCE_TYPE"
    ].eq(
        "LOANDEPO"
    ),
    0,
    1,
)

exposure_current = (
    exposure
    .sort_values(
        [
            "INSTRUMENT_ID",
            "RFO_DATE",
            "SOURCE_PRIORITY",
            "REPORT_VERSIONID",
        ]
    )
    .drop_duplicates(
        subset=[
            "INSTRUMENT_ID",
            "RFO_DATE",
        ],
        keep="first",
    )[
        [
            "INSTRUMENT_ID",
            "RFO_DATE",
            "NOMINAL_AMOUNT_EUR",
        ]
    ]
    .copy()
)

portfolio_instruments = portfolio_instruments.merge(
    exposure_current,
    how="left",
    on="INSTRUMENT_ID",
    validate="m:1",
)

portfolio_instruments[
    "NOMINAL_AMOUNT_EUR"
] = pd.to_numeric(
    portfolio_instruments[
        "NOMINAL_AMOUNT_EUR"
    ],
    errors="coerce",
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Classify RRE and Other Retail instruments
# Distinguish mortgage-related loans from other Retail instruments using their purpose codes.

portfolio_instruments[
    "IS_RRE_INSTRUMENT"
] = portfolio_instruments[
    "CONTRACT_INTERNAL_PURPOSE_CODE"
].isin(
    MORTGAGE_PURPOSE_CODES
)

portfolio_instruments[
    "IS_OTHER_RETAIL_INSTRUMENT"
] = ~portfolio_instruments[
    "IS_RRE_INSTRUMENT"
]

portfolio_instruments[
    "RRE_NOMINAL_EXPOSURE"
] = np.where(
    portfolio_instruments[
        "IS_RRE_INSTRUMENT"
    ],
    portfolio_instruments[
        "NOMINAL_AMOUNT_EUR"
    ].fillna(
        0.0
    ),
    0.0,
)

portfolio_instruments[
    "OTHER_RETAIL_NOMINAL_EXPOSURE"
] = np.where(
    portfolio_instruments[
        "IS_OTHER_RETAIL_INSTRUMENT"
    ],
    portfolio_instruments[
        "NOMINAL_AMOUNT_EUR"
    ].fillna(
        0.0
    ),
    0.0,
)

portfolio = (
    portfolio_instruments
    .groupby(
        "ENTITY_ID",
        as_index=False,
    )
    .agg(
        RRE_NOMINAL_EXPOSURE=(
            "RRE_NOMINAL_EXPOSURE",
            "sum",
        ),
        OTHER_RETAIL_NOMINAL_EXPOSURE=(
            "OTHER_RETAIL_NOMINAL_EXPOSURE",
            "sum",
        ),
        RRE_INSTRUMENT_COUNT=(
            "IS_RRE_INSTRUMENT",
            "sum",
        ),
        OTHER_RETAIL_INSTRUMENT_COUNT=(
            "IS_OTHER_RETAIL_INSTRUMENT",
            "sum",
        ),
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### RRE LTV
# Calculate LTV by dividing the total relevant outstanding amount by allocated collateral.

ltv_source = dataiku.Dataset(
    "loan_RRE_ltv_previous"
).get_dataframe()

if ltv_source.empty:
    raise ValueError(
        "loan_RRE_ltv_previous is empty."
    )

ltv_date_column = resolve_column(
    ltv_source,
    "loan_RRE_ltv_previous",
    [
        "REFERENCE_DATE",
        "MAP_DTE",
        "reference_date",
    ],
)

ltv_collateral_column = resolve_column(
    ltv_source,
    "loan_RRE_ltv_previous",
    [
        "ALLOCATEDCOLLATERALVALUE",
        "ALLOCATED_COLLATERAL_VALUE",
    ],
)

required_ltv_columns = [
    "ENTITY_ID",
    ltv_date_column,
    "OUTSTANDING_NOMINAL_AMOUNT",
    ltv_collateral_column,
]

missing_ltv_columns = [
    column
    for column in required_ltv_columns
    if column not in ltv_source.columns
]

if missing_ltv_columns:
    raise KeyError(
        "loan_RRE_ltv_previous is missing required column(s): {}".format(
            ", ".join(
                missing_ltv_columns
            )
        )
    )

ltv_source = ltv_source[
    required_ltv_columns
].copy()

ltv_source = ltv_source.rename(
    columns={
        ltv_date_column: "MAP_DTE",
        ltv_collateral_column: "ALLOCATEDCOLLATERALVALUE",
    }
)

ltv_source[
    "ENTITY_ID"
] = normalize_id(
    ltv_source[
        "ENTITY_ID"
    ]
)

ltv_source[
    "MAP_DTE"
] = normalize_date(
    ltv_source[
        "MAP_DTE"
    ]
)

ltv_source = ltv_source.loc[
    ltv_source[
        "MAP_DTE"
    ].eq(
        REFERENCE_DATE
    )
].copy()

if ltv_source.empty:
    raise ValueError(
        "loan_RRE_ltv_previous has no rows for reference date {}.".format(
            REFERENCE_DATE_STR
        )
    )

for column in [
    "OUTSTANDING_NOMINAL_AMOUNT",
    "ALLOCATEDCOLLATERALVALUE",
]:
    ltv_source[
        column
    ] = pd.to_numeric(
        ltv_source[
            column
        ],
        errors="coerce",
    )

ltv_entity = (
    ltv_source
    .dropna(
        subset=[
            "ENTITY_ID",
        ]
    )
    .groupby(
        "ENTITY_ID",
        as_index=False,
    )
    .agg(
        RRE_OUTSTANDING_NOMINAL_AMOUNT=(
            "OUTSTANDING_NOMINAL_AMOUNT",
            "sum",
        ),
        RRE_ALLOCATED_COLLATERAL_VALUE=(
            "ALLOCATEDCOLLATERALVALUE",
            "sum",
        ),
    )
)

ltv_entity[
    "RRE_LTV"
] = np.where(
    ltv_entity[
        "RRE_ALLOCATED_COLLATERAL_VALUE"
    ].gt(
        0
    ),
    (
        ltv_entity[
            "RRE_OUTSTANDING_NOMINAL_AMOUNT"
        ]
        / ltv_entity[
            "RRE_ALLOCATED_COLLATERAL_VALUE"
        ]
    ),
    np.nan,
)

portfolio = portfolio.merge(
    ltv_entity[
        [
            "ENTITY_ID",
            "RRE_LTV",
        ]
    ],
    how="left",
    on="ENTITY_ID",
    validate="1:1",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Apply portfolio thresholds
# Apply the RRE minimum exposure and mortgage/LTV conditions, and the Other Retail exposure minimum.

portfolio[
    "RRE_FLAG"
] = (
    portfolio[
        "RRE_NOMINAL_EXPOSURE"
    ].gt(
        MIN_EXPOSURE
    )
    & (
        portfolio[
            "RRE_NOMINAL_EXPOSURE"
        ].gt(
            RRE_MIN_MORTGAGE_EXPOSURE
        )
        | portfolio[
            "RRE_LTV"
        ].gt(
            RRE_MIN_LTV
        )
    )
).astype(
    "int8"
)

portfolio[
    "OTHER_RETAIL_FLAG"
] = (
    portfolio[
        "OTHER_RETAIL_NOMINAL_EXPOSURE"
    ].ge(
        MIN_EXPOSURE
    )
).astype(
    "int8"
)

entities_before_scope = int(
    portfolio[
        "ENTITY_ID"
    ].nunique()
)

portfolio = portfolio.loc[
    portfolio[
        "RRE_FLAG"
    ].eq(
        1
    )
    | portfolio[
        "OTHER_RETAIL_FLAG"
    ].eq(
        1
    )
].copy()

entities_after_scope = int(
    portfolio[
        "ENTITY_ID"
    ].nunique()
)

portfolio[
    "MAP_DTE"
] = REFERENCE_DATE

for column in [
    "RRE_INSTRUMENT_COUNT",
    "OTHER_RETAIL_INSTRUMENT_COUNT",
]:
    portfolio[
        column
    ] = (
        portfolio[
            column
        ]
        .fillna(
            0
        )
        .astype(
            "int32"
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Quality controls
# Check excluded loans, missing exposure or FX, and the size and overlap of both portfolios.

if portfolio[
    "ENTITY_ID"
].duplicated().any():
    raise ValueError(
        "Retail_Portfolio_Daily contains duplicate ENTITY_ID rows."
    )

rre_entities = int(
    portfolio[
        "RRE_FLAG"
    ].sum()
)

other_retail_entities = int(
    portfolio[
        "OTHER_RETAIL_FLAG"
    ].sum()
)

entities_in_both = int(
    (
        portfolio[
            "RRE_FLAG"
        ].eq(
            1
        )
        & portfolio[
            "OTHER_RETAIL_FLAG"
        ].eq(
            1
        )
    ).sum()
)

rre_only = int(
    (
        portfolio[
            "RRE_FLAG"
        ].eq(
            1
        )
        & portfolio[
            "OTHER_RETAIL_FLAG"
        ].eq(
            0
        )
    ).sum()
)

other_retail_only = int(
    (
        portfolio[
            "RRE_FLAG"
        ].eq(
            0
        )
        & portfolio[
            "OTHER_RETAIL_FLAG"
        ].eq(
            1
        )
    ).sum()
)

matched_exposure_rows = int(
    portfolio_instruments[
        "NOMINAL_AMOUNT_EUR"
    ].notna().sum()
)

missing_exposure_rows = int(
    portfolio_instruments[
        "NOMINAL_AMOUNT_EUR"
    ].isna().sum()
)

missing_fx_rows = int(
    exposure[
        "NOMINAL_AMOUNT"
    ].notna()
    .mul(
        exposure[
            "RATE"
        ].isna()
    )
    .sum()
)

rre_ltv_available = int(
    portfolio[
        "RRE_LTV"
    ].notna().sum()
)

print()
print("=" * 100)
print("RETAIL PORTFOLIO DAILY")
print("=" * 100)

print(
    "Reference date                       :",
    REFERENCE_DATE_STR,
)
print(
    "Retail entities                      :",
    "{:,}".format(
        entity[
            "ENTITY_ID"
        ].nunique()
    ),
)
print(
    "Loan-level rows                      :",
    "{:,}".format(
        len(
            loan_level
        )
    ),
)
print(
    "Excluded Lombard instruments         :",
    "{:,}".format(
        excluded_lombard_instruments
    ),
)
print(
    "Rows matched to nominal exposure     :",
    "{:,}".format(
        matched_exposure_rows
    ),
)
print(
    "Rows without nominal exposure        :",
    "{:,}".format(
        missing_exposure_rows
    ),
)
print(
    "RFO exposure rows missing FX         :",
    "{:,}".format(
        missing_fx_rows
    ),
)

print()
print("Portfolio thresholds")
print("--------------------")
print(
    "RRE minimum mortgage exposure (EUR)  :",
    "{:,.0f}".format(
        RRE_MIN_MORTGAGE_EXPOSURE
    ),
)
print(
    "RRE minimum LTV                      :",
    "{:.0%}".format(
        RRE_MIN_LTV
    ),
)
print(
    "Minimum exposure (EUR)               :",
    "{:,.0f}".format(
        MIN_EXPOSURE
    ),
)

print()
print("Scope QA")
print("--------")
print(
    "Entities before materiality scope    :",
    "{:,}".format(
        entities_before_scope
    ),
)
print(
    "Entities after materiality scope     :",
    "{:,}".format(
        entities_after_scope
    ),
)
print(
    "Entities removed by materiality      :",
    "{:,}".format(
        entities_before_scope
        - entities_after_scope
    ),
)
print(
    "Entities with RRE LTV available      :",
    "{:,}".format(
        rre_ltv_available
    ),
)

print()
print("Final portfolio population")
print("--------------------------")
print(
    "Entities in RRE                      :",
    "{:,}".format(
        rre_entities
    ),
)
print(
    "Entities in Other Retail             :",
    "{:,}".format(
        other_retail_entities
    ),
)
print(
    "RRE-only entities                    :",
    "{:,}".format(
        rre_only
    ),
)
print(
    "Other Retail-only entities           :",
    "{:,}".format(
        other_retail_only
    ),
)
print(
    "Entities in both portfolios          :",
    "{:,}".format(
        entities_in_both
    ),
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Prepare output
# Keep only the entity IDs, dates and portfolio membership flags needed downstream.

portfolio = portfolio[
    [
        "ENTITY_ID",
        "MAP_DTE",
        "RRE_FLAG",
        "OTHER_RETAIL_FLAG",
        "RRE_INSTRUMENT_COUNT",
        "OTHER_RETAIL_INSTRUMENT_COUNT",
    ]
].copy()

portfolio = (
    portfolio
    .sort_values(
        "ENTITY_ID"
    )
    .reset_index(
        drop=True
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the two Retail portfolio flags for the selected snapshot.

dataiku.Dataset(
    "Retail_Portfolio_Daily_Previous"
).write_with_schema(
    portfolio
)