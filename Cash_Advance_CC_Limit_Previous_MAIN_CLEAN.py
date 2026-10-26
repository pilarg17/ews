# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import numpy as np
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation dates
# Read the previous branch reference and preceding business dates and ensure they are
# ordered.

implementation_dates_df = dataiku.Dataset(
    "Implementation_Reference_Date_previous"
).get_dataframe()

if implementation_dates_df.empty:
    raise ValueError("Implementation_Reference_Date_previous is empty.")

required_date_columns = ["REFERENCE_DATE", "PREVIOUS_DATE"]
missing_date_columns = [
    column
    for column in required_date_columns
    if column not in implementation_dates_df.columns
]

if missing_date_columns:
    raise KeyError(
        "Implementation_Reference_Date_previous is missing required column(s): {}".format(
            ", ".join(missing_date_columns)
        )
    )

if len(implementation_dates_df) != 1:
    raise ValueError(
        "Implementation_Reference_Date_previous must contain exactly one row."
    )

REFERENCE_DATE = (
    pd.to_datetime(
        implementation_dates_df.loc[
            implementation_dates_df.index[0],
            "REFERENCE_DATE",
        ],
        errors="coerce",
        utc=True,
    )
    .tz_localize(None)
    .normalize()
)

PREVIOUS_DATE = (
    pd.to_datetime(
        implementation_dates_df.loc[
            implementation_dates_df.index[0],
            "PREVIOUS_DATE",
        ],
        errors="coerce",
        utc=True,
    )
    .tz_localize(None)
    .normalize()
)

if pd.isna(REFERENCE_DATE):
    raise ValueError("REFERENCE_DATE could not be parsed.")

if pd.isna(PREVIOUS_DATE):
    raise ValueError("PREVIOUS_DATE could not be parsed.")

if PREVIOUS_DATE >= REFERENCE_DATE:
    raise ValueError("Expected PREVIOUS_DATE < REFERENCE_DATE.")

REFERENCE_DATE_STR = REFERENCE_DATE.strftime("%Y-%m-%d")
PREVIOUS_DATE_STR = PREVIOUS_DATE.strftime("%Y-%m-%d")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration
# Retain the distinct 94% RRE and 97% Other Retail utilisation crossing thresholds.

RRE_CASH_ADVANCE_LIMIT_THRESHOLD = 0.94
OTHER_CASH_ADVANCE_LIMIT_THRESHOLD = 0.97


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### SQL source utility
# Resolve the SQL tables behind the card transaction and related Dataiku source datasets.

def get_sql_source(dataset_name):
    dataset = dataiku.Dataset(dataset_name)
    location = dataset.get_location_info()
    info = location.get("info", {})

    schema = info.get("schema") or info.get("schemaName")
    table = info.get("table") or info.get("tableName")

    if not table:
        raise ValueError("{} must be SQL-backed.".format(dataset_name))

    if schema and "." not in table:
        table = "{}.{}".format(schema, table)

    return dataset, table


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Resolve source tables
# Identify card cash-advance transactions, card ownership, card usage and entity mapping
# sources.

transaction_dataset, transaction_table = get_sql_source(
    "CARDS_ATM_TRANSACTIONS_VW"
)
_, card_table = get_sql_source("CARD_REFERENTIAL_VW")
_, usage_table = get_sql_source("CARD_USAGE_VW")
_, entity_customer_table = get_sql_source("ENTITY_CUSTOMER_VW_1")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Cash-advance transaction and card-limit data
# Retrieve reference-day credit-card cash advances and their card-level current and
# previous-business-day balances and limits. Keep the one-transaction-one-card matching
# logic.
#
# Strict daily rule:
#   1) cash advance must occur on REFERENCE_DATE;
#   2) card utilisation must cross the portfolio threshold on REFERENCE_DATE.
#
# Utilisation is based only on outstanding / max(credit limit, temporary limit).
# The cash-advance amount is not added to outstanding.

query = """
WITH filtered_transactions AS (
    SELECT
        t.CARD_ATM_TRANSACTION_ID,
        t.ACCOUNT_ID,
        TRUNC(t.OPERATION_DATE) AS TRANSACTION_DTE,
        t.OPERATION_GROSS_AMOUNT_CTV
    FROM {transaction_table} t
    WHERE TRUNC(t.OPERATION_DATE) = DATE '{reference_date}'
      AND t.SOURCE_TRANSACTION_CODE = 1091
      AND UPPER(TRIM(t.CARD_TYPE)) = 'CREDIT CARD'
),

card_candidates AS (
    SELECT
        t.CARD_ATM_TRANSACTION_ID,
        t.ACCOUNT_ID,
        t.TRANSACTION_DTE,
        t.OPERATION_GROSS_AMOUNT_CTV,
        c.CARD_OWNER_CUSTOMER_ID,
        c.CARD_ID,
        c.CREDIT_LIMIT,
        c.TEMPORARY_CREDIT_LIMIT,
        ROW_NUMBER() OVER (
            PARTITION BY t.CARD_ATM_TRANSACTION_ID
            ORDER BY
                c.TEK_STRT_DTE DESC,
                c.CARD_ID
        ) AS RN
    FROM filtered_transactions t
    INNER JOIN {card_table} c
        ON TRIM(CAST(t.ACCOUNT_ID AS VARCHAR2(100)))
         = TRIM(CAST(c.LINKED_ACCOUNT_ID AS VARCHAR2(100)))
       AND TRUNC(c.TEK_STRT_DTE) <= DATE '{reference_date}'
       AND DATE '{reference_date}' < TRUNC(
            NVL(c.TEK_END_DTE, DATE '9999-12-31')
       )
       AND UPPER(TRIM(c.CARD_TYPE)) = 'CREDIT CARD'
       AND c.CARD_OWNER_CUSTOMER_ID IS NOT NULL
       AND c.CARD_ID IS NOT NULL
),

card_mapping AS (
    SELECT *
    FROM card_candidates
    WHERE RN = 1
),

entity_candidates AS (
    SELECT
        c.CARD_ATM_TRANSACTION_ID,
        c.ACCOUNT_ID,
        c.TRANSACTION_DTE,
        c.OPERATION_GROSS_AMOUNT_CTV,
        c.CARD_OWNER_CUSTOMER_ID,
        c.CARD_ID,
        c.CREDIT_LIMIT,
        c.TEMPORARY_CREDIT_LIMIT,
        ec.ENTITY_ID,
        ROW_NUMBER() OVER (
            PARTITION BY c.CARD_ATM_TRANSACTION_ID
            ORDER BY
                ec.TEK_STRT_DTE DESC,
                ec.ENTITY_ID
        ) AS RN
    FROM card_mapping c
    INNER JOIN {entity_customer_table} ec
        ON TRIM(CAST(c.CARD_OWNER_CUSTOMER_ID AS VARCHAR2(100)))
         = TRIM(CAST(ec.CUSTOMER_ID AS VARCHAR2(100)))
       AND TRUNC(ec.TEK_STRT_DTE) <= DATE '{reference_date}'
       AND DATE '{reference_date}' < TRUNC(
            NVL(ec.TEK_END_DTE, DATE '9999-12-31')
       )
       AND ec.ENTITY_ID IS NOT NULL
),

entity_mapping AS (
    SELECT *
    FROM entity_candidates
    WHERE RN = 1
),

current_usage_candidates AS (
    SELECT
        e.ENTITY_ID,
        e.TRANSACTION_DTE,
        e.CARD_ATM_TRANSACTION_ID,
        e.CARD_ID,
        e.CREDIT_LIMIT,
        e.TEMPORARY_CREDIT_LIMIT,
        e.OPERATION_GROSS_AMOUNT_CTV,
        u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT,
        ROW_NUMBER() OVER (
            PARTITION BY e.CARD_ATM_TRANSACTION_ID
            ORDER BY u.TEK_STRT_DTE DESC NULLS LAST
        ) AS RN
    FROM entity_mapping e
    LEFT JOIN {usage_table} u
        ON TRIM(CAST(e.CARD_ID AS VARCHAR2(100)))
         = TRIM(CAST(u.CARD_ID AS VARCHAR2(100)))
       AND TRUNC(u.TEK_STRT_DTE) <= DATE '{reference_date}'
       AND DATE '{reference_date}' < TRUNC(
            NVL(u.TEK_END_DTE, DATE '9999-12-31')
       )
),

current_usage AS (
    SELECT *
    FROM current_usage_candidates
    WHERE RN = 1
),

previous_usage_candidates AS (
    SELECT
        c.*,
        u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT
            AS PREVIOUS_PERIOD_TOTAL_OUTSTANDING_AMOUNT,
        ROW_NUMBER() OVER (
            PARTITION BY c.CARD_ATM_TRANSACTION_ID
            ORDER BY u.TEK_STRT_DTE DESC NULLS LAST
        ) AS PREVIOUS_USAGE_RN
    FROM current_usage c
    LEFT JOIN {usage_table} u
        ON TRIM(CAST(c.CARD_ID AS VARCHAR2(100)))
         = TRIM(CAST(u.CARD_ID AS VARCHAR2(100)))
       AND TRUNC(u.TEK_STRT_DTE) <= DATE '{previous_date}'
       AND DATE '{previous_date}' < TRUNC(
            NVL(u.TEK_END_DTE, DATE '9999-12-31')
       )
),

previous_usage AS (
    SELECT *
    FROM previous_usage_candidates
    WHERE PREVIOUS_USAGE_RN = 1
),

previous_limit_candidates AS (
    SELECT
        p.*,
        c.CREDIT_LIMIT AS PREVIOUS_CREDIT_LIMIT,
        c.TEMPORARY_CREDIT_LIMIT AS PREVIOUS_TEMPORARY_CREDIT_LIMIT,
        ROW_NUMBER() OVER (
            PARTITION BY p.CARD_ATM_TRANSACTION_ID
            ORDER BY
                c.TEK_STRT_DTE DESC NULLS LAST,
                c.CARD_ID
        ) AS PREVIOUS_LIMIT_RN
    FROM previous_usage p
    LEFT JOIN {card_table} c
        ON TRIM(CAST(p.CARD_ID AS VARCHAR2(100)))
         = TRIM(CAST(c.CARD_ID AS VARCHAR2(100)))
       AND TRUNC(c.TEK_STRT_DTE) <= DATE '{previous_date}'
       AND DATE '{previous_date}' < TRUNC(
            NVL(c.TEK_END_DTE, DATE '9999-12-31')
       )
       AND UPPER(TRIM(c.CARD_TYPE)) = 'CREDIT CARD'
)

SELECT
    ENTITY_ID,
    DATE '{reference_date}' AS MAP_DTE,
    TRANSACTION_DTE,
    CARD_ATM_TRANSACTION_ID,
    CARD_ID,
    CREDIT_LIMIT,
    TEMPORARY_CREDIT_LIMIT,
    PREVIOUS_CREDIT_LIMIT,
    PREVIOUS_TEMPORARY_CREDIT_LIMIT,
    OPERATION_GROSS_AMOUNT_CTV,
    CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT,
    PREVIOUS_PERIOD_TOTAL_OUTSTANDING_AMOUNT
FROM previous_limit_candidates
WHERE PREVIOUS_LIMIT_RN = 1
""".format(
    reference_date=REFERENCE_DATE_STR,
    previous_date=PREVIOUS_DATE_STR,
    transaction_table=transaction_table,
    card_table=card_table,
    usage_table=usage_table,
    entity_customer_table=entity_customer_table,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Execute query
# Run the source query and report elapsed time.

start_time = time.time()

cash_limit_df = SQLExecutor2(
    dataset=transaction_dataset
).query_to_df(query)

print(
    "Time to perform query: {:.1f} seconds".format(
        time.time() - start_time
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise identifiers and dates
# Normalise mapped entity, card and transaction identifiers and date fields.

for column in ["ENTITY_ID", "CARD_ID", "CARD_ATM_TRANSACTION_ID"]:
    cash_limit_df[column] = (
        cash_limit_df[column]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

for column in ["MAP_DTE", "TRANSACTION_DTE"]:
    cash_limit_df[column] = (
        pd.to_datetime(
            cash_limit_df[column],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise numeric fields
# Convert current and previous limits, outstanding and transaction amounts to numeric
# values.

for column in [
    "CREDIT_LIMIT",
    "TEMPORARY_CREDIT_LIMIT",
    "PREVIOUS_CREDIT_LIMIT",
    "PREVIOUS_TEMPORARY_CREDIT_LIMIT",
    "OPERATION_GROSS_AMOUNT_CTV",
    "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT",
    "PREVIOUS_PERIOD_TOTAL_OUTSTANDING_AMOUNT",
]:
    cash_limit_df[column] = pd.to_numeric(
        cash_limit_df[column],
        errors="coerce",
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Strict same-day timing
# Require a credit-card cash advance on the reference date for a qualifying same-day event.

cash_limit_df["TRANSACTION_LAG_BUSINESS_DAYS"] = np.where(
    cash_limit_df["TRANSACTION_DTE"].eq(REFERENCE_DATE),
    0,
    np.nan,
)

cash_limit_df["TRANSACTION_WITHIN_1D_WINDOW"] = (
    cash_limit_df["TRANSACTION_DTE"]
    .eq(REFERENCE_DATE)
    .astype("int8")
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current and previous-business-day utilisation
# Calculate utilisation as absolute outstanding divided by the higher of regular and
# temporary limits, separately for current and previous business dates.

cash_limit_df["OVERARCHING_CREDIT_LIMIT"] = (
    cash_limit_df[
        ["CREDIT_LIMIT", "TEMPORARY_CREDIT_LIMIT"]
    ]
    .max(axis=1, skipna=True)
)

cash_limit_df["PREVIOUS_OVERARCHING_CREDIT_LIMIT"] = (
    cash_limit_df[
        ["PREVIOUS_CREDIT_LIMIT", "PREVIOUS_TEMPORARY_CREDIT_LIMIT"]
    ]
    .max(axis=1, skipna=True)
)

transaction_amount = cash_limit_df[
    "OPERATION_GROSS_AMOUNT_CTV"
].abs()

current_outstanding = cash_limit_df[
    "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT"
].abs()

previous_outstanding = cash_limit_df[
    "PREVIOUS_PERIOD_TOTAL_OUTSTANDING_AMOUNT"
].abs()

current_limit = cash_limit_df[
    "OVERARCHING_CREDIT_LIMIT"
]

previous_limit = cash_limit_df[
    "PREVIOUS_OVERARCHING_CREDIT_LIMIT"
]

valid_current_limit = current_limit.notna() & current_limit.gt(0)
valid_previous_limit = previous_limit.notna() & previous_limit.gt(0)

cash_limit_df["TRANSACTION_LIMIT_UTILIZATION"] = np.where(
    valid_current_limit,
    transaction_amount / current_limit,
    np.nan,
)

# BEFORE = previous business-day utilisation
# AFTER = reference-date utilisation
cash_limit_df["LIMIT_UTILIZATION_BEFORE_TRANSACTION"] = np.where(
    valid_previous_limit,
    previous_outstanding / previous_limit,
    np.nan,
)

cash_limit_df["LIMIT_UTILIZATION_AFTER_TRANSACTION"] = np.where(
    valid_current_limit,
    current_outstanding / current_limit,
    np.nan,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Strict same-day crossing flags
# Flag a threshold crossing only when a reference-day cash advance exists and utilisation
# moves from at or below the threshold to above it.
#
# Both conditions must hold on REFERENCE_DATE:
#   - a credit-card cash advance exists; and
#   - utilisation crosses the threshold from the previous business day
#     to the current day
#
# The cash-advance amount is not added to outstanding

before = cash_limit_df[
    "LIMIT_UTILIZATION_BEFORE_TRANSACTION"
]
after = cash_limit_df[
    "LIMIT_UTILIZATION_AFTER_TRANSACTION"
]
same_day_cash_advance = cash_limit_df[
    "TRANSACTION_DTE"
].eq(REFERENCE_DATE)

cash_limit_df["RRE_CASH_ADVANCE_LIMIT_CROSS_94_1D"] = (
    same_day_cash_advance
    & before.notna()
    & after.notna()
    & before.le(RRE_CASH_ADVANCE_LIMIT_THRESHOLD)
    & after.gt(RRE_CASH_ADVANCE_LIMIT_THRESHOLD)
).astype("int8")

cash_limit_df["OTHER_CASH_ADVANCE_LIMIT_CROSS_97_1D"] = (
    same_day_cash_advance
    & before.notna()
    & after.notna()
    & before.le(OTHER_CASH_ADVANCE_LIMIT_THRESHOLD)
    & after.gt(OTHER_CASH_ADVANCE_LIMIT_THRESHOLD)
).astype("int8")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep implementation fields
# Select the transaction identifiers, timing flags, before/after ratios and crossing
# results used by the indicator logic.

cash_limit_df = cash_limit_df[
    [
        "ENTITY_ID",
        "MAP_DTE",
        "TRANSACTION_DTE",
        "TRANSACTION_LAG_BUSINESS_DAYS",
        "TRANSACTION_WITHIN_1D_WINDOW",
        "CARD_ATM_TRANSACTION_ID",
        "CARD_ID",
        "TRANSACTION_LIMIT_UTILIZATION",
        "LIMIT_UTILIZATION_BEFORE_TRANSACTION",
        "LIMIT_UTILIZATION_AFTER_TRANSACTION",
        "RRE_CASH_ADVANCE_LIMIT_CROSS_94_1D",
        "OTHER_CASH_ADVANCE_LIMIT_CROSS_97_1D",
    ]
].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Source transaction diagnostics
# Compare mapped transactions with the raw source and print utilisation and threshold-
# crossing coverage.

source_count_query = """
SELECT
    COUNT(*) AS RAW_ROWS,
    COUNT(DISTINCT CARD_ATM_TRANSACTION_ID) AS RAW_TRANSACTIONS
FROM {transaction_table}
WHERE TRUNC(OPERATION_DATE) = DATE '{reference_date}'
  AND SOURCE_TRANSACTION_CODE = 1091
  AND UPPER(TRIM(CARD_TYPE)) = 'CREDIT CARD'
""".format(
    transaction_table=transaction_table,
    reference_date=REFERENCE_DATE_STR,
)

source_count_df = SQLExecutor2(
    dataset=transaction_dataset
).query_to_df(source_count_query)

raw_rows = int(
    pd.to_numeric(
        source_count_df.get("RAW_ROWS", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()
)

raw_transactions = int(
    pd.to_numeric(
        source_count_df.get("RAW_TRANSACTIONS", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()
)

unique_transaction_keys = int(
    cash_limit_df[
        ["TRANSACTION_DTE", "CARD_ATM_TRANSACTION_ID"]
    ]
    .drop_duplicates()
    .shape[0]
)

transaction_duplicates = int(
    cash_limit_df.duplicated(
        subset=["TRANSACTION_DTE", "CARD_ATM_TRANSACTION_ID"],
        keep=False,
    ).sum()
)

entities = int(cash_limit_df["ENTITY_ID"].nunique())

rows_with_previous_util = int(
    cash_limit_df["LIMIT_UTILIZATION_BEFORE_TRANSACTION"]
    .notna()
    .sum()
)

rows_with_current_util = int(
    cash_limit_df["LIMIT_UTILIZATION_AFTER_TRANSACTION"]
    .notna()
    .sum()
)

rre_crossings = int(
    cash_limit_df["RRE_CASH_ADVANCE_LIMIT_CROSS_94_1D"].sum()
)
other_crossings = int(
    cash_limit_df["OTHER_CASH_ADVANCE_LIMIT_CROSS_97_1D"].sum()
)

same_day_transactions = int(
    cash_limit_df["TRANSACTION_DTE"].eq(REFERENCE_DATE).sum()
)

distinct_map_dates = int(cash_limit_df["MAP_DTE"].nunique())

print()
print("=" * 90)
print("CASH ADVANCE CC LIMIT QA - STRICT SAME DAY")
print("=" * 90)
print("Reference date                         :", REFERENCE_DATE_STR)
print("Previous business date                 :", PREVIOUS_DATE_STR)
print("Raw same-day credit-card cash rows     :", "{:,}".format(raw_rows))
print("Raw same-day credit-card transactions  :", "{:,}".format(raw_transactions))
print("Mapped same-day transaction rows       :", "{:,}".format(len(cash_limit_df)))
print("Unique transaction keys                :", "{:,}".format(unique_transaction_keys))
print("Entities                               :", "{:,}".format(entities))
print("Same-day transactions                  :", "{:,}".format(same_day_transactions))
print("Rows with previous-day utilisation     :", "{:,}".format(rows_with_previous_util))
print("Rows with current-day utilisation      :", "{:,}".format(rows_with_current_util))
print("RRE same-day crossings <=94% -> >94%   :", "{:,}".format(rre_crossings))
print("Other same-day crossings <=97% -> >97% :", "{:,}".format(other_crossings))
print("Duplicate transaction keys             :", "{:,}".format(transaction_duplicates))
print("Distinct MAP_DTE                       :", distinct_map_dates)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final validations
# Check transaction key uniqueness, expected reference dates and report any empty output.

if transaction_duplicates > 0:
    raise ValueError(
        "Cash_Advance_CC_Limit contains duplicate transaction-date keys."
    )

if not cash_limit_df.empty and distinct_map_dates != 1:
    raise ValueError(
        "Cash_Advance_CC_Limit should contain exactly one MAP_DTE."
    )

if (
    not cash_limit_df.empty
    and cash_limit_df["MAP_DTE"].max().normalize()
    != REFERENCE_DATE.normalize()
):
    raise ValueError(
        "Cash_Advance_CC_Limit MAP_DTE does not match REFERENCE_DATE."
    )

invalid_transaction_dates = int(
    cash_limit_df["TRANSACTION_DTE"]
    .ne(REFERENCE_DATE)
    .sum()
)

if invalid_transaction_dates > 0:
    raise ValueError(
        "Cash_Advance_CC_Limit contains a cash-advance transaction outside REFERENCE_DATE."
    )

if cash_limit_df.empty:
    print(
        "WARNING: Cash_Advance_CC_Limit returned 0 mapped same-day "
        "credit-card cash advances for {}.".format(REFERENCE_DATE_STR)
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the previous-branch same-day cash-advance utilisation features.

dataiku.Dataset(
    "Cash_Advance_CC_Limit_Previous"
).write_with_schema(cash_limit_df)