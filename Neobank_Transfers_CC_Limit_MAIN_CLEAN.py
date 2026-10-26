# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import numpy as np
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation dates
# Set the current and previous dates that define the transaction window.

implementation_dates_df = dataiku.Dataset(
    "Implementation_Reference_Date"
).get_dataframe()

if implementation_dates_df.empty:
    raise ValueError("Implementation_Reference_Date is empty.")

required_date_columns = ["REFERENCE_DATE", "PREVIOUS_DATE"]
missing_date_columns = [
    column
    for column in required_date_columns
    if column not in implementation_dates_df.columns
]

if missing_date_columns:
    raise KeyError(
        "Implementation_Reference_Date is missing required column(s): {}".format(
            ", ".join(missing_date_columns)
        )
    )

if len(implementation_dates_df) != 1:
    raise ValueError(
        "Implementation_Reference_Date must contain exactly one row."
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
### SQL source utility
# Locate the physical tables linked to the configured inputs.

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
# Use card transactions, owner details, usage and entity/customer mappings.

transaction_dataset, transaction_table = get_sql_source(
    "CARDS_ATM_TRANSACTIONS_VW"
)
_, card_table = get_sql_source("CARD_REFERENTIAL_VW")
_, usage_table = get_sql_source("CARD_USAGE_VW")
_, entity_customer_table = get_sql_source("ENTITY_CUSTOMER_VW_1")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Neobank transfer / card-limit utilisation
# Find qualifying neobank transfers on either date and retrieve card outstanding amounts and limits on the transaction date.
#
# Neobank transactions are identified using merchant category code 6012
# and operation locality REVOLUT.COM.
#
# TRANSACTION_DTE keeps the transaction date and MAP_DTE is set to REFERENCE_DATE.

query = """
WITH filtered_transactions AS (
    SELECT
        t.CARD_ATM_TRANSACTION_ID,
        t.ACCOUNT_ID,
        TRUNC(t.OPERATION_DATE) AS TRANSACTION_DTE,
        t.OPERATION_AMOUNT
    FROM {transaction_table} t
    WHERE TRUNC(t.OPERATION_DATE) IN (
        DATE '{previous_date}',
        DATE '{reference_date}'
    )
      AND t.OPERATION_MERCHANT_CATEGORY_CODE = 6012
      AND UPPER(TRIM(t.OPERATION_LOCALITY)) = 'REVOLUT.COM'
),

card_candidates AS (
    SELECT
        t.CARD_ATM_TRANSACTION_ID,
        t.ACCOUNT_ID,
        t.TRANSACTION_DTE,
        t.OPERATION_AMOUNT,
        c.CARD_OWNER_CUSTOMER_ID,
        c.CARD_ID,
        c.CREDIT_LIMIT,
        c.TEMPORARY_CREDIT_LIMIT,
        ROW_NUMBER() OVER (
            PARTITION BY
                t.TRANSACTION_DTE,
                t.CARD_ATM_TRANSACTION_ID
            ORDER BY
                c.TEK_STRT_DTE DESC,
                c.CARD_ID
        ) AS RN
    FROM filtered_transactions t
    INNER JOIN {card_table} c
        ON TRIM(CAST(t.ACCOUNT_ID AS VARCHAR2(100)))
         = TRIM(CAST(c.LINKED_ACCOUNT_ID AS VARCHAR2(100)))
       AND TRUNC(c.TEK_STRT_DTE) <= t.TRANSACTION_DTE
       AND t.TRANSACTION_DTE < TRUNC(
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
        c.OPERATION_AMOUNT,
        c.CARD_OWNER_CUSTOMER_ID,
        c.CARD_ID,
        c.CREDIT_LIMIT,
        c.TEMPORARY_CREDIT_LIMIT,
        ec.ENTITY_ID,
        ROW_NUMBER() OVER (
            PARTITION BY
                c.TRANSACTION_DTE,
                c.CARD_ATM_TRANSACTION_ID
            ORDER BY
                ec.TEK_STRT_DTE DESC,
                ec.ENTITY_ID
        ) AS RN
    FROM card_mapping c
    INNER JOIN {entity_customer_table} ec
        ON TRIM(CAST(c.CARD_OWNER_CUSTOMER_ID AS VARCHAR2(100)))
         = TRIM(CAST(ec.CUSTOMER_ID AS VARCHAR2(100)))
       AND TRUNC(ec.TEK_STRT_DTE) <= c.TRANSACTION_DTE
       AND c.TRANSACTION_DTE < TRUNC(
            NVL(ec.TEK_END_DTE, DATE '9999-12-31')
       )
       AND ec.ENTITY_ID IS NOT NULL
),

entity_mapping AS (
    SELECT *
    FROM entity_candidates
    WHERE RN = 1
),

usage_candidates AS (
    SELECT
        e.ENTITY_ID,
        e.TRANSACTION_DTE,
        e.CARD_ATM_TRANSACTION_ID,
        e.CARD_ID,
        e.CREDIT_LIMIT,
        e.TEMPORARY_CREDIT_LIMIT,
        e.OPERATION_AMOUNT,
        u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT,
        ROW_NUMBER() OVER (
            PARTITION BY
                e.TRANSACTION_DTE,
                e.CARD_ATM_TRANSACTION_ID
            ORDER BY
                u.TEK_STRT_DTE DESC NULLS LAST
        ) AS RN
    FROM entity_mapping e
    LEFT JOIN {usage_table} u
        ON TRIM(CAST(e.CARD_ID AS VARCHAR2(100)))
         = TRIM(CAST(u.CARD_ID AS VARCHAR2(100)))
       AND TRUNC(u.TEK_STRT_DTE) <= e.TRANSACTION_DTE
       AND e.TRANSACTION_DTE < TRUNC(
            NVL(u.TEK_END_DTE, DATE '9999-12-31')
       )
)

SELECT
    ENTITY_ID,
    DATE '{reference_date}' AS MAP_DTE,
    TRANSACTION_DTE,
    CARD_ATM_TRANSACTION_ID,
    CARD_ID,
    CREDIT_LIMIT,
    TEMPORARY_CREDIT_LIMIT,
    OPERATION_AMOUNT,
    CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT
FROM usage_candidates
WHERE RN = 1
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
# Retrieve the transaction and card observations for the two-day window.

start_time = time.time()

neobank_limit_df = SQLExecutor2(
    dataset=transaction_dataset
).query_to_df(query)

print(
    "Time to perform query: {:.1f} seconds".format(
        time.time() - start_time
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise identifiers and dates
# Prepare IDs and dates so transactions can be compared consistently.

for column in ["ENTITY_ID", "CARD_ID", "CARD_ATM_TRANSACTION_ID"]:
    neobank_limit_df[column] = (
        neobank_limit_df[column]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

for column in ["MAP_DTE", "TRANSACTION_DTE"]:
    neobank_limit_df[column] = (
        pd.to_datetime(
            neobank_limit_df[column],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

for column in [
    "CREDIT_LIMIT",
    "TEMPORARY_CREDIT_LIMIT",
    "OPERATION_AMOUNT",
    "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT",
]:
    neobank_limit_df[column] = pd.to_numeric(
        neobank_limit_df[column],
        errors="coerce",
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Transaction timing
# Identify whether each transfer took place today or on the previous reference day.

neobank_limit_df["TRANSACTION_LAG_BUSINESS_DAYS"] = np.select(
    [
        neobank_limit_df["TRANSACTION_DTE"].eq(REFERENCE_DATE),
        neobank_limit_df["TRANSACTION_DTE"].eq(PREVIOUS_DATE),
    ],
    [0, 1],
    default=np.nan,
)

neobank_limit_df["TRANSACTION_WITHIN_1D_WINDOW"] = (
    neobank_limit_df["TRANSACTION_LAG_BUSINESS_DAYS"]
    .isin([0, 1])
    .astype("int8")
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Credit-limit utilisation around each neobank transfer
# Compare card outstanding and transfer amount with the higher available credit limit.

neobank_limit_df["OVERARCHING_CREDIT_LIMIT"] = (
    neobank_limit_df[
        ["CREDIT_LIMIT", "TEMPORARY_CREDIT_LIMIT"]
    ]
    .max(axis=1, skipna=True)
)

transaction_amount = neobank_limit_df["OPERATION_AMOUNT"].abs()
outstanding_before = neobank_limit_df[
    "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT"
].abs()
limit_amount = neobank_limit_df["OVERARCHING_CREDIT_LIMIT"]

valid_limit = limit_amount.notna() & limit_amount.gt(0)

neobank_limit_df["TRANSACTION_LIMIT_UTILIZATION"] = np.where(
    valid_limit,
    transaction_amount / limit_amount,
    np.nan,
)

neobank_limit_df["LIMIT_UTILIZATION_BEFORE_TRANSACTION"] = np.where(
    valid_limit,
    outstanding_before / limit_amount,
    np.nan,
)

neobank_limit_df["LIMIT_UTILIZATION_AFTER_TRANSACTION"] = np.where(
    valid_limit,
    (outstanding_before + transaction_amount) / limit_amount,
    np.nan,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep implementation fields
# Keep transfer dates, card IDs and utilisation measures required downstream.

neobank_limit_df = neobank_limit_df[
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
    ]
].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check transaction uniqueness, date coverage and how many records contain utilisation.

transaction_duplicates = int(
    neobank_limit_df.duplicated(
        subset=["TRANSACTION_DTE", "CARD_ATM_TRANSACTION_ID"],
        keep=False,
    ).sum()
)

entities = int(neobank_limit_df["ENTITY_ID"].nunique())
rows_with_after = int(
    neobank_limit_df["LIMIT_UTILIZATION_AFTER_TRANSACTION"]
    .notna()
    .sum()
)
same_day_transactions = int(
    neobank_limit_df["TRANSACTION_LAG_BUSINESS_DAYS"].eq(0).sum()
)
previous_day_transactions = int(
    neobank_limit_df["TRANSACTION_LAG_BUSINESS_DAYS"].eq(1).sum()
)
distinct_map_dates = int(neobank_limit_df["MAP_DTE"].nunique())

print()
print("=" * 90)
print("NEOBANK TRANSFERS CC LIMIT QA")
print("=" * 90)
print("Reference date                     :", REFERENCE_DATE_STR)
print("Previous business date             :", PREVIOUS_DATE_STR)
print("Mapped transaction rows            :", "{:,}".format(len(neobank_limit_df)))
print("Entities                           :", "{:,}".format(entities))
print("Same-day transactions              :", "{:,}".format(same_day_transactions))
print("Previous-business-day transactions :", "{:,}".format(previous_day_transactions))
print("Rows with limit-after utilisation  :", "{:,}".format(rows_with_after))
print("Duplicate transaction keys         :", "{:,}".format(transaction_duplicates))
print("Distinct MAP_DTE                   :", distinct_map_dates)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final validations
# Reject duplicate transaction keys or dates outside the selected window.

if transaction_duplicates > 0:
    raise ValueError(
        "Neobank_Transfers_CC_Limit contains duplicate transaction-date keys."
    )

if not neobank_limit_df.empty and distinct_map_dates != 1:
    raise ValueError(
        "Neobank_Transfers_CC_Limit should contain exactly one MAP_DTE."
    )

if (
    not neobank_limit_df.empty
    and neobank_limit_df["MAP_DTE"].max().normalize()
    != REFERENCE_DATE.normalize()
):
    raise ValueError(
        "Neobank_Transfers_CC_Limit MAP_DTE does not match REFERENCE_DATE."
    )

invalid_transaction_dates = int(
    (~neobank_limit_df["TRANSACTION_DTE"].isin(
        [REFERENCE_DATE, PREVIOUS_DATE]
    )).sum()
)

if invalid_transaction_dates > 0:
    raise ValueError(
        "Neobank_Transfers_CC_Limit contains transaction dates outside the 0/1-step window."
    )

if neobank_limit_df.empty:
    print(
        "WARNING: Neobank_Transfers_CC_Limit returned 0 mapped transactions "
        "for {} / {}.".format(PREVIOUS_DATE_STR, REFERENCE_DATE_STR)
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save neobank-related card-limit utilisation observations.

dataiku.Dataset(
    "Neobank_Transfers_CC_Limit"
).write_with_schema(neobank_limit_df)