# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation dates
# Use the previous-branch reference date for the source query and output.

implementation_dates_df = dataiku.Dataset(
    "Implementation_Reference_Date_previous"
).get_dataframe()

if implementation_dates_df.empty:
    raise ValueError(
        "Implementation_Reference_Date_previous is empty."
    )

required_date_columns = [
    "REFERENCE_DATE",
    "PREVIOUS_DATE",
]

missing_date_columns = [
    column
    for column in required_date_columns
    if column not in implementation_dates_df.columns
]

if missing_date_columns:
    raise KeyError(
        "Implementation_Reference_Date_previous is missing "
        "required column(s): {}".format(
            ", ".join(
                missing_date_columns
            )
        )
    )

if len(implementation_dates_df) != 1:
    raise ValueError(
        "Implementation_Reference_Date_previous must contain "
        "exactly one row."
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
    raise ValueError(
        "REFERENCE_DATE could not be parsed."
    )

if pd.isna(PREVIOUS_DATE):
    raise ValueError(
        "PREVIOUS_DATE could not be parsed."
    )

REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)

PREVIOUS_DATE_STR = (
    PREVIOUS_DATE.strftime(
        "%Y-%m-%d"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### SQL source utility
# Look up the SQL table behind each Dataiku input dataset.

def get_sql_source(dataset_name):

    dataset = dataiku.Dataset(
        dataset_name
    )

    location = dataset.get_location_info()

    info = location.get(
        "info",
        {},
    )

    schema = (
        info.get("schema")
        or info.get("schemaName")
    )

    table = (
        info.get("table")
        or info.get("tableName")
    )

    if not table:
        raise ValueError(
            "{} must be a SQL-backed dataset.".format(
                dataset_name
            )
        )

    if (
        schema
        and "." not in table
    ):
        table = "{}.{}".format(
            schema,
            table,
        )

    return dataset, table


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Resolve source tables
# Link teller transactions with account owners and their entities.

transaction_dataset, transaction_table = (
    get_sql_source(
        "TELLER_TRANSACTIONS_VW"
    )
)

account_dataset, account_table = (
    get_sql_source(
        "ACCOUNT_REFERENTIAL_VW"
    )
)

entity_customer_dataset, entity_customer_table = (
    get_sql_source(
        "ENTITY_CUSTOMER_VW_1"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Cash deposits
# Select deposits made on the reference date and total them by entity.
# Account and entity/customer mappings must be valid on the same date.

query = """
WITH account_mapping AS (

    SELECT DISTINCT
        ACCOUNT_ID,
        ACCOUNT_MAIN_CUSTOMER_ID

    FROM {account_table}

    WHERE TRUNC(
    TEK_STRT_DTE
) <= DATE '{source_date}'

AND DATE '{source_date}'
    < TRUNC(
        NVL(
            TEK_END_DTE,
            DATE '9999-12-31'
        )
    )

      AND ACCOUNT_ID IS NOT NULL
      AND ACCOUNT_MAIN_CUSTOMER_ID IS NOT NULL
),

entity_mapping AS (

    SELECT DISTINCT
        ENTITY_ID,
        CUSTOMER_ID

    FROM {entity_customer_table}

    WHERE TRUNC(
    TEK_STRT_DTE
) <= DATE '{source_date}'

AND DATE '{source_date}'
    < TRUNC(
        NVL(
            TEK_END_DTE,
            DATE '9999-12-31'
        )
    )

      AND ENTITY_ID IS NOT NULL
      AND CUSTOMER_ID IS NOT NULL
)

SELECT
    ec.ENTITY_ID
        AS ENTITY_ID,

    DATE '{reference_date}'
        AS MAP_DTE,

    COUNT(*)
        AS TRANSACTION_COUNT,

    SUM(
        NVL(
            t.OPERATION_AMOUNT,
            0
        )
    ) AS TOTAL_OPERATION_AMOUNT

FROM {transaction_table} t

INNER JOIN account_mapping a

    ON TRIM(
        CAST(
            t.ACCOUNT_ID
            AS VARCHAR2(100)
        )
    )
    =
    TRIM(
        CAST(
            a.ACCOUNT_ID
            AS VARCHAR2(100)
        )
    )

INNER JOIN entity_mapping ec

    ON TRIM(
        CAST(
            a.ACCOUNT_MAIN_CUSTOMER_ID
            AS VARCHAR2(100)
        )
    )
    =
    TRIM(
        CAST(
            ec.CUSTOMER_ID
            AS VARCHAR2(100)
        )
    )

WHERE TRUNC(
    t.OPERATION_DATE
) = DATE '{source_date}'

  AND UPPER(
      TRIM(
          t.SOURCE_TRANSACTION_CODE_DESCRIPTION
      )
  ) = 'DEPOSIT'

GROUP BY
    ec.ENTITY_ID

""".format(
    reference_date=REFERENCE_DATE_STR,
    source_date=REFERENCE_DATE_STR,
    transaction_table=transaction_table,
    account_table=account_table,
    entity_customer_table=entity_customer_table,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Execute query
# Read the deposit totals for the selected day.

start_time = time.time()

cash_deposits_df = SQLExecutor2(
    dataset=transaction_dataset
).query_to_df(
    query
)

query_elapsed = (
    time.time()
    - start_time
)

print(
    "Time to perform the query {:.1f} seconds".format(
        query_elapsed
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise output
# Align entity IDs, dates, transaction counts and deposit amounts.

cash_deposits_df[
    "ENTITY_ID"
] = (
    cash_deposits_df[
        "ENTITY_ID"
    ]
    .astype("string")
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True,
    )
)

cash_deposits_df[
    "MAP_DTE"
] = (
    pd.to_datetime(
        cash_deposits_df[
            "MAP_DTE"
        ],
        errors="coerce",
        utc=True,
    )
    .dt.tz_localize(None)
    .dt.normalize()
)

cash_deposits_df[
    "TRANSACTION_COUNT"
] = (
    pd.to_numeric(
        cash_deposits_df[
            "TRANSACTION_COUNT"
        ],
        errors="coerce",
    )
    .fillna(
        0
    )
)

cash_deposits_df[
    "TOTAL_OPERATION_AMOUNT"
] = (
    pd.to_numeric(
        cash_deposits_df[
            "TOTAL_OPERATION_AMOUNT"
        ],
        errors="coerce",
    )
    .fillna(
        0.0
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check entity coverage, volume, amounts and duplicate entity-day records.

duplicate_entity_days = int(
    cash_deposits_df
    .duplicated(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
        ]
    )
    .sum()
)

n_rows = int(
    len(
        cash_deposits_df
    )
)

n_entities = int(
    cash_deposits_df[
        "ENTITY_ID"
    ]
    .nunique()
)

n_dates = int(
    cash_deposits_df[
        "MAP_DTE"
    ]
    .nunique()
)

transaction_count = int(
    cash_deposits_df[
        "TRANSACTION_COUNT"
    ]
    .sum()
)

total_amount = float(
    cash_deposits_df[
        "TOTAL_OPERATION_AMOUNT"
    ]
    .sum()
)

print()
print("=" * 80)
print("CASH DEPOSITS QA")
print("=" * 80)

print(
    "Reference date      :",
    REFERENCE_DATE_STR,
)

print(
    "Source business date:",
    REFERENCE_DATE_STR,
)

print(
    "Rows                :",
    "{:,}".format(
        n_rows
    ),
)

print(
    "Entities            :",
    "{:,}".format(
        n_entities
    ),
)

print(
    "Transactions        :",
    "{:,}".format(
        transaction_count
    ),
)

print(
    "Total amount        :",
    "{:,.2f}".format(
        total_amount
    ),
)

print(
    "Distinct MAP_DTE    :",
    n_dates,
)

print(
    "Duplicated ENTITY_ID + MAP_DTE:",
    duplicate_entity_days,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final validations
# Reject duplicates or dates outside the expected daily snapshot.

if duplicate_entity_days > 0:
    raise ValueError(
        "Cash_Deposits_Previous contains duplicate "
        "ENTITY_ID + MAP_DTE rows."
    )

if (
    not cash_deposits_df.empty
    and n_dates != 1
):
    raise ValueError(
        "Cash_Deposits_Previous should contain exactly "
        "one MAP_DTE when data is available."
    )

if (
    not cash_deposits_df.empty
    and (
        cash_deposits_df[
            "MAP_DTE"
        ]
        .max()
        .normalize()
        != REFERENCE_DATE.normalize()
    )
):
    raise ValueError(
        "Cash_Deposits_Previous MAP_DTE does not match "
        "the implementation reference date."
    )

if cash_deposits_df.empty:
    print(
        "WARNING: Cash_Deposits_Previous returned 0 rows "
        "for source business date {}.".format(
            REFERENCE_DATE_STR
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the cash deposit counts and amounts for the previous-day branch.

dataiku.Dataset(
    "Cash_Deposits_Previous"
).write_with_schema(
    cash_deposits_df
)
