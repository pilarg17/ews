# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation date
# Use the reference date to calculate both daily and month-to-date payment activity.

reference_date_df = dataiku.Dataset(
    "Implementation_Reference_Date"
).get_dataframe()


if reference_date_df.empty:
    raise ValueError(
        "Implementation_Reference_Date is empty."
    )


if "REFERENCE_DATE" not in reference_date_df.columns:
    raise KeyError(
        "Implementation_Reference_Date is missing "
        "required column: REFERENCE_DATE"
    )


if len(reference_date_df) != 1:
    raise ValueError(
        "Implementation_Reference_Date must contain "
        "exactly one row."
    )


REFERENCE_DATE = (
    pd.to_datetime(
        reference_date_df.loc[
            reference_date_df.index[0],
            "REFERENCE_DATE",
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


MONTH_START_DATE = (
    REFERENCE_DATE
    .replace(
        day=1
    )
)


REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)


MONTH_START_DATE_STR = (
    MONTH_START_DATE.strftime(
        "%Y-%m-%d"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### SQL source utility
# Locate the payment table configured in Dataiku.

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
### Resolve source
# Use the payment records from the source system.

payor_dataset, payor_table = (
    get_sql_source(
        "PAYOR_VW"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Daily and month-to-date failed direct debits
# Count failed SEPA direct debits for the reference day and from the start of the month.
#
# Source transactions are read from the first day of the current month
# up to REFERENCE_DATE.
#
# Daily metrics represent activity occurring exactly on REFERENCE_DATE.
#
# MTD metrics represent cumulative activity from the first day of the
# month up to and including REFERENCE_DATE.


query = """
WITH payments AS (

    SELECT
        p.DEBIT_ACCOUNT_CUSTOMER_ID,
        p.DEBIT_PAYMENT_ID,
        p.DEBIT_VALUE_DATE,
        p.DEBIT_AMOUNT_EUR,
        p.STATUS_CODE,

        ROW_NUMBER() OVER (
            PARTITION BY
                p.DEBIT_PAYMENT_ID,
                p.DEBIT_VALUE_DATE

            ORDER BY
                p.DEBIT_PAYMENT_ID
        ) AS RN

    FROM {payor_table} p

    WHERE TRUNC(
        p.DEBIT_VALUE_DATE
    ) BETWEEN DATE '{month_start_date}'
          AND DATE '{reference_date}'

      AND UPPER(
          TRIM(
              p.SOURCE_DESCRIPTION
          )
      ) = 'SEPA DIRECT DEBIT'

      AND p.DEBIT_ACCOUNT_CUSTOMER_ID IS NOT NULL
),

deduplicated_payments AS (

    SELECT
        DEBIT_ACCOUNT_CUSTOMER_ID,
        DEBIT_PAYMENT_ID,
        DEBIT_VALUE_DATE,
        DEBIT_AMOUNT_EUR,
        STATUS_CODE

    FROM payments

    WHERE RN = 1
)

SELECT
    DEBIT_ACCOUNT_CUSTOMER_ID
        AS CUSTOMER_ID,

    DATE '{reference_date}'
        AS MAP_DTE,

    /* -------------------------------------------------------------- */
    /* Daily metrics: REFERENCE_DATE only                              */
    /* -------------------------------------------------------------- */

    SUM(
        CASE
            WHEN STATUS_CODE = 997
             AND TRUNC(
                    DEBIT_VALUE_DATE
                 ) = DATE '{reference_date}'
                THEN 1
            ELSE 0
        END
    ) AS FAILED_DEBIT_COUNT,

    SUM(
        CASE
            WHEN STATUS_CODE = 997
             AND TRUNC(
                    DEBIT_VALUE_DATE
                 ) = DATE '{reference_date}'
                THEN NVL(
                    DEBIT_AMOUNT_EUR,
                    0
                )
            ELSE 0
        END
    ) AS FAILED_DEBIT_AMOUNT,

    /* -------------------------------------------------------------- */
    /* Month-to-date metrics                                           */
    /* -------------------------------------------------------------- */

    SUM(
        CASE
            WHEN STATUS_CODE = 997
                THEN 1
            ELSE 0
        END
    ) AS FAILED_DEBIT_COUNT_MTD,

    SUM(
        CASE
            WHEN STATUS_CODE = 997
                THEN NVL(
                    DEBIT_AMOUNT_EUR,
                    0
                )
            ELSE 0
        END
    ) AS FAILED_DEBIT_AMOUNT_MTD

FROM deduplicated_payments

GROUP BY
    DEBIT_ACCOUNT_CUSTOMER_ID

""".format(
    payor_table=payor_table,
    reference_date=REFERENCE_DATE_STR,
    month_start_date=MONTH_START_DATE_STR,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Execute query
# Retrieve the payment totals for each customer.

start_time = time.time()


vars_payment_level_df = SQLExecutor2(
    dataset=payor_dataset
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
# Align customer IDs and dates, and convert payment amounts and counts to numbers.

vars_payment_level_df[
    "CUSTOMER_ID"
] = (
    vars_payment_level_df[
        "CUSTOMER_ID"
    ]
    .astype("string")
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True,
    )
)


vars_payment_level_df[
    "MAP_DTE"
] = (
    pd.to_datetime(
        vars_payment_level_df[
            "MAP_DTE"
        ],
        errors="coerce",
        utc=True,
    )
    .dt.tz_localize(None)
    .dt.normalize()
)


numeric_columns = [
    "FAILED_DEBIT_COUNT",
    "FAILED_DEBIT_AMOUNT",
    "FAILED_DEBIT_COUNT_MTD",
    "FAILED_DEBIT_AMOUNT_MTD",
]


for column in numeric_columns:

    vars_payment_level_df[
        column
    ] = (
        pd.to_numeric(
            vars_payment_level_df[
                column
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final columns
# Keep the daily and month-to-date payment measures used by the indicators.

vars_payment_level_df = (
    vars_payment_level_df[
        [
            "CUSTOMER_ID",
            "MAP_DTE",
            "FAILED_DEBIT_COUNT",
            "FAILED_DEBIT_AMOUNT",
            "FAILED_DEBIT_COUNT_MTD",
            "FAILED_DEBIT_AMOUNT_MTD",
        ]
    ]
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Review payment volumes, the six-failure threshold and duplicate customer records.

duplicate_customer_days = int(
    vars_payment_level_df
    .duplicated(
        subset=[
            "CUSTOMER_ID",
            "MAP_DTE",
        ]
    )
    .sum()
)


n_rows = int(
    len(
        vars_payment_level_df
    )
)


n_customers = int(
    vars_payment_level_df[
        "CUSTOMER_ID"
    ]
    .nunique()
)


n_dates = int(
    vars_payment_level_df[
        "MAP_DTE"
    ]
    .nunique()
)


daily_failed_debits = int(
    vars_payment_level_df[
        "FAILED_DEBIT_COUNT"
    ]
    .sum()
)


daily_failed_debit_amount = float(
    vars_payment_level_df[
        "FAILED_DEBIT_AMOUNT"
    ]
    .sum()
)


mtd_failed_debits = int(
    vars_payment_level_df[
        "FAILED_DEBIT_COUNT_MTD"
    ]
    .sum()
)


mtd_failed_debit_amount = float(
    vars_payment_level_df[
        "FAILED_DEBIT_AMOUNT_MTD"
    ]
    .sum()
)


customers_above_6_mtd = int(
    (
        vars_payment_level_df[
            "FAILED_DEBIT_COUNT_MTD"
        ] > 6
    )
    .sum()
)


max_daily_failed_debits = int(
    vars_payment_level_df[
        "FAILED_DEBIT_COUNT"
    ]
    .max()
) if not vars_payment_level_df.empty else 0


max_mtd_failed_debits = int(
    vars_payment_level_df[
        "FAILED_DEBIT_COUNT_MTD"
    ]
    .max()
) if not vars_payment_level_df.empty else 0


print()
print("=" * 80)
print("VARS_PAYMENT_LEVEL QA")
print("=" * 80)

print(
    "Reference date        :",
    REFERENCE_DATE_STR,
)

print(
    "Month start           :",
    MONTH_START_DATE_STR,
)

print(
    "Rows                  :",
    "{:,}".format(
        n_rows
    ),
)

print(
    "Customers             :",
    "{:,}".format(
        n_customers
    ),
)

print()
print("-" * 80)
print("DAILY FAILED DIRECT DEBITS")
print("-" * 80)

print(
    "Failed debits         :",
    "{:,}".format(
        daily_failed_debits
    ),
)

print(
    "Failed debit amount   :",
    "{:,.2f}".format(
        daily_failed_debit_amount
    ),
)

print(
    "Maximum per customer  :",
    "{:,}".format(
        max_daily_failed_debits
    ),
)

print()
print("-" * 80)
print("MONTH-TO-DATE FAILED DIRECT DEBITS")
print("-" * 80)

print(
    "MTD failed debits     :",
    "{:,}".format(
        mtd_failed_debits
    ),
)

print(
    "MTD failed amount     :",
    "{:,.2f}".format(
        mtd_failed_debit_amount
    ),
)

print(
    "Maximum per customer  :",
    "{:,}".format(
        max_mtd_failed_debits
    ),
)

print(
    "Customers > 6 MTD     :",
    "{:,}".format(
        customers_above_6_mtd
    ),
)

print()
print("-" * 80)
print("STRUCTURAL QA")
print("-" * 80)

print(
    "Distinct MAP_DTE      :",
    n_dates,
)

print(
    "Duplicate customer days:",
    duplicate_customer_days,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final validations
# Check that the output contains one row per customer and that monthly totals are consistent.

if duplicate_customer_days > 0:
    raise ValueError(
        "vars_payment_level contains duplicate "
        "CUSTOMER_ID + MAP_DTE rows."
    )


if (
    not vars_payment_level_df.empty
    and n_dates != 1
):
    raise ValueError(
        "vars_payment_level should contain exactly "
        "one MAP_DTE when data is available."
    )


if (
    not vars_payment_level_df.empty
    and (
        vars_payment_level_df[
            "MAP_DTE"
        ].max().normalize()
        != REFERENCE_DATE.normalize()
    )
):
    raise ValueError(
        "vars_payment_level MAP_DTE does not match "
        "the implementation reference date."
    )


if (
    (
        vars_payment_level_df[
            "FAILED_DEBIT_COUNT_MTD"
        ]
        <
        vars_payment_level_df[
            "FAILED_DEBIT_COUNT"
        ]
    )
    .any()
):
    raise ValueError(
        "FAILED_DEBIT_COUNT_MTD cannot be lower "
        "than FAILED_DEBIT_COUNT."
    )


if (
    (
        vars_payment_level_df[
            "FAILED_DEBIT_AMOUNT_MTD"
        ]
        <
        vars_payment_level_df[
            "FAILED_DEBIT_AMOUNT"
        ]
    )
    .any()
):
    raise ValueError(
        "FAILED_DEBIT_AMOUNT_MTD cannot be lower "
        "than FAILED_DEBIT_AMOUNT."
    )


if vars_payment_level_df.empty:
    print(
        "WARNING: vars_payment_level returned 0 rows "
        "between {} and {}.".format(
            MONTH_START_DATE_STR,
            REFERENCE_DATE_STR,
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the customer payment measures for the daily indicator calculation.

dataiku.Dataset(
    "vars_payment_level"
).write_with_schema(
    vars_payment_level_df
)