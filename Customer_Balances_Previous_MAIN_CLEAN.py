# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration
# Set the current-account category and the overdraft level shown in the checks.

OUTPUT_DATASET = "Customer_Balances_Previous"

CURRENT_ACCOUNT_CATEGORY = "1001"

OVERDRAFT_THRESHOLD = 500.0
EUR_FORMAT = "EUR {:,.2f}"


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation reference date
# Select the day used for account balances and overdrafts.

reference_date_df = dataiku.Dataset(
    "Implementation_Reference_Date_previous"
).get_dataframe()


if reference_date_df.empty:
    raise ValueError(
        "Implementation_Reference_Date_previous is empty."
    )


if "REFERENCE_DATE" not in reference_date_df.columns:
    raise KeyError(
        "Implementation_Reference_Date_previous is missing REFERENCE_DATE."
    )


reference_dates = (
    pd.to_datetime(
        reference_date_df["REFERENCE_DATE"],
        errors="coerce",
        utc=True,
    )
    .dt.tz_localize(None)
    .dt.normalize()
    .dropna()
    .drop_duplicates()
)


if len(reference_dates) != 1:
    raise ValueError(
        "Implementation_Reference_Date_previous must contain "
        "exactly one valid REFERENCE_DATE."
    )


REFERENCE_DATE = reference_dates.iloc[0]

REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)


print(
    "Reference date:",
    REFERENCE_DATE_STR,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### SQL source utility
# Locate the account and entity-to-customer SQL tables.

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
# Identify the source accounts and their customer relationships.

account_dataset, account_table = (
    get_sql_source(
        "PLY_T24_ACCOUNT_VW"
    )
)


entity_customer_dataset, entity_customer_table = (
    get_sql_source(
        "ENTITY_CUSTOMER_VW_1"
    )
)


print(
    "Account source:",
    account_table,
)

print(
    "Entity/customer source:",
    entity_customer_table,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Customer balances and current-account overdraft
# Sum account balances by entity and add up negative balances for current accounts.

query = """
SELECT
    ec.ENTITY_ID AS ENTITY_ID,

    DATE '{reference_date}'
        AS MAP_DTE,

    SUM(
        NVL(
            acc.WORKING_BALANCE,
            0
        )
    ) AS ASSETS,

    SUM(
        CASE
            WHEN TRIM(
                TO_CHAR(
                    acc.CATEGORY
                )
            ) = '{current_account_category}'

            AND NVL(
                acc.WORKING_BALANCE,
                0
            ) < 0

            THEN ABS(
                NVL(
                    acc.WORKING_BALANCE,
                    0
                )
            )

            ELSE 0
        END
    ) AS OVERDRAFT_AMOUNT

FROM {account_table} acc

INNER JOIN
(
    SELECT DISTINCT
        ENTITY_ID,
        CUSTOMER_ID

    FROM {entity_customer_table}

    WHERE TRUNC(
        TEK_STRT_DTE
    ) <= DATE '{reference_date}'

    AND DATE '{reference_date}'
        < TRUNC(
            NVL(
                TEK_END_DTE,
                DATE '9999-12-31'
            )
        )

    AND ENTITY_ID IS NOT NULL
    AND CUSTOMER_ID IS NOT NULL

) ec

    ON TRIM(
        CAST(
            acc.CUSTOMER
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
    acc.TEK_STRT_DTE
) <= DATE '{reference_date}'

AND DATE '{reference_date}'
    < TRUNC(
        NVL(
            acc.TEK_END_DTE,
            DATE '9999-12-31'
        )
    )

AND acc.CUSTOMER IS NOT NULL

GROUP BY
    ec.ENTITY_ID

""".format(
    reference_date=REFERENCE_DATE_STR,
    current_account_category=CURRENT_ACCOUNT_CATEGORY,
    account_table=account_table,
    entity_customer_table=entity_customer_table,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Execute query
# Retrieve the balances recorded on the reference date.

start_time = time.time()


customer_balances_df = SQLExecutor2(
    dataset=account_dataset
).query_to_df(
    query
)


balance_query_elapsed = (
    time.time()
    - start_time
)


print(
    "Time to perform balance query: "
    "{:.1f} seconds".format(
        balance_query_elapsed
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Safety check
# Stop if the source query returned no records.

if customer_balances_df.empty:
    raise ValueError(
        "Customer_Balances_Previous query returned 0 rows "
        "for reference date {}. "
        "Output dataset will NOT be overwritten.".format(
            REFERENCE_DATE_STR
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise output
# Prepare entity IDs, dates and numeric balances.

customer_balances_df[
    "ENTITY_ID"
] = (
    customer_balances_df[
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

customer_balances_df[
    "MAP_DTE"
] = (
    pd.to_datetime(
        customer_balances_df[
            "MAP_DTE"
        ],
        errors="coerce",
        utc=True,
    )
    .dt.tz_localize(None)
    .dt.normalize()
)


for column in [
    "ASSETS",
    "OVERDRAFT_AMOUNT",
]:

    customer_balances_df[
        column
    ] = (
        pd.to_numeric(
            customer_balances_df[
                column
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Remove invalid keys
# Keep only rows with a usable entity ID and date.

customer_balances_df = (
    customer_balances_df
    .dropna(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
        ]
    )
    .copy()
)


customer_balances_df = (
    customer_balances_df.loc[
        customer_balances_df[
            "ENTITY_ID"
        ]
        .astype("string")
        .str.len()
        .gt(0)
    ]
    .copy()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check balance and overdraft coverage, including the EUR 500 threshold.

duplicate_entity_days = int(
    customer_balances_df
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
        customer_balances_df
    )
)


n_entities = int(
    customer_balances_df[
        "ENTITY_ID"
    ]
    .nunique()
)


n_dates = int(
    customer_balances_df[
        "MAP_DTE"
    ]
    .nunique()
)


min_date = (
    customer_balances_df[
        "MAP_DTE"
    ]
    .min()
)


max_date = (
    customer_balances_df[
        "MAP_DTE"
    ]
    .max()
)


assets_non_null = int(
    customer_balances_df[
        "ASSETS"
    ]
    .notna()
    .sum()
)


assets_positive = int(
    customer_balances_df[
        "ASSETS"
    ]
    .gt(0)
    .sum()
)


assets_zero_or_negative = int(
    customer_balances_df[
        "ASSETS"
    ]
    .le(0)
    .sum()
)


entities_with_overdraft = int(
    customer_balances_df[
        "OVERDRAFT_AMOUNT"
    ]
    .gt(0)
    .sum()
)


entities_overdraft_above_500 = int(
    customer_balances_df[
        "OVERDRAFT_AMOUNT"
    ]
    .gt(
        OVERDRAFT_THRESHOLD
    )
    .sum()
)


total_overdraft_amount = float(
    customer_balances_df[
        "OVERDRAFT_AMOUNT"
    ]
    .sum()
)


overdraft_positive = (
    customer_balances_df.loc[
        customer_balances_df[
            "OVERDRAFT_AMOUNT"
        ].gt(0),
        "OVERDRAFT_AMOUNT",
    ]
)


if overdraft_positive.empty:
    median_overdraft_amount = 0.0
    max_overdraft_amount = 0.0

else:
    median_overdraft_amount = float(
        overdraft_positive.median()
    )

    max_overdraft_amount = float(
        overdraft_positive.max()
    )


negative_overdraft_rows = int(
    customer_balances_df[
        "OVERDRAFT_AMOUNT"
    ]
    .lt(0)
    .sum()
)


print()
print("=" * 100)
print("CUSTOMER BALANCES - QA")
print("=" * 100)

print(
    "Reference date              :",
    REFERENCE_DATE_STR,
)

print(
    "Rows                        :",
    "{:,}".format(
        n_rows
    ),
)

print(
    "Entities                    :",
    "{:,}".format(
        n_entities
    ),
)

print(
    "Distinct dates              :",
    n_dates,
)

print(
    "Minimum MAP_DTE             :",
    min_date,
)

print(
    "Maximum MAP_DTE             :",
    max_date,
)

print(
    "Rows with ASSETS            :",
    "{:,}".format(
        assets_non_null
    ),
)

print(
    "Rows with positive ASSETS   :",
    "{:,}".format(
        assets_positive
    ),
)

print(
    "Rows with ASSETS <= 0       :",
    "{:,}".format(
        assets_zero_or_negative
    ),
)

print(
    "Duplicated ENTITY_ID + date :",
    duplicate_entity_days,
)


print()
print("=" * 100)
print("CURRENT-ACCOUNT OVERDRAFT - QA")
print("=" * 100)

print(
    "T24 current-account category:",
    CURRENT_ACCOUNT_CATEGORY,
)

print(
    "Entities with overdraft > 0 :",
    "{:,}".format(
        entities_with_overdraft
    ),
)

print(
    "Entities with overdraft >500:",
    "{:,}".format(
        entities_overdraft_above_500
    ),
)

print(
    "Total overdraft amount      :",
    EUR_FORMAT.format(
        total_overdraft_amount
    ),
)

print(
    "Median positive overdraft   :",
    EUR_FORMAT.format(
        median_overdraft_amount
    ),
)

print(
    "Maximum overdraft           :",
    EUR_FORMAT.format(
        max_overdraft_amount
    ),
)

print(
    "Negative overdraft rows     :",
    "{:,}".format(
        negative_overdraft_rows
    ),
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final validations
# Confirm there is one valid date and no duplicate entity or negative overdraft values.

if n_rows == 0:
    raise ValueError(
        "Customer_Balances is empty after "
        "standardisation. "
        "Output dataset will NOT be overwritten."
    )


if duplicate_entity_days > 0:
    raise ValueError(
        "Customer_Balances_Previous contains duplicate "
        "ENTITY_ID + MAP_DTE rows."
    )


if n_dates != 1:
    raise ValueError(
        "Customer_Balances_Previous should contain exactly "
        "one snapshot date, but found {}.".format(
            n_dates
        )
    )


if (
    pd.Timestamp(
        max_date
    ).normalize()
    != REFERENCE_DATE.normalize()
):
    raise ValueError(
        "Customer_Balances_Previous output date {} "
        "does not match reference date {}.".format(
            pd.Timestamp(
                max_date
            ).strftime(
                "%Y-%m-%d"
            ),
            REFERENCE_DATE_STR,
        )
    )


if negative_overdraft_rows > 0:
    raise ValueError(
        "OVERDRAFT_AMOUNT must not contain "
        "negative values."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final clean schema
# Keep the entity, date, assets and overdraft amounts.

customer_balances_df = (
    customer_balances_df[
        [
            "ENTITY_ID",
            "MAP_DTE",
            "ASSETS",
            "OVERDRAFT_AMOUNT",
        ]
    ]
    .sort_values(
        [
            "ENTITY_ID",
            "MAP_DTE",
        ]
    )
    .reset_index(
        drop=True
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the daily balances for the previous snapshot.

dataiku.Dataset(
    OUTPUT_DATASET
).write_with_schema(
    customer_balances_df
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final log
# Report the output size and query duration.

print()
print("=" * 100)
print("CUSTOMER BALANCES - COMPLETED")
print("=" * 100)

print(
    "Output dataset :",
    OUTPUT_DATASET,
)

print(
    "Snapshot date  :",
    REFERENCE_DATE_STR,
)

print(
    "Rows written   :",
    "{:,}".format(
        len(
            customer_balances_df
        )
    ),
)

print(
    "Entities       :",
    "{:,}".format(
        customer_balances_df[
            "ENTITY_ID"
        ]
        .nunique()
    ),
)

print(
    "Overdraft >500:",
    "{:,}".format(
        customer_balances_df[
            "OVERDRAFT_AMOUNT"
        ]
        .gt(
            OVERDRAFT_THRESHOLD
        )
        .sum()
    ),
)

print(
    "Balance query  : {:.1f}s".format(
        balance_query_elapsed
    )
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE