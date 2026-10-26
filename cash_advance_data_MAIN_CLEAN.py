# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation dates
# Read the dates used for daily cash activity and subsequent comparisons.

implementation_dates_df = dataiku.Dataset(
    "Implementation_Reference_Date"
).get_dataframe()


if implementation_dates_df.empty:
    raise ValueError(
        "Implementation_Reference_Date is empty."
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
        "Implementation_Reference_Date is missing "
        "required column(s): {}".format(
            ", ".join(
                missing_date_columns
            )
        )
    )


if len(implementation_dates_df) != 1:
    raise ValueError(
        "Implementation_Reference_Date must contain "
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
# Locate the physical SQL tables configured for each Dataiku dataset.

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
# Use ATM transactions, card ownership and customer-to-entity relationships.

transaction_dataset, transaction_table = (
    get_sql_source(
        "CARDS_ATM_TRANSACTIONS_VW"
    )
)


_, card_table = get_sql_source(
    "CARD_REFERENTIAL_VW"
)


_, entity_customer_table = get_sql_source(
    "ENTITY_CUSTOMER_VW_1"
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Cash-advance transactions
# Select credit-card cash advances on the reference date, link them to an entity, and total the transactions and amounts per entity.
#
# Cash advances are identified using SOURCE_TRANSACTION_CODE = 1091.
# Transactions and reference mappings are evaluated on REFERENCE_DATE.
# MAP_DTE is set to REFERENCE_DATE.

query = """
WITH filtered_transactions AS (

    SELECT
        t.CARD_ATM_TRANSACTION_ID,
        t.ACCOUNT_ID,
        t.OPERATION_AMOUNT

    FROM {transaction_table} t

    WHERE TRUNC(
        t.OPERATION_DATE
    ) = DATE '{source_date}'

      AND t.SOURCE_TRANSACTION_CODE = 1091

      AND UPPER(
          TRIM(
              t.CARD_TYPE
          )
      ) = 'CREDIT CARD'
),

card_mapping_ranked AS (

    SELECT
        c.LINKED_ACCOUNT_ID,
        c.CARD_OWNER_CUSTOMER_ID,

        ROW_NUMBER() OVER (

            PARTITION BY
                TRIM(
                    CAST(
                        c.LINKED_ACCOUNT_ID
                        AS VARCHAR2(100)
                    )
                )

            ORDER BY
                c.TEK_STRT_DTE DESC,
                c.CARD_ID

        ) AS RN

    FROM {card_table} c

    WHERE TRUNC(
    c.TEK_STRT_DTE
) <= DATE '{source_date}'

AND DATE '{source_date}'
    < TRUNC(
        NVL(
            c.TEK_END_DTE,
            DATE '9999-12-31'
        )
    )

      AND UPPER(
          TRIM(
              c.CARD_TYPE
          )
      ) = 'CREDIT CARD'

      AND c.LINKED_ACCOUNT_ID IS NOT NULL
      AND c.CARD_OWNER_CUSTOMER_ID IS NOT NULL
),

card_mapping AS (

    SELECT
        LINKED_ACCOUNT_ID,
        CARD_OWNER_CUSTOMER_ID

    FROM card_mapping_ranked

    WHERE RN = 1
),

entity_mapping_ranked AS (

    SELECT
        ec.CUSTOMER_ID,
        ec.ENTITY_ID,

        ROW_NUMBER() OVER (

            PARTITION BY
                TRIM(
                    CAST(
                        ec.CUSTOMER_ID
                        AS VARCHAR2(100)
                    )
                )

            ORDER BY
                ec.TEK_STRT_DTE DESC,
                ec.ENTITY_ID

        ) AS RN

    FROM {entity_customer_table} ec

    WHERE TRUNC(
    ec.TEK_STRT_DTE
) <= DATE '{source_date}'

AND DATE '{source_date}'
    < TRUNC(
        NVL(
            ec.TEK_END_DTE,
            DATE '9999-12-31'
        )
    )

      AND ec.CUSTOMER_ID IS NOT NULL
      AND ec.ENTITY_ID IS NOT NULL
),

entity_mapping AS (

    SELECT
        CUSTOMER_ID,
        ENTITY_ID

    FROM entity_mapping_ranked

    WHERE RN = 1
),

mapped_transactions AS (

    SELECT
        e.ENTITY_ID,
        t.CARD_ATM_TRANSACTION_ID,
        t.OPERATION_AMOUNT

    FROM filtered_transactions t

    INNER JOIN card_mapping c

        ON TRIM(
            CAST(
                t.ACCOUNT_ID
                AS VARCHAR2(100)
            )
        )
        =
        TRIM(
            CAST(
                c.LINKED_ACCOUNT_ID
                AS VARCHAR2(100)
            )
        )

    INNER JOIN entity_mapping e

        ON TRIM(
            CAST(
                c.CARD_OWNER_CUSTOMER_ID
                AS VARCHAR2(100)
            )
        )
        =
        TRIM(
            CAST(
                e.CUSTOMER_ID
                AS VARCHAR2(100)
            )
        )
)

SELECT
    ENTITY_ID
        AS ENTITY_ID,

    DATE '{reference_date}'
        AS MAP_DTE,

    COUNT(*)
        AS TRANSACTION_COUNT,

    SUM(
        NVL(
            OPERATION_AMOUNT,
            0
        )
    ) AS TOTAL_OPERATION_AMOUNT

FROM mapped_transactions

GROUP BY
    ENTITY_ID

""".format(
    reference_date=REFERENCE_DATE_STR,
    source_date=REFERENCE_DATE_STR,
    transaction_table=transaction_table,
    card_table=card_table,
    entity_customer_table=entity_customer_table,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Execute query
# Retrieve the daily cash-advance activity from the source tables.

start_time = time.time()


cash_advance_df = SQLExecutor2(
    dataset=transaction_dataset
).query_to_df(
    query
)


print(
    "Time to perform query: {:.1f} seconds".format(
        time.time() - start_time
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Handle empty result
# Keep the expected output columns even when no transactions meet the conditions.

if cash_advance_df.empty:

    cash_advance_df = pd.DataFrame(
        columns=[
            "ENTITY_ID",
            "MAP_DTE",
            "TRANSACTION_COUNT",
            "TOTAL_OPERATION_AMOUNT",
        ]
    )

else:

    cash_advance_df[
        "ENTITY_ID"
    ] = (
        cash_advance_df[
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


    cash_advance_df[
        "MAP_DTE"
    ] = (
        pd.to_datetime(
            cash_advance_df[
                "MAP_DTE"
            ],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )


    cash_advance_df[
        "TRANSACTION_COUNT"
    ] = (
        pd.to_numeric(
            cash_advance_df[
                "TRANSACTION_COUNT"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
    )


    cash_advance_df[
        "TOTAL_OPERATION_AMOUNT"
    ] = (
        pd.to_numeric(
            cash_advance_df[
                "TOTAL_OPERATION_AMOUNT"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Source transaction QA
# Count the eligible cash advances before joining them to their owners.

source_query = """
SELECT
    COUNT(*) AS RAW_ROWS,

    COUNT(
        DISTINCT CARD_ATM_TRANSACTION_ID
    ) AS RAW_TRANSACTIONS

FROM {transaction_table}

WHERE TRUNC(
    OPERATION_DATE
) = DATE '{source_date}'

  AND SOURCE_TRANSACTION_CODE = 1091

  AND UPPER(
      TRIM(
          CARD_TYPE
      )
  ) = 'CREDIT CARD'

""".format(
    transaction_table=transaction_table,
    source_date=REFERENCE_DATE_STR,
)


source_df = SQLExecutor2(
    dataset=transaction_dataset
).query_to_df(
    source_query
)


raw_rows = int(
    source_df[
        "RAW_ROWS"
    ].iloc[0]
)


raw_transactions = int(
    source_df[
        "RAW_TRANSACTIONS"
    ].iloc[0]
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Compare source transactions with mapped transactions and check output uniqueness.

duplicate_entity_days = int(
    cash_advance_df
    .duplicated(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
        ]
    )
    .sum()
)


mapped_transactions = int(
    pd.to_numeric(
        cash_advance_df[
            "TRANSACTION_COUNT"
        ],
        errors="coerce",
    )
    .fillna(
        0
    )
    .sum()
)


entities = int(
    cash_advance_df[
        "ENTITY_ID"
    ]
    .nunique()
)


total_amount = float(
    pd.to_numeric(
        cash_advance_df[
            "TOTAL_OPERATION_AMOUNT"
        ],
        errors="coerce",
    )
    .fillna(
        0.0
    )
    .sum()
)


distinct_dates = int(
    cash_advance_df[
        "MAP_DTE"
    ]
    .nunique()
)


unmapped_transactions = (
    raw_transactions
    - mapped_transactions
)


mapping_rate = (
    mapped_transactions
    / raw_transactions
    if raw_transactions > 0
    else None
)


print()
print("=" * 80)
print("CASH ADVANCE QA")
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
    "Raw source rows      :",
    "{:,}".format(
        raw_rows
    ),
)

print(
    "Raw source transactions:",
    "{:,}".format(
        raw_transactions
    ),
)

print(
    "Rows / entities      :",
    "{:,}".format(
        len(
            cash_advance_df
        )
    ),
)

print(
    "Entities             :",
    "{:,}".format(
        entities
    ),
)

print(
    "Mapped transactions  :",
    "{:,}".format(
        mapped_transactions
    ),
)

print(
    "Unmapped transactions:",
    "{:,}".format(
        unmapped_transactions
    ),
)

if mapping_rate is not None:

    print(
        "Transaction mapping rate:",
        "{:.2%}".format(
            mapping_rate
        ),
    )


print(
    "Total source amount  :",
    "{:,.2f}".format(
        total_amount
    ),
)

print(
    "Distinct MAP_DTE     :",
    distinct_dates,
)

print(
    "Duplicated ENTITY_ID + MAP_DTE:",
    duplicate_entity_days,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final validations
# Reject duplicate records or an unexpected output date.

if duplicate_entity_days > 0:
    raise ValueError(
        "cash_advance_data contains duplicate "
        "ENTITY_ID + MAP_DTE rows."
    )


if (
    not cash_advance_df.empty
    and distinct_dates != 1
):
    raise ValueError(
        "cash_advance_data should contain exactly "
        "one MAP_DTE when data is available."
    )


if (
    not cash_advance_df.empty
    and (
        cash_advance_df[
            "MAP_DTE"
        ]
        .max()
        .normalize()
        != REFERENCE_DATE.normalize()
    )
):
    raise ValueError(
        "cash_advance_data MAP_DTE does not match "
        "the implementation reference date."
    )


if cash_advance_df.empty:

    print(
        "WARNING: cash_advance_data returned "
        "0 rows for source business date {}.".format(
            REFERENCE_DATE_STR
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the daily cash-advance counts and amounts by entity.

dataiku.Dataset(
    "cash_advance_data"
).write_with_schema(
    cash_advance_df
)