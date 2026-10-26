# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation dates
# Set the dates used to identify transactions and label the daily output.

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
# Locate the underlying SQL table for each configured input dataset.

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
# Use card transactions, card owners and the entity/customer mapping.

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
### Neobank transactions
# Find qualifying neobank top-ups, link each transaction to an entity and total the activity for the day.
#
# Neobank transactions are identified using merchant category code 6012
# and operation locality REVOLUT.COM.
#
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

      AND t.OPERATION_MERCHANT_CATEGORY_CODE = 6012

      AND UPPER(
          TRIM(
              t.OPERATION_LOCALITY
          )
      ) = 'REVOLUT.COM'
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

        ) AS rn

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

    WHERE rn = 1
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

        ) AS rn

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

    WHERE rn = 1
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
        AS NEOBANK_TOPUP_COUNT,

    SUM(
        NVL(
            OPERATION_AMOUNT,
            0
        )
    ) AS NEOBANK_TOPUP_AMOUNT

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
# Retrieve the mapped neobank transactions and their amounts.

start_time = time.time()


neobank_df = SQLExecutor2(
    dataset=transaction_dataset
).query_to_df(
    query
)


query_elapsed = (
    time.time()
    - start_time
)


print(
    "Time to perform query: {:.1f} seconds".format(
        query_elapsed
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Handle empty result
# Preserve the output structure if no matching transactions are found.

if neobank_df.empty:

    neobank_df = pd.DataFrame(
        columns=[
            "ENTITY_ID",
            "MAP_DTE",
            "neobank_topup_count",
            "neobank_topup_amount",
        ]
    )

else:

    neobank_df[
        "ENTITY_ID"
    ] = (
        neobank_df[
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


    neobank_df[
        "MAP_DTE"
    ] = (
        pd.to_datetime(
            neobank_df[
                "MAP_DTE"
            ],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )


    neobank_df[
        "NEOBANK_TOPUP_COUNT"
    ] = (
        pd.to_numeric(
            neobank_df[
                "NEOBANK_TOPUP_COUNT"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
    )


    neobank_df[
        "NEOBANK_TOPUP_AMOUNT"
    ] = (
        pd.to_numeric(
            neobank_df[
                "NEOBANK_TOPUP_AMOUNT"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
    )


    neobank_df.rename(
        columns={
            "NEOBANK_TOPUP_COUNT":
                "neobank_topup_count",
            "NEOBANK_TOPUP_AMOUNT":
                "neobank_topup_amount",
        },
        inplace=True,
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check the daily number of entities, top-ups, transaction amounts and duplicate rows.

duplicate_entity_days = int(
    neobank_df
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
        neobank_df
    )
)


n_entities = int(
    neobank_df[
        "ENTITY_ID"
    ]
    .nunique()
)


transaction_count = int(
    pd.to_numeric(
        neobank_df[
            "neobank_topup_count"
        ],
        errors="coerce",
    )
    .fillna(
        0
    )
    .sum()
)


total_amount = float(
    pd.to_numeric(
        neobank_df[
            "neobank_topup_amount"
        ],
        errors="coerce",
    )
    .fillna(
        0.0
    )
    .sum()
)


n_dates = int(
    neobank_df[
        "MAP_DTE"
    ]
    .nunique()
)


print()
print("=" * 80)
print("NEOBANK TRANSACTIONS QA")
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
    "Total source amount :",
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
### Empty-output diagnostics
# If no top-ups are mapped, count the records remaining after each source join.

if neobank_df.empty:

    diagnostic_query = """
    WITH raw_transactions AS (

        SELECT
            t.ACCOUNT_ID

        FROM {transaction_table} t

        WHERE TRUNC(
            t.OPERATION_DATE
        ) = DATE '{source_date}'

          AND t.OPERATION_MERCHANT_CATEGORY_CODE = 6012

          AND UPPER(
              TRIM(
                  t.OPERATION_LOCALITY
              )
          ) = 'REVOLUT.COM'
    ),

    card_mapped AS (

        SELECT
            r.ACCOUNT_ID,
            c.CARD_OWNER_CUSTOMER_ID

        FROM raw_transactions r

        INNER JOIN {card_table} c

            ON TRIM(
                CAST(
                    r.ACCOUNT_ID
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
    ),

    entity_mapped AS (

        SELECT
            c.ACCOUNT_ID,
            ec.ENTITY_ID

        FROM card_mapped c

        INNER JOIN {entity_customer_table} ec

            ON TRIM(
                CAST(
                    c.CARD_OWNER_CUSTOMER_ID
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
    ec.TEK_STRT_DTE
) <= DATE '{source_date}'

AND DATE '{source_date}'
    < TRUNC(
        NVL(
            ec.TEK_END_DTE,
            DATE '9999-12-31'
        )
    )

          AND ec.ENTITY_ID IS NOT NULL
    )

    SELECT

        (
            SELECT COUNT(*)
            FROM raw_transactions
        ) AS RAW_TRANSACTIONS,

        (
            SELECT COUNT(*)
            FROM card_mapped
        ) AS AFTER_ACCOUNT_CARD_JOIN,

        (
            SELECT COUNT(*)
            FROM entity_mapped
        ) AS AFTER_ENTITY_JOIN

    FROM dual

    """.format(
        source_date=REFERENCE_DATE_STR,
        transaction_table=transaction_table,
        card_table=card_table,
        entity_customer_table=entity_customer_table,
    )


    diagnostic_df = SQLExecutor2(
        dataset=transaction_dataset
    ).query_to_df(
        diagnostic_query
    )


    print()
    print("NEOBANK EMPTY-OUTPUT DIAGNOSTICS")
    print("--------------------------------")

    print(
        diagnostic_df.to_string(
            index=False
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate QA
# Ensure that the result has one record per entity and the correct date.

if duplicate_entity_days > 0:
    raise ValueError(
        "CC_Transactions_Neobanks contains duplicate "
        "ENTITY_ID + MAP_DTE rows."
    )


if (
    not neobank_df.empty
    and n_dates != 1
):
    raise ValueError(
        "CC_Transactions_Neobanks should contain "
        "exactly one MAP_DTE when data is available."
    )


if (
    not neobank_df.empty
    and (
        neobank_df[
            "MAP_DTE"
        ]
        .max()
        .normalize()
        != REFERENCE_DATE.normalize()
    )
):
    raise ValueError(
        "CC_Transactions_Neobanks MAP_DTE does not "
        "match the implementation reference date."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the neobank top-up counts and amounts used by the behavioural indicators.

dataiku.Dataset(
    "CC_Transactions_Neobanks"
).write_with_schema(
    neobank_df
)