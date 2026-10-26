# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import numpy as np
import pandas as pd
import time
from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Reference date
# Set the day for which card usage and limits are prepared.

reference_df = dataiku.Dataset(
    "Implementation_Reference_Date"
).get_dataframe()

if reference_df.empty:
    raise ValueError(
        "Implementation_Reference_Date is empty."
    )

if "REFERENCE_DATE" not in reference_df.columns:
    raise KeyError(
        "Implementation_Reference_Date is missing REFERENCE_DATE."
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

MONTH_START_DATE = REFERENCE_DATE.replace(day=1)
MONTH_START_DATE_STR = MONTH_START_DATE.strftime("%Y-%m-%d")


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### SQL source utility
# Find the SQL sources configured in Dataiku.

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
            "{} must be SQL-backed.".format(
                dataset_name
            )
        )

    if schema and "." not in table:
        table = "{}.{}".format(
            schema,
            table,
        )

    return dataset, table


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Resolve source tables
# Locate card usage, card ownership and customer entity links.

usage_dataset, usage_table = get_sql_source(
    "CARD_USAGE_VW"
)

_, card_table = get_sql_source(
    "CARD_REFERENTIAL_VW"
)

_, customer_table = get_sql_source(
    "CUSTOMER_VW"
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Credit-card source diagnostics
# Check availability and accounting signs of the outstanding balances.

source_qa_query = """
SELECT
    COUNT(*) AS USAGE_ROWS,

    COUNT(DISTINCT u.CARD_ID) AS USAGE_CARDS,

    SUM(
        CASE
            WHEN u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT IS NULL
                THEN 1
            ELSE 0
        END
    ) AS OUTSTANDING_NULL_ROWS,

    SUM(
        CASE
            WHEN u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT IS NOT NULL
                THEN 1
            ELSE 0
        END
    ) AS OUTSTANDING_POPULATED_ROWS,

    SUM(
        CASE
            WHEN u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT < 0
                THEN 1
            ELSE 0
        END
    ) AS OUTSTANDING_NEGATIVE_ROWS,

    SUM(
        CASE
            WHEN u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT > 0
                THEN 1
            ELSE 0
        END
    ) AS OUTSTANDING_POSITIVE_ROWS,

    SUM(
        CASE
            WHEN u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT = 0
                THEN 1
            ELSE 0
        END
    ) AS OUTSTANDING_ZERO_ROWS

FROM {usage_table} u

WHERE TRUNC(
    u.TEK_STRT_DTE
) <= DATE '{reference_date}'

AND DATE '{reference_date}'
    < TRUNC(
        NVL(
            u.TEK_END_DTE,
            DATE '9999-12-31'
        )
    )
""".format(
    reference_date=REFERENCE_DATE_STR,
    usage_table=usage_table,
)

source_qa_df = SQLExecutor2(
    dataset=usage_dataset
).query_to_df(
    source_qa_query
)

print()
print("CARD_USAGE SOURCE QA")
print("--------------------")

if source_qa_df.empty:
    print("No rows returned.")
else:
    print(
        source_qa_df.to_string(
            index=False
        )
    )

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Daily credit-card data
# Retrieve credit-card usage, link cards to their owners, and select month-start limits or the first available limits.

query = """
WITH current_usage AS (

    SELECT
        u.CARD_ID,
        u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT

    FROM {usage_table} u

    WHERE TRUNC(u.TEK_STRT_DTE) <= DATE '{reference_date}'
      AND DATE '{reference_date}' < TRUNC(
            NVL(
                u.TEK_END_DTE,
                DATE '9999-12-31'
            )
      )
),

current_cards AS (

    SELECT
        CARD_ID,
        CARD_OWNER_CUSTOMER_ID

    FROM (
        SELECT
            c.CARD_ID,
            c.CARD_OWNER_CUSTOMER_ID,
            ROW_NUMBER() OVER (
                PARTITION BY c.CARD_ID
                ORDER BY
                    c.TEK_STRT_DTE DESC,
                    NVL(c.TEK_END_DTE, DATE '9999-12-31') DESC
            ) AS RN

        FROM {card_table} c

        WHERE TRUNC(c.TEK_STRT_DTE) <= DATE '{reference_date}'
          AND DATE '{reference_date}' < TRUNC(
                NVL(
                    c.TEK_END_DTE,
                    DATE '9999-12-31'
                )
          )
          AND UPPER(TRIM(c.CARD_TYPE)) = 'CREDIT CARD'
    )

    WHERE RN = 1
),

month_start_limits AS (

    SELECT
        CARD_ID,
        CREDIT_LIMIT,
        TEMPORARY_CREDIT_LIMIT

    FROM (
        SELECT
            c.CARD_ID,
            c.CREDIT_LIMIT,
            c.TEMPORARY_CREDIT_LIMIT,
            ROW_NUMBER() OVER (
                PARTITION BY c.CARD_ID
                ORDER BY
                    c.TEK_STRT_DTE DESC,
                    NVL(c.TEK_END_DTE, DATE '9999-12-31') DESC
            ) AS RN

        FROM {card_table} c

        WHERE TRUNC(c.TEK_STRT_DTE) <= DATE '{month_start_date}'
          AND DATE '{month_start_date}' < TRUNC(
                NVL(
                    c.TEK_END_DTE,
                    DATE '9999-12-31'
                )
          )
          AND UPPER(TRIM(c.CARD_TYPE)) = 'CREDIT CARD'
    )

    WHERE RN = 1
),

first_available_limits AS (

    SELECT
        CARD_ID,
        CREDIT_LIMIT,
        TEMPORARY_CREDIT_LIMIT

    FROM (
        SELECT
            c.CARD_ID,
            c.CREDIT_LIMIT,
            c.TEMPORARY_CREDIT_LIMIT,
            ROW_NUMBER() OVER (
                PARTITION BY c.CARD_ID
                ORDER BY
                    c.TEK_STRT_DTE ASC,
                    NVL(c.TEK_END_DTE, DATE '9999-12-31') ASC
            ) AS RN

        FROM {card_table} c

        WHERE TRUNC(c.TEK_STRT_DTE) > DATE '{month_start_date}'
          AND TRUNC(c.TEK_STRT_DTE) <= DATE '{reference_date}'
          AND UPPER(TRIM(c.CARD_TYPE)) = 'CREDIT CARD'
          AND (
              c.CREDIT_LIMIT IS NOT NULL
              OR c.TEMPORARY_CREDIT_LIMIT IS NOT NULL
          )
    )

    WHERE RN = 1
),

current_customers AS (

    SELECT
        CUSTOMER_ID,
        COUNTERPARTY_IDENTIFIER_GLOBE

    FROM (
        SELECT
            cust.CUSTOMER_ID,
            cust.COUNTERPARTY_IDENTIFIER_GLOBE,
            ROW_NUMBER() OVER (
                PARTITION BY cust.CUSTOMER_ID
                ORDER BY
                    cust.TEK_STRT_DTE DESC,
                    NVL(cust.TEK_END_DTE, DATE '9999-12-31') DESC
            ) AS RN

        FROM {customer_table} cust

        WHERE TRUNC(cust.TEK_STRT_DTE) <= DATE '{reference_date}'
          AND DATE '{reference_date}' < TRUNC(
                NVL(
                    cust.TEK_END_DTE,
                    DATE '9999-12-31'
                )
          )
    )

    WHERE RN = 1
)

SELECT
    cust.COUNTERPARTY_IDENTIFIER_GLOBE AS ENTITY_ID,
    DATE '{reference_date}' AS MAP_DTE,
    c.CARD_ID AS CARD_ID,

    CASE
        WHEN ms.CREDIT_LIMIT IS NOT NULL OR ms.TEMPORARY_CREDIT_LIMIT IS NOT NULL THEN ms.CREDIT_LIMIT
        ELSE fa.CREDIT_LIMIT
    END AS CREDIT_LIMIT,

    CASE
        WHEN ms.CREDIT_LIMIT IS NOT NULL OR ms.TEMPORARY_CREDIT_LIMIT IS NOT NULL THEN ms.TEMPORARY_CREDIT_LIMIT
        ELSE fa.TEMPORARY_CREDIT_LIMIT
    END AS TEMPORARY_CREDIT_LIMIT,

    u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT
        AS CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT,

    CASE
        WHEN ms.CREDIT_LIMIT IS NOT NULL OR ms.TEMPORARY_CREDIT_LIMIT IS NOT NULL THEN 'MONTH_START'
        WHEN fa.CARD_ID IS NOT NULL THEN 'FIRST_AVAILABLE'
        ELSE 'MISSING'
    END AS LIMIT_SOURCE

FROM current_usage u

INNER JOIN current_cards c
    ON TRIM(CAST(u.CARD_ID AS VARCHAR2(100)))
     = TRIM(CAST(c.CARD_ID AS VARCHAR2(100)))

INNER JOIN current_customers cust
    ON TRIM(CAST(c.CARD_OWNER_CUSTOMER_ID AS VARCHAR2(100)))
     = TRIM(CAST(cust.CUSTOMER_ID AS VARCHAR2(100)))

LEFT JOIN month_start_limits ms
    ON TRIM(CAST(c.CARD_ID AS VARCHAR2(100)))
     = TRIM(CAST(ms.CARD_ID AS VARCHAR2(100)))

LEFT JOIN first_available_limits fa
    ON TRIM(CAST(c.CARD_ID AS VARCHAR2(100)))
     = TRIM(CAST(fa.CARD_ID AS VARCHAR2(100)))

WHERE cust.COUNTERPARTY_IDENTIFIER_GLOBE IS NOT NULL
""".format(
    reference_date=REFERENCE_DATE_STR,
    month_start_date=MONTH_START_DATE_STR,
    usage_table=usage_table,
    card_table=card_table,
    customer_table=customer_table,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Execute query
# Retrieve the card records and available limits for the selected date.

start_time = time.time()

credit_cards_raw_df = SQLExecutor2(
    dataset=usage_dataset
).query_to_df(
    query
)

print(
    "Time to perform query: {:.1f} seconds".format(
        time.time() - start_time
    )
)

if credit_cards_raw_df.empty:
    raise ValueError(
        "No credit-card observations were returned "
        "for {}.".format(
            REFERENCE_DATE_STR
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate source columns
# Confirm that cards, limits and outstanding balances are available.

required_columns = [
    "ENTITY_ID",
    "MAP_DTE",
    "CARD_ID",
    "CREDIT_LIMIT",
    "TEMPORARY_CREDIT_LIMIT",
    "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT",
]

missing_columns = [
    column
    for column in required_columns
    if column not in credit_cards_raw_df.columns
]

if missing_columns:
    raise KeyError(
        "Credit-card query is missing required columns: {}".format(
            ", ".join(
                missing_columns
            )
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise identifiers
# Align card and entity IDs for the daily data.

credit_cards_raw_df["ENTITY_ID"] = (
    credit_cards_raw_df[
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

credit_cards_raw_df["CARD_ID"] = (
    credit_cards_raw_df[
        "CARD_ID"
    ]
    .astype("string")
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True,
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise date
# Use a consistent daily date for all records.

credit_cards_raw_df["MAP_DTE"] = (
    pd.to_datetime(
        credit_cards_raw_df[
            "MAP_DTE"
        ],
        errors="coerce",
        utc=True,
    )
    .dt.tz_localize(None)
    .dt.normalize()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise numeric fields
# Convert credit limits and outstanding balances into numbers.

for column in [
    "CREDIT_LIMIT",
    "TEMPORARY_CREDIT_LIMIT",
    "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT",
]:
    credit_cards_raw_df[column] = (
        pd.to_numeric(
            credit_cards_raw_df[
                column
            ],
            errors="coerce",
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Limit-source QA
# Show whether each card uses a month-start limit, first-available limit or has no limit.

limit_source_counts = (
    credit_cards_raw_df["LIMIT_SOURCE"]
    .fillna("MISSING")
    .value_counts(dropna=False)
)

print()
print("CREDIT CARD LIMIT SOURCE")
print("------------------------")
print("Month-start limit rows :", "{:,}".format(int(limit_source_counts.get("MONTH_START", 0))))
print("First-available rows   :", "{:,}".format(int(limit_source_counts.get("FIRST_AVAILABLE", 0))))
print("Missing limit rows     :", "{:,}".format(int(limit_source_counts.get("MISSING", 0))))


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Source QA before deduplication
# Check raw record counts, source balance signs and repeated card keys.

raw_rows = len(
    credit_cards_raw_df
)

raw_entities = (
    credit_cards_raw_df[
        "ENTITY_ID"
    ].nunique()
)

raw_cards = (
    credit_cards_raw_df[
        "CARD_ID"
    ].nunique()
)

raw_duplicate_rows = int(
    credit_cards_raw_df.duplicated(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
            "CARD_ID",
        ],
        keep=False,
    ).sum()
)

missing_entity_rows = int(
    credit_cards_raw_df[
        "ENTITY_ID"
    ].isna().sum()
)

missing_card_rows = int(
    credit_cards_raw_df[
        "CARD_ID"
    ].isna().sum()
)

source_outstanding = (
    credit_cards_raw_df[
        "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT"
    ]
)

negative_outstanding_rows = int(
    (
        source_outstanding < 0
    ).sum()
)

positive_outstanding_rows = int(
    (
        source_outstanding > 0
    ).sum()
)

zero_outstanding_rows = int(
    (
        source_outstanding == 0
    ).sum()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Apply calibrated credit-card definitions
# Use the higher credit limit and the absolute outstanding amount for each card.

credit_cards_raw_df[
    "OVERARCHING_CREDIT_LIMIT"
] = (
    credit_cards_raw_df[
        [
            "CREDIT_LIMIT",
            "TEMPORARY_CREDIT_LIMIT",
        ]
    ]
    .max(
        axis=1,
        skipna=True,
    )
)

# Accounting signs are removed before card-level deduplication.
credit_cards_raw_df[
    "CC_OUTSTANDING_AMOUNT"
] = (
    credit_cards_raw_df[
        "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT"
    ]
    .abs()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep one contribution per card and day
# Avoid counting the same card more than once for an entity on a day.

credit_card_day = (
    credit_cards_raw_df
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
            "CARD_ID",
        ],
        as_index=False,
    )
    .agg(
        CREDIT_LIMIT=(
            "CREDIT_LIMIT",
            "max",
        ),
        TEMPORARY_CREDIT_LIMIT=(
            "TEMPORARY_CREDIT_LIMIT",
            "max",
        ),
        CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT=(
            "CC_OUTSTANDING_AMOUNT",
            "max",
        ),
        OVERARCHING_CREDIT_LIMIT=(
            "OVERARCHING_CREDIT_LIMIT",
            "max",
        ),
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Entity-level utilisation QA
# Sum card balances and limits by entity, then calculate overall utilisation.

entity_utilization_df = (
    credit_card_day[
        [
            "ENTITY_ID",
            "MAP_DTE",
            "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT",
            "OVERARCHING_CREDIT_LIMIT",
        ]
    ]
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
        ],
        as_index=False,
    )
    .sum(
        min_count=1
    )
    .rename(
        columns={
            "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT":
                "CC_OUTSTANDING_AMOUNT",
            "OVERARCHING_CREDIT_LIMIT":
                "CC_CREDIT_LIMIT",
        }
    )
)

entity_utilization_df[
    "CC_LIMIT_UTILIZATION"
] = np.where(
    entity_utilization_df[
        "CC_CREDIT_LIMIT"
    ] > 0,
    (
        entity_utilization_df[
            "CC_OUTSTANDING_AMOUNT"
        ]
        /
        entity_utilization_df[
            "CC_CREDIT_LIMIT"
        ]
    ),
    np.nan,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Credit-card QA
# Check missing limits, missing balances and cards linked to several entities.

duplicate_card_day_rows = int(
    credit_card_day.duplicated(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
            "CARD_ID",
        ]
    ).sum()
)

cards_without_limit = int(
    credit_card_day[
        "OVERARCHING_CREDIT_LIMIT"
    ].isna().sum()
)

cards_with_non_positive_limit = int(
    (
        credit_card_day[
            "OVERARCHING_CREDIT_LIMIT"
        ]
        .notna()
        &
        (
            credit_card_day[
                "OVERARCHING_CREDIT_LIMIT"
            ] <= 0
        )
    ).sum()
)

cards_without_outstanding = int(
    credit_card_day[
        "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT"
    ].isna().sum()
)

cards_with_positive_outstanding = int(
    (
        credit_card_day[
            "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT"
        ] > 0
    ).sum()
)

card_entity_counts = (
    credit_card_day
    .groupby(
        "CARD_ID"
    )[
        "ENTITY_ID"
    ]
    .nunique()
)

cards_mapped_to_multiple_entities = int(
    (
        card_entity_counts > 1
    ).sum()
)

utilization = (
    entity_utilization_df[
        "CC_LIMIT_UTILIZATION"
    ]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
)

usable_utilization = (
    utilization.dropna()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA output
# Report coverage and the distribution of utilisation against the main thresholds.

print()
print("CREDIT CARDS QA")
print("----------------")

print(
    "Reference date:",
    REFERENCE_DATE_STR,
)

print()
print("SOURCE / JOIN RESULT")

print(
    "Raw rows:",
    "{:,}".format(
        raw_rows
    ),
)

print(
    "Raw entities:",
    "{:,}".format(
        raw_entities
    ),
)

print(
    "Raw cards:",
    "{:,}".format(
        raw_cards
    ),
)

print(
    "Rows belonging to duplicated entity/card/day keys:",
    "{:,}".format(
        raw_duplicate_rows
    ),
)

print(
    "Missing ENTITY_ID rows:",
    "{:,}".format(
        missing_entity_rows
    ),
)

print(
    "Missing CARD_ID rows:",
    "{:,}".format(
        missing_card_rows
    ),
)

print()
print("SOURCE OUTSTANDING SIGN")

print(
    "Negative outstanding rows:",
    "{:,}".format(
        negative_outstanding_rows
    ),
)

print(
    "Positive outstanding rows:",
    "{:,}".format(
        positive_outstanding_rows
    ),
)

print(
    "Zero outstanding rows:",
    "{:,}".format(
        zero_outstanding_rows
    ),
)

print()
print("CARD-DAY DATA")

print(
    "Card-day rows:",
    "{:,}".format(
        len(
            credit_card_day
        )
    ),
)

print(
    "Entities with credit cards:",
    "{:,}".format(
        credit_card_day[
            "ENTITY_ID"
        ].nunique()
    ),
)

print(
    "Unique cards:",
    "{:,}".format(
        credit_card_day[
            "CARD_ID"
        ].nunique()
    ),
)

print(
    "Duplicate card-day rows after aggregation:",
    "{:,}".format(
        duplicate_card_day_rows
    ),
)

print(
    "Cards without usable credit limit:",
    "{:,}".format(
        cards_without_limit
    ),
)

print(
    "Cards with non-positive credit limit:",
    "{:,}".format(
        cards_with_non_positive_limit
    ),
)

print(
    "Cards without outstanding amount:",
    "{:,}".format(
        cards_without_outstanding
    ),
)

print(
    "Cards with positive outstanding:",
    "{:,}".format(
        cards_with_positive_outstanding
    ),
)

print(
    "Cards mapped to multiple entities:",
    "{:,}".format(
        cards_mapped_to_multiple_entities
    ),
)

print()
print("ENTITY-LEVEL CC UTILISATION")

print(
    "Entities:",
    "{:,}".format(
        len(
            entity_utilization_df
        )
    ),
)

print(
    "Entities with usable utilisation:",
    "{:,}".format(
        usable_utilization.shape[0]
    ),
)

if not usable_utilization.empty:

    print(
        "Minimum utilisation: {:.4f}".format(
            usable_utilization.min()
        )
    )

    print(
        "Median utilisation: {:.4f}".format(
            usable_utilization.median()
        )
    )

    print(
        "90th percentile: {:.4f}".format(
            usable_utilization.quantile(
                0.90
            )
        )
    )

    print(
        "95th percentile: {:.4f}".format(
            usable_utilization.quantile(
                0.95
            )
        )
    )

    print(
        "99th percentile: {:.4f}".format(
            usable_utilization.quantile(
                0.99
            )
        )
    )

    print(
        "Maximum utilisation: {:.4f}".format(
            usable_utilization.max()
        )
    )

    print()

    for threshold in [
        0.80,
        0.90,
        0.94,
        0.97,
        1.00,
    ]:

        count = int(
            (
                usable_utilization
                > threshold
            ).sum()
        )

        percentage = (
            100.0
            * count
            / len(
                usable_utilization
            )
        )

        print(
            "Entities > {:.0f}% utilisation: {:,} ({:.2f}%)".format(
                threshold * 100,
                count,
                percentage,
            )
        )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate QA
# Reject missing keys, duplicate cards or conflicting entity mappings.

if missing_entity_rows > 0:
    raise ValueError(
        "Credit-card query returned rows without ENTITY_ID."
    )

if missing_card_rows > 0:
    raise ValueError(
        "Credit-card query returned rows without CARD_ID."
    )

if duplicate_card_day_rows > 0:
    raise ValueError(
        "Credit_Cards contains duplicate "
        "ENTITY_ID + MAP_DTE + CARD_ID rows."
    )

if cards_mapped_to_multiple_entities > 0:
    raise ValueError(
        "At least one credit card is mapped to more than "
        "one ENTITY_ID on the implementation reference date."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Prepare final output
# Keep the card-level balances and limits required by the behavioural calculation.

credit_cards_df = (
    credit_card_day[
        [
            "ENTITY_ID",
            "MAP_DTE",
            "CARD_ID",
            "CREDIT_LIMIT",
            "TEMPORARY_CREDIT_LIMIT",
            "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT",
        ]
    ]
    .copy()
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the daily credit-card source records.

dataiku.Dataset(
    "Credit_Cards"
).write_with_schema(
    credit_cards_df
)