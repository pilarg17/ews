# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import numpy as np
import pandas as pd
import time
from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Reference date

reference_df = dataiku.Dataset(
    "Implementation_Reference_Date_previous"
).get_dataframe()

if reference_df.empty:
    raise ValueError(
        "Implementation_Reference_Date_previous is empty."
    )

reference_dates = (
    pd.to_datetime(
        reference_df["REFERENCE_DATE"],
        errors="coerce",
    )
    .dropna()
    .dt.normalize()
    .drop_duplicates()
)

if len(reference_dates) != 1:
    raise ValueError(
        "Implementation_Reference_Date_previous must contain one valid reference date."
    )

REFERENCE_DATE = reference_dates.iloc[0]

REFERENCE_DATE_STR = REFERENCE_DATE.strftime(
    "%Y-%m-%d"
)

if "PREVIOUS_DATE" not in reference_df.columns:
    raise KeyError(
        "Implementation_Reference_Date_previous is missing PREVIOUS_DATE."
    )

previous_dates = (
    pd.to_datetime(
        reference_df["PREVIOUS_DATE"],
        errors="coerce",
    )
    .dropna()
    .dt.normalize()
    .drop_duplicates()
)

if len(previous_dates) != 1:
    raise ValueError(
        "Implementation_Reference_Date_previous must contain one valid previous date."
    )

PREVIOUS_DATE = previous_dates.iloc[0]

if PREVIOUS_DATE >= REFERENCE_DATE:
    raise ValueError(
        "Expected PREVIOUS_DATE < REFERENCE_DATE."
    )

PREVIOUS_DATE_STR = PREVIOUS_DATE.strftime(
    "%Y-%m-%d"
)

PREVIOUS_MONTH_START_DATE = PREVIOUS_DATE.replace(day=1)

PREVIOUS_MONTH_START_DATE_STR = PREVIOUS_MONTH_START_DATE.strftime(
    "%Y-%m-%d"
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Historical reference dates

# Previous 12 completed months.
previous_month = (
    REFERENCE_DATE.to_period("M") - 1
)

history_months = pd.period_range(
    end=previous_month,
    periods=12,
    freq="M",
)

HISTORY_DATES = [
    period.end_time.normalize()
    for period in history_months
]


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers

def normalize_id(series):
    return (
        series.astype("string")
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


def prepare_card_data(frame):
    frame = frame.copy()

    frame.columns = [
        column.upper()
        for column in frame.columns
    ]

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
        if column not in frame.columns
    ]

    if missing_columns:
        raise KeyError(
            "Credit-card data is missing required column(s): {}".format(
                ", ".join(
                    missing_columns
                )
            )
        )

    frame["ENTITY_ID"] = normalize_id(
        frame["ENTITY_ID"]
    )

    frame["CARD_ID"] = normalize_id(
        frame["CARD_ID"]
    )

    frame["MAP_DTE"] = normalize_date(
        frame["MAP_DTE"]
    )

    for column in [
        "CREDIT_LIMIT",
        "TEMPORARY_CREDIT_LIMIT",
        "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    return frame[
        required_columns
    ].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Resolve SQL sources

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
### Current credit-card data

current_cards = dataiku.Dataset(
    "Credit_Cards_Previous"
).get_dataframe()

current_cards = prepare_card_data(
    current_cards
)

# Exclude provisional cards carrying an effective EUR 1 credit limit.
_current_provisional = current_cards[["CREDIT_LIMIT", "TEMPORARY_CREDIT_LIMIT"]].max(axis=1, skipna=True).eq(1.0)
print("Excluded provisional EUR 1 cards (current):", int(_current_provisional.sum()))
current_cards = current_cards.loc[~_current_provisional].copy()

current_cards = current_cards[
    current_cards["MAP_DTE"].eq(
        REFERENCE_DATE
    )
].copy()

if current_cards.empty:
    raise ValueError(
        "Credit_Cards_Previous has no rows for reference date {}.".format(
            REFERENCE_DATE_STR
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Previous-business-day credit-card utilisation

previous_cards_query = """
WITH previous_usage AS (

    SELECT
        u.CARD_ID,
        u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT

    FROM {usage_table} u

    WHERE u.TEK_STRT_DTE < (DATE '{previous_date}' + 1)
      AND NVL(u.TEK_END_DTE, DATE '9999-12-31') >= (DATE '{previous_date}' + 1)
),

previous_cards AS (

    SELECT
        CARD_ID,
        CARD_OWNER_CUSTOMER_ID,
        CREDIT_LIMIT,
        TEMPORARY_CREDIT_LIMIT

    FROM (
        SELECT
            c.CARD_ID,
            c.CARD_OWNER_CUSTOMER_ID,
            c.CREDIT_LIMIT,
            c.TEMPORARY_CREDIT_LIMIT,
            ROW_NUMBER() OVER (
                PARTITION BY c.CARD_ID
                ORDER BY c.TEK_STRT_DTE DESC,
                         NVL(c.TEK_END_DTE, DATE '9999-12-31') DESC
            ) AS RN
        FROM {card_table} c
        WHERE c.TEK_STRT_DTE < (DATE '{previous_date}' + 1)
          AND NVL(c.TEK_END_DTE, DATE '9999-12-31') >= (DATE '{previous_date}' + 1)
          AND UPPER(TRIM(c.CARD_TYPE)) = 'CREDIT CARD'
    )
    WHERE RN = 1
),

previous_customers AS (

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

        WHERE cust.TEK_STRT_DTE < (DATE '{previous_date}' + 1)
          AND NVL(cust.TEK_END_DTE, DATE '9999-12-31') >= (DATE '{previous_date}' + 1)
    )

    WHERE RN = 1
)

SELECT
    cust.COUNTERPARTY_IDENTIFIER_GLOBE AS ENTITY_ID,
    DATE '{previous_date}' AS MAP_DTE,
    c.CARD_ID AS CARD_ID,

    c.CREDIT_LIMIT AS CREDIT_LIMIT,
    c.TEMPORARY_CREDIT_LIMIT AS TEMPORARY_CREDIT_LIMIT,

    u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT
        AS CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT

FROM previous_usage u

INNER JOIN previous_cards c
    ON TRIM(CAST(u.CARD_ID AS VARCHAR2(100)))
     = TRIM(CAST(c.CARD_ID AS VARCHAR2(100)))

INNER JOIN previous_customers cust
    ON TRIM(CAST(c.CARD_OWNER_CUSTOMER_ID AS VARCHAR2(100)))
     = TRIM(CAST(cust.CUSTOMER_ID AS VARCHAR2(100)))

WHERE cust.COUNTERPARTY_IDENTIFIER_GLOBE IS NOT NULL
""".format(
    previous_date=PREVIOUS_DATE_STR,
    previous_month_start=PREVIOUS_MONTH_START_DATE_STR,
    usage_table=usage_table,
    card_table=card_table,
    customer_table=customer_table,
)

previous_cards = SQLExecutor2(
    dataset=usage_dataset
).query_to_df(
    previous_cards_query
)

if previous_cards.empty:
    raise ValueError(
        "No credit-card observations were returned for previous date {}.".format(
            PREVIOUS_DATE_STR
        )
    )

previous_cards = prepare_card_data(
    previous_cards
)

_previous_provisional = previous_cards[["CREDIT_LIMIT", "TEMPORARY_CREDIT_LIMIT"]].max(axis=1, skipna=True).eq(1.0)
print("Excluded provisional EUR 1 cards (previous business day):", int(_previous_provisional.sum()))
previous_cards = previous_cards.loc[~_previous_provisional].copy()

previous_cards[
    "OVERARCHING_CREDIT_LIMIT"
] = (
    previous_cards[
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

previous_cards[
    "CC_OUTSTANDING_AMOUNT"
] = (
    previous_cards[
        "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT"
    ]
    .abs()
)

previous_card_level = (
    previous_cards
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
            "CARD_ID",
        ],
        as_index=False,
    )
    .agg(
        cc_card_outstanding_amount=(
            "CC_OUTSTANDING_AMOUNT",
            "max",
        ),
        cc_card_credit_limit=(
            "OVERARCHING_CREDIT_LIMIT",
            "max",
        ),
    )
)

previous_entity_utilization = (
    previous_card_level
    .groupby(
        "ENTITY_ID",
        as_index=False,
    )
    .agg(
        PREVIOUS_CC_OUTSTANDING_AMOUNT=(
            "cc_card_outstanding_amount",
            "sum",
        ),
        PREVIOUS_CC_CREDIT_LIMIT=(
            "cc_card_credit_limit",
            "sum",
        ),
    )
)

previous_entity_utilization[
    "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION"
] = np.where(
    previous_entity_utilization[
        "PREVIOUS_CC_CREDIT_LIMIT"
    ].gt(0),
    (
        previous_entity_utilization[
            "PREVIOUS_CC_OUTSTANDING_AMOUNT"
        ]
        /
        previous_entity_utilization[
            "PREVIOUS_CC_CREDIT_LIMIT"
        ]
    ),
    np.nan,
)

previous_entity_utilization = previous_entity_utilization[
    [
        "ENTITY_ID",
        "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION",
    ]
].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Previous 12 months

# One historical extraction for all twelve snapshots. Oracle receives dates once,
# preventing twelve separate remote query round-trips.
month_date_selects = [
    "SELECT DATE '{}' AS SNAPSHOT_DATE FROM dual".format(d.strftime("%Y-%m-%d"))
    for d in HISTORY_DATES
]

historical_sql_cte = """
WITH snapshot_dates AS (
    {snapshot_dates}
), raw_history AS (
    SELECT
        cust.COUNTERPARTY_IDENTIFIER_GLOBE AS ENTITY_ID,
        d.SNAPSHOT_DATE AS MAP_DTE,
        c.CARD_ID AS CARD_ID,
        c.CREDIT_LIMIT AS CREDIT_LIMIT,
        c.TEMPORARY_CREDIT_LIMIT AS TEMPORARY_CREDIT_LIMIT,
        u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT
            AS CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT
    FROM snapshot_dates d
    INNER JOIN {usage_table} u
        ON u.TEK_STRT_DTE < (d.SNAPSHOT_DATE + 1)
       AND NVL(u.TEK_END_DTE, DATE '9999-12-31') >= (d.SNAPSHOT_DATE + 1)
    INNER JOIN {card_table} c
        ON TRIM(CAST(u.CARD_ID AS VARCHAR2(100)))
         = TRIM(CAST(c.CARD_ID AS VARCHAR2(100)))
       AND c.TEK_STRT_DTE < (d.SNAPSHOT_DATE + 1)
       AND NVL(c.TEK_END_DTE, DATE '9999-12-31') >= (d.SNAPSHOT_DATE + 1)
       AND UPPER(TRIM(c.CARD_TYPE)) = 'CREDIT CARD'
    INNER JOIN {customer_table} cust
        ON TRIM(CAST(c.CARD_OWNER_CUSTOMER_ID AS VARCHAR2(100)))
         = TRIM(CAST(cust.CUSTOMER_ID AS VARCHAR2(100)))
       AND cust.TEK_STRT_DTE < (d.SNAPSHOT_DATE + 1)
       AND NVL(cust.TEK_END_DTE, DATE '9999-12-31') >= (d.SNAPSHOT_DATE + 1)
    WHERE cust.COUNTERPARTY_IDENTIFIER_GLOBE IS NOT NULL
)
""".format(
    snapshot_dates=" UNION ALL ".join(month_date_selects),
    usage_table=usage_table,
    card_table=card_table,
    customer_table=customer_table,
)

historical_agg_sql = historical_sql_cte + """
SELECT
    ENTITY_ID,
    MAP_DTE,
    CARD_ID,
    MAX(CREDIT_LIMIT) AS CREDIT_LIMIT,
    MAX(TEMPORARY_CREDIT_LIMIT) AS TEMPORARY_CREDIT_LIMIT,
    MAX(CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT)
        AS CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT,
    COUNT(*) AS SOURCE_CARD_ROWS
FROM raw_history
GROUP BY ENTITY_ID, MAP_DTE, CARD_ID
"""

history_query_start = time.perf_counter()
historical_card_rows = SQLExecutor2(dataset=usage_dataset).query_to_df(historical_agg_sql)
print("CC 12-month SQL aggregated query seconds:",
      round(time.perf_counter() - history_query_start, 1), flush=True)

if historical_card_rows.empty:
    historical_frames = []
else:
    # The old algorithm keeps the last raw row per card/month before aggregation.
    # Pushing MAX down is equivalent only if each card/month has one source row.
    # If a collision exists, retrieve raw rows and let the unchanged pandas
    # processing choose the row, instead of silently changing financial values.
    historical_card_rows.columns = [c.upper() for c in historical_card_rows.columns]
    duplicate_card_groups = int(
        pd.to_numeric(historical_card_rows["SOURCE_CARD_ROWS"], errors="coerce")
        .fillna(0).gt(1).sum()
    )
    if duplicate_card_groups:
        print("CC historical duplicate card groups:", duplicate_card_groups,
              "- using full raw rows to preserve original selection logic.", flush=True)
        raw_sql = historical_sql_cte + """
SELECT
    ENTITY_ID, MAP_DTE, CARD_ID, CREDIT_LIMIT,
    TEMPORARY_CREDIT_LIMIT,
    CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT
FROM raw_history
ORDER BY MAP_DTE
"""
        raw_started = time.perf_counter()
        historical_card_rows = SQLExecutor2(dataset=usage_dataset).query_to_df(raw_sql)
        print("CC raw historical fallback SQL seconds:",
              round(time.perf_counter() - raw_started, 1), flush=True)
    else:
        historical_card_rows = historical_card_rows.drop(columns=["SOURCE_CARD_ROWS"])
    _historical_clean = prepare_card_data(historical_card_rows)
    _historical_provisional = _historical_clean[["CREDIT_LIMIT", "TEMPORARY_CREDIT_LIMIT"]].max(axis=1, skipna=True).eq(1.0)
    print("Excluded provisional EUR 1 historical card rows:", int(_historical_provisional.sum()))
    historical_frames = [_historical_clean.loc[~_historical_provisional].copy()]

overall_start_time = history_query_start


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Combine current and historical observations

frames = historical_frames + [
    current_cards
]

if not frames:
    raise ValueError(
        "No credit-card observations are available."
    )

credit_cards = pd.concat(
    frames,
    ignore_index=True,
)

credit_cards = (
    credit_cards
    .drop_duplicates(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
            "CARD_ID",
        ],
        keep="last",
    )
    .sort_values(
        [
            "ENTITY_ID",
            "MAP_DTE",
            "CARD_ID",
        ]
    )
    .reset_index(
        drop=True
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Credit-card utilisation

credit_cards[
    "OVERARCHING_CREDIT_LIMIT"
] = (
    credit_cards[
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

credit_cards[
    "CC_OUTSTANDING_AMOUNT"
] = (
    credit_cards[
        "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT"
    ]
    .abs()
)


# One contribution per card and date.
card_level = (
    credit_cards
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
            "CARD_ID",
        ],
        as_index=False,
    )
    .agg(
        cc_card_outstanding_amount=(
            "CC_OUTSTANDING_AMOUNT",
            "max",
        ),
        cc_card_credit_limit=(
            "OVERARCHING_CREDIT_LIMIT",
            "max",
        ),
    )
)


# Aggregate all cards held by the entity.
cc_history = (
    card_level
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
        ],
        as_index=False,
    )
    .agg(
        cc_outstanding_amount=(
            "cc_card_outstanding_amount",
            "sum",
        ),
        cc_credit_limit=(
            "cc_card_credit_limit",
            "sum",
        ),
    )
)

cc_history[
    "cc_limit_utilization"
] = np.where(
    cc_history[
        "cc_credit_limit"
    ].gt(0),
    (
        cc_history[
            "cc_outstanding_amount"
        ]
        /
        cc_history[
            "cc_credit_limit"
        ]
    ),
    np.nan,
)

cc_history = (
    cc_history
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
### Previous-month utilisation

previous_date = (
    cc_history
    .groupby(
        "ENTITY_ID"
    )[
        "MAP_DTE"
    ]
    .shift(1)
)

previous_utilization = (
    cc_history
    .groupby(
        "ENTITY_ID"
    )[
        "cc_limit_utilization"
    ]
    .shift(1)
)

month_gap = (
    (
        cc_history[
            "MAP_DTE"
        ].dt.year
        -
        previous_date.dt.year
    )
    * 12
    +
    (
        cc_history[
            "MAP_DTE"
        ].dt.month
        -
        previous_date.dt.month
    )
)

cc_history[
    "cc_diff_limit_util"
] = np.where(
    month_gap.eq(1),
    (
        cc_history[
            "cc_limit_utilization"
        ]
        -
        previous_utilization
    ),
    np.nan,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Previous 3, 6 and 12 months

_utilisation_groups = cc_history.groupby("ENTITY_ID", sort=False)["cc_limit_utilization"]

for months in [
    3,
    6,
    12,
]:

    average_column = (
        "cc_average_utilization_{}m".format(
            months
        )
    )

    difference_column = (
        "cc_diff_limit_util_{}m".format(
            months
        )
    )

    maximum_column = (
        "cc_max_utilization_{}m".format(
            months
        )
    )

    maximum_difference_column = (
        "cc_diff_limit_util_{}m_max".format(
            months
        )
    )

    cc_history[
        average_column
    ] = (
        _utilisation_groups.rolling(
            months,
            min_periods=1,
            closed="left",
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    cc_history[
        difference_column
    ] = (
        cc_history[
            "cc_limit_utilization"
        ]
        -
        cc_history[
            average_column
        ]
    )

    cc_history[
        maximum_column
    ] = (
        _utilisation_groups.rolling(
            months,
            min_periods=1,
            closed="left",
        )
        .max()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    cc_history[
        maximum_difference_column
    ] = (
        cc_history[
            "cc_limit_utilization"
        ]
        -
        cc_history[
            maximum_column
        ]
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current reference date

cc_history = cc_history.merge(
    previous_entity_utilization,
    how="left",
    on="ENTITY_ID",
    validate="m:1",
)

cc_history[
    "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION"
] = np.where(
    cc_history["MAP_DTE"].eq(REFERENCE_DATE),
    cc_history[
        "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION"
    ],
    np.nan,
)

cc_history[
    "CC_UTILIZATION_DAILY_CHANGE"
] = np.where(
    cc_history["MAP_DTE"].eq(REFERENCE_DATE),
    (
        cc_history["cc_limit_utilization"]
        -
        cc_history[
            "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION"
        ]
    ),
    np.nan,
)

cc_history[
    "CC_UTILIZATION_DAILY_WORSENING"
] = (
    cc_history["MAP_DTE"].eq(REFERENCE_DATE)
    & cc_history["cc_limit_utilization"].notna()
    & cc_history[
        "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION"
    ].notna()
    & cc_history["cc_limit_utilization"].gt(
        cc_history[
            "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION"
        ]
    )
).astype("int8")

current_history = cc_history[
    cc_history[
        "MAP_DTE"
    ].eq(
        REFERENCE_DATE
    )
].copy()

if current_history.empty:
    raise ValueError(
        "No credit-card utilisation was calculated "
        "for reference date {}.".format(
            REFERENCE_DATE_STR
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA

duplicate_rows = int(
    cc_history.duplicated(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
        ]
    ).sum()
)

print()
print("CC UTILISATION HISTORY QA")
print("-------------------------")

print(
    "Reference date:",
    REFERENCE_DATE_STR,
)

print(
    "Historical months:",
    len(
        HISTORY_DATES
    ),
)

print(
    "Rows:",
    "{:,}".format(
        len(
            cc_history
        )
    ),
)

print(
    "Entities:",
    "{:,}".format(
        cc_history[
            "ENTITY_ID"
        ].nunique()
    ),
)

print(
    "Current entities:",
    "{:,}".format(
        current_history[
            "ENTITY_ID"
        ].nunique()
    ),
)

print(
    "Current utilisation available:",
    "{:,}".format(
        current_history[
            "cc_limit_utilization"
        ].notna().sum()
    ),
)

print(
    "Current 12m average available:",
    "{:,}".format(
        current_history[
            "cc_average_utilization_12m"
        ].notna().sum()
    ),
)

print(
    "Current 12m max available:",
    "{:,}".format(
        current_history[
            "cc_max_utilization_12m"
        ].notna().sum()
    ),
)

print(
    "Duplicate ENTITY_ID + MAP_DTE:",
    duplicate_rows,
)

print(
    "Runtime: {:.2f} minutes".format(
        (
            time.time()
            -
            overall_start_time
        )
        / 60
    )
)


if duplicate_rows > 0:
    raise ValueError(
        "CC_Utilisation_History contains duplicate "
        "ENTITY_ID + MAP_DTE rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final columns

final_columns = [
    "ENTITY_ID",
    "MAP_DTE",
    "cc_outstanding_amount",
    "cc_credit_limit",
    "cc_limit_utilization",
    "cc_diff_limit_util",
    "cc_average_utilization_3m",
    "cc_diff_limit_util_3m",
    "cc_average_utilization_6m",
    "cc_diff_limit_util_6m",
    "cc_average_utilization_12m",
    "cc_diff_limit_util_12m",
    "cc_max_utilization_3m",
    "cc_diff_limit_util_3m_max",
    "cc_max_utilization_6m",
    "cc_diff_limit_util_6m_max",
    "cc_max_utilization_12m",
    "cc_diff_limit_util_12m_max",
    "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION",
    "CC_UTILIZATION_DAILY_CHANGE",
    "CC_UTILIZATION_DAILY_WORSENING",
]

cc_history = cc_history[
    final_columns
].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output

_write_started = time.perf_counter()

dataiku.Dataset(
    "CC_Utilisation_History_Previous"
).write_with_schema(
    cc_history
)

print("Credit-card history write seconds:", round(time.perf_counter() - _write_started, 1))
