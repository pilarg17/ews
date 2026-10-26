# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import dataiku
import numpy as np
import pandas as pd
import time

recipe_start = time.perf_counter()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read inputs
# Read the date that the behavioural features should represent.

reference_df = dataiku.Dataset("Implementation_Reference_Date").get_dataframe()

if reference_df.empty:
    raise ValueError("Implementation_Reference_Date is empty.")
REFERENCE_DATE = pd.to_datetime(reference_df["REFERENCE_DATE"].iloc[0], errors="raise").normalize()
REFERENCE_DATE_STR = REFERENCE_DATE.strftime("%Y-%m-%d")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### SQL pushdown for daily inputs
# Aggregate daily records in the database when the source supports it, reducing the data transferred to Python.

from dataiku.core.sql import SQLExecutor2
import time


def sql_numeric_type(type_name):
    t = str(type_name).lower()
    return any(k in t for k in ("int", "float", "double", "decimal", "numeric", "bigint", "smallint", "tinyint", "long"))


def load_daily_compact(dataset_name, reference_date, mode, numeric_columns=()):
    dataset = dataiku.Dataset(dataset_name)
    schema = {col["name"].upper(): col for col in dataset.read_schema()}
    required = ["ENTITY_ID", "MAP_DTE"]
    missing = [c for c in required if c not in schema]
    if missing:
        raise KeyError("{} is missing columns {}".format(dataset_name, missing))

    def col(name):
        value = schema[name.upper()]["name"]
        return 't."{}"'.format(value.replace('"', '""'))

    def alias(name):
        return '"{}"'.format(name.replace('"', '""'))

    location = dataset.get_location_info()
    info = location.get("info", {}) or {}
    table = info.get("table") or info.get("tableName")
    db_schema = info.get("schema") or info.get("schemaName")
    all_numeric = all(
        c.upper() in schema and sql_numeric_type(schema[c.upper()].get("type", ""))
        for c in numeric_columns
    )
    if not table or not all_numeric:
        why = "non-SQL dataset" if not table else "numeric column not SQL numeric"
        print("{}: pandas fallback ({})".format(dataset_name, why), flush=True)
        return dataset.get_dataframe()

    table_full = "{}.{}".format(db_schema, table) if db_schema and "." not in table else table
    start_date = reference_date.strftime("%Y-%m-%d")
    end_date = (reference_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    id_expr = "REGEXP_REPLACE(TRIM(TO_CHAR({})), '[.]0$', '')".format(col("ENTITY_ID"))
    card_expr = None
    if mode == "cards":
        card_expr = "REGEXP_REPLACE(TRIM(TO_CHAR({})), '[.]0$', '')".format(col("CARD_ID"))
    w = "{map_dte} >= DATE '{start}' AND {map_dte} < DATE '{end}' AND {entity_id} IS NOT NULL".format(
        map_dte=col("MAP_DTE"), start=start_date, end=end_date, entity_id=col("ENTITY_ID")
    )

    if mode == "entity":
        exprs = ["DISTINCT {} AS {}".format(id_expr, alias("ENTITY_ID"))]
        group_by = ""
    elif mode == "cards":
        exprs = [
            "{} AS {}".format(id_expr, alias("ENTITY_ID")),
            "{} AS {}".format(card_expr, alias("CARD_ID")),
            "MAX({}) AS {}".format(col("CREDIT_LIMIT"), alias("CREDIT_LIMIT")),
            "MAX({}) AS {}".format(col("TEMPORARY_CREDIT_LIMIT"), alias("TEMPORARY_CREDIT_LIMIT")),
            "MAX(ABS({})) AS {}".format(col("CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT"), alias("CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT")),
        ]
        group_by = " GROUP BY {}, {}".format(id_expr, card_expr)
    else:
        cols = {
            "balances": [("MAX", "ASSETS")],
            "cash": [("SUM", "TRANSACTION_COUNT"), ("SUM", "TOTAL_OPERATION_AMOUNT")],
            "neobank": [("SUM", "neobank_topup_count"), ("SUM", "neobank_topup_amount")],
            "cash_limit": [("MAX", c) for c in (
                "TRANSACTION_LIMIT_UTILIZATION", "LIMIT_UTILIZATION_BEFORE_TRANSACTION",
                "LIMIT_UTILIZATION_AFTER_TRANSACTION", "RRE_CASH_ADVANCE_LIMIT_CROSS_94_1D",
                "OTHER_CASH_ADVANCE_LIMIT_CROSS_97_1D")],
            "neobank_limit": [("MAX", c) for c in (
                "TRANSACTION_LIMIT_UTILIZATION", "LIMIT_UTILIZATION_BEFORE_TRANSACTION",
                "LIMIT_UTILIZATION_AFTER_TRANSACTION")],
        }[mode]
        exprs = ["{} AS {}".format(id_expr, alias("ENTITY_ID"))]
        exprs += ["{}({}) AS {}".format(op, col(name), alias(name)) for op, name in cols]
        group_by = " GROUP BY {}".format(id_expr)

    query = "SELECT {} FROM {} t WHERE {}{}".format(
        ", ".join(exprs), table_full, w, group_by
    )
    started = time.monotonic()
    try:
        df = SQLExecutor2(dataset=dataset).query_to_df(query)
    except Exception as exc:
        # Report SQL errors rather than loading a full table unexpectedly.
        raise RuntimeError("SQL pushdown failed on {}: {}".format(dataset_name, exc)) from exc
    elapsed = time.monotonic() - started
    df.columns = [c.upper() if c.upper() in ("ENTITY_ID", "MAP_DTE", "CARD_ID", "ASSETS", "CREDIT_LIMIT", "TEMPORARY_CREDIT_LIMIT", "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT", "TRANSACTION_COUNT", "TOTAL_OPERATION_AMOUNT", "TRANSACTION_LIMIT_UTILIZATION", "LIMIT_UTILIZATION_BEFORE_TRANSACTION", "LIMIT_UTILIZATION_AFTER_TRANSACTION", "RRE_CASH_ADVANCE_LIMIT_CROSS_94_1D", "OTHER_CASH_ADVANCE_LIMIT_CROSS_97_1D") else c for c in df.columns]
    if mode == "neobank":
        expected_columns = ("neobank_topup_count", "neobank_topup_amount")
        rename_columns = {}
        for expected in expected_columns:
            matches = [name for name in df.columns if str(name).casefold() == expected.casefold()]
            if len(matches) != 1:
                raise ValueError(
                    "{}: expected exactly one column for {}. Actual columns: {}".format(
                        dataset_name, expected, list(df.columns)
                    )
                )
            rename_columns[matches[0]] = expected
        df = df.rename(columns=rename_columns)
    df["MAP_DTE"] = reference_date
    print("{}: SQL pushdown {:.2f}s, {:,} rows".format(dataset_name, elapsed, len(df)), flush=True)
    return df

entity_df = load_daily_compact('Entity_Information', REFERENCE_DATE, 'entity', [])
balances_df = load_daily_compact('Customer_Balances', REFERENCE_DATE, 'balances', ['ASSETS'])
credit_cards_df = load_daily_compact('Credit_Cards', REFERENCE_DATE, 'cards', ['CREDIT_LIMIT', 'TEMPORARY_CREDIT_LIMIT', 'CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT'])
cash_advance_df = load_daily_compact('cash_advance_data', REFERENCE_DATE, 'cash', ['TRANSACTION_COUNT', 'TOTAL_OPERATION_AMOUNT'])
neobank_df = load_daily_compact('CC_Transactions_Neobanks', REFERENCE_DATE, 'neobank', ['neobank_topup_count', 'neobank_topup_amount'])
cash_limit_df = load_daily_compact('Cash_Advance_CC_Limit', REFERENCE_DATE, 'cash_limit', ['TRANSACTION_LIMIT_UTILIZATION', 'LIMIT_UTILIZATION_BEFORE_TRANSACTION', 'LIMIT_UTILIZATION_AFTER_TRANSACTION', 'RRE_CASH_ADVANCE_LIMIT_CROSS_94_1D', 'OTHER_CASH_ADVANCE_LIMIT_CROSS_97_1D'])
neobank_limit_df = load_daily_compact('Neobank_Transfers_CC_Limit', REFERENCE_DATE, 'neobank_limit', ['TRANSACTION_LIMIT_UTILIZATION', 'LIMIT_UTILIZATION_BEFORE_TRANSACTION', 'LIMIT_UTILIZATION_AFTER_TRANSACTION'])


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers
# Keep identifiers and dates aligned across the input datasets.

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


def prepare_entity_date(frame):
    frame = frame.copy()

    if "ENTITY_ID" in frame.columns:
        frame["ENTITY_ID"] = normalize_id(
            frame["ENTITY_ID"]
        )

    if "MAP_DTE" in frame.columns:
        frame["MAP_DTE"] = normalize_date(
            frame["MAP_DTE"]
        )

    return frame


def keep_reference_date(
    frame,
    dataset_name,
    reference_date,
):
    if frame.empty:
        return frame.copy()

    if "MAP_DTE" not in frame.columns:
        raise KeyError(
            "{} is missing MAP_DTE.".format(
                dataset_name
            )
        )

    available_dates = (
        frame["MAP_DTE"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if reference_date not in set(
        available_dates
    ):
        raise ValueError(
            "{} does not contain the implementation "
            "reference date {}. Available dates: {}".format(
                dataset_name,
                reference_date.strftime(
                    "%Y-%m-%d"
                ),
                [
                    date.strftime(
                        "%Y-%m-%d"
                    )
                    for date in available_dates
                ],
            )
        )

    return frame[
        frame["MAP_DTE"]
        == reference_date
    ].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Reference date
# The daily date is already defined before loading the sources.

# REFERENCE_DATE is defined before SQL pushdown.


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise identifiers and dates
# Prepare all input datasets for merging.

entity_df = prepare_entity_date(
    entity_df
)

balances_df = prepare_entity_date(
    balances_df
)

credit_cards_df = prepare_entity_date(
    credit_cards_df
)

cash_advance_df = prepare_entity_date(
    cash_advance_df
)

neobank_df = prepare_entity_date(
    neobank_df
)

cash_limit_df = prepare_entity_date(
    cash_limit_df
)

neobank_limit_df = prepare_entity_date(
    neobank_limit_df
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Keep reference date
# Use only the records belonging to the current daily run.

entity_df = keep_reference_date(
    entity_df,
    "Entity_Information",
    REFERENCE_DATE,
)

balances_df = keep_reference_date(
    balances_df,
    "Customer_Balances",
    REFERENCE_DATE,
)

credit_cards_df = keep_reference_date(
    credit_cards_df,
    "Credit_Cards",
    REFERENCE_DATE,
)

cash_advance_df = keep_reference_date(
    cash_advance_df,
    "cash_advance_data",
    REFERENCE_DATE,
)

neobank_df = keep_reference_date(
    neobank_df,
    "CC_Transactions_Neobanks",
    REFERENCE_DATE,
)

cash_limit_df = keep_reference_date(
    cash_limit_df,
    "Cash_Advance_CC_Limit",
    REFERENCE_DATE,
)

neobank_limit_df = keep_reference_date(
    neobank_limit_df,
    "Neobank_Transfers_CC_Limit",
    REFERENCE_DATE,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Base entity population
# Start with one record per entity and date.

required_entity_columns = [
    "ENTITY_ID",
    "MAP_DTE",
]

missing = [
    column
    for column in required_entity_columns
    if column not in entity_df.columns
]

if missing:
    raise KeyError(
        "Entity_Information is missing: {}".format(
            ", ".join(missing)
        )
    )

base = (
    entity_df[
        [
            "ENTITY_ID",
            "MAP_DTE",
        ]
    ]
    .dropna(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
        ]
    )
    .drop_duplicates()
)

if base.duplicated(
    subset=[
        "ENTITY_ID",
        "MAP_DTE",
    ]
).any():
    raise ValueError(
        "Base entity population is not unique "
        "by ENTITY_ID + MAP_DTE."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Customer balances / assets
# Prepare the balance used in behavioural ratios.

if "ASSETS" not in balances_df.columns:
    raise KeyError(
        "Customer_Balances is missing ASSETS."
    )

balances_agg = (
    balances_df[
        [
            "ENTITY_ID",
            "MAP_DTE",
            "ASSETS",
        ]
    ]
    .assign(
        ASSETS=lambda x: pd.to_numeric(
            x["ASSETS"],
            errors="coerce",
        )
    )
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
        ],
        as_index=False,
    )
    .agg(
        ASSETS=(
            "ASSETS",
            "max",
        )
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Credit-card utilisation
# Aggregate outstanding amounts and credit limits by card, then by entity.

for column in [
    "CREDIT_LIMIT",
    "TEMPORARY_CREDIT_LIMIT",
    "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT",
]:
    credit_cards_df[column] = (
        pd.to_numeric(
            credit_cards_df[column],
            errors="coerce",
        )
    )

credit_cards_df[
    "OVERARCHING_CREDIT_LIMIT"
] = (
    credit_cards_df[
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

credit_cards_df[
    "CC_OUTSTANDING_AMOUNT"
] = (
    credit_cards_df[
        "CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT"
    ].abs()
)

credit_card_day = (
    credit_cards_df
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

credit_card_entity = (
    credit_card_day
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

credit_card_entity[
    "cc_limit_utilization"
] = np.where(
    credit_card_entity[
        "cc_credit_limit"
    ].gt(0),
    (
        credit_card_entity[
            "cc_outstanding_amount"
        ]
        /
        credit_card_entity[
            "cc_credit_limit"
        ]
    ),
    np.nan,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Cash-advance activity
# Summarise cash withdrawals and their amounts by entity.

cash_advance_df[
    "TRANSACTION_COUNT"
] = (
    pd.to_numeric(
        cash_advance_df[
            "TRANSACTION_COUNT"
        ],
        errors="coerce",
    )
    .fillna(0)
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
    .fillna(0.0)
)

cash_advance_entity = (
    cash_advance_df
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
        ],
        as_index=False,
    )
    .agg(
        cash_advance_count=(
            "TRANSACTION_COUNT",
            "sum",
        ),
        cash_advance_source_amount=(
            "TOTAL_OPERATION_AMOUNT",
            "sum",
        ),
    )
)

cash_advance_entity[
    "cash_advance_amount"
] = (
    -cash_advance_entity[
        "cash_advance_source_amount"
    ]
).clip(
    lower=0
)

cash_advance_entity.drop(
    columns=[
        "cash_advance_source_amount"
    ],
    inplace=True,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Neobank activity
# Summarise top-ups and transfers to neobanks.

neobank_df[
    "neobank_topup_count"
] = (
    pd.to_numeric(
        neobank_df[
            "neobank_topup_count"
        ],
        errors="coerce",
    )
    .fillna(0)
)

neobank_df[
    "neobank_topup_amount"
] = (
    pd.to_numeric(
        neobank_df[
            "neobank_topup_amount"
        ],
        errors="coerce",
    )
    .fillna(0.0)
)

neobank_entity = (
    neobank_df
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
        ],
        as_index=False,
    )
    .agg(
        neobank_topup_count=(
            "neobank_topup_count",
            "sum",
        ),
        neobank_topup_source_amount=(
            "neobank_topup_amount",
            "sum",
        ),
    )
)

neobank_entity[
    "neobank_topup_amount"
] = (
    -neobank_entity[
        "neobank_topup_source_amount"
    ]
).clip(
    lower=0
)

neobank_entity.drop(
    columns=[
        "neobank_topup_source_amount"
    ],
    inplace=True,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Cash-advance limit utilisation
# Keep utilisation before and after transactions and their crossing flags.

required_cash_limit_columns = [
    "ENTITY_ID",
    "MAP_DTE",
    "TRANSACTION_LIMIT_UTILIZATION",
    "LIMIT_UTILIZATION_BEFORE_TRANSACTION",
    "LIMIT_UTILIZATION_AFTER_TRANSACTION",
    "RRE_CASH_ADVANCE_LIMIT_CROSS_94_1D",
    "OTHER_CASH_ADVANCE_LIMIT_CROSS_97_1D",
]

missing_cash_limit_columns = [
    column
    for column in required_cash_limit_columns
    if column not in cash_limit_df.columns
]

if missing_cash_limit_columns:
    raise KeyError(
        "Cash_Advance_CC_Limit is missing required column(s): {}".format(
            ", ".join(missing_cash_limit_columns)
        )
    )

for column in [
    "TRANSACTION_LIMIT_UTILIZATION",
    "LIMIT_UTILIZATION_BEFORE_TRANSACTION",
    "LIMIT_UTILIZATION_AFTER_TRANSACTION",
    "RRE_CASH_ADVANCE_LIMIT_CROSS_94_1D",
    "OTHER_CASH_ADVANCE_LIMIT_CROSS_97_1D",
]:
    cash_limit_df[column] = pd.to_numeric(
        cash_limit_df[column],
        errors="coerce",
    )

cash_limit_entity = (
    cash_limit_df
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
        ],
        as_index=False,
    )
    .agg(
        cash_advance_transaction_limit_utilization=(
            "TRANSACTION_LIMIT_UTILIZATION",
            "max",
        ),
        cash_advance_limit_utilization_before_transaction=(
            "LIMIT_UTILIZATION_BEFORE_TRANSACTION",
            "max",
        ),
        cash_advance_limit_utilization_after_transaction=(
            "LIMIT_UTILIZATION_AFTER_TRANSACTION",
            "max",
        ),
        cash_advance_limit_cross_94_1d=(
            "RRE_CASH_ADVANCE_LIMIT_CROSS_94_1D",
            "max",
        ),
        cash_advance_limit_cross_97_1d=(
            "OTHER_CASH_ADVANCE_LIMIT_CROSS_97_1D",
            "max",
        ),
    )
)

for column in [
    "cash_advance_limit_cross_94_1d",
    "cash_advance_limit_cross_97_1d",
]:
    cash_limit_entity[column] = (
        pd.to_numeric(
            cash_limit_entity[column],
            errors="coerce",
        )
        .fillna(0)
        .astype("int8")
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Neobank limit utilisation
# Prepare card-limit usage associated with neobank transfers.

for column in [
    "TRANSACTION_LIMIT_UTILIZATION",
    "LIMIT_UTILIZATION_BEFORE_TRANSACTION",
    "LIMIT_UTILIZATION_AFTER_TRANSACTION",
]:
    neobank_limit_df[column] = (
        pd.to_numeric(
            neobank_limit_df[column],
            errors="coerce",
        )
    )

neobank_limit_entity = (
    neobank_limit_df
    .groupby(
        [
            "ENTITY_ID",
            "MAP_DTE",
        ],
        as_index=False,
    )
    .agg(
        neobank_transaction_limit_utilization=(
            "TRANSACTION_LIMIT_UTILIZATION",
            "max",
        ),
        neobank_limit_utilization_before_transaction=(
            "LIMIT_UTILIZATION_BEFORE_TRANSACTION",
            "max",
        ),
        neobank_limit_utilization_after_transaction=(
            "LIMIT_UTILIZATION_AFTER_TRANSACTION",
            "max",
        ),
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Merge daily behavioural features
# Bring balances, card activity and transaction features together for each entity.

features = (
    base
    .merge(
        balances_agg,
        how="left",
        on=[
            "ENTITY_ID",
            "MAP_DTE",
        ],
        validate="1:1",
    )
    .merge(
        credit_card_entity,
        how="left",
        on=[
            "ENTITY_ID",
            "MAP_DTE",
        ],
        validate="1:1",
    )
    .merge(
        cash_advance_entity,
        how="left",
        on=[
            "ENTITY_ID",
            "MAP_DTE",
        ],
        validate="1:1",
    )
    .merge(
        neobank_entity,
        how="left",
        on=[
            "ENTITY_ID",
            "MAP_DTE",
        ],
        validate="1:1",
    )
    .merge(
        cash_limit_entity,
        how="left",
        on=[
            "ENTITY_ID",
            "MAP_DTE",
        ],
        validate="1:1",
    )
    .merge(
        neobank_limit_entity,
        how="left",
        on=[
            "ENTITY_ID",
            "MAP_DTE",
        ],
        validate="1:1",
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Derived behavioural variables
# Calculate combined activity and ratios against positive balances.

for column in [
    "cash_advance_count",
    "cash_advance_amount",
    "neobank_topup_count",
    "neobank_topup_amount",
]:
    features[column] = (
        pd.to_numeric(
            features[column],
            errors="coerce",
        )
        .fillna(0.0)
    )

features[
    "cash_neobank_count"
] = (
    features[
        "cash_advance_count"
    ]
    +
    features[
        "neobank_topup_count"
    ]
)

features[
    "cash_neobank_amount"
] = (
    features[
        "cash_advance_amount"
    ]
    +
    features[
        "neobank_topup_amount"
    ]
)

features[
    "cash_advance_balance_ratio"
] = np.where(
    pd.to_numeric(
        features["ASSETS"],
        errors="coerce",
    ).gt(0),
    (
        features[
            "cash_advance_amount"
        ]
        /
        pd.to_numeric(
            features["ASSETS"],
            errors="coerce",
        )
    ),
    np.nan,
)

features[
    "neobank_topup_balance_ratio"
] = np.where(
    pd.to_numeric(
        features["ASSETS"],
        errors="coerce",
    ).gt(0),
    (
        features[
            "neobank_topup_amount"
        ]
        /
        pd.to_numeric(
            features["ASSETS"],
            errors="coerce",
        )
    ),
    np.nan,
)

features[
    "cash_neobank_balance_ratio"
] = np.where(
    pd.to_numeric(
        features["ASSETS"],
        errors="coerce",
    ).gt(0),
    (
        features[
            "cash_neobank_amount"
        ]
        /
        pd.to_numeric(
            features["ASSETS"],
            errors="coerce",
        )
    ),
    np.nan,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Final columns
# Keep only the fields needed by the risk indicators.

final_columns = [
    "ENTITY_ID",
    "MAP_DTE",
    "ASSETS",
    "cc_outstanding_amount",
    "cc_credit_limit",
    "cc_limit_utilization",
    "cash_advance_count",
    "cash_advance_amount",
    "neobank_topup_count",
    "neobank_topup_amount",
    "cash_neobank_count",
    "cash_neobank_amount",
    "cash_advance_balance_ratio",
    "neobank_topup_balance_ratio",
    "cash_neobank_balance_ratio",
    "cash_advance_transaction_limit_utilization",
    "cash_advance_limit_utilization_before_transaction",
    "cash_advance_limit_utilization_after_transaction",
    "cash_advance_limit_cross_94_1d",
    "cash_advance_limit_cross_97_1d",
    "neobank_transaction_limit_utilization",
    "neobank_limit_utilization_before_transaction",
    "neobank_limit_utilization_after_transaction",
]

features = features[
    final_columns
].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check coverage, activity counts and duplicate entity-date records.

duplicate_rows = int(
    features.duplicated(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
        ]
    ).sum()
)

print()
print(
    "BEHAVIOURAL FEATURES QA"
)
print(
    "-----------------------"
)
print(
    "Reference date:",
    REFERENCE_DATE_STR,
)
print(
    "Rows:",
    "{:,}".format(
        len(features)
    ),
)
print(
    "Entities:",
    "{:,}".format(
        features[
            "ENTITY_ID"
        ].nunique()
    ),
)
print(
    "Rows with credit-card utilisation:",
    "{:,}".format(
        features[
            "cc_limit_utilization"
        ].notna().sum()
    ),
)
print(
    "Rows with cash advance activity:",
    "{:,}".format(
        features[
            "cash_advance_count"
        ].gt(0).sum()
    ),
)
print(
    "Rows with neobank activity:",
    "{:,}".format(
        features[
            "neobank_topup_count"
        ].gt(0).sum()
    ),
)
print(
    "Rows with cash/neobank balance ratio:",
    "{:,}".format(
        features[
            "cash_neobank_balance_ratio"
        ].notna().sum()
    ),
)
print(
    "Rows with cash-advance limit-after utilisation:",
    "{:,}".format(
        features[
            "cash_advance_limit_utilization_after_transaction"
        ].notna().sum()
    ),
)
print(
    "Entities crossing cash-advance limit 94% (same day / +1 step):",
    "{:,}".format(
        int(
            pd.to_numeric(
                features["cash_advance_limit_cross_94_1d"],
                errors="coerce",
            ).fillna(0).sum()
        )
    ),
)
print(
    "Entities crossing cash-advance limit 97% (same day / +1 step):",
    "{:,}".format(
        int(
            pd.to_numeric(
                features["cash_advance_limit_cross_97_1d"],
                errors="coerce",
            ).fillna(0).sum()
        )
    ),
)
print(
    "Duplicate entity/day rows:",
    duplicate_rows,
)

if duplicate_rows > 0:
    raise ValueError(
        "vars_behavioural_daily contains duplicate "
        "ENTITY_ID + MAP_DTE rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the consolidated daily behavioural features.

print("Behavioural processing before write: {:.2f}s".format(time.perf_counter() - recipe_start), flush=True)
write_start = time.perf_counter()
dataiku.Dataset(
    "vars_behavioural_daily"
).write_with_schema(
    features
)
print("Behavioural output write: {:.2f}s".format(time.perf_counter() - write_start), flush=True)
print("Behavioural total: {:.2f}s".format(time.perf_counter() - recipe_start), flush=True)