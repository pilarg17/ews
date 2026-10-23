# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

# Daily Retail EWS indicators for RRE and Other Retail.

import re
import time
from concurrent.futures import ThreadPoolExecutor

import dataiku
import numpy as np
import pandas as pd

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration

DB_CONNECTION = "PRD_BDWH"

# Rating thresholds
CLIENT_WORSE_THAN_BBB_MINUS = 11.0
CLIENT_WORSE_THAN_BB_MINUS = 14.0
CLIENT_WORSE_THAN_B_PLUS = 15.0
CLIENT_WORSE_THAN_B = 16.0
CLIENT_WORSE_THAN_B_MINUS = 17.0

GEC_HEAD_WORSE_THAN_B = 16.0
GEC_HEAD_WORSE_THAN_B_MINUS = 17.0

# Payment thresholds
DPD_15_THRESHOLD = 15.0
DPD_30_THRESHOLD = 30.0
DPD_60_THRESHOLD = 60.0
OVERDRAFT_THRESHOLD = 500.0
GEC_DPD_15_THRESHOLD = 1000000.0
GEC_DPD_30_THRESHOLD = 250000.0
GEC_DPD_60_THRESHOLD = 50000.0
FAILED_DEBIT_COUNT_MTD_THRESHOLD = 6.0

# Behavioural thresholds
RRE_CASH_NEOBANK_RATIO_THRESHOLD = 4.0
OTHER_CASH_NEOBANK_RATIO_THRESHOLD = 0.5
RRE_CASH_ADVANCE_LIMIT_THRESHOLD = 0.94
OTHER_CASH_ADVANCE_LIMIT_THRESHOLD = 0.97
OTHER_CASH_ADVANCE_AMOUNT_THRESHOLD = 5000.0
OTHER_NEOBANK_TOPUP_AMOUNT_THRESHOLD = 1000.0
OTHER_CC_UTILIZATION_THRESHOLD = 1.00
OTHER_CC_ABS_CHANGE_12M_THRESHOLD = 0.95
OTHER_CC_ABS_CHANGE_CURRENT_THRESHOLD = 0.95
OTHER_CC_REL_CHANGE_3M_MAX_THRESHOLD = 1.00
OTHER_CC_REL_CHANGE_CURRENT_THRESHOLD = 0.90

# RRE LTV thresholds
RRE_LTV_THRESHOLD = 0.95
RRE_LTV_MONITORING_THRESHOLD = 1.05

# Stage 2 materiality thresholds
# Keep the currently validated setup until the 50k / 250k sensitivity can be
# reconstructed reliably on the correctly scoped Retail population.
CONNECTED_STAGE2_GEC_MATERIALITY_THRESHOLD = 150000.0
STAGE2_DOWNGRADE_GEC_MATERIALITY_THRESHOLD = 150000.0

RATING_PERSISTENCE_BUSINESS_DAYS = 5
RATING_HISTORY_QUERY_CHUNK_SIZE = 1000
RATING_HISTORY_PARALLEL_WORKERS = 2

PURE_GEC_HEAD_DOWNGRADE_CODES = {
    "RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
    "RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M",
    "OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
    "OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M",
}


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helper functions

def normalize_id(series):
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
            }
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


def normalized_column_name(column):
    return re.sub(
        r"[^A-Z0-9]+",
        "_",
        str(column).upper(),
    ).strip("_")


def first_non_null(series):
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.iloc[0]


def flag(condition):
    return condition.fillna(False).astype("int8")


def ensure_columns(frame, dataset_name, required_columns):
    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(
            "{} is missing required column(s): {}".format(
                dataset_name,
                ", ".join(missing),
            )
        )


def keep_reference_date(frame, dataset_name, reference_date, date_column="MAP_DTE"):
    ensure_columns(
        frame,
        dataset_name,
        [date_column],
    )

    frame = frame.copy()
    frame[date_column] = normalize_date(frame[date_column])
    frame = frame.loc[
        frame[date_column].eq(reference_date)
    ].copy()

    if frame.empty:
        raise ValueError(
            "{} has no rows for reference date {}.".format(
                dataset_name,
                reference_date.strftime("%Y-%m-%d"),
            )
        )

    return frame


def resolve_frame_column(frame, dataset_name, aliases):
    lookup = {
        normalized_column_name(column): column
        for column in frame.columns
    }

    for alias in aliases:
        key = normalized_column_name(alias)
        if key in lookup:
            return lookup[key]

    raise KeyError(
        "Could not identify required field in {}. Tried aliases: {}. "
        "Available columns: {}".format(
            dataset_name,
            aliases,
            list(frame.columns),
        )
    )


def resolve_history_column(
    frame,
    label,
    aliases,
    required_tokens,
    excluded_tokens=None,
):
    excluded_tokens = excluded_tokens or []

    lookup = {
        normalized_column_name(column): column
        for column in frame.columns
    }

    for alias in aliases:
        key = normalized_column_name(alias)
        if key in lookup:
            return lookup[key]

    matches = []

    for column in frame.columns:
        normalized = normalized_column_name(column)

        if not all(
            token.upper() in normalized
            for token in required_tokens
        ):
            continue

        if any(
            token.upper() in normalized
            for token in excluded_tokens
        ):
            continue

        matches.append(column)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            "Multiple columns match {}: {}".format(
                label,
                matches,
            )
        )

    raise KeyError(
        "Could not identify {}. Available columns: {}".format(
            label,
            list(frame.columns),
        )
    )


def get_sql_source(dataset_name):
    dataset = dataiku.Dataset(dataset_name)
    location = dataset.get_location_info()
    info = location.get("info", {})

    schema = info.get("schema") or info.get("schemaName")
    table = info.get("table") or info.get("tableName")

    if not table:
        raise ValueError(
            "{} must be a SQL-backed dataset.".format(dataset_name)
        )

    if schema and "." not in table:
        table = "{}.{}".format(schema, table)

    return dataset, table


def resolve_schema_column(dataset, dataset_name, aliases):
    schema = dataset.read_schema()
    lookup = {
        normalized_column_name(column["name"]): column
        for column in schema
    }

    for alias in aliases:
        key = normalized_column_name(alias)
        if key in lookup:
            return lookup[key]

    raise KeyError(
        "Could not identify required field in {}. Tried aliases: {}. "
        "Available columns: {}".format(
            dataset_name,
            aliases,
            [column["name"] for column in schema],
        )
    )


def portfolio_flag(condition, portfolio_series):
    return flag(
        condition
        & portfolio_series.eq(1)
    )


def materiality_mask(exposure_series, threshold):
    if threshold is None:
        raise ValueError(
            "A materiality threshold must be configured before this condition is activated."
        )

    threshold = float(threshold)

    return (
        pd.to_numeric(
            exposure_series,
            errors="coerce",
        )
        .fillna(0.0)
        .gt(threshold)
    )


# Rating master scale
MASTER_SCALE = pd.DataFrame(
    [
        ("AAA+", 0.00000, 0.00005, 1),
        ("AAA",  0.00005, 0.00015, 2),
        ("AA+",  0.00015, 0.00025, 3),
        ("AA",   0.00025, 0.00035, 4),
        ("AA-",  0.00035, 0.00045, 5),
        ("A+",   0.00045, 0.00055, 6),
        ("A",    0.00055, 0.00065, 7),
        ("A-",   0.00065, 0.00125, 8),
        ("BBB+", 0.00125, 0.00260, 9),
        ("BBB",  0.00260, 0.00525, 10),
        ("BBB-", 0.00525, 0.00795, 11),
        ("BB+",  0.00795, 0.01015, 12),
        ("BB",   0.01015, 0.01915, 13),
        ("BB-",  0.01915, 0.03315, 14),
        ("B+",   0.03315, 0.06510, 15),
        ("B",    0.06510, 0.11455, 16),
        ("B-",   0.11455, 0.22355, 17),
        ("CCC",  0.22355, 0.99975, 18),
        ("D",    0.99975, 1.00500, 19),
    ],
    columns=[
        "RATING",
        "LOWER_BOUND",
        "UPPER_BOUND",
        "NOTCH",
    ],
)

MASTER_SCALE_BOUNDS = np.r_[
    MASTER_SCALE["LOWER_BOUND"].to_numpy(),
    MASTER_SCALE["UPPER_BOUND"].iloc[-1],
]

MASTER_SCALE_LABELS = MASTER_SCALE["RATING"].tolist()
MASTER_SCALE_NOTCH_MAP = dict(
    zip(
        MASTER_SCALE["RATING"],
        MASTER_SCALE["NOTCH"],
    )
)


def map_pd_to_master_scale(pd_series):
    values = pd.to_numeric(
        pd_series,
        errors="coerce",
    )

    rating = pd.cut(
        values,
        bins=MASTER_SCALE_BOUNDS,
        labels=MASTER_SCALE_LABELS,
        right=False,
        include_lowest=True,
    ).astype("string")

    notch = (
        rating
        .map(MASTER_SCALE_NOTCH_MAP)
        .astype(float)
    )

    return rating, notch


def load_rating_history(
    entity_ids,
    entity_dataset,
    entity_table,
    current_start_date,
    current_end_date,
    baseline_start_date,
    baseline_end_date,
):
    numeric_ids = sorted({
        int(entity_id)
        for entity_id in entity_ids
        if pd.notna(entity_id) and str(entity_id).strip().isdigit()
    })

    if not numeric_ids:
        return pd.DataFrame(columns=[
            "ENTITY_ID", "PD_CRR", "TEK_STRT_DTE",
            "TEK_END_DTE", "RATING_NOTCH"
        ])

    chunks = [
        numeric_ids[i:i + RATING_HISTORY_QUERY_CHUNK_SIZE]
        for i in range(0, len(numeric_ids), RATING_HISTORY_QUERY_CHUNK_SIZE)
    ]

    def fetch_chunk(chunk):
        query = """
        SELECT
            TRIM(TO_CHAR(ENTITY_ID)) AS ENTITY_ID,
            PD_CRR,
            TEK_STRT_DTE,
            TEK_END_DTE
        FROM {entity_table}
        WHERE ENTITY_ID IN ({entity_ids})
          AND (
                (
                    TEK_STRT_DTE < (DATE '{current_end}' + 1)
                    AND DATE '{current_start}' <= NVL(
                        TEK_END_DTE, DATE '9999-12-31'
                    )
                )
                OR
                (
                    TEK_STRT_DTE < (DATE '{baseline_end}' + 1)
                    AND DATE '{baseline_start}' <= NVL(
                        TEK_END_DTE, DATE '9999-12-31'
                    )
                )
              )
        """.format(
            entity_table=entity_table,
            entity_ids=",".join(map(str, chunk)),
            current_start=current_start_date.strftime("%Y-%m-%d"),
            current_end=current_end_date.strftime("%Y-%m-%d"),
            baseline_start=baseline_start_date.strftime("%Y-%m-%d"),
            baseline_end=baseline_end_date.strftime("%Y-%m-%d"),
        )
        return SQLExecutor2(dataset=entity_dataset).query_to_df(query)

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(RATING_HISTORY_PARALLEL_WORKERS, len(chunks))) as executor:
        history_parts = list(executor.map(fetch_chunk, chunks))

    print("Rating history batches:", len(chunks))
    print("Rating history SQL elapsed seconds:", round(time.perf_counter() - start_time, 1))

    history = pd.concat(history_parts, ignore_index=True)
    history["ENTITY_ID"] = normalize_id(history["ENTITY_ID"])
    history["PD_CRR"] = pd.to_numeric(history["PD_CRR"], errors="coerce")
    history["TEK_STRT_DTE"] = normalize_date(history["TEK_STRT_DTE"])
    history["TEK_END_DTE"] = normalize_date(history["TEK_END_DTE"])
    _, history["RATING_NOTCH"] = map_pd_to_master_scale(history["PD_CRR"])
    return history


def rating_snapshot(history, snapshot_date):
    snapshot_date = pd.Timestamp(
        snapshot_date
    ).normalize()

    snapshot = history.loc[
        history["TEK_STRT_DTE"].le(snapshot_date)
        & (
            history["TEK_END_DTE"].isna()
            | history["TEK_END_DTE"].ge(snapshot_date)
        ),
        [
            "ENTITY_ID",
            "TEK_STRT_DTE",
            "TEK_END_DTE",
            "RATING_NOTCH",
        ],
    ].copy()

    snapshot = (
        snapshot
        .sort_values(
            [
                "ENTITY_ID",
                "TEK_STRT_DTE",
                "TEK_END_DTE",
            ]
        )
        .drop_duplicates(
            subset=["ENTITY_ID"],
            keep="last",
        )
    )

    return snapshot[
        [
            "ENTITY_ID",
            "RATING_NOTCH",
        ]
    ]


def build_rating_persistence_flags(
    current_frame,
    entity_dataset,
    entity_table,
    reference_date,
):
    # The reference date is the confirmation date (Day 5).
    # The trigger is identified on Day 1 and must remain active for
    # five consecutive business days, including the trigger day.
    #
    # Example for reference date 2026-10-01:
    # previous state = 2026-09-24
    # Day 1 / trigger = 2026-09-25
    # Day 2 = 2026-09-28
    # Day 3 = 2026-09-29
    # Day 4 = 2026-09-30
    # Day 5 / confirmation = 2026-10-01
    business_dates = list(
        pd.bdate_range(
            end=reference_date,
            periods=RATING_PERSISTENCE_BUSINESS_DAYS + 1,
        )
    )

    previous_state_date = business_dates[0]
    persistence_dates = business_dates[1:]
    trigger_date = persistence_dates[0]

    baseline_dates = [
        date - pd.DateOffset(years=1)
        for date in business_dates
    ]

    portfolio_scope = current_frame.loc[
        current_frame["RRE_FLAG"].eq(1)
        | current_frame["OTHER_RETAIL_FLAG"].eq(1),
        [
            "ENTITY_ID",
            "HEAD_GEC_PARENT_ID",
            "GEC_NET_EXPOSURE",
            "RRE_FLAG",
            "OTHER_RETAIL_FLAG",
        ],
    ].copy()

    history_ids = pd.concat(
        [
            portfolio_scope["ENTITY_ID"],
            portfolio_scope["HEAD_GEC_PARENT_ID"],
        ],
        ignore_index=True,
    ).dropna().drop_duplicates()

    rating_history = load_rating_history(
        history_ids,
        entity_dataset,
        entity_table,
        min(business_dates),
        max(business_dates),
        min(baseline_dates),
        max(baseline_dates),
    )

    all_snapshot_dates = sorted(
        set(business_dates + baseline_dates)
    )

    snapshots = {
        date: rating_snapshot(
            rating_history,
            date,
        )
        for date in all_snapshot_dates
    }

    daily_parts = []

    for date in business_dates:
        baseline_date = date - pd.DateOffset(years=1)

        client_current = snapshots[date].rename(
            columns={
                "RATING_NOTCH": "CLIENT_CURRENT_NOTCH",
            }
        )
        client_12m = snapshots[baseline_date].rename(
            columns={
                "RATING_NOTCH": "CLIENT_NOTCH_12M",
            }
        )

        gec_current = snapshots[date].rename(
            columns={
                "ENTITY_ID": "HEAD_GEC_PARENT_ID",
                "RATING_NOTCH": "GEC_CURRENT_NOTCH",
            }
        )
        gec_12m = snapshots[baseline_date].rename(
            columns={
                "ENTITY_ID": "HEAD_GEC_PARENT_ID",
                "RATING_NOTCH": "GEC_NOTCH_12M",
            }
        )

        day = portfolio_scope.merge(
            client_current,
            how="left",
            on="ENTITY_ID",
            validate="1:1",
        )
        day = day.merge(
            client_12m,
            how="left",
            on="ENTITY_ID",
            validate="1:1",
        )
        day = day.merge(
            gec_current,
            how="left",
            on="HEAD_GEC_PARENT_ID",
            validate="m:1",
        )
        day = day.merge(
            gec_12m,
            how="left",
            on="HEAD_GEC_PARENT_ID",
            validate="m:1",
        )

        day["DATE"] = date
        day["CLIENT_VALID"] = (
            day["CLIENT_CURRENT_NOTCH"].notna()
            & day["CLIENT_NOTCH_12M"].notna()
        )
        day["CLIENT_CURRENT_VALID"] = (
            day["CLIENT_CURRENT_NOTCH"].notna()
        )
        day["CLIENT_DOWNGRADE_12M"] = (
            day["CLIENT_CURRENT_NOTCH"]
            - day["CLIENT_NOTCH_12M"]
        )

        day["GEC_VALID"] = (
            day["GEC_CURRENT_NOTCH"].notna()
            & day["GEC_NOTCH_12M"].notna()
        )
        day["GEC_DOWNGRADE_12M"] = (
            day["GEC_CURRENT_NOTCH"]
            - day["GEC_NOTCH_12M"]
        )

        client_rules = {
            "RRE_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M": (
                day["CLIENT_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["CLIENT_DOWNGRADE_12M"].ge(6)
            ),
            "RRE_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M": (
                day["CLIENT_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["CLIENT_CURRENT_NOTCH"].gt(CLIENT_WORSE_THAN_BBB_MINUS)
                & day["CLIENT_DOWNGRADE_12M"].ge(5)
            ),
            "RRE_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M": (
                day["CLIENT_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["CLIENT_CURRENT_NOTCH"].gt(CLIENT_WORSE_THAN_BB_MINUS)
                & day["CLIENT_DOWNGRADE_12M"].ge(4)
            ),
            "RRE_CLIENT_WORSE_B_PLUS_DOWNGRADE_AT_LEAST_2_VS_12M": (
                day["CLIENT_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["CLIENT_CURRENT_NOTCH"].gt(CLIENT_WORSE_THAN_B_PLUS)
                & day["CLIENT_DOWNGRADE_12M"].ge(2)
            ),
            "RRE_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_1_VS_12M": (
                day["CLIENT_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["CLIENT_CURRENT_NOTCH"].gt(CLIENT_WORSE_THAN_B)
                & day["CLIENT_DOWNGRADE_12M"].ge(1)
            ),
            "OTHER_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M": (
                day["CLIENT_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["CLIENT_DOWNGRADE_12M"].ge(6)
            ),
            "OTHER_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M": (
                day["CLIENT_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["CLIENT_CURRENT_NOTCH"].gt(CLIENT_WORSE_THAN_BBB_MINUS)
                & day["CLIENT_DOWNGRADE_12M"].ge(5)
            ),
            "OTHER_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M": (
                day["CLIENT_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["CLIENT_CURRENT_NOTCH"].gt(CLIENT_WORSE_THAN_BB_MINUS)
                & day["CLIENT_DOWNGRADE_12M"].ge(4)
            ),
            "OTHER_CLIENT_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M": (
                day["CLIENT_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["CLIENT_CURRENT_NOTCH"].gt(CLIENT_WORSE_THAN_B_MINUS)
                & day["CLIENT_DOWNGRADE_12M"].ge(1)
            ),
            "OTHER_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_3_VS_12M": (
                day["CLIENT_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["CLIENT_CURRENT_NOTCH"].gt(CLIENT_WORSE_THAN_B)
                & day["CLIENT_DOWNGRADE_12M"].ge(3)
            ),
        }

        gec_rules = {
            "RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M": (
                day["GEC_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["GEC_CURRENT_NOTCH"].gt(GEC_HEAD_WORSE_THAN_B)
                & day["GEC_DOWNGRADE_12M"].ge(4)
            ),
            "RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M": (
                day["GEC_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["GEC_CURRENT_NOTCH"].gt(GEC_HEAD_WORSE_THAN_B_MINUS)
                & day["GEC_DOWNGRADE_12M"].ge(1)
            ),
            "RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M": (
                day["GEC_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["GEC_DOWNGRADE_12M"].gt(7)
            ),
            "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_PLUS": (
                day["GEC_VALID"]
                & day["CLIENT_CURRENT_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["GEC_DOWNGRADE_12M"].ge(3)
                & day["CLIENT_CURRENT_NOTCH"].gt(CLIENT_WORSE_THAN_B_PLUS)
            ),
            "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_1_CLIENT_DOWNGRADE_AT_LEAST_6": (
                day["GEC_VALID"]
                & day["CLIENT_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["GEC_DOWNGRADE_12M"].ge(1)
                & day["CLIENT_DOWNGRADE_12M"].ge(6)
            ),
            "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5": (
                day["GEC_VALID"]
                & day["CLIENT_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["GEC_DOWNGRADE_12M"].ge(3)
                & day["CLIENT_DOWNGRADE_12M"].ge(5)
            ),
            "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3": (
                day["GEC_VALID"]
                & day["CLIENT_VALID"]
                & day["RRE_FLAG"].eq(1)
                & day["GEC_DOWNGRADE_12M"].ge(6)
                & day["CLIENT_DOWNGRADE_12M"].ge(3)
            ),
            "OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M": (
                day["GEC_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["GEC_CURRENT_NOTCH"].gt(GEC_HEAD_WORSE_THAN_B)
                & day["GEC_DOWNGRADE_12M"].ge(4)
            ),
            "OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M": (
                day["GEC_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["GEC_CURRENT_NOTCH"].gt(GEC_HEAD_WORSE_THAN_B_MINUS)
                & day["GEC_DOWNGRADE_12M"].ge(1)
            ),
            "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M": (
                day["GEC_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["GEC_DOWNGRADE_12M"].ge(6)
            ),
            "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B": (
                day["GEC_VALID"]
                & day["CLIENT_CURRENT_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["GEC_DOWNGRADE_12M"].ge(3)
                & day["CLIENT_CURRENT_NOTCH"].gt(CLIENT_WORSE_THAN_B)
            ),
            "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3": (
                day["GEC_VALID"]
                & day["CLIENT_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["GEC_DOWNGRADE_12M"].ge(6)
                & day["CLIENT_DOWNGRADE_12M"].ge(3)
            ),
            "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5": (
                day["GEC_VALID"]
                & day["CLIENT_VALID"]
                & day["OTHER_RETAIL_FLAG"].eq(1)
                & day["GEC_DOWNGRADE_12M"].ge(3)
                & day["CLIENT_DOWNGRADE_12M"].ge(5)
            ),
        }

        for rule_name, condition in client_rules.items():
            day[rule_name] = flag(condition)
            day["{}_VALID".format(rule_name)] = flag(
                day["CLIENT_VALID"]
            )

        for rule_name, condition in gec_rules.items():
            day[rule_name] = flag(condition)

            if "CLIENT_WORSE" in rule_name:
                valid = (
                    day["GEC_VALID"]
                    & day["CLIENT_CURRENT_VALID"]
                )
            elif "CLIENT_DOWNGRADE" in rule_name:
                valid = (
                    day["GEC_VALID"]
                    & day["CLIENT_VALID"]
                )
            else:
                valid = day["GEC_VALID"]

            day["{}_VALID".format(rule_name)] = flag(valid)

        daily_parts.append(day)

    daily = pd.concat(
        daily_parts,
        ignore_index=True,
    )

    client_rule_names = [
        column
        for column in daily.columns
        if column.startswith("RRE_CLIENT_")
        or column.startswith("OTHER_CLIENT_")
    ]
    client_rule_names = [
        column
        for column in client_rule_names
        if not column.endswith("_VALID")
    ]

    gec_rule_names = [
        column
        for column in daily.columns
        if column.startswith("RRE_GEC_HEAD_")
        or column.startswith("OTHER_GEC_HEAD_")
    ]
    gec_rule_names = [
        column
        for column in gec_rule_names
        if not column.endswith("_VALID")
    ]

    result = current_frame[["ENTITY_ID"]].copy()
    qa_rows = []

    exposure_map = (
        current_frame[
            [
                "ENTITY_ID",
                "GEC_NET_EXPOSURE",
            ]
        ]
        .drop_duplicates("ENTITY_ID")
        .set_index("ENTITY_ID")["GEC_NET_EXPOSURE"]
    )

    all_rule_names = client_rule_names + gec_rule_names

    client_notch = (
        daily
        .pivot(
            index="ENTITY_ID",
            columns="DATE",
            values="CLIENT_CURRENT_NOTCH",
        )
        .reindex(columns=business_dates)
    )

    gec_notch = (
        daily
        .pivot(
            index="ENTITY_ID",
            columns="DATE",
            values="GEC_CURRENT_NOTCH",
        )
        .reindex(columns=business_dates)
    )

    for rule_name in all_rule_names:
        active = (
            daily
            .pivot(
                index="ENTITY_ID",
                columns="DATE",
                values=rule_name,
            )
            .reindex(columns=business_dates)
            .fillna(0)
            .astype(bool)
        )

        valid = (
            daily
            .pivot(
                index="ENTITY_ID",
                columns="DATE",
                values="{}_VALID".format(rule_name),
            )
            .reindex(columns=business_dates)
            .fillna(0)
            .astype(bool)
        )

        raw_trigger = (
            valid[previous_state_date]
            & valid[trigger_date]
            & ~active[previous_state_date]
            & active[trigger_date]
        )

        if rule_name in client_rule_names:
            recent_worsening = (
                client_notch[previous_state_date].notna()
                & client_notch[trigger_date].notna()
                & client_notch[trigger_date].gt(
                    client_notch[previous_state_date]
                )
            )
        else:
            recent_worsening = (
                gec_notch[previous_state_date].notna()
                & gec_notch[trigger_date].notna()
                & gec_notch[trigger_date].gt(
                    gec_notch[previous_state_date]
                )
            )

        trigger = raw_trigger & recent_worsening

        complete_history = (
            valid[persistence_dates]
            .all(axis=1)
        )
        persisted = (
            active[persistence_dates]
            .all(axis=1)
        )

        matured = (
            trigger
            & complete_history
            & persisted
        )

        final = matured

        result[rule_name] = (
            result["ENTITY_ID"]
            .map(final.astype("int8"))
            .fillna(0)
            .astype("int8")
        )

        qa_rows.append(
            {
                "INDICATOR": rule_name,
                "DAY_1_TRIGGERS": int(trigger.sum()),
                "MATURED_DAY_5": int(matured.sum()),
                "FINAL_AFTER_PERSISTENCE": int(final.sum()),
                "BLOCKED_NO_RECENT_WORSENING": int(
                    (raw_trigger & ~recent_worsening).sum()
                ),
                "BLOCKED_MISSING_DATA": int(
                    (trigger & ~complete_history).sum()
                ),
            }
        )

    qa = pd.DataFrame(qa_rows)

    # Current rating worsening vs the immediately previous business day.
    # Used by historical-comparison indicators outside the 5-day rating block.
    previous_business_date = business_dates[-2]
    current_business_date = business_dates[-1]

    client_daily_worsening = (
        client_notch[previous_business_date].notna()
        & client_notch[current_business_date].notna()
        & client_notch[current_business_date].gt(
            client_notch[previous_business_date]
        )
    )

    rating_daily_change = result[["ENTITY_ID"]].copy()
    rating_daily_change["CLIENT_RATING_DAILY_WORSENING"] = (
        rating_daily_change["ENTITY_ID"]
        .map(client_daily_worsening.astype("int8"))
        .fillna(0)
        .astype("int8")
    )

    return (
        result,
        qa,
        rating_daily_change,
        previous_state_date,
        trigger_date,
        persistence_dates,
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Reference and previous business dates

_indicator_started = time.perf_counter()

reference_df = dataiku.Dataset(
    "Implementation_Reference_Date_previous"
).get_dataframe()

if reference_df.empty or len(reference_df) != 1:
    raise ValueError(
        "Implementation_Reference_Date_previous must contain exactly one row."
    )

ensure_columns(
    reference_df,
    "Implementation_Reference_Date_previous",
    [
        "REFERENCE_DATE",
        "PREVIOUS_DATE",
    ],)

REFERENCE_DATE = normalize_date(
    reference_df["REFERENCE_DATE"]
).iloc[0]

PREVIOUS_DATE = normalize_date(
    reference_df["PREVIOUS_DATE"]
).iloc[0]

if pd.isna(REFERENCE_DATE) or pd.isna(PREVIOUS_DATE):
    raise ValueError(
        "REFERENCE_DATE and PREVIOUS_DATE must both be valid dates."
    )

if PREVIOUS_DATE >= REFERENCE_DATE:
    raise ValueError(
        "PREVIOUS_DATE must be earlier than REFERENCE_DATE."
    )

REFERENCE_12M_DATE = (
    REFERENCE_DATE
    - pd.DateOffset(years=1)
).normalize()

REFERENCE_DATE_STR = REFERENCE_DATE.strftime("%Y-%m-%d")
PREVIOUS_DATE_STR = PREVIOUS_DATE.strftime("%Y-%m-%d")
REFERENCE_12M_DATE_STR = REFERENCE_12M_DATE.strftime("%Y-%m-%d")

print("Reference date       :", REFERENCE_DATE_STR)
print("Previous business day:", PREVIOUS_DATE_STR)
print("Reference date - 12m :", REFERENCE_12M_DATE_STR)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current entity population

entity_source = dataiku.Dataset(
    "Entity_Information_Previous"
).get_dataframe()

required_entity_columns = [
    "ENTITY_ID",
    "ENTITY_NAME",
    "CUSTOMER_ID",
    "MAP_DTE",
    "DEFAULT_STATUS",
    "PD_MASTER_RATING",
    "PD_MASTER_NOTCH",
    "GCC_PARENT_ID",
    "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE",
    "PROBATION_PERIOD_FLAG",
]

ensure_columns(
    entity_source,
    "Entity_Information_Previous",
    required_entity_columns,
)

entity_source = entity_source[
    required_entity_columns
].copy()

entity_source["ENTITY_ID"] = normalize_id(
    entity_source["ENTITY_ID"]
)
entity_source["CUSTOMER_ID"] = normalize_id(
    entity_source["CUSTOMER_ID"]
)
entity_source["GCC_PARENT_ID"] = normalize_id(
    entity_source["GCC_PARENT_ID"]
)

entity_source = keep_reference_date(
    entity_source,
    "Entity_Information_Previous",
    REFERENCE_DATE,
)

entity_source["DEFAULT_STATUS"] = (
    entity_source["DEFAULT_STATUS"]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

default_entity_ids = set(
    entity_source.loc[
        entity_source["DEFAULT_STATUS"].isin(["18", "19", "20"]),
        "ENTITY_ID",
    ]
    .dropna()
    .astype(str)
)

print(
    "Default entities identified:",
    "{:,}".format(len(default_entity_ids)),
)

for column in [
    "PD_MASTER_NOTCH",
    "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE",
]:
    entity_source[column] = pd.to_numeric(
        entity_source[column],
        errors="coerce",
    )

entity_source["PROBATION_PERIOD_FLAG"] = (
    entity_source["PROBATION_PERIOD_FLAG"]
    .astype("string")
    .str.strip()
    .str.upper()
)

entity_source["PD_MASTER_RATING"] = (
    entity_source["PD_MASTER_RATING"]
    .astype("string")
    .str.strip()
    .str.upper()
)

# Joint Obligors are deliberately retained in Entity_Information_Previous because one JO
# can be linked to multiple customers. However, customer-level behavioural/payment
# data must keep the original direct customer -> entity mapping to avoid duplicating
# the same customer event into both the direct entity and the JO entity.
entity_source["IS_JOINT_OBLIGOR"] = (
    entity_source["ENTITY_NAME"]
    .astype("string")
    .str.strip()
    .str.upper()
    .str.startswith("JO_", na=False)
)

joint_obligor_customer_links = (
    entity_source.loc[
        entity_source["IS_JOINT_OBLIGOR"],
        [
            "CUSTOMER_ID",
            "ENTITY_ID",
        ],
    ]
    .dropna()
    .drop_duplicates()
)

entity_customer_map = (
    entity_source.loc[
        ~entity_source["IS_JOINT_OBLIGOR"],
        [
            "CUSTOMER_ID",
            "ENTITY_ID",
        ],
    ]
    .dropna()
    .drop_duplicates()
)

customer_multiplicity = (
    entity_customer_map
    .groupby("CUSTOMER_ID")["ENTITY_ID"]
    .nunique()
)

if customer_multiplicity.gt(1).any():
    raise ValueError(
        "At least one CUSTOMER_ID maps to more than one non-JO ENTITY_ID "
        "on the reference date."
    )

entity_customer_map = entity_customer_map.drop_duplicates(
    subset=["CUSTOMER_ID"]
)

print()
print("ENTITY / CUSTOMER MAPPING QA")
print("----------------------------")
print(
    "Direct customer mappings used for customer-level data:",
    "{:,}".format(len(entity_customer_map)),
)
print(
    "JO customer links retained in Entity_Information_Previous       :",
    "{:,}".format(len(joint_obligor_customer_links)),
)
print(
    "JO entities represented                               :",
    "{:,}".format(
        joint_obligor_customer_links["ENTITY_ID"].nunique()
    ),
)

entity = (
    entity_source
    .groupby(
        "ENTITY_ID",
        as_index=False,
    )
    .agg(
        MAP_DTE=("MAP_DTE", "first"),
        CLIENT_RATING=("PD_MASTER_RATING", first_non_null),
        CLIENT_RATING_NOTCH=("PD_MASTER_NOTCH", first_non_null),
        GCC_PARENT_ID=("GCC_PARENT_ID", first_non_null),
        UTP_SOURCE_FLAG=(
            "FLAG_UTP_INSOLVENCY_CONCILIATION_PROCEDURE",
            first_non_null,
        ),
        PROBATION_PERIOD_FLAG=("PROBATION_PERIOD_FLAG", first_non_null),
    )
)

entity["PROBATION_FLAG"] = flag(
    entity["PROBATION_PERIOD_FLAG"].eq("Y")
)

entity["UTP_FLAG"] = flag(
    pd.to_numeric(
        entity["UTP_SOURCE_FLAG"],
        errors="coerce",
    ).gt(0.5)
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Daily RRE / Other Retail portfolio membership

portfolio = dataiku.Dataset(
    "Retail_Portfolio_Daily_Previous"
).get_dataframe()

ensure_columns(
    portfolio,
    "Retail_Portfolio_Daily_Previous",
    [
        "ENTITY_ID",
        "MAP_DTE",
        "RRE_FLAG",
        "OTHER_RETAIL_FLAG",
        "RRE_INSTRUMENT_COUNT",
        "OTHER_RETAIL_INSTRUMENT_COUNT",
    ],
)

portfolio["ENTITY_ID"] = normalize_id(
    portfolio["ENTITY_ID"]
)
portfolio = keep_reference_date(
    portfolio,
    "Retail_Portfolio_Daily_Previous",
    REFERENCE_DATE,
)

if portfolio["ENTITY_ID"].duplicated().any():
    raise ValueError(
        "Retail_Portfolio_Daily_Previous contains duplicate ENTITY_ID rows."
    )

for column in [
    "RRE_FLAG",
    "OTHER_RETAIL_FLAG",
    "RRE_INSTRUMENT_COUNT",
    "OTHER_RETAIL_INSTRUMENT_COUNT",
]:
    portfolio[column] = pd.to_numeric(
        portfolio[column],
        errors="coerce",
    ).fillna(0)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Client rating history

history = dataiku.Dataset(
    "Rating_GCC_History_previous"
).get_dataframe()

ensure_columns(
    history,
    "Rating_GCC_History_previous",
    ["ENTITY_ID"],
)

history["ENTITY_ID"] = normalize_id(
    history["ENTITY_ID"]
)

if history["ENTITY_ID"].duplicated().any():
    raise ValueError(
        "Rating_GCC_History_previous contains duplicate ENTITY_ID rows."
    )

client_downgrade_12m_col = resolve_history_column(
    history,
    label="client downgrade vs 12 months",
    aliases=[
        "rating_granular_scale_downgrade_12m",
        "CLIENT_RATING_DOWNGRADE_12M",
        "CLIENT_DOWNGRADE_12M",
    ],
    required_tokens=["RATING", "DOWNGRADE", "12M"],
    excluded_tokens=["GCC", "BEST", "WORST"],
)

history = history[
    [
        "ENTITY_ID",
        client_downgrade_12m_col,
    ]
].rename(
    columns={
        client_downgrade_12m_col: "CLIENT_DOWNGRADE_12M",
    }
)

history["CLIENT_DOWNGRADE_12M"] = pd.to_numeric(
    history["CLIENT_DOWNGRADE_12M"],
    errors="coerce",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Daily behavioural features

behavioural = dataiku.Dataset(
    "vars_behavioural_daily_Previous"
).get_dataframe()

required_behavioural_columns = [
    "ENTITY_ID",
    "MAP_DTE",
    "cc_limit_utilization",
    "cash_advance_amount",
    "neobank_topup_amount",
    "cash_neobank_balance_ratio",
    "cash_advance_limit_utilization_before_transaction",
    "cash_advance_limit_utilization_after_transaction",
    "cash_advance_limit_cross_94_1d",
    "cash_advance_limit_cross_97_1d",
]

ensure_columns(
    behavioural,
    "vars_behavioural_daily_Previous",
    required_behavioural_columns,
)

behavioural = behavioural[
    required_behavioural_columns
].copy()

behavioural["ENTITY_ID"] = normalize_id(
    behavioural["ENTITY_ID"]
)
behavioural = keep_reference_date(
    behavioural,
    "vars_behavioural_daily_Previous",
    REFERENCE_DATE,
)

if behavioural["ENTITY_ID"].duplicated().any():
    raise ValueError(
        "vars_behavioural_daily_Previous contains duplicate ENTITY_ID rows on the reference date."
    )

for column in required_behavioural_columns:
    if column not in ["ENTITY_ID", "MAP_DTE"]:
        behavioural[column] = pd.to_numeric(
            behavioural[column],
            errors="coerce",
        )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Credit-card utilisation history

cc_history = dataiku.Dataset(
    "CC_Utilisation_History_Previous"
).get_dataframe()

ensure_columns(
    cc_history,
    "CC_Utilisation_History_Previous",
    ["ENTITY_ID"],
)

cc_history["ENTITY_ID"] = normalize_id(
    cc_history["ENTITY_ID"]
)

cc_history_date_column = next(
    (
        column
        for column in [
            "MAP_DTE",
            "REFERENCE_DATE",
            "reference_date",
            "SNAPSHOT_DATE",
            "snapshot_date",
        ]
        if column in cc_history.columns
    ),
    None,
)

if cc_history_date_column is not None:
    cc_history = keep_reference_date(
        cc_history,
        "CC_Utilisation_History_Previous",
        REFERENCE_DATE,
        date_column=cc_history_date_column,
    )

cc_diff_12m_col = resolve_frame_column(
    cc_history,
    "CC_Utilisation_History_Previous",
    [
        "cc_diff_limit_util_12m",
        "cc_limit_utilization_abs_change_12m",
    ],
)

cc_max_3m_col = resolve_frame_column(
    cc_history,
    "CC_Utilisation_History_Previous",
    [
        "cc_max_utilization_3m",
    ],
)

cc_diff_3m_max_col = resolve_frame_column(
    cc_history,
    "CC_Utilisation_History_Previous",
    [
        "cc_diff_limit_util_3m_max",
        "cc_limit_utilization_abs_change_3m_max",
    ],
)

cc_previous_day_col = resolve_frame_column(
    cc_history,
    "CC_Utilisation_History_Previous",
    [
        "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION",
        "cc_previous_business_day_utilization",
    ],
)

cc_history = cc_history[
    [
        "ENTITY_ID",
        cc_diff_12m_col,
        cc_max_3m_col,
        cc_diff_3m_max_col,
        cc_previous_day_col,
    ]
].rename(
    columns={
        cc_diff_12m_col: "CC_UTILIZATION_ABS_CHANGE_12M",
        cc_max_3m_col: "CC_MAX_UTILIZATION_3M",
        cc_diff_3m_max_col: "CC_UTILIZATION_ABS_CHANGE_3M_MAX",
        cc_previous_day_col: "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION",
    }
)

if cc_history["ENTITY_ID"].duplicated().any():
    raise ValueError(
        "CC_Utilisation_History_Previous contains duplicate ENTITY_ID rows for the selected snapshot."
    )

for column in [
    "CC_UTILIZATION_ABS_CHANGE_12M",
    "CC_MAX_UTILIZATION_3M",
    "CC_UTILIZATION_ABS_CHANGE_3M_MAX",
    "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION",
]:
    cc_history[column] = pd.to_numeric(
        cc_history[column],
        errors="coerce",
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Daily DPD and arrears

dpd_arrears = dataiku.Dataset(
    "DPD_Arrears_Daily_Entity_Previous"
).get_dataframe()

ensure_columns(
    dpd_arrears,
    "DPD_Arrears_Daily_Entity_Previous",
    [
        "ENTITY_ID",
        "MAP_DTE",
        "DPD",
        "ARREARS_AMOUNT",
    ],
)

dpd_arrears["ENTITY_ID"] = normalize_id(
    dpd_arrears["ENTITY_ID"]
)
dpd_arrears = keep_reference_date(
    dpd_arrears,
    "DPD_Arrears_Daily_Entity_Previous",
    REFERENCE_DATE,
)

for column in ["DPD", "ARREARS_AMOUNT"]:
    dpd_arrears[column] = pd.to_numeric(
        dpd_arrears[column],
        errors="coerce",
    )

dpd_arrears = (
    dpd_arrears
    .groupby("ENTITY_ID", as_index=False)
    .agg(
        DPD=("DPD", "max"),
        ARREARS_AMOUNT=("ARREARS_AMOUNT", "max"),
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current loan state

loan_state = dataiku.Dataset(
    "vars_loan_Previous"
).get_dataframe()

required_loan_columns = [
    "ENTITY_ID",
    "MAP_DTE",
    "IND_FORBEARANCE",
    "STAGE",
    "NET_EXPOSURE",
]

ensure_columns(
    loan_state,
    "vars_loan_Previous",
    required_loan_columns,
)

loan_state = loan_state[
    required_loan_columns
].copy()

loan_state["ENTITY_ID"] = normalize_id(
    loan_state["ENTITY_ID"]
)
loan_state = keep_reference_date(
    loan_state,
    "vars_loan_Previous",
    REFERENCE_DATE,
)

loan_state["IND_FORBEARANCE"] = (
    loan_state["IND_FORBEARANCE"]
    .astype("string")
    .str.strip()
    .str.upper()
)
loan_state["STAGE"] = pd.to_numeric(
    loan_state["STAGE"],
    errors="coerce",
)
loan_state["NET_EXPOSURE"] = pd.to_numeric(
    loan_state["NET_EXPOSURE"],
    errors="coerce",
)

loan_state["FORBEARANCE_FLAG"] = flag(
    loan_state["IND_FORBEARANCE"].eq("T")
)
loan_state["CURRENT_STAGE2_INSTRUMENT_FLAG"] = flag(
    loan_state["STAGE"].eq(2.0)
)

loan_entity = (
    loan_state
    .dropna(subset=["ENTITY_ID"])
    .groupby("ENTITY_ID", as_index=False)
    .agg(
        FORBEARANCE_FLAG=("FORBEARANCE_FLAG", "max"),
        CURRENT_STAGE2_FLAG=("CURRENT_STAGE2_INSTRUMENT_FLAG", "max"),
        NET_EXPOSURE=(
            "NET_EXPOSURE",
            lambda series: series.sum(min_count=1),
        ),
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Connected-client current Stage 2 and UTP state

connected = (
    entity[
        [
            "ENTITY_ID",
            "GCC_PARENT_ID",
            "UTP_FLAG",
        ]
    ]
    .merge(
        loan_entity[
            [
                "ENTITY_ID",
                "CURRENT_STAGE2_FLAG",
            ]
        ],
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
)

connected["CURRENT_STAGE2_FLAG"] = (
    connected["CURRENT_STAGE2_FLAG"]
    .fillna(0)
    .astype("int8")
)
connected["UTP_FLAG"] = (
    connected["UTP_FLAG"]
    .fillna(0)
    .astype("int8")
)

connected_group = (
    connected
    .dropna(subset=["GCC_PARENT_ID"])
    .groupby("GCC_PARENT_ID", as_index=False)
    .agg(
        GCC_MEMBER_COUNT=("ENTITY_ID", "nunique"),
        GCC_STAGE2_GROUP_ACTIVE_COUNT=("CURRENT_STAGE2_FLAG", "sum"),
        GCC_UTP_GROUP_ACTIVE_COUNT=("UTP_FLAG", "sum"),
    )
)

connected = connected.merge(
    connected_group,
    how="left",
    on="GCC_PARENT_ID",
    validate="m:1",
)
for column in [
    "GCC_MEMBER_COUNT",
    "GCC_STAGE2_GROUP_ACTIVE_COUNT",
    "GCC_UTP_GROUP_ACTIVE_COUNT",
]:
    connected[column] = (
        connected[column]
        .fillna(0)
        .astype("int32")
    )

connected["GCC_STAGE2_CONNECTED_MEMBER_COUNT"] = (
    connected["GCC_STAGE2_GROUP_ACTIVE_COUNT"]
    - connected["CURRENT_STAGE2_FLAG"]
)
connected["GCC_UTP_CONNECTED_MEMBER_COUNT"] = (
    connected["GCC_UTP_GROUP_ACTIVE_COUNT"]
    - connected["UTP_FLAG"]
)

connected["GCC_STAGE2_CONNECTED_BASE"] = flag(
    connected["GCC_PARENT_ID"].notna()
    & connected["GCC_STAGE2_CONNECTED_MEMBER_COUNT"].gt(0)
)
connected["GCC_UTP_CONNECTED_BASE"] = flag(
    connected["GCC_PARENT_ID"].notna()
    & connected["GCC_UTP_CONNECTED_MEMBER_COUNT"].gt(0)
)

if (
    connected["GCC_STAGE2_CONNECTED_MEMBER_COUNT"].lt(0).any()
    or connected["GCC_UTP_CONNECTED_MEMBER_COUNT"].lt(0).any()
):
    raise ValueError(
        "Connected-client active-member counts must not be negative."
    )

connected = connected[
    [
        "ENTITY_ID",
        "GCC_MEMBER_COUNT",
        "GCC_STAGE2_CONNECTED_MEMBER_COUNT",
        "GCC_UTP_CONNECTED_MEMBER_COUNT",
        "GCC_STAGE2_CONNECTED_BASE",
        "GCC_UTP_CONNECTED_BASE",
    ]
].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Failed direct debit features

payment = dataiku.Dataset(
    "vars_payment_level_Previous"
).get_dataframe()

required_payment_columns = [
    "CUSTOMER_ID",
    "MAP_DTE",
    "FAILED_DEBIT_COUNT",
    "FAILED_DEBIT_AMOUNT",
    "FAILED_DEBIT_COUNT_MTD",
    "FAILED_DEBIT_AMOUNT_MTD",
]

ensure_columns(
    payment,
    "vars_payment_level_Previous",
    required_payment_columns,
)

payment = payment[
    required_payment_columns
].copy()
payment["CUSTOMER_ID"] = normalize_id(
    payment["CUSTOMER_ID"]
)
payment = keep_reference_date(
    payment,
    "vars_payment_level_Previous",
    REFERENCE_DATE,
)

if payment.duplicated(
    subset=["CUSTOMER_ID", "MAP_DTE"]
).any():
    raise ValueError(
        "vars_payment_level_Previous contains duplicate CUSTOMER_ID + MAP_DTE rows."
    )

for column in [
    "FAILED_DEBIT_COUNT",
    "FAILED_DEBIT_AMOUNT",
    "FAILED_DEBIT_COUNT_MTD",
    "FAILED_DEBIT_AMOUNT_MTD",
]:
    payment[column] = pd.to_numeric(
        payment[column],
        errors="coerce",
    ).fillna(0.0)

payment = payment.merge(
    entity_customer_map,
    how="left",
    on="CUSTOMER_ID",
    validate="1:1",
)

payment_entity = (
    payment
    .dropna(subset=["ENTITY_ID"])
    .groupby("ENTITY_ID", as_index=False)
    .agg(
        FAILED_DEBIT_COUNT=("FAILED_DEBIT_COUNT", "sum"),
        FAILED_DEBIT_AMOUNT=("FAILED_DEBIT_AMOUNT", "sum"),
        FAILED_DEBIT_COUNT_MTD=("FAILED_DEBIT_COUNT_MTD", "sum"),
        FAILED_DEBIT_AMOUNT_MTD=("FAILED_DEBIT_AMOUNT_MTD", "sum"),
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Previous-business-day DPD

instrument_dataset, instrument_table = get_sql_source(
    "INSTRUMENT_VW"
)
_, instrument_entity_table = get_sql_source(
    "INSTRUMENT_ENTITY_VW"
)

previous_dpd_query = """
SELECT
    TRIM(TO_CHAR(i2e.ENTITY_ID)) AS ENTITY_ID,
    MAX(
        CASE
            WHEN ABS(NVL(ins.DPD, 0)) >= 10000000000 THEN 0
            ELSE NVL(ins.DPD, 0)
        END
    ) AS PREVIOUS_DPD
FROM {instrument_table} ins
INNER JOIN {instrument_entity_table} i2e
    ON TRIM(CAST(ins.INSTRUMENT_ID AS VARCHAR2(100)))
     = TRIM(CAST(i2e.INSTRUMENT_ID AS VARCHAR2(100)))
WHERE ins.TEK_STRT_DTE < (DATE '{previous_date}' + 1)
  AND NVL(ins.TEK_END_DTE, DATE '9999-12-31') >= (DATE '{previous_date}' + 1)
  AND i2e.TEK_STRT_DTE < (DATE '{previous_date}' + 1)
  AND NVL(i2e.TEK_END_DTE, DATE '9999-12-31') >= (DATE '{previous_date}' + 1)
  AND i2e.ENTITY_ID IS NOT NULL
GROUP BY TRIM(TO_CHAR(i2e.ENTITY_ID))
""".format(
    instrument_table=instrument_table,
    instrument_entity_table=instrument_entity_table,
    previous_date=PREVIOUS_DATE_STR,
)

previous_dpd = SQLExecutor2(
    dataset=instrument_dataset
).query_to_df(
    previous_dpd_query
)

if previous_dpd.empty:
    raise ValueError(
        "Previous DPD query returned no rows for {}.".format(
            PREVIOUS_DATE_STR
        )
    )

previous_dpd["ENTITY_ID"] = normalize_id(
    previous_dpd["ENTITY_ID"]
)
previous_dpd["PREVIOUS_DPD"] = pd.to_numeric(
    previous_dpd["PREVIOUS_DPD"],
    errors="coerce",
)

if previous_dpd["ENTITY_ID"].duplicated().any():
    raise ValueError(
        "Previous DPD preparation contains duplicate ENTITY_ID rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Client downgrade from Stage 1 to Stage 2

stage2_dataset, stage2_table = get_sql_source(
    "X_CDS_RESULTS_VW"
)

stage2_contract_meta = resolve_schema_column(
    stage2_dataset,
    "X_CDS_RESULTS_VW",
    ["CONTRACT_REFERENCE", "INSTRUMENT_ID"],
)
stage2_entity_meta = resolve_schema_column(
    stage2_dataset,
    "X_CDS_RESULTS_VW",
    ["COUNTERPARTY_RISK", "ENTITY_ID"],
)
stage2_date_meta = resolve_schema_column(
    stage2_dataset,
    "X_CDS_RESULTS_VW",
    ["REPORT_DATEKEY", "MAP_DTE"],
)
stage2_stage_meta = resolve_schema_column(
    stage2_dataset,
    "X_CDS_RESULTS_VW",
    ["STAGE", "IFRS9_STAGE", "DIM1"],
)

try:
    stage2_version_meta = resolve_schema_column(
        stage2_dataset,
        "X_CDS_RESULTS_VW",
        ["REPORT_VERSIONID", "REPORT_VERSION_ID"],
    )
except KeyError:
    stage2_version_meta = None

stage2_contract_col = stage2_contract_meta["name"]
stage2_entity_col = stage2_entity_meta["name"]
stage2_date_col = stage2_date_meta["name"]
stage2_stage_col = stage2_stage_meta["name"]
stage2_version_col = (
    stage2_version_meta["name"]
    if stage2_version_meta is not None
    else None
)

stage2_date_type = str(
    stage2_date_meta.get("type", "")
).lower()

stage2_numeric_date = (
    normalized_column_name(stage2_date_col) == "REPORT_DATEKEY"
    or any(
        token in stage2_date_type
        for token in [
            "int",
            "long",
            "bigint",
            "double",
            "float",
            "decimal",
            "number",
        ]
    )
)

reference_date_int = int(
    REFERENCE_DATE.strftime("%Y%m%d")
)

if stage2_numeric_date:
    stage2_date_filter = "{column} <= {date_value}".format(
        column=stage2_date_col,
        date_value=reference_date_int,
    )
else:
    stage2_date_filter = "TRUNC({column}) <= DATE '{date_value}'".format(
        column=stage2_date_col,
        date_value=REFERENCE_DATE_STR,
    )

stage2_dates_query = """
SELECT DISTINCT
    {date_column} AS REPORT_DATE_VALUE
FROM {table}
WHERE {date_filter}
ORDER BY {date_column} DESC
FETCH FIRST 2 ROWS ONLY
""".format(
    date_column=stage2_date_col,
    table=stage2_table,
    date_filter=stage2_date_filter,
)

stage2_dates = SQLExecutor2(
    dataset=stage2_dataset
).query_to_df(
    stage2_dates_query
)

if len(stage2_dates) < 2:
    raise ValueError(
        "Two Stage 2 source snapshots are required on or before {}.".format(
            REFERENCE_DATE_STR
        )
    )

if stage2_numeric_date:
    stage2_values = (
        pd.to_numeric(
            stage2_dates["REPORT_DATE_VALUE"],
            errors="coerce",
        )
        .dropna()
        .astype("int64")
        .tolist()
    )
    stage2_current_value = stage2_values[0]
    stage2_previous_value = stage2_values[1]

    STAGE2_CURRENT_SOURCE_DATE = pd.to_datetime(
        str(stage2_current_value),
        format="%Y%m%d",
        errors="raise",
    ).normalize()
    STAGE2_PREVIOUS_SOURCE_DATE = pd.to_datetime(
        str(stage2_previous_value),
        format="%Y%m%d",
        errors="raise",
    ).normalize()

    stage2_snapshot_filter = "{column} IN ({current_value}, {previous_value})".format(
        column=stage2_date_col,
        current_value=stage2_current_value,
        previous_value=stage2_previous_value,
    )
else:
    stage2_dates["REPORT_DATE_VALUE"] = normalize_date(
        stage2_dates["REPORT_DATE_VALUE"]
    )
    stage2_dates = stage2_dates.dropna(
        subset=["REPORT_DATE_VALUE"]
    )

    STAGE2_CURRENT_SOURCE_DATE = pd.Timestamp(
        stage2_dates.iloc[0]["REPORT_DATE_VALUE"]
    ).normalize()
    STAGE2_PREVIOUS_SOURCE_DATE = pd.Timestamp(
        stage2_dates.iloc[1]["REPORT_DATE_VALUE"]
    ).normalize()

    stage2_current_value = STAGE2_CURRENT_SOURCE_DATE.strftime("%Y-%m-%d")
    stage2_previous_value = STAGE2_PREVIOUS_SOURCE_DATE.strftime("%Y-%m-%d")

    stage2_snapshot_filter = (
        "TRUNC({column}) IN (DATE '{current_value}', DATE '{previous_value}')"
        .format(
            column=stage2_date_col,
            current_value=stage2_current_value,
            previous_value=stage2_previous_value,
        )
    )

if stage2_version_col is not None:
    version_select = ", {version_col} AS REPORT_VERSIONID".format(
        version_col=stage2_version_col
    )
    version_order = "REPORT_VERSIONID DESC"
else:
    version_select = ""
    version_order = "INSTRUMENT_ID"

stage2_query = """
WITH source_rows AS (
    SELECT
        TRIM(CAST({contract_col} AS VARCHAR(100))) AS INSTRUMENT_ID,
        TRIM(CAST({entity_col} AS VARCHAR(100))) AS ENTITY_ID,
        {date_col} AS REPORT_DATE_VALUE,
        {stage_col} AS STAGE
        {version_select}
    FROM {table}
    WHERE {snapshot_filter}
      AND {contract_col} IS NOT NULL
      AND {entity_col} IS NOT NULL
),
ranked AS (
    SELECT
        source_rows.*,
        ROW_NUMBER() OVER (
            PARTITION BY INSTRUMENT_ID, REPORT_DATE_VALUE
            ORDER BY {version_order}
        ) AS RN
    FROM source_rows
)
SELECT
    INSTRUMENT_ID,
    ENTITY_ID,
    REPORT_DATE_VALUE,
    STAGE
FROM ranked
WHERE RN = 1
""".format(
    contract_col=stage2_contract_col,
    entity_col=stage2_entity_col,
    date_col=stage2_date_col,
    stage_col=stage2_stage_col,
    version_select=version_select,
    table=stage2_table,
    snapshot_filter=stage2_snapshot_filter,
    version_order=version_order,
)

stage2_snapshot = SQLExecutor2(
    dataset=stage2_dataset
).query_to_df(
    stage2_query
)

if stage2_snapshot.empty:
    raise ValueError(
        "Stage 2 source query returned no rows."
    )

stage2_snapshot["INSTRUMENT_ID"] = normalize_id(
    stage2_snapshot["INSTRUMENT_ID"]
)
stage2_snapshot["ENTITY_ID"] = normalize_id(
    stage2_snapshot["ENTITY_ID"]
)
stage2_snapshot["STAGE"] = pd.to_numeric(
    stage2_snapshot["STAGE"],
    errors="coerce",
)

if stage2_numeric_date:
    stage2_snapshot["REPORT_DATE_PARSED"] = pd.to_datetime(
        pd.to_numeric(
            stage2_snapshot["REPORT_DATE_VALUE"],
            errors="coerce",
        )
        .round()
        .astype("Int64")
        .astype("string"),
        format="%Y%m%d",
        errors="coerce",
    )
else:
    stage2_snapshot["REPORT_DATE_PARSED"] = normalize_date(
        stage2_snapshot["REPORT_DATE_VALUE"]
    )

stage2_current = stage2_snapshot.loc[
    stage2_snapshot["REPORT_DATE_PARSED"].eq(
        STAGE2_CURRENT_SOURCE_DATE
    ),
    [
        "INSTRUMENT_ID",
        "ENTITY_ID",
        "STAGE",
    ],
].rename(
    columns={"STAGE": "CURRENT_STAGE"}
)

stage2_previous = stage2_snapshot.loc[
    stage2_snapshot["REPORT_DATE_PARSED"].eq(
        STAGE2_PREVIOUS_SOURCE_DATE
    ),
    [
        "INSTRUMENT_ID","STAGE",
    ],
].rename(
    columns={"STAGE": "PREVIOUS_STAGE"}
)


if stage2_current["INSTRUMENT_ID"].duplicated().any():
    raise ValueError(
        "Current Stage 2 source snapshot contains duplicate INSTRUMENT_ID rows."
    )
if stage2_previous["INSTRUMENT_ID"].duplicated().any():
    raise ValueError(
        "Previous Stage 2 source snapshot contains duplicate INSTRUMENT_ID rows."
    )

stage2_transition = stage2_current.merge(
    stage2_previous,
    how="left",
    on="INSTRUMENT_ID",
    validate="1:1",
)

stage2_transition["STAGE2_PREVIOUS_STAGE_AVAILABLE"] = flag(
    stage2_transition["PREVIOUS_STAGE"].notna()
)
stage2_transition["STAGE2_S1_TO_S2_RAW"] = flag(
    stage2_transition["PREVIOUS_STAGE"].eq(1)
    & stage2_transition["CURRENT_STAGE"].eq(2)
)
stage2_transition["STAGE2_S3_TO_S2_EXCLUDED"] = flag(
    stage2_transition["PREVIOUS_STAGE"].eq(3)
    & stage2_transition["CURRENT_STAGE"].eq(2)
)

stage2_entity = (
    stage2_transition
    .dropna(subset=["ENTITY_ID"])
    .groupby("ENTITY_ID", as_index=False)
    .agg(
        STAGE2_SOURCE_COVERAGE_FLAG=(
            "INSTRUMENT_ID",
            lambda series: 1 if len(series) > 0 else 0,
        ),
        STAGE2_CURRENT_INSTRUMENT_COUNT=("INSTRUMENT_ID", "nunique"),
        STAGE2_PREVIOUS_STAGE_COVERAGE_COUNT=(
            "STAGE2_PREVIOUS_STAGE_AVAILABLE",
            "sum",
        ),
        STAGE2_S1_TO_S2_RAW=("STAGE2_S1_TO_S2_RAW", "max"),
        STAGE2_S1_TO_S2_INSTRUMENT_COUNT=("STAGE2_S1_TO_S2_RAW", "sum"),
        STAGE2_S3_TO_S2_EXCLUDED_COUNT=("STAGE2_S3_TO_S2_EXCLUDED", "sum"),
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### RRE LTV features

ltv_source = dataiku.Dataset(
    "loan_RRE_ltv_previous"
).get_dataframe()

ltv_entity_col = resolve_frame_column(
    ltv_source,
    "loan_RRE_ltv_previous",
    ["ENTITY_ID"],
)
ltv_date_col = resolve_frame_column(
    ltv_source,
    "loan_RRE_ltv_previous",
    ["REFERENCE_DATE", "MAP_DTE", "reference_date"],
)
ltv_outstanding_col = resolve_frame_column(
    ltv_source,
    "loan_RRE_ltv_previous",
    ["OUTSTANDING_NOMINAL_AMOUNT"],
)
ltv_collateral_col = resolve_frame_column(
    ltv_source,
    "loan_RRE_ltv_previous",
    ["ALLOCATEDCOLLATERALVALUE", "ALLOCATED_COLLATERAL_VALUE"],
)

ltv_source = ltv_source[
    [
        ltv_entity_col,
        ltv_date_col,
        ltv_outstanding_col,
        ltv_collateral_col,
    ]
].rename(
    columns={
        ltv_entity_col: "ENTITY_ID",
        ltv_date_col: "MAP_DTE",
        ltv_outstanding_col: "OUTSTANDING_NOMINAL_AMOUNT",
        ltv_collateral_col: "ALLOCATEDCOLLATERALVALUE",
    }
)

ltv_source["ENTITY_ID"] = normalize_id(
    ltv_source["ENTITY_ID"]
)
ltv_source = keep_reference_date(
    ltv_source,
    "loan_RRE_ltv_previous",
    REFERENCE_DATE,
)

for column in [
    "OUTSTANDING_NOMINAL_AMOUNT",
    "ALLOCATEDCOLLATERALVALUE",
]:
    ltv_source[column] = pd.to_numeric(
        ltv_source[column],
        errors="coerce",
    )

ltv_entity = (
    ltv_source
    .groupby("ENTITY_ID", as_index=False)
    .agg(
        RRE_OUTSTANDING_NOMINAL_AMOUNT=(
            "OUTSTANDING_NOMINAL_AMOUNT",
            "sum",
        ),
        RRE_ALLOCATED_COLLATERAL_VALUE=(
            "ALLOCATEDCOLLATERALVALUE",
            "sum",
        ),
    )
)

ltv_entity["LTV"] = np.where(
    ltv_entity["RRE_ALLOCATED_COLLATERAL_VALUE"].gt(0),
    ltv_entity["RRE_OUTSTANDING_NOMINAL_AMOUNT"]
    / ltv_entity["RRE_ALLOCATED_COLLATERAL_VALUE"],
    np.nan,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Current overdraft and GEC exposure

balances = dataiku.Dataset(
    "Customer_Balances_Previous"
).get_dataframe()

ensure_columns(
    balances,
    "Customer_Balances_Previous",
    [
        "ENTITY_ID",
        "MAP_DTE",
        "OVERDRAFT_AMOUNT",
    ],
)

balances["ENTITY_ID"] = normalize_id(
    balances["ENTITY_ID"]
)
balances = keep_reference_date(
    balances,
    "Customer_Balances_Previous",
    REFERENCE_DATE,
)
balances["OVERDRAFT_AMOUNT"] = pd.to_numeric(
    balances["OVERDRAFT_AMOUNT"],
    errors="coerce",
).fillna(0.0)

if balances["ENTITY_ID"].duplicated().any():
    raise ValueError(
        "Customer_Balances_Previous contains duplicate ENTITY_ID rows on the reference date."
    )

gec = dataiku.Dataset(
    "GEC_Exposure_Current_Previous"
).get_dataframe()

ensure_columns(
    gec,
    "GEC_Exposure_Current_Previous",
    [
        "ENTITY_ID",
        "HEAD_GEC_PARENT_ID",
        "GEC_NET_EXPOSURE",
        "MAP_DTE",
    ],
)

gec["ENTITY_ID"] = normalize_id(
    gec["ENTITY_ID"]
)
gec["HEAD_GEC_PARENT_ID"] = normalize_id(
    gec["HEAD_GEC_PARENT_ID"]
)
gec = keep_reference_date(
    gec,
    "GEC_Exposure_Current_Previous",
    REFERENCE_DATE,
)
gec["GEC_NET_EXPOSURE"] = pd.to_numeric(
    gec["GEC_NET_EXPOSURE"],
    errors="coerce",
)

if gec["ENTITY_ID"].duplicated().any():
    raise ValueError(
        "GEC_Exposure_Current_Previous contains duplicate ENTITY_ID rows on the reference date."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### GEC head rating and 12-month downgrade

# GEC head rating is retrieved directly from ENTITY_VW.
entity_dataset, entity_table = get_sql_source(
    "ENTITY_VW"
)

gec_head_rating_query = """
WITH snapshot_dates AS (

    SELECT
        'CURRENT' AS SNAPSHOT,
        DATE '{reference_date}' AS SNAPSHOT_DATE
    FROM dual

    UNION ALL

    SELECT
        'REFERENCE_12M' AS SNAPSHOT,
        DATE '{reference_12m_date}' AS SNAPSHOT_DATE
    FROM dual

),

entity_ranked AS (

    SELECT
        s.SNAPSHOT,
        e.ENTITY_ID,
        e.PD_CRR,

        ROW_NUMBER() OVER (
            PARTITION BY
                s.SNAPSHOT,
                e.ENTITY_ID
            ORDER BY
                e.TEK_STRT_DTE DESC,
                NVL(
                    e.TEK_END_DTE,
                    DATE '9999-12-31'
                ) DESC
        ) AS RN

    FROM snapshot_dates s

    INNER JOIN {entity_table} e
        ON e.TEK_STRT_DTE < (s.SNAPSHOT_DATE + 1)
        AND NVL(e.TEK_END_DTE, DATE '9999-12-31') >= (s.SNAPSHOT_DATE + 1)

    WHERE e.ENTITY_ID IS NOT NULL

)

SELECT
    SNAPSHOT,
    ENTITY_ID,
    PD_CRR

FROM entity_ranked

WHERE RN = 1
""".format(
    entity_table=entity_table,
    reference_date=REFERENCE_DATE_STR,
    reference_12m_date=REFERENCE_12M_DATE_STR,
)


gec_head_ids = (
    gec["HEAD_GEC_PARENT_ID"].dropna().astype("string").str.strip()
    .drop_duplicates().tolist()
)

# IDs are numeric in Oracle. Fall back to the original query if another format exists.
if gec_head_ids and all(re.fullmatch(r"[0-9]+", str(value)) for value in gec_head_ids):
    head_chunks = [
        gec_head_ids[i:i + 900]
        for i in range(0, len(gec_head_ids), 900)
    ]

    def fetch_gec_head_ratings(ids):
        sql_ids = ",".join("'{}'".format(value) for value in ids)
        query_for_heads = gec_head_rating_query.replace(
            "WHERE e.ENTITY_ID IS NOT NULL",
            "WHERE e.ENTITY_ID IN ({})".format(sql_ids)
        )
        if query_for_heads == gec_head_rating_query:
            raise ValueError("Could not add GEC head population filter.")
        return SQLExecutor2(dataset=entity_dataset).query_to_df(query_for_heads)

    begin = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(2, len(head_chunks))) as executor:
        gec_head_parts = list(executor.map(fetch_gec_head_ratings, head_chunks))
    gec_head_rating_source = pd.concat(gec_head_parts, ignore_index=True)
    print("GEC head IDs:", len(gec_head_ids))
    print("GEC head SQL seconds:", round(time.perf_counter() - begin, 1))
elif not gec_head_ids:
    gec_head_rating_source = pd.DataFrame(columns=["ENTITY_ID", "SNAPSHOT", "PD_CRR"])
else:
    print("Non-numeric GEC head IDs detected; using unchanged GEC head query.")
    gec_head_rating_source = SQLExecutor2(dataset=entity_dataset).query_to_df(gec_head_rating_query)

gec_head_rating_source["ENTITY_ID"] = normalize_id(
    gec_head_rating_source["ENTITY_ID"]
)
gec_head_rating_source["PD_CRR"] = pd.to_numeric(
    gec_head_rating_source["PD_CRR"],
    errors="coerce",
)

gec_head_rating_source[
    "RATING"
], gec_head_rating_source[
    "RATING_NOTCH"
] = map_pd_to_master_scale(
    gec_head_rating_source["PD_CRR"]
)

gec_head_rating = (
    gec_head_rating_source[
        [
            "ENTITY_ID",
            "SNAPSHOT",
            "RATING",
            "RATING_NOTCH",
        ]
    ]
    .pivot(
        index="ENTITY_ID",
        columns="SNAPSHOT",
        values=[
            "RATING",
            "RATING_NOTCH",
        ],
    )
)



gec_head_rating.columns = [
    "{}_{}".format(value, snapshot)
    for value, snapshot in gec_head_rating.columns
]
gec_head_rating = gec_head_rating.reset_index()

for column in [
    "RATING_CURRENT",
    "RATING_REFERENCE_12M",
    "RATING_NOTCH_CURRENT",
    "RATING_NOTCH_REFERENCE_12M",
]:
    if column not in gec_head_rating.columns:
        gec_head_rating[column] = np.nan

gec_head_rating = gec_head_rating.rename(
    columns={
        "ENTITY_ID": "HEAD_GEC_PARENT_ID",
        "RATING_CURRENT": "GEC_HEAD_RATING",
        "RATING_REFERENCE_12M": "GEC_HEAD_RATING_12M",
        "RATING_NOTCH_CURRENT": "GEC_HEAD_RATING_NOTCH",
        "RATING_NOTCH_REFERENCE_12M": "GEC_HEAD_RATING_NOTCH_12M",
    }
)

gec_head_rating["GEC_HEAD_RATING_NOTCH"] = pd.to_numeric(
    gec_head_rating["GEC_HEAD_RATING_NOTCH"],
    errors="coerce",
)
gec_head_rating["GEC_HEAD_RATING_NOTCH_12M"] = pd.to_numeric(
    gec_head_rating["GEC_HEAD_RATING_NOTCH_12M"],
    errors="coerce",
)
gec_head_rating["GEC_HEAD_DOWNGRADE_12M"] = (
    gec_head_rating["GEC_HEAD_RATING_NOTCH"]
    - gec_head_rating["GEC_HEAD_RATING_NOTCH_12M"]
)

if gec_head_rating["HEAD_GEC_PARENT_ID"].duplicated().any():
    raise ValueError(
        "GEC head rating lookup contains duplicate HEAD_GEC_PARENT_ID rows."
    )


gec = gec.merge(
    gec_head_rating,
    how="left",
    on="HEAD_GEC_PARENT_ID",
    validate="m:1",
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Employment-status transition

occupation_dataset, occupation_table = get_sql_source(
    "OCCUPATION_VW"
)

_, customer_table = get_sql_source(
    "CUSTOMER_VW"
)

employment_query = """
WITH snapshot_dates AS (

    SELECT
        'PREVIOUS' AS SNAPSHOT,
        DATE '{previous_date}' AS SNAPSHOT_DATE
    FROM dual

    UNION ALL

    SELECT
        'CURRENT' AS SNAPSHOT,
        DATE '{reference_date}' AS SNAPSHOT_DATE
    FROM dual
),

occupation_ranked AS (

    SELECT
        d.SNAPSHOT,
        o.CUSTOMER_SK,
        o.EMPLOYMENT_STATUS,
        o.EMPLOYMENT_STATUS_LABEL,
        ROW_NUMBER() OVER (
            PARTITION BY
                d.SNAPSHOT,
                o.CUSTOMER_SK
            ORDER BY
                o.TEK_STRT_DTE DESC,
                NVL(o.TEK_END_DTE, DATE '9999-12-31') DESC,
                o.OCCUPATION_SK DESC
        ) AS RN

    FROM snapshot_dates d

    INNER JOIN {occupation_table} o
        ON o.TEK_STRT_DTE < (d.SNAPSHOT_DATE + 1)
       AND NVL(o.TEK_END_DTE, DATE '9999-12-31') >= (d.SNAPSHOT_DATE + 1)

    WHERE o.CUSTOMER_SK IS NOT NULL
),

occupation_snapshots AS (

    SELECT
        SNAPSHOT,
        CUSTOMER_SK,
        EMPLOYMENT_STATUS,
        EMPLOYMENT_STATUS_LABEL

    FROM occupation_ranked

    WHERE RN = 1
),

customer_ranked AS (

    SELECT
        cust.CUSTOMER_SK,
        cust.CUSTOMER_ID,
        ROW_NUMBER() OVER (
            PARTITION BY cust.CUSTOMER_SK
            ORDER BY
                cust.TEK_STRT_DTE DESC,
                NVL(cust.TEK_END_DTE, DATE '9999-12-31') DESC
        ) AS RN

    FROM {customer_table} cust

    WHERE cust.TEK_STRT_DTE < (DATE '{reference_date}' + 1)
      AND NVL(cust.TEK_END_DTE, DATE '9999-12-31') >= (DATE '{reference_date}' + 1)
      AND cust.CUSTOMER_SK IS NOT NULL
      AND cust.CUSTOMER_ID IS NOT NULL
)

SELECT
    p.CUSTOMER_SK,
    cust.CUSTOMER_ID,
    p.EMPLOYMENT_STATUS AS PREVIOUS_EMPLOYMENT_STATUS,
    p.EMPLOYMENT_STATUS_LABEL AS PREVIOUS_EMPLOYMENT_STATUS_LABEL,
    c.EMPLOYMENT_STATUS AS CURRENT_EMPLOYMENT_STATUS,
    c.EMPLOYMENT_STATUS_LABEL AS CURRENT_EMPLOYMENT_STATUS_LABEL

FROM occupation_snapshots p

INNER JOIN occupation_snapshots c
    ON p.CUSTOMER_SK = c.CUSTOMER_SK
   AND p.SNAPSHOT = 'PREVIOUS'
   AND c.SNAPSHOT = 'CURRENT'

INNER JOIN customer_ranked cust
    ON p.CUSTOMER_SK = cust.CUSTOMER_SK
   AND cust.RN = 1
""".format(
    previous_date=PREVIOUS_DATE_STR,
    reference_date=REFERENCE_DATE_STR,
    occupation_table=occupation_table,
    customer_table=customer_table,
)

employment_source = SQLExecutor2(
    dataset=occupation_dataset
).query_to_df(
    employment_query
)

ensure_columns(
    employment_source,
    "Employment status source",
    [
        "CUSTOMER_ID",
        "PREVIOUS_EMPLOYMENT_STATUS",
        "PREVIOUS_EMPLOYMENT_STATUS_LABEL",
        "CURRENT_EMPLOYMENT_STATUS",
    ],
)

employment_source["CUSTOMER_ID"] = normalize_id(
    employment_source["CUSTOMER_ID"]
)

for column in [
    "PREVIOUS_EMPLOYMENT_STATUS",
    "PREVIOUS_EMPLOYMENT_STATUS_LABEL",
    "CURRENT_EMPLOYMENT_STATUS",
    "CURRENT_EMPLOYMENT_STATUS_LABEL",
]:
    if column in employment_source.columns:
        employment_source[column] = (
            employment_source[column]
            .astype("string")
            .str.strip()
            .str.upper()
        )

previous_employed_text = (
    employment_source["PREVIOUS_EMPLOYMENT_STATUS"].fillna("")
    + " "
    + employment_source["PREVIOUS_EMPLOYMENT_STATUS_LABEL"].fillna("")
)

previous_employed = previous_employed_text.str.contains(
    r"SELF[ ._-]*EMPLOY|WAGE|SALARI",
    regex=True,
    na=False,
)

employment_source["EMPLOYMENT_LOSS_EVENT"] = flag(
    previous_employed
    & employment_source["CURRENT_EMPLOYMENT_STATUS"].eq(
        "700.NO.OCCUPATION"
    )
)

employment_event_origins = (
    employment_source.loc[
        employment_source["EMPLOYMENT_LOSS_EVENT"].eq(1),
        [
            "PREVIOUS_EMPLOYMENT_STATUS",
            "PREVIOUS_EMPLOYMENT_STATUS_LABEL",
        ],
    ]
    .value_counts(dropna=False)
    .reset_index(name="EVENT_COUNT")
)

employment_customer = (
    employment_source
    .merge(
        entity_customer_map,
        how="inner",
        on="CUSTOMER_ID",
        validate="m:1",
    )
)

employment_entity = (
    employment_customer
    .groupby(
        "ENTITY_ID",
        as_index=False,
    )
    .agg(
        EMPLOYMENT_LOSS_EVENT_BASE=(
            "EMPLOYMENT_LOSS_EVENT",
            "max",
        ),
        EMPLOYMENT_LOSS_CUSTOMER_COUNT=(
            "EMPLOYMENT_LOSS_EVENT",
            "sum",
        ),
    )
)

employment_entity["EMPLOYMENT_LOSS_EVENT_BASE"] = (
    pd.to_numeric(
        employment_entity["EMPLOYMENT_LOSS_EVENT_BASE"],
        errors="coerce",
    )
    .fillna(0)
    .astype("int8")
)




# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Consolidated daily feature set

current = (
    entity
    .merge(
        employment_entity,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        portfolio.drop(columns=["MAP_DTE"]),
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        history,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        behavioural.drop(columns=["MAP_DTE"]),
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        cc_history,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        dpd_arrears,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        previous_dpd,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        payment_entity,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        loan_entity,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        connected,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        stage2_entity,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        ltv_entity,
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        balances[
            [
                "ENTITY_ID",
                "OVERDRAFT_AMOUNT",
            ]
        ],
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
    .merge(
        gec[
            [
                "ENTITY_ID",
                "HEAD_GEC_PARENT_ID",
                "GEC_NET_EXPOSURE",
                "GEC_HEAD_RATING",
                "GEC_HEAD_RATING_12M",
                "GEC_HEAD_RATING_NOTCH",
                "GEC_HEAD_RATING_NOTCH_12M",
                "GEC_HEAD_DOWNGRADE_12M",
            ]
        ],
        how="left",
        on="ENTITY_ID",
        validate="1:1",
    )
)

current["SALARY_3M_COMPLETE_HISTORY"] = np.int8(0)
current["SALARY_YOY_DECREASE_20PCT_3M"] = np.int8(0)

for column in [
    "RRE_FLAG",
    "OTHER_RETAIL_FLAG",
    "RRE_INSTRUMENT_COUNT",
    "OTHER_RETAIL_INSTRUMENT_COUNT",
    "FAILED_DEBIT_COUNT",
    "FAILED_DEBIT_AMOUNT",
    "FAILED_DEBIT_COUNT_MTD",
    "FAILED_DEBIT_AMOUNT_MTD",
    "FORBEARANCE_FLAG",
    "PROBATION_FLAG",
    "UTP_FLAG",
    "CURRENT_STAGE2_FLAG",
    "GCC_MEMBER_COUNT",
    "GCC_STAGE2_CONNECTED_MEMBER_COUNT",
    "GCC_UTP_CONNECTED_MEMBER_COUNT",
    "GCC_STAGE2_CONNECTED_BASE",
    "GCC_UTP_CONNECTED_BASE",
    "STAGE2_SOURCE_COVERAGE_FLAG",
    "STAGE2_CURRENT_INSTRUMENT_COUNT",
    "STAGE2_PREVIOUS_STAGE_COVERAGE_COUNT",
    "STAGE2_S1_TO_S2_RAW",
    "STAGE2_S1_TO_S2_INSTRUMENT_COUNT",
    "STAGE2_S3_TO_S2_EXCLUDED_COUNT",
    "OVERDRAFT_AMOUNT",
    "EMPLOYMENT_LOSS_EVENT_BASE",
    "EMPLOYMENT_LOSS_CUSTOMER_COUNT",
    "SALARY_3M_COMPLETE_HISTORY",
    "SALARY_YOY_DECREASE_20PCT_3M",
]:
    if column in current.columns:
        current[column] = pd.to_numeric(
            current[column],
            errors="coerce",
        ).fillna(0)

for column in [
    "RRE_FLAG",
    "OTHER_RETAIL_FLAG",
    "FORBEARANCE_FLAG",
    "PROBATION_FLAG",
    "UTP_FLAG",
    "CURRENT_STAGE2_FLAG",
    "GCC_STAGE2_CONNECTED_BASE",
    "GCC_UTP_CONNECTED_BASE",
    "STAGE2_SOURCE_COVERAGE_FLAG",
    "STAGE2_S1_TO_S2_RAW",
    "EMPLOYMENT_LOSS_EVENT_BASE",
    "SALARY_3M_COMPLETE_HISTORY",
    "SALARY_YOY_DECREASE_20PCT_3M",
]:
    current[column] = current[column].astype("int8")

current["MAP_DTE"] = REFERENCE_DATE
current["PREVIOUS_MAP_DTE"] = PREVIOUS_DATE
current["STAGE2_CURRENT_SOURCE_DATE"] = STAGE2_CURRENT_SOURCE_DATE
current["STAGE2_PREVIOUS_SOURCE_DATE"] = STAGE2_PREVIOUS_SOURCE_DATE


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Shared payment and distress conditions

# DPD / overdraft / GEC active conditions
current["DPD_15_ACTIVE"] = flag(
    current["DPD"].gt(DPD_15_THRESHOLD)
    & current["OVERDRAFT_AMOUNT"].gt(OVERDRAFT_THRESHOLD)
    & current["GEC_NET_EXPOSURE"].gt(GEC_DPD_15_THRESHOLD)
)

current["DPD_30_ACTIVE"] = flag(
    current["DPD"].gt(DPD_30_THRESHOLD)
    & current["OVERDRAFT_AMOUNT"].gt(OVERDRAFT_THRESHOLD)
    & current["GEC_NET_EXPOSURE"].gt(GEC_DPD_30_THRESHOLD)
)

current["DPD_60_ACTIVE"] = flag(
    current["DPD"].gt(DPD_60_THRESHOLD)
    & current["OVERDRAFT_AMOUNT"].gt(OVERDRAFT_THRESHOLD)
    & current["GEC_NET_EXPOSURE"].gt(GEC_DPD_60_THRESHOLD)
)

# Crossing events. Missing PREVIOUS_DPD is not converted to zero: without a
# previous observation we cannot prove that the threshold was crossed.
current["DPD_15_CROSSING_RAW"] = flag(
    current["PREVIOUS_DPD"].notna()
    & current["PREVIOUS_DPD"].le(DPD_15_THRESHOLD)
    & current["DPD_15_ACTIVE"].eq(1)
)
current["DPD_30_CROSSING_RAW"] = flag(
    current["PREVIOUS_DPD"].notna()
    & current["PREVIOUS_DPD"].le(DPD_30_THRESHOLD)
    & current["DPD_30_ACTIVE"].eq(1)
)
current["DPD_60_CROSSING_RAW"] = flag(
    current["PREVIOUS_DPD"].notna()
    & current["PREVIOUS_DPD"].le(DPD_60_THRESHOLD)
    & current["DPD_60_ACTIVE"].eq(1)
)

# Keep only the highest qualifying crossing when several thresholds are crossed
# on the same day. Priority: 60 > 30 > 15.
current["DPD_EVENT_TIER"] = np.select(
    [
        current["DPD_60_CROSSING_RAW"].eq(1),
        current["DPD_30_CROSSING_RAW"].eq(1),
        current["DPD_15_CROSSING_RAW"].eq(1),
    ],
    [60, 30, 15],
    default=0,
).astype("int16")

current["PAYMENT_DPD_15_EVENT_BASE"] = flag(
    current["DPD_EVENT_TIER"].eq(15)
)
current["PAYMENT_DPD_30_EVENT_BASE"] = flag(
    current["DPD_EVENT_TIER"].eq(30)
)
current["PAYMENT_DPD_60_EVENT_BASE"] = flag(
    current["DPD_EVENT_TIER"].eq(60)
)

current["FORBEARANCE_PROBATION_DPD_GT_0_BASE"] = flag(
    (
        current["FORBEARANCE_FLAG"].eq(1)
        | current["PROBATION_FLAG"].eq(1)
    )
    & current["DPD"].gt(0)
)

current["STAGE2_DOWNGRADE_BASE"] = flag(
    current["STAGE2_S1_TO_S2_RAW"].eq(1)
)

stage2_downgrade_materiality = materiality_mask(
    current["GEC_NET_EXPOSURE"],
    STAGE2_DOWNGRADE_GEC_MATERIALITY_THRESHOLD,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### RRE payment and distress indicators

current["RRE_FORBEARANCE_PROBATION_DPD_GT_0"] = portfolio_flag(
    current["FORBEARANCE_PROBATION_DPD_GT_0_BASE"].eq(1),
    current["RRE_FLAG"],
)

for source, target in [
    ("PAYMENT_DPD_15_EVENT_BASE", "RRE_DPD_15_EVENT"),
    ("PAYMENT_DPD_30_EVENT_BASE", "RRE_DPD_30_EVENT"),
    ("PAYMENT_DPD_60_EVENT_BASE", "RRE_DPD_60_EVENT"),
]:
    current[target] = portfolio_flag(
        current[source].eq(1),
        current["RRE_FLAG"],
    )

current["RRE_DOWNGRADE_TO_STAGE2"] = np.int8(0)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Other Retail payment and distress indicators

current["OTHER_FORBEARANCE_PROBATION_DPD_GT_0"] = portfolio_flag(
    current["FORBEARANCE_PROBATION_DPD_GT_0_BASE"].eq(1),
    current["OTHER_RETAIL_FLAG"],
)

for source, target in [
    ("PAYMENT_DPD_15_EVENT_BASE", "OTHER_DPD_15_EVENT"),
    ("PAYMENT_DPD_30_EVENT_BASE", "OTHER_DPD_30_EVENT"),
    ("PAYMENT_DPD_60_EVENT_BASE", "OTHER_DPD_60_EVENT"),
]:
    current[target] = portfolio_flag(
        current[source].eq(1),
        current["OTHER_RETAIL_FLAG"],
    )

current["OTHER_DOWNGRADE_TO_STAGE2"] = np.int8(0)

current["OTHER_FAILED_DIRECT_DEBITS_MORE_THAN_6_MTD"] = portfolio_flag(
    current["FAILED_DEBIT_COUNT_MTD"].gt(
        FAILED_DEBIT_COUNT_MTD_THRESHOLD
    ),
    current["OTHER_RETAIL_FLAG"],
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Personal circumstances

current["RRE_EMPLOYMENT_LOSS_EVENT"] = portfolio_flag(
    current["EMPLOYMENT_LOSS_EVENT_BASE"].eq(1),
    current["RRE_FLAG"],
)

current["OTHER_EMPLOYMENT_LOSS_EVENT"] = portfolio_flag(
    current["EMPLOYMENT_LOSS_EVENT_BASE"].eq(1),
    current["OTHER_RETAIL_FLAG"],
)

current["RRE_SALARY_YOY_DECREASE_20PCT_3M"] = np.int8(0)
current["OTHER_SALARY_YOY_DECREASE_20PCT_3M"] = np.int8(0)

current["RRE_PERSONAL_CIRCUMSTANCES_EVENT"] = portfolio_flag(
    current["EMPLOYMENT_LOSS_EVENT_BASE"].eq(1),
    current["RRE_FLAG"],
)

current["OTHER_PERSONAL_CIRCUMSTANCES_EVENT"] = portfolio_flag(
    current["EMPLOYMENT_LOSS_EVENT_BASE"].eq(1),
    current["OTHER_RETAIL_FLAG"],
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### RRE behavioural indicator

current["RRE_CASH_NEOBANK_ABOVE_4X_POSITIVE_BALANCES"] = portfolio_flag(
    current["cash_neobank_balance_ratio"].gt(
        RRE_CASH_NEOBANK_RATIO_THRESHOLD
    ),
    current["RRE_FLAG"],
)

current["RRE_CASH_ADVANCE_LIMIT_AFTER_ABOVE_94"] = portfolio_flag(
    current["cash_advance_limit_cross_94_1d"].eq(1),
    current["RRE_FLAG"],
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Other Retail behavioural indicators

current["OTHER_CASH_NEOBANK_ABOVE_HALF_POSITIVE_BALANCES"] = portfolio_flag(
    current["cash_neobank_balance_ratio"].gt(
        OTHER_CASH_NEOBANK_RATIO_THRESHOLD
    ),
    current["OTHER_RETAIL_FLAG"],
)

current["OTHER_CASH_ADVANCE_LIMIT_AFTER_ABOVE_97"] = portfolio_flag(
    current["cash_advance_limit_cross_97_1d"].eq(1),
    current["OTHER_RETAIL_FLAG"],
)

current["OTHER_CASH_ADVANCE_AMOUNT_ABOVE_5000"] = portfolio_flag(
    current["cash_advance_amount"].gt(
        OTHER_CASH_ADVANCE_AMOUNT_THRESHOLD
    ),
    current["OTHER_RETAIL_FLAG"],
)

current["OTHER_NEOBANK_TOPUP_AMOUNT_ABOVE_1000"] = portfolio_flag(
    current["neobank_topup_amount"].gt(
        OTHER_NEOBANK_TOPUP_AMOUNT_THRESHOLD
    ),
    current["OTHER_RETAIL_FLAG"],
)

current["OTHER_CC_UTILIZATION_ABOVE_100"] = portfolio_flag(
    current["cc_limit_utilization"].gt(
        OTHER_CC_UTILIZATION_THRESHOLD
    ),
    current["OTHER_RETAIL_FLAG"],
)

current["CC_UTILIZATION_DAILY_WORSENING"] = flag(
    current["cc_limit_utilization"].notna()
    & current[
        "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION"
    ].notna()
    & current["cc_limit_utilization"].gt(
        current[
            "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION"
        ]
    )
)

current["OTHER_CC_ABS_CHANGE_12M_ABOVE_95_CURRENT_ABOVE_95"] = portfolio_flag(
    current["CC_UTILIZATION_ABS_CHANGE_12M"].gt(
        OTHER_CC_ABS_CHANGE_12M_THRESHOLD
    )
    & current["cc_limit_utilization"].gt(
        OTHER_CC_ABS_CHANGE_CURRENT_THRESHOLD
    )
    & current["CC_UTILIZATION_DAILY_WORSENING"].eq(1),
    current["OTHER_RETAIL_FLAG"],
)

current["CC_UTILIZATION_REL_CHANGE_3M_MAX"] = np.where(
    current["CC_MAX_UTILIZATION_3M"].gt(0),
    current["CC_UTILIZATION_ABS_CHANGE_3M_MAX"]
    / current["CC_MAX_UTILIZATION_3M"],
    np.nan,
)

current["OTHER_CC_REL_CHANGE_3M_MAX_ABOVE_100_CURRENT_ABOVE_90"] = portfolio_flag(
    pd.Series(
        current["CC_UTILIZATION_REL_CHANGE_3M_MAX"],
        index=current.index,
    ).gt(
        OTHER_CC_REL_CHANGE_3M_MAX_THRESHOLD
    )
    & current["cc_limit_utilization"].gt(
        OTHER_CC_REL_CHANGE_CURRENT_THRESHOLD
    )
    & current["CC_UTILIZATION_DAILY_WORSENING"].eq(1),
    current["OTHER_RETAIL_FLAG"],
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Client rating indicators

# RRE
current["RRE_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M_BASE"] = portfolio_flag(
    current["CLIENT_DOWNGRADE_12M"].ge(6),
    current["RRE_FLAG"],
)
current["RRE_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M_BASE"] = portfolio_flag(
    current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_BBB_MINUS)
    & current["CLIENT_DOWNGRADE_12M"].ge(5),
    current["RRE_FLAG"],
)
current["RRE_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M_BASE"] = portfolio_flag(
    current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_BB_MINUS)
    & current["CLIENT_DOWNGRADE_12M"].ge(4),
    current["RRE_FLAG"],
)
current["RRE_CLIENT_WORSE_B_PLUS_DOWNGRADE_AT_LEAST_2_VS_12M_BASE"] = portfolio_flag(
    current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_B_PLUS)
    & current["CLIENT_DOWNGRADE_12M"].ge(2),
    current["RRE_FLAG"],
)
current["RRE_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_1_VS_12M_BASE"] = portfolio_flag(
    current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_B)
    & current["CLIENT_DOWNGRADE_12M"].ge(1),
    current["RRE_FLAG"],
)

# Other Retail
current["OTHER_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M_BASE"] = portfolio_flag(
    current["CLIENT_DOWNGRADE_12M"].ge(6),
    current["OTHER_RETAIL_FLAG"],
)
current["OTHER_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M_BASE"] = portfolio_flag(
    current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_BBB_MINUS)
    & current["CLIENT_DOWNGRADE_12M"].ge(5),
    current["OTHER_RETAIL_FLAG"],
)
current["OTHER_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M_BASE"] = portfolio_flag(
    current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_BB_MINUS)
    & current["CLIENT_DOWNGRADE_12M"].ge(4),
    current["OTHER_RETAIL_FLAG"],
)
current["OTHER_CLIENT_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M_BASE"] = portfolio_flag(
    current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_B_MINUS)
    & current["CLIENT_DOWNGRADE_12M"].ge(1),
    current["OTHER_RETAIL_FLAG"],
)
current["OTHER_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_3_VS_12M_BASE"] = portfolio_flag(
    current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_B)
    & current["CLIENT_DOWNGRADE_12M"].ge(3),
    current["OTHER_RETAIL_FLAG"],
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Connected-client indicators

connected_stage2_materiality = materiality_mask(
    current["GEC_NET_EXPOSURE"],
    CONNECTED_STAGE2_GEC_MATERIALITY_THRESHOLD,
)

# GEC head rating conditions

# RRE
current["RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M_BASE"] = portfolio_flag(
    current["GEC_HEAD_RATING_NOTCH"].gt(GEC_HEAD_WORSE_THAN_B)
    & current["GEC_HEAD_DOWNGRADE_12M"].ge(4),
    current["RRE_FLAG"],
)
current["RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M_BASE"] = portfolio_flag(
    current["GEC_HEAD_RATING_NOTCH"].gt(GEC_HEAD_WORSE_THAN_B_MINUS)
    & current["GEC_HEAD_DOWNGRADE_12M"].ge(1),
    current["RRE_FLAG"],
)
current["RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M_BASE"] = portfolio_flag(
    current["GEC_HEAD_DOWNGRADE_12M"].gt(7),
    current["RRE_FLAG"],
)
current["RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_PLUS_BASE"] = portfolio_flag(
    current["GEC_HEAD_DOWNGRADE_12M"].ge(3)
    & current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_B_PLUS),
    current["RRE_FLAG"],
)
current["RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_1_CLIENT_DOWNGRADE_AT_LEAST_6_BASE"] = portfolio_flag(
    current["GEC_HEAD_DOWNGRADE_12M"].ge(1)
    & current["CLIENT_DOWNGRADE_12M"].ge(6),
    current["RRE_FLAG"],
)
current["RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5_BASE"] = portfolio_flag(
    current["GEC_HEAD_DOWNGRADE_12M"].ge(3)
    & current["CLIENT_DOWNGRADE_12M"].ge(5),
    current["RRE_FLAG"],
)
current["RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3_BASE"] = portfolio_flag(
    current["GEC_HEAD_DOWNGRADE_12M"].ge(6)
    & current["CLIENT_DOWNGRADE_12M"].ge(3),
    current["RRE_FLAG"],
)

# Other Retail
current["OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M_BASE"] = portfolio_flag(
    current["GEC_HEAD_RATING_NOTCH"].gt(GEC_HEAD_WORSE_THAN_B)
    & current["GEC_HEAD_DOWNGRADE_12M"].ge(4),
    current["OTHER_RETAIL_FLAG"],
)
current["OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M_BASE"] = portfolio_flag(
    current["GEC_HEAD_RATING_NOTCH"].gt(GEC_HEAD_WORSE_THAN_B_MINUS)
    & current["GEC_HEAD_DOWNGRADE_12M"].ge(1),
    current["OTHER_RETAIL_FLAG"],
)
current["OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M_BASE"] = portfolio_flag(
    current["GEC_HEAD_DOWNGRADE_12M"].ge(6),
    current["OTHER_RETAIL_FLAG"],
)
current["OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_BASE"] = portfolio_flag(
    current["GEC_HEAD_DOWNGRADE_12M"].ge(3)
    & current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_B),
    current["OTHER_RETAIL_FLAG"],
)
current["OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3_BASE"] = portfolio_flag(
    current["GEC_HEAD_DOWNGRADE_12M"].ge(6)
    & current["CLIENT_DOWNGRADE_12M"].ge(3),
    current["OTHER_RETAIL_FLAG"],
)
current["OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5_BASE"] = portfolio_flag(
    current["GEC_HEAD_DOWNGRADE_12M"].ge(3)
    & current["CLIENT_DOWNGRADE_12M"].ge(5),
    current["OTHER_RETAIL_FLAG"],
)

# Apply materiality threshold
gec_head_rating_pairs = [
    (
        "RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M_BASE",
        "RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
    ),
    (
        "RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M_BASE",
        "RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    ),
    (
        "RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M_BASE",
        "RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M",
    ),
    (
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_PLUS_BASE",
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_PLUS",
    ),
    (
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_1_CLIENT_DOWNGRADE_AT_LEAST_6_BASE",
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_1_CLIENT_DOWNGRADE_AT_LEAST_6",
    ),
    (
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5_BASE",
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5",
    ),
    (
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3_BASE",
        "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3",
    ),
    (
        "OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M_BASE",
        "OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
    ),
    (
        "OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M_BASE",
        "OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    ),
    (
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M_BASE",
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M",
    ),
    (
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_BASE",
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B",
    ),
    (
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3_BASE",
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3",
    ),
    (
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5_BASE",
        "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5",
    ),
]

rating_persistence, rating_persistence_qa, rating_daily_change, rating_previous_state_date, rating_trigger_date, rating_persistence_dates = (
    build_rating_persistence_flags(
        current,
        entity_dataset,
        entity_table,
        REFERENCE_DATE,
    )
)

rating_final_columns = [
    "RRE_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M",
    "RRE_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M",
    "RRE_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M",
    "RRE_CLIENT_WORSE_B_PLUS_DOWNGRADE_AT_LEAST_2_VS_12M",
    "RRE_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_1_VS_12M",
    "OTHER_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M",
    "OTHER_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M",
    "OTHER_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M",
    "OTHER_CLIENT_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "OTHER_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_3_VS_12M",
] + [
    final_column
    for _, final_column in gec_head_rating_pairs
]

current = current.merge(
    rating_persistence,
    how="left",
    on="ENTITY_ID",
    validate="1:1",
)

current = current.merge(
    rating_daily_change,
    how="left",
    on="ENTITY_ID",
    validate="1:1",
)

current["CLIENT_RATING_DAILY_WORSENING"] = pd.to_numeric(
    current["CLIENT_RATING_DAILY_WORSENING"],
    errors="coerce",
).fillna(0).astype("int8")

for column in rating_final_columns:
    current[column] = pd.to_numeric(
        current[column],
        errors="coerce",
    ).fillna(0).astype("int8")

default_mask = current["ENTITY_ID"].isin(default_entity_ids)

for column in PURE_GEC_HEAD_DOWNGRADE_CODES:
    current.loc[default_mask, column] = np.int8(0)

print(
    "Default GEC head downgrade flags excluded:",
    int(default_mask.sum()),
)

# Connected Stage 2
current["RRE_GCC_STAGE2_CONNECTED"] = portfolio_flag(
    current["GCC_STAGE2_CONNECTED_BASE"].eq(1)
    & connected_stage2_materiality,
    current["RRE_FLAG"],
)
current["OTHER_GCC_STAGE2_CONNECTED"] = portfolio_flag(
    current["GCC_STAGE2_CONNECTED_BASE"].eq(1)
    & connected_stage2_materiality,
    current["OTHER_RETAIL_FLAG"],
)

# Connected UTP
current["RRE_GCC_UTP_CONNECTED"] = portfolio_flag(
    current["GCC_UTP_CONNECTED_BASE"].eq(1),
    current["RRE_FLAG"],
)
current["OTHER_GCC_UTP_CONNECTED"] = portfolio_flag(
    current["GCC_UTP_CONNECTED_BASE"].eq(1),
    current["OTHER_RETAIL_FLAG"],
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### RRE property-linked indicators

current["RRE_LTV_ABOVE_95_CLIENT_DOWNGRADE_MORE_THAN_4_VS_12M"] = portfolio_flag(
    current["LTV"].gt(RRE_LTV_THRESHOLD)
    & current["CLIENT_DOWNGRADE_12M"].gt(4)
    & current["CLIENT_RATING_DAILY_WORSENING"].eq(1),
    current["RRE_FLAG"],
)

current["RRE_LTV_ABOVE_95_CLIENT_RATING_WORSE_B"] = portfolio_flag(
    current["LTV"].gt(RRE_LTV_THRESHOLD)
    & current["CLIENT_RATING_NOTCH"].gt(CLIENT_WORSE_THAN_B),
    current["RRE_FLAG"],
)

current["RRE_LTV_ABOVE_105_MONITORING"] = portfolio_flag(
    current["LTV"].gt(RRE_LTV_MONITORING_THRESHOLD),
    current["RRE_FLAG"],
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Daily implementation controls

RRE_ALERT_COLUMNS = [
    "RRE_FORBEARANCE_PROBATION_DPD_GT_0",
    "RRE_DPD_15_EVENT",
    "RRE_DPD_30_EVENT",
    "RRE_DPD_60_EVENT",
    "RRE_DOWNGRADE_TO_STAGE2",
    "RRE_PERSONAL_CIRCUMSTANCES_EVENT",
    "RRE_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M",
    "RRE_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M",
    "RRE_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M",
    "RRE_CLIENT_WORSE_B_PLUS_DOWNGRADE_AT_LEAST_2_VS_12M",
    "RRE_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_1_VS_12M",
    "RRE_CASH_NEOBANK_ABOVE_4X_POSITIVE_BALANCES",
    "RRE_CASH_ADVANCE_LIMIT_AFTER_ABOVE_94",
    "RRE_LTV_ABOVE_95_CLIENT_DOWNGRADE_MORE_THAN_4_VS_12M",
    "RRE_LTV_ABOVE_95_CLIENT_RATING_WORSE_B",
    "RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
    "RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M",
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_PLUS",
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_1_CLIENT_DOWNGRADE_AT_LEAST_6",
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5",
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3",
    "RRE_GCC_STAGE2_CONNECTED",
    "RRE_GCC_UTP_CONNECTED",
]


OTHER_RETAIL_ALERT_COLUMNS = [
    "OTHER_FORBEARANCE_PROBATION_DPD_GT_0",
    "OTHER_DPD_15_EVENT",
    "OTHER_DPD_30_EVENT",
    "OTHER_DPD_60_EVENT",
    "OTHER_DOWNGRADE_TO_STAGE2",
    "OTHER_FAILED_DIRECT_DEBITS_MORE_THAN_6_MTD",
    "OTHER_PERSONAL_CIRCUMSTANCES_EVENT",
    "OTHER_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M",
    "OTHER_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M",
    "OTHER_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M",
    "OTHER_CLIENT_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "OTHER_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_3_VS_12M",
    "OTHER_CASH_NEOBANK_ABOVE_HALF_POSITIVE_BALANCES",
    "OTHER_CASH_ADVANCE_LIMIT_AFTER_ABOVE_97",
    "OTHER_CASH_ADVANCE_AMOUNT_ABOVE_5000",
    "OTHER_NEOBANK_TOPUP_AMOUNT_ABOVE_1000",
    "OTHER_CC_UTILIZATION_ABOVE_100",
    "OTHER_CC_ABS_CHANGE_12M_ABOVE_95_CURRENT_ABOVE_95",
    "OTHER_CC_REL_CHANGE_3M_MAX_ABOVE_100_CURRENT_ABOVE_90",
    "OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
    "OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M",
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B",
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3",
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5",
    "OTHER_GCC_STAGE2_CONNECTED",
    "OTHER_GCC_UTP_CONNECTED",
]

all_alert_columns = RRE_ALERT_COLUMNS + OTHER_RETAIL_ALERT_COLUMNS

if len(RRE_ALERT_COLUMNS) != 24:
    raise ValueError(
        "RRE indicator catalogue contains {} indicators; expected 24.".format(
            len(RRE_ALERT_COLUMNS)
        )
    )

if len(OTHER_RETAIL_ALERT_COLUMNS) != 27:
    raise ValueError(
        "Other Retail indicator catalogue contains {} indicators; expected 27.".format(
            len(OTHER_RETAIL_ALERT_COLUMNS)
        )
    )

missing_alert_columns = [
    column
    for column in all_alert_columns
    if column not in current.columns
]

if missing_alert_columns:
    raise KeyError(
        "Daily EWS output is missing indicator column(s): {}".format(
            ", ".join(missing_alert_columns)
        )
    )

for column in all_alert_columns:
    current[column] = pd.to_numeric(
        current[column],
        errors="coerce",
    ).fillna(0).astype("int8")

current["RRE_ACTIVE_INDICATOR_COUNT"] = (
    current[RRE_ALERT_COLUMNS]
    .sum(axis=1)
    .astype("int16")
)
current["OTHER_RETAIL_ACTIVE_INDICATOR_COUNT"] = (
    current[OTHER_RETAIL_ALERT_COLUMNS]
    .sum(axis=1)
    .astype("int16")
)
current["RETAIL_ACTIVE_INDICATOR_COUNT"] = (
    current[
        [
            "RRE_ACTIVE_INDICATOR_COUNT",
            "OTHER_RETAIL_ACTIVE_INDICATOR_COUNT",
        ]
    ]
    .sum(axis=1)
    .astype("int16")
)

# Keep indicators within the relevant portfolio
invalid_rre_flags = int(
    (
        current[RRE_ALERT_COLUMNS]
        .sum(axis=1)
        .gt(0)
        & current["RRE_FLAG"].ne(1)
    ).sum()
)
invalid_other_flags = int(
    (
        current[OTHER_RETAIL_ALERT_COLUMNS]
        .sum(axis=1)
        .gt(0)
        & current["OTHER_RETAIL_FLAG"].ne(1)
    ).sum()
)

if invalid_rre_flags > 0 or invalid_other_flags > 0:
    raise ValueError(
        "Portfolio gating failed for one or more indicator rows."
    )

if current.duplicated(
    subset=["ENTITY_ID", "MAP_DTE"]
).any():
    raise ValueError(
        "Available_Indicators_Previous contains duplicate ENTITY_ID + MAP_DTE rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA summary

print()
print("=" * 110)
print("DAILY EWS INDICATORS - QA")
print("=" * 110)
print("Reference date                     :", REFERENCE_DATE_STR)
print("Previous business day              :", PREVIOUS_DATE_STR)
print("Rows                               :", "{:,}".format(len(current)))
print("Entities                           :", "{:,}".format(current["ENTITY_ID"].nunique()))
print("RRE entities                       :", "{:,}".format(int(current["RRE_FLAG"].sum())))
print("Other Retail entities              :", "{:,}".format(int(current["OTHER_RETAIL_FLAG"].sum())))
print("Entities in both portfolios        :", "{:,}".format(int((current["RRE_FLAG"].eq(1) & current["OTHER_RETAIL_FLAG"].eq(1)).sum())))
print("Rating additional materiality      : none")
print("Rating persistence (business days) :", RATING_PERSISTENCE_BUSINESS_DAYS)
print("Rating previous state date         :", rating_previous_state_date.strftime("%Y-%m-%d"))
print("Rating trigger / Day 1             :", rating_trigger_date.strftime("%Y-%m-%d"))
print("Rating confirmation / Day 5        :", rating_persistence_dates[-1].strftime("%Y-%m-%d"))
print("Connected Stage 2 threshold (EUR)  :", "{:,.0f}".format(CONNECTED_STAGE2_GEC_MATERIALITY_THRESHOLD))
print(
    "Downgrade to Stage 2 threshold (EUR):",
    "{:,.0f}".format(STAGE2_DOWNGRADE_GEC_MATERIALITY_THRESHOLD),
)

portfolio_scope = (
    current["RRE_FLAG"].eq(1)
    | current["OTHER_RETAIL_FLAG"].eq(1)
)
gec_history_coverage = (
    current.loc[
        portfolio_scope,
        "GEC_HEAD_DOWNGRADE_12M",
    ]
    .notna()
    .mean()
    * 100.0
)
print("GEC-head 12m rating coverage (%)    :", "{:.2f}".format(gec_history_coverage))
print("Rating indicator Day 1 triggers    :", "{:,}".format(
    int(rating_persistence_qa["DAY_1_TRIGGERS"].sum())
))
print("Rating indicator matured Day 5     :", "{:,}".format(
    int(rating_persistence_qa["MATURED_DAY_5"].sum())
))
print("Rating indicator final Day 5       :", "{:,}".format(
    int(rating_persistence_qa["FINAL_AFTER_PERSISTENCE"].sum())
))
print("Rating blocked - no recent worsening :", "{:,}".format(
    int(rating_persistence_qa["BLOCKED_NO_RECENT_WORSENING"].sum())
))
print("Rating indicator blocked by history :", "{:,}".format(
    int(rating_persistence_qa["BLOCKED_MISSING_DATA"].sum())
))
print("Client rating worsening vs previous BD :", "{:,}".format(
    int(current["CLIENT_RATING_DAILY_WORSENING"].sum())
))
print("Employment-loss source transitions  :", "{:,}".format(
    int(employment_source["EMPLOYMENT_LOSS_EVENT"].sum())
))
print("Employment-loss mapped entities     :", "{:,}".format(
    int(employment_entity["EMPLOYMENT_LOSS_EVENT_BASE"].sum())
))
if not employment_event_origins.empty:
    print("Employment-loss origin statuses:")
    print(employment_event_origins.to_string(index=False))
print("Stage 2 current source snapshot    :", STAGE2_CURRENT_SOURCE_DATE.strftime("%Y-%m-%d"))
print("Stage 2 previous source snapshot   :", STAGE2_PREVIOUS_SOURCE_DATE.strftime("%Y-%m-%d"))
print("DPD 15 crossing events             :", "{:,}".format(int(current["PAYMENT_DPD_15_EVENT_BASE"].sum())))
print("DPD 30 crossing events             :", "{:,}".format(int(current["PAYMENT_DPD_30_EVENT_BASE"].sum())))
print("DPD 60 crossing events             :", "{:,}".format(int(current["PAYMENT_DPD_60_EVENT_BASE"].sum())))
print("Cash crossing 94% same/+1 step     :", "{:,}".format(int(current["cash_advance_limit_cross_94_1d"].fillna(0).sum())))
print("Cash crossing 97% same/+1 step     :", "{:,}".format(int(current["cash_advance_limit_cross_97_1d"].fillna(0).sum())))
print()
print("-" * 110)
print("RRE ACTIVE CONDITIONS")
print("-" * 110)
for column in RRE_ALERT_COLUMNS:
    print("{:<80} {:>10,}".format(column, int(current[column].sum())))

print()
print("-" * 110)
print("OTHER RETAIL ACTIVE CONDITIONS")
print("-" * 110)
for column in OTHER_RETAIL_ALERT_COLUMNS:
    print("{:<80} {:>10,}".format(column, int(current[column].sum())))

print()
print("Unique RRE entities with >=1 active condition         : {:,}".format(
    int(current["RRE_ACTIVE_INDICATOR_COUNT"].gt(0).sum())
))
print("Unique Other Retail entities with >=1 active condition: {:,}".format(
    int(current["OTHER_RETAIL_ACTIVE_INDICATOR_COUNT"].gt(0).sum())
))
print("Unique Retail entities with >=1 active condition      : {:,}".format(
    int(current["RETAIL_ACTIVE_INDICATOR_COUNT"].gt(0).sum())
))

print()
print("Connected-client Stage 2 base population               : {:,}".format(
    int(current["GCC_STAGE2_CONNECTED_BASE"].sum())
))
print("RRE connected-client Stage 2 after materiality         : {:,}".format(
    int(current["RRE_GCC_STAGE2_CONNECTED"].sum())
))
print("Other Retail connected-client Stage 2 after materiality: {:,}".format(
    int(current["OTHER_GCC_STAGE2_CONNECTED"].sum())
))

print()
print("RRE LTV > 105% monitoring population                  : {:,}".format(
    int(current["RRE_LTV_ABOVE_105_MONITORING"].sum())
))


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output

output_core_columns = [
    "ENTITY_ID",
    "MAP_DTE",
    "RRE_FLAG",
    "OTHER_RETAIL_FLAG",
    "HEAD_GEC_PARENT_ID",
    "GEC_NET_EXPOSURE",
    "CLIENT_RATING",
    "CLIENT_RATING_NOTCH",
    "CLIENT_DOWNGRADE_12M",
    "GEC_HEAD_RATING",
    "GEC_HEAD_RATING_NOTCH",
    "GEC_HEAD_DOWNGRADE_12M",
    "DPD",
    "ARREARS_AMOUNT",
    "OVERDRAFT_AMOUNT",
    "EMPLOYMENT_LOSS_EVENT_BASE",
    "SALARY_3M_COMPLETE_HISTORY",
    "SALARY_YOY_DECREASE_20PCT_3M",
    "cash_advance_amount",
    "cash_advance_limit_utilization_before_transaction",
    "cash_advance_limit_utilization_after_transaction",
    "RRE_ACTIVE_INDICATOR_COUNT",
    "OTHER_RETAIL_ACTIVE_INDICATOR_COUNT",
    "RETAIL_ACTIVE_INDICATOR_COUNT",
]

output_columns = list(
    dict.fromkeys(
        output_core_columns
        + all_alert_columns
    )
)

missing_output_columns = [
    column
    for column in output_columns
    if column not in current.columns
]

cash_output_columns = [
    "cash_advance_amount",
    "cash_advance_limit_utilization_before_transaction",
    "cash_advance_limit_utilization_after_transaction",
]

print()
print("RATING OUTPUT FIELD QA")
print("----------------------")
print(
    "Client ratings populated:",
    "{:,}".format(int(current["CLIENT_RATING"].notna().sum())),
)
print(
    "GEC head ratings populated:",
    "{:,}".format(int(current["GEC_HEAD_RATING"].notna().sum())),
)

print()
print("CASH OUTPUT FIELD QA")
print("--------------------")
for column in cash_output_columns:
    available_values = int(
        pd.to_numeric(
            current[column],
            errors="coerce",
        )
        .notna()
        .sum()
    )
    print(
        "{:<55} {:>10,}".format(
            column,
            available_values,
        )
    )

if missing_output_columns:
    raise KeyError(
        "Daily EWS state output is missing required column(s): {}".format(
            ", ".join(missing_output_columns)
        )
    )

# Selecting the final columns also consolidates the dataframe and materially
# reduces serialisation / network write time.
current = current[
    output_columns
].copy()

print(
    "Persisted state columns             :",
    len(output_columns),
)

print("Available indicators before write seconds:", round(time.perf_counter() - _indicator_started, 1))
_write_started = time.perf_counter()

dataiku.Dataset(
    "Available_Indicators_Previous"
).write_with_schema(
    current
)

print("Available indicators write seconds:", round(time.perf_counter() - _write_started, 1))
print("Available indicators total seconds:", round(time.perf_counter() - _indicator_started, 1))