# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration
# Set the database connection and the upper limit used to ignore invalid amounts.

DB_CONNECTION = "PRD_BDWH"

DUMMY_HIGH_VALUE_CUTOFF = 1e10


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers
# Keep identifiers, dates and SQL source names consistent.

def execute_query(database, query):

    start_time = time.time()

    result = SQLExecutor2(
        connection=database
    ).query_to_df(
        query
    )

    print(
        "Time to perform the query {:.1f} seconds".format(
            time.time() - start_time
        )
    )

    return result


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


def normalize_id(series):

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    normalized = normalized.mask(
        normalized.isin(
            [
                "",
                "nan",
                "None",
                "<NA>",
            ]
        )
    )

    return normalized


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

    if schema and "." not in table:
        table = "{}.{}".format(
            schema,
            table,
        )

    return table


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read implementation reference date
# Use the date selected for the daily run.

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


reference_dates = (
    normalize_date(
        reference_df[
            "REFERENCE_DATE"
        ]
    )
    .dropna()
    .drop_duplicates()
)


if len(reference_dates) != 1:
    raise ValueError(
        "Implementation_Reference_Date must contain "
        "exactly one valid REFERENCE_DATE."
    )


REFERENCE_DATE = (
    reference_dates.iloc[0]
)

REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)


print()
print("DAILY DPD / ARREARS - REFERENCE DATE")
print("------------------------------------")

print(
    "Reference date:",
    REFERENCE_DATE_STR,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Resolve historized Credit Insight sources
# Find the instrument tables used for DPD and arrears.

instrument_table = get_sql_source(
    "INSTRUMENT_VW"
)

instrument_entity_table = get_sql_source(
    "INSTRUMENT_ENTITY_VW"
)


print(
    "Instrument source:",
    instrument_table,
)

print(
    "Instrument-entity source:",
    instrument_entity_table,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### DPD and arrears aggregated to entity level
# Read valid instrument records for the reference date. Keep the highest DPD and add up arrears for each entity.
#
query = """
SELECT
    i2e.ENTITY_ID AS ENTITY_ID,

    DATE '{reference_date}' AS MAP_DTE,

    MAX(
        CASE
            WHEN ABS(
                NVL(
                    ins.DPD,
                    0
                )
            ) >= {dummy_cutoff}
                THEN 0

            ELSE NVL(
                ins.DPD,
                0
            )
        END
    ) AS DPD,

    SUM(
        CASE
            WHEN ABS(
                NVL(
                    ins.ARREARS_AMOUNT,
                    0
                )
            ) >= {dummy_cutoff}
                THEN 0

            ELSE NVL(
                ins.ARREARS_AMOUNT,
                0
            )
        END
    ) AS ARREARS_AMOUNT

FROM {instrument_table} ins

INNER JOIN {instrument_entity_table} i2e

    ON TRIM(
        CAST(
            ins.INSTRUMENT_ID
            AS VARCHAR2(100)
        )
    )
    =
    TRIM(
        CAST(
            i2e.INSTRUMENT_ID
            AS VARCHAR2(100)
        )
    )

WHERE
    TRUNC(
        ins.TEK_STRT_DTE
    ) <= DATE '{reference_date}'

    AND DATE '{reference_date}'
        < TRUNC(
            NVL(
                ins.TEK_END_DTE,
                DATE '9999-12-31'
            )
        )

    AND TRUNC(
        i2e.TEK_STRT_DTE
    ) <= DATE '{reference_date}'

    AND DATE '{reference_date}'
        < TRUNC(
            NVL(
                i2e.TEK_END_DTE,
                DATE '9999-12-31'
            )
        )

    AND i2e.ENTITY_ID IS NOT NULL

GROUP BY
    i2e.ENTITY_ID

""".format(
    reference_date=REFERENCE_DATE_STR,
    instrument_table=instrument_table,
    instrument_entity_table=instrument_entity_table,
    dummy_cutoff=DUMMY_HIGH_VALUE_CUTOFF,
)


dpd_arrears_daily_df = execute_query(
    DB_CONNECTION,
    query,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Validate source data
# Do not continue with an empty or incomplete source result.

if dpd_arrears_daily_df.empty:

    raise ValueError(
        "No valid DPD / arrears records were returned "
        "from INSTRUMENT_VW / INSTRUMENT_ENTITY_VW "
        "for reference date {}. "
        "Check source refresh timing before continuing.".format(
            REFERENCE_DATE_STR
        )
    )


required_columns = [
    "ENTITY_ID",
    "MAP_DTE",
    "DPD",
    "ARREARS_AMOUNT",
]


missing_columns = [
    column
    for column in required_columns
    if column not in dpd_arrears_daily_df.columns
]


if missing_columns:
    raise KeyError(
        "DPD / arrears query is missing required column(s): {}".format(
            ", ".join(
                missing_columns
            )
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Standardise output
# Make sure entity identifiers, dates and amounts have the expected formats.

dpd_arrears_daily_df[
    "ENTITY_ID"
] = normalize_id(
    dpd_arrears_daily_df[
        "ENTITY_ID"
    ]
)


dpd_arrears_daily_df[
    "MAP_DTE"
] = normalize_date(
    dpd_arrears_daily_df[
        "MAP_DTE"
    ]
)


dpd_arrears_daily_df = (
    dpd_arrears_daily_df
    .dropna(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
        ]
    )
    .copy()
)


if dpd_arrears_daily_df.empty:

    raise ValueError(
        "DPD / arrears rows were returned from Oracle, "
        "but no valid ENTITY_ID + MAP_DTE rows remain "
        "after standardisation."
    )


for col in [
    "DPD",
    "ARREARS_AMOUNT",
]:

    dpd_arrears_daily_df[col] = (
        pd.to_numeric(
            dpd_arrears_daily_df[col],
            errors="coerce",
        )
        .fillna(
            0.0
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Derived arrears flags
# Flag entities with arrears above EUR 250 and EUR 500.

dpd_arrears_daily_df[
    "arrears_above_250"
] = (
    dpd_arrears_daily_df[
        "ARREARS_AMOUNT"
    ]
    > 250
).astype(
    "int8"
)


dpd_arrears_daily_df[
    "arrears_above_500"
] = (
    dpd_arrears_daily_df[
        "ARREARS_AMOUNT"
    ]
    > 500
).astype(
    "int8"
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check coverage, dates, duplicates and the number of entities with overdue amounts.

duplicate_entity_days = int(
    dpd_arrears_daily_df
    .duplicated(
        subset=[
            "ENTITY_ID",
            "MAP_DTE",
        ]
    )
    .sum()
)


output_dates = (
    dpd_arrears_daily_df[
        "MAP_DTE"
    ]
    .dropna()
    .drop_duplicates()
)


reference_date_mismatch = int(
    (
        dpd_arrears_daily_df[
            "MAP_DTE"
        ]
        != REFERENCE_DATE
    )
    .sum()
)


print()
print(
    "DAILY DPD / ARREARS QA"
)
print(
    "-----------------------"
)

print(
    "Reference date:",
    REFERENCE_DATE_STR,
)

print(
    "Entity-day rows:",
    f"{len(dpd_arrears_daily_df):,}",
)

print(
    "Entities:",
    f"{dpd_arrears_daily_df['ENTITY_ID'].nunique():,}",
)

print(
    "Output dates:",
    [
        date.strftime(
            "%Y-%m-%d"
        )
        for date in output_dates
    ],
)

print(
    "Duplicated ENTITY_ID + MAP_DTE:",
    duplicate_entity_days,
)

print(
    "Rows outside reference date:",
    reference_date_mismatch,
)

print(
    "Entities with DPD > 0:",
    f"{int((dpd_arrears_daily_df['DPD'] > 0).sum()):,}",
)

print(
    "Entities with arrears > 0:",
    f"{int((dpd_arrears_daily_df['ARREARS_AMOUNT'] > 0).sum()):,}",
)

print(
    "Entity-days with arrears > 250:",
    f"{int(dpd_arrears_daily_df['arrears_above_250'].sum()):,}",
)

print(
    "Entity-days with arrears > 500:",
    f"{int(dpd_arrears_daily_df['arrears_above_500'].sum()):,}",
)


if duplicate_entity_days > 0:

    raise ValueError(
        "DPD_Arrears_Daily_Entity contains duplicate "
        "ENTITY_ID + MAP_DTE rows."
    )


if reference_date_mismatch > 0:

    raise ValueError(
        "DPD_Arrears_Daily_Entity contains rows outside "
        "the implementation reference date."
    )


if len(output_dates) != 1:

    raise ValueError(
        "DPD_Arrears_Daily_Entity must contain exactly "
        "one MAP_DTE."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save one daily DPD and arrears record per entity.

dataiku.Dataset(
    "DPD_Arrears_Daily_Entity"
).write_with_schema(
    dpd_arrears_daily_df
)