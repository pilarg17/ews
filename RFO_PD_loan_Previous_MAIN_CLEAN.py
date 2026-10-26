# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import time

import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Configuration
# Identify the previous-branch reference date, RFO source and output datasets.

REFERENCE_DATASET = "Implementation_Reference_Date_previous"
SOURCE_DATASET = "X_CDS_RESULTS_VW"
OUTPUT_DATASET = "RFO_PD_loan_Previous"


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers
# Parse dates, resolve physical SQL tables and identify the expected source columns.

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
    info = location.get("info", {})

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

    return dataset, table


def resolve_column(
    dataset,
    candidates,
):

    schema = {
        column["name"].upper():
            column["name"]
        for column in dataset.read_schema()
    }

    for candidate in candidates:

        if candidate.upper() in schema:
            return schema[
                candidate.upper()
            ]

    raise KeyError(
        "None of the expected columns were found in "
        "{}: {}".format(
            dataset.name,
            ", ".join(candidates),
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Reference date
# Read exactly one reference date for the previous-day calculation.

reference_df = dataiku.Dataset(
    REFERENCE_DATASET
).get_dataframe()


if len(reference_df) != 1:
    raise ValueError(
        "{} must contain exactly one row.".format(
            REFERENCE_DATASET
        )
    )


if "REFERENCE_DATE" not in reference_df.columns:
    raise KeyError(
        "{} is missing REFERENCE_DATE.".format(
            REFERENCE_DATASET
        )
    )


REFERENCE_DATE = normalize_date(
    reference_df[
        "REFERENCE_DATE"
    ]
).iloc[0]


if pd.isna(REFERENCE_DATE):
    raise ValueError(
        "REFERENCE_DATE could not be parsed."
    )


REFERENCE_DATE_STR = (
    REFERENCE_DATE.strftime(
        "%Y-%m-%d"
    )
)

REFERENCE_DATE_INT = int(
    REFERENCE_DATE.strftime(
        "%Y%m%d"
    )
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Resolve SQL source and required fields
# Select the source fields for contracts, counterparties, exposure, Stage, DPD and
# forbearance; allow the listed source-column aliases.

source_dataset, source_table = (
    get_sql_source(
        SOURCE_DATASET
    )
)


contract_col = resolve_column(
    source_dataset,
    [
        "CONTRACT_REFERENCE",
        "INSTRUMENT_ID",
    ],
)

counterparty_col = resolve_column(
    source_dataset,
    [
        "COUNTERPARTY_RISK",
        "ENTITY_ID",
    ],
)

report_date_col = resolve_column(
    source_dataset,
    [
        "REPORT_DATEKEY",
        "REFERENCE_DATE",
        "MAP_DTE",
    ],
)

version_col = resolve_column(
    source_dataset,
    [
        "REPORT_VERSIONID",
        "REPORT_VERSION_ID",
        "VERSION_ID",
    ],
)

net_exposure_col = resolve_column(
    source_dataset,
    [
        "NET_EXPOSURE",
    ],
)

stage_col = resolve_column(
    source_dataset,
    [
        "STAGE",
        "IFRS9_STAGE",
        "DIM1",
    ],
)

past_due_col = resolve_column(
    source_dataset,
    [
        "PAST_DUE_DAYS",
        "DAYS_PAST_DUE",
        "DPD",
    ],
)

performing_col = resolve_column(
    source_dataset,
    [
        "PERFORMING_STATUS",
    ],
)

gross_exposure_col = resolve_column(
    source_dataset,
    [
        "GROSS_EXPOSURE",
    ],
)

collateral_col = resolve_column(
    source_dataset,
    [
        "COLLATERAL_ASSIGNED_AMOUNT",
    ],
)

forbearance_col = resolve_column(
    source_dataset,
    [
        "IND_FORBEARANCE",
        "FORBEARANCE_IND",
        "FORBEARANCE_FLAG",
    ],
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Source date handling
# Choose the latest available RFO report snapshot on or before the reference date,
# whether the source date is numeric or a SQL date.

report_date_is_numeric = (
    report_date_col.upper()
    == "REPORT_DATEKEY"
)


if report_date_is_numeric:

    max_date_query = """
    SELECT
        MAX({report_date_col}) AS REPORT_DATE_VALUE
    FROM {source_table}
    WHERE {report_date_col} <= {reference_date}
    """.format(
        report_date_col=report_date_col,
        source_table=source_table,
        reference_date=REFERENCE_DATE_INT,
    )

else:

    max_date_query = """
    SELECT
        MAX(TRUNC({report_date_col})) AS REPORT_DATE_VALUE
    FROM {source_table}
    WHERE TRUNC({report_date_col}) <= DATE '{reference_date}'
    """.format(
        report_date_col=report_date_col,
        source_table=source_table,
        reference_date=REFERENCE_DATE_STR,
    )


latest_date_df = SQLExecutor2(
    dataset=source_dataset
).query_to_df(
    max_date_query
)


if (
    latest_date_df.empty
    or latest_date_df[
        "REPORT_DATE_VALUE"
    ].isna().all()
):
    raise ValueError(
        "No RFO snapshot is available on or before {}.".format(
            REFERENCE_DATE_STR
        )
    )


latest_value = latest_date_df[
    "REPORT_DATE_VALUE"
].iloc[0]


if report_date_is_numeric:

    latest_snapshot_int = int(
        pd.to_numeric(
            pd.Series(
                [latest_value]
            ),
            errors="raise",
        ).iloc[0]
    )

    latest_snapshot = pd.to_datetime(
        str(latest_snapshot_int),
        format="%Y%m%d",
        errors="raise",
    ).normalize()

    snapshot_filter = (
        "{column} = {value}".format(
            column=report_date_col,
            value=latest_snapshot_int,
        )
    )

else:

    latest_snapshot = pd.Timestamp(
        latest_value
    ).normalize()

    snapshot_filter = (
        "TRUNC({column}) = DATE '{value}'".format(
            column=report_date_col,
            value=latest_snapshot.strftime(
                "%Y-%m-%d"
            ),
        )
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Read only the latest RFO snapshot and only required columns
# Read the selected snapshot, retain the latest version per contract, and limit the
# result to the fields used downstream.

query = """
SELECT
    INSTRUMENT_ID,
    COUNTERPARTY_RISK,
    RFO_SOURCE_DATE,
    NET_EXPOSURE,
    STAGE,
    PAST_DUE_DAYS,
    PERFORMING_STATUS,
    GROSS_EXPOSURE,
    COLLATERAL_ASSIGNED_AMOUNT,
    IND_FORBEARANCE

FROM (

    SELECT
        TRIM(TO_CHAR(x.{contract_col})) AS INSTRUMENT_ID,
        TRIM(TO_CHAR(x.{counterparty_col})) AS COUNTERPARTY_RISK,
        x.{report_date_col} AS RFO_SOURCE_DATE,
        x.{net_exposure_col} AS NET_EXPOSURE,
        x.{stage_col} AS STAGE,
        x.{past_due_col} AS PAST_DUE_DAYS,
        x.{performing_col} AS PERFORMING_STATUS,
        x.{gross_exposure_col} AS GROSS_EXPOSURE,
        x.{collateral_col} AS COLLATERAL_ASSIGNED_AMOUNT,
        x.{forbearance_col} AS IND_FORBEARANCE,

        ROW_NUMBER() OVER (
            PARTITION BY x.{contract_col}
            ORDER BY x.{version_col} DESC
        ) AS RN

    FROM {source_table} x

    WHERE {snapshot_filter}
      AND x.{contract_col} IS NOT NULL

)

WHERE RN = 1
""".format(
    contract_col=contract_col,
    counterparty_col=counterparty_col,
    report_date_col=report_date_col,
    net_exposure_col=net_exposure_col,
    stage_col=stage_col,
    past_due_col=past_due_col,
    performing_col=performing_col,
    gross_exposure_col=gross_exposure_col,
    collateral_col=collateral_col,
    forbearance_col=forbearance_col,
    version_col=version_col,
    source_table=source_table,
    snapshot_filter=snapshot_filter,
)


start_time = time.time()


rfo = SQLExecutor2(
    dataset=source_dataset
).query_to_df(
    query
)


elapsed = (
    time.time()
    - start_time
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Reject an empty extract or duplicate instrument keys, then print population, snapshot
# date and runtime.

if rfo.empty:
    raise ValueError(
        "RFO_PD_loan_Previous returned 0 rows."
    )


duplicate_instruments = int(
    rfo[
        "INSTRUMENT_ID"
    ]
    .duplicated()
    .sum()
)


if duplicate_instruments > 0:
    raise ValueError(
        "RFO_PD_loan_Previous contains duplicate INSTRUMENT_ID rows."
    )


print()
print("=" * 90)
print("RFO_PD_LOAN PREVIOUS - QA")
print("=" * 90)

print(
    "EWS reference date :",
    REFERENCE_DATE_STR,
)

print(
    "RFO snapshot       :",
    latest_snapshot.strftime(
        "%Y-%m-%d"
    ),
)

print(
    "Rows               :",
    "{:,}".format(
        len(rfo)
    ),
)

print(
    "Instruments        :",
    "{:,}".format(
        rfo[
            "INSTRUMENT_ID"
        ].nunique(
            dropna=True
        )
    ),
)

print(
    "Entities           :",
    "{:,}".format(
        rfo[
            "COUNTERPARTY_RISK"
        ].nunique(
            dropna=True
        )
    ),
)

print(
    "Duplicate instruments:",
    "{:,}".format(
        duplicate_instruments
    ),
)

print(
    "SQL runtime (sec)  :",
    "{:.1f}".format(
        elapsed
    ),
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the previous-branch RFO loan extract.

rfo = rfo.copy()

dataiku.Dataset(
    OUTPUT_DATASET
).write_with_schema(
    rfo
)