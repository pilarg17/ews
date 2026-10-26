# -*- coding: utf-8 -*-

import dataiku
import numpy as np
import pandas as pd
import time

from dataiku.core.sql import SQLExecutor2


# Settings for the six-month history included with each alert.

DB_CONNECTION = "PRD_BDWH"

REFERENCE_DATE_DATASET = "Implementation_Reference_Date"
BREACH_DATASET = "ews_ewi_breach_information"
OUTPUT_DATASET = "ews_ewi_historic_information"

N_HISTORY_MONTHS = 6
ENTITY_CHUNK_SIZE = 800


# Small functions for reading source data and keeping IDs, dates and values consistent.

# Run an Oracle query and log how long it takes.
def execute_query(database, query):
    start_time = time.time()

    result = SQLExecutor2(
        connection=database
    ).query_to_df(query)

    print(
        "Time to perform query: {:.1f} seconds".format(
            time.time() - start_time
        )
    )

    return result


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


def scalar_date(value):
    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        return pd.NaT

    return parsed.tz_localize(None).normalize()


def sql_string(value):
    return (
        "'"
        + str(value).replace("'", "''")
        + "'"
    )


# Split larger entity lists into smaller batches for the historical queries.
def entity_chunks(values, chunk_size):
    values = list(values)

    for start in range(
        0,
        len(values),
        chunk_size,
    ):
        yield values[
            start:start + chunk_size
        ]


def round_or_nan(value, digits=4):
    if pd.isna(value):
        return np.nan

    return round(
        float(value),
        digits,
    )


def normalize_text(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .casefold()
    )


def numeric_or_nan(value):
    return pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]


# Read the run, reference and previous dates used throughout the output.

reference_df = dataiku.Dataset(
    REFERENCE_DATE_DATASET
).get_dataframe()

required_reference_columns = [
    "RUN_DATE",
    "REFERENCE_DATE",
    "PREVIOUS_DATE",
]

missing_reference_columns = [
    column
    for column in required_reference_columns
    if column not in reference_df.columns
]

if missing_reference_columns:
    raise KeyError(
        "{} is missing required column(s): {}".format(
            REFERENCE_DATE_DATASET,
            ", ".join(missing_reference_columns),
        )
    )

if len(reference_df) != 1:
    raise ValueError(
        "{} must contain exactly one row.".format(
            REFERENCE_DATE_DATASET
        )
    )

RUN_DATE = scalar_date(
    reference_df.loc[
        reference_df.index[0],
        "RUN_DATE",
    ]
)

REFERENCE_DATE = scalar_date(
    reference_df.loc[
        reference_df.index[0],
        "REFERENCE_DATE",
    ]
)

PREVIOUS_DATE = scalar_date(
    reference_df.loc[
        reference_df.index[0],
        "PREVIOUS_DATE",
    ]
)

if (
    pd.isna(RUN_DATE)
    or pd.isna(REFERENCE_DATE)
    or pd.isna(PREVIOUS_DATE)
):
    raise ValueError(
        "RUN_DATE, REFERENCE_DATE and PREVIOUS_DATE must be valid dates."
    )

if not (
    PREVIOUS_DATE
    < REFERENCE_DATE
    < RUN_DATE
):
    raise ValueError(
        "Expected PREVIOUS_DATE < REFERENCE_DATE < RUN_DATE."
    )

RUN_DATE_STR = RUN_DATE.strftime("%Y-%m-%d")
REFERENCE_DATE_STR = REFERENCE_DATE.strftime("%Y-%m-%d")
PREVIOUS_DATE_STR = PREVIOUS_DATE.strftime("%Y-%m-%d")


# Define the columns expected by the historical EWS export.

historic_columns = [
    "BREACH_ID",
    "EWI_1",
    "VALUE_EWI_1",
    "THRESHOLD_EWI_1",
    "DEVIATION_EWI_1",
    "CHANGE_EWI_1",
    "EWI_2",
    "VALUE_EWI_2",
    "THRESHOLD_EWI_2",
    "DEVIATION_EWI_2",
    "CHANGE_EWI_2",
    "HISTORIC_DATE",
    "ALERT_DATE",
    "DAY",
    "error",
]


# Start with the breaches generated for the reference date.

breaches = dataiku.Dataset(
    BREACH_DATASET
).get_dataframe()

required_breach_columns = [
    "BREACH_ID",
    "EWI_NAME",
    "ENTITY_ID",
    "ALERT_DATE",
    "RM",
    "BUSINESS_LINE",
]

missing_breach_columns = [
    column
    for column in required_breach_columns
    if column not in breaches.columns
]

if missing_breach_columns:
    raise KeyError(
        "{} is missing required column(s): {}".format(
            BREACH_DATASET,
            ", ".join(missing_breach_columns),
        )
    )

for column in [
    "EWI_VALUE_1",
    "THRESHOLD_1",
    "EWI_VALUE_2",
    "THRESHOLD2",
]:
    if column not in breaches.columns:
        breaches[column] = np.nan


# If there are no breaches, write an empty file with the expected columns.

if breaches.empty:
    ews_ewi_historic_information_df = pd.DataFrame(
        columns=historic_columns
    )

    dataiku.Dataset(
        OUTPUT_DATASET
    ).write_with_schema(
        ews_ewi_historic_information_df
    )

    print()
    print("=" * 90)
    print("EWS HISTORICAL TRACKING QA")
    print("=" * 90)
    print("Run date             :", RUN_DATE_STR)
    print("Reference date       :", REFERENCE_DATE_STR)
    print("Current breach rows  : 0")
    print("Historic rows written: 0")

else:

    # Check that the breaches belong to the correct date and exclude remediation cases.

    exit_text = (
        breaches[["RM", "BUSINESS_LINE"]]
        .astype("string")
        .fillna("")
        .agg(" ".join, axis=1)
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    remediation_exit_rows = int(
        (
            exit_text.str.contains(
                "KYC EXIT CIC",
                regex=False,
                na=False,
            )
            | exit_text.str.contains(
                "BIL REMEDIATION EXIT",
                regex=False,
                na=False,
            )
        ).sum()
    )

    if remediation_exit_rows > 0:
        raise ValueError(
            "{} contains KYC / remediation exit rows.".format(
                BREACH_DATASET
            )
        )

    breaches["ENTITY_ID"] = normalize_id(
        breaches["ENTITY_ID"]
    )

    breaches["_ALERT_DATE"] = normalize_date(
        breaches["ALERT_DATE"]
    )

    alert_dates = (
        breaches["_ALERT_DATE"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(alert_dates) != 1:
        raise ValueError(
            "Expected exactly one ALERT_DATE in {}.".format(
                BREACH_DATASET
            )
        )

    ALERT_DATE = alert_dates.iloc[0]

    if ALERT_DATE != REFERENCE_DATE:
        raise ValueError(
            "ALERT_DATE ({}) does not match REFERENCE_DATE ({}).".format(
                ALERT_DATE.strftime("%Y-%m-%d"),
                REFERENCE_DATE_STR,
            )
        )

    ALERT_DATE_STR = ALERT_DATE.strftime(
        "%Y-%m-%d"
    )

    breached_entities = (
        breaches["ENTITY_ID"]
        .dropna()
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    if not breached_entities:
        raise ValueError(
            "No breached entities are available."
        )


    # Use the last five month-ends and the current reference date as history points.

    reference_month_start = (
        REFERENCE_DATE
        .to_period("M")
        .to_timestamp()
    )

    history_checkpoints = []

    for offset in range(
        N_HISTORY_MONTHS - 1,
        -1,
        -1,
    ):
        month_start = (
            reference_month_start
            - pd.DateOffset(
                months=offset
            )
        )

        if (
            month_start.year == REFERENCE_DATE.year
            and month_start.month == REFERENCE_DATE.month
        ):
            checkpoint = REFERENCE_DATE
        else:
            checkpoint = (
                month_start
                + pd.offsets.MonthEnd(0)
            )

        history_checkpoints.append(
            checkpoint.normalize()
        )

    checkpoint_df = pd.DataFrame({
        "HISTORIC_DATE":
            history_checkpoints,
    })


    # Retrieve the probability of default at each history point for the entities with breaches.

    date_cte_sql = "\nUNION ALL\n".join(
        [
            "SELECT DATE '{}' AS HISTORIC_DATE FROM DUAL".format(
                date.strftime("%Y-%m-%d")
            )
            for date in history_checkpoints
        ]
    )

    pd_parts = []

    for chunk_no, chunk in enumerate(
        entity_chunks(
            breached_entities,
            ENTITY_CHUNK_SIZE,
        ),
        start=1,
    ):
        entity_list_sql = ", ".join(
            sql_string(entity_id)
            for entity_id in chunk
        )

        query = """
WITH history_dates AS (
    {date_cte_sql}
)
SELECT
    TO_CHAR(e.entity_id) AS ENTITY_ID,
    d.HISTORIC_DATE,
    e.pd_crr AS PD_CRR
FROM history_dates d
INNER JOIN BDL_CREDIT_INSIGHT_PLY.ENTITY_VW e
    ON TRUNC(e.tek_strt_dte) <= d.HISTORIC_DATE
   AND d.HISTORIC_DATE
       < TRUNC(
           NVL(
               e.tek_end_dte,
               DATE '9999-12-31'
           )
       )
WHERE TO_CHAR(e.entity_id) IN ({entity_list_sql})
""".format(
            date_cte_sql=date_cte_sql,
            entity_list_sql=entity_list_sql,
        )

        print(
            "Probability of default history chunk {}: {} entities".format(
                chunk_no,
                len(chunk),
            )
        )

        pd_parts.append(
            execute_query(
                DB_CONNECTION,
                query,
            )
        )

    if pd_parts:
        pd_history = pd.concat(
            pd_parts,
            ignore_index=True,
        )
    else:
        pd_history = pd.DataFrame(
            columns=[
                "ENTITY_ID",
                "HISTORIC_DATE",
                "PD_CRR",
            ]
        )

    pd_history["ENTITY_ID"] = normalize_id(
        pd_history["ENTITY_ID"]
    )

    pd_history["HISTORIC_DATE"] = normalize_date(
        pd_history["HISTORIC_DATE"]
    )

    pd_history["PD_CRR"] = pd.to_numeric(
        pd_history["PD_CRR"],
        errors="coerce",
    )

    # Overlapping rating records are reduced to one value for each history point.
    if pd_history.duplicated(
        subset=[
            "ENTITY_ID",
            "HISTORIC_DATE",
        ],
        keep=False,
    ).any():
        pd_history = (
            pd_history
            .groupby(
                [
                    "ENTITY_ID",
                    "HISTORIC_DATE",
                ],
                as_index=False,
            )["PD_CRR"]
            .max()
        )


    # Retrieve DPD and arrears at the same dates, combining instrument data by entity.

    dpd_parts = []

    for chunk_no, chunk in enumerate(
        entity_chunks(
            breached_entities,
            ENTITY_CHUNK_SIZE,
        ),
        start=1,
    ):
        entity_list_sql = ", ".join(
            sql_string(entity_id)
            for entity_id in chunk
        )

        query = """
WITH history_dates AS (
    {date_cte_sql}
)
SELECT
    TO_CHAR(i2e.entity_id) AS ENTITY_ID,
    d.HISTORIC_DATE,

    MAX(
        CASE
            WHEN ABS(NVL(ins.dpd, 0)) >= 10000000000
                THEN 0
            ELSE NVL(ins.dpd, 0)
        END
    ) AS DPD,

    SUM(
        CASE
            WHEN ABS(NVL(ins.arrears_amount, 0)) >= 10000000000
                THEN 0
            ELSE NVL(ins.arrears_amount, 0)
        END
    ) AS ARREARS_AMOUNT

FROM history_dates d

INNER JOIN BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_ENTITY_VW i2e
    ON TRUNC(i2e.tek_strt_dte) <= d.HISTORIC_DATE
   AND d.HISTORIC_DATE
       < TRUNC(
           NVL(
               i2e.tek_end_dte,
               DATE '9999-12-31'
           )
       )
   AND i2e.entity_id IS NOT NULL
   AND TO_CHAR(i2e.entity_id) IN ({entity_list_sql})

INNER JOIN BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_VW ins
    ON ins.instrument_id = i2e.instrument_id
   AND TRUNC(ins.tek_strt_dte) <= d.HISTORIC_DATE
   AND d.HISTORIC_DATE
       < TRUNC(
           NVL(
               ins.tek_end_dte,
               DATE '9999-12-31'
           )
       )

GROUP BY
    TO_CHAR(i2e.entity_id),
    d.HISTORIC_DATE
""".format(
            date_cte_sql=date_cte_sql,
            entity_list_sql=entity_list_sql,
        )

        print(
            "Days past due / arrears history chunk {}: {} entities".format(
                chunk_no,
                len(chunk),
            )
        )

        dpd_parts.append(
            execute_query(
                DB_CONNECTION,
                query,
            )
        )

    if dpd_parts:
        dpd_history = pd.concat(
            dpd_parts,
            ignore_index=True,
        )
    else:
        dpd_history = pd.DataFrame(
            columns=[
                "ENTITY_ID",
                "HISTORIC_DATE",
                "DPD",
                "ARREARS_AMOUNT",
            ]
        )

    dpd_history["ENTITY_ID"] = normalize_id(
        dpd_history["ENTITY_ID"]
    )

    dpd_history["HISTORIC_DATE"] = normalize_date(
        dpd_history["HISTORIC_DATE"]
    )

    dpd_history["DPD"] = pd.to_numeric(
        dpd_history["DPD"],
        errors="coerce",
    )

    dpd_history["ARREARS_AMOUNT"] = pd.to_numeric(
        dpd_history["ARREARS_AMOUNT"],
        errors="coerce",
    )

    # A duplicate DPD record would make the exported history unreliable.
    if dpd_history.duplicated(
        subset=[
            "ENTITY_ID",
            "HISTORIC_DATE",
        ]
    ).any():
        raise ValueError(
            "Historical days past due / arrears contains duplicate "
            "ENTITY_ID + HISTORIC_DATE rows."
        )


    # Create one row per breach and history date, then add the available source values.

    breach_history_columns = [
        "BREACH_ID",
        "ENTITY_ID",
        "EWI_NAME",
        "EWI_VALUE_1",
        "THRESHOLD_1",
        "EWI_VALUE_2",
        "THRESHOLD2",
        "_ALERT_DATE",
    ]

    # Match every current breach to the six historical reporting dates.
    base = (
        breaches[
            breach_history_columns
        ]
        .merge(
            checkpoint_df,
            how="cross",
        )
    )

    base = base.merge(
        pd_history,
        how="left",
        on=[
            "ENTITY_ID",
            "HISTORIC_DATE",
        ],
        validate="m:1",
    )

    base = base.merge(
        dpd_history,
        how="left",
        on=[
            "ENTITY_ID",
            "HISTORIC_DATE",
        ],
        validate="m:1",
    )


    # Select which historical measure to show based on the type of alert.

    # Pick the historic value displayed for this alert type.
    def historic_value(row):
        ewi_name = normalize_text(
            row["EWI_NAME"]
        )

        breach_threshold = numeric_or_nan(
            row["THRESHOLD_1"]
        )

        if (
            "days past due" in ewi_name
            or ewi_name.startswith("rre_dpd_")
            or ewi_name.startswith("other_dpd_")
        ):
            value = row["DPD"]

            return (
                "DAYS PAST DUE",
                value,
                breach_threshold,
                (
                    ""
                    if pd.notna(value)
                    else "Historical days past due not available"
                ),
            )

        if "arrear" in ewi_name:
            value = row["ARREARS_AMOUNT"]

            return (
                "ARREARS AMOUNT",
                value,
                breach_threshold,
                (
                    ""
                    if pd.notna(value)
                    else "Historical arrears amount not available"
                ),
            )

        if (
            "rating" in ewi_name
            or "downgrade" in ewi_name
            or "client_worse" in ewi_name
            or "gcc_worse" in ewi_name
        ):
            value = (
                row["PD_CRR"] * 100.0
                if pd.notna(
                    row["PD_CRR"]
                )
                else np.nan
            )

            return (
                "PROBABILITY OF DEFAULT",
                value,
                np.nan,
                (
                    ""
                    if pd.notna(value)
                    else "Historical probability of default not available"
                ),
            )

        return (
            row["EWI_NAME"],
            np.nan,
            breach_threshold,
            "",
        )


    mapped = base.apply(
        historic_value,
        axis=1,
        result_type="expand",
    )

    mapped.columns = [
        "EWI_1",
        "VALUE_EWI_1",
        "THRESHOLD_EWI_1",
        "error",
    ]

    base = pd.concat(
        [
            base.reset_index(drop=True),
            mapped.reset_index(drop=True),
        ],
        axis=1,
    )

    base["VALUE_EWI_1"] = pd.to_numeric(
        base["VALUE_EWI_1"],
        errors="coerce",
    )

    base["THRESHOLD_EWI_1"] = pd.to_numeric(
        base["THRESHOLD_EWI_1"],
        errors="coerce",
    )

    base["DEVIATION_EWI_1"] = (
        base["VALUE_EWI_1"]
        - base["THRESHOLD_EWI_1"]
    )


    # Compare each value with the preceding history point for the same breach.

    base = (
        base
        .sort_values(
            [
                "BREACH_ID",
                "HISTORIC_DATE",
            ]
        )
        .reset_index(drop=True)
    )

    base["PREVIOUS_VALUE_EWI_1"] = (
        base
        .groupby("BREACH_ID")[
            "VALUE_EWI_1"
        ]
        .shift(1)
    )

    # When there is no prior observed value, use the current historical value.
    base["CHANGE_EWI_1"] = np.where(
        base[
            "PREVIOUS_VALUE_EWI_1"
        ].notna()
        & base[
            "VALUE_EWI_1"
        ].notna(),
        (
            base["VALUE_EWI_1"]
            - base["PREVIOUS_VALUE_EWI_1"]
        ),
        base["VALUE_EWI_1"],
    )


    # Arrange the results in the format expected by the EWS tool.

    # Only one historical measure is populated for each alert.
    base["EWI_2"] = np.nan
    base["VALUE_EWI_2"] = np.nan
    base["THRESHOLD_EWI_2"] = np.nan
    base["DEVIATION_EWI_2"] = np.nan
    base["CHANGE_EWI_2"] = np.nan

    base["ALERT_DATE"] = ALERT_DATE_STR
    base["DAY"] = RUN_DATE_STR

    ews_ewi_historic_information_df = (
        base[
            historic_columns
        ]
        .copy()
    )

    for column in [
        "VALUE_EWI_1",
        "THRESHOLD_EWI_1",
        "DEVIATION_EWI_1",
        "CHANGE_EWI_1",
        "VALUE_EWI_2",
        "THRESHOLD_EWI_2",
        "DEVIATION_EWI_2",
        "CHANGE_EWI_2",
    ]:
        ews_ewi_historic_information_df[
            column
        ] = (
            ews_ewi_historic_information_df[
                column
            ]
            .apply(
                round_or_nan
            )
        )

    ews_ewi_historic_information_df[
        "HISTORIC_DATE"
    ] = (
        pd.to_datetime(
            ews_ewi_historic_information_df[
                "HISTORIC_DATE"
            ],
            errors="coerce",
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )


    # Check the expected number of rows and make sure breach/date pairs are unique.

    # Six history records are expected for every breach in the input.
    expected_rows = (
        len(breaches)
        * N_HISTORY_MONTHS
    )

    actual_rows = len(
        ews_ewi_historic_information_df
    )

    duplicate_rows = int(
        ews_ewi_historic_information_df
        .duplicated(
            subset=[
                "BREACH_ID",
                "HISTORIC_DATE",
            ]
        )
        .sum()
    )

    breaches_with_history = int(
        ews_ewi_historic_information_df.loc[
            ews_ewi_historic_information_df[
                "VALUE_EWI_1"
            ].notna(),
            "BREACH_ID",
        ]
        .nunique()
    )

    breaches_without_numeric_history = (
        len(breaches)
        - breaches_with_history
    )

    print()
    print("=" * 90)
    print("EWS HISTORICAL TRACKING QA")
    print("=" * 90)
    print("Run date                         :", RUN_DATE_STR)
    print("Reference date                   :", REFERENCE_DATE_STR)
    print("Current breach rows              :", len(breaches))
    print("Expected historic rows           :", expected_rows)
    print("Actual historic rows             :", actual_rows)
    print("Breaches with numeric history    :", breaches_with_history)
    print("Breaches without numeric history :", breaches_without_numeric_history)
    print("Duplicate breach/date rows       :", duplicate_rows)

    if actual_rows != expected_rows:
        raise ValueError(
            "Historic output row count does not match breach rows x {}.".format(
                N_HISTORY_MONTHS
            )
        )

    if duplicate_rows > 0:
        raise ValueError(
            "Duplicate BREACH_ID + HISTORIC_DATE rows detected."
        )


    # Save the completed history for export.

    dataiku.Dataset(
        OUTPUT_DATASET
    ).write_with_schema(
        ews_ewi_historic_information_df
    )

    print()
    print(
        "{} rows written to {}.".format(
            actual_rows,
            OUTPUT_DATASET,
        )
    )