# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-

import time

import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2
from pandas.tseries.offsets import BDay


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Parameters
# Define the database connection and the current-day LTV output.

DB_CONNECTION = "PRD_BDWH"
OUTPUT_DATASET = "loan_RRE_ltv"


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Helpers
# Execute SQL and print the time spent retrieving source information.


def execute_query(connection, query):
    start_time = time.time()

    result = SQLExecutor2(
        connection=connection
    ).query_to_df(
        query
    )

    print(
        "Query runtime: {:.1f} seconds".format(
            time.time() - start_time
        )
    )

    return result


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Reference date
# Read the current implementation reference date and convert it to both SQL date formats.

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
    pd.to_datetime(
        reference_df["REFERENCE_DATE"],
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
        "Implementation_Reference_Date must contain one valid REFERENCE_DATE."
    )

REFERENCE_DATE = reference_dates.iloc[0]
REFERENCE_DATE_INT = int(
    REFERENCE_DATE.strftime("%Y%m%d")
)
REFERENCE_DATE_STR = REFERENCE_DATE.strftime(
    "%Y-%m-%d"
)

print(
    "Implementation reference date:",
    REFERENCE_DATE_STR,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Latest available LTV snapshot
# Select the most recent available LTV snapshot that is not after the reference date.
# Print the corresponding protection-metrics date and snapshot age.

snapshot_query = """
SELECT
    MAX(reference_date) AS ltv_snapshot_date
FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_DAY_VW
WHERE reference_date <= {reference_date}
""".format(
    reference_date=REFERENCE_DATE_INT
)

snapshot_df = execute_query(
    DB_CONNECTION,
    snapshot_query,
)

if snapshot_df.empty:
    raise ValueError(
        "No LTV snapshot date could be resolved."
    )

snapshot_df.columns = [
    str(column).upper()
    for column in snapshot_df.columns
]

snapshot_value = snapshot_df[
    "LTV_SNAPSHOT_DATE"
].iloc[0]

if pd.isna(snapshot_value):
    raise ValueError(
        "No INSTRUMENT_REGULATORY_METRICS_DAY_VW snapshot is available on or before {}.".format(
            REFERENCE_DATE_STR
        )
    )

LTV_SNAPSHOT_DATE_INT = int(
    float(snapshot_value)
)

LTV_SNAPSHOT_DATE = pd.to_datetime(
    str(LTV_SNAPSHOT_DATE_INT),
    format="%Y%m%d",
    errors="raise",
).normalize()

if LTV_SNAPSHOT_DATE > REFERENCE_DATE:
    raise ValueError(
        "LTV snapshot date cannot be later than the implementation reference date."
    )

LTV_SNAPSHOT_DATE_STR = LTV_SNAPSHOT_DATE.strftime(
    "%Y-%m-%d"
)

PROTECTION_METRICS_DATE = (
    LTV_SNAPSHOT_DATE
    if BDay().is_on_offset(LTV_SNAPSHOT_DATE)
    else LTV_SNAPSHOT_DATE - BDay(1)
)

PROTECTION_METRICS_DATE_STR = (
    PROTECTION_METRICS_DATE.strftime(
        "%Y-%m-%d"
    )
)

SNAPSHOT_AGE_DAYS = int(
    (REFERENCE_DATE - LTV_SNAPSHOT_DATE).days
)

print(
    "LTV snapshot date:",
    LTV_SNAPSHOT_DATE_STR,
)
print(
    "Protection metrics date:",
    PROTECTION_METRICS_DATE_STR,
)
print(
    "LTV snapshot age in days:",
    SNAPSHOT_AGE_DAYS,
)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### RRE LTV
# Use instrument, protection and regulatory-metrics data to allocate collateral across
# loans and calculate the loan-to-value ratio. Keep the established source filters and
# date-validity conditions.

query = f"""
WITH allocratio AS (

    SELECT
        prt.protection_id,

        CASE
            WHEN NULLIF(
                prt.net_market_value_lcy,
                1
            ) > 2
            THEN prt.net_market_value_lcy

            ELSE
                CASE
                    WHEN NVL(
                        prt.collateral_cover_amount_lcy,
                        0
                    ) <= 2
                    THEN NULL

                    ELSE prt.collateral_cover_amount_lcy
                END
        END AS collateral_value,

        CASE
            WHEN SUM(
                irg.outstanding_nominal_amount
            ) >= 11111111111
            THEN NULL

            ELSE SUM(
                irg.outstanding_nominal_amount
                + NVL(
                    irg.off_balance_sheet_amount,
                    0
                )
            )
        END AS loan_value,

        CASE
            WHEN NULLIF(
                prt.net_market_value_lcy,
                1
            ) > 2
            THEN
                prt.net_market_value_lcy
                / NULLIF(
                    SUM(
                        irg.outstanding_nominal_amount
                        + NVL(
                            irg.off_balance_sheet_amount,
                            0
                        )
                    ),
                    0
                )

            ELSE
                CASE
                    WHEN NVL(
                        prt.collateral_cover_amount_lcy,
                        0
                    ) <= 2
                    THEN NULL

                    ELSE
                        prt.collateral_cover_amount_lcy
                        / NULLIF(
                            SUM(
                                irg.outstanding_nominal_amount
                                + NVL(
                                    irg.off_balance_sheet_amount,
                                    0
                                )
                            ),
                            0
                        )
                END
        END AS allocratio

    FROM (
        SELECT *
        FROM BDL_CREDIT_INSIGHT_PLY.PROTECTION_DAY_VW
        WHERE TRUNC(map_dte) = DATE '{LTV_SNAPSHOT_DATE_STR}'
          AND protection_type = 8
    ) p

    LEFT JOIN (
        SELECT *
        FROM BDL_CREDIT_INSIGHT_PLY.PROTECTION_METRICS_DAY_VW
        WHERE TRUNC(valid_date) = DATE '{PROTECTION_METRICS_DATE_STR}'
    ) prt
        ON p.protection_id = prt.protection_id

    LEFT JOIN (
        SELECT *
        FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_PROTECTION_DAY_VW
        WHERE TRUNC(map_dte) = DATE '{LTV_SNAPSHOT_DATE_STR}'
    ) i2p
        ON prt.protection_id = i2p.protection_id

    LEFT JOIN (
        SELECT *
        FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_DAY_VW
        WHERE reference_date = {LTV_SNAPSHOT_DATE_INT}
          AND source_contract_reference NOT LIKE '211MD%'
    ) irg
        ON i2p.instrument_id = irg.source_contract_reference

    GROUP BY
        prt.protection_id,
        prt.net_market_value_lcy,
        prt.protection_execution_value_lcy,
        prt.collateral_cover_amount_lcy
),

allocated_value AS (

    SELECT
        i2p.instrument_id,
        i2p.protection_id,

        CASE
            WHEN irg.outstanding_nominal_amount >= 11111111111
            THEN NULL

            ELSE
                irg.outstanding_nominal_amount
                + NVL(
                    irg.off_balance_sheet_amount,
                    0
                )
        END AS outstanding_nominal_amount,

        allocratio.allocratio,

        CASE
            WHEN irg.outstanding_nominal_amount >= 11111111111
            THEN NULL

            ELSE
                allocratio.allocratio
                * (
                    irg.outstanding_nominal_amount
                    + NVL(
                        irg.off_balance_sheet_amount,
                        0
                    )
                )
        END AS allocatedcollvalue

    FROM (
        SELECT *
        FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_PROTECTION_DAY_VW
        WHERE TRUNC(map_dte) = DATE '{LTV_SNAPSHOT_DATE_STR}'
    ) i2p

    LEFT JOIN (
        SELECT *
        FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_DAY_VW
        WHERE reference_date = {LTV_SNAPSHOT_DATE_INT}
    ) irg
        ON i2p.instrument_id = irg.source_contract_reference

    LEFT JOIN allocratio
        ON i2p.protection_id = allocratio.protection_id

    GROUP BY
        i2p.instrument_id,
        i2p.protection_id,
        irg.off_balance_sheet_amount,
        irg.outstanding_nominal_amount,
        allocratio.allocratio
)

SELECT
    irg.source_contract_reference,
    ins.contract_id,

    CASE
        WHEN irg.outstanding_nominal_amount >= 11111111111
        THEN NULL

        ELSE
            irg.outstanding_nominal_amount
            + NVL(
                irg.off_balance_sheet_amount,
                0
            )
    END AS outstanding_nominal_amount,

    SUM(
        alco_value.allocatedcollvalue
    ) AS allocatedcollateralvalue,

    CASE
        WHEN irg.outstanding_nominal_amount >= 11111111111
        THEN NULL

        ELSE
            (
                irg.outstanding_nominal_amount
                + NVL(
                    irg.off_balance_sheet_amount,
                    0
                )
            )
            / NULLIF(
                SUM(
                    alco_value.allocatedcollvalue
                ),
                0
            )
    END AS ltv,

    ent.entity_id,
    ent.entity_name

FROM (
    SELECT *
    FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_DAY_VW
    WHERE reference_date = {LTV_SNAPSHOT_DATE_INT}
) irg

LEFT JOIN allocated_value alco_value
    ON irg.source_contract_reference = alco_value.instrument_id

LEFT JOIN (
    SELECT *
    FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_VW
    WHERE TRUNC(TEK_STRT_DTE) <= DATE '{LTV_SNAPSHOT_DATE_STR}'
      AND DATE '{LTV_SNAPSHOT_DATE_STR}' < TRUNC(
            NVL(TEK_END_DTE, DATE '9999-12-31')
          )
) ins
    ON irg.source_contract_reference = ins.instrument_id

LEFT JOIN (
    SELECT *
    FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_ENTITY_VW
    WHERE TRUNC(TEK_STRT_DTE) <= DATE '{LTV_SNAPSHOT_DATE_STR}'
      AND DATE '{LTV_SNAPSHOT_DATE_STR}' < TRUNC(
            NVL(TEK_END_DTE, DATE '9999-12-31')
          )
) i2e
    ON irg.source_contract_reference = i2e.instrument_id

LEFT JOIN (
    SELECT *
    FROM BDL_CREDIT_INSIGHT_PLY.ENTITY_VW
    WHERE TRUNC(TEK_STRT_DTE) <= DATE '{LTV_SNAPSHOT_DATE_STR}'
      AND DATE '{LTV_SNAPSHOT_DATE_STR}' < TRUNC(
            NVL(TEK_END_DTE, DATE '9999-12-31')
          )
) ent
    ON i2e.entity_id = ent.entity_id

GROUP BY
    irg.source_contract_reference,
    ins.contract_id,
    ent.entity_id,
    ent.entity_name,
    irg.outstanding_nominal_amount,
    irg.off_balance_sheet_amount
"""

loan_ltv_df = execute_query(
    DB_CONNECTION,
    query,
)

if loan_ltv_df.empty:
    raise ValueError(
        "RRE LTV query returned 0 rows for snapshot date {}.".format(
            LTV_SNAPSHOT_DATE_STR
        )
    )

loan_ltv_df.columns = [
    str(column).upper()
    for column in loan_ltv_df.columns
]

required_columns = [
    "SOURCE_CONTRACT_REFERENCE",
    "CONTRACT_ID",
    "ENTITY_ID",
    "ENTITY_NAME",
    "OUTSTANDING_NOMINAL_AMOUNT",
    "ALLOCATEDCOLLATERALVALUE",
    "LTV",
]

missing_columns = [
    column
    for column in required_columns
    if column not in loan_ltv_df.columns
]

if missing_columns:
    raise KeyError(
        "RRE LTV query is missing required column(s): {}".format(
            ", ".join(missing_columns)
        )
    )

for column in [
    "OUTSTANDING_NOMINAL_AMOUNT",
    "ALLOCATEDCOLLATERALVALUE",
    "LTV",
]:
    loan_ltv_df[column] = pd.to_numeric(
        loan_ltv_df[column],
        errors="coerce",
    )

loan_ltv_df["LTV_SNAPSHOT_DATE"] = LTV_SNAPSHOT_DATE
loan_ltv_df["REFERENCE_DATE"] = REFERENCE_DATE

loan_ltv_df = loan_ltv_df[
    [
        "SOURCE_CONTRACT_REFERENCE",
        "CONTRACT_ID",
        "ENTITY_ID",
        "ENTITY_NAME",
        "OUTSTANDING_NOMINAL_AMOUNT",
        "ALLOCATEDCOLLATERALVALUE",
        "LTV",
        "LTV_SNAPSHOT_DATE",
        "REFERENCE_DATE",
    ]
].copy()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### QA
# Check LTV and allocated-collateral coverage, record counts and unique contract-date
# keys.

duplicate_contracts = int(
    loan_ltv_df.duplicated(
        subset=[
            "SOURCE_CONTRACT_REFERENCE",
            "REFERENCE_DATE",
        ]
    ).sum()
)

print()
print("=" * 90)
print("RRE LTV QA")
print("=" * 90)
print(
    "Implementation reference date: {}".format(
        REFERENCE_DATE_STR
    )
)
print(
    "LTV snapshot date: {}".format(
        LTV_SNAPSHOT_DATE_STR
    )
)
print(
    "LTV snapshot age in days: {:,}".format(
        SNAPSHOT_AGE_DAYS
    )
)
print(
    "Rows: {:,}".format(
        len(loan_ltv_df)
    )
)
print(
    "Entities: {:,}".format(
        loan_ltv_df["ENTITY_ID"].nunique()
    )
)
print(
    "Rows with ENTITY_ID: {:,}".format(
        loan_ltv_df["ENTITY_ID"].notna().sum()
    )
)
print(
    "Rows with allocated collateral: {:,}".format(
        loan_ltv_df["ALLOCATEDCOLLATERALVALUE"].notna().sum()
    )
)
print(
    "Rows with LTV: {:,}".format(
        loan_ltv_df["LTV"].notna().sum()
    )
)
print(
    "Duplicate contract/date rows: {:,}".format(
        duplicate_contracts
    )
)

if duplicate_contracts > 0:
    raise ValueError(
        "RRE LTV output contains duplicate contract/date rows."
    )


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
### Write output
# Save the LTV results used in the current RRE population calculation.

output = dataiku.Dataset(
    OUTPUT_DATASET
)

output.write_with_schema(
    loan_ltv_df
)