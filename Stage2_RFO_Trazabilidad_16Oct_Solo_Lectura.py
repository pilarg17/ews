# -*- coding: utf-8 -*-
import re
import pandas as pd
import dataiku
from dataiku.core.sql import SQLExecutor2

# Replace the three placeholders with the ENTITY_IDs from the archived breach CSV.
TARGET_ENTITY_IDS = """
ENTITY_ID_1
ENTITY_ID_2
ENTITY_ID_3
"""
ALERT_DATE = "2026-10-16"
SOURCE_DATASET = "X_CDS_RESULTS_VW"

ids = [v.strip() for v in re.split(r"[,;\s]+", TARGET_ENTITY_IDS) if v.strip()]
if len(ids) != 3 or any(v.startswith("ENTITY_ID_") or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", v) for v in ids):
    raise ValueError("Replace ENTITY_ID_1, ENTITY_ID_2, ENTITY_ID_3 with the three IDs from the CSV.")
if len(set(ids)) != 3:
    raise ValueError("The three ENTITY_IDs must be different.")

source = dataiku.Dataset(SOURCE_DATASET)
metadata = source.read_schema()
columns = {re.sub(r"[^A-Z0-9]+", "_", x["name"].upper()).strip("_"): x for x in metadata}

def field(*aliases, optional=False):
    for candidate in aliases:
        key = re.sub(r"[^A-Z0-9]+", "_", candidate.upper()).strip("_")
        if key in columns:
            return columns[key]
    if optional:
        return None
    raise KeyError("Source field not found: " + ", ".join(aliases))

contract_meta = field("CONTRACT_REFERENCE", "INSTRUMENT_ID")
entity_meta = field("COUNTERPARTY_RISK", "ENTITY_ID")
date_meta = field("REPORT_DATEKEY", "MAP_DTE")
stage_meta = field("STAGE", "IFRS9_STAGE", "DIM1")
version_meta = field("REPORT_VERSIONID", "REPORT_VERSION_ID", optional=True)
contract_col, entity_col = contract_meta["name"], entity_meta["name"]
date_col, stage_col = date_meta["name"], stage_meta["name"]
version_col = version_meta["name"] if version_meta is not None else None

location = source.get_location_info().get("info", {})
table = location.get("table") or location.get("tableName")
schema_name = location.get("schema") or location.get("schemaName")
if not table:
    raise ValueError("The source dataset is not backed by a SQL table/view.")
if schema_name and "." not in table:
    table = schema_name + "." + table

numeric_date = (date_col.upper() == "REPORT_DATEKEY" or any(
    token in str(date_meta.get("type", "")).lower()
    for token in ["int", "long", "bigint", "decimal", "number", "double", "float"]
))
ref_date = pd.Timestamp(ALERT_DATE)
if numeric_date:
    before_date = f"{date_col} <= {ref_date:%Y%m%d}"
else:
    before_date = f"TRUNC({date_col}) <= DATE '{ALERT_DATE}'"

sql = SQLExecutor2(dataset=source)
dates = sql.query_to_df(f"""
SELECT DISTINCT {date_col} AS REPORT_DATE_VALUE
FROM {table}
WHERE {before_date}
ORDER BY {date_col} DESC
FETCH FIRST 2 ROWS ONLY
""")
dates.columns = [str(c).upper() for c in dates.columns]
if len(dates) != 2:
    raise ValueError("RFO does not return two snapshots as of the alert date.")

if numeric_date:
    values = [int(x) for x in dates["REPORT_DATE_VALUE"].tolist()]
    display_dates = [pd.to_datetime(str(x), format="%Y%m%d").strftime("%Y-%m-%d") for x in values]
    date_filter = f"{date_col} IN ({values[0]}, {values[1]})"
else:
    display_dates = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in dates["REPORT_DATE_VALUE"]]
    date_filter = (
        f"TRUNC({date_col}) IN (DATE '{display_dates[0]}', DATE '{display_dates[1]}')"
    )

print("=" * 90)
print("STAGE 2 - RFO SOURCE TRACEABILITY (READ ONLY)")
print("Alert date:", ALERT_DATE)
print("Source snapshots:", display_dates[1], "->", display_dates[0])
print("RFO entity column:", entity_col, "| contract column:", contract_col)
print("=" * 90)

id_values = ", ".join("'" + x + "'" for x in ids)
# First determine the contracts linked to the requested source entities.
contracts_df = sql.query_to_df(f"""
SELECT DISTINCT TRIM(CAST({contract_col} AS VARCHAR2(100))) AS INSTRUMENT_ID
FROM {table}
WHERE {date_filter}
  AND {contract_col} IS NOT NULL
  AND TRIM(CAST({entity_col} AS VARCHAR2(100))) IN ({id_values})
""")
contracts_df.columns = [str(c).upper() for c in contracts_df.columns]
contracts = contracts_df["INSTRUMENT_ID"].dropna().astype(str).str.strip().unique().tolist()
print("Contracts linked to the supplied IDs in these snapshots:", len(contracts))

if not contracts:
    print("NO SOURCE MATCH. The breach ENTITY_ID may not match RFO COUNTERPARTY_RISK, or the source may have changed.")
    print("Do not conclude the alert is incorrect; investigate identifier mapping or historical source availability.")
else:
    # Match the recipe's global latest-version selection for each instrument and source date.
    if len(contracts) > 900:
        raise ValueError("More than 900 contracts: split into smaller groups before querying Oracle.")
    contract_values = ", ".join("'" + x.replace("'", "''") + "'" for x in contracts)
    version_select = f", {version_col} AS REPORT_VERSIONID" if version_col else ""
    version_sort = "REPORT_VERSIONID DESC" if version_col else "INSTRUMENT_ID"
    detail_df = sql.query_to_df(f"""
WITH source_rows AS (
    SELECT TRIM(CAST({contract_col} AS VARCHAR2(100))) AS INSTRUMENT_ID,
           TRIM(CAST({entity_col} AS VARCHAR2(100))) AS ENTITY_ID,
           {date_col} AS REPORT_DATE_VALUE,
           {stage_col} AS STAGE
           {version_select}
    FROM {table}
    WHERE {date_filter}
      AND {contract_col} IS NOT NULL
      AND {entity_col} IS NOT NULL
      AND TRIM(CAST({contract_col} AS VARCHAR2(100))) IN ({contract_values})
), ranked AS (
    SELECT source_rows.*,
           ROW_NUMBER() OVER (
               PARTITION BY INSTRUMENT_ID, REPORT_DATE_VALUE
               ORDER BY {version_sort}
           ) AS RN
    FROM source_rows
)
SELECT INSTRUMENT_ID, ENTITY_ID, REPORT_DATE_VALUE, STAGE
FROM ranked WHERE RN = 1
""")
    detail_df.columns = [str(c).upper() for c in detail_df.columns]
    if numeric_date:
        detail_df["SNAPSHOT"] = pd.to_datetime(
            pd.to_numeric(detail_df["REPORT_DATE_VALUE"], errors="coerce")
            .astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    else:
        detail_df["SNAPSHOT"] = pd.to_datetime(
            detail_df["REPORT_DATE_VALUE"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    detail_df["STAGE"] = pd.to_numeric(detail_df["STAGE"], errors="coerce")
    current = detail_df.loc[
        detail_df["SNAPSHOT"].eq(display_dates[0]),
        ["INSTRUMENT_ID", "ENTITY_ID", "STAGE"]
    ].rename(columns={"STAGE": "CURRENT_STAGE"})
    previous = detail_df.loc[
        detail_df["SNAPSHOT"].eq(display_dates[1]),
        ["INSTRUMENT_ID", "STAGE"]
    ].rename(columns={"STAGE": "PREVIOUS_STAGE"})
    comparison = current.merge(previous, on="INSTRUMENT_ID", how="left", validate="1:1")
    comparison["S1_TO_S2"] = comparison["PREVIOUS_STAGE"].eq(1) & comparison["CURRENT_STAGE"].eq(2)
    comparison = comparison.loc[comparison["ENTITY_ID"].isin(ids)].copy()
    print("\nCONTRACT LEVEL COMPARISON (CURRENT SNAPSHOT ENTITY)")
    if comparison.empty:
        print("No current-snapshot instrument is assigned to these source entity IDs after version selection.")
    else:
        print(comparison.sort_values(["ENTITY_ID", "S1_TO_S2"], ascending=[True, False]).to_string(index=False))
    print("\nSUMMARY BY BREACH ENTITY_ID")
    for entity_id in ids:
        subset = comparison.loc[comparison["ENTITY_ID"].eq(entity_id)]
        transitions = subset.loc[subset["S1_TO_S2"]]
        print("ENTITY_ID:", entity_id, "| source contracts:", len(subset), "| Stage 1 -> 2:", len(transitions))
        if len(transitions):
            print(transitions[["INSTRUMENT_ID", "PREVIOUS_STAGE", "CURRENT_STAGE"]].to_string(index=False))
    print("\nNOTE: RFO CONTRACT_REFERENCE may not equal the account identifier visible in T24/ZFO.")
    print("NOTE: Snapshot dates do not establish the exact date of the stage change.")

print("\nCompleted. No source data or Dataiku datasets were modified.")
