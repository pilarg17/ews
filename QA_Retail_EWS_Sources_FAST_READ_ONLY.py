# -*- coding: utf-8 -*-
# Retail EWS source availability check - read only.
# Counts are capped to avoid scanning entire source views.

import time
import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2

DATES = ["2026-10-21", "2026-10-22", "2026-10-23"]
CAP = 200
CONNECTION = "PRD_BDWH"

# Sources referenced by the current Retail EWS Python recipes.
DATASETS = [
    "ACCOUNT_REFERENTIAL_VW",
    "CARDS_ATM_TRANSACTIONS_VW",
    "CARD_REFERENTIAL_VW",
    "CARD_USAGE_VW",
    "CUSTOMER_VW",
    "ENTITY_CUSTOMER_VW_1",
    "ENTITY_VW",
    "INDIVIDUAL_CUSTOMER_VW",
    "INSTRUMENT_ENTITY_VW",
    "INSTRUMENT_VW",
    "OCCUPATION_VW",
    "PAYOR_VW",
    "PLY_T24_ACCOUNT_VW",
    "TELLER_TRANSACTIONS_VW",
    "X_CDS_RESULTS_VW",
    "ACCOUNT_VW_1",
]

# Physical views referenced directly in the Retail EWS SQL recipes.
PHYSICAL = [
    "BDL_CREDIT_INSIGHT_PLY.ENTITY_VW",
    "BDL_CREDIT_INSIGHT_PLY.ENTITY_CUSTOMER_VW",
    "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_VW",
    "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_ENTITY_VW",
    "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_PROTECTION_DAY_VW",
    "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_DAY_VW",
    "BDL_CREDIT_INSIGHT_PLY.PROTECTION_DAY_VW",
    "BDL_CREDIT_INSIGHT_PLY.PROTECTION_METRICS_DAY_VW",
    "DTM_RFO_PLY.LOANDEPO_VW",
    "DTM_RFO_PLY.ACCOUNT_VW",
    "DTM_RFO_PLY.MKT_FX_INPUT_VW",
    "BDWH_PLY.REF_PRTY_VW",
    "BDWH_PLY.REF_DAO_VW",
    "BDWH_PLY.REF_EMPL_VW",
    "BDWH_PLY.LNK_EUC_HCHY_FLAT_VW",
]

# Date key is a reporting snapshot, not a daily validity window.
SNAPSHOT_PHYSICAL = {
    "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_PROTECTION_DAY_VW": "MAP_DTE",
    "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_DAY_VW": "REFERENCE_DATE",
    "BDL_CREDIT_INSIGHT_PLY.PROTECTION_DAY_VW": "MAP_DTE",
    "BDL_CREDIT_INSIGHT_PLY.PROTECTION_METRICS_DAY_VW": "VALID_DATE",
    "DTM_RFO_PLY.LOANDEPO_VW": "REPORT_DATEKEY",
    "DTM_RFO_PLY.ACCOUNT_VW": "REPORT_DATEKEY",
    "DTM_RFO_PLY.MKT_FX_INPUT_VW": "REPORT_DATEKEY",
}

DATE_FIELDS = ["DEBIT_VALUE_DATE", "OPERATION_DATE", "TRANSACTION_DATE", "VALID_DATE", "MAP_DTE"]
NUMERIC_KEYS = {"REFERENCE_DATE", "REPORT_DATEKEY"}
results = []


def next_day(date_str):
    return (pd.Timestamp(date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def limited_count(executor, table, predicate):
    # Oracle stops after CAP matching rows; 200 means "at least 200".
    sql = (
        "SELECT COUNT(*) AS N FROM ("
        "SELECT 1 FROM {t} WHERE {p} AND ROWNUM <= {cap}"
        ")"
    ).format(t=table, p=predicate, cap=CAP)
    df = executor.query_to_df(sql)
    return int(df.iloc[0, 0])


def detect_date_type(dataset_schema, field):
    return next(
        (str(c.get("type", "")).lower() for c in dataset_schema
         if c["name"].upper() == field),
        ""
    )


def test_source(name, table, executor, columns, column_types=None, forced_snapshot=None):
    columns = {str(x).upper() for x in columns}
    column_types = column_types or {}
    if {"TEK_STRT_DTE", "TEK_END_DTE"}.issubset(columns):
        mode = "TEMPORAL"
        def condition(d):
            following = next_day(d)
            return (
                "TEK_STRT_DTE < DATE '{n}' AND "
                "(TEK_END_DTE IS NULL OR TEK_END_DTE >= DATE '{n}')"
            ).format(n=following)
    else:
        field = forced_snapshot or next((x for x in DATE_FIELDS if x in columns), None)
        if not field:
            field = next((x for x in ["REPORT_DATEKEY", "REFERENCE_DATE"] if x in columns), None)
        if not field:
            return {"SOURCE": name, "MODE": "NO_DATE_COLUMN", "21": None, "22": None, "23": None, "STATUS": "NOT_VERIFIED"}
        typ = column_types.get(field, "")
        is_number = field == "REPORT_DATEKEY" or (
            field == "REFERENCE_DATE" and ("int" in typ or "number" in typ or "double" in typ or "bigint" in typ)
        )
        snapshot = bool(forced_snapshot) or field in NUMERIC_KEYS
        if snapshot:
            # Only verify that at least one reporting snapshot exists up to 23/10.
            predicate = (
                "{f} <= {d}".format(f=field, d=DATES[-1].replace("-", ""))
                if is_number else
                "{f} < DATE '{d}'".format(f=field, d=next_day(DATES[-1]))
            )
            n = limited_count(executor, table, predicate)
            return {"SOURCE": name, "MODE": "REPORTING_SNAPSHOT", "21": None, "22": None, "23": None,
                    "STATUS": "SNAPSHOT_PRESENT_NOT_DAILY" if n else "NO_SNAPSHOT_BEFORE_23"}
        mode = "DAILY:" + field
        def condition(d):
            return "{f} >= DATE '{d}' AND {f} < DATE '{n}'".format(f=field, d=d, n=next_day(d))

    counts = [limited_count(executor, table, condition(d)) for d in DATES]
    a, b, c = counts
    if a and not b and not c:
        status = "MISSING_FROM_22"
    elif a and not b:
        status = "MISSING_22"
    elif a and not c:
        status = "MISSING_23"
    elif a >= CAP and b < CAP // 2 and c >= CAP:
        status = "SHARP_DROP_22"
    elif a and b and c:
        status = "PRESENT_ALL_DAYS"
    else:
        status = "REVIEW_PATTERN"
    return {"SOURCE": name, "MODE": mode, "21": a, "22": b, "23": c, "STATUS": status}


def execute_check(name, is_physical):
    started = time.time()
    try:
        if is_physical:
            executor = SQLExecutor2(connection=CONNECTION)
            table = name
            empty = executor.query_to_df("SELECT * FROM {} WHERE 1=0".format(table))
            columns = list(empty.columns)
            forced = SNAPSHOT_PHYSICAL.get(name)
            # Known physical REFERENCE_DATE reporting key is YYYYMMDD numeric.
            types = {"REFERENCE_DATE": "number"} if name.endswith("INSTRUMENT_REGULATORY_METRICS_DAY_VW") else {}
        else:
            dataset = dataiku.Dataset(name)
            loc = dataset.get_location_info().get("info", {})
            table = loc.get("table") or loc.get("tableName")
            owner = loc.get("schema") or loc.get("schemaName")
            if not table:
                raise ValueError("Not a SQL-backed source dataset")
            if owner and "." not in table:
                table = owner + "." + table
            dataset_schema = dataset.read_schema()
            columns = [col["name"] for col in dataset_schema]
            types = {col["name"].upper(): str(col.get("type", "")).lower() for col in dataset_schema}
            executor = SQLExecutor2(dataset=dataset)
            forced = None
        record = test_source(name, table, executor, columns, types, forced)
        record["PHYSICAL_TABLE"] = table
    except Exception as exc:
        record = {"SOURCE": name, "MODE": "ERROR", "21": None, "22": None, "23": None,
                  "STATUS": "QUERY_ERROR", "DETAIL": str(exc)[:220]}
    record["SECONDS"] = round(time.time() - started, 1)
    results.append(record)
    print("{:2d}/{:2d} | {:55s} | {:25s} | {:6.1f}s | {}".format(
        len(results), len(DATASETS) + len(PHYSICAL), name, record["STATUS"], record["SECONDS"],
        [record.get("21"), record.get("22"), record.get("23")]
    ), flush=True)
    if record.get("DETAIL"):
        print("    Error:", record["DETAIL"], flush=True)


print("RETAIL EWS SOURCE QA - MAX {} MATCHES PER DATE (200 = 200+)".format(CAP), flush=True)
print("This checks only references found in Retail EWS recipes.", flush=True)
for item in DATASETS:
    execute_check(item, False)
for item in PHYSICAL:
    execute_check(item, True)

qa = pd.DataFrame(results)
print("\n" + "=" * 110)
print("SOURCES WITH POSSIBLE ISSUES")
print("=" * 110)
flags = qa[qa["STATUS"].isin([
    "MISSING_FROM_22", "MISSING_22", "MISSING_23", "SHARP_DROP_22", "REVIEW_PATTERN", "NO_SNAPSHOT_BEFORE_23", "QUERY_ERROR", "NOT_VERIFIED"
])]
print(flags.to_string(index=False) if len(flags) else "No gaps detected in checked date windows.")
print("\nFULL SUMMARY")
print(qa[["SOURCE", "MODE", "21", "22", "23", "STATUS", "SECONDS"]].to_string(index=False))
print("\nREAD ONLY - no dataset modifications.")
