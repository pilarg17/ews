# -*- coding: utf-8 -*-
# Read-only check of physical sources referenced by Retail EWS.
# Results are capped at 200 observations per date/source.

import time
import pandas as pd
from dataiku.core.sql import SQLExecutor2

CONNECTION = "PRD_BDWH"
DATES = ["2026-10-21", "2026-10-22", "2026-10-23"]
CAP = 200

TEMPORAL = [
    "BDL_CREDIT_INSIGHT_PLY.ENTITY_VW",
    "BDL_CREDIT_INSIGHT_PLY.ENTITY_CUSTOMER_VW",
    "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_VW",
    "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_ENTITY_VW",
]

# These sources use reporting snapshots, not necessarily daily records.
SNAPSHOTS = {
    "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_PROTECTION_DAY_VW": ("MAP_DTE", "DATE"),
    "BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_DAY_VW": ("REFERENCE_DATE", "NUMBER"),
    "BDL_CREDIT_INSIGHT_PLY.PROTECTION_DAY_VW": ("MAP_DTE", "DATE"),
    "BDL_CREDIT_INSIGHT_PLY.PROTECTION_METRICS_DAY_VW": ("VALID_DATE", "DATE"),
    "DTM_RFO_PLY.LOANDEPO_VW": ("REPORT_DATEKEY", "NUMBER"),
    "DTM_RFO_PLY.ACCOUNT_VW": ("REPORT_DATEKEY", "NUMBER"),
    "DTM_RFO_PLY.MKT_FX_INPUT_VW": ("REPORT_DATEKEY", "NUMBER"),
}

OTHERS = [
    "BDWH_PLY.LNK_EUC_HCHY_FLAT_VW",
    "BDWH_PLY.REF_DAO_VW",
    "BDWH_PLY.REF_EMPL_VW",
    "BDWH_PLY.REF_PRTY_VW",
]

executor = SQLExecutor2(connection=CONNECTION)
results = []


def sample_count(source, condition):
    query = (
        "SELECT COUNT(*) AS N FROM ("
        "SELECT 1 FROM {source} WHERE {condition} AND ROWNUM <= {cap}"
        ")"
    ).format(source=source, condition=condition, cap=CAP)
    frame = executor.query_to_df(query)
    return int(frame.iloc[0, 0])


def valid_on(day):
    return (
        "TRUNC(TEK_STRT_DTE) <= DATE '{day}' "
        "AND DATE '{day}' < TRUNC(NVL(TEK_END_DTE, DATE '9999-12-31'))"
    ).format(day=day)


def check(source, category, snapshot=None):
    row = {"SOURCE": source, "TYPE": category, "21/10": None,
           "22/10": None, "23/10": None, "STATUS": "NOT_CHECKED"}
    start = time.time()
    try:
        if category == "TEMPORAL":
            counts = [sample_count(source, valid_on(day)) for day in DATES]
            row.update(zip(["21/10", "22/10", "23/10"], counts))
            if counts[0] > 0 and counts[1] == 0 and counts[2] == 0:
                row["STATUS"] = "MISSING_FROM_22"
            elif 0 in counts:
                row["STATUS"] = "CHECK_MISSING_DATE"
            elif counts[0] == CAP and counts[1] < CAP / 2 and counts[2] == CAP:
                row["STATUS"] = "SHARP_DROP_22"
            else:
                row["STATUS"] = "PRESENT_ALL_DAYS"
        elif category == "SNAPSHOT":
            field, data_type = snapshot
            if data_type == "NUMBER":
                condition = "{} <= 20261023".format(field)
            else:
                condition = "{} < DATE '2026-10-24'".format(field)
            n = sample_count(source, condition)
            row["STATUS"] = "SNAPSHOT_PRESENT" if n else "NO_SNAPSHOT_UP_TO_23"
        else:
            n = sample_count(source, "1=1")
            row["STATUS"] = "TABLE_PRESENT" if n else "EMPTY_TABLE"
    except Exception as exc:
        row["STATUS"] = "QUERY_ERROR: {}".format(str(exc)[:180])
    results.append(row)
    print("{}/15 | {} | {} | {:.1f}s".format(
        len(results), source, row["STATUS"], time.time() - start
    ), flush=True)


print("RETAIL EWS - PHYSICAL SOURCE QA (READ ONLY)", flush=True)
for source in TEMPORAL:
    check(source, "TEMPORAL")
for source, config in SNAPSHOTS.items():
    check(source, "SNAPSHOT", config)
for source in OTHERS:
    check(source, "OTHER")

summary = pd.DataFrame(results)
print("\nFULL SOURCE CHECK")
print(summary.to_string(index=False))
print("\nNEEDS REVIEW")
review = summary[~summary["STATUS"].isin([
    "PRESENT_ALL_DAYS", "SNAPSHOT_PRESENT", "TABLE_PRESENT"
])]
print(review.to_string(index=False) if not review.empty else "No issues detected by this test.")
print("\nRead-only check completed. Nothing was modified.")
