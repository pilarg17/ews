# -*- coding: utf-8 -*-
"""Corporate B01 | GEC mapping source locator | ONE Dataiku notebook cell | READ ONLY.
Only metadata and 0-row SELECT queries. No datasets or recipes are modified.
"""
import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2
from IPython.display import display

PROJECT = "VALIDATIONEWSCORPORATEPORTFOLIODAILY"
DATASETS = ["vars_entity_level", "ENTITY_DAY_VW", "ENTITY_VW", "vars_loan_level"]
VIEWS = ["BDL_CREDIT_INSIGHT_PLY.ENTITY_VW", "BDL_CREDIT_INSIGHT_PLY.ENTITY_DAY_VW"]
GEC_KEYS = ("HEAD_GEC_PARENT__ID", "HEAD_GEC_PARENT_ID")
client = dataiku.api_client()
project = client.get_project(PROJECT)
report, candidate_connections = [], []

def matched(cols, *needles):
    return [c for c in cols if any(n in str(c).upper() for n in needles)]

print("=" * 100)
print("CORPORATE B01 - FIND THE AUTHORITATIVE GEC MAPPING | READ ONLY")
print("Checks Dataiku dataset metadata and source-connection schema only; no data extraction.")
print("=" * 100)

for name in DATASETS:
    try:
        ds = dataiku.Dataset(name, project_key=PROJECT)
        colnames = [r["name"] for r in ds.read_schema()]
        settings = project.get_dataset(name).get_settings().get_raw()
        params = settings.get("params") or {}
        conn = params.get("connection", "")
        loc = ds.get_location_info()
        info = loc.get("info") or {}
        conn = conn or info.get("connectionName", "")
        table = info.get("quotedResolvedTableName") or (
            str(params.get("schema", "")) + "." + str(params.get("table", ""))
        )
        if conn and conn not in candidate_connections:
            candidate_connections.append(conn)
        gec = matched(colnames, "GEC")
        row = {
            "DATASET": name, "CONNECTION": conn or "UNKNOWN", "TABLE": table,
            "GEC_COLUMNS": ", ".join(gec) or "NONE IN DATASET",
            "GCC_COLUMNS": ", ".join(matched(colnames, "GCC")) or "-",
            "PD_COLUMNS": ", ".join(matched(colnames, "PD_CRR")) or "-",
            "ENTITY_ID": "ENTITY_ID" in [c.upper() for c in colnames],
            "VALIDITY_OR_DAILY": ", ".join(matched(colnames, "MAP_DTE", "TEK_STRT_DTE", "TEK_END_DTE")) or "-"
        }
        report.append(row)
    except Exception as exc:
        report.append({"DATASET": name, "CONNECTION": "NOT FOUND / ERROR",
                       "TABLE": "-", "GEC_COLUMNS": "-", "GCC_COLUMNS": "-",
                       "PD_COLUMNS": "-", "ENTITY_ID": False,
                       "VALIDITY_OR_DAILY": "-"})
        print("Dataset", name, "error:", str(exc)[:220])

print("\n1. REGISTERED DATASET INVENTORY")
display(pd.DataFrame(report))

# The upstream entity source may live on a DIFFERENT connection from DEV_AA_TEAMS.
# PRD_BDWH is only a fallback for source discovery if the mapped Dataiku source
# dataset does not expose a connection. Every query returns ZERO source rows.
if "PRD_BDWH" not in candidate_connections:
    candidate_connections.append("PRD_BDWH")

checks = []
for conn in candidate_connections:
    for view in VIEWS:
        try:
            ex = SQLExecutor2(connection=conn)
            df = ex.query_to_df("SELECT * FROM " + view + " WHERE 1=0")
            cols = [str(c) for c in df.columns]
            gec = matched(cols, "GEC")
            checks.append({
                "CONNECTION": conn, "VIEW": view, "STATUS": "ACCESSIBLE",
                "GEC_COLUMNS": ", ".join(gec) or "NONE",
                "GCC_COLUMNS": ", ".join(matched(cols, "GCC")) or "-",
                "PD_COLUMNS": ", ".join(matched(cols, "PD_CRR")) or "-",
                "DATE_COLUMNS": ", ".join(matched(cols, "MAP_DTE", "TEK_STRT_DTE", "TEK_END_DTE")) or "-",
                "SCOPE_COLUMN": ", ".join(matched(cols, "COUNTERPARTY_SUBTYPE")) or "-",
                "ERROR": ""
            })
        except Exception as exc:
            checks.append({
                "CONNECTION": conn, "VIEW": view, "STATUS": "ERROR",
                "GEC_COLUMNS": "-", "GCC_COLUMNS": "-", "PD_COLUMNS": "-",
                "DATE_COLUMNS": "-", "SCOPE_COLUMN": "-",
                "ERROR": str(exc).replace("\n", " ")[:240]
            })

print("\n2. ORIGINAL ENTITY VIEW CHECKS (ZERO ROWS READ)")
display(pd.DataFrame(checks))

found = [x for x in checks if x["STATUS"] == "ACCESSIBLE"
         and any(k in x["GEC_COLUMNS"].upper() for k in GEC_KEYS)]
if found:
    print("\nSOURCE FOUND: the mapping exists in the view(s) above.")
    print("Next: verify full GEC membership and PD/loan completeness before recomputing B01.")
else:
    print("\nBLOCKED: exact GEC mapping not yet confirmed from available source connections.")
    print("Ask the data owner for an accessible complete GEC-to-entity historical mapping.")
print("No dataset/recipe/Flow was modified.")
