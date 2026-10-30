# -*- coding: utf-8 -*-
"""Dataiku DSS 13 - Corporate EWS project inventory, READ ONLY.

Run this code in a Python notebook of VALIDATIONEWSCORPORATEPORTFOLIODAILY.
It does not build recipes, change settings, read customer records, or write datasets.
It writes only a local ZIP into the notebook's working directory.
"""

import dataiku
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from IPython.display import FileLink, display
from dataiku.core.sql import SQLExecutor2

PROJECT_KEY = "VALIDATIONEWSCORPORATEPORTFOLIODAILY"
OUTPUT_NAME = "corporate_ews_auditoria_solo_lectura.zip"
CRITICAL_DATASETS = [
    "DPD_Arrears_Daily_Loan",
    "DPD_Arrears_Daily_Entity",
    "vars_loan_level",
    "vars_entity_level",
    "Haircut_Loan",
    "loan_RRE_ltv",
    "vars_entity_level_merged",
    "vars_loan_level_merged",
    "EWS_training",
    "EWS_validation",
]

client = dataiku.api_client()
project = client.get_project(PROJECT_KEY)


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def sql_table(name):
    ds = dataiku.Dataset(name)
    info = (ds.get_location_info().get("info") or {})
    table = info.get("table") or info.get("tableName")
    schema = info.get("schema") or info.get("schemaName")
    if not table:
        return ds, None
    raw = table if not schema or "." in table else schema + "." + table
    quoted = ".".join('"' + s.strip('"').replace('"', '""') + '"' for s in raw.split("."))
    return ds, quoted


manifest = {
    "project": PROJECT_KEY,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "read_only": True,
    "recipes": [],
    "datasets": [],
    "physical_table_checks": [],
    "recent_jobs": [],
    "errors": [],
    "recent_failure_excerpts": [],
}
payloads = {}

print("1/4 Listing recipes and capturing their source code...", flush=True)
for item in project.list_recipes():
    name = item.name
    try:
        settings = project.get_recipe(name).get_settings()
        recipe_type = settings.type
        code = settings.str_payload or ""
        extension = ".py" if recipe_type == "python" else (
            ".sql" if "sql" in recipe_type.lower() else ".txt"
        )
        filename = "recipes/" + safe_filename(name) + extension
        payloads[filename] = code
        refs_in = list(settings.get_flat_input_refs() or [])
        refs_out = list(settings.get_flat_output_refs() or [])
        manifest["recipes"].append({
            "name": name,
            "type": recipe_type,
            "inputs": refs_in,
            "outputs": refs_out,
            "source_file_in_zip": filename,
            "lines": len(code.splitlines()),
            "risk_markers": {
                "full_dataframe_reads": len(re.findall(r"\.get_dataframe\s*\(", code)),
                "writes_from_python": len(re.findall(r"\.write_with_schema\s*\(", code)),
                "pandas_explode": len(re.findall(r"\.explode\s*\(", code)),
                "pandas_apply": len(re.findall(r"\.apply\s*\(", code)),
                "pd_merge_asof": "merge_asof" in code,
                "sampling_switch": "SAMPLE_TEST" in code,
                "partitioned_by_date": "MAP_DTE" in code or "REFERENCE_DATE" in code,
            },
        })
    except Exception as exc:
        manifest["errors"].append("Recipe {}: {}".format(name, str(exc)[:350]))

print("2/4 Reading dataset schemas (not the data)...", flush=True)
for item in project.list_datasets():
    name = item.name
    try:
        schema = item.schema or {}
        columns = schema.get("columns", []) if isinstance(schema, dict) else []
        manifest["datasets"].append({
            "name": name,
            "type": item.type,
            "connection": item.connection,
            "columns": [{"name": c.get("name"), "type": c.get("type")} for c in columns],
        })
    except Exception as exc:
        manifest["errors"].append("Dataset {}: {}".format(name, str(exc)[:350]))

print("3/4 Checking whether essential SQL tables physically exist...", flush=True)
known = {x["name"] for x in manifest["datasets"]}
for name in CRITICAL_DATASETS:
    status = {"name": name, "status": "NOT_IN_PROJECT"}
    if name in known:
        try:
            ds, table = sql_table(name)
            if table is None:
                status = {"name": name, "status": "NOT_SQL_BACKED"}
            else:
                SQLExecutor2(dataset=ds).query_to_df(
                    "SELECT 1 AS CHECK_OK FROM {} WHERE 1=0".format(table)
                )
                status = {"name": name, "status": "TABLE_ACCESSIBLE", "table": table}
        except Exception as exc:
            status = {"name": name, "status": "CHECK_FAILED", "error": str(exc)[-600:]}
    manifest["physical_table_checks"].append(status)
    print("  {}: {}".format(name, status["status"]), flush=True)

print("4/4 Reading recent job statuses (not running jobs)...", flush=True)
try:
    jobs = project.list_jobs()
    jobs.sort(key=lambda x: (x.get("def") or {}).get("initiationTimestamp") or 0, reverse=True)
    for job in jobs[:40]:
        definition = job.get("def") or {}
        manifest["recent_jobs"].append({
            "id": definition.get("id"),
            "status": job.get("state"),
            "stable": job.get("stableState"),
            "initiation_timestamp_ms": definition.get("initiationTimestamp"),
        })
except Exception as exc:
    manifest["errors"].append("Jobs: {}".format(str(exc)[:350]))

# Only a few recent FAILED jobs; retrieve exception lines, not entire logs.
print("Collecting concise error excerpts for the latest failed jobs...", flush=True)
failures = [
    j for j in manifest["recent_jobs"]
    if (j.get("status") or "").upper() in ("FAILED", "ERROR", "ABORTED")
]
for j in failures[:4]:
    try:
        raw_log = project.get_job(j["id"]).get_log()
        suspicious = re.compile(
            r"(?i)(ERROR|EXCEPTION|Caused by:|OUT OF MEMORY|\bOOM\b|"
            r"Kubernetes|K8S|POD|EVICTED|FAILED|RESOURCE EXHAUSTED|KILLED)"
        )
        excerpts = []
        for line in raw_log.splitlines():
            if suspicious.search(line) and not re.search(
                r"(?i)(JWT|access.token|refresh.token|bearer|password|api.key)", line
            ):
                excerpts.append(line[:500])
        manifest["recent_failure_excerpts"].append({
            "job_id": j["id"],
            "excerpt": excerpts[-35:],
        })
    except Exception as exc:
        manifest["errors"].append("Job log {}: {}".format(j["id"], str(exc)[:250]))

summary = [
    "CORPORATE EWS - READ-ONLY INVENTORY",
    "Project: {}".format(PROJECT_KEY),
    "Recipes: {}".format(len(manifest["recipes"])),
    "Datasets: {}".format(len(manifest["datasets"])),
    "",
    "ESSENTIAL PHYSICAL TABLES",
]
for item in manifest["physical_table_checks"]:
    summary.append("{}: {}".format(item["name"], item["status"]))
summary.extend(["", "RECENT JOB STATUSES"])
for item in manifest["recent_jobs"][:15]:
    summary.append("{}: {}".format(item["id"], item["status"]))
summary.extend(["", "RECENT FAILURE EXCERPTS"] )
for item in manifest["recent_failure_excerpts"]:
    summary.append("JOB: " + str(item["job_id"]))
    summary.extend(item["excerpt"][-8:])
summary.extend(["", "ERRORS", *manifest["errors"]])

path = os.path.join(os.getcwd(), OUTPUT_NAME)
with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    z.writestr("summary.txt", "\n".join(summary))
    for filename, payload in payloads.items():
        z.writestr(filename, payload)

print("\n".join(summary), flush=True)
print("\nLOCAL ZIP:", path, flush=True)
print("ZIP SIZE MB:", round(os.path.getsize(path) / 1e6, 2), flush=True)
print("Review recipe source code for any company secrets before sharing.", flush=True)
try:
    display(FileLink(path))
except Exception:
    print("If the link is unavailable, download the file through the notebook's file browser.")
