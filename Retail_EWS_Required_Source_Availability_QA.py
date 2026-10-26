# -*- coding: utf-8 -*-
# Read-only inventory of external sources needed by the Retail EWS flow.
# Run in a notebook in the Retail EWS Dataiku project (not Corporate).

import re
import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2

DATES = ["2026-10-21", "2026-10-22", "2026-10-23"]
FINAL_OUTPUTS = {
    "ews_ewi_breach_information",
    "ews_ewi_historic_information",
    "ews_entity_static_data",
    "ews_exposure_static_data",
}
DATE_COLUMNS = ["OPERATION_DATE", "TRANSACTION_DATE", "VALID_DATE", "REFERENCE_DATE", "MAP_DTE"]

project = dataiku.api_client().get_default_project()
project_key = dataiku.default_project_key()


def local_ref(ref):
    if ref.startswith(project_key + "."):
        return ref[len(project_key) + 1:]
    return ref


recipes = []
for recipe_item in project.list_recipes():
    try:
        settings = project.get_recipe(recipe_item.name).get_settings()
        code = settings.str_payload or ""
        recipes.append({
            "name": recipe_item.name,
            "inputs": {local_ref(x) for x in settings.get_flat_input_refs()},
            "outputs": {local_ref(x) for x in settings.get_flat_output_refs()},
            "code": code or "",
        })
    except Exception as e:
        print("WARNING: could not inspect recipe {}: {}".format(recipe_item.name, e))

# Find the upstream recipes that actually feed the final EWS outputs.
required_outputs = set(FINAL_OUTPUTS)
selected = []
changed = True
while changed:
    changed = False
    for recipe in recipes:
        if recipe not in selected and recipe["outputs"] & required_outputs:
            selected.append(recipe)
            required_outputs.update(recipe["inputs"])
            changed = True

if not selected:
    raise ValueError("No recipes found feeding the final EWS outputs. Check the project and dataset names.")

all_outputs = set().union(*(x["outputs"] for x in selected))
source_names = set().union(*(x["inputs"] for x in selected)) - all_outputs

# Add dataset references accessed in Python without being declared as Flow inputs.
for recipe in selected:
    source_names.update(
        x for x in re.findall(r'dataiku\.Dataset\(\s*["\']([\w.]+)["\']', recipe["code"])
        if x not in all_outputs
    )
    # Source names passed as literal arguments to get_sql_source(...).
    source_names.update(
        x for x in re.findall(r'get_sql_source\(\s*["\']([\w.]+)["\']', recipe["code"])
        if x not in all_outputs
    )

# Collect physical Oracle views referenced directly within recipe SQL.
# These are dependencies even when there is no separate Dataiku source dataset.
direct_sources = {}
for recipe in selected:
    code = recipe["code"]
    connection_match = re.search(r'\bDB_CONNECTION\s*=\s*["\']([^"\']+)', code)
    connection = connection_match.group(1) if connection_match else None
    for physical in re.findall(r'\b(?:FROM|JOIN)\s+((?:BDL_|BDWH_|PRD_)[A-Za-z0-9_]+\.[A-Za-z0-9_]+)', code, re.I):
        direct_sources.setdefault(physical.upper(), set()).add(connection)

records = []
registered_tables = set()


def check_counts(name, kind, table, executor, columns, types):
    names = {x.upper() for x in columns}
    if {"TEK_STRT_DTE", "TEK_END_DTE"}.issubset(names):
        mode = "TEMPORAL_VALIDITY"
        conditions = [
            "TRUNC(TEK_STRT_DTE) <= DATE '{d}' AND DATE '{d}' < "
            "TRUNC(NVL(TEK_END_DTE, DATE '9999-12-31'))".format(d=d)
            for d in DATES
        ]
    else:
        field = next((x for x in DATE_COLUMNS if x in names), None)
        if field is None:
            records.append({"SOURCE": name, "KIND": kind, "MODE": "OTHER / NO DATE FIELD", "21/10": None, "22/10": None, "23/10": None, "STATUS": "NEEDS_SOURCE_SPECIFIC_CHECK"})
            return
        dtype = (types.get(field) or "").lower()
        if field == "REFERENCE_DATE" and any(x in dtype for x in ("int", "double", "decimal", "number", "bigint", "float")):
            conditions = ["{} = {}".format(field, d.replace("-", "")) for d in DATES]
        elif dtype and any(x in dtype for x in ("date", "timestamp")):
            conditions = ["TRUNC({}) = DATE '{}'".format(field, d) for d in DATES]
        else:
            records.append({"SOURCE": name, "KIND": kind, "MODE": "DATE TYPE TO VERIFY: " + field, "21/10": None, "22/10": None, "23/10": None, "STATUS": "NEEDS_SOURCE_SPECIFIC_CHECK"})
            return
        mode = "DAILY_DATE: " + field
    sums = ", ".join(
        "SUM(CASE WHEN {} THEN 1 ELSE 0 END) AS N{}".format(cond, i)
        for i, cond in enumerate(conditions, 1)
    )
    query = "SELECT {} FROM {}".format(sums, table)
    try:
        df = executor.query_to_df(query)
        df.columns = df.columns.str.upper()
        counts = [int(pd.to_numeric(df.iloc[0]["N{}".format(i)], errors="coerce") or 0)
                  if pd.notna(df.iloc[0]["N{}".format(i)]) else 0
                  for i in range(1, 4)]
        a, b, c = counts
        status = ("DROP_FROM_22" if a > 0 and b == 0 and c == 0
                  else "AVAILABLE_ALL" if a > 0 and b > 0 and c > 0
                  else "CHECK_PATTERN")
        records.append({"SOURCE": name, "KIND": kind, "MODE": mode, "21/10": a, "22/10": b, "23/10": c, "STATUS": status})
    except Exception as e:
        records.append({"SOURCE": name, "KIND": kind, "MODE": mode, "21/10": None, "22/10": None, "23/10": None, "STATUS": "QUERY_ERROR: " + str(e)[:150]})


for name in sorted(source_names):
    try:
        ds = dataiku.Dataset(name)
        info = ds.get_location_info().get("info", {})
        table = info.get("table") or info.get("tableName")
        schema = info.get("schema") or info.get("schemaName")
        if not table:
            records.append({"SOURCE": name, "KIND": "DATASET", "MODE": "NOT A SQL TABLE", "21/10": None, "22/10": None, "23/10": None, "STATUS": "NOT_CHECKED"})
            continue
        if schema and "." not in table:
            table = "{}.{}".format(schema, table)
        registered_tables.add(table.upper())
        column_schema = ds.read_schema()
        columns = [x["name"].upper() for x in column_schema]
        types = {x["name"].upper(): str(x.get("type", "")) for x in column_schema}
        check_counts(name, "FLOW_SOURCE", table, SQLExecutor2(dataset=ds), columns, types)
    except Exception as e:
        records.append({"SOURCE": name, "KIND": "DATASET", "MODE": "ERROR", "21/10": None, "22/10": None, "23/10": None, "STATUS": "SOURCE_ERROR: " + str(e)[:150]})

# Report direct SQL dependencies not covered by a registered input dataset.
for physical, conns in sorted(direct_sources.items()):
    if physical in registered_tables:
        continue
    connections = sorted(x for x in conns if x)
    if len(connections) != 1:
        records.append({"SOURCE": physical, "KIND": "DIRECT_SQL", "MODE": "CONNECTION TO CONFIRM", "21/10": None, "22/10": None, "23/10": None, "STATUS": "NOT_CHECKED"})
        continue
    try:
        executor = SQLExecutor2(connection=connections[0])
        sample = executor.query_to_df("SELECT * FROM {} WHERE 1=0".format(physical))
        cols = [str(c).upper() for c in sample.columns]
        if {"TEK_STRT_DTE", "TEK_END_DTE"}.issubset(cols):
            check_counts(physical, "DIRECT_SQL", physical, executor, cols, {})
        else:
            # For Oracle views, read date column types from Oracle metadata.
            owner, view_name = physical.split(".", 1)
            metadata = executor.query_to_df(
                "SELECT COLUMN_NAME, DATA_TYPE FROM ALL_TAB_COLUMNS "
                "WHERE OWNER = '{}' AND TABLE_NAME = '{}'".format(owner, view_name)
            )
            metadata.columns = metadata.columns.str.upper()
            types = dict(zip(metadata["COLUMN_NAME"].str.upper(), metadata["DATA_TYPE"].str.lower()))
            # Oracle DATE/TIMESTAMP columns can be compared by TRUNC.
            for col, dtype in list(types.items()):
                if dtype == "number":
                    types[col] = "number"
                elif "timestamp" in dtype or "date" in dtype:
                    types[col] = "date"
            check_counts(physical, "DIRECT_SQL", physical, executor, cols, types)
    except Exception as e:
        records.append({"SOURCE": physical, "KIND": "DIRECT_SQL", "MODE": "ERROR", "21/10": None, "22/10": None, "23/10": None, "STATUS": "QUERY_ERROR: " + str(e)[:150]})

qa_df = pd.DataFrame(records)
print("\nRETAIL EWS - REQUIRED SOURCE AVAILABILITY")
print("Recipes feeding exports:", len(selected))
print("Sources discovered:", len(qa_df))
print(qa_df.to_string(index=False))
print("\nPOSSIBLE DATA LOSS FROM 22/10")
print(qa_df.loc[qa_df["STATUS"] == "DROP_FROM_22"].to_string(index=False))
print("\nNOT FULLY CHECKED / ERRORS")
print(qa_df.loc[qa_df["STATUS"].str.contains("CHECK|ERROR|NOT_CHECKED", regex=True)].to_string(index=False))
print("\nREAD ONLY - NOTHING WRITTEN")
