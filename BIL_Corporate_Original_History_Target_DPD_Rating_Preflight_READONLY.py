# BIL Corporate EWS - preflight SOLO LECTURA
# Objetivo: comprobar FUENTES ORIGINALES (no _QA) para TARGET + DPD + Rating/GCC
# No ejecuta recipes, no reconstruye históricos, no crea tablas/datasets.

import re
import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2

PROJECT_KEY = "VALIDATIONEWSCORPORATEPORTFOLIODAILY"
CHECK_DATE = "2026-10-26"  # dia ya validado; se utiliza solo para una comprobacion puntual
RUN_ONE_DAY_COUNTS = True  # si tarda, poner False: solo schema + SELECT 1

# Campos que aparecen realmente en los codigos de Training/EWS_Results.
# RFO se muestra solo como fuente opcional para ratings de contratos.
SOURCES = {
    "vars_entity_level": {
        "block": "TARGET_PD_CRR + RATING/GCC",
        "required": ["ENTITY_ID", "PD_CRR", "GCC_PARENT_ID", "TEK_STRT_DTE", "TEK_END_DTE"],
        "optional": ["RATING_SCALE", "GCC_PD_CRR", "GCC_HEAD_OF_GROUP_FLAG"],
        "kind": "history",
    },
    "vars_loan_level": {
        "block": "TARGET_DEFAULT/FORBEARANCE/DIM1 + DPD cross-check",
        "required": ["ENTITY_ID", "INSTRUMENT_ID", "MAP_DTE", "DEFAULT_STATUS", "FORBEARANCE_STATUS", "DIM1"],
        "optional": ["DPD", "ARREARS_AMOUNT", "NOMINAL_AMOUNT", "GROSS_CARRYING_AMOUNT"],
        "kind": "daily_loan",
    },
    "DPD_Arrears_Daily_Loan": {
        "block": "DPD + ARREARS historical daily source",
        "required": ["ENTITY_ID", "INSTRUMENT_ID", "MAP_DTE", "DPD", "ARREARS_AMOUNT"],
        "optional": ["NOMINAL_AMOUNT", "LIMIT_AMOUNT_CTV"],
        "kind": "daily_dpd",
    },
    "RFO_PD_loan": {
        "block": "RFO - optional, no bloquear target / primera fase",
        "required": [],
        "optional": ["INSTRUMENT_ID", "CONTRACT_REFERENCE", "REPORT_DATEKEY", "PD_FINAL_12M", "RATING_CURRENT"],
        "kind": "optional",
    },
}

project = dataiku.api_client().get_project(PROJECT_KEY)
results = []
columns_by_dataset = {}

def identifier(value):
    # No se utiliza contenido de datasets para construir SQL.
    s = str(value).replace("${projectKey}", PROJECT_KEY)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", s):
        raise ValueError("Identificador SQL no valido: " + repr(s))
    return '"' + s + '"'

for ds_name, spec in SOURCES.items():
    row = {
        "DATASET": ds_name,
        "BLOQUE": spec["block"],
        "COLUMNAS": None,
        "REQUIRED_MISSING": "",
        "OPTIONAL_FOUND": "",
        "TABLA_ACCESIBLE": "NO COMPROBADA",
        "ROWS_26OCT": None,
        "ENTITIES_26OCT": None,
        "COVERAGE_FIELD_26OCT": None,
        "ERROR": "",
    }
    try:
        ds = dataiku.Dataset(ds_name, project_key=PROJECT_KEY)
        cols = {c["name"].upper() for c in ds.read_schema()}
        columns_by_dataset[ds_name] = cols
        row["COLUMNAS"] = len(cols)
        missing = [c for c in spec["required"] if c.upper() not in cols]
        row["REQUIRED_MISSING"] = ", ".join(missing)
        row["OPTIONAL_FOUND"] = ", ".join(c for c in spec["optional"] if c.upper() in cols)
        params = project.get_dataset(ds_name).get_settings().get_raw().get("params", {})
        conn = params.get("connection")
        schema = params.get("schema")
        table = params.get("table")
        if not all([conn, schema, table]):
            raise ValueError("No hay conexion, schema o tabla Snowflake en la configuracion")
        fq_table = identifier(schema) + "." + identifier(table)
        sql = SQLExecutor2(connection=conn)
        sql.query_to_df("SELECT 1 AS VISIBLE FROM " + fq_table + " LIMIT 1")
        row["TABLA_ACCESIBLE"] = "SI (SELECT 1)"

        if RUN_ONE_DAY_COUNTS and not missing and spec["kind"] != "optional":
            if spec["kind"] == "history":
                where = ("TEK_STRT_DTE <= DATE '" + CHECK_DATE + "' "
                         "AND (TEK_END_DTE >= DATE '" + CHECK_DATE + "' OR TEK_END_DTE IS NULL)")
                quality = "COUNT_IF(PD_CRR IS NOT NULL)"
            elif spec["kind"] == "daily_loan":
                where = "MAP_DTE >= DATE '" + CHECK_DATE + "' AND MAP_DTE < DATEADD(day, 1, DATE '" + CHECK_DATE + "')"
                quality = "COUNT_IF(DEFAULT_STATUS IS NOT NULL AND FORBEARANCE_STATUS IS NOT NULL AND DIM1 IS NOT NULL)"
            else:
                where = "MAP_DTE >= DATE '" + CHECK_DATE + "' AND MAP_DTE < DATEADD(day, 1, DATE '" + CHECK_DATE + "')"
                quality = "COUNT_IF(DPD IS NOT NULL)"

            check_sql = ("SELECT COUNT(*) AS N_ROWS, COUNT(DISTINCT ENTITY_ID) AS N_ENTITIES, "
                         + quality + " AS N_WITH_CORE_VALUE FROM " + fq_table + " WHERE " + where)
            d = sql.query_to_df(check_sql)
            if not d.empty:
                d.columns = [str(c).upper() for c in d.columns]
                row["ROWS_26OCT"] = int(d.iloc[0]["N_ROWS"])
                row["ENTITIES_26OCT"] = int(d.iloc[0]["N_ENTITIES"])
                row["COVERAGE_FIELD_26OCT"] = int(d.iloc[0]["N_WITH_CORE_VALUE"])
    except Exception as exc:
        row["ERROR"] = str(exc).replace("\n", " ")[:260]
        columns_by_dataset.setdefault(ds_name, set())
    results.append(row)

print("=== CORPORATE ORIGINAL HISTORY: SOURCE / TARGET PREFLIGHT ===")
print("CHECK_DATE:", CHECK_DATE)
print(pd.DataFrame(results).to_string(index=False))
print("\n=== TARGET: DEFINITION IN CURRENT EWS_RESULTS RECIPE ===")
print("PD_CRR==1 | DEFAULT_STATUS in {18,19,20} | DPD>=20 | FORBEARANCE_STATUS in {3,4,5} | DIM1 in {'S2','S3','S3S','S3R'}")
print("WARNING: DPD is both a predictor candidate and part of same-day TGT_ANY; do NOT use same-date target to judge prediction.")
print("WARNING: entity history must be matched as-of TEK_STRT_DTE/TEK_END_DTE, not MAP_DTE equality.")
print("WARNING: coverage for one day does NOT prove full historical coverage; there is no full-table download here.")
print("\nSOLO LECTURA. NO QA. NO ORIGINAL TABLES MODIFIED.")
