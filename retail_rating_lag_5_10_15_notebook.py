# BIL Retail EWS | Simulación read-only del lag de rating (5 / 10 / 15 días hábiles)
# Copiar TODO este archivo en UNA celda de notebook de Dataiku (proyecto Retail).
import ast
import re
import time
from concurrent.futures import ThreadPoolExecutor

import dataiku
import numpy as np
import pandas as pd
from dataiku.core.sql import SQLExecutor2
from IPython.display import display

# ------------------------- CONFIGURACIÓN -------------------------
ALERT_DATE = pd.Timestamp("2026-11-06")
INITIAL_TRIGGER_DATE = pd.Timestamp("2026-11-02")
PREVIOUS_STATE_DATE = pd.Timestamp("2026-10-30")
COHORT_DATASET = "EWS_Daily_Alerts"  # debe conservar las alertas originales del 06/11
PRODUCER_RECIPE = "compute_Daily_EWS_Indicators"
LAGS = (5, 10, 15)
RATING_PREFIXES = ("RRE_CLIENT_", "OTHER_CLIENT_", "RRE_GEC_HEAD_", "OTHER_GEC_HEAD_")

# ------------------------- LÓGICA ORIGINAL ------------------------
# No hacemos copy/paste manual de umbrales. Extraemos las funciones y constantes
# de la receta actual, sin ejecutarla y sin modificar el Flow.
project = dataiku.api_client().get_default_project()
source = project.get_recipe(PRODUCER_RECIPE).get_settings().get_code()
if not isinstance(source, str):
    raise RuntimeError(f"No se ha podido leer el código de {PRODUCER_RECIPE}.")

wanted_functions = {
    "normalize_id", "normalize_date", "flag", "get_sql_source",
    "map_pd_to_master_scale", "load_rating_history", "rating_snapshot",
    "build_rating_persistence_flags",
}
wanted_constants = {
    "MASTER_SCALE", "MASTER_SCALE_BOUNDS", "MASTER_SCALE_LABELS",
    "MASTER_SCALE_NOTCH_MAP", "RATING_HISTORY_QUERY_CHUNK_SIZE",
    "RATING_HISTORY_PARALLEL_WORKERS", "RATING_PERSISTENCE_BUSINESS_DAYS",
    "CLIENT_WORSE_THAN_BBB_MINUS", "CLIENT_WORSE_THAN_BB_MINUS",
    "CLIENT_WORSE_THAN_B_PLUS", "CLIENT_WORSE_THAN_B",
    "CLIENT_WORSE_THAN_B_MINUS", "GEC_HEAD_WORSE_THAN_B",
    "GEC_HEAD_WORSE_THAN_B_MINUS",
}
parsed = ast.parse(source)
nodes = []
for node in parsed.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted_functions:
        nodes.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in wanted_constants for t in node.targets
    ):
        nodes.append(node)
loaded_functions = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
loaded_constants = {
    t.id for n in nodes if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)
}
if wanted_functions - loaded_functions or wanted_constants - loaded_constants:
    raise RuntimeError(
        "El código actual de Retail no coincide con el recuperado. "
        f"Funciones ausentes: {sorted(wanted_functions-loaded_functions)}. "
        f"Constantes ausentes: {sorted(wanted_constants-loaded_constants)}. "
        "Se detiene para no inventar la metodología."
    )
ns = {
    "dataiku": dataiku, "pd": pd, "np": np, "re": re,
    "time": time, "ThreadPoolExecutor": ThreadPoolExecutor,
    "SQLExecutor2": SQLExecutor2,
}
module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
exec(compile(module, f"<source:{PRODUCER_RECIPE}>", "exec"), ns)
if int(ns["RATING_PERSISTENCE_BUSINESS_DAYS"]) != 5:
    raise RuntimeError("La receta consultada no implementa 5 días: revisar la versión antes de simular.")

# ------------------------- COHORTE REAL 06/11 ---------------------
alerts = dataiku.Dataset(COHORT_DATASET).get_dataframe()
required = {
    "ALERT_DATE", "ENTITY_ID", "HEAD_GEC_PARENT_ID", "GEC_NET_EXPOSURE",
    "RRE_FLAG", "OTHER_RETAIL_FLAG", "TRIGGERED_INDICATOR_CODES",
}
missing = sorted(required - set(alerts.columns))
if missing:
    raise KeyError(f"Faltan columnas en {COHORT_DATASET}: {missing}.")
alerts["ALERT_DATE"] = ns["normalize_date"](alerts["ALERT_DATE"])
available_dates = sorted(alerts["ALERT_DATE"].dropna().dt.strftime("%Y-%m-%d").unique())
alerts = alerts.loc[alerts["ALERT_DATE"].eq(ALERT_DATE)].copy()
if alerts.empty:
    raise RuntimeError(
        f"{COHORT_DATASET} no conserva la cohorte del 06/11/2026. "
        f"Fechas encontradas (últimas 12): {available_dates[-12:]}. "
        "Restaurar una copia histórica del dataset de alertas o cambiar COHORT_DATASET; "
        "no es correcto inferir las 67 alertas a partir del output actual."
    )

def rating_codes(value):
    if pd.isna(value):
        return []
    return sorted({
        c.strip() for c in str(value).split("|")
        if c.strip().startswith(RATING_PREFIXES)
    })

alerts["ORIGINAL_RATING_CODES"] = alerts["TRIGGERED_INDICATOR_CODES"].map(rating_codes)
cohort = alerts.loc[alerts["ORIGINAL_RATING_CODES"].map(bool)].copy()
if cohort.empty:
    raise RuntimeError("En 06/11 no hay indicadores del bloque de downgrade de rating en este dataset.")
cohort["ENTITY_ID"] = ns["normalize_id"](cohort["ENTITY_ID"])
cohort["HEAD_GEC_PARENT_ID"] = ns["normalize_id"](cohort["HEAD_GEC_PARENT_ID"])
if cohort["ENTITY_ID"].isna().any() or cohort["ENTITY_ID"].duplicated().any():
    raise RuntimeError("Hay ENTITY_ID nulos/duplicados en la cohorte: revisar datos antes de contar.")
for c in ("RRE_FLAG", "OTHER_RETAIL_FLAG"):
    cohort[c] = pd.to_numeric(cohort[c], errors="coerce").fillna(0).astype("int8")
cohort["GEC_NET_EXPOSURE"] = pd.to_numeric(cohort["GEC_NET_EXPOSURE"], errors="coerce")
if not (cohort["RRE_FLAG"].eq(1) | cohort["OTHER_RETAIL_FLAG"].eq(1)).all():
    raise RuntimeError("Hay alertas de rating fuera de RRE/Other Retail; revisar población.")

scope_cols = ["ENTITY_ID", "HEAD_GEC_PARENT_ID", "GEC_NET_EXPOSURE", "RRE_FLAG", "OTHER_RETAIL_FLAG"]
scope = cohort[scope_cols].copy()
entity_dataset, entity_table = ns["get_sql_source"]("ENTITY_VW")
print("Fuente:", COHORT_DATASET, "| 06/11/2026, entidades con rating:", len(cohort))
print("Método:", PRODUCER_RECIPE, "| Oracle:", entity_table)
print("No se escribe ni se modifica ningún dataset.")

# ------------------------- SIMULACIÓN EXACTA ----------------------
# El día inicial del downgrade NO cambia entre escenarios: 02/11.
# Se prolonga su persistencia hasta 06/11, 13/11 o 20/11, sin saltarse
# días hábiles intermedios ni permitir datos de rating ausentes.
result = cohort[["ENTITY_ID", "ORIGINAL_RATING_CODES", "TRIGGERED_INDICATOR_CODES"]].copy()
all_flags = {}
all_qas = {}
for lag in LAGS:
    confirmation = pd.bdate_range(start=INITIAL_TRIGGER_DATE, periods=lag)[-1]
    ns["RATING_PERSISTENCE_BUSINESS_DAYS"] = lag
    flags, qa, _, previous, trigger, dates = ns["build_rating_persistence_flags"](
        scope, entity_dataset, entity_table, confirmation
    )
    if (previous != PREVIOUS_STATE_DATE or trigger != INITIAL_TRIGGER_DATE
            or dates[-1] != confirmation or len(dates) != lag):
        raise AssertionError(f"Ventana de {lag} días distinta de la del 6/11: {previous}, {trigger}, {dates[-1]}")
    rule_cols = [c for c in flags.columns if c.startswith(RATING_PREFIXES)]
    found = set(rule_cols)
    unexpected = sorted({c for cs in result["ORIGINAL_RATING_CODES"] for c in cs} - found)
    if unexpected:
        raise RuntimeError(f"Códigos originales no reproducibles con la receta actual: {unexpected}")
    flags = flags.set_index("ENTITY_ID").reindex(result["ENTITY_ID"])
    flags[rule_cols] = flags[rule_cols].fillna(0).astype("int8")
    flag_matrix = flags[rule_cols]
    result[f"ANY_RATING_{lag}D"] = flag_matrix.any(axis=1).astype("int8").to_numpy()
    result[f"ORIGINAL_EWI_{lag}D"] = [
        int(any(flag_matrix.loc[eid, code] == 1 for code in codes))
        for eid, codes in zip(result["ENTITY_ID"], result["ORIGINAL_RATING_CODES"])
    ]
    all_flags[lag] = flags
    all_qas[lag] = qa
    print(f"Lag {lag:>2}d | Day 1: {trigger.date()} | Day {lag}: {confirmation.date()} "
          f"| cualquier indicador: {result[f'ANY_RATING_{lag}D'].sum()} "
          f"| indicador original: {result[f'ORIGINAL_EWI_{lag}D'].sum()}")

# El método actualizado y la cohorte exportada deben reconciliarse en el caso base.
n_original = len(result)
n_base = int(result["ANY_RATING_5D"].sum())
n_base_ewi = int(result["ORIGINAL_EWI_5D"].sum())
print(f"QA 5d: cohorte observada={n_original}; reproducida (rating)={n_base}; "
      f"mismo EWI={n_base_ewi}; cliente mencionó un pico de 67 alertas.")
if n_base != n_original or n_base_ewi != n_original:
    raise RuntimeError(
        "No cuadra el lag original de 5 días con las alertas del 06/11. "
        "Revisar si el dataset histórico usa otra versión de la receta, una "
        "regla adicional de alertado/bundling o si cambiaron las fuentes de rating. "
        "No presentar cifras de 10/15 días al cliente hasta reconciliar."
    )

summary = []
for lag in LAGS:
    survivors = int(result[f"ANY_RATING_{lag}D"].sum())
    same_ewi = int(result[f"ORIGINAL_EWI_{lag}D"].sum())
    summary.append({
        "LAG_BUSINESS_DAYS": lag,
        "CONFIRMATION_DATE": pd.bdate_range(INITIAL_TRIGGER_DATE, periods=lag)[-1].date(),
        "COHORT_6_NOV": n_original,
        "RATING_ALERTS_RETAINED": survivors,
        "RATING_ALERTS_AVOIDED": n_original - survivors,
        "REDUCTION_PCT": round(100 * (n_original - survivors) / n_original, 2),
        "SAME_ORIGINAL_EWI_RETAINED": same_ewi,
    })
summary = pd.DataFrame(summary)
result["SUPPRESSED_WITH_10D"] = (result["ANY_RATING_5D"].eq(1) & result["ANY_RATING_10D"].eq(0))
result["SUPPRESSED_WITH_15D"] = (result["ANY_RATING_5D"].eq(1) & result["ANY_RATING_15D"].eq(0))
print("\nRESUMEN - COHORTE DEL 6 NOVIEMBRE, UNA ALERTA RATING POR ENTIDAD")
display(summary)
print("\nDETALLE POR CLIENTE/ENTIDAD (solo lectura)")
display(result.sort_values(["SUPPRESSED_WITH_15D", "SUPPRESSED_WITH_10D"], ascending=False))
