# Diagnostico READ ONLY, para ejecutar tras reconstruir alertas y outputs
import dataiku
import pandas as pd

BLOCKED = {"RRE_DOWNGRADE_TO_STAGE2", "OTHER_DOWNGRADE_TO_STAGE2"}
DATASETS = [
    "EWS_Daily_Alerts",
    "ews_ewi_breach_information",
    "ews_ewi_historic_information",
    "ews_exposure_static_data",
    "ews_entity_static_data",
]

ref = dataiku.Dataset("Implementation_Reference_Date").get_dataframe()
if len(ref) != 1:
    raise ValueError("Implementation_Reference_Date debe tener una fila")
print("REFERENCE_DATE:", ref["REFERENCE_DATE"].iloc[0])
print("PREVIOUS_DATE:", ref["PREVIOUS_DATE"].iloc[0])
print("RUN_DATE:", ref["RUN_DATE"].iloc[0] if "RUN_DATE" in ref else "sin columna")

for name in DATASETS:
    ds = dataiku.Dataset(name)
    cols = [x["name"] for x in ds.read_schema()]
    print("\n", name)
    print("Columnas:", len(cols), "| SEGMENT presente:", "SEGMENT" in {x.upper() for x in cols})
    if name != "EWS_Daily_Alerts" and "SEGMENT" in {x.upper() for x in cols}:
        raise ValueError(name + " todavia contiene SEGMENT")

alerts = dataiku.Dataset("EWS_Daily_Alerts").get_dataframe()
print("\nAlertas:", len(alerts))
if "TRIGGERED_INDICATOR_CODES" not in alerts.columns:
    raise KeyError("EWS_Daily_Alerts sin TRIGGERED_INDICATOR_CODES")
strings = alerts["TRIGGERED_INDICATOR_CODES"].fillna("").astype(str)
blocked_rows = strings.map(
    lambda v: any(code.strip() in BLOCKED for code in v.split("|"))
)
print("Alertas que contienen Stage2 downgrade suspendido:", int(blocked_rows.sum()))
if blocked_rows.any():
    raise ValueError("Hay Stage 2 downgrade en EWS_Daily_Alerts: reconstruir alertas")
for c in ["ALERT_DATE", "REFERENCE_DATE"]:
    if c in alerts.columns:
        print("Fechas distintas en alertas (",c,"):",
              sorted(alerts[c].dropna().astype(str).unique().tolist()))
print("\nDiagnostico terminado. Confirmar tambien fechas MAP_DTE y los CSV fisicos del SFTP.")
