# -*- coding: utf-8 -*-
"""Retail EWS | Read-only alert and breach-export reconciliation.

Run in the existing Retail Dataiku notebook after the 5/10/15-day calculation.
Reads the existing datasets only; no Oracle calls and no Flow writes.
"""
import pandas as pd
import dataiku
from IPython.display import display

REFERENCE_DATE = "2026-11-06"
ALERT_DATASET = "EWS_Daily_Alerts"
BREACH_DATASET = "ews_ewi_breach_information"
REPORTED_PEAK = 67
RATING_PREFIXES = (
    "RRE_CLIENT_", "OTHER_CLIENT_",
    "RRE_GEC_HEAD_", "OTHER_GEC_HEAD_",
)


def heading(text):
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100, flush=True)


def show(title, records):
    heading(title)
    frame = pd.DataFrame(records)
    if frame.empty:
        print("NO ROWS")
    else:
        display(frame.reset_index(drop=True))
    return frame


def norm_id(values):
    return (values.astype("string").str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA}))


def norm_date(values):
    dates = values.astype("string").str.strip().str.slice(0, 10)
    return pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce")


def codes(value):
    if pd.isna(value):
        return set()
    return {x.strip() for x in str(value).split("|") if x.strip()}


def is_rating_export(value):
    label = str(value).strip().lower()
    return label.startswith("client rating ") or label.startswith("gec head ")


heading("RETAIL EWS | RATING PERSISTENCE / BREACH EXPORT RECONCILIATION")
print("Reference date          :", REFERENCE_DATE)
print("Reported rating peak    :", REPORTED_PEAK, "(external statement; denominator unverified)")
print("Daily alert dataset     :", ALERT_DATASET)
print("Breach export dataset   :", BREACH_DATASET)
print("Execution mode          : READ ONLY")

alerts = dataiku.Dataset(ALERT_DATASET).get_dataframe()
needed = {"ENTITY_ID", "TRIGGERED_INDICATOR_CODES", "ALERT_DATE"}
if needed - set(alerts.columns):
    raise RuntimeError("Missing daily alert columns: " + ", ".join(sorted(needed - set(alerts.columns))))
alerts = alerts.copy()
alerts["ENTITY_ID"] = norm_id(alerts["ENTITY_ID"])
alerts["ALERT_DATE"] = norm_date(alerts["ALERT_DATE"])
alerts = alerts.loc[alerts["ALERT_DATE"].eq(pd.Timestamp(REFERENCE_DATE))].copy()
if alerts.empty:
    raise RuntimeError("Daily alert dataset does not contain the specified reference date")
if alerts["ENTITY_ID"].isna().any():
    raise RuntimeError("Daily alerts contain missing client IDs")
alerts["ALL_CODES"] = alerts["TRIGGERED_INDICATOR_CODES"].map(codes)
alerts["RATING_CODES"] = alerts["ALL_CODES"].map(
    lambda a: {v for v in a if v.startswith(RATING_PREFIXES)})
alerts["NON_RATING_CODES"] = alerts["ALL_CODES"].map(
    lambda a: {v for v in a if not v.startswith(RATING_PREFIXES)})
all_pairs = {(eid, code) for eid, ss in zip(alerts["ENTITY_ID"], alerts["ALL_CODES"]) for code in ss}
rating_pairs = {(eid, code) for eid, ss in zip(alerts["ENTITY_ID"], alerts["RATING_CODES"]) for code in ss}
nonrating_pairs = all_pairs - rating_pairs
rating_entities = {eid for eid, _ in rating_pairs}
nonrating_entities = {eid for eid, _ in nonrating_pairs}
all_entities = set(alerts["ENTITY_ID"].dropna())

# The source export is read directly from its verified Retail Flow dataset.
try:
    exported = dataiku.Dataset(BREACH_DATASET).get_dataframe()
except Exception as exc:
    raise RuntimeError(
        "Breach export could not be read. Confirm that " + BREACH_DATASET +
        " contains the archived 2026-11-06 run. Dataset reading error: " + type(exc).__name__
    ) from None
necessary = {"BREACH_ID", "ENTITY_ID", "EWI_NAME", "ALERT_DATE"}
if necessary - set(exported.columns):
    raise RuntimeError("Missing breach export columns: " + ", ".join(sorted(necessary - set(exported.columns))))
exported = exported.copy()
exported["ENTITY_ID"] = norm_id(exported["ENTITY_ID"])
exported["ALERT_DATE"] = norm_date(exported["ALERT_DATE"])
exported = exported.loc[exported["ALERT_DATE"].eq(pd.Timestamp(REFERENCE_DATE))].copy()
if exported.empty:
    raise RuntimeError(
        "Breach export has no rows on 2026-11-06. Restore the archived output "
        "for that run rather than comparing with a different run."
    )
exported["IS_RATING"] = exported["EWI_NAME"].map(is_rating_export)
exported["IS_STAGE2_CONNECTED"] = exported["EWI_NAME"].astype("string").str.contains(
    "Connected client classified as Stage 2", case=False, regex=False, na=False
)
exported["BREACH_ID"] = exported["BREACH_ID"].astype("string").str.strip()
export_entities = set(exported["ENTITY_ID"].dropna())
export_rating_entities = set(exported.loc[exported.IS_RATING, "ENTITY_ID"].dropna())
export_nonrating_entities = set(exported.loc[~exported.IS_RATING, "ENTITY_ID"].dropna())

# Daily alert codes reflect the post-bundling daily alert dataset. The export
# can apply subsequent filtering and deprecated-code exclusions.
source_summary = {
    "DAILY_ROWS": len(alerts),
    "DAILY_UNIQUE_CLIENTS": len(all_entities),
    "DAILY_UNIQUE_CLIENT_EWI": len(all_pairs),
    "DAILY_RATING_CLIENTS": len(rating_entities),
    "DAILY_RATING_CLIENT_EWI": len(rating_pairs),
    "DAILY_NONRATING_CLIENTS": len(nonrating_entities),
    "DAILY_NONRATING_CLIENT_EWI": len(nonrating_pairs),
}
export_summary = {
    "EXPORT_ROWS": len(exported),
    "EXPORT_UNIQUE_BREACH_ID": exported["BREACH_ID"].nunique(),
    "EXPORT_UNIQUE_CLIENTS": len(export_entities),
    "EXPORT_RATING_ROWS": int(exported["IS_RATING"].sum()),
    "EXPORT_RATING_CLIENTS": len(export_rating_entities),
    "EXPORT_NONRATING_ROWS": int((~exported["IS_RATING"]).sum()),
    "EXPORT_NONRATING_CLIENTS": len(export_nonrating_entities),
    "EXPORT_STAGE2_CONNECTED_ROWS": int(exported["IS_STAGE2_CONNECTED"].sum()),
}

show("01 | REFERENCE-DATE COUNTS AND COUNTING UNITS", [
    {"SOURCE": "Client-reported peak", "METRIC": "Reported alerts", "VALUE": REPORTED_PEAK, "NOTE": "Unverified count basis"},
    *({"SOURCE": "Daily alerts", "METRIC": k, "VALUE": v, "NOTE": "Post daily alert selection"}
      for k, v in source_summary.items()),
    *({"SOURCE": "Breach export", "METRIC": k, "VALUE": v, "NOTE": "Saved external export"}
      for k, v in export_summary.items()),
])

show("02 | BREACH EXPORT BY INDICATOR", [
    {"FAMILY": "RATING" if rating else "OTHER", "EWI_NAME": name,
     "BREACH_ROWS": len(group), "UNIQUE_CLIENTS": group["ENTITY_ID"].nunique()}
    for (rating, name), group in exported.groupby(["IS_RATING", "EWI_NAME"], dropna=False)
])

show("03 | DAILY ALERTS VS SAVED BREACH EXPORT", [
    {"CHECK": "Daily clients also in breach export", "COUNT": len(all_entities & export_entities)},
    {"CHECK": "Daily clients not in breach export", "COUNT": len(all_entities - export_entities)},
    {"CHECK": "Breach export clients not in daily alerts", "COUNT": len(export_entities - all_entities)},
    {"CHECK": "Rating clients in both sources", "COUNT": len(rating_entities & export_rating_entities)},
    {"CHECK": "Rating clients only in daily alerts", "COUNT": len(rating_entities - export_rating_entities)},
    {"CHECK": "Rating clients only in breach export", "COUNT": len(export_rating_entities - rating_entities)},
    {"CHECK": "Duplicate breach IDs", "COUNT": int(exported["BREACH_ID"].duplicated().sum())},
    {"CHECK": "Duplicate client / EWI names", "COUNT": int(exported.duplicated(["ENTITY_ID", "EWI_NAME"]).sum())},
])

# Validate the existing persistence simulation without querying Oracle again.
scenario_status = "NOT AVAILABLE IN THIS KERNEL"
scenario_data = []
if "result" in globals():
    scenario = result.copy()
    required_scenario_cols = {"ENTITY_ID"} | {
        f"ANY_RATING_{n}D" for n in (5, 10, 15)} | {
        f"ORIGINAL_EWI_{n}D" for n in (5, 10, 15)}
    if required_scenario_cols.issubset(scenario.columns):
        scenario["ENTITY_ID"] = norm_id(scenario["ENTITY_ID"])
        sim_entities = set(scenario["ENTITY_ID"].dropna())
        if sim_entities != rating_entities or len(scenario) != len(rating_entities):
            scenario_status = "REVIEW: SIMULATION COHORT DOES NOT MATCH DAILY RATING COHORT"
        elif (scenario["ANY_RATING_5D"] == 1).all() and (scenario["ORIGINAL_EWI_5D"] == 1).all():
            scenario_status = "PASS: SAME ORIGINAL RATING COHORT"
            for n in (5, 10, 15):
                surviving = set(scenario.loc[
                    pd.to_numeric(scenario[f"ANY_RATING_{n}D"], errors="coerce").eq(1), "ENTITY_ID"])
                scenario_data.append({
                    "PERSISTENCE_BD": n,
                    "RATING_CLIENTS_RETAINED": len(surviving),
                    "RATING_CLIENTS_AVOIDED": len(rating_entities) - len(surviving),
                    "RATING_REDUCTION_PCT": round(100 * (len(rating_entities) - len(surviving)) / len(rating_entities), 2),
                    "SAME_ORIGINAL_EWI_RETAINED": int(pd.to_numeric(scenario[f"ORIGINAL_EWI_{n}D"], errors="coerce").sum()),
                    "HYPOTHETICAL_ALL_ALERT_CLIENTS": len(surviving | nonrating_entities),
                })
        else:
            scenario_status = "REVIEW: 5-DAY BASELINE DOES NOT REPRODUCE ALL RATING CLIENTS"

show("04 | RATING PERSISTENCE / FIXED-COHORT SENSITIVITY", scenario_data)
print("Simulation reconciliation :", scenario_status)
print("Note: Hypothetical total retains other 6 Nov indicators unchanged.")
print("Note: This is NOT a forecast of alerts on 13 or 20 November.")

show("05 | 67-ALERT PEAK: EXACT METRIC COMPARISONS", [
    {"METRIC": "Daily all clients", "OBSERVED": len(all_entities), "DELTA_VS_67": len(all_entities) - REPORTED_PEAK},
    {"METRIC": "Daily rating clients", "OBSERVED": len(rating_entities), "DELTA_VS_67": len(rating_entities) - REPORTED_PEAK},
    {"METRIC": "Daily rating client/EWI instances", "OBSERVED": len(rating_pairs), "DELTA_VS_67": len(rating_pairs) - REPORTED_PEAK},
    {"METRIC": "Export all breach rows", "OBSERVED": len(exported), "DELTA_VS_67": len(exported) - REPORTED_PEAK},
    {"METRIC": "Export rating breach rows", "OBSERVED": int(exported["IS_RATING"].sum()), "DELTA_VS_67": int(exported["IS_RATING"].sum()) - REPORTED_PEAK},
    {"METRIC": "Export rating clients", "OBSERVED": len(export_rating_entities), "DELTA_VS_67": len(export_rating_entities) - REPORTED_PEAK},
])

heading("06 | FINAL VALIDATION AND PRESENTATION STATUS")
valid_export = bool(exported["BREACH_ID"].notna().all() and
                    (exported["BREACH_ID"].str.len() > 0).all() and
                    exported["BREACH_ID"].is_unique and
                    exported["ENTITY_ID"].notna().all())
print("Saved breach export integrity       :", "PASS" if valid_export else "REVIEW")
print("Stage 2 connected rows in export   :", int(exported["IS_STAGE2_CONNECTED"].sum()))
print("Export recipe/version alignment    : VERIFY if Stage 2 was excluded in deployed export")
print("5/10/15-day scenario reconciliation :", scenario_status)
print("Daily date and export date          :", REFERENCE_DATE)
print("Client peak of 67                  : NOT PROVEN FROM CURRENT SOURCES")
print("False-positive impact               : NOT TESTED")
print("Future daily-volume forecast        : NOT SIMULATED")
print("Dataset modifications              : NONE")
print("Next evidence: Archived original peak run / pre-bundling counts if 67 is not matched and defined.")
