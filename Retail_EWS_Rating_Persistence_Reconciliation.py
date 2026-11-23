# -*- coding: utf-8 -*-
"""Retail EWS | Rating persistence sensitivity and alert-volume reconciliation.

Run after the read-only 5/10/15 business-day simulation in the same Dataiku notebook.
Uses the existing `result` and `cohort` DataFrames; does not rerun Oracle history.
Reads EWS_Daily_Alerts only if the daily `alerts` DataFrame is not available.
Does not modify any dataset, recipe, source, project configuration, or file.
"""

import html
import pandas as pd
import dataiku
from IPython.display import display, HTML

REFERENCE_DATE = "2026-11-06"
TRIGGER_DATE = "2026-11-02"
REPORTED_PEAK = 67  # Client-reported rating-downgrade alerts; unit not yet verified
DATASET = "EWS_Daily_Alerts"
LAGS = (5, 10, 15)
RATING_PREFIXES = ("RRE_CLIENT_", "OTHER_CLIENT_", "RRE_GEC_HEAD_", "OTHER_GEC_HEAD_")


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100, flush=True)


def show(frame, columns=None):
    table = frame.copy()
    if columns is not None:
        table = table[columns]
    def fmt(value):
        if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
            return ""
        if isinstance(value, float):
            return f"{value:,.2f}"
        if isinstance(value, (int,)):
            return f"{value:,}"
        return str(value)
    headings = "".join(f"<th>{html.escape(str(c))}</th>" for c in table.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(fmt(v))}</td>" for v in row) + "</tr>"
        for row in table.itertuples(index=False, name=None)
    )
    css = (
        "<style>.ewsqa{border-collapse:collapse;width:100%;font-family:Arial,sans-serif;"
        "font-size:12px;margin:7px 0 16px 0}"
        ".ewsqa th{background:#18324d;color:white;text-align:left;padding:9px 10px;}"
        ".ewsqa td{padding:8px 10px;border-bottom:1px solid #dce3eb;}"
        ".ewsqa tr:nth-child(even) td{background:#f3f6f9;}"
        "</style>"
    )
    display(HTML(css + '<table class="ewsqa"><thead><tr>' + headings +
                 '</tr></thead><tbody>' + body + '</tbody></table>'))


def normal_id(values):
    return (values.astype("string").str.strip()
            .str.replace(r"\.0+$", "", regex=True))


def parse_iso_dates(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    s = series.astype("string").str.strip()
    short = s.str.slice(0, 10)
    date = pd.to_datetime(short, format="%Y-%m-%d", errors="coerce")
    numeric = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return date.fillna(numeric)


def parse_codes(value):
    if value is None or pd.isna(value):
        return []
    return sorted(set(code.strip() for code in str(value).split("|") if code.strip()))


def peak_message(metrics, peak):
    rating_instances = metrics["rating_instances"]
    rating_entities = metrics["rating_entities"]
    if rating_instances == peak and rating_entities != peak:
        return ("RECONCILED (EXPORTED EWI COUNT)",
                "67 rating indicator instances correspond to 56 unique rating entities; "
                "the difference reflects more than one rating EWI on some entities. "
                "The original pre-bundling number is not established.")
    if rating_entities == peak:
        return ("RECONCILED (UNIQUE ENTITY COUNT)",
                "The client-reported peak matches distinct rating entities on this date.")
    if metrics["all_instances"] == peak:
        return ("NUMERIC MATCH ONLY (ALL EWI)",
                "67 matches all exported indicator instances, not rating-only indicators. "
                "Confirm the scope of the client's reported peak.")
    if metrics["all_rows"] == peak:
        return ("NUMERIC MATCH ONLY (ALL ALERT ROWS)",
                "67 matches all exported alert rows, not rating-only indicators. "
                "Confirm whether the client referred to all categories.")
    if metrics["all_entities"] == peak:
        return ("NUMERIC MATCH ONLY (ALL CLIENTS)",
                "67 matches unique clients across all indicators, not rating-only clients. "
                "Confirm the scope of the client's reported peak.")
    return ("NOT RECONCILED",
            "The exported 6 November counts do not reproduce 67. "
            "Check the original daily execution log, pre-bundling indicator counts, "
            "date and recipe version; do not manufacture 11 missing alerts.")


section("RETAIL EWS | RATING PERSISTENCE AND ALERT RECONCILIATION")
print("Reference date           : " + REFERENCE_DATE)
print("First deterioration day  : " + TRIGGER_DATE)
print("Scenario confirmation    : 2026-11-06 / 2026-11-13 / 2026-11-20")
print("Source                   : " + DATASET)
print("Run mode                 : READ ONLY; no Oracle history rerun")
print("Reported peak            : 67 rating-downgrade alerts (client statement)")

# The scenario simulation must have completed successfully in the same kernel.
if "result" not in globals() or "cohort" not in globals():
    raise RuntimeError("Missing result/cohort: run the existing 5/10/15-day simulation first.")
scenarios = globals()["result"].copy()
cohort_ref = globals()["cohort"].copy()
needed_scenario = {"ENTITY_ID", "ORIGINAL_RATING_CODES", "TRIGGERED_INDICATOR_CODES"}
for lag in LAGS:
    needed_scenario.update({f"ANY_RATING_{lag}D", f"ORIGINAL_EWI_{lag}D"})
if needed_scenario - set(scenarios.columns):
    raise RuntimeError("Missing scenario columns: " + ", ".join(sorted(needed_scenario - set(scenarios.columns))))
if {"ENTITY_ID", "RRE_FLAG", "OTHER_RETAIL_FLAG"} - set(cohort_ref.columns):
    raise RuntimeError("Missing Retail portfolio columns in cohort.")

scenarios["ENTITY_ID"] = normal_id(scenarios["ENTITY_ID"])
cohort_ref["ENTITY_ID"] = normal_id(cohort_ref["ENTITY_ID"])
if scenarios["ENTITY_ID"].isna().any() or scenarios["ENTITY_ID"].duplicated().any():
    raise RuntimeError("Scenario entity keys are null or duplicated.")
if cohort_ref["ENTITY_ID"].isna().any() or cohort_ref["ENTITY_ID"].duplicated().any():
    raise RuntimeError("Cohort entity keys are null or duplicated.")
for lag in LAGS:
    for field in (f"ANY_RATING_{lag}D", f"ORIGINAL_EWI_{lag}D"):
        scenarios[field] = pd.to_numeric(scenarios[field], errors="coerce")
        if not scenarios[field].isin([0, 1]).all():
            raise RuntimeError(f"Invalid or missing binary scenario result: {field}")

# Use the snapshot already loaded by the simulation whenever possible.
if "alerts" in globals() and isinstance(globals()["alerts"], pd.DataFrame):
    daily = globals()["alerts"].copy()
    daily_source = "Existing in-memory alerts DataFrame"
else:
    daily = dataiku.Dataset(DATASET).get_dataframe()
    daily_source = "Dataiku dataset read"
required_daily = {"ALERT_DATE", "ENTITY_ID", "TRIGGERED_INDICATOR_CODES"}
if required_daily - set(daily.columns):
    raise RuntimeError("Missing alert columns: " + ", ".join(sorted(required_daily - set(daily.columns))))
daily["ALERT_DATE"] = parse_iso_dates(daily["ALERT_DATE"])
daily = daily.loc[daily["ALERT_DATE"].eq(pd.Timestamp(REFERENCE_DATE))].copy()
if daily.empty:
    raise RuntimeError("No source alert records for 2026-11-06; original output must be recovered.")
daily["ENTITY_ID"] = normal_id(daily["ENTITY_ID"])
if daily["ENTITY_ID"].isna().any():
    raise RuntimeError("Some daily alert records have no entity identifier.")
daily["ALL_CODES"] = daily["TRIGGERED_INDICATOR_CODES"].map(parse_codes)
daily["RATING_CODES"] = daily["ALL_CODES"].map(
    lambda codes: [x for x in codes if x.startswith(RATING_PREFIXES)])
daily["NONRATING_CODES"] = daily["ALL_CODES"].map(
    lambda codes: [x for x in codes if not x.startswith(RATING_PREFIXES)])

# One distinct client / EWI / reference date is counted once.
all_codes = daily[["ENTITY_ID", "ALL_CODES"]].explode("ALL_CODES").rename(columns={"ALL_CODES": "EWI_CODE"})
all_codes = all_codes.dropna(subset=["EWI_CODE"]).drop_duplicates(["ENTITY_ID", "EWI_CODE"])
rating_codes = daily[["ENTITY_ID", "RATING_CODES"]].explode("RATING_CODES").rename(columns={"RATING_CODES": "EWI_CODE"})
rating_codes = rating_codes.dropna(subset=["EWI_CODE"]).drop_duplicates(["ENTITY_ID", "EWI_CODE"])
nonrating_codes = daily[["ENTITY_ID", "NONRATING_CODES"]].explode("NONRATING_CODES").rename(columns={"NONRATING_CODES": "EWI_CODE"})
nonrating_codes = nonrating_codes.dropna(subset=["EWI_CODE"]).drop_duplicates(["ENTITY_ID", "EWI_CODE"])
source_rating_ids = set(rating_codes["ENTITY_ID"])
scenario_ids = set(scenarios["ENTITY_ID"])
cohort_ids = set(cohort_ref["ENTITY_ID"])
metrics = {
    "all_rows": len(daily),
    "all_entities": daily["ENTITY_ID"].nunique(),
    "all_instances": len(all_codes),
    "rating_entities": len(source_rating_ids),
    "rating_instances": len(rating_codes),
    "nonrating_entities": nonrating_codes["ENTITY_ID"].nunique(),
    "nonrating_instances": len(nonrating_codes),
}

section("01 | ALERT COUNTS: UNITS AND BUNDLING")
print("Daily alert source: " + daily_source)
count_report = pd.DataFrame([
    ("Client-reported peak", REPORTED_PEAK, "Unverified unit; email statement"),
    ("Source rows on 6 Nov", metrics["all_rows"], "Rows in EWS_Daily_Alerts"),
    ("Distinct clients: all alerts", metrics["all_entities"], "Unique ENTITY_ID"),
    ("Distinct client/EWI: all indicators", metrics["all_instances"], "Post-export codes, de-duplicated"),
    ("Distinct clients: rating EWI", metrics["rating_entities"], "Rating prefixes only"),
    ("Distinct client/EWI: rating", metrics["rating_instances"], "Post-export rating code instances"),
    ("Distinct clients: non-rating EWI", metrics["nonrating_entities"], "May overlap with rating clients"),
    ("Distinct client/EWI: non-rating", metrics["nonrating_instances"], "Other exported codes"),
    ("Rating instances above unique clients", metrics["rating_instances"] - metrics["rating_entities"], "Multiple rating codes per client"),
], columns=["MEASURE", "COUNT", "DEFINITION"])
show(count_report)

multi = rating_codes.groupby("ENTITY_ID").size()
show(pd.DataFrame([
    ("One rating EWI", int(multi.eq(1).sum())),
    ("Two rating EWIs", int(multi.eq(2).sum())),
    ("Three or more rating EWIs", int(multi.ge(3).sum())),
], columns=["RATING EWIs PER CLIENT", "CLIENTS"]))

by_ewi = rating_codes.groupby("EWI_CODE")["ENTITY_ID"].nunique().sort_values(ascending=False)
section("02 | RATING INDICATOR DISTRIBUTION")
show(by_ewi.rename("DISTINCT_CLIENTS").reset_index())

peak_status, peak_note = peak_message(metrics, REPORTED_PEAK)
section("03 | CLIENT-REPORTED PEAK: RECONCILIATION")
print("Status: " + peak_status)
print("Finding: " + peak_note)

section("04 | PERSISTENCE SENSITIVITY: FIXED ORIGINAL RATING COHORT")
base_n = len(scenarios)
scenario_rows = []
for lag in LAGS:
    retained = int(scenarios[f"ANY_RATING_{lag}D"].sum())
    identical = int(scenarios[f"ORIGINAL_EWI_{lag}D"].sum())
    date = pd.bdate_range(TRIGGER_DATE, periods=lag)[-1].strftime("%Y-%m-%d")
    scenario_rows.append((lag, date, base_n, retained, base_n-retained,
                          round(100*(base_n-retained)/base_n, 2), identical,
                          retained-identical))
summary = pd.DataFrame(scenario_rows, columns=[
    "BUSINESS DAYS", "CONFIRMATION DATE", "ORIGINAL CLIENTS", "RATING RETAINED",
    "RATING NOT RETAINED", "REDUCTION %", "SAME ORIGINAL EWI", "OTHER RATING EWI ONLY"])
show(summary)
print("Additional clients not retained between 10d and 15d: " +
      str(int(scenarios["ANY_RATING_10D"].sum() - scenarios["ANY_RATING_15D"].sum())))
print("Scope: fixed 6-Nov rating cohort; not a simulation of all future daily EWS alerts.")

section("05 | PORTFOLIO QA: OVERLAP IS ALLOWED")
ports = cohort_ref[["ENTITY_ID", "RRE_FLAG", "OTHER_RETAIL_FLAG"]].copy()
for field in ("RRE_FLAG", "OTHER_RETAIL_FLAG"):
    ports[field] = pd.to_numeric(ports[field], errors="coerce").fillna(0).astype(int)
ports["PORTFOLIO_SCOPE"] = "NEITHER"
a = ports["RRE_FLAG"].eq(1)
b = ports["OTHER_RETAIL_FLAG"].eq(1)
ports.loc[a & ~b, "PORTFOLIO_SCOPE"] = "RRE only"
ports.loc[~a & b, "PORTFOLIO_SCOPE"] = "Other Retail only"
ports.loc[a & b, "PORTFOLIO_SCOPE"] = "Both RRE and Other Retail"
merged = scenarios.merge(ports[["ENTITY_ID", "PORTFOLIO_SCOPE"]],
                         on="ENTITY_ID", how="left", validate="one_to_one")
rows = []
for label in ("RRE only", "Other Retail only", "Both RRE and Other Retail", "NEITHER"):
    g = merged.loc[merged["PORTFOLIO_SCOPE"].eq(label)]
    rows.append((label, len(g), int(g["ANY_RATING_10D"].sum()),
                 int(g["ANY_RATING_15D"].sum())))
portfolio_summary = pd.DataFrame(rows, columns=["EXCLUSIVE CLIENT SCOPE", "BASELINE 5D", "RETAINED 10D", "RETAINED 15D"])
show(portfolio_summary)
print("RRE members (including overlaps)          : " + str(int(a.sum())))
print("Other Retail members (including overlaps) : " + str(int(b.sum())))
print("Clients in both portfolios                : " + str(int((a & b).sum())))
print("Note: overlapping portfolio memberships must not be summed to compute unique clients.")

section("06 | 6-NOV COHORT OVERLAY: NON-RATING CODES HELD CONSTANT")
# This is only a hypothetical overlay on the SAME original-day cohort. It is NOT
# a forecast of actual alert totals on 13 or 20 November.
all_ids = set(daily["ENTITY_ID"])
nonrating_ids = set(nonrating_codes["ENTITY_ID"])
comparison = []
for lag in LAGS:
    kept_rating = set(scenarios.loc[scenarios[f"ANY_RATING_{lag}D"].eq(1), "ENTITY_ID"])
    surviving_entities = len(nonrating_ids | kept_rating)
    comparison.append((lag, len(all_ids), surviving_entities, len(all_ids)-surviving_entities))
overlay = pd.DataFrame(comparison, columns=[
    "RATING LAG (DAYS)", "ORIGINAL ALL-EWI CLIENTS", "RETAINED CLIENTS (HYPOTHETICAL)", "REMOVED (HYPOTHETICAL)"])
show(overlay)
print("Assumption: original non-rating EWIs are frozen; future dates and newly arising alerts are not modelled.")

section("07 | CONSISTENCY QA")
qa_checks = [
    ("Rating cohort equals exported rating entities", scenario_ids == source_rating_ids),
    ("Scenario IDs equal original cohort IDs", scenario_ids == cohort_ids),
    ("5-day scenario reproduces all original rating clients", int(scenarios["ANY_RATING_5D"].sum()) == base_n),
    ("5-day scenario reproduces original rating EWI", int(scenarios["ORIGINAL_EWI_5D"].sum()) == base_n),
    ("Original rating EWI remains subset of any rating flag",
     all((scenarios[f"ORIGINAL_EWI_{lag}D"] <= scenarios[f"ANY_RATING_{lag}D"]).all() for lag in LAGS)),
    ("Longer persistence cannot increase retained rating clients",
     (scenarios["ANY_RATING_15D"] <= scenarios["ANY_RATING_10D"]).all()
     and (scenarios["ANY_RATING_10D"] <= scenarios["ANY_RATING_5D"]).all()),
    ("Every rated entity belongs to at least one Retail portfolio",
     ports["PORTFOLIO_SCOPE"].ne("NEITHER").all()),
    ("Portfolio scope is one row per entity", len(ports) == ports["ENTITY_ID"].nunique()),
    ("No duplicate client/EWI keys in reported counts",
     not all_codes.duplicated(["ENTITY_ID", "EWI_CODE"]).any()),
]
qa_df = pd.DataFrame(qa_checks, columns=["VALIDATION CHECK", "PASS"])
qa_df["STATUS"] = qa_df["PASS"].map({True: "PASS", False: "REVIEW"})
show(qa_df[["VALIDATION CHECK", "STATUS"]])
qa_status = "PASS" if qa_df["PASS"].all() else "REVIEW"

section("08 | PRESENTATION-READY CONCLUSION")
print(f"Rating cohort                    : {base_n} unique entities on 6 November 2026")
print(f"5-day confirmation              : {int(scenarios['ANY_RATING_5D'].sum())} / {base_n} entities")
print(f"10-day confirmation             : {int(scenarios['ANY_RATING_10D'].sum())} / {base_n} entities")
print(f"15-day confirmation             : {int(scenarios['ANY_RATING_15D'].sum())} / {base_n} entities")
print("Incremental gain (10d to 15d)  : " +
      str(int(scenarios["ANY_RATING_10D"].sum()-scenarios["ANY_RATING_15D"].sum())) + " fewer entities")
print("Technical scenario validation   : " + qa_status)
print("Client peak reconciliation      : " + peak_status)
print("False-positive identification   : NOT TESTED")
print("Actual future daily alert count : NOT SIMULATED")
print("Operational approval            : PENDING")
print("Dataset modifications          : NONE")

# Notebook variables for subsequent checks; nothing is written to the Flow.
retail_peak_reconciliation = count_report
retail_rating_lag_summary = summary
retail_portfolio_lag_summary = portfolio_summary
retail_qa = qa_df
retail_peak_status = peak_status
