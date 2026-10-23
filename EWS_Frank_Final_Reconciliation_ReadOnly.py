import re
import pandas as pd
import numpy as np
import dataiku
from dataiku.core.sql import SQLExecutor2

TARGET_ENTITY_ID = "211001530170"  # Verified case under investigation
ALERT_DATE = "2026-10-16"
FRANK_VALUES = {
    "2026-07-25": (1500.0, 1468.0),
    "2026-08-22": (500.0, 410.0),
    "2026-09-19": (1500.0, 1499.0),
}

if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", str(TARGET_ENTITY_ID).strip()):
    raise ValueError("Enter TARGET_ENTITY_ID for the credit-card case (not the Stage 2 case).")

alert_date = pd.Timestamp(ALERT_DATE).normalize()
previous_date = alert_date - pd.offsets.BDay(1)
months = pd.period_range(end=alert_date.to_period("M") - 1, periods=12, freq="M")
month_ends = [p.end_time.normalize() for p in months]
points = [(pd.Timestamp(d), "FRANK_DATE") for d in FRANK_VALUES]
points += [(d, "MONTH_END") for d in month_ends]
points += [(alert_date, "CURRENT"), (previous_date, "PREVIOUS_DAY")]
points += [(p.start_time.normalize(), "MONTH_START") for p in list(months) + [alert_date.to_period("M")]]
points = list(dict.fromkeys(points))


def sql_source(name):
    dataset = dataiku.Dataset(name)
    info = dataset.get_location_info().get("info", {})
    table = info.get("table") or info.get("tableName")
    schema = info.get("schema") or info.get("schemaName")
    if not table:
        raise ValueError("SQL table not available for " + name)
    if schema and "." not in table:
        table = schema + "." + table
    return dataset, table


usage_ds, usage_table = sql_source("CARD_USAGE_VW")
_, card_table = sql_source("CARD_REFERENTIAL_VW")
_, customer_table = sql_source("CUSTOMER_VW")
card_fields = {c["name"].upper(): c["name"] for c in dataiku.Dataset("CARD_REFERENTIAL_VW").read_schema()}
status_expr = "c." + card_fields["CARD_STATUS"] if "CARD_STATUS" in card_fields else "CAST(NULL AS VARCHAR2(60))"
points_sql = " UNION ALL ".join(
    "SELECT DATE '{}' AS OBS_DATE, '{}' AS POINT_KIND FROM dual".format(d.strftime("%Y-%m-%d"), k)
    for d, k in points
)

query = """
WITH sample_dates AS (
    {points_sql}
), owners AS (
    SELECT d.OBS_DATE, d.POINT_KIND,
           cust.CUSTOMER_ID, cust.COUNTERPARTY_IDENTIFIER_GLOBE AS ENTITY_ID
    FROM sample_dates d
    JOIN {customer_table} cust
      ON TRIM(CAST(cust.COUNTERPARTY_IDENTIFIER_GLOBE AS VARCHAR2(100))) = '{entity}'
     AND cust.TEK_STRT_DTE < d.OBS_DATE + 1
     AND NVL(cust.TEK_END_DTE, DATE '9999-12-31') >= d.OBS_DATE + 1
)
SELECT o.ENTITY_ID, o.OBS_DATE, o.POINT_KIND,
       c.CARD_ID, c.CARD_TYPE, {status_expr} AS CARD_STATUS,
       c.CREDIT_LIMIT, c.TEMPORARY_CREDIT_LIMIT,
       u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT AS OUTSTANDING,
       c.TEK_STRT_DTE AS CARD_VALID_FROM,
       u.TEK_STRT_DTE AS USAGE_VALID_FROM
FROM owners o
JOIN {card_table} c
  ON TRIM(CAST(c.CARD_OWNER_CUSTOMER_ID AS VARCHAR2(100))) =
     TRIM(CAST(o.CUSTOMER_ID AS VARCHAR2(100)))
 AND c.TEK_STRT_DTE < o.OBS_DATE + 1
 AND NVL(c.TEK_END_DTE, DATE '9999-12-31') >= o.OBS_DATE + 1
 AND UPPER(TRIM(c.CARD_TYPE)) = 'CREDIT CARD'
LEFT JOIN {usage_table} u
  ON TRIM(CAST(u.CARD_ID AS VARCHAR2(100))) = TRIM(CAST(c.CARD_ID AS VARCHAR2(100)))
 AND u.TEK_STRT_DTE < o.OBS_DATE + 1
 AND NVL(u.TEK_END_DTE, DATE '9999-12-31') >= o.OBS_DATE + 1
ORDER BY o.OBS_DATE, o.POINT_KIND, c.CARD_ID, c.TEK_STRT_DTE, u.TEK_STRT_DTE
""".format(points_sql=points_sql, customer_table=customer_table,
           card_table=card_table, usage_table=usage_table,
           entity=TARGET_ENTITY_ID, status_expr=status_expr)

print("=" * 105)
print("CREDIT CARD - SOURCE RECONCILIATION - READ ONLY")
print("=" * 105)
print("ENTITY_ID:", TARGET_ENTITY_ID, "| Current:", ALERT_DATE, "| Previous:", previous_date.date())
print("Sources: CUSTOMER_VW / CARD_REFERENTIAL_VW / CARD_USAGE_VW")
print("Date comparison uses source rows valid at each observation date (not an archived EWS run).")
raw = SQLExecutor2(dataset=usage_ds).query_to_df(query)
raw.columns = raw.columns.str.upper()
if raw.empty:
    raise ValueError("No source rows found. Check ENTITY_ID and historical customer/card relationships.")

raw["CARD_ID"] = raw["CARD_ID"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
raw["OBS_DATE"] = pd.to_datetime(raw["OBS_DATE"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
for c in ["CARD_VALID_FROM", "USAGE_VALID_FROM"]:
    raw[c] = pd.to_datetime(raw[c], errors="coerce", utc=True).dt.tz_localize(None)
for c in ["OUTSTANDING", "CREDIT_LIMIT", "TEMPORARY_CREDIT_LIMIT"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw["EFFECTIVE_LIMIT"] = raw[["CREDIT_LIMIT", "TEMPORARY_CREDIT_LIMIT"]].max(axis=1, skipna=True)
raw["OUTSTANDING_ABS"] = raw["OUTSTANDING"].abs()
raw["MONTH"] = raw["OBS_DATE"].dt.to_period("M").astype(str)

print("\n1. SOURCE COVERAGE AND DUPLICATES")
print("Raw rows:", len(raw), "| Distinct cards:", raw["CARD_ID"].nunique())
dups = raw.groupby(["POINT_KIND", "OBS_DATE", "CARD_ID"], dropna=False).size().reset_index(name="SOURCE_ROWS")
print("Card/date combinations with multiple candidate source rows:", int(dups["SOURCE_ROWS"].gt(1).sum()))
print(dups[dups["SOURCE_ROWS"].gt(1)].to_string(index=False) if dups["SOURCE_ROWS"].gt(1).any() else "None")
print("Missing OUTSTANDING source rows:", int(raw["OUTSTANDING"].isna().sum()))
print("Observation-date timezone:", raw["OBS_DATE"].dtype)
print("Available point kinds:", raw["POINT_KIND"].value_counts().to_dict())
print("Actual zero OUTSTANDING source rows:", int(raw["OUTSTANDING"].eq(0).sum()))

print("\n2. ALL SOURCE CANDIDATES - FRANK DATES AND RECENT MONTH ENDS")
focus_dates = set(pd.to_datetime(list(FRANK_VALUES))) | set(month_ends[-3:]) | {alert_date, previous_date}
raw_focus = raw[raw["OBS_DATE"].isin(focus_dates) & raw["POINT_KIND"].ne("MONTH_START")].copy()
cols = ["POINT_KIND", "OBS_DATE", "CARD_ID", "CARD_STATUS", "OUTSTANDING", "CREDIT_LIMIT", "TEMPORARY_CREDIT_LIMIT", "EFFECTIVE_LIMIT", "CARD_VALID_FROM", "USAGE_VALID_FROM"]
print(raw_focus[cols].sort_values(["OBS_DATE", "CARD_ID", "CARD_VALID_FROM", "USAGE_VALID_FROM"]).to_string(index=False))

# Diagnostic representative selection only. A tie or multiple candidates means historical EWS selection is unverified.
one = (raw.sort_values(["OBS_DATE", "POINT_KIND", "CARD_ID", "CARD_VALID_FROM", "USAGE_VALID_FROM"], kind="stable")
       .drop_duplicates(["POINT_KIND", "OBS_DATE", "CARD_ID"], keep="last").copy())
one["EXCLUDED_EUR_1"] = one["EFFECTIVE_LIMIT"].eq(1.0)
first = (one[one["POINT_KIND"].eq("MONTH_START")][["CARD_ID", "MONTH", "EFFECTIVE_LIMIT"]]
         .rename(columns={"EFFECTIVE_LIMIT": "MONTH_START_LIMIT"}))
main = one[one["POINT_KIND"].ne("MONTH_START")].merge(first, on=["CARD_ID", "MONTH"], how="left", validate="m:1")
main["LIMIT_DIFFERS_FROM_MONTH_START"] = (main["MONTH_START_LIMIT"].notna() & main["EFFECTIVE_LIMIT"].notna()
                                           & main["MONTH_START_LIMIT"].ne(main["EFFECTIVE_LIMIT"]))
print("\n3. SELECTED CARD VALUES - OBSERVATION VS MONTH-START LIMIT")
print(main[main["OBS_DATE"].isin(focus_dates)][["POINT_KIND", "OBS_DATE", "CARD_ID", "OUTSTANDING", "EFFECTIVE_LIMIT", "MONTH_START_LIMIT", "LIMIT_DIFFERS_FROM_MONTH_START", "EXCLUDED_EUR_1"]].sort_values(["OBS_DATE", "CARD_ID"]).to_string(index=False))

valid = main[~main["EXCLUDED_EUR_1"].fillna(False)].copy()
valid["HAS_BALANCE"] = valid["OUTSTANDING"].notna()
summary = (valid.groupby(["POINT_KIND", "OBS_DATE"], as_index=False)
           .agg(CARDS=("CARD_ID", "nunique"), CARDS_WITH_BALANCE=("HAS_BALANCE", "sum"),
                BALANCE_EUR=("OUTSTANDING_ABS", lambda s: s.sum(min_count=1)),
                LIMIT_EUR=("EFFECTIVE_LIMIT", lambda s: s.sum(min_count=1))))
summary["UTILISATION_PCT"] = np.where(summary["LIMIT_EUR"].gt(0) & summary["BALANCE_EUR"].notna(),
                                      100 * summary["BALANCE_EUR"] / summary["LIMIT_EUR"], np.nan)
expected = pd.DataFrame(points, columns=["OBS_DATE", "POINT_KIND"])
expected = expected[expected["POINT_KIND"].ne("MONTH_START")].copy()
expected["OBS_DATE"] = pd.to_datetime(expected["OBS_DATE"], utc=True).dt.tz_localize(None).dt.normalize()
summary = expected.merge(summary, on=["POINT_KIND", "OBS_DATE"], how="left", validate="1:1")
summary = summary.sort_values(["OBS_DATE", "POINT_KIND"])
summary.loc[summary["CARDS_WITH_BALANCE"].eq(0), ["BALANCE_EUR", "UTILISATION_PCT"]] = np.nan
print("\n4. ENTITY-LEVEL VALUES: EXACT DATES, MONTH ENDS AND DAILY COMPARISON")
print(summary.to_string(index=False))

print("\n5. FRANK RECONCILIATION (amounts supplied in his email)")
compare = summary[summary["POINT_KIND"].eq("FRANK_DATE")].copy()
compare["FRANK_LIMIT_EUR"] = compare["OBS_DATE"].dt.strftime("%Y-%m-%d").map({k: v[0] for k, v in FRANK_VALUES.items()})
compare["FRANK_BALANCE_EUR"] = compare["OBS_DATE"].dt.strftime("%Y-%m-%d").map({k: v[1] for k, v in FRANK_VALUES.items()})
compare["FRANK_UTILISATION_PCT"] = 100 * compare["FRANK_BALANCE_EUR"] / compare["FRANK_LIMIT_EUR"]
compare["BALANCE_DIFF_EUR"] = compare["BALANCE_EUR"] - compare["FRANK_BALANCE_EUR"]
compare["LIMIT_DIFF_EUR"] = compare["LIMIT_EUR"] - compare["FRANK_LIMIT_EUR"]
print(compare[["OBS_DATE", "CARDS", "BALANCE_EUR", "FRANK_BALANCE_EUR", "BALANCE_DIFF_EUR", "LIMIT_EUR", "FRANK_LIMIT_EUR", "LIMIT_DIFF_EUR", "UTILISATION_PCT", "FRANK_UTILISATION_PCT"]].to_string(index=False))

print("\n6. ZERO VS MISSING AND MONTH-END RECONCILIATION")
recent = main[main["POINT_KIND"].eq("MONTH_END") & main["OBS_DATE"].isin(month_ends[-3:])]
recent = recent.copy()
recent["BALANCE_STATUS"] = np.select([recent["OUTSTANDING"].isna(), recent["OUTSTANDING"].eq(0)],
                                     ["MISSING", "SOURCE_ZERO"], default="NONZERO")
print(recent[["OBS_DATE", "CARD_ID", "OUTSTANDING", "EFFECTIVE_LIMIT", "BALANCE_STATUS"]].to_string(index=False) if not recent.empty else "No recent month-end card rows found")

print("\n7. DIAGNOSTIC EWS METRICS - NOT HISTORICAL RUN REPRODUCTION")
month_summary = summary[summary["POINT_KIND"].eq("MONTH_END")].sort_values("OBS_DATE")
curr = summary[summary["POINT_KIND"].eq("CURRENT")]["UTILISATION_PCT"].iloc[0]
prev = summary[summary["POINT_KIND"].eq("PREVIOUS_DAY")]["UTILISATION_PCT"].iloc[0]
avg12 = month_summary["UTILISATION_PCT"].mean()
max3 = month_summary.tail(3)["UTILISATION_PCT"].max()
print("Current %:", curr, "| Previous business day %:", prev)
print("12m average %:", avg12, "| Last 3m maximum %:", max3)
print("12m months with observation:", month_summary["UTILISATION_PCT"].notna().sum(), "/ 12")
print("3m months with observation:", month_summary.tail(3)["UTILISATION_PCT"].notna().sum(), "/ 3")
print("Current > previous:", bool(curr > prev) if pd.notna(curr) and pd.notna(prev) else "UNKNOWN")
print("Current > 95% and delta vs 12m > 95 percentage points:", bool(curr > 95 and curr - avg12 > 95) if pd.notna(curr) and pd.notna(avg12) else "UNKNOWN")
print("Current > 90% and relative increase vs 3m max > 100%:", bool(curr > 90 and max3 > 0 and (curr - max3) / max3 > 1) if pd.notna(curr) and pd.notna(max3) else "UNKNOWN")
print("\nIMPORTANT: Source balance is EUR, utilisation is a percentage.")
print("Missing historic observations must not be interpreted as actual 0% utilisation.")
print("Any multiple source candidates require verification against original EWS selection logic.")
print("Frank's dates and month-end snapshots are not interchangeable.")
print("Only reads were executed. No datasets or recipes were changed.")


print("\n" + "=" * 105)
print("8. STORED CC HISTORY - OBSERVED EWS VALUES")
print("=" * 105)
history_columns = [
    "ENTITY_ID", "MAP_DTE", "cc_outstanding_amount", "cc_credit_limit",
    "cc_limit_utilization", "cc_average_utilization_12m",
    "cc_max_utilization_3m", "cc_diff_limit_util_12m",
    "cc_diff_limit_util_3m_max",
    "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION",
    "CC_UTILIZATION_DAILY_CHANGE", "CC_UTILIZATION_DAILY_WORSENING"
]
hist_ds = dataiku.Dataset("CC_Utilisation_History")
hist_schema = [c["name"] for c in hist_ds.read_schema()]
hist_cols = [c for c in history_columns if c in hist_schema]
if "ENTITY_ID" not in hist_cols or "MAP_DTE" not in hist_cols:
    raise ValueError("CC_Utilisation_History lacks ENTITY_ID or MAP_DTE; inspect the schema.")
hist = hist_ds.get_dataframe(columns=hist_cols)
hist = hist[
    hist["ENTITY_ID"].astype(str).str.strip()
    .str.replace(r"\.0$", "", regex=True).eq(TARGET_ENTITY_ID)
].copy()
hist["MAP_DTE"] = pd.to_datetime(
    hist["MAP_DTE"], errors="coerce", utc=True
).dt.tz_localize(None).dt.normalize()
hist = hist.sort_values("MAP_DTE")
print("Stored history records for entity:", len(hist))
print(hist.to_string(index=False) if not hist.empty else "NONE")

print("\n9. EWS HISTORY VS SOURCE ON EXACT STORED DATES")
src_month = summary[summary["POINT_KIND"].eq("MONTH_END")].copy()
src_month = src_month[["OBS_DATE", "BALANCE_EUR", "LIMIT_EUR", "UTILISATION_PCT"]]
src_month = src_month.rename(columns={
    "BALANCE_EUR": "SOURCE_BALANCE_EUR",
    "LIMIT_EUR": "SOURCE_LIMIT_EUR",
    "UTILISATION_PCT": "SOURCE_UTILISATION_PCT"
})
h_compare = hist.merge(
    src_month, left_on="MAP_DTE", right_on="OBS_DATE", how="left", validate="m:1"
)
for name in ["cc_outstanding_amount", "cc_credit_limit", "cc_limit_utilization"]:
    if name in h_compare:
        h_compare[name] = pd.to_numeric(h_compare[name], errors="coerce")
if "cc_limit_utilization" in h_compare:
    h_compare["EWS_UTILISATION_PCT"] = 100 * h_compare["cc_limit_utilization"]
    h_compare["DIFF_UTILISATION_PP"] = h_compare["EWS_UTILISATION_PCT"] - h_compare["SOURCE_UTILISATION_PCT"]
for k, source in [
    ("DIFF_BALANCE_EUR", "cc_outstanding_amount"),
    ("DIFF_LIMIT_EUR", "cc_credit_limit")
]:
    if source in h_compare:
        h_compare[k] = h_compare[source] - h_compare[
            "SOURCE_BALANCE_EUR" if k == "DIFF_BALANCE_EUR" else "SOURCE_LIMIT_EUR"
        ]
compare_cols = [
    "MAP_DTE", "cc_outstanding_amount", "SOURCE_BALANCE_EUR", "DIFF_BALANCE_EUR",
    "cc_credit_limit", "SOURCE_LIMIT_EUR", "DIFF_LIMIT_EUR",
    "EWS_UTILISATION_PCT", "SOURCE_UTILISATION_PCT", "DIFF_UTILISATION_PP"
]
print(h_compare[[c for c in compare_cols if c in h_compare.columns]].to_string(index=False)
      if not h_compare.empty else "No records")

print("\n10. CURRENT 16 OCTOBER - STORED EWS VS SOURCE")
current_hist = hist[hist["MAP_DTE"].eq(alert_date)]
current_source = summary[summary["POINT_KIND"].eq("CURRENT")]
if len(current_hist) == 1 and len(current_source) == 1:
    h = current_hist.iloc[0]
    s = current_source.iloc[0]
    print("EWS stored balance:", h.get("cc_outstanding_amount"))
    print("Source diagnostic balance:", s["BALANCE_EUR"])
    print("EWS stored total limit:", h.get("cc_credit_limit"))
    print("Source diagnostic total limit (including card with missing balance):", s["LIMIT_EUR"])
    print("EWS stored utilisation (%):",
          pd.to_numeric(h.get("cc_limit_utilization"), errors="coerce") * 100)
    print("Source diagnostic utilisation (%):", s["UTILISATION_PCT"])
    print("EWS previous-day utilisation (stored):",
          h.get("CC_PREVIOUS_BUSINESS_DAY_UTILIZATION"))
    print("EWS daily worsening (stored):", h.get("CC_UTILIZATION_DAILY_WORSENING"))
    print("NOTE: EWS and this source diagnostic may have different card/limit aggregation rules.")
else:
    print("Cannot uniquely match a stored EWS record for 16 October.")

print("\n11. AVAILABLE INDICATORS - DATE SAFETY AND ACTUAL FLAGS")
flag_cols = [
    "OTHER_CC_UTILIZATION_ABOVE_100",
    "OTHER_CC_ABS_CHANGE_12M_ABOVE_95_CURRENT_ABOVE_95",
    "OTHER_CC_REL_CHANGE_3M_MAX_ABOVE_100_CURRENT_ABOVE_90",
]
for dname in ["Available_Indicators_Current", "Available_Indicators_Previous"]:
    ds = dataiku.Dataset(dname)
    schema = [c["name"] for c in ds.read_schema()]
    fields = [c for c in ["ENTITY_ID", "MAP_DTE"] + flag_cols if c in schema]
    if "ENTITY_ID" not in fields or "MAP_DTE" not in fields:
        print(dname, "missing date/entity columns")
        continue
    x = ds.get_dataframe(columns=fields)
    x = x[x["ENTITY_ID"].astype(str).str.strip()
          .str.replace(r"\.0$", "", regex=True).eq(TARGET_ENTITY_ID)].copy()
    x["MAP_DTE"] = pd.to_datetime(
        x["MAP_DTE"], errors="coerce", utc=True
    ).dt.tz_localize(None).dt.normalize()
    print("\n", dname, "rows:", len(x))
    print(x.to_string(index=False) if not x.empty else "No matching row")
    expected_date = alert_date if dname.endswith("Current") else previous_date
    print("TARGET DATE VALID:",
          bool(len(x) == 1 and x["MAP_DTE"].iloc[0] == expected_date))

print("\n12. AUDIT CONCLUSIONS")
print("Frank's dates use exact observation dates; EWS monthly history uses month-end dates.")
print("Source NULL outstanding and stored 0 utilisation are not equivalent without proving the transformation.")
print("The blocked card is included by this source diagnostic; EWS's precise limit aggregation remains to verify.")
print("Stored history can be compared to Oracle here, but this does not recreate archived indicator flags.")
print("Any Current dataset dated 19 October is not evidence of the 16 October historical indicator flag.")
print("No SQL UPDATE/INSERT/DELETE and no Dataiku dataset write executed.")
