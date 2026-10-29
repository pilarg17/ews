# -*- coding: utf-8 -*-

import dataiku
import pandas as pd

# Configuration
REFERENCE_DATASET = "Implementation_Reference_Date"
CURRENT_DATASET = "Available_Indicators_Current"
PREVIOUS_DATASET = "Available_Indicators_Previous"
ALERT_DATASET = "EWS_Daily_Alerts"
BREACH_DATASET = "ews_ewi_breach_information"

CLIENT_RATING_CODES = [
    "RRE_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M",
    "RRE_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M",
    "RRE_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M",
    "RRE_CLIENT_WORSE_B_PLUS_DOWNGRADE_AT_LEAST_2_VS_12M",
    "RRE_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_1_VS_12M",
    "OTHER_CLIENT_DOWNGRADE_AT_LEAST_6_VS_12M",
    "OTHER_CLIENT_WORSE_BBB_MINUS_DOWNGRADE_AT_LEAST_5_VS_12M",
    "OTHER_CLIENT_WORSE_BB_MINUS_DOWNGRADE_AT_LEAST_4_VS_12M",
    "OTHER_CLIENT_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "OTHER_CLIENT_WORSE_B_DOWNGRADE_AT_LEAST_3_VS_12M",
]

GEC_RATING_CODES = [
    "RRE_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
    "RRE_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "RRE_GEC_HEAD_DOWNGRADE_MORE_THAN_7_VS_12M",
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B_PLUS",
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_1_CLIENT_DOWNGRADE_AT_LEAST_6",
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5",
    "RRE_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3",
    "OTHER_GEC_HEAD_WORSE_B_DOWNGRADE_AT_LEAST_4_VS_12M",
    "OTHER_GEC_HEAD_WORSE_B_MINUS_DOWNGRADE_AT_LEAST_1_VS_12M",
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_VS_12M",
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_WORSE_B",
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_6_CLIENT_DOWNGRADE_AT_LEAST_3",
    "OTHER_GEC_HEAD_DOWNGRADE_AT_LEAST_3_CLIENT_DOWNGRADE_AT_LEAST_5",
]

RATING_CODES = CLIENT_RATING_CODES + GEC_RATING_CODES
RATING_CODE_SET = set(RATING_CODES)
BASE_COLUMNS = [code + "_BASE" for code in RATING_CODES]
SOURCE_FIELDS = [
    "CLIENT_RATING_NOTCH",
    "CLIENT_DOWNGRADE_12M",
    "GEC_HEAD_RATING_NOTCH",
    "GEC_HEAD_DOWNGRADE_12M",
    "GEC_NET_EXPOSURE",
    "HEAD_GEC_PARENT_ID",
    "CLIENT_RATING_DAILY_WORSENING",
]


def columns_in_dataset(name):
    schema = dataiku.Dataset(name).read_schema()
    return {column["name"] for column in schema}


def read_columns(name, wanted, required):
    available = columns_in_dataset(name)
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError("{} is missing columns: {}".format(name, ", ".join(missing)))
    present = [col for col in dict.fromkeys(wanted) if col in available]
    result = dataiku.Dataset(name).get_dataframe(columns=present, parse_dates=False)
    return result, available


def normalize_ids(values):
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def date_values(values):
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def flag(values):
    return pd.to_numeric(values, errors="coerce").fillna(0).eq(1)


def count_active(frame, col):
    return int(flag(frame[col]).sum()) if col in frame.columns else None


def fmt(value):
    return "N/A" if value is None else "{:,}".format(int(value))


reference = dataiku.Dataset(REFERENCE_DATASET).get_dataframe()
if len(reference) != 1 or not {"REFERENCE_DATE", "PREVIOUS_DATE"}.issubset(reference.columns):
    raise ValueError("Implementation_Reference_Date must have one row and both date columns.")

reference_date = date_values(reference["REFERENCE_DATE"]).iloc[0]
previous_date = date_values(reference["PREVIOUS_DATE"]).iloc[0]
if pd.isna(reference_date) or pd.isna(previous_date):
    raise ValueError("Invalid reference or previous date.")

core = ["ENTITY_ID", "MAP_DTE", "RRE_FLAG", "OTHER_RETAIL_FLAG"]
needed = core + RATING_CODES + BASE_COLUMNS + SOURCE_FIELDS
current, current_schema = read_columns(CURRENT_DATASET, needed, core)
previous, previous_schema = read_columns(PREVIOUS_DATASET, core + RATING_CODES, core)

missing_current = sorted(RATING_CODE_SET - current_schema)
missing_previous = sorted(RATING_CODE_SET - previous_schema)
if missing_current or missing_previous:
    raise ValueError(
        "Rating flag columns are missing. Current: {} / Previous: {}".format(
            ", ".join(missing_current) or "none",
            ", ".join(missing_previous) or "none",
        )
    )

for frame, date in [(current, reference_date), (previous, previous_date)]:
    frame["ENTITY_ID"] = normalize_ids(frame["ENTITY_ID"])
    frame["MAP_DTE"] = date_values(frame["MAP_DTE"])
    if frame["MAP_DTE"].notna().sum() > 0:
        observed = sorted(frame["MAP_DTE"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
        if observed != [date.strftime("%Y-%m-%d")]:
            raise ValueError("Snapshot date mismatch: expected {}, found {}".format(date.date(), observed[:8]))
    if frame["ENTITY_ID"].isna().any() or frame["ENTITY_ID"].duplicated().any():
        raise ValueError("Missing or duplicated ENTITY_ID in indicator snapshot.")

if current.empty or previous.empty:
    raise ValueError("An indicator snapshot is empty.")

current = current.set_index("ENTITY_ID", drop=False)
previous = previous.set_index("ENTITY_ID", drop=False)
previous_exists = pd.Series(current.index.isin(previous.index), index=current.index)
previous_aligned = previous.reindex(current.index)

print("=" * 102)
print("RETAIL EWS - RATING QA (READ ONLY)")
print("=" * 102)
print("Reference date             :", reference_date.strftime("%Y-%m-%d"))
print("Previous date              :", previous_date.strftime("%Y-%m-%d"))
print("Current entities           :", fmt(len(current)))
print("Previous entities          :", fmt(len(previous)))
print("Current missing in previous:", fmt((~previous_exists).sum()))
print("Rating persistence         : 5 business days (checked upstream in indicator recipe)")
print("Rating Day 1 / Day 5       :", pd.bdate_range(end=reference_date, periods=5)[0].strftime("%Y-%m-%d"), "/", reference_date.strftime("%Y-%m-%d"))

scope = flag(current["RRE_FLAG"]) | flag(current["OTHER_RETAIL_FLAG"])
print("Retail scope entities      :", fmt(scope.sum()))
print()
print("RATING SOURCE COVERAGE")
print("----------------------")
for source in SOURCE_FIELDS:
    if source not in current.columns:
        print("{}: column not available".format(source))
        continue
    values = current.loc[scope, source]
    valid = values.notna() & values.astype("string").str.strip().ne("")
    print("{:<32} {:>9,} / {:>9,} ({:5.1f}%)".format(
        source, int(valid.sum()), int(scope.sum()),
        100 * float(valid.mean()) if len(valid) else 0,
    ))

alerts, alert_schema = read_columns(
    ALERT_DATASET,
    ["ENTITY_ID", "ALERT_DATE", "TRIGGERED_INDICATOR_CODES"],
    ["ENTITY_ID", "TRIGGERED_INDICATOR_CODES"],
)
if "ALERT_DATE" in alerts.columns and not alerts.empty:
    alerts["ALERT_DATE"] = date_values(alerts["ALERT_DATE"])
    observed = sorted(alerts["ALERT_DATE"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    if observed != [reference_date.strftime("%Y-%m-%d")]:
        raise ValueError("EWS_Daily_Alerts does not match the reference date: {}".format(observed[:8]))

selected_rows = []
for _, alert in alerts.iterrows():
    codes = [] if pd.isna(alert["TRIGGERED_INDICATOR_CODES"]) else [
        item.strip() for item in str(alert["TRIGGERED_INDICATOR_CODES"]).split("|") if item.strip()
    ]
    for code in set(codes):
        if code in RATING_CODE_SET:
            selected_rows.append((str(alert["ENTITY_ID"]).strip(), code))
selected_rating = pd.DataFrame(selected_rows, columns=["ENTITY_ID", "INDICATOR_CODE"])
selected_counts = selected_rating["INDICATOR_CODE"].value_counts().to_dict() if len(selected_rating) else {}

breach, breach_schema = read_columns(
    BREACH_DATASET,
    ["ENTITY_ID", "EWI_NAME", "ALERT_DATE"],
    ["ENTITY_ID", "EWI_NAME"],
)
if "ALERT_DATE" in breach.columns and not breach.empty:
    breach["ALERT_DATE"] = date_values(breach["ALERT_DATE"])
    observed = sorted(breach["ALERT_DATE"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    if observed != [reference_date.strftime("%Y-%m-%d")]:
        raise ValueError("Breach output does not match the reference date: {}".format(observed[:8]))

export_rating_mask = breach["EWI_NAME"].astype("string").str.contains(
    r"(?i)^(?:client rating|client downgrade|gec head rating|gec head downgrade)",
    regex=True, na=False,
)

rows = []
raw_new_entities = set()
for code in RATING_CODES:
    family = "CLIENT" if code in CLIENT_RATING_CODES else "GEC_HEAD"
    cur = flag(current[code])
    prev = flag(previous_aligned[code])
    prev_portfolio_code = "RRE_FLAG" if code.startswith("RRE_") else "OTHER_RETAIL_FLAG"
    valid_prev_portfolio = previous_exists & flag(previous_aligned[prev_portfolio_code])
    new = cur & valid_prev_portfolio & ~prev
    blocked = cur & ~valid_prev_portfolio
    ongoing = cur & valid_prev_portfolio & prev
    raw_new_entities.update(current.index[new].astype("string").tolist())
    rows.append({
        "FAMILY": family,
        "INDICATOR_CODE": code,
        "BASE_ACTIVE": count_active(current, code + "_BASE"),
        "CURRENT_ACTIVE": int(cur.sum()),
        "PREVIOUS_ACTIVE": int(flag(previous[code]).sum()),
        "ACTIVE_BOTH_DAYS": int(ongoing.sum()),
        "NEW_0_TO_1": int(new.sum()),
        "BLOCKED_NO_PREVIOUS_PORTFOLIO": int(blocked.sum()),
        "DAILY_SELECTED": int(selected_counts.get(code, 0)),
    })

summary = pd.DataFrame(rows)
family_summary = summary.groupby("FAMILY", as_index=False).agg({
    "CURRENT_ACTIVE": "sum",
    "PREVIOUS_ACTIVE": "sum",
    "ACTIVE_BOTH_DAYS": "sum",
    "NEW_0_TO_1": "sum",
    "BLOCKED_NO_PREVIOUS_PORTFOLIO": "sum",
    "DAILY_SELECTED": "sum",
})
if summary["BASE_ACTIVE"].notna().any():
    family_summary.insert(1, "BASE_ACTIVE", summary.groupby("FAMILY")["BASE_ACTIVE"].sum(min_count=1).to_numpy())

print()
print("RATING FUNNEL (INDICATOR COUNTS, NOT UNIQUE CLIENTS)")
print("----------------------------------------------------")
print(family_summary.to_string(index=False))
print()
print("RATING DETAIL")
print("-------------")
interesting = (
    summary[["CURRENT_ACTIVE", "PREVIOUS_ACTIVE", "NEW_0_TO_1", "DAILY_SELECTED"]].sum(axis=1).gt(0)
    | summary["BASE_ACTIVE"].fillna(0).gt(0)
)
if interesting.any():
    print(summary.loc[interesting].to_string(index=False))
else:
    print("No active rating rules in either snapshot (all 23 rating codes checked).")

raw_new_count = int(summary["NEW_0_TO_1"].sum())
selected_count = int(len(selected_rating))
selected_entities = int(selected_rating["ENTITY_ID"].nunique()) if selected_count else 0
exported_count = int(export_rating_mask.sum())

print()
print("FINAL RATING COUNTS")
print("-------------------")
print("Reconstructed 0->1 rating signals  :", fmt(raw_new_count))
print("Unique entities with new rating    :", fmt(len(raw_new_entities)))
print("Rating codes in EWS_Daily_Alerts   :", fmt(selected_count))
print("Unique clients in EWS_Daily_Alerts :", fmt(selected_entities))
print("Rating-looking EWI_NAME in export  :", fmt(exported_count))
print("Total EWS_Daily_Alerts clients     :", fmt(len(alerts)))
print("Total breach rows                  :", fmt(len(breach)))

print()
print("ASSESSMENT")
print("----------")
if raw_new_count > 0 and selected_count == 0:
    print("CHECK: Rating 0->1 transitions exist but none reached EWS_Daily_Alerts. Review alert generation/bundling.")
elif raw_new_count == 0 and selected_count > 0:
    print("CHECK: EWS_Daily_Alerts contains rating codes not reproduced by the Current/Previous transition check.")
elif raw_new_count == 0 and selected_count == 0:
    print("CONSISTENT: No rating transitions qualify for new daily alerts on this snapshot.")
    print("This does not independently validate the upstream 5-day persistence or source history.")
else:
    print("Rating signals are present. Check expected bundling and any output exclusions.")

if selected_count > 0 and exported_count == 0:
    print("CHECK: Rating codes reached Daily Alerts but no rating EWI names appear in the breach export.")
elif selected_count == 0 and exported_count > 0:
    print("CHECK: Breach export contains rating EWI names although no rating codes were selected.")

print("Review the indicator recipe log for: Day 1 triggers, matured Day 5, history coverage and missing history.")
print("No datasets were modified.")
