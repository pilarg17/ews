# -*- coding: utf-8 -*-
"""Retail EWS | Alert-count reconciliation and rating persistence reporting.

Run as ONE notebook cell in the SAME kernel after the existing 5/10/15-day simulation.
Reads EWS_Daily_Alerts once; does not query Oracle history or modify any dataset.
No claim is made about pre-bundling alert counts or false positives.
"""
import html
import pandas as pd
import dataiku
from IPython.display import display, HTML

DATASET = "EWS_Daily_Alerts"
TARGET_DATE = pd.Timestamp("2026-11-06")
FIRST_DAY = pd.Timestamp("2026-11-02")
CLIENT_REPORTED_PEAK = 67
LAGS = (5, 10, 15)
RATING_PREFIXES = ("RRE_CLIENT_", "OTHER_CLIENT_", "RRE_GEC_HEAD_", "OTHER_GEC_HEAD_")


def heading(title):
    print("\n" + "=" * 98 + "\n" + title + "\n" + "=" * 98, flush=True)


def show(df):
    out = df.copy()
    css = """<style>
    table.ews-peak {border-collapse:collapse;width:100%;font:12px Arial,sans-serif;margin:8px 0 14px}
    .ews-peak th{background:#1b3954;color:#fff;text-align:left;padding:9px;border:0}
    .ews-peak td{border-bottom:1px solid #dce3ea;padding:7px 9px}
    .ews-peak tr:nth-child(even) td{background:#f3f6fa}
    </style>"""
    def printable(v):
        if isinstance(v, (list, dict, tuple)):
            return str(v)
        if v is None or pd.isna(v):
            return ""
        if isinstance(v, float):
            return f"{v:,.2f}"
        if isinstance(v, (int,)):
            return f"{v:,}"
        return str(v)
    th = "".join("<th>%s</th>" % html.escape(str(c)) for c in out.columns)
    trs = []
    for values in out.itertuples(index=False, name=None):
        trs.append("<tr>"+"".join("<td>%s</td>" % html.escape(printable(v)) for v in values)+"</tr>")
    display(HTML(css+'<table class="ews-peak"><thead><tr>'+th+'</tr></thead><tbody>'+''.join(trs)+'</tbody></table>'))


def parse_dates(values):
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    raw = values.astype("string").str.strip()
    iso = pd.to_datetime(raw.str.slice(0, 10), errors="coerce", format="%Y-%m-%d")
    compact = pd.to_datetime(raw, errors="coerce", format="%Y%m%d")
    return iso.fillna(compact).dt.normalize()


def norm_ids(values):
    return values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)


def codes(value):
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        return []
    return sorted(set(x.strip() for x in str(value).split("|") if x.strip()))


heading("RETAIL EWS | RATING PERSISTENCE AND REPORTED PEAK RECONCILIATION")
print("Original alert date         : 2026-11-06")
print("Rating trigger date        : 2026-11-02")
print("Client-reported peak       : 67 (original counting basis unverified)")
print("Source                     : EWS_Daily_Alerts")
print("Execution                  : READ ONLY; no Oracle rating-history queries")

if "result" not in globals() or "cohort" not in globals():
    raise RuntimeError("Existing rating lag simulation variables 'result' and 'cohort' are required")
scenarios = result.copy()
original = cohort.copy()
needed = {"ENTITY_ID"}
for lag in LAGS:
    needed.update({f"ANY_RATING_{lag}D", f"ORIGINAL_EWI_{lag}D"})
if needed - set(scenarios.columns):
    raise RuntimeError("Missing rating simulation fields: " + ", ".join(sorted(needed - set(scenarios.columns))))
scenarios["ENTITY_ID"] = norm_ids(scenarios["ENTITY_ID"])
if scenarios["ENTITY_ID"].isna().any() or scenarios["ENTITY_ID"].duplicated().any():
    raise RuntimeError("Duplicate or missing scenario client identifiers")
for lag in LAGS:
    for col in (f"ANY_RATING_{lag}D", f"ORIGINAL_EWI_{lag}D"):
        scenarios[col] = pd.to_numeric(scenarios[col], errors="coerce")
        if not scenarios[col].isin([0, 1]).all():
            raise RuntimeError("Non-binary rating scenario field: " + col)

# Retrieve the stored daily export to inspect other dates and reconcile count definitions.
daily = dataiku.Dataset(DATASET).get_dataframe()
required = {"ALERT_DATE", "ENTITY_ID", "TRIGGERED_INDICATOR_CODES"}
if required - set(daily.columns):
    raise RuntimeError("Missing EWS_Daily_Alerts fields: " + ", ".join(sorted(required - set(daily.columns))))
daily = daily[["ALERT_DATE", "ENTITY_ID", "TRIGGERED_INDICATOR_CODES"]].copy()
daily["ALERT_DATE"] = parse_dates(daily["ALERT_DATE"])
daily["ENTITY_ID"] = norm_ids(daily["ENTITY_ID"])
if daily["ALERT_DATE"].isna().any() or daily["ENTITY_ID"].isna().any():
    raise RuntimeError("Invalid source alert dates or client identifiers")
daily["CODES"] = daily["TRIGGERED_INDICATOR_CODES"].map(codes)
current = daily.loc[daily["ALERT_DATE"].eq(TARGET_DATE)].copy()
if current.empty:
    raise RuntimeError("No stored alerts dated 2026-11-06")

# No counting of repeated same client/EWI codes.
def count_day(frame):
    all_e = frame[["ENTITY_ID", "CODES"]].explode("CODES").rename(columns={"CODES": "CODE"})
    all_e = all_e.dropna(subset=["CODE"]).drop_duplicates(["ENTITY_ID", "CODE"])
    is_rating = all_e["CODE"].astype(str).str.startswith(RATING_PREFIXES)
    r = all_e.loc[is_rating]
    n = all_e.loc[~is_rating]
    rids, nids = set(r["ENTITY_ID"]), set(n["ENTITY_ID"])
    return {
        "SOURCE_ROWS": len(frame),
        "ALL_CLIENTS": frame["ENTITY_ID"].nunique(),
        "ALL_CLIENT_EWI": len(all_e),
        "RATING_CLIENTS": len(rids),
        "RATING_CLIENT_EWI": len(r),
        "NONRATING_CLIENTS": len(nids),
        "NONRATING_CLIENT_EWI": len(n),
        "BOTH_CATEGORY_CLIENTS": len(rids & nids),
        "UNCLASSIFIED_SOURCE_CLIENTS": len(set(frame["ENTITY_ID"]) - (rids | nids)),
    }, all_e, r, n

m, all_pairs, rating_pairs, nonrating_pairs = count_day(current)
heading("01 | 6 NOVEMBER | COUNTS BY UNIT")
show(pd.DataFrame([
    ("Source rows", m["SOURCE_ROWS"], "Stored entity-level rows"),
    ("Unique clients, any indicator", m["ALL_CLIENTS"], "Distinct ENTITY_ID"),
    ("Unique client/indicator pairs, any category", m["ALL_CLIENT_EWI"], "Post-export, de-duplicated"),
    ("Unique clients with rating indicator", m["RATING_CLIENTS"], "Distinct ENTITY_ID with rating EWI"),
    ("Unique client/rating indicator pairs", m["RATING_CLIENT_EWI"], "Post-export rating EWI count"),
    ("Unique clients with non-rating indicator", m["NONRATING_CLIENTS"], "Distinct ENTITY_ID with non-rating EWI"),
    ("Unique client/non-rating indicator pairs", m["NONRATING_CLIENT_EWI"], "Post-export other EWI count"),
    ("Clients with rating AND non-rating EWI", m["BOTH_CATEGORY_CLIENTS"], "Overlap in two categories"),
    ("Clients with no readable indicator code", m["UNCLASSIFIED_SOURCE_CLIENTS"], "Data quality exception"),
    ("Reported peak", CLIENT_REPORTED_PEAK, "Client statement, counting unit unconfirmed"),
], columns=["MEASURE", "VALUE", "DEFINITION"]))

heading("02 | 6 NOVEMBER | POST-EXPORT INDICATOR BREAKDOWN")
per_indicator = all_pairs.groupby("CODE")["ENTITY_ID"].nunique().sort_values(ascending=False).reset_index(name="CLIENTS")
per_indicator.insert(0, "TYPE", per_indicator["CODE"].map(lambda x: "RATING" if str(x).startswith(RATING_PREFIXES) else "NON-RATING"))
show(per_indicator)

heading("03 | REPORTED PEAK 67 | SOURCE DATE SCAN")
# Scan stored dates only. The daily export cannot reconstruct pre-bundling signals.
window = daily.loc[daily["ALERT_DATE"].between(pd.Timestamp("2026-10-26"), pd.Timestamp("2026-11-20"))].copy()
scan_rows = []
for date, part in window.groupby("ALERT_DATE", sort=True):
    d, _, _, _ = count_day(part)
    scan_rows.append({"DATE": date.strftime("%Y-%m-%d"), **d})
timeline = pd.DataFrame(scan_rows)
if timeline.empty:
    print("No stored dates in 26 Oct to 20 Nov window")
else:
    timeline = timeline[["DATE", "SOURCE_ROWS", "ALL_CLIENTS", "ALL_CLIENT_EWI", "RATING_CLIENTS", "RATING_CLIENT_EWI", "NONRATING_CLIENTS", "NONRATING_CLIENT_EWI"]]
    show(timeline)
    matching = timeline.loc[(timeline.iloc[:, 1:] == CLIENT_REPORTED_PEAK).any(axis=1)]
    print("\nDates with at least one metric equal to 67:")
    if matching.empty:
        print("NONE in the stored export window")
    else:
        show(matching)
        print("A matching number on another date or metric is NOT proof of identical client reporting scope.")

heading("04 | RATING PERSISTENCE | ORIGINAL 6 NOV RATING COHORT")
base = len(scenarios)
comparison = []
for lag in LAGS:
    retained = int(scenarios[f"ANY_RATING_{lag}D"].sum())
    original_ewi = int(scenarios[f"ORIGINAL_EWI_{lag}D"].sum())
    comparison.append({
        "BUSINESS_DAYS": lag,
        "CONFIRMATION_DATE": pd.bdate_range(FIRST_DAY, periods=lag)[-1].strftime("%Y-%m-%d"),
        "ORIGINAL_RATING_CLIENTS": base,
        "RETAINED_ANY_RATING": retained,
        "NOT_RETAINED": base - retained,
        "REDUCTION_PCT": round(100 * (base-retained) / base, 2) if base else None,
        "RETAINED_ORIGINAL_EWI": original_ewi,
        "OTHER_RATING_ONLY": retained-original_ewi,
    })
comparison = pd.DataFrame(comparison)
show(comparison)
print("Note: fixed original 6-Nov cohort, NOT total alert counts generated on future dates.")

heading("05 | EXPLICIT OVERLAY | NON-RATING SIGNALS HELD CONSTANT")
nonrating_ids = set(nonrating_pairs["ENTITY_ID"])
base_all = set(current["ENTITY_ID"])
overlay = []
for lag in LAGS:
    kept = set(scenarios.loc[scenarios[f"ANY_RATING_{lag}D"].eq(1), "ENTITY_ID"])
    saved = nonrating_ids | kept
    overlay.append({"BUSINESS_DAYS": lag, "ORIGINAL_ALL_CLIENTS": len(base_all),
                    "HYPOTHETICAL_RETAINED": len(saved), "HYPOTHETICAL_NOT_RETAINED": len(base_all-saved)})
show(pd.DataFrame(overlay))
print("Non-rating indicators are frozen from 6 Nov. This is NOT a future production alert simulation.")

heading("06 | RECONCILIATION AND CONSISTENCY QA")
source_rating_ids = set(rating_pairs["ENTITY_ID"])
checks = [
    ("Source 6-Nov rating clients equal simulated cohort", source_rating_ids == set(scenarios["ENTITY_ID"])),
    ("5-day alert and original EWI reproduced", int(scenarios["ANY_RATING_5D"].sum()) == base and int(scenarios["ORIGINAL_EWI_5D"].sum()) == base),
    ("10/15-day persistence monotonic", (scenarios["ANY_RATING_15D"] <= scenarios["ANY_RATING_10D"]).all() and (scenarios["ANY_RATING_10D"] <= scenarios["ANY_RATING_5D"]).all()),
    ("Original EWI subset of all rating EWIs", all((scenarios[f"ORIGINAL_EWI_{lag}D"] <= scenarios[f"ANY_RATING_{lag}D"]).all() for lag in LAGS)),
    ("All source clients accounted for by indicator category", m["UNCLASSIFIED_SOURCE_CLIENTS"] == 0),
    ("Unique-client union is arithmetically consistent", m["RATING_CLIENTS"]+m["NONRATING_CLIENTS"]-m["BOTH_CATEGORY_CLIENTS"] == m["ALL_CLIENTS"]),
    ("Indicator pair totals equal rating plus non-rating", m["RATING_CLIENT_EWI"]+m["NONRATING_CLIENT_EWI"] == m["ALL_CLIENT_EWI"]),
]
checks_df = pd.DataFrame(checks, columns=["CHECK", "PASSED"])
checks_df["STATUS"] = checks_df["PASSED"].map({True: "PASS", False: "REVIEW"})
show(checks_df[["CHECK", "STATUS"]])

heading("07 | AUDIT CONCLUSION")
qa_ok = bool(checks_df["PASSED"].all())
print("Technical 5/10/15 sensitivity QA :", "PASS" if qa_ok else "REVIEW")
print("6-Nov post-export source rows     :", m["SOURCE_ROWS"])
print("6-Nov exported rating clients    :", m["RATING_CLIENTS"])
print("6-Nov exported rating pairs      :", m["RATING_CLIENT_EWI"])
print("Client-reported peak             :", CLIENT_REPORTED_PEAK)
if CLIENT_REPORTED_PEAK in (m["SOURCE_ROWS"], m["ALL_CLIENTS"], m["ALL_CLIENT_EWI"], m["RATING_CLIENTS"], m["RATING_CLIENT_EWI"]):
    print("Reported peak count status       : NUMERIC MATCH; IDENTICAL SCOPE NOT PROVEN")
else:
    print("Reported peak count status       : NOT RECONCILED FOR 6 NOV")
print("Required evidence                : Original client alert list or execution log with date, pre/post-bundling and counting unit")
print("False-positive impact            : NOT TESTED")
print("Live/production policy           : UNCHANGED")
print("Dataset, recipe or Flow writes   : NONE")

# Results remain in notebook memory only.
retail_6nov_count_audit = pd.DataFrame([m])
retail_67_date_scan = timeline
retail_5_10_15_comparison = comparison
retail_67_audit_qa = checks_df
