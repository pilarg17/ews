import os
import re
import pandas as pd
import numpy as np
import dataiku

ALERT_DATE = "2026-10-16"
CSV_PATH = ""  # Optional: full path to the original ews_ewi_breach_information CSV for 16 Oct
DATASET_NAME = "ews_ewi_breach_information"

FRANK_CURRENT_BALANCE = 1462.0
FRANK_CURRENT_LIMIT = 1500.0
BALANCE_TOLERANCE = 20.0
LIMIT_TOLERANCE = 0.01

def first_col(frame, options):
    names = {str(c).strip().upper(): c for c in frame.columns}
    return next((names[x] for x in options if x in names), None)

def to_number(x):
    if pd.isna(x):
        return np.nan
    t = str(x).strip().replace(" ", "")
    if "," in t and "." in t:
        t = t.replace(",", "") if t.rfind(".") > t.rfind(",") else t.replace(".", "").replace(",", ".")
    elif "," in t:
        t = t.replace(",", ".")
    return pd.to_numeric(t, errors="coerce")

print("=" * 108)
print("CREDIT CARD - 16 OCT EXPORT IDENTIFICATION - READ ONLY")
print("=" * 108)

if CSV_PATH:
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError("CSV_PATH does not exist: " + CSV_PATH)
    alerts = pd.read_csv(CSV_PATH, sep=None, engine="python", dtype=str)
    source_name = "Original CSV: " + CSV_PATH
else:
    ds = dataiku.Dataset(DATASET_NAME)
    alerts = ds.get_dataframe()
    source_name = "Dataiku dataset (may have been overwritten): " + DATASET_NAME

print("Source:", source_name)
print("Columns:", list(alerts.columns))
print("Rows:", len(alerts))

entity_col = first_col(alerts, ["ENTITY_ID", "_ENTITY", "CUSTOMER_ID"])
breach_col = first_col(alerts, ["BREACH_ID"])
date_col = first_col(alerts, ["ALERT_DATE", "BREACH_DATE", "MAP_DTE", "REFERENCE_DATE", "EWI_DATE"])
name_col = first_col(alerts, ["EWI_NAME", "EWI_DESCRIPTION", "INDICATOR_NAME", "EWI_LABEL", "EWI", "INDICATOR"])
if not entity_col:
    raise ValueError("No ENTITY_ID field found. Check displayed column names.")
if not name_col:
    print("WARNING: No EWI name/description field recognised; inspection will use all textual columns.")

date_evidence = pd.Series(False, index=alerts.index)
if date_col:
    date_evidence |= pd.to_datetime(alerts[date_col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d").eq(ALERT_DATE)
if breach_col:
    date_evidence |= alerts[breach_col].astype(str).str.startswith("20261016_")

if not date_evidence.any():
    print("\nNO VERIFIED 16 OCT ROWS IN SELECTED EXPORT.")
    print("This dataset could have been overwritten. Set CSV_PATH to the archived 16 Oct CSV.")
    print("No 16 Oct attribution will be made.")
else:
    a = alerts.loc[date_evidence].copy()
    text_cols = [name_col] if name_col else [
        c for c in a.columns
        if a[c].dtype == object and
        any(k in c.upper() for k in ["EWI", "INDICATOR", "DESCRIPTION", "LABEL", "TYPE"])
    ]
    matches = pd.Series(False, index=a.index)
    for c in text_cols:
        if c:
            matches |= a[c].astype(str).str.contains(
                r"credit[\s_-]*card|card[\s_-]*utili[sz]ation|(^|_)CC(_|$)",
                flags=re.IGNORECASE, regex=True, na=False
            )
    if not matches.any():
        print("\nNo card alerts identified from text names; showing distinct descriptions:")
        for c in text_cols:
            print(c, ":", a[c].dropna().astype(str).unique()[:40].tolist())
    else:
        cc = a.loc[matches].copy()
        cc["_ENTITY_KEY"] = cc[entity_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        cols = [c for c in cc.columns if c != "_ENTITY_KEY"]
        print("\n1. ALL CARD ALERT ROWS FROM VERIFIED 16 OCT EXPORT:", len(cc))
        print(cc[cols].to_string(index=False, max_colwidth=130))
        print("\n2. DISTINCT ENTITIES / BREACH IDS / INDICATORS")
        shortlist_cols = [c for c in [entity_col, breach_col, date_col, name_col] if c]
        shortlist_cols = list(dict.fromkeys(shortlist_cols))
        print(cc[shortlist_cols].drop_duplicates().to_string(index=False))

        print("\n3. RECONCILE EACH ALERT ENTITY WITH STORED CC HISTORY ON 16 OCT")
        hist_ds = dataiku.Dataset("CC_Utilisation_History")
        hist_fields = [x["name"] for x in hist_ds.read_schema()]
        wanted = ["ENTITY_ID", "MAP_DTE", "cc_outstanding_amount", "cc_credit_limit",
                  "cc_limit_utilization", "CC_PREVIOUS_BUSINESS_DAY_UTILIZATION"]
        available = [c for c in wanted if c in hist_fields]
        if "ENTITY_ID" not in available or "MAP_DTE" not in available:
            print("History schema missing identifier/date; cannot reconcile.")
        else:
            hist = hist_ds.get_dataframe(columns=available)
            hist["_ENTITY_KEY"] = hist["ENTITY_ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            hist = hist[hist["_ENTITY_KEY"].isin(cc["_ENTITY_KEY"].unique())].copy()
            hist["MAP_DTE"] = pd.to_datetime(
                hist["MAP_DTE"], errors="coerce", utc=True
            ).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
            hist = hist[hist["MAP_DTE"].eq(ALERT_DATE)].copy()
            for c in ["cc_outstanding_amount", "cc_credit_limit", "cc_limit_utilization"]:
                if c in hist.columns:
                    hist[c] = pd.to_numeric(hist[c], errors="coerce")
            if "cc_outstanding_amount" in hist.columns and "cc_credit_limit" in hist.columns:
                hist["BALANCE_MATCH"] = hist["cc_outstanding_amount"].sub(FRANK_CURRENT_BALANCE).abs().le(BALANCE_TOLERANCE)
                hist["LIMIT_MATCH"] = hist["cc_credit_limit"].sub(FRANK_CURRENT_LIMIT).abs().le(LIMIT_TOLERANCE)
                hist["BOTH_MATCH"] = hist["BALANCE_MATCH"] & hist["LIMIT_MATCH"]
                hist["BALANCE_DIFF_EUR"] = hist["cc_outstanding_amount"] - FRANK_CURRENT_BALANCE
                hist["LIMIT_DIFF_EUR"] = hist["cc_credit_limit"] - FRANK_CURRENT_LIMIT
            if "cc_limit_utilization" in hist:
                hist["UTILISATION_PCT"] = hist["cc_limit_utilization"] * 100
            out = cc[shortlist_cols + ["_ENTITY_KEY"]].drop_duplicates().merge(
                hist.drop(columns=["ENTITY_ID"], errors="ignore"), on="_ENTITY_KEY",
                how="left", validate="m:m")
            print(out.drop(columns="_ENTITY_KEY").to_string(index=False))
            candidates = out[out.get("BOTH_MATCH", pd.Series(False, index=out.index)).eq(True)] if "BOTH_MATCH" in out else pd.DataFrame()
            print("\n4. BOTH FRANK AMOUNTS MATCHED:", len(candidates), "joined alert row(s)")
            if not candidates.empty:
                print(candidates[shortlist_cols].drop_duplicates().to_string(index=False))
                print("MATCH IS A LEAD, NOT DEFINITIVE PROOF OF WHICH CARD FRANK CHECKED.")
            else:
                print("No joint match in available archived history. Do not identify Frank's card from this alone.")

print("\nREAD ONLY. No Flow/PROD changes. For final identification use archived 16 Oct export.")
