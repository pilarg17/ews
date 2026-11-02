# -*- coding: utf-8 -*-
"""
BIL EWS / Phase 2 / UAT load-test — ONE EXPORT / FOUR CSV
Run as ONE Python notebook cell in an isolated Dataiku DEV project.
READ ONLY for Dataiku datasets; never writes/overwrites EWS datasets or runs the Flow.

Reads the four existing final-output datasets as TEMPLATES. Uses original EWI names,
methodologies, enums, and column order from the bank's actual outputs. Does not
copy customer identities or exact source amounts into the synthetic delivery.
Creates 5,000 Retail + 5,000 Corporate synthetic breaches (incl. two named CIB
QA cases among Corporate), six historic rows and one entity/exposure per breach.

Fails closed if either Retail/Corporate input template, or a required output column,
is missing. This prevents inventing a Corporate EWI/catalogue/field.

THIS IS FOR UAT VOLUME IMPORT TESTING ONLY, not real EWI validation.
IT must approve fake IDs, allowed field values, formats, import location and names.
"""

import csv
import io
import math
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import dataiku
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIG — change only in a dedicated DEV notebook
# -----------------------------------------------------------------------------
RETAIL_ALERTS = 5_000
CORPORATE_ALERTS = 5_000
CIB_QA_CASES = 2                    # part of Corporate, NOT additional alerts
HISTORY_MONTHS = 6
ALERT_DATE = "2026-11-02"          # change if UAT agreed on another date
DRY_RUN_COUNT = None                # None=10,000; 10=small trial (5+5)

# DO NOT point this script to an automatic upload folder or PROD location.
# Files/ZIP are staged in the notebook's temporary disk for IT-controlled handoff.
STAGING_PARENT = Path(tempfile.gettempdir())

# Canonical column order taken from the Retail EWS export-output recipes.
# Compare against the ACTUAL dataiku output columns and STOP if different.
SCHEMAS = {
    "ews_ewi_breach_information": [
        "BREACH_ID", "EWI_NAME", "EWI_VALUE_1", "THRESHOLD_1", "DEVIATION_1",
        "EWI_VALUE_2", "THRESHOLD2", "DEVIATION2", "ENTITY_ID", "RM",
        "ID_RM", "RM_N+1", "ID_RM_N+1", "BUSINESS_LINE", "MAIN_EXPLN",
        "TEAM", "ALERT_DATE", "DAY", "Entity_Breaches_12M",
        "ENTITY_BREACHES_TOTAL", "EWI_methodology", "SEGMENT", "error",
    ],
    "ews_ewi_historic_information": [
        "BREACH_ID", "EWI_1", "VALUE_EWI_1", "THRESHOLD_EWI_1",
        "DEVIATION_EWI_1", "CHANGE_EWI_1", "EWI_2", "VALUE_EWI_2",
        "THRESHOLD_EWI_2", "DEVIATION_EWI_2", "CHANGE_EWI_2",
        "HISTORIC_DATE", "ALERT_DATE", "DAY", "error",
    ],
    "ews_entity_static_data": [
        "ENTITY_ID", "SEGMENT", "ENTITY_NAME", "CUSTOMER_IDs",
        "INTERNAL_RATING", "NACE2", "GCC_ID", "GCC_NAME", "TOTAL_DEBT",
        "TOTAL_ASSETS", "DAY", "error",
    ],
    "ews_exposure_static_data": [
        "EXPOSURE_ID", "ENTITY_ID", "PRODUCT_TYPE", "NET_EXPOSURE_AMOUNT_EUR",
        "CURRENCY", "IFRS9_STAGE", "DAYS_PAST_DUE", "PERFORMING_STATUS",
        "GROSS_EXPOSURE_AMOUNT_EUR", "COLLATERAL_AMOUNT",
        "COLLATERAL_LAST_APPRAISAL_DATE", "LIMIT_AMOUNT", "DAY", "error",
    ],
}


def safe(value):
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def number(value):
    try:
        v = float(str(value).strip())
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def fmt(value):
    return "" if value is None else ("{:.4f}".format(value)).rstrip("0").rstrip(".")


def load_template(name):
    ds = dataiku.Dataset(name)
    frame = ds.get_dataframe()
    actual = list(frame.columns)
    expected = SCHEMAS[name]
    if actual != expected:
        missing = [c for c in expected if c not in actual]
        added = [c for c in actual if c not in expected]
        raise RuntimeError(
            "{}: export structure differs from documented recipe.\n"
            "Missing: {}\nUnexpected: {}\nActual order: {}\n"
            "STOP. Send the latest export recipe before generating any files.".format(
                name, missing, added, actual
            )
        )
    return frame.fillna("")


def monthly_dates():
    anchor = pd.Timestamp(ALERT_DATE)
    # Six previous month-end points, not current/future calendar dates.
    return [
        (anchor.replace(day=1) - pd.offsets.MonthEnd(i)).strftime("%Y-%m-%d")
        for i in range(HISTORY_MONTHS, 0, -1)
    ]


def synthetic_values(seed, index, value_key, threshold_key, deviation_key):
    # Keep the official threshold from the observed source, not client values.
    source_value = safe(seed.get(value_key, ""))
    source_threshold = safe(seed.get(threshold_key, ""))
    if not source_value and not source_threshold:
        return "", "", ""
    threshold = number(source_threshold)
    if threshold is not None:
        value = threshold + max(1.0, abs(threshold) * (0.08 + (index % 7) / 100.0))
        return fmt(value), source_threshold, fmt(value - threshold)
    # Non-numeric code/label from the existing EWS catalogue.
    # Its use as a synthetic test value still requires UAT/IT validation.
    return source_value, source_threshold, ""


def main():
    frames = {name: load_template(name) for name in SCHEMAS}
    breach_src = frames["ews_ewi_breach_information"]
    if breach_src.empty:
        raise RuntimeError("Source breach dataset is empty: need a real DEV export to use as a template.")
    breach_src = breach_src.copy()
    breach_src["SEGMENT"] = breach_src["SEGMENT"].astype(str).str.strip()
    by_segment = {
        segment: breach_src.loc[breach_src["SEGMENT"].eq(segment)].reset_index(drop=True)
        for segment in ("Retail", "Corporate")
    }
    for seg, df in by_segment.items():
        if df.empty:
            raise RuntimeError(
                "No '{}' EWI sample in the CURRENT DEV breach output. "
                "Cannot invent a valid EWI catalogue for this segment. "
                "Use an existing valid {} export as a template (send current recipe/output).".format(
                    seg, seg
                )
            )

    if frames["ews_ewi_historic_information"].empty or frames["ews_entity_static_data"].empty:
        raise RuntimeError("Historic or entity template dataset is empty.")
    if frames["ews_exposure_static_data"].empty:
        raise RuntimeError("Exposure template dataset is empty; do not fabricate product/status enums.")

    n_retail, n_corporate = RETAIL_ALERTS, CORPORATE_ALERTS
    if DRY_RUN_COUNT is not None:
        if DRY_RUN_COUNT < 2 or DRY_RUN_COUNT % 2:
            raise ValueError("DRY_RUN_COUNT must be an even integer >=2")
        n_retail = n_corporate = DRY_RUN_COUNT // 2
    n_total = n_retail + n_corporate
    if n_total > 99_999:
        raise ValueError("Protect the 6-digit BREACH_ID sequence format")

    # Only use observed input enums/EWI names; do not reuse real customer names/IDs.
    hist_src = frames["ews_ewi_historic_information"]
    entity_src = frames["ews_entity_static_data"]
    exp_src = frames["ews_exposure_static_data"]
    hist_lookup = {safe(k): g.iloc[0].to_dict() for k, g in hist_src.groupby("BREACH_ID")}
    entity_lookup = {safe(k): g.iloc[0].to_dict() for k, g in entity_src.groupby("ENTITY_ID")}
    exp_lookup = {safe(k): g.iloc[0].to_dict() for k, g in exp_src.groupby("ENTITY_ID")}
    history_dates = monthly_dates()
    out = {name: [] for name in SCHEMAS}
    special_cib_ids = []

    for i in range(n_total):
        seg = "Retail" if i < n_retail else "Corporate"
        seg_idx = i if seg == "Retail" else i - n_retail
        seed = by_segment[seg].iloc[seg_idx % len(by_segment[seg])].to_dict()
        old_breach = safe(seed["BREACH_ID"])
        old_entity = safe(seed["ENTITY_ID"])
        if not old_breach or not old_entity or not safe(seed["EWI_NAME"]):
            raise RuntimeError("Source EWI has a missing BREACH_ID, ENTITY_ID or EWI_NAME.")
        special_cib = (seg == "Corporate" and seg_idx < min(CIB_QA_CASES, n_corporate))
        ix = i + 1
        new_entity = "99{:010d}".format(ix)
        new_breach = "{}_{:06d}".format(ALERT_DATE.replace("-", ""), 900000 + ix)
        net = 150000 + (ix % 101) * 150
        v1, th1, dev1 = synthetic_values(
            seed, ix, "EWI_VALUE_1", "THRESHOLD_1", "DEVIATION_1")
        v2, th2, dev2 = synthetic_values(
            seed, ix, "EWI_VALUE_2", "THRESHOLD2", "DEVIATION2")

        # BREACH: metadata enums/EWI names come from real template;
        # names, RM fields, IDs and explanatory free text are synthetic.
        br = dict(seed)
        br.update({
            "BREACH_ID": new_breach,
            "ENTITY_ID": new_entity,
            "EWI_VALUE_1": v1, "THRESHOLD_1": th1, "DEVIATION_1": dev1,
            "EWI_VALUE_2": v2, "THRESHOLD2": th2, "DEVIATION2": dev2,
            "RM": "UAT_TEST", "ID_RM": "", "RM_N+1": "UAT_TEST", "ID_RM_N+1": "",
            "MAIN_EXPLN": "UAT synthetic CIB QA case" if special_cib else "UAT synthetic volume test",
            "ALERT_DATE": ALERT_DATE, "DAY": ALERT_DATE,
            "Entity_Breaches_12M": "0", "ENTITY_BREACHES_TOTAL": "1",
            "SEGMENT": seg, "error": "",
        })
        out["ews_ewi_breach_information"].append(br)
        if special_cib:
            special_cib_ids.append(new_breach)

        # HISTORIC: 6 rows per breach, same original schema / approved EWI label.
        hs = hist_lookup.get(old_breach, hist_src.iloc[0].to_dict())
        for j, date in enumerate(history_dates):
            h = {col: "" for col in SCHEMAS["ews_ewi_historic_information"]}
            h["BREACH_ID"] = new_breach
            h["EWI_1"] = safe(seed["EWI_NAME"])
            h["VALUE_EWI_1"] = v1
            h["THRESHOLD_EWI_1"] = th1
            h["DEVIATION_EWI_1"] = dev1
            h["CHANGE_EWI_1"] = "0" if j else ""
            # Respect whether optional EWI_2 data exist in template, without
            # copying original customer-specific values.
            if safe(hs.get("EWI_2", "")):
                h["EWI_2"] = safe(hs["EWI_2"])
                h["VALUE_EWI_2"] = v2
                h["THRESHOLD_EWI_2"] = th2
                h["DEVIATION_EWI_2"] = dev2
                h["CHANGE_EWI_2"] = "0" if j else ""
            h.update({"HISTORIC_DATE": date, "ALERT_DATE": ALERT_DATE, "DAY": ALERT_DATE, "error": ""})
            out["ews_ewi_historic_information"].append(h)

        # ENTITY STATIC: preserve original allowed categorical classifications.
        es = entity_lookup.get(old_entity, entity_src.iloc[0].to_dict())
        entity = dict(es)
        entity.update({
            "ENTITY_ID": new_entity, "SEGMENT": seg,
            "ENTITY_NAME": "UAT_SYNTHETIC_ENTITY_{:05d}".format(ix),
            "CUSTOMER_IDs": "98{:010d}".format(ix),
            "GCC_ID": "97{:010d}".format(ix),
            "GCC_NAME": "UAT_SYNTHETIC_GROUP_{:05d}".format(ix),
            "TOTAL_DEBT": fmt(net), "TOTAL_ASSETS": fmt(net * 1.8),
            "DAY": ALERT_DATE, "error": "",
        })
        out["ews_entity_static_data"].append(entity)

        # EXPOSURE STATIC: product/status/currency are taken from an actual
        # source row, never fabricated as a new bank enum.
        ex = dict(exp_lookup.get(old_entity, exp_src.iloc[0].to_dict()))
        ex.update({
            "EXPOSURE_ID": "96{:012d}".format(ix), "ENTITY_ID": new_entity,
            "NET_EXPOSURE_AMOUNT_EUR": fmt(net),
            "GROSS_EXPOSURE_AMOUNT_EUR": fmt(net * 1.08),
            "COLLATERAL_AMOUNT": fmt(net * 0.7),
            "LIMIT_AMOUNT": fmt(net * 1.2),
            "DAYS_PAST_DUE": "0", "DAY": ALERT_DATE, "error": "",
        })
        if safe(ex.get("COLLATERAL_LAST_APPRAISAL_DATE", "")):
            ex["COLLATERAL_LAST_APPRAISAL_DATE"] = "2026-09-30"
        out["ews_exposure_static_data"].append(ex)

    breaches = out["ews_ewi_breach_information"]
    historic = out["ews_ewi_historic_information"]
    entities = out["ews_entity_static_data"]
    exposure = out["ews_exposure_static_data"]
    breach_ids = {x["BREACH_ID"] for x in breaches}
    entity_ids = {x["ENTITY_ID"] for x in breaches}
    assert len(breaches) == n_total == len(breach_ids) == len(entity_ids)
    assert len(historic) == HISTORY_MONTHS * n_total
    assert {h["BREACH_ID"] for h in historic} == breach_ids
    assert {e["ENTITY_ID"] for e in entities} == entity_ids
    assert {e["ENTITY_ID"] for e in exposure} == entity_ids
    assert Counter(b["SEGMENT"] for b in breaches) == {"Retail": n_retail, "Corporate": n_corporate}
    assert not any(set(row) != set(SCHEMAS[name]) for name, rows in out.items() for row in rows)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = STAGING_PARENT / ("BIL_EWS_UAT_SINGLE_EXPORT_" + stamp)
    target.mkdir(parents=True, exist_ok=False)
    file_paths = []
    for name, rows in out.items():
        csv_path = target / "{}_{}.csv".format(name, stamp)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMAS[name], extrasaction="raise", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: safe(row[k]) for k in SCHEMAS[name]})
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            if next(reader) != SCHEMAS[name]:
                raise AssertionError("CSV header changed: {}".format(csv_path))
            if sum(1 for _ in reader) != len(rows):
                raise AssertionError("CSV row count changed: {}".format(csv_path))
        file_paths.append(csv_path)

    # SINGLE delivery containing the FOUR standard files (NO Retail/Corp split).
    zip_path = target / "BIL_EWS_UAT_SINGLE_EXPORT_{}.zip".format(stamp)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in file_paths:
            zf.write(path, arcname=path.name)
    with zipfile.ZipFile(zip_path) as zf:
        assert len(zf.namelist()) == 4 and zf.testzip() is None

    print("ONE EXPORT / FOUR CSV; NO FLOW EXECUTION")
    print("Breaches: {} | Retail: {} | Corporate: {}".format(n_total, n_retail, n_corporate))
    print("History rows: {} | Entity rows: {} | Exposure rows: {}".format(
        len(historic), len(entities), len(exposure)))
    print("Two designated CIB test cases (within Corporate):", special_cib_ids)
    print("Existing EWI names used:", sorted({b["EWI_NAME"] for b in breaches}))
    print("Files:")
    for p in file_paths:
        print("   {} | {:,} rows".format(p.name, len(out[p.name.rsplit("_" + stamp + ".csv", 1)[0]])))
    print("ZIP:", zip_path)
    print("NOT approved for direct UAT ingestion. IT must validate a small sample first.")
    return zip_path


if __name__ == "__main__":
    main()
