# -*- coding: utf-8 -*-
"""BIL EWS Phase 2 - ONE synthetic UAT export from validated DEV templates.

Scope (import/load test ONLY, not EWS model validation):
  * 5,000 Retail alerts from the Retail-model DEV output
  * 4,998 Corporate alerts from Corporate-model output templates
  * 2 CIB alerts from the Retail model, SEGMENT = Corporate
  * ONE delivery set consisting of the four existing CSV formats.

This script NEVER executes a Flow and NEVER overwrites existing Dataiku datasets.
It writes ONLY to a NEW dedicated DEV test Managed Folder when CHECK_ONLY=False.
It deliberately fails if valid Corporate and Retail/CIB templates are missing.
It uses existing EWI labels and schema, with synthetic identities and amounts.

Prerequisites:
  A. Retail Phase 2 outputs already built (incl. CIB and SEGMENT in both files).
  B. Corporate-model EWS export templates in a DEV Managed Folder, or mapped
     to separate DEV datasets; both modes are supported below.
  C. Corporate source format compatible with the four EWS interface schemas.
  D. IT approval of synthetic ID conventions and CSV dialect before transfer.

Keep CHECK_ONLY=True for the initial validation. This code does NOT amend
compute_EWS_Output_Datasets or create additional indicator definitions.
"""

from collections import Counter
from datetime import datetime
import csv
import io
import math

import dataiku
import pandas as pd

# ======================================================================
# CONFIGURE IN DEV
# ======================================================================
CHECK_ONLY = True  # True = read/validate, NEVER upload; False = generate/upload

RETAIL_ALERTS = 5_000
CORPORATE_ALERTS = 5_000  # includes the 2 CIB alerts
CIB_FROM_RETAIL = 2
N_HISTORY_MONTHS = 6
ALERT_DATE = "2026-11-02"

# Dedicated NEW, ISOLATED test folder. NEVER the operational export folder.
TEST_OUTPUT_FOLDER_ID = None

# Retail sources are the existing Phase 2 DEV output datasets.
RETAIL_DATASETS = {
    "ews_ewi_breach_information": "ews_ewi_breach_information",
    "ews_ewi_historic_information": "ews_ewi_historic_information",
    "ews_entity_static_data": "ews_entity_static_data",
    "ews_exposure_static_data": "ews_exposure_static_data",
}

# Corporate MODEL, not Retail CIB, source.
# Select "folder" for existing Corporate CSVs, or "datasets" when the
# Corporate outputs exist as independently named datasets in this project.
CORPORATE_INPUT_MODE = "folder"  # "folder" or "datasets"
CORPORATE_SOURCE_FOLDER_ID = None
CORPORATE_CSV_PATHS = {  # actual file names/paths from the Corporate-model export
    "ews_ewi_breach_information": None,
    "ews_ewi_historic_information": None,
    "ews_entity_static_data": None,
    "ews_exposure_static_data": None,
}
CORPORATE_DATASETS = {  # only used if CORPORATE_INPUT_MODE="datasets"
    "ews_ewi_breach_information": None,
    "ews_ewi_historic_information": None,
    "ews_entity_static_data": None,
    "ews_exposure_static_data": None,
}

# Confirm these THREE fields with the REAL production export recipe before UAT.
CSV_DELIMITER = ","
CSV_ENCODING = "utf-8"
CSV_LINE_TERMINATOR = "\n"

# ======================================================================
# CONTRACT: existing Phase 2 columns, order unchanged, no extra columns
# ======================================================================
SCHEMA = {
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
        "ENTITY_ID", "SEGMENT", "ENTITY_NAME", "CUSTOMER_IDs", "INTERNAL_RATING",
        "NACE2", "GCC_ID", "GCC_NAME", "TOTAL_DEBT", "TOTAL_ASSETS",
        "DAY", "error",
    ],
    "ews_exposure_static_data": [
        "EXPOSURE_ID", "ENTITY_ID", "PRODUCT_TYPE", "NET_EXPOSURE_AMOUNT_EUR",
        "CURRENCY", "IFRS9_STAGE", "DAYS_PAST_DUE", "PERFORMING_STATUS",
        "GROSS_EXPOSURE_AMOUNT_EUR", "COLLATERAL_AMOUNT",
        "COLLATERAL_LAST_APPRAISAL_DATE", "LIMIT_AMOUNT", "DAY", "error",
    ],
}

CIB_BUSINESS_LINES = {
    "corporate & institutional banking",
    "corporate and institutional banking",
}

# These are the display names associated with the Stage 2 codes excluded
# from the Retail Phase 2 export; do not introduce them through templates.
EXCLUDED_RETAIL_EWI_NAMES = {
    "downgrade from stage 1 to stage 2",
    "connected client classified as stage 2",
}


def clean(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    return "" if value.casefold() in {"nan", "none", "<na>", "miss"} else value


def _check_schema(df, name, tag):
    actual = list(df.columns)
    if actual != SCHEMA[name]:
        raise ValueError(
            "{} / {}: columns or order DIFFER from Phase 2 interface.\n"
            "Expected: {}\nActual: {}".format(tag, name, SCHEMA[name], actual)
        )


def _normalize(df):
    return df.apply(lambda col: col.map(clean))


def _read_datasets(names, tag):
    if not all(names.values()):
        raise ValueError("{}: configure all four dataset names".format(tag))
    data = {}
    for name, dataset_name in names.items():
        frame = dataiku.Dataset(dataset_name).get_dataframe()
        _check_schema(frame, name, tag)
        data[name] = _normalize(frame)
        print("{} / {}: {} rows".format(tag, name, len(frame)))
    return data


def _read_corporate():
    if CORPORATE_INPUT_MODE == "datasets":
        if any(CORPORATE_DATASETS[k] == RETAIL_DATASETS[k] for k in SCHEMA):
            raise ValueError("Corporate MODEL datasets must not reuse Retail output datasets")
        return _read_datasets(CORPORATE_DATASETS, "Corporate model")
    if CORPORATE_INPUT_MODE != "folder":
        raise ValueError("CORPORATE_INPUT_MODE must be 'folder' or 'datasets'")
    if not CORPORATE_SOURCE_FOLDER_ID or not all(CORPORATE_CSV_PATHS.values()):
        raise ValueError(
            "Corporate-model output not configured. Set CORPORATE_SOURCE_FOLDER_ID "
            "and all four CORPORATE_CSV_PATHS to the actual Corporate export; "
            "do not substitute Retail CIB for Corporate model alerts."
        )
    folder = dataiku.Folder(CORPORATE_SOURCE_FOLDER_ID)
    data = {}
    for name, path in CORPORATE_CSV_PATHS.items():
        with folder.get_download_stream(path) as stream:
            raw = stream.read()
        frame = pd.read_csv(
            io.BytesIO(raw), sep=CSV_DELIMITER, encoding=CSV_ENCODING,
            dtype=str, keep_default_na=False,
        )
        _check_schema(frame, name, "Corporate model")
        data[name] = _normalize(frame)
        print("Corporate model / {}: {} rows".format(name, len(frame)))
    return data


def _valid_candidates(source, tag):
    b = source["ews_ewi_breach_information"]
    h = source["ews_ewi_historic_information"]
    e = source["ews_entity_static_data"]
    x = source["ews_exposure_static_data"]
    if any(v.empty for v in (b, h, e, x)):
        raise ValueError("{}: one of the four source files is empty".format(tag))
    for column in ("BREACH_ID", "ENTITY_ID", "EWI_NAME", "SEGMENT"):
        if b[column].eq("").any():
            raise ValueError("{}: blank {} in breach source".format(tag, column))
    if b["BREACH_ID"].duplicated().any():
        raise ValueError("{}: duplicated BREACH_ID in source".format(tag))
    if h.duplicated(["BREACH_ID", "HISTORIC_DATE"]).any():
        raise ValueError("{}: duplicated historic BREACH_ID+HISTORIC_DATE".format(tag))
    if e["ENTITY_ID"].duplicated().any():
        raise ValueError("{}: duplicated entity static ENTITY_ID".format(tag))
    b_seg = b[["ENTITY_ID", "SEGMENT"]].drop_duplicates()
    e_seg = e[["ENTITY_ID", "SEGMENT"]].drop_duplicates()
    mismatches = b_seg.merge(e_seg, how="inner", on="ENTITY_ID", suffixes=("_B", "_E"))
    if (mismatches["SEGMENT_B"] != mismatches["SEGMENT_E"]).any():
        raise ValueError("{}: inconsistent entity SEGMENT in breach/entity static".format(tag))
    ids = set(e["ENTITY_ID"]) & set(x["ENTITY_ID"])
    history = set(h["BREACH_ID"])
    valid = b.loc[b["ENTITY_ID"].isin(ids) & b["BREACH_ID"].isin(history)].copy()
    print("{}: {} linked breach templates / {} total".format(tag, len(valid), len(b)))
    if valid.empty:
        raise ValueError("{}: no breach with linked history, entity and exposure".format(tag))
    return valid.reset_index(drop=True)


def preflight():
    print("=== BIL PHASE 2 UAT PREFLIGHT / READ ONLY ===")
    retail_src = _read_datasets(RETAIL_DATASETS, "Retail DEV")
    retail_valid = _valid_candidates(retail_src, "Retail DEV")
    eligible_retail = retail_valid.loc[
        retail_valid["SEGMENT"].eq("Retail")
        & ~retail_valid["EWI_NAME"].str.casefold().isin(EXCLUDED_RETAIL_EWI_NAMES)
    ].reset_index(drop=True)
    eligible_cib = retail_valid.loc[
        retail_valid["SEGMENT"].eq("Corporate")
        & retail_valid["BUSINESS_LINE"].str.casefold().isin(CIB_BUSINESS_LINES)
        & ~retail_valid["EWI_NAME"].str.casefold().isin(EXCLUDED_RETAIL_EWI_NAMES)
    ].reset_index(drop=True)
    print("Retail-model Retail seeds:", len(eligible_retail))
    print("Retail-model CIB seeds:", len(eligible_cib))
    if eligible_retail.empty or eligible_cib.empty:
        raise ValueError(
            "Need >=1 Retail-model Retail and >=1 Retail-model CIB alert "
            "with linked outputs. The code will not fabricate an EWI_NAME."
        )

    corp_src = _read_corporate()
    corp_valid = _valid_candidates(corp_src, "Corporate model")
    eligible_corp = corp_valid.loc[
        corp_valid["SEGMENT"].eq("Corporate")
    ].reset_index(drop=True)
    print("Corporate-model Corporate seeds:", len(eligible_corp))
    if eligible_corp.empty:
        raise ValueError("Corporate MODEL export needs >=1 valid Corporate alert")
    return [
        (retail_src, eligible_retail, "Retail model / Retail", RETAIL_ALERTS),
        (corp_src, eligible_corp, "Corporate model / Corporate", CORPORATE_ALERTS - CIB_FROM_RETAIL),
        (retail_src, eligible_cib, "Retail model / CIB", CIB_FROM_RETAIL),
    ]


def _numeric(s):
    try:
        value = float(clean(s))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _fmt(n):
    return ("{:.4f}".format(float(n)).rstrip("0").rstrip(".") or "0")


def _synthetic_measure(template_value, template_threshold, n):
    """Generate a synthetic value while preserving categorical/numeric field type."""
    v, t = clean(template_value), clean(template_threshold)
    v_num, t_num = _numeric(v), _numeric(t)
    if not v and not t:
        return "", t, ""
    if t_num is not None:
        # Test data, not a proof of the indicator's business rule.
        generated = t_num + max(0.1, abs(t_num) * 0.01) + (n % 3)
        return _fmt(generated), t, _fmt(generated - t_num)
    if v_num is not None:
        generated = 1 + (n % 7)
        return _fmt(generated), t, ""
    # For rating values, reuse an existing allowed categorical label only.
    return v, t, ""


def _month_end_dates(test_date, count):
    date = pd.Timestamp(test_date)
    if date.strftime("%Y-%m-%d") != test_date:
        raise ValueError("ALERT_DATE must be YYYY-MM-DD")
    # Six last completed month-ends, oldest first, excluding the month of ALERT_DATE.
    return [
        (date.replace(day=1) - pd.offsets.MonthEnd(months)).strftime("%Y-%m-%d")
        for months in range(count, 0, -1)
    ]


def build(templates):
    output = {name: [] for name in SCHEMA}
    origins = Counter()
    date_list = _month_end_dates(ALERT_DATE, N_HISTORY_MONTHS)
    n = 0
    for source, candidates, origin, number in templates:
        h = source["ews_ewi_historic_information"]
        e = source["ews_entity_static_data"].drop_duplicates("ENTITY_ID").set_index("ENTITY_ID")
        x = source["ews_exposure_static_data"].groupby("ENTITY_ID")
        h_grouped = {key: grp.sort_values("HISTORIC_DATE") for key, grp in h.groupby("BREACH_ID")}
        for i in range(number):
            n += 1
            template = candidates.iloc[i % len(candidates)]
            old_entity, old_breach = template["ENTITY_ID"], template["BREACH_ID"]
            segment = "Retail" if origin == "Retail model / Retail" else "Corporate"
            entity_id = "99{:010d}".format(n)
            breach_id = "{}_{:06d}".format(ALERT_DATE.replace("-", ""), 900000 + n)
            customer_id = "98{:010d}".format(n)
            gcc_id = "97{:010d}".format(n)

            v1, t1, dev1 = _synthetic_measure(template["EWI_VALUE_1"], template["THRESHOLD_1"], n)
            v2, t2, dev2 = _synthetic_measure(template["EWI_VALUE_2"], template["THRESHOLD2"], n)
            breach = dict(template)
            breach.update({
                "BREACH_ID": breach_id, "ENTITY_ID": entity_id,
                "SEGMENT": segment, "EWI_VALUE_1": v1, "THRESHOLD_1": t1,
                "DEVIATION_1": dev1, "EWI_VALUE_2": v2, "THRESHOLD2": t2,
                "DEVIATION2": dev2, "ALERT_DATE": ALERT_DATE, "DAY": ALERT_DATE,
                "RM": "", "ID_RM": "", "RM_N+1": "", "ID_RM_N+1": "",
                "TEAM": "", "Entity_Breaches_12M": "", "ENTITY_BREACHES_TOTAL": "",
                "error": "",
            })
            output["ews_ewi_breach_information"].append(breach)

            source_hist = h_grouped[old_breach]
            for j, hist_date in enumerate(date_list):
                src_hist = source_hist.iloc[j % len(source_hist)]
                hist = {col: "" for col in SCHEMA["ews_ewi_historic_information"]}
                hist["BREACH_ID"] = breach_id
                hist["EWI_1"] = clean(src_hist["EWI_1"]) or clean(template["EWI_NAME"])
                hist["THRESHOLD_EWI_1"] = clean(src_hist["THRESHOLD_EWI_1"])
                hist["THRESHOLD_EWI_2"] = clean(src_hist["THRESHOLD_EWI_2"])
                hist["EWI_2"] = clean(src_hist["EWI_2"])
                value_h1, _, deviation_h1 = _synthetic_measure(
                    src_hist["VALUE_EWI_1"], src_hist["THRESHOLD_EWI_1"], n + j
                )
                value_h2, _, deviation_h2 = _synthetic_measure(
                    src_hist["VALUE_EWI_2"], src_hist["THRESHOLD_EWI_2"], n + j
                )
                hist.update({
                    "VALUE_EWI_1": value_h1, "DEVIATION_EWI_1": deviation_h1,
                    "CHANGE_EWI_1": "0" if value_h1 else "",
                    "VALUE_EWI_2": value_h2, "DEVIATION_EWI_2": deviation_h2,
                    "CHANGE_EWI_2": "0" if value_h2 else "",
                    "HISTORIC_DATE": hist_date, "ALERT_DATE": ALERT_DATE, "DAY": ALERT_DATE,
                })
                output["ews_ewi_historic_information"].append(hist)

            ent = dict(e.loc[old_entity])
            ent.update({
                "ENTITY_ID": entity_id, "SEGMENT": segment,
                "ENTITY_NAME": "UAT_TEST_ENTITY_{:05d}".format(n),
                "CUSTOMER_IDs": customer_id, "INTERNAL_RATING": "", "NACE2": "",
                "GCC_ID": gcc_id, "GCC_NAME": "UAT_TEST_GCC_{:05d}".format(n),
                "TOTAL_DEBT": "", "TOTAL_ASSETS": "", "DAY": ALERT_DATE, "error": "",
            })
            output["ews_entity_static_data"].append(ent)

            exposures = x.get_group(old_entity).head(3)
            for k, (_, src_exp) in enumerate(exposures.iterrows(), start=1):
                exp = dict(src_exp)
                exp.update({
                    "EXPOSURE_ID": "96{:010d}{:02d}".format(n, k),
                    "ENTITY_ID": entity_id, "NET_EXPOSURE_AMOUNT_EUR": str(100000 + n % 100 * 1000 + k * 100),
                    "GROSS_EXPOSURE_AMOUNT_EUR": str(110000 + n % 100 * 1000 + k * 100),
                    "COLLATERAL_AMOUNT": "", "COLLATERAL_LAST_APPRAISAL_DATE": "",
                    "LIMIT_AMOUNT": "", "DAYS_PAST_DUE": str(n % 5),
                    "DAY": ALERT_DATE, "error": "",
                })
                output["ews_exposure_static_data"].append(exp)
            origins[origin] += 1

    _validate_outputs(output, origins)
    print("PASS: output counts", {k: len(v) for k, v in output.items()})
    print("PASS: source breakdown (QA only; not added as columns):", dict(origins))
    return output


def _validate_outputs(output, origins):
    b = output["ews_ewi_breach_information"]
    h = output["ews_ewi_historic_information"]
    e = output["ews_entity_static_data"]
    x = output["ews_exposure_static_data"]
    if len(b) != RETAIL_ALERTS + CORPORATE_ALERTS:
        raise ValueError("Incorrect number of alerts")
    if Counter(row["SEGMENT"] for row in b) != {"Retail": RETAIL_ALERTS, "Corporate": CORPORATE_ALERTS}:
        raise ValueError("Incorrect Retail/Corporate distribution")
    if origins["Retail model / CIB"] != CIB_FROM_RETAIL:
        raise ValueError("Incorrect CIB count")
    ids = [row["BREACH_ID"] for row in b]
    entity_ids = [row["ENTITY_ID"] for row in b]
    if len(ids) != len(set(ids)) or len(entity_ids) != len(set(entity_ids)):
        raise ValueError("Synthetic BREACH_ID or ENTITY_ID duplicated")
    if {row["BREACH_ID"] for row in h} != set(ids) or len(h) != len(b) * N_HISTORY_MONTHS:
        raise ValueError("Historic linkage/size mismatch")
    if len({(row["BREACH_ID"], row["HISTORIC_DATE"]) for row in h}) != len(h):
        raise ValueError("Duplicate historic BREACH_ID+HISTORIC_DATE")
    if {row["ENTITY_ID"] for row in e} != set(entity_ids):
        raise ValueError("Entity static linkage mismatch")
    if {row["ENTITY_ID"] for row in x} != set(entity_ids):
        raise ValueError("Exposure linkage mismatch")
    if len({row["EXPOSURE_ID"] for row in x}) != len(x):
        raise ValueError("EXPOSURE_ID duplicated")
    for name, rows in output.items():
        for row in rows:
            if set(row) != set(SCHEMA[name]):
                raise ValueError("Output columns changed: {}".format(name))
    if any(row["SEGMENT"] not in {"Retail", "Corporate"} for row in e):
        raise ValueError("Invalid segment on entity static")
    if any(row["EWI_NAME"].casefold() in EXCLUDED_RETAIL_EWI_NAMES for row in b[:RETAIL_ALERTS]):
        raise ValueError("Excluded Stage 2 EWI introduced to Retail export")


def export_to_test_folder(output):
    if not TEST_OUTPUT_FOLDER_ID:
        raise ValueError("Set TEST_OUTPUT_FOLDER_ID to a NEW dedicated DEV test Managed Folder")
    folder = dataiku.Folder(TEST_OUTPUT_FOLDER_ID)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, rows in output.items():
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            csv_buffer, fieldnames=SCHEMA[name], extrasaction="raise",
            delimiter=CSV_DELIMITER, quoting=csv.QUOTE_MINIMAL,
            lineterminator=CSV_LINE_TERMINATOR,
        )
        writer.writeheader()
        writer.writerows([{column: clean(row.get(column)) for column in SCHEMA[name]}
                          for row in rows])
        output_name = "{}_{}.csv".format(name, timestamp)
        folder.upload_stream(output_name, io.BytesIO(csv_buffer.getvalue().encode(CSV_ENCODING)))
        print("UPLOADED:", output_name, "rows:", len(rows))
    print("EXPORT DONE: one timestamp, exactly four CSVs in test folder")
    print("NOT YET approved for UAT: confirm synthetic identifiers, CSV dialect, and importer rules with IT")


def main():
    templates = preflight()
    if CHECK_ONLY:
        print("CHECK_ONLY=True: preflight completed; NO files written")
        return
    output = build(templates)
    export_to_test_folder(output)


if __name__ == "__main__":
    main()
