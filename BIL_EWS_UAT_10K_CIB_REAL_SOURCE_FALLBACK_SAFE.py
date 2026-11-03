# -*- coding: utf-8 -*-
"""BIL EWS Phase 2 - synthetic import/load test (NOT a model calculation).

Exactly FOUR files in ONE delivery batch:
  5,000 Retail model-style alerts        -> SEGMENT Retail
  4,998 Corporate model-style alerts     -> SEGMENT Corporate
      2 records derived from real Retail-origin CIB templates -> SEGMENT Corporate
  Total 10,000 breaches.

Corporate field names and EWI examples come from pasted Excel extracts of the
4 historic exports. Retail EWI names come from the latest validated DEV log.
NO real customer/person/entity IDs or RM names are copied from the extracts.
CIB test rows use genuine EWI + value + 6-month historic data from
historical Retail DEV exports. If only one distinct real Phase 2 source is
available, the same template can be reused in two synthetic test rows,
clearly reported as 1 distinct real source (NOT 2 distinct real alerts). Personal names/IDs and static financial values
are NOT copied. Alert/daily dates are rebased for the UAT load-test batch.
These cases are real-derived, NOT the original unmodified bank records.

No dataset reads and no Flow execution. CHECK_ONLY=True is read/write safe.
With CHECK_ONLY=False, writes ONLY into the NEW managed folder identified below.
IT must confirm ID validity, CSV dialect and acceptance before any UAT ingestion.
"""

import csv
import io
import re
from collections import Counter
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

# ======================== CONFIG ========================
CHECK_ONLY = True
RETAIL_CIB_FOLDER_ID = "tFtFJ7YO"  # Source: actual Retail DEV export folder.
MAX_CIB_EXPORTS_TO_SCAN = 250
TEST_OUTPUT_FOLDER_ID = None    # Insert ID of NEW/ISOLATED DEV managed folder.
ALERT_DATE = "2026-11-02"     # Set test batch date agreed with IT.
N_HISTORY_MONTHS = 6
RETAIL_ALERTS = 5000
CORPORATE_MODEL_ALERTS = 4998
CIB_ROUTING_ALERTS = 2
# Load test only. Reuse ONE authentic Phase 2 CIB template twice, but
# report 1 distinct real source / 2 derived test rows. Switch to False
# when two independent genuine source cases are a mandatory requirement.
ALLOW_SINGLE_REAL_SOURCE_FOR_TWO_TEST_ROWS = True
EXPORT_DELIMITER = ","        # Confirm against bank's actual exporter.
EXPORT_ENCODING = "utf-8"    # Confirm against bank's actual exporter.
# ========================================================

FIELDS = {
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

# Real EWI LABELS from user-provided Corporate sample; synthetic values.
CORPORATE_EWIS = [
    ("PD CHANGE", "Absolute increase in probability of default compared with the best PD observed in the previous 12 months.",
     "Substantial increase in probability of default assessment of client"),
    ("DPD & PD", "Days past due exceeding 7 days combined with a probability of default at or above 5%.",
     "Substantial increase in probability of default assessment of client with missed installment on loans / credit cards"),
    ("CASH ADVANCE & LIMIT AFTER TRANSACTION", "Cash advance with credit limit utilisation after the transaction above 94%.",
     "Maxing out credit card limits with cash advance(s)"),
]

# Exact RETAIL EWI labels observed in successful DEV run; no Stage 2 alerts.
RETAIL_EWIS = [
    ("Credit card limit utilization above 100%", "Credit card utilization exceeds the available credit limit.",
     "Credit card utilization > 100%"),
    ("More than 6 failed direct debits in the current month", "The number of failed direct debits in the current month is greater than 6.",
     "Failed direct debit count MTD > 6"),
    ("Forbearance / probation period and days past due > 0", "Forbearance or probation is active and days past due are above 0.",
     "Forbearance or probation active and days past due > 0"),
]

# PRODUCT_TYPE examples copied as types only, not IDs/names from Corporate sample.
CORPORATE_PRODUCTS = [
    "112 HOUSE ACQUISITION OWNER OCCUPIER",
    "114 APARTMENT CONSTRUCTION OWNER OCCUPIER",
    "321 COMMERCIAL OPERATIONS",
    "121 ACQUIS./CONSTR. PROPERTY FOR RENTAL",
    "41 MISCELLANEOUS NEEDS",
    "140 RENEWABLE ENERGY",
    "211 SHOP PREMISES",
]
RETAIL_PRODUCTS = ["Credit card debt", "Overdrafts", "Current Account"]
BANNED_EWIS = {"downgrade from stage 1 to stage 2", "connected client classified as stage 2"}


def number(value):
    result = ("%.2f" % float(value)).rstrip("0").rstrip(".")
    return result if result else "0"


def row_for(name, **values):
    if set(values) - set(FIELDS[name]):
        raise ValueError("Unknown output fields in {}: {}".format(name, set(values) - set(FIELDS[name])))
    return {column: str(values.get(column, "")) for column in FIELDS[name]}


def metric(ewi_name, i):
    """Interface examples only: NOT a calculation/validation of an EWS EWI."""
    if ewi_name == "PD CHANGE":
        v = 3.1 + (i % 120) * 0.13
        return number(v), "3", number(v - 3), "", "", ""
    if ewi_name == "DPD & PD":
        dpd = 8 + i % 70
        pd_pct = 5.1 + (i % 80) * 0.17
        return str(dpd), "7", str(dpd - 7), number(pd_pct), "5", number(pd_pct - 5)
    if ewi_name == "CASH ADVANCE & LIMIT AFTER TRANSACTION":
        v = 94.2 + (i % 30) * 0.17
        return number(v), "94", number(v - 94), "", "", ""
    if ewi_name == "Credit card limit utilization above 100%":
        v = 1.01 + (i % 25) * .012
        return number(v), "1", number(v - 1), "", "", ""
    if ewi_name == "More than 6 failed direct debits in the current month":
        count = 7 + i % 7
        return str(count), "6", str(count - 6), "", "", ""
    if ewi_name == "Forbearance / probation period and days past due > 0":
        dpd = 1 + i % 25
        return str(dpd), "0", str(dpd), "", "", ""
    raise ValueError("Unapproved EWI label: {}".format(ewi_name))


def history_names(ewi_name):
    if ewi_name == "PD CHANGE":
        return "PD CHANGE", ""
    if ewi_name == "DPD & PD":
        return "DPD", "PD"
    # No confirmed history nomenclature for these EWIs in the pasted sample.
    # Retain the EWI label, rather than inventing a different bank code.
    return ewi_name, ""


def build():
    if N_HISTORY_MONTHS != 6:
        raise ValueError("The tested six-month load scenario requires N_HISTORY_MONTHS=6")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", ALERT_DATE):
        raise ValueError("ALERT_DATE must be YYYY-MM-DD")
    test_date = pd.Timestamp(ALERT_DATE)
    dates = [(test_date.replace(day=1) - pd.offsets.MonthEnd(m)).strftime("%Y-%m-%d")
             for m in range(N_HISTORY_MONTHS, 0, -1)]
    output = {name: [] for name in FIELDS}
    origins = Counter()
    n_total = RETAIL_ALERTS + CORPORATE_MODEL_ALERTS + CIB_ROUTING_ALERTS
    if n_total != 10000 or (CORPORATE_MODEL_ALERTS + CIB_ROUTING_ALERTS) != 5000:
        raise ValueError("Expected 5000 Retail + 4998 Corporate model + 2 CIB test cases")

    specs = [("Retail model", RETAIL_ALERTS, "Retail", RETAIL_EWIS),
             ("Corporate model", CORPORATE_MODEL_ALERTS, "Corporate", CORPORATE_EWIS),
             ("Retail-origin CIB simulation", CIB_ROUTING_ALERTS, "Corporate", RETAIL_EWIS[:2])]
    n = 0
    for origin, qty, segment, ewiset in specs:
        for j in range(qty):
            n += 1
            # Synthetic IDs are character strings; no real IDs and no Excel precision loss.
            entity_id = "99{:010d}".format(n)
            customer_id = "98{:010d}".format(n)
            gcc_id = "97{:010d}".format(n)
            breach_id = "{}_{:06d}".format(test_date.strftime("%Y%m%d"), 900000 + n)
            ewi_name, expl, methodology = ewiset[j % len(ewiset)]
            v1, t1, d1, v2, t2, d2 = metric(ewi_name, n)
            line = ("Corporate & Institutional Banking" if segment == "Corporate"
                    else ("Digital Banking" if j % 2 == 0 else "Wealth Management"))
            b = row_for("ews_ewi_breach_information", BREACH_ID=breach_id,
                EWI_NAME=ewi_name, EWI_VALUE_1=v1, THRESHOLD_1=t1, DEVIATION_1=d1,
                EWI_VALUE_2=v2, THRESHOLD2=t2, DEVIATION2=d2, ENTITY_ID=entity_id,
                RM="UAT_TEST_RM", ID_RM="UTRM01",
                BUSINESS_LINE=line, MAIN_EXPLN=expl, TEAM="UAT_TEST_TEAM",
                ALERT_DATE=ALERT_DATE, DAY=ALERT_DATE,
                Entity_Breaches_12M=(str(1 + n % 6) if segment == "Corporate" else ""),
                ENTITY_BREACHES_TOTAL=(str(1 + n % 20) if segment == "Corporate" else ""),
                EWI_methodology=methodology, SEGMENT=segment)
            # Exact contract names containing + cannot be Python keyword args.
            b["RM_N+1"] = "UAT_TEST_SUPERVISOR"
            b["ID_RM_N+1"] = "UTSUP1"
            output["ews_ewi_breach_information"].append(b)

            h1, h2 = history_names(ewi_name)
            prev1, prev2 = None, None
            for idx, hist_date in enumerate(dates):
                # Synthetic historic series based on numeric interface values.
                if v1:
                    value1 = number(max(0.0, float(v1) - (5 - idx) * 0.05))
                    dev1 = number(float(value1) - float(t1)) if t1 else ""
                    change1 = number(float(value1) - prev1) if prev1 is not None else value1
                    prev1 = float(value1)
                else:
                    value1 = dev1 = change1 = ""
                if h2 and v2:
                    value2 = number(max(0.0, float(v2) - (5 - idx) * 0.03))
                    dev2 = number(float(value2) - float(t2)) if t2 else ""
                    change2 = number(float(value2) - prev2) if prev2 is not None else value2
                    prev2 = float(value2)
                else:
                    value2 = dev2 = change2 = ""
                output["ews_ewi_historic_information"].append(row_for(
                    "ews_ewi_historic_information", BREACH_ID=breach_id,
                    EWI_1=h1, VALUE_EWI_1=value1, THRESHOLD_EWI_1=t1,
                    DEVIATION_EWI_1=dev1, CHANGE_EWI_1=change1,
                    EWI_2=h2, VALUE_EWI_2=value2,
                    THRESHOLD_EWI_2=t2 if h2 else "",
                    DEVIATION_EWI_2=dev2, CHANGE_EWI_2=change2,
                    HISTORIC_DATE=hist_date, ALERT_DATE=ALERT_DATE, DAY=ALERT_DATE))

            output["ews_entity_static_data"].append(row_for(
                "ews_entity_static_data", ENTITY_ID=entity_id, SEGMENT=segment,
                ENTITY_NAME="UAT_TEST_ENTITY_{:05d}".format(n), CUSTOMER_IDs=customer_id,
                INTERNAL_RATING=["B+", "BBB", "B-", "D"][n % 4],
                NACE2=["6804", "9998", "6501", "9603"][n % 4],
                GCC_ID=gcc_id, GCC_NAME="UAT_TEST_GCC_{:05d}".format(n),
                TOTAL_DEBT=number(1000 + n % 97 * 1057.33),
                TOTAL_ASSETS=number(5000 + n % 130 * 2042.37), DAY=ALERT_DATE))

            # 1-3 exposures/entity; based on field labels and example product values.
            exp_count = 1 + n % 3
            products = CORPORATE_PRODUCTS if segment == "Corporate" else RETAIL_PRODUCTS
            for k in range(1, exp_count + 1):
                amount = round(700 + ((n * 17 + k * 131) % 1400000) / 3, 2)
                gross = amount + round(20 + (n + k) % 400, 2)
                days = (v1 if ewi_name in {"DPD & PD", "Forbearance / probation period and days past due > 0"} and k == 1
                        else ("" if k % 3 == 0 else str((n + k) % 4)))
                output["ews_exposure_static_data"].append(row_for(
                    "ews_exposure_static_data", EXPOSURE_ID="95{:012d}{:02d}".format(n, k),
                    ENTITY_ID=entity_id, PRODUCT_TYPE=products[(n + k) % len(products)],
                    NET_EXPOSURE_AMOUNT_EUR=number(amount), CURRENCY="EUR",
                    IFRS9_STAGE="S1", DAYS_PAST_DUE=days, PERFORMING_STATUS="PERFORMING",
                    GROSS_EXPOSURE_AMOUNT_EUR=number(gross), COLLATERAL_AMOUNT="",
                    COLLATERAL_LAST_APPRAISAL_DATE="2026-08-28",
                    LIMIT_AMOUNT=number(gross + 2500), DAY=ALERT_DATE))
            origins[origin] += 1
    return output, origins


def validate(output, origins):
    b, h, e, x = (output[name] for name in FIELDS)
    counts = Counter(z["SEGMENT"] for z in b)
    if counts != Counter({"Retail": 5000, "Corporate": 5000}):
        raise AssertionError("Wrong segment counts: {}".format(counts))
    cib_provenances = {k: v for k, v in origins.items()
                       if k.startswith('Retail-origin CIB derived from ')}
    if (origins.get('Retail model') != 5000 or
        origins.get('Corporate model') != 4998 or
        sum(cib_provenances.values()) != 2):
        raise AssertionError('Wrong provenance breakdown: {}'.format(dict(origins)))
    if len(h) != 60000 or len(e) != 10000 or not (10000 <= len(x) <= 30000):
        raise AssertionError("Unexpected output row counts")
    bid = [z["BREACH_ID"] for z in b]
    eid = [z["ENTITY_ID"] for z in b]
    xid = [z["EXPOSURE_ID"] for z in x]
    if len(set(bid)) != 10000 or len(set(eid)) != 10000 or len(set(xid)) != len(x):
        raise AssertionError("Duplicated synthetic IDs")
    if {z["BREACH_ID"] for z in h} != set(bid):
        raise AssertionError("Historic BREACH_ID mismatch")
    if {z["ENTITY_ID"] for z in e} != set(eid) or {z["ENTITY_ID"] for z in x} != set(eid):
        raise AssertionError("Entity/exposure references not linked")
    if len({(z["BREACH_ID"], z["HISTORIC_DATE"]) for z in h}) != len(h):
        raise AssertionError("Duplicated historic keys")
    if any(z["EWI_NAME"].casefold() in BANNED_EWIS for z in b):
        raise AssertionError("Stage2 indicator included")
    mapping = {z["ENTITY_ID"]: z["SEGMENT"] for z in e}
    if any(mapping[z["ENTITY_ID"]] != z["SEGMENT"] for z in b):
        raise AssertionError("Entity segment mismatch")
    if any(z["BUSINESS_LINE"] != "Corporate & Institutional Banking" for z in b if z["SEGMENT"] == "Corporate"):
        raise AssertionError("Corporate BUSINESS_LINE mismatch")
    if any(len(z["BREACH_ID"].split("_")[-1]) != 6 for z in b):
        raise AssertionError("BREACH_ID is not fixed 6-digits")
    for name, records in output.items():
        for record in records:
            if list(record.keys()) != FIELDS[name]:
                raise AssertionError("Column order mismatch: {}".format(name))
            if any("MISS" == str(value).upper() for value in record.values()):
                raise AssertionError("MISS sentinel leaked")
    print("VALIDATION PASSED")
    print("Breaches:", len(b), dict(counts))
    print("Historical records:", len(h))
    print("Entity static:", len(e), "(SEGMENT present)")
    print("Exposures:", len(x))
    print("Sources (only QA, not CSV fields):", dict(origins))
    print("Excluded Stage 2 display EWIs: verified")
    real_cib = [z for z in b if z["ENTITY_ID"] in {"99{:010d}".format(9999), "99{:010d}".format(10000)}]
    if len(real_cib) != 2 or not all(z["EWI_NAME"].casefold() in ALLOWED_PHASE2_EWIS for z in real_cib):
        raise AssertionError("CIB EWI names do not match the current Phase 2 catalogue")
    print('Real-derived CIB TEST ROWS:', len(real_cib), '(synthetic IDs + date-rebased)')



# Current Phase 2 EWI display names were extracted from the validated
# compute_EWS_Output_Datasets_FINAL_STAGE2_EXCLUDED_CIB_SEGMENT.py recipe.
# This excludes Stage 2 + obsolete indicator codes at the source.
ALLOWED_PHASE2_EWIS = frozenset(('cash advance / transfer to neobanks > 0.5x positive balances', 'cash advance / transfer to neobanks > 4x positive balances', 'cash advance amount above eur 5,000', 'cash advance and same-day limit utilization crossing above 94%', 'cash advance and same-day limit utilization crossing above 97%', 'client rating downgrade by at least 6 notches vs 12 months ago', 'client rating worse than b and downgrade by at least 1 notch vs 12 months ago', 'client rating worse than b and downgrade by at least 3 notches vs 12 months ago', 'client rating worse than b+ and downgrade by at least 2 notches vs 12 months ago', 'client rating worse than b- and downgrade by at least 1 notch vs 12 months ago', 'client rating worse than bb- and downgrade by at least 4 notches vs 12 months ago', 'client rating worse than bbb- and downgrade by at least 5 notches vs 12 months ago', 'connected client unlikely to pay', 'credit card limit utilization above 100%', 'credit card utilization absolute change > 95% vs average credit card utilization over the last 12 months and current utilization > 95%', 'credit card utilization relative change > 100% vs 3-month maximum and current utilization > 90%', 'days past due > 15, overdraft > eur 500 and gec exposure > eur 1m', 'days past due > 30, overdraft > eur 500 and gec exposure > eur 250k', 'days past due > 60, overdraft > eur 500 and gec exposure > eur 50k', 'forbearance / probation period and days past due > 0', 'gec head downgrade by at least 1 notch and client downgrade by at least 6 notches', 'gec head downgrade by at least 3 notches and client downgrade by at least 5 notches', 'gec head downgrade by at least 3 notches and client rating worse than b', 'gec head downgrade by at least 3 notches and client rating worse than b+', 'gec head downgrade by at least 6 notches and client downgrade by at least 3 notches', 'gec head downgrade by at least 6 notches vs 12 months ago', 'gec head downgrade by more than 7 notches vs 12 months ago', 'gec head rating worse than b and downgrade by at least 4 notches vs 12 months ago', 'gec head rating worse than b- and downgrade by at least 1 notch vs 12 months ago', 'loan-to-value above 95% and client downgrade by more than 4 notches', 'loan-to-value above 95% and client rating worse than b', 'more than 6 failed direct debits in the current month', 'neobank top-up amount above eur 1,000', 'personal circumstances deterioration'))

# Do not depend on a previous notebook cell. Discover the actual cases again
# from original Retail DEV exports and take only cases with all four references.
import re as _re

CIB_FILENAME_RE = _re.compile(
    r'^(ews_ewi_breach_information|ews_ewi_historic_information|'
    r'ews_entity_static_data|ews_exposure_static_data)_(\d{8}_\d{6})\.csv$',
    _re.IGNORECASE,
)
CIB_BUSINESS_LINE_RE = r'^corporate\s*(?:(?:&|and)\s*)?institutional\s*banking$'


def _read_original_csv(folder, path):
    with folder.get_download_stream(path) as stream:
        raw = stream.read()
    if not raw:
        return pd.DataFrame()
    header = raw.splitlines()[0].decode('utf-8-sig', errors='replace')
    sep = max([',', ';', '\t', '|'], key=header.count)
    if header.count(sep) == 0:
        raise ValueError('Could not identify CSV delimiter in source export')
    for enc in ['utf-8-sig', 'cp1252']:
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str,
                               keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError('Could not decode the original Retail export')


def _single_source(frame, col, value):
    if col not in frame.columns:
        return frame.iloc[:0].copy()
    return frame.loc[frame[col].astype(str).str.strip().eq(str(value).strip())].copy()


def _parse_source_history_dates(series):
    # Mixed date formats: historic exports may use YYYY-MM-DD or 28-Feb-26.
    # Avoid pandas dayfirst warnings for ISO dates.
    text = series.astype(str).str.strip()
    iso = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso.any():
        parsed.loc[iso] = pd.to_datetime(text.loc[iso], format="%Y-%m-%d", errors="coerce")
    if (~iso).any():
        other = text.loc[~iso]
        for fmt in ("%d-%b-%y", "%d-%b-%Y", "%m/%d/%Y", "%d/%m/%Y"):
            pending = other.loc[parsed.loc[other.index].isna()]
            if pending.empty:
                break
            parsed.loc[pending.index] = pd.to_datetime(pending, format=fmt,
                                                       errors="coerce")
    return parsed


def discover_real_cib():
    """Read only: scan historical Retail DEV folder, no client data in the log."""
    import dataiku
    folder = dataiku.Folder(RETAIL_CIB_FOLDER_ID)
    paths = {}
    for path in folder.list_paths_in_partition():
        match = CIB_FILENAME_RE.fullmatch(path.rsplit('/', 1)[-1])
        if match:
            parent = path.rsplit('/', 1)[0] if '/' in path else ''
            paths[(parent, match.group(2), match.group(1).lower())] = path
    batches = sorted([
        (stamp, parent, path)
        for (parent, stamp, name), path in paths.items()
        if name == 'ews_ewi_breach_information'
    ], reverse=True)
    print('Retail DEV historic breach exports:', len(batches))
    candidates, seen, errors = [], set(), 0
    for stamp, parent, path in batches[:MAX_CIB_EXPORTS_TO_SCAN]:
        try:
            b = _read_original_csv(folder, path)
            if not {'BREACH_ID', 'ENTITY_ID', 'EWI_NAME', 'BUSINESS_LINE'}.issubset(b.columns):
                continue
            bus = (b['BUSINESS_LINE'].astype(str).str.strip()
                   .str.replace(r'\s+', ' ', regex=True).str.casefold())
            ew = b['EWI_NAME'].astype(str).str.strip().str.casefold()
            mask = bus.str.fullmatch(CIB_BUSINESS_LINE_RE, na=False)
            for record in b.loc[mask].to_dict('records'):
                ewi = str(record['EWI_NAME']).strip().casefold()
                if ewi not in ALLOWED_PHASE2_EWIS:
                    continue
                key = (record['ENTITY_ID'], ewi, record.get('ALERT_DATE', ''))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((stamp, parent, record))
        except Exception:
            errors += 1
    print('CIB real alerts compatible with current Phase 2:', len(candidates))
    if errors:
        print('Unreadable exports:', errors)

    # Check completeness before selection and prefer two distinct entities.
    checked, cache = [], {}
    for stamp, parent, record in candidates:
        src = {}
        names = tuple(FIELDS)
        pair = {(name, paths.get((parent, stamp, name))) for name in names}
        if any(p is None for _, p in pair):
            continue
        try:
            for name, p in pair:
                if p not in cache:
                    cache[p] = _read_original_csv(folder, p)
                src[name] = cache[p]
            bid, eid = record['BREACH_ID'], record['ENTITY_ID']
            b = _single_source(src['ews_ewi_breach_information'], 'BREACH_ID', bid)
            h = _single_source(src['ews_ewi_historic_information'], 'BREACH_ID', bid)
            e = _single_source(src['ews_entity_static_data'], 'ENTITY_ID', eid)
            x = _single_source(src['ews_exposure_static_data'], 'ENTITY_ID', eid)
            if len(b) != 1 or len(h) != 6 or len(e) != 1 or not (1 <= len(x) <= 100):
                continue
            if not all(k in b.columns for k in ('EWI_NAME', 'MAIN_EXPLN', 'EWI_methodology')):
                continue
            if 'HISTORIC_DATE' not in h.columns:
                continue
            # HISTORIC_DATE parse, reject if ambiguous or duplicated.
            hd = _parse_source_history_dates(h['HISTORIC_DATE'])
            if hd.isna().any() or hd.nunique() != 6:
                continue
            checked.append({'stamp': stamp, 'eid': eid, 'breach': b,
                            'historic': h, 'entity': e, 'exposure': x})
        except Exception:
            continue

    print('Phase 2 CIB alerts with complete 4-file references:', len(checked))
    selected, used_eids = [], set()
    for item in checked:
        if item['eid'] not in used_eids:
            selected.append(item)
            used_eids.add(item['eid'])
        if len(selected) == 2:
            break
    if len(selected) < 2:
        selected_stamps_and_ids = {(z['stamp'], z['breach'].iloc[0]['BREACH_ID']) for z in selected}
        for item in checked:
            key = (item['stamp'], item['breach'].iloc[0]['BREACH_ID'])
            if key not in selected_stamps_and_ids:
                selected.append(item)
                selected_stamps_and_ids.add(key)
            if len(selected) == 2:
                break
    if not selected:
        raise ValueError(
            'No genuine Retail-origin CIB alert with current Phase 2 EWI and complete '
            'four-file references was found. Nothing written.'
        )
    distinct_source_count = len(selected)
    if distinct_source_count == 1:
        if not ALLOW_SINGLE_REAL_SOURCE_FOR_TWO_TEST_ROWS:
            raise ValueError(
                'Only 1 distinct genuine Phase 2 CIB source found. 2 distinct '
                'real cases were required; NOTHING written.'
            )
        # Explicit load-test fallback: same original values/history, but two
        # synthetic entities and BREACH_IDs. NOT two independent genuine cases.
        selected = [selected[0], selected[0]]
        print('IMPORTANT: ONE distinct real CIB source -> TWO synthetic test records.')
        print('These TWO records do NOT count as two independent real alerts.')
    print('Distinct real Retail CIB sources used:', distinct_source_count)
    print('Real-derived CIB rows planned:', len(selected))
    for index, case in enumerate(selected, 1):
        print('CIB template {}: export {} | EWI {} | history {} | exposures {}'.format(
            index, case['stamp'], case['breach'].iloc[0]['EWI_NAME'],
            len(case['historic']), len(case['exposure'])))
    return selected


def _val(value):
    if pd.isna(value):
        return ''
    value = str(value).strip()
    return '' if value.upper() in ('MISS', 'NAN', 'NONE', '<NA>') else value


def inject_real_cib(output, origins, cases):
    """Replace the 2 simulated CIB entries; keep real indicators/metrics/history.

    All personal names, real IDs, GCC/customer references and exposure amounts
    remain synthetic. Rebase daily/monthly dates to a single UAT batch date.
    """
    if len(cases) != 2:
        raise ValueError('Exactly 2 genuine Retail-origin CIB cases required')
    bn, hn, en, xn = tuple(FIELDS)
    bnrows, hnrows, enrows, xnrows = (output[n] for n in FIELDS)
    test_day = pd.Timestamp(ALERT_DATE)
    months = [(test_day.replace(day=1) - pd.offsets.MonthEnd(m)).strftime('%Y-%m-%d')
              for m in range(6, 0, -1)]
    ids = []
    for case_index, case in enumerate(cases):
        alert_index = RETAIL_ALERTS + CORPORATE_MODEL_ALERTS + case_index
        base = bnrows[alert_index]
        entity_id, breach_id = base['ENTITY_ID'], base['BREACH_ID']
        ids.append((entity_id, breach_id))
        real = case['breach'].iloc[0]
        # Keep real alert logic and real EWI values without customer identifiers.
        for col in ('EWI_NAME','EWI_VALUE_1','THRESHOLD_1','DEVIATION_1',
                    'EWI_VALUE_2','THRESHOLD2','DEVIATION2','MAIN_EXPLN',
                    'EWI_methodology','Entity_Breaches_12M','ENTITY_BREACHES_TOTAL'):
            if col in real.index:
                base[col] = _val(real[col])
        base['BUSINESS_LINE'] = 'Corporate & Institutional Banking'
        base['SEGMENT'] = 'Corporate'
        base['ALERT_DATE'] = ALERT_DATE
        base['DAY'] = ALERT_DATE
        base['error'] = ''  # no real RM validation errors on synthetic RM IDs
        # Real historic values + thresholds + changes, with rebased month ends.
        source_hist = case['historic'].copy()
        source_hist['_DATE'] = _parse_source_history_dates(source_hist['HISTORIC_DATE'])
        source_hist = source_hist.sort_values('_DATE')
        for hist_idx, (_, rec) in enumerate(source_hist.iterrows()):
            new_row = row_for(hn, BREACH_ID=breach_id, HISTORIC_DATE=months[hist_idx],
                              ALERT_DATE=ALERT_DATE, DAY=ALERT_DATE)
            for col in FIELDS[hn]:
                if col not in {'BREACH_ID','HISTORIC_DATE','ALERT_DATE','DAY','error'} and col in rec.index:
                    new_row[col] = _val(rec[col])
            hnrows.append(new_row)
        # Preserve only categoric non-identifying static attributes; all personal,
        # financial and customer ID fields stay synthetic from the base generator.
        entity_out = enrows[alert_index]
        real_entity = case['entity'].iloc[0]
        for col in ('INTERNAL_RATING', 'NACE2'):
            if col in real_entity.index:
                entity_out[col] = _val(real_entity[col])
        entity_out['SEGMENT'] = 'Corporate'
        # Replace synthetic source exposures by the actual count/product/status,
        # while keeping synthetic exposure IDs and synthetic money figures.
        synthetic_out = [z for z in xnrows if z['ENTITY_ID'] == entity_id]
        for expidx, (_, erow) in enumerate(case['exposure'].iterrows(), 1):
            z = dict(synthetic_out[(expidx - 1) % len(synthetic_out)])
            z['EXPOSURE_ID'] = '95{:012d}{:02d}'.format(alert_index + 1, expidx)
            z['ENTITY_ID'] = entity_id
            for col in ('PRODUCT_TYPE','CURRENCY','IFRS9_STAGE',
                        'DAYS_PAST_DUE','PERFORMING_STATUS'):
                if col in erow.index:
                    z[col] = _val(erow[col])
            z['DAY'] = ALERT_DATE
            z['error'] = ''
            # Monetary values are still synthetic; preserve coherent net/gross/limit.
            amt = 2000.0 + (alert_index % 179) * 117.39 + expidx * 333.17
            z['NET_EXPOSURE_AMOUNT_EUR'] = number(amt)
            z['GROSS_EXPOSURE_AMOUNT_EUR'] = number(amt + 250)
            z['LIMIT_AMOUNT'] = number(amt + 1500)
            z['COLLATERAL_AMOUNT'] = ''
            xnrows.append(z)
    target_eids = {x[0] for x in ids}
    # Keep the 59,988 non-CIB synthetic historic rows and replace the 12 CIB
    # placeholders with the 12 records derived from the original real alerts.
    original_h = hnrows[:RETAIL_ALERTS * 6 + CORPORATE_MODEL_ALERTS * 6]
    new_cib_h = hnrows[-12:]
    output[hn] = original_h + new_cib_h
    real_exposure_count = sum(len(c['exposure']) for c in cases)
    old_exposures = xnrows[:-real_exposure_count]
    real_exposures = xnrows[-real_exposure_count:]
    output[xn] = [z for z in old_exposures if z['ENTITY_ID'] not in target_eids] + real_exposures
    origins.pop('Retail-origin CIB simulation')
    distinct_sources = len({(case['stamp'], str(case['breach'].iloc[0]['BREACH_ID']))
                            for case in cases})
    origins['Retail-origin CIB derived from {} distinct genuine source(s)'.format(
        distinct_sources)] = 2
    print('AUDIT: {} distinct REAL CIB alert source(s) used to create 2 UAT rows'.format(
        distinct_sources))
    return output, origins

def csv_bytes(records, columns):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise",
                            delimiter=EXPORT_DELIMITER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode(EXPORT_ENCODING)


def write_to_dataiku_folder(output):
    if not TEST_OUTPUT_FOLDER_ID:
        raise ValueError("Set TEST_OUTPUT_FOLDER_ID to a NEW, ISOLATED managed folder in DEV")
    import dataiku
    folder = dataiku.Folder(TEST_OUTPUT_FOLDER_ID)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in FIELDS:
        filename = "{}_{}.csv".format(name, timestamp)
        folder.upload_stream(filename, io.BytesIO(csv_bytes(output[name], FIELDS[name])))
        print("UPLOADED", filename, len(output[name]), "rows")
    print("FOUR CSV written to test folder ONLY. NOT submitted to UAT.")


def main():
    print("=== BIL EWS PHASE2 10K SYNTHETIC LOAD TEST ===")
    cases = discover_real_cib()
    output, origins = build()
    output, origins = inject_real_cib(output, origins, cases)
    validate(output, origins)
    if CHECK_ONLY:
        print("CHECK_ONLY=True: NO files generated or uploaded")
    else:
        write_to_dataiku_folder(output)


if __name__ == "__main__":
    main()
