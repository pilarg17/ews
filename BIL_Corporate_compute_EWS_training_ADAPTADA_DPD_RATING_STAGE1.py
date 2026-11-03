# -*- coding: utf-8 -*-
"""BIL Corporate Validation: adapt the EXISTING compute_EWS_training recipe.

Stage 1 (DPD + Rating only), using COMPLETE ORIGINAL HISTORIES in Snowflake:
vars_entity_level, vars_loan_level, DPD_Arrears_Daily_Loan.

No QA data. No need for Retail-only sources that currently fail.
CHECK and PILOT are read-only. BUILD replaces the EXISTING EWS_training table
and its schema; the code requires an explicit safety switch to do this.

This is a STAGE-1 training feature panel, not the complete Retail training
schema, final Corporate EWI methodology, or a trained model.

Backup the original Python recipe before pasting this replacement.


Limits / caveats:
  * Universe = legal persons found in at least one daily DPD or loan source.
    This does NOT include legal persons with neither input on a given day.
  * GEC exposure from GCC_PARENT_ID is a candidate, not bank-certified GEC.
    It is NULL when any known loan in that source-group is missing the amount.
    It cannot establish that all group members are present in the loan source.
  * GCC_PD_CRR may reflect the entity's stored group PD at an SCD version
    rather than a separately reconstructed parent state per day.
  * Rating 12M compares against a real historical version as-of DAY-12 months.
    The "best 12M" rules and 5-business-day persistence are NOT claimed.
  * Same-day deterioration fields are diagnostics, NOT a predictive target.
  * CR01 and GCC02 depend on exact best rating over 12 months and are not
    implemented in this first build. 5-business-day persistence is pending.
"""

import re
import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2

PROJECT = "VALIDATIONEWSCORPORATEPORTFOLIODAILY"
CONNECTION = "DEV_AA_TEAMS"
OUTPUT_NAME = "EWS_training"  # EXISTING output of original training recipe

# The historical preflight already passed; start with PILOT. BUILD is guarded.
RUN_MODE = "PILOT"  # PILOT first; BUILD only after validation (CHECK optional)
CONFIRM_REPLACE_EWS_TRAINING = False  # set True ONLY after inspecting PILOT
PILOT_START = "2026-10-20"
PILOT_END = "2026-10-26"

# BUILD None = entire available daily history (large Snowflake table).
BUILD_START = None
BUILD_END = None

assert RUN_MODE in {"CHECK", "PILOT", "BUILD"}

proj = dataiku.api_client().get_project(PROJECT)


def ds_table(dataset_name):
    conf = proj.get_dataset(dataset_name).get_settings().get_raw()
    params = conf.get("params", {})
    if params.get("connection") != CONNECTION:
        raise ValueError("Unexpected connection for " + dataset_name)

    def quote_ident(x):
        x = str(x).replace("${projectKey}", PROJECT)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", x):
            raise ValueError("Unsafe SQL identifier: " + repr(x))
        return '"{}"'.format(x)

    cols = {str(c["name"]).upper() for c in dataiku.Dataset(dataset_name,
                         project_key=PROJECT).read_schema()}
    return quote_ident(params["schema"]) + "." + quote_ident(params["table"]), cols


E, ec = ds_table("vars_entity_level")
L, lc = ds_table("vars_loan_level")
D, dc = ds_table("DPD_Arrears_Daily_Loan")

for name, cols, required in [
    ("vars_entity_level", ec,
       {"ENTITY_ID", "TEK_STRT_DTE", "TEK_END_DTE", "PD_CRR", "GCC_PARENT_ID", "GCC_PD_CRR"}),
    ("vars_loan_level", lc,
       {"ENTITY_ID", "INSTRUMENT_ID", "MAP_DTE", "NOMINAL_AMOUNT", "GROSS_CARRYING_AMOUNT",
        "DEFAULT_STATUS", "FORBEARANCE_STATUS", "DIM1"}),
    ("DPD_Arrears_Daily_Loan", dc, {"ENTITY_ID", "INSTRUMENT_ID", "MAP_DTE", "DPD", "ARREARS_AMOUNT"}),
]:
    absent = sorted(required - cols)
    if absent:
        raise ValueError(name + " missing: " + ", ".join(absent))


# The rating scale below is copied from the current Corporate daily-rating recipe,
# NOT invented for this notebook. It maps PD (0..1) to increasing risk notches.
RATING_SCALE = [
    (0.00000, 0.00005, "AAA+", 1),
    (0.00005, 0.00015, "AAA",  2),
    (0.00015, 0.00025, "AA+",  3),
    (0.00025, 0.00035, "AA",   4),
    (0.00035, 0.00045, "AA-",  5),
    (0.00045, 0.00055, "A+",   6),
    (0.00055, 0.00065, "A",    7),
    (0.00065, 0.00125, "A-",   8),
    (0.00125, 0.00260, "BBB+", 9),
    (0.00260, 0.00525, "BBB", 10),
    (0.00525, 0.00795, "BBB-",11),
    (0.00795, 0.01015, "BB+", 12),
    (0.01015, 0.01915, "BB",  13),
    (0.01915, 0.03315, "BB-", 14),
    (0.03315, 0.06510, "B+",  15),
    (0.06510, 0.11455, "B",   16),
    (0.11455, 0.22355, "B-",  17),
    (0.22355, 0.99975, "CCC", 18),
    (0.99975, 1.00500, "D",   19),
]


def rating_case(expr, col="NOTCH"):
    parts = []
    for low, high, name, notch in RATING_SCALE:
        value = str(notch) if col == "NOTCH" else "'{}'".format(name)
        parts.append("WHEN {x} >= {lo} AND {x} < {hi} THEN {v}".format(
            x=expr, lo=low, hi=high, v=value))
    return "CASE " + " ".join(parts) + " ELSE NULL END"


def date_condition(alias, start, end):
    out = []
    if start is not None:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", start)
        out.append("{a}.MAP_DTE >= DATE '{d}'".format(a=alias, d=start))
    if end is not None:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", end)
        out.append("{a}.MAP_DTE < DATEADD(day, 1, DATE '{d}')".format(a=alias, d=end))
    return " AND ".join(out) if out else "1=1"


def build_sql(start=None, end=None):
    filt_d = date_condition("d", start, end)
    filt_l = date_condition("l", start, end)
    pd_expr = "TRY_TO_DOUBLE(NULLIF(e.PD_CRR, 11111111111))"
    gcc_expr = "TRY_TO_DOUBLE(e.GCC_PD_CRR)"
    # Point-in-time value 12 months ago, not best rating during 12 months.
    # A missing baseline is kept NULL.
    query = r'''
WITH
D_LOAN_DAY AS (
 SELECT TO_DATE(d.MAP_DTE) AS MAP_DTE,
        d.ENTITY_ID::VARCHAR AS ENTITY_ID,
        d.INSTRUMENT_ID::VARCHAR AS INSTRUMENT_ID,
        MAX(TRY_TO_DOUBLE(d.DPD)) AS DPD,
        MAX(TRY_TO_DOUBLE(d.ARREARS_AMOUNT)) AS ARREARS_AMOUNT,
        COUNT(*) AS SOURCE_ROWS
 FROM {D} d WHERE {filt_d}
 GROUP BY 1,2,3
),
D_DAILY AS (
 SELECT MAP_DTE, ENTITY_ID,
        MAX(DPD) AS DPD_MAX,
        SUM(ARREARS_AMOUNT) AS ARREARS_AMOUNT,
        SUM(SOURCE_ROWS) AS DPD_SOURCE_ROWS,
        COUNT_IF(DPD IS NOT NULL) AS DPD_KNOWN_ROWS,
        SUM(SOURCE_ROWS - 1) AS DPD_SOURCE_DUPLICATES
 FROM D_LOAN_DAY GROUP BY 1,2
),
L_INSTRUMENT AS (
 SELECT TO_DATE(l.MAP_DTE) AS MAP_DTE, l.ENTITY_ID::VARCHAR AS ENTITY_ID,
        l.INSTRUMENT_ID::VARCHAR AS INSTRUMENT_ID,
        MAX(TRY_TO_DOUBLE(l.NOMINAL_AMOUNT)) AS NOMINAL_AMOUNT,
        MAX(TRY_TO_DOUBLE(l.GROSS_CARRYING_AMOUNT)) AS GCA_AMOUNT,
        MAX(IFF(TRY_TO_NUMBER(l.DEFAULT_STATUS) IN (18,19,20),1,0)) AS DEFAULT_FLAG,
        MAX(IFF(TRY_TO_NUMBER(l.FORBEARANCE_STATUS) IN (3,4,5),1,0)) AS FORBEARANCE_FLAG,
        MAX(IFF(l.DIM1::VARCHAR IN ('S2','S3','S3S','S3R'),1,0)) AS DIM1_FLAG,
        COUNT(*) AS SOURCE_DUP_ROWS
 FROM {L} l WHERE {filt_l}
 GROUP BY 1,2,3
),
L_ENTITY AS (
 SELECT MAP_DTE,ENTITY_ID,
        COUNT(*) AS LOAN_COUNT,
        SUM(NOMINAL_AMOUNT) AS NOMINAL_SUM,
        SUM(GCA_AMOUNT) AS GCA_SUM,
        COUNT_IF(NOMINAL_AMOUNT IS NOT NULL) AS NOMINAL_KNOWN_LOANS,
        COUNT_IF(GCA_AMOUNT IS NOT NULL) AS GCA_KNOWN_LOANS,
        SUM(SOURCE_DUP_ROWS-1) AS SOURCE_DUPLICATE_LOAN_ROWS,
        MAX(DEFAULT_FLAG) AS DEFAULT_FLAG,
        MAX(FORBEARANCE_FLAG) AS FORBEARANCE_FLAG,
        MAX(DIM1_FLAG) AS DIM1_FLAG
 FROM L_INSTRUMENT GROUP BY 1,2
),
KEYS AS (
 SELECT MAP_DTE, ENTITY_ID FROM D_DAILY
 UNION
 SELECT MAP_DTE, ENTITY_ID FROM L_ENTITY
),
HIST AS (
 SELECT e.ENTITY_ID::VARCHAR AS ENTITY_ID,
        TO_DATE(e.TEK_STRT_DTE) AS START_DATE,
        IFF(e.TEK_END_DTE IS NULL, DATE '9999-12-31', TO_DATE(e.TEK_END_DTE)) AS END_DATE,
        NULLIF(TRIM(e.GCC_PARENT_ID::VARCHAR),'') AS GCC_PARENT_ID,
        IFF({pd_expr} BETWEEN 0 AND 1, {pd_expr}, NULL) AS PD_CRR,
        IFF({gcc_expr} BETWEEN 0 AND 1, {gcc_expr}, NULL) AS GCC_PD_CRR
 FROM {E} e
 WHERE e.ENTITY_ID IS NOT NULL AND e.TEK_STRT_DTE IS NOT NULL
),
HIST_NOTCH AS (
 SELECT h.*,
        {client_case} AS CLIENT_NOTCH,
        {gcc_case} AS GCC_NOTCH,
        {client_label} AS CLIENT_RATING,
        {gcc_label} AS GCC_RATING
 FROM HIST h
),
ENTITY_CURRENT AS (
 SELECT k.MAP_DTE,k.ENTITY_ID,
        h.GCC_PARENT_ID,h.PD_CRR,h.GCC_PD_CRR,
        h.CLIENT_NOTCH,h.GCC_NOTCH,
        h.CLIENT_RATING,h.GCC_RATING,
        h.START_DATE AS ENTITY_VERSION_START,
        h.END_DATE AS ENTITY_VERSION_END
 FROM KEYS k
 JOIN HIST_NOTCH h ON k.ENTITY_ID=h.ENTITY_ID
   AND k.MAP_DTE>=h.START_DATE AND k.MAP_DTE<=h.END_DATE
 QUALIFY ROW_NUMBER() OVER (
   PARTITION BY k.MAP_DTE,k.ENTITY_ID
   ORDER BY h.START_DATE DESC,h.END_DATE DESC
 )=1
),
ENTITY_LOOKBACK AS (
 SELECT a.MAP_DTE,a.ENTITY_ID,
        h.CLIENT_NOTCH AS CLIENT_NOTCH_12M,
        h.GCC_NOTCH AS GCC_NOTCH_12M,
        h.START_DATE AS LOOKBACK_VERSION_START,
        IFF(h.ENTITY_ID IS NOT NULL AND h.END_DATE>=a.REF_DATE,1,0) AS LOOKBACK_VALID
 FROM (
   SELECT MAP_DTE,ENTITY_ID,DATEADD(month,-12,MAP_DTE) AS REF_DATE
   FROM ENTITY_CURRENT
 ) a
 ASOF JOIN HIST_NOTCH h
   MATCH_CONDITION (a.REF_DATE >= h.START_DATE)
   ON a.ENTITY_ID=h.ENTITY_ID
),
GEC_DAY AS (
 SELECT l.MAP_DTE,COALESCE(e.GCC_PARENT_ID,e.ENTITY_ID) AS GEC_KEY,
        SUM(l.NOMINAL_SUM) AS GEC_NOMINAL_SOURCE_SUM,
        SUM(l.GCA_SUM) AS GEC_GCA_SOURCE_SUM,
        SUM(l.LOAN_COUNT) AS GEC_SOURCE_LOAN_COUNT,
        SUM(l.NOMINAL_KNOWN_LOANS) AS GEC_NOMINAL_KNOWN_LOANS,
        SUM(l.GCA_KNOWN_LOANS) AS GEC_GCA_KNOWN_LOANS,
        SUM(l.SOURCE_DUPLICATE_LOAN_ROWS) AS GEC_SOURCE_DUPLICATES
 FROM L_ENTITY l
 JOIN ENTITY_CURRENT e ON l.MAP_DTE=e.MAP_DTE AND l.ENTITY_ID=e.ENTITY_ID
 GROUP BY 1,2
),
BASE AS (
 SELECT e.MAP_DTE,e.ENTITY_ID,
        e.GCC_PARENT_ID,COALESCE(e.GCC_PARENT_ID,e.ENTITY_ID) AS GEC_KEY,
        e.PD_CRR,e.GCC_PD_CRR,e.CLIENT_RATING,e.GCC_RATING,
        e.CLIENT_NOTCH,e.GCC_NOTCH,
        IFF(lb.LOOKBACK_VALID=1,lb.CLIENT_NOTCH_12M,NULL) AS CLIENT_NOTCH_12M,
        IFF(lb.LOOKBACK_VALID=1,lb.GCC_NOTCH_12M,NULL) AS GCC_NOTCH_12M,
        IFF(lb.LOOKBACK_VALID=1,1,0) AS HAS_12M_HISTORY,
        d.DPD_MAX, d.ARREARS_AMOUNT,d.DPD_SOURCE_ROWS,d.DPD_KNOWN_ROWS,d.DPD_SOURCE_DUPLICATES,
        l.LOAN_COUNT,l.NOMINAL_SUM,l.GCA_SUM,
        l.SOURCE_DUPLICATE_LOAN_ROWS,
        l.DEFAULT_FLAG,l.FORBEARANCE_FLAG,l.DIM1_FLAG,
        g.GEC_SOURCE_LOAN_COUNT,g.GEC_NOMINAL_KNOWN_LOANS,g.GEC_GCA_KNOWN_LOANS,
        g.GEC_SOURCE_DUPLICATES,
        IFF(g.GEC_SOURCE_LOAN_COUNT>0
            AND g.GEC_SOURCE_LOAN_COUNT=g.GEC_NOMINAL_KNOWN_LOANS
            AND g.GEC_SOURCE_DUPLICATES=0,
            g.GEC_NOMINAL_SOURCE_SUM,NULL) AS GEC_NOMINAL_CANDIDATE,
        IFF(g.GEC_SOURCE_LOAN_COUNT>0
            AND g.GEC_SOURCE_LOAN_COUNT=g.GEC_GCA_KNOWN_LOANS
            AND g.GEC_SOURCE_DUPLICATES=0,
            g.GEC_GCA_SOURCE_SUM,NULL) AS GEC_GCA_CANDIDATE
 FROM ENTITY_CURRENT e
 LEFT JOIN ENTITY_LOOKBACK lb ON e.MAP_DTE=lb.MAP_DTE AND e.ENTITY_ID=lb.ENTITY_ID
 LEFT JOIN D_DAILY d ON e.MAP_DTE=d.MAP_DTE AND e.ENTITY_ID=d.ENTITY_ID
 LEFT JOIN L_ENTITY l ON e.MAP_DTE=l.MAP_DTE AND e.ENTITY_ID=l.ENTITY_ID
 LEFT JOIN GEC_DAY g ON e.MAP_DTE=g.MAP_DTE AND COALESCE(e.GCC_PARENT_ID,e.ENTITY_ID)=g.GEC_KEY
),
FEATURES AS (
 SELECT b.*,
        b.CLIENT_NOTCH-b.CLIENT_NOTCH_12M AS CLIENT_DOWNGRADE_12M,
        b.GCC_NOTCH-b.GCC_NOTCH_12M AS GCC_DOWNGRADE_12M,
        LAG(b.MAP_DTE) OVER (PARTITION BY b.ENTITY_ID ORDER BY b.MAP_DTE) AS PREV_MAP_DTE,
        LAG(b.DPD_MAX) OVER (PARTITION BY b.ENTITY_ID ORDER BY b.MAP_DTE) AS PREV_DPD_MAX,
        LAG(b.GEC_NOMINAL_CANDIDATE) OVER (PARTITION BY b.ENTITY_ID ORDER BY b.MAP_DTE) AS PREV_GEC_NOMINAL,
        LAG(b.GEC_GCA_CANDIDATE) OVER (PARTITION BY b.ENTITY_ID ORDER BY b.MAP_DTE) AS PREV_GEC_GCA
 FROM BASE b
)
SELECT f.*,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 5 THEN 1 ELSE 0 END AS DPD_GT_5,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 10 THEN 1 ELSE 0 END AS DPD_GT_10,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 15 THEN 1 ELSE 0 END AS DPD_GT_15,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 20 THEN 1 ELSE 0 END AS DPD_GT_20,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 25 THEN 1 ELSE 0 END AS DPD_GT_25,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 30 THEN 1 ELSE 0 END AS DPD_GT_30,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 35 THEN 1 ELSE 0 END AS DPD_GT_35,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 40 THEN 1 ELSE 0 END AS DPD_GT_40,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 45 THEN 1 ELSE 0 END AS DPD_GT_45,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 50 THEN 1 ELSE 0 END AS DPD_GT_50,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 55 THEN 1 ELSE 0 END AS DPD_GT_55,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX > 60 THEN 1 ELSE 0 END AS DPD_GT_60,
       CASE WHEN f.CLIENT_NOTCH IS NULL OR f.CLIENT_DOWNGRADE_12M IS NULL THEN NULL WHEN f.CLIENT_NOTCH>11 AND f.CLIENT_DOWNGRADE_12M>=5 THEN 1 ELSE 0 END AS CR02_CANDIDATE,
       CASE WHEN f.CLIENT_NOTCH IS NULL OR f.CLIENT_DOWNGRADE_12M IS NULL THEN NULL WHEN f.CLIENT_NOTCH>13 AND f.CLIENT_DOWNGRADE_12M>=3 THEN 1 ELSE 0 END AS CR03_CANDIDATE,
       CASE WHEN f.CLIENT_NOTCH IS NULL OR f.CLIENT_DOWNGRADE_12M IS NULL THEN NULL WHEN f.CLIENT_NOTCH>14 AND f.CLIENT_DOWNGRADE_12M>=2 THEN 1 ELSE 0 END AS CR04_CANDIDATE,
       CASE WHEN f.CLIENT_NOTCH IS NULL THEN NULL WHEN f.CLIENT_NOTCH>16 THEN 1 ELSE 0 END AS CR05_CANDIDATE,
       CASE WHEN f.GCC_NOTCH IS NULL OR f.GCC_DOWNGRADE_12M IS NULL THEN NULL WHEN f.GCC_NOTCH>15 AND f.GCC_DOWNGRADE_12M>=1 THEN 1 ELSE 0 END AS GCC01_CANDIDATE,
       CASE WHEN f.GCC_DOWNGRADE_12M IS NULL OR f.CLIENT_NOTCH IS NULL THEN NULL WHEN f.GCC_DOWNGRADE_12M>=3 AND f.CLIENT_NOTCH>13 THEN 1 ELSE 0 END AS GCC03_CANDIDATE,
       CASE WHEN f.GCC_DOWNGRADE_12M IS NULL OR f.CLIENT_DOWNGRADE_12M IS NULL THEN NULL WHEN f.GCC_DOWNGRADE_12M>=1 AND f.CLIENT_DOWNGRADE_12M>=5 THEN 1 ELSE 0 END AS GCC04_CANDIDATE,
       CASE WHEN f.GCC_DOWNGRADE_12M IS NULL OR f.CLIENT_DOWNGRADE_12M IS NULL THEN NULL WHEN f.GCC_DOWNGRADE_12M>=2 AND f.CLIENT_DOWNGRADE_12M>=4 THEN 1 ELSE 0 END AS GCC05_CANDIDATE,
       CASE WHEN f.GCC_DOWNGRADE_12M IS NULL OR f.CLIENT_DOWNGRADE_12M IS NULL THEN NULL WHEN f.GCC_DOWNGRADE_12M>=4 AND f.CLIENT_DOWNGRADE_12M>=3 THEN 1 ELSE 0 END AS GCC06_CANDIDATE,
       CASE WHEN f.GCC_DOWNGRADE_12M IS NULL OR f.CLIENT_DOWNGRADE_12M IS NULL THEN NULL WHEN f.GCC_DOWNGRADE_12M>=5 AND f.CLIENT_DOWNGRADE_12M>=1 THEN 1 ELSE 0 END AS GCC07_CANDIDATE,
       CASE WHEN f.PD_CRR IS NULL THEN NULL WHEN f.PD_CRR=1 THEN 1 ELSE 0 END AS TGT_PD_CRR_SAMEDAY,
       CASE WHEN f.DPD_MAX IS NULL THEN NULL WHEN f.DPD_MAX>=20 THEN 1 ELSE 0 END AS TGT_DPD_20_SAMEDAY,
       f.DEFAULT_FLAG AS TGT_DEFAULT_SAMEDAY,
       f.FORBEARANCE_FLAG AS TGT_FORBEARANCE_SAMEDAY,
       f.DIM1_FLAG AS TGT_DIM1_SAMEDAY
FROM FEATURES f
'''
    return query.format(
        D=D, L=L, E=E, filt_d=filt_d, filt_l=filt_l,
        pd_expr=pd_expr, gcc_expr=gcc_expr,
        client_case=rating_case('h.PD_CRR','NOTCH'),
        gcc_case=rating_case('h.GCC_PD_CRR','NOTCH'),
        client_label=rating_case('h.PD_CRR','LABEL'),
        gcc_label=rating_case('h.GCC_PD_CRR','LABEL'),
    ).strip()


print("=== CORPORATE: EXISTING EWS_training | DPD + Rating, stage 1 ===")
print("Mode:", RUN_MODE)
print("Sources:", "vars_entity_level", "vars_loan_level", "DPD_Arrears_Daily_Loan")
print("No QA sources, no Retail-only sources. Historical sources never overwritten.")
print("WARNING: BUILD replaces existing EWS_training table AND schema.")

executor = SQLExecutor2(connection=CONNECTION)

if RUN_MODE == "CHECK":
    for name, table in [("Entity", E), ("Loan", L), ("DPD", D)]:
        q = "SELECT 1 FROM {} LIMIT 1".format(table)
        x = executor.query_to_df(q)
        print(name, "source accessible:", bool(len(x)))
    print("CHECK passed. Change RUN_MODE='PILOT' to test a few days.")

elif RUN_MODE == "PILOT":
    q = build_sql(PILOT_START, PILOT_END)
    # Limit the returned rows; computation remains in Snowflake.
    preview = executor.query_to_df("SELECT * FROM (" + q + ") LIMIT 5")
    # Do not print actual client identifiers in notebook logs.
    privacy_cols = [x for x in preview.columns if str(x).upper() in
                    {"ENTITY_ID", "GCC_PARENT_ID", "GEC_KEY"}]
    print("PILOT rows shown:", len(preview), "/ hidden identifier columns:", privacy_cols)
    print(preview.drop(columns=privacy_cols).to_string(index=False))
    check = executor.query_to_df("""
        SELECT COUNT(*) AS N_ROWS,
               COUNT(DISTINCT ENTITY_ID || '|' || TO_VARCHAR(MAP_DTE)) AS N_ENTITY_DAYS,
               MIN(MAP_DTE) AS MIN_DAY, MAX(MAP_DTE) AS MAX_DAY,
               COUNT_IF(DPD_MAX IS NOT NULL) AS N_DPD_KNOWN,
               COUNT_IF(CLIENT_NOTCH IS NOT NULL) AS N_CLIENT_RATING,
               COUNT_IF(HAS_12M_HISTORY=1) AS N_HAS_12M,
               COUNT_IF(SOURCE_DUPLICATE_LOAN_ROWS>0) AS N_LOAN_DUP_DAY
        FROM (""" + q + ")")
    print("PILOT aggregate QA (no client identifiers):")
    print(check.to_string(index=False))
    if len(check) and int(check.iloc[0]['N_ROWS']) != int(check.iloc[0]['N_ENTITY_DAYS']):
        raise ValueError("Duplicated ENTITY_ID + MAP_DTE in PILOT")
    print("PILOT complete. No output written.")

elif RUN_MODE == "BUILD":
    # Existing training recipe must declare EWS_training as its ONLY output
    # for this stage. Back up code and Flow config before modifying the recipe.
    # exec_recipe_fragment replaces the existing Snowflake table AND schema.
    if not CONFIRM_REPLACE_EWS_TRAINING:
        raise RuntimeError(
            "BUILD prevented. Set CONFIRM_REPLACE_EWS_TRAINING=True "
            "only after PILOT and checking downstream schema compatibility."
        )
    from dataiku import recipe
    # Normalize dataset objects and strings.
    normalized = {str(getattr(x, "name", x)).split(".")[-1] for x in
                  recipe.get_outputs(object_type="DATASET")}
    if normalized != {OUTPUT_NAME}:
        raise RuntimeError("Wrong recipe output, refusing write: " + repr(normalized))
    output = dataiku.Dataset(OUTPUT_NAME)
    print("REPLACING existing EWS_training with DPD + Rating stage 1:", OUTPUT_NAME)
    SQLExecutor2.exec_recipe_fragment(
        output_dataset=output,
        query=build_sql(BUILD_START, BUILD_END),
        overwrite_output_schema=True,
    )
    print("BUILD finished. Verify row counts, date coverage, uniqueness and nulls.")
    print("Do NOT run downstream EWS_Results without adapting it to the stage-1 schema.")
