# -*- coding: utf-8 -*-
"""
CORPORATE B01 | FULL GEC SOURCE & METHODOLOGY QA | SINGLE DATAIKU NOTEBOOK CELL
READ ONLY. Never calls recipe/build, never writes datasets.
Checks historical GEC grouping, loan coverage, duplicate instruments, missing nominal,
all-member PD/rating, simple vs exposure-weighted averages, parent-vs-group differences,
existing training reconciliation, and DPD >15 x 350k materiality sensitivity.
Only aggregate statistics are printed. No customer identifiers are displayed.
"""
import re
import dataiku
import pandas as pd
from IPython.display import display
from dataiku.core.sql import SQLExecutor2

PROJECT = "VALIDATIONEWSCORPORATEPORTFOLIODAILY"
CONNECTION = "DEV_AA_TEAMS"
DATES = ["2024-11-11", "2025-10-29", "2026-04-29", "2026-08-31", "2026-09-30", "2026-10-15", "2026-10-26", "2026-10-29"]
TRAIN_NAME = "EWS_training_B01_DPD_RATING"
LOAN_NAME = "vars_loan_level"
ENTITY_NAME = "vars_entity_level"
GEC_NAMES = ("HEAD_GEC_PARENT__ID", "HEAD_GEC_PARENT_ID")

pd.set_option("display.max_columns", 35)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 80)
client = dataiku.api_client()
project = client.get_project(PROJECT)
ex = SQLExecutor2(connection=CONNECTION)

def qi(x):
    x = str(x).replace("${projectKey}", PROJECT)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", x):
        raise ValueError("Unsafe SQL identifier: " + repr(x))
    return '"' + x + '"'

def name_map(cols):
    return {str(c).upper(): str(c) for c in cols}

def get_col(cols, *names):
    d = name_map(cols)
    for n in names:
        if n.upper() in d:
            return d[n.upper()]
    return None

def dataset_source(name):
    ds = dataiku.Dataset(name, project_key=PROJECT)
    cols = [c["name"] for c in ds.read_schema()]
    cfg = project.get_dataset(name).get_settings().get_raw()
    pars = cfg.get("params") or {}
    if pars.get("connection") != CONNECTION:
        raise ValueError(name + ": source not on " + CONNECTION)
    table = qi(pars["schema"]) + "." + qi(pars["table"])
    return {"name": name, "table": table, "cols": cols}

def direct_source(table):
    # LIMIT 0: introspects schema without reading rows
    df = ex.query_to_df("SELECT * FROM " + table + " WHERE 1=0")
    return {"name": table, "table": table, "cols": list(df.columns)}

def require(src, cols):
    missing = [x for x in cols if get_col(src["cols"], x) is None]
    if missing:
        raise ValueError(src["name"] + " missing " + str(missing))

def run(title, sql):
    print("\n" + "="*105 + "\n" + title + "\n" + "="*105)
    try:
        df = ex.query_to_df(sql)
        if df.empty:
            print("No rows returned")
        else:
            display(df)
        return df
    except Exception as err:
        print("BLOCKED / QUERY ERROR:", str(err)[:900])
        return None

def expr_col(alias, actual):
    return alias + "." + qi(actual)

print("="*110)
print("CORPORATE B01 - FULL GEC QA | READ ONLY |", ", ".join(DATES))
print("Connection:", CONNECTION, "| No dataset writes / no Flow runs / no identifiers shown")
print("="*110)
entity_ds, loan_ds, train_ds = [dataset_source(x) for x in (ENTITY_NAME, LOAN_NAME, TRAIN_NAME)]
require(loan_ds, ("ENTITY_ID", "INSTRUMENT_ID", "MAP_DTE", "NOMINAL_AMOUNT"))
require(train_ds, ("ENTITY_ID", "MAP_DTE", "GEC_NOMINAL_CANDIDATE", "GEC_KEY", "GCC_PD_CRR", "GCC_NOTCH", "DPD_MAX", "DPD_CROSS_GT_15"))

# Inspect both the original entity source and the prepared entity data.
source_inventory = []
for s in [entity_ds]:
    source_inventory.append({"SOURCE": s["name"], "GEC_COLUMN": get_col(s["cols"], *GEC_NAMES) or "MISSING",
                             "GCC_COLUMN": get_col(s["cols"], "GCC_PARENT_ID") or "MISSING",
                             "PD_COLUMN": get_col(s["cols"], "PD_CRR") or "MISSING",
                             "TIME_MODE": "ASOF_VALIDITY" if get_col(s["cols"], "TEK_STRT_DTE") and get_col(s["cols"], "TEK_END_DTE") else "OTHER"})
raw_sources = []
for table in ["BDL_CREDIT_INSIGHT_PLY.ENTITY_VW", "BDL_CREDIT_INSIGHT_PLY.ENTITY_DAY_VW"]:
    try:
        s = direct_source(table)
        raw_sources.append(s)
        source_inventory.append({"SOURCE": table, "GEC_COLUMN": get_col(s["cols"], *GEC_NAMES) or "MISSING",
                                 "GCC_COLUMN": get_col(s["cols"], "GCC_PARENT_ID") or "MISSING",
                                 "PD_COLUMN": get_col(s["cols"], "PD_CRR") or "MISSING",
                                 "TIME_MODE": "ASOF_VALIDITY" if get_col(s["cols"], "TEK_STRT_DTE") and get_col(s["cols"], "TEK_END_DTE") else ("EXACT_DAILY" if get_col(s["cols"], "MAP_DTE") else "OTHER")})
    except Exception as err:
        source_inventory.append({"SOURCE": table, "GEC_COLUMN": "NOT ACCESSIBLE", "GCC_COLUMN": "-", "PD_COLUMN": "-", "TIME_MODE": str(err)[:100]})

print("\n1. EXACT SOURCE/COLUMN INVENTORY")
display(pd.DataFrame(source_inventory))

# Prefer full raw ENTITY_VW rather than the Legal-Person-filtered vars_entity_level.
candidates = raw_sources + [entity_ds]
usable = [s for s in candidates if get_col(s["cols"], *GEC_NAMES)
          and get_col(s["cols"], "ENTITY_ID")
          and ((get_col(s["cols"], "TEK_STRT_DTE") and get_col(s["cols"], "TEK_END_DTE"))
               or get_col(s["cols"], "MAP_DTE"))]
if not usable:
    print("\nBLOCKER: No accessible entity history contains HEAD_GEC_PARENT_ID / HEAD_GEC_PARENT__ID.")
    print("Need the authoritative GEC mapping field in an accessible source before calculating full-GEC figures.")
    print("No calculations have been silently substituted with GCC_PARENT_ID.")
else:
    es = usable[0]
    member_scope = "FULL ORIGINAL ENTITY SOURCE" if es in raw_sources else "PREPARED ENTITY DATASET - MAY EXCLUDE GEC MEMBERS"
    gec_col = get_col(es["cols"], *GEC_NAMES)
    gcc_col = get_col(es["cols"], "GCC_PARENT_ID")
    pd_col = get_col(es["cols"], "PD_CRR")
    print("\nSelected GEC source:", es["name"])
    print("GEC identifier:", gec_col, "| scope:", member_scope)
    if not pd_col:
        print("WARNING: selected GEC source does not expose PD_CRR: PD/rating checks will be coverage-only.")
    print("IMPORTANT: null GEC key is provisionally considered standalone (self-key), separately counted below.")

    dates_sql = "SELECT TO_DATE(column1) AS DT FROM VALUES " + ",".join("('" + d + "')" for d in DATES)
    a = "e"
    e_id = "TRIM(" + expr_col(a, get_col(es["cols"], "ENTITY_ID")) + "::VARCHAR)"
    e_gec = "NULLIF(TRIM(" + expr_col(a, gec_col) + "::VARCHAR),'')"
    e_gcc = "NULLIF(TRIM(" + expr_col(a, gcc_col) + "::VARCHAR),'')" if gcc_col else "NULL::VARCHAR"
    e_pd = ("IFF(TRY_TO_DOUBLE(" + expr_col(a,pd_col) + "::VARCHAR) BETWEEN 0 AND 1, "
            + "TRY_TO_DOUBLE(" + expr_col(a,pd_col) + "::VARCHAR), NULL)") if pd_col else "NULL::DOUBLE"
    start_col = get_col(es["cols"], "TEK_STRT_DTE")
    end_col = get_col(es["cols"], "TEK_END_DTE")
    if start_col and end_col:
        date_join = "TO_DATE(" + expr_col(a,start_col) + ") <= d.DT AND COALESCE(TO_DATE(" + expr_col(a,end_col) + "),DATE '9999-12-31') >= d.DT"
        order_vers = "TO_DATE(" + expr_col(a,start_col) + ") DESC, COALESCE(TO_DATE(" + expr_col(a,end_col) + "),DATE '9999-12-31') DESC"
    else:
        date_join = "TO_DATE(" + expr_col(a, get_col(es["cols"], "MAP_DTE")) + ")=d.DT"
        order_vers = e_id

    # This is the same 19-notch PD scale used in the existing B01 training.
    rating_intervals = [0, .00005, .00015, .00025, .00035, .00045, .00055, .00065,
                        .00125, .00260, .00525, .00795, .01015, .01915, .03315,
                        .06510, .11455, .22355, .99975, 1.005]
    notch_sql = "CASE " + " ".join("WHEN PD >= {lo} AND PD < {hi} THEN {n}".format(
        lo=rating_intervals[i-1], hi=rating_intervals[i], n=i) for i in range(1,20)) + " ELSE NULL END"

    loan_id = expr_col("l", get_col(loan_ds["cols"], "ENTITY_ID"))
    loan_inst = expr_col("l", get_col(loan_ds["cols"], "INSTRUMENT_ID"))
    loan_nom = "TRY_TO_DOUBLE(" + expr_col("l", get_col(loan_ds["cols"], "NOMINAL_AMOUNT")) + "::VARCHAR)"
    loan_date = expr_col("l", get_col(loan_ds["cols"], "MAP_DTE"))
    cte = """
WITH DATES AS ({DATES}),
E_RAW AS (
    SELECT d.DT, {E_ID} AS ENTITY_ID,
           {E_GEC} AS RAW_GEC_KEY,
           {E_GCC} AS RAW_GCC_KEY,
           {E_PD} AS PD,
           ROW_NUMBER() OVER(PARTITION BY d.DT,{E_ID} ORDER BY {VERS}) AS RN,
           COUNT(*) OVER(PARTITION BY d.DT,{E_ID}) AS ASOF_VERSIONS
    FROM DATES d JOIN {E_SOURCE} e ON {DATE_JOIN}
    WHERE {E_ID} IS NOT NULL
),
E_ONE AS (
    SELECT DT,ENTITY_ID,RAW_GEC_KEY,RAW_GCC_KEY,PD,ASOF_VERSIONS,
           COALESCE(RAW_GEC_KEY,ENTITY_ID) AS GEC_KEY,
           COALESCE(RAW_GCC_KEY,ENTITY_ID) AS GCC_KEY,
           {RATING} AS NOTCH
    FROM E_RAW WHERE RN=1
),
LOAN_RAW AS (
    SELECT TO_DATE({LOAN_DATE}) AS DT,
           TRIM({LOAN_ID}::VARCHAR) ENTITY_ID,
           TRIM({LOAN_INST}::VARCHAR) INSTRUMENT_ID,
           {LOAN_NOM} NOMINAL
    FROM {LOAN_TABLE} l
    JOIN DATES d ON TO_DATE({LOAN_DATE})=d.DT
    WHERE {LOAN_ID} IS NOT NULL AND {LOAN_INST} IS NOT NULL
),
LOAN_INSTR AS (
    SELECT DT,ENTITY_ID,INSTRUMENT_ID,COUNT(*) SOURCE_ROWS,
           MIN(NOMINAL) MIN_NOMINAL,MAX(NOMINAL) MAX_NOMINAL,
           MAX(NOMINAL) NOMINAL,
           COUNT_IF(NOMINAL IS NOT NULL) N_NOMINAL_NON_NULL
    FROM LOAN_RAW GROUP BY 1,2,3
),
LOAN_ENTITY AS (
    SELECT DT,ENTITY_ID,COUNT(*) N_LOANS,
           COUNT_IF(NOMINAL IS NOT NULL) N_NOMINAL_KNOWN,
           COUNT_IF(SOURCE_ROWS>1) N_DUP_INSTRUMENTS,
           COUNT_IF(MIN_NOMINAL<>MAX_NOMINAL) N_CONFLICTING_NOMINAL,
           SUM(NOMINAL) KNOWN_NOMINAL
    FROM LOAN_INSTR GROUP BY 1,2
),
MEMBERS AS (
    SELECT e.*,l.N_LOANS,l.N_NOMINAL_KNOWN,l.KNOWN_NOMINAL,
           l.N_DUP_INSTRUMENTS,l.N_CONFLICTING_NOMINAL
    FROM E_ONE e LEFT JOIN LOAN_ENTITY l ON e.DT=l.DT AND e.ENTITY_ID=l.ENTITY_ID
),
GEC_GROUPS AS (
    SELECT DT,GEC_KEY,COUNT(*) N_MEMBERS,
           COUNT_IF(RAW_GEC_KEY IS NULL) N_SELF_FALLBACK,
           COUNT_IF(ASOF_VERSIONS>1) N_OVERLAPPING_ENTITY_VERSIONS,
           COUNT_IF(PD IS NOT NULL) N_PD_KNOWN,
           COUNT_IF(NOTCH IS NOT NULL) N_NOTCH_KNOWN,
           COUNT_IF(N_LOANS>0) N_LOAN_MEMBERS,
           SUM(COALESCE(N_LOANS,0)) N_LOANS,
           SUM(COALESCE(N_NOMINAL_KNOWN,0)) N_NOMINAL_KNOWN,
           SUM(COALESCE(N_DUP_INSTRUMENTS,0)) N_DUP_INSTRUMENTS,
           SUM(COALESCE(N_CONFLICTING_NOMINAL,0)) N_CONFLICTING_NOMINAL,
           SUM(KNOWN_NOMINAL) NOMINAL_KNOWN_SUM,
           AVG(PD) PD_SIMPLE,
           AVG(NOTCH) NOTCH_SIMPLE,
           SUM(IFF(PD IS NOT NULL AND KNOWN_NOMINAL>0,PD*KNOWN_NOMINAL,0))
              /NULLIF(SUM(IFF(PD IS NOT NULL AND KNOWN_NOMINAL>0,KNOWN_NOMINAL,0)),0) PD_WEIGHTED,
           SUM(IFF(NOTCH IS NOT NULL AND KNOWN_NOMINAL>0,NOTCH*KNOWN_NOMINAL,0))
              /NULLIF(SUM(IFF(NOTCH IS NOT NULL AND KNOWN_NOMINAL>0,KNOWN_NOMINAL,0)),0) NOTCH_WEIGHTED,
           SUM(IFF(PD IS NOT NULL AND KNOWN_NOMINAL>0,KNOWN_NOMINAL,0)) NOMINAL_PD_WEIGHT_KNOWN,
           IFF(SUM(COALESCE(N_LOANS,0))>0
               AND SUM(COALESCE(N_LOANS,0))=SUM(COALESCE(N_NOMINAL_KNOWN,0))
               AND SUM(COALESCE(N_DUP_INSTRUMENTS,0))=0
               AND SUM(COALESCE(N_CONFLICTING_NOMINAL,0))=0
               AND SUM(IFF(ASOF_VERSIONS>1,1,0))=0,
               SUM(KNOWN_NOMINAL),NULL) NOMINAL_STRICT
    FROM MEMBERS GROUP BY 1,2
),
TRAIN AS (
    SELECT TO_DATE(t.MAP_DTE) DT,t.ENTITY_ID::VARCHAR ENTITY_ID,
           TRY_TO_DOUBLE(t.GEC_NOMINAL_CANDIDATE::VARCHAR) OLD_GEC_NOMINAL,
           TRY_TO_DOUBLE(t.GCC_PD_CRR::VARCHAR) OLD_GROUP_PD,
           TRY_TO_DOUBLE(t.GCC_NOTCH::VARCHAR) OLD_GROUP_NOTCH,
           TRY_TO_DOUBLE(t.DPD_MAX::VARCHAR) DPD_MAX,
           TRY_TO_DOUBLE(t.DPD_CROSS_GT_15::VARCHAR) DPD_CROSS_GT_15,
           t.GEC_KEY::VARCHAR OLD_GEC_KEY
    FROM {TRAIN_TABLE} t
    JOIN DATES d ON TO_DATE(t.MAP_DTE)=d.DT
)
""".format(DATES=dates_sql,E_ID=e_id,E_GEC=e_gec,E_GCC=e_gcc,E_PD=e_pd,
           VERS=order_vers,E_SOURCE=es["table"],DATE_JOIN=date_join,RATING=notch_sql,
           LOAN_DATE=loan_date,LOAN_ID=loan_id,LOAN_INST=loan_inst,LOAN_NOM=loan_nom,
           LOAN_TABLE=loan_ds["table"],TRAIN_TABLE=train_ds["table"])

    run("2. HISTORICAL GEC COVERAGE - ALL MEMBERS (not only entities with loans)", cte+"""
SELECT DT,COUNT(*) N_ENTITY_MEMBERS,
       COUNT(DISTINCT GEC_KEY) N_GECS,
       COUNT_IF(RAW_GEC_KEY IS NULL) N_WITHOUT_HEAD_GEC,
       COUNT_IF(ASOF_VERSIONS>1) N_MULTIPLE_ACTIVE_VERSIONS,
       COUNT_IF(PD IS NOT NULL) N_PD_KNOWN,
       COUNT_IF(NOTCH IS NOT NULL) N_RATING_KNOWN,
       COUNT_IF(N_LOANS>0) N_MEMBERS_WITH_LOANS,
       COUNT_IF(N_LOANS IS NULL) N_MEMBERS_WITHOUT_LOANS,
       COUNT_IF(RAW_GEC_KEY IS NOT NULL AND RAW_GCC_KEY IS NOT NULL
                AND RAW_GEC_KEY<>RAW_GCC_KEY) N_DIFFERENT_GEC_VS_GCC
FROM MEMBERS GROUP BY 1 ORDER BY 1
""")

    run("3. LOAN SOURCE - DUPLICATES, MISSING NOMINAL, MISSING GEC ENTITY MAPPING", cte+"""
SELECT l.DT,COUNT(*) N_LOAN_ENTITIES,
       SUM(l.N_LOANS) N_INSTRUMENTS,
       SUM(l.N_NOMINAL_KNOWN) N_INSTRUMENTS_WITH_NOMINAL,
       SUM(l.N_LOANS-l.N_NOMINAL_KNOWN) N_MISSING_NOMINAL,
       SUM(l.N_DUP_INSTRUMENTS) N_DUPLICATED_INSTRUMENT_KEYS,
       SUM(l.N_CONFLICTING_NOMINAL) N_CONFLICTING_DUPLICATES,
       COUNT_IF(e.ENTITY_ID IS NULL) N_LOAN_ENTITIES_WITHOUT_GEC_MAPPING,
       COUNT_IF(e.ENTITY_ID IS NOT NULL AND e.RAW_GEC_KEY IS NULL) N_SELF_FALLBACK
FROM LOAN_ENTITY l LEFT JOIN E_ONE e ON l.DT=e.DT AND l.ENTITY_ID=e.ENTITY_ID
GROUP BY 1 ORDER BY 1
""")

    run("4. GROUP-LEVEL COMPLETENESS - GEC NOMINAL AND PD/RATING", cte+"""
SELECT DT,COUNT(*) N_GECS,
       COUNT_IF(N_MEMBERS>1) N_MULTI_MEMBER_GECS,
       COUNT_IF(N_LOANS>0) N_GECS_WITH_LOANS,
       COUNT_IF(N_LOANS>0 AND NOMINAL_STRICT IS NULL) N_INCOMPLETE_NOMINAL_GECS,
       COUNT_IF(NOMINAL_STRICT IS NOT NULL) N_STRICT_NOMINAL_GECS,
       COUNT_IF(N_PD_KNOWN=N_MEMBERS) N_ALL_MEMBERS_PD_KNOWN,
       COUNT_IF(N_PD_KNOWN<N_MEMBERS) N_INCOMPLETE_PD_GECS,
       COUNT_IF(N_MEMBERS>1 AND N_PD_KNOWN>0 AND N_PD_KNOWN<N_MEMBERS) N_PARTIAL_PD_MULTI_GECS,
       COUNT_IF(N_MEMBERS>1 AND ABS(PD_SIMPLE-PD_WEIGHTED)>0.005) N_PD_METHOD_DIFFERENCES_GT_0_5PP,
       COUNT_IF(N_MEMBERS>1 AND ABS(NOTCH_SIMPLE-NOTCH_WEIGHTED)>1) N_RATING_METHOD_DIFFERENCES_GT_1_NOTCH,
       COUNT_IF(N_MEMBERS>1 AND N_SELF_FALLBACK>0) N_MULTI_GECS_WITH_NULL_PARENT_KEY
FROM GEC_GROUPS GROUP BY 1 ORDER BY 1
""")

    run("5. IMPACT OF GEC VS CURRENT GCC TRAINING - EXACT-DATE RECONCILIATION", cte+"""
SELECT t.DT,COUNT(*) N_TRAIN_ROWS,
       COUNT_IF(e.ENTITY_ID IS NULL) N_NO_GEC_ENTITY_HISTORY,
       COUNT_IF(e.ENTITY_ID IS NOT NULL AND t.OLD_GEC_KEY<>e.GEC_KEY) N_DIFFERENT_GROUP_KEY,
       COUNT_IF(t.OLD_GEC_NOMINAL IS NOT NULL) N_OLD_EXPOSURE_KNOWN,
       COUNT_IF(g.NOMINAL_STRICT IS NOT NULL) N_NEW_STRICT_GEC_KNOWN,
       COUNT_IF(t.OLD_GEC_NOMINAL IS NOT NULL AND g.NOMINAL_STRICT IS NOT NULL
                AND ABS(t.OLD_GEC_NOMINAL-g.NOMINAL_STRICT)>0.01) N_CHANGED_EXPOSURE,
       COUNT_IF(t.OLD_GROUP_PD IS NOT NULL AND g.PD_SIMPLE IS NOT NULL
                AND ABS(t.OLD_GROUP_PD-g.PD_SIMPLE)>0.000001) N_CHANGED_GROUP_PD_SIMPLE,
       COUNT_IF(t.OLD_GROUP_NOTCH IS NOT NULL AND g.NOTCH_SIMPLE IS NOT NULL
                AND ABS(t.OLD_GROUP_NOTCH-g.NOTCH_SIMPLE)>0.01) N_CHANGED_GROUP_NOTCH_SIMPLE,
       COUNT_IF(g.N_PD_KNOWN<g.N_MEMBERS) N_GEC_RATING_AVG_PARTIAL
FROM TRAIN t
LEFT JOIN E_ONE e ON t.DT=e.DT AND t.ENTITY_ID=e.ENTITY_ID
LEFT JOIN GEC_GROUPS g ON e.DT=g.DT AND e.GEC_KEY=g.GEC_KEY
GROUP BY 1 ORDER BY 1
""")

    run("6. DPD >15 x GEC >=350k: CURRENT VS RECOMPUTED GROUP EXPOSURE (NOT FINAL ALERTS)", cte+"""
SELECT t.DT,
       COUNT_IF(t.DPD_CROSS_GT_15=1) N_DPD15_CROSSES_TRAIN,
       COUNT_IF(t.DPD_CROSS_GT_15=1 AND t.OLD_GEC_NOMINAL>=350000) N_OLD_GCC_PROXY_GE_350K,
       COUNT_IF(t.DPD_CROSS_GT_15=1 AND g.NOMINAL_STRICT>=350000) N_NEW_GEC_STRICT_GE_350K,
       COUNT_IF(t.DPD_CROSS_GT_15=1 AND g.NOMINAL_STRICT IS NULL) N_DPD_CROSSES_NEW_EXPOSURE_UNKNOWN,
       COUNT_IF(t.DPD_CROSS_GT_15=1 AND t.OLD_GEC_NOMINAL>=350000
                AND g.NOMINAL_STRICT IS NOT NULL AND g.NOMINAL_STRICT<350000) N_OLD_YES_NEW_NO,
       COUNT_IF(t.DPD_CROSS_GT_15=1 AND t.OLD_GEC_NOMINAL<350000
                AND g.NOMINAL_STRICT>=350000) N_OLD_NO_NEW_YES
FROM TRAIN t
LEFT JOIN E_ONE e ON t.DT=e.DT AND t.ENTITY_ID=e.ENTITY_ID
LEFT JOIN GEC_GROUPS g ON e.DT=g.DT AND e.GEC_KEY=g.GEC_KEY
GROUP BY 1 ORDER BY 1
""")

    run("7. GROUP MEANS: SIMPLE vs NOMINAL-WEIGHTED and HEAD'S PD (aggregates only)", cte+"""
SELECT g.DT,COUNT_IF(g.N_MEMBERS>1 AND g.N_PD_KNOWN>=2) N_MULTI_MEMBER_GROUPS_PD_COMPARABLE,
       ROUND(AVG(IFF(g.N_MEMBERS>1,ABS(g.PD_SIMPLE-g.PD_WEIGHTED),NULL)),6) AVG_ABS_PD_SIMPLE_VS_WEIGHTED,
       ROUND(MAX(IFF(g.N_MEMBERS>1,ABS(g.PD_SIMPLE-g.PD_WEIGHTED),NULL)),6) MAX_ABS_PD_SIMPLE_VS_WEIGHTED,
       ROUND(AVG(IFF(g.N_MEMBERS>1,ABS(g.NOTCH_SIMPLE-g.NOTCH_WEIGHTED),NULL)),3) AVG_ABS_NOTCH_SIMPLE_VS_WEIGHTED,
       COUNT_IF(g.N_MEMBERS>1 AND g.N_PD_KNOWN=g.N_MEMBERS AND g.N_LOANS>0
                AND g.N_NOMINAL_KNOWN=g.N_LOANS AND g.N_DUP_INSTRUMENTS=0) N_COMPLETE_GROUPS_FOR_BOTH
FROM GEC_GROUPS g GROUP BY 1 ORDER BY 1
""")

    run("8. SIX/TWELVE-MONTH GEC HISTORY FEASIBILITY (exact as-of dates, group composition may change)", cte+"""
SELECT c.DT,
       COUNT(*) N_CURRENT_GEC,
       COUNT_IF(p6.GEC_KEY IS NOT NULL) N_WITH_SAME_GEC_KEY_6M_AGO,
       COUNT_IF(p12.GEC_KEY IS NOT NULL) N_WITH_SAME_GEC_KEY_12M_AGO,
       COUNT_IF(c.PD_SIMPLE IS NOT NULL AND p6.PD_SIMPLE IS NOT NULL) N_PD_SIMPLE_6M_COMPARABLE,
       COUNT_IF(c.PD_SIMPLE IS NOT NULL AND p12.PD_SIMPLE IS NOT NULL) N_PD_SIMPLE_12M_COMPARABLE,
       COUNT_IF(c.NOTCH_SIMPLE IS NOT NULL AND p12.NOTCH_SIMPLE IS NOT NULL) N_NOTCH_SIMPLE_12M_COMPARABLE,
       COUNT_IF(p12.GEC_KEY IS NOT NULL AND c.N_MEMBERS<>p12.N_MEMBERS) N_DIFFERENT_GROUP_SIZE_12M
FROM GEC_GROUPS c
LEFT JOIN GEC_GROUPS p6 ON p6.GEC_KEY=c.GEC_KEY AND p6.DT=DATEADD(month,-6,c.DT)
LEFT JOIN GEC_GROUPS p12 ON p12.GEC_KEY=c.GEC_KEY AND p12.DT=DATEADD(month,-12,c.DT)
WHERE c.DT=TO_DATE('2026-10-29')
GROUP BY 1
""")

    print("\n" + "="*105)
    print("9. INTERPRETATION / BLOCKERS BEFORE REBUILD")
    print("="*105)
    print("* Official GEC mapping found in:", es["name"], "column:", gec_col)
    print("* Source scope:", member_scope)
    print("* Head missing -> entity self-fallback is a QA assumption, NOT confirmed GEC mapping.")
    print("* PD / rating averages IGNORE null PD; review partial-GEC coverage before production.")
    print("* Weighted averages here use positive known NOMINAL; missing/zero weight means exclusions.")
    print("* A simple average of notches differs from the notch derived from average PD: method pending Timo.")
    print("* Nominal completeness is checked only within vars_loan_level, NOT against an independent all-bank loan inventory.")
    print("* Dates above are SOURCE DIAGNOSTICS, NOT historical-model re-calibration or final alert counts.")
    print("* No dataset, recipe, schema or Flow was modified.")

print("\nQA completed. Any BLOCKED section above needs resolution; it is not a passed check.")
