# -*- coding: utf-8 -*-
"""Corporate EWS | Training B01 | DPD & Rating/GCC.

Inputs:
    vars_entity_level
    vars_loan_level
    DPD_Arrears_Daily_Loan
Output:
    EWS_training_B01_DPD_RATING

Builds the historical entity-day training panel in Snowflake.
Feature evaluation and model calibration are handled separately.
"""

import re
import dataiku
from dataiku import recipe
from dataiku.core.sql import SQLExecutor2

PROJECT = "VALIDATIONEWSCORPORATEPORTFOLIODAILY"
CONNECTION = "DEV_AA_TEAMS"
OUTPUT_DATASET = "EWS_training_B01_DPD_RATING"

project = dataiku.api_client().get_project(PROJECT)


def quote_identifier(value):
    value = str(value).replace("${projectKey}", PROJECT)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Invalid Snowflake identifier: " + repr(value))
    return '"' + value + '"'


def source_table(name, required_columns):
    cfg = project.get_dataset(name).get_settings().get_raw()
    params = cfg.get("params") or {}
    if params.get("connection") != CONNECTION:
        raise ValueError("Unexpected connection for " + name)

    existing = {
        col["name"].upper()
        for col in dataiku.Dataset(name, project_key=PROJECT).read_schema()
    }
    missing = set(required_columns) - existing
    if missing:
        raise ValueError("Missing columns in {}: {}".format(name, sorted(missing)))

    return quote_identifier(params["schema"]) + "." + quote_identifier(params["table"])


ENTITY = source_table("vars_entity_level", [
    "ENTITY_ID", "TEK_STRT_DTE", "TEK_END_DTE", "PD_CRR", "GCC_PARENT_ID"
])
LOAN = source_table("vars_loan_level", [
    "ENTITY_ID", "INSTRUMENT_ID", "MAP_DTE", "NOMINAL_AMOUNT",
    "GROSS_CARRYING_AMOUNT", "DEFAULT_STATUS", "FORBEARANCE_STATUS", "DIM1"
])
DPD = source_table("DPD_Arrears_Daily_Loan", [
    "ENTITY_ID", "INSTRUMENT_ID", "MAP_DTE", "DPD", "ARREARS_AMOUNT"
])

SCALE = [
    ("AAA+", 0.00000, 0.00005), ("AAA", 0.00005, 0.00015),
    ("AA+", 0.00015, 0.00025), ("AA", 0.00025, 0.00035),
    ("AA-", 0.00035, 0.00045), ("A+", 0.00045, 0.00055),
    ("A", 0.00055, 0.00065), ("A-", 0.00065, 0.00125),
    ("BBB+", 0.00125, 0.00260), ("BBB", 0.00260, 0.00525),
    ("BBB-", 0.00525, 0.00795), ("BB+", 0.00795, 0.01015),
    ("BB", 0.01015, 0.01915), ("BB-", 0.01915, 0.03315),
    ("B+", 0.03315, 0.06510), ("B", 0.06510, 0.11455),
    ("B-", 0.11455, 0.22355), ("CCC", 0.22355, 0.99975),
    ("D", 0.99975, 1.00500)
]


def rating_case(pd_expr, return_label=False):
    terms = []
    for notch, (rating, lower, upper) in enumerate(SCALE, 1):
        value = "'{}'".format(rating) if return_label else str(notch)
        terms.append("WHEN {e} >= {l} AND {e} < {u} THEN {v}".format(
            e=pd_expr, l=lower, u=upper, v=value))
    return "CASE {} ELSE NULL END".format(" ".join(terms))



# DPD thresholds and upward crossings.
DPD_FLAGS = []
for t in range(5, 61, 5):
    DPD_FLAGS.append(
        "IFF(f.DPD_MAX IS NULL,NULL,IFF(f.DPD_MAX>{t},1,0)) AS DPD_GT_{t}".format(t=t)
    )
    DPD_FLAGS.append(
        "CASE WHEN f.DPD_MAX IS NULL OR f.PREV_DPD_MAX IS NULL THEN NULL "
        "WHEN f.DPD_MAX>{t} AND f.PREV_DPD_MAX<={t} THEN 1 ELSE 0 END "
        "AS DPD_CROSS_GT_{t}".format(t=t)
    )

DPD_SQL = ",\n       ".join(DPD_FLAGS)

SQL = """
WITH
DPD_LOAN_DAY AS (
    SELECT TO_DATE(d.MAP_DTE) MAP_DTE,
           d.ENTITY_ID::VARCHAR ENTITY_ID,
           d.INSTRUMENT_ID::VARCHAR INSTRUMENT_ID,
           MAX(TRY_TO_DOUBLE(d.DPD::VARCHAR)) DPD,
           MAX(TRY_TO_DOUBLE(d.ARREARS_AMOUNT::VARCHAR)) ARREARS,
           COUNT(*) SOURCE_ROWS
    FROM {DPD} d
    GROUP BY 1,2,3
),
DPD_ENTITY_DAY AS (
    SELECT MAP_DTE, ENTITY_ID,
           MAX(DPD) DPD_MAX,
           SUM(ARREARS) ARREARS_AMOUNT,
           COUNT(*) DPD_LOANS,
           COUNT_IF(DPD IS NOT NULL) DPD_KNOWN_LOANS,
           SUM(SOURCE_ROWS-1) DPD_SOURCE_DUPLICATES
    FROM DPD_LOAN_DAY
    GROUP BY 1,2
),
LOAN_INSTRUMENT_DAY AS (
    SELECT TO_DATE(l.MAP_DTE) MAP_DTE,
           l.ENTITY_ID::VARCHAR ENTITY_ID,
           l.INSTRUMENT_ID::VARCHAR INSTRUMENT_ID,
           MAX(TRY_TO_DOUBLE(l.NOMINAL_AMOUNT::VARCHAR)) NOMINAL,
           MAX(TRY_TO_DOUBLE(l.GROSS_CARRYING_AMOUNT::VARCHAR)) GCA,
           MAX(CASE WHEN TRY_TO_DOUBLE(l.DEFAULT_STATUS::VARCHAR) IS NULL THEN NULL
                    WHEN TRY_TO_DOUBLE(l.DEFAULT_STATUS::VARCHAR) IN (18,19,20) THEN 1 ELSE 0 END) DEFAULT_FLAG,
           MAX(CASE WHEN TRY_TO_DOUBLE(l.FORBEARANCE_STATUS::VARCHAR) IS NULL THEN NULL
                    WHEN TRY_TO_DOUBLE(l.FORBEARANCE_STATUS::VARCHAR) IN (3,4,5) THEN 1 ELSE 0 END) FORBEARANCE_FLAG,
           MAX(CASE WHEN l.DIM1 IS NULL THEN NULL
                    WHEN l.DIM1::VARCHAR IN ('S2','S3','S3S','S3R') THEN 1 ELSE 0 END) DIM1_FLAG,
           COUNT(*) SOURCE_ROWS
    FROM {LOAN} l
    GROUP BY 1,2,3
),
LOAN_ENTITY_DAY AS (
    SELECT MAP_DTE, ENTITY_ID, COUNT(*) LOAN_COUNT,
           SUM(NOMINAL) NOMINAL_KNOWN_SUM, SUM(GCA) GCA_KNOWN_SUM,
           COUNT_IF(NOMINAL IS NOT NULL) NOMINAL_KNOWN_LOANS,
           COUNT_IF(GCA IS NOT NULL) GCA_KNOWN_LOANS,
           SUM(SOURCE_ROWS-1) LOAN_SOURCE_DUPLICATES,
           MAX(DEFAULT_FLAG) TGT_DEFAULT_SAMEDAY,
           MAX(FORBEARANCE_FLAG) TGT_FORBEARANCE_SAMEDAY,
           MAX(DIM1_FLAG) TGT_DIM1_SAMEDAY
    FROM LOAN_INSTRUMENT_DAY
    GROUP BY 1,2
),
PANEL_KEYS AS (
    SELECT MAP_DTE, ENTITY_ID FROM DPD_ENTITY_DAY
    UNION
    SELECT MAP_DTE, ENTITY_ID FROM LOAN_ENTITY_DAY
),
ENTITY_HISTORY AS (
    SELECT e.ENTITY_ID::VARCHAR ENTITY_ID,
           TO_DATE(e.TEK_STRT_DTE) START_DATE,
           COALESCE(TO_DATE(e.TEK_END_DTE),DATE '9999-12-31') END_DATE,
           NULLIF(TRIM(e.GCC_PARENT_ID::VARCHAR),'') GCC_PARENT_ID,
           IFF(TRY_TO_DOUBLE(e.PD_CRR::VARCHAR) BETWEEN 0 AND 1,
               TRY_TO_DOUBLE(e.PD_CRR::VARCHAR),NULL) PD_CRR
    FROM {ENTITY} e
    WHERE e.ENTITY_ID IS NOT NULL AND e.TEK_STRT_DTE IS NOT NULL
),
-- Deterministically remove identical-start versions; keep widest validity.
ENTITY_HISTORY_DEDUP AS (
    SELECT * FROM ENTITY_HISTORY
    QUALIFY ROW_NUMBER() OVER (
       PARTITION BY ENTITY_ID,START_DATE
       ORDER BY END_DATE DESC, GCC_PARENT_ID DESC NULLS LAST, PD_CRR DESC NULLS LAST
    )=1
),
ENTITY_RATINGS AS (
    SELECT h.*,
           {RATING_NOTCH} AS RATING_NOTCH,
           {RATING_LABEL} AS RATING_LABEL
    FROM ENTITY_HISTORY_DEDUP h
),
CLIENT_ASOF_RAW AS (
    SELECT k.MAP_DTE,k.ENTITY_ID,
           h.END_DATE,h.GCC_PARENT_ID,h.PD_CRR,
           h.RATING_NOTCH AS CLIENT_NOTCH,
           h.RATING_LABEL AS CLIENT_RATING
    FROM PANEL_KEYS k ASOF JOIN ENTITY_RATINGS h
       MATCH_CONDITION(k.MAP_DTE >= h.START_DATE)
       ON k.ENTITY_ID=h.ENTITY_ID
),
CLIENT_ASOF AS (
    SELECT MAP_DTE,ENTITY_ID,
           IFF(END_DATE>=MAP_DTE,1,0) HAS_VALID_ENTITY_HISTORY,
           IFF(END_DATE>=MAP_DTE,GCC_PARENT_ID,NULL) GCC_PARENT_ID,
           IFF(END_DATE>=MAP_DTE,PD_CRR,NULL) PD_CRR,
           IFF(END_DATE>=MAP_DTE,CLIENT_NOTCH,NULL) CLIENT_NOTCH,
           IFF(END_DATE>=MAP_DTE,CLIENT_RATING,NULL) CLIENT_RATING
    FROM CLIENT_ASOF_RAW
),
HEAD_ASOF_RAW AS (
    SELECT c.*,
           h.END_DATE AS HEAD_END_DATE,
           h.PD_CRR AS HEAD_PD_CRR,
           h.RATING_NOTCH AS HEAD_NOTCH,
           h.RATING_LABEL AS HEAD_RATING
    FROM CLIENT_ASOF c ASOF JOIN ENTITY_RATINGS h
       MATCH_CONDITION(c.MAP_DTE >= h.START_DATE)
       ON c.GCC_PARENT_ID=h.ENTITY_ID
),
RATING_CURRENT AS (
    SELECT MAP_DTE,ENTITY_ID,HAS_VALID_ENTITY_HISTORY,
           GCC_PARENT_ID,PD_CRR,CLIENT_NOTCH,CLIENT_RATING,
           CASE WHEN GCC_PARENT_ID IS NULL THEN PD_CRR
                WHEN HEAD_END_DATE>=MAP_DTE THEN HEAD_PD_CRR ELSE NULL END GCC_PD_CRR,
           CASE WHEN GCC_PARENT_ID IS NULL THEN CLIENT_NOTCH
                WHEN HEAD_END_DATE>=MAP_DTE THEN HEAD_NOTCH ELSE NULL END GCC_NOTCH,
           CASE WHEN GCC_PARENT_ID IS NULL THEN CLIENT_RATING
                WHEN HEAD_END_DATE>=MAP_DTE THEN HEAD_RATING ELSE NULL END GCC_RATING,
           IFF(GCC_PARENT_ID IS NULL OR HEAD_END_DATE>=MAP_DTE,1,0) HAS_VALID_GCC_HEAD
    FROM HEAD_ASOF_RAW
),
BEST_CLIENT_12M AS (
    SELECT c.MAP_DTE,c.ENTITY_ID,
           MIN(h.RATING_NOTCH) BEST_CLIENT_NOTCH_12M,
           MAX(IFF(h.START_DATE<=DATEADD(month,-12,c.MAP_DTE)
                   AND h.END_DATE>=DATEADD(month,-12,c.MAP_DTE),1,0)) CLIENT_HAS_12M_START
    FROM RATING_CURRENT c LEFT JOIN ENTITY_RATINGS h
      ON c.ENTITY_ID=h.ENTITY_ID
     AND h.START_DATE<=c.MAP_DTE
     AND h.END_DATE>=DATEADD(month,-12,c.MAP_DTE)
    GROUP BY 1,2
),
BEST_HEAD_12M AS (
    SELECT c.MAP_DTE,c.ENTITY_ID,
           MIN(h.RATING_NOTCH) BEST_HEAD_NOTCH_12M,
           MAX(IFF(h.START_DATE<=DATEADD(month,-12,c.MAP_DTE)
                   AND h.END_DATE>=DATEADD(month,-12,c.MAP_DTE),1,0)) HEAD_HAS_12M_START
    FROM RATING_CURRENT c LEFT JOIN ENTITY_RATINGS h
      ON c.GCC_PARENT_ID=h.ENTITY_ID
     AND h.START_DATE<=c.MAP_DTE
     AND h.END_DATE>=DATEADD(month,-12,c.MAP_DTE)
    GROUP BY 1,2
),
RATING_BEST AS (
    SELECT c.*,
           cb.BEST_CLIENT_NOTCH_12M,
           CASE WHEN c.GCC_PARENT_ID IS NULL THEN cb.BEST_CLIENT_NOTCH_12M
                ELSE hb.BEST_HEAD_NOTCH_12M END BEST_GCC_NOTCH_12M,
           cb.CLIENT_HAS_12M_START,
           CASE WHEN c.GCC_PARENT_ID IS NULL THEN cb.CLIENT_HAS_12M_START
                ELSE hb.HEAD_HAS_12M_START END GCC_HAS_12M_START
    FROM RATING_CURRENT c
    LEFT JOIN BEST_CLIENT_12M cb ON c.ENTITY_ID=cb.ENTITY_ID AND c.MAP_DTE=cb.MAP_DTE
    LEFT JOIN BEST_HEAD_12M hb ON c.ENTITY_ID=hb.ENTITY_ID AND c.MAP_DTE=hb.MAP_DTE
),
RATING_FEATURES AS (
    SELECT r.*,
           IFF(r.CLIENT_HAS_12M_START=1 AND r.CLIENT_NOTCH IS NOT NULL
               AND r.BEST_CLIENT_NOTCH_12M IS NOT NULL,
               r.CLIENT_NOTCH-r.BEST_CLIENT_NOTCH_12M,NULL) CLIENT_DOWNGRADE_12M,
           IFF(r.GCC_HAS_12M_START=1 AND r.GCC_NOTCH IS NOT NULL
               AND r.BEST_GCC_NOTCH_12M IS NOT NULL,
               r.GCC_NOTCH-r.BEST_GCC_NOTCH_12M,NULL) GCC_DOWNGRADE_12M
    FROM RATING_BEST r
),
GEC_EXPOSURE_DAY AS (
    SELECT l.MAP_DTE,COALESCE(r.GCC_PARENT_ID,l.ENTITY_ID) GEC_KEY,
           SUM(l.NOMINAL_KNOWN_SUM) GEC_NOMINAL_KNOWN_SUM,
           SUM(l.GCA_KNOWN_SUM) GEC_GCA_KNOWN_SUM,
           SUM(l.LOAN_COUNT) GEC_SOURCE_LOANS,
           SUM(l.NOMINAL_KNOWN_LOANS) GEC_NOMINAL_KNOWN_LOANS,
           SUM(l.GCA_KNOWN_LOANS) GEC_GCA_KNOWN_LOANS,
           SUM(l.LOAN_SOURCE_DUPLICATES) GEC_SOURCE_DUPLICATES,
           MIN(r.HAS_VALID_ENTITY_HISTORY) GEC_ALL_ENTITY_MAPPED
    FROM LOAN_ENTITY_DAY l
    JOIN RATING_CURRENT r ON l.ENTITY_ID=r.ENTITY_ID AND l.MAP_DTE=r.MAP_DTE
    GROUP BY 1,2
),
BASE AS (
    SELECT k.ENTITY_ID,k.MAP_DTE,
           r.GCC_PARENT_ID,COALESCE(r.GCC_PARENT_ID,k.ENTITY_ID) GEC_KEY,
           r.PD_CRR,r.GCC_PD_CRR,r.CLIENT_RATING,r.GCC_RATING,
           r.CLIENT_NOTCH,r.GCC_NOTCH,
           r.BEST_CLIENT_NOTCH_12M,r.BEST_GCC_NOTCH_12M,
           r.CLIENT_HAS_12M_START,r.GCC_HAS_12M_START,
           r.CLIENT_DOWNGRADE_12M,r.GCC_DOWNGRADE_12M,
           r.HAS_VALID_ENTITY_HISTORY,r.HAS_VALID_GCC_HEAD,
           d.DPD_MAX,d.ARREARS_AMOUNT,d.DPD_LOANS,d.DPD_KNOWN_LOANS,
           d.DPD_SOURCE_DUPLICATES,
           l.LOAN_COUNT,l.NOMINAL_KNOWN_SUM,l.GCA_KNOWN_SUM,
           l.NOMINAL_KNOWN_LOANS,l.GCA_KNOWN_LOANS,l.LOAN_SOURCE_DUPLICATES,
           l.TGT_DEFAULT_SAMEDAY,l.TGT_FORBEARANCE_SAMEDAY,l.TGT_DIM1_SAMEDAY,
           g.GEC_SOURCE_LOANS,g.GEC_NOMINAL_KNOWN_LOANS,g.GEC_GCA_KNOWN_LOANS,
           g.GEC_SOURCE_DUPLICATES,
           IFF(g.GEC_SOURCE_LOANS>0 AND g.GEC_ALL_ENTITY_MAPPED=1
               AND g.GEC_SOURCE_DUPLICATES=0
               AND g.GEC_SOURCE_LOANS=g.GEC_NOMINAL_KNOWN_LOANS,
               g.GEC_NOMINAL_KNOWN_SUM,NULL) GEC_NOMINAL_CANDIDATE,
           IFF(g.GEC_SOURCE_LOANS>0 AND g.GEC_ALL_ENTITY_MAPPED=1
               AND g.GEC_SOURCE_DUPLICATES=0
               AND g.GEC_SOURCE_LOANS=g.GEC_GCA_KNOWN_LOANS,
               g.GEC_GCA_KNOWN_SUM,NULL) GEC_GCA_CANDIDATE
    FROM PANEL_KEYS k
    LEFT JOIN RATING_FEATURES r ON k.ENTITY_ID=r.ENTITY_ID AND k.MAP_DTE=r.MAP_DTE
    LEFT JOIN DPD_ENTITY_DAY d ON k.ENTITY_ID=d.ENTITY_ID AND k.MAP_DTE=d.MAP_DTE
    LEFT JOIN LOAN_ENTITY_DAY l ON k.ENTITY_ID=l.ENTITY_ID AND k.MAP_DTE=l.MAP_DTE
    LEFT JOIN GEC_EXPOSURE_DAY g ON k.MAP_DTE=g.MAP_DTE
       AND COALESCE(r.GCC_PARENT_ID,k.ENTITY_ID)=g.GEC_KEY
),
WITH_PREVIOUS AS (
    SELECT b.*,
           LAG(b.MAP_DTE) OVER (PARTITION BY b.ENTITY_ID ORDER BY b.MAP_DTE) PREV_MAP_DTE,
           LAG(b.DPD_MAX) OVER (PARTITION BY b.ENTITY_ID ORDER BY b.MAP_DTE) PREV_DPD_MAX,
           LAG(b.GEC_NOMINAL_CANDIDATE) OVER
               (PARTITION BY b.ENTITY_ID ORDER BY b.MAP_DTE) PREV_GEC_NOMINAL,
           LAG(b.GEC_GCA_CANDIDATE) OVER
               (PARTITION BY b.ENTITY_ID ORDER BY b.MAP_DTE) PREV_GEC_GCA
    FROM BASE b
),
FEATURES AS (
    SELECT f.*,
           IFF(f.PD_CRR IS NULL,NULL,IFF(f.PD_CRR=1,1,0)) TGT_PD_CRR_SAMEDAY,
           IFF(f.DPD_MAX IS NULL,NULL,IFF(f.DPD_MAX>=20,1,0)) TGT_DPD20_SAMEDAY,
           CASE WHEN f.PD_CRR=1 OR f.DPD_MAX>=20 OR f.TGT_DEFAULT_SAMEDAY=1
                     OR f.TGT_FORBEARANCE_SAMEDAY=1 OR f.TGT_DIM1_SAMEDAY=1 THEN 1
                WHEN f.PD_CRR IS NULL OR f.DPD_MAX IS NULL OR f.TGT_DEFAULT_SAMEDAY IS NULL
                     OR f.TGT_FORBEARANCE_SAMEDAY IS NULL OR f.TGT_DIM1_SAMEDAY IS NULL THEN NULL
                ELSE 0 END AS TGT_ANY_SAMEDAY
    FROM WITH_PREVIOUS f
),
FINAL_FEATURES AS (
    SELECT f.*,
           CASE
               WHEN MOD(MOD(HASH(f.ENTITY_ID), 10000) + 10000, 10000) < 7000
                   THEN 'TRAIN'
               WHEN MOD(MOD(HASH(f.ENTITY_ID), 10000) + 10000, 10000) < 8500
                   THEN 'VALIDATION'
               ELSE 'TEST'
           END AS DATA_SPLIT
    FROM FEATURES f
)
SELECT r.*,
       {DPD_FLAGS}
FROM FINAL_FEATURES r
"""

DPD_SQL = DPD_SQL.replace("f.", "r.")

SQL = SQL.format(
    DPD=DPD,
    LOAN=LOAN,
    ENTITY=ENTITY,
    RATING_NOTCH=rating_case("h.PD_CRR"),
    RATING_LABEL=rating_case("h.PD_CRR", return_label=True),
    DPD_FLAGS=DPD_SQL
)


# Write only the designated Batch 01 output.
configured_outputs = {
    str(getattr(item, "name", item)).split(".")[-1]
    for item in recipe.get_outputs(object_type="DATASET")
}
if configured_outputs != {OUTPUT_DATASET}:
    raise ValueError("Configure only {} as the recipe output".format(OUTPUT_DATASET))

output_config = project.get_dataset(OUTPUT_DATASET).get_settings().get_raw()
output_params = output_config.get("params") or {}
expected_table = PROJECT + "_EWS_TRAINING_B01_DPD_RATING"
actual_table = str(output_params.get("table", "")).replace("${projectKey}", PROJECT)

if (
    output_params.get("connection") != CONNECTION
    or str(output_params.get("schema", "")).upper() != "RRG"
    or actual_table.upper() != expected_table
):
    raise ValueError("Unexpected output table configuration for " + OUTPUT_DATASET)

output = dataiku.Dataset(OUTPUT_DATASET, project_key=PROJECT)
SQLExecutor2.exec_recipe_fragment(output, SQL, overwrite_output_schema=True)
