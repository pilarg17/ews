# BIL CORPORATE - BATCH 01 FULL HISTORY DPD + RATING/GCC
# PURPOSE: one output dataset `results` with all available DPD/loan entity-days.
# Use a separate Python recipe with 3 inputs and only one output `results`.
# EXECUTES in Snowflake via SQLExecutor2.exec_recipe_fragment; no massive Pandas.
# PREVIEW/CHECK: set MODE='CHECK', then MODE='BUILD' for full history.
# IMPORTANT: 'results' is rebuilt/overwritten in BUILD; original sources untouched.
# WARNING: PD-to-rating mapping, GEC exposure and rating thresholds are CANDIDATES
# until validated with BIL. Do not treat SAME-DAY targets as predictive labels.
# WARNING: source dates are not restricted, but the universe is based on
# observed DPD/loan entity-days (not all legal persons); dates with no source
# observation are not invented.

# -*- coding: utf-8 -*-
"""
BIL Corporate EWS | BATCH 01: DPD + RATING/GCC | UNICO OUTPUT: results
==============================================================
New Dataiku DSS Python recipe with 3 original inputs:
    vars_entity_level, vars_loan_level, DPD_Arrears_Daily_Loan
And ONE output Snowflake dataset named exactly `results`.

SQL design: BASE_PANEL + RATING_PANEL are CTEs inside a single query.
No intermediate datasets. Future EWI groups should be created as independent SQL update recipes that
MERGE only new columns into results on ENTITY_ID/MAP_DTE; this first recipe
itself does NOT incrementally update existing results.
This initial BUILD computes full history in Snowflake.

LIMITATIONS (important):
- Entity/day universe: keys present in original DPD or loan snapshots and
  with valid historized entity version. Not all LEGAL_PERSON days.
- GEC exposures are CANDIDATE proxies from observed loan rows and
  GCC_PARENT_ID, not confirmed official full-group exposures.
- Rating rules are CANDIDATES: five-business-day persistence not added.
- Same-day target components are diagnostics, not future prediction labels.
- BUILD writes/replaces only `results`. Original histories and EWS_training
  are not modified. Full historical query can be expensive in Snowflake.
"""
import re
import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
MODE = 'BUILD'  # or 'CHECK' for no-write connection checks
START_DATE = None  # None means complete available source history
END_DATE = None
OUTPUT_NAME = 'results'
EXPECTED_TABLE = PROJECT + '_RESULTS'

assert MODE in ('BUILD', 'CHECK')
if START_DATE and END_DATE and START_DATE > END_DATE:
    raise ValueError('START_DATE greater than END_DATE')

project = dataiku.api_client().get_project(PROJECT)
executor = SQLExecutor2(connection=CONNECTION)


def qident(s):
    s = str(s).replace('${projectKey}', PROJECT)
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', s):
        raise ValueError('Unsafe SQL identifier: ' + repr(s))
    return '"{}"'.format(s)


def table_for(name, required):
    config = project.get_dataset(name).get_settings().get_raw()
    pars = config.get('params') or {}
    if pars.get('connection') != CONNECTION:
        raise ValueError('{} requires connection {}'.format(name, CONNECTION))
    columns = {c['name'].upper() for c in dataiku.Dataset(
        name, project_key=PROJECT).read_schema()}
    missing = set(required) - columns
    if missing:
        raise ValueError('{} missing {}'.format(name, sorted(missing)))
    return qident(pars['schema']) + '.' + qident(pars['table'])


E = table_for('vars_entity_level', [
    'ENTITY_ID', 'TEK_STRT_DTE', 'TEK_END_DTE',
    'PD_CRR', 'GCC_PARENT_ID', 'GCC_PD_CRR'])
L = table_for('vars_loan_level', [
    'MAP_DTE', 'ENTITY_ID', 'INSTRUMENT_ID', 'NOMINAL_AMOUNT',
    'GROSS_CARRYING_AMOUNT', 'DEFAULT_STATUS', 'FORBEARANCE_STATUS', 'DIM1'])
D = table_for('DPD_Arrears_Daily_Loan', [
    'MAP_DTE', 'ENTITY_ID', 'INSTRUMENT_ID', 'DPD', 'ARREARS_AMOUNT'])


def date_filter(alias):
    where = []
    for dt, comparison in ((START_DATE, '>='), (END_DATE, '<=')):
        if dt is not None:
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', dt):
                raise ValueError('Date must be YYYY-MM-DD: ' + repr(dt))
            where.append("TO_DATE({}.MAP_DTE) {} DATE '{}'".format(alias, comparison, dt))
    return ' AND '.join(where) if where else '1=1'

# PROVISIONAL rating mapping from the pilot; verify against the official bank scale.
SCALE = [
    (0.0, 0.00005, 'AAA+',1), (0.00005,0.00015,'AAA',2),
    (0.00015,0.00025,'AA+',3), (0.00025,0.00035,'AA',4),
    (0.00035,0.00045,'AA-',5), (0.00045,0.00055,'A+',6),
    (0.00055,0.00065,'A',7), (0.00065,0.00125,'A-',8),
    (0.00125,0.00260,'BBB+',9), (0.00260,0.00525,'BBB',10),
    (0.00525,0.00795,'BBB-',11), (0.00795,0.01015,'BB+',12),
    (0.01015,0.01915,'BB',13), (0.01915,0.03315,'BB-',14),
    (0.03315,0.06510,'B+',15), (0.06510,0.11455,'B',16),
    (0.11455,0.22355,'B-',17), (0.22355,0.99975,'CCC',18),
    (0.99975,1.005,'D',19),
]


def scale_case(expr, label=False):
    parts = []
    for low, high, name, notch in SCALE:
        result = "'{}'".format(name) if label else str(notch)
        parts.append('WHEN {e}>={low} AND {e}<{high} THEN {r}'.format(
            e=expr, low=low, high=high, r=result))
    return 'CASE ' + ' '.join(parts) + ' ELSE NULL END'


# The two feature families remain separate CTE blocks in ONE Snowflake SQL.
# For future EWIs use targeted SQL updates/MERGE to this same physical table;
# rerunning initial BUILD overwrites results and recomputes the full history.
FULL_SQL = r"""
WITH
D_LOAN AS (
 SELECT TO_DATE(d.MAP_DTE) MAP_DTE,
        d.ENTITY_ID::VARCHAR ENTITY_ID, d.INSTRUMENT_ID::VARCHAR INSTRUMENT_ID,
        MAX(TRY_TO_DOUBLE(d.DPD::VARCHAR)) DPD,
        MAX(TRY_TO_DOUBLE(d.ARREARS_AMOUNT::VARCHAR)) ARREARS,
        COUNT(*) N_SOURCE
 FROM {D} d WHERE {date_d}
 GROUP BY 1,2,3
),
D_ENTITY AS (
 SELECT MAP_DTE, ENTITY_ID, MAX(DPD) DPD_MAX,
        SUM(ARREARS) ARREARS_AMOUNT, COUNT(*) DPD_LOANS,
        COUNT_IF(DPD IS NOT NULL) DPD_KNOWN_LOANS,
        SUM(N_SOURCE-1) DPD_DUPLICATE_SOURCE_ROWS
 FROM D_LOAN GROUP BY 1,2
),
L_LOAN AS (
 SELECT TO_DATE(l.MAP_DTE) MAP_DTE,
        l.ENTITY_ID::VARCHAR ENTITY_ID, l.INSTRUMENT_ID::VARCHAR INSTRUMENT_ID,
        MAX(TRY_TO_DOUBLE(l.NOMINAL_AMOUNT::VARCHAR)) NOMINAL,
        MAX(TRY_TO_DOUBLE(l.GROSS_CARRYING_AMOUNT::VARCHAR)) GCA,
        MAX(IFF(TRY_TO_DOUBLE(l.DEFAULT_STATUS::VARCHAR) IN (18,19,20),1,0)) IS_DEFAULT,
        MAX(IFF(TRY_TO_DOUBLE(l.FORBEARANCE_STATUS::VARCHAR) IN (3,4,5),1,0)) IS_FORBEARANCE,
        MAX(IFF(l.DIM1::VARCHAR IN ('S2','S3','S3S','S3R'),1,0)) IS_DIM1,
        COUNT(*) N_SOURCE
 FROM {L} l WHERE {date_l}
 GROUP BY 1,2,3
),
L_ENTITY AS (
 SELECT MAP_DTE,ENTITY_ID,COUNT(*) LOAN_COUNT,
        SUM(NOMINAL) NOMINAL_KNOWN_SUM, SUM(GCA) GCA_KNOWN_SUM,
        COUNT_IF(NOMINAL IS NOT NULL) NOMINAL_KNOWN_LOANS,
        COUNT_IF(GCA IS NOT NULL) GCA_KNOWN_LOANS,
        SUM(N_SOURCE-1) DUPLICATE_LOAN_SOURCE_ROWS,
        MAX(IS_DEFAULT) TGT_DEFAULT_SAMEDAY,
        MAX(IS_FORBEARANCE) TGT_FORBEARANCE_SAMEDAY,
        MAX(IS_DIM1) TGT_DIM1_SAMEDAY
 FROM L_LOAN GROUP BY 1,2
),
KEYS AS (
 SELECT MAP_DTE,ENTITY_ID FROM D_ENTITY
 UNION
 SELECT MAP_DTE,ENTITY_ID FROM L_ENTITY
),
ENTITY_HIST AS (
 SELECT e.ENTITY_ID::VARCHAR ENTITY_ID,
        TO_DATE(e.TEK_STRT_DTE) START_DATE,
        COALESCE(TO_DATE(e.TEK_END_DTE),DATE '9999-12-31') END_DATE,
        NULLIF(TRIM(e.GCC_PARENT_ID::VARCHAR),'') GCC_PARENT_ID,
        TRY_TO_DOUBLE(NULLIF(TRIM(e.PD_CRR::VARCHAR),'11111111111')) PD_CRR_RAW
 FROM {E} e
 WHERE e.ENTITY_ID IS NOT NULL AND e.TEK_STRT_DTE IS NOT NULL
),
ENTITY_ASOF AS (
 SELECT k.MAP_DTE,k.ENTITY_ID,h.GCC_PARENT_ID,
        IFF(h.PD_CRR_RAW BETWEEN 0 AND 1,h.PD_CRR_RAW,NULL) PD_CRR,
        COUNT(h.ENTITY_ID) OVER (PARTITION BY k.MAP_DTE,k.ENTITY_ID) ENTITY_VERSION_MATCHES
 FROM KEYS k LEFT JOIN ENTITY_HIST h ON k.ENTITY_ID=h.ENTITY_ID
    AND k.MAP_DTE>=h.START_DATE AND k.MAP_DTE<=h.END_DATE
 QUALIFY ROW_NUMBER() OVER (PARTITION BY k.MAP_DTE,k.ENTITY_ID
                            ORDER BY h.START_DATE DESC,h.END_DATE DESC)=1
),
GEC_DAY AS (
 SELECT l.MAP_DTE,COALESCE(e.GCC_PARENT_ID,e.ENTITY_ID) GEC_KEY,
        SUM(l.NOMINAL_KNOWN_SUM) GEC_NOMINAL_KNOWN_SUM,
        SUM(l.GCA_KNOWN_SUM) GEC_GCA_KNOWN_SUM,
        SUM(l.LOAN_COUNT) GEC_SOURCE_LOANS,
        SUM(l.NOMINAL_KNOWN_LOANS) GEC_NOMINAL_KNOWN_LOANS,
        SUM(l.GCA_KNOWN_LOANS) GEC_GCA_KNOWN_LOANS,
        SUM(l.DUPLICATE_LOAN_SOURCE_ROWS) GEC_DUPLICATE_LOAN_ROWS,
        MIN(IFF(e.ENTITY_VERSION_MATCHES>0,1,0)) GEC_ALL_MAPPED
 FROM L_ENTITY l JOIN ENTITY_ASOF e
   ON l.MAP_DTE=e.MAP_DTE AND l.ENTITY_ID=e.ENTITY_ID
 GROUP BY 1,2
),
BASE AS (
 SELECT e.MAP_DTE,e.ENTITY_ID,e.GCC_PARENT_ID,
        COALESCE(e.GCC_PARENT_ID,e.ENTITY_ID) GEC_KEY,
        e.ENTITY_VERSION_MATCHES,e.PD_CRR,
        d.DPD_MAX,d.ARREARS_AMOUNT,d.DPD_LOANS,d.DPD_KNOWN_LOANS,
        d.DPD_DUPLICATE_SOURCE_ROWS,
        l.LOAN_COUNT,l.NOMINAL_KNOWN_SUM,l.GCA_KNOWN_SUM,
        l.NOMINAL_KNOWN_LOANS,l.GCA_KNOWN_LOANS,l.DUPLICATE_LOAN_SOURCE_ROWS,
        l.TGT_DEFAULT_SAMEDAY,l.TGT_FORBEARANCE_SAMEDAY,l.TGT_DIM1_SAMEDAY,
        g.GEC_SOURCE_LOANS,g.GEC_NOMINAL_KNOWN_LOANS,
        g.GEC_GCA_KNOWN_LOANS,g.GEC_DUPLICATE_LOAN_ROWS,
        IFF(g.GEC_SOURCE_LOANS>0 AND g.GEC_ALL_MAPPED=1
            AND g.GEC_SOURCE_LOANS=g.GEC_NOMINAL_KNOWN_LOANS
            AND g.GEC_DUPLICATE_LOAN_ROWS=0,
            g.GEC_NOMINAL_KNOWN_SUM,NULL) GEC_NOMINAL_CANDIDATE,
        IFF(g.GEC_SOURCE_LOANS>0 AND g.GEC_ALL_MAPPED=1
            AND g.GEC_SOURCE_LOANS=g.GEC_GCA_KNOWN_LOANS
            AND g.GEC_DUPLICATE_LOAN_ROWS=0,
            g.GEC_GCA_KNOWN_SUM,NULL) GEC_GCA_CANDIDATE
 FROM ENTITY_ASOF e
 LEFT JOIN D_ENTITY d ON e.ENTITY_ID=d.ENTITY_ID AND e.MAP_DTE=d.MAP_DTE
 LEFT JOIN L_ENTITY l ON e.ENTITY_ID=l.ENTITY_ID AND e.MAP_DTE=l.MAP_DTE
 LEFT JOIN GEC_DAY g ON g.MAP_DTE=e.MAP_DTE
       AND g.GEC_KEY=COALESCE(e.GCC_PARENT_ID,e.ENTITY_ID)
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
)
,
BASE_PANEL AS (
SELECT q.*,
       IFF(q.PD_CRR IS NULL,NULL,IFF(q.PD_CRR=1,1,0)) TGT_PD_CRR_SAMEDAY,
       IFF(q.DPD_MAX IS NULL,NULL,IFF(q.DPD_MAX>=20,1,0)) TGT_DPD_20_SAMEDAY,
       CASE WHEN q.PD_CRR=1 OR q.DPD_MAX>=20 OR q.TGT_DEFAULT_SAMEDAY=1
                  OR q.TGT_FORBEARANCE_SAMEDAY=1 OR q.TGT_DIM1_SAMEDAY=1 THEN 1
            WHEN q.PD_CRR IS NULL AND q.DPD_MAX IS NULL AND q.LOAN_COUNT IS NULL THEN NULL
            ELSE 0 END AS TGT_ANY_SAMEDAY,
       IFF(q.DPD_MAX IS NOT NULL,1,0) HAS_DPD_VALUE,
       IFF(q.GEC_NOMINAL_CANDIDATE IS NOT NULL,1,0) HAS_GEC_NOMINAL_CANDIDATE
FROM WITH_PREVIOUS q
),
HIST0 AS (
 SELECT e.ENTITY_ID::VARCHAR ENTITY_ID,
        TO_DATE(e.TEK_STRT_DTE) START_DATE,
        COALESCE(TO_DATE(e.TEK_END_DTE),DATE '9999-12-31') END_DATE,
        TRY_TO_DOUBLE(NULLIF(TRIM(e.PD_CRR::VARCHAR),'11111111111')) RAW_PD,
        TRY_TO_DOUBLE(e.GCC_PD_CRR::VARCHAR) RAW_GCC_PD
 FROM {E} e WHERE e.ENTITY_ID IS NOT NULL AND e.TEK_STRT_DTE IS NOT NULL
),
RATING_HIST AS (
 SELECT h.ENTITY_ID,h.START_DATE,h.END_DATE,
        IFF(h.RAW_PD BETWEEN 0 AND 1,h.RAW_PD,NULL) PD_CRR,
        IFF(h.RAW_GCC_PD BETWEEN 0 AND 1,h.RAW_GCC_PD,NULL) GCC_PD_CRR
 FROM HIST0 h
),
SCD AS (
 SELECT h.*,
        {client_notch} CLIENT_NOTCH,
        {gcc_notch} GCC_NOTCH,
        {client_rating} CLIENT_RATING,
        {gcc_rating} GCC_RATING
 FROM RATING_HIST h
),
CURRENT_RATING AS (
 SELECT c.ENTITY_ID,c.MAP_DTE,
        h.PD_CRR,h.GCC_PD_CRR,
        h.CLIENT_NOTCH,h.GCC_NOTCH,h.CLIENT_RATING,h.GCC_RATING,
        COUNT(h.ENTITY_ID) OVER (PARTITION BY c.ENTITY_ID,c.MAP_DTE) CURRENT_VERSION_MATCHES
 FROM BASE_PANEL c
 LEFT JOIN SCD h ON c.ENTITY_ID=h.ENTITY_ID
   AND c.MAP_DTE>=h.START_DATE AND c.MAP_DTE<=h.END_DATE
 QUALIFY ROW_NUMBER() OVER (PARTITION BY c.ENTITY_ID,c.MAP_DTE
                             ORDER BY h.START_DATE DESC,h.END_DATE DESC)=1
),
BEST_12M AS (
 SELECT c.ENTITY_ID,c.MAP_DTE,
        MIN(h.CLIENT_NOTCH) BEST_CLIENT_NOTCH_12M,
        MIN(h.GCC_NOTCH) BEST_GCC_NOTCH_12M,
        MAX(IFF(h.START_DATE<=DATEADD(month,-12,c.MAP_DTE)
                AND h.END_DATE>=DATEADD(month,-12,c.MAP_DTE),1,0)) HAS_LOOKBACK_START,
        COUNT_IF(h.CLIENT_NOTCH IS NOT NULL) CLIENT_RATING_VERSIONS_12M,
        COUNT_IF(h.GCC_NOTCH IS NOT NULL) GCC_RATING_VERSIONS_12M
 FROM BASE_PANEL c
 LEFT JOIN SCD h ON c.ENTITY_ID=h.ENTITY_ID
      AND h.START_DATE<=c.MAP_DTE
      AND h.END_DATE>=DATEADD(month,-12,c.MAP_DTE)
 GROUP BY 1,2
),
R0 AS (
 SELECT c.ENTITY_ID,c.MAP_DTE,
        c.PD_CRR,c.GCC_PD_CRR,c.CLIENT_RATING,c.GCC_RATING,
        c.CLIENT_NOTCH,c.GCC_NOTCH,
        b.BEST_CLIENT_NOTCH_12M,b.BEST_GCC_NOTCH_12M,
        b.HAS_LOOKBACK_START,b.CLIENT_RATING_VERSIONS_12M,
        b.GCC_RATING_VERSIONS_12M,
        c.CURRENT_VERSION_MATCHES,
        IFF(b.HAS_LOOKBACK_START=1 AND b.BEST_CLIENT_NOTCH_12M IS NOT NULL,
            c.CLIENT_NOTCH-b.BEST_CLIENT_NOTCH_12M,NULL) CLIENT_DOWNGRADE_BEST_12M,
        IFF(b.HAS_LOOKBACK_START=1 AND b.BEST_GCC_NOTCH_12M IS NOT NULL,
            c.GCC_NOTCH-b.BEST_GCC_NOTCH_12M,NULL) GCC_DOWNGRADE_BEST_12M
 FROM CURRENT_RATING c JOIN BEST_12M b
   ON c.ENTITY_ID=b.ENTITY_ID AND c.MAP_DTE=b.MAP_DTE
)
,
RATING_PANEL AS (
SELECT r.*,
       IFF(r.CLIENT_NOTCH IS NULL OR r.CLIENT_DOWNGRADE_BEST_12M IS NULL,NULL,
           IFF(r.CLIENT_NOTCH>11 AND r.CLIENT_DOWNGRADE_BEST_12M>=5,1,0)) CR02_CANDIDATE,
       IFF(r.CLIENT_NOTCH IS NULL OR r.CLIENT_DOWNGRADE_BEST_12M IS NULL,NULL,
           IFF(r.CLIENT_NOTCH>13 AND r.CLIENT_DOWNGRADE_BEST_12M>=3,1,0)) CR03_CANDIDATE,
       IFF(r.CLIENT_NOTCH IS NULL OR r.CLIENT_DOWNGRADE_BEST_12M IS NULL,NULL,
           IFF(r.CLIENT_NOTCH>14 AND r.CLIENT_DOWNGRADE_BEST_12M>=2,1,0)) CR04_CANDIDATE,
       IFF(r.CLIENT_NOTCH IS NULL,NULL,IFF(r.CLIENT_NOTCH>16,1,0)) CR05_CANDIDATE,
       IFF(r.GCC_NOTCH IS NULL OR r.GCC_DOWNGRADE_BEST_12M IS NULL,NULL,
           IFF(r.GCC_NOTCH>15 AND r.GCC_DOWNGRADE_BEST_12M>=1,1,0)) GCC01_CANDIDATE,
       IFF(r.GCC_DOWNGRADE_BEST_12M IS NULL OR r.CLIENT_NOTCH IS NULL,NULL,
           IFF(r.GCC_DOWNGRADE_BEST_12M>=3 AND r.CLIENT_NOTCH>13,1,0)) GCC03_CANDIDATE,
       IFF(r.GCC_DOWNGRADE_BEST_12M IS NULL OR r.CLIENT_DOWNGRADE_BEST_12M IS NULL,NULL,
           IFF(r.GCC_DOWNGRADE_BEST_12M>=1 AND r.CLIENT_DOWNGRADE_BEST_12M>=5,1,0)) GCC04_CANDIDATE,
       IFF(r.GCC_DOWNGRADE_BEST_12M IS NULL OR r.CLIENT_DOWNGRADE_BEST_12M IS NULL,NULL,
           IFF(r.GCC_DOWNGRADE_BEST_12M>=2 AND r.CLIENT_DOWNGRADE_BEST_12M>=4,1,0)) GCC05_CANDIDATE
FROM R0 r
)
SELECT b.*,r.GCC_PD_CRR,
       r.CLIENT_RATING,r.GCC_RATING,r.CLIENT_NOTCH,r.GCC_NOTCH,
       r.BEST_CLIENT_NOTCH_12M,r.BEST_GCC_NOTCH_12M,
       r.HAS_LOOKBACK_START,r.CURRENT_VERSION_MATCHES,
       r.CLIENT_DOWNGRADE_BEST_12M,r.GCC_DOWNGRADE_BEST_12M,
       r.CR02_CANDIDATE,r.CR03_CANDIDATE,r.CR04_CANDIDATE,
       r.CR05_CANDIDATE,r.GCC01_CANDIDATE,r.GCC03_CANDIDATE,
       r.GCC04_CANDIDATE,r.GCC05_CANDIDATE,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>5,1,0)) DPD_GT_5,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>5 AND b.PREV_DPD_MAX<=5 THEN 1 ELSE 0 END DPD_NEW_GT_5,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>10,1,0)) DPD_GT_10,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>10 AND b.PREV_DPD_MAX<=10 THEN 1 ELSE 0 END DPD_NEW_GT_10,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>15,1,0)) DPD_GT_15,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>15 AND b.PREV_DPD_MAX<=15 THEN 1 ELSE 0 END DPD_NEW_GT_15,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>20,1,0)) DPD_GT_20,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>20 AND b.PREV_DPD_MAX<=20 THEN 1 ELSE 0 END DPD_NEW_GT_20,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>25,1,0)) DPD_GT_25,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>25 AND b.PREV_DPD_MAX<=25 THEN 1 ELSE 0 END DPD_NEW_GT_25,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>30,1,0)) DPD_GT_30,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>30 AND b.PREV_DPD_MAX<=30 THEN 1 ELSE 0 END DPD_NEW_GT_30,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>35,1,0)) DPD_GT_35,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>35 AND b.PREV_DPD_MAX<=35 THEN 1 ELSE 0 END DPD_NEW_GT_35,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>40,1,0)) DPD_GT_40,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>40 AND b.PREV_DPD_MAX<=40 THEN 1 ELSE 0 END DPD_NEW_GT_40,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>45,1,0)) DPD_GT_45,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>45 AND b.PREV_DPD_MAX<=45 THEN 1 ELSE 0 END DPD_NEW_GT_45,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>50,1,0)) DPD_GT_50,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>50 AND b.PREV_DPD_MAX<=50 THEN 1 ELSE 0 END DPD_NEW_GT_50,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>55,1,0)) DPD_GT_55,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>55 AND b.PREV_DPD_MAX<=55 THEN 1 ELSE 0 END DPD_NEW_GT_55,
       IFF(b.DPD_MAX IS NULL,NULL,IFF(b.DPD_MAX>60,1,0)) DPD_GT_60,
       CASE WHEN b.DPD_MAX IS NULL OR b.PREV_DPD_MAX IS NULL THEN NULL
            WHEN b.DPD_MAX>60 AND b.PREV_DPD_MAX<=60 THEN 1 ELSE 0 END DPD_NEW_GT_60
FROM BASE_PANEL b LEFT JOIN RATING_PANEL r
    ON b.ENTITY_ID=r.ENTITY_ID AND b.MAP_DTE=r.MAP_DTE
"""


def build_sql():
    return FULL_SQL.format(
        E=E, L=L, D=D, date_d=date_filter('d'), date_l=date_filter('l'),
        client_notch=scale_case('h.PD_CRR'),
        gcc_notch=scale_case('h.GCC_PD_CRR'),
        client_rating=scale_case('h.PD_CRR', True),
        gcc_rating=scale_case('h.GCC_PD_CRR', True))


print('=== BIL CORPORATE | DPD + RATING | RESULTS ===')
print('MODE:', MODE, '| DATE RANGE:', START_DATE, END_DATE)
print('ONE output:', OUTPUT_NAME)
print('Indicators: DPD_GT_5..60 and DPD_NEW_GT_5..60, rating rule candidates')
print('GEC materiality is a proxy, no official group exposure claim.')

if MODE == 'CHECK':
    for label, table in (('entity', E), ('loan', L), ('dpd', D)):
        if executor.query_to_df('SELECT 1 FROM {} LIMIT 1'.format(table)).empty:
            raise ValueError(label + ' is empty')
        print(label, 'OK')
    print('CHECK completed. No output written.')
else:
    out_cfg = project.get_dataset(OUTPUT_NAME).get_settings().get_raw()
    pars = out_cfg.get('params') or {}
    if pars.get('connection') != CONNECTION:
        raise ValueError('results output must use Snowflake ' + CONNECTION)
    if str(pars.get('schema', '')).upper() != 'RRG':
        raise ValueError('results output schema must be RRG')
    table_name = str(pars.get('table', '')).replace('${projectKey}', PROJECT).upper()
    if table_name != EXPECTED_TABLE:
        raise ValueError('results physical table must be ' + EXPECTED_TABLE)
    from dataiku import recipe
    outputs = {str(getattr(o, 'name', o)).split('.')[-1]
               for o in recipe.get_outputs(object_type='DATASET')}
    if outputs != {OUTPUT_NAME}:
        raise ValueError('Recipe must have EXACTLY 1 output named results: ' + str(outputs))

    output = dataiku.Dataset(OUTPUT_NAME, project_key=PROJECT)
    # Compile the full SELECT before allowing the recipe to replace results.
    # LIMIT 0 verifies Snowflake syntax, column resolution and type compatibility.
    sql = build_sql()
    print('Compiling full-history SELECT with LIMIT 0 (no output write) ...')
    executor.query_to_df('SELECT * FROM (\n' + sql + '\n) AS SQL_PREFLIGHT LIMIT 0')
    print('SQL compiled. Building full history directly into', EXPECTED_TABLE)
    SQLExecutor2.exec_recipe_fragment(output, sql, overwrite_output_schema=True)

    qc = executor.query_to_df('''
        SELECT COUNT(*) AS N_ROWS,
               COUNT(DISTINCT ENTITY_ID, MAP_DTE) AS N_ENTITY_DAYS,
               COUNT(DISTINCT MAP_DTE) AS N_DATES,
               MIN(MAP_DTE) AS MIN_DATE,
               MAX(MAP_DTE) AS MAX_DATE,
               COUNT_IF(DPD_MAX IS NOT NULL) AS N_DPD_KNOWN,
               COUNT_IF(CLIENT_NOTCH IS NOT NULL) AS N_RATING_KNOWN,
               COUNT_IF(HAS_LOOKBACK_START=1) AS N_12M_LOOKBACK,
               COUNT_IF(GEC_NOMINAL_CANDIDATE IS NOT NULL) AS N_NOMINAL_GEC_KNOWN,
               COUNT_IF(ENTITY_VERSION_MATCHES=0) AS N_NO_ENTITY_HISTORY,
               COUNT_IF(DPD_MAX>5) AS N_DPD_GT_5,
               COUNT_IF(CR03_CANDIDATE=1) AS N_CR03_CANDIDATES
        FROM {}'''.format(qident('RRG') + '.' + qident(EXPECTED_TABLE)))
    print('\n=== OUTPUT RESULTS - QA ===')
    print(qc.to_string(index=False))
    if qc.empty or int(qc.iloc[0]['N_ROWS']) == 0:
        raise ValueError('results is empty')
    if int(qc.iloc[0]['N_ROWS']) != int(qc.iloc[0]['N_ENTITY_DAYS']):
        raise ValueError('Duplicate ENTITY_ID + MAP_DTE in results')
    print('SUCCESS. Batch 01 only: DPD and Rating/GCC, in results.')
    print('WARNING: provisional rating rules and proxy GEC, same-day targets only.')
