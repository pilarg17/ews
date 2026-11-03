# -*- coding: utf-8 -*-
"""BIL Corporate EWS | Batch 01 | DPD, Client Rating and GCC Rating.

Snowflake-native, event-based threshold calibration. No pandas or full history
transfers. Reuses the existing B01 entity/day training; writes two NEW outputs.

Inputs : EWS_training_B01_DPD_RATING
Outputs: results_B01_DPD_RATING, test_results_B01_DPD_RATING

Retail methodology retained: deterioration event (5 target components),
3-month event grouping, 1-11 calendar-month lead window, 2-month signal
merging, crossings for levels / active starts for changes, TP/FP/FN, coverage
1/2/3, and original cost function.

Deliberately not certified as final Corporate calibration: upstream target,
LEGAL_PERSON and mortgage exclusions, and GEC materiality must be validated.
The second, collateral-adjusted cost function remains NULL because its required
collateral input is absent from the B01 training: no imputation is invented.
"""

import re
import dataiku
from dataiku import recipe
from dataiku.core.sql import SQLExecutor2

PROJECT = "VALIDATIONEWSCORPORATEPORTFOLIODAILY"
CONNECTION = "DEV_AA_TEAMS"
TRAINING = "EWS_training_B01_DPD_RATING"
RESULTS = "results_B01_DPD_RATING"
TEST_RESULTS = "test_results_B01_DPD_RATING"

# Retail calibration parameters. Corporate GEC thresholds are NOT assumed here.
MIN_LEAD_MONTHS = 1
MAX_LEAD_MONTHS = 11
MAX_GAP_MONTHS = 3
MAX_SIGNAL_GAP_MONTHS = 2
MIN_EXPOSURE = 0.0       # positive, COMPLETE entity nominal only (not GEC)
COST_TP_FIXED = 1000.0
COST_FP = 200.0
COST_FN_PCT = 0.20
COST_TP_PCT_LATE = 0.20
COST_TP_PCT_EARLY = 0.10

# Grids corresponding to Retail final version, restricted to the B01 scope.
INDICATORS = [
    "DPD_MAX", "CLIENT_NOTCH", "GCC_NOTCH", "PD_CRR", "GCC_PD_CRR",
    "CLIENT_DOWNGRADE_12M", "GCC_DOWNGRADE_12M",
]
PAIRS = [
    ("CLIENT_NOTCH", "CLIENT_DOWNGRADE_12M"),
    ("GCC_NOTCH", "GCC_DOWNGRADE_12M"),
    ("DPD_MAX", "CLIENT_NOTCH"),
]
ACTIVE_INDICATORS = {"CLIENT_DOWNGRADE_12M", "GCC_DOWNGRADE_12M"}

THRESHOLDS = {
    "DPD_MAX": list(range(0, 19)),
    "CLIENT_NOTCH": list(range(1, 20)),
    "GCC_NOTCH": list(range(1, 20)),
    "PD_CRR": [round(i * .01, 8) for i in range(1, 51)],
    "GCC_PD_CRR": [round(i * .01, 8) for i in range(1, 51)],
    "CLIENT_DOWNGRADE_12M": list(range(0, 19)),
    "GCC_DOWNGRADE_12M": list(range(0, 19)),
}
RATING_NAMES = [
    "AAA+", "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+",
    "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC", "D",
]


def q(value):
    value = str(value).replace("${projectKey}", PROJECT)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Invalid Snowflake identifier: " + repr(value))
    return '"' + value + '"'


project = dataiku.api_client().get_project(PROJECT)


def get_table(dataset_name, required):
    cfg = project.get_dataset(dataset_name).get_settings().get_raw()
    p = cfg.get("params") or {}
    if p.get("connection") != CONNECTION:
        raise ValueError("Unexpected connection for " + dataset_name)
    found = {c["name"].upper() for c in dataiku.Dataset(
        dataset_name, project_key=PROJECT).read_schema()}
    missing = set(required) - found
    if missing:
        raise ValueError("Missing fields in {}: {}".format(
            dataset_name, sorted(missing)))
    return q(p["schema"]) + "." + q(p["table"])


required_source = [
    "ENTITY_ID", "MAP_DTE", "DPD_MAX", "PD_CRR", "GCC_PD_CRR",
    "CLIENT_NOTCH", "GCC_NOTCH", "CLIENT_DOWNGRADE_12M",
    "GCC_DOWNGRADE_12M", "LOAN_COUNT", "HAS_VALID_ENTITY_HISTORY",
    "NOMINAL_KNOWN_SUM", "NOMINAL_KNOWN_LOANS", "LOAN_SOURCE_DUPLICATES",
    "TGT_ANY_SAMEDAY", "TGT_DEFAULT_SAMEDAY",
    "TGT_FORBEARANCE_SAMEDAY", "TGT_DIM1_SAMEDAY",
]
SOURCE = get_table(TRAINING, required_source)

# Enforce exact, isolated batch outputs. No original recipes/tables are overwritten.
expected_outputs = {RESULTS, TEST_RESULTS}
configured_outputs = {
    str(getattr(item, "name", item)).split(".")[-1]
    for item in recipe.get_outputs(object_type="DATASET")
}
if configured_outputs != expected_outputs:
    raise ValueError("Configure exactly these outputs: " + ", ".join(sorted(expected_outputs)))

for dataset_name in sorted(expected_outputs):
    settings = project.get_dataset(dataset_name).get_settings().get_raw()
    p = settings.get("params") or {}
    physical_name = str(p.get("table", "")).replace("${projectKey}", PROJECT)
    expected_physical = PROJECT + "_" + dataset_name.upper()
    if (p.get("connection") != CONNECTION
            or str(p.get("schema", "")).upper() != "RRG"
            or physical_name.upper() != expected_physical):
        raise ValueError(
            "Invalid output table for {}. Use connection {}, schema RRG, "
            "table {}".format(dataset_name, CONNECTION, expected_physical)
        )

# New output datasets have no Dataiku schema before their first build.
# Their physical Snowflake table names were checked above, so resolve the
# results table directly rather than calling read_schema() on an empty output.
TARGET_TABLE = q("RRG") + "." + q(PROJECT + "_" + RESULTS.upper())

# Thresholds are tiny numeric literals, never read through a Python dataframe.
grid_values = ",\n        ".join(
    "('{}', {:.8f})".format(ind, float(t))
    for ind in INDICATORS for t in THRESHOLDS[ind]
)

single_fields = ", ".join(INDICATORS)
previous_fields = ",\n           ".join(
    "LAG({name}) OVER(PARTITION BY ENTITY_ID ORDER BY MAP_DTE) AS PREV_{name}".format(name=i)
    for i in INDICATORS
)
previous_choice = "CASE u.PREDICTOR " + " ".join(
    "WHEN '{}' THEN u.PREV_{}".format(i, i) for i in INDICATORS
) + " ELSE NULL END"

pair_grids = "\nUNION ALL\n".join(
    "SELECT '{a} x {b}' AS INDICATOR, '{a}' AS PREDICTOR_A, "
    "'{b}' AS PREDICTOR_B, a.THRESHOLD AS THRESHOLD_A, "
    "b.THRESHOLD AS THRESHOLD_B "
    "FROM GRID_SINGLE a JOIN GRID_SINGLE b "
    "ON a.PREDICTOR='{a}' AND b.PREDICTOR='{b}'".format(a=a, b=b)
    for a, b in PAIRS
)

pair_starts = []
for a, b in PAIRS:
    pair_name = a + " x " + b
    # Either predictor may provide a new start, while the other is active.
    pair_starts.append("""
SELECT '{pair}' AS INDICATOR, '{a}' AS PREDICTOR_A, '{b}' AS PREDICTOR_B,
       s.ENTITY_ID, s.MAP_DTE, s.THRESHOLD_A, g.THRESHOLD AS THRESHOLD_B
FROM SINGLE_START s
JOIN PREP p ON p.ENTITY_ID=s.ENTITY_ID AND p.MAP_DTE=s.MAP_DTE
JOIN GRID_SINGLE g ON g.PREDICTOR='{b}' AND p.{b}>g.THRESHOLD
WHERE s.INDICATOR='{a}' AND p.TARGET=0
UNION ALL
SELECT '{pair}' AS INDICATOR, '{a}' AS PREDICTOR_A, '{b}' AS PREDICTOR_B,
       s.ENTITY_ID, s.MAP_DTE, g.THRESHOLD AS THRESHOLD_A, s.THRESHOLD_A AS THRESHOLD_B
FROM SINGLE_START s
JOIN PREP p ON p.ENTITY_ID=s.ENTITY_ID AND p.MAP_DTE=s.MAP_DTE
JOIN GRID_SINGLE g ON g.PREDICTOR='{a}' AND p.{a}>g.THRESHOLD
WHERE s.INDICATOR='{b}' AND p.TARGET=0
""".format(pair=pair_name, a=a, b=b))
pair_starts_sql = "\nUNION ALL\n".join(pair_starts)

pair_eligible = "\nUNION ALL\n".join(
    "SELECT DISTINCT '{a} x {b}' AS INDICATOR, e.EVENT_ID, "
    "e.EVENT_EXPOSURE FROM EVENTS e JOIN PREP p "
    "ON p.ENTITY_ID=e.ENTITY_ID "
    "AND p.MAP_DTE BETWEEN DATEADD(month,-{lead_hi},e.EVENT_START) "
    "AND DATEADD(month,-{lead_lo},e.EVENT_START) "
    "WHERE p.TARGET=0 AND p.{a} IS NOT NULL AND p.{b} IS NOT NULL".format(
        a=a, b=b, lead_hi=MAX_LEAD_MONTHS, lead_lo=MIN_LEAD_MONTHS)
    for a, b in PAIRS
)

active_grid_names = "(" + ",".join("'{}'".format(v) for v in ACTIVE_INDICATORS) + ")"
rating_a_sql = "CASE WHEN m.PREDICTOR_A IN ('CLIENT_NOTCH','GCC_NOTCH') THEN " \
    + "CASE m.THRESHOLD_A " + " ".join(
        "WHEN {} THEN '{}'".format(i, name) for i, name in enumerate(RATING_NAMES, 1)
    ) + " ELSE NULL END ELSE NULL END"
rating_b_sql = "CASE WHEN m.PREDICTOR_B IN ('CLIENT_NOTCH','GCC_NOTCH') THEN " \
    + "CASE m.THRESHOLD_B " + " ".join(
        "WHEN {} THEN '{}'".format(i, name) for i, name in enumerate(RATING_NAMES, 1)
    ) + " ELSE NULL END ELSE NULL END"

SQL = r"""
WITH
LAST_VALID_DPD AS (
    SELECT MAX(TO_DATE(MAP_DTE)) AS LAST_DPD_DATE FROM {SOURCE}
    WHERE DPD_MAX IS NOT NULL AND LOAN_COUNT>0
),
CORPORATE_PANEL AS (
    SELECT
        t.ENTITY_ID::VARCHAR AS ENTITY_ID,
        TO_DATE(t.MAP_DTE) AS MAP_DTE,
        TRY_TO_DOUBLE(t.DPD_MAX::VARCHAR) AS DPD_MAX,
        TRY_TO_DOUBLE(t.CLIENT_NOTCH::VARCHAR) AS CLIENT_NOTCH,
        TRY_TO_DOUBLE(t.GCC_NOTCH::VARCHAR) AS GCC_NOTCH,
        TRY_TO_DOUBLE(t.PD_CRR::VARCHAR) AS PD_CRR,
        TRY_TO_DOUBLE(t.GCC_PD_CRR::VARCHAR) AS GCC_PD_CRR,
        TRY_TO_DOUBLE(t.CLIENT_DOWNGRADE_12M::VARCHAR) AS CLIENT_DOWNGRADE_12M,
        TRY_TO_DOUBLE(t.GCC_DOWNGRADE_12M::VARCHAR) AS GCC_DOWNGRADE_12M,
        CASE WHEN t.LOAN_COUNT>0
                AND t.LOAN_SOURCE_DUPLICATES=0
                AND t.NOMINAL_KNOWN_LOANS=t.LOAN_COUNT
             THEN TRY_TO_DOUBLE(t.NOMINAL_KNOWN_SUM::VARCHAR) ELSE NULL END AS EXPOSURE,
        TRY_TO_NUMBER(t.TGT_ANY_SAMEDAY::VARCHAR) AS TARGET,
        t.LOAN_COUNT AS LOAN_COUNT
    FROM {SOURCE} t
    WHERE t.LOAN_COUNT>0 AND t.HAS_VALID_ENTITY_HISTORY=1
      AND t.DPD_MAX IS NOT NULL
      AND TO_DATE(t.MAP_DTE)<=(SELECT LAST_DPD_DATE FROM LAST_VALID_DPD)
),
PANEL_DATES AS (
    SELECT MAX(MAP_DTE) AS PANEL_END FROM CORPORATE_PANEL
),
FIRST_OBS AS (
    SELECT ENTITY_ID, MIN(MAP_DTE) AS FIRST_DATE FROM CORPORATE_PANEL
    GROUP BY ENTITY_ID
),
EVENT_ACTIVE AS (
    SELECT ENTITY_ID,MAP_DTE,
           LAG(MAP_DTE) OVER(PARTITION BY ENTITY_ID ORDER BY MAP_DTE) AS PREV_ACTIVE
    FROM CORPORATE_PANEL WHERE TARGET=1
),
EVENT_GROUPS AS (
    SELECT ENTITY_ID,MAP_DTE,
           SUM(CASE WHEN PREV_ACTIVE IS NULL OR MAP_DTE>DATEADD(month,{event_gap},PREV_ACTIVE)
                    THEN 1 ELSE 0 END)
               OVER(PARTITION BY ENTITY_ID ORDER BY MAP_DTE ROWS UNBOUNDED PRECEDING) AS EVENT_GROUP
    FROM EVENT_ACTIVE
),
EVENT_DATES AS (
    SELECT ENTITY_ID,EVENT_GROUP,MIN(MAP_DTE) AS EVENT_START,
           MAX(MAP_DTE) AS EVENT_END
    FROM EVENT_GROUPS GROUP BY ENTITY_ID,EVENT_GROUP
),
EVENTS AS (
    SELECT CONCAT(e.ENTITY_ID,'|',TO_VARCHAR(e.EVENT_START)) AS EVENT_ID,
           e.ENTITY_ID,e.EVENT_START,e.EVENT_END,p.EXPOSURE AS EVENT_EXPOSURE
    FROM EVENT_DATES e
    JOIN FIRST_OBS f ON f.ENTITY_ID=e.ENTITY_ID
    JOIN CORPORATE_PANEL p ON p.ENTITY_ID=e.ENTITY_ID AND p.MAP_DTE=e.EVENT_START
    WHERE e.EVENT_START>=DATEADD(month,{lead_hi},f.FIRST_DATE)
      AND p.EXPOSURE>{min_exposure}
),
# A new threshold evaluation uses only observations with fully known
# entity-level nominal; unknown exposures never become zeroes.
PREP AS (
    SELECT ENTITY_ID,MAP_DTE,TARGET,EXPOSURE,
           {single_fields},
           LAG(MAP_DTE) OVER(PARTITION BY ENTITY_ID ORDER BY MAP_DTE) AS PREV_DTE,
           LAG(TARGET) OVER(PARTITION BY ENTITY_ID ORDER BY MAP_DTE) AS PREV_TARGET,
           {previous_fields}
    FROM CORPORATE_PANEL
    WHERE EXPOSURE>{min_exposure}
),
GRID_SINGLE AS (
    SELECT COLUMN1::VARCHAR AS PREDICTOR,COLUMN2::DOUBLE AS THRESHOLD
    FROM VALUES {grid_values}
),
GRID_ALL AS (
    SELECT PREDICTOR AS INDICATOR,PREDICTOR AS PREDICTOR_A,
           CAST(NULL AS VARCHAR) AS PREDICTOR_B,
           THRESHOLD AS THRESHOLD_A,CAST(NULL AS DOUBLE) AS THRESHOLD_B
    FROM GRID_SINGLE
    UNION ALL
    {pair_grids}
),
FLAT_FEATURES AS (
    SELECT u.ENTITY_ID,u.MAP_DTE,u.TARGET,u.PREV_TARGET,u.PREV_DTE,
           u.EXPOSURE,u.PREDICTOR,u.FEATURE_VALUE,
           {previous_choice} AS PREVIOUS_VALUE
    FROM PREP f
    UNPIVOT EXCLUDE NULLS (FEATURE_VALUE FOR PREDICTOR IN ({single_fields})) u
),
SINGLE_START AS (
    SELECT f.PREDICTOR AS INDICATOR,f.PREDICTOR AS PREDICTOR_A,
           CAST(NULL AS VARCHAR) AS PREDICTOR_B,
           f.ENTITY_ID,f.MAP_DTE,g.THRESHOLD AS THRESHOLD_A,
           CAST(NULL AS DOUBLE) AS THRESHOLD_B
    FROM FLAT_FEATURES f
    JOIN GRID_SINGLE g ON g.PREDICTOR=f.PREDICTOR
      AND f.FEATURE_VALUE>g.THRESHOLD
    WHERE f.TARGET=0 AND (
       (f.PREDICTOR NOT IN {active_grid_names}
        AND f.PREV_TARGET=0
        AND DATEDIFF(day,f.PREV_DTE,f.MAP_DTE)=1
        AND f.PREVIOUS_VALUE IS NOT NULL
        AND f.PREVIOUS_VALUE<=g.THRESHOLD)
       OR
       (f.PREDICTOR IN {active_grid_names}
        AND (f.PREV_TARGET IS NULL OR f.PREV_TARGET<>0
             OR f.PREV_DTE IS NULL
             OR DATEDIFF(day,f.PREV_DTE,f.MAP_DTE)<>1
             OR f.PREVIOUS_VALUE IS NULL
             OR f.PREVIOUS_VALUE<=g.THRESHOLD))
    )
),
PAIR_START AS (
    {pair_starts_sql}
),
SIGNAL_STARTS AS (
    SELECT DISTINCT INDICATOR,PREDICTOR_A,PREDICTOR_B,
           ENTITY_ID,MAP_DTE,THRESHOLD_A,THRESHOLD_B
    FROM (
        SELECT * FROM SINGLE_START
        UNION ALL
        SELECT * FROM PAIR_START
    )
    WHERE MAP_DTE<=DATEADD(month,-{lead_lo},(SELECT PANEL_END FROM PANEL_DATES))
),
SIGNAL_PREVIOUS AS (
    SELECT s.*,
           LAG(MAP_DTE) OVER(PARTITION BY INDICATOR,THRESHOLD_A,THRESHOLD_B,
                                     ENTITY_ID ORDER BY MAP_DTE) AS PREV_START
    FROM SIGNAL_STARTS s
),
SIGNAL_GROUPS AS (
    SELECT s.*,
           SUM(CASE WHEN PREV_START IS NULL
                          OR MAP_DTE>DATEADD(month,{signal_gap},PREV_START)
                    THEN 1 ELSE 0 END)
               OVER(PARTITION BY INDICATOR,THRESHOLD_A,THRESHOLD_B,
                                     ENTITY_ID ORDER BY MAP_DTE
                    ROWS UNBOUNDED PRECEDING) AS SIGNAL_GROUP
    FROM SIGNAL_PREVIOUS s
),
SIGNALS AS (
    SELECT INDICATOR,PREDICTOR_A,PREDICTOR_B,THRESHOLD_A,THRESHOLD_B,
           ENTITY_ID,MIN(MAP_DTE) AS SIGNAL_START
    FROM SIGNAL_GROUPS
    GROUP BY 1,2,3,4,5,6,SIGNAL_GROUP
),
INDICATOR_EVENTS AS (
    SELECT DISTINCT f.PREDICTOR AS INDICATOR,e.EVENT_ID,e.EVENT_EXPOSURE
    FROM EVENTS e
    JOIN FLAT_FEATURES f ON f.ENTITY_ID=e.ENTITY_ID
       AND f.MAP_DTE BETWEEN DATEADD(month,-{lead_hi},e.EVENT_START)
                          AND DATEADD(month,-{lead_lo},e.EVENT_START)
    WHERE f.TARGET=0
    UNION ALL
    {pair_eligible}
),
EVENT_COUNTS AS (
    SELECT INDICATOR,COUNT(DISTINCT EVENT_ID) AS N_EVENTS,
           SUM(EVENT_EXPOSURE) AS SUM_GCA_EVENTS
    FROM INDICATOR_EVENTS GROUP BY INDICATOR
),
VALID_PAIRS AS (
    SELECT s.INDICATOR,s.THRESHOLD_A,s.THRESHOLD_B,
           s.ENTITY_ID,s.SIGNAL_START,
           e.EVENT_ID,e.EVENT_START,e.EVENT_EXPOSURE
    FROM SIGNALS s JOIN EVENTS e ON e.ENTITY_ID=s.ENTITY_ID
       AND s.SIGNAL_START BETWEEN DATEADD(month,-{lead_hi},e.EVENT_START)
                              AND DATEADD(month,-{lead_lo},e.EVENT_START)
    JOIN INDICATOR_EVENTS ie ON ie.INDICATOR=s.INDICATOR
                             AND ie.EVENT_ID=e.EVENT_ID
),
FIRST_MATCH AS (
    SELECT * FROM VALID_PAIRS
    QUALIFY ROW_NUMBER() OVER(
        PARTITION BY INDICATOR,THRESHOLD_A,THRESHOLD_B,EVENT_ID
        ORDER BY SIGNAL_START
    )=1
),
TP AS (
    SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
           COUNT(*) AS N_TP,
           SUM(EVENT_EXPOSURE) AS SUM_GCA_TP,
           SUM(EVENT_EXPOSURE*(
               {cost_tp_late} + ({cost_tp_early}-{cost_tp_late}) *
               (LEAST({lead_hi},GREATEST({lead_lo},
                 DATEDIFF(day,SIGNAL_START,EVENT_START)/(365.25/12.0)))-{lead_lo})
                /({lead_hi}-{lead_lo})
           )) AS SUM_GCA_TP_WTD
    FROM FIRST_MATCH GROUP BY 1,2,3
),
DETECTED AS (
    SELECT INDICATOR,COUNT(DISTINCT EVENT_ID) AS N_EVENTS_DETECTABLE
    FROM VALID_PAIRS GROUP BY INDICATOR
),
MATCHED_SIGNALS AS (
    SELECT DISTINCT INDICATOR,THRESHOLD_A,THRESHOLD_B,ENTITY_ID,SIGNAL_START
    FROM VALID_PAIRS
),
SIGNAL_COUNTS AS (
    SELECT s.INDICATOR,s.THRESHOLD_A,s.THRESHOLD_B,
           COUNT(*) AS N_ALERTS,
           COUNT_IF(m.SIGNAL_START IS NULL) AS N_FP
    FROM SIGNALS s LEFT JOIN MATCHED_SIGNALS m
      ON s.INDICATOR=m.INDICATOR AND s.THRESHOLD_A=m.THRESHOLD_A
     AND EQUAL_NULL(s.THRESHOLD_B,m.THRESHOLD_B)
     AND s.ENTITY_ID=m.ENTITY_ID AND s.SIGNAL_START=m.SIGNAL_START
    GROUP BY 1,2,3
),
SIGNAL_MONTHS AS (
    SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
           DATE_TRUNC(month,SIGNAL_START) AS SIGNAL_MONTH,
           COUNT(*) AS N_MONTHLY_SIGNALS
    FROM SIGNALS GROUP BY 1,2,3,4
),
MONTHLY_MAX AS (
    SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
           MAX(N_MONTHLY_SIGNALS) AS MAX_MONTHLY_SIGNALS
    FROM SIGNAL_MONTHS GROUP BY 1,2,3
),
SIGNAL_DAYS AS (
    SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
           SIGNAL_START,COUNT(*) AS N_DAILY_SIGNALS
    FROM SIGNALS GROUP BY 1,2,3,4
),
DAILY_MAX AS (
    SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
           MAX(N_DAILY_SIGNALS) AS MAX_DAILY_SIGNALS
    FROM SIGNAL_DAYS GROUP BY 1,2,3
),
SCOPE_COUNTS AS (
    SELECT COUNT(DISTINCT ENTITY_ID) AS N_ENTITIES,
           GREATEST(COUNT(DISTINCT DATE_TRUNC(month,MAP_DTE)),1) AS N_MONTHS,
           GREATEST(COUNT(DISTINCT MAP_DTE),1) AS N_DAYS
    FROM PREP WHERE TARGET=0
),
METRICS AS (
    SELECT g.INDICATOR,g.PREDICTOR_A,g.PREDICTOR_B,
           g.THRESHOLD_A,g.THRESHOLD_B,
           c.N_ENTITIES,
           COALESCE(t.N_TP,0) AS N_TP,
           COALESCE(s.N_FP,0) AS N_FP,
           GREATEST(COALESCE(e.N_EVENTS,0)-COALESCE(t.N_TP,0),0) AS N_FN,
           COALESCE(t.N_TP,0)+COALESCE(s.N_FP,0) AS N_ALERTS,
           COALESCE(e.N_EVENTS,0) AS N_EVENTS,
           COALESCE(d.N_EVENTS_DETECTABLE,0) AS N_EVENTS_DETECTABLE,
           COALESCE(t.SUM_GCA_TP,0) AS SUM_GCA_TP,
           COALESCE(t.SUM_GCA_TP_WTD,0) AS SUM_GCA_TP_WTD,
           GREATEST(COALESCE(e.SUM_GCA_EVENTS,0)-COALESCE(t.SUM_GCA_TP,0),0) AS SUM_GCA_FN,
           COALESCE(mm.MAX_MONTHLY_SIGNALS,0) AS MAX_MONTHLY_ALERTS,
           COALESCE(dm.MAX_DAILY_SIGNALS,0) AS MAX_DAILY_ALERTS,
           c.N_MONTHS,c.N_DAYS
    FROM GRID_ALL g
    CROSS JOIN SCOPE_COUNTS c
    LEFT JOIN TP t ON t.INDICATOR=g.INDICATOR
      AND t.THRESHOLD_A=g.THRESHOLD_A
      AND EQUAL_NULL(t.THRESHOLD_B,g.THRESHOLD_B)
    LEFT JOIN SIGNAL_COUNTS s ON s.INDICATOR=g.INDICATOR
      AND s.THRESHOLD_A=g.THRESHOLD_A
      AND EQUAL_NULL(s.THRESHOLD_B,g.THRESHOLD_B)
    LEFT JOIN EVENT_COUNTS e ON e.INDICATOR=g.INDICATOR
    LEFT JOIN DETECTED d ON d.INDICATOR=g.INDICATOR
    LEFT JOIN MONTHLY_MAX mm ON mm.INDICATOR=g.INDICATOR
      AND mm.THRESHOLD_A=g.THRESHOLD_A
      AND EQUAL_NULL(mm.THRESHOLD_B,g.THRESHOLD_B)
    LEFT JOIN DAILY_MAX dm ON dm.INDICATOR=g.INDICATOR
      AND dm.THRESHOLD_A=g.THRESHOLD_A
      AND EQUAL_NULL(dm.THRESHOLD_B,g.THRESHOLD_B)
),
COSTED AS (
    SELECT m.*,
           {cost_tp_fixed}*N_TP + {cost_fp}*N_FP
              + SUM_GCA_TP_WTD + {cost_fn_pct}*SUM_GCA_FN AS TOTAL_COST,
           IFF(N_ALERTS>0,N_TP/N_ALERTS,NULL) AS ACCURACY,
           IFF(N_EVENTS>0,N_TP/N_EVENTS,NULL) AS COVERAGE,
           IFF(N_EVENTS_DETECTABLE>0,N_TP/N_EVENTS_DETECTABLE,NULL) AS COVERAGE_2,
           IFF(MAX(N_TP) OVER (PARTITION BY INDICATOR)>0,
               N_TP/MAX(N_TP) OVER (PARTITION BY INDICATOR),NULL) AS COVERAGE_3
    FROM METRICS m
),
FINAL AS (
    SELECT m.*,
           ROW_NUMBER() OVER(PARTITION BY INDICATOR
               ORDER BY TOTAL_COST ASC,THRESHOLD_A ASC,THRESHOLD_B ASC NULLS FIRST) AS COST_RANK,
           MAX(N_TP) OVER(PARTITION BY INDICATOR) AS N_EVENTS_REACHABLE
    FROM COSTED m
)
SELECT
    INDICATOR,PREDICTOR_A,PREDICTOR_B,N_ENTITIES,THRESHOLD_A,THRESHOLD_B,
    {rating_a_sql} AS THRESHOLD_A_RATING_LABEL,
    {rating_b_sql} AS THRESHOLD_B_RATING_LABEL,
    N_TP,N_FP,N_FN,N_ALERTS,MAX_MONTHLY_ALERTS,MAX_DAILY_ALERTS,
    N_EVENTS,N_EVENTS_DETECTABLE,N_EVENTS_REACHABLE,
    ROUND(ACCURACY,3) AS ACCURACY,
    ROUND(COVERAGE,3) AS COVERAGE,
    ROUND(COVERAGE_2,3) AS COVERAGE_2,
    ROUND(COVERAGE_3,3) AS COVERAGE_3,
    SUM_GCA_TP,SUM_GCA_TP_WTD,SUM_GCA_FN,
    ROUND(TOTAL_COST,2) AS TOTAL_COST,
    IFF(COST_RANK=1,TRUE,FALSE) AS OPTIMAL,
    CAST(NULL AS DOUBLE) AS SUM_COST2_TP,
    CAST(NULL AS DOUBLE) AS SUM_COST2_FN,
    CAST(NULL AS DOUBLE) AS TOTAL_COST_2,
    CAST(NULL AS BOOLEAN) AS OPTIMAL_COST_2,
    ROUND(N_ALERTS/N_MONTHS,0) AS AV_NR_MON_ALERTS,
    ROUND(N_ALERTS/N_MONTHS,0) AS AV_EXP_NR_MON_ALERTS,
    ROUND(MAX_MONTHLY_ALERTS,0) AS MAX_EXP_NR_MON_ALERTS,
    ROUND(N_ALERTS/N_DAYS,3) AS AV_NR_DAY_ALERTS,
    'OWN_NOMINAL_COMPLETE' AS EXPOSURE_BASIS,
    'NOT_BUSINESS_VALIDATED' AS CALIBRATION_STATUS
FROM FINAL m
""".replace("# A new threshold evaluation uses only observations with fully known\n# entity-level nominal; unknown exposures never become zeroes.\n", "-- Exposure must be complete to evaluate monetary costs.\n").format(
    SOURCE=SOURCE,
    event_gap=MAX_GAP_MONTHS,
    lead_hi=MAX_LEAD_MONTHS,
    lead_lo=MIN_LEAD_MONTHS,
    signal_gap=MAX_SIGNAL_GAP_MONTHS,
    min_exposure=MIN_EXPOSURE,
    single_fields=single_fields,
    previous_fields=previous_fields,
    grid_values=grid_values,
    pair_grids=pair_grids,
    previous_choice=previous_choice,
    active_grid_names=active_grid_names,
    pair_starts_sql=pair_starts_sql,
    pair_eligible=pair_eligible,
    cost_tp_fixed=COST_TP_FIXED,
    cost_fp=COST_FP,
    cost_fn_pct=COST_FN_PCT,
    cost_tp_late=COST_TP_PCT_LATE,
    cost_tp_early=COST_TP_PCT_EARLY,
    rating_a_sql=rating_a_sql,
    rating_b_sql=rating_b_sql,
)

# Populate the official Retail-style results structure, with auditable
# Corporate-specific fields; then build QA as a separate small SQL query.
output_results = dataiku.Dataset(RESULTS, project_key=PROJECT)
SQLExecutor2.exec_recipe_fragment(output_results, SQL, overwrite_output_schema=True)

QA_SQL = r"""
WITH SOURCE_QA AS (
 SELECT
  COUNT(*) AS TOTAL_ROWS,
  COUNT_IF(LOAN_COUNT>0) AS LOAN_ROWS,
  COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1) AS CORP_MAPPED_ROWS,
  COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
       AND DPD_MAX IS NOT NULL) AS CORP_DPD_ROWS,
  COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
       AND DPD_MAX IS NOT NULL AND TGT_ANY_SAMEDAY IS NULL) AS UNKNOWN_TARGET_ROWS,
  COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
       AND DPD_MAX IS NOT NULL AND TGT_ANY_SAMEDAY=1) AS POSITIVE_TARGET_ROWS,
  COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
       AND DPD_MAX IS NOT NULL AND TGT_ANY_SAMEDAY=0) AS NEGATIVE_TARGET_ROWS,
  COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
       AND DPD_MAX IS NOT NULL AND CLIENT_NOTCH IS NOT NULL) AS CLIENT_RATING_ROWS,
  COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
       AND DPD_MAX IS NOT NULL AND GCC_NOTCH IS NOT NULL) AS GCC_RATING_ROWS,
  COUNT_IF(LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
       AND DPD_MAX IS NOT NULL
       AND NOMINAL_KNOWN_LOANS=LOAN_COUNT
       AND LOAN_SOURCE_DUPLICATES=0
       AND NOMINAL_KNOWN_SUM>0) AS COMPLETE_EXPOSURE_ROWS,
  MAX(IFF(DPD_MAX IS NOT NULL AND LOAN_COUNT>0,TO_DATE(MAP_DTE),NULL)) AS LAST_DPD_DATE,
  MAX(TO_DATE(MAP_DTE)) AS LAST_TRAINING_DATE
 FROM {SOURCE}
),
RESULT_QA AS (
 SELECT COUNT(*) AS RESULT_ROWS,
        COUNT_IF(N_TP+N_FP<>N_ALERTS) AS BAD_ALERT_IDENTITIES,
        COUNT_IF(N_TP+N_FN<>N_EVENTS) AS BAD_EVENT_IDENTITIES,
        COUNT_IF(ACCURACY<0 OR ACCURACY>1 OR COVERAGE<0 OR COVERAGE>1
                 OR COVERAGE_2<0 OR COVERAGE_2>1 OR COVERAGE_3<0 OR COVERAGE_3>1) AS BAD_METRIC_RANGES,
        COUNT_IF(COVERAGE_2<COVERAGE OR COVERAGE_3<COVERAGE_2) AS BAD_COVERAGE_ORDER,
        COUNT(DISTINCT INDICATOR) AS INDICATORS,
        COUNT_IF(OPTIMAL) AS OPTIMAL_ROWS,
        COUNT_IF(N_EVENTS>0) AS RESULTS_WITH_EVENTS
 FROM {RESULTS_TABLE}
),
CHECKS AS (
 SELECT 'MODEL_CHECK' AS QA_TYPE,'At least one result' AS CHECK_NAME,
        IFF(r.RESULT_ROWS>0,TRUE,FALSE) AS CHECK_PASSED,
        r.RESULT_ROWS::DOUBLE AS CHECK_VALUE,1.0 AS EXPECTED_VALUE
 FROM RESULT_QA r
 UNION ALL SELECT 'MODEL_CHECK','TP+FP equals alerts',IFF(r.BAD_ALERT_IDENTITIES=0,TRUE,FALSE),r.BAD_ALERT_IDENTITIES::DOUBLE,0.0 FROM RESULT_QA r
 UNION ALL SELECT 'MODEL_CHECK','TP+FN equals events',IFF(r.BAD_EVENT_IDENTITIES=0,TRUE,FALSE),r.BAD_EVENT_IDENTITIES::DOUBLE,0.0 FROM RESULT_QA r
 UNION ALL SELECT 'MODEL_CHECK','Metrics within 0-1',IFF(r.BAD_METRIC_RANGES=0,TRUE,FALSE),r.BAD_METRIC_RANGES::DOUBLE,0.0 FROM RESULT_QA r
 UNION ALL SELECT 'MODEL_CHECK','Coverage order',IFF(r.BAD_COVERAGE_ORDER=0,TRUE,FALSE),r.BAD_COVERAGE_ORDER::DOUBLE,0.0 FROM RESULT_QA r
 UNION ALL SELECT 'MODEL_CHECK','Exactly one optimal per indicator',IFF(r.OPTIMAL_ROWS=r.INDICATORS,TRUE,FALSE),r.OPTIMAL_ROWS::DOUBLE,r.INDICATORS::DOUBLE FROM RESULT_QA r
 UNION ALL SELECT 'MODEL_CHECK','At least one indicator with historical events',IFF(r.RESULTS_WITH_EVENTS>0,TRUE,FALSE),r.RESULTS_WITH_EVENTS::DOUBLE,1.0 FROM RESULT_QA r
 UNION ALL SELECT 'DATA_QUALITY','Corporate mapping rows',NULL,s.CORP_MAPPED_ROWS::DOUBLE,NULL FROM SOURCE_QA s
 UNION ALL SELECT 'DATA_QUALITY','DPD available on Corporate-mapped loan rows',NULL,s.CORP_DPD_ROWS::DOUBLE,NULL FROM SOURCE_QA s
 UNION ALL SELECT 'DATA_QUALITY','Unknown target rows (Corporate DPD scope)',NULL,s.UNKNOWN_TARGET_ROWS::DOUBLE,NULL FROM SOURCE_QA s
 UNION ALL SELECT 'DATA_QUALITY','Positive target rows (Corporate DPD scope)',NULL,s.POSITIVE_TARGET_ROWS::DOUBLE,NULL FROM SOURCE_QA s
 UNION ALL SELECT 'DATA_QUALITY','Negative target rows (Corporate DPD scope)',NULL,s.NEGATIVE_TARGET_ROWS::DOUBLE,NULL FROM SOURCE_QA s
 UNION ALL SELECT 'DATA_QUALITY','Client rating available rows',NULL,s.CLIENT_RATING_ROWS::DOUBLE,NULL FROM SOURCE_QA s
 UNION ALL SELECT 'DATA_QUALITY','GCC rating available rows',NULL,s.GCC_RATING_ROWS::DOUBLE,NULL FROM SOURCE_QA s
 UNION ALL SELECT 'DATA_QUALITY','Complete own nominal exposure rows',NULL,s.COMPLETE_EXPOSURE_ROWS::DOUBLE,NULL FROM SOURCE_QA s
 UNION ALL SELECT 'DATA_QUALITY','Target unknown share in mapped DPD rows',NULL,
     IFF(s.CORP_DPD_ROWS>0,s.UNKNOWN_TARGET_ROWS/s.CORP_DPD_ROWS,NULL),NULL FROM SOURCE_QA s
 UNION ALL SELECT 'METHODOLOGY','1-11 month lead window',TRUE,1.0,1.0 FROM SOURCE_QA
 UNION ALL SELECT 'METHODOLOGY','Retail 3-month event grouping',TRUE,3.0,3.0 FROM SOURCE_QA
 UNION ALL SELECT 'METHODOLOGY','Retail 2-month signal merging',TRUE,2.0,2.0 FROM SOURCE_QA
 UNION ALL SELECT 'BUSINESS_REVIEW','Mortgage exclusions require source-level verification',NULL,NULL,NULL FROM SOURCE_QA
 UNION ALL SELECT 'BUSINESS_REVIEW','GEC exposure/materiality not certified in B01',NULL,NULL,NULL FROM SOURCE_QA
 UNION ALL SELECT 'BUSINESS_REVIEW','Target DIM1 and default codes require bank sign-off',NULL,NULL,NULL FROM SOURCE_QA
 UNION ALL SELECT 'BUSINESS_REVIEW','Collateral cost 2 not computed without collateral',NULL,NULL,NULL FROM SOURCE_QA
),
RESULT_BEST AS (
 SELECT INDICATOR,PREDICTOR_A,PREDICTOR_B,THRESHOLD_A,THRESHOLD_B,
        N_ALERTS,AV_NR_MON_ALERTS,AV_EXP_NR_MON_ALERTS,MAX_EXP_NR_MON_ALERTS,
        ACCURACY,COVERAGE,COVERAGE_2,COVERAGE_3,TOTAL_COST
 FROM {RESULTS_TABLE} WHERE OPTIMAL=TRUE
)
SELECT c.QA_TYPE,c.CHECK_NAME,
       CAST(NULL AS VARCHAR) AS INDICATOR,
       CAST(NULL AS VARCHAR) AS PREDICTOR_A,
       CAST(NULL AS VARCHAR) AS PREDICTOR_B,
       CAST(NULL AS VARCHAR) AS UNAVAILABLE_COLUMNS,
       CAST(NULL AS VARCHAR) AS ENTITY_ID,
       CAST(NULL AS DATE) AS MAP_DTE,
       CAST(NULL AS VARCHAR) AS SIGNAL_LOGIC,
       CAST(NULL AS VARCHAR) AS DIRECTION,
       CAST(NULL AS DOUBLE) AS THRESHOLD_A,
       CAST(NULL AS DOUBLE) AS THRESHOLD_B,
       CAST(NULL AS DOUBLE) AS PREDICTOR_VALUE,
       CAST(NULL AS DOUBLE) AS PREVIOUS_VALUE,
       CAST(NULL AS BOOLEAN) AS ALERT_GENERATED,
       c.CHECK_PASSED,c.CHECK_VALUE,c.EXPECTED_VALUE,
       CAST(NULL AS NUMBER) AS N_ALERTS,
       CAST(NULL AS DOUBLE) AS AV_NR_MON_ALERTS,
       CAST(NULL AS DOUBLE) AS AV_EXP_NR_MON_ALERTS,
       CAST(NULL AS DOUBLE) AS MAX_EXP_NR_MON_ALERTS,
       CAST(NULL AS DOUBLE) AS ACCURACY,
       CAST(NULL AS DOUBLE) AS COVERAGE,
       CAST(NULL AS DOUBLE) AS COVERAGE_2,
       CAST(NULL AS DOUBLE) AS COVERAGE_3,
       CAST(NULL AS DOUBLE) AS TOTAL_COST
FROM CHECKS c
UNION ALL
SELECT 'OPTIMAL_THRESHOLD',CAST(NULL AS VARCHAR),
       b.INDICATOR,b.PREDICTOR_A,b.PREDICTOR_B,
       CAST(NULL AS VARCHAR),CAST(NULL AS VARCHAR),CAST(NULL AS DATE),
       CAST(NULL AS VARCHAR),CAST(NULL AS VARCHAR),
       b.THRESHOLD_A,b.THRESHOLD_B,
       CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),CAST(NULL AS BOOLEAN),
       CAST(NULL AS BOOLEAN),CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),
       b.N_ALERTS,b.AV_NR_MON_ALERTS,b.AV_EXP_NR_MON_ALERTS,
       b.MAX_EXP_NR_MON_ALERTS,b.ACCURACY,b.COVERAGE,b.COVERAGE_2,
       b.COVERAGE_3,b.TOTAL_COST
FROM RESULT_BEST b
""".format(SOURCE=SOURCE, RESULTS_TABLE=TARGET_TABLE)

output_tests = dataiku.Dataset(TEST_RESULTS, project_key=PROJECT)
SQLExecutor2.exec_recipe_fragment(output_tests, QA_SQL, overwrite_output_schema=True)
print("B01 model completed. Model outputs are NOT business-certified until QA sign-off.")
