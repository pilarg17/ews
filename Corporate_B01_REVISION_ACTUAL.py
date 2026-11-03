# Corporate EWS - B01 - DPD / PD / Rating / GCC | Review (read-only)
import re
import dataiku
import pandas as pd
from IPython.display import display
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
project = dataiku.api_client().get_project(PROJECT)
executor = SQLExecutor2(connection=CONNECTION)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 120)

def table(dataset_name):
    params = project.get_dataset(dataset_name).get_settings().get_raw().get('params', {})
    if params.get('connection') != CONNECTION:
        raise ValueError('Incorrect connection for '+dataset_name)
    def ident(value):
        value = str(value).replace('${projectKey}', PROJECT)
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
            raise ValueError('Invalid identifier: '+value)
        return '"'+value+'"'
    return ident(params['schema'])+'.'+ident(params['table'])

R = table('results_B01_DPD_RATING')
Q = table('test_results_B01_DPD_RATING')

print('\n1 - Optimal results: DPD, Rating, PD and GCC (Retail names)')
display(executor.query_to_df(f'''
SELECT INDICATOR, THRESHOLD_A, THRESHOLD_A_RATING_LABEL,
       THRESHOLD_B, THRESHOLD_B_RATING_LABEL,
       N_EVENTS, N_TP, N_FP, N_FN,
       N_ALERTS, N_ALERTS_SIGNAL_STARTS,
       ACCURACY, COVERAGE, COVERAGE_2, COVERAGE_3,
       AV_NR_MON_ALERTS, MAX_MONTHLY_ALERTS, MAX_DAILY_ALERTS,
       AV_EXP_NR_MON_ALERTS, MAX_EXP_NR_MON_ALERTS,
       TOTAL_COST, CALIBRATION_STATUS
FROM {R}
WHERE OPTIMAL = TRUE AND INDICATOR <> 'dpd x GEC_exposure'
ORDER BY INDICATOR
'''))

print('\n2 - DPD: best cutoff at each GEC exposure level (20 levels)')
display(executor.query_to_df(f'''
SELECT GEC_EXPOSURE_THRESHOLD, THRESHOLD_A AS DPD_DAYS,
       N_EVENTS, N_TP, N_FP, N_FN,
       N_ALERTS, N_ALERTS_SIGNAL_STARTS,
       ACCURACY, COVERAGE, COVERAGE_2, COVERAGE_3,
       AV_NR_MON_ALERTS, MAX_MONTHLY_ALERTS, MAX_DAILY_ALERTS,
       TOTAL_COST, CALIBRATION_STATUS
FROM {R}
WHERE INDICATOR = 'dpd x GEC_exposure' AND OPTIMAL_AT_GEC=TRUE
ORDER BY GEC_EXPOSURE_THRESHOLD
'''))

print('\n3 - DPD: compare all 12 cutoff values at GEC 50k')
display(executor.query_to_df(f'''
SELECT THRESHOLD_A AS DPD_DAYS,
       N_EVENTS, N_TP, N_FP, N_FN,
       N_ALERTS, N_ALERTS_SIGNAL_STARTS,
       ACCURACY, COVERAGE, COVERAGE_2, COVERAGE_3,
       AV_NR_MON_ALERTS, MAX_MONTHLY_ALERTS, MAX_DAILY_ALERTS,
       TOTAL_COST
FROM {R}
WHERE INDICATOR = 'dpd x GEC_exposure'
  AND GEC_EXPOSURE_THRESHOLD = 50000
ORDER BY DPD_DAYS
'''))

print('\n4 - All configurations: indicators, volume and availability')
display(executor.query_to_df(f'''
SELECT CASE WHEN INDICATOR IN ('dpd','dpd x GEC_exposure') THEN 'DPD'
            WHEN INDICATOR ILIKE '%RATING%' THEN 'RATING'
            WHEN INDICATOR ILIKE '%PD%' THEN 'PD'
            WHEN INDICATOR ILIKE '%ARREARS%' THEN 'ARREARS'
            ELSE 'OTHER' END AS FAMILY,
       COUNT(DISTINCT INDICATOR) AS N_INDICATORS,
       COUNT(*) AS N_CONFIGURATIONS,
       COUNT_IF(N_EVENTS > 0) AS WITH_EVENTS,
       COUNT_IF(N_SIGNALS > 0) AS WITH_SIGNALS,
       COUNT_IF(OPTIMAL) AS N_OPTIMAL
FROM {R}
GROUP BY 1 ORDER BY 1
'''))

print('\n5 - QA checks: failed controls first')
display(executor.query_to_df(f'''
SELECT QA_TYPE,CHECK_NAME,CHECK_PASSED,CHECK_VALUE,EXPECTED_VALUE
FROM {Q}
WHERE QA_TYPE <> 'OPTIMAL_THRESHOLD'
ORDER BY CASE WHEN CHECK_PASSED = FALSE THEN 0 ELSE 1 END,
         QA_TYPE,CHECK_NAME
'''))

print('\nNotes: N_ALERTS = TP + FP (Retail method); '
      'N_ALERTS_SIGNAL_STARTS = actual historical signal starts. '
      'Maximum daily/monthly counts refer to those signal starts. '
      'Metrics require business validation. No datasets modified.')
