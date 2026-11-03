# Corporate EWS B01 | Notebook de validacion (solo lectura)
# Ejecutar tras el Build de compute_results_B01_DPD_RATING.
# No crea tablas, no cambia los datasets ni reejecuta el training.

import re
import dataiku
import pandas as pd
from IPython.display import display
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
R_NAME = 'results_B01_DPD_RATING'
Q_NAME = 'test_results_B01_DPD_RATING'
ex = SQLExecutor2(connection=CONNECTION)
prj = dataiku.api_client().get_project(PROJECT)
pd.set_option('display.max_rows', 125)
pd.set_option('display.max_columns', 32)
pd.set_option('display.max_colwidth', 110)

RETAIL_RESULTS = [
    'INDICATOR', 'PREDICTOR_A', 'PREDICTOR_B', 'N_ENTITIES',
    'THRESHOLD_A', 'THRESHOLD_B', 'N_TP', 'N_FP', 'N_FN',
    'SUM_GCA_TP', 'SUM_GCA_TP_WTD', 'SUM_GCA_FN',
    'SUM_COST2_TP', 'SUM_COST2_FN', 'N_ALERTS', 'MAX_MONTHLY_ALERTS',
    'N_EVENTS', 'N_EVENTS_DETECTABLE', 'ACCURACY', 'COVERAGE',
    'COVERAGE_2', 'COVERAGE_3', 'TOTAL_COST', 'OPTIMAL',
    'TOTAL_COST_2', 'OPTIMAL_COST_2',
    'THRESHOLD_A_RATING_LABEL', 'THRESHOLD_B_RATING_LABEL',
    'AV_NR_MON_ALERTS', 'AV_EXP_NR_MON_ALERTS', 'MAX_EXP_NR_MON_ALERTS'
]
RETAIL_QA = [
    'QA_TYPE', 'CHECK_NAME', 'INDICATOR', 'PREDICTOR_A',
    'PREDICTOR_B', 'UNAVAILABLE_COLUMNS', 'ENTITY_ID', 'MAP_DTE',
    'SIGNAL_LOGIC', 'DIRECTION', 'THRESHOLD_A', 'THRESHOLD_B',
    'PREDICTOR_VALUE', 'PREVIOUS_VALUE', 'ALERT_GENERATED',
    'CHECK_PASSED', 'CHECK_VALUE', 'EXPECTED_VALUE', 'N_ALERTS',
    'AV_NR_MON_ALERTS', 'AV_EXP_NR_MON_ALERTS',
    'MAX_EXP_NR_MON_ALERTS', 'ACCURACY', 'COVERAGE',
    'COVERAGE_2', 'COVERAGE_3', 'TOTAL_COST'
]

def table(name):
    settings = prj.get_dataset(name).get_settings().get_raw()
    p = settings.get('params') or {}
    if p.get('connection') != CONNECTION:
        raise ValueError('Incorrect SQL connection for ' + name)
    def ident(val):
        val = str(val).replace('${projectKey}', PROJECT)
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', val):
            raise ValueError('Unsafe identifier: ' + repr(val))
        return '"' + val + '"'
    return ident(p['schema']) + '.' + ident(p['table'])

R = table(R_NAME)
Q = table(Q_NAME)

def query(sql):
    return ex.query_to_df(sql)

print('=' * 82)
print('0 / RETAIL OUTPUT CONTRACT: EXACT COLUMNS')
print('=' * 82)
for name, expected in [(R_NAME, RETAIL_RESULTS), (Q_NAME, RETAIL_QA)]:
    actual = [c['name'].upper() for c in dataiku.Dataset(name, project_key=PROJECT).read_schema()]
    print('{}: {} columns; Retail expected {}; match={}'.format(
        name, len(actual), len(expected), actual == expected))
    if actual != expected:
        print('  Missing:', [x for x in expected if x not in actual])
        print('  Extra:', [x for x in actual if x not in expected])
        print('  If columns match but order differs, inspect the recipe.')

print('\n' + '=' * 82)
print('1 / SCOPE: WHAT WAS ACTUALLY CALIBRATED?')
print('=' * 82)
display(query(f'''
SELECT COUNT(*) AS CONFIGURATIONS,
       COUNT(DISTINCT INDICATOR) AS INDICATOR_NAMES,
       COUNT_IF(INDICATOR='dpd x GEC_exposure') AS DPD_GEC_CONFIGURATIONS,
       COUNT(DISTINCT IFF(INDICATOR='dpd x GEC_exposure',THRESHOLD_A,NULL)) AS DPD_CUTOFFS,
       COUNT(DISTINCT IFF(INDICATOR='dpd x GEC_exposure',THRESHOLD_B,NULL)) AS GEC_CUTOFFS,
       COUNT_IF(INDICATOR='dpd x GEC_exposure' AND OPTIMAL=TRUE) AS DPD_GEC_BEST_ROWS,
       COUNT_IF(INDICATOR<>'dpd x GEC_exposure' AND OPTIMAL=TRUE) AS OTHER_BEST_ROWS
FROM {R}
'''))
print('Expected: 50,554 configurations; 12 DPD cutoffs; 20 GEC cutoffs;')
print('240 DPD x GEC rows; 20 best DPD rows; 88 best Retail rows.')

print('\n' + '=' * 82)
print('2 / DPD + GEC: BEST DPD AT EACH GEC MATERIALITY')
print('=' * 82)
display(query(f'''
SELECT THRESHOLD_B AS GEC_MIN_EUR, THRESHOLD_A AS DPD_DAYS,
       N_EVENTS, N_TP, N_FP, N_FN, N_ALERTS,
       ACCURACY, COVERAGE, COVERAGE_2, COVERAGE_3,
       MAX_MONTHLY_ALERTS, AV_NR_MON_ALERTS,
       AV_EXP_NR_MON_ALERTS, MAX_EXP_NR_MON_ALERTS
FROM {R}
WHERE INDICATOR='dpd x GEC_exposure' AND OPTIMAL=TRUE
ORDER BY THRESHOLD_B
'''))

print('\n' + '=' * 82)
print('3 / DPD + GEC: ALL 12 DPD CUTOFFS AT 50,000 EUR')
print('=' * 82)
display(query(f'''
SELECT THRESHOLD_A AS DPD_DAYS,
       N_EVENTS, N_TP, N_FP, N_FN, N_ALERTS,
       ACCURACY, COVERAGE, MAX_MONTHLY_ALERTS, TOTAL_COST, OPTIMAL
FROM {R}
WHERE INDICATOR='dpd x GEC_exposure' AND THRESHOLD_B=50000
ORDER BY THRESHOLD_A
'''))

print('\n' + '=' * 82)
print('4 / RATING CLIENT + GCC: OPTIMAL RETAIL CANDIDATES')
print('=' * 82)
display(query(f'''
SELECT INDICATOR, THRESHOLD_A, THRESHOLD_A_RATING_LABEL,
       THRESHOLD_B, THRESHOLD_B_RATING_LABEL,
       N_EVENTS, N_TP, N_FP, N_FN, ACCURACY, COVERAGE,
       N_ALERTS, MAX_MONTHLY_ALERTS, AV_EXP_NR_MON_ALERTS
FROM {R}
WHERE OPTIMAL=TRUE AND INDICATOR <> 'dpd x GEC_exposure'
  AND (INDICATOR ILIKE '%rating%' OR INDICATOR ILIKE '%gcc%')
ORDER BY MAX_MONTHLY_ALERTS ASC, INDICATOR
'''))
print('A downgrade threshold of 0 means any strictly positive deterioration.')

print('\n' + '=' * 82)
print('5 / QA: TECHNICAL CHECKS, DATA COVERAGE AND OPEN BUSINESS ITEMS')
print('=' * 82)
checks = query(f'''
SELECT QA_TYPE, CHECK_NAME, CHECK_PASSED, CHECK_VALUE, EXPECTED_VALUE
FROM {Q}
WHERE QA_TYPE IN ('MODEL_CHECK','DATA_QUALITY','BUSINESS_REVIEW')
ORDER BY CASE WHEN CHECK_PASSED=FALSE THEN 0 ELSE 1 END,
         QA_TYPE, CHECK_NAME
''')
display(checks)
model_checks = checks[checks['QA_TYPE']=='MODEL_CHECK']
failed = model_checks[model_checks['CHECK_PASSED'].astype(str).str.lower()=='false']
print('TECHNICAL MODEL CHECKS FAILED:',len(failed))
print('Business review rows with blank CHECK_PASSED are OPEN issues, not passed checks.')
print('\nRetail monthly extrapolations (/0.7) are estimates, not observed signals.')
print('TOTAL_COST_2 stays empty until reliable collateral coverage is available.')
print('Business sign-off still needs GEC exposure, target codes and Corporate exclusions.')
print('No dataset has been changed.')
