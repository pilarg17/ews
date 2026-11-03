# Corporate EWS B01 - QA de validacion (solo lectura, una celda Dataiku)
# No crea datasets ni modifica el training ni la recipe.
import re
import dataiku
from dataiku.core.sql import SQLExecutor2
from IPython.display import display

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
prj = dataiku.api_client().get_project(PROJECT)
ex = SQLExecutor2(connection=CONNECTION)

def table(name):
    p = prj.get_dataset(name).get_settings().get_raw().get('params', {})
    if p.get('connection') != CONNECTION:
        raise ValueError('Incorrect connection: ' + name)
    def quote(value):
        value = str(value).replace('${projectKey}', PROJECT)
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',value):
            raise ValueError('Invalid table identifier')
        return '"' + value + '"'
    return quote(p['schema'])+'.'+quote(p['table'])

TRAIN=table('EWS_training_B01_DPD_RATING')
RESULT=table('results_B01_DPD_RATING')
TEST=table('test_results_B01_DPD_RATING')

def view(title,sql):
    print('\n'+'='*90+'\n'+title+'\n'+'='*90)
    display(ex.query_to_df(sql))

# 1. No extra columns: the original 31 / 27 positions, four monthly fields
# replaced by the daily equivalents in both outputs.
cols_result=[
 'INDICATOR','PREDICTOR_A','PREDICTOR_B','N_ENTITIES','THRESHOLD_A',
 'THRESHOLD_B','N_TP','N_FP','N_FN','SUM_GCA_TP','SUM_GCA_TP_WTD',
 'SUM_GCA_FN','SUM_COST2_TP','SUM_COST2_FN','N_ALERTS','MAX_DAILY_ALERTS',
 'N_EVENTS','N_EVENTS_DETECTABLE','ACCURACY','COVERAGE','COVERAGE_2',
 'COVERAGE_3','TOTAL_COST','OPTIMAL','TOTAL_COST_2','OPTIMAL_COST_2',
 'THRESHOLD_A_RATING_LABEL','THRESHOLD_B_RATING_LABEL',
 'AV_NR_DAY_ALERTS','AV_EXP_NR_DAY_ALERTS','MAX_EXP_NR_DAY_ALERTS'
]
cols_test=[
 'QA_TYPE','CHECK_NAME','INDICATOR','PREDICTOR_A','PREDICTOR_B',
 'UNAVAILABLE_COLUMNS','ENTITY_ID','MAP_DTE','SIGNAL_LOGIC','DIRECTION',
 'THRESHOLD_A','THRESHOLD_B','PREDICTOR_VALUE','PREVIOUS_VALUE',
 'ALERT_GENERATED','CHECK_PASSED','CHECK_VALUE','EXPECTED_VALUE',
 'N_ALERTS','AV_NR_DAY_ALERTS','AV_EXP_NR_DAY_ALERTS',
 'MAX_EXP_NR_DAY_ALERTS','ACCURACY','COVERAGE','COVERAGE_2',
 'COVERAGE_3','TOTAL_COST'
]
print('\n1 / SCHEMA DE SALIDA - 31 y 27 columnas, ninguna auxiliar')
for ds,expect in [('results_B01_DPD_RATING',cols_result),('test_results_B01_DPD_RATING',cols_test)]:
    actual=[c['name'].upper() for c in dataiku.Dataset(ds,project_key=PROJECT).read_schema()]
    print(ds, 'OK' if actual==expect else 'REVISAR',
          'columnas:',len(actual),'esperadas:',len(expect))
    if actual!=expect:
        print('Faltan:',[v for v in expect if v not in actual]);print('Sobran:',[v for v in actual if v not in expect])

# 2. Full grid / constraints / count definitions.
view('2 / CONTROL DE REJILLA, CALCULOS Y OPTIMOS', f'''
SELECT COUNT(*) AS CONFIGURACIONES,
       COUNT(DISTINCT INDICATOR) AS INDICADORES,
       COUNT_IF(INDICATOR='dpd x GEC_exposure') AS DPD_X_GEC,
       COUNT(DISTINCT IFF(INDICATOR='dpd x GEC_exposure',THRESHOLD_A,NULL)) AS UMBRALES_DPD,
       COUNT(DISTINCT IFF(INDICATOR='dpd x GEC_exposure',THRESHOLD_B,NULL)) AS UMBRALES_GEC,
       COUNT_IF(INDICATOR='dpd x GEC_exposure' AND OPTIMAL=TRUE) AS OPTIMOS_GEC,
       COUNT_IF(N_TP+N_FP<>N_ALERTS) AS ERRORES_ALERTAS,
       COUNT_IF(N_TP+N_FN<>N_EVENTS) AS ERRORES_EVENTOS,
       COUNT_IF(ACCURACY NOT BETWEEN 0 AND 1 OR COVERAGE NOT BETWEEN 0 AND 1) AS ERRORES_METRICAS,
       COUNT_IF(MAX_DAILY_ALERTS < 0 OR AV_NR_DAY_ALERTS < 0) AS ERRORES_VOL_DIARIO
FROM {RESULT}
''')
print('Esperado: 50554 configuraciones, 240 DPD x GEC, 12 DPD, 20 GEC, 20 optimos; errores=0.')

view('3 / DPD: UMBRAL SELECCIONADO PARA CADA MATERIALIDAD NOMINAL GEC',f'''
SELECT THRESHOLD_B AS GEC_NOMINAL_MIN_EUR,THRESHOLD_A AS DPD,
       N_TP,N_FP,N_FN,N_ALERTS,ACCURACY,COVERAGE,
       AV_NR_DAY_ALERTS,MAX_DAILY_ALERTS,
       AV_EXP_NR_DAY_ALERTS,MAX_EXP_NR_DAY_ALERTS
FROM {RESULT}
WHERE INDICATOR='dpd x GEC_exposure' AND OPTIMAL=TRUE
ORDER BY THRESHOLD_B
''')

view('4 / RATING + PD + GCC: OPTIMOS CON VOLUMEN DIARIO',f'''
SELECT INDICATOR,THRESHOLD_A,THRESHOLD_A_RATING_LABEL,
       THRESHOLD_B,THRESHOLD_B_RATING_LABEL,
       N_EVENTS,N_TP,N_FP,N_FN,ACCURACY,COVERAGE,
       AV_NR_DAY_ALERTS,MAX_DAILY_ALERTS
FROM {RESULT}
WHERE OPTIMAL=TRUE AND INDICATOR<>'dpd x GEC_exposure'
ORDER BY MAX_DAILY_ALERTS DESC,INDICATOR
LIMIT 35
''')

view('5 / CONTROLES QA (FALSE = fallo; NULL = pendiente/informativo)',f'''
SELECT QA_TYPE,CHECK_NAME,CHECK_PASSED,CHECK_VALUE,EXPECTED_VALUE
FROM {TEST}
WHERE QA_TYPE IN ('MODEL_CHECK','DATA_QUALITY','METHODOLOGY','BUSINESS_REVIEW')
ORDER BY CASE WHEN CHECK_PASSED=FALSE THEN 0 ELSE 1 END,QA_TYPE,CHECK_NAME
''')

# 6. Coverage is evaluated only on the Corporate-mapped DPD/loan scope;
# NULL in target is tracked separately even though Retail boolean .any()
# converts unknown components with no positive component to FALSE.
view('6 / FUENTES: 10 ULTIMOS SNAPSHOTS DPD, NOMINAL GEC Y TARGET',f'''
SELECT TO_DATE(MAP_DTE) AS DIA,COUNT(*) AS FILAS,
       COUNT(DISTINCT ENTITY_ID) AS ENTIDADES,
       COUNT_IF(GEC_NOMINAL_CANDIDATE IS NOT NULL) AS GEC_NOMINAL_DISPONIBLE,
       ROUND(100.0*COUNT_IF(GEC_NOMINAL_CANDIDATE IS NOT NULL)/NULLIF(COUNT(*),0),1) AS GEC_COBERTURA_PCT,
       COUNT_IF(CLIENT_NOTCH IS NOT NULL) AS RATING_CLIENTE_DISPONIBLE,
       COUNT_IF(GCC_NOTCH IS NOT NULL) AS RATING_GCC_DISPONIBLE,
       COUNT_IF(TGT_ANY_SAMEDAY=1) AS TARGET_POSITIVO,
       COUNT_IF(TGT_ANY_SAMEDAY IS NULL) AS TARGET_DESCONOCIDO_EN_FUENTE,
       COUNT_IF(LOAN_SOURCE_DUPLICATES>0) AS FILAS_LOAN_DUPLICADOS
FROM {TRAIN}
WHERE LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1 AND DPD_MAX IS NOT NULL
GROUP BY 1 ORDER BY DIA DESC LIMIT 10
''')

# 7. Show a dated sample of raw daily DPD threshold crossings.
# This query deliberately counts CANDIDATE daily crossings, before 2-month
# signal merging. It must not be compared directly with historical TP+FP.
view('7 / CRUCES CANDIDATOS DPD DIARIOS (NO SON TP+FP NI ALERTAS OPERATIVAS CERTIFICADAS)',f'''
WITH P AS (
 SELECT TO_DATE(MAP_DTE) AS DIA,ENTITY_ID,
        TRY_TO_DOUBLE(DPD_MAX::VARCHAR) AS DPD,
        TRY_TO_DOUBLE(GEC_NOMINAL_CANDIDATE::VARCHAR) AS GEC_NOMINAL,
        CASE WHEN PD_CRR=1 OR TGT_DEFAULT_SAMEDAY=1
             OR TGT_FORBEARANCE_SAMEDAY=1 OR TGT_DIM1_SAMEDAY=1
             THEN 1 ELSE 0 END AS TARGET_EXCL_DPD,
        LAG(TO_DATE(MAP_DTE)) OVER(PARTITION BY ENTITY_ID ORDER BY TO_DATE(MAP_DTE)) AS PREV_DIA,
        LAG(TRY_TO_DOUBLE(DPD_MAX::VARCHAR)) OVER(PARTITION BY ENTITY_ID ORDER BY TO_DATE(MAP_DTE)) AS PREV_DPD
 FROM {TRAIN}
 WHERE LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
       AND LOAN_SOURCE_DUPLICATES=0
       AND NOMINAL_KNOWN_LOANS=LOAN_COUNT AND NOMINAL_KNOWN_SUM>0
       AND TO_DATE(MAP_DTE)>=DATEADD(day,-45,
          (SELECT MAX(TO_DATE(MAP_DTE)) FROM {TRAIN}
           WHERE DPD_MAX IS NOT NULL AND LOAN_COUNT>0))
),
OPTS AS (
 SELECT THRESHOLD_A AS CUT_DPD,THRESHOLD_B AS CUT_GEC
 FROM {RESULT} WHERE INDICATOR='dpd x GEC_exposure' AND OPTIMAL=TRUE
),
X AS (
 SELECT p.DIA,o.CUT_GEC,o.CUT_DPD,COUNT(*) AS CRUCES_CANDIDATOS
 FROM P p JOIN OPTS o
  ON p.DPD>o.CUT_DPD AND p.PREV_DPD<=o.CUT_DPD
  AND p.GEC_NOMINAL>=o.CUT_GEC
 WHERE DATEDIFF(day,p.PREV_DIA,p.DIA)=1 AND p.TARGET_EXCL_DPD=0
   AND p.DIA>DATEADD(day,-14,(SELECT MAX(TO_DATE(MAP_DTE)) FROM {TRAIN}
                      WHERE DPD_MAX IS NOT NULL AND LOAN_COUNT>0))
 GROUP BY 1,2,3
)
SELECT DIA,CUT_GEC,CUT_DPD,CRUCES_CANDIDATOS
FROM X ORDER BY DIA DESC,CUT_GEC LIMIT 180
''')

# 8. Explicitly quantify the calendar gaps that the Retail crossing rule
# excludes, particularly weekend / holiday returns.
view('8 / DIAS SIN SNAPSHOT CONSECUTIVO - CUANTOS CRUCES PODRIAMOS OMITIR',f'''
WITH H AS (
 SELECT ENTITY_ID,TO_DATE(MAP_DTE) AS DIA,TRY_TO_DOUBLE(DPD_MAX::VARCHAR) AS DPD,
        LAG(TO_DATE(MAP_DTE)) OVER(PARTITION BY ENTITY_ID ORDER BY TO_DATE(MAP_DTE)) AS PREV_DIA,
        LAG(TRY_TO_DOUBLE(DPD_MAX::VARCHAR)) OVER(PARTITION BY ENTITY_ID ORDER BY TO_DATE(MAP_DTE)) AS PREV_DPD
 FROM {TRAIN}
 WHERE LOAN_COUNT>0 AND HAS_VALID_ENTITY_HISTORY=1
   AND TO_DATE(MAP_DTE)>=DATEADD(day,-45,
       (SELECT MAX(TO_DATE(MAP_DTE)) FROM {TRAIN}
        WHERE DPD_MAX IS NOT NULL AND LOAN_COUNT>0))
),B AS (
 SELECT *,DATEDIFF(day,PREV_DIA,DIA) AS GAP_DIAS
 FROM H WHERE DIA>(SELECT DATEADD(day,-30,MAX(TO_DATE(MAP_DTE))) FROM {TRAIN}
                    WHERE DPD_MAX IS NOT NULL AND LOAN_COUNT>0)
)
SELECT GAP_DIAS,COUNT(*) AS FILAS,
       COUNT_IF(DPD>5 AND PREV_DPD<=5) AS POTENCIALES_CRUCES_DPD5,
       COUNT_IF(DPD>15 AND PREV_DPD<=15) AS POTENCIALES_CRUCES_DPD15
FROM B WHERE GAP_DIAS IS NOT NULL
GROUP BY GAP_DIAS ORDER BY GAP_DIAS
''')

print('\nFINAL: QA ejecutada sin escribir ni crear datasets.')
print('IMPORTANTE: los 31 resultados son historicos (1-11 meses de anticipacion).')
print('Las senales diarias recientes de la tabla 7 son candidatas sin fusion de 2 meses.')
print('Verificar poblacion LEGAL_PERSON, codigos de eventos y cobertura GEC antes de certificacion de negocio.')
