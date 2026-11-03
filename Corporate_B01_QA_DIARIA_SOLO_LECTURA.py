# Corporate B01 - QA diaria de metodologia y senales
# Ejecutar como una unica celda de notebook Dataiku. SOLO LECTURA.
# No crea datasets, no modifica schemas y no reconstruye el modelo.

import re
import dataiku
from dataiku.core.sql import SQLExecutor2
from IPython.display import display

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
prj = dataiku.api_client().get_project(PROJECT)
ex = SQLExecutor2(connection=CONNECTION)


def table(dataset):
    config = prj.get_dataset(dataset).get_settings().get_raw()
    p = config.get('params') or {}
    if p.get('connection') != CONNECTION:
        raise ValueError('Connection mismatch for ' + dataset)
    def ident(s):
        s = str(s).replace('${projectKey}', PROJECT)
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', s):
            raise ValueError('Invalid SQL identifier: ' + repr(s))
        return '"' + s + '"'
    return ident(p['schema']) + '.' + ident(p['table'])

TRAIN = table('EWS_training_B01_DPD_RATING')
RES = table('results_B01_DPD_RATING')
TEST = table('test_results_B01_DPD_RATING')


def show(title, sql):
    print('\n' + '='*80)
    print(title)
    print('='*80)
    display(ex.query_to_df(sql))


# 1. QA tecnica del modelo y reconciliacion de la funcion coste 1.
show('1 / GRID Y FUNCION DE COSTES - comprobaciones internas', f'''
SELECT COUNT(*) AS CONFIGURACIONES,
       COUNT(DISTINCT INDICATOR) AS INDICADORES,
       COUNT_IF(INDICATOR = 'dpd x GEC_exposure') AS DPD_GEC_CONFIGURACIONES,
       COUNT_IF(INDICATOR = 'dpd x GEC_exposure' AND OPTIMAL = TRUE)
           AS DPD_GEC_OPTIMOS,
       COUNT_IF(N_TP + N_FP <> N_ALERTS) AS FALLOS_TP_FP,
       COUNT_IF(N_TP + N_FN <> N_EVENTS) AS FALLOS_TP_FN,
       COUNT_IF(ABS(TOTAL_COST - ROUND(
            1000 * N_TP + 200 * N_FP
            + SUM_GCA_TP_WTD + 0.20 * SUM_GCA_FN, 2)) > 0.02)
           AS FALLOS_FUNCION_COSTE
FROM {RES}
''')

show('2 / CONTROLES DEL OUTPUT - fallos y aspectos pendientes', f'''
SELECT QA_TYPE, CHECK_NAME, CHECK_PASSED, CHECK_VALUE, EXPECTED_VALUE
FROM {TEST}
WHERE (QA_TYPE = 'MODEL_CHECK' AND CHECK_PASSED = FALSE)
   OR QA_TYPE = 'BUSINESS_REVIEW'
ORDER BY QA_TYPE, CHECK_NAME
''')

# 3. La frecuencia es diaria. Cuantos clientes tienen datos para el analisis?
show('3 / COBERTURA EN LOS ULTIMOS 10 SNAPSHOTS CON DPD', f'''
SELECT TO_DATE(MAP_DTE) AS DIA,
       COUNT(*) AS FILAS,
       COUNT(DISTINCT ENTITY_ID) AS ENTIDADES,
       COUNT_IF(TGT_ANY_SAMEDAY = 1) AS TARGET_POSITIVO,
       COUNT_IF(TGT_ANY_SAMEDAY = 0) AS TARGET_NEGATIVO,
       COUNT_IF(TGT_ANY_SAMEDAY IS NULL) AS TARGET_DESCONOCIDO,
       COUNT_IF(GEC_NOMINAL_CANDIDATE IS NOT NULL) AS GEC_NOMINAL_DISPONIBLE,
       COUNT_IF(CLIENT_NOTCH IS NOT NULL) AS RATING_CLIENTE_DISPONIBLE,
       COUNT_IF(GCC_NOTCH IS NOT NULL) AS RATING_GCC_DISPONIBLE
FROM {TRAIN}
WHERE LOAN_COUNT > 0
  AND HAS_VALID_ENTITY_HISTORY = 1
  AND DPD_MAX IS NOT NULL
GROUP BY 1
ORDER BY 1 DESC
LIMIT 10
''')

# 4. Auditar > contra >= en datos reales, con las mismas condiciones
#    de elegibilidad de las senales DPD Corporate, en los ultimos 30 dias.
#    No confundir cruces en bruto con senales fusionadas 2 meses.
BASE = f'''
WITH ASOF_DATE AS (
    SELECT MAX(TO_DATE(MAP_DTE)) AS D
    FROM {TRAIN} WHERE LOAN_COUNT > 0 AND DPD_MAX IS NOT NULL
),
RAW AS (
    SELECT t.ENTITY_ID::VARCHAR AS ENTITY_ID,
           TO_DATE(t.MAP_DTE) AS DIA,
           TRY_TO_DOUBLE(t.DPD_MAX::VARCHAR) AS DPD,
           TRY_TO_DOUBLE(t.GEC_NOMINAL_CANDIDATE::VARCHAR) AS GEC_EUR,
           CASE WHEN TRY_TO_DOUBLE(t.PD_CRR::VARCHAR) = 1
                   OR t.TGT_DEFAULT_SAMEDAY = 1
                   OR t.TGT_FORBEARANCE_SAMEDAY = 1
                   OR t.TGT_DIM1_SAMEDAY = 1 THEN 1
                WHEN t.PD_CRR IS NULL OR t.TGT_DEFAULT_SAMEDAY IS NULL
                   OR t.TGT_FORBEARANCE_SAMEDAY IS NULL
                   OR t.TGT_DIM1_SAMEDAY IS NULL THEN NULL
                ELSE 0 END AS TARGET,
           CASE WHEN t.LOAN_COUNT > 0 AND t.LOAN_SOURCE_DUPLICATES = 0
                    AND t.NOMINAL_KNOWN_LOANS = t.LOAN_COUNT
                THEN TRY_TO_DOUBLE(t.NOMINAL_KNOWN_SUM::VARCHAR)
                ELSE NULL END AS OWN_EXPOSURE
    FROM {TRAIN} t
    WHERE t.LOAN_COUNT > 0 AND t.HAS_VALID_ENTITY_HISTORY = 1
      AND TO_DATE(t.MAP_DTE) <= (SELECT D FROM ASOF_DATE)
),
PANEL AS (
    SELECT * FROM RAW WHERE OWN_EXPOSURE > 0 AND DPD IS NOT NULL
),
PR AS (
    SELECT *,
           LAG(DIA) OVER(PARTITION BY ENTITY_ID ORDER BY DIA) AS PREV_DIA,
           LAG(DPD) OVER(PARTITION BY ENTITY_ID ORDER BY DIA) AS PREV_DPD,
           LAG(TARGET) OVER(PARTITION BY ENTITY_ID ORDER BY DIA) AS PREV_TARGET
    FROM PANEL
)
'''

show('4 / DPD: diferencia entre Retail (>) y Corporate (>=), ultimos 30 dias',
     BASE + f'''
, CUTS AS (SELECT COLUMN1::DOUBLE AS DPD_UMBRAL
           FROM VALUES (5),(10),(15),(20),(25),(30),(35),(40),(45),(50),(55),(60))
SELECT c.DPD_UMBRAL,
       COUNT_IF(p.PREV_DPD <= c.DPD_UMBRAL AND p.DPD > c.DPD_UMBRAL)
           AS CRUCES_RETAIL_ESTRICTO,
       COUNT_IF(p.PREV_DPD < c.DPD_UMBRAL AND p.DPD >= c.DPD_UMBRAL)
           AS CRUCES_CORPORATE_INCLUSIVO
FROM PR p
CROSS JOIN CUTS c
WHERE p.DIA > DATEADD(day,-30,(SELECT D FROM ASOF_DATE))
  AND p.PREV_DIA = DATEADD(day,-1,p.DIA)
  AND p.TARGET = 0 AND p.PREV_TARGET = 0
  AND p.GEC_EUR >= 50000
GROUP BY 1 ORDER BY 1
''')

# 5. Alertas diarias: modelamos los mismos cruces y la fusion de 2 meses,
#    PERO sin excluir el ultimo mes, porque esto es una revision OPERATIVA.
#    La calibracion historica si excluye el ultimo mes por la anticipacion 1-11m.
#    Son alertas candidatas DPD x GEC, no el export final de produccion.
show('5 / SENALES DIARIAS CANDIDATAS - 4 niveles GEC optimos, ultimos 30 dias',
     BASE + f'''
, OPT AS (
    SELECT THRESHOLD_A AS DPD_UMBRAL, THRESHOLD_B AS GEC_MIN_EUR
    FROM {RES}
    WHERE INDICATOR = 'dpd x GEC_exposure'
      AND OPTIMAL = TRUE
      AND THRESHOLD_B IN (50000,250000,500000,1000000)
),
STARTS AS (
    SELECT p.ENTITY_ID,p.DIA,o.DPD_UMBRAL,o.GEC_MIN_EUR
    FROM PR p JOIN OPT o
      ON p.DPD >= o.DPD_UMBRAL
     AND p.PREV_DPD < o.DPD_UMBRAL
     AND p.GEC_EUR >= o.GEC_MIN_EUR
    WHERE p.TARGET = 0 AND p.PREV_TARGET = 0
      AND p.PREV_DIA = DATEADD(day,-1,p.DIA)
),
LAGGED AS (
    SELECT *, LAG(DIA) OVER (
      PARTITION BY ENTITY_ID,DPD_UMBRAL,GEC_MIN_EUR ORDER BY DIA
    ) AS PREV_START FROM STARTS
),
GROUPED AS (
    SELECT *, SUM(IFF(PREV_START IS NULL
        OR DIA > DATEADD(month,2,PREV_START),1,0)) OVER (
        PARTITION BY ENTITY_ID,DPD_UMBRAL,GEC_MIN_EUR
        ORDER BY DIA ROWS UNBOUNDED PRECEDING
    ) AS SIGNAL_GROUP
    FROM LAGGED
),
MERGED AS (
    SELECT ENTITY_ID,DPD_UMBRAL,GEC_MIN_EUR,SIGNAL_GROUP,
           MIN(DIA) AS SIGNAL_START
    FROM GROUPED
    GROUP BY 1,2,3,4
),
DAILY AS (
    SELECT SIGNAL_START AS DIA,GEC_MIN_EUR,DPD_UMBRAL,
           COUNT(*) AS N_SENALES_NUEVAS
    FROM MERGED GROUP BY 1,2,3
),
CALENDAR AS (
    SELECT DATEADD(day,-SEQ4(),(SELECT D FROM ASOF_DATE)) AS DIA
    FROM TABLE(GENERATOR(ROWCOUNT=>31))
)
SELECT cal.DIA,o.GEC_MIN_EUR,o.DPD_UMBRAL,
       COALESCE(d.N_SENALES_NUEVAS,0) AS N_SENALES_NUEVAS
FROM CALENDAR cal CROSS JOIN OPT o
LEFT JOIN DAILY d ON d.DIA=cal.DIA AND d.GEC_MIN_EUR=o.GEC_MIN_EUR
       AND d.DPD_UMBRAL=o.DPD_UMBRAL
ORDER BY cal.DIA DESC,o.GEC_MIN_EUR
''')

# 6. Huecos de snapshots: la recipe actual exige diferencia exacta de 1 dia.
#    Esto puede omitir cruces despues de fines de semana o festivos.
show('6 / CRUCES DPD POTENCIALES OMITIDOS POR HUECOS DE CALENDARIO',
     BASE + f'''
, CUTS AS (SELECT COLUMN1::DOUBLE AS DPD_UMBRAL
           FROM VALUES (5),(10),(15),(20),(25),(30),(35),(40),(45),(50),(55),(60))
SELECT c.DPD_UMBRAL,
       COUNT_IF(p.PREV_DIA < DATEADD(day,-1,p.DIA)
          AND p.PREV_DPD < c.DPD_UMBRAL
          AND p.DPD >= c.DPD_UMBRAL)
          AS CRUCES_POTENCIALES_TRAS_HUECO,
       COUNT(DISTINCT IFF(p.PREV_DIA < DATEADD(day,-1,p.DIA)
          AND p.PREV_DPD < c.DPD_UMBRAL
          AND p.DPD >= c.DPD_UMBRAL,p.ENTITY_ID,NULL))
          AS ENTIDADES_AFECTADAS
FROM PR p CROSS JOIN CUTS c
WHERE p.DIA > DATEADD(day,-30,(SELECT D FROM ASOF_DATE))
  AND p.TARGET = 0 AND p.PREV_TARGET = 0
  AND p.GEC_EUR >= 50000
GROUP BY 1 ORDER BY 1
''')

print('\nFINAL: QA solo lectura completada.')
print('IMPORTANTE: la tabla 5 muestra senales operativas candidatas para los ultimos')
print('snapshots DPD disponibles, sin imponer 1 mes de anticipacion futura.')
print('NO equivale al export diario de alertas de produccion y NO modifica el modelo.')
