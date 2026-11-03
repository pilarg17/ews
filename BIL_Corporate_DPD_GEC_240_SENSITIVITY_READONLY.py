# BIL Corporate EWS | DPD x GEC Exposure: sensitivity (READ-ONLY)
# Solo tablas ORIGINALES, sin _QA, sin training, sin escrituras.
# DPD thresholds: 5,10,...,60 (strictly >). GEC exposures: 50k,...,1M EUR (strictly >).
# EXPOSURE methodology is PROVISIONAL: group by GCC_PARENT_ID as-of date;
# compute NOMINAL_AMOUNT and GROSS_CARRYING_AMOUNT as distinct candidates.
# This is exploratory sensitivity, NOT validated official GEC methodology.

import re
import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
START_DATE = '2026-09-15'
END_DATE = '2026-10-26'
BASELINE_DATE = '2026-09-14'   # previous observation for first analysis date

assert BASELINE_DATE < START_DATE <= END_DATE
proj = dataiku.api_client().get_project(PROJECT)

def sf_dataset(name):
    raw = proj.get_dataset(name).get_settings().get_raw()
    p = raw.get('params', {})
    if p.get('connection') != 'DEV_AA_TEAMS':
        raise ValueError('Unexpected SQL connection for {}: {}'.format(name,p.get('connection')))
    def ident(v):
        v = str(v).replace('${projectKey}', PROJECT)
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',v):
            raise ValueError('Unexpected identifier: '+repr(v))
        return '"{}"'.format(v)
    cols = {c['name'].upper() for c in dataiku.Dataset(name, project_key=PROJECT).read_schema()}
    return '{}.{}'.format(ident(p['schema']),ident(p['table'])),cols

ent, ent_cols = sf_dataset('vars_entity_level')
loan, loan_cols = sf_dataset('vars_loan_level')
dpd, dpd_cols = sf_dataset('DPD_Arrears_Daily_Loan')
for name,cols,needed in [
    ('vars_entity_level',ent_cols, ['ENTITY_ID','TEK_STRT_DTE','TEK_END_DTE','GCC_PARENT_ID']),
    ('vars_loan_level',loan_cols,['ENTITY_ID','INSTRUMENT_ID','MAP_DTE','NOMINAL_AMOUNT','GROSS_CARRYING_AMOUNT']),
    ('DPD_Arrears_Daily_Loan',dpd_cols,['ENTITY_ID','MAP_DTE','DPD'])
]:
    missing = set(needed)-cols
    if missing:
        raise ValueError(name+' missing '+repr(sorted(missing)))

# Deduplication is NOT assumed for loan-level amounts; verify before running grid.
connection = 'DEV_AA_TEAMS'
executor = SQLExecutor2(connection=connection)
check = executor.query_to_df('''
 SELECT COUNT(*) AS N_LOAN_ROWS,
        COUNT(DISTINCT ENTITY_ID::VARCHAR || '|' || INSTRUMENT_ID::VARCHAR) AS N_UNIQUE_ENTITY_INSTRUMENT,
        COUNT_IF(TRY_TO_DOUBLE(NOMINAL_AMOUNT) IS NOT NULL) AS N_NOMINAL_KNOWN,
        COUNT_IF(TRY_TO_DOUBLE(GROSS_CARRYING_AMOUNT) IS NOT NULL) AS N_GCA_KNOWN
 FROM {loan}
 WHERE MAP_DTE >= DATE '{end}' AND MAP_DTE < DATEADD(day,1,DATE '{end}')
 '''.format(loan=loan,end=END_DATE))
check.columns = [str(c).upper() for c in check.columns]
print('=== PRECHECK: LOAN SOURCE ON END DATE ===')
print(check.to_string(index=False))
if not check.empty and int(check.iloc[0]['N_LOAN_ROWS']) != int(check.iloc[0]['N_UNIQUE_ENTITY_INSTRUMENT']):
    raise ValueError('Repeated entity/instrument in loan source on check date. Need a deterministic de-duplication rule before adding exposure. No grid run.')

# Scans only the requested 30 snapshots (+ baseline) and the entity validity history.
# Aggregate loan amounts once per entity/day, then once per GEC/day.
# Expose an amount only when every source loan contributing to that group has a
# non-null metric AND the historical entity->GEC mapping exists.
# Still cannot prove that ALL bank-wide GEC members have exposures in this dataset.
DPD_CUTS = list(range(5,61,5))
GEC_CUTS = list(range(50000,1000001,50000))
assert len(DPD_CUTS)*len(GEC_CUTS)==240

dpd_values = ', '.join('({})'.format(x) for x in DPD_CUTS)
gec_values = ', '.join('({})'.format(x) for x in GEC_CUTS)

query = '''
WITH
 D AS (
  SELECT TO_DATE(MAP_DTE) AS DAY, ENTITY_ID::VARCHAR AS ENTITY_ID,
         MAX(TRY_TO_DOUBLE(DPD)) AS MAX_DPD
  FROM {dpd}
  WHERE MAP_DTE >= DATE '{baseline}' AND MAP_DTE < DATEADD(day,1,DATE '{end}')
  GROUP BY 1,2
 ),
 L AS (
  SELECT TO_DATE(MAP_DTE) AS DAY, ENTITY_ID::VARCHAR AS ENTITY_ID,
         COUNT(*) AS N_LOANS,
         COUNT_IF(TRY_TO_DOUBLE(NOMINAL_AMOUNT) IS NOT NULL) AS N_NOM,
         COUNT_IF(TRY_TO_DOUBLE(GROSS_CARRYING_AMOUNT) IS NOT NULL) AS N_GCA,
         SUM(TRY_TO_DOUBLE(NOMINAL_AMOUNT)) AS NOM,
         SUM(TRY_TO_DOUBLE(GROSS_CARRYING_AMOUNT)) AS GCA
  FROM {loan}
  WHERE MAP_DTE >= DATE '{baseline}' AND MAP_DTE < DATEADD(day,1,DATE '{end}')
  GROUP BY 1,2
 ),
 ALL_KEYS AS (
   SELECT DAY, ENTITY_ID FROM D
   UNION
   SELECT DAY, ENTITY_ID FROM L
 ),
 HIST AS (
   SELECT k.DAY, k.ENTITY_ID,
      IFF(e.ENTITY_ID IS NULL, 0, 1) AS HAS_ENTITY_MAPPING,
      COALESCE(NULLIF(TRIM(e.GCC_PARENT_ID::VARCHAR), ''), k.ENTITY_ID) AS CANDIDATE_GEC_ID
   FROM ALL_KEYS k LEFT JOIN {ent} e
     ON k.ENTITY_ID=e.ENTITY_ID::VARCHAR
    AND TO_DATE(e.TEK_STRT_DTE)<=k.DAY
    AND (e.TEK_END_DTE IS NULL OR TO_DATE(e.TEK_END_DTE)>=k.DAY)
   QUALIFY ROW_NUMBER() OVER (
     PARTITION BY k.DAY,k.ENTITY_ID
     ORDER BY e.TEK_STRT_DTE DESC NULLS LAST,e.TEK_END_DTE DESC NULLS LAST
   )=1
 ),
 GEC_EXPOSURE AS (
   SELECT h.DAY,h.CANDIDATE_GEC_ID,
      SUM(l.N_LOANS) AS N_SOURCE_LOANS,
      SUM(l.N_NOM) AS N_NOM_KNOWN,
      SUM(l.N_GCA) AS N_GCA_KNOWN,
      MIN(h.HAS_ENTITY_MAPPING) AS ALL_MAPPED,
      SUM(l.NOM) AS NOM_SUM,
      SUM(l.GCA) AS GCA_SUM
   FROM L l JOIN HIST h ON h.DAY=l.DAY AND h.ENTITY_ID=l.ENTITY_ID
   GROUP BY 1,2
 ),
 PANEL AS (
   SELECT d.DAY,d.ENTITY_ID,d.MAX_DPD,
     IFF(g.ALL_MAPPED=1 AND g.N_SOURCE_LOANS=g.N_NOM_KNOWN,
         g.NOM_SUM, NULL) AS GEC_NOMINAL,
     IFF(g.ALL_MAPPED=1 AND g.N_SOURCE_LOANS=g.N_GCA_KNOWN,
         g.GCA_SUM, NULL) AS GEC_GCA
   FROM D d LEFT JOIN HIST h ON h.DAY=d.DAY AND h.ENTITY_ID=d.ENTITY_ID
            LEFT JOIN GEC_EXPOSURE g ON g.DAY=d.DAY
                   AND g.CANDIDATE_GEC_ID=h.CANDIDATE_GEC_ID
 ),
 PREVIOUS AS (
   SELECT p.*,
     LAG(DAY) OVER (PARTITION BY ENTITY_ID ORDER BY DAY) AS PREV_DAY,
     LAG(MAX_DPD) OVER (PARTITION BY ENTITY_ID ORDER BY DAY) AS PREV_DPD,
     LAG(GEC_NOMINAL) OVER (PARTITION BY ENTITY_ID ORDER BY DAY) AS PREV_GEC_NOMINAL,
     LAG(GEC_GCA) OVER (PARTITION BY ENTITY_ID ORDER BY DAY) AS PREV_GEC_GCA
   FROM PANEL p
 ),
 METRICS AS (
   SELECT DAY,ENTITY_ID,'NOMINAL' AS EXPOSURE_METRIC,
          MAX_DPD,PREV_DAY,PREV_DPD,
          GEC_NOMINAL AS EXPOSURE,PREV_GEC_NOMINAL AS PREV_EXPOSURE
   FROM PREVIOUS WHERE DAY >= DATE '{start}'
   UNION ALL
   SELECT DAY,ENTITY_ID,'GCA' AS EXPOSURE_METRIC,
          MAX_DPD,PREV_DAY,PREV_DPD,
          GEC_GCA AS EXPOSURE,PREV_GEC_GCA AS PREV_EXPOSURE
   FROM PREVIOUS WHERE DAY >= DATE '{start}'
 ),
 DPD_T AS (SELECT COLUMN1 AS DPD_THRESHOLD FROM VALUES {dpd_values}),
 GEC_T AS (SELECT COLUMN1 AS GEC_THRESHOLD FROM VALUES {gec_values})
SELECT m.DAY,m.EXPOSURE_METRIC,t.DPD_THRESHOLD,x.GEC_THRESHOLD,
 COUNT_IF(m.MAX_DPD > t.DPD_THRESHOLD AND m.EXPOSURE > x.GEC_THRESHOLD) AS ACTIVE,
 COUNT_IF(m.MAX_DPD > t.DPD_THRESHOLD AND m.EXPOSURE > x.GEC_THRESHOLD
    AND m.PREV_DAY IS NOT NULL AND m.PREV_DPD IS NOT NULL AND m.PREV_EXPOSURE IS NOT NULL
    AND NOT (m.PREV_DPD > t.DPD_THRESHOLD AND m.PREV_EXPOSURE > x.GEC_THRESHOLD)) AS NEW_OBSERVED,
 COUNT_IF(m.MAX_DPD > t.DPD_THRESHOLD AND m.EXPOSURE > x.GEC_THRESHOLD
    AND (m.PREV_DAY IS NULL OR m.PREV_DPD IS NULL OR m.PREV_EXPOSURE IS NULL)) AS NO_VALID_PREVIOUS,
 COUNT_IF(m.MAX_DPD > t.DPD_THRESHOLD AND m.EXPOSURE IS NULL) AS MISSING_GEC_EXPOSURE
FROM METRICS m CROSS JOIN DPD_T t CROSS JOIN GEC_T x
WHERE m.MAX_DPD > 5
GROUP BY 1,2,3,4
ORDER BY 1,2,3,4
'''.format(dpd=dpd,loan=loan,ent=ent,
           baseline=BASELINE_DATE,start=START_DATE,end=END_DATE,
           dpd_values=dpd_values,gec_values=gec_values)

print('\n=== RUNNING READ-ONLY 240 x 2 DPD/GEC SENSITIVITY ===')
print('Dates:',START_DATE,'to',END_DATE,'; baseline:',BASELINE_DATE)
print('DPD strict > threshold; group exposure strict > threshold')
print('GEC is a PROXY via GCC_PARENT_ID; NOMINAL/GCA compared separately')

grid = executor.query_to_df(query)
grid.columns=[str(c).upper() for c in grid.columns]
for c in ['ACTIVE','NEW_OBSERVED','NO_VALID_PREVIOUS','MISSING_GEC_EXPOSURE']:
    grid[c]=pd.to_numeric(grid[c],errors='coerce').fillna(0).astype('int64')
DPD_GEC_SENSITIVITY = grid
print('Aggregate result rows:',len(grid))
print('Dates:',grid['DAY'].nunique() if not grid.empty else 0)
if grid.empty:
    print('No data for this period.')
else:
    last = grid['DAY'].max()
    for m in ['NOMINAL','GCA']:
        z=grid[(grid.DAY==last)&(grid.EXPOSURE_METRIC==m)]
        if z.empty:
            continue
        print('\n=== ACTIVE SIGNALS: {} | {} | rows DPD, cols GEC kEUR ==='.format(m,last))
        p=z.pivot(index='DPD_THRESHOLD',columns='GEC_THRESHOLD',values='ACTIVE')
        p.columns=[int(c/1000) for c in p.columns]
        print(p.fillna(0).astype(int).to_string())
        print('\nSelected materialities:')
        print(z[z.GEC_THRESHOLD.isin([50000,250000,500000,750000,1000000])]
              [['DPD_THRESHOLD','GEC_THRESHOLD','ACTIVE','NEW_OBSERVED','NO_VALID_PREVIOUS','MISSING_GEC_EXPOSURE']]
              .to_string(index=False))

print('\nREAD ONLY. No existing datasets rebuilt. NO target quality/accuracy claims.')
print('CAUTION: GEC mapping and chosen exposure field require official confirmation.')
