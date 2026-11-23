# -*- coding: utf-8 -*-
# V16 | One Dataiku Python notebook cell | READ ONLY
# Purpose: 15 real missing-nominal loan IDs from the 2026-10-29 Corporate snapshot.
# No files, recipes, datasets, sources or backtests are modified.
import re
import pandas as pd
import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
SOURCE = 'vars_loan_level'
DAY = '2026-10-29'
N_SAMPLE = 15

p = dataiku.api_client().get_project(PROJECT)
conf = p.get_dataset(SOURCE).get_settings().get_raw().get('params') or {}
if conf.get('connection') != CONNECTION:
    raise RuntimeError('Unexpected connection on '+SOURCE)

def identifier(x):
    x = str(x).replace('${projectKey}', PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', x):
        raise RuntimeError('Unsafe SQL identifier: '+x)
    return '"'+x+'"'

table = identifier(conf['schema'])+'.'+identifier(conf['table'])
cols = {str(c['name']).upper() for c in dataiku.Dataset(SOURCE, project_key=PROJECT).read_schema()}
needed = {'MAP_DTE','INSTRUMENT_ID','ENTITY_ID','NOMINAL_AMOUNT'}
if needed-cols:
    raise RuntimeError('Missing source fields: '+', '.join(sorted(needed-cols)))
extra = [x for x in ('CONTRACT_REFERENCE','CONTRACT_ID_SHORT','CONTRACT_ID','CURRENCY',
                     'LOAN_CURRENCY','OFF_BALANCE_SHEET_AMOUNT','INTERNAL_SEGMENT') if x in cols]

ex = SQLExecutor2(connection=CONNECTION)
def query(label, statement, max_rows):
    if not statement.lstrip().upper().startswith(('SELECT','WITH')):
        raise RuntimeError('Read-only SELECT required')
    print('\n=== '+label+' ===', flush=True)
    df = ex.query_to_df(statement)
    df.columns = [str(c).upper() for c in df.columns]
    if len(df)>max_rows:
        raise RuntimeError('Unexpected volume from '+label)
    print(df.to_csv(index=False, sep=';', na_rep='NULL').strip() if not df.empty else 'NO_ROWS', flush=True)
    return df

base = query('CORPORATE_EXPOSURE_BASELINE_20261029', '''
SELECT COUNT(*) N_LOAN_ROWS,
       COUNT(DISTINCT TRIM(TO_VARCHAR(ENTITY_ID))) N_ENTITIES,
       COUNT_IF(TRY_TO_DOUBLE(TO_VARCHAR(NOMINAL_AMOUNT)) IS NOT NULL) N_KNOWN_NOMINAL,
       COUNT_IF(TRY_TO_DOUBLE(TO_VARCHAR(NOMINAL_AMOUNT)) IS NULL) N_MISSING_NOMINAL,
       COUNT_IF(NOMINAL_AMOUNT IS NULL) N_SQL_NULL_NOMINAL,
       COUNT_IF(NOMINAL_AMOUNT IS NOT NULL AND TRY_TO_DOUBLE(TO_VARCHAR(NOMINAL_AMOUNT)) IS NULL) N_NONNUMERIC_NOMINAL
FROM {table}
WHERE TO_DATE(MAP_DTE) = DATE '{day}'
'''.format(table=table,day=DAY),5)
if base.empty or int(base.iloc[0]['N_MISSING_NOMINAL']) == 0:
    raise RuntimeError('No missing nominal values found for '+DAY)

pattern = query('MISSING_NOMINAL_ID_PATTERNS', '''
SELECT CASE WHEN INSTRUMENT_ID IS NULL OR TRIM(TO_VARCHAR(INSTRUMENT_ID)) = '' THEN 'NO_ID'
            WHEN REGEXP_LIKE(TRIM(TO_VARCHAR(INSTRUMENT_ID)),'[0-9]+') THEN 'NUMERIC_ID'
            ELSE 'ALPHANUMERIC_OR_OTHER_ID' END ID_PATTERN,
       COUNT(*) N_ROWS,
       COUNT(DISTINCT TRIM(TO_VARCHAR(INSTRUMENT_ID))) N_DISTINCT_IDS
FROM {table}
WHERE TO_DATE(MAP_DTE) = DATE '{day}'
  AND TRY_TO_DOUBLE(TO_VARCHAR(NOMINAL_AMOUNT)) IS NULL
GROUP BY 1 ORDER BY 1
'''.format(table=table,day=DAY),5)

fields = ['TRIM(TO_VARCHAR(INSTRUMENT_ID)) AS INSTRUMENT_ID',
          'TRIM(TO_VARCHAR(ENTITY_ID)) AS ENTITY_ID',
          "IFF(NOMINAL_AMOUNT IS NULL,'SQL_NULL','NON_NUMERIC_SOURCE_VALUE') AS MISSING_REASON"]
for field in extra:
    fields.append('TRIM(TO_VARCHAR('+identifier(field)+')) AS '+identifier(field))
# Sample up to 30 IDs from each pattern, then select 15; these are actual snapshot rows.
# Deduplicate INSTRUMENT_ID: a loan ID must not be sent repeatedly.
s = '''
WITH SOURCE_ROWS AS (
 SELECT {fields}
 FROM {table}
 WHERE TO_DATE(MAP_DTE)=DATE '{day}'
   AND TRY_TO_DOUBLE(TO_VARCHAR(NOMINAL_AMOUNT)) IS NULL
   AND INSTRUMENT_ID IS NOT NULL AND TRIM(TO_VARCHAR(INSTRUMENT_ID))<>''
), DISTINCT_LOANS AS (
 SELECT * FROM SOURCE_ROWS
 QUALIFY ROW_NUMBER() OVER(PARTITION BY INSTRUMENT_ID ORDER BY HASH(ENTITY_ID,INSTRUMENT_ID))=1
), CATEGORIZED AS (
 SELECT *, CASE WHEN REGEXP_LIKE(INSTRUMENT_ID,'[0-9]+')
                   THEN 'NUMERIC_ID' ELSE 'ALPHANUMERIC_OR_OTHER_ID' END AS ID_PATTERN,
        ROW_NUMBER() OVER(PARTITION BY CASE WHEN REGEXP_LIKE(INSTRUMENT_ID,'[0-9]+')
                   THEN 'NUMERIC_ID' ELSE 'ALPHANUMERIC_OR_OTHER_ID' END
                   ORDER BY CASE WHEN MISSING_REASON='SQL_NULL' THEN 0 ELSE 1 END,
                            HASH(INSTRUMENT_ID,ENTITY_ID)) AS PATTERN_RANK
 FROM DISTINCT_LOANS
)
SELECT {chosen} FROM CATEGORIZED WHERE PATTERN_RANK<=30 ORDER BY ID_PATTERN,PATTERN_RANK
'''.format(fields=',\n        '.join(fields),table=table,day=DAY,
           chosen=', '.join(['INSTRUMENT_ID','ENTITY_ID','MISSING_REASON','ID_PATTERN','PATTERN_RANK']+extra))
# Do not print the larger candidate pool, only the final 15 ID sample.
pool = ex.query_to_df(s)
pool.columns=[str(c).upper() for c in pool.columns]
if pool.empty:
    raise RuntimeError('No valid loan IDs in the missing population')
pool = pool.drop_duplicates(subset=['INSTRUMENT_ID']).sort_values(['PATTERN_RANK','ID_PATTERN'])
# Mix numeric and alphanumeric IDs so business can assess potential off-balance patterns.
a = pool[pool.ID_PATTERN == 'ALPHANUMERIC_OR_OTHER_ID'].head(8)
b = pool[pool.ID_PATTERN == 'NUMERIC_ID'].head(7)
chosen = pd.concat([a,b],ignore_index=True).drop_duplicates(subset=['INSTRUMENT_ID'])
if len(chosen)<N_SAMPLE:
    remaining=pool[~pool.INSTRUMENT_ID.isin(chosen.INSTRUMENT_ID)]
    chosen=pd.concat([chosen,remaining.head(N_SAMPLE-len(chosen))],ignore_index=True)
chosen=chosen.head(N_SAMPLE)
chosen.insert(0,'SAMPLE_NO',range(1,len(chosen)+1))
chosen = chosen.drop(columns=['PATTERN_RANK'])
print('\n=== 15_MISSING_NOMINAL_LOANS_FOR_BUSINESS_REVIEW ===',flush=True)
print(chosen.to_csv(index=False,sep=';',na_rep='NULL').strip(),flush=True)
print('\n=== LOAN_ID_LIST_ONLY ===',flush=True)
for v in chosen.INSTRUMENT_ID:
    print(str(v),flush=True)
print('\nQA | READ_ONLY=YES | SNAPSHOT='+DAY+' | N_SAMPLED='+str(len(chosen)),flush=True)
print('QA | ALPHANUMERIC_IDS_ARE_NOT_PROOF_OF_OFF_BALANCE; BUSINESS_REVIEW_REQUIRED',flush=True)
print('QA | MISSING NOMINAL IN VARS_LOAN_LEVEL IS NOT PROOF OF ZERO LOAN EXPOSURE',flush=True)
print('QA | NO DATASETS, RECIPES OR BACKTESTS MODIFIED',flush=True)
