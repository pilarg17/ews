import dataiku
import pandas as pd
import time
import re
from dataiku.core.sql import SQLExecutor2

print('=' * 88)
print('CREDIT CARDS - TARGETED ROOT CAUSE / RUNTIME CHECK (READ ONLY)')
print('=' * 88, flush=True)

PROJECT = dataiku.api_client().get_default_project()
RECIPES = [
    'compute_Credit_Cards', 'compute_Credit_Cards_Previous',
    'compute_CC_Utilisation_History', 'compute_CC_Utilisation_History_Previous',
    'compute_Daily_Behavioural_Features', 'compute_Daily_Behavioural_Features_1',
    'compute_Cash_Advance_CC_Limit', 'compute_Cash_Advance_CC_Limit_Previous',
]

for name in ('Implementation_Reference_Date', 'Implementation_Reference_Date_previous'):
    try:
        df = dataiku.Dataset(name).get_dataframe()
        print('DATE INPUT:', name, df[['REFERENCE_DATE', 'PREVIOUS_DATE']].head(1).to_dict('records'), flush=True)
    except Exception as exc:
        print('DATE INPUT ERROR:', name, type(exc).__name__, str(exc)[:200], flush=True)

print('\nRECIPE SQL / CLASSIFICATION CHECK', flush=True)
for name in RECIPES:
    try:
        code = PROJECT.get_recipe(name).get_settings().get_code()
        patterns = {
            'G00': len(re.findall(r'\bG00\b', code, re.I)),
            'GRADE_OR_CLASS': len(re.findall(r'GRADE|CLASSIF|CATEGORY', code, re.I)),
            'TRUNC_DATE': len(re.findall(r'TRUNC\s*\(\s*\w+\.TEK_', code, re.I)),
            'ROW_NUMBER': len(re.findall(r'ROW_NUMBER\s*\(', code, re.I)),
            'CARD_ID': len(re.findall(r'CARD_ID', code, re.I)),
            'SQL_EXECUTIONS': len(re.findall(r'\.query_to_df\s*\(', code, re.I)),
            'FULL_DATAFRAME_READS': len(re.findall(r'\.get_dataframe\s*\(', code, re.I)),
        }
        print(name, patterns, flush=True)
        for line in code.splitlines():
            if re.search(r'G00|\bGRADE\b|CLASSIF', line, re.I):
                print('  MATCH:', line.strip()[:180], flush=True)
    except Exception as exc:
        print(name, 'ERROR:', type(exc).__name__, str(exc)[:200], flush=True)

print('\nORACLE SOURCE METADATA', flush=True)
for name in ['CARD_USAGE_VW', 'CARD_REFERENTIAL_VW', 'CUSTOMER_VW']:
    ds = dataiku.Dataset(name)
    fields = [x['name'] for x in ds.read_schema()]
    print(name, 'classification-like columns:', [x for x in fields if re.search('GRADE|G00|CLASS|CATEGORY', x, re.I)], flush=True)

ref = dataiku.Dataset('Implementation_Reference_Date').get_dataframe()
if len(ref) != 1:
    raise ValueError('Implementation_Reference_Date must have exactly one row')
date = pd.to_datetime(ref['REFERENCE_DATE'].iloc[0]).strftime('%Y-%m-%d')

def table_of(name):
    ds = dataiku.Dataset(name)
    info = ds.get_location_info().get('info', {})
    schema = info.get('schema') or info.get('schemaName')
    table = info.get('table') or info.get('tableName')
    if not table:
        raise ValueError(name + ' is not SQL-backed')
    return ds, (schema + '.' + table if schema and '.' not in table else table)

usage_ds, usage = table_of('CARD_USAGE_VW')
_, card = table_of('CARD_REFERENTIAL_VW')
_, cust = table_of('CUSTOMER_VW')

# 1% deterministic subset of cards; the DB returns only aggregate statistics.
# Change 99 to 19 if the sample is too small (5%), but expect more work.
SQL = '''
WITH
u0 AS (
 SELECT TRIM(CAST(u.CARD_ID AS VARCHAR2(100))) AS CARD_ID,
        u.CURRENT_PERIOD_TOTAL_OUTSTANDING_AMOUNT AS OUTSTANDING,
        u.TEK_STRT_DTE AS START_DTE,
        u.TEK_END_DTE AS END_DTE
 FROM {usage} u
 WHERE u.TEK_STRT_DTE < DATE '{date}' + 1
   AND NVL(u.TEK_END_DTE, DATE '9999-12-31') >= DATE '{date}' + 1
   AND ORA_HASH(u.CARD_ID, 99) = 0
),
c0 AS (
 SELECT TRIM(CAST(c.CARD_ID AS VARCHAR2(100))) AS CARD_ID,
        TRIM(CAST(c.CARD_OWNER_CUSTOMER_ID AS VARCHAR2(100))) AS OWNER_ID,
        c.CREDIT_LIMIT AS CREDIT_LIMIT,
        c.TEMPORARY_CREDIT_LIMIT AS TEMPORARY_LIMIT,
        c.TEK_STRT_DTE AS START_DTE,
        c.TEK_END_DTE AS END_DTE
 FROM {card} c
 WHERE c.TEK_STRT_DTE < DATE '{date}' + 1
   AND NVL(c.TEK_END_DTE, DATE '9999-12-31') >= DATE '{date}' + 1
   AND UPPER(TRIM(c.CARD_TYPE)) = 'CREDIT CARD'
   AND ORA_HASH(c.CARD_ID, 99) = 0
),
u_counts AS (SELECT CARD_ID, COUNT(*) AS N,
             COUNT(DISTINCT OUTSTANDING) AS N_VALUES
             FROM u0 GROUP BY CARD_ID),
c_counts AS (SELECT CARD_ID, COUNT(*) AS N,
             COUNT(DISTINCT CREDIT_LIMIT) AS N_LIMITS,
             COUNT(DISTINCT OWNER_ID) AS N_OWNERS
             FROM c0 GROUP BY CARD_ID)
SELECT 'USAGE' AS SOURCE,
       COUNT(*) AS SAMPLE_CARD_IDS,
       NVL(SUM(N), 0) AS SAMPLE_ROWS,
       NVL(SUM(CASE WHEN N > 1 THEN 1 ELSE 0 END), 0) AS MULTIVERSION_CARDS,
       NVL(SUM(CASE WHEN N_VALUES > 1 THEN 1 ELSE 0 END), 0) AS CONFLICTING_CARDS,
       CAST(NULL AS NUMBER) AS MULTI_OWNER_CARDS
FROM u_counts
UNION ALL
SELECT 'CARD_REFERENTIAL' AS SOURCE,
       COUNT(*) AS SAMPLE_CARD_IDS,
       NVL(SUM(N), 0) AS SAMPLE_ROWS,
       NVL(SUM(CASE WHEN N > 1 THEN 1 ELSE 0 END), 0) AS MULTIVERSION_CARDS,
       NVL(SUM(CASE WHEN N_LIMITS > 1 THEN 1 ELSE 0 END), 0) AS CONFLICTING_CARDS,
       NVL(SUM(CASE WHEN N_OWNERS > 1 THEN 1 ELSE 0 END), 0) AS MULTI_OWNER_CARDS
FROM c_counts
'''.format(usage=usage, card=card, date=date)

print('\nORACLE 1% CARD-ID SAMPLE FOR', date, flush=True)
print('Only aggregate rows returned. This query may still scan Oracle history.', flush=True)
started = time.perf_counter()
try:
    stats = SQLExecutor2(dataset=usage_ds).query_to_df(SQL)
    print(stats.to_string(index=False), flush=True)
    print('ORACLE QUERY SECONDS:', round(time.perf_counter() - started, 1), flush=True)
    print('NOTE: Sample estimates duplication; it does not prove whole-population equivalence.', flush=True)
except Exception as exc:
    print('ORACLE ERROR:', type(exc).__name__, str(exc)[:800], flush=True)

print('\nINTERPRETATION', flush=True)
print('USAGE MULTIVERSION > 0: deduplicate usage rows by CARD_ID and observation date before joining.', flush=True)
print('CONFLICTING_CARDS > 0: never use MAX(outstanding) to choose a temporal row without rule validation.', flush=True)
print('REFERENTIAL MULTI_OWNER > 0: selection must respect effective date and owner mapping.', flush=True)
print('G00 absent: do not filter it without identifying the actual source / column.', flush=True)
print('READ ONLY. NO WRITES, RECIPE EXECUTION OR BUNDLE CHANGES.', flush=True)
