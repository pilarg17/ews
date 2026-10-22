# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import re
from dataiku.core.sql import SQLExecutor2

ALERT_DATE = pd.Timestamp('2026-10-16')
SOURCE_DATASET = 'X_CDS_RESULTS_VW'


def norm_id(series):
    return series.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)


def norm_date(series):
    return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()


def normalized_col(name):
    return re.sub(r'[^A-Z0-9]+', '_', str(name).upper()).strip('_')


def read_source_output(dataset_name, date_field, code_field, description_field):
    try:
        ds = dataiku.Dataset(dataset_name)
        schema_names = [x['name'] for x in ds.read_schema()]
        selected = [x for x in ['ENTITY_ID', date_field, code_field, description_field]
                    if x in schema_names]
        if 'ENTITY_ID' not in selected or date_field not in selected:
            print(dataset_name, '- cannot identify date and entity with available columns')
            return set()
        df = ds.get_dataframe(columns=selected)
        df['ENTITY_ID'] = norm_id(df['ENTITY_ID'])
        dated = df[norm_date(df[date_field]).eq(ALERT_DATE)].copy()
        if dated.empty:
            print(dataset_name, '- the available output does not contain 16 Oct data (may have been overwritten)')
            return set()
        checks = pd.Series(False, index=dated.index)
        if code_field in dated:
            checks |= dated[code_field].astype('string').str.contains('DOWNGRADE_TO_STAGE2', case=False, regex=False, na=False)
        if description_field in dated:
            checks |= dated[description_field].astype('string').str.contains('Downgrade from Stage 1 to Stage 2', case=False, regex=False, na=False)
        hits = dated.loc[checks].copy()
        print(dataset_name, '- dated rows:', len(dated), '| Stage 2 breaches:', len(hits))
        if not hits.empty:
            print(hits.to_string(index=False))
        return set(hits['ENTITY_ID'].dropna().astype(str))
    except Exception as err:
        print(dataset_name, '- unavailable:', str(err)[:220])
        return set()


print('=' * 96)
print('STAGE 2 TRACEABILITY | 16 OCT 2026 | READ ONLY')
print('=' * 96)
print('\nSTEP 1 - EXISTING OUTPUT (ONLY IF STILL STORED FOR THE RIGHT DATE)')

ids = read_source_output(
    'ews_ewi_breach_information', 'ALERT_DATE', 'INDICATOR_CODE', 'EWI_NAME'
)

print('\nSTEP 2 - HISTORICAL RFO SOURCE')
src = dataiku.Dataset(SOURCE_DATASET)
meta = {normalized_col(x['name']): x for x in src.read_schema()}


def col(*aliases, optional=False):
    for name in aliases:
        value = meta.get(normalized_col(name))
        if value is not None:
            return value
    if optional:
        return None
    raise KeyError('Required field not found: ' + ', '.join(aliases))

contract = col('CONTRACT_REFERENCE', 'INSTRUMENT_ID')['name']
entity = col('COUNTERPARTY_RISK', 'ENTITY_ID')['name']
source_date_meta = col('REPORT_DATEKEY', 'MAP_DTE')
source_date = source_date_meta['name']
stage = col('STAGE', 'IFRS9_STAGE', 'DIM1')['name']
version_meta = col('REPORT_VERSIONID', 'REPORT_VERSION_ID', optional=True)

loc = src.get_location_info().get('info', {})
table = loc.get('table') or loc.get('tableName')
schema = loc.get('schema') or loc.get('schemaName')
if not table:
    raise ValueError('The Stage source dataset is not SQL backed')
if schema and '.' not in table:
    table = schema + '.' + table

numeric_date = normalized_col(source_date) == 'REPORT_DATEKEY' or any(
    x in str(source_date_meta.get('type', '')).lower()
    for x in ['int', 'long', 'number', 'decimal', 'double', 'float']
)
if numeric_date:
    date_condition = '{field} <= {value}'.format(field=source_date, value=ALERT_DATE.strftime('%Y%m%d'))
else:
    date_condition = 'TRUNC({field}) <= DATE \'{value}\''.format(field=source_date, value=ALERT_DATE.strftime('%Y-%m-%d'))

executor = SQLExecutor2(dataset=src)
date_sql = '''
SELECT DISTINCT {date_field} AS SOURCE_DATE
FROM {table}
WHERE {condition}
ORDER BY {date_field} DESC
FETCH FIRST 2 ROWS ONLY
'''.format(date_field=source_date, table=table, condition=date_condition)

dates = executor.query_to_df(date_sql)
if len(dates) != 2:
    raise ValueError('Two source snapshots were not found on or before 16 October')
if numeric_date:
    vals = [str(int(x)) for x in dates['SOURCE_DATE']]
    source_dates = [pd.to_datetime(x, format='%Y%m%d').strftime('%Y-%m-%d') for x in vals]
    compare_condition = '{field} IN ({vals})'.format(field=source_date, vals=','.join(vals))
else:
    source_dates = [pd.Timestamp(x).strftime('%Y-%m-%d') for x in dates['SOURCE_DATE']]
    compare_condition = 'TRUNC({field}) IN (DATE \'{one}\', DATE \'{two}\')'.format(
        field=source_date, one=source_dates[0], two=source_dates[1]
    )

print('Requested alert date:', ALERT_DATE.strftime('%Y-%m-%d'))
print('Source snapshots:', source_dates[1], '->', source_dates[0])
if source_dates[0] != ALERT_DATE.strftime('%Y-%m-%d'):
    print('NOTE: The source snapshot date differs from the alert date.')

version_sel = ', {field} AS VERSION_ID'.format(field=version_meta['name']) if version_meta else ''
version_order = 'VERSION_ID DESC' if version_meta else 'INSTRUMENT_ID'

# Apply ranking BEFORE restricting to Stage 1 -> 2, consistent with the EWS source logic.
transition_sql = '''
WITH source_rows AS (
 SELECT TRIM(CAST({contract} AS VARCHAR2(100))) AS INSTRUMENT_ID,
        TRIM(CAST({entity} AS VARCHAR2(100))) AS ENTITY_ID,
        {datefield} AS SOURCE_DATE,
        {stage} AS STAGE{version}
 FROM {table}
 WHERE {cond}
   AND {contract} IS NOT NULL
   AND {entity} IS NOT NULL
), ranked AS (
 SELECT source_rows.*,
        ROW_NUMBER() OVER (PARTITION BY INSTRUMENT_ID, SOURCE_DATE ORDER BY {order_by}) AS RN
 FROM source_rows
), selected AS (
 SELECT INSTRUMENT_ID, ENTITY_ID, SOURCE_DATE, STAGE
 FROM ranked WHERE RN = 1
)
SELECT c.ENTITY_ID, c.INSTRUMENT_ID AS CONTRACT_REFERENCE,
       p.STAGE AS PREVIOUS_STAGE, c.STAGE AS CURRENT_STAGE
FROM selected c
JOIN selected p ON c.INSTRUMENT_ID = p.INSTRUMENT_ID
WHERE {cdate_expr} = {curr}
  AND {pdate_expr} = {prev}
  AND TRIM(CAST(p.STAGE AS VARCHAR2(20))) = '1'
  AND TRIM(CAST(c.STAGE AS VARCHAR2(20))) = '2'
ORDER BY c.ENTITY_ID, c.INSTRUMENT_ID
'''.format(
    contract=contract, entity=entity, datefield=source_date, stage=stage,
    version=version_sel, table=table, cond=compare_condition, order_by=version_order,
    curr=vals[0] if numeric_date else "DATE '" + source_dates[0] + "'",
    prev=vals[1] if numeric_date else "DATE '" + source_dates[1] + "'",
    cdate_expr='c.SOURCE_DATE' if numeric_date else 'TRUNC(c.SOURCE_DATE)',
    pdate_expr='p.SOURCE_DATE' if numeric_date else 'TRUNC(p.SOURCE_DATE)',
)

try:
    found = executor.query_to_df(transition_sql)
except Exception as err:
    print('Oracle rejected the transition query:', str(err)[:500])
    raise

print('\nInstrument-level source transitions Stage 1 -> Stage 2:', len(found))
if found.empty:
    print('No transition in the two source snapshots. Check date coverage or source revisions.')
else:
    found['ENTITY_ID'] = norm_id(found['ENTITY_ID'])
    print('Distinct source entities:', found['ENTITY_ID'].nunique())
    if ids:
        matched = found[found['ENTITY_ID'].isin(ids)].copy()
        print('\nSOURCE TRANSITIONS MATCHING 16 OCT FINAL BREACH ENTITIES:', len(matched))
        print(matched.to_string(index=False))
        missing = ids - set(found['ENTITY_ID'].dropna().astype(str))
        if missing:
            print('16 Oct breach entities with no matching source transition:', len(missing))
            print('These require further investigation before replying to the bank.')
    else:
        print('\nNo historical 16 Oct breach entity list is available in the current output.')
        print('The following are SOURCE-LEVEL candidates, not proven alert recipients:')
        print(found.head(100).to_string(index=False))
        if len(found) > 100:
            print('Additional transitions not shown:', len(found) - 100)

print('\nNOTE: CONTRACT_REFERENCE is not necessarily the T24 account number.')
print('NOTE: This query does not prove which exact case Frank meant without a matching alert record.')
print('NOTE: The output remains unchanged; all SQL statements are SELECT-only.')
