# -*- coding: utf-8 -*-
"""Prepare monthly recurring incoming payments by Corporate entity and payer."""

import re
import time

import dataiku
import pandas as pd
from dataiku import recipe
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
ORACLE_CONNECTION = 'PRD_BDWH'
SNOWFLAKE_CONNECTION = 'DEV_AA_TEAMS'
OUTPUT = 'payment_recurring_payers_monthly_Corporate'
FIRST_MONTH = '2024-09-01'
LAST_MONTH = '2026-09-01'
MAX_MONTH_ROWS = 600000

start_time = time.monotonic()
project = dataiku.api_client().get_project(PROJECT)
datasets = {item['name'] for item in project.list_datasets()}


def log(message):
    print('[%6.1fs] %s' % (time.monotonic() - start_time, message), flush=True)


def settings(name):
    if name not in datasets:
        raise RuntimeError('Missing Dataiku dataset: ' + name)
    return project.get_dataset(name).get_settings().get_raw().get('params') or {}


def quoted(value):
    value = str(value).replace('${projectKey}', PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', value):
        raise RuntimeError('Invalid physical dataset identifier')
    return '"' + value + '"'


source_tables = {
    'PAYEE_VW': ('BDL_PAYMENT_PLY', 'PAYEE_VW'),
    'PAYOR_VW': ('BDL_PAYMENT_PLY', 'PAYOR_VW'),
    'ENTITY_CUSTOMER_MON_VW': ('BDL_CREDIT_INSIGHT_PLY', 'ENTITY_CUSTOMER_MON_VW'),
}
for name, (schema, table) in source_tables.items():
    config = settings(name)
    if (config.get('connection') != ORACLE_CONNECTION or
            str(config.get('schema', '')).upper() != schema or
            str(config.get('table', '')).upper() != table):
        raise RuntimeError('Unexpected Oracle source for ' + name)

training_config = settings('EWS_training_Corporate')
output_config = settings(OUTPUT)
if training_config.get('connection') != SNOWFLAKE_CONNECTION:
    raise RuntimeError('Unexpected Corporate training connection')
if (output_config.get('connection') != SNOWFLAKE_CONNECTION or
        str(output_config.get('schema', '')).upper() != 'RRG'):
    raise RuntimeError('Expected a managed Snowflake output in RRG')

inputs = {str(getattr(item, 'name', item)).split('.')[-1]
          for item in recipe.get_inputs(object_type='DATASET')}
outputs = {str(getattr(item, 'name', item)).split('.')[-1]
           for item in recipe.get_outputs(object_type='DATASET')}
required_inputs = set(source_tables) | {'EWS_training_Corporate'}
if not required_inputs.issubset(inputs):
    raise RuntimeError('Missing declared Flow inputs: ' +
                       ', '.join(sorted(required_inputs - inputs)))
if outputs != {OUTPUT}:
    raise RuntimeError('The recipe must have exactly one output: ' + OUTPUT)

training_table = (quoted(training_config['schema']) + '.' +
                  quoted(training_config['table']))
output_table = quoted(output_config['schema']) + '.' + quoted(output_config['table'])
if output_table.upper() == training_table.upper():
    raise RuntimeError('The output cannot share the training table')

for protected in ('results_Corporate_Payments', 'results_Corporate_Recurring_Payments',
                  'results_Corporate_EWS_validation', 'results_Corporate_EWS_consolidated',
                  'payment_movements_monthly_Corporate',
                  'payment_movements_signals_Corporate'):
    if protected not in datasets:
        continue
    config = settings(protected)
    if config.get('connection') == SNOWFLAKE_CONNECTION and config.get('schema') and config.get('table'):
        protected_table = quoted(config['schema']) + '.' + quoted(config['table'])
        if output_table.upper() == protected_table.upper():
            raise RuntimeError('Output shares a protected table: ' + protected)

oracle = SQLExecutor2(connection=ORACLE_CONNECTION)
snowflake = SQLExecutor2(connection=SNOWFLAKE_CONNECTION)
required_columns = {
    'PAYEE_VW': {'CREDIT_PAYMENT_ID', 'LINKED_DEBIT_PAYMENT_ID',
                 'CREDIT_ACCOUNT_CUSTOMER_ID', 'CREDIT_VALUE_DATE',
                 'CREDIT_AMOUNT_EUR', 'PAYMENT_DIRECTION'},
    'PAYOR_VW': {'DEBIT_PAYMENT_ID', 'DEBIT_VALUE_DATE',
                 'DEBIT_ACCOUNT_CUSTOMER_ID', 'ORDER_GIVER_ACCOUNT',
                 'TRANSACTION_PURPOSE_DESCRIPTION'},
    'ENTITY_CUSTOMER_MON_VW': {'ENTITY_ID', 'CUSTOMER_ID', 'MAP_DTE'},
}
for name, (schema, table) in source_tables.items():
    metadata = oracle.query_to_df(
        "SELECT COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE OWNER='%s' AND TABLE_NAME='%s'"
        % (schema, table)
    )
    available = set(metadata.iloc[:, 0].astype(str).str.upper()) if not metadata.empty else set()
    missing = required_columns[name] - available
    if missing:
        raise RuntimeError('Missing columns in %s: %s' % (name, sorted(missing)))

corporate = snowflake.query_to_df(
    'SELECT DISTINCT TRIM(TO_VARCHAR(ENTITY_ID)) ENTITY_ID FROM ' +
    training_table + ' WHERE ENTITY_ID IS NOT NULL'
)
corporate_entities = {str(value).strip() for value in corporate.iloc[:, 0].dropna()}
if not corporate_entities:
    raise RuntimeError('Corporate training population is empty')
log('Corporate entities: %d' % len(corporate_entities))

# This is the existing descriptive exclusion rule. Its completeness must be
# confirmed against the bank's transaction classification before client sign-off.
loan_disbursement = (
    "REGEXP_LIKE(UPPER(NVL(TO_CHAR(o.TRANSACTION_PURPOSE_DESCRIPTION),' ')), "
    "'(LOAN DISBURS|CREDIT DRAWDOWN|DECAISSEMENT|DEBLOCAGE|DÉBLOCAGE|DARLEHEN|KREDITAUSZAHL)')"
)

monthly_sql = """
WITH MAP_PAIRS AS (
    SELECT DISTINCT TRIM(TO_CHAR(ENTITY_ID)) ENTITY_ID,
           TRIM(TO_CHAR(CUSTOMER_ID)) CUSTOMER_ID
    FROM BDL_CREDIT_INSIGHT_PLY.ENTITY_CUSTOMER_MON_VW
    WHERE MAP_DTE >= DATE '{snapshot}' AND MAP_DTE < DATE '{snapshot}' + 1
      AND ENTITY_ID IS NOT NULL AND CUSTOMER_ID IS NOT NULL
), MAP_OK AS (
    SELECT CUSTOMER_ID, MIN(ENTITY_ID) ENTITY_ID
    FROM MAP_PAIRS
    GROUP BY CUSTOMER_ID
    HAVING COUNT(DISTINCT ENTITY_ID) = 1
), CREDIT_PAYMENTS AS (
    SELECT TRIM(TO_CHAR(p.CREDIT_PAYMENT_ID)) CREDIT_PAYMENT_ID,
           MIN(c.ENTITY_ID) ENTITY_ID,
           MIN(p.CREDIT_AMOUNT_EUR) AMOUNT_EUR,
           CASE WHEN COUNT(DISTINCT TRIM(TO_CHAR(p.LINKED_DEBIT_PAYMENT_ID))) = 1
                THEN MIN(TRIM(TO_CHAR(p.LINKED_DEBIT_PAYMENT_ID)))
                ELSE NULL END DEBIT_PAYMENT_ID
    FROM BDL_PAYMENT_PLY.PAYEE_VW p
    JOIN MAP_OK c
      ON c.CUSTOMER_ID = TRIM(TO_CHAR(p.CREDIT_ACCOUNT_CUSTOMER_ID))
    WHERE p.CREDIT_VALUE_DATE >= DATE '{month_start}'
      AND p.CREDIT_VALUE_DATE < DATE '{month_end}'
      AND UPPER(TRIM(p.PAYMENT_DIRECTION)) = 'INCOMING'
      AND p.CREDIT_AMOUNT_EUR > 0
      AND p.CREDIT_PAYMENT_ID IS NOT NULL
    GROUP BY TRIM(TO_CHAR(p.CREDIT_PAYMENT_ID))
    HAVING COUNT(DISTINCT c.ENTITY_ID) = 1
       AND COUNT(DISTINCT p.CREDIT_AMOUNT_EUR) = 1
), PAYOR_PAYMENTS AS (
    SELECT TRIM(TO_CHAR(o.DEBIT_PAYMENT_ID)) DEBIT_PAYMENT_ID,
           MIN(TRIM(TO_CHAR(o.ORDER_GIVER_ACCOUNT))) PAYER_ACCOUNT,
           MIN(TRIM(TO_CHAR(o.DEBIT_ACCOUNT_CUSTOMER_ID))) PAYER_CUSTOMER,
           COUNT(DISTINCT TRIM(TO_CHAR(o.ORDER_GIVER_ACCOUNT))) N_PAYER_ACCOUNTS,
           COUNT(DISTINCT TRIM(TO_CHAR(o.DEBIT_ACCOUNT_CUSTOMER_ID))) N_PAYER_CUSTOMERS,
           MAX(CASE WHEN {loan_disbursement} THEN 1 ELSE 0 END) LOAN_DISBURSEMENT
    FROM BDL_PAYMENT_PLY.PAYOR_VW o
    JOIN (SELECT DISTINCT DEBIT_PAYMENT_ID FROM CREDIT_PAYMENTS
          WHERE DEBIT_PAYMENT_ID IS NOT NULL) p
      ON TRIM(TO_CHAR(o.DEBIT_PAYMENT_ID)) = p.DEBIT_PAYMENT_ID
    WHERE o.DEBIT_VALUE_DATE >= DATE '{month_start}' - 7
      AND o.DEBIT_VALUE_DATE < DATE '{month_end}' + 7
    GROUP BY TRIM(TO_CHAR(o.DEBIT_PAYMENT_ID))
), RESOLVED AS (
    SELECT c.ENTITY_ID, c.AMOUNT_EUR,
           p.PAYER_ACCOUNT, p.N_PAYER_ACCOUNTS, p.N_PAYER_CUSTOMERS,
           p.LOAN_DISBURSEMENT,
           sender.ENTITY_ID SENDER_ENTITY_ID
    FROM CREDIT_PAYMENTS c
    LEFT JOIN PAYOR_PAYMENTS p ON p.DEBIT_PAYMENT_ID = c.DEBIT_PAYMENT_ID
    LEFT JOIN MAP_OK sender ON sender.CUSTOMER_ID = p.PAYER_CUSTOMER
), CLASSIFIED AS (
    SELECT ENTITY_ID, AMOUNT_EUR,
           CASE WHEN PAYER_ACCOUNT IS NOT NULL AND N_PAYER_ACCOUNTS = 1
                     AND N_PAYER_CUSTOMERS <= 1 AND LOAN_DISBURSEMENT = 0
                     AND (SENDER_ENTITY_ID IS NULL OR SENDER_ENTITY_ID <> ENTITY_ID)
                THEN RAWTOHEX(STANDARD_HASH(UPPER(TRIM(PAYER_ACCOUNT)), 'SHA256'))
                ELSE NULL END PAYER_KEY_HASH,
           CASE WHEN LOAN_DISBURSEMENT = 1 OR SENDER_ENTITY_ID = ENTITY_ID
                THEN 1 ELSE 0 END IS_EXCLUDED
    FROM RESOLVED
), ENTITY_TOTALS AS (
    SELECT ENTITY_ID,
           SUM(AMOUNT_EUR) TOTAL_EUR,
           SUM(CASE WHEN PAYER_KEY_HASH IS NOT NULL THEN AMOUNT_EUR ELSE 0 END) KNOWN_EUR,
           SUM(CASE WHEN IS_EXCLUDED = 1 THEN AMOUNT_EUR ELSE 0 END) EXCLUDED_EUR
    FROM CLASSIFIED GROUP BY ENTITY_ID
), PAYER_TOTALS AS (
    SELECT ENTITY_ID, PAYER_KEY_HASH, SUM(AMOUNT_EUR) INCOMING_EUR
    FROM CLASSIFIED
    WHERE PAYER_KEY_HASH IS NOT NULL
    GROUP BY ENTITY_ID, PAYER_KEY_HASH
)
SELECT t.ENTITY_ID,
       NVL(p.PAYER_KEY_HASH, '__SOURCE__') PAYER_KEY_HASH,
       NVL(p.INCOMING_EUR, 0) INCOMING_EUR,
       t.TOTAL_EUR, t.KNOWN_EUR, t.EXCLUDED_EUR
FROM ENTITY_TOTALS t
LEFT JOIN PAYER_TOTALS p ON t.ENTITY_ID = p.ENTITY_ID
"""

months = pd.date_range(FIRST_MONTH, LAST_MONTH, freq='MS')
if len(months) != 25:
    raise RuntimeError('Unexpected number of months')

# Check all historical customer mappings before replacing the existing stage.
snapshots = {}
for month in months:
    following_month = month + pd.offsets.MonthBegin(1)
    month_start = month.strftime('%Y-%m-%d')
    month_end = following_month.strftime('%Y-%m-%d')
    result = oracle.query_to_df(
        "SELECT MAX(TRUNC(MAP_DTE)) SNAPSHOT FROM "
        "BDL_CREDIT_INSIGHT_PLY.ENTITY_CUSTOMER_MON_VW "
        "WHERE MAP_DTE < DATE '%s'" % month_end
    )
    if result.empty or pd.isna(result.iloc[0, 0]):
        raise RuntimeError('No customer mapping for ' + month_start)
    snapshot = pd.Timestamp(result.iloc[0, 0])
    if snapshot.tzinfo is not None:
        snapshot = snapshot.tz_localize(None)
    snapshot = snapshot.normalize()
    if ((following_month - pd.Timedelta(days=1)) - snapshot).days > 35:
        raise RuntimeError('Customer mapping is outdated for ' + month_start)
    snapshots[month_start] = snapshot.strftime('%Y-%m-%d')

output_columns = [
    'ENTITY_ID', 'PAYMENT_MONTH', 'AVAILABLE_DATE', 'PAYER_KEY_HASH',
    'INCOMING_EUR', 'ENTITY_INCOMING_EUR', 'IS_SOURCE_COMPLETE', 'SOURCE_STATUS',
]
output_schema = pd.DataFrame({
    'ENTITY_ID': pd.Series(dtype='object'),
    'PAYMENT_MONTH': pd.Series(dtype='object'),
    'AVAILABLE_DATE': pd.Series(dtype='object'),
    'PAYER_KEY_HASH': pd.Series(dtype='object'),
    'INCOMING_EUR': pd.Series(dtype='float64'),
    'ENTITY_INCOMING_EUR': pd.Series(dtype='float64'),
    'IS_SOURCE_COMPLETE': pd.Series(dtype='int64'),
    'SOURCE_STATUS': pd.Series(dtype='object'),
})[output_columns]

stage = dataiku.Dataset(OUTPUT, project_key=PROJECT)
stage.write_schema_from_dataframe(output_schema)
total_rows = 0

with stage.get_writer() as writer:
    for month in months:
        following_month = month + pd.offsets.MonthBegin(1)
        month_start = month.strftime('%Y-%m-%d')
        month_end = following_month.strftime('%Y-%m-%d')
        statement = monthly_sql.format(
            snapshot=snapshots[month_start],
            month_start=month_start,
            month_end=month_end,
            loan_disbursement=loan_disbursement,
        )
        data = oracle.query_to_df(statement)
        data.columns = [str(column).upper() for column in data.columns]
        if data.empty or len(data) > MAX_MONTH_ROWS:
            raise RuntimeError('Unexpected source volume for ' + month_start)

        data['ENTITY_ID'] = data['ENTITY_ID'].astype(str).str.strip()
        data = data[data['ENTITY_ID'].isin(corporate_entities)].copy()
        if data.empty:
            raise RuntimeError('No Corporate payments for ' + month_start)

        amounts = ['INCOMING_EUR', 'TOTAL_EUR', 'KNOWN_EUR', 'EXCLUDED_EUR']
        for column in amounts:
            data[column] = pd.to_numeric(data[column], errors='coerce')
        if data[amounts].isna().any().any():
            raise RuntimeError('Invalid source amount for ' + month_start)
        if (data[amounts] < 0).any().any():
            raise RuntimeError('Negative source amount for ' + month_start)
        if data.duplicated(['ENTITY_ID', 'PAYER_KEY_HASH']).any():
            raise RuntimeError('Duplicate payer group for ' + month_start)

        # A missing payer is treated as zero only for months in which every
        # available incoming payment was identified or explicitly excluded.
        unresolved = data['TOTAL_EUR'] - data['KNOWN_EUR'] - data['EXCLUDED_EUR']
        if (unresolved < -0.01).any():
            raise RuntimeError('Payment classification is inconsistent for ' + month_start)
        data['IS_SOURCE_COMPLETE'] = unresolved.abs().le(0.01).astype(int)
        data['ENTITY_INCOMING_EUR'] = data['KNOWN_EUR']
        data['PAYMENT_MONTH'] = month_start
        data['AVAILABLE_DATE'] = (
            following_month - pd.Timedelta(days=1)
        ).strftime('%Y-%m-%d')
        data['SOURCE_STATUS'] = 'PAYMENT_EXCLUSIONS_QA_PENDING'

        writer.write_dataframe(data[output_columns])
        total_rows += len(data)
        log('%s | %d payer groups | %d complete' %
            (month_start, len(data), int(data['IS_SOURCE_COMPLETE'].sum())))

if total_rows == 0:
    raise RuntimeError('No recurring payment rows written')

persisted = snowflake.query_to_df(
    'SELECT COUNT(*) N_ROWS, COUNT(DISTINCT PAYMENT_MONTH) N_MONTHS, '
    'COUNT(DISTINCT IFF(IS_SOURCE_COMPLETE=1,PAYMENT_MONTH,NULL)) N_COMPLETE_MONTHS '
    'FROM ' + output_table
)
if (persisted.empty or int(persisted.iloc[0, 0]) != total_rows or
        int(persisted.iloc[0, 1]) != 25 or int(persisted.iloc[0, 2]) != 25):
    raise RuntimeError('Recurring payer stage validation failed')

log('Recurring incoming payment stage built successfully')
