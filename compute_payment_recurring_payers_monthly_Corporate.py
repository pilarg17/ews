# -*- coding: utf-8 -*-
"""Monthly recurring incoming payments by Corporate entity and payer account."""

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
MIN_COVERAGE = 0.95

STARTED = time.monotonic()
project = dataiku.api_client().get_project(PROJECT)
known_datasets = {item['name'] for item in project.list_datasets()}


def log(message):
    print('[%6.1fs] %s' % (time.monotonic() - STARTED, message), flush=True)


def dataset_params(name):
    if name not in known_datasets:
        raise RuntimeError('Missing dataset: ' + name)
    return project.get_dataset(name).get_settings().get_raw().get('params') or {}


def clean_identifier(value):
    value = str(value).replace('${projectKey}', PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', value):
        raise RuntimeError('Unexpected dataset identifier')
    return '"' + value + '"'


source_tables = {
    'PAYEE_VW': ('BDL_PAYMENT_PLY', 'PAYEE_VW'),
    'PAYOR_VW': ('BDL_PAYMENT_PLY', 'PAYOR_VW'),
    'ENTITY_CUSTOMER_MON_VW': ('BDL_CREDIT_INSIGHT_PLY', 'ENTITY_CUSTOMER_MON_VW'),
}
for name, (owner, table) in source_tables.items():
    params = dataset_params(name)
    if (params.get('connection'), params.get('schema'), params.get('table')) != (
            ORACLE_CONNECTION, owner, table):
        raise RuntimeError('Unexpected input source: ' + name)

training_params = dataset_params('EWS_training_Corporate')
output_params = dataset_params(OUTPUT)
if training_params.get('connection') != SNOWFLAKE_CONNECTION:
    raise RuntimeError('Unexpected training connection')
if (output_params.get('connection') != SNOWFLAKE_CONNECTION or
        str(output_params.get('schema', '')).upper() != 'RRG'):
    raise RuntimeError('Unexpected output connection')

expected_outputs = {str(getattr(item, 'name', item)).split('.')[-1]
                    for item in recipe.get_outputs(object_type='DATASET')}
if expected_outputs != {OUTPUT}:
    raise RuntimeError('The recipe must have one output: ' + OUTPUT)

expected_inputs = set(source_tables) | {'EWS_training_Corporate'}
configured_inputs = {str(getattr(item, 'name', item)).split('.')[-1]
                     for item in recipe.get_inputs(object_type='DATASET')}
if not expected_inputs.issubset(configured_inputs):
    raise RuntimeError('Required recipe inputs are not configured')

training_table = (clean_identifier(training_params['schema']) + '.' +
                  clean_identifier(training_params['table']))
output_table = (clean_identifier(output_params['schema']) + '.' +
                clean_identifier(output_params['table']))
if output_table.upper() == training_table.upper():
    raise RuntimeError('The output must differ from the training dataset')

ora = SQLExecutor2(connection=ORACLE_CONNECTION)
snow = SQLExecutor2(connection=SNOWFLAKE_CONNECTION)
required_fields = {
    'PAYEE_VW': {
        'CREDIT_PAYMENT_ID', 'LINKED_DEBIT_PAYMENT_ID',
        'CREDIT_ACCOUNT_CUSTOMER_ID', 'CREDIT_VALUE_DATE',
        'CREDIT_AMOUNT_EUR', 'PAYMENT_DIRECTION', 'STATUS_CODE',
    },
    'PAYOR_VW': {
        'DEBIT_PAYMENT_ID', 'DEBIT_ACCOUNT_CUSTOMER_ID',
        'ORDER_GIVER_ACCOUNT', 'TRANSACTION_PURPOSE_DESCRIPTION',
    },
    'ENTITY_CUSTOMER_MON_VW': {'ENTITY_ID', 'CUSTOMER_ID', 'MAP_DTE'},
}
for name, (owner, table) in source_tables.items():
    statement = (
        "SELECT COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE OWNER='%s' "
        "AND TABLE_NAME='%s'" % (owner, table)
    )
    available = {str(row[0]).upper() for row in ora.query_to_df(statement).itertuples(index=False, name=None)}
    missing = required_fields[name] - available
    if missing:
        raise RuntimeError('Missing source fields in %s: %s' % (name, sorted(missing)))

entities = snow.query_to_df(
    'SELECT DISTINCT TRIM(TO_VARCHAR(ENTITY_ID)) ENTITY_ID FROM ' +
    training_table + ' WHERE ENTITY_ID IS NOT NULL'
)
corporate_entities = {str(value).strip() for value in entities.iloc[:, 0].dropna()}
if not corporate_entities:
    raise RuntimeError('Corporate population is empty')

output_columns = [
    'ENTITY_ID', 'PAYMENT_MONTH', 'AVAILABLE_DATE', 'PAYER_KEY_HASH',
    'INCOMING_EUR', 'ENTITY_INCOMING_EUR', 'IS_SOURCE_COMPLETE', 'SOURCE_STATUS',
]
output_dataset = dataiku.Dataset(OUTPUT, project_key=PROJECT)
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

# The source-purpose description is used for an initial disbursement filter.
# Confirm its completeness against the bank's transaction classification.
loan_disbursement = (
    "REGEXP_LIKE(UPPER(NVL(TO_CHAR(o.TRANSACTION_PURPOSE_DESCRIPTION),' ')), "
    "'(LOAN DISBURS|CREDIT DRAWDOWN|DECAISSEMENT|DÉCAISSEMENT|DEBLOCAGE|DÉBLOCAGE|DARLEHEN|KREDITAUSZAHL)')"
)

payment_query = """
WITH MAPPING_PAIRS AS (
    SELECT DISTINCT TRIM(TO_CHAR(ENTITY_ID)) ENTITY_ID,
           TRIM(TO_CHAR(CUSTOMER_ID)) CUSTOMER_ID
    FROM BDL_CREDIT_INSIGHT_PLY.ENTITY_CUSTOMER_MON_VW
    WHERE MAP_DTE >= DATE '{snapshot}'
      AND MAP_DTE < DATE '{snapshot}' + 1
      AND ENTITY_ID IS NOT NULL AND CUSTOMER_ID IS NOT NULL
),
MAPPING AS (
    SELECT CUSTOMER_ID, MIN(ENTITY_ID) ENTITY_ID
    FROM MAPPING_PAIRS
    GROUP BY CUSTOMER_ID
    HAVING COUNT(DISTINCT ENTITY_ID) = 1
),
CREDIT_PAYMENTS AS (
    SELECT TRIM(TO_CHAR(p.CREDIT_PAYMENT_ID)) CREDIT_PAYMENT_ID,
           c.ENTITY_ID,
           MIN(TRIM(TO_CHAR(p.LINKED_DEBIT_PAYMENT_ID))) DEBIT_PAYMENT_ID,
           COUNT(DISTINCT TRIM(TO_CHAR(p.LINKED_DEBIT_PAYMENT_ID))) N_DEBIT_IDS,
           MIN(p.CREDIT_AMOUNT_EUR) AMOUNT_EUR
    FROM BDL_PAYMENT_PLY.PAYEE_VW p
    JOIN MAPPING c
      ON c.CUSTOMER_ID = TRIM(TO_CHAR(p.CREDIT_ACCOUNT_CUSTOMER_ID))
    WHERE p.CREDIT_VALUE_DATE >= DATE '{month_start}'
      AND p.CREDIT_VALUE_DATE < DATE '{month_end}'
      AND UPPER(TRIM(p.PAYMENT_DIRECTION)) = 'INCOMING'
      AND TRIM(TO_CHAR(p.STATUS_CODE)) NOT IN ('404','996','997','998')
      AND p.CREDIT_AMOUNT_EUR > 0
      AND p.CREDIT_PAYMENT_ID IS NOT NULL
    GROUP BY TRIM(TO_CHAR(p.CREDIT_PAYMENT_ID)), c.ENTITY_ID
    HAVING COUNT(DISTINCT p.CREDIT_AMOUNT_EUR) = 1
),
PAYOR_PAYMENTS AS (
    SELECT TRIM(TO_CHAR(o.DEBIT_PAYMENT_ID)) DEBIT_PAYMENT_ID,
           MIN(TRIM(TO_CHAR(o.ORDER_GIVER_ACCOUNT))) PAYER_ACCOUNT,
           MIN(TRIM(TO_CHAR(o.DEBIT_ACCOUNT_CUSTOMER_ID))) PAYER_CUSTOMER,
           COUNT(DISTINCT TRIM(TO_CHAR(o.ORDER_GIVER_ACCOUNT))) N_PAYER_ACCOUNTS,
           COUNT(DISTINCT TRIM(TO_CHAR(o.DEBIT_ACCOUNT_CUSTOMER_ID))) N_PAYER_CUSTOMERS,
           MAX(CASE WHEN {loan_disbursement} THEN 1 ELSE 0 END) LOAN_DISBURSEMENT
    FROM BDL_PAYMENT_PLY.PAYOR_VW o
    JOIN (
        SELECT DISTINCT DEBIT_PAYMENT_ID
        FROM CREDIT_PAYMENTS
        WHERE N_DEBIT_IDS = 1 AND DEBIT_PAYMENT_ID IS NOT NULL
    ) p ON p.DEBIT_PAYMENT_ID = TRIM(TO_CHAR(o.DEBIT_PAYMENT_ID))
    GROUP BY TRIM(TO_CHAR(o.DEBIT_PAYMENT_ID))
),
RESOLVED AS (
    SELECT c.ENTITY_ID, c.AMOUNT_EUR, p.PAYER_ACCOUNT,
           p.N_PAYER_ACCOUNTS, p.N_PAYER_CUSTOMERS,
           NVL(p.LOAN_DISBURSEMENT,0) LOAN_DISBURSEMENT,
           sender.ENTITY_ID SENDER_ENTITY_ID
    FROM CREDIT_PAYMENTS c
    LEFT JOIN PAYOR_PAYMENTS p
      ON c.N_DEBIT_IDS = 1 AND p.DEBIT_PAYMENT_ID = c.DEBIT_PAYMENT_ID
    LEFT JOIN MAPPING sender
      ON sender.CUSTOMER_ID = p.PAYER_CUSTOMER
),
CLASSIFIED AS (
    SELECT ENTITY_ID, AMOUNT_EUR,
           CASE WHEN PAYER_ACCOUNT IS NOT NULL AND N_PAYER_ACCOUNTS = 1
                     AND (N_PAYER_CUSTOMERS IS NULL OR N_PAYER_CUSTOMERS <= 1)
                     AND LOAN_DISBURSEMENT = 0
                     AND (SENDER_ENTITY_ID IS NULL OR SENDER_ENTITY_ID <> ENTITY_ID)
                THEN RAWTOHEX(STANDARD_HASH(UPPER(TRIM(PAYER_ACCOUNT)), 'SHA256'))
           END PAYER_KEY_HASH,
           CASE WHEN N_PAYER_ACCOUNTS = 1
                     AND (N_PAYER_CUSTOMERS IS NULL OR N_PAYER_CUSTOMERS <= 1)
                     AND (LOAN_DISBURSEMENT = 1 OR SENDER_ENTITY_ID = ENTITY_ID)
                THEN 1 ELSE 0 END EXCLUDED
    FROM RESOLVED
),
ENTITY_TOTALS AS (
    SELECT ENTITY_ID,
           SUM(AMOUNT_EUR) TOTAL_EUR,
           SUM(CASE WHEN PAYER_KEY_HASH IS NOT NULL THEN AMOUNT_EUR ELSE 0 END) KNOWN_EUR,
           SUM(CASE WHEN EXCLUDED = 1 THEN AMOUNT_EUR ELSE 0 END) EXCLUDED_EUR
    FROM CLASSIFIED GROUP BY ENTITY_ID
),
PAYER_TOTALS AS (
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
    raise RuntimeError('Unexpected monthly history length')

output_dataset.write_schema_from_dataframe(output_schema)
total_rows = 0
with output_dataset.get_writer() as writer:
    for month in months:
        end = month + pd.offsets.MonthBegin(1)
        month_start = month.strftime('%Y-%m-%d')
        month_end = end.strftime('%Y-%m-%d')
        latest_mapping = ora.query_to_df(
            "SELECT MAX(TRUNC(MAP_DTE)) SNAPSHOT "
            "FROM BDL_CREDIT_INSIGHT_PLY.ENTITY_CUSTOMER_MON_VW "
            "WHERE MAP_DTE < DATE '%s'" % month_end
        )
        if latest_mapping.empty or pd.isna(latest_mapping.iloc[0, 0]):
            raise RuntimeError('Customer mapping missing for ' + month_start)
        snapshot = pd.Timestamp(latest_mapping.iloc[0, 0]).tz_localize(None).normalize()
        if ((end - pd.Timedelta(days=1)) - snapshot).days > 35:
            raise RuntimeError('Customer mapping is stale for ' + month_start)
        statement = payment_query.format(
            snapshot=snapshot.strftime('%Y-%m-%d'),
            month_start=month_start,
            month_end=month_end,
            loan_disbursement=loan_disbursement,
        )
        monthly = ora.query_to_df(statement)
        monthly.columns = [str(column).upper() for column in monthly.columns]
        if monthly.empty:
            raise RuntimeError('No payments for ' + month_start)
        if len(monthly) > MAX_MONTH_ROWS:
            raise RuntimeError('Monthly source output exceeds expected limit')

        monthly['ENTITY_ID'] = monthly['ENTITY_ID'].astype(str).str.strip().str.replace(
            r'\.0+$', '', regex=True
        )
        monthly = monthly[monthly['ENTITY_ID'].isin(corporate_entities)].copy()
        if monthly.empty:
            raise RuntimeError('No Corporate payment rows for ' + month_start)

        for field in ('INCOMING_EUR', 'TOTAL_EUR', 'KNOWN_EUR', 'EXCLUDED_EUR'):
            monthly[field] = pd.to_numeric(monthly[field], errors='coerce')
        if monthly[['INCOMING_EUR','TOTAL_EUR','KNOWN_EUR','EXCLUDED_EUR']].isna().any().any():
            raise RuntimeError('Invalid amount in ' + month_start)
        covered = ((monthly['KNOWN_EUR'] + monthly['EXCLUDED_EUR']) /
                   monthly['TOTAL_EUR'].replace(0, float('nan')))
        monthly['IS_SOURCE_COMPLETE'] = covered.ge(MIN_COVERAGE).astype(int)
        monthly['ENTITY_INCOMING_EUR'] = monthly['KNOWN_EUR']
        monthly['PAYMENT_MONTH'] = month_start
        monthly['AVAILABLE_DATE'] = (end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        monthly['SOURCE_STATUS'] = 'DISBURSEMENT_CLASSIFICATION_PENDING_CONFIRMATION'
        if monthly.duplicated(['ENTITY_ID','PAYMENT_MONTH','PAYER_KEY_HASH']).any():
            raise RuntimeError('Duplicate payer rows for ' + month_start)
        writer.write_dataframe(monthly[output_columns])
        total_rows += len(monthly)
        log('%s | %d rows' % (month_start, len(monthly)))

persisted = snow.query_to_df(
    'SELECT COUNT(*) N_ROWS, COUNT(DISTINCT PAYMENT_MONTH) N_MONTHS, '
    'COUNT(DISTINCT CASE WHEN IS_SOURCE_COMPLETE=1 THEN PAYMENT_MONTH END) N_COMPLETE_MONTHS '
    'FROM ' + output_table
)
if (persisted.empty or int(persisted.iloc[0, 0]) != total_rows or
        int(persisted.iloc[0, 1]) != len(months)):
    raise RuntimeError('Monthly stage integrity check failed')
log('Recurring incoming payment staging complete')
