# -*- coding: utf-8 -*-
"""Read-only validation of Corporate account-movement results."""

import re

import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
DATASETS = (
    'payment_movements_monthly_Corporate',
    'payment_movements_signals_Corporate',
    'results_Corporate_Payments',
    'payment_recurring_payers_monthly_Corporate',
    'results_Corporate_Recurring_Payments',
)
PORTFOLIOS = {
    'GLOBAL', 'LARGE_CORPORATE', 'MID_CORPORATE',
    'ADC', 'IP', 'OTHER_CORPORATE',
}
project = dataiku.api_client().get_project(PROJECT)
known = {row['name'] for row in project.list_datasets()}
engine = SQLExecutor2(connection=CONNECTION)


def identifier(value):
    value = str(value).replace('${projectKey}', PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', value):
        raise RuntimeError('Unexpected identifier')
    return '"' + value + '"'


def table_name(name):
    if name not in known:
        raise RuntimeError('Missing dataset: ' + name)
    params = project.get_dataset(name).get_settings().get_raw().get('params') or {}
    if params.get('connection') != CONNECTION:
        raise RuntimeError('Unexpected connection: ' + name)
    return identifier(params['schema']) + '.' + identifier(params['table'])


def run(label, sql):
    print('\n' + label)
    result = engine.query_to_df(sql)
    print(result.to_csv(index=False, sep=';', na_rep='NULL').strip())
    return result


for dataset in DATASETS:
    table = table_name(dataset)
    run(dataset, 'SELECT COUNT(*) N_ROWS FROM ' + table)

staging = table_name('payment_recurring_payers_monthly_Corporate')
source = run(
    'Recurring payer source coverage',
    'SELECT COUNT(DISTINCT PAYMENT_MONTH) N_MONTHS, '
    'COUNT(DISTINCT CASE WHEN IS_SOURCE_COMPLETE=1 THEN PAYMENT_MONTH END) N_COMPLETE_MONTHS, '
    'COUNT(DISTINCT ENTITY_ID) N_ENTITIES '
    'FROM ' + staging,
)
if (source.empty or int(source.iloc[0]['N_MONTHS']) != 25 or
        int(source.iloc[0]['N_COMPLETE_MONTHS']) != 25):
    raise RuntimeError('Recurring payer history is incomplete')

for dataset in ('results_Corporate_Payments', 'results_Corporate_Recurring_Payments'):
    table = table_name(dataset)
    result = run(
        dataset + ' by portfolio',
        'SELECT PORTFOLIO, COUNT(*) N_CONFIGURATIONS, '
        'COUNT(DISTINCT INDICATOR) N_INDICATORS, '
        'SUM(N_ALERTS) N_ALERTS_SUM_ACROSS_CONFIGURATIONS '
        'FROM ' + table + ' GROUP BY PORTFOLIO ORDER BY PORTFOLIO',
    )
    actual = set(result['PORTFOLIO'].astype(str).str.upper())
    if actual != PORTFOLIOS:
        raise RuntimeError('Unexpected portfolio coverage in ' + dataset)

print('\nAccount movement result checks completed.')
print('Alert totals across alternative configurations are not unique customer alerts.')
print('Loan-disbursement classification remains subject to business confirmation.')
