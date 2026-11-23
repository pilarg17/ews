# -*- coding: utf-8 -*-
"""Read-only checks for Corporate payment data and existing backtests."""

import re

import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
DATASETS = (
    'payment_movements_monthly_Corporate',
    'payment_movements_signals_Corporate',
    'payment_recurring_payers_monthly_Corporate',
    'results_Corporate_Payments',
    'results_Corporate_Recurring_Payments',
)
PORTFOLIOS = {
    'GLOBAL', 'LARGE_CORPORATE', 'MID_CORPORATE',
    'ADC', 'IP', 'OTHER_CORPORATE',
}

project = dataiku.api_client().get_project(PROJECT)
known = {item['name'] for item in project.list_datasets()}
sql = SQLExecutor2(connection=CONNECTION)


def quoted(value):
    value = str(value).replace('${projectKey}', PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', value):
        raise RuntimeError('Unexpected dataset identifier')
    return '"' + value + '"'


def physical(name):
    if name not in known:
        raise RuntimeError('Missing dataset: ' + name)
    config = project.get_dataset(name).get_settings().get_raw().get('params') or {}
    if config.get('connection') != CONNECTION:
        raise RuntimeError('Unexpected dataset connection: ' + name)
    return quoted(config['schema']) + '.' + quoted(config['table'])


def report(label, query):
    data = sql.query_to_df(query)
    print('\n' + label)
    print(data.to_csv(index=False, sep=';', na_rep='NULL').strip())
    return data


for name in DATASETS:
    report('Rows | ' + name, 'SELECT COUNT(*) N_ROWS FROM ' + physical(name))

monthly = physical('payment_recurring_payers_monthly_Corporate')
quality = report(
    'Recurring payer history quality',
    '''WITH ENTITY_MONTH AS (
        SELECT ENTITY_ID, PAYMENT_MONTH, MIN(IS_SOURCE_COMPLETE) COMPLETE
        FROM %s
        GROUP BY ENTITY_ID, PAYMENT_MONTH
    )
    SELECT COUNT(DISTINCT PAYMENT_MONTH) N_MONTHS,
           COUNT(*) N_ENTITY_MONTHS,
           COUNT_IF(COMPLETE = 1) N_COMPLETE_ENTITY_MONTHS,
           COUNT_IF(COMPLETE <> 1) N_INCOMPLETE_ENTITY_MONTHS
    FROM ENTITY_MONTH''' % monthly,
)
if quality.empty or int(quality.iloc[0]['N_MONTHS']) != 25:
    raise RuntimeError('The recurring payer source does not cover 25 months')

for name, expected in (('results_Corporate_Payments', 666),
                       ('results_Corporate_Recurring_Payments', 12)):
    data = report(
        'Backtest | ' + name,
        'SELECT PORTFOLIO, COUNT(*) N_CONFIGS, SUM(N_TP) SUM_TP, '
        'SUM(N_FP) SUM_FP, SUM(N_ALERTS) SUM_ALERTS '
        'FROM ' + physical(name) + ' GROUP BY PORTFOLIO ORDER BY PORTFOLIO',
    )
    if set(data['PORTFOLIO'].astype(str).str.upper()) != PORTFOLIOS:
        raise RuntimeError('Unexpected portfolios in ' + name)
    if int(data['N_CONFIGS'].sum()) != expected:
        raise RuntimeError('Unexpected number of configurations in ' + name)

print('\nValidation completed. Source exclusions still require business approval.')
print('Backtest summaries count alternative configurations, not distinct customers.')
