# -*- coding: utf-8 -*-
"""Read-only comparison for GLOBAL DPD_GEC_ARREARS: DPD 15 / GEC 500000 / Arrears 10000."""
import re
import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
DATASETS = (
    'results_Corporate_EWS',
    'results_Corporate_EWS_validation',
    'results_Corporate_DPD',
    'results_Corporate_EWS_consolidated',
)


def ident(raw):
    value = str(raw).replace('${projectKey}', PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', value):
        raise ValueError('Unexpected Snowflake identifier')
    return '"' + value + '"'

project = dataiku.api_client().get_project(PROJECT)
executor = SQLExecutor2(connection=CONNECTION)
for name in DATASETS:
    try:
        params = project.get_dataset(name).get_settings().get_raw().get('params') or {}
        if params.get('connection') != CONNECTION:
            raise RuntimeError('Unexpected connection for ' + name)
        physical = ident(params['schema']) + '.' + ident(params['table'])
        columns = {str(c).upper() for c in executor.query_to_df(
            'SELECT * FROM ' + physical + ' WHERE 1=0').columns}
        fields = ('PORTFOLIO','INDICATOR','TARGET_SCOPE','TARGET_SCOPE_ORIGINAL',
                  'THRESHOLD_A','THRESHOLD_B','THRESHOLD_C','N_EVENTS','N_TP',
                  'N_TP_ALERTS','N_FP','N_ALERTS','ACCURACY','ALERT_PRECISION')
        select = ', '.join(ident(c) if c in columns else 'NULL AS '+ident(c) for c in fields)
        sql = (f'SELECT {select} FROM {physical} '
               "WHERE UPPER(TRIM(TO_VARCHAR(\"PORTFOLIO\")))='GLOBAL' "
               "AND UPPER(TRIM(TO_VARCHAR(\"INDICATOR\")))='DPD_GEC_ARREARS' "
               'AND TRY_TO_DOUBLE(TO_VARCHAR("THRESHOLD_A"))=15 '
               'AND TRY_TO_DOUBLE(TO_VARCHAR("THRESHOLD_B"))=500000 '
               'AND TRY_TO_DOUBLE(TO_VARCHAR("THRESHOLD_C"))=10000')
        df = executor.query_to_df(sql)
        print('\n=== EXACT_GRID | ' + name + ' | N_ROWS=' + str(len(df)) + ' ===', flush=True)
        print(df.to_csv(index=False, sep=';', na_rep='NULL'), flush=True)
    except Exception as exc:
        print('\nERROR | ' + name + ' | ' + str(exc), flush=True)
print('\nREAD_ONLY_COMPLETED | No datasets modified', flush=True)
