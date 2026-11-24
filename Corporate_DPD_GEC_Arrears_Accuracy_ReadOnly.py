# -*- coding: utf-8 -*-
"""Dataiku notebook: read-only reconciliation for one Corporate DPD/GEC/arrears grid.
No datasets, recipes, schemas, or jobs are modified.
"""
import re
import pandas as pd
import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
INDICATOR = 'DPD_GEC_ARREARS'
DPD_DAYS = 15
GEC_EUR = 500000
DATASETS = [
    'results_Corporate_DPD',
    'results_Corporate_EWS_validation',
    'results_Corporate_EWS',
    'results_Corporate_EWS_consolidated',
]

def ident(value):
    v = str(value).replace('${projectKey}', PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', v):
        raise RuntimeError('Unsafe SQL identifier')
    return '"' + v + '"'

api = dataiku.api_client().get_project(PROJECT)
known = {r['name'] for r in api.list_datasets()}
exe = SQLExecutor2(connection=CONNECTION)
frames = []
for name in DATASETS:
    if name not in known:
        print('SOURCE_NOT_AVAILABLE |', name, flush=True)
        continue
    try:
        prm = api.get_dataset(name).get_settings().get_raw().get('params') or {}
        if prm.get('connection') != CONNECTION:
            print('SOURCE_CONNECTION_DIFFERENT |', name, flush=True)
            continue
        table = ident(prm['schema']) + '.' + ident(prm['table'])
        cols = {str(v).upper() for v in exe.query_to_df('SELECT * FROM ' + table + ' WHERE 1=0').columns}
        mandatory = {'PORTFOLIO', 'INDICATOR', 'THRESHOLD_A', 'THRESHOLD_B', 'THRESHOLD_C', 'N_TP', 'N_FP', 'ACCURACY'}
        if not mandatory.issubset(cols):
            print('SOURCE_MISSING_FIELDS |', name, ','.join(sorted(mandatory - cols)), flush=True)
            continue
        wanted = ['PORTFOLIO', 'INDICATOR', 'TARGET_SCOPE', 'TARGET_SCOPE_ORIGINAL',
                  'THRESHOLD_A', 'THRESHOLD_B', 'THRESHOLD_C', 'N_EVENTS',
                  'N_TP', 'N_TP_ALERTS', 'N_FP', 'N_ALERTS', 'ACCURACY', 'ALERT_PRECISION']
        expressions = [((ident(c) if c in cols else 'NULL') + ' AS ' + ident(c)) for c in wanted]
        sql = ("SELECT " + ', '.join(expressions) + ' FROM ' + table +
               " WHERE UPPER(TRIM(TO_VARCHAR(INDICATOR)))='DPD_GEC_ARREARS'" +
               ' AND TRY_TO_DOUBLE(THRESHOLD_A)=15 AND TRY_TO_DOUBLE(THRESHOLD_B)=500000')
        df = exe.query_to_df(sql)
        df.columns = [str(c).upper() for c in df.columns]
        df.insert(0, 'SOURCE', name)
        for c in ('N_EVENTS', 'N_TP', 'N_TP_ALERTS', 'N_FP', 'N_ALERTS', 'ACCURACY', 'ALERT_PRECISION', 'THRESHOLD_C'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['EVENT_BASED_3DP'] = (df['N_TP'] / (df['N_TP'] + df['N_FP']).replace(0, float('nan'))).round(3)
        df['ALERT_BASED_3DP'] = (df['N_TP_ALERTS'] / (df['N_TP_ALERTS'] + df['N_FP']).replace(0, float('nan'))).round(3)
        df['STORED_MATCHES_EVENT_3DP'] = (df['ACCURACY'] - df['EVENT_BASED_3DP']).abs().le(0.0001)
        df['STORED_MATCHES_ALERT_3DP'] = (df['ACCURACY'] - df['ALERT_BASED_3DP']).abs().le(0.0001)
        frames.append(df)
        print('SOURCE_ROWS | %s | %d' % (name, len(df)), flush=True)
    except Exception as exc:
        print('SOURCE_READ_ERROR | %s | %s' % (name, str(exc)[:250]), flush=True)

if not frames:
    raise RuntimeError('No eligible results found; no writes performed')
all_rows = pd.concat(frames, ignore_index=True)
keys = ['PORTFOLIO', 'THRESHOLD_C']
columns = ['SOURCE', 'PORTFOLIO', 'TARGET_SCOPE', 'TARGET_SCOPE_ORIGINAL',
           'THRESHOLD_C', 'N_EVENTS', 'N_TP', 'N_TP_ALERTS', 'N_FP', 'N_ALERTS',
           'ACCURACY', 'ALERT_PRECISION', 'EVENT_BASED_3DP', 'ALERT_BASED_3DP',
           'STORED_MATCHES_EVENT_3DP', 'STORED_MATCHES_ALERT_3DP']
print('\n=== ROWS_WITH_REPORTED_ACCURACY_0_255 ===', flush=True)
flag = all_rows['ACCURACY'].sub(0.255).abs().le(0.0005)
print(all_rows.loc[flag, columns].sort_values(['PORTFOLIO','THRESHOLD_C','SOURCE']).to_csv(sep=';', index=False, na_rep='NULL'), flush=True)

print('\n=== CURRENT_DPD_SAME_GRID_ALL_ARREARS_THRESHOLDS ===', flush=True)
current = all_rows.loc[all_rows['SOURCE'].eq('results_Corporate_DPD'), columns]
print(current.sort_values(['PORTFOLIO','THRESHOLD_C']).to_csv(sep=';', index=False, na_rep='NULL'), flush=True)

print('\n=== SELECTED_GRID_COMPARISON ===', flush=True)
summary = all_rows.groupby(['SOURCE','TARGET_SCOPE'], dropna=False).agg(
    N_ROWS=('ACCURACY','size'),
    N_ACC_EQUALS_0_255=('ACCURACY',lambda s:int(s.sub(0.255).abs().le(0.0005).sum())),
    N_STORED_EQUALS_EVENT=('STORED_MATCHES_EVENT_3DP','sum'),
    N_STORED_EQUALS_ALERT=('STORED_MATCHES_ALERT_3DP','sum'),
).reset_index()
print(summary.to_csv(sep=';', index=False, na_rep='NULL'), flush=True)
print('READ_ONLY_COMPLETED | No source modified | Compare exact portfolio, arrears threshold, target, TP and FP', flush=True)
