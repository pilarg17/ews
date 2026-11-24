# -*- coding: utf-8 -*-
"""Read-only Corporate EWS quality gate: source vs consolidated accuracy and scope.

Run in a Dataiku notebook attached to the Corporate project. No recipe builds,
no dataset writes and no state mutation. This does NOT compare with the previous
emailed workbook: run compare_accuracy_exports.py with both exports for that.
"""
import re
import pandas as pd
import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
CONSOLIDATED = 'results_Corporate_EWS_consolidated'
SOURCES = (
    'results_Corporate_Payments',
    'results_Corporate_Recurring_Payments',
    'results_Corporate_LTV',
    'results_Corporate_Forbearance',
    'results_Corporate_EWS_review',
    'results_Corporate_EWS_validation',
    'results_Corporate_DPD',
)
PCT_TOL = 0.00011      # stored percentages rounded to 4 decimals
SHOW_EXAMPLES = 20
executor = SQLExecutor2(connection=CONNECTION)
project = dataiku.api_client().get_project(PROJECT)
names = {x['name'] for x in project.list_datasets()}


def sql_id(x):
    val = str(x).replace('${projectKey}', PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9$]*', val):
        raise ValueError('Unexpected physical identifier: ' + repr(val))
    return '"' + val + '"'


def literal(x):
    return "'" + str(x).replace("'", "''") + "'"


def info(name):
    if name not in names:
        return None
    ds = project.get_dataset(name)
    prm = ds.get_settings().get_raw().get('params') or {}
    if prm.get('connection') != CONNECTION:
        print('SKIP_DIFFERENT_CONNECTION', name, prm.get('connection'))
        return None
    if not prm.get('schema') or not prm.get('table'):
        print('SKIP_NO_PHYSICAL_TABLE', name)
        return None
    physical = sql_id(prm['schema']) + '.' + sql_id(prm['table'])
    schema = ds.get_schema()
    if isinstance(schema, dict):
        schema = schema.get('columns', [])
    columns = {x['name'].upper() for x in schema or [] if isinstance(x, dict) and x.get('name')}
    if not columns:
        columns = set(executor.query_to_df('SELECT * FROM ' + physical + ' WHERE 1=0').columns.str.upper())
    return {'name': name, 'table': physical, 'cols': columns}


def select_col(meta, c, alias='t', typ='number'):
    if c not in meta['cols']:
        return 'NULL::VARCHAR' if typ == 'text' else 'NULL::DOUBLE'
    ref = alias + '.' + sql_id(c)
    return 'TO_VARCHAR(' + ref + ')' if typ == 'text' else 'TRY_TO_DOUBLE(' + ref + '::VARCHAR)'


def run(title, query, show=SHOW_EXAMPLES):
    print('\n###', title, flush=True)
    frame = executor.query_to_df(query)
    frame.columns = [str(c).upper() for c in frame.columns]
    if len(frame):
        print(frame.head(show).to_csv(index=False, sep=';', na_rep='NULL'), flush=True)
    else:
        print('NO_ROWS', flush=True)
    return frame


metadata = {name: info(name) for name in (CONSOLIDATED,) + SOURCES}

# 1. Evaluate each stored metric against its own numerator/denominator.
summary = []
for name in (CONSOLIDATED,) + SOURCES:
    meta = metadata[name]
    if not meta:
        continue
    cols = meta['cols']
    if not {'N_TP', 'N_FP', 'ACCURACY'}.issubset(cols):
        print('NO_EVENT_ACCURACY_COLUMNS', name)
        continue
    t = meta['table']
    # Source review table contains statuses as well: only evaluate real backtest rows.
    filt = "WHERE ROW_TYPE='BACKTEST'" if 'ROW_TYPE' in cols else ''
    sig = 'N_TP_ALERTS' in cols and 'N_EVALUATED_EPISODES' in cols and 'ALERT_PRECISION' in cols
    alert_mismatch = ("COUNT_IF(N_EVALUATED_EPISODES>0 AND ALERT_PRECISION IS NOT NULL "
        "AND ABS(ALERT_PRECISION - N_TP_ALERTS/NULLIF(N_EVALUATED_EPISODES,0))>" + str(PCT_TOL) + ")") if sig else 'NULL::NUMBER'
    q = '''SELECT COUNT(*) N_ROWS,
       COUNT_IF(N_TP + N_FP>0 AND ACCURACY IS NOT NULL
            AND ABS(ACCURACY-N_TP/NULLIF(N_TP+N_FP,0))>{tol}) BAD_ACCURACY,
       COUNT_IF(N_TP+N_FP=0 AND ACCURACY IS NOT NULL) ZERO_DENOM_WITH_ACCURACY,
       COUNT_IF(N_TP+N_FP>0 AND ACCURACY IS NULL) MISSING_ACCURACY,
       COUNT_IF(N_TP<0 OR N_FP<0 OR N_TP>N_EVENTS) BAD_COUNTS,
       {alert_mismatch} BAD_ALERT_PRECISION
       FROM {table} {filter}'''.format(tol=PCT_TOL, alert_mismatch=alert_mismatch,
                                       table=t, filter=filt)
    z = run('METRIC_FORMULA | ' + name, q, show=2)
    if len(z):
        rec = z.iloc[0].to_dict()
        rec['DATASET'] = name
        summary.append(rec)

# 2. Exactly reconcile each consolidated backtest row with its ORIGINAL source.
combined = metadata.get(CONSOLIDATED)
if combined and {'SOURCE_DATASET', 'ROW_TYPE', 'TARGET_SCOPE_ORIGINAL'}.issubset(combined['cols']):
    for name in SOURCES:
        src = metadata[name]
        if not src or not {'PORTFOLIO','INDICATOR','N_TP','N_FP','ACCURACY'}.issubset(src['cols']):
            continue
        base = combined['table']; original = src['table']
        filters = []
        if name == 'results_Corporate_EWS_review':
            if not {'FAMILY', 'ROW_TYPE'}.issubset(src['cols']):
                print('REVIEW_MISSING_FAMILY_ROW_TYPE: reconciliation skipped')
                continue
            filters.append("s.FAMILY='LIMIT_UTILISATION' AND s.ROW_TYPE='BACKTEST'")
        if name == 'results_Corporate_EWS_validation':
            # The consolidated script deliberately excludes duplicated legacy DPD,
            # payment and forbearance families; compare only the rows it retained.
            pass
        source_filter = ' AND '.join(filters) if filters else '1=1'
        def key_expr(col):
            if col in src['cols']:
                return select_col(src, col, 's')
            if name == 'results_Corporate_Forbearance' and col in ('THRESHOLD_A','THRESHOLD_B'):
                fallback = {'THRESHOLD_A':'THRESHOLD_GEC_EXPOSURE',
                            'THRESHOLD_B':'THRESHOLD_ARREARS'}[col]
                return select_col(src, fallback, 's')
            return 'NULL::DOUBLE'
        ckey = ['PORTFOLIO', 'INDICATOR', 'TARGET_SCOPE_ORIGINAL',
                'THRESHOLD_A', 'THRESHOLD_B', 'THRESHOLD_C']
        skey = [select_col(src,'PORTFOLIO','s','text'), select_col(src,'INDICATOR','s','text'),
                select_col(src,'TARGET_SCOPE','s','text'), key_expr('THRESHOLD_A'),
                key_expr('THRESHOLD_B'), key_expr('THRESHOLD_C')]
        conditions = ["EQUAL_NULL(c.\"%s\", k.K%s)" % (col, i)
                      for i,col in enumerate(ckey)]
        sq = '''WITH k AS (
          SELECT {keys},
                 {tp} S_TP,{fp} S_FP,{acc} S_ACCURACY,
                 COUNT(*) OVER(PARTITION BY {partition}) S_KEY_ROWS
          FROM {source} s WHERE {source_filter}
        ), c AS (
          SELECT * FROM {consolidated}
          WHERE SOURCE_DATASET={label} AND ROW_TYPE='BACKTEST'
        )
        SELECT c.PORTFOLIO,c.INDICATOR,c.TARGET_SCOPE,
               c.TARGET_SCOPE_ORIGINAL,c.THRESHOLD_A,c.THRESHOLD_B,
               c.N_TP C_TP,k.S_TP,c.N_FP C_FP,k.S_FP,
               c.ACCURACY C_ACCURACY,k.S_ACCURACY,k.S_KEY_ROWS,
               CASE WHEN k.S_KEY_ROWS IS NULL THEN 'NO_SOURCE_MATCH'
                    WHEN k.S_KEY_ROWS<>1 THEN 'AMBIGUOUS_SOURCE_KEY'
                    WHEN NOT EQUAL_NULL(c.N_TP,k.S_TP) OR NOT EQUAL_NULL(c.N_FP,k.S_FP) THEN 'CHANGED_TP_FP'
                    WHEN NOT EQUAL_NULL(c.ACCURACY,k.S_ACCURACY)
                      AND (c.ACCURACY IS NULL OR k.S_ACCURACY IS NULL
                           OR ABS(c.ACCURACY-k.S_ACCURACY)>{tol}) THEN 'CHANGED_ACCURACY'
                    ELSE 'MATCH' END QA_STATUS
        FROM c LEFT JOIN k ON {on}
        '''.format(keys=', '.join('(%s) K%s' % (e,i) for i,e in enumerate(skey)),
                   partition=', '.join(skey), tp=select_col(src,'N_TP','s'),
                   fp=select_col(src,'N_FP','s'),acc=select_col(src,'ACCURACY','s'),
                   source=original, source_filter=source_filter,
                   consolidated=base, label=literal(name),tol=PCT_TOL,
                   on=' AND '.join(conditions))
        try:
            z=run('LINEAGE_DIFF_COUNTS | '+name,
                  'SELECT QA_STATUS,COUNT(*) N FROM ('+sq+') q GROUP BY QA_STATUS ORDER BY 1',show=15)
            bad = z[z['QA_STATUS']!='MATCH'] if len(z) else pd.DataFrame()
            if len(bad):
                run('LINEAGE_EXAMPLES | '+name,
                    'SELECT * FROM ('+sq+") q WHERE QA_STATUS<>'MATCH' ORDER BY QA_STATUS,PORTFOLIO,INDICATOR LIMIT 25")
        except Exception as e:
            # An unsupported source grid or schema is NOT a pass; report explicitly.
            print('LINEAGE_QA_ERROR | '+name+' | '+str(e)[:650], flush=True)
else:
    print('BLOCKER: consolidated output or original-target/source lineage columns unavailable')

# 3. Core partner questions, without inventing product classifications.
lim = metadata.get('results_Corporate_EWS_review')
if lim and 'FAMILY' in lim['cols']:
    run('LIMIT_REVIEW_CONFIGURATIONS', '''
      SELECT INDICATOR,THRESHOLD_A,COUNT(*) N_ROWS,
             COUNT_IF(ACCURACY IS NOT NULL) N_ACCURACY_AVAILABLE
      FROM {table} WHERE FAMILY='LIMIT_UTILISATION' AND ROW_TYPE='BACKTEST'
      GROUP BY 1,2 ORDER BY 1,2
    '''.format(table=lim['table']),show=60)

# 4. Requested payment candidates. Only read the stored backtest results.
pay = metadata.get('results_Corporate_Payments')
if pay and {'PORTFOLIO','INDICATOR','THRESHOLD_A','N_TP','N_FP','ACCURACY'}.issubset(pay['cols']):
    wanted = [
      ('payment_gross_avg3_vs_avg12_decline',30,('ADC','LARGE_CORPORATE')),
      ('payment_incoming_avg3_vs_avg12_decline',20,('ADC','IP','LARGE_CORPORATE')),
      ('payment_top10_floor_prev12_max',20,('IP',)),
      ('payment_net_max3_vs_avg12_decline',30,('ADC',)),
    ]
    req=[]
    for ind,threshold,ports in wanted:
        for portfolio in ports:
            req.append("(%s,%d,%s)"%(literal(ind),threshold,literal(portfolio)))
    q = """WITH requested AS (
       SELECT COLUMN1::VARCHAR INDICATOR,COLUMN2::DOUBLE THRESHOLD_A,
              COLUMN3::VARCHAR PORTFOLIO FROM VALUES {rows}
    )
    SELECT r.INDICATOR,r.THRESHOLD_A,r.PORTFOLIO,
           COUNT(p.INDICATOR) CONFIGURATIONS,
           CASE WHEN COUNT(p.INDICATOR)=0 THEN 'MISSING' ELSE 'EXISTS' END STATUS,
           MIN(p.ACCURACY) MIN_ACCURACY,MAX(p.ACCURACY) MAX_ACCURACY,
           MIN(p.N_TP) MIN_TP,MAX(p.N_TP) MAX_TP,
           MIN(p.N_FP) MIN_FP,MAX(p.N_FP) MAX_FP
    FROM requested r LEFT JOIN {table} p ON p.INDICATOR=r.INDICATOR
       AND p.THRESHOLD_A=r.THRESHOLD_A AND p.PORTFOLIO=r.PORTFOLIO
    GROUP BY 1,2,3 ORDER BY 1,2,3""".format(rows=','.join(req),table=pay['table'])
    run('PAYMENTS_REQUESTED_SHORTLIST | current stored backtest',q,show=30)
    print('ADDITIONAL_REQUEST | payment_top10 prev3 mean at 30% is NOT in this existing grid; requires a separately approved backtest.')

ltv = metadata.get('results_Corporate_LTV')
if ltv and {'INDICATOR','THRESHOLD_A'}.issubset(ltv['cols']):
    run('LTV_EXISTING_GRID_AND_STATUS', '''
      SELECT INDICATOR,THRESHOLD_A,THRESHOLD_B,
             COUNT(*) CONFIGURATIONS,MAX(RESULT_STATUS) STATUS,
             COUNT_IF(ACCURACY IS NOT NULL) N_EVALUATED_ACCURACY
      FROM {table} GROUP BY 1,2,3 ORDER BY 1,2,3
    '''.format(table=ltv['table']),show=40)

# 5. Date and distinct target checks. Normalization may make old and new labels
# appear identical despite differing source definitions.
if combined and {'TARGET_SCOPE','TARGET_SCOPE_ORIGINAL','ROW_TYPE'}.issubset(combined['cols']):
    run('TARGET_LABEL_CHANGES_IN_CONSOLIDATION', '''
      SELECT SOURCE_DATASET,TARGET_SCOPE_ORIGINAL,TARGET_SCOPE,
             COUNT(*) N_ROWS
      FROM {table} WHERE ROW_TYPE='BACKTEST'
        AND NOT EQUAL_NULL(TARGET_SCOPE_ORIGINAL,TARGET_SCOPE)
      GROUP BY 1,2,3 ORDER BY 1,2,3
    '''.format(table=combined['table']),show=35)

print('\nREAD_ONLY_COMPLETED | No datasets modified. Previous emailed export must still be compared separately.',flush=True)
