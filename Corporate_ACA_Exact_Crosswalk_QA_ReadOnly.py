# -*- coding: utf-8 -*-
"""Corporate EWS | ACA exact key follow-up, read-only, one Dataiku notebook cell.
Checks exact source keys, not fuzzy/name matches. No writes or recipe changes.
Prints only aggregated QA; private client identifiers never printed.
"""
from collections import defaultdict
import re
import pandas as pd
import dataiku
from dataiku.core.sql import SQLExecutor2

CONNECTION='PRD_BDWH'
PROJECT='VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CUT_OFF='2026-10-29'
BATCH=150
oracle=SQLExecutor2(connection=CONNECTION)

def query(label,sql):
    if not re.match(r'^\s*(SELECT|WITH)\b',sql,re.I):
        raise ValueError('SELECT only')
    print('QUERY | '+label,flush=True)
    f=oracle.query_to_df(sql)
    f.columns=[str(c).upper() for c in f.columns]
    return f

def batch(seq):
    seq=sorted(set(s for s in seq if s))
    for j in range(0,len(seq),BATCH):yield seq[j:j+BATCH]

def literal(s):
    s=str(s)
    if not re.fullmatch(r'[a-zA-Z0-9_.\-]{1,160}',s):
        raise ValueError('Unexpected identifier characters; stop rather than silently altering IDs')
    return "'"+s+"'"

def clean_id(x):
    if pd.isna(x):return None
    t=str(x).strip()
    if re.fullmatch(r'\d+\.0+',t):t=t.split('.')[0]
    return t if t and t.upper() not in ('NULL','NONE','NAN') else None

def summary(title,rows):
    print('\n=== '+title+' ===',flush=True)
    print(pd.DataFrame(rows).to_csv(sep=';',index=False).strip(),flush=True)

st=dataiku.Dataset('ACA_slotting_entity_day',project_key=PROJECT).get_dataframe(columns=['ENTITY_ID'])
ids={clean_id(v) for v in st.ENTITY_ID.unique()}
ids.discard(None)
assert len(ids)==233, 'Unexpected ACA staging entity count; do not infer mapping'
print('READ ONLY | ACA EXACT KEY QA | STAGING_ENTITIES',len(ids),flush=True)

owner='BDL_CREDIT_INSIGHT_PLY'
source_names=('ENTITY_VW','ENTITY_CUSTOMER_MON_VW')
meta=query('official bridge metadata',
  "SELECT TABLE_NAME,COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE OWNER='"+owner+
  "' AND TABLE_NAME IN ('ENTITY_VW','ENTITY_CUSTOMER_MON_VW') "+
  "AND COLUMN_NAME IN ('ENTITY_ID','CUSTOMER_ID','MAP_DTE')")
fields={t:set(v.COLUMN_NAME) for t,v in meta.groupby('TABLE_NAME')}

# map candidate source keys to exact, official EWS entity IDs
key_map=defaultdict(set)
method_map=defaultdict(set)
for e in ids:
    key_map[e].add(e)
    method_map[('DIRECT_ENTITY_ID',e)].add(e)
bridge_stats=[]
for table in source_names:
    if not {'ENTITY_ID','CUSTOMER_ID'}.issubset(fields.get(table,set())):
        bridge_stats.append({'BRIDGE_TABLE':table,'STATUS':'FIELDS_MISSING','ENTITY_MATCHES':None,'AMBIGUOUS_CUSTOMERS':None})
        continue
    rows=[]
    for chunk in batch(ids):
        filt=" AND MAP_DTE < DATE '2026-10-30'" if 'MAP_DTE' in fields[table] else ''
        # Prefix with K to preserve source IDs in DataFrame conversion.
        sql=("SELECT DISTINCT 'K'||TRIM(TO_CHAR(ENTITY_ID)) EWS_KEY, "+
             "'K'||TRIM(TO_CHAR(CUSTOMER_ID)) CUSTOMER_KEY "+
             'FROM '+owner+'.'+table+' WHERE CUSTOMER_ID IS NOT NULL'+filt+
             ' AND TRIM(TO_CHAR(ENTITY_ID)) IN ('+','.join(literal(x) for x in chunk)+')')
        x=query(table+' bridge',sql)
        if not x.empty:rows.append(x)
    if not rows:
        bridge_stats.append({'BRIDGE_TABLE':table,'STATUS':'NO_KEYS','ENTITY_MATCHES':0,'AMBIGUOUS_CUSTOMERS':0})
        continue
    m=pd.concat(rows,ignore_index=True).drop_duplicates()
    m['ENTITY']=m.EWS_KEY.astype(str).str.removeprefix('K').map(clean_id)
    m['CUSTOMER']=m.CUSTOMER_KEY.astype(str).str.removeprefix('K').map(clean_id)
    m=m.dropna(subset=['ENTITY','CUSTOMER'])
    cardinality=m.groupby('ENTITY').CUSTOMER.nunique()
    accepted=cardinality[cardinality.eq(1)].index
    for e,c in m.loc[m.ENTITY.isin(accepted),['ENTITY','CUSTOMER']].itertuples(index=False):
        key_map[c].add(e)
        method_map[(table+'_CUSTOMER_ID',c)].add(e)
    bridge_stats.append({'BRIDGE_TABLE':table,'STATUS':'READ_ONLY',
        'ENTITY_MATCHES':int(m.ENTITY.nunique()),
        'AMBIGUOUS_CUSTOMERS':int(cardinality.gt(1).sum()),
        'UNIQUE_MAPPED_ENTITIES':len(accepted)})
summary('01_OFFICIAL_ENTITY_CUSTOMER_BRIDGES',bridge_stats)

views=('PLY_ACA_LEGAL_ENTITY_DAY_VW','PLY_ACA_LEGAL_ENTITY_VW')
aca_meta=query('ACA key columns',
 "SELECT TABLE_NAME,COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE OWNER='BDWH_PLY' "+
 "AND TABLE_NAME IN ('PLY_ACA_LEGAL_ENTITY_DAY_VW','PLY_ACA_LEGAL_ENTITY_VW') "+
 "AND COLUMN_NAME IN ('LEGAL_ENTITY_ID','LEGAL_ENTITY_REF_CODE','MAP_DTE','RISKASSESSMENT_SCORE','FINAL_RATING')")
aca_fields={t:set(v.COLUMN_NAME) for t,v in aca_meta.groupby('TABLE_NAME')}

results=[]
for view in views:
    opts=aca_fields.get(view,set())
    for acol in ('LEGAL_ENTITY_ID','LEGAL_ENTITY_REF_CODE'):
        if acol not in opts:continue
        date_modes = ([('ASOF'," AND MAP_DTE >= DATE '2026-10-29' AND MAP_DTE < DATE '2026-10-30'"),
                       ('HISTORY_ASOF'," AND MAP_DTE < DATE '2026-10-30'")]
                      if view.endswith('_DAY_VW') and 'MAP_DTE' in opts else [('AVAILABLE_HISTORY','')])
        for date_mode in date_modes:
            # Read in chunks and project only source keys, never client attributes.
            matched=defaultdict(set)
            matched_source_keys=set()
            n_source_rows=0
            for chunk in batch(key_map.keys()):
                sql=("SELECT 'K'||TRIM(TO_CHAR("+acol+")) SOURCE_KEY "+
                     'FROM BDWH_PLY.'+view+' WHERE '+acol+' IS NOT NULL'+date_mode[1]+
                     ' AND TRIM(TO_CHAR('+acol+')) IN ('+','.join(literal(x) for x in chunk)+')')
                z=query(view+' '+acol+' '+date_mode[0],sql)
                n_source_rows+=len(z)
                for token in z.SOURCE_KEY.dropna().astype(str).unique():
                    v=clean_id(token[1:]) if token.startswith('K') else None
                    if v in key_map:
                        matched_source_keys.add(v)
                        for e in key_map[v]:matched[e].add(v)
            ambiguities=sum(len(s)>1 for s in matched.values())
            shared_source_keys=sum(len(key_map[v])>1 for v in matched_source_keys)
            results.append({'ACA_VIEW':view,'ACA_KEY_COLUMN':acol,'DATE_SCOPE':date_mode[0],
                            'MATCHED_EWS_ENTITIES':len(matched),
                            'MATCHED_SOURCE_KEYS':len(matched_source_keys),
                            'ENTITIES_MULTIPLE_SOURCE_KEYS':ambiguities,
                            'SOURCE_KEYS_SHARED_BETWEEN_ENTITIES':shared_source_keys,
                            'MATCHED_SOURCE_ROWS':n_source_rows})
summary('02_EXACT_ACA_KEY_MATCHES',results)
print('NOTE: matches using CUSTOMER_ID bridges are candidate technical joins; validate cardinality and the bank key definition.')
print('NOTE: AVAILABLE_HISTORY view is NOT as-of verified if no date filter exists.')
print('NOTE: historical staging may be based on unapproved name matching; do not count it as exact source-key validation.')
print('NOTE: Excel column CP and the anomalous ACA score still need independent reconciliation.')
print('READ ONLY | NO DATASETS OR RECIPES CHANGED')
