# -*- coding: utf-8 -*-
"""Read-only diagnostic: which loan/product types enter Corporate Limit Utilisation?

Runs in a Dataiku notebook with Snowflake access. Does not categorise a purpose
code as revolving unless a validated banking product mapping exists.
"""
import re
import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT='VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION='DEV_AA_TEAMS'
SOURCE='vars_loan_level'
CANDIDATE_FIELDS=(
  'IS_REVOLVING','REVOLVING_FLAG','REVOLVING_IND','CREDIT_TYPE',
  'FACILITY_TYPE','PRODUCT_TYPE','PRODUCT_CATEGORY','LOAN_TYPE',
  'CONTRACT_TYPE','CONTRACT_INTERNAL_PURPOSE_CODE','CONTRACT_PRODUCT_CODE',
  'INSTRUMENT_TYPE','SOURCE_PRODUCT_CODE',
)

def ident(z):
    z=str(z).replace('${projectKey}',PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9$]*',z):raise ValueError(z)
    return '"'+z+'"'

project=dataiku.api_client().get_project(PROJECT)
if SOURCE not in {x['name'] for x in project.list_datasets()}:
    raise RuntimeError('Loan source not present: '+SOURCE)
ds=project.get_dataset(SOURCE)
raw=ds.get_settings().get_raw();p=raw.get('params') or {}
if p.get('connection')!=CONNECTION:raise RuntimeError('Unexpected source connection')
physical=ident(p['schema'])+'.'+ident(p['table'])
sc=ds.get_schema();sc=sc.get('columns',[]) if isinstance(sc,dict) else sc
cols={x['name'].upper() for x in sc}
query=SQLExecutor2(connection=CONNECTION)
print('SOURCE',SOURCE,'PHYSICAL',physical)
found=[x for x in CANDIDATE_FIELDS if x in cols]
print('CANDIDATE_FIELDS_FOUND',','.join(found) if found else 'NONE')
print('WARNING | No codes or categorical labels are treated as confirmed revolving/term classification.')
# A single latest-month snapshot; do not scan every daily loan row into Python.
latest=query.query_to_df('SELECT MAX(TO_DATE(MAP_DTE)) AS D FROM '+physical)
if latest.empty or latest.iloc[0,0] is None:raise RuntimeError('No snapshot')
latest_day=str(latest.iloc[0,0])[:10]
print('LATEST_SNAPSHOT',latest_day)
for col in found:
    c=ident(col)
    sql="""
      SELECT COALESCE(TO_VARCHAR({c}),'<NULL>') CATEGORY,
             COUNT(*) LOAN_ROWS,
             COUNT(DISTINCT INSTRUMENT_ID) INSTRUMENTS,
             COUNT(DISTINCT ENTITY_ID) ENTITIES
      FROM {t}
      WHERE TO_DATE(MAP_DTE)=DATE '{dt}'
      GROUP BY 1 ORDER BY LOAN_ROWS DESC LIMIT 35
    """.format(c=c,t=physical,dt=latest_day)
    try:
        df=query.query_to_df(sql)
        print('\nPRODUCT_SOURCE_FIELD',col,'\n'+df.to_csv(index=False,sep=';'),flush=True)
    except Exception as exc:
        print('FIELD_QUERY_ERROR',col,str(exc)[:300])
print('\nNEXT_GATE: obtain business-approved revolving product code mapping; do not label all historical Limit ratios as revolving-only.')
