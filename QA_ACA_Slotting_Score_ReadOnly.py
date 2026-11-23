# -*- coding: utf-8 -*-
"""Corporate ACA slotting | staging/history and risk score field discovery.

Paste into a SINGLE Dataiku Python notebook cell.
READ ONLY: no writes, recipe or backtest runs.
Timo referred to COLUMN CP in an ACA Excel extract. This script discovers Oracle
candidate fields; it cannot prove which one corresponds to Excel CP without that extract.
"""
import re
import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2

PROJECT='VALIDATIONEWSCORPORATEPORTFOLIODAILY'
DATASET='ACA_slotting_entity_day'
ora=SQLExecutor2(connection='PRD_BDWH')
print('ACA | READ ONLY | CHECK CURRENT HISTORICAL STAGING DATASET',flush=True)
try:
    df=dataiku.Dataset(DATASET,project_key=PROJECT).get_dataframe()
    df.columns=[str(c).upper() for c in df.columns]
    print('DATASET:',DATASET,'ROWS:',len(df),'COLUMNS:',len(df.columns),flush=True)
    print('AVAILABLE_COLUMNS:',', '.join(df.columns),flush=True)
    if 'ENTITY_ID' in df:
        print('UNIQUE_ENTITIES:',df['ENTITY_ID'].nunique(),flush=True)
    if 'MAP_DTE' in df:
        dt=pd.to_datetime(df['MAP_DTE'],errors='coerce')
        print('DATES: ',dt.min(), 'to',dt.max(),'N_DATES:',dt.nunique(),flush=True)
        if 'ENTITY_ID' in df:
            print('DUPLICATE_ENTITY_DAY:',int(df.duplicated(['ENTITY_ID','MAP_DTE']).sum()),flush=True)
    candidates=[x for x in df if any(z in x for z in ('SCORE','RISK','ASSESS','FACTOR','DSCR','RATING','SLOT'))]
    print('STAGING_SCORE_OR_RATING_COLUMNS:',candidates,flush=True)
    for col in candidates[:15]:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(col,'NONNULL',int(df[col].notna().sum()),
                  'DISTINCT',int(df[col].nunique(dropna=True)),
                  'MIN',df[col].min(),'MAX',df[col].max(),flush=True)
except Exception as exc:
    print('STAGING_DATASET_NOT_READABLE:',type(exc).__name__,str(exc)[:320],flush=True)
    print('Continue to Oracle source discovery. Do not rename/materialize existing tables here.',flush=True)

source_tables=['PLY_ACA_LEGAL_ENTITY_VW','PLY_ACA_LEGAL_ENTITY_DAY_VW']
q="""SELECT OWNER,TABLE_NAME,COLUMN_NAME,DATA_TYPE,DATA_SCALE
FROM ALL_TAB_COLUMNS
WHERE OWNER='BDWH_PLY'
 AND TABLE_NAME IN ('PLY_ACA_LEGAL_ENTITY_VW','PLY_ACA_LEGAL_ENTITY_DAY_VW')
ORDER BY TABLE_NAME,COLUMN_ID"""
meta=ora.query_to_df(q)
meta.columns=[str(c).upper() for c in meta.columns]
if meta.empty:
    raise RuntimeError('Oracle ACA source views not visible; verify PRD_BDWH access')
print('\n=== ORACLE_ACA_POSSIBLE_SCORE_FIELDS ===',flush=True)
find=meta.COLUMN_NAME.astype(str).str.contains('SCORE|RISK|ASSESS|FACTOR|DSCR|WEIGHT|SLOT|RATING',case=False,regex=True,na=False)
print(meta.loc[find].to_csv(index=False,sep=';',na_rep='NULL').strip(),flush=True)
for table in source_tables:
    table_meta=meta.loc[meta.TABLE_NAME.eq(table)]
    candidates=table_meta.loc[table_meta.COLUMN_NAME.astype(str).str.contains('SCORE|ASSESS|RISK',case=False,regex=True,na=False),'COLUMN_NAME'].tolist()
    for col in candidates[:10]:
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*',col):
            continue
        sql=f'SELECT {col} FROM BDWH_PLY.{table} WHERE {col} IS NOT NULL AND ROWNUM<=5'
        try:
            samples=ora.query_to_df(sql)
            print('\nSAMPLE',table,col, samples.to_csv(index=False,sep=';').strip(),flush=True)
        except Exception as exc:
            print('SAMPLE_FAILED',table,col,type(exc).__name__,str(exc)[:130],flush=True)
print('\nNEXT: compare named candidate field values to COLUMN CP of Timo\'s original ACA extract.',flush=True)
print('Do not treat a matching name/scale as validated correspondence without ID + value comparison.',flush=True)
print('NO WRITES / NO RECIPES CHANGED.',flush=True)
