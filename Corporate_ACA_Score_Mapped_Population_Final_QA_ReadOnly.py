# -*- coding: utf-8 -*-
"""Corporate EWS | ACA score, exact-key population QA | READ ONLY.
Paste this entire file into ONE Dataiku Python notebook cell.
Reads the current Corporate ACA staging and Oracle source views only.
No writes, recipe edits, data refreshes, or changes to previous results.
Output contains only aggregate counts; entity and customer IDs are not printed.
"""
import re
from collections import defaultdict
import pandas as pd
import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
DATE = '2026-10-29'
BATCH = 120
ex = SQLExecutor2(connection='PRD_BDWH')

def clean(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if re.fullmatch(r'\d+\.0+', s): s = s.split('.')[0]
    return s if s and s.upper() not in ('NONE','NULL','NAN','<NA>') else None

def lit(s):
    s = clean(s)
    if s is None or not re.fullmatch(r'[A-Za-z0-9_.\-]{1,160}', s):
        raise ValueError('Unexpected source identifier; stopping, not altering IDs')
    return "'" + s + "'"

def chunks(x):
    v = sorted(set(x))
    for i in range(0, len(v), BATCH):
        yield v[i:i+BATCH]

def read(label, sql):
    if not re.match(r'^\s*SELECT\b', sql, re.I):
        raise RuntimeError('Only SELECT queries permitted')
    print('QUERY | ' + label, flush=True)
    d = ex.query_to_df(sql)
    d.columns = [str(c).upper() for c in d.columns]
    return d

def show(label, frame):
    print('\n=== ' + label + ' ===', flush=True)
    print(frame.to_csv(index=False, sep=';', na_rep='NULL').strip(), flush=True)

print('READ ONLY | ACA SCORE IN EXACTLY MAPPED CORPORATE ENTITIES | START', flush=True)
st = dataiku.Dataset('ACA_slotting_entity_day', project_key=PROJECT).get_dataframe(
    columns=['ENTITY_ID', 'MAP_DTE', 'ACA_FINAL_RATING_BEST', 'ACA_FINAL_RATING_WORST'])
st['ENTITY_ID'] = st['ENTITY_ID'].map(clean)
st['MAP_DTE'] = pd.to_datetime(st['MAP_DTE'], errors='coerce').dt.normalize()
ids = sorted(set(st['ENTITY_ID'].dropna()))
assert len(ids) == 233, 'Expected 233 distinct staging entities, stop if scope changed'
if st.duplicated(['ENTITY_ID', 'MAP_DTE']).any():
    raise RuntimeError('Unexpected staging entity/day duplicates')
st_day = st.loc[st['MAP_DTE'].eq(pd.Timestamp(DATE))].copy()

# The authoritative bank-side source identified in the previous QA.
bridge=[]
for i, b in enumerate(chunks(ids), 1):
    sql = ("SELECT DISTINCT 'K'||TRIM(TO_CHAR(ENTITY_ID)) ENTITY_LOCKED, "
           "'K'||TRIM(TO_CHAR(CUSTOMER_ID)) CUSTOMER_LOCKED "
           "FROM BDL_CREDIT_INSIGHT_PLY.ENTITY_CUSTOMER_MON_VW "
           "WHERE CUSTOMER_ID IS NOT NULL AND MAP_DTE < DATE '2026-10-30' "
           "AND TRIM(TO_CHAR(ENTITY_ID)) IN (" + ','.join(map(lit,b)) + ')')
    bridge.append(read('BANK ENTITY-CUSTOMER BRIDGE '+str(i),sql))
br = pd.concat(bridge, ignore_index=True).drop_duplicates()
br['ENTITY_ID'] = br['ENTITY_LOCKED'].astype(str).map(lambda z: clean(z[1:]) if z.startswith('K') else None)
br['ACA_REF'] = br['CUSTOMER_LOCKED'].astype(str).map(lambda z: clean(z[1:]) if z.startswith('K') else None)
br = br.dropna(subset=['ENTITY_ID', 'ACA_REF'])[['ENTITY_ID','ACA_REF']].drop_duplicates()
by_entity = br.groupby('ENTITY_ID')['ACA_REF'].nunique()
by_ref = br.groupby('ACA_REF')['ENTITY_ID'].nunique()
show('01_VERIFIED_BRIDGE_CARDINALITY', pd.DataFrame([{
    'STAGING_ENTITIES':len(ids),
    'ENTITIES_WITH_BANK_BRIDGE':int(br['ENTITY_ID'].nunique()),
    'ENTITIES_MULTIPLE_CUSTOMER_KEYS':int((by_entity>1).sum()),
    'CUSTOMER_KEYS_SHARED_BY_ENTITIES':int((by_ref>1).sum()),
    'STAGING_ENTITIES_ON_REFERENCE_DATE':int(st_day['ENTITY_ID'].nunique())
}]))
if len(br['ENTITY_ID'].unique()) != len(ids) or (by_entity>1).any() or (by_ref>1).any():
    raise RuntimeError('Bank bridge is not one-to-one for all 233 entities. STOP, no fabricated score mapping.')

# Pull actual 29-Oct ACA score/rating rows ONLY for those official customer references.
src=[]
for i,b in enumerate(chunks(br['ACA_REF'].unique()),1):
    sql=("SELECT 'K'||TRIM(TO_CHAR(LEGAL_ENTITY_REF_CODE)) ACA_KEY_LOCKED, "
         "TRIM(TO_CHAR(FINAL_RATING)) FINAL_RATING, "
         "RISKASSESSMENT_SCORE SCORE, "
         "'P'||TRIM(TO_CHAR(PROJECT_NUMBER)) PROJECT_LOCKED, "
         "'M'||TRIM(TO_CHAR(MODEL_ID)) MODEL_LOCKED "
         "FROM BDWH_PLY.PLY_ACA_LEGAL_ENTITY_DAY_VW "
         "WHERE MAP_DTE >= DATE '2026-10-29' AND MAP_DTE < DATE '2026-10-30' "
         "AND TRIM(TO_CHAR(LEGAL_ENTITY_REF_CODE)) IN ("+','.join(map(lit,b))+')')
    src.append(read('ACA SCORE AS-OF BATCH '+str(i), sql))
a = pd.concat(src,ignore_index=True) if src else pd.DataFrame()
if a.empty: raise RuntimeError('No ACA score rows for matched refs. Stop, no approved integration')
a['ACA_REF'] = a['ACA_KEY_LOCKED'].astype(str).map(lambda z:clean(z[1:]) if z.startswith('K') else None)
a['SCORE'] = pd.to_numeric(a['SCORE'],errors='coerce')
a['FINAL_RATING_NUM'] = pd.to_numeric(a['FINAL_RATING'],errors='coerce')
a = a.merge(br,on='ACA_REF',how='inner',validate='many_to_one')

p=a.groupby('ENTITY_ID',dropna=False).agg(
    N_SOURCE_ROWS=('SCORE','size'),
    N_DISTINCT_SCORES=('SCORE',lambda x:x.dropna().nunique()),
    N_DISTINCT_RATINGS=('FINAL_RATING',lambda x:x.dropna().nunique()),
    N_DISTINCT_PROJECTS=('PROJECT_LOCKED',lambda x:x.dropna().nunique()),
    SCORE_MIN=('SCORE','min'), SCORE_MAX=('SCORE','max'),
    SOURCE_RATING_MIN=('FINAL_RATING_NUM','min'),SOURCE_RATING_MAX=('FINAL_RATING_NUM','max')
).reset_index()
show('02_ACA_MAPPED_SCORE_COVERAGE', pd.DataFrame([{
    'ENTITIES_EXPECTED':len(ids),
    'ENTITIES_WITH_ASOF_ACA_ROWS':int(p.ENTITY_ID.nunique()),
    'ENTITIES_WITH_NUMERIC_SCORE':int(p.SCORE_MAX.notna().sum()),
    'SOURCE_ROWS':len(a),
    'SOURCE_ROWS_WITH_NULL_SCORE':int(a.SCORE.isna().sum()),
    'SOURCE_ROWS_WITH_SCORE_GT_10':int((a.SCORE>10).sum()),
    'SOURCE_ROWS_WITH_SCORE_LE_0':int((a.SCORE<=0).sum()),
    'ENTITIES_WITH_MULTIPLE_SCORES_SAME_DAY':int((p.N_DISTINCT_SCORES>1).sum()),
    'ENTITIES_WITH_MULTIPLE_FINAL_RATINGS_SAME_DAY':int((p.N_DISTINCT_RATINGS>1).sum()),
    'ENTITIES_WITH_MULTIPLE_PROJECTS_SAME_DAY':int((p.N_DISTINCT_PROJECTS>1).sum()),
    'MIN_SCORE_IN_MAPPED_POPULATION':a.SCORE.min(),
    'MAX_SCORE_IN_MAPPED_POPULATION':a.SCORE.max()
}]))

# Independent grade QA against existing staging only where the exact date exists.
grade=st_day[['ENTITY_ID','ACA_FINAL_RATING_BEST','ACA_FINAL_RATING_WORST']].merge(
    p[['ENTITY_ID','SOURCE_RATING_MIN','SOURCE_RATING_MAX']],on='ENTITY_ID',how='inner',validate='one_to_one')
for c in ('ACA_FINAL_RATING_BEST','ACA_FINAL_RATING_WORST'):
    grade[c] = pd.to_numeric(grade[c], errors='coerce')
show('03_FINAL_RATING_VS_EXISTING_STAGING_ON_SAME_DATE', pd.DataFrame([{
    'STAGING_ENTITIES_ASOF':int(st_day.ENTITY_ID.nunique()),
    'JOINED_ENTITIES_SAME_DATE':len(grade),
    'MATCH_STAGING_BEST_TO_SOURCE_MIN':int(grade['ACA_FINAL_RATING_BEST'].eq(grade.SOURCE_RATING_MIN).sum()),
    'MATCH_STAGING_WORST_TO_SOURCE_MAX':int(grade['ACA_FINAL_RATING_WORST'].eq(grade.SOURCE_RATING_MAX).sum()),
    'SOURCE_RATINGS_NUMERIC_EXPECTED_1_TO_5':bool(a['FINAL_RATING_NUM'].dropna().between(1,5).all())
}]))

print('\n=== 04_NEXT_DECISION ===', flush=True)
if int((a.SCORE>10).sum()) > 0:
    print('SCORE OUTLIER INSIDE MAPPED COHORT: inspect its source semantics before selecting a score.',flush=True)
else:
    print('THE PREVIOUS OUTLIER DOES NOT APPEAR IN THE MAPPED 233-ENTITY SNAPSHOT.',flush=True)
if (p.N_DISTINCT_SCORES>1).any():
    print('MULTIPLE SCORES PER ENTITY-DAY: define a business rule for projects/models; do not take arbitrary min/max.',flush=True)
else:
    print('ONE SCORE PER MAPPED ENTITY-DAY IN THIS SNAPSHOT, subject to source validation.',flush=True)
print('BANK BRIDGE KEY MEANING + EXCEL COLUMN CP COMPARISON STILL REQUIRE CONFIRMATION.',flush=True)
print('READ ONLY | NO RECIPES, DATASETS, BACKTESTS OR RESULTS MODIFIED.',flush=True)
