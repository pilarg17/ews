# -*- coding: utf-8 -*-
"""CORPORATE EWS | VERSION SEMANTICS AND AMOUNT RECONCILIATION
One Dataiku Python notebook cell, SAME KERNEL after regulatory resume QA.
READ ONLY: comparisons of already loaded R, V, M, C; optional bounded source-type sample.
No recipe, dataset, training, model or backtest modifications.
This is an exploratory cross-version comparison, not source substitution approval.
"""
import re
import numpy as np
import pandas as pd
from dataiku.core.sql import SQLExecutor2

for var in ('R','V','M','C','ASOF','reg_day'):
    if var not in globals():
        raise RuntimeError('Missing '+var+'; execute after regulatory resume in SAME notebook kernel')
if str(ASOF)!='2026-10-29' or str(reg_day)!='2026-09-30':
    raise RuntimeError('Unexpected date; no comparison executed')

def norm_qa(value):
    if value is None or pd.isna(value): return None
    value=str(value).strip()
    if not value or value.lower() in ('none','null','nan','<na>'): return None
    if re.fullmatch(r'\d+\.0+',value): value=value.split('.')[0]
    return value

def report_qa(name,df,max_rows=40):
    print('\n=== '+name+' ===',flush=True)
    print('NO_ROWS' if df.empty else df.head(max_rows).to_csv(index=False,sep=';',na_rep='NULL').strip(),flush=True)

r=R.copy(); v=V.copy()
required_r={'ROW_ID','SOURCE','REF_KEY','REPORT_VERSIONID','RAW_AMOUNT','CCY','REPORT_DATEKEY'}
required_v={'REF_KEY','REG_SOURCE','REG_VERSION','N_AMOUNTS_VERSION','N_AMOUNT_ROWS_VERSION','MIN_AMOUNT_VERSION','MAX_AMOUNT_VERSION'}
if required_r-set(r.columns) or required_v-set(v.columns):
    raise RuntimeError('Unexpected in-memory frame schema; cannot safely reconcile')
for col in ('REF_KEY','SOURCE'):
    r[col]=r[col].map(norm_qa).astype('string')
for col in ('REF_KEY','REG_SOURCE','REG_VERSION'):
    v[col]=v[col].map(norm_qa).astype('string')
r['RFO_VERSION']=r.REPORT_VERSIONID.map(norm_qa).astype('string')
r['RAW_AMOUNT']=pd.to_numeric(r.RAW_AMOUNT,errors='coerce')
r['REPORT_DATEKEY']=pd.to_numeric(r.REPORT_DATEKEY,errors='coerce')
for col in ('N_AMOUNTS_VERSION','N_AMOUNT_ROWS_VERSION','MIN_AMOUNT_VERSION','MAX_AMOUNT_VERSION'):
    v[col]=pd.to_numeric(v[col],errors='coerce')

# Reconcile on the *source reference* and product family, intentionally WITHOUT
# equating SOURCE_VERSION_ID to REPORT_VERSIONID. Compare version IDs separately.
if r.duplicated(['ROW_ID','SOURCE','REF_KEY']).any():
    raise RuntimeError('Multiple RFO candidates per loan/source/ref; stop to avoid inflation')
if v.duplicated(['REF_KEY','REG_SOURCE','REG_VERSION']).any():
    raise RuntimeError('Repeated regulatory source/version keys; cannot uniquely assess')
q=r.merge(v,left_on=['REF_KEY','SOURCE'],right_on=['REF_KEY','REG_SOURCE'],how='left',validate='one_to_many',indicator='_LINK')
q['HAS_REG_VERSION']=q._LINK.eq('both')
q['SAME_VERSION']=q.HAS_REG_VERSION & q.RFO_VERSION.eq(q.REG_VERSION).fillna(False)
q['REG_SINGLE_AMOUNT']=q.N_AMOUNTS_VERSION.eq(1) & q.N_AMOUNT_ROWS_VERSION.gt(0)
q['RFO_CCY_EUR']=q.CCY.astype('string').str.upper().eq('EUR').fillna(False)
q['OVERDRAFT_CANDIDATE']=q.SOURCE.eq('ACCOUNT_VW') & q.RAW_AMOUNT.lt(0) & q.RFO_CCY_EUR
q['COMPARE_AMOUNT']=np.where(q.OVERDRAFT_CANDIDATE,-q.RAW_AMOUNT,q.RAW_AMOUNT)
q['ABS_DIFF_EUR']=(q.COMPARE_AMOUNT-q.MIN_AMOUNT_VERSION).abs()
q['MATCH_AMOUNT_ANY_VERSION']=(q.HAS_REG_VERSION & q.REG_SINGLE_AMOUNT & q.RFO_CCY_EUR & q.ABS_DIFF_EUR.le(1))
q['MATCH_AMOUNT_AND_VERSION']=q.MATCH_AMOUNT_ANY_VERSION & q.SAME_VERSION
q['LOANDEPO_POSITIVE_EUR']=(q.SOURCE.eq('LOANDEPO_VW') & q.RAW_AMOUNT.gt(0) & q.RFO_CCY_EUR)
# Per loan and source, avoid counting its many regulatory versions as extra loans.
agg=q.groupby(['ROW_ID','SOURCE','REF_KEY'],dropna=False).agg(
    N_REG_VERSIONS=('HAS_REG_VERSION','sum'),
    N_VERSION_EXACT=('SAME_VERSION','sum'),
    N_SINGLE_AMOUNT_VERSIONS=('REG_SINGLE_AMOUNT','sum'),
    N_AMOUNT_AGREES_ANY_VERSION=('MATCH_AMOUNT_ANY_VERSION','sum'),
    N_AMOUNT_AND_VERSION_AGREES=('MATCH_AMOUNT_AND_VERSION','sum'),
    MIN_ABS_DIFF_EUR=('ABS_DIFF_EUR','min'),
    RFO_VERSION=('RFO_VERSION','first'),
    RFO_AMOUNT=('RAW_AMOUNT','first'),
    CCY=('CCY','first'),
    OVERDRAFT_CANDIDATE=('OVERDRAFT_CANDIDATE','max'),
    LOANDEPO_POSITIVE_EUR=('LOANDEPO_POSITIVE_EUR','max'),
).reset_index()
agg['REG_FOUND']=agg.N_REG_VERSIONS.gt(0)
agg['ANY_VERSION_MATCH']=agg.N_VERSION_EXACT.gt(0)
agg['ANY_AMOUNT_MATCH']=agg.N_AMOUNT_AGREES_ANY_VERSION.gt(0)
agg['SAME_VERSION_AND_AMOUNT']=agg.N_AMOUNT_AND_VERSION_AGREES.gt(0)
agg['MULTI_VERSION']=agg.N_REG_VERSIONS.gt(1)
report_qa('01_CROSS_VERSION_AMOUNT_RECONCILIATION',pd.DataFrame([
    {'SOURCE':src,
     'N_LOANS_RFO':len(g),
     'N_WITH_REGULATORY_ROWS_SAME_DATE':int(g.REG_FOUND.sum()),
     'N_WITH_MULTIPLE_REG_VERSIONS':int(g.MULTI_VERSION.sum()),
     'N_WITH_EXACT_VERSION_ID':int(g.ANY_VERSION_MATCH.sum()),
     'N_WITH_AMOUNT_MATCH_ANY_VERSION_EUR':int(g.ANY_AMOUNT_MATCH.sum()),
     'N_WITH_VERSION_AND_AMOUNT_MATCH_EUR':int(g.SAME_VERSION_AND_AMOUNT.sum()),
     'N_POSITIVE_LOANDEPO_EUR_AND_ANY_AMOUNT_MATCH':int((g.LOANDEPO_POSITIVE_EUR & g.ANY_AMOUNT_MATCH).sum()),
     'N_ACCOUNT_OVERDRAFT_AND_ANY_AMOUNT_MATCH':int((g.OVERDRAFT_CANDIDATE & g.ANY_AMOUNT_MATCH).sum()),
     'N_SOURCE_SWITCH_APPROVED':0}
    for src,g in agg.groupby('SOURCE')]))

# Source version profiles: show what is actually represented, without guessing
# about equivalence of the two version identifiers.
profiles=[]
for src,g in agg.groupby('SOURCE'):
    rs=r[r.SOURCE.eq(src)]
    vs=v[v.REG_SOURCE.eq(src)]
    nums_r=pd.to_numeric(rs.RFO_VERSION,errors='coerce')
    nums_v=pd.to_numeric(vs.REG_VERSION,errors='coerce')
    profiles.append({'SOURCE':src,'N_RFO_VERSION_IDS':rs.RFO_VERSION.nunique(),
       'RFO_VERSION_MIN':nums_r.min(),'RFO_VERSION_MAX':nums_r.max(),
       'N_REG_VERSION_IDS':vs.REG_VERSION.nunique(),
       'REG_VERSION_MIN':nums_v.min(),'REG_VERSION_MAX':nums_v.max(),
       'COMMON_VERSION_IDS_GLOBAL':len(set(rs.RFO_VERSION.dropna()) & set(vs.REG_VERSION.dropna())),
       'N_REF_PAIRS_WITH_EXACT_VERSION_ID':int(g.ANY_VERSION_MATCH.sum())})
report_qa('02_VERSION_ID_PROFILES',pd.DataFrame(profiles))

watch={'2110000000049295','2110000091015150'}
w=q[q.REF_KEY.isin(watch)].copy()
w['REG_VERSION_NUMBER']=pd.to_numeric(w.REG_VERSION,errors='coerce')
report_qa('03_TWO_EXAMPLES_RFO_VS_REG_AMOUNTS',
    w.sort_values(['REF_KEY','REG_VERSION_NUMBER'],ascending=[True,False])[
      ['SOURCE','REF_KEY','RFO_VERSION','REG_VERSION','RAW_AMOUNT',
       'COMPARE_AMOUNT','MIN_AMOUNT_VERSION','MAX_AMOUNT_VERSION',
       'N_AMOUNTS_VERSION','SAME_VERSION','MATCH_AMOUNT_ANY_VERSION','ABS_DIFF_EUR']
    ],35)

m=M.copy()
report_qa('04_FULL_MISSING_COHORT_REGULATORY_AVAILABILITY',pd.DataFrame([{
    'N_BASELINE_MISSING':len(m),
    'N_WITH_REG_LINK':int(m.REG_LINK.sum()),
    'N_WITH_POSITIVE_REG_AMOUNT_ANY_VERSION':int(m.REG_AMT_POS.sum()),
    'N_WITH_AMBIGUOUS_REG_AMOUNTS_ACROSS_VERSIONS':int(m.REG_AMT_MULTIVERSION_DIFFERENCE.sum()),
    'N_REG_OTHER_ON_OR_FACILITY':int((m.REG_LINK & ~m.REG_LOANDEPO & ~m.REG_ACCOUNT).sum()),
    'N_NO_REG_LINK':int((~m.REG_LINK).sum()),
    'N_APPROVED_REPLACEMENTS':0,
    'STATUS':'Availability only; no source/amount sign-off'
}]))

# A bounded source-name inspection of the large ON/FACILITY group, not a full scan.
others=m[m.REG_LINK & ~m.REG_LOANDEPO & ~m.REG_ACCOUNT]
if not others.empty:
    sample=sorted(set(others.INSTRUMENT_ID.dropna().map(norm_qa)))[:24]
    if sample:
        esc=lambda x:"'"+x.replace("'","''")+"'"
        check=("SELECT TRIM(TO_CHAR(INSTRUMENT_ID)) REG_INSTRUMENT_ID, "
               "TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) SOURCE_REF, "
               "TRIM(TO_CHAR(SOURCE_TABLE_NAME)) SOURCE_TABLE_NAME, "
               "TRIM(TO_CHAR(SOURCE_VERSION_ID)) SOURCE_VERSION_ID, "
               "TRIM(TO_CHAR(LAST_VERSION_FLAG)) LAST_VERSION_FLAG, "
               "OUTSTANDING_NOMINAL_AMOUNT REG_AMOUNT "
               "FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_MON_VW "
               "WHERE MAP_DTE >= DATE '2026-09-30' AND MAP_DTE < DATE '2026-10-01' "
               "AND TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) IN ("+','.join(esc(x) for x in sample)+") "
               "AND ROWNUM <= 120")
        extra=SQLExecutor2(connection='PRD_BDWH').query_to_df(check)
        extra.columns=[str(col).upper() for col in extra.columns]
        if not extra.empty:
            extra['REG_PREFIX']=extra.REG_INSTRUMENT_ID.map(
                lambda x: norm_qa(x).split('_',1)[0] if norm_qa(x) and '_' in norm_qa(x) else 'NO_PREFIX')
            report_qa('05_ON_FACILITY_SOURCE_SAMPLE',extra[
                ['REG_PREFIX','REG_INSTRUMENT_ID','SOURCE_REF','SOURCE_TABLE_NAME','SOURCE_VERSION_ID','LAST_VERSION_FLAG','REG_AMOUNT']],30)

print('\nCONCLUSION | Exact version IDs are not currently a validated cross-source join.')
print('CONCLUSION | An amount match across versions is diagnostic evidence, NOT approval.')
print('CONCLUSION | Need authoritative mapping of RFO REPORT_VERSIONID to regulatory SOURCE_VERSION_ID and source classification.')
print('READ ONLY | NO DATASETS, RECIPES, TRAINING, MODELS OR BACKTESTS MODIFIED.',flush=True)
