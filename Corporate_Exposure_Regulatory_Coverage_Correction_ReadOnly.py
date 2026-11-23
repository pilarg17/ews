# CORPORATE EWS | REGULATORY CROSSWALK & COVERAGE CORRECTION
# One Dataiku Python notebook cell; run AFTER the two prior QA cells, SAME kernel.
# SELECT-only. Does not modify datasets, recipes, outputs, training or backtests.
import re
import time
import numpy as np
import pandas as pd
from dataiku.core.sql import SQLExecutor2

_clock = time.monotonic()
def log(s):
    print('[%6.1fs] %s' % (time.monotonic()-_clock, s), flush=True)
def view(label, frame, n=50):
    print('\n=== '+label+' ===', flush=True)
    if frame is None or frame.empty:
        print('NO_ROWS', flush=True)
    else:
        print(frame.head(n).to_csv(index=False, sep=';', na_rep='NULL').strip(), flush=True)
def norm(v):
    if v is None or pd.isna(v): return None
    s = str(v).strip()
    if not s or s.lower() in ('null','none','nan'): return None
    if re.fullmatch(r'\d+\.0+',s): s=s.split('.')[0]
    return s
def lit(v):
    s = norm(v)
    if not s or len(s)>128 or any(ord(c)<32 for c in s):
        raise ValueError('Invalid input identifier')
    return "'"+s.replace("'","''")+"'"
def chunks(seq,n):
    seq=list(seq)
    for i in range(0,len(seq),n): yield seq[i:i+n]
def oracle_sql(label,q,maxrows=50000):
    if not re.match(r'^\s*(SELECT|WITH)\b',q,re.I):
        raise RuntimeError('SELECT queries only')
    log('QUERY | '+label)
    d=SQLExecutor2(connection='PRD_BDWH').query_to_df(q)
    d.columns=[str(c).upper() for c in d.columns]
    if len(d)>maxrows: raise RuntimeError('Unbounded result: '+label)
    return d

for name in ('loans','matched','keys','reg_day','ASOF'):
    if name not in globals():
        raise RuntimeError('Missing %s: run the FULL portfolio QA in the same notebook kernel'%name)
if not all(isinstance(globals()[name],pd.DataFrame) for name in ('loans','matched','keys')):
    raise RuntimeError('Previous QA results are not accessible')
if str(ASOF)!='2026-10-29':
    raise RuntimeError('Unexpected EWS cutoff; no comparison attempted')
if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',str(reg_day)):
    raise RuntimeError('Unexpected regulatory snapshot date')
required_loans={'ROW_ID','INSTRUMENT_ID','NOMINAL_AMOUNT','MISSING'}
required_match={'ROW_ID','SOURCE','REF_KEY','MATCHED_SOURCE','STRICT_ONE_TO_ONE',
                'FRESH','REPORT_DATEKEY','REPORT_VERSIONID','RAW_AMOUNT','CCY'}
if required_loans-set(loans) or required_match-set(matched):
    raise RuntimeError('Prior QA results are missing required columns')

log('START | Full-population regulatory crosswalk by SOURCE_CONTRACT_REFERENCE, not REG.INSTRUMENT_ID')
log('Population=%d; EWS cutoff=%s; regulatory snapshot=%s' %(len(loans),ASOF,reg_day))
refs=sorted(set(loans.INSTRUMENT_ID.dropna().map(norm)))
if len(refs)>25000: raise RuntimeError('Unexpected Corporate population size')
reg_date=reg_day
frames=[]
# Aggregate at Oracle source to avoid transferring all 1.17M versioned monthly rows.
for i,part in enumerate(chunks(refs,400),1):
    q="""
SELECT TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) REF_KEY,
       COUNT(*) N_REG_ROWS,
       COUNT(DISTINCT TRIM(TO_CHAR(INSTRUMENT_ID))) N_REG_INSTRUMENT_IDS,
       COUNT(DISTINCT TRIM(TO_CHAR(SOURCE_VERSION_ID))) N_SOURCE_VERSIONS,
       COUNT(OUTSTANDING_NOMINAL_AMOUNT) N_REG_AMOUNT_ROWS,
       COUNT(DISTINCT OUTSTANDING_NOMINAL_AMOUNT) N_REG_DISTINCT_AMOUNTS,
       SUM(CASE WHEN OUTSTANDING_NOMINAL_AMOUNT > 0 THEN 1 ELSE 0 END) N_REG_POSITIVE_ROWS,
       SUM(CASE WHEN OUTSTANDING_NOMINAL_AMOUNT = 0 THEN 1 ELSE 0 END) N_REG_ZERO_ROWS,
       SUM(CASE WHEN OUTSTANDING_NOMINAL_AMOUNT < 0 THEN 1 ELSE 0 END) N_REG_NEGATIVE_ROWS,
       MIN(OUTSTANDING_NOMINAL_AMOUNT) MIN_REG_AMOUNT,
       MAX(OUTSTANDING_NOMINAL_AMOUNT) MAX_REG_AMOUNT,
       MAX(CASE WHEN SUBSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),1,9) = 'LOANDEPO_' THEN 1 ELSE 0 END) HAS_REG_LOANDEPO,
       MAX(CASE WHEN SUBSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),1,8) = 'ACCOUNT_' THEN 1 ELSE 0 END) HAS_REG_ACCOUNT
FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_MON_VW
WHERE MAP_DTE >= DATE '%s' AND MAP_DTE < DATE '%s'+1
  AND TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) IN (%s)
GROUP BY TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE))
"""%(reg_date,reg_date,','.join(lit(x) for x in part))
    z=oracle_sql('regulatory source-reference batch %d'%i,q,1000)
    if not z.empty: frames.append(z)
    if i%5==0 or i==(len(refs)+399)//400:
        log('BATCH | %d/%d' %(i,(len(refs)+399)//400))
agg=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['REF_KEY'])
if agg.REF_KEY.duplicated().any():
    raise RuntimeError('Unexpected duplicate aggregated regulatory source reference')
agg['REF_KEY']=agg.REF_KEY.map(norm)
for c in agg.columns:
    if c!='REF_KEY': agg[c]=pd.to_numeric(agg[c],errors='coerce')
C=loans.merge(agg,left_on='INSTRUMENT_ID',right_on='REF_KEY',how='left',validate='many_to_one')
C['REG_LINK']=C.N_REG_ROWS.notna()
C['REG_AMT_ANY']=C.N_REG_AMOUNT_ROWS.fillna(0).gt(0)
C['REG_AMT_POS']=C.N_REG_POSITIVE_ROWS.fillna(0).gt(0)
C['REG_AMT_ZERO_ONLY']=(C.N_REG_ZERO_ROWS.fillna(0).gt(0) &
                       ~C.REG_AMT_POS & ~C.N_REG_NEGATIVE_ROWS.fillna(0).gt(0))
C['REG_AMT_MULTIVERSION_DIFFERENCE']=C.N_REG_DISTINCT_AMOUNTS.fillna(0).gt(1)
C['REG_LOANDEPO']=C.HAS_REG_LOANDEPO.fillna(0).eq(1)
C['REG_ACCOUNT']=C.HAS_REG_ACCOUNT.fillna(0).eq(1)
M=C[C.MISSING].copy()
K=C[~C.MISSING].copy()

view('01_CORRECTED_REGULATORY_CROSSWALK', pd.DataFrame([{
    'AS_OF':ASOF,'REG_SNAPSHOT':reg_date,'N_LOANS':len(C),'N_BASELINE_WITH_NOMINAL':len(K),
    'N_BASELINE_MISSING':len(M),
    'ALL_LOANS_WITH_REG_SOURCE_REFERENCE':int(C.REG_LINK.sum()),
    'MISSING_WITH_REG_SOURCE_REFERENCE':int(M.REG_LINK.sum()),
    'MISSING_WITH_REG_OUTSTANDING_AMOUNT':int(M.REG_AMT_ANY.sum()),
    'MISSING_WITH_ANY_POSITIVE_REG_OUTSTANDING':int(M.REG_AMT_POS.sum()),
    'MISSING_WITH_ZERO_ONLY_REG_OUTSTANDING':int(M.REG_AMT_ZERO_ONLY.sum()),
    'MISSING_WITH_MULTIPLE_REG_AMOUNTS_ACROSS_VERSIONS':int(M.REG_AMT_MULTIVERSION_DIFFERENCE.sum()),
    'MISSING_WITHOUT_REG_SOURCE_REFERENCE':int((~M.REG_LINK).sum()),
    'REG_AMOUNTS_APPROVED_FOR_REPLACEMENT':'NO'
}]))

buckets=[]
for name,condition in (
    ('REG_LOANDEPO_ONLY',M.REG_LOANDEPO&~M.REG_ACCOUNT),
    ('REG_ACCOUNT_ONLY',M.REG_ACCOUNT&~M.REG_LOANDEPO),
    ('REG_BOTH_SOURCE_TYPES',M.REG_LOANDEPO&M.REG_ACCOUNT),
    ('REG_OTHER_SOURCE_TYPE',M.REG_LINK&~M.REG_LOANDEPO&~M.REG_ACCOUNT),
    ('NO_REG_LINK',~M.REG_LINK)):
    p=M[condition]
    buckets.append({'REG_SOURCE_CLASS':name,'N_MISSING':len(p),
        'N_REG_POSITIVE_ANY_VERSION':int(p.REG_AMT_POS.sum()),
        'N_REG_MULTI_AMOUNT':int(p.REG_AMT_MULTIVERSION_DIFFERENCE.sum())})
view('02_MISSING_LOANS_BY_REGULATORY_SOURCE_TYPE',pd.DataFrame(buckets))

# Reconcile the regulatory version and amount with the actual RFO reporting version.
# Only use observed RFO records for missing loans, with the SAME 30-Sep reporting date.
miss_ids=set(M.ROW_ID)
R=matched[matched.ROW_ID.isin(miss_ids)].copy()
R=R[(R.MATCHED_SOURCE==True)&(R.STRICT_ONE_TO_ONE==True)&(R.FRESH==True)]
R=R[pd.to_numeric(R.REPORT_DATEKEY,errors='coerce').eq(int(reg_date.replace('-','')))]
R=R[R.SOURCE.isin(['LOANDEPO_VW','ACCOUNT_VW'])].drop_duplicates(['ROW_ID','SOURCE','REF_KEY'])
if R.empty:
    view('03_SAME_DATE_RFO_REGULATORY_VERSION_RECONCILIATION',pd.DataFrame())
else:
    rkeys=sorted(set(R.REF_KEY.dropna().map(norm)))
    vframes=[]
    for i,part in enumerate(chunks(rkeys,250),1):
        q="""
SELECT TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) REF_KEY,
       TRIM(TO_CHAR(SOURCE_VERSION_ID)) REG_VERSION,
       CASE WHEN SUBSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),1,9) = 'LOANDEPO_' THEN 'LOANDEPO_VW'
            WHEN SUBSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),1,8) = 'ACCOUNT_' THEN 'ACCOUNT_VW'
            ELSE 'OTHER' END REG_SOURCE,
       COUNT(*) N_ROWS_VERSION,
       COUNT(OUTSTANDING_NOMINAL_AMOUNT) N_AMOUNT_ROWS_VERSION,
       COUNT(DISTINCT OUTSTANDING_NOMINAL_AMOUNT) N_AMOUNTS_VERSION,
       MIN(OUTSTANDING_NOMINAL_AMOUNT) MIN_AMOUNT_VERSION,
       MAX(OUTSTANDING_NOMINAL_AMOUNT) MAX_AMOUNT_VERSION
FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_MON_VW
WHERE MAP_DTE >= DATE '%s' AND MAP_DTE < DATE '%s'+1
  AND TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) IN (%s)
GROUP BY TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)), TRIM(TO_CHAR(SOURCE_VERSION_ID)),
         CASE WHEN SUBSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),1,9) = 'LOANDEPO_' THEN 'LOANDEPO_VW'
              WHEN SUBSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),1,8) = 'ACCOUNT_' THEN 'ACCOUNT_VW'
              ELSE 'OTHER' END
"""%(reg_date,reg_date,','.join(lit(x) for x in part))
        z=oracle_sql('regulatory versioned reference batch %d'%i,q,20000)
        if not z.empty: vframes.append(z)
        if i%5==0 or i==(len(rkeys)+249)//250:
            log('VERSION | %d/%d'%(i,(len(rkeys)+249)//250))
    V=pd.concat(vframes,ignore_index=True) if vframes else pd.DataFrame(
        columns=['REF_KEY','REG_VERSION','REG_SOURCE','N_AMOUNT_ROWS_VERSION',
                 'N_AMOUNTS_VERSION','MIN_AMOUNT_VERSION','MAX_AMOUNT_VERSION'])
    if not V.empty and V.duplicated(['REF_KEY','REG_VERSION','REG_SOURCE']).any():
        raise RuntimeError('Duplicate grouped regulatory rows by source/version')
    V['REG_VERSION']=V.REG_VERSION.map(norm)
    R['RFO_VERSION']=R.REPORT_VERSIONID.map(norm)
    R['REF_KEY']=R.REF_KEY.map(norm)
    R=R.merge(V,left_on=['REF_KEY','SOURCE','RFO_VERSION'],
              right_on=['REF_KEY','REG_SOURCE','REG_VERSION'],how='left',validate='many_to_one')
    R['SAME_VERSION']=R.REG_VERSION.notna()
    R['SINGLE_REG_AMOUNT']=R.N_AMOUNTS_VERSION.fillna(0).eq(1)
    R['SOURCE_CCY_EUR']=R.CCY.eq('EUR')
    R['RFO_AMOUNT_COMPARABLE']=np.where(R.SOURCE.eq('ACCOUNT_VW')&R.RAW_AMOUNT.lt(0),
                                         -R.RAW_AMOUNT,R.RAW_AMOUNT)
    R['AMOUNT_AGREES_EUR']=(R.SAME_VERSION & R.SINGLE_REG_AMOUNT & R.SOURCE_CCY_EUR &
        (R.RFO_AMOUNT_COMPARABLE-R.MIN_AMOUNT_VERSION).abs().le(1))
    compare=[]
    for source,g in R.groupby('SOURCE'):
        compare.append({'SOURCE':source,'N_MISSING_WITH_RFO_SOURCE_DATE_AND_UNIQUE_LINK':g.ROW_ID.nunique(),
            'N_SAME_REGULATORY_SOURCE_VERSION':int(g.SAME_VERSION.sum()),
            'N_VERSION_WITH_SINGLE_REG_AMOUNT':int((g.SAME_VERSION&g.SINGLE_REG_AMOUNT).sum()),
            'N_REG_AMOUNT_AGREES_WITH_RFO_EUR_ABS_1':int(g.AMOUNT_AGREES_EUR.sum()),
            'N_VERSION_MISSING_OR_AMBIGUOUS':int((~g.SAME_VERSION|~g.SINGLE_REG_AMOUNT).sum())})
    view('03_SAME_DATE_RFO_REGULATORY_VERSION_RECONCILIATION',pd.DataFrame(compare))
    targets=['2110000000049295','2110000091015150']
    det=R[R.REF_KEY.isin(targets)][['ROW_ID','SOURCE','REF_KEY','REPORT_DATEKEY',
         'RFO_VERSION','RAW_AMOUNT','CCY','REG_VERSION','N_AMOUNTS_VERSION',
         'MIN_AMOUNT_VERSION','MAX_AMOUNT_VERSION','SAME_VERSION','AMOUNT_AGREES_EUR']]
    view('04_TWO_SOURCE_OWNER_REFERENCES_VERSION_QA',det,20)

# This is information availability only; do not add to the numerator.
view('05_COVERAGE_INTERPRETATION',pd.DataFrame([{
    'EXISTING_COVERAGE_PCT':round(100*len(K)/len(C),2),
    'EXISTING_KNOWN':len(K),'EXISTING_MISSING':len(M),
    'NEWLY_APPROVED_EXPOSURES':0,
    'FINAL_APPROVED_COVERAGE_PCT':round(100*len(K)/len(C),2),
    'WHY_NOT_APPROVED':'Regulatory row may have multiple versions / different exposure semantics; overdrafts need business classification and date/FX QA'
}]))
print('\nREAD ONLY | NO RECIPE, DATASET, TRAINING, MODEL OR BACKTEST MODIFIED',flush=True)
