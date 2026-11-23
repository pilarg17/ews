# CORPORATE EWS | RESUME REGULATORY/RFO VERSION QA
# Paste as ONE NEW Dataiku Python notebook cell immediately after the failed cell.
# SAME KERNEL. ALL read-only. Does not rerun the previous 33+6 Oracle batches.

import re
import pandas as pd
import numpy as np
from dataiku.core.sql import SQLExecutor2

for name in ('C','M','R','V','reg_day','ASOF'):
    if name not in globals():
        raise RuntimeError('Missing '+name+'. Keep the same notebook kernel; the previous query state is required.')
if str(ASOF) != '2026-10-29' or str(reg_day) != '2026-09-30':
    raise RuntimeError('Unexpected cutoff / regulatory source date.')

def n(v):
    if v is None or pd.isna(v): return None
    s = str(v).strip()
    if not s or s.lower() in ('nan','null','none','<na>'): return None
    if re.fullmatch(r'\d+\.0+',s): s=s.split('.')[0]
    return s

def show(label, frame, top=60):
    print('\n=== '+label+' ===',flush=True)
    if frame is None or frame.empty: print('NO_ROWS',flush=True)
    else: print(frame.head(top).to_csv(index=False,sep=';',na_rep='NULL').strip(),flush=True)

# Sanitize BOTH SIDES of every merge key. Oracle's SQL driver may load
# numeric-looking SOURCE_CONTRACT_REFERENCE as an int64.
left=R.copy()
right=V.copy()
for f in ('REF_KEY','SOURCE','RFO_VERSION'):
    if f not in left.columns: raise RuntimeError('R missing '+f)
    left[f]=left[f].map(n).astype('string')
for f in ('REF_KEY','REG_SOURCE','REG_VERSION'):
    if f not in right.columns: raise RuntimeError('V missing '+f)
    right[f]=right[f].map(n).astype('string')

# Verify the right-hand keys are unique: no silent aggregation or arbitrary dedup.
reg_k=['REF_KEY','REG_SOURCE','REG_VERSION']
if right[reg_k].isna().any().any():
    raise RuntimeError('Null grouped regulatory reference/source/version. Investigate the version data.')
if right.duplicated(reg_k).any():
    show('DUPLICATE_REG_SOURCE_VERSION_KEYS',right[right.duplicated(reg_k,keep=False)])
    raise RuntimeError('Multiple groups have identical reference/source/version; cannot match 1-to-1.')

# Compare against the pre-existing, version-selected RFO candidates, not
# the entire RFO history or a newest version from a different report date.
for f in ('RAW_AMOUNT','REPORT_DATEKEY','N_LOANS_FOR_REF'):
    if f in left: left[f]=pd.to_numeric(left[f],errors='coerce')
for f in ('N_AMOUNTS_VERSION','N_AMOUNT_ROWS_VERSION','MIN_AMOUNT_VERSION','MAX_AMOUNT_VERSION'):
    if f in right: right[f]=pd.to_numeric(right[f],errors='coerce')

Q=left.merge(right,left_on=['REF_KEY','SOURCE','RFO_VERSION'],
             right_on=['REF_KEY','REG_SOURCE','REG_VERSION'],
             how='left',validate='many_to_one',indicator='_REG_MERGE')
Q['SAME_VERSION']=Q['_REG_MERGE'].eq('both')
Q['SINGLE_REG_AMOUNT']=(Q.N_AMOUNTS_VERSION.eq(1) & Q.N_AMOUNT_ROWS_VERSION.gt(0))
Q['EUR']=Q.CCY.astype('string').str.upper().eq('EUR').fillna(False)
Q['RFO_AMOUNT_COMPARABLE']=np.where(
    Q.SOURCE.eq('ACCOUNT_VW') & pd.to_numeric(Q.RAW_AMOUNT,errors='coerce').lt(0),
    -pd.to_numeric(Q.RAW_AMOUNT,errors='coerce'),
    pd.to_numeric(Q.RAW_AMOUNT,errors='coerce'))
Q['AMOUNT_AGREES_EUR']=(Q.SAME_VERSION & Q.SINGLE_REG_AMOUNT & Q.EUR &
    (Q.RFO_AMOUNT_COMPARABLE-Q.MIN_AMOUNT_VERSION).abs().le(1))
Q['RAW_RFO_POSITIVE']=pd.to_numeric(Q.RAW_AMOUNT,errors='coerce').gt(0)
Q['OVERDRAFT_CANDIDATE']=(Q.SOURCE.eq('ACCOUNT_VW') &
    pd.to_numeric(Q.RAW_AMOUNT,errors='coerce').lt(0) & Q.EUR)

rows=[]
for source,g in Q.groupby('SOURCE'):
    rows.append({
        'SOURCE':source,
        'N_MISSING_LOANS_WITH_RFO_AT_20260930':g.ROW_ID.nunique(),
        'N_SAME_SOURCE_VERSION':g.loc[g.SAME_VERSION,'ROW_ID'].nunique(),
        'N_SAME_VERSION_SINGLE_REG_AMOUNT':g.loc[g.SAME_VERSION & g.SINGLE_REG_AMOUNT,'ROW_ID'].nunique(),
        'N_SAME_VERSION_AMOUNT_MATCH_EUR':g.loc[g.AMOUNT_AGREES_EUR,'ROW_ID'].nunique(),
        'N_ACCOUNT_NEGATIVE_EUR_REQUIRING_CLASSIFICATION':g.loc[g.OVERDRAFT_CANDIDATE,'ROW_ID'].nunique(),
        'N_VERSION_NOT_MATCHED_OR_REG_AMOUNT_AMBIGUOUS':g.loc[~g.SAME_VERSION | ~g.SINGLE_REG_AMOUNT,'ROW_ID'].nunique(),
        'N_RFO_REFERENCE_PAIRS':len(g),
        'N_REPLACEMENTS_APPROVED':0
    })
show('03_SAME_DATE_RFO_REGULATORY_VERSION_RECONCILIATION',pd.DataFrame(rows))

watch=['2110000000049295','2110000091015150']
columns=['ROW_ID','SOURCE','REF_KEY','REPORT_DATEKEY','RFO_VERSION','RAW_AMOUNT','CCY',
         'REG_VERSION','N_AMOUNTS_VERSION','MIN_AMOUNT_VERSION','MAX_AMOUNT_VERSION',
         'SAME_VERSION','AMOUNT_AGREES_EUR','OVERDRAFT_CANDIDATE']
show('04_TWO_SOURCE_OWNER_REFERENCES_VERSION_QA',Q[Q.REF_KEY.isin(watch)][columns],30)

# Complement: the regulatory 'OTHER' group is large. Inspect observed prefixes
# only if necessary; this prints a bounded diagnostic without changing coverage.
other=M[M.REG_LINK & ~M.REG_LOANDEPO & ~M.REG_ACCOUNT]
show('05_COVERAGE_INTERPRETATION',pd.DataFrame([{
    'AS_OF':ASOF,
    'LOANS_TOTAL':len(C),
    'BASELINE_KNOWN':int((~C.MISSING).sum()),
    'BASELINE_MISSING':int(C.MISSING.sum()),
    'BASELINE_COVERAGE_PCT':round(100*(~C.MISSING).mean(),2),
    'MISSING_WITH_REGULATORY_LINK':int(M.REG_LINK.sum()),
    'MISSING_WITH_ANY_POSITIVE_REG_VERSION':int(M.REG_AMT_POS.sum()),
    'MISSING_REG_OTHER_SOURCE_TYPE':len(other),
    'MISSING_WITHOUT_REGULATORY_LINK':int((~M.REG_LINK).sum()),
    'NEWLY_APPROVED_EXPOSURES':0,
    'COVERAGE_APPROVED_PCT':round(100*(~C.MISSING).mean(),2),
    'METHODOLOGY':'Do not sum RFO/regulatory versions. Account negatives need classification; regulatory field not approved as loan nominal.'
}]))

# Extra QA: examine the source-type prefixes behind REG_OTHER_SOURCE_TYPE
# by reading only a bounded number of matching source rows; no full-row pulls.
if len(other)>0:
    vals=sorted(set(other.INSTRUMENT_ID.dropna().map(n)))
    def lit(s): return "'"+str(s).replace("'","''")+"'"
    distributions=[]
    for j in range(0,len(vals),400):
        part=vals[j:j+400]
        sql=("SELECT CASE WHEN INSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),'_')>0 "
             "THEN SUBSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),1,INSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),'_')-1) "
             "ELSE 'NO_PREFIX' END AS REG_PREFIX, "
             "COUNT(DISTINCT TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE))) N_REFERENCES "
             "FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_MON_VW "
             "WHERE MAP_DTE>=DATE '2026-09-30' AND MAP_DTE<DATE '2026-10-01' "
             "AND TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) IN ("+','.join(lit(x) for x in part)+") "
             "GROUP BY CASE WHEN INSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),'_')>0 "
             "THEN SUBSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),1,INSTR(TRIM(TO_CHAR(INSTRUMENT_ID)),'_')-1) "
             "ELSE 'NO_PREFIX' END")
        d=SQLExecutor2(connection='PRD_BDWH').query_to_df(sql)
        d.columns=[str(c).upper() for c in d.columns]
        if not d.empty: distributions.append(d)
    if distributions:
        p=pd.concat(distributions,ignore_index=True)
        p.N_REFERENCES=pd.to_numeric(p.N_REFERENCES,errors='coerce')
        p=p.groupby('REG_PREFIX',dropna=False,as_index=False).N_REFERENCES.sum()
        show('06_OTHER_REGULATORY_SOURCE_PREFIX_DIAGNOSTIC',
             p.sort_values('N_REFERENCES',ascending=False),40)

print('\nREAD ONLY | No datasets, recipes, models, training or backtests modified.',flush=True)
