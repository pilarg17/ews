# -*- coding: utf-8 -*-
"""Corporate EWS | final pending source QAs, ONE Dataiku Python notebook cell.

READ ONLY: dataiku dataset metadata + SELECT SQL only. No writes, recipes or backtests.
Scope: RFO candidate coverage for full 29-Oct loan snapshot, full-month payer
account linkage for Jul/Aug/Sep, ACA score distribution and slot relationship.
ACA Excel column-CP identity remains unverified without the original extract.
"""
import re
import time
import pandas as pd
import numpy as np
import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
ORACLE = 'PRD_BDWH'
SNOW = 'DEV_AA_TEAMS'
CORPORATE_DATE = '2026-10-29'
RFO_DATEKEY = 20260930  # latest month end <= Corporate cutoff; do not use October 31
PAYMENT_MONTHS = ['2026-07-01', '2026-08-01', '2026-09-01']
PAYMENT_MAX_ROWS_PER_MONTH = None  # None = full-month, NOT 20k sample
BATCH = 450  # Oracle IN list well below 1,000
START = time.monotonic()
ORA = SQLExecutor2(connection=ORACLE)
SNF = SQLExecutor2(connection=SNOW)


def log(message):
    print('[%5.1fs] %s' % (time.monotonic() - START, message), flush=True)


def view(title, frame):
    print('\n=== %s ===' % title, flush=True)
    print(frame.to_csv(sep=';', index=False, na_rep='NULL').strip() if not frame.empty else 'NO ROWS', flush=True)


def norm(value):
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.upper() in ('NULL', 'NONE', 'NAN'):
        return None
    return re.sub(r'\.0+$', '', s) if re.fullmatch(r'\d+\.0+', s) else s


def lit(value):
    s = norm(value)
    if not s or len(s) > 160 or any(ord(c) < 32 for c in s):
        raise ValueError('Invalid source reference')
    return "'" + s.replace("'", "''") + "'"


def ident(s):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', str(s)):
        raise ValueError('Unverified SQL identifier')
    return '"' + str(s) + '"'


def query(engine, label, sql, max_rows=50000):
    if not re.match(r'^\s*(SELECT|WITH)\b', sql, re.I):
        raise RuntimeError('Only SELECT statements allowed')
    log('QUERY | ' + label)
    frame = engine.query_to_df(sql)
    frame.columns = [str(c).upper() for c in frame.columns]
    if len(frame) > max_rows:
        raise RuntimeError('Unexpected result size for %s: %d' % (label, len(frame)))
    return frame


def chunks(arr, n=BATCH):
    arr = list(arr)
    for i in range(0, len(arr), n):
        yield arr[i:i+n]


log('READ ONLY | BEGIN CORPORATE SOURCE VALIDATION')

# =============================================================================
# 1) RFO: Full loan-population denominator, as-of monthly global version.
# =============================================================================
log('SECTION 1 / RFO FULL POPULATION')
project = dataiku.api_client().get_project(PROJECT)
ds = dataiku.Dataset('vars_loan_level', project_key=PROJECT)
fields = {str(c['name']).upper() for c in ds.read_schema()}
required = {'ENTITY_ID', 'INSTRUMENT_ID', 'NOMINAL_AMOUNT', 'MAP_DTE'}
if required - fields:
    raise RuntimeError('Missing vars_loan_level fields: %s' % sorted(required-fields))
raw_settings = project.get_dataset('vars_loan_level').get_settings().get_raw()
settings = raw_settings.get('params', {})
if settings.get('connection') != SNOW:
    raise RuntimeError('Unexpected vars_loan_level connection')
schema = str(settings['schema']).replace('${projectKey}', PROJECT)
table = str(settings['table']).replace('${projectKey}', PROJECT)
physical = ident(schema) + '.' + ident(table)
optional = [c for c in ('CONTRACT_REFERENCE', 'CONTRACT_ID', 'CONTRACT_ID_SHORT') if c in fields]
select = ["TRIM(TO_VARCHAR(ENTITY_ID)) ENTITY_ID", "TRIM(TO_VARCHAR(INSTRUMENT_ID)) INSTRUMENT_ID",
          "TRY_TO_DOUBLE(TO_VARCHAR(NOMINAL_AMOUNT)) NOMINAL_AMOUNT"]
select.extend('TRIM(TO_VARCHAR(%s)) %s' % (ident(c), ident(c)) for c in optional)
loans = query(SNF, 'all Corporate loan rows 29-Oct',
              'SELECT '+','.join(select)+' FROM '+physical+
              " WHERE TO_DATE(MAP_DTE)=DATE '"+CORPORATE_DATE+"'", 50000)
if loans.empty:
    raise RuntimeError('Empty Corporate snapshot')
for c in ['ENTITY_ID', 'INSTRUMENT_ID'] + optional:
    loans[c] = loans[c].map(norm)
loans['ROW_ID'] = np.arange(len(loans))
loans['MISSING'] = loans.NOMINAL_AMOUNT.isna()
loans['DUP_LOAN_INSTRUMENT'] = loans.INSTRUMENT_ID.isna() | loans.INSTRUMENT_ID.duplicated(keep=False)
base = len(loans)
known = int((~loans.MISSING).sum())
missing = base-known
view('01_CORPORATE_BASELINE', pd.DataFrame([{
    'ASOF':CORPORATE_DATE,'N_LOAN_ROWS':base,'N_WITH_NOMINAL':known,
    'N_MISSING_NOMINAL':missing,'COVERAGE_PCT':round(100*known/base,2),
    'N_DUPLICATE_OR_NULL_INSTRUMENT_IDS':int(loans.DUP_LOAN_INSTRUMENT.sum())
}]))
if base != 12878 or known != 6664:
    raise RuntimeError('Baseline differs from reconciled 12,878 / 6,664; stop before comparison')

# RFO source metadata (do not assume more fields than the original source has).
meta = query(ORA, 'RFO columns',
    "SELECT TABLE_NAME,COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE OWNER='DTM_RFO_PLY' "
    "AND TABLE_NAME IN ('LOANDEPO_VW','ACCOUNT_VW')", 1000)
columns = {n:set(g.COLUMN_NAME.astype(str)) for n,g in meta.groupby('TABLE_NAME')}
specs = {
    'LOANDEPO_VW':('CONTRACT_REFERENCE','OUTSTANDING'),
    'ACCOUNT_VW':('ACCOUNT_REFERENCE','BALANCE'),
}
versions={}
for src,(ref,amt) in specs.items():
    must = {ref,amt,'ACCRUALS','CURRENCY','REPORT_DATEKEY','PARTITION_KEY','REPORT_VERSIONID'}
    if must - columns.get(src,set()):
        raise RuntimeError('RFO source fields missing: '+src+' '+str(sorted(must-columns.get(src,set()))))
    v = query(ORA,'global monthly version '+src,
        'SELECT PARTITION_KEY,COUNT(DISTINCT REPORT_VERSIONID) N_VERSIONS, '
        'MAX(REPORT_VERSIONID) LATEST_REPORT_VERSIONID '
        'FROM DTM_RFO_PLY.'+src+' WHERE REPORT_DATEKEY='+str(RFO_DATEKEY)+
        ' GROUP BY PARTITION_KEY', 50)
    if len(v) != 1 or pd.isna(v.iloc[0].LATEST_REPORT_VERSIONID):
        raise RuntimeError('Expected one RFO partition/version per source; inspect: '+src)
    versions[src] = (norm(v.iloc[0].PARTITION_KEY), int(v.iloc[0].LATEST_REPORT_VERSIONID))
    view('02_LATEST_MONTHLY_VERSION_'+src, v.assign(SOURCE=src))

# Include every missing instrument and 100 known for comparison.
focus = pd.concat([loans.loc[loans.MISSING], loans.loc[~loans.MISSING].head(100)],ignore_index=True)
key_rows=[]
for rec in focus.to_dict('records'):
    for field in ['INSTRUMENT_ID'] + optional:
        key=norm(rec.get(field))
        if key:
            for source in specs:
                key_rows.append((int(rec['ROW_ID']),source,key,field))
keys = pd.DataFrame(key_rows, columns=['ROW_ID','SOURCE','REF_KEY','KEY_ORIGIN']).drop_duplicates()
if keys.empty:
    raise RuntimeError('No RFO reference keys found')
keys['N_LOANS_FOR_KEY'] = keys.groupby(['SOURCE','REF_KEY']).ROW_ID.transform('nunique')
all_rows=[]
for src,(ref,amount) in specs.items():
    part,version=versions[src]
    values=sorted(set(keys.loc[keys.SOURCE.eq(src),'REF_KEY']))
    for i,block in enumerate(chunks(values),1):
        # Global monthly latest version, NOT max among the matched reference rows.
        sql=('SELECT TRIM(TO_CHAR('+ref+')) REF_KEY, '+amount+' RAW_AMOUNT, '
             'ACCRUALS,TRIM(TO_CHAR(CURRENCY)) CCY,REPORT_VERSIONID '
             'FROM DTM_RFO_PLY.'+src+
             ' WHERE REPORT_DATEKEY='+str(RFO_DATEKEY)+
             ' AND PARTITION_KEY='+lit(part)+
             ' AND REPORT_VERSIONID='+str(version)+
             ' AND TRIM(TO_CHAR('+ref+')) IN ('+','.join(map(lit,block))+')')
        x=query(ORA,src+' references '+str(i),sql,25000)
        if not x.empty:
            x['SOURCE']=src
            all_rows.append(x)
        if i%5 == 0 or i==(len(values)+BATCH-1)//BATCH:
            log('RFO '+src+': '+str(i)+'/'+str((len(values)+BATCH-1)//BATCH))
rfo = pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame(
    columns=['REF_KEY','RAW_AMOUNT','ACCRUALS','CCY','REPORT_VERSIONID','SOURCE'])
if not rfo.empty:
    rfo['REF_KEY']=rfo.REF_KEY.map(norm)
    rfo['CCY']=rfo.CCY.astype('string').str.strip().str.upper()
    rfo['RAW_AMOUNT']=pd.to_numeric(rfo.RAW_AMOUNT,errors='coerce')
    rfo['ACCRUALS']=pd.to_numeric(rfo.ACCRUALS,errors='coerce')
    rfo['N_RFO_ROWS_FOR_REF']=rfo.groupby(['SOURCE','REF_KEY']).REF_KEY.transform('size')
else:
    rfo['N_RFO_ROWS_FOR_REF']=pd.Series(dtype='int')
found = keys.merge(rfo,on=['SOURCE','REF_KEY'],how='left',validate='many_to_many')
found = found.merge(loans[['ROW_ID','MISSING','DUP_LOAN_INSTRUMENT','NOMINAL_AMOUNT']],on='ROW_ID',how='left',validate='many_to_one')
found['RAW_AMOUNT']=pd.to_numeric(found.RAW_AMOUNT,errors='coerce')
found['ACCRUALS']=pd.to_numeric(found.ACCRUALS,errors='coerce')
found['MATCHED']=found.REPORT_VERSIONID.notna()
found['UNIQUE_LINK']=(found.N_LOANS_FOR_KEY.eq(1)&found.N_RFO_ROWS_FOR_REF.eq(1)&
                      ~found.DUP_LOAN_INSTRUMENT)
found['LD_VALID_EUR']=(found.SOURCE.eq('LOANDEPO_VW')&found.MATCHED&found.UNIQUE_LINK&
                       found.CCY.eq('EUR')&found.RAW_AMOUNT.gt(0)&
                       (found.ACCRUALS.fillna(0)+found.RAW_AMOUNT).gt(0))
found['ACC_VALID_EUR']=(found.SOURCE.eq('ACCOUNT_VW')&found.MATCHED&found.UNIQUE_LINK&
                        found.CCY.eq('EUR')&found.RAW_AMOUNT.lt(0))
found['OLD_FILTER']=(found.ACCRUALS+found.RAW_AMOUNT).gt(0)
found['LOANDEPO_NULL_ACCRUALS']=(found.SOURCE.eq('LOANDEPO_VW')&found.RAW_AMOUNT.gt(0)&
                                 found.ACCRUALS.isna()&found.MATCHED)
missing_hits=found[found.MISSING].copy()
# Multiple distinct source references and/or opposing source types => not approved.
candidates = missing_hits[missing_hits.LD_VALID_EUR|missing_hits.ACC_VALID_EUR].copy()
uniques=[]
for row_id,group in candidates.groupby('ROW_ID'):
    dedup=group.drop_duplicates(['SOURCE','REF_KEY'])
    all_source_matches=missing_hits[(missing_hits.ROW_ID==row_id)&missing_hits.MATCHED].drop_duplicates(['SOURCE','REF_KEY'])
    if len(dedup)==1 and len(all_source_matches)==1:
        r=dedup.iloc[0]
        uniques.append({'ROW_ID':row_id,'SOURCE':str(r.SOURCE),
                        'EUR_AMOUNT':float(r.RAW_AMOUNT if r.SOURCE=='LOANDEPO_VW' else -r.RAW_AMOUNT)})
unique = pd.DataFrame(uniques,columns=['ROW_ID','SOURCE','EUR_AMOUNT'])
count_ld=int(unique.SOURCE.eq('LOANDEPO_VW').sum()) if not unique.empty else 0
count_acc=int(unique.SOURCE.eq('ACCOUNT_VW').sum()) if not unique.empty else 0
valid_ids=set(unique.ROW_ID) if not unique.empty else set()
view('03_FULL_POPULATION_RFO_CANDIDATES',pd.DataFrame([{
    'N_MISSING_LOAN_ROWS':missing,
    'MISSING_WITH_ANY_RFO_SOURCE':int(missing_hits.loc[missing_hits.MATCHED,'ROW_ID'].nunique()),
    'MISSING_LOANDEPO_POSITIVE_NULL_ACCRUALS':int(missing_hits.loc[missing_hits.LOANDEPO_NULL_ACCRUALS,'ROW_ID'].nunique()),
    'MISSING_ACCOUNT_NEGATIVE':int(missing_hits.loc[missing_hits.SOURCE.eq('ACCOUNT_VW')&missing_hits.RAW_AMOUNT.lt(0),'ROW_ID'].nunique()),
    'STRICT_UNIQUE_LOANDEPO_EUR_CANDIDATES':count_ld,
    'STRICT_UNIQUE_ACCOUNT_NEGATIVE_EUR_CANDIDATES':count_acc,
    'STRICT_UNIQUE_ADDITIONAL_LOANS':len(valid_ids),
    'ILLUSTRATIVE_COVERAGE_EUR_ONLY_PCT':round(100*(known+len(valid_ids))/base,2),
    'REMAINING_NULL_BEFORE_FX_OR_OTHER_SOURCES':missing-len(valid_ids),
    'SOURCE_COMPARABILITY_APPROVED':'NO, QA ONLY'
}]))
if not unique.empty:
    view('04_CANDIDATES_BY_SOURCE',unique.groupby('SOURCE',as_index=False).agg(
        N_INSTRUMENTS=('ROW_ID','nunique'), TOTAL_EUR=('EUR_AMOUNT','sum')))
known_matches=found[(~found.MISSING)&found.MATCHED&found.UNIQUE_LINK&found.CCY.eq('EUR')&
                    found.SOURCE.eq('LOANDEPO_VW')&found.NOMINAL_AMOUNT.gt(0)].copy()
if not known_matches.empty:
    known_matches['REL_DIFF']=(known_matches.RAW_AMOUNT-known_matches.NOMINAL_AMOUNT).abs()/known_matches.NOMINAL_AMOUNT
    view('05_KNOWN_NOMINAL_SANITY_SAMPLE',pd.DataFrame([{
        'N_KNOWN_COMPARABLE_PAIRS':len(known_matches),
        'N_WITHIN_10PCT':int(known_matches.REL_DIFF.le(.1).sum()),
        'MEDIAN_ABS_REL_DIFF_PCT':round(100*known_matches.REL_DIFF.median(),2)}]))

# =============================================================================
# 2) Incoming payment payer linkage: three FULL months, not source samples.
# =============================================================================
log('SECTION 2 / FULL MONTH PAYER ACCOUNT LINKAGE')
pay_meta=query(ORA,'PAYEE/PAYOR metadata',
    "SELECT TABLE_NAME,COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE OWNER='BDL_PAYMENT_PLY' "
    "AND TABLE_NAME IN ('PAYEE_VW','PAYOR_VW')",250)
pay_cols={n:set(g.COLUMN_NAME.astype(str)) for n,g in pay_meta.groupby('TABLE_NAME')}
for t,req in {'PAYEE_VW':{'CREDIT_PAYMENT_ID','LINKED_DEBIT_PAYMENT_ID','CREDIT_VALUE_DATE','PAYMENT_DIRECTION'},
              'PAYOR_VW':{'DEBIT_PAYMENT_ID','ORDER_GIVER_ACCOUNT'}}.items():
    if req-pay_cols.get(t,set()):
        raise RuntimeError('Payment fields missing: '+t+str(req-pay_cols.get(t,set())))
pay_months=[]
for start in PAYMENT_MONTHS:
    end=(pd.Timestamp(start)+pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d')
    cap=('AND ROWNUM <= '+str(int(PAYMENT_MAX_ROWS_PER_MONTH)) if PAYMENT_MAX_ROWS_PER_MONTH is not None else '')
    pay_sql="""
WITH P_RAW AS (
 SELECT TRIM(TO_CHAR(CREDIT_PAYMENT_ID)) CREDIT_ID,
        TRIM(TO_CHAR(LINKED_DEBIT_PAYMENT_ID)) DEBIT_ID
 FROM BDL_PAYMENT_PLY.PAYEE_VW
 WHERE CREDIT_VALUE_DATE >= DATE '{start}' AND CREDIT_VALUE_DATE < DATE '{end}'
   AND UPPER(TRIM(PAYMENT_DIRECTION))='INCOMING'
   AND CREDIT_PAYMENT_ID IS NOT NULL {cap}
), P AS (
 SELECT CREDIT_ID,MIN(DEBIT_ID) DEBIT_ID,COUNT(*) N_P_ROWS,
        COUNT(DISTINCT DEBIT_ID) N_DIRECT_IDS FROM P_RAW GROUP BY CREDIT_ID
), V AS (
 SELECT TRIM(TO_CHAR(v.DEBIT_PAYMENT_ID)) DEBIT_ID,
        COUNT(*) N_V_ROWS,
        COUNT(DISTINCT TRIM(TO_CHAR(v.ORDER_GIVER_ACCOUNT))) N_DISTINCT_ACCOUNTS,
        MAX(TRIM(TO_CHAR(v.ORDER_GIVER_ACCOUNT))) ORDER_GIVER_ACCOUNT
 FROM BDL_PAYMENT_PLY.PAYOR_VW v
 JOIN (SELECT DISTINCT DEBIT_ID FROM P WHERE DEBIT_ID IS NOT NULL) d
   ON TRIM(TO_CHAR(v.DEBIT_PAYMENT_ID))=d.DEBIT_ID
 GROUP BY TRIM(TO_CHAR(v.DEBIT_PAYMENT_ID))
)
SELECT COUNT(*) N_UNIQUE_CREDIT_PAYMENTS,
 NVL(SUM(N_P_ROWS),0) N_PAYEE_ROWS,
 NVL(SUM(CASE WHEN P.N_DIRECT_IDS>1 THEN 1 ELSE 0 END),0) N_AMBIGUOUS_DIRECT_IDS,
 NVL(SUM(CASE WHEN P.N_P_ROWS>1 THEN 1 ELSE 0 END),0) N_DUPLICATE_CREDIT_ID,
 NVL(SUM(CASE WHEN P.DEBIT_ID IS NOT NULL THEN 1 ELSE 0 END),0) N_WITH_DEBIT_ID,
 NVL(SUM(CASE WHEN V.DEBIT_ID IS NOT NULL THEN 1 ELSE 0 END),0) N_MATCHED_PAYOR,
 NVL(SUM(CASE WHEN V.ORDER_GIVER_ACCOUNT IS NOT NULL THEN 1 ELSE 0 END),0) N_WITH_PAYER_ACCOUNT,
 NVL(SUM(CASE WHEN V.DEBIT_ID IS NOT NULL AND V.ORDER_GIVER_ACCOUNT IS NULL THEN 1 ELSE 0 END),0) N_MATCH_WITHOUT_ACCOUNT,
 NVL(SUM(CASE WHEN V.N_V_ROWS>1 THEN 1 ELSE 0 END),0) N_MULTIPLE_PAYOR_ROWS,
 NVL(SUM(CASE WHEN V.N_DISTINCT_ACCOUNTS>1 THEN 1 ELSE 0 END),0) N_CONFLICTING_PAYER_ACCOUNTS
FROM P LEFT JOIN V ON P.DEBIT_ID=V.DEBIT_ID
""".format(start=start,end=end,cap=cap)
    x=query(ORA,'full payment payer linkage '+start,pay_sql,5)
    if x.empty:
        raise RuntimeError('No payment QA returned for '+start)
    r=x.iloc[0].to_dict()
    r['MONTH']=start[:7]
    r['PAYOR_MATCH_PCT']=round(100*float(r['N_MATCHED_PAYOR'])/max(1,float(r['N_UNIQUE_CREDIT_PAYMENTS'])),2)
    r['PAYER_ACCOUNT_COVERAGE_PCT']=round(100*float(r['N_WITH_PAYER_ACCOUNT'])/max(1,float(r['N_UNIQUE_CREDIT_PAYMENTS'])),2)
    r['SCOPE']='FULL_MONTH' if PAYMENT_MAX_ROWS_PER_MONTH is None else 'CAPPED_SAMPLE'
    pay_months.append(r)
pay=pd.DataFrame(pay_months)
view('06_MONTHLY_PAYER_ACCOUNT_COVERAGE',pay)

# =============================================================================
# 3) ACA: candidate score, granularity, rating mapping (not Excel CP proof).
# =============================================================================
log('SECTION 3 / ACA SCORE QUALITY')
stage=dataiku.Dataset('ACA_slotting_entity_day',project_key=PROJECT).get_dataframe()
stage.columns=[str(c).upper() for c in stage.columns]
if {'ENTITY_ID','MAP_DTE','ACA_FINAL_RATING_WORST'}-set(stage):
    raise RuntimeError('ACA staging missing expected identity/date/rating columns')
view('07_ACA_STAGING',pd.DataFrame([{
    'N_ROWS':len(stage),'N_ENTITIES':stage.ENTITY_ID.nunique(),
    'N_DATES':pd.to_datetime(stage.MAP_DTE,errors='coerce').nunique(),
    'DUP_ENTITY_DAY':int(stage.duplicated(['ENTITY_ID','MAP_DTE']).sum()),
    'RATING_WORST_DISTINCT':stage.ACA_FINAL_RATING_WORST.nunique(dropna=True),
    'HAS_RISKASSESSMENT_SCORE':'RISKASSESSMENT_SCORE' in stage.columns
}]))
aca_meta=query(ORA,'ACA source columns',
    "SELECT TABLE_NAME,COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE OWNER='BDWH_PLY' "
    "AND TABLE_NAME='PLY_ACA_LEGAL_ENTITY_DAY_VW'",350)
aca_fields=set(aca_meta.COLUMN_NAME.astype(str))
required_aca={'MAP_DTE','LEGAL_ENTITY_REF_CODE','RISKASSESSMENT_SCORE','FINAL_RATING'}
if required_aca-aca_fields:
    raise RuntimeError('Missing ACA source columns '+str(sorted(required_aca-aca_fields)))
aca_day=query(ORA,'ACA most recent snapshot <= Corporate as-of',
    'SELECT MAX(MAP_DTE) ACA_SOURCE_DATE FROM BDWH_PLY.PLY_ACA_LEGAL_ENTITY_DAY_VW '
    "WHERE MAP_DTE<DATE '"+CORPORATE_DATE+"'+1",5)
if aca_day.empty or pd.isna(aca_day.iloc[0].ACA_SOURCE_DATE):
    raise RuntimeError('No ACA source snapshot before cutoff')
aca_date=pd.Timestamp(aca_day.iloc[0].ACA_SOURCE_DATE).strftime('%Y-%m-%d')
aca_sql="""
SELECT TRIM(TO_CHAR(FINAL_RATING)) FINAL_RATING,
 COUNT(*) N_ROWS,
 COUNT(DISTINCT TRIM(TO_CHAR(LEGAL_ENTITY_REF_CODE))) N_SOURCE_REFERENCES,
 COUNT(RISKASSESSMENT_SCORE) N_WITH_SCORE,
 COUNT(DISTINCT RISKASSESSMENT_SCORE) N_DISTINCT_SCORES,
 COUNT(DISTINCT ROUND(RISKASSESSMENT_SCORE,2)) N_DISTINCT_ROUNDED_2DP,
 MIN(RISKASSESSMENT_SCORE) MIN_SCORE,
 MAX(RISKASSESSMENT_SCORE) MAX_SCORE,
 SUM(CASE WHEN RISKASSESSMENT_SCORE IS NOT NULL AND
           RISKASSESSMENT_SCORE<>TRUNC(RISKASSESSMENT_SCORE) THEN 1 ELSE 0 END) N_FRACTIONAL_SCORES,
 SUM(CASE WHEN RISKASSESSMENT_SCORE IS NOT NULL AND
           RISKASSESSMENT_SCORE<>ROUND(RISKASSESSMENT_SCORE,2) THEN 1 ELSE 0 END) N_SCORES_WITH_MORE_THAN_2DP
FROM BDWH_PLY.PLY_ACA_LEGAL_ENTITY_DAY_VW
WHERE MAP_DTE>=DATE '{date}' AND MAP_DTE<DATE '{date}'+1
GROUP BY TRIM(TO_CHAR(FINAL_RATING)) ORDER BY FINAL_RATING
""".format(date=aca_date)
aca=query(ORA,'ACA risk assessment score by final rating',aca_sql,25)
aca.insert(0,'SOURCE_DATE',aca_date)
view('08_ACA_SCORE_BY_RATING',aca)

print('\n=== 09_DECISION_STATUS ===',flush=True)
print('RFO: full 29-Oct denominator; EUR-only candidates from latest global September RFO versions; no source change approved.',flush=True)
print('PAYMENTS: '+('three full months' if PAYMENT_MAX_ROWS_PER_MONTH is None else 'three capped monthly samples')+
      ' of linkage; recurring-payer signal (6 months/top-3/>20%/90% drop and exclusions) NOT tested here.',flush=True)
print('ACA: RISKASSESSMENT_SCORE distribution tested, but Excel column CP identity cannot be proved without the original ACA extract and reference-ID/value comparison.',flush=True)
print('Source mismatch, duplicate, FX, source semantics or score mismatches must be reviewed BEFORE any recipe edit.',flush=True)
print('READ ONLY / NO RECIPES / NO DATASETS / NO BACKTEST MODIFICATIONS',flush=True)
log('END')
