# Corporate EWS | FULL exposure coverage reconciliation | one Dataiku Python notebook cell
# Read-only, full 2026-10-29 portfolio, strict as-of, no writes
import re
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
SNOWFLAKE = 'DEV_AA_TEAMS'
ORACLE = 'PRD_BDWH'
ASOF = '2026-10-29'
RFO_MAX_AGE = 31
REG_MAX_LOOKBACK = 92
BATCH = 450
STARTED = time.monotonic()


def log(msg):
    print('[%6.1fs] %s' % (time.monotonic()-STARTED, msg), flush=True)


def show(label, frame):
    print('\n=== '+label+' ===', flush=True)
    if frame is None or frame.empty:
        print('NO_ROWS', flush=True)
    else:
        print(frame.to_csv(index=False, sep=';', na_rep='NULL').strip(), flush=True)


def norm(x):
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in ('nan', 'none', 'null'):
        return None
    return s.split('.')[0] if re.fullmatch(r'\d+\.0+', s) else s


def ident(x):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', str(x)):
        raise RuntimeError('Unverified table identifier')
    return '"'+str(x)+'"'


def lit(x):
    x = norm(x)
    if not x or len(x) > 128 or any(ord(c)<32 for c in x):
        raise RuntimeError('Invalid source reference')
    return "'"+x.replace("'", "''")+"'"


def batches(xs, n=BATCH):
    xs = list(xs)
    for offset in range(0, len(xs), n):
        yield xs[offset:offset+n]


def sql(connection, label, query, max_rows=250000):
    if not re.match(r'^\s*(SELECT|WITH)\b', query, re.I):
        raise RuntimeError('Non-read-only statement refused')
    log('QUERY | '+label)
    out = SQLExecutor2(connection=connection).query_to_df(query)
    out.columns = [str(c).upper() for c in out.columns]
    if len(out)>max_rows:
        raise RuntimeError('Unbounded result for '+label)
    return out


log('START | FULL CORPORATE SNAPSHOT | NO SAMPLING | READ ONLY')
# 1. Real Corporate population, not a 15-loan or 750-loan sample.
loan_ds = dataiku.Dataset('vars_loan_level', project_key=PROJECT)
cols = {str(c['name']).upper() for c in loan_ds.read_schema()}
needed = {'ENTITY_ID','INSTRUMENT_ID','NOMINAL_AMOUNT','MAP_DTE'}
if needed-cols:
    raise RuntimeError('Missing Corporate fields: '+str(sorted(needed-cols)))
settings = dataiku.api_client().get_project(PROJECT).get_dataset('vars_loan_level').get_settings().get_raw().get('params', {})
if settings.get('connection') != SNOWFLAKE:
    raise RuntimeError('Unexpected Corporate source connection')
loan_table = ident(settings['schema'])+'.'+ident(settings['table'].replace('${projectKey}', PROJECT))
optional = ['CONTRACT_REFERENCE', 'CONTRACT_ID', 'CONTRACT_ID_SHORT']
select = ["TRIM(TO_VARCHAR(ENTITY_ID)) ENTITY_ID",
          "TRIM(TO_VARCHAR(INSTRUMENT_ID)) INSTRUMENT_ID",
          "TRY_TO_DOUBLE(TO_VARCHAR(NOMINAL_AMOUNT)) NOMINAL_AMOUNT"]
for c in optional:
    if c in cols:
        select.append('TRIM(TO_VARCHAR('+c+')) '+c)
loans = sql(SNOWFLAKE, 'Corporate full loan snapshot',
    'SELECT '+','.join(select)+' FROM '+loan_table+
    " WHERE TO_DATE(MAP_DTE)=DATE '"+ASOF+"'", max_rows=50000)
if loans.empty:
    raise RuntimeError('Missing Corporate snapshot')
for c in ['ENTITY_ID','INSTRUMENT_ID']+[x for x in optional if x in loans]:
    loans[c]=loans[c].map(norm)
loans['ROW_ID']=np.arange(len(loans))
loans['MISSING']=loans.NOMINAL_AMOUNT.isna()
loans['DUP_INSTRUMENT']=loans.INSTRUMENT_ID.duplicated(keep=False) | loans.INSTRUMENT_ID.isna()
base_known = int((~loans.MISSING).sum())
base_missing = int(loans.MISSING.sum())
base_n = len(loans)
show('01_BASELINE',pd.DataFrame([{'AS_OF':ASOF,'N_LOAN_ROWS':base_n,
    'N_WITH_NOMINAL':base_known,'N_WITHOUT_NOMINAL':base_missing,
    'BASELINE_COVERAGE_PCT':round(100*base_known/base_n,2),
    'N_POSITIVE_NOMINAL':int(loans.NOMINAL_AMOUNT.gt(0).sum()),
    'N_ZERO_NOMINAL':int(loans.NOMINAL_AMOUNT.eq(0).sum()),
    'N_NEGATIVE_NOMINAL':int(loans.NOMINAL_AMOUNT.lt(0).sum()),
    'N_ENTITIES':loans.ENTITY_ID.nunique(),
    'N_ENTITIES_WITH_ANY_MISSING':int(loans.groupby('ENTITY_ID').MISSING.any().sum()),
    'N_ROWS_DUPLICATE_INSTRUMENT':int(loans.DUP_INSTRUMENT.sum())}]))

# 2. Dated regulatory instrument -> source-contract crosswalk.
meta = sql(ORACLE, 'Oracle schema checks',
    "SELECT OWNER,TABLE_NAME,COLUMN_NAME FROM ALL_TAB_COLUMNS "
    "WHERE (OWNER='BDL_CREDIT_INSIGHT_PLY' AND TABLE_NAME='INSTRUMENT_REGULATORY_METRICS_MON_VW') "
    "OR (OWNER='DTM_RFO_PLY' AND TABLE_NAME IN ('LOANDEPO_VW','ACCOUNT_VW'))", max_rows=900)
fields = {(o,t):set(g.COLUMN_NAME.astype(str)) for (o,t),g in meta.groupby(['OWNER','TABLE_NAME'])}
reg_cols = fields.get(('BDL_CREDIT_INSIGHT_PLY','INSTRUMENT_REGULATORY_METRICS_MON_VW'),set())
required_reg = {'MAP_DTE','INSTRUMENT_ID','SOURCE_CONTRACT_REFERENCE','OUTSTANDING_NOMINAL_AMOUNT'}
if required_reg-reg_cols:
    raise RuntimeError('Regulatory crosswalk missing: '+str(sorted(required_reg-reg_cols)))
reg_table = 'BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_MON_VW'
reg_date = sql(ORACLE,'Regulatory last available dated snapshot',
    "SELECT MAX(MAP_DTE) REG_DATE FROM "+reg_table+" WHERE MAP_DTE < DATE '"+ASOF+"' + 1",10)
if reg_date.empty or pd.isna(reg_date.iloc[0].REG_DATE):
    raise RuntimeError('No regulatory monthly snapshot at or before cutoff')
reg_day = pd.to_datetime(reg_date.iloc[0].REG_DATE).strftime('%Y-%m-%d')
log('Regulatory snapshot <= cutoff: '+reg_day)
reg_sel = ["TRIM(TO_CHAR(INSTRUMENT_ID)) INSTRUMENT_ID",
    "TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) SOURCE_CONTRACT_REFERENCE",
    "OUTSTANDING_NOMINAL_AMOUNT REG_AMOUNT",
    "TRUNC(MAP_DTE) REG_OBS_DATE"]
for c in ('LEGACY_CONTRACT_REFERENCE','SOURCE_TABLE_NAME','SOURCE_VERSION_ID','LAST_VERSION_FLAG'):
    if c in reg_cols:
        reg_sel.append('TRIM(TO_CHAR('+c+')) '+c)
reg_frames=[]
ids=sorted(set(loans.INSTRUMENT_ID.dropna()))
reg_start=(pd.Timestamp(ASOF)-pd.Timedelta(days=REG_MAX_LOOKBACK)).strftime('%Y-%m-%d')
for i,part in enumerate(batches(ids),1):
    q=("SELECT * FROM (SELECT "+','.join(reg_sel)+", "
        "DENSE_RANK() OVER (PARTITION BY TRIM(TO_CHAR(INSTRUMENT_ID)) ORDER BY MAP_DTE DESC) REG_RANK "
        "FROM "+reg_table+" WHERE MAP_DTE >= DATE '"+reg_start+
        "' AND MAP_DTE < DATE '"+ASOF+"' + 1 AND TRIM(TO_CHAR(INSTRUMENT_ID)) IN ("+
        ','.join(lit(x) for x in part)+')) WHERE REG_RANK=1')
    z=sql(ORACLE,'Regulatory instruments batch %d'%i,q,max_rows=20000)
    if not z.empty:
        reg_frames.append(z)
    if i%5==0:
        log('REG | %d/%d batches'%(i,(len(ids)+BATCH-1)//BATCH))
reg = pd.concat(reg_frames,ignore_index=True) if reg_frames else pd.DataFrame(
    columns=['INSTRUMENT_ID','SOURCE_CONTRACT_REFERENCE','REG_AMOUNT','REG_OBS_DATE'])
for c in ['INSTRUMENT_ID','SOURCE_CONTRACT_REFERENCE','LEGACY_CONTRACT_REFERENCE']:
    if c in reg:
        reg[c]=reg[c].map(norm)
reg['REG_OBS_DATE']=pd.to_datetime(reg.REG_OBS_DATE,errors='coerce').dt.tz_localize(None)
reg['REG_AGE_DAYS']=(pd.Timestamp(ASOF)-reg.REG_OBS_DATE).dt.days
reg['REG_LINK_CURRENT']=reg.REG_AGE_DAYS.between(0,RFO_MAX_AGE)
reg['REG_AMOUNT']=pd.to_numeric(reg.REG_AMOUNT,errors='coerce')
regstat=[]
for inst,g in reg.groupby('INSTRUMENT_ID'):
    r={ 'INSTRUMENT_ID':inst,'REG_ROWS':len(g),
        'REG_REF_COUNT':g.SOURCE_CONTRACT_REFERENCE.dropna().nunique(),
        'REG_AMOUNT_COUNT':g.REG_AMOUNT.notna().sum(),
        'REG_AMOUNT_DISTINCT':g.REG_AMOUNT.dropna().nunique(),
        'REG_ASOF_FRESH':bool(g.REG_LINK_CURRENT.any())}
    regstat.append(r)
regstat=pd.DataFrame(regstat) if regstat else pd.DataFrame(columns=['INSTRUMENT_ID','REG_ROWS','REG_REF_COUNT','REG_AMOUNT_COUNT','REG_AMOUNT_DISTINCT','REG_ASOF_FRESH'])
cohort=loans.merge(regstat,on='INSTRUMENT_ID',how='left',validate='many_to_one')
show('02_REGULATORY_CROSSWALK',pd.DataFrame([{'REG_ASOF':reg_day,
    'LOANS_WITH_REG_RECORD':int(cohort.REG_ROWS.notna().sum()),
    'MISSING_WITH_REG_RECORD':int((cohort.MISSING & cohort.REG_ROWS.notna()).sum()),
    'MISSING_WITH_ONE_SOURCE_REF':int((cohort.MISSING & cohort.REG_REF_COUNT.eq(1)).sum()),
    'MISSING_WITH_FRESH_REGULATORY_LINK':int((cohort.MISSING & cohort.REG_ASOF_FRESH.fillna(False)).sum()),
    'MISSING_WITH_NON_NULL_REG_AMOUNT':int((cohort.MISSING & cohort.REG_AMOUNT_COUNT.gt(0)).sum()),
    'REG_SOURCE_REFERENCE_AMBIGUOUS_MISSING':int((cohort.MISSING & cohort.REG_REF_COUNT.gt(1)).sum())}]))

# 3. Exact reference strategies ONLY (no fuzzy joins, no account balance as automatic loan nominal).
ref_rows=[]
for _,r in loans.iterrows():
    for src,field in [('LOANDEPO_VW','INSTRUMENT_ID'),('ACCOUNT_VW','INSTRUMENT_ID'),
                      ('LOANDEPO_VW','CONTRACT_REFERENCE')]:
        key=norm(r.get(field))
        if key:
            ref_rows.append((r.ROW_ID,r.INSTRUMENT_ID,src,field,key))
# Regulatory crosswalk may link the instrument to a different source reference.
inst_rows=loans[['ROW_ID','INSTRUMENT_ID']].dropna().drop_duplicates()
if not reg.empty:
    regkeys=reg.loc[reg.REG_LINK_CURRENT,['INSTRUMENT_ID','SOURCE_CONTRACT_REFERENCE']].dropna().drop_duplicates()
    bridge=inst_rows.merge(regkeys,on='INSTRUMENT_ID',how='inner')
    for row in bridge.itertuples(index=False):
        for src in ['LOANDEPO_VW','ACCOUNT_VW']:
            ref_rows.append((row.ROW_ID,row.INSTRUMENT_ID,src,'REG_SOURCE_CONTRACT_REFERENCE',row.SOURCE_CONTRACT_REFERENCE))
    if 'LEGACY_CONTRACT_REFERENCE' in reg:
        leg=reg.loc[reg.REG_LINK_CURRENT,['INSTRUMENT_ID','LEGACY_CONTRACT_REFERENCE']].dropna().drop_duplicates()
        bridge=inst_rows.merge(leg,on='INSTRUMENT_ID',how='inner')
        for row in bridge.itertuples(index=False):
            for src in ['LOANDEPO_VW','ACCOUNT_VW']:
                ref_rows.append((row.ROW_ID,row.INSTRUMENT_ID,src,'REG_LEGACY_CONTRACT_REFERENCE',row.LEGACY_CONTRACT_REFERENCE))
keys=pd.DataFrame(ref_rows,columns=['ROW_ID','INSTRUMENT_ID','SOURCE','KEY_ORIGIN','REF_KEY']).drop_duplicates()
key_count=keys.groupby(['SOURCE','REF_KEY']).ROW_ID.nunique().rename('N_LOANS_FOR_REF')
keys=keys.join(key_count,on=['SOURCE','REF_KEY'])

# 4. Global version per RFO partition, not maximum inside the matched sample.
start=(pd.Timestamp(ASOF)-pd.Timedelta(days=RFO_MAX_AGE)).strftime('%Y%m%d')
end=ASOF.replace('-','')
records=[]
for src,ref,amount in [('LOANDEPO_VW','CONTRACT_REFERENCE','OUTSTANDING'),
                       ('ACCOUNT_VW','ACCOUNT_REFERENCE','BALANCE')]:
    c=fields.get(('DTM_RFO_PLY',src),set())
    req={ref,amount,'ACCRUALS','PARTITION_KEY','REPORT_VERSIONID','REPORT_DATEKEY','CURRENCY'}
    if req-c:
        raise RuntimeError('Missing RFO columns for '+src+': '+str(sorted(req-c)))
    vers=sql(ORACLE,src+' global version per partition',
        'SELECT PARTITION_KEY,MAX(REPORT_VERSIONID) MAX_VERSION FROM DTM_RFO_PLY.'+src+
        ' WHERE REPORT_DATEKEY BETWEEN '+start+' AND '+end+' GROUP BY PARTITION_KEY',max_rows=200)
    if vers.empty:
        log('NO RFO partitions in period: '+src)
        continue
    # Constrain to authoritative versions before deduplicating references.
    version_filters=[]
    for v in vers.itertuples(index=False):
        if pd.isna(v.PARTITION_KEY) or pd.isna(v.MAX_VERSION):
            continue
        version_filters.append('(PARTITION_KEY='+lit(v.PARTITION_KEY)+' AND REPORT_VERSIONID='+str(int(v.MAX_VERSION))+')')
    if not version_filters:
        continue
    version_pred=' OR '.join(version_filters)
    wanted=sorted(set(keys.loc[keys.SOURCE.eq(src),'REF_KEY']))
    kind='ACCOUNT_TYPE' if src=='ACCOUNT_VW' and 'ACCOUNT_TYPE' in c else 'CAST(NULL AS VARCHAR2(100))'
    for i,part in enumerate(batches(wanted),1):
        q=('SELECT TRIM(TO_CHAR('+ref+')) REF_KEY,REPORT_DATEKEY,REPORT_VERSIONID,PARTITION_KEY,'+
           'UPPER(TRIM(TO_CHAR(CURRENCY))) CCY,'+amount+' RAW_AMOUNT,ACCRUALS,'+kind+' ACCOUNT_TYPE '
           'FROM DTM_RFO_PLY.'+src+' WHERE TRIM(TO_CHAR('+ref+')) IN ('+
           ','.join(lit(x) for x in part)+') AND REPORT_DATEKEY BETWEEN '+start+' AND '+end+
           ' AND ('+version_pred+')')
        got=sql(ORACLE,src+' references %d'%i,q,max_rows=15000)
        if not got.empty:
            got['SOURCE']=src
            records.append(got)
        if i%5==0:
            log('RFO | %s | %d/%d batches'%(src,i,(len(wanted)+BATCH-1)//BATCH))
if records:
    rfo=pd.concat(records,ignore_index=True)
    for c in ['REF_KEY','SOURCE','CCY','ACCOUNT_TYPE']:
        rfo[c]=rfo[c].map(norm)
    rfo['RAW_AMOUNT']=pd.to_numeric(rfo.RAW_AMOUNT,errors='coerce')
    rfo['ACCRUALS']=pd.to_numeric(rfo.ACCRUALS,errors='coerce')
    rfo['REPORT_DATEKEY']=pd.to_numeric(rfo.REPORT_DATEKEY,errors='coerce')
    # More than one physical row in the latest eligible partition makes the reference ambiguous.
    rfo['N_ON_DATE']=rfo.groupby(['SOURCE','REF_KEY','REPORT_DATEKEY']).REF_KEY.transform('size')
    rfo=rfo.sort_values(['SOURCE','REF_KEY','REPORT_DATEKEY'],ascending=[True,True,False])
    latest=rfo.drop_duplicates(['SOURCE','REF_KEY'],keep='first').copy()
    latest['REPORT_DATE']=pd.to_datetime(latest.REPORT_DATEKEY.astype('Int64').astype(str),format='%Y%m%d',errors='coerce')
    latest['AGE_DAYS']=(pd.Timestamp(ASOF)-latest.REPORT_DATE).dt.days
    latest['OLD_FILTER_PASSES']=(latest.ACCRUALS+latest.RAW_AMOUNT).gt(0)
    latest['NULL_ACCRUALS_POSITIVE']=(latest.RAW_AMOUNT.gt(0) & latest.ACCRUALS.isna())
    latest['NEGATIVE_ACCOUNT']=(latest.SOURCE.eq('ACCOUNT_VW') & latest.RAW_AMOUNT.lt(0))
else:
    latest=pd.DataFrame(columns=['SOURCE','REF_KEY','RAW_AMOUNT','ACCRUALS','CCY','ACCOUNT_TYPE','AGE_DAYS',
        'N_ON_DATE','OLD_FILTER_PASSES','NULL_ACCRUALS_POSITIVE','NEGATIVE_ACCOUNT','REPORT_DATEKEY'])
matched=keys.merge(latest,on=['SOURCE','REF_KEY'],how='left',validate='many_to_one')
for c in ['RAW_AMOUNT','ACCRUALS','AGE_DAYS','N_ON_DATE']:
    matched[c]=pd.to_numeric(matched[c],errors='coerce')
matched['MATCHED_SOURCE']=matched.REPORT_DATEKEY.notna()
matched['STRICT_ONE_TO_ONE']=(matched.N_LOANS_FOR_REF.eq(1) & matched.N_ON_DATE.eq(1) &
    ~matched.ROW_ID.isin(loans.loc[loans.DUP_INSTRUMENT,'ROW_ID']))
matched['FRESH']=matched.AGE_DAYS.between(0,RFO_MAX_AGE)
matched['RFO_POSITIVE']=matched.RAW_AMOUNT.gt(0)
matched['EUR']=matched.CCY.eq('EUR')
matched['ELIGIBLE_LOANDEPO_SOURCE_CCY']=(matched.SOURCE.eq('LOANDEPO_VW') & matched.STRICT_ONE_TO_ONE &
    matched.FRESH & matched.RFO_POSITIVE & matched.CCY.notna())
matched['ELIGIBLE_LOANDEPO_EUR']=(matched.ELIGIBLE_LOANDEPO_SOURCE_CCY & matched.EUR)
matched['AMBIGUOUS_LINK']=(matched.MATCHED_SOURCE & ~matched.STRICT_ONE_TO_ONE)
matched['CONDITIONAL_ACCOUNT_OVERDRAFT_EUR']=(matched.SOURCE.eq('ACCOUNT_VW') & matched.STRICT_ONE_TO_ONE &
    matched.FRESH & matched.RAW_AMOUNT.lt(0) & matched.EUR)

# Do not count multiple source references twice; if one loan maps to several unique RFO keys,
# recovery is NOT approved in this technical QA.
loan_matches=matched[matched.ELIGIBLE_LOANDEPO_EUR].copy()
source_ccy_matches=matched[matched.ELIGIBLE_LOANDEPO_SOURCE_CCY].copy()
unique_source=source_ccy_matches.groupby('ROW_ID').REF_KEY.nunique()
recoverable_source_ids=set(unique_source[unique_source.eq(1)].index)
unique_ld=loan_matches.groupby('ROW_ID').REF_KEY.nunique()
recoverable_ids=set(unique_ld[unique_ld.eq(1)].index)
# Even if two source strategies return the same ref, deduplicate it by reference.
recoverable=loans.ROW_ID.isin(recoverable_ids) & loans.MISSING
conditional_acc=matched.loc[matched.CONDITIONAL_ACCOUNT_OVERDRAFT_EUR,'ROW_ID'].drop_duplicates()
account_ids=set(conditional_acc)-set(recoverable_ids)
missing_rfo=matched[matched.ROW_ID.isin(loans.loc[loans.MISSING,'ROW_ID'])]

summary=pd.DataFrame([{'AS_OF':ASOF,'N_LOAN_ROWS':base_n,
    'BASELINE_KNOWN':base_known,'BASELINE_MISSING':base_missing,
    'RFO_LOANDEPO_MATCHED_MISSING':int(missing_rfo.loc[missing_rfo.SOURCE.eq('LOANDEPO_VW') & missing_rfo.MATCHED_SOURCE,'ROW_ID'].nunique()),
    'RFO_ACCOUNT_MATCHED_MISSING':int(missing_rfo.loc[missing_rfo.SOURCE.eq('ACCOUNT_VW') & missing_rfo.MATCHED_SOURCE,'ROW_ID'].nunique()),
    'RFO_LOANDEPO_NULL_ACCRUALS_POSITIVE_MISSING':int(missing_rfo.loc[missing_rfo.SOURCE.eq('LOANDEPO_VW') & missing_rfo.NULL_ACCRUALS_POSITIVE.eq(True),'ROW_ID'].nunique()),
    'RFO_ACCOUNT_NEGATIVE_MISSING':int(missing_rfo.loc[missing_rfo.SOURCE.eq('ACCOUNT_VW') & missing_rfo.NEGATIVE_ACCOUNT.eq(True),'ROW_ID'].nunique()),
    'STRICT_UNIQUE_LOANDEPO_ANY_CURRENCY_CANDIDATES':len(recoverable_source_ids & set(loans.loc[loans.MISSING,'ROW_ID'])),
    'STRICT_UNIQUE_LOANDEPO_EUR_CANDIDATES':int(recoverable.sum()),
    'CONDITIONAL_ACCOUNT_OVERDRAFT_CANDIDATES':len(account_ids & set(loans.loc[loans.MISSING,'ROW_ID'])),
    'POTENTIAL_SOURCE_CURRENCY_COVERAGE_PCT':round(100*(base_known+len(recoverable_source_ids & set(loans.loc[loans.MISSING,'ROW_ID'])))/base_n,2),
    'POTENTIAL_KNOWN_WITH_LOANDEPO_ONLY':base_known+int(recoverable.sum()),
    'POTENTIAL_COVERAGE_PCT':round(100*(base_known+int(recoverable.sum()))/base_n,2),
    'STILL_MISSING_AFTER_LOANDEPO_CANDIDATES':base_missing-int(recoverable.sum()),
    'FX_NON_EUR_PENDING':int(missing_rfo.loc[missing_rfo.MATCHED_SOURCE & ~missing_rfo.EUR.fillna(False),'ROW_ID'].nunique()),
    'SOURCE_CHANGE_APPROVED':'NO'}])
show('03_COVERAGE_BEFORE_AFTER_CANDIDATES',summary)
show('04_MISSING_RFO_BY_SOURCE_AND_ORIGIN',missing_rfo.groupby(['SOURCE','KEY_ORIGIN'],dropna=False).agg(
    N_LOANS_TESTED=('ROW_ID','nunique'),
    N_MATCHED_LINKS=('MATCHED_SOURCE',lambda s: int(s.sum())),
    N_MATCHED_AMBIGUOUS_LINKS=('AMBIGUOUS_LINK','sum'),
    N_EUR_POSITIVE_CANDIDATES=('ELIGIBLE_LOANDEPO_EUR',lambda s:int(s.sum())),
    N_OVERDRAFT_CANDIDATES=('CONDITIONAL_ACCOUNT_OVERDRAFT_EUR',lambda s:int(s.sum()))).reset_index())

# Known-nominal sanity check (not a replacement authorization).
known_compare=loan_matches.merge(loans[['ROW_ID','NOMINAL_AMOUNT']],on='ROW_ID',how='left',validate='many_to_one')
known_compare=known_compare[known_compare.NOMINAL_AMOUNT.gt(0)]
if not known_compare.empty:
    known_compare['REL_DIFF']=(known_compare.RAW_AMOUNT-known_compare.NOMINAL_AMOUNT).abs()/known_compare.NOMINAL_AMOUNT
    show('05_KNOWN_NOMINAL_RECONCILIATION',pd.DataFrame([{
        'KNOWN_REFERENCE_PAIRS':len(known_compare),
        'N_WITHIN_10_PCT':int(known_compare.REL_DIFF.le(.10).sum()),
        'MEDIAN_RELATIVE_DIFFERENCE_PCT':round(100*known_compare.REL_DIFF.median(),2)}]))

# Show a small, inspectable set of unresolved IDs, without exposing personal names.
diagnostics=cohort[cohort.MISSING][['INSTRUMENT_ID','ENTITY_ID','REG_ROWS','REG_REF_COUNT','REG_AMOUNT_COUNT']].copy()
linked=missing_rfo.groupby('ROW_ID',as_index=False).agg(
    RFO_MATCH=('MATCHED_SOURCE','max'),
    LOANDEPO_ELIGIBLE=('ELIGIBLE_LOANDEPO_EUR','max'),
    ACCOUNT_OVERDRAFT_CANDIDATE=('CONDITIONAL_ACCOUNT_OVERDRAFT_EUR','max'))
diagnostics['ROW_ID']=cohort.loc[cohort.MISSING,'ROW_ID'].values
diagnostics=diagnostics.merge(linked,on='ROW_ID',how='left',validate='one_to_one')
diagnostics['LOANDEPO_ELIGIBLE']=diagnostics.LOANDEPO_ELIGIBLE.fillna(False)
diagnostics['REMAINING_UNRESOLVED']=~diagnostics.ROW_ID.isin(recoverable_ids)
# Mutually exclusive root-cause categories for missing exposure.
diagnostics['ROOT_CAUSE_GROUP'] = np.select([
    ~diagnostics.REMAINING_UNRESOLVED,
    diagnostics.ACCOUNT_OVERDRAFT_CANDIDATE.fillna(False),
    diagnostics.REG_AMOUNT_COUNT.fillna(0).gt(0),
    diagnostics.RFO_MATCH.fillna(False),
    diagnostics.REG_REF_COUNT.fillna(0).gt(0),
], [
    'LOANDEPO_EUR_1TO1_CANDIDATE',
    'ACCOUNT_OVERDRAFT_REQUIRES_VALIDATION',
    'REGULATORY_AMOUNT_PRESENT_RECONCILE',
    'RFO_MATCH_BUT_NOT_ELIGIBLE',
    'REGULATORY_REFERENCE_BUT_NO_ELIGIBLE_RFO',
], default='NO_VALIDATED_SOURCE_LINK')
root_summary=diagnostics.groupby('ROOT_CAUSE_GROUP',dropna=False).size().reset_index(name='N_MISSING_LOAN_ROWS')
root_summary['PCT_OF_MISSING']=(100*root_summary.N_MISSING_LOAN_ROWS/base_missing).round(2)
show('06_ROOT_CAUSES_FULL_MISSING_COHORT',root_summary)
show('06A_FIRST_25_UNRESOLVED',diagnostics.loc[diagnostics.REMAINING_UNRESOLVED].head(25))

# Provided RFO sample references are independent until linked to an instrument.
show('07_REFERENCE_LINK_CHECK',pd.DataFrame([
    {'SOURCE':'LOANDEPO_VW','RFO_REFERENCE':'2110000000049295',
     'N_CORPORATE_INSTRUMENTS_USING_THIS_REF':int(keys.loc[keys.SOURCE.eq('LOANDEPO_VW') & keys.REF_KEY.eq('2110000000049295'),'INSTRUMENT_ID'].nunique())},
    {'SOURCE':'ACCOUNT_VW','RFO_REFERENCE':'2110000091015150',
     'N_CORPORATE_INSTRUMENTS_USING_THIS_REF':int(keys.loc[keys.SOURCE.eq('ACCOUNT_VW') & keys.REF_KEY.eq('2110000091015150'),'INSTRUMENT_ID'].nunique())}]))
print('\nFINAL_QA | This measures a full-population recovery CANDIDATE, not approved exposures.',flush=True)
print('FINAL_QA | EUR LoanDepo only for potential recovery; no overdraft or other currency auto-added.',flush=True)
print('FINAL_QA | 31-day vintage, regulatory crosswalk and RFO version/uniqueness checks applied.',flush=True)
print('FINAL_QA | NO WRITES. No training, datasets, recipes or production outputs changed.',flush=True)
log('FINISHED')
