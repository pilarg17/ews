# RETAIL EWS - DPD > 15 | AUDITORIA MULTIFECHA INTEGRADA | SOLO LECTURA
# Ejecutar como UNA celda en el notebook de Dataiku.
# No ejecuta el Flow, no crea datasets, no escribe tablas.
import re
import pandas as pd
import numpy as np
import dataiku
from dataiku.core.sql import SQLExecutor2
from IPython.display import display

PROJECTS = [
    'TEST15_CREDITRISKEARLYWARNINGSYSTEMRETAILPORTFOLIO',
    'CREDIT_RISK_EARLY_WARNING_SYSTEM_RETAIL_PORTFOLIO_FEATURE_CREDITRISKEARLYWARNINGSYSTEMRETAILPORTFOLIO',
]
DAYS = [('2026-10-15','2026-10-14'),('2026-10-16','2026-10-15'),('2026-10-19','2026-10-16')]
ALL_DATES = sorted({x for pair in DAYS for x in pair})
GEC_LIMITS = [1_000_000,750_000,500_000,250_000]
EXPECTED_15 = {'DPD_GT_0':1925,'DPD_GT_15':1335,'RETAIL_CROSSINGS':10,
               'RRE_CROSSINGS':3,'OTHER_RETAIL_CROSSINGS':7,'AFTER_OD_500':3,
               'GEC_GT_1000000':0,'GEC_GT_750000':0,'GEC_GT_500000':0,
               'GEC_GT_250000':1}
# These are from the PREVIOUS multifecha notebook result, for comparative QA only.
OLD_ORACLE = {'2026-10-15':(4353,79,8,2),
              '2026-10-16':(4596,281,None,None),
              '2026-10-19':(4990,449,None,None)}
ALIASES = {
 'ENTITY_ID': ['ENTITY_ID','_ENTITY','COUNTERPARTY_ID','CUSTOMER_ID'],
 'MAP_DTE':['MAP_DTE','REPORT_DATE','AS_OF_DATE','SNAPSHOT_DATE'],
 'DPD':['DPD','DPD_MAX','MAX_DPD','DAYS_PAST_DUE'],
 'OVERDRAFT_AMOUNT':['OVERDRAFT_AMOUNT','OD_AMOUNT'],
 'GEC_NET_EXPOSURE':['GEC_NET_EXPOSURE','GEC_EXPOSURE','GEC_TOTAL_EXPOSURE'],
 'RRE_FLAG':['RRE_FLAG','IS_RRE'],
 'OTHER_RETAIL_FLAG':['OTHER_RETAIL_FLAG','IS_OTHER_RETAIL']
}
DS_NAMES = [
 'Available_Indicators_Current','Available_Indicators_Previous',
 'Available_Indicators_Daily','Retail_Portfolio_Daily',
 'DPD_Arrears_Daily_Entity','DPD_Arrears_Daily_Loan',
 'Customer_Balances_Daily','Customer_Balances_Current',
 'GEC_Exposure_Daily','GEC_Exposure_Current'
]

client = dataiku.api_client()
projects = {}
for pk in PROJECTS:
    try:
        p = client.get_project(pk)
        names = {x['name'] for x in p.list_datasets()}
        projects[pk] = (p, names)
    except Exception as err:
        print('Project unavailable:',pk, str(err)[:140])
if not projects: raise RuntimeError('No se puede acceder a ninguno de los proyectos Retail.')

def present(df):
    if df is None: return pd.DataFrame(columns=['ENTITY_ID','MAP_DTE'])
    x = df.copy()
    x.columns = [str(c).upper() for c in x.columns]
    rename = {}
    for canonical, options in ALIASES.items():
        if canonical not in x.columns:
            for option in options:
                if option in x.columns:
                    rename[option] = canonical
                    break
    x = x.rename(columns=rename)
    if 'MAP_DTE' in x.columns: x['MAP_DTE'] = pd.to_datetime(x['MAP_DTE'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
    if 'ENTITY_ID' in x.columns: x['ENTITY_ID'] = x['ENTITY_ID'].astype('string').str.strip()
    for c in ('DPD','OVERDRAFT_AMOUNT','GEC_NET_EXPOSURE'):
        if c in x.columns: x[c] = pd.to_numeric(x[c],errors='coerce')
    for c in ('RRE_FLAG','OTHER_RETAIL_FLAG'):
        if c in x.columns:
            x[c] = x[c].astype('string').str.upper().str.strip().map({'TRUE':1,'FALSE':0,'1':1,'0':0,'YES':1,'NO':0})
    return x

def read_dataset(pk, p, name):
    ds = dataiku.Dataset(name, project_key=pk)
    sc = ds.read_schema()
    names = {str(c['name']).upper():str(c['name']) for c in sc}
    wanted = []
    for candidates in ALIASES.values():
        for n in candidates:
            if n in names:
                wanted.append(names[n]);break
    if not (any(n in names for n in ALIASES['ENTITY_ID']) and any(n in names for n in ALIASES['MAP_DTE'])):
        return None, 'Missing entity/date fields'
    s = p.get_dataset(name).get_settings().get_raw()
    prm = s.get('params') or {}
    conn, table, schema = prm.get('connection'),prm.get('table'),prm.get('schema')
    table = str(table or '').replace('${projectKey}',pk)
    schema = str(schema or '').replace('${projectKey}',pk)
    def quote(v):
        if not re.fullmatch(r'[A-Za-z0-9_]+',str(v)): raise ValueError('Unexpected SQL identifier')
        return '"'+str(v)+'"'
    date_col = next(names[n] for n in ALIASES['MAP_DTE'] if n in names)
    days = ','.join("DATE '"+v+"'" for v in ALL_DATES)
    if conn and table and schema:
        try:
            ref = quote(schema)+'.'+quote(table)
            if prm.get('database'):
                ref = quote(prm['database'])+'.'+ref
            sql = ('SELECT '+', '.join(quote(n) for n in dict.fromkeys(wanted))+
                   ' FROM '+ref+' WHERE CAST('+quote(date_col)+' AS DATE) IN ('+days+')')
            df = SQLExecutor2(connection=conn).query_to_df(sql)
            return present(df), 'SQL pushdown: '+str(conn)
        except Exception as err:
            sql_err = str(err)[:170]
    else:
        sql_err = 'not a SQL-table dataset'
    # The two snapshot datasets are intentionally small enough to read once.
    if name in ('Available_Indicators_Current','Available_Indicators_Previous'):
        try:
            return present(ds.get_dataframe(columns=list(dict.fromkeys(wanted)))), 'Dataiku dataframe (snapshot)'
        except Exception as err:
            return None, 'Read error: '+str(err)[:160]
    return None, 'SQL read not possible: '+sql_err

found = {}; log=[]
for pk,(p,names) in projects.items():
    for dsname in DS_NAMES:
        if dsname not in names: continue
        try:
            df, method = read_dataset(pk,p,dsname)
            if df is not None and not df.empty and {'ENTITY_ID','MAP_DTE'}.issubset(df.columns):
                df = df[df['MAP_DTE'].isin(ALL_DATES)].copy()
                for d,g in df.groupby('MAP_DTE'):
                    found[(pk,dsname,d)] = g
            log.append((dsname,pk,len(df) if df is not None else None,method))
        except Exception as err:
            log.append((dsname,pk,None,'Error: '+str(err)[:180]))

logdf = pd.DataFrame(log,columns=['DATASET','PROJECT','ROWS_IN_DATES','READ_STATUS'])
print('='*102)
print('RETAIL EWS | DPD > 15 | HISTORICAL MULTI-DATE | READ ONLY')
print('Dates: 15, 16 and 19 October 2026 | Previous working snapshots: 14, 15 and 16 October')
print('No Flow runs, no output writes, no imputation of missing Retail/GEC history.')
print('='*102)
print('\n0. SOURCE INVENTORY / DATE AVAILABILITY')
display(logdf)

# Merge duplicate entity/day observations in source, always report duplicates.
def dedup(df):
    if df is None or df.empty or 'ENTITY_ID' not in df.columns: return None, 0
    df = df[df['ENTITY_ID'].notna()].copy()
    duplicates = int(df.duplicated(['ENTITY_ID','MAP_DTE']).sum())
    out = df.groupby(['ENTITY_ID','MAP_DTE'],as_index=False).agg({
        c:'max' for c in df.columns if c not in ('ENTITY_ID','MAP_DTE')
    })
    return out, duplicates

def choose(date, col, preferred):
    options=[]
    for ix,name in enumerate(preferred):
        for pk in PROJECTS:
            df=found.get((pk,name,date))
            if df is not None and col in df.columns and df[col].notna().any():
                options.append((ix,pk,name,df))
    if not options: return None,'MISSING',0
    _,pk,name,df=options[0]
    clean,duplicates=dedup(df[['ENTITY_ID','MAP_DTE',col]].copy())
    return clean[['ENTITY_ID',col]],pk+'/'+name,duplicates

# Exact-date priority: archived indicator snapshots, then genuine daily sources.
DPD_SOURCES=['Available_Indicators_Current','Available_Indicators_Previous',
             'Available_Indicators_Daily','DPD_Arrears_Daily_Entity',
             'DPD_Arrears_Daily_Loan']
SCOPE_SOURCES=['Available_Indicators_Current','Available_Indicators_Previous',
               'Available_Indicators_Daily','Retail_Portfolio_Daily']
OD_SOURCES=['Available_Indicators_Current','Available_Indicators_Previous',
            'Available_Indicators_Daily','Customer_Balances_Daily','Customer_Balances_Current']
GEC_SOURCES=['Available_Indicators_Current','Available_Indicators_Previous',
             'Available_Indicators_Daily','GEC_Exposure_Daily','GEC_Exposure_Current']

def describe(num): return 'N/A' if num is None or pd.isna(num) else int(num)

outrows=[]; source_rows=[]; crossing_sets={}; mismatch_rows=[]
for day,prev in DAYS:
    cur,cs,cd=choose(day,'DPD',DPD_SOURCES)
    old,ps,pd=choose(prev,'DPD',DPD_SOURCES)
    rr,rs,rd=choose(day,'RRE_FLAG',SCOPE_SOURCES)
    oth,os,odup=choose(day,'OTHER_RETAIL_FLAG',SCOPE_SOURCES)
    overdraft,ods,odd=choose(day,'OVERDRAFT_AMOUNT',OD_SOURCES)
    gec,gs,gd=choose(day,'GEC_NET_EXPOSURE',GEC_SOURCES)
    source_rows.append(dict(DATE=day,DPD_SOURCE=cs,PREV_DPD_SOURCE=ps,
                            RRE_SOURCE=rs,OTHER_SOURCE=os,OD_SOURCE=ods,GEC_SOURCE=gs,
                            DPD_DUPLICATE_KEYS=cd,PREV_DPD_DUPLICATE_KEYS=pd,
                            RRE_DUPLICATE_KEYS=rd,OTHER_DUPLICATE_KEYS=odup,
                            OD_DUPLICATE_KEYS=odd,GEC_DUPLICATE_KEYS=gd))
    if cur is None:
        outrows.append(dict(DATE=day,PREVIOUS=prev,STATUS='DPD_HISTORY_MISSING'))
        continue
    panel=cur.rename(columns={'DPD':'DPD_CURRENT'}).copy()
    if old is not None: panel=panel.merge(old.rename(columns={'DPD':'DPD_PREVIOUS'}),on='ENTITY_ID',how='left',validate='one_to_one')
    else: panel['DPD_PREVIOUS']=np.nan
    for df,col in ((rr,'RRE_FLAG'),(oth,'OTHER_RETAIL_FLAG'),(overdraft,'OVERDRAFT_AMOUNT'),(gec,'GEC_NET_EXPOSURE')):
        panel=panel.merge(df,on='ENTITY_ID',how='left',validate='one_to_one') if df is not None else panel.assign(**{col:np.nan})
    panel['RAW_CROSS']=panel.DPD_CURRENT.gt(15)&panel.DPD_PREVIOUS.le(15)&panel.DPD_PREVIOUS.notna()
    scope_ok=(rr is not None and oth is not None)
    panel['IN_RETAIL']=(panel.RRE_FLAG.eq(1)|panel.OTHER_RETAIL_FLAG.eq(1)).fillna(False) if scope_ok else False
    panel['RETAIL_CROSS']=panel.RAW_CROSS&panel.IN_RETAIL
    panel['OD_CROSS']=panel.RETAIL_CROSS&panel.OVERDRAFT_AMOUNT.gt(500).fillna(False)
    crossing_sets[day]=panel.copy()
    n=lambda mask: int(mask.sum())
    result=dict(DATE=day,PREVIOUS=prev,
        STATUS='|'.join([s for s,v in [('PREV_DPD_MISSING',old is None),
             ('RETAIL_SCOPE_MISSING',not scope_ok),('OD_MISSING',overdraft is None),
             ('GEC_MISSING',gec is None),('PARTIAL_GEC',gec is not None and panel.loc[panel.OD_CROSS,'GEC_NET_EXPOSURE'].isna().any())] if v]) or 'ALL_SOURCES_PRESENT',
        DPD_KNOWN=n(panel.DPD_CURRENT.notna()),
        DPD_GT_0=n(panel.DPD_CURRENT.gt(0)),DPD_GT_15=n(panel.DPD_CURRENT.gt(15)),
        DPD_PREVIOUS_MISSING=n(panel.DPD_PREVIOUS.isna()),
        RAW_CROSSINGS=n(panel.RAW_CROSS) if old is not None else None,
        RETAIL_CROSSINGS=n(panel.RETAIL_CROSS) if old is not None and scope_ok else None,
        RRE_CROSSINGS=n(panel.RETAIL_CROSS&panel.RRE_FLAG.eq(1).fillna(False)) if old is not None and scope_ok else None,
        OTHER_RETAIL_CROSSINGS=n(panel.RETAIL_CROSS&panel.OTHER_RETAIL_FLAG.eq(1).fillna(False)) if old is not None and scope_ok else None,
        RETAIL_OVERLAP=n(panel.RETAIL_CROSS&panel.RRE_FLAG.eq(1).fillna(False)&panel.OTHER_RETAIL_FLAG.eq(1).fillna(False)) if old is not None and scope_ok else None,
        AFTER_OD_500=n(panel.OD_CROSS) if old is not None and scope_ok and overdraft is not None else None,
        REMOVED_BY_OD=n(panel.RETAIL_CROSS)-n(panel.OD_CROSS) if old is not None and scope_ok and overdraft is not None else None,
        GEC_AVAILABLE=n(panel.OD_CROSS&panel.GEC_NET_EXPOSURE.notna()) if old is not None and scope_ok and overdraft is not None and gec is not None else None)
    for lim in GEC_LIMITS:
        key='GEC_GT_'+str(lim)
        subset=panel.loc[panel.OD_CROSS]
        # Complete ONLY when all qualifying OD crossings have an exact-date GEC.
        result[key]=(n(panel.OD_CROSS&panel.GEC_NET_EXPOSURE.gt(lim))
                     if gec is not None and old is not None and scope_ok and overdraft is not None
                     and subset.GEC_NET_EXPOSURE.notna().all() else None)
    outrows.append(result)

summary=pd.DataFrame(outrows)
print('\n1. DATA SOURCE TRACEABILITY (EXACT DATE, NO AS-OF SUBSTITUTION)')
display(pd.DataFrame(source_rows))
print('\n2. ALL DATES - SAME RETAIL DPD 15 FORMAT')
fields=['DATE','PREVIOUS','STATUS','DPD_GT_0','DPD_GT_15','RAW_CROSSINGS',
        'RETAIL_CROSSINGS','RRE_CROSSINGS','OTHER_RETAIL_CROSSINGS','RETAIL_OVERLAP',
        'REMOVED_BY_OD','AFTER_OD_500','GEC_AVAILABLE'] + ['GEC_GT_'+str(x) for x in GEC_LIMITS]
display(summary.reindex(columns=fields).fillna('N/A'))

for day,prev in DAYS:
    print('\n'+'='*93+'\nDPD > 15 | '+day+' versus '+prev+' | RETAIL FUNNEL\n'+'='*93)
    rows=summary[summary.DATE.eq(day)]
    if rows.empty: continue
    r=rows.iloc[0]
    for key,label in [
        ('DPD_GT_0','DPD > 0'),('DPD_GT_15','DPD > 15 (active)'),
        ('RAW_CROSSINGS','Raw crossings: previous <=15 and current >15'),
        ('RETAIL_CROSSINGS','Crossings in Retail scope'),
        ('RRE_CROSSINGS','  RRE'),('OTHER_RETAIL_CROSSINGS','  Other Retail'),
        ('RETAIL_OVERLAP','  RRE / Other Retail overlap'),
        ('REMOVED_BY_OD','Removed by overdraft <= EUR 500'),
        ('AFTER_OD_500','Remaining after overdraft > EUR 500'),
        ('GEC_AVAILABLE','GEC known among remaining candidates'),
        ('GEC_GT_1000000','  GEC > EUR 1,000,000'),
        ('GEC_GT_750000','  GEC > EUR 750,000'),
        ('GEC_GT_500000','  GEC > EUR 500,000'),
        ('GEC_GT_250000','  GEC > EUR 250,000')]:
        v=r.get(key,None)
        print(f'{label:<57} {"N/A" if pd.isna(v) else int(v)}')
    print('SOURCE STATUS:',r.STATUS)
    if day in crossing_sets:
        x=crossing_sets[day]
        print('DETAIL - Retail crossings (IDs and amounts, no client names):')
        if r.get('RETAIL_CROSSINGS') is not None and pd.notna(r.get('RETAIL_CROSSINGS')):
            detail=x.loc[x.RETAIL_CROSS,[
                'ENTITY_ID','DPD_PREVIOUS','DPD_CURRENT','RRE_FLAG','OTHER_RETAIL_FLAG',
                'OVERDRAFT_AMOUNT','GEC_NET_EXPOSURE','OD_CROSS']].copy()
            detail['GEC_250K']=np.where(detail.GEC_NET_EXPOSURE.notna(),detail.GEC_NET_EXPOSURE.gt(250000),pd.NA)
            display(detail.sort_values(['OD_CROSS','DPD_CURRENT'],ascending=False).reset_index(drop=True))
        else:
            print('N/A: Retail scope missing, cannot classify raw crossings as Retail.')

print('\n3. CONTROL - EXPECTED 15 OCTOBER RESULTS FROM THE ORIGINAL RETAIL NOTEBOOK')
ctl=summary.loc[summary.DATE.eq('2026-10-15')]
if not ctl.empty:
    actual=ctl.iloc[0]
    checks=[]
    for k,target in EXPECTED_15.items():
        got=actual.get(k,None)
        checks.append({'METRIC':k,'ORIGINAL_15':target,'THIS_RUN':describe(got),
                       'DELTA':None if got is None or pd.isna(got) else int(got)-target,
                       'MATCH':'N/A' if got is None or pd.isna(got) else ('YES' if got==target else 'NO')})
    display(pd.DataFrame(checks))
print('If there are mismatches, compare source snapshots, population, flags, DPD aggregation and previous-date mapping first.')

print('\n4. CROSS-CHECK AGAINST PREVIOUS ORACLE HISTORICAL CELL (REFERENCE, NOT RECOMPUTED)')
rows=[]
for day,(active,raw,retail,od) in OLD_ORACLE.items():
    our=summary.loc[summary.DATE.eq(day)]
    now=our.iloc[0] if not our.empty else pd.Series(dtype='object')
    rows.append({'DATE':day,'ORACLE_GT15_PREVIOUS_RUN':active,'ORACLE_RAW_PREVIOUS_RUN':raw,
                 'ORACLE_RETAIL_PREVIOUS_RUN':retail,'ORACLE_OD_PREVIOUS_RUN':od,
                 'CURRENT_DPD_GT15':now.get('DPD_GT_15',None),
                 'CURRENT_RAW_CROSSINGS':now.get('RAW_CROSSINGS',None),
                 'CURRENT_RETAIL_CROSSINGS':now.get('RETAIL_CROSSINGS',None),
                 'CURRENT_AFTER_OD':now.get('AFTER_OD_500',None)})
display(pd.DataFrame(rows).fillna('N/A'))
print('\nIMPORTANT: Oracle counts and Retail Indicator counts have different populations. Do not equate them.')
print('N/A means the required dated data is absent, NOT zero alerts. Historical sensitivity candidates are NOT final EWS breaches.')
print('This cell did not write any dataset, table, file or Flow output.')

print('\n5. 15 OCTOBER: SOURCE-LEVEL RECONCILIATION (WHEN BOTH HISTORIES ARE AVAILABLE)')
# Compare stored Retail indicator DPD with a separately materialized historical
# DPD entity source; show IDs only for mismatched records, not customer names.
def directly_from(name, date, col):
    for pk in PROJECTS:
        df = found.get((pk,name,date))
        if df is not None and col in df.columns and df[col].notna().any():
            x,_ = dedup(df[['ENTITY_ID','MAP_DTE',col]])
            return x[['ENTITY_ID',col]]
    return None

cur15 = directly_from('Available_Indicators_Current','2026-10-15','DPD')
pre14 = directly_from('Available_Indicators_Previous','2026-10-14','DPD')
hist15 = directly_from('DPD_Arrears_Daily_Entity','2026-10-15','DPD')
hist14 = directly_from('DPD_Arrears_Daily_Entity','2026-10-14','DPD')
if all(x is not None for x in [cur15,pre14,hist15,hist14]):
    legacy=cur15.rename(columns={'DPD':'CUR_RETAIL'}).merge(
        pre14.rename(columns={'DPD':'PREV_RETAIL'}),on='ENTITY_ID',how='outer')
    historical=hist15.rename(columns={'DPD':'CUR_HIST'}).merge(
        hist14.rename(columns={'DPD':'PREV_HIST'}),on='ENTITY_ID',how='outer')
    z=legacy.merge(historical,on='ENTITY_ID',how='outer',indicator=True)
    z['RETAIL_CROSS']=z.PREV_RETAIL.le(15)&z.CUR_RETAIL.gt(15)
    z['HIST_CROSS']=z.PREV_HIST.le(15)&z.CUR_HIST.gt(15)
    z['CROSS_DISAGREEMENT']=(z.RETAIL_CROSS!=z.HIST_CROSS)
    z['DPD_DISAGREEMENT']=(z.CUR_RETAIL.ne(z.CUR_HIST).fillna(False)|
                           z.PREV_RETAIL.ne(z.PREV_HIST).fillna(False))
    print('Current/previous Retail vs historical entity-day DPD:')
    display(pd.DataFrame([{
        'RETAIL_CROSSINGS_DPD_ONLY':int(z.RETAIL_CROSS.sum()),
        'HIST_CROSSINGS_DPD_ONLY':int(z.HIST_CROSS.sum()),
        'CROSSINGS_NOT_AGREEING':int(z.CROSS_DISAGREEMENT.sum()),
        'DPD_NOT_AGREEING':int(z.DPD_DISAGREEMENT.sum()),
        'RETAIL_ONLY_ENTITIES':int((z._merge=='left_only').sum()),
        'HIST_ONLY_ENTITIES':int((z._merge=='right_only').sum())
    }]))
    display(z.loc[z.CROSS_DISAGREEMENT|z.DPD_DISAGREEMENT,
                  ['ENTITY_ID','CUR_RETAIL','PREV_RETAIL','CUR_HIST','PREV_HIST',
                   'RETAIL_CROSS','HIST_CROSS','_merge']].head(100))
    print('This contrasts raw DPD crossings; the 8 vs 10 Retail-scoped difference')
    print('also requires the exact-date portfolio, overdraft and GEC filters.')
else:
    print('Not evaluable: the 14/15 snapshot pair and an independent historical')
    print('entity-level DPD pair are not both accessible. No entity-level cause invented.')
