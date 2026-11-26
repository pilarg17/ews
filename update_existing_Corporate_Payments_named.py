import ast, re, string, json
from pathlib import Path
import dataiku
import pandas as pd

PROJECT='VALIDATIONEWSCORPORATEPORTFOLIODAILY'
p=dataiku.api_client().get_project(PROJECT)
RESULT='results_Corporate_Payments'
RECIPE='compute_results_Corporate_Payments'
AM_RECIPE='compute_account_movements_signals_Corporate'
AM='account_movements_signals_Corporate'
AR='results_Corporate_Account_Movements'
MONTH='payment_movements_monthly_Corporate'
EXTRA={
 'compute_payment_extended_monthly_Corporate':'payment_extended_monthly_Corporate',
 'compute_payment_ranked_percentage_monthly_Corporate':'payment_ranked_percentage_monthly_Corporate'}
EXTRA_DS=set(EXTRA.values())
KEYS=['PORTFOLIO','INDICATOR','TARGET_SCOPE','THRESHOLD_A','THRESHOLD_B','THRESHOLD_C']
PORTS={'GLOBAL','LARGE_CORPORATE','MID_CORPORATE','OTHER_CORPORATE','ADC','IP'}
IN12='account_movements_incoming_12m_vs_prev12m_decline'
OUT12='account_movements_outgoing_12m_vs_prev12m_increase'

TOP_NAME_MAP={
 f'payment_top{n}{suffix}_{ref}': f'payment_top{n}{label}_{ref}'
 for n in (10,20,30)
 for suffix,label in (('_floor','pct_floor'),('_ceil','pct_ceil'),('','fixed'))
 for ref in ('prev3_mean','prev12_mean','prev12_max')
}
assert len(TOP_NAME_MAP)==27 and len(set(TOP_NAME_MAP.values()))==27
TOP_NAME_REVERSE={v:k for k,v in TOP_NAME_MAP.items()}

def read(n):
 x=dataiku.Dataset(n,project_key=PROJECT).get_dataframe();x.columns=x.columns.str.upper();return x

def settings(n): return p.get_recipe(n).get_settings()
def local(ref): return ref.split('.')[-1] if ref.startswith(PROJECT+'.') else ref

def literal(code,n):
 return next(ast.literal_eval(x.value) for x in ast.parse(code).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==n for t in x.targets))

def has_input(s,n): return n in {local(x) for x in s.get_flat_input_refs()}

def once(x,a,b):
 assert x.count(a)==1, 'Unexpected source structure: '+a[:100]
 return x.replace(a,b,1)

old=read(RESULT); old_am=read(AR)
old_original=old.copy()
old['INDICATOR']=old.INDICATOR.replace(TOP_NAME_REVERSE)
assert len(old) in (828,840,1092) and not old.duplicated(KEYS).any()
assert len(old_am)==12 and not old_am.duplicated(KEYS).any()
assert set(old.PORTFOLIO)==PORTS
keep=[f'{i}_{r}' for i in ['payment_turnover']+[f'payment_top{n}_{z}' for n in (10,20,30) for z in ('floor','ceil')]+[f'payment_top{n}' for n in (10,20,30)] for r in ('prev3_mean','prev12_mean','prev12_max')]
keep += [f'payment_{f}_{s}_vs_avg12_{d}' for f in ('incoming','outgoing','net','gross') for s in ('avg3','max3') for d in ('decline','increase') if not (f=='incoming' and d=='increase' or f=='outgoing' and d=='decline')]
new=[f'payment_{f}_{s}_vs_prev12_{r}_{d}' for f,r,d in [('incoming','min','decline'),('outgoing','max','increase')] for s in ('avg3','max3')]
names=keep+new+[IN12,OUT12]
assert len(keep)==42 and len(set(names))==48
ref=pd.concat([old[old.INDICATOR.isin(keep)],old_am[old_am.INDICATOR.eq(IN12)]],ignore_index=True)
assert len(ref)==762 and not ref.duplicated(KEYS).any()
assert set(ref.columns)==set(old.columns)
old_original.to_pickle('payments_before_simplification.pkl')
for n in (RECIPE,AM_RECIPE):
 s=settings(n)
 Path(n+'_before_simplification.py').write_text(s.get_code(),encoding='utf-8')
 Path(n+'_before_simplification.json').write_text(json.dumps(s.get_recipe_raw_definition(),default=str),encoding='utf-8')

# Reuse the existing monthly extraction code for the eight additional months only.
source=settings('compute_payment_movements_monthly_Corporate').get_code()
assert source.count('# Full consecutive months only.')==1
am_code=source.split('# Full consecutive months only.')[0]
am_code=once(am_code,"FIRST_MONTH='2024-09-01'","FIRST_MONTH='2024-01-01'")
am_code=once(am_code,"LAST_COMPLETE_MONTH='2026-09-01'","LAST_COMPLETE_MONTH='2024-08-01'")
am_code+=r'''
cols=['ENTITY_ID','PAYMENT_MONTH','AVAILABLE_DATE','INCOMING_EUR','OUTGOING_EUR']
old=dataiku.Dataset('payment_movements_monthly_Corporate',project_key=PROJECT).get_dataframe(columns=cols)
x=pd.concat([monthly[cols],old[cols]],ignore_index=True)
x.ENTITY_ID=x.ENTITY_ID.map(norm)
for c in ('PAYMENT_MONTH','AVAILABLE_DATE'):
 x[c]=pd.to_datetime(x[c],errors='raise',utc=True).dt.tz_localize(None).dt.normalize()
for c in ('INCOMING_EUR','OUTGOING_EUR'): x[c]=pd.to_numeric(x[c],errors='raise')
assert not x[cols].isna().any().any() and not x.duplicated(['ENTITY_ID','PAYMENT_MONTH']).any()
assert (x[['INCOMING_EUR','OUTGOING_EUR']]>=0).all().all()
assert set(x.PAYMENT_MONTH)==set(pd.date_range('2024-01-01','2026-09-01',freq='MS'))
assert x.AVAILABLE_DATE.eq(x.PAYMENT_MONTH+pd.offsets.MonthEnd(0)).all()
x=x.sort_values(['ENTITY_ID','PAYMENT_MONTH']).reset_index(drop=True)
x['MONTH_SERIAL']=x.PAYMENT_MONTH.dt.year*12+x.PAYMENT_MONTH.dt.month
g=x.groupby('ENTITY_ID',sort=False)
complete=x.MONTH_SERIAL.sub(g.MONTH_SERIAL.shift(23)).eq(23)
consecutive=x.MONTH_SERIAL.sub(g.MONTH_SERIAL.shift()).eq(1)
frames=[]
for f,c,side in [('incoming','INCOMING_EUR','decline'),('outgoing','OUTGOING_EUR','decline'),('outgoing','OUTGOING_EUR','increase')]:
 latest=g[c].transform(lambda s:s.rolling(12,min_periods=12).sum())
 prior=latest.groupby(x.ENTITY_ID,sort=False).shift(12)
 mag=(100*(1-latest/prior) if side=='decline' else 100*(latest/prior-1)).where(complete & prior.gt(0))
 prev=mag.groupby(x.ENTITY_ID,sort=False).shift()
 hit=consecutive & mag.gt(25) & prev.le(25)
 z=x.loc[hit,['ENTITY_ID','PAYMENT_MONTH','AVAILABLE_DATE']].copy()
 z['INDICATOR']='account_movements_'+f+'_12m_vs_prev12m_'+side
 z['THRESHOLD_A']=25.0; z['DECLINE_PCT']=mag.loc[hit].astype(float)
 frames.append(z)
 print('[QA]',z.INDICATOR.iloc[0] if len(z) else f+'_'+side,len(z),flush=True)
out=pd.concat(frames,ignore_index=True)
assert not out.duplicated(['ENTITY_ID','PAYMENT_MONTH','INDICATOR','THRESHOLD_A']).any()
for c in ('PAYMENT_MONTH','AVAILABLE_DATE'): out[c]=out[c].dt.strftime('%Y-%m-%d')
dataiku.Dataset('account_movements_signals_Corporate',project_key=PROJECT).write_with_schema(out)
print('FINISHED | Rolling 12M signals',flush=True)
'''

# Keep the existing portfolio/event/target/cost SQL unchanged.
current=settings(RECIPE).get_code(); q=literal(current,'SQL_TEMPLATE')
head=q[:q.index('FIXED_MONTHLY AS (')]
fixed_end=min(q.index(marker) for marker in ('EXT_MONTHLY AS (','MOV_LONG AS (','PAYMENT_STAGE_COMBINED AS (') if marker in q)
fixed=q[q.index('FIXED_MONTHLY AS ('):fixed_end]
fixed=re.sub(r'(FIXED_MONTHLY\s+AS\s*\(\s*SELECT)\s+(?:DISTINCT\s+)?',r'\1 DISTINCT ',fixed,count=1,flags=re.I)
tail=q[q.index('SIGNALS_PAYMENT AS ('):]
a=tail.index(' WHERE '); b=tail.index('   AND s.AVAILABLE_DATE>=',a)
tail=tail[:a]+''' WHERE EXISTS (SELECT 1 FROM GRID_PAYMENTS g WHERE g.INDICATOR=s.INDICATOR AND g.THRESHOLD_A=s.THRESHOLD_A)\n'''+tail[b:]
W='PARTITION BY ENTITY_ID,FAMILY ORDER BY PAYMENT_MONTH'
flow=f'''
MOV_LONG AS (
 SELECT TRIM(ENTITY_ID::VARCHAR) ENTITY_ID,TO_DATE(PAYMENT_MONTH) PAYMENT_MONTH,TO_DATE(AVAILABLE_DATE) AVAILABLE_DATE,'incoming' FAMILY,INCOMING_EUR AMOUNT FROM {{monthly}}
 UNION ALL SELECT TRIM(ENTITY_ID::VARCHAR),TO_DATE(PAYMENT_MONTH),TO_DATE(AVAILABLE_DATE),'outgoing',OUTGOING_EUR FROM {{monthly}}
), MOV_HIST AS (
 SELECT *, LAG(PAYMENT_MONTH,14) OVER({W}) M14,
 COUNT(AMOUNT) OVER({W} ROWS BETWEEN 14 PRECEDING AND CURRENT ROW) N15,
 AVG(AMOUNT) OVER({W} ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) RAVG,
 MAX(AMOUNT) OVER({W} ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) RMAX,
 MIN(AMOUNT) OVER({W} ROWS BETWEEN 14 PRECEDING AND 3 PRECEDING) BMIN,
 MAX(AMOUNT) OVER({W} ROWS BETWEEN 14 PRECEDING AND 3 PRECEDING) BMAX FROM MOV_LONG
), MOV_FEATURES AS (
'''
selects=[]
for f,r,d,base in [('incoming','min','decline','BMIN'),('outgoing','max','increase','BMAX')]:
 for s,recent in [('avg3','RAVG'),('max3','RMAX')]:
  expr=f'100*(1-{recent}/NULLIF({base},0))' if d=='decline' else f'100*({recent}/NULLIF({base},0)-1)'
  selects.append(f"SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,'payment_{f}_{s}_vs_prev12_{r}_{d}' INDICATOR,IFF(N15=15 AND M14=DATEADD('MONTH',-14,PAYMENT_MONTH) AND {base}>0,{expr},NULL) MAG FROM MOV_HIST WHERE FAMILY='{f}'")
flow+='\n UNION ALL\n'.join(selects)+'''
), MOV_PREV AS (
 SELECT *,LAG(PAYMENT_MONTH) OVER(PARTITION BY ENTITY_ID,INDICATOR ORDER BY PAYMENT_MONTH) PM,
 LAG(MAG) OVER(PARTITION BY ENTITY_ID,INDICATOR ORDER BY PAYMENT_MONTH) PREV_MAG FROM MOV_FEATURES
), MOV_SIGNALS AS (
 SELECT m.ENTITY_ID,m.PAYMENT_MONTH,m.AVAILABLE_DATE,m.INDICATOR,g.THRESHOLD_A
 FROM MOV_PREV m JOIN GRID_PAYMENTS g ON m.INDICATOR=g.INDICATOR
 WHERE PM=DATEADD('MONTH',-1,m.PAYMENT_MONTH) AND MAG>=g.THRESHOLD_A AND PREV_MAG<g.THRESHOLD_A
), PAYMENT_STAGE_COMBINED AS (
 SELECT TRIM(ENTITY_ID::VARCHAR) ENTITY_ID,TO_DATE(PAYMENT_MONTH) PAYMENT_MONTH,TO_DATE(AVAILABLE_DATE) AVAILABLE_DATE,INDICATOR,TRY_TO_DOUBLE(THRESHOLD_A::VARCHAR) THRESHOLD_A FROM {legacy_stage}
 UNION ALL SELECT TRIM(ENTITY_ID::VARCHAR),TO_DATE(PAYMENT_MONTH),TO_DATE(AVAILABLE_DATE),INDICATOR,TRY_TO_DOUBLE(THRESHOLD_A::VARCHAR) FROM {new_stage}
 UNION ALL SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,INDICATOR,THRESHOLD_A FROM FIXED_SIGNALS
 UNION ALL SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,INDICATOR,THRESHOLD_A FROM MOV_SIGNALS
 UNION ALL SELECT TRIM(ENTITY_ID::VARCHAR),TO_DATE(PAYMENT_MONTH),TO_DATE(AVAILABLE_DATE),INDICATOR,TRY_TO_DOUBLE(THRESHOLD_A::VARCHAR) FROM {account_stage}
),
'''
new_sql=head+fixed+flow+tail
values=','.join(f"('{n}',{t})" for n in names for t in ((25,) if n in (IN12,OUT12) else (20,30,50)))
prefix=current.split('OLD=physical(')[0]
assert 'def physical(' in prefix and 'F=physical(' in prefix
result_code=prefix+f'''
OLD=physical('payment_signals_Corporate')
NEW=physical('payment_movements_signals_Corporate')
M=physical('payment_fixed_top_monthly_Corporate')
X=physical({MONTH!r})
A=physical({AM!r})
AR=physical({AR!r})
O=physical(OUTPUT)
from dataiku import recipe
assert {{str(getattr(x,'name',x)).split('.')[-1] for x in recipe.get_outputs(object_type='DATASET')}}=={{OUTPUT}}
assert O.upper() not in {{x.upper() for x in (T,B,L,E,H,F,OLD,NEW,M,X,A,AR,physical('results_EWS_Corporate'),physical('results_Corporate_EWS_validation'))}}
SQL_TEMPLATE={new_sql!r}
PAYMENT_GRID_VALUES={values!r}
NAMES={names!r}
KEEP={keep!r}
IN12={IN12!r}
KEYS={KEYS!r}
PORTS={sorted(PORTS)!r}
TOP_NAME_MAP={TOP_NAME_MAP!r}
'''+r'''
engine=SQLExecutor2(connection=CONNECTION)
def query(q):
 d=engine.query_to_df(q);d.columns=d.columns.str.upper();return d

def unique(table,cols):
 assert int(query('SELECT COUNT(*) FROM (SELECT '+cols+',COUNT(*) N FROM '+table+' GROUP BY '+cols+' HAVING COUNT(*)>1)').iloc[0,0])==0,'Duplicate keys: '+table
fields='ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,MAPPING_SOURCE_DATE,INCOMING_EUR,N_INCOMING,FIXED_INCOMING_EUR,FIXED_N_INCOMING,TOP10_EUR,TOP20_EUR,TOP30_EUR,N_CONFLICTS,IS_SOURCE_RECONCILED'
unique('(SELECT DISTINCT '+fields+' FROM '+M+')','ENTITY_ID,PAYMENT_MONTH')
unique(X,'ENTITY_ID,PAYMENT_MONTH')
unique(A,'ENTITY_ID,PAYMENT_MONTH,INDICATOR,THRESHOLD_A')
assert int(query('SELECT COUNT(DISTINCT TO_DATE(PAYMENT_MONTH)) FROM '+X).iloc[0,0])==25
assert int(query('SELECT COUNT(DISTINCT TO_DATE(PAYMENT_MONTH)) FROM '+M+' WHERE IS_SOURCE_RECONCILED=1').iloc[0,0])==25
old=query('SELECT * FROM '+O); ar=query('SELECT * FROM '+AR)
old['INDICATOR']=old.INDICATOR.replace({v:k for k,v in TOP_NAME_MAP.items()})
ref=pd.concat([old[old.INDICATOR.isin(KEEP)],ar[ar.INDICATOR.eq(IN12)]],ignore_index=True)
assert len(ref)==762 and not ref.duplicated(KEYS).any()
sql=SQL_TEMPLATE.format(T=T,B=B,L=L,E=E,H=H,F=F,start=START_DATE,end=END_DATE,adc=nums(ADC_CODES),ip=nums(IP_CODES),other=nums(OTHER_CODES),lead_lo=LEAD_MIN_MONTHS,lead_hi=LEAD_MAX_MONTHS,payment_grid_values=PAYMENT_GRID_VALUES,legacy_stage=OLD,new_stage=NEW,fixed_monthly=M,monthly=X,account_stage=A)
c=query(sql)
assert len(c)==840 and set(c.INDICATOR)==set(NAMES) and set(c.PORTFOLIO)==set(PORTS)
assert not c.duplicated(KEYS).any() and c.groupby('PORTFOLIO').size().eq(140).all()
assert set(c.columns)==set(old.columns)==set(ref.columns)
assert (c.N_TP+c.N_FN==c.N_EVENTS).all() and (c.N_TP+c.N_FP==c.N_ALERTS).all()
pd.testing.assert_frame_equal(ref[old.columns].sort_values(KEYS).reset_index(drop=True),c[c.INDICATOR.isin(KEEP+[IN12])][old.columns].sort_values(KEYS).reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-7)
print('[PASS] 762 retained configurations unchanged',flush=True)
published=c[old.columns].copy()
published['INDICATOR']=published.INDICATOR.replace(TOP_NAME_MAP)
assert published.INDICATOR.nunique()==48 and not published.duplicated(KEYS).any()
assert published.INDICATOR.isin(TOP_NAME_MAP.values()).sum()==486
qa=published.copy(); qa['INDICATOR']=qa.INDICATOR.replace({v:k for k,v in TOP_NAME_MAP.items()})
pd.testing.assert_frame_equal(qa,c[old.columns])
dataiku.Dataset(OUTPUT,project_key=PROJECT).write_dataframe(published,infer_schema=False)
print('[FINISHED] 48 indicators | 840 configurations',flush=True)
'''
compile(am_code,AM_RECIPE,'exec');compile(result_code,RECIPE,'exec')
new_sql.format(**{k:'X' for k in ('T','B','L','E','H','F','start','end','adc','ip','other','lead_lo','lead_hi','payment_grid_values','legacy_stage','new_stage','fixed_monthly','monthly','account_stage')})
# Check all other recipe dependencies before making changes or deleting anything.
known={r['name'] for r in p.list_recipes()}
for n in known:
 s=settings(n)
 refs={local(x) for x in s.get_flat_input_refs()+s.get_flat_output_refs()}
 if n not in set(EXTRA)|{RECIPE}: assert not refs&EXTRA_DS, 'Extra dataset used by '+n
 if n in EXTRA: assert {local(x) for x in s.get_flat_output_refs()}=={EXTRA[n]}, 'Unexpected output: '+n
s=settings(AM_RECIPE)
assert {local(x) for x in s.get_flat_output_refs()}=={AM}
assert has_input(s,MONTH)
s.set_code(am_code);s.save()
print('RUNNING | '+AM_RECIPE,flush=True)
p.get_recipe(AM_RECIPE).run(job_type='NON_RECURSIVE_FORCED_BUILD',wait=True)
s=settings(RECIPE)
for role in s.get_recipe_raw_definition()['inputs'].values():
 role['items'][:]=[i for i in role['items'] if local(i['ref']) not in EXTRA_DS]
for inp in (MONTH,AM,AR):
 if not has_input(s,inp):s.add_input('main',inp)
s.set_code(result_code);s.save()
assert not ({local(x) for x in settings(RECIPE).get_flat_input_refs()}&EXTRA_DS)
print('RUNNING | '+RECIPE,flush=True)
p.get_recipe(RECIPE).run(job_type='NON_RECURSIVE_FORCED_BUILD',wait=True)
result=read(RESULT)
assert result.INDICATOR.isin(TOP_NAME_MAP.values()).sum()==486
result['INDICATOR']=result.INDICATOR.replace(TOP_NAME_REVERSE)
assert len(result)==840 and result.INDICATOR.nunique()==48 and not result.duplicated(KEYS).any()
pd.testing.assert_frame_equal(ref[result.columns].sort_values(KEYS).reset_index(drop=True),result[result.INDICATOR.isin(keep+[IN12])].sort_values(KEYS).reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-7)
print('PASS | 840 configurations / 48 indicators',flush=True)
# Remove only the two extra recipes and their dataset definitions. Retain physical tables.
for n in known-set(EXTRA):
 s=settings(n)
 assert not ({local(x) for x in s.get_flat_input_refs()+s.get_flat_output_refs()}&EXTRA_DS), 'Still in use: '+n
for n,d in EXTRA.items():
 if n in {r['name'] for r in p.list_recipes()}:p.get_recipe(n).delete();print('REMOVED RECIPE | '+n)
 if d in {x['name'] for x in p.list_datasets()}:p.get_dataset(d).delete(drop_data=False);print('REMOVED DATASET | '+d)
print(result.loc[result.PORTFOLIO.eq('GLOBAL')&result.INDICATOR.isin(new+[IN12,OUT12]),['INDICATOR','THRESHOLD_A','N_TP','N_FP','N_ALERTS','ACCURACY','COVERAGE']].to_string(index=False))
print('FINISHED | Existing Payments updated; additional Flow objects removed')
