import ast
import re
import dataiku
import pandas as pd

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONNECTION = 'DEV_AA_TEAMS'
p = dataiku.api_client().get_project(PROJECT)
RESULT = 'results_Corporate_Payments'
RECIPE = 'compute_results_Corporate_Payments'
EXT = 'payment_extended_monthly_Corporate'
RANK = 'payment_ranked_percentage_monthly_Corporate'
MONTH = 'payment_movements_monthly_Corporate'
FIXED = 'payment_fixed_top_monthly_Corporate'
ACCOUNT = 'account_movements_signals_Corporate'
ACCOUNT_RESULT = 'results_Corporate_Account_Movements'
KEYS = ['PORTFOLIO','INDICATOR','TARGET_SCOPE','THRESHOLD_A','THRESHOLD_B','THRESHOLD_C']
PORTS = {'GLOBAL','LARGE_CORPORATE','MID_CORPORATE','OTHER_CORPORATE','ADC','IP'}

# Read and retain the previous result and code before changing anything.
def read(name, **kwargs):
    x = dataiku.Dataset(name, project_key=PROJECT).get_dataframe(**kwargs)
    x.columns = x.columns.str.upper()
    return x

def get_code(name):
    return p.get_recipe(name).get_settings().get_code()

def once(s, old, new):
    assert s.count(old) == 1, 'Unexpected source structure: '+old[:100]
    return s.replace(old,new,1)

def literal(s,name):
    return next(ast.literal_eval(n.value) for n in ast.parse(s).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))

previous = read(RESULT)
previous_recipe_code = get_code(RECIPE)
assert len(previous) in (828,840,1092)
assert not previous.duplicated(KEYS).any()
assert set(previous.PORTFOLIO)==PORTS
previous.to_pickle('payment_results_before_update.pkl')
with open('payment_recipe_before_update.py','w',encoding='utf-8') as f:
    f.write(previous_recipe_code)

# Extend the monthly cache: only January-August 2024 needs Oracle extraction.
source = get_code('compute_payment_movements_monthly_Corporate')
assert source.count('# Full consecutive months only.')==1
ext_code = source.split('# Full consecutive months only.')[0]
ext_code = once(ext_code,"FIRST_MONTH='2024-09-01'","FIRST_MONTH='2024-01-01'")
ext_code = once(ext_code,"LAST_COMPLETE_MONTH='2026-09-01'","LAST_COMPLETE_MONTH='2024-08-01'")
ext_code += r'''
cols=['ENTITY_ID','PAYMENT_MONTH','AVAILABLE_DATE','INCOMING_EUR','OUTGOING_EUR']
old=dataiku.Dataset('payment_movements_monthly_Corporate',project_key=PROJECT).get_dataframe(columns=cols)
x=pd.concat([monthly[cols],old[cols]],ignore_index=True)
x['ENTITY_ID']=x.ENTITY_ID.map(norm)
for c in ['PAYMENT_MONTH','AVAILABLE_DATE']:
    x[c]=pd.to_datetime(x[c],errors='raise',utc=True).dt.strftime('%Y-%m-%d')
for c in ['INCOMING_EUR','OUTGOING_EUR']:
    x[c]=pd.to_numeric(x[c],errors='raise')
assert not x[cols].isna().any().any()
assert not x.duplicated(['ENTITY_ID','PAYMENT_MONTH']).any()
assert (x[['INCOMING_EUR','OUTGOING_EUR']]>=0).all().all()
assert set(x.PAYMENT_MONTH)==set(pd.date_range('2024-01-01','2026-09-01',freq='MS').strftime('%Y-%m-%d'))
assert x.AVAILABLE_DATE.eq((pd.to_datetime(x.PAYMENT_MONTH)+pd.offsets.MonthEnd(0)).dt.strftime('%Y-%m-%d')).all()
dataiku.Dataset('payment_extended_monthly_Corporate',project_key=PROJECT).write_with_schema(x)
print('FINISHED | 33 monthly observations available',flush=True)
'''

# Reuse the verified incoming ranking/mapping logic. Add percentage aggregates.
rank_code = get_code('compute_payment_fixed_top_monthly_Corporate')
rank_code = once(rank_code,"OUTPUT='payment_fixed_top_monthly_Corporate'",f"OUTPUT='{RANK}'")
rank_code = once(rank_code,'ROW_NUMBER() OVER(PARTITION BY ENTITY_ID ORDER BY AMOUNT_EUR DESC,PAYMENT_ID) RN',
    'ROW_NUMBER() OVER(PARTITION BY ENTITY_ID ORDER BY AMOUNT_EUR DESC,PAYMENT_ID) RN,\n COUNT(*) OVER(PARTITION BY ENTITY_ID) N_TX')
pct_cols = [f'TOP{n}_{z}_EUR' for n in (10,20,30) for z in ('FLOOR','CEIL')]
agg = ''.join(f' SUM(CASE WHEN RN<={z}(N_TX*{n}/100) THEN AMOUNT_EUR ELSE 0 END) TOP{n}_{z}_EUR,\n' for n in (10,20,30) for z in ('FLOOR','CEIL'))
rank_code = once(rank_code,' SUM(CASE WHEN RN<=10',agg+' SUM(CASE WHEN RN<=10')
rank_code = once(rank_code,'NVL(a.TOP10_EUR,0) TOP10_EUR,',''.join(f'NVL(a.{c},0) {c},\n ' for c in pct_cols)+'NVL(a.TOP10_EUR,0) TOP10_EUR,')
rank_code = once(rank_code,'    for c in numeric:',f'    numeric += {pct_cols!r}\n    for c in numeric:')
rank_code = once(rank_code,'    frames.append(z)',r'''
    ok=z.IS_SOURCE_RECONCILED.eq(1)
    for n in (10,20,30):
        a=z[f'TOP{n}_FLOOR_EUR']; b=z[f'TOP{n}_CEIL_EUR']
        assert (a[ok]>=0).all() and (a[ok]<=b[ok]).all()
        assert (b[ok]<=z.loc[ok,'FIXED_INCOMING_EUR']+0.01).all()
    frames.append(z)'''.lstrip('\n'))

# Preserve legacy indicator names; new rules have new names.
legacy = [f'{i}_{r}' for i in ['payment_turnover']+[f'payment_top{n}_{z}' for n in (10,20,30) for z in ('floor','ceil')] for r in ('prev3_mean','prev12_mean','prev12_max')]
fixed = [f'payment_top{n}_{r}' for n in (10,20,30) for r in ('prev3_mean','prev12_mean','prev12_max')]
movement = [f'payment_{f}_{s}_vs_avg12_{d}' for f in ('incoming','outgoing','net','gross') for s in ('avg3','max3') for d in ('decline','increase') if not (f=='incoming' and d=='increase' or f=='outgoing' and d=='decline')]
flow_new = [f'payment_{f}_{s}_vs_prev12_{r}_{d}' for f,rs,d in [('incoming',('mean','min'),'decline'),('outgoing',('mean','max'),'increase')] for s in ('avg3','max3') for r in rs]
min_new = [f'{i}_prev12_min' for i in ['payment_turnover']+[f'payment_top{n}' for n in (10,20,30)]+[f'payment_top{n}_{z}' for n in (10,20,30) for z in ('floor','ceil')]]
in12='account_movements_incoming_12m_vs_prev12m_decline'
out12='account_movements_outgoing_12m_vs_prev12m_increase'
keep=legacy+fixed+movement
names=keep+flow_new+min_new+[in12,out12]
assert len(keep)==42 and len(set(names))==62
values=','.join(f"('{n}',{t})" for n in names for t in ((25,) if n in (in12,out12) else (20,30,50)))

# Reuse the common validated portfolio/event backtest SQL.
q=literal(previous_recipe_code,'SQL_TEMPLATE')
head=q[:q.index('FIXED_MONTHLY AS (')]
tail=q[q.index('SIGNALS_PAYMENT AS ('):]
a=tail.index(' WHERE ')
b=tail.index('   AND s.AVAILABLE_DATE>=',a)
tail=tail[:a]+''' WHERE EXISTS (SELECT 1 FROM GRID_PAYMENTS g
    WHERE g.INDICATOR=s.INDICATOR AND g.THRESHOLD_A=s.THRESHOLD_A)
'''+tail[b:]
# Keep the old fixed Top N calculation intact, including exact deduplication.
fixed_cte=q[q.index('FIXED_MONTHLY AS ('):q.index('PAYMENT_STAGE_COMBINED AS (')]
# Strip new CTEs if rerunning this installer.
if 'EXT_MONTHLY AS (' in fixed_cte:
    fixed_cte=fixed_cte[:fixed_cte.index('EXT_MONTHLY AS (')]
fixed_cte=re.sub(r'(FIXED_MONTHLY\s+AS\s*\(\s*SELECT)\s+(?:DISTINCT\s+)?',r'\1 DISTINCT ',fixed_cte,count=1,flags=re.I)

# New Snowflake features; all amounts come from complete entity-month windows.
W='PARTITION BY ENTITY_ID,FAMILY ORDER BY PAYMENT_MONTH'
A=lambda fn,lo,hi: f'{fn}(AMOUNT) OVER({W} ROWS BETWEEN {lo} AND {hi})'
flow_cte=f'''
EXT_MONTHLY AS (
 SELECT TRIM(ENTITY_ID::VARCHAR) ENTITY_ID,TO_DATE(PAYMENT_MONTH) PAYMENT_MONTH,
 TO_DATE(AVAILABLE_DATE) AVAILABLE_DATE,INCOMING_EUR,OUTGOING_EUR FROM {{extended_monthly}}
), FLOW_LONG AS (
 SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,'incoming' FAMILY,INCOMING_EUR AMOUNT FROM EXT_MONTHLY
 UNION ALL SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,'outgoing',OUTGOING_EUR FROM EXT_MONTHLY
), FLOW_HIST AS (
 SELECT *, LAG(PAYMENT_MONTH,14) OVER({W}) M14,LAG(PAYMENT_MONTH,23) OVER({W}) M23,
 {A('COUNT','14 PRECEDING','CURRENT ROW')} N15,
 {A('COUNT','23 PRECEDING','CURRENT ROW')} N24,
 {A('AVG','2 PRECEDING','CURRENT ROW')} RAVG,
 {A('MAX','2 PRECEDING','CURRENT ROW')} RMAX,
 {A('AVG','14 PRECEDING','3 PRECEDING')} BAVG,
 {A('MIN','14 PRECEDING','3 PRECEDING')} BMIN,
 {A('MAX','14 PRECEDING','3 PRECEDING')} BMAX,
 {A('SUM','11 PRECEDING','CURRENT ROW')} SUM12,
 {A('SUM','23 PRECEDING','12 PRECEDING')} PRIOR12
 FROM FLOW_LONG
), FLOW_FEATURES AS (
'''
flow_selects=[]
for recent,rc in [('avg3','RAVG'),('max3','RMAX')]:
    for family,refs,side,sign in [('incoming',[('mean','BAVG'),('min','BMIN')],'decline',-1),('outgoing',[('mean','BAVG'),('max','BMAX')],'increase',1)]:
        for label,bc in refs:
            flow_selects.append(f"SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,'payment_{family}_{recent}_vs_prev12_{label}_{side}' INDICATOR, IFF(N15=15 AND M14=DATEADD('MONTH',-14,PAYMENT_MONTH) AND {bc}>0,{sign}*100*({rc}-{bc})/NULLIF({bc},0),NULL) MAG FROM FLOW_HIST WHERE FAMILY='{family}'")
flow_selects.append(f"SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,'{out12}', IFF(N24=24 AND M23=DATEADD('MONTH',-23,PAYMENT_MONTH) AND PRIOR12>0,100*(SUM12-PRIOR12)/NULLIF(PRIOR12,0),NULL) FROM FLOW_HIST WHERE FAMILY='outgoing'")
flow_cte+='\n UNION ALL\n'.join(flow_selects)+'\n), MIN_LONG AS (\n'
minimum=["SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,'payment_turnover' FAMILY,INCOMING_EUR AMOUNT FROM EXT_MONTHLY WHERE PAYMENT_MONTH>=DATE '2024-09-01'"]
for n in (10,20,30):
    minimum.append(f"SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,'payment_top{n}',TOP{n}_EUR FROM FIXED_MONTHLY")
    for z in ('floor','ceil'):
        minimum.append(f"SELECT TRIM(ENTITY_ID::VARCHAR),TO_DATE(PAYMENT_MONTH),TO_DATE(AVAILABLE_DATE),'payment_top{n}_{z}',TOP{n}_{z.upper()}_EUR FROM {{ranked_monthly}} WHERE IS_SOURCE_RECONCILED=1")
flow_cte+='\n UNION ALL\n'.join(minimum)+f'''
), MIN_HIST AS (
 SELECT *, LAG(PAYMENT_MONTH,12) OVER({W}) M12,
 {A('COUNT','12 PRECEDING','1 PRECEDING')} N12,
 {A('MIN','12 PRECEDING','1 PRECEDING')} BASE_MIN
 FROM MIN_LONG
), NEW_FEATURES AS (
 SELECT * FROM FLOW_FEATURES
 UNION ALL SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,FAMILY||'_prev12_min',
 IFF(N12=12 AND M12=DATEADD('MONTH',-12,PAYMENT_MONTH) AND BASE_MIN>0,100*(1-AMOUNT/NULLIF(BASE_MIN,0)),NULL)
 FROM MIN_HIST
), NEW_PREV AS (
 SELECT *,LAG(PAYMENT_MONTH) OVER(PARTITION BY ENTITY_ID,INDICATOR ORDER BY PAYMENT_MONTH) PM,
 LAG(MAG) OVER(PARTITION BY ENTITY_ID,INDICATOR ORDER BY PAYMENT_MONTH) PREV_MAG FROM NEW_FEATURES
), NEW_SIGNALS AS (
 SELECT n.ENTITY_ID,n.PAYMENT_MONTH,n.AVAILABLE_DATE,n.INDICATOR,g.THRESHOLD_A
 FROM NEW_PREV n JOIN GRID_PAYMENTS g ON n.INDICATOR=g.INDICATOR
 WHERE n.PM=DATEADD('MONTH',-1,n.PAYMENT_MONTH)
 AND ((g.THRESHOLD_A=25 AND n.MAG>25 AND n.PREV_MAG<=25)
 OR (g.THRESHOLD_A<>25 AND n.MAG>=g.THRESHOLD_A AND n.PREV_MAG<g.THRESHOLD_A))
), PAYMENT_STAGE_COMBINED AS (
 SELECT TRIM(ENTITY_ID::VARCHAR) ENTITY_ID,TO_DATE(PAYMENT_MONTH) PAYMENT_MONTH,TO_DATE(AVAILABLE_DATE) AVAILABLE_DATE,INDICATOR,TRY_TO_DOUBLE(THRESHOLD_A::VARCHAR) THRESHOLD_A FROM {{legacy_stage}}
 UNION ALL SELECT TRIM(ENTITY_ID::VARCHAR),TO_DATE(PAYMENT_MONTH),TO_DATE(AVAILABLE_DATE),INDICATOR,TRY_TO_DOUBLE(THRESHOLD_A::VARCHAR) FROM {{new_stage}}
 UNION ALL SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,INDICATOR,THRESHOLD_A FROM FIXED_SIGNALS
 UNION ALL SELECT ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,INDICATOR,THRESHOLD_A FROM NEW_SIGNALS
 UNION ALL SELECT TRIM(ENTITY_ID::VARCHAR),TO_DATE(PAYMENT_MONTH),TO_DATE(AVAILABLE_DATE),INDICATOR,TRY_TO_DOUBLE(THRESHOLD_A::VARCHAR) FROM {{account_stage}} WHERE INDICATOR='{in12}'
),
'''
new_sql=head+fixed_cte+flow_cte+tail

# Replace the output recipe, retaining the same training, target and costs.
prefix=previous_recipe_code.split('OLD=physical(')[0]
assert 'def physical(' in prefix and 'F=physical(' in prefix
result_code=prefix+f'''
OLD=physical('payment_signals_Corporate')
NEW=physical('payment_movements_signals_Corporate')
M=physical({FIXED!r})
X=physical({EXT!r})
R=physical({RANK!r})
A=physical({ACCOUNT!r})
AR=physical({ACCOUNT_RESULT!r})
O=physical(OUTPUT)
from dataiku import recipe
assert {{str(getattr(x,'name',x)).split('.')[-1] for x in recipe.get_outputs(object_type='DATASET')}}=={{OUTPUT}}
assert O.upper() not in {{x.upper() for x in (T,B,L,E,H,F,OLD,NEW,M,X,R,A,AR,physical('results_EWS_Corporate'),physical('results_Corporate_EWS_validation'))}}
SQL_TEMPLATE={new_sql!r}
PAYMENT_GRID_VALUES={values!r}
NAMES={names!r}
KEEP={keep!r}
IN12={in12!r}
KEYS={KEYS!r}
PORTS={sorted(PORTS)!r}
'''+r'''
engine=SQLExecutor2(connection=CONNECTION)
def query(q):
    d=engine.query_to_df(q); d.columns=d.columns.str.upper(); return d

def check_unique(table,cols):
    q='SELECT COUNT(*) FROM (SELECT '+cols+',COUNT(*) N FROM '+table+' GROUP BY '+cols+' HAVING COUNT(*)>1)'
    assert int(query(q).iloc[0,0])==0, 'Duplicate keys: '+table

# Fixed source contains identical physical duplicates; reject conflicting records.
fixed_fields='ENTITY_ID,PAYMENT_MONTH,AVAILABLE_DATE,MAPPING_SOURCE_DATE,INCOMING_EUR,N_INCOMING,FIXED_INCOMING_EUR,FIXED_N_INCOMING,TOP10_EUR,TOP20_EUR,TOP30_EUR,N_CONFLICTS,IS_SOURCE_RECONCILED'
check_unique('(SELECT DISTINCT '+fixed_fields+' FROM '+M+')','ENTITY_ID,PAYMENT_MONTH')
check_unique(X,'ENTITY_ID,PAYMENT_MONTH')
check_unique(R,'ENTITY_ID,PAYMENT_MONTH')
assert int(query('SELECT COUNT(DISTINCT TO_DATE(PAYMENT_MONTH)) FROM '+X).iloc[0,0])==33
assert int(query('SELECT COUNT(DISTINCT TO_DATE(PAYMENT_MONTH)) FROM '+R+' WHERE IS_SOURCE_RECONCILED=1').iloc[0,0])==25
old=query('SELECT * FROM '+O)
account=query('SELECT * FROM '+AR)
assert len(old) in (828,840,1092)
assert not old.duplicated(KEYS).any()
ref=pd.concat([old[old.INDICATOR.isin(KEEP)],account[account.INDICATOR.eq(IN12)]],ignore_index=True)
assert len(ref)==762 and not ref.duplicated(KEYS).any()
sql=SQL_TEMPLATE.format(T=T,B=B,L=L,E=E,H=H,F=F,start=START_DATE,end=END_DATE,
 adc=nums(ADC_CODES),ip=nums(IP_CODES),other=nums(OTHER_CODES),lead_lo=LEAD_MIN_MONTHS,lead_hi=LEAD_MAX_MONTHS,
 payment_grid_values=PAYMENT_GRID_VALUES,legacy_stage=OLD,new_stage=NEW,fixed_monthly=M,extended_monthly=X,ranked_monthly=R,account_stage=A)
print('[RUN] Unified Payments backtest',flush=True)
candidate=query(sql)
assert len(candidate)==1092 and set(candidate.INDICATOR)==set(NAMES)
assert set(candidate.PORTFOLIO)==set(PORTS) and candidate.groupby('PORTFOLIO').size().eq(182).all()
assert not candidate.duplicated(KEYS).any()
assert set(candidate.columns)==set(old.columns)==set(ref.columns)
assert (candidate.N_TP+candidate.N_FN==candidate.N_EVENTS).all()
assert (candidate.N_TP+candidate.N_FP==candidate.N_ALERTS).all()
check=candidate[candidate.INDICATOR.isin(KEEP+[IN12])]
pd.testing.assert_frame_equal(ref[old.columns].sort_values(KEYS).reset_index(drop=True),check[old.columns].sort_values(KEYS).reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-7)
print('[PASS] 762 retained configurations unchanged',flush=True)
dataiku.Dataset(OUTPUT,project_key=PROJECT).write_dataframe(candidate[old.columns],infer_schema=False)
print('[FINISHED] 62 indicators | 1092 configurations',flush=True)
'''

# Create technical caches with distinct Snowflake destinations and explicit schema.
specs=[('compute_payment_extended_monthly_Corporate',EXT,['PAYEE_VW','PAYOR_VW','ENTITY_CUSTOMER_MON_VW','EWS_training_Corporate',MONTH,'payment_movements_signals_Corporate'],ext_code),('compute_payment_ranked_percentage_monthly_Corporate',RANK,['PAYEE_VW','ENTITY_CUSTOMER_MON_VW',MONTH],rank_code)]
for name,out,inputs,code in specs:
    compile(code,name,'exec')
compile(result_code,RECIPE,'exec')
# Check formatting before creating anything.
literal(result_code,'SQL_TEMPLATE').format(**{k:'X' for k in ('T','B','L','E','H','F','start','end','adc','ip','other','lead_lo','lead_hi','payment_grid_values','legacy_stage','new_stage','fixed_monthly','extended_monthly','ranked_monthly','account_stage')})
reference=p.get_dataset(RESULT).get_settings().get_raw()['params']
assert reference['connection']==CONNECTION and reference['schema'].upper()=='RRG'
known_r={x['name'] for x in p.list_recipes()}; known_d={x['name'] for x in p.list_datasets()}
def loc(x):
    return (x.get('connection'),str(x.get('schema','')).upper(),str(x.get('table','')).replace('${projectKey}',PROJECT).upper())
for name,out,inputs,code in specs:
    assert set(inputs)<=known_d
    assert (name in known_r)==(out in known_d),'Partial creation: '+name
    if name in known_r:
        assert get_code(name).strip()==code.strip(),'Existing recipe differs: '+name
    target=(CONNECTION,'RRG',(PROJECT+'_'+out).upper())
    for d in known_d-{out}:
        assert loc(p.get_dataset(d).get_settings().get_raw().get('params',{}))!=target,'Physical collision: '+d
for name,out,inputs,code in specs:
    if name not in known_r:
        b=p.new_recipe('python',name)
        for inp in inputs: b.with_input(inp)
        b.with_new_output_dataset(out,CONNECTION,overwrite=False)
        b.with_script(code); b.create()
        settings=p.get_dataset(out).get_settings(); params=settings.get_raw()['params']
        for k in ('database','catalog','schema'):
            if reference.get(k): params[k]=reference[k]
        params['table']=PROJECT+'_'+out
        settings.save()
    assert loc(p.get_dataset(out).get_settings().get_raw()['params'])==(CONNECTION,'RRG',(PROJECT+'_'+out).upper())
    try: ready=not read(out,limit=1).empty
    except Exception: ready=False
    if not ready:
        print('RUNNING | '+name,flush=True)
        p.get_recipe(name).run(job_type='NON_RECURSIVE_FORCED_BUILD',wait=True)
    else: print('REUSING | '+out,flush=True)
    assert not read(out,limit=1).empty

settings=p.get_recipe(RECIPE).get_settings()
for inp in (EXT,RANK,ACCOUNT,ACCOUNT_RESULT):
    if not settings.has_input(inp) and not settings.has_input(PROJECT+'.'+inp):
        settings.add_input('main',inp)
settings.set_code(result_code); settings.save()
assert get_code(RECIPE).strip()==result_code.strip()
print('RUNNING | '+RECIPE,flush=True)
p.get_recipe(RECIPE).run(job_type='NON_RECURSIVE_FORCED_BUILD',wait=True)
result=read(RESULT)
assert len(result)==1092 and result.INDICATOR.nunique()==62
assert not result.duplicated(KEYS).any()
assert set(result.PORTFOLIO)==PORTS and result.groupby('PORTFOLIO').size().eq(182).all()
print('PASS | 62 indicators / 1092 configurations')
print(result.loc[result.PORTFOLIO.eq('GLOBAL') & result.INDICATOR.isin(flow_new+min_new+[in12,out12]),['INDICATOR','THRESHOLD_A','N_TP','N_FP','N_ALERTS','ACCURACY','COVERAGE']].sort_values(['INDICATOR','THRESHOLD_A']).to_string(index=False))
