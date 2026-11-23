# Corporate EWS: 5 exact-key examples requested for source reconciliation
# One Dataiku Python notebook cell; SELECT only, no recipe/dataset changes.
import re
import pandas as pd
import dataiku
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
ASOF = '2026-10-29'
RFO_DAY = '2026-09-30'
MAX_LOANS = 2000
BATCH_SIZE = 200
N_EXAMPLES = 5

project = dataiku.api_client().get_project(PROJECT)
ds = dataiku.Dataset('vars_loan_level', project_key=PROJECT)
settings = project.get_dataset('vars_loan_level').get_settings().get_raw().get('params', {})
assert settings.get('connection') == 'DEV_AA_TEAMS', 'Unexpected Corporate source'
def qid(s):
    s = str(s).replace('${projectKey}', PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', s):
        raise ValueError('Unverified source identifier')
    return '"' + s + '"'
phys = qid(settings['schema']) + '.' + qid(settings['table'])
sf = SQLExecutor2(connection='DEV_AA_TEAMS')
ora = SQLExecutor2(connection='PRD_BDWH')
base = sf.query_to_df("""
SELECT TRIM(TO_VARCHAR(INSTRUMENT_ID)) INSTRUMENT_ID
FROM {table}
WHERE TO_DATE(MAP_DTE) = DATE '{asof}'
  AND NOMINAL_AMOUNT IS NULL AND INSTRUMENT_ID IS NOT NULL
GROUP BY 1
ORDER BY 1
LIMIT {limit}
""".format(table=phys, asof=ASOF, limit=MAX_LOANS))
base.columns = [str(x).upper() for x in base.columns]
ids = base['INSTRUMENT_ID'].dropna().astype(str).str.strip().tolist()
if not ids:
    raise RuntimeError('No missing-nominal Corporate instruments found')

def in_sql(values):
    if len(values) > 500:
        raise RuntimeError('Unexpected query batch size')
    return '(' + ','.join("'" + s.replace("'", "''") + "'" for s in values) + ')'

def norm(d):
    d.columns = [str(x).upper() for x in d.columns]
    if 'REF_KEY' in d:
        d['REF_KEY'] = d.REF_KEY.astype(str).str.strip()
    return d

found = []
examined = 0
for offset in range(0, len(ids), BATCH_SIZE):
    batch = ids[offset: offset + BATCH_SIZE]
    quoted = in_sql(batch)
    # CRI: candidate source versions, kept separate from RFO report versions.
    reg = norm(ora.query_to_df("""
SELECT TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) REF_KEY,
       TRIM(TO_CHAR(SOURCE_VERSION_ID)) REG_VERSION,
       TRIM(TO_CHAR(LAST_VERSION_FLAG)) LAST_FLAG,
       COUNT(*) N_REG_ROWS,
       COUNT(DISTINCT OUTSTANDING_NOMINAL_AMOUNT) N_AMOUNTS,
       MAX(OUTSTANDING_NOMINAL_AMOUNT) REG_AMOUNT
FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_MON_VW
WHERE MAP_DTE >= DATE '{day}' AND MAP_DTE < DATE '{day}' + 1
  AND TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) IN {keys}
GROUP BY TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)),
         TRIM(TO_CHAR(SOURCE_VERSION_ID)), TRIM(TO_CHAR(LAST_VERSION_FLAG))
""".format(day=RFO_DAY, keys=quoted)))
    if reg.empty:
        examined += len(batch)
        continue
    # Only a unique positive amount in the greatest Y-flagged CRI source version.
    reg = reg[reg.LAST_FLAG.astype(str).str.upper().eq('Y')].copy()
    if reg.empty:
        examined += len(batch)
        continue
    reg['NUM_VERSION'] = pd.to_numeric(reg.REG_VERSION, errors='coerce')
    reg['REG_AMOUNT'] = pd.to_numeric(reg.REG_AMOUNT, errors='coerce')
    reg['N_AMOUNTS'] = pd.to_numeric(reg.N_AMOUNTS, errors='coerce')
    reg = reg.sort_values(['REF_KEY', 'NUM_VERSION'], ascending=[True,False])
    reg = reg.drop_duplicates('REF_KEY', keep='first')
    reg = reg[reg.N_AMOUNTS.eq(1) & reg.REG_AMOUNT.gt(0)]
    if reg.empty:
        examined += len(batch)
        continue
    keys = in_sql(reg.REF_KEY.tolist())
    rfo_presence = {}
    for view, keycol in [('LOANDEPO_VW','CONTRACT_REFERENCE'),
                         ('ACCOUNT_VW','ACCOUNT_REFERENCE')]:
        r = norm(ora.query_to_df("""
SELECT TRIM(TO_CHAR({ref})) REF_KEY,
       COUNT(*) N_ANY_VERSION_ROWS,
       SUM(CASE WHEN REPORT_VERSIONID = 1096 THEN 1 ELSE 0 END) N_LATEST_ROWS
FROM DTM_RFO_PLY.{view}
WHERE REPORT_DATEKEY = 20260930
  AND TRIM(TO_CHAR({ref})) IN {keys}
GROUP BY TRIM(TO_CHAR({ref}))
""".format(ref=keycol, view=view, keys=keys)))
        rfo_presence[view] = ({str(row['REF_KEY']): row for _,row in r.iterrows()}
                              if not r.empty else {})
    for _,row in reg.iterrows():
        ref = str(row.REF_KEY)
        ld = rfo_presence['LOANDEPO_VW'].get(ref)
        ac = rfo_presence['ACCOUNT_VW'].get(ref)
        nld = int(ld.N_ANY_VERSION_ROWS) if ld is not None else 0
        nac = int(ac.N_ANY_VERSION_ROWS) if ac is not None else 0
        if nld == 0 and nac == 0:
            found.append({
                'INSTRUMENT_ID': ref,
                'VARS_LOAN_NOMINAL_29_OCT': 'NULL',
                'CRI_REG_VERSION': str(row.REG_VERSION),
                'CRI_LAST_VERSION_FLAG': str(row.LAST_FLAG),
                'CRI_OUTSTANDING_NOMINAL_AMOUNT_30_SEP': round(float(row.REG_AMOUNT),2),
                'RFO_LOANDEPO_ROWS_30_SEP_ALL_VERSIONS': nld,
                'RFO_ACCOUNT_ROWS_30_SEP_ALL_VERSIONS': nac,
                'FINDING': 'NO_RFO_ROW_ON_30_SEP_BY_EXACT_SOURCE_REFERENCE'
            })
        if len(found) >= N_EXAMPLES:
            break
    examined += len(batch)
    print('Checked %d missing-nominal instruments; verified examples: %d' %
          (examined, len(found)), flush=True)
    if len(found) >= N_EXAMPLES:
        break
print('\n=== VERIFIED_CRI_POSITIVE_NO_RFO_ROW_SAME_DATE ===', flush=True)
if found:
    print(pd.DataFrame(found[:N_EXAMPLES]).to_csv(index=False,sep=';'), flush=True)
else:
    print('No exact-key examples confirmed in the scanned subset; do NOT claim absence in RFO.', flush=True)
print('Scope: Corporate vars_loan_level as of 2026-10-29; CRI and both RFO tables as of 2026-09-30.',flush=True)
print('Note: exact source-reference absence only; a different RFO key mapping or another reporting date is not ruled out.',flush=True)
print('READ ONLY | NO RECIPE OR DATASET CHANGES', flush=True)
