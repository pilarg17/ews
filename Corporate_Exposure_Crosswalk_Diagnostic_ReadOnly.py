# Corporate EWS | exposure link diagnosis after full-population QA
# Run in the SAME Dataiku Python notebook/kernel as full exposure QA.
# SELECT only: no datasets, recipes or results are changed.
import pandas as pd
from dataiku.core.sql import SQLExecutor2

REQUIRED = ('loans', 'keys', 'matched')
for name in REQUIRED:
    if name not in globals() or not isinstance(globals()[name], pd.DataFrame):
        raise RuntimeError('Run the full exposure QA first in the same kernel. Missing: ' + name)

ORACLE = SQLExecutor2(connection='PRD_BDWH')
AS_OF = '2026-10-29'
TIMO_REFS = [('LOANDEPO_VW', '2110000000049295'),
             ('ACCOUNT_VW', '2110000091015150')]

def show(title, frame, n=50):
    print('\n=== '+title+' ===', flush=True)
    if frame is None or frame.empty:
        print('NO_ROWS', flush=True)
    else:
        print(frame.head(n).to_csv(index=False, sep=';', na_rep='NULL').strip(), flush=True)

def lit(x):
    return "'" + str(x).replace("'", "''") + "'"

# 01 | Identify the actual Corporate instrument linked to each supplied RFO ref.
for src, ref in TIMO_REFS:
    subset = keys[(keys['SOURCE'] == src) & (keys['REF_KEY'] == ref)]
    if subset.empty:
        show('01 | '+src+' | '+ref, pd.DataFrame())
        continue
    det = subset.merge(loans[['ROW_ID', 'ENTITY_ID', 'NOMINAL_AMOUNT']],
                       on='ROW_ID', how='left', validate='many_to_one')
    matched_cols = ['ROW_ID','SOURCE','REF_KEY','REPORT_DATEKEY','RAW_AMOUNT',
                    'ACCRUALS','CCY','STRICT_ONE_TO_ONE','FRESH']
    det = det.merge(matched[matched_cols].drop_duplicates(),
                    on=['ROW_ID','SOURCE','REF_KEY'], how='left', validate='many_to_many')
    show('01 | LINKED CORPORATE LOAN | '+src+' | '+ref, det)

# 02 | Are regulatory monthly records actually present on the selected date?
reg_table = 'BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_MON_VW'
q = """
SELECT TO_CHAR(MAP_DTE, 'YYYY-MM-DD') MAP_DTE,
       COUNT(*) N_ROWS,
       COUNT(INSTRUMENT_ID) N_WITH_INSTRUMENT_ID,
       COUNT(SOURCE_CONTRACT_REFERENCE) N_WITH_SOURCE_REF,
       COUNT(OUTSTANDING_NOMINAL_AMOUNT) N_WITH_AMOUNT
FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_MON_VW
WHERE MAP_DTE >= DATE '2026-08-01' AND MAP_DTE < DATE '2026-10-30'
GROUP BY MAP_DTE ORDER BY MAP_DTE DESC
"""
source_dates = ORACLE.query_to_df(q)
source_dates.columns = [str(c).upper() for c in source_dates.columns]
show('02 | REGULATORY SOURCE SNAPSHOTS', source_dates, 15)

# 03 | Test both identifier columns on the same source, for the actual missing
# instrument and a bounded sample of the current snapshot. This diagnoses the
# suspicious zero regulatory crosswalk; it is not a portfolio replacement.
first_missing = '2118155658.0054650.01.9586315'
missing_ids = loans.loc[loans.NOMINAL_AMOUNT.isna(), 'INSTRUMENT_ID'].dropna().astype(str).head(15).tolist()
known_ids = loans.loc[loans.NOMINAL_AMOUNT.notna(), 'INSTRUMENT_ID'].dropna().astype(str).head(15).tolist()
sample = sorted(set(missing_ids + known_ids + [first_missing]))
ref_list = ','.join(lit(x) for x in sample)
q = f"""
SELECT TO_CHAR(MAP_DTE,'YYYY-MM-DD') REG_DATE,
       TRIM(TO_CHAR(INSTRUMENT_ID)) REG_INSTRUMENT_ID,
       TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) REG_SOURCE_REF,
       TRIM(TO_CHAR(LEGACY_CONTRACT_REFERENCE)) REG_LEGACY_REF,
       OUTSTANDING_NOMINAL_AMOUNT,
       SOURCE_VERSION_ID, LAST_VERSION_FLAG
FROM {reg_table}
WHERE MAP_DTE >= DATE '2026-08-01' AND MAP_DTE < DATE '2026-10-30'
  AND (TRIM(TO_CHAR(INSTRUMENT_ID)) IN ({ref_list})
       OR TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) IN ({ref_list})
       OR TRIM(TO_CHAR(LEGACY_CONTRACT_REFERENCE)) IN ({ref_list}))
FETCH FIRST 100 ROWS ONLY
"""
reg_sample = ORACLE.query_to_df(q)
reg_sample.columns = [str(c).upper() for c in reg_sample.columns]
show('03 | REGULATORY LINK SAMPLE: EITHER ID FIELD', reg_sample, 100)

# 04 | Specifically test the two supplied RFO reference values in the
# regulatory source: does it map them to any other instrument IDs?
ref_two = ','.join(lit(x) for _,x in TIMO_REFS)
q = f"""
SELECT TO_CHAR(MAP_DTE,'YYYY-MM-DD') REG_DATE,
       TRIM(TO_CHAR(INSTRUMENT_ID)) REG_INSTRUMENT_ID,
       TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) REG_SOURCE_REF,
       TRIM(TO_CHAR(LEGACY_CONTRACT_REFERENCE)) REG_LEGACY_REF,
       OUTSTANDING_NOMINAL_AMOUNT,
       SOURCE_VERSION_ID, LAST_VERSION_FLAG
FROM {reg_table}
WHERE MAP_DTE >= DATE '2026-08-01' AND MAP_DTE < DATE '2026-10-30'
  AND (TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) IN ({ref_two})
       OR TRIM(TO_CHAR(INSTRUMENT_ID)) IN ({ref_two})
       OR TRIM(TO_CHAR(LEGACY_CONTRACT_REFERENCE)) IN ({ref_two}))
FETCH FIRST 100 ROWS ONLY
"""
rfo_bridge = ORACLE.query_to_df(q)
rfo_bridge.columns = [str(c).upper() for c in rfo_bridge.columns]
show('04 | REGULATORY REFERENCES PROVIDED BY SOURCE OWNER', rfo_bridge, 100)

print('\nCONCLUSION | This diagnoses the crosswalk and identifies the two exact loans.')
print('No exposure has been added; no data or recipe modifications.')
