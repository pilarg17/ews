# -*- coding: utf-8 -*-
"""BIL Corporate EWS | RFO monthly versions and Timo's two examples.

One cell in Dataiku Python notebook with PRD_BDWH access.
Read-only diagnostic: selects only, no Dataiku dataset or recipe changes.
This validates SOURCE DATA and proposed treatment; does NOT recalculate full portfolio coverage.
"""
import pandas as pd
from dataiku.core.sql import SQLExecutor2

ora = SQLExecutor2(connection='PRD_BDWH')
REPORT_DATEKEY = 20260930  # Latest month-end prior to the 2026-10-29 EWS snapshot
EXAMPLES = (
    ('LOANDEPO_VW', 'CONTRACT_REFERENCE', 'OUTSTANDING', '2110000000049295'),
    ('ACCOUNT_VW', 'ACCOUNT_REFERENCE', 'BALANCE', '2110000091015150'),
)

def q(label, sql):
    print('\n=== '+label+' ===', flush=True)
    df = ora.query_to_df(sql)
    df.columns = [str(c).upper() for c in df.columns]
    print(df.to_csv(index=False, sep=';', na_rep='NULL').strip(), flush=True)
    return df

metadata = q('00_SOURCE_COLUMN_CHECK', """
SELECT TABLE_NAME, COLUMN_NAME FROM ALL_TAB_COLUMNS
WHERE OWNER='DTM_RFO_PLY' AND TABLE_NAME IN ('LOANDEPO_VW','ACCOUNT_VW')
AND COLUMN_NAME IN ('CONTRACT_REFERENCE','ACCOUNT_REFERENCE','OUTSTANDING','BALANCE',
                    'ACCRUALS','REPORT_DATEKEY','REPORT_VERSIONID','PARTITION_KEY','CURRENCY')
ORDER BY TABLE_NAME,COLUMN_NAME
""")
for tbl, ref_col, amount_col, ref in EXAMPLES:
    required = {ref_col, amount_col, 'ACCRUALS', 'REPORT_DATEKEY',
                'REPORT_VERSIONID', 'PARTITION_KEY', 'CURRENCY'}
    found = set(metadata.loc[metadata.TABLE_NAME.eq(tbl), 'COLUMN_NAME'].astype(str))
    if required - found:
        raise RuntimeError(f'{tbl}: missing expected fields {sorted(required-found)}')
    # Global max version per monthly partition: not merely max for a particular instrument.
    versions = q('01_GLOBAL_MONTHLY_VERSIONS_'+tbl, f"""
SELECT REPORT_DATEKEY, PARTITION_KEY,
       COUNT(DISTINCT REPORT_VERSIONID) N_VERSIONS,
       MIN(REPORT_VERSIONID) MIN_VERSION,
       MAX(REPORT_VERSIONID) LATEST_REPORT_VERSIONID,
       COUNT(*) N_SOURCE_ROWS
FROM DTM_RFO_PLY.{tbl}
WHERE REPORT_DATEKEY = {REPORT_DATEKEY}
GROUP BY REPORT_DATEKEY,PARTITION_KEY
ORDER BY PARTITION_KEY
""")
    if versions.empty:
        print(f'NO MONTHLY DATA for {tbl} and date {REPORT_DATEKEY}', flush=True)
        continue
    data = q('02_EXAMPLE_VERSION_HISTORY_'+tbl, f"""
SELECT TO_CHAR({ref_col}) REFERENCE,
       REPORT_DATEKEY, PARTITION_KEY, REPORT_VERSIONID,
       {amount_col} RAW_AMOUNT, ACCRUALS, CURRENCY,
       CASE WHEN {amount_col}>0 THEN 1 ELSE 0 END POSITIVE_RAW_AMOUNT,
       CASE WHEN {amount_col}<0 THEN 1 ELSE 0 END NEGATIVE_RAW_AMOUNT,
       CASE WHEN ACCRUALS IS NULL THEN 1 ELSE 0 END NULL_ACCRUALS,
       CASE WHEN ({amount_col} + ACCRUALS)>0 THEN 1 ELSE 0 END OLD_FILTER_PASSES,
       CASE WHEN ({amount_col} + NVL(ACCRUALS,0))>0 THEN 1 ELSE 0 END NULL_SAFE_FILTER_PASSES
FROM DTM_RFO_PLY.{tbl}
WHERE REPORT_DATEKEY = {REPORT_DATEKEY}
  AND TRIM(TO_CHAR({ref_col}))='{ref}'
ORDER BY PARTITION_KEY,REPORT_VERSIONID DESC
""")
    if data.empty:
        print('EXAMPLE NOT FOUND for requested reporting month', flush=True)
        continue
    latest = versions[['REPORT_DATEKEY','PARTITION_KEY','LATEST_REPORT_VERSIONID']].copy()
    latest['PARTITION_KEY'] = latest.PARTITION_KEY.astype(str)
    data['PARTITION_KEY'] = data.PARTITION_KEY.astype(str)
    data = data.merge(latest, on=['REPORT_DATEKEY','PARTITION_KEY'], how='left', validate='many_to_one')
    data['IS_GLOBAL_LATEST_MONTHLY_VERSION'] = data.REPORT_VERSIONID.eq(data.LATEST_REPORT_VERSIONID)
    print('\n=== 03_EXAMPLE_ONLY_ON_GLOBAL_LATEST_MONTHLY_VERSION_'+tbl+' ===', flush=True)
    selected=data.loc[data.IS_GLOBAL_LATEST_MONTHLY_VERSION]
    if selected.empty:
        print('NO MATCH: this example is absent from the global latest monthly version; review before assuming source coverage', flush=True)
    else:
        print(selected.to_csv(index=False,sep=';',na_rep='NULL').strip(),flush=True)
    if tbl=='ACCOUNT_VW':
        print('DIAGNOSTIC ONLY: Timo confirmed negative BALANCE should be represented as -BALANCE for overdraft exposure.', flush=True)
    else:
        print('DIAGNOSTIC ONLY: positive OUTSTANDING with NULL ACCRUALS must not be lost to NULL arithmetic.', flush=True)
print('\nNO WRITES / NO RECIPES CHANGED. This is not the recomputed full-population nominal coverage.', flush=True)
