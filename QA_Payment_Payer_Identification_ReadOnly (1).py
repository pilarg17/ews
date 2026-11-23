# -*- coding: utf-8 -*-
"""Corporate EWS | PAYEE/PAYOR payer identification QA | read-only, one cell.

Paste into one Dataiku Python notebook cell using PRD_BDWH.
Default: first 20,000 incoming credit-payment source rows for September 2026.
After validating, set MAX_PAYEE_ROWS = None to test the full month.
No dataset, recipe or backtest is modified.
"""
import pandas as pd
from dataiku.core.sql import SQLExecutor2

START = '2026-09-01'
END = '2026-10-01'
MAX_PAYEE_ROWS = 20000  # Set None for full-month coverage after sample QA
ORACLE = 'PRD_BDWH'
ora = SQLExecutor2(connection=ORACLE)

required = {
    'PAYEE_VW': {'CREDIT_PAYMENT_ID', 'LINKED_DEBIT_PAYMENT_ID', 'CREDIT_VALUE_DATE', 'PAYMENT_DIRECTION'},
    'PAYOR_VW': {'DEBIT_PAYMENT_ID', 'ORDER_GIVER_ACCOUNT'},
}
meta = ora.query_to_df("""
SELECT TABLE_NAME, COLUMN_NAME FROM ALL_TAB_COLUMNS
WHERE OWNER = 'BDL_PAYMENT_PLY' AND TABLE_NAME IN ('PAYEE_VW', 'PAYOR_VW')
""")
meta.columns = [str(c).upper() for c in meta.columns]
found = {str(t): set(g['COLUMN_NAME'].astype(str).str.upper()) for t, g in meta.groupby('TABLE_NAME')}
missing = {t: sorted(cols - found.get(t, set())) for t, cols in required.items() if cols - found.get(t, set())}
if missing:
    raise RuntimeError('Oracle fields missing; stop before QA: ' + str(missing))

cap = (f'AND ROWNUM <= {int(MAX_PAYEE_ROWS)}' if MAX_PAYEE_ROWS is not None else '')
q = f"""
WITH P_RAW AS (
    SELECT TRIM(CREDIT_PAYMENT_ID) AS CREDIT_ID,
           TRIM(LINKED_DEBIT_PAYMENT_ID) AS DEBIT_ID
    FROM BDL_PAYMENT_PLY.PAYEE_VW
    WHERE CREDIT_VALUE_DATE >= DATE '{START}'
      AND CREDIT_VALUE_DATE < DATE '{END}'
      AND UPPER(TRIM(PAYMENT_DIRECTION)) = 'INCOMING'
      AND CREDIT_PAYMENT_ID IS NOT NULL
      {cap}
),
P AS (
    SELECT CREDIT_ID,
           MIN(DEBIT_ID) AS DEBIT_ID,
           COUNT(*) AS PAYEE_ROWS,
           COUNT(DISTINCT DEBIT_ID) AS N_DIRECT_DEBITS
    FROM P_RAW
    GROUP BY CREDIT_ID
),
V AS (
    SELECT TRIM(v.DEBIT_PAYMENT_ID) AS DEBIT_ID,
           COUNT(*) AS PAYOR_ROWS,
           COUNT(DISTINCT TRIM(v.ORDER_GIVER_ACCOUNT)) AS N_ORDER_GIVER_ACCOUNTS,
           MAX(TRIM(v.ORDER_GIVER_ACCOUNT)) AS ORDER_GIVER_ACCOUNT
    FROM BDL_PAYMENT_PLY.PAYOR_VW v
    JOIN (SELECT DISTINCT DEBIT_ID FROM P WHERE DEBIT_ID IS NOT NULL) d
      ON TRIM(v.DEBIT_PAYMENT_ID) = d.DEBIT_ID
    GROUP BY TRIM(v.DEBIT_PAYMENT_ID)
),
J AS (
    SELECT p.CREDIT_ID, p.DEBIT_ID, p.PAYEE_ROWS, p.N_DIRECT_DEBITS,
           v.PAYOR_ROWS, v.N_ORDER_GIVER_ACCOUNTS, v.ORDER_GIVER_ACCOUNT
    FROM P p LEFT JOIN V v ON p.DEBIT_ID = v.DEBIT_ID
)
SELECT COUNT(*) AS N_UNIQUE_CREDIT_PAYMENTS,
       NVL(SUM(PAYEE_ROWS),0) AS N_PAYEE_SOURCE_ROWS,
       NVL(SUM(CASE WHEN PAYEE_ROWS > 1 THEN 1 ELSE 0 END),0) AS N_CREDIT_IDS_REPEATED_IN_PAYEE,
       NVL(SUM(CASE WHEN N_DIRECT_DEBITS > 1 THEN 1 ELSE 0 END),0) AS N_AMBIGUOUS_PAYEE_DIRECT_IDS,
       NVL(SUM(CASE WHEN DEBIT_ID IS NOT NULL THEN 1 ELSE 0 END),0) AS N_WITH_LINKED_DEBIT_ID,
       NVL(SUM(CASE WHEN PAYOR_ROWS IS NOT NULL THEN 1 ELSE 0 END),0) AS N_MATCHED_TO_PAYOR,
       NVL(SUM(CASE WHEN PAYOR_ROWS IS NOT NULL AND ORDER_GIVER_ACCOUNT IS NOT NULL THEN 1 ELSE 0 END),0) AS N_WITH_ORDER_GIVER_ACCOUNT,
       NVL(SUM(CASE WHEN PAYOR_ROWS IS NOT NULL AND ORDER_GIVER_ACCOUNT IS NULL THEN 1 ELSE 0 END),0) AS N_PAYOR_MATCH_WITHOUT_ACCOUNT,
       NVL(SUM(CASE WHEN PAYOR_ROWS > 1 THEN 1 ELSE 0 END),0) AS N_LINKS_TO_MULTIPLE_PAYOR_ROWS,
       NVL(SUM(CASE WHEN N_ORDER_GIVER_ACCOUNTS > 1 THEN 1 ELSE 0 END),0) AS N_LINKS_TO_DIFFERENT_ACCOUNTS
FROM J
"""
print('PAYEE -> PAYOR | PAYER IDENTIFICATION QA | READ ONLY', flush=True)
print(f'SOURCE: BDL_PAYMENT_PLY.PAYEE_VW / BDL_PAYMENT_PLY.PAYOR_VW | MONTH={START}', flush=True)
print('JOIN: PAYEE.LINKED_DEBIT_PAYMENT_ID = PAYOR.DEBIT_PAYMENT_ID', flush=True)
print('SCOPE:', 'FULL MONTH' if MAX_PAYEE_ROWS is None else f'FIRST {MAX_PAYEE_ROWS:,} PAYEE SOURCE ROWS (sample only)', flush=True)
out = ora.query_to_df(q)
out.columns = [str(c).upper() for c in out.columns]
print(out.to_csv(index=False, sep=';', na_rep='NULL').strip(), flush=True)
if not out.empty:
    r = out.iloc[0]
    n = int(r['N_UNIQUE_CREDIT_PAYMENTS'] or 0)
    print(f"PAYER_ACCOUNT_COVERAGE_OF_UNIQUE_CREDITS: {int(r['N_WITH_ORDER_GIVER_ACCOUNT'])}/{n} = {(100*float(r['N_WITH_ORDER_GIVER_ACCOUNT'])/n if n else 0):.2f}%", flush=True)
    print(f"PAYOR_LINK_COVERAGE_OF_UNIQUE_CREDITS: {int(r['N_MATCHED_TO_PAYOR'])}/{n} = {(100*float(r['N_MATCHED_TO_PAYOR'])/n if n else 0):.2f}%", flush=True)
    if int(r['N_AMBIGUOUS_PAYEE_DIRECT_IDS']) or int(r['N_LINKS_TO_DIFFERENT_ACCOUNTS']):
        print('QA STATUS: AMBIGUOUS REFERENCES/ACCOUNTS FOUND; DO NOT SWITCH SOURCE YET', flush=True)
    elif int(r['N_LINKS_TO_MULTIPLE_PAYOR_ROWS']):
        print('QA STATUS: REPEATED PAYOR ROWS; DEDUPLICATE BEFORE ANY AMOUNT AGGREGATION', flush=True)
    else:
        print('QA STATUS: NO MULTIPLE DIRECT IDS / MULTIPLE PAYER ACCOUNTS IN THIS SCOPE', flush=True)
print('NO WRITES. NO RECIPE OR BACKTEST CHANGES.', flush=True)
