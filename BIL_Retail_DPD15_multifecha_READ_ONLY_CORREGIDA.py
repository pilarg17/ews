# Retail EWS | DPD 15 sensitivity, multi-date | READ ONLY
# Exact dates: 15 Oct control, 16-19 Oct analysis
import re
import numpy as np
import pandas as pd
import dataiku
from dataiku.core.sql import SQLExecutor2

FIRST = '2026-10-15'
LAST = '2026-10-19'
START_BUFFER = '2026-10-14'
THRESHOLDS = [1000000, 750000, 500000, 250000]
ORACLE = SQLExecutor2(connection='PRD_BDWH')
PROJECTS = [
    'TEST15_CREDITRISKEARLYWARNINGSYSTEMRETAILPORTFOLIO',
    'CREDIT_RISK_EARLY_WARNING_SYSTEM_RETAIL_PORTFOLIO_FEATURE_CREDITRISKEARLYWARNINGSYSTEMRETAILPORTFOLIO',
]


def norm_id(series):
    return series.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)


def source_frame(dataset_name, day, ids, requested):
    """Only exact-date rows, selected by the source's MAP_DTE, no writes."""
    if not ids:
        return pd.DataFrame(), 'NO_CANDIDATES'
    for key in PROJECTS:
        try:
            ds = dataiku.Dataset(dataset_name, project_key=key, ignore_flow=True)
            columns = {c['name'].upper(): c['name'] for c in ds.read_schema()}
            if any(c not in columns for c in ['MAP_DTE', 'ENTITY_ID'] + requested):
                continue
            loc = ds.get_location_info()
            info = loc.get('info', {})
            table = info.get('quotedResolvedTableName')
            if not table:
                table = info.get('table') or info.get('tableName')
                schema = info.get('schema') or info.get('schemaName')
                if schema and table and '.' not in table:
                    table = '"{}"."{}"'.format(schema, table)
            conn = info.get('connectionName')
            if not table or not conn:
                continue
            # All IDs are validated numeric keys returned from the Oracle query
            safe_ids = [str(v) for v in ids if re.fullmatch(r'\d{1,30}', str(v))]
            if not safe_ids:
                continue
            frames = []
            for i in range(0, len(safe_ids), 300):
                batch = safe_ids[i:i + 300]
                cols = [columns[c] for c in ['ENTITY_ID', 'MAP_DTE'] + requested]
                projections = ', '.join('"{}"'.format(c.replace('"', '""')) for c in cols)
                entity = '"{}"'.format(columns['ENTITY_ID'])
                dt = '"{}"'.format(columns['MAP_DTE'])
                sql = ("SELECT {} FROM {} WHERE TRY_TO_DATE(LEFT(TO_VARCHAR({}),10)) = DATE '{}' "
                       "AND REGEXP_REPLACE(TRIM(TO_VARCHAR({})), '[.]0$', '') IN ({})").format(
                          projections, table, dt, day, entity,
                          ','.join("'{}'".format(v) for v in batch))
                frames.append(SQLExecutor2(connection=conn).query_to_df(sql))
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if not frame.empty:
                frame.columns = [str(c).upper() for c in frame.columns]
                frame['ENTITY_ID'] = norm_id(frame['ENTITY_ID'])
                frame = frame.drop_duplicates(subset=['ENTITY_ID'])
                return frame, key
        except Exception as exc:
            print('Source check {} / {}: {}'.format(dataset_name, key, str(exc)[-150:]), flush=True)
    return pd.DataFrame(), 'NO_EXACT_DATE_DATA'


def flag_values(s):
    return (pd.to_numeric(s, errors='coerce').eq(1) |
            s.astype('string').str.strip().str.upper().isin(['TRUE', 'T', 'Y', 'YES']))


def get_overdraft(day, ids):
    """Replicate Customer_Balances overdraft: T24 category 1001, negative working balance."""
    if not ids:
        return pd.DataFrame(columns=['ENTITY_ID', 'OVERDRAFT_AMOUNT'])
    frames = []
    ids = [str(x) for x in ids if re.fullmatch(r'\d{1,30}', str(x))]
    for i in range(0, len(ids), 250):
        batch = ids[i:i + 250]
        sql = """
        SELECT TRIM(TO_CHAR(ec.ENTITY_ID)) AS ENTITY_ID,
               SUM(CASE WHEN TRIM(TO_CHAR(a.CATEGORY)) = '1001'
                         AND NVL(a.WORKING_BALANCE,0) < 0
                        THEN ABS(NVL(a.WORKING_BALANCE,0)) ELSE 0 END) AS OVERDRAFT_AMOUNT
        FROM BDL_CREDIT_INSIGHT_PLY.ENTITY_CUSTOMER_VW ec
        LEFT JOIN BDWH_PLY.PLY_T24_ACCOUNT_VW a
          ON TRIM(CAST(a.CUSTOMER AS VARCHAR2(100))) = TRIM(CAST(ec.CUSTOMER_ID AS VARCHAR2(100)))
         AND TRUNC(a.TEK_STRT_DTE) <= DATE '{day}'
         AND DATE '{day}' < TRUNC(NVL(a.TEK_END_DTE,DATE '9999-12-31'))
        WHERE TRIM(TO_CHAR(ec.ENTITY_ID)) IN ({entity_ids})
          AND TRUNC(ec.TEK_STRT_DTE) <= DATE '{day}'
          AND DATE '{day}' < TRUNC(NVL(ec.TEK_END_DTE,DATE '9999-12-31'))
        GROUP BY TRIM(TO_CHAR(ec.ENTITY_ID))
        """.format(day=day, entity_ids=','.join("'{}'".format(s) for s in batch))
        frames.append(ORACLE.query_to_df(sql))
    result = pd.concat(frames, ignore_index=True)
    result.columns = [str(c).upper() for c in result.columns]
    result['ENTITY_ID'] = norm_id(result['ENTITY_ID'])
    return result.groupby('ENTITY_ID', as_index=False)['OVERDRAFT_AMOUNT'].sum()


print('DPD 15 | HISTORICAL MULTI-DATE | READ ONLY', flush=True)
print('Source: Oracle daily instrument DPD; T24 as-of overdraft; exact-date Retail/GEC only.', flush=True)
print('No Flow runs, no dataset writes.\n', flush=True)

# Historic daily snapshots: 14-19 October, including weekends only if Oracle has usable records.
sql_dpd = """
SELECT TRUNC(ins.MAP_DTE) AS MAP_DTE,
       TRIM(TO_CHAR(link.ENTITY_ID)) AS ENTITY_ID,
       MAX(CASE WHEN ABS(NVL(ins.DPD,0)) >= 10000000000 THEN 0
                ELSE NVL(ins.DPD,0) END) AS DPD,
       COUNT(ins.DPD) AS DPD_SOURCE_ROWS
FROM BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_DAY_VW ins
JOIN BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_ENTITY_DAY_VW link
  ON ins.INSTRUMENT_ID = link.INSTRUMENT_ID AND ins.MAP_DTE = link.MAP_DTE
WHERE ins.MAP_DTE >= DATE '{start}'
  AND ins.MAP_DTE < DATE '{end}'
  AND link.ENTITY_ID IS NOT NULL
GROUP BY TRUNC(ins.MAP_DTE), TRIM(TO_CHAR(link.ENTITY_ID))
""".format(start=START_BUFFER, end=(pd.Timestamp(LAST) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

source = ORACLE.query_to_df(sql_dpd)
if source.empty:
    raise RuntimeError('Oracle returned no daily DPD observations. Nothing to report.')
source.columns = [str(c).upper() for c in source.columns]
source['ENTITY_ID'] = norm_id(source['ENTITY_ID'])
source['MAP_DTE'] = pd.to_datetime(source['MAP_DTE'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
source['DPD'] = pd.to_numeric(source['DPD'], errors='coerce')
source['DPD_SOURCE_ROWS'] = pd.to_numeric(source['DPD_SOURCE_ROWS'], errors='coerce').fillna(0)
coverage = source.groupby('MAP_DTE', as_index=False).agg(
    ENTITIES=('ENTITY_ID','nunique'), NON_NULL_DPD_ROWS=('DPD_SOURCE_ROWS','sum'))
print('ORACLE DPD SNAPSHOT COVERAGE (including baseline):')
print(coverage.to_string(index=False), flush=True)
usable = set(coverage.loc[coverage.NON_NULL_DPD_ROWS.gt(0), 'MAP_DTE'])
results = []

for day in pd.date_range(FIRST, LAST):
    label = day.strftime('%Y-%m-%d')
    if day.weekday() >= 5:
        results.append(dict(DATE=label, PREVIOUS='-', STATUS='WEEKEND_NO_SCHEDULED_EWS_RUN',
                            DPD_ABOVE_15=np.nan, RAW_CROSSINGS=np.nan, RETAIL_CROSSINGS=np.nan,
                            RRE_CROSSINGS=np.nan, OTHER_RETAIL_CROSSINGS=np.nan,
                            AFTER_OD_500=np.nan, GEC_AVAILABLE=np.nan,
                            **{'GEC_GT_{}'.format(k): np.nan for k in THRESHOLDS}))
        continue
    prev_expected = day - pd.offsets.BDay(1)
    if day not in usable or prev_expected not in usable:
        results.append(dict(DATE=label, PREVIOUS='-', STATUS='NO_USABLE_DPD_SNAPSHOT',
                            DPD_ABOVE_15=np.nan, RAW_CROSSINGS=np.nan, RETAIL_CROSSINGS=np.nan,
                            RRE_CROSSINGS=np.nan, OTHER_RETAIL_CROSSINGS=np.nan,
                            AFTER_OD_500=np.nan, GEC_AVAILABLE=np.nan,
                            **{'GEC_GT_{}'.format(k): np.nan for k in THRESHOLDS}))
        continue
    prev_date = prev_expected
    prev_str = prev_date.strftime('%Y-%m-%d')
    curr = source.loc[source.MAP_DTE.eq(day), ['ENTITY_ID','DPD']].drop_duplicates('ENTITY_ID')
    prev = source.loc[source.MAP_DTE.eq(prev_date), ['ENTITY_ID','DPD']].rename(columns={'DPD':'PREVIOUS_DPD'}).drop_duplicates('ENTITY_ID')
    merged = curr.merge(prev, how='left', on='ENTITY_ID', validate='one_to_one')
    crossed = merged.loc[merged.DPD.gt(15) & merged.PREVIOUS_DPD.notna()
                         & merged.PREVIOUS_DPD.le(15)].copy()
    row = dict(DATE=label, PREVIOUS=prev_str, STATUS='RAW_ONLY',
               DPD_ABOVE_15=int(curr.DPD.gt(15).sum()),
               RAW_CROSSINGS=len(crossed), RETAIL_CROSSINGS=np.nan,
               RRE_CROSSINGS=np.nan, OTHER_RETAIL_CROSSINGS=np.nan,
               AFTER_OD_500=np.nan, GEC_AVAILABLE=np.nan,
               **{'GEC_GT_{}'.format(k): np.nan for k in THRESHOLDS})
    if crossed.empty:
        row.update(STATUS='NO_CROSSINGS', RETAIL_CROSSINGS=0, RRE_CROSSINGS=0,
                   OTHER_RETAIL_CROSSINGS=0, AFTER_OD_500=0,
                   GEC_AVAILABLE=0, **{'GEC_GT_{}'.format(k): 0 for k in THRESHOLDS})
        results.append(row)
        continue

    # Scope is deliberately date-specific. Never use a current Retail snapshot for a historical date.
    portfolio, p_source = source_frame('Retail_Portfolio_Daily', label,
                                       crossed.ENTITY_ID.tolist(), ['RRE_FLAG','OTHER_RETAIL_FLAG'])
    if not portfolio.empty:
        portfolio['IS_RRE'] = flag_values(portfolio.RRE_FLAG)
        portfolio['IS_OTHER'] = flag_values(portfolio.OTHER_RETAIL_FLAG)
        portfolio['RETAIL_INCLUDED'] = portfolio.IS_RRE | portfolio.IS_OTHER
        crossed = crossed.merge(portfolio[['ENTITY_ID','RETAIL_INCLUDED','IS_RRE','IS_OTHER']], how='left', on='ENTITY_ID')
        crossed = crossed.loc[crossed.RETAIL_INCLUDED.eq(True)].copy()
        row['RETAIL_CROSSINGS'] = len(crossed)
        row['RRE_CROSSINGS'] = int(crossed.IS_RRE.sum())
        row['OTHER_RETAIL_CROSSINGS'] = int(crossed.IS_OTHER.sum())
        row['STATUS'] = 'RETAIL_SCOPED'
    else:
        row['STATUS'] = 'RETAIL_SCOPE_MISSING'
        # Continue with unscoped numbers; never label them as Retail alerts.

    if crossed.empty:
        row['AFTER_OD_500'] = 0
        row['GEC_AVAILABLE'] = 0
        row.update({'GEC_GT_{}'.format(k): 0 for k in THRESHOLDS})
        results.append(row)
        continue

    try:
        od = get_overdraft(label, crossed.ENTITY_ID.tolist())
        crossed = crossed.merge(od, how='left', on='ENTITY_ID', validate='one_to_one')
        crossed['OVERDRAFT_AMOUNT'] = pd.to_numeric(crossed['OVERDRAFT_AMOUNT'], errors='coerce')
        crossed = crossed.loc[crossed.OVERDRAFT_AMOUNT.gt(500)].copy()
        if not portfolio.empty:
            row['AFTER_OD_500'] = len(crossed)
        else:
            row['STATUS'] += '|OD_UNSCOPED'
    except Exception as exc:
        print('{} | Historical overdraft unavailable: {}'.format(label,str(exc)[-250:]),flush=True)
        row['STATUS'] += '|OD_MISSING'
        results.append(row)
        continue

    if crossed.empty:
        row['GEC_AVAILABLE'] = 0
        row.update({'GEC_GT_{}'.format(k): 0 for k in THRESHOLDS})
        results.append(row)
        continue

    gec, g_source = source_frame('GEC_Exposure_Current', label,
                                 crossed.ENTITY_ID.tolist(), ['GEC_NET_EXPOSURE'])
    if gec.empty:
        row['STATUS'] += '|GEC_HISTORY_MISSING'
    else:
        crossed = crossed.merge(gec[['ENTITY_ID','GEC_NET_EXPOSURE']], how='left', on='ENTITY_ID', validate='one_to_one')
        crossed['GEC_NET_EXPOSURE'] = pd.to_numeric(crossed.GEC_NET_EXPOSURE, errors='coerce')
        if not portfolio.empty:
            row['GEC_AVAILABLE'] = int(crossed.GEC_NET_EXPOSURE.notna().sum())
            for k in THRESHOLDS:
                row['GEC_GT_{}'.format(k)] = int(crossed.GEC_NET_EXPOSURE.gt(k).sum()) if crossed.GEC_NET_EXPOSURE.notna().all() else np.nan
            if crossed.GEC_NET_EXPOSURE.isna().any():
                row['STATUS'] += '|PARTIAL_GEC'
        else:
            row['STATUS'] += '|UNSCOPED_GEC'
    results.append(row)

summary = pd.DataFrame(results)
print('\nDPD 15 SENSITIVITY:')
print(summary.to_string(index=False, na_rep='N/A'), flush=True)
print('\nINTERPRETATION:')
print('RAW_CROSSINGS = DPD crossing among all mapped Oracle entities, before Retail scope.')
print('RETAIL_CROSSINGS = only when Retail_Portfolio_Daily exists for that exact date.')
print('AFTER_OD_500 = positive current-account overdraft > EUR 500, Retail-scoped when available.')
print('GEC_GT_* = only when a complete exact-date GEC source exists; N/A is NOT zero.')
print('These are DPD15 sensitivity candidates, NOT final EWS breaches (bundling/exclusions not reproduced).')
print('No datasets or recipes changed.', flush=True)
