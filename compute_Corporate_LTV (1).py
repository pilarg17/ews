# -*- coding: utf-8 -*-
"""Corporate EWS: monthly LTV and eligible protection subtypes."""
import re
import time
import numpy as np
import pandas as pd
import dataiku
from dataiku.core.sql import SQLExecutor2
from dataiku import recipe

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
CONN_SNOW = 'DEV_AA_TEAMS'
CONN_ORA = 'PRD_BDWH'
SOURCE_LOANS = 'vars_loan_level'
OUTPUT = 'ltv_monthly_Corporate'
TRAIN_START = '2024-11-07'
TRAIN_END = '2026-10-29'
ELIGIBLE_PROTECTION_TYPES = ('8', '9', '10')
MAX_SOURCE_LAG_DAYS = 7
MAX_LOANS_PER_MONTH = 120000
MAX_ORACLE_INSTRUMENTS = 250000
T0 = time.monotonic()

ORACLE_REG = 'BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_REGULATORY_METRICS_DAY_VW'
ORACLE_LINK = 'BDL_CREDIT_INSIGHT_PLY.INSTRUMENT_PROTECTION_DAY_VW'
ORACLE_PROTECTION = 'BDL_CREDIT_INSIGHT_PLY.PROTECTION_DAY_VW'
ORACLE_METRICS = 'BDL_CREDIT_INSIGHT_PLY.PROTECTION_METRICS_DAY_VW'
FIELD = 'PROTECTION_INTERNAL_SUBTYPE_DESC'
NEW_COLUMNS = [FIELD, 'N_DISTINCT_PROTECTION_SUBTYPES', 'N_MISSING_PROTECTION_SUBTYPES',
    'CONTRACT_INTERNAL_PURPOSE_CODE', 'LTV_POLICY_CATEGORY', 'LTV_POLICY_THRESHOLD',
    'LTV_POLICY_BREACH', 'LTV_POLICY_STATUS']
BASE_COLUMNS = ['MAP_DTE','INSTRUMENT_ID','ENTITY_ID','LOAN_CURRENCY','NOMINAL_AMOUNT',
    'REGULATORY_EXPOSURE','LTV_SNAPSHOT_DATE','PROTECTION_METRICS_DATE',
    'N_TOTAL_PROTECTION_LINKS','N_REAL_ESTATE_LINKS','N_UNCLASSIFIED_LINKS',
    'N_SHARED_REAL_ESTATE_LINKS','N_VALID_REAL_ESTATE_LINKS',
    'N_INCOMPLETE_REAL_ESTATE_LINKS','N_AMBIGUOUS_VALUATIONS',
    'N_BAD_SHARED_EXPOSURES','N_DUPLICATE_LINKS','ALLOCATED_COLLATERAL_VALUE',
    'LTV_RETAIL_METHOD','LTV_STATUS']
KEYS = ['MAP_DTE','INSTRUMENT_ID']
# Preserve the historical loan population while recalculating collateral values.
class MissingSnapshot(RuntimeError):
    pass


def log(*parts):
    print('[%8.1fs] %s' % (time.monotonic()-T0, ' | '.join(map(str,parts))), flush=True)


def get(con, label, sql, cap):
    log('QUERY START', label)
    tic = time.monotonic()
    df = SQLExecutor2(connection=con).query_to_df(sql)
    if len(df) > cap:
        raise RuntimeError('%s returned %d rows (limit=%d)' % (label,len(df),cap))
    df.columns = [str(k).upper() for k in df.columns]
    log('QUERY DONE',label,'rows=%d'%len(df),'seconds=%.1f'%(time.monotonic()-tic))
    return df


def sqlident(v):
    v = str(v).replace('${projectKey}',PROJECT)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*',v):
        raise RuntimeError('Unsafe physical identifier')
    return '"'+v+'"'


def norm(v):
    if pd.isna(v): return None
    s = str(v).strip()
    if s.startswith('I|'): s = s[2:]
    if re.fullmatch(r'[0-9]+[.]0+',s): s=s.split('.')[0]
    return s or None


def required_dataiku(name, columns):
    ds = dataiku.Dataset(name,project_key=PROJECT)
    if columns:
        present = {str(x['name']).upper() for x in ds.read_schema()}
        if set(columns)-present:
            raise RuntimeError('Dataset missing columns: '+name+' '+str(sorted(set(columns)-present)))
    prm = dataiku.api_client().get_project(PROJECT).get_dataset(name).get_settings().get_raw().get('params') or {}
    if prm.get('connection') != CONN_SNOW:
        raise RuntimeError('Unexpected Snowflake connection: '+name)
    return sqlident(prm['schema'])+'.'+sqlident(prm['table'])


def date_value(df, col, label, target):
    if df.empty or pd.isna(df[col].iloc[0]):
        raise MissingSnapshot('Missing as-of date for '+label+' at '+target)
    day = pd.Timestamp(df[col].iloc[0])
    if day.tzinfo is not None: day=day.tz_localize(None)
    day=day.normalize()
    lag=(pd.Timestamp(target)-day).days
    if not 0<=lag<=MAX_SOURCE_LAG_DAYS:
        raise RuntimeError('Invalid/stale as-of date: '+label+' '+str(day.date())+' lag='+str(lag))
    return day.strftime('%Y-%m-%d')


def ora_dates(target):
    d_int = int(target.replace('-',''))
    reg = get(CONN_ORA,'REGULATORY AS-OF '+target,
        "SELECT MAX(REFERENCE_DATE) SOURCE_DATE FROM %s "
        "WHERE REFERENCE_DATE <= %d AND REFERENCE_DATE >= %d" %
        (ORACLE_REG,d_int,int((pd.Timestamp(target)-pd.Timedelta(days=MAX_SOURCE_LAG_DAYS)).strftime('%Y%m%d'))),2)
    if reg.empty or pd.isna(reg['SOURCE_DATE'].iloc[0]):
        raise MissingSnapshot('No regulatory reference date for '+target)
    reg_int = str(int(float(reg['SOURCE_DATE'].iloc[0])))
    reg_day = pd.to_datetime(reg_int,format='%Y%m%d').strftime('%Y-%m-%d')
    if (pd.Timestamp(target)-pd.Timestamp(reg_day)).days>MAX_SOURCE_LAG_DAYS:
        raise RuntimeError('Regulatory snapshot too old for '+target)
    source_dates = {}
    for label,source in [('LINK',ORACLE_LINK),('PROTECTION',ORACLE_PROTECTION)]:
        sql = "SELECT MAX(TRUNC(MAP_DTE)) SOURCE_DATE FROM %s " \
              "WHERE MAP_DTE >= DATE '%s' - %d AND MAP_DTE < DATE '%s' + 1" % \
              (source,reg_day,MAX_SOURCE_LAG_DAYS,reg_day)
        f = get(CONN_ORA,label+' AS-OF '+target,sql,2)
        source_dates[label]=date_value(f,'SOURCE_DATE',label,reg_day)
    if source_dates['LINK'] != source_dates['PROTECTION']:
        raise RuntimeError('Link/protection source date mismatch: '+str(source_dates))
    if source_dates['LINK'] != reg_day:
        raise RuntimeError('Retail baseline requires matching regulatory and protection day: '+str({'REG':reg_day,**source_dates}))
    q = "SELECT MAX(TRUNC(VALID_DATE)) SOURCE_DATE FROM %s WHERE " \
        "VALID_DATE >= DATE '%s' - %d AND VALID_DATE < DATE '%s' + 1 " \
        "AND MAP_DTE < DATE '%s' + 1" % \
        (ORACLE_METRICS,reg_day,MAX_SOURCE_LAG_DAYS,reg_day,reg_day)
    m = get(CONN_ORA,'METRICS VALID_DATE AS-OF '+target,q,2)
    metrics_day=date_value(m,'SOURCE_DATE','METRICS VALID_DATE',reg_day)
    log('SOURCE_DATES','TARGET='+target,'REG='+reg_day,'PROTECTION='+source_dates['LINK'],'METRICS_VALID='+metrics_day)
    return reg_day,reg_int,metrics_day


def retail_ltv_sql(reg_d,reg_int,metrics_d):
    # All bank instruments linked to a protection participate in its allocation.
    allowed=','.join("'%s'"%x for x in ELIGIBLE_PROTECTION_TYPES)
    return r"""
WITH
REG_BASE AS (
    SELECT TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE)) INSTRUMENT_ID,
           COUNT(*) N_ROWS,
           COUNT(DISTINCT OUTSTANDING_NOMINAL_AMOUNT) N_NOMINAL_VALUES,
           COUNT(DISTINCT NVL(OFF_BALANCE_SHEET_AMOUNT,0)) N_OFF_VALUES,
           SUM(CASE WHEN OUTSTANDING_NOMINAL_AMOUNT IS NULL OR
                         OUTSTANDING_NOMINAL_AMOUNT < 0 OR
                         OUTSTANDING_NOMINAL_AMOUNT >= 11111111111 OR
                         NVL(OFF_BALANCE_SHEET_AMOUNT,0) < 0
                    THEN 1 ELSE 0 END) N_INVALID_EXPOSURE,
           MAX(OUTSTANDING_NOMINAL_AMOUNT) NOMINAL_REG,
           MAX(NVL(OFF_BALANCE_SHEET_AMOUNT,0)) OFF_BALANCE_REG
    FROM {reg}
    WHERE REFERENCE_DATE={reg_int}
      AND SOURCE_CONTRACT_REFERENCE IS NOT NULL
      AND SOURCE_CONTRACT_REFERENCE NOT LIKE '211MD%'
    GROUP BY TRIM(TO_CHAR(SOURCE_CONTRACT_REFERENCE))
),
REG AS (
    SELECT INSTRUMENT_ID,
           CASE WHEN N_INVALID_EXPOSURE=0 AND N_NOMINAL_VALUES=1 AND
                     N_OFF_VALUES<=1 AND NOMINAL_REG+OFF_BALANCE_REG>0
                THEN NOMINAL_REG+OFF_BALANCE_REG END EXPOSURE,
           N_ROWS,
           CASE WHEN N_INVALID_EXPOSURE=0 AND N_NOMINAL_VALUES=1 AND
                     N_OFF_VALUES<=1 AND NOMINAL_REG+OFF_BALANCE_REG>0
                THEN 1 ELSE 0 END EXPOSURE_VALID
    FROM REG_BASE
),
LINKS AS (
    SELECT TRIM(TO_CHAR(INSTRUMENT_ID)) INSTRUMENT_ID,
           TRIM(TO_CHAR(PROTECTION_ID)) PROTECTION_ID,
           COUNT(*) N_RAW_LINKS
    FROM {links}
    WHERE MAP_DTE >= DATE '{d}' AND MAP_DTE < DATE '{d}'+1
      AND INSTRUMENT_ID IS NOT NULL AND PROTECTION_ID IS NOT NULL
    GROUP BY TRIM(TO_CHAR(INSTRUMENT_ID)),TRIM(TO_CHAR(PROTECTION_ID))
),
PROTECTION_BASE AS (
    SELECT TRIM(TO_CHAR(PROTECTION_ID)) PROTECTION_ID,
           COUNT(DISTINCT TRIM(TO_CHAR(PROTECTION_TYPE))) N_TYPES,
           MAX(TRIM(TO_CHAR(PROTECTION_TYPE))) PROTECTION_TYPE,
           COUNT(*) N_PROTECTION_ROWS,
           CASE WHEN COUNT(DISTINCT TRIM(PROTECTION_INTERNAL_SUBTYPE_DESC))=1
                 AND COUNT(TRIM(PROTECTION_INTERNAL_SUBTYPE_DESC))=COUNT(*)
                THEN MAX(TRIM(PROTECTION_INTERNAL_SUBTYPE_DESC))
           END PROTECTION_INTERNAL_SUBTYPE_DESC
    FROM {prot}
    WHERE MAP_DTE >= DATE '{d}' AND MAP_DTE < DATE '{d}'+1
      AND PROTECTION_ID IS NOT NULL
    GROUP BY TRIM(TO_CHAR(PROTECTION_ID))
),
METRICS_BASE AS (
    SELECT TRIM(TO_CHAR(PROTECTION_ID)) PROTECTION_ID,
           COUNT(*) N_METRIC_ROWS,
           COUNT(DISTINCT NET_MARKET_VALUE_LCY) N_NET_VALUES,
           COUNT(DISTINCT COLLATERAL_COVER_AMOUNT_LCY) N_COVER_VALUES,
           MAX(NET_MARKET_VALUE_LCY) NET_VALUE,
           MAX(COLLATERAL_COVER_AMOUNT_LCY) COVER_VALUE
    FROM {metrics}
    WHERE VALID_DATE >= DATE '{m}' AND VALID_DATE < DATE '{m}'+1
      AND MAP_DTE < DATE '{d}'+1
      AND PROTECTION_ID IS NOT NULL
    GROUP BY TRIM(TO_CHAR(PROTECTION_ID))
),
PROTECTION_VALUES AS (
    SELECT P.PROTECTION_ID, P.PROTECTION_TYPE, P.N_TYPES,
           P.PROTECTION_INTERNAL_SUBTYPE_DESC,
           CASE WHEN P.PROTECTION_TYPE IN ({allowed}) AND P.N_TYPES=1
                     THEN 1 ELSE 0 END ELIGIBLE,
           CASE WHEN M.N_NET_VALUES>1 OR M.N_COVER_VALUES>1 THEN 1 ELSE 0 END AMBIGUOUS_METRICS,
           CASE WHEN M.N_NET_VALUES<=1 AND M.NET_VALUE>2 THEN M.NET_VALUE
                WHEN M.N_COVER_VALUES<=1 AND M.COVER_VALUE>2 THEN M.COVER_VALUE
                ELSE NULL END MARKET_OR_COVER_VALUE
    FROM PROTECTION_BASE P
    LEFT JOIN METRICS_BASE M ON P.PROTECTION_ID=M.PROTECTION_ID
),
SHARED AS (
    SELECT L.PROTECTION_ID,
           COUNT(*) N_LINKED_INSTRUMENTS,
           SUM(CASE WHEN L.N_RAW_LINKS<>1 THEN 1 ELSE 0 END) N_DUP_LINKS,
           SUM(CASE WHEN R.EXPOSURE_VALID=1 THEN R.EXPOSURE ELSE 0 END) SHARED_EXPOSURE,
           SUM(CASE WHEN NVL(R.EXPOSURE_VALID,0)<>1 THEN 1 ELSE 0 END) N_BAD_EXPOSURES
    FROM LINKS L
    LEFT JOIN REG R ON L.INSTRUMENT_ID=R.INSTRUMENT_ID
    GROUP BY L.PROTECTION_ID
),
PAIR AS (
    SELECT L.INSTRUMENT_ID, L.PROTECTION_ID, L.N_RAW_LINKS,
           P.PROTECTION_INTERNAL_SUBTYPE_DESC,
           NVL(P.ELIGIBLE,0) ELIGIBLE,
           CASE WHEN P.PROTECTION_ID IS NULL OR P.N_TYPES<>1 THEN 1 ELSE 0 END UNKNOWN_PROTECTION_TYPE,
           NVL(P.AMBIGUOUS_METRICS,0) AMBIGUOUS_METRICS,
           P.MARKET_OR_COVER_VALUE COLLATERAL_VALUE,
           S.N_LINKED_INSTRUMENTS, S.N_DUP_LINKS,
           S.N_BAD_EXPOSURES, S.SHARED_EXPOSURE,
           R.EXPOSURE,
           CASE WHEN P.ELIGIBLE=1 AND L.N_RAW_LINKS=1
                     AND S.N_DUP_LINKS=0 AND S.N_BAD_EXPOSURES=0
                     AND S.SHARED_EXPOSURE>0 AND P.AMBIGUOUS_METRICS=0
                     AND P.MARKET_OR_COVER_VALUE>2 AND R.EXPOSURE_VALID=1
                THEN P.MARKET_OR_COVER_VALUE * R.EXPOSURE / S.SHARED_EXPOSURE
                ELSE NULL END ALLOCATED_COLLATERAL_VALUE
    FROM LINKS L
    LEFT JOIN PROTECTION_VALUES P ON L.PROTECTION_ID=P.PROTECTION_ID
    JOIN SHARED S ON L.PROTECTION_ID=S.PROTECTION_ID
    LEFT JOIN REG R ON L.INSTRUMENT_ID=R.INSTRUMENT_ID
),
AGG AS (
    SELECT INSTRUMENT_ID,
           COUNT(*) N_TOTAL_PROTECTION_LINKS,
           COUNT(DISTINCT CASE WHEN ELIGIBLE=1 THEN PROTECTION_INTERNAL_SUBTYPE_DESC END)
                N_DISTINCT_PROTECTION_SUBTYPES,
           SUM(CASE WHEN ELIGIBLE=1 AND PROTECTION_INTERNAL_SUBTYPE_DESC IS NULL THEN 1 ELSE 0 END)
                N_MISSING_PROTECTION_SUBTYPES,
           SUM(CASE WHEN ELIGIBLE=1 THEN 1 ELSE 0 END) N_REAL_ESTATE_LINKS,
           SUM(CASE WHEN UNKNOWN_PROTECTION_TYPE=1 THEN 1 ELSE 0 END) N_UNCLASSIFIED_LINKS,
           SUM(CASE WHEN ELIGIBLE=1 AND N_LINKED_INSTRUMENTS>1 THEN 1 ELSE 0 END) N_SHARED_REAL_ESTATE_LINKS,
           SUM(CASE WHEN ELIGIBLE=1 AND ALLOCATED_COLLATERAL_VALUE IS NULL THEN 1 ELSE 0 END) N_INCOMPLETE_REAL_ESTATE_LINKS,
           SUM(CASE WHEN ELIGIBLE=1 AND AMBIGUOUS_METRICS=1 THEN 1 ELSE 0 END) N_AMBIGUOUS_VALUATIONS,
           SUM(CASE WHEN ELIGIBLE=1 AND N_BAD_EXPOSURES>0 THEN 1 ELSE 0 END) N_BAD_SHARED_EXPOSURES,
           SUM(CASE WHEN ELIGIBLE=1 AND N_DUP_LINKS>0 THEN 1 ELSE 0 END) N_DUPLICATE_LINKS,
           SUM(CASE WHEN ELIGIBLE=1 THEN ALLOCATED_COLLATERAL_VALUE END) ALLOCATED_COLLATERAL_VALUE,
           SUM(CASE WHEN ELIGIBLE=1 AND ALLOCATED_COLLATERAL_VALUE>0 THEN 1 ELSE 0 END) N_VALID_REAL_ESTATE_LINKS
    FROM PAIR GROUP BY INSTRUMENT_ID
),
SUBTYPES_DISTINCT AS (
    SELECT DISTINCT INSTRUMENT_ID, PROTECTION_INTERNAL_SUBTYPE_DESC
    FROM PAIR WHERE ELIGIBLE=1 AND PROTECTION_INTERNAL_SUBTYPE_DESC IS NOT NULL
),
SUBTYPES_AGG AS (
    SELECT INSTRUMENT_ID,
           LISTAGG(PROTECTION_INTERNAL_SUBTYPE_DESC, ' | ')
             WITHIN GROUP (ORDER BY PROTECTION_INTERNAL_SUBTYPE_DESC) PROTECTION_INTERNAL_SUBTYPE_DESC
    FROM SUBTYPES_DISTINCT GROUP BY INSTRUMENT_ID
)
SELECT 'I|'||A.INSTRUMENT_ID INSTRUMENT_ID,
       R.EXPOSURE AS REGULATORY_EXPOSURE,
       A.N_TOTAL_PROTECTION_LINKS, A.N_REAL_ESTATE_LINKS, A.N_UNCLASSIFIED_LINKS,
       A.N_SHARED_REAL_ESTATE_LINKS, A.N_INCOMPLETE_REAL_ESTATE_LINKS,
       A.N_AMBIGUOUS_VALUATIONS, A.N_BAD_SHARED_EXPOSURES,
       A.N_DUPLICATE_LINKS, A.N_VALID_REAL_ESTATE_LINKS,
       CASE WHEN A.N_REAL_ESTATE_LINKS>0 AND A.N_INCOMPLETE_REAL_ESTATE_LINKS=0
                AND A.N_VALID_REAL_ESTATE_LINKS=A.N_REAL_ESTATE_LINKS
                AND A.ALLOCATED_COLLATERAL_VALUE>0
                THEN A.ALLOCATED_COLLATERAL_VALUE END ALLOCATED_COLLATERAL_VALUE,
       CASE WHEN R.EXPOSURE_VALID=1 AND A.N_REAL_ESTATE_LINKS>0
                AND A.N_INCOMPLETE_REAL_ESTATE_LINKS=0
                AND A.N_VALID_REAL_ESTATE_LINKS=A.N_REAL_ESTATE_LINKS
                AND A.ALLOCATED_COLLATERAL_VALUE>0
                THEN R.EXPOSURE/A.ALLOCATED_COLLATERAL_VALUE END LTV_RECONSTRUCTED,
       S.PROTECTION_INTERNAL_SUBTYPE_DESC,
       A.N_DISTINCT_PROTECTION_SUBTYPES,
       A.N_MISSING_PROTECTION_SUBTYPES
FROM AGG A
LEFT JOIN REG R ON A.INSTRUMENT_ID=R.INSTRUMENT_ID
LEFT JOIN SUBTYPES_AGG S ON A.INSTRUMENT_ID=S.INSTRUMENT_ID
""".format(reg=ORACLE_REG, reg_int=reg_int, links=ORACLE_LINK,
           prot=ORACLE_PROTECTION,metrics=ORACLE_METRICS,d=reg_d,m=metrics_d,allowed=allowed)


def fetch_loan_snapshot(table, d):
    q = "SELECT TRIM(TO_VARCHAR(INSTRUMENT_ID)) INSTRUMENT_ID, " \
        "TRIM(TO_VARCHAR(ENTITY_ID)) ENTITY_ID, " \
        "UPPER(TRIM(TO_VARCHAR(CURRENCY))) LOAN_CURRENCY, " \
        "TRY_TO_DOUBLE(TO_VARCHAR(NOMINAL_AMOUNT)) NOMINAL_AMOUNT, " \
        "TRIM(TO_VARCHAR(CONTRACT_INTERNAL_PURPOSE_CODE)) CONTRACT_INTERNAL_PURPOSE_CODE " \
        "FROM %s WHERE TO_DATE(MAP_DTE)=DATE '%s' AND INSTRUMENT_ID IS NOT NULL" % (table,d)
    frame=get(CONN_SNOW,'CORPORATE LOANS '+d,q,MAX_LOANS_PER_MONTH)
    if frame.empty: raise RuntimeError('No Corporate loans '+d)
    frame['INSTRUMENT_ID']=frame['INSTRUMENT_ID'].map(norm)
    frame['ENTITY_ID']=frame['ENTITY_ID'].map(norm)
    frame['NOMINAL_AMOUNT']=pd.to_numeric(frame['NOMINAL_AMOUNT'],errors='coerce')
    frame['LOAN_CURRENCY']=frame['LOAN_CURRENCY'].fillna('UNKNOWN').replace('','UNKNOWN')
    if frame['INSTRUMENT_ID'].isna().any() or frame['INSTRUMENT_ID'].duplicated().any():
        raise RuntimeError('Corporate loan keys null or duplicated at '+d)
    return frame


# ADC policy mapping. Only explicit observed purpose codes are used.
# Residential land projects are treated as existing projects for now.
# The code-to-policy correspondence remains a candidate mapping.
ADC_RESIDENTIAL_LAND = {402, 412}
ADC_COMMERCIAL_LAND = {403, 414, 415}
ADC_CONSTRUCTION = {422, 423, 424, 432, 433, 435, 442, 443}
LAND_SUBTYPES = {'Building lot', 'Land for building projects', 'Land for professional usage'}
RESIDENTIAL_LAND_SUBTYPES = {'Building lot', 'Land for building projects'}
CONSTRUCTION_SUBTYPES = LAND_SUBTYPES | {
    'House', 'Studio / apartment', 'Residential building', 'Duplex apartment',
    'Penthouse', 'Triplex apartment', 'Commercial building', 'Commercial premises',
    'Industrial / artisanal building', 'Administration building', 'Storehouse building'
}
POLICY = {'ADC_RESIDENTIAL_LAND_EXISTING_PROJECT': 0.80,
          'ADC_COMMERCIAL_LAND': 0.80, 'ADC_CONSTRUCTION': 0.70}


def apply_policy(df):
    purpose = pd.to_numeric(df['CONTRACT_INTERNAL_PURPOSE_CODE'], errors='coerce')
    valid = df['LTV_STATUS'].eq('CANDIDATE_RETAIL_METHOD') & df['LTV_RETAIL_METHOD'].notna()
    complete = (df['N_DISTINCT_PROTECTION_SUBTYPES'].eq(1)
                & df['N_MISSING_PROTECTION_SUBTYPES'].eq(0))
    subtype = df[FIELD]
    df['LTV_POLICY_CATEGORY'] = None
    df['LTV_POLICY_STATUS'] = 'OUTSIDE_ADC_MAPPING_OR_UNMAPPED'
    df.loc[~valid, 'LTV_POLICY_STATUS'] = 'NO_VALID_LTV'
    df.loc[valid & ~complete, 'LTV_POLICY_STATUS'] = 'COLLATERAL_CLASSIFICATION_PENDING'
    rules = [
        (purpose.isin(ADC_RESIDENTIAL_LAND) & subtype.isin(RESIDENTIAL_LAND_SUBTYPES),
         'ADC_RESIDENTIAL_LAND_EXISTING_PROJECT'),
        (purpose.isin(ADC_COMMERCIAL_LAND) & subtype.isin(LAND_SUBTYPES),
         'ADC_COMMERCIAL_LAND'),
        (purpose.isin(ADC_CONSTRUCTION) & subtype.isin(CONSTRUCTION_SUBTYPES),
         'ADC_CONSTRUCTION'),
    ]
    for condition, category in rules:
        mapped = valid & complete & condition
        df.loc[mapped, 'LTV_POLICY_CATEGORY'] = category
        df.loc[mapped, 'LTV_POLICY_STATUS'] = 'PROVISIONAL_MAPPING'
    df['LTV_POLICY_THRESHOLD'] = df['LTV_POLICY_CATEGORY'].map(POLICY).astype(float)
    df['LTV_POLICY_BREACH'] = np.nan
    mapped = df['LTV_POLICY_THRESHOLD'].notna()
    df.loc[mapped, 'LTV_POLICY_BREACH'] = (
        df.loc[mapped, 'LTV_RETAIL_METHOD'] > df.loc[mapped, 'LTV_POLICY_THRESHOLD']).astype(float)
    return df


def build_month(table, target):
    log('MONTH START',target)
    loan=fetch_loan_snapshot(table,target)
    try:
        ref,ref_int,metric=ora_dates(target)
    except MissingSnapshot as exc:
        log('SOURCE UNAVAILABLE',target,str(exc))
        loan['MAP_DTE']=target
        loan['LTV_STATUS']='SOURCE_SNAPSHOT_UNAVAILABLE'
        loan['LTV_POLICY_STATUS']='NO_VALID_LTV'
        return loan.reindex(columns=BASE_COLUMNS+NEW_COLUMNS)
    rat=get(CONN_ORA,'RETAIL-METHOD ALLOCATION '+target,
            retail_ltv_sql(ref,ref_int,metric),MAX_ORACLE_INSTRUMENTS)
    if rat.empty: raise RuntimeError('Retail-method query returned no bank instruments')
    rat['INSTRUMENT_ID']=rat['INSTRUMENT_ID'].map(norm)
    if rat['INSTRUMENT_ID'].isna().any() or rat['INSTRUMENT_ID'].duplicated().any():
        raise RuntimeError('Nonunique Oracle aggregate instrument key at '+target)
    for x in rat.columns:
        if x not in ('INSTRUMENT_ID',FIELD): rat[x]=pd.to_numeric(rat[x],errors='coerce')
    df=loan.merge(rat,on='INSTRUMENT_ID',how='left',validate='one_to_one')
    df['MAP_DTE']=target
    df['LTV_SNAPSHOT_DATE']=ref
    df['PROTECTION_METRICS_DATE']=metric
    df['LTV_RETAIL_METHOD']=np.nan
    valid_r = df['LTV_RECONSTRUCTED'].gt(0) & np.isfinite(df['LTV_RECONSTRUCTED']) & df['N_UNCLASSIFIED_LINKS'].fillna(0).eq(0)
    safe=valid_r & df['LOAN_CURRENCY'].eq('EUR')
    df.loc[safe,'LTV_RETAIL_METHOD']=df.loc[safe,'LTV_RECONSTRUCTED']
    df['LTV_STATUS']=np.select([
        df['N_TOTAL_PROTECTION_LINKS'].isna(),
        df['N_UNCLASSIFIED_LINKS'].fillna(0).gt(0),
        df['N_REAL_ESTATE_LINKS'].fillna(0).eq(0),
        df['N_INCOMPLETE_REAL_ESTATE_LINKS'].fillna(0).gt(0),
        df['REGULATORY_EXPOSURE'].isna() | df['REGULATORY_EXPOSURE'].le(0),
        df['ALLOCATED_COLLATERAL_VALUE'].isna() | df['ALLOCATED_COLLATERAL_VALUE'].le(0),
        ~df['LOAN_CURRENCY'].eq('EUR'),
        ~valid_r,
    ],[
        'NO_PROTECTION_LINK','UNKNOWN_PROTECTION_TYPE','NO_ELIGIBLE_REAL_ESTATE','INCOMPLETE_COLLATERAL',
        'UNKNOWN_REGULATORY_EXPOSURE','NO_VALID_ALLOCATED_COLLATERAL',
        'FX_CURRENCY_REVIEW','LTV_REVIEW',
    ],default='CANDIDATE_RETAIL_METHOD')
    df['LTV_RETAIL_METHOD']=pd.to_numeric(df['LTV_RETAIL_METHOD'],errors='coerce')
    df=apply_policy(df)[BASE_COLUMNS+NEW_COLUMNS].copy()
    sub=df[df['LTV_STATUS'].eq('CANDIDATE_RETAIL_METHOD')]['LTV_RETAIL_METHOD']
    log('MONTH QA',target,'N_LOANS=%d'%len(df),'VALID_LTV=%d'%len(sub),
        'LTV_MEDIAN=%s'%(str(round(float(sub.median()),4)) if len(sub) else 'NULL'))
    return df


def canonical(frame):
    f=frame[BASE_COLUMNS].copy()
    dates=['MAP_DTE','LTV_SNAPSHOT_DATE','PROTECTION_METRICS_DATE']
    strings=['INSTRUMENT_ID','ENTITY_ID','LOAN_CURRENCY','LTV_STATUS']
    for c in dates:
        f[c]=pd.to_datetime(f[c],errors='raise',utc=True).dt.strftime('%Y-%m-%d')
    for c in strings:
        f[c]=f[c].map(lambda v: None if pd.isna(v) else str(v))
    for c in set(BASE_COLUMNS)-set(dates)-set(strings):
        f[c]=pd.to_numeric(f[c],errors='raise').astype(float)
    if f[KEYS].isna().any().any() or f.duplicated(KEYS).any():
        raise RuntimeError('Missing or duplicate monthly loan keys')
    return f.sort_values(KEYS).reset_index(drop=True)


def subtype_qa(result):
    valid=result.loc[result['LTV_STATUS'].eq('CANDIDATE_RETAIL_METHOD') & result['LTV_RETAIL_METHOD'].notna()].copy()
    valid['SUBTYPE_STATUS']='UNIQUE'
    valid.loc[valid['N_DISTINCT_PROTECTION_SUBTYPES'].gt(1),'SUBTYPE_STATUS']='MULTIPLE'
    valid.loc[valid[FIELD].isna() | valid['N_MISSING_PROTECTION_SUBTYPES'].gt(0),'SUBTYPE_STATUS']='MISSING_OR_INCOMPLETE'
    def counts(grouped):
        return grouped.agg(N_LOAN_MONTHS=('INSTRUMENT_ID','size'),N_INSTRUMENTS=('INSTRUMENT_ID','nunique'),N_ENTITIES=('ENTITY_ID','nunique'),MEDIAN_LTV=('LTV_RETAIL_METHOD','median')).reset_index()
    print('\n=== SUBTYPE COVERAGE ===\n'+counts(valid.groupby('SUBTYPE_STATUS')).to_csv(index=False,sep=';',na_rep='NULL'),flush=True)
    print('\n=== DISTINCT PROTECTION SUBTYPES ===\n'+counts(valid.groupby([FIELD,'SUBTYPE_STATUS'],dropna=False)).sort_values('N_LOAN_MONTHS',ascending=False).to_csv(index=False,sep=';',na_rep='NULL'),flush=True)
    latest=valid.loc[valid['MAP_DTE'].eq(valid['MAP_DTE'].max())]
    print('\n=== LATEST SNAPSHOT SUBTYPES ===\n'+counts(latest.groupby([FIELD,'SUBTYPE_STATUS'],dropna=False)).sort_values('N_LOAN_MONTHS',ascending=False).to_csv(index=False,sep=';',na_rep='NULL'),flush=True)


# Historical calculation only; no pilot write to the existing output.
outputs = {str(getattr(x, 'name', x)).split('.')[-1]
           for x in recipe.get_outputs(object_type='DATASET')}
if outputs != {OUTPUT}:
    raise RuntimeError('Expected sole output '+OUTPUT)
log('START','Corporate LTV','HISTORY')
LOANS=required_dataiku(SOURCE_LOANS,['MAP_DTE','ENTITY_ID','INSTRUMENT_ID','NOMINAL_AMOUNT','CURRENCY','CONTRACT_INTERNAL_PURPOSE_CODE'])
OUT=required_dataiku(OUTPUT,[])
if OUT==LOANS: raise RuntimeError('Output must not share physical table with input')
# Check the requested source field before historical calculations.
get(CONN_ORA,'PROTECTION SUBTYPE SCHEMA','SELECT '+FIELD+' FROM '+ORACLE_PROTECTION+' WHERE 1=0',1)

q="SELECT TO_CHAR(MAX(TO_DATE(MAP_DTE)),'YYYY-MM-DD') AS SNAPSHOT " \
  "FROM %s WHERE TO_DATE(MAP_DTE) BETWEEN DATE '%s' AND DATE '%s' " \
  "GROUP BY DATE_TRUNC('MONTH',TO_DATE(MAP_DTE)) ORDER BY SNAPSHOT"%(LOANS,TRAIN_START,TRAIN_END)
dates=get(CONN_SNOW,'TRAINING MONTH-END SNAPSHOTS',q,40)
months=dates['SNAPSHOT'].astype(str).tolist()
if len(months)<20: raise RuntimeError('Too few training months: '+str(len(months)))

# Reconcile population and historical availability before writing.
before = canonical(get(CONN_SNOW,'EXISTING LTV HISTORY',
    'SELECT '+','.join(BASE_COLUMNS)+' FROM '+OUT,MAX_LOANS_PER_MONTH*len(months)))
if before.empty:
    raise RuntimeError('Existing monthly history unavailable')

all_frames=[]
for idx,m in enumerate(months,1):
    log('PROCESS','%d/%d'%(idx,len(months)),m)
    all_frames.append(build_month(LOANS,m))
result=pd.concat(all_frames,ignore_index=True)
if result[KEYS].isna().any().any() or result.duplicated(KEYS).any():
    raise RuntimeError('Missing or duplicate monthly loan keys')
if result['LTV_RETAIL_METHOD'].notna().sum()==0:
    raise RuntimeError('No eligible LTV; output not written')
current=canonical(result)
pd.testing.assert_frame_equal(before[KEYS+['ENTITY_ID']],current[KEYS+['ENTITY_ID']],check_dtype=False)
previous_months=set(before.loc[before['LTV_RETAIL_METHOD'].notna(),'MAP_DTE'])
current_months=set(current.loc[current['LTV_RETAIL_METHOD'].notna(),'MAP_DTE'])
if not previous_months.issubset(current_months):
    raise RuntimeError('Previously available LTV months lost: '+str(sorted(previous_months-current_months)))
comparison=before[KEYS+['LTV_RETAIL_METHOD']].merge(
    current[KEYS+['LTV_RETAIL_METHOD']],on=KEYS,validate='one_to_one',suffixes=('_OLD','_NEW'))
old= comparison['LTV_RETAIL_METHOD_OLD']
new= comparison['LTV_RETAIL_METHOD_NEW']
print('LTV RECONCILIATION | previous_valid=%d | current_valid=%d | gained=%d | lost=%d' % (
      old.notna().sum(),new.notna().sum(),(old.isna() & new.notna()).sum(),(old.notna() & new.isna()).sum()))
print('MONTHLY COVERAGE')
print(result.groupby('MAP_DTE').agg(N_LOANS=('INSTRUMENT_ID','size'),
      N_VALID_LTV=('LTV_RETAIL_METHOD','count'),N_MAPPED=('LTV_POLICY_THRESHOLD','count')).to_string())
print('POLICY MAPPING COVERAGE')
print(result.groupby(['LTV_POLICY_STATUS','LTV_POLICY_CATEGORY','LTV_POLICY_THRESHOLD'],dropna=False)
      .agg(N_LOAN_MONTHS=('INSTRUMENT_ID','size'),N_ABOVE_POLICY=('LTV_POLICY_BREACH',lambda x:x.sum(min_count=1))).to_string())
if result['LTV_POLICY_THRESHOLD'].notna().sum()==0:
    raise RuntimeError('No mapped ADC policy observations; output not written')

subtype_qa(result)
# Guarantee a string schema even if the new field happens to be entirely null.
result[FIELD]=result[FIELD].map(lambda v: None if pd.isna(v) else str(v)).astype(object)
for c in ['LTV_SNAPSHOT_DATE','PROTECTION_METRICS_DATE','CONTRACT_INTERNAL_PURPOSE_CODE','LTV_POLICY_CATEGORY','LTV_POLICY_STATUS']:
    result[c]=result[c].map(lambda v: None if pd.isna(v) else str(v)).astype(object)
log('WRITE START',OUTPUT,'rows=%d'%len(result))
dataiku.Dataset(OUTPUT,project_key=PROJECT).write_with_schema(result)
log('FINISHED',OUTPUT,'rows=%d'%len(result),'valid_candidates=%d'%result['LTV_RETAIL_METHOD'].notna().sum())
log('FINISHED','Protection types 8, 9 and 10; ADC candidate policy mapping; EUR only')
