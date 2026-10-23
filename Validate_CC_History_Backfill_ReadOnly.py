import dataiku
import pandas as pd
import numpy as np
import hashlib
import json
import time

print('=' * 100)
print('CREDIT CARD HISTORY - BACKFILL PRE-MAIN VALIDATION (READ ONLY)')
print('=' * 100)

project_key = dataiku.api_client().get_default_project().project_key
expected_project = 'RETAIL_EWS_BACKFILL_20261020'
print('Project:', project_key)
if project_key != expected_project:
    raise RuntimeError('STOP: run this validation in ' + expected_project)

pairs = [
    ('CURRENT', 'Implementation_Reference_Date', 'CC_Utilisation_History', 'Credit_Cards'),
    ('PREVIOUS', 'Implementation_Reference_Date_previous', 'CC_Utilisation_History_Previous', 'Credit_Cards_Previous'),
]

expected_columns = [
    'ENTITY_ID','MAP_DTE','cc_outstanding_amount','cc_credit_limit',
    'cc_limit_utilization','cc_diff_limit_util','cc_average_utilization_3m',
    'cc_diff_limit_util_3m','cc_average_utilization_6m',
    'cc_diff_limit_util_6m','cc_average_utilization_12m',
    'cc_diff_limit_util_12m','cc_max_utilization_3m',
    'cc_diff_limit_util_3m_max','cc_max_utilization_6m',
    'cc_diff_limit_util_6m_max','cc_max_utilization_12m',
    'cc_diff_limit_util_12m_max','CC_PREVIOUS_BUSINESS_DAY_UTILIZATION',
    'CC_UTILIZATION_DAILY_CHANGE','CC_UTILIZATION_DAILY_WORSENING'
]

metric_cols = [
    'cc_diff_limit_util_12m', 'cc_max_utilization_3m',
    'cc_diff_limit_util_3m_max', 'CC_PREVIOUS_BUSINESS_DAY_UTILIZATION'
]

summary = []
failures = []

def normalise_value(value):
    if pd.isna(value):
        return '<NULL>'
    try:
        return format(float(value), '.12g')
    except (TypeError, ValueError):
        return str(value)

for label, reference_name, output_name, cards_name in pairs:
    print('\n' + '-' * 100)
    print(label, '|', output_name)
    start = time.perf_counter()
    reference = dataiku.Dataset(reference_name).get_dataframe()
    if reference.empty or 'REFERENCE_DATE' not in reference:
        failures.append(label + ': invalid reference dataset')
        continue
    ref_date = pd.to_datetime(reference['REFERENCE_DATE'].iloc[0], errors='coerce', utc=True)
    if pd.isna(ref_date):
        failures.append(label + ': invalid REFERENCE_DATE')
        continue
    ref_date = ref_date.tz_localize(None).normalize()
    print('Reference date:', ref_date.date())

    output_ds = dataiku.Dataset(output_name)
    schema = [c['name'] for c in output_ds.read_schema()]
    missing = sorted(set(expected_columns) - set(schema))
    extra = sorted(set(schema) - set(expected_columns))
    print('Schema columns:', len(schema), '| missing:', missing, '| extra:', extra)
    if missing or extra or len(schema) != 21:
        failures.append(label + ': 21-column schema mismatch')
        continue

    df = output_ds.get_dataframe(columns=expected_columns)
    dt = pd.to_datetime(df['MAP_DTE'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    invalid_dates = int(dt.isna().sum())
    wrong_dates = int(dt.ne(ref_date).sum())
    entity = df['ENTITY_ID'].astype('string').str.strip()
    missing_ids = int(entity.isna().sum() + entity.eq('').sum())
    duplicate_keys = int(pd.DataFrame({'ENTITY_ID': entity, 'MAP_DTE': dt}).duplicated().sum())
    print('Output rows:', len(df), '| entities:', entity.nunique())
    print('Invalid dates:', invalid_dates, '| non-reference dates:', wrong_dates)
    print('Missing ENTITY_ID:', missing_ids, '| duplicate entity/date:', duplicate_keys)
    if df.empty or invalid_dates or wrong_dates or missing_ids or duplicate_keys:
        failures.append(label + ': output grain/date/ID issue')

    print('Metric coverage (non-null / numeric):')
    for col in metric_cols:
        numeric = pd.to_numeric(df[col], errors='coerce')
        print('  {}: {:,} / {:,}'.format(col, int(numeric.notna().sum()), len(df)))

    util = pd.to_numeric(df['cc_limit_utilization'], errors='coerce')
    limit = pd.to_numeric(df['cc_credit_limit'], errors='coerce')
    amount = pd.to_numeric(df['cc_outstanding_amount'], errors='coerce')
    valid_ratio = limit.gt(0) & util.notna() & amount.notna()
    differences = (util.loc[valid_ratio] - amount.loc[valid_ratio] / limit.loc[valid_ratio]).abs()
    ratio_mismatches = int((differences > 1e-8).sum())
    print('Current utilisation ratio inconsistencies:', ratio_mismatches)
    if ratio_mismatches:
        failures.append(label + ': utilisation does not match amount / limit')

    cards_ds = dataiku.Dataset(cards_name)
    card_columns = [c['name'] for c in cards_ds.read_schema()]
    card_dates = None
    if 'MAP_DTE' in card_columns and 'ENTITY_ID' in card_columns:
        cards = cards_ds.get_dataframe(columns=['ENTITY_ID', 'MAP_DTE'])
        card_dates = pd.to_datetime(cards['MAP_DTE'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
        card_entities = cards.loc[card_dates.eq(ref_date), 'ENTITY_ID'].astype('string').str.strip()
        absent_output = int((~entity.dropna().drop_duplicates().isin(card_entities.dropna().unique())).sum())
        print('Credit_Cards rows on reference date:', int(card_dates.eq(ref_date).sum()))
        print('Output entities absent from daily Credit_Cards:', absent_output)
        if absent_output:
            print('CHECK: investigate absent output entities (could be legitimate depending on inputs).')
    else:
        print('Credit_Cards schema not suitable for coverage check; skipped.')

    hash_frame = df[['ENTITY_ID', 'MAP_DTE'] + metric_cols].copy()
    hash_frame['ENTITY_ID'] = entity
    hash_frame['MAP_DTE'] = dt
    hash_frame = hash_frame.sort_values(['ENTITY_ID', 'MAP_DTE'], kind='mergesort')
    digest = hashlib.sha256()
    for rec in hash_frame.itertuples(index=False, name=None):
        line = json.dumps([normalise_value(v) for v in rec])
        digest.update((line + '\n').encode('utf-8'))
    print('Four-metric SHA256:', digest.hexdigest())
    print('Validation seconds:', round(time.perf_counter() - start, 1))
    summary.append({'SIDE': label, 'DATE': str(ref_date.date()), 'OUTPUT_ROWS': len(df),
                    'ENTITIES': entity.nunique(), 'HASH': digest.hexdigest(),
                    'RATIO_ERRORS': ratio_mismatches})

print('\n' + '=' * 100)
print('SUMMARY')
for item in summary:
    print(item)
if failures:
    print('CHECKS FAILED:', *failures, sep='\n - ')
else:
    print('STRUCTURAL / INTERNAL CHECKS: PASS')
print('\nIMPORTANT: This does NOT prove equivalence to the old recipe on identical source snapshots.')
print('Earlier baseline hashes may have been obtained in a different project or run.')
print('Still required before Main: confirm the original-vs-optimized version differs only in the final output filter,')
print('check downstream/external dependencies, and verify Backfill and Main write destinations are isolated.')
print('NO DATASETS WRITTEN / NO RECIPES EXECUTED / NO BUNDLES CREATED')
