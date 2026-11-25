import re
import dataiku
import pandas as pd
from dataiku.core.sql import SQLExecutor2

PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
project = dataiku.api_client().get_project(PROJECT)
known = {d['name'] for d in project.list_datasets()}


def show(label, fn):
    print('\n=== ' + label + ' ===', flush=True)
    try:
        value = fn()
        if isinstance(value, pd.DataFrame):
            print(value.to_string(index=False), flush=True)
        elif value is not None:
            print(value, flush=True)
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc)[:500], flush=True)


def schema(name):
    return [c['name'] for c in dataiku.Dataset(name, project_key=PROJECT).read_schema()]


def query(conn, sql):
    f = SQLExecutor2(connection=conn).query_to_df(sql)
    f.columns = f.columns.str.upper()
    return f


pattern = re.compile(r'slotting|speciali[sz]ed|aca|rating|instrument_type|limit_id|limit_amount|report_version|reference_date|residence_country|economic_activity|risk_country|nominal_known|nominal_amount|incoming|outgoing|inflow|outflow|payment|map_dte|tek_strt_dte|tek_end_dte|entity_id|instrument_id|contract_id', re.I)

names = {'EWS_training_Corporate', 'EWS_training_Corporate_GEC_Rating_DPD_Arrears_B02',
         'vars_entity_level', 'vars_entity_level_merged', 'vars_loan_level',
         'vars_loan_level_merged', 'corporate_Systematic_Rating_Downgrades',
         'limit_utilisation_monthly_Corporate', 'limit_utilisation_features_Corporate',
         'limit_utilisation_signals_Corporate', 'payment_movements_monthly_Corporate',
         'INSTRUMENT_MON_VW', 'LIMIT_MON_VW', 'GEC_links_B01_daily'}
names |= {n for n in known if re.search(r'slotting|speciali[sz]ed|aca|bullet', n, re.I)}
for name in sorted(names & known):
    show('SCHEMA | ' + name, lambda n=name: ', '.join(c for c in schema(n) if pattern.search(c)))

# Actual recipe code for existing producers. No mutation.
for item in project.list_recipes():
    name = item['name']
    low = name.lower()
    if (('limit' in low and 'results' not in low) or 'systematic' in low
        or name in {'compute_results_Corporate_GEC_Contagion', 'compute_results_Corporate_Payments'}):
        show('RECIPE | ' + name, lambda n=name: project.get_recipe(n).get_settings().get_code())

# Inspect exact Oracle field names instead of assuming a grade or report key.
show('ORACLE | INSTRUMENT, LIMIT AND SLOTTING FIELDS', lambda: query('PRD_BDWH', """
SELECT OWNER, TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM ALL_TAB_COLUMNS
WHERE (OWNER='BDL_CREDIT_INSIGHT_PLY' AND TABLE_NAME IN ('INSTRUMENT_DAY_VW','INSTRUMENT_MON_VW','INSTRUMENT_REGULATORY_METRICS_DAY_VW'))
   OR (OWNER='BDL_LOANS_PLY' AND TABLE_NAME IN ('LIMIT_DAY_VW','LIMIT_MON_VW'))
   OR (OWNER IN ('BDL_CREDIT_INSIGHT_PLY','BDWH_PLY','DTM_RFO_PLY')
       AND (COLUMN_NAME LIKE '%SLOTTING%' OR COLUMN_NAME LIKE '%ASSESSMENT_SCORE%'))
ORDER BY OWNER,TABLE_NAME,COLUMN_ID
""").loc[lambda d: d.COLUMN_NAME.str.contains(r'SLOTTING|ASSESSMENT_SCORE|INSTRUMENT_TYPE|LIMIT_ID|LIMIT_AMOUNT|REPORT_VERSION|SOURCE_VERSION|LAST_VERSION|REFERENCE_DATE|MAP_DTE|INTERNAL_RATING|RATING_METHOD|RATING_SCALE|INSTRUMENT_ID|CONTRACT_REFERENCE', regex=True)])

print('\nFINISHED | READ ONLY', flush=True)
