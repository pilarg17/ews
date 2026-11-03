# BIL Corporate Validation - Training prerequisites | READ ONLY
# Run in a notebook of project VALIDATIONEWSCORPORATEPORTFOLIODAILY.
# Does NOT build recipes, write data, download entire datasets or expose customer rows.
import dataiku
import pandas as pd

EXPECTED_PROJECT = 'VALIDATIONEWSCORPORATEPORTFOLIODAILY'
actual_project = dataiku.default_project_key()
if actual_project != EXPECTED_PROJECT:
    raise RuntimeError('Open this notebook in the Corporate Validation project; current project = ' + str(actual_project))

GROUPS = {
    'BASE DIARIA (ya comprobada anteriormente)': [
        'vars_entity_level', 'vars_loan_level',
        'DPD_Arrears_Daily_Loan', 'DPD_Arrears_Daily_Entity_SQL_QA'
    ],
    'MERGES CRITICOS PARA TRAINING': [
        'vars_entity_level_merged', 'vars_loan_level_merged',
        'Haircut_Loan', 'loan_RRE_ltv', 'DPD_Arrears_Daily_Entity',
    ],
    'OTRAS ENTRADAS DIRECTAS DE TRAINING': [
        'Proxy_Customer_Entity_Level', 'Customer_Balances',
        'RFO_PD_loan', 'Credit_Cards', 'cash_advance_data',
        'Cash_Deposits', 'CC_Transactions_Neobanks',
        'Cash_Advance_CC_Limit', 'Neobank_Transfers_CC_Limit',
    ],
    'SOURCES PROXY': [
        '20260710_PROXY_SALARY_FINAL_OUTPUT',
        '20260710_PROXY_NUMBER_CHILDREN_FINAL_OUTPUT',
    ],
    'OUTPUTS DE TRAINING': [
        'EWS_master_scale_QA', 'EWS_training',
        'EWS_testing', 'EWS_validation', 'non_mortgage_training',
    ],
    'SALIDAS DE QA (NO SUSTITUIR ORIGINALES)': [
        'vars_loan_level_merged_QA', 'vars_entity_level_merged_QA',
        'Haircut_Loan_QA', 'loan_RRE_ltv_QA',
    ]
}

# Massive historic datasets: inspect schema only, never read even one record.
METADATA_ONLY = {
    'DPD_Arrears_Daily_Loan', 'DPD_Arrears_Daily_Entity_SQL_QA',
    'vars_loan_level', 'vars_entity_level',
}
rows = []
for group, names in GROUPS.items():
    for name in names:
        result = {'BLOQUE': group, 'DATASET': name, 'ESTADO': '', 'COLUMNAS': None, 'CLAVES': '', 'ERROR': ''}
        try:
            ds = dataiku.Dataset(name)
            cols = [x['name'] for x in ds.read_schema()]
            result['COLUMNAS'] = len(cols)
            result['CLAVES'] = ', '.join(x for x in ['ENTITY_ID','INSTRUMENT_ID','MAP_DTE'] if x in cols)
            if name in METADATA_ONLY:
                result['ESTADO'] = 'ESQUEMA (sin comprobar datos actuales)'
            else:
                # One row maximum; checks physical readability, never prints real values.
                keys = [x for x in ['ENTITY_ID','INSTRUMENT_ID','MAP_DTE'] if x in cols]
                one = ds.get_dataframe(columns=keys if keys else None, sampling='head', limit=1)
                result['ESTADO'] = 'OK: accesible, con datos' if len(one) else 'VACIO / sin filas'
        except Exception as e:
            result['ESTADO'] = 'ERROR / NO DISPONIBLE'
            result['ERROR'] = (str(e).splitlines()[0] or type(e).__name__)[:180]
        rows.append(result)

report = pd.DataFrame(rows)
for group in GROUPS:
    print('\n=== {} ==='.format(group))
    subset = report.loc[report['BLOQUE'] == group, ['DATASET','ESTADO','COLUMNAS','CLAVES','ERROR']]
    print(subset.to_string(index=False))

print('\nSOLO LECTURA: no se ha ejecutado ninguna recipe ni descargado ninguna tabla completa.')
print('El estado ESQUEMA no prueba que la tabla física exista o esté actualizada.')
print('Si un dataset es particionado y exige seleccionar NP, el error indicará la partición.')
