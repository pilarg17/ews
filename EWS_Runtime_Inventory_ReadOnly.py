import dataiku

DATASETS = [
    'Credit_Cards', 'Credit_Cards_Previous',
    'CC_Utilisation_History', 'CC_Utilisation_History_Previous',
    'vars_behavioural_daily', 'CARD_USAGE_VW',
    'CARD_REFERENTIAL_VW', 'CUSTOMER_VW',
]
KEYWORDS = ('CARD', 'BEHAVIOUR', 'CASH', 'UTILIS', 'REFERENT', 'FEATURE')

print('=' * 90)
print('RETAIL EWS - RUNTIME INVENTORY (METADATA ONLY)')
print('=' * 90)
for name in DATASETS:
    print('\nDATASET:', name)
    try:
        ds = dataiku.Dataset(name)
        location = ds.get_location_info()
        info = location.get('info', {})
        print('STORAGE:', location.get('locationInfoType'))
        print('CONNECTION:', info.get('connectionName', 'not provided'))
        print('TABLE:', info.get('quotedResolvedTableName') or info.get('table') or 'not provided')
        cols = ds.read_schema()
        print('COLUMNS ({}):'.format(len(cols)))
        for col in cols:
            print('  {} [{}]'.format(col.get('name'), col.get('type')))
    except Exception as e:
        print('METADATA ERROR:', type(e).__name__, str(e)[:500])

print('\n' + '=' * 90)
print('RELEVANT RECIPES - NAMES ONLY')
print('=' * 90)
try:
    project = dataiku.api_client().get_default_project()
    recipes = project.list_recipes()
    found = 0
    for recipe in recipes:
        name = recipe.get('name', '')
        if any(token in name.upper() for token in KEYWORDS):
            print('{} | type={}'.format(name, recipe.get('type', 'unknown')))
            found += 1
    print('Matching recipe names:', found)
except Exception as e:
    print('RECIPE LIST ERROR:', type(e).__name__, str(e)[:500])

print('\nMETADATA ONLY: no row reads, no writes, no recipe execution.')
