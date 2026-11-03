# UNA CELDA EN UN NOTEBOOK DE RETAIL DEV. SOLO LECTURA.
import io, re
import pandas as pd
import dataiku

FOLDER_ID = 'tFtFJ7YO'  # Carpeta de exports Retail DEV observada en logs previos
MAX_EXPORTS = 250
PREFIX = ['ews_ewi_breach_information', 'ews_ewi_historic_information',
          'ews_entity_static_data', 'ews_exposure_static_data']
regex = re.compile(r'^(ews_ewi_breach_information|ews_ewi_historic_information|ews_entity_static_data|ews_exposure_static_data)_(\d{8}_\d{6})\.csv$', re.I)
CIB_RE = r'^corporate\s*(?:(?:&|and)\s*)?institutional\s*banking$'
EXCLUDE_STAGE2 = {'downgrade from stage 1 to stage 2',
                  'connected client classified as stage 2'}

def read_csv(folder, path):
    with folder.get_download_stream(path) as stream:
        raw = stream.read()
    header = raw.splitlines()[0].decode('utf-8-sig', errors='replace')
    sep = max([',', ';', '\t', '|'], key=header.count)
    if header.count(sep) == 0:
        raise ValueError('Separador CSV no reconocido: ' + path)
    try:
        return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str,
                           keep_default_na=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str,
                           keep_default_na=False, encoding='cp1252')

def rows(df, field, value):
    if field not in df.columns: return df.iloc[:0].copy()
    return df[df[field].astype(str).str.strip().eq(str(value).strip())].copy()

folder = dataiku.Folder(FOLDER_ID)
paths = {}
for p in folder.list_paths_in_partition():
    m = regex.fullmatch(p.rsplit('/', 1)[-1])
    if m:
        parent = p.rsplit('/', 1)[0] if '/' in p else ''
        paths[(parent, m.group(2), m.group(1).lower())] = p
exports = sorted([(t, d, p) for (d, t, n), p in paths.items()
                  if n == PREFIX[0]], reverse=True)
print('Exports historicos Retail DEV encontrados:', len(exports))

candidatos, vistos, fallos = [], set(), []
for fecha, carpeta, f in exports[:MAX_EXPORTS]:
    try:
        df = read_csv(folder, f)
        if not {'BREACH_ID','ENTITY_ID','EWI_NAME','BUSINESS_LINE'}.issubset(df.columns):
            continue
        bl = (df['BUSINESS_LINE'].astype(str).str.strip()
              .str.replace(r'\s+', ' ', regex=True).str.casefold())
        ew = df['EWI_NAME'].astype(str).str.strip().str.casefold()
        cib = df[bl.str.fullmatch(CIB_RE, na=False) & ~ew.isin(EXCLUDE_STAGE2)]
        for r in cib.to_dict('records'):
            clave = (r['ENTITY_ID'], r['EWI_NAME'], r.get('ALERT_DATE', ''))
            if clave not in vistos:
                vistos.add(clave)
                candidatos.append((fecha, carpeta, f, r))
    except Exception as exc:
        fallos.append((f.rsplit('/', 1)[-1], str(exc)[:100]))

print('Alertas CIB reales diferentes:', len(candidatos))
print('Entidades diferentes:', len({x[3]['ENTITY_ID'] for x in candidatos}))
if fallos: print('Lecturas fallidas:', len(fallos), '| ejemplo:', fallos[0])

# Preferir 2 entidades distintas; si no, aceptar 2 alertas distintas.
elegidos, entidades = [], set()
for item in candidatos:
    if item[3]['ENTITY_ID'] not in entidades:
        elegidos.append(item)
        entidades.add(item[3]['ENTITY_ID'])
    if len(elegidos) == 2: break
if len(elegidos) < 2:
    for item in candidatos:
        if item not in elegidos: elegidos.append(item)
        if len(elegidos) == 2: break

# Estas variables quedan en memoria del notebook; NO salen del banco.
CIB_REAL_CASES = []
cache = {}
for num, (fecha, carpeta, f, r) in enumerate(elegidos, 1):
    paquete, ausentes = {}, []
    for name in PREFIX:
        path = paths.get((carpeta, fecha, name))
        if path is None:
            paquete[name] = pd.DataFrame()
            ausentes.append(name)
        else:
            if path not in cache: cache[path] = read_csv(folder, path)
            paquete[name] = cache[path]
    b = rows(paquete[PREFIX[0]], 'BREACH_ID', r['BREACH_ID'])
    h = rows(paquete[PREFIX[1]], 'BREACH_ID', r['BREACH_ID'])
    e = rows(paquete[PREFIX[2]], 'ENTITY_ID', r['ENTITY_ID'])
    x = rows(paquete[PREFIX[3]], 'ENTITY_ID', r['ENTITY_ID'])
    if len(e):
        if 'SEGMENT' not in e.columns: e.insert(1, 'SEGMENT', 'Corporate')
        else: e['SEGMENT'] = 'Corporate'
    CIB_REAL_CASES.append({'source': f, 'stamp': fecha,
                           'breach': b, 'historic': h, 'entity': e, 'exposure': x})
    print('CIB', num, '| export', fecha, '| EWI:', r['EWI_NAME'])
    print('   breach:', len(b), 'historico:', len(h),
          'entity:', len(e), 'exposiciones:', len(x))
    if ausentes: print('   Archivos que faltan:', ', '.join(ausentes))

if len(CIB_REAL_CASES) != 2:
    print('FALTAN CASOS: no inventar ni duplicar CIB; ampliar MAX_EXPORTS o buscar otro archivo Retail DEV.')
else:
    print('DOS ALERTAS CIB REALES LOCALIZADAS. Revisar si sus 4 referencias estan completas.')
print('SOLO LECTURA: no se ha escrito ningun dataset ni CSV.')
