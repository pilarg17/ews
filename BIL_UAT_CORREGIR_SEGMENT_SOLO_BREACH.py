# EWS Phase 2 UAT 10k — retirar SEGMENT SOLO de entity static
# Ejecutar en el MISMO notebook/kernel de la subida anterior.
# Modifica únicamente el CSV de prueba en la subcarpeta aislada.
import csv
import io
import hashlib
import dataiku

FOLDER_ID_UAT = 'tFtFJ7YO'
SUBFOLDER_UAT = 'UAT_PHASE2_10K_TEST'
ENTITY_EXPORT = 'ews_entity_static_data'

assert '_UAT_REPAIR_CACHE' in globals(), 'Falta el cache del lote en memoria'
files = _UAT_REPAIR_CACHE['payloads']
matching = [fn for fn in files if fn.startswith(ENTITY_EXPORT + '_') and fn.endswith('.csv')]
assert len(matching) == 1, 'Debe existir un único CSV de entity static en el cache: {}'.format(matching)
filename = matching[0]
path = SUBFOLDER_UAT + '/' + filename
folder = dataiku.Folder(FOLDER_ID_UAT)

with folder.get_download_stream(path) as stream:
    original = stream.read()

reader = csv.DictReader(io.StringIO(original.decode('utf-8-sig'), newline=''))
old_columns = list(reader.fieldnames or [])
assert old_columns == [
    'ENTITY_ID','SEGMENT','ENTITY_NAME','CUSTOMER_IDs','INTERNAL_RATING','NACE2',
    'GCC_ID','GCC_NAME','TOTAL_DEBT','TOTAL_ASSETS','DAY','error'
], 'Cabecera inesperada: {}'.format(old_columns)
rows = list(reader)
assert len(rows) == 10000, 'Se esperaban 10000 entidades, hay {}'.format(len(rows))
new_columns = [col for col in old_columns if col != 'SEGMENT']

buf = io.StringIO(newline='')
writer = csv.DictWriter(buf, fieldnames=new_columns, extrasaction='ignore', lineterminator='\n')
writer.writeheader()
writer.writerows(rows)
new_content = buf.getvalue().encode('utf-8')
new_sha = hashlib.sha256(new_content).hexdigest()

# Escribe solo el entity static en la carpeta de PRUEBAS (nunca en raíz).
with folder.get_writer(path) as stream:
    for start in range(0, len(new_content), 1024 * 1024):
        stream.write(new_content[start:start + 1024 * 1024])

with folder.get_download_stream(path) as stream:
    uploaded = stream.read()
assert hashlib.sha256(uploaded).hexdigest() == new_sha, 'No coincide el CSV guardado'

# Actualizar cache en memoria, para no reponer accidentalmente la version con SEGMENT.
files[filename] = (new_content, new_sha, len(rows))
print('CORRECTO:', path)
print('ENTIDADES:', len(rows))
print('COLUMNAS ENTITY STATIC:', len(new_columns), '(sin SEGMENT)')
print('BREACH, HISTORIC y EXPOSURE: sin cambios')
print('El contrato UAT corregido queda en 23 / 15 / 11 / 14 columnas.')
