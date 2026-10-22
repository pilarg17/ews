# SEGMENT y escenario DEV MAIN — 22/10/2026

## Columna SEGMENT

- **DEV MAIN:** la recipe `EWS_Output_Datasets_NO_CAMBIAR.py` (copia de referencia del MAIN) crea `SEGMENT` en `ews_entity_static_data` a partir de `BUSINESS_LINE` de las filas de breach. Digital Banking y Wealth Management se convierten en `Retail`; CIB en `Corporate` (aunque el proceso de Retail excluye CIB de breaches). No modificar las recipes de `Entity_Information` para añadir `SEGMENT`.
- **PROD si el sistema receptor NO acepta aún SEGMENT:** usar **solo para ese despliegue**, y tras pruebas, `EWS_Output_Datasets_COMPAT_PROD_SIN_SEGMENT_SOLO.py`. Este fichero difiere de la copia de MAIN únicamente en los cuatro bloques dedicados a `SEGMENT`; **mantiene la regla original de exclusión de Default**, que NO elimina indiscriminadamente todo cliente Default. No pegarlo en DEV MAIN donde SEGMENT debe permanecer.
- La versión antigua `Build_EWS_Output_Datasets_NO_SEGMENT(1).py` NO es equivalente: además cambia la exclusión de Default y por tanto NO se recomienda usarla para quitar solo SEGMENT.
- Log del backfill 19/10: confirmó 11 breaches y 11 registros de entity static, PERO no contiene la línea `Missing SEGMENT:`, por lo que **no demuestra el contenido de SEGMENT**. Validar columna y no nulos en `ews_entity_static_data` de DEV MAIN antes de dar SEGMENT por cerrado.

## Escenario existente: Step 1 Current + Previous + RM

- Mantener el escenario existente y las tres ramas tal como están. No añadir otro Step 1.
- Sustituir/validar la recipe optimizada en el Flow de cada rama; **la ejecución de Step 1 utilizará ese código** solo si su acción está configurada para reconstruir sus datasets y upstreams relevantes. Revisar en la configuración del escenario las opciones de `Build` y de dependencias, en vez de asumirlo.
- `Current`: `Credit_Cards`, `CC_Utilisation_History`, `Entity_Information`, `Rating_GCC_History`, `vars_behavioural_daily`, `Available_Indicators_Current` y otras dependencias ya existentes.
- `Previous`: contrapartes con sufijo Previous/`_previous`, usando su fecha propia y la misma ventana lógica.
- `RM`: `RM_Data_Daily_Run`; la recipe `RM_Data_Daily_Run_Previous` está incluida como candidata pero solo debe instalarse/ejecutarse si el Flow realmente la utiliza.
- Tras Step 1, comprobar que los pasos existentes reconstruyen `EWS_Daily_Alerts` y `EWS_Output_Datasets`. **No asumir que Step 1 los genera**: depende de los targets configurados. No activar export SFTP/PROD durante la medición.
- Evitar doble construcción simultánea de una misma salida si los 3 `Build` tienen dependencias compartidas con `rebuild upstream`.
- Para analizar runtimes, comparar escenario completo anterior/después, actividades por recipe, SQL de las ramas, tiempo de Step 1 y ruta crítica, sin sumar duraciones concurrentes.

## Estado de validación

- Rating + CC bundling probado: 23 indicadores -> 11 breaches finales en la referencia del 19/10.
- SEGMENT en el código MAIN: confirmado. SEGMENT no nulo en la salida real de MAIN: **pendiente de comprobar**.
- Optimizaciones SQL y límites TEK/EUR 1: candidatas que requieren equivalencia funcional con Oracle; no se han probado remotamente aquí. No mover a PROD sin pruebas.
