# DEV MAIN — Paquete integral de recetas | 22-10-2026

## Alcance

Cubre las 12 recipes optimizadas de Current/Previous (CC, Entity, Rating GCC, RM, Daily Behavioural, Available Indicators); 2 recipes de tarjetas actualizadas al límite vigente según TEK; y el bundling Rating+CC confirmado en el backfill. Mantiene sin cambios la recipe EWS_Output_Datasets que ya validó el output del 19/10 y la columna SEGMENT.

Los nombres de fichero contienen el destino. Son códigos completos para SUSTITUIR en la correspondiente recipe **solo después de guardar el código original**. Los nombres de datasets de entrada y salida se mantienen. El código es un candidato para DEV MAIN, NO un cambio en Dataiku. No se ha ejecutado contra el Oracle del banco desde aquí.

## Qué incluye

| Recipes | Modificación |
|---|---|
| Credit_Cards_Current y Previous | límite de tarjeta vigente a fecha de cálculo, sin month-start/fallback; descarta límite efectivo exactamente 1 EUR en pruebas |
| CC_Utilisation_History_Current y Previous | query única de 12 fechas con fallback ante duplicidades, límite vigente para Previous Day y 12 snapshots, filtrado coherente de EUR 1 |
| Entity_Information_Current y Previous | fechas optimizadas SQL |
| Rating_GCC_History_Current y Previous | fechas optimizadas SQL |
| Daily_Behavioural_Features_Current y Previous | filtros y agregación SQL cuando es seguro; fallback pandas |
| RM_Data_Daily_Run_Current y Previous | condiciones fechas optimizadas SQL; conserva lookup por customer |
| Available_Indicators_Current y Previous | selección Oracle de historia de rating por lotes y filtros GEC, sin alterar reglas de EWI |
| EWS_Daily_Alerts | categoría única para ratings RRE/Other Retail y CC; probado en DEV 16/10 y backfill 19/10 |
| EWS_Output_Datasets | NO CAMBIAR: copia de referencia con SEGMENT y exclusiones originales |

## Precauciones funcionales

- TEK_STRT_DTE/TEK_END_DTE indica **vigencia técnica**, no hay constancia de que acredite una APROBACIÓN del límite. Si el banco tiene un indicador de aprobación aparte, debe integrarse tras identificarlo. No afirmar que «latest approved» está confirmado.
- La exclusión del límite efectivo exactamente 1 EUR es una aproximación al caso de tarjeta provisional comunicado por el banco; no se dispone del código de estado del plástico. Confirmar que EUR 1 identifica de forma inequívoca tarjetas provisionales (especialmente temporal limits).
- Los 12 meses: se mantienen los snapshots mensuales disponibles, incluidos ceros registrados. La petición de excluir meses con importe 0 NO está implementada: falta decisión metodológica y afecta a triggers.
- No se renombra el EWI_NAME de CC al texto genérico, porque no consta que la herramienta admita ese nuevo código/nombre. Se conserva el indicador priorizado de los códigos fuente.
- El ranking entre indicadores CC y los casos Check Head/Client Rating necesita confirmación funcional para generalizar a PROD; el bundling técnico ya pasa QA.
- No se cambian DPD, Stage 2, UTP, salary, Default, product type ni la política de continuidad de cartera.
- Las dos versiones del paquete UAT con/sin SEGMENT son una tarea de despliegue, no de estas recipes.

## Secuencia de implementación en MAIN — copia y pegado CON CONTROL

1. Guardar copia del código de las 15 recipes de trabajo antes de sustituirlas y exportar snapshots actuales a datasets temporales. No editar una recipe durante una ejecución.
2. Instalar primero Credit_Cards_Current/Previous y CC_Utilisation_History_Current/Previous como bloque funcional atómico. RECONSTRUIR ambos historiales y los indicadores después; no mezclar snapshot nuevo con viejo.
3. Instalar Entity_Information_Current/Previous, Rating_GCC_History_Current/Previous, Daily_Behavioural_Features_Current/Previous, RM_Data_Daily_Run_Current/Previous, Available_Indicators_Current/Previous; comprobar que los nombres de salida indicados coinciden exactamente con los del Flow.
4. EWS_Daily_Alerts ya quedó instalado y validado en DEV MAIN; dejarlo SI coincide con este fichero; si no, pegar esta versión conservando backup.
5. EWS_Output_Datasets no requiere cambio. El archivo _NO_CAMBIAR.py es solo copia de referencia.
6. Ejecutar scenario de prueba en MAIN con fechas consistentemente sincronizadas en ambas ramas y SIN SFTP/PROD, reconstruyendo dependencias una única vez. Mantener en MAIN los dos flujos y los datasets originales previos hasta completar comparaciones.
7. QA: fechas, cardinalidades, ENTITY_ID, valores y flags por indicador, medias 3m/12m, límites, NULL, Rating Day1/Day5, bundles, Default, SEGMENT, salida y runtimes por actividad. Los recuentos de tarjetas pueden cambiar deliberadamente por los nuevos límites y la exclusión EUR1: documentar las diferencias.
8. Tras ello, preparar evidencia y SQL timings/EXPLAIN PLAN con Jonas; no afirmar mejora antes de medir Oracle.

## Validación local efectuada

Sintaxis Python mediante `ast.parse` para todos los archivos del paquete. NO ejecutados contra Dataiku, Oracle, entorno de banco, ni probados para equivalencia de outputs. Los logs anteriores del bundling en 16/10 y 19/10 corresponden a la versión EWS_Daily_Alerts incluida; no validan las otras 14 recipes.

## NOTA de ingeniería

El paquete profundo hace muchas consultas SQL y depende de detalles de Oracle (tipos de keys, permisos, planificación). Cualquier SQL de origen distinto a los vistos/fechas exclusivas o con timestamp exige prueba de equivalencia. El uso de GROUP BY y `MAX` en la SQL histórica tiene fallback ante duplicados pero la selección original `keep=last` no garantizaba orden en Oracle. Cualquier discrepancia debe investigarse antes de UAT/PROD.
