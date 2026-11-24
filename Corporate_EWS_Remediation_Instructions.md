# Corporate EWS — reconciliación y correcciones propuestas

## Principio de ejecución

**No reconstruir el training, el staging de pagos, los indicadores B01/B02 ni los backtests existentes para investigar una diferencia de `ACCURACY`.** Primero hacer QA **solo lectura**. Conservar los outputs vigentes como baseline.

## Paso 1 — Diferencia de ACCURACY (bloquea la entrega)

Ejecutar `Corporate_EWS_Accuracy_Reconciliation_ReadOnly.py` **en una celda de un notebook Dataiku con acceso al proyecto**, no como recipe con output. No escribe datasets. Verificar:

- `METRIC_FORMULA`: `ACCURACY = N_TP / (N_TP + N_FP)` cuando el denominador es positivo; si no hay alertas, resultado nulo. `ALERT_PRECISION = N_TP_ALERTS / N_EVALUATED_EPISODES` cuando existe esa métrica. Ambas son medidas distintas. OJO: en varios backtests `N_TP` cuenta **eventos detectados**, mientras `N_FP` cuenta **episodios de falsas alertas**. Por tanto, `ACCURACY` es el indicador histórico con denominador mixto, **no debe presentarse sin matiz como precisión estadística estándar**. La métrica de precisión por episodio es `ALERT_PRECISION`.
- `LINEAGE_DIFF_COUNTS`: compara por cartera, indicador, target original y umbrales cada fila consolidada con su **backtest fuente**. `CHANGED_TP_FP`, `CHANGED_ACCURACY`, `AMBIGUOUS_SOURCE_KEY` o `NO_SOURCE_MATCH` deben investigarse; el indicador no puede darse por reconciliado.
- `TARGET_LABEL_CHANGES_IN_CONSOLIDATION`: la consolidación normaliza algunas etiquetas (`RATING` → `FULL_EWS`, `DPD_EARLY` → `FULL_EWS`, etc.), pero conserva `TARGET_SCOPE_ORIGINAL`. **La etiqueta normalizada no prueba equivalencia de los eventos objetivo.**
- Para cerrar la discrepancia exacta del correo, se necesitan ambos archivos históricos: **Excel/CSV anterior enviado** y **Excel/CSV nuevo**. Exportar a CSV la misma hoja/granularidad de ambos Excel y ejecutar localmente:

  `python Corporate_EWS_Compare_Accuracy_Exports.py anterior.csv nuevo.csv reporte.csv`

  El informe compara N_EVENTS, N_TP, N_FP, ACCURACY, target, thresholds y otras métricas fila por fila. Conservar el CSV resultante.

Interpretación:

| Resultado QA | Conclusión y acción |
|---|---|
| Fuente y consolidado coinciden; anterior y nuevo coinciden | No hay cambio numérico; revisar filtros, pestaña o definición de columna visualizada |
| Fuente y consolidado coinciden; anterior y nuevo difieren en TP/FP | El **backtest almacenado cambió**, o se están comparando distinto target/población/configuración; no cambiar la fórmula para ocultarlo |
| TP/FP iguales, ACCURACY diferente por encima del redondeo | Error en cálculo/exportación de la métrica: corregir solo la proyección afectada tras aislar la fila |
| Consolidado y fuente no coinciden en TP/FP/ACCURACY | Error de lineage/selección/versión de datos; no publicar el consolidado hasta reconciliar |
| `ACCURACY` coincide con el antiguo `ALERT_PRECISION` | Probable comparación entre métricas distintas; confirmar la columna exacta |

**No afirmar que se ha resuelto la discrepancia** hasta tener el fichero anterior y el resultado de esta prueba.

## Paso 2 — Limit Utilisation: term loans vs revolving

Ejecutar `Corporate_EWS_Limit_Product_Scope_ReadOnly.py` en notebook. Lee un snapshot de `vars_loan_level` y muestra campos/códigos de clasificación del producto sin decidir por inferencia qué es revolving.

- El cálculo histórico actual de Limit utiliza préstamos del dataset general sin un filtro revolving explícito. No rotular los **216 resultados existentes** como revolving-only.
- Solicitar/conseguir el mapeo de negocio de códigos de producto revolving vs term loans (y si la utilización debe medirse a nivel de facilidad o préstamo). Controlar límites compartidos entre entidades, moneda, importe nominal, denominador y fecha.
- **Solo tras validar ese mapeo**, construir una cadena **técnica separada** de utilización revolving y evaluar su backtest, sin sustituir los resultados actuales. No introducir una lista de códigos supuesta.
- La media ponderada disponible para Limit es un *proxy mensual*, no una verdadera media diaria ponderada por tiempo; mantenerlo etiquetado como provisional.

## Paso 3 — LTV más granular

El código revisado `compute_results_Corporate_LTV.py` actualmente prueba LTV 80%, 100% y 120%, solo o combinado con downgrades de 2 y 3 notches en 3 meses, y **detecta el cruce** del umbral entre snapshots consecutivos elegibles. `THRESHOLD_B=0` en `ltv_high_loan` significa **sin condición de rating**. Un umbral 100% está almacenado como `1.0`, no `100`.

La utilidad `Corporate_EWS_LTV_Granular_Patch.py` **no ejecuta Dataiku ni sobrescribe la recipe existente**. Con una copia local del código original:

`python Corporate_EWS_LTV_Granular_Patch.py compute_results_Corporate_LTV.py`

Genera un nuevo archivo de código `compute_results_Corporate_LTV_granular.py` para un output técnico distinto `results_Corporate_LTV_granular`. Prueba:
- LTV-only: **85, 90, 95, 100, 105 y 110%** (6 alternativas).
- LTV + rating downgrade a **2, 3 y 4 notches**, para cada umbral (18 alternativas). Los 4 notches solicitados quedan explícitos; 2/3 se retienen como comparadores.
- Total teórico: **24 configuraciones por cartera**, 144 en seis carteras, sujeto a cobertura de datos, validez del target y reglas de backtest.

**No ejecutarlo aún** si la reconstrucción de las garantías sigue etiquetada `SOURCE_QA_ONLY` o si hay fechas/FX/garantías compartidas sin validar. El cálculo actual exige fuentes EUR y continuidad entre snapshots, lo que limita la cobertura de LTV. La variante con slotting/score para **ADC/IPRE** requiere el score ACA previo a slotting, mapping préstamo-entidad-fecha y definición comparable de deterioro; **no se puede sustituir una diferencia de slots por cuatro notches de rating**.

## Paso 4 — Shortlist de Payments

El script de reconciliación informa si las configuraciones solicitadas existen en las 666 filas ya almacenadas:

- `payment_gross_avg3_vs_avg12_decline` 30% → ADC y Large Corporate.
- `payment_incoming_avg3_vs_avg12_decline` 20% → ADC, IP (IPRE) y Large Corporate.
- `payment_top10_floor_prev12_max` 20% → IP (IPRE).
- `payment_net_max3_vs_avg12_decline` 30% → ADC.

No sumar alertas de las 111 configuraciones alternativas. `payment_incoming_avg3_vs_avg12_increase` compara media mensual últimos 3 meses con media últimos 12 meses **incluyendo los últimos tres**. La alternativa de `payment_top10` con **media previa de tres meses y 30%** necesita validación de su definición y un backtest separado si no existe en la grid actual.

## Paso 5 — Rating, slotting y GEC

- `gec_member_rating_downgrade_best_12m` y la metodología Head requieren la correspondencia **miembro ↔ GEC Head ENTITY_ID ↔ fecha** correcta antes de evaluar contagio. No reportar métricas fabricadas.
- *Systematic downgrades*, según el criterio de negocio comunicado: `N_REVIEWS_WITH_DOWNGRADE_6M / N_RATING_REVIEWS_6M > 0.60`. Contar **revisiones reales**, no días/snapshots con un mismo rating. Revisión sin downgrade cuenta en denominador; ausencia de revisiones → NULL/no evaluable. No inventar un mínimo de revisiones ni cambiar la ventana si no se aprueba.
- ADC e IPRE: evaluar slotting/score solo tras validar mapping y evolución temporal de score ACA; los scores no son automáticamente equivalentes a notches de rating.

## Control previo a enviar al cliente

1. QA de accuracy sin errores y reconciliación cerrada contra el Excel anterior.
2. Resultados por las seis carteras; distinguir GLOBAL del desglose sin doble conteo.
3. Definición de `ACCURACY` (*legacy metric: detected events / detected events + FP episodes*) y `ALERT_PRECISION` (*episode precision*) visible, o en vista de negocio mostrar `LEGACY_ACCURACY` y, si se conserva, `EPISODE_PRECISION` hasta que negocio confirme qué métrica denominar `PRECISION`. **No cambiar las series históricas silenciosamente**.
4. Fuentes LTV y Limit claramente separadas: `EVALUATED` vs `SOURCE_QA_ONLY`/`PROVISIONAL`.
5. No modificar ni borrar los outputs antiguos de training, B01/B02, Payments, Recurring ni el Excel de resultados hasta superar la reconciliación.

**Estado del paquete:** utilidades de diagnóstico y parche de LTV generadas y compiladas localmente; las consultas SQL requieren ejecutarse en el proyecto Dataiku real. No hay conexión desde este entorno a la base bancaria ni al Excel anterior mencionado en el correo, por lo que ningún backtest nuevo ni corrección numérica de accuracy se ha certificado.
