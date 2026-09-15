# Propuesta de implementación: altura trófica validada para WLNM_dir_neg

Estado: propuesta basada en el código actual. No se ha implementado el cambio de producción ni ejecutado el piloto. El objetivo es recuperar soluciones matemáticamente definidas y numéricamente verificables, con causas explícitas para los casos que sigan siendo inválidos. La corrección se aplicaría según propiedades de la matriz, sin excepciones por nombre de food web.

## 1. Definición y compatibilidad

La métrica del manuscrito procede de `PseudoNetworkXMeanTrophicLevel`, que el procesador de retención presenta como `PseudoMeanTrophicHeight`. Se mantendrían la orientación recurso→consumidor, la eliminación de autoenlaces, el mayor componente débilmente conectado, los basales a nivel 1 y la media de todos los nodos de ese componente. Se conservaría también el criterio actual de desempate cuando hay componentes máximos del mismo tamaño.

Se introducirían dos protocolos explícitos:

- `legacy_v1`: reproducción del cálculo actual, incluyendo sus controles históricos.
- `validated_v2`: comprobación de accesibilidad basal, eliminación del límite superior heurístico y validación numérica con diagnósticos.

Las llamadas existentes a `compute_foodweb_metrics(A)` seguirían usando `legacy_v1` durante la transición. El rerun de WLNM_dir_neg activaría `validated_v2` mediante configuración. Se aplicaría el mismo protocolo a matrices empíricas, de entrenamiento y reconstruidas. El cálculo separado llamado `MeanTrophicLevel` no se sustituiría ni se utilizaría como relleno de la métrica del manuscrito.

## 2. Archivos e interfaces propuestas

| Archivo | Cambio propuesto |
|---|---|
| `src/matlab/metrics/compute_foodweb_metrics.m` | Aceptar una estructura opcional de opciones tróficas; seleccionar el protocolo; trasladar los diagnósticos al resultado. Conservar la rutina actual como implementación legacy. |
| `src/matlab/metrics/compute_networkx_trophic_levels_v2.m` — nuevo | Implementar el cálculo validado de niveles y diagnósticos. Función accesible directamente para pruebas y recálculo desde matrices. |
| `src/matlab/wlnm_version_runners/wlnm_dir_neg/WLNM_dir_neg.m` | Recibir las opciones, pasarlas a las tres llamadas de métricas y guardar matrices reconstruidas dentro del bucle de umbrales. |
| `src/matlab/wlnm_version_runners/wlnm_dir_neg/run_wlnm_dir_neg.m` | Propagar opciones, ampliar valores por defecto y copia de diagnósticos para Empirical/Train/Pseudo, y exportar las columnas nuevas. |
| `src/matlab/Main.m` | Añadir selección explícita del protocolo y opciones del piloto; registrar sus valores efectivos en el manifiesto de ejecución. |
| `src/matlab/tests/test_networkx_trophic_levels_v2.m` — nuevo | Pruebas matemáticas, casos límite y compatibilidad. |
| `src/matlab/tests/test_wlnm_trophic_diagnostic_export.m` — nuevo | Comprobar exportación, prefijos, metadatos y separación de protocolos. |
| `docs/stats/compare_trophic_level_protocols.py` — nuevo | Resumir comparaciones por ejecución y retención v1/v2 desde CSV, sin entrenar modelos. |
| `docs/stats/apply_wlnm_tukey_retention.py` | Leer y propagar el protocolo desde el manifiesto; rechazar mezclas y conservar trazabilidad. Mantener el cálculo y los umbrales de Tukey. |

Interfaz propuesta, todavía no ejecutable:

```matlab
opts = struct('protocol', 'validated_v2', ...
              'highPrecision', 'auto', ...
              'compareLegacy', true);
metrics = compute_foodweb_metrics(A, opts);
```

Las opciones se validarían; un nombre de protocolo desconocido produciría un error explícito. El helper de métricas sería determinista y no consumiría números aleatorios.

## 3. Algoritmo propuesto

### Comprobación estructural

1. Identificar el componente máximo con la misma regla que v1 y conservar sus índices originales.
2. Identificar basales por grado de entrada cero dentro de ese componente. Un componente de un único nodo tiene nivel 1.
3. Recorrer los enlaces dirigidos desde todos los basales. Si no hay basales o quedan nodos no alcanzables, devolver la métrica como NaN y un motivo estructural específico.
4. En esos casos no se añadirían enlaces, no se eliminarían nodos adicionales y no se sustituiría el sistema por una pseudoinversa. La salida completa del componente quedaría inválida, evitando que una media parcial se presente como la media del componente original.

La accesibilidad basal es la condición estructural utilizada por [NetworkX](https://networkx.org/documentation/stable/_modules/networkx/algorithms/centrality/trophic.html). Un ciclo alimentado desde un basal puede tener niveles finitos; detectar simplemente ciclos no basta para rechazar una red.

### Cálculo y verificación numérica

Construir P como el promedio de recursos y resolver `(I-P)t = 1` mediante un solucionador lineal. Eliminar la condición `all(t <= max(20,m))`; registrar el máximo calculado y si habría superado el límite histórico.

Como configuración inicial para validar en las pruebas:

| Control | Propuesta inicial | Uso |
|---|---:|---|
| Nivel mínimo | `1 - 1e-8` | Tolerancia numérica, conservada del código actual |
| Umbral de condicionamiento | `rcond <= 1e-10` | Activa verificación adicional; deja de ser un rechazo definitivo automático |
| Residuo escalado | `<= 1e-12` | Control de consistencia del sistema |
| Acuerdo entre precisiones | `<= 1e-10` relativo, por nodo | Control adicional de estabilidad para casos dudosos |
| Precisión adicional | 50 y 100 dígitos | Comparación independiente del cálculo en doble precisión |

El residuo escalado se definiría como `norm(M*t-b,1) / (norm(M,1)*norm(t,1)+norm(b,1))`, con `M=I-P` y `b=ones(m,1)`. Se conservaría también el residuo sin escalar. Un residuo pequeño por sí solo no demostraría precisión suficiente si el sistema está mal condicionado. Las advertencias del solucionador, valores no finitos, incumplimientos del nivel mínimo, condicionamiento dudoso o residuo elevado activarían la comprobación adicional.

Para esa comprobación, se reconstruiría el sistema en alta precisión a partir de la adyacencia binaria y los grados enteros, antes de dividir. Convertir únicamente la matriz double ya redondeada a alta precisión no recuperaría la información perdida. Se compararían las soluciones de 50 y 100 dígitos mediante `max(abs(t50-t100)./max(1,abs(t100)))`, además de verificar sus residuos, niveles y representabilidad al exportar a double.

`vpa` requiere Symbolic Math Toolbox: se comprobarían disponibilidad y licencia antes del piloto. `highPrecision='auto'` permitiría probar el cálculo básico cuando no esté disponible, pero los casos que requieran esa comprobación quedarían como `numerical_unresolved`, con matrices guardadas. `highPrecision='required'` fallaría al inicio si falta la dependencia. El rerun definitivo fijaría una política y backend comunes, registrados en el manifiesto; no dependería silenciosamente de diferencias entre el Mac y Apocrita.

Los valores de tolerancia son una propuesta de ingeniería que se validaría con problemas de solución conocida. No se elegirían buscando incluir un número concreto de redes. Una solución alta y estable podría conservarse en el CSV y aun así ser excluida posteriormente por Tukey.

Referencias técnicas: [rcond de MATLAB](https://www.mathworks.com/help/matlab/ref/rcond.html), [precisión variable](https://www.mathworks.com/help/symbolic/sym.vpa.html) y [aumento de precisión](https://www.mathworks.com/help/symbolic/increase-precision-of-numeric-calculations.html).

## 4. Diagnósticos y esquema de resultados

Se conservaría `StatusCode=0` para resultados válidos y la interpretación histórica de los códigos existentes. Se añadirían códigos para ausencia de basales y falta de accesibilidad; los casos numéricos no resueltos conservarían una categoría de fallo numérico. La correspondencia completa de códigos y motivos se registraría en el manifiesto de v2.

Campos adicionales, con prefijos `Empirical`, `Train` y `Pseudo`:

- `NetworkXTrophicLevelProtocolVersion`.
- `NetworkXTrophicLevelNumBasalLargest` y `NetworkXTrophicLevelNumUnreachableFromBasal`.
- `NetworkXTrophicLevelReciprocalCondition`.
- `NetworkXTrophicLevelScaledResidual` y `NetworkXTrophicLevelResidualNorm`.
- `NetworkXTrophicLevelMinCandidate` y `NetworkXTrophicLevelMaxCandidate`.
- `NetworkXTrophicLevelAboveLegacyCap`.
- `NetworkXTrophicLevelPrecisionDigits` y `NetworkXTrophicLevelSolverCode`.
- `NetworkXTrophicLevelFailureReason`, como texto, con tratamiento específico en el exportador.

Los extremos de candidatos rechazados servirían solo como diagnóstico. No alimentarían la media trófica. Los campos numéricos se añadirían a `networkx_trophic_diagnostic_suffixes`; el motivo textual usaría una inicialización y copia separadas para evitar errores de tipos en tablas MATLAB. Los diagnósticos no se incluirían en la lista de métricas para calcular deltas estadísticos.

En el piloto, `compareLegacy=true` calcularía ambos protocolos sobre cada misma matriz ya generada. El CSV de comparación incluiría `Foodweb`, `TrainRatio`, `ExperimentID`, `Seed`, `Threshold`, valores v1/v2, estados y motivo de transición. Los resultados legacy irían en columnas de comparación y no se mezclarían con la columna principal que consume Tukey.

## 5. Exportación de matrices

La exportación actual mediante `save_confusion` y `auxiliaryExportExperimentID` está diseñada para una repetición seleccionada. No basta para diagnosticar las 50.

Se añadiría una opción independiente, `ecologicalSnapshotMode`, con valores `none`, `all` o `failed`. El piloto usaría `all`, incluso cuando `save_confusion=false`. La exportación se haría cuando `pseudo_full` todavía está disponible, dentro del bucle de umbrales.

Cada snapshot MAT contendría las matrices dispersas empírica, de entrenamiento y reconstruida, el orden/identidad de nodos, la configuración efectiva, identificadores de ejecución y el protocolo trófico. El nombre identificaría de forma inequívoca red, K, train ratio, semilla, repetición, umbral y protocolo. La escritura sería temporal seguida de renombrado, compatible con ejecuciones paralelas y con rechazo de colisiones. Se evitaría cambiar el estado del generador aleatorio al crear nombres o exportar.

Esto permitiría volver a calcular las métricas sin volver a entrenar. Para el barrido completo se mediría primero el espacio del piloto y se elegiría conscientemente la política de conservación. Guardar solo fallos abarata almacenamiento, pero impide volver a calcular todos los casos aceptados sin reproducirlos.

## 6. Pruebas y criterios de aceptación

1. Cadena basal→intermedio→superior: niveles `[1,2,3]`.
2. Ciclo con alimentación basal: solución finita coincidente con una referencia independiente.
3. Ciclo sin basales: fallo estructural explícito.
4. Red débilmente conectada con un basal y un ciclo no alcanzable desde él: fallo por accesibilidad.
5. Contraejemplo de seis nodos del audit: niveles `[1,22,26,26,26,26]`, aceptado por v2 y rechazado por el límite de v1.
6. Componentes desconectados, empate en tamaño, nodo aislado y autoenlaces: misma selección/preprocesamiento que v1.
7. Permutación de nodos con componente máximo único: niveles equivalentes tras invertir la permutación.
8. Caso de mal condicionamiento con referencia de precisión suficiente: fallback y estado final correctos. Caso sin backend de precisión adicional: motivo explícito, sin valor aparentemente válido.
9. Matrices ordinarias aceptadas por v1: acuerdo de medias y niveles a una tolerancia propuesta de `1e-10` relativa; diferencias mayores se investigan.
10. Exportación: 50 snapshots distintos por red al 60 %, prefijos y tipos correctos, protocolo presente y ningún archivo original sobrescrito.
11. Aislamiento: el cálculo de métricas conserva el estado RNG; las matrices y predicciones del piloto compartido son exactamente las mismas para v1 y v2. La configuración legacy reproduce el comportamiento previo.
12. Integración con retención: la misma regla 1,5×IQR y mínimo 25/50; rechazo de entradas que mezclen protocolos y exclusión por métrica de valores no válidos.

La aceptación no exigiría que las 290 redes terminen incluidas. Exigiría recuperar correctamente los casos verificables y explicar de forma reproducible los restantes.

## 7. Piloto y despliegue

Primero, pruebas unitarias y comparación local. Después, PlotB y Ythan al 60 %, 50 repeticiones cada una: 100 ejecuciones objetivo. Añadir tres redes de control, cinco repeticiones por red, para otras 15 ejecuciones de integración. Se mantendrían K=10, umbral 0,5, las semillas y el protocolo negativo original. El piloto produciría conteos de transiciones v1/v2, cambios de medias, retención y costes de cálculo/almacenamiento.

Directorio propuesto, independiente del histórico:

`src/matlab/data/result_wlnm_dir_neg_roleonly_train60_trophic_v2_pilot/`

El manifiesto guardaría commit/hash de código, MATLAB/toolboxes, política de precisión, tolerancias, selección de redes, IDs de experimento, semillas y hashes de matrices. Un piloto en el mismo proceso compararía ambos cálculos sobre la misma matriz; reproducir solo la semilla en sistemas distintos no bastaría para demostrar igualdad bit a bit de las predicciones históricas.

Cuando la validación pase, activar el mismo protocolo en todas las redes y proporciones que alimenten los resultados definitivos, desde un directorio nuevo en Apocrita. Volver a generar retención y contrastes a partir de esos resultados homogéneos. Un cambio adicional en el enmascaramiento del enlace focal requeriría su propia validación y quedaría identificado en la versión del experimento; el piloto trófico debe poder atribuir sus diferencias exclusivamente al cálculo ecológico.

## 8. Resultado esperado y límites

En PlotB, las 23 ejecuciones inválidas al 60 % estaban clasificadas como singulares/mal condicionadas; eliminar el límite superior no las recuperaría por sí solo. En Ythan, cinco rechazos de solución son candidatos a revisión de límites, pero los registros no identifican todavía el control exacto que falló. La mejora aporta capacidad de recuperación y diagnóstico, no una garantía de resultados finitos para todas las reconstrucciones.

Las matrices sin niveles definidos seguirían siendo inválidas bajo esta definición. Imputación, regularización, enlaces basales artificiales o cálculo sobre un subconjunto serían procedimientos científicos distintos y no formarían parte de esta corrección.
