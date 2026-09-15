# Auditoría de altura trófica de WLNM_dir_neg

Se verificaron los 290 CSV originales del escenario `result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_Apocrita_neg_const`: 130.500 ejecuciones, K=10, umbral 0,5 y nueve proporciones de entrenamiento. El cálculo independiente reprodujo la retención existente de los 2.610 grupos red/proporción. No se modificaron resultados, notebooks, manuscritos ni código de producción.

La métrica es `PseudoNetworkXMeanTrophicLevel`, presentada como `PseudoMeanTrophicHeight` en los archivos de retención. Todos los valores empíricos de referencia son finitos. La pérdida ocurre en las realizaciones reconstruidas y en su retención, antes del contraste pareado de Wilcoxon.

| Entrenamiento | Redes incluidas | Redes excluidas | Ejecuciones inválidas antes de Tukey |
|---|---:|---:|---:|
| 10 % | 275 | 15 | 2225 |
| 20 % | 286 | 4 | 1678 |
| 30 % | 288 | 2 | 1088 |
| 40 % | 288 | 2 | 782 |
| 50 % | 289 | 1 | 553 |
| 60 % | 288 | 2 | 495 |
| 70 % | 289 | 1 | 380 |
| 80 % | 290 | 0 | 329 |
| 90 % | 290 | 0 | 214 |

Hay ejecuciones inválidas en todos los porcentajes. Al 80 % y 90 %, todas las redes conservan suficientes ejecuciones para obtener un resumen válido. Esto no significa que sus 50 ejecuciones sean válidas.

## Las dos redes excluidas al 60 %

| Red | Ejecuciones válidas antes de Tukey | Inválidas | Atípicas descartadas | Retenidas | Mínimo |
|---|---:|---:|---:|---:|---:|
| Dutch Microfauna food web PlotB | 27 | 23 | 6 | 21 | 25 |
| Ythan Estuary | 21 | 29 | 2 | 19 | 25 |

Por tanto, ambas redes sí producen algunos resultados válidos. PlotB cae por debajo del mínimo después de filtrar atípicos; Ythan ya tiene menos de 25 valores finitos antes del filtro.

| Entrenamiento | PlotB retenidas/50 | Ythan retenidas/50 |
|---|---:|---:|
| 10 % | 41 | 28 |
| 20 % | 38 | 22 |
| 30 % | 37 | 16 |
| 40 % | 28 | 16 |
| 50 % | 28 | 22 |
| 60 % | 21 | 19 |
| 70 % | 32 | 18 |
| 80 % | 38 | 28 |
| 90 % | 47 | 36 |

PlotB queda excluida únicamente al 60 %. Ythan queda excluida del 20 % al 70 %, ambos incluidos. Los detalles del resto de redes están en `excluded_foodwebs.csv`.

## Causas registradas

En `src/matlab/metrics/compute_foodweb_metrics.m:317`, la función de nivel trófico forma el sistema (I−P)t=1 en el mayor componente débilmente conectado. Rechaza la solución si el recíproco de la condición es no finito o ≤1e−10 (estado 2), o si la solución incumple los controles de finitud o los límites 1−1e−8 y max(20,m), siendo m el número de nodos del componente (estado 3).

Al 60 %, las 23 ejecuciones inválidas de PlotB tienen estado 2. En Ythan, 24 tienen estado 2 y cinco estado 3. El estado 2 combina singularidad y mal condicionamiento; el CSV no guarda el valor de `rcond`. El estado 3 tampoco identifica cuál de sus controles falló. Por ello no es posible determinar a partir de estos resúmenes cuántos casos son recuperables numéricamente.

Una causa estructural posible es que haya nodos que no puedan alcanzarse desde ningún basal, por ejemplo un ciclo sin alimentación basal. Tener algún basal en otra parte de la red, o que el componente sea débilmente conectado, no garantiza esa accesibilidad dirigida. La definición de [NetworkX](https://networkx.org/documentation/stable/_modules/networkx/algorithms/centrality/trophic.html) exige que todos los nodos sean alcanzables desde basales. No todos los ciclos invalidan el nivel trófico: pueden tener solución finita si reciben alimentación basal.

## Sensibilidad a parámetros, sin alterar el análisis principal

| Multiplicador IQR | PlotB retenidas al 60 % | Ythan retenidas al 60 % | Ambas alcanzan 25 |
|---|---:|---:|---|
| 1,5 | 21 | 19 | No |
| 2 | 21 | 19 | No |
| 3 | 21 | 20 | No |
| 5 | 24 | 21 | No |
| 15 | 26 | 21 | No |
| Sin excluir atípicos | 27 | 21 | No |

PlotB necesitaría aproximadamente 11,0812×IQR para retener 25 valores con los cuartiles actuales. Es un cambio grande del filtro. Ythan no puede alcanzar 25 modificando únicamente Tukey, pues solo tiene 21 valores finitos.

Con 1,5×IQR, bajar la fracción mínima de retención del 50 % al 40 % admitiría PlotB (21≥20), pero no Ythan (19<20). Al 38 %, ambas serían admitidas (mínimo 19). Esto cambia el criterio de inclusión; no convierte ejecuciones inválidas en válidas. No se recomienda elegir el corte después de observar qué valor permite incluir esas dos redes. Los CSV de sensibilidad muestran el efecto de aplicar los mismos parámetros a todas las redes y proporciones, sin recalcular p-valores ni alterar los resultados principales.

Los controles `1e−10` y `max(20,m)` están fijados en la función de métricas, no expuestos como parámetros del análisis Wilcoxon. Hay una mejora de implementación que merece estudio: distinguir la inexistencia estructural de niveles, el condicionamiento numérico y un nivel alto pero finito. El límite superior actual es una decisión heurística adicional, no una condición matemática de existencia del nivel trófico.

Se construyó un contraejemplo aislado: una red de seis nodos con un basal y cinco consumidores conectados entre sí, a la que el basal alimenta mediante un consumidor. Todos son alcanzables desde el basal y el sistema tiene solución [1,22,26,26,26,26], con recíproco de condición 0,02. El límite actual de 20 la rechazaría. Esto demuestra la posibilidad de rechazar soluciones finitas; no demuestra que las cinco ejecuciones históricas de Ythan con estado 3 fallen por ese motivo, ni valida ecológicamente niveles tan altos. La matriz y el residuo se conservan en `upper_bound_counterexample.json`.

## Qué puede corregirse

1. Mantener el criterio principal actual implica usar las redes con resúmenes válidos para cada proporción: al 60 %, 288 pares para altura trófica. No se deben imputar ceros ni reutilizar pares inválidos para completar 290. Cambiar parámetros de Wilcoxon no resuelve la falta de una medida válida.
2. Una mejora legítima del cálculo requiere guardar la matriz reconstruida, su accesibilidad desde basales, `rcond`, el residuo del sistema y los extremos de los niveles calculados. Con ello se podrían recuperar soluciones rechazadas solo por un control numérico o por el límite heurístico, cuando esté justificado. Una matriz realmente singular no se arregla aflojando la tolerancia.
3. Las matrices reconstruidas históricas no están exportadas en este directorio: `confusion_matrix_csv` está vacío y los CSV de resultados contienen resúmenes por ejecución. Se necesitarían los artefactos de aquellas ejecuciones, si existen en otra ubicación, o una reproducción controlada para cuantificar el efecto de cambiar el cálculo.
4. Cambiar el umbral de clasificación, la división de entrenamiento o las restricciones sobre las reconstrucciones cambia las redes que se evalúan y requiere un nuevo análisis. Más repeticiones no garantizan cumplir un mínimo del 50 %, porque también aumenta el número mínimo exigido. La conectividad débil por sí sola tampoco garantiza alimentación basal para todos los nodos.

## Reproducción y archivos

Ejecutar desde cualquier directorio:

```bash
/Users/acw792/miniconda3/envs/Foodweb/bin/python /Users/acw792/Developer/qmul-phd-framework/docs/manuscript_review/2026-09-09/trophic_height_audit/audit_trophic_height.py
```

El script solo escribe los diagnósticos junto a sí mismo. `source_hashes.csv` registra los archivos consultados. `retention_all_ratios.csv`, `focus_foodwebs_all_ratios.csv`, `focus_status_all_ratios.csv` y `focus_train60_runs.csv` permiten revisar los recuentos. `sensitivity_all_ratios.csv` y `sensitivity_focus_train60.csv` documentan los escenarios de sensibilidad; no son nuevos resultados principales.
