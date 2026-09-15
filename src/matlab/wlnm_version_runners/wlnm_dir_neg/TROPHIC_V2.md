# Cálculo trófico validado para WLNM dirigido-negativo

`validated_v2` es el protocolo configurado para `WLNM_dir_neg` y
`WLNM_dir_neg_kfold`. Se aplica de forma general, sin condiciones por food web.
K-fold conserva `computeEcologicalMetrics=false`, por lo que registra el protocolo
pero no calcula métricas tróficas en esa ejecución. El piloto aislado continúa
centrado en PlotB y Ythan para completar el análisis pendiente.

## Comportamiento

- Conserva orientación recurso → consumidor, eliminación de autoenlaces,
  componente débil mayor y desempate por el primer componente, igual que v1.
- Exige que todos los nodos de ese componente sean accesibles desde algún basal.
  Un ciclo alimentado por un basal puede ser válido. Un ciclo sin alimentación
  basal no se convierte artificialmente en válido.
- Resuelve `(I - D^-1 A') t = 1`, con basales en nivel 1. Elimina únicamente
  el límite superior `max(20, numero_de_nodos)` del criterio de aceptación.
- Comprueba valores finitos, mínimo ≥ `1-1e-8` y residuo escalado ≤ `1e-12`.
  `rcond ≤ 1e-10`, advertencias del solver o una solución inválida requieren
  verificación adicional; un residuo pequeño por sí solo no basta en esos casos.
- La verificación adicional reconstruye coeficientes racionales exactos desde
  la adyacencia binaria y los grados, resuelve con 50 y 100 dígitos y exige
  concordancia relativa por nodo ≤ `1e-10`, además de los controles del residuo
  y de la conversión a `double`. No usa pseudoinversa ni regularización.
- `HighPrecision='auto'`: utiliza Symbolic Math Toolbox si está disponible;
  si se necesita y no está disponible, conserva el estado numérico no resuelto.
  `'off'` desactiva esa recuperación; `'required'` comprueba la disponibilidad
  antes de entrenar y falla si falta. No se instala ninguna toolbox.
- v1 y v2 se calculan sobre **las mismas matrices** empírica, de entrenamiento
  y reconstruida. Los campos `NetworkX*` contienen v2; `TrophicV2LegacyMean` y
  `TrophicV2LegacyStatusCode`, con prefijos Empirical/Train/Pseudo, guardan v1.
  Las otras métricas, el muestreo negativo, la partición y el entrenamiento
  conservan su implementación anterior.

`TrophicV2FailureReason` explica el resultado. Estado 0: válido; 1: vacío;
2: numérico no resuelto; 6: sin basales; 7: nodos inaccesibles desde basales.
Los códigos anteriores permanecen disponibles en las columnas de comparación v1.
Una reconstrucción estructuralmente inválida conserva todos sus niveles como NaN;
no se calcula una media parcial para intentar incluirla en Wilcoxon.

## Ejecución local

Desde la raíz del repositorio, usando una carpeta de salida **nueva**:

```sh
/Applications/MATLAB_R2025a.app/bin/matlab -batch "addpath('src/matlab/wlnm_version_runners/wlnm_dir_neg'); run_wlnm_dir_neg_trophic_pilot;"
```

Ejecuta exclusivamente las dos foodwebs, 100 repeticiones cada una, train ratio
60 %, K=10 y umbral 0.5. Mantiene semilla base 12345, identificadores de
experimento 1–100, particiones aleatorias por repetición, negativos `role_only`
2:1 con el mismo complemento aleatorio y conectividad/backbone desactivados.
La ejecución es secuencial en CPU para el piloto local. Las semillas son las
que calcula el runner existente; no garantizan igualdad bit a bit con otro
hardware o versión de MATLAB. La comparación v1/v2 dentro del piloto sí usa
exactamente la misma reconstrucción.

Salida predeterminada:
`src/matlab/data/result_wlnm_dir_neg_roleonly_train60_trophic_v2_pilot/`.
Una carpeta existente provoca un error, para preservar resultados.

Comprobación rápida, con **una** repetición por foodweb:

```matlab
addpath('src/matlab/wlnm_version_runners/wlnm_dir_neg');
run_wlnm_dir_neg_trophic_pilot('/tmp/nuevo_trophic_smoke', 'ExperimentIDs', 1);
```

La comparación exige los 100 identificadores para considerar completo un piloto.
Un smoke test no puede justificar incluir una red en el análisis estadístico.

## Archivos y comparación

Cada repetición guarda un MAT independiente en `ecological_snapshots`, con las
matrices dispersas, orden de nodos y taxonomía, foodweb, semilla, identificador,
K, ratio, umbral, protocolo negativo, protocolo trófico y métricas/diagnósticos.
Esto es independiente de la exportación habitual de CSV auxiliares. Los MAT
conservan toda la precisión numérica y permiten recalcular sin entrenar.

`pilot_manifest.json` y `.mat` registran configuración, toolboxes, hashes SHA-256
del código MATLAB/MEX y de las dos entradas, y estado de ejecución. Los resultados
por foodweb se guardan como MAT y CSV con todos los campos. `RUN_MANIFEST.txt`
conserva `NumExperiments=100` incluso en una comprobación parcial.

```sh
python docs/stats/compare_trophic_level_protocols.py \
  --result-root src/matlab/data/result_wlnm_dir_neg_roleonly_train60_trophic_v2_pilot
```

Genera `trophic_protocol_comparison/retention_comparison.csv` y
`status_transitions.csv`. Reutiliza las fronteras Tukey del procesador existente:
1.5×IQR, observaciones válidas y al menos 25/100 retenidas, separadamente para cada
protocolo. Los campos de diagnóstico no se tratan como métricas ecológicas.

Este piloto no actualiza automáticamente los insumos históricos de Wilcoxon.
Su revisión debe determinar qué fallos son recuperables y qué redes cumplen
la retención. Mezclar protocolos requiere dejar constancia del cambio y de su
alcance antes de producir nuevas cifras del manuscrito.

## Validación

```matlab
addpath('src/matlab/tests','src/matlab/logging');
r = runtests({'src/matlab/tests/test_networkx_trophic_levels_v2.m', ...
             'src/matlab/tests/test_wlnm_dir_neg_protocol_logging.m'});
assertSuccess(r);
```

Las pruebas cubren cadenas, ciclos sin basales, accesibilidad parcial, una solución
finita por encima del límite antiguo, componentes/desempates, permutación con
componente mayor único, autoenlaces, red vacía, compatibilidad del cálculo anterior,
ausencia de cambios en otras métricas, validación de versiones, matriz mal
condicionada, persistencia de matrices y estado del generador aleatorio.
La rama simbólica se prueba cuando está disponible; si falta la toolbox, se
comprueba el rechazo explícito de `required`.
