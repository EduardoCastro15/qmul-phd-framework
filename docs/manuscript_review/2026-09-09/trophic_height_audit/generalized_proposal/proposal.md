# Propuesta: altura trófica generalizada para WLNM_dir_neg

## Decisión metodológica necesaria

El cálculo clásico sigue indefinido en reconstrucciones sin acceso basal. No
existe una corrección numérica que preserve esa definición y garantice valores
finitos para todas las topologías. NetworkX comprueba explícitamente esa condición:
https://networkx.org/documentation/stable/_modules/networkx/algorithms/centrality/trophic.html

Propongo evaluar como medida de jerarquía la altura trófica generalizada de
MacKay, Johnson y Sansom (2020), DOI 10.1098/rsos.201138. Está definida para redes
dirigidas generales y no requiere basales. Es un cambio de definición, no una
imputación de los resultados clásicos que faltan. Si el objetivo científico
requiere específicamente posición trófica clásica respecto a recursos basales,
esta sustitución no responde exactamente a esa pregunta.

Fuente primaria y ecuaciones:
https://arxiv.org/pdf/2001.05173
Artículo publicado: https://doi.org/10.1098/rsos.201138

## Definición propuesta

Mantener A(recurso, consumidor), adyacencia binaria sin autoenlaces y la selección
actual del componente débil mayor. Para ese componente:

- kin = suma de columnas; kout = suma de filas.
- L = diag(kin + kout) - A - A'.
- Resolver L h = kin - kout.
- Fijar temporalmente un nodo en cero para resolver el sistema reducido.
- Normalizar después h := h - min(h).
- Reportar mean(h), con el nombre explícito Mean generalized trophic height.

La normalización es parte de la definición de la media. Fijar simplemente un
nodo arbitrario sin normalizar daría medias dependientes del nodo elegido.
Un cero representa el mínimo jerárquico del componente, no necesariamente una
especie basal. No se añadirá 1 para aparentar equivalencia con el cálculo clásico.
Un ciclo dirigido equilibrado puede dar alturas todas cero: significa ausencia
de jerarquía según esta medida, no que las especies sean productores primarios.

En un componente conectado no vacío el sistema reducido tiene solución única;
la singularidad del Laplaciano completo corresponde a la libertad de sumar una
constante. No se precisa un coeficiente de regularización. Singleton: altura cero
por convención, con un indicador de caso degenerado. Una red sin nodos sigue sin
tener media definida. Reportar también tamaño y fracción del componente utilizado.

## Prueba exploratoria realizada

Recalculé las matrices empírica, de entrenamiento y reconstruida de los 100
snapshots del piloto al 60 %, sin entrenamiento ni modificaciones del pipeline.
Usé el componente débil mayor y min(h)=0; apliqué las fronteras del procesador
Tukey existente, 1.5 IQR. Las referencias empíricas son finitas.

| Foodweb | Válidos clásicos v2 | Válidos generalizados | Retenidos clásicos v2 | Retenidos generalizados |
|---|---:|---:|---:|---:|
| PlotB | 37/50 | 50/50 | 31/50 | 49/50 |
| Ythan | 22/50 | 50/50 | 18/50 | 47/50 |

Máximo residuo escalado en los 300 cálculos: 1.0540046334054393e-16.
Los detalles están en exploratory_results.json. Esta prueba confirma cobertura
para estas matrices, no certifica todavía la implementación MATLAB ni todos los
ratios y foodwebs. Los valores no deben mezclarse con las medias clásicas.

## Implementación propuesta

1. Añadir un helper independiente compute_generalized_trophic_levels.m con
   validación de entrada, componente, resolución dispersa y normalización. Guardar
   vector, media, dispersión, rango, residuo, estado, cobertura y degeneración.
2. Introducir el protocolo generalized_mackay2020_v1 como salida adicional en
   WLNM_dir_neg. Conservar legacy_v1 y validated_v2 como resultados diferenciados.
   Campos nuevos Empirical/Train/PseudoGeneralizedMeanTrophicHeight; referencias
   y deltas calculados únicamente entre medidas de la misma definición.
3. Añadir recálculo desde snapshots que exporte CSV/MAT y manifiesto con hashes,
   normalización y selección de componentes. No repetir entrenamientos si existen
   las tres matrices necesarias de cada repetición. Usar salidas nuevas.
4. Probar cadenas, ciclos equilibrados y no equilibrados, grafos sin basales,
   componentes desconectados, singleton y vacío; verificar permutaciones con
   componente mayor único, invariancia al nodo usado como referencia, orientación
   y residuo. Cotejar MATLAB con el prototipo independiente y ejemplos de la fuente.
5. Aplicar la misma definición a todas las foodwebs, ratios y referencias empíricas
   de WLNM_dir_neg incluidas en el análisis. Evitar sustituir únicamente los NaN.
   Auditar primero la disponibilidad de matrices; donde solo haya estadísticas
   agregadas, habrá que recuperar las matrices o regenerar las reconstrucciones.
   El recálculo de la métrica puede ser local; un rerun amplio puede planificarse
   en Apocrita. Guardar snapshots en todas las ejecuciones regeneradas.
6. Generar un informe por foodweb y ratio con ejecuciones esperadas, completadas,
   finitas, degeneradas y retenidas; comparar cambios de orden y dispersión entre
   ambas definiciones en redes donde la clásica existe. Mantener resultados
   clásicos como análisis de sensibilidad y explicar el cambio en el manuscrito.

No alterar el muestreo, umbral, semillas ni enlaces para conseguir un valor
finito. Tampoco elegir reconstrucciones o repetir semillas hasta que desaparezcan
los fallos. Si se comparan modelos diferentes con la nueva métrica, sus matrices
necesitan el mismo cálculo para mantener comparabilidad.

## Criterio de éxito y límites

Para redes válidas no vacías, obtener niveles finitos con residuo dentro de la
tolerancia, normalización reproducible y diagnóstico explícito; validar esto en
todas las reconstrucciones disponibles. La definición elimina la restricción de
acceso basal. La cobertura observada se comprobará en toda la colección antes
de afirmar 100 %.

Mantener el criterio de al menos 25/50 después de Tukey. Tener 50 valores finitos
no garantiza automáticamente superar dicho filtro. No ajustar las fronteras para
forzar inclusión. Redes degeneradas y comparaciones sin diferencias informativas
necesitan un tratamiento declarado, aunque sus métricas sean numéricamente finitas.
El objetivo de cobertura para las demás métricas requiere además auditar sus
propias condiciones de definición; este cambio no las valida automáticamente.

Estado: propuesta con prototipo exploratorio; no aplicada al código de producción,
resultados oficiales, Wilcoxon, figuras ni manuscrito.
