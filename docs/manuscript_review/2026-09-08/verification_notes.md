# Verificación del manuscrito — 8 de septiembre de 2026

Se completaron las correcciones verificables en dos copias DOCX: 23 párrafos del manuscrito y 14 del documento de figuras y tablas. Las inserciones y sustituciones aparecen en rojo. Los originales permanecen intactos. El registro de cambios conserva los textos completos anterior y final, el motivo y la fuente; las eliminaciones se ven tachadas en su versión HTML.

## Procedencia

La fuente principal es `/Users/acw792/Developer/qmul-phd-framework/src/matlab/data/result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_Apocrita_neg_const`. Se verificó la condición `role_only`, cinco reglas por rol, masa corporal desactivada, razón objetivo de dos negativos por positivo, clasificación a 0.5, K=10, 50 repeticiones y entrenamiento del 10 al 90 %. Los resúmenes usan `retention_protocol/tukey_iqr_1p5_min50pct_threshold0p50_v1` y las medias retenidas por red y métrica.

Las dos fuentes auxiliares se conservaron con autorización expresa: `/Users/acw792/Developer/qmul-phd-framework/src/matlab/data/result_wlnm_dir_neg_roleonly_figure2_clcontrol2_train60_bestfn_local` solo para la Figura 2 y su tabla de especies, y `/Users/acw792/Developer/qmul-phd-framework/src/matlab/data/result_wlnm_dir_neg_kfold_sweep_train_ratios_10-90_pseudo_properties_Apocrita_neg_const` solo para la Figura suplementaria 3. La comparación con el WLNM original procede por separado de `/Users/acw792/Developer/qmul-phd-framework/src/matlab/data/result_wlnm_original_50x290_train90_thresh0p50_legacy_uppertriangular_checkconnfalse_adaptivefalse_Apocrita`. Las diez imágenes de resultados del DOCX coinciden píxel a píxel con los outputs identificados en los notebooks; la correspondencia está en `verified_values/figure_image_matches.csv`.

La Figura 2 representa la ejecución 21, semilla 1314098572, seleccionada entre 50 por menor número de falsos negativos y los desempates registrados. Al 60 % se usaron 460 positivos de entrenamiento; el test contiene TP=284, FP=8, FN=24 y TN=608. Es un ejemplo favorable seleccionado. Sus 97 especies y masas se contrastaron con la fuente; la tabla usa miligramos.

## Correcciones metodológicas

- Los enlaces van de recurso a consumidor. Se eliminan autoenlaces antes de dividir y contar los positivos utilizables. Los negativos se extraen uniformemente sin reemplazo del conjunto elegible; si resulta insuficiente, se retiene completo y se cubre el déficit con no-enlaces válidos restantes. Las cinco combinaciones son top→top, basal→basal, top→basal, top→intermediate e intermediate→basal. Los logs verifican 207 redes con conjunto suficiente y 83 con complemento aleatorio.
- La evaluación considera positivos reservados y negativos de prueba muestreados. La red reconstruida reúne los positivos de entrenamiento y los candidatos de prueba clasificados como positivos; no representa una evaluación exhaustiva de todos los pares no observados.
- El código expande el subgrafo hasta dos pasos, trunca antes del orden canónico y rellena el vector con ceros. Usa distancias sobre el triángulo superior tratado como grafo no dirigido para iniciar el coloreado, y vectoriza entradas dirigidas ponderadas fuera de la diagonal.
- Tukey se aplica por red, condición y métrica a valores finitos. Los límites son inclusivos: [Q1−1.5 IQR, Q3+1.5 IQR], con percentiles interpolados linealmente. Se exige conservar al menos 25/50 ejecuciones. La media se calcula sobre las retenidas; la retención de una métrica no excluye automáticamente las demás. Los errores estándar del texto se calculan entre las medias por red.
- En validación cruzada, las particiones positivas se fijan una vez por red y k con semilla 12345. Las 20 repeticiones cambian el ajuste y el muestreo negativo; la semilla por fold f y repetición e es 12345+1000f+e. Se promedian primero los folds completos de cada repetición, se filtran después esas medias y se exige conservar 10/20.
- La conectancia ecológica es L/[S(S−1)]. La generalidad promedia los recursos solo entre taxa con recursos, y la vulnerabilidad los consumidores solo entre taxa con consumidores. La altura trófica usa el mayor componente débilmente conectado, incluye basales a nivel 1 y excluye soluciones inválidas. La fuente retenida es `PseudoNetworkXMeanTrophicLevel`, calculada en MATLAB con una definición compatible con NetworkX.

## Resultados y Wilcoxon

Se recalcularon los resúmenes a partir de los CSV retenidos sin ejecutar notebooks completos ni entrenar modelos. Los CSV de `verified_values` conservan la precisión numérica disponible, las rutas y los filtros empleados. Los valores provienen de archivos que ya contienen redondeo de exportación; no se presentan como precisión adicional de las ejecuciones MATLAB originales.

Al 90 %, las medias dirigidas son ROC-AUC=0.878820, PR-AUC=0.503538, F1=0.665219, MCC=0.620619, precisión=0.904026, recall=0.582965 y TSS=0.544390. El manuscrito distingue las mejoras frente al original de las reducciones en PR-AUC y recall, y reconoce que los protocolos dirigido y no dirigido difieren.

La Figura 4 y la suplementaria 8 muestran **medianas e intervalos percentil 25–75** entre redes. Los boxplots usan bigotes hasta las observaciones extremas dentro de 1.5 IQR; los puntos fuera de ellos son valores por red y son distintos del filtrado previo de ejecuciones.

Wilcoxon se calculó sobre diferencias emparejadas de media pseudo posterior a Tukey menos referencia empírica, redondeadas a 12 decimales. Se usó contraste bilateral, exclusión de ceros (`wilcox`), rangos medios en empates, selección automática del método y ausencia de corrección de continuidad, con SciPy 1.18.0. W es el menor de las sumas de rangos positivos y negativos; el efecto es la correlación biserial por rangos (W+−W−)/(W++W−). Se reportan exclusivamente p-valores sin ajustar y α=0.05. Los contrastes globales y por ecosistema se distinguen en `verified_values/ecological_wilcoxon.csv`.

| Métrica | W | p bilateral | Efecto biserial |
|---|---:|---:|---:|
| Conectancia | 4857 | 6.432540082616343e−30 | −0.7697831497 |
| Generalidad | 7430 | 1.151982191895760e−21 | −0.6478255718 |
| Altura trófica | 18499 | 0.1026190720790153 | +0.1109669358 |
| Vulnerabilidad | 7317 | 5.348935410154284e−22 | −0.6531816566 |

La ausencia de significación global en altura trófica no demuestra equivalencia. Las diferencias relativas medias dentro de red son −11.30 % en conectancia, +4.61 % en altura trófica y −7.33 % en generalidad y vulnerabilidad. Se moderaron las afirmaciones sobre recuperación completa y conservación de estructura.

## Correspondencia e integridad

La tabla `figure_table_mapping.csv` recoge las cinco figuras principales, ocho suplementarias y una tabla, sus paneles y las menciones modificadas. La comparación de siete métricas se remite a la suplementaria 7, y el barrido de siete métricas a la suplementaria 8. Se corrigieron los paneles de extracción y coloreado del esquema, el signo de interrogación de la Figura 1 y la descripción de isomorfismo de la suplementaria 1.

Se verificaron los contenedores ZIP/XML, el texto exacto de cada párrafo corregido y la identidad de los párrafos restantes. En cada DOCX se conservaron byte a byte los otros 41 componentes del archivo, incluidas imágenes, estilos y relaciones. Se conservaron los campos y metadatos Mendeley, sus números, las tablas, los marcadores y las revisiones previas. Los campos de citas desplazados conservan el superíndice. Los hashes de ambos originales coinciden con los de la lectura inicial. `document_checks.json` y `source_manifest.csv` registran estas comprobaciones.

La revisión visual utilizó vistas previas Quick Look y la inspección directa de las imágenes. Los tres esquemas se inspeccionaron extrayendo los PDF internos de sus EMF, sin modificar los DOCX. Quick Look no reproduce fielmente todas las ecuaciones, imágenes EMF, revisiones ni saltos de página; por ello sus PDF temporales no se entregan como una versión final paginada.

## Pendientes que requieren validación adicional

1. **Enmascaramiento del enlace focal.** `graph2vector_dir_neg.m:169–198` elimina el candidato en la matriz binaria, pero construye la matriz ponderada por separado y sustituye únicamente el primer elemento del vector. Una comprobación aislada del código actual, con una red sintética de 8 nodos y semilla 23, encontró el enlace focal positivo fuera de esa primera posición en 7 de 18 ejemplos; su peso permanecía en el vector. El log y la copia instrumentada están en `audit_evidence`. Esto demuestra que el enmascaramiento no está garantizado, pero no estima su frecuencia ni su efecto en los resultados históricos. La última modificación registrada de ese archivo es `4bc67f2330` (14 de mayo de 2026); los manifiestos consultados no vinculan de forma completa cada ejecución histórica a un binario o commit archivado. La repercusión cuantitativa requiere una auditoría de aquellas ejecuciones y, si procede, experimentos posteriores. Esta revisión no modifica la implementación ni repite el entrenamiento.
2. **Paginación final en Word.** La integridad estructural y el contenido visual se comprobaron, pero la automatización local de Word no produjo una exportación utilizable. Quedan pendientes los saltos de página definitivos y la presentación de ecuaciones y revisiones en Word. Por este motivo, los documentos se entregan como copias científicamente corregidas con esta comprobación editorial pendiente, no como versiones completamente validadas para envío.

Las afirmaciones metodológicas describen las fuentes comprobadas; la revisión no presenta como ejecutados diagnósticos adicionales de supuestos ni validaciones de equivalencia.

## Fuentes externas de contraste

Se verificaron las cifras atribuidas a la compilación publicada de [Brose et al.](https://www.nature.com/articles/s41559-019-0899-x), la comparación de TSS en el [registro institucional del estudio citado](https://opus.lib.uts.edu.au/handle/10453/179160) y la distinción entre media y desviación estándar del [modelo publicado comparado](https://www.nature.com/articles/s41467-026-68769-7). Se mantuvo la bibliografía del manuscrito. La parametrización del contraste se contrastó con la [documentación oficial de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
