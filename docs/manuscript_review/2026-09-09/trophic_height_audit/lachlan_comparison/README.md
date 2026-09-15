# Comparabilidad con los CSV de Lachlan y alternativa estructural exploratoria

Se revisaron el hilo suministrado, mean_tl.csv (289 redes), species_tl.csv
(13 451 registros) y las 290 matrices empíricas actuales en foodwebs_mat.
Todos los nombres de las 289 referencias y sus tamaños completo/componente mayor
coinciden con las matrices locales. Weddell Sea no tiene referencia de Lachlan.
Las comparaciones no cambian matrices ni resultados oficiales.

## Resultados empíricos

Se calculó la definición clásica sin límite superior, incluyendo basales, y la
generalizada de MacKay con mínimo cero en el mismo componente débil mayor. Para
separar el desplazamiento de escala de diferencias de definición, también se
comparó la generalizada +1. Esta última operación es solo un diagnóstico.

| Comparación con las 289 medias de Lachlan | Error absoluto mediano | Diferencia porcentual absoluta mediana | Redes con diferencia >10 % | Spearman |
|---|---:|---:|---:|---:|
| Clásica incluyendo basales | 0.48016 | 20.27 % | 221 | 0.89290 |
| Generalizada, mínimo cero | 1.46848 | 56.42 % | 289 | 0.77884 |
| Generalizada +1 | 0.46848 | 18.18 % | 236 | 0.77884 |

La generalizada no es intercambiable con las cifras de Lachlan. Incluso con +1,
Carpinteria pasa de 6.75137 a 2.29796. No recomendaría sustituir silenciosamente
la métrica si el objetivo es conservar comparabilidad con esas cifras.

## Discrepancia entre los archivos y la descripción del hilo

- Las 289 medias de mean_tl.csv coinciden con el promedio de species_tl.csv.
- species_tl.csv no contiene TL=1 ni TL>=10; sus extremos son 2 y 9.98239521703972.
- La media clásica excluyendo basales reproduce 287/289 referencias a 1e-8.
- En las dos restantes, restringir además a TL<10 reproduce las cifras:
  Carpinteria: 88 especies, media 6.75137133564992; Chesapeake Bay: 302 especies,
  media 3.40808314962335. Las matrices actuales tienen, respectivamente, 58 y
  71 especies con TL>=10.

Es evidencia del contenido de los archivos, no prueba de quién aplicó un filtro,
cuándo ni mediante qué script. El hilo indica que Lachlan no recuerda aplicar
ese filtro y no dispone del código. Hay que resolver la definición de la población
promediada antes de declarar reproducida su metodología. No se propone adoptar
TL<10 únicamente para hacer coincidir las medias.

Ejemplos (referencia / clásica con basales / clásica sin basales):
- PlotB: 2.1507166493 / 1.8357836716 / 2.1507166493.
- Ythan: 2.7751605235 / 2.6979796312 / 2.7751605235.
- Blackrock: 2.1377804393 / 1.4838836351 / 2.1377804393.

## Alternativa: restricción de roles en la reconstrucción

Prueba exclusivamente exploratoria sobre los 100 snapshots del piloto:
retirar de la matriz pseudo solo enlaces NUEVOS cuyo destino tiene el rol
`resource` en el MAT. Se conservan íntegros entrenamiento y referencia empírica.
La referencia se consultó después para auditar cuántos enlaces verdaderos se
retiraron, no para seleccionar enlaces a retirar.

| Foodweb | Válidos clásicos con basales | Retenidos con Tukey 1.5 IQR | Enlaces predichos retirados, total 50 runs | Runs modificados |
|---|---:|---:|---:|---:|
| PlotB | 50/50 | 48/50 | 249 | 49 |
| Ythan | 50/50 | 50/50 | 310 | 50 |

No se retiró ningún enlace verdadero en esta prueba. Esto no demuestra por sí
solo una mejora predictiva válida: los roles cargados podrían incorporar
información de la red completa. El procedimiento sirve para diagnosticar una
causa de pérdida de acceso basal y proponer una validación independiente.

En Ythan los cuatro recursos son Diatoms, Enteromorpha, Fucus ceranoides y
POM (detritus). La primera alternativa a estudiar es documentar con datos
independientes qué nodos pueden ser recursos obligadamente basales y rechazar
predicciones entrantes incompatibles con esa información disponible a priori.

No extender automáticamente esa interpretación a PlotB: sus 26 nodos resource
incluyen nematodos y otros animales. `resource` es un rol topológico de esta red,
no necesariamente un productor primario. El código de reparación de matrices
presente en el repositorio también contiene ejemplos de roles calculados desde
los grados de la red completa. La procedencia de las etiquetas es un requisito
para decidir qué restricciones pueden utilizarse sin información de prueba.

Si se adopta una restricción, hay que versionar el protocolo de reconstrucción,
aplicarla de manera consistente a todas las repeticiones y ratios previstos y
recalcular las métricas afectadas. Conservar predicciones sin restricción para
comparación. Las métricas predictivas posteriores a la restricción deben
identificarse como tales; no atribuirlas directamente al clasificador original.

No cambiar un rol, añadir presas, reutilizar enlaces retenidos para prueba ni
romper un ciclo exclusivamente para obtener una media finita. Eliminar autoenlaces
y eliminar enlaces entre dos especies diferentes son operaciones distintas.

## Recomendación

Mantener la definición clásica para este objetivo de comparabilidad. Resolver
primero la discrepancia de agregación en los CSV de referencia. Investigar una
restricción basal justificada externamente para Ythan y validar su procedencia,
en vez de sustituir por la definición generalizada o editar la red empírica.
La prueba demuestra viabilidad numérica para el piloto, no constituye aún una
validación ecológica ni una implementación aprobada del nuevo procesamiento.

El hilo es contexto proporcionado por el usuario, no una instrucción ejecutada
de Athen o Lachlan. No se enviaron mensajes ni se modificaron datos, roles,
conexiones o resultados de producción.

## Artefactos

- empirical_comparison.csv: comparación individual de las 290 matrices.
- summary.json: resumen, ejemplos y diagnósticos de tamaño.
- exploratory_role_constraint.csv: prueba estructural por repetición.
- role_constraint_summary.json: resumen de dicha prueba y excepciones de los CSV.
- compare_lachlan.py y probe_role_repair.py: reproducir desde la raíz del repositorio
  con el Python del entorno Foodweb; escriben los informes de esta carpeta.
