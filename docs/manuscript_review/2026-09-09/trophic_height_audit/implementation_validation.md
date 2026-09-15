# Validación de la implementación limitada a PlotB y Ythan

Fecha: 2026-09-09. Implementado únicamente en el runner de `WLNM_dir_neg`.
La activación es explícita mediante `run_wlnm_dir_neg_trophic_pilot`.
Guía: `src/matlab/wlnm_version_runners/wlnm_dir_neg/TROPHIC_V2.md`.

## Comprobado

- 12 pruebas MATLAB aprobadas: 11 del cálculo nuevo y 1 de compatibilidad del
  registro del protocolo negativo. MATLAB R2025a.
- 3 pruebas Python aprobadas: piloto incompleto, separación de invalidez y
  exclusión Tukey, e invalidez de la referencia empírica.
- Ejecución real completa del experimento 1, train ratio 60 %, para cada una
  de las dos foodwebs. Entrenamiento, métricas, diagnósticos CSV y snapshots MAT
  completados. Se usa una sola reconstrucción para ambos cálculos.
- Reconstrucción independiente del sistema trófico con SciPy/NumPy sobre las
  matrices exportadas: niveles finitos coincidentes dentro de tolerancia
  relativa/absoluta `1e-10`; comprobación de diagnósticos, deltas, identificadores
  y semillas entre MAT y CSV.
- Verificados SHA-256 de los 96 archivos de código registrados y las 2 entradas.
- `git diff --check` sin errores para el código modificado.
- Sin cambios en `compute_foodweb_metrics.m`, `Main.m` ni el logger compartido.

## Resultado de la comprobación real (una repetición por red)

| Foodweb | Estado v1 | Estado v2 | Media v1 | Media v2 | Residuo escalado v2 |
|---|---:|---:|---:|---:|---:|
| Dutch Microfauna food web PlotB | 0 | 0 | 1.91846606532281 | 1.91846606532281 | 1.04369207928414e-18 |
| Ythan Estuary | 3 | 0 | NaN | 204.634210863608 | 1.27445343637621e-18 |

La solución de Ythan tiene máximo `215.032426556361`, componente de 90 nodos,
y `rcond=5.82615343417512e-06`. Supera el límite antiguo de 90, pero satisface
la ecuación y los controles numéricos nuevos. El valor alto debe conservarse
para la posterior evaluación Tukey; no se limita ni se sustituye por otro valor.
Esto verifica recuperación en esta nueva reconstrucción, no identifica la
matriz exacta de una ejecución histórica de Apocrita.

## Límites y trabajo posterior

- El piloto estadístico completo de 50 repeticiones por foodweb **no se ejecutó**
  en esta implementación. La comprobación de una repetición no permite evaluar
  retención ni actualizar Wilcoxon. El comparador marca ambos grupos como
  incompletos y mantiene el mínimo de 25/50.
- Symbolic Math Toolbox no está instalada localmente. Se comprobó el rechazo
  temprano de `required`; la rama de recuperación a 50/100 dígitos necesita
  validación en una instalación que disponga de esa toolbox.
- Los fallos de acceso basal siguen siendo inválidos. El nuevo cálculo no
  garantiza la inclusión de ambas foodwebs tras Tukey.
- Resultados históricos, notebooks, manuscrito y figuras conservados.

Artefactos de comprobación:
`src/matlab/data/result_wlnm_dir_neg_roleonly_train60_trophic_v2_smoke/`.
Contiene configuración y hashes, dos snapshots, dos CSV con todos los campos,
MAT de resultados y la comparación de protocolos. El comando de 50 repeticiones
utiliza otra carpeta, terminada en `_pilot`, para mantener esta evidencia.
