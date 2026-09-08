# 08 · Registro del reinicio y respaldo

**Fecha:** 5 de septiembre de 2026.

El usuario autorizó retirar el trabajo anterior para replantear ModelBuilder desde cero y pidió documentar la nueva implementación antes de decidir si se construye.

## Operación realizada

Se trasladaron fuera de la carpeta activa los siguientes elementos anteriores: `src/`, `tests/`, `scripts/`, `docs/`, `data/`, `projects/`, `artifacts/`, `.demo-cache/`, `.pytest_cache/`, `.gitignore`, `README.md`, `pyproject.toml` y `run.sh`.

**Respaldo recuperable:** [/home/randy/Python/ModelBuilder-respaldo-2026-09-05-GPzMRe](../../ModelBuilder-respaldo-2026-09-05-GPzMRe/).

Fue un traslado local, sin borrado irreversible. El respaldo incluye proyectos, pesos/datos, capturas, código y documentación anterior, incluida la decisión de stack. No se tocó el entorno conda ni otros proyectos. No se encontró repositorio Git dentro de la carpeta activa al revisar el trabajo previo.

Se conservaron sin modificar `references/LLM.md` y las cuatro imágenes WebP en `references/`. Sirven como fuentes visuales y de producto, no como implementación nueva.

## Estado al cerrar la etapa documental

La carpeta activa contenía exclusivamente `README.md`, `docs/` con la propuesta nueva y `references/` con las referencias originales. Después de la aprobación del usuario se añadió la implementación React/Tauri/Python descrita en el README actual.

El reinicio permanece documentado para distinguir el respaldo PyQt de la nueva base. El código actual no procede de copiar los widgets anteriores.

## Recuperación del material anterior

Todo el material retirado se puede consultar en el respaldo. Para recuperar sólo un proyecto/dataset, copiar su ruta concreta a otra ubicación elegida. Para restaurar toda la aplicación anterior, hacerlo en una carpeta separada o conservar primero esta propuesta; no mezclar ambas versiones sobrescribiendo el nuevo README/docs.
