# Punto de continuación: Crear dataset

2026-11-08. Pausa solicitada por el usuario por cuota. Implementación parcial: no declarar entrega terminada.

## Referencias

- Producto, alcance y cinco pantallas: `19-modo-crear-dataset.md`.
- Implementación y pendientes de Terra medium: `20-implementacion-datasets.md`.
- Entorno de toda la aplicación, Luna xhigh: `18-entorno-datasets.md`.
- No LLM ni tareas de texto. No modificar Conda compartido; solo Luna instala dependencias.

## Aportes del coordinador

- `backend/modelbuilder/data.py`: reimportación de clasificación `{train,val,test}/{clase}` y YOLO `images/{train,val,test}` / `labels/{train,val,test}`. Conserva particiones oficiales, nombres repetidos y clases. Una clase plana llamada train sigue funcionando. Las etiquetas ausentes en YOLO particionado no cuentan como negativas. Tolerancia de 1e-7 en bordes por redondeo de exportaciones.
- `tests/test_dataset_reimport.py`: ocho casos nuevos; mensajes de error ES/EN.
- `scripts/test-dataset-ui.mjs`: smoke con backend real, proyectos temporales y sin descargas de modelos. Capturas/fallos en `capturas/dataset-ui/`.
- Baseline anterior: 157 pruebas backend verdes. Suite completa posterior del coordinador: **169 pruebas verdes** en Python Conda base (`python -m pytest -q -p no:cacheprovider`). Esto no verifica el entorno nuevo ni inferencia EG2. Terra también ejecutó otras pruebas; su suite completa separada fue interrumpida.
- Smoke UI detectó arrastre nativo de imagen y botones ilegibles. Terra corrigió ambos; no considerar smoke verde hasta revisar última repetición.

## Prioridades al retomar

1. Leer docs 18/20 y comprobar procesos y estado del entorno antes de ejecutar. `.venv` estaba en preparación; no asumir instalación completa.
2. Terminar con Luna el entorno único de TODA ModelBuilder y probar entrenamiento existente. No actualizar Conda base.
3. EG2 aún no tiene inferencia real verificada. Comprobar revisión fijada del modelo y API processor/salida/proyección de Transformers 5.22.2: vectores imagen/texto finitos, normalizados y dimensión común 768. El test de caché no valida pesos. Luna no debe descargar pesos hasta reanudar.
4. Resolver el smoke UI y repetir build tras últimas ediciones. Comprobar propuestas, filtros globales/paginación, guardado al avanzar, recuperación de jobs y exportación/reimportación real.
5. Comparar funciones entregadas contra primera entrega del doc19; documentar pendientes sin ocultarlos.

```bash
# Solo después de validar el entorno con Luna:
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q -p no:cacheprovider
npm run build
MODELBUILDER_PYTHON="$PWD/.venv/bin/python" node scripts/test-dataset-ui.mjs
```

No commits/reset/checkout: hay muchos cambios previos del usuario que deben conservarse.

Último smoke UI finalizó con timeout esperando exportación; hubo actualizaciones HMR de `DatasetWorkspace.tsx` durante la ejecución y la pantalla quedó de nuevo en Fuentes. La creación de caja pasó antes del timeout. Repetir con archivos estables para distinguir reinicio por HMR de un fallo funcional de exportación. El script cerró su navegador y servidor de prueba.

Luna detuvo pip con SIGINT: PID 113587 terminado, sin instaladores activos, Conda intacto. El entorno solo tenía Python 3.12.3 y pip24; imports, pip check e inferencia no validados. Se solicitó mover el entorno incompleto fuera de `.venv` para evitar que los scripts de arranque lo seleccionen automáticamente; consultar docs18 para su ubicación final y recreación.
