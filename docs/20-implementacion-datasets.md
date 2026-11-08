# Estado de implementación: Crear dataset

Actualización: 2026-11-08. Alcance de referencia: `19-modo-crear-dataset.md`. El proyecto y el estado de entrenamiento permanecen independientes del manifiesto de curación.

## Entregado

- Tipo de proyecto dataset en Inicio; creación, búsqueda, apertura y eliminación, con protección del autoguardado y la validación de entrenamiento.
- Cinco secciones: importación de imágenes, extracción de vídeo, organización, cajas manuales, revisión y exportación. Miniaturas paginadas con concurrencia limitada; imágenes originales solo bajo demanda. El diálogo Tauri de vídeo tiene filtro propio.
- Fuentes y trabajos duraderos con progreso, cancelación, reanudación, recuperación de procesos interrumpidos, exclusión interproceso y detección de conflictos al guardar muestras modificadas simultáneamente.
- Clases estables, grupos y propuestas, filtros globales por fuente, grupo, clase, estado y alertas; selección y aceptación/descarto por lote, negativas explícitas y duplicados exactos. Edición manual de cajas normalizadas y selección de clase junto a la caja.
- Exportación durable y cancelable de clasificación y detección YOLO con vista previa, omisiones y particiones reproducibles por origen y duplicados; reimportación de formatos particionados conservando splits en el importador de entrenamiento.
- YOLO por lote en CPU con catálogo, consulta de clases del checkpoint elegido, selección de clases, propuestas y recortes con padding. Pesos almacenados bajo caché del proyecto. EmbeddingGemma 2 implementado con revisión fija, cache de vectores, K-means y propuestas de similitud; la interfaz informa disponibilidad real de runtime y pesos.

## Comprobaciones

- `npm run build` pasó tras los cambios de interfaz.
- `python -m pytest -q -p no:cacheprovider`: **179 pruebas** correctas en el Python base. `PYTHONPATH=backend pytest -q tests/test_dataset_project.py tests/test_dataset_reimport.py tests/test_dataset_embeddings.py tests/test_dataset_model_jobs.py`: **22 pruebas** correctas. Las pruebas de caché de embeddings usan vectores artificiales y no validan pesos.
- Smoke web con backend real: importación de ocho imágenes, caja y geometría, exportación y reapertura correctas. El script `scripts/test-dataset-ui.mjs` espera a que se habilite el botón Exportar tras el guardado, evitando una carrera de la prueba.
- Inferencia real YOLO11n en CPU: 13 imágenes sintéticas sin errores y la imagen `bus.jpg` de Ultralytics produjo cinco cajas y cinco recortes; se verificaron sus dimensiones con padding. Una consulta real del checkpoint recuperó 80 clases.
- EmbeddingGemma 2: en `.venv` la inferencia real de una imagen y dos clases con la revisión fijada produjo vectores finitos de 768 dimensiones, norma de imagen 0.99999994 y similitudes 0.72348 y 0.64191; propuso la clase esperada. Una segunda prueba real agrupó dos imágenes en dos grupos propuestos. El Python base informa correctamente de la incompatibilidad de su Transformers. Ver `18-entorno-datasets.md` para el entorno unificado.
- `cargo check --offline` no pudo compilar Tauri porque faltan los archivos de desarrollo de sistema de Pango/Cairo. El comando y diálogo de vídeo se añadieron, pero el binario Tauri no se ha validado en esta máquina.

## Límites pendientes de comprobación o mejora

- Ampliar la evaluación real de EG2 a más imágenes y dominios; la prueba real existente comprueba una imagen y dos clases, no la calidad general del modelo.
- Los 36 nombres del catálogo se contrastaron con los assets declarados por Ultralytics local; verificar licencias y ejecución de cada checkpoint. Solo YOLO11n tiene inferencia real comprobada aquí. El detector usa CPU para evitar contención con entrenamiento.
- Completar QA en escritorio Tauri con bibliotecas de sistema disponibles, y casos de UI adicionales para vídeo, reanudación, selección en múltiples páginas y revisión de propuestas.
- El visor admite corrección numérica de cajas y zoom; conviene ampliar pruebas de EXIF, navegación entre páginas y flujos de edición concurrente. Los recuentos y filtros son globales, aunque la selección visual se hace por página.
- La API síncrona de exportación se conserva para compatibilidad; la interfaz usa trabajos persistentes cancelables. El manifiesto JSON completo se reescribe por operación; se recomienda migración de esquema a almacenamiento incremental si el inventario crece mucho.
