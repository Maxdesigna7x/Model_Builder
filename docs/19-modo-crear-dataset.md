# Crear dataset: especificación de producto e implementación

Fecha: 2026-11-08. Alcance acordado: imágenes y vídeos como fuente de imágenes. **Sin LLM y sin tareas de texto**; las etiquetas escritas son instrucciones para un encoder multimodal, no generación de texto.

Este documento describe el producto objetivo. No implica que todas las funciones estén implementadas. La entrega debe registrar el estado real y las limitaciones en una sección final, sin presentar controles simulados como funciones terminadas.

## 1. Objetivo y decisiones

ModelBuilder tendrá dos tipos de proyecto seleccionables desde Inicio: **Entrenar modelo** y **Crear dataset**. Crear dataset no exige arquitectura, grafo, optimizador ni entrenamiento. Su salida es un dataset revisable y exportable, que luego puede importarse en un proyecto de entrenamiento.

Los proyectos actuales se interpretan como `training` cuando no incluyen tipo. Los nuevos de preparación usan `projectType: "dataset"`. Se conserva el formato de proyectos existentes y sus modelos, corridas y checkpoints. Los proyectos dataset tendrán estado independiente del autoguardado del editor de modelos.

Todas las operaciones son locales salvo la descarga inicial de pesos. Un único entorno Python sirve para TODA la aplicación: entrenamiento, importación de datos, embeddings, vídeos y YOLO. No se creará un runtime exclusivo de datasets. No se actualiza el Conda compartido sin comprobar riesgos; se prefiere un entorno propio de la aplicación si hace falta aislar dependencias.

## 2. Funciones principales de la primera entrega

1. Importar una carpeta de imágenes, conservando originales y procedencia.
2. Extraer imágenes de uno o varios vídeos, seleccionando un fotograma cada N frames o cada T segundos.
3. Descubrir grupos de imágenes similares mediante embeddings; revisar y nombrar grupos.
4. Clasificar imágenes según clases definidas por el usuario mediante similitud imagen-texto.
5. Dibujar y corregir cajas manualmente, con selección de clase al terminar cada caja.
6. Detectar objetos con YOLO, proponer anotaciones y/o generar recortes con padding.
7. Revisar muestras, asignaciones y cajas; excluir muestras sin borrar originales.
8. Detectar duplicados exactos y problemas básicos de calidad; inspeccionar distribución de clases.
9. Crear particiones manteniendo unidos los derivados de un mismo origen.
10. Exportar clasificación por carpetas o detección en formato YOLO; poder reutilizarlo en entrenamiento.

La primera entrega debe resolver estos flujos completos. Las ampliaciones de la sección 13 no son condiciones de esta primera entrega.

## 3. Interfaz: Inicio y cinco partes del espacio de dataset

### Inicio

Inicio presenta dos tarjetas en columnas: Modelos a la izquierda y Datasets a la derecha, cada una con búsqueda y botón de creación propios. El botón determina el tipo; el modal solicita nombre y descripción sin selector de tipo. Las tarjetas del inicio muestran distintivo de tipo; buscar/abrir/eliminar funcionan para ambos. Los filtros por arquitectura no esconden accidentalmente los proyectos dataset. No se envía un proyecto dataset al selector de arquitectura al abrirlo.

### Estructura común

El espacio Crear dataset tiene **cinco partes**, como navegación lateral propia:

1. **Fuentes**
2. **Organizar**
3. **Anotar y extraer**
4. **Revisar**
5. **Exportar**

Cabecera: nombre del proyecto, tipo, estado de guardado y regreso a Inicio. Pie o panel persistente de trabajo: operación activa, avance real, elementos procesados/total, errores y acciones de cancelar/reanudar cuando sean aplicables. Solo un trabajo que use GPU por vez como decisión inicial; un conflicto se explica al usuario. Navegar por el proyecto no pierde el trabajo persistido.

Se conserva el tema, acento, iconografía y traducciones ES/EN de la app. Los controles tienen etiquetas, estados vacíos, errores accionables, foco visible y confirmación para descartar anotaciones sin guardar. Evitar cargar todas las imágenes en DOM o enviar sus originales en base64 en una única respuesta.

### Parte 1: Fuentes

- Zona para añadir carpeta de imágenes o vídeo(s). Escritorio: diálogo nativo. Modo web local: ruta de servidor explícita y/o subida adecuada; nunca confundir una ruta del navegador con una accesible por Python.
- Opciones: incluir subcarpetas y usar nombres de carpetas como etiquetas iniciales cuando se solicite.
- Inventario: fuente, tipo, número de imágenes/frames, resolución y problemas de lectura.
- Mosaico paginado de miniaturas; original solo bajo demanda.
- Resumen de cantidad y espacio previsto, sin modificar ni mover los archivos de origen.

Vídeos: elegir **cada N fotogramas** (N entero >= 1) o **cada T segundos** (T > 0), rango de tiempo opcional y límite máximo opcional. La primera muestra dentro del rango se incluye y luego se aplica el intervalo; documentar el redondeo temporal. Vista previa de cantidad estimada antes de ejecutar, y cantidad real al terminar. Cada frame conserva identificador de vídeo, índice real y timestamp disponible. Los vídeos con FPS desconocido admiten el muestreo por frames; el muestreo temporal debe fallar de forma explicativa si el decoder no puede ofrecer timestamps fiables. Cancelar preserva las muestras ya extraídas y registra el punto de progreso. No duplicar las ya extraídas al reanudar o volver a ejecutar los mismos parámetros.

El vídeo es una **fuente**, no otra tarea de entrenamiento: sus imágenes pueden pasar por cualquiera de las herramientas siguientes.

### Parte 2: Organizar

Dos pestañas: **Descubrir grupos** y **Clases definidas**. Comparten selector del subconjunto de imágenes y catálogo del modelo de embeddings con estado disponible/pendiente de descarga/incompatible.

Descubrir grupos: calcular embeddings y agrupar con número de grupos configurable (K-means como algoritmo inicial). Presentar grupos con miniaturas representativas, recuento y nombre editable. Admitir mover muestras y unir grupos; separar seleccionadas en un grupo nuevo. El resultado inicial se llama Grupo 1, Grupo 2, etc.; el modelo de embeddings NO inventa nombres. La elección automática de cantidad de grupos y algoritmos de densidad quedan como ampliación.

Clases definidas: crear lista estable de clases con nombre y descripción opcional; comparar embeddings de imágenes con los textos de clase. Usar un umbral configurable y mostrar las mejores alternativas y la diferencia entre similitudes. Las puntuaciones se llaman **similitud**, nunca probabilidad calibrada. Las imágenes ambiguas o bajo umbral quedan Por revisar/Sin asignar. Una imagen puede mostrar varias candidatas; la clasificación exclusiva es el formato de exportación inicial, y no se exportan asignaciones ambiguas como si fueran confirmadas.

Cambiar los textos o el algoritmo reutiliza embeddings de imagen compatibles; cambiar modelo, revisión o preprocesamiento invalida esa caché. Los resultados automáticos se guardan como propuestas; el usuario puede aceptarlos por lote, editar o descartar. No reemplazar silenciosamente anotaciones humanas.

### Parte 3: Anotar y extraer

Tres pestañas: **Cajas manuales**, **YOLO** y **Recortes** (pueden compartir un visor).

Cajas manuales: lista previa de clases; imagen grande con ajustar/zoom; anterior/siguiente; estado Sin revisar, Anotada, Negativa o Excluida. Para crear caja se arrastra entre esquinas o se marcan dos esquinas. Al soltar se abre un menú de las clases definidas junto a la caja; Escape cancela la caja pendiente. No ofrecer un clic inteligente que prometa ajustar el objeto sin modelo de segmentación. Las cajas se seleccionan, corrigen y eliminan; guardar antes de avanzar y permitir volver a imágenes anteriores. Marcar explícitamente Negativa significa revisada sin objetos; no equivale a una imagen todavía sin anotar. Convertir coordenadas entre pantalla y tamaño original, también con zoom, orientación EXIF y bordes. Cajas normalizadas válidas y clases estables.

YOLO: elegir familia/versión, tamaño, dispositivo, umbral de detección y clases ofrecidas por el checkpoint. Mostrar si los pesos están descargados. Elegir **Proponer cajas**, **Extraer recortes**, o ambas. Inferencia por lotes o streaming en imágenes originales, no en miniaturas. Los resultados son revisables. No sobrescribir cajas humanas; identificar su origen automático, modelo y configuración.

Recortes: clase(s), padding en píxeles por lado (ejemplo 20), confianza mínima y tamaño mínimo opcional. Un objeto = un archivo, con nombre sin colisiones; ajustar el padding a los límites. Guardar imagen y caja de origen. Los recortes pueden añadirse como muestras de clasificación dentro del proyecto y se identifican como derivados. Las imágenes sin detecciones no se fabrican como recortes ni se marcan automáticamente como negativas humanas.

### Parte 4: Revisar

Vista de mosaico/lista con filtros por clase, grupo, fuente, estado y alertas. Selección múltiple para asignar clase, aceptar propuestas, excluir o restaurar. Visor de imagen y cajas. Estadísticas por clase, muestras sin asignar, negativas y excluidas.

Duplicados exactos: SHA-256 del contenido; mostrar grupos y elegir cuál conservar/incluir. No eliminar los originales. Alertas básicas: imagen ilegible, dimensiones pequeñas y etiquetas/cajas inválidas. Umbrales de tamaño configurables; las alertas no excluyen silenciosamente. Un resumen de revisión debe distinguir propuesta automática de etiqueta confirmada. Duplicados perceptuales, borrosidad avanzada y vecinos visuales quedan en la ampliación.

### Parte 5: Exportar

Elegir clasificación o detección YOLO, nombre de exportación y reparto train/validation/test. Mostrar cantidad de muestras que se exportarán y por qué se omiten las demás. Rechazar clasificación exclusiva si una muestra tiene varias asignaciones finales. Para detección solo exportar imágenes revisadas con cajas aceptadas o negativas explícitas; nunca tratar Sin revisar como negativa.

Particiones con semilla reproducible. Mantener unidos original y derivados; todos los frames de un vídeo van a una misma partición por defecto. Si solo hay uno o dos grupos de origen, advertir que no pueden existir tres particiones no vacías sin romper la separación; no dividir automáticamente un vídeo. Duplicados exactos incluidos comparten grupo de partición. Informar desviaciones de los porcentajes solicitados cuando hay grupos grandes.

Exportación en un directorio nuevo/versionado bajo el proyecto, con clases, procedencia, configuración y resumen. No pisar exportaciones anteriores. Mostrar ruta copiable y cómo importarla en un proyecto de entrenamiento; la integración directa puede crear/importar un nuevo proyecto, pero no convertir destructivamente el proyecto dataset.

## 4. Catálogo de modelos

### Embeddings

Modelo solicitado: `google/embeddinggemma-2`. Imágenes y texto únicamente, sin encoder de audio cuando el runtime lo permita. Cargar con revisión fijada y processor coherente. `bfloat16` si hay soporte nativo, `float32` como fallback; no usar `float16` para este modelo. Normalizar los embeddings antes de similitud coseno y registrar dimensionalidad.

Estado conocido al redactar: el Python Conda compartido tiene Transformers 5.10.2 y su CONFIG_MAPPING no reconoce `embedding_gemma2`; el config publicado del modelo referencia 5.18.0.dev0, lo cual NO establece por sí solo una versión mínima. La compatibilidad debe verificarse con la versión instalada realmente, no deducirse de un rango amplio. SentenceTransformers está ausente. No instalar ni actualizar globalmente para solucionar esto. El análisis y preparación del entorno se documentan en `18-entorno-datasets.md`.

Si no existe un runtime compatible validado, la interfaz explica el motivo y conserva disponibles los demás flujos. No usar embeddings aleatorios ni un modelo distinto ocultamente para aparentar soporte. No añadir un LLM para resolver esta limitación.

### YOLO

Catálogo curado de detección, condicionado al soporte del Ultralytics fijado:

| Familia | Tamaños de detección |
| --- | --- |
| YOLO26 | n, s, m, l, x |
| YOLO12 | n, s, m, l, x |
| YOLO11 | n, s, m, l, x |
| YOLOv10 | n, s, m, b, l, x |
| YOLOv9 | t, s, m, c, e |
| YOLOv8 | n, s, m, l, x |
| YOLOv5u, adicional/heredado | n, s, m, l, x; variantes P6 solo si se incorporan y verifican explícitamente |

N nano, S pequeño, M mediano, L grande y X extragrande. Las excepciones B/T/C/E conservan su nombre real. No fabricar combinaciones como v9n. YOLO27 no se ofrece mientras no tenga soporte y pesos publicados comprobados. Versiones antiguas que no cargan de forma nativa en el paquete no se anuncian como disponibles. No descargar todos los modelos: solo el seleccionado.

El checkpoint determina las clases detectables. Un nombre de clase nuevo no amplía un detector de vocabulario cerrado. YOLOE/World quedan como ampliación opcional con dependencias y pesos de su encoder informados. La licencia de cada modelo se muestra conforme al catálogo; no afirmar que todos los pesos YOLO son intercambiables ni que el editor de redes admite automáticamente esos checkpoints.

## 5. Modelo de datos y persistencia

El proyecto dataset mantiene un manifiesto versionado propio (ejemplo `dataset-project.json` o base SQLite) independiente de `state.json` de entrenamiento. Para una primera versión puede usarse JSON con escritura atómica y operaciones serializadas; si crece, migrar mediante schema explícito.

Entidades mínimas:

- Fuente: id, ruta/tipo, huella, parámetros de importación y estado.
- Muestra: id estable, ubicación gestionada/original, dimensiones, hash, source_id, parent_sample_id, vídeo/frame/timestamp cuando aplique, etiquetas finales y propuestas, estado de revisión e inclusión.
- Clase: id estable, nombre, descripción opcional y orden de exportación. Eliminar/reordenar clases no debe reinterpretar índices de cajas existentes.
- Caja: id, clase estable, coordenadas normalizadas, origen humano/YOLO, aceptación y metadata del modelo.
- Trabajo: id, operación, parámetros, estado, progreso, cursor y error. Estados pending/running/completed/cancelled/error; recuperación explícita de trabajos interrumpidos.
- Exportación: id, fecha, opciones, semilla, esquema de clases, lista de muestras/particiones y ubicación.

Ejemplo de carpetas: `sources/`, `media/`, `thumbnails/`, `embeddings/`, `jobs/`, `exports/`. Los recursos están acotados al proyecto. No insertar datos masivos de píxeles o vectores en localStorage. La caché de embeddings usa muestra/huella + modelo/revisión + processor/configuración; guardar por lotes con persistencia parcial. Guardar miniaturas para listado, originales para exportación y recortes.

Los originales nunca se modifican. Si se referencian archivos externos, detectar su ausencia/cambio y explicarlo; las extracciones y subidas se guardan gestionadas en el proyecto. Los ids y nombres de salida evitan colisiones por archivos con el mismo basename. Se valida que las rutas relativas no escapen del proyecto.

## 6. Backend, procesos y recursos

Implementar módulo separado de curación y rutas `dataset.*` o equivalente. La llamada normal actual arranca Python por solicitud: no ejecutar una inferencia por imagen mediante esa ruta ni recargar pesos a cada caja. Usar un trabajo duradero por lote con progreso persistido y polling o eventos independientes del entrenamiento. Cancelación cooperativa entre muestras y limpieza de recursos; reanudación sin duplicar resultados ya confirmados.

En web y Tauri debe existir el mismo contrato. No reutilizar el único canal/proceso global de entrenamiento para resultados de datasets. Las dependencias opcionales se importan al ejecutar la operación; un fallo de YOLO no debe impedir abrir un proyecto de entrenamiento. Controlar exclusión de trabajos GPU entre modos o comunicar el conflicto. Todo cambio de manifiesto desde trabajo y UI debe preservar las modificaciones concurrentes o bloquearlas explícitamente con error recuperable.

Vídeo: OpenCV como decoder inicial si el entorno validado lo permite; no exigir FFmpeg de sistema sin necesidad comprobada. Codecs no soportados se reportan con archivo y motivo. Las estimaciones por FPS no sustituyen el índice/timestamp de frame real. Nunca cargar un vídeo entero en RAM.

## 7. Contratos de exportación e integración

Clasificación: carpetas por clase (y por partición si se eligió), nombres sanitizados y un manifiesto con id original. Detección: `images/{train,val,test}`, `labels/{train,val,test}`, `classes.txt`, `data.yaml` y procedencia. Cada TXT contiene `class_index cx cy width height` normalizados. Las negativas explícitas producen TXT vacío; las imágenes pendientes no se incluyen.

El importador de detección actual lee imágenes planas y `labels/`, y ya admite TXT YOLO o JSON. Ampliarlo de manera compatible para exportaciones con particiones, sin romper los formatos planos ni perder el reparto si se declara oficial. El importador de clasificación también necesita resolver explícitamente carpetas de partición si se exportan así. Añadir controles para que un conjunto de recortes de una sola clase sea exportable aunque el entrenamiento actual exija más de una clase; explicar esa limitación al importarlo, sin inventar una segunda clase.

## 8. Compatibilidad y entorno compartido de la aplicación

La revisión/instalación se delega a GPT-6 Luna con razonamiento muy alto; la implementación de producto a GPT-6.1 Terra medium. Ningún otro agente ejecuta instalaciones. El entorno dedicado, si se crea, debe satisfacer entrenamiento existente y nuevas operaciones; se integra mediante `MODELBUILDER_PYTHON` y scripts existentes. No sobrescribir cambios previos del usuario ni regenerar lockfiles ajenos sin necesidad.

Declarar versiones compatibles y reproducibles de torch/torchvision, Transformers, SentenceTransformers, scikit-learn y Ultralytics/decoder. No instalar extras de audio, TensorFlow, TensorRT u ONNX por defecto. Probar imports, resolución de configuración del modelo y regresiones existentes; separar disponibilidad de librería, disponibilidad de pesos y prueba real de inferencia. No declarar inferencia validada solo por importar un paquete.

## 9. Criterios de aceptación

- Crear, cerrar y reabrir ambos tipos de proyecto; no perder estados existentes.
- Importar imágenes de subcarpetas y nombres repetidos; manejo de archivos corruptos.
- Extraer frames con intervalo y rango comprobables; cancelar/reanudar sin duplicados.
- Agrupar y clasificar con embeddings reales cuando el modelo esté disponible; errores explicativos si no lo está.
- Dibujar varias cajas, elegir clase, editar/borrar, navegar y recuperar lo guardado; diferenciar negativas y pendientes.
- Seleccionar únicamente versiones/tamaños reales de YOLO; recortar con padding correcto junto a los bordes; conservar procedencia.
- Revisar/asignar/aceptar/excluir por lote sin sobrescribir anotaciones humanas.
- Detectar duplicados exactos; particiones reproducibles sin fuentes compartidas entre train/val/test.
- Exportación YOLO numéricamente válida y clasificación legible; reimportación compatible en entrenamiento.
- Trabajo cancelable con progreso real; UI usable con miniaturas y paginación.
- Build TypeScript y pruebas del motor actuales; pruebas nuevas enfocadas en geometría, vídeo, particiones, persistencia, importación/exportación y control de trabajos. Las pruebas con modelos simulados no cuentan como comprobación de pesos reales.

## 10. Orden de implementación

1. Tipo de proyecto, navegación, estado, inventario y miniaturas.
2. Importación de imágenes y extracción de frames con procedencia.
3. Clases, revisión y anotador manual persistente.
4. Exportación y adaptadores de reimportación; particiones por origen y duplicados.
5. Trabajos persistentes y YOLO para cajas/recortes.
6. EmbeddingGemma 2, caché, agrupación y clasificación; condicionado a runtime validado.
7. Verificación integrada en web y contrato Tauri; documentación de estado real.

## 11. Fuentes técnicas consultadas

- [EmbeddingGemma 2: ficha](https://huggingface.co/google/embeddinggemma-2) y [configuración](https://huggingface.co/google/embeddinggemma-2/blob/main/config.json).
- [SentenceTransformers: dependencias](https://github.com/huggingface/sentence-transformers/blob/main/pyproject.toml).
- [Ultralytics: catálogo](https://docs.ultralytics.com/models/), [v9](https://docs.ultralytics.com/models/yolov9/), [v10](https://docs.ultralytics.com/models/yolov10/) y [v5u](https://docs.ultralytics.com/models/yolov5/).

Las versiones y pesos deben comprobarse en la implementación; este listado no sustituye la validación del entorno local.

## 12. Estado de entrega

Pendiente de implementación y verificación. Completar aquí o en un informe enlazado las funciones entregadas, pruebas ejecutadas, runtime elegido y limitaciones reales.

## 13. Ampliaciones útiles, fuera de la primera entrega

- Duplicados perceptuales y por embeddings, revisión de vecinos y detección configurable de desenfoque.
- Clasificar por imágenes de ejemplo positivas/negativas.
- Agrupación de cantidad automática; buscar mediante texto e imagen dentro del dataset.
- Segmentación manual y asistida por puntos/cajas, con runtime/modelo opcional; no confundir con el anotador de cajas.
- Detección de vocabulario abierto con YOLOE/World.
- Propagación de anotaciones entre frames con revisión, no tracking silencioso como verdad.
- Etiquetado multiclase por imagen con formatos de exportación adecuados; COCO y otros formatos.
- Versiones comparables del dataset, historial completo con deshacer y métricas de acuerdo entre anotadores.

No forman parte del producto: LLM, generación de nombres/descripciones por LLM, tareas de texto, audio, entrenamiento de un YOLO externo dentro del editor actual o publicación automática de datasets.
