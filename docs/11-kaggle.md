# Importación de imágenes desde Kaggle

En **Datos**, al elegir la tarea **Clasificación de imágenes**, la tarjeta Kaggle
permite introducir `owner/dataset` o una URL `https://www.kaggle.com/datasets/...`.
La app usa la biblioteca oficial [kagglehub](https://github.com/Kaggle/kagglehub).
Está declarada en `pyproject.toml`; actualizar el entorno con `python -m pip install -e .`.

1. Introducir el identificador o la URL.
2. Si el paquete contiene varias colecciones, elegir una subcarpeta en **Opciones
   de estructura**. Ahí también se puede seleccionar un CSV concreto de etiquetas.
3. Elegir tamaño de imagen y canales.
4. Pulsar **Descargar y analizar**. La app informa de las clases, cantidades,
   estructura detectada y carpetas de particiones encontradas.
5. Pulsar **Importar todas las imágenes etiquetadas**.
6. Definir los porcentajes en el panel **Particiones** y aplicar el pipeline.
   Los índices iniciales son provisionales; nunca se preservan los splits de Kaggle.

## Estructuras admitidas

- `clase/imagen.jpg`, con carpetas envolventes opcionales.
- `train/clase/imagen.jpg`, `validation/clase/...`, `test/clase/...` y variantes
  `training`, `testing`, `val`, `valid`, `*_set`, `train_data`, `test_data` y
  `validation_data`.
- `clase/train/imagen.jpg` y `clase/test/imagen.jpg`.
- Imágenes con un CSV de etiquetas: una columna de ruta/identificador y una columna
  de clase. Ejemplo: `filename,label` o `image_id,target`. Las columnas admitidas
  están documentadas en los mensajes de la app. Se omiten automáticamente los CSV
  llamados `sample_submission` y otras tablas cuyo nombre contiene `submission`.
- PNG, JPEG, WebP, BMP y TIFF, con al menos dos clases.

Las carpetas de split se ignoran para obtener las clases. Se reúnen todas las
imágenes etiquetadas sin truncar la cantidad ni sobrescribir archivos cuyos nombres
se repitan entre splits. Se conserva la procedencia y el informe de importación en
el manifiesto del dataset, incluso después de aplicar transformaciones.

Si un test contiene imágenes sin etiqueta, la importación falla indicando cuántas
faltan: se debe seleccionar una colección etiquetada o proporcionar su CSV. No hay
forma de inferir etiquetas correctas a partir del nombre `test`. Las rutas ambiguas
necesitan referencias completas en el CSV.

La integración se centra en **clasificación por clase**. Datasets de máscaras,
cajas, pares, formatos binarios especializados o archivos ZIP internos necesitan
su adaptación/importador de tarea; no se presentan como clases automáticamente.
La descarga no convierte por sí sola cualquier formato de Kaggle en un dataset
compatible. Todas las imágenes se materializan como tensores al tamaño elegido;
la memoria requerida depende de su cantidad y resolución.

## Credenciales y condiciones

Muchos datasets públicos permiten descarga directa. Los privados y algunos públicos
con condiciones requieren una cuenta autorizada y aceptación en Kaggle. Configurar
las credenciales en el equipo donde corre el backend mediante los mecanismos de
kagglehub, por ejemplo `python -c "import kagglehub; kagglehub.login()"`, o su archivo
`~/.kaggle/access_token`. No introducir tokens en campos de la app ni en el proyecto.
La caché la gestiona kagglehub y se reutiliza al importar después del análisis.

## Verificación

`tests/test_kaggle_datasets.py` cubre estructuras con/sin split, clases unificadas,
CSV, duplicación de nombres, etiquetas ausentes y contradictorias, rutas inválidas,
metadatos persistidos y regeneración de particiones elegidas por el usuario.
Las descargas se sustituyen por fixtures locales; no se descargó un dataset remoto
real ni se comprobó una cuenta privada de Kaggle durante esta implementación.

`npm run test:kaggle-ui` utiliza Chromium con respuestas simuladas para comprobar
análisis, importación y reaplicación después de editar porcentajes de particiones.

Verificado el 2026-11-07: 134 pruebas de backend aprobadas (25 de Kaggle), build
TypeScript/Vite aprobado y recorrido de navegador aprobado a 1280×800. Vite conserva
su aviso de chunks mayores de 500 kB. Se revisaron las capturas del formulario,
informe e importación; el panel Particiones sigue visible y editable.
