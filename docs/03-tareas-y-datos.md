# 03 · Tareas, compatibilidad y datos

**Estado:** contrato implementado salvo las ampliaciones señaladas expresamente. A/B/C remiten a [01](01-producto-y-alcance.md).

## 1. Vocabulario y convención de tensores

`B` = batch; `F` = variables; `C` = canales; `H,W` = dimensiones espaciales; `T` = pasos/tokens; `K` = clases; `Q` = objetivos continuos; `P` = horizonte; `D` = ancho interno; `V` = vocabulario. La UI puede mostrar las dimensiones por muestra, pero identifica B al inspeccionar el batch.

Tabular usa `[B,F]`, visión `[B,C,H,W]`, secuencias `[B,T,F]`, tokens `[B,T]`. El dtype es parte del contrato: valores continuos flotantes; IDs de tokens/clases enteros; máscaras de validez booleanas. Layout no se infiere sólo contando ejes.

## 2. Matriz de compatibilidad del producto

| ID de tarea | Familias admitidas | Entrada admitida | Objetivo / salida | Vista de inferencia |
| --- | --- | --- | --- | --- |
| `tabular.classification` | `mlp` | CSV con variables numéricas | Etiqueta binaria/multiclase → logits `[B,K]`, K≥2 | Clase + probabilidades |
| `tabular.regression` | `mlp` | CSV con variables numéricas y una columna objetivo | Valor `[B,1]` | Valor, residual |
| `tabular.reconstruction` | `autoencoder` | CSV con variables numéricas | Reconstrucción `[B,F]` | Original, reconstrucción, error |
| `image.classification` | `cnn`, `vit` | PNG/JPEG/WebP en `raíz/clase/imagen` | Una clase por imagen → `[B,K]` | Imagen + probabilidades |
| `image.regression` | `cnn`, `vit` | Imágenes + `labels.csv` con `filename,target` | Valor `[B,1]` | Imagen + valor |
| `image.reconstruction` | `autoencoder` | Carpeta de imágenes | Reconstrucción `[B,C,H,W]` | Original, reconstrucción, error |
| `sequence.classification` | `cnn1d`, `lstm`, `transformer` | CSV con secuencias de igual longitud | Una etiqueta por secuencia → `[B,K]` | Serie + clase |
| `sequence.regression` | `cnn1d`, `lstm`, `transformer` | CSV con secuencias de igual longitud y una columna objetivo | Valor `[B,1]` | Serie + valor |
| `sequence.forecast` | `cnn1d`, `lstm`, `transformer` | CSV con historia y columnas `future_1…future_P` | Horizonte `[B,P]` | Pasado + horizonte |
| `text.classification` | `transformer` | TXT/TSV con una muestra por línea o `label<TAB>texto` | Una etiqueta por texto → `[B,K]` | Texto + probabilidades |
| `text.language_model` | `transformer_causal` | TXT con una muestra por línea | Logits `[B,T,V]`, targets desplazados `[B,T]` | Predicción token a token |
| `image.segmentation.binary` | `unet` | Carpetas `images/` y `masks/` alineadas | Máscara 0/1, logits `[B,1,H,W]` | Overlay y probabilidad de foreground |
| `image.segmentation.multiclass` | `unet` | Carpetas `images/` y `masks/` alineadas | Una clase por píxel, logits `[B,K,H,W]` | Overlay por clase |

> **Nota sobre bloques recurrentes:** `rnn` y `gru` existen como bloques atómicos en `models.py`, pero actualmente ninguna tarea del catálogo los habilita. Se pueden añadir a `sequence.*` ampliando `backend/modelbuilder/catalog.py`.
>
> **No implementados:** clasificación multietiqueta, `image.denoising`, VAE como tarea, traducción encoder–decoder y streaming recurrente con estado persistente. Estos requieren contratos de datos, pérdidas o ejecución adicionales y no se muestran como disponibles.

Las compatibilidades son decisiones de interfaz y adapters soportados, no limitaciones teóricas de una familia.

## 3. Tarjeta de selección antes de importar

Cada opción expone modalidad, tareas, formato y salida con un ejemplo. «CNN / Regresión visual» debe decir «imágenes + archivo con valores numéricos por imagen»; «U-Net / Segmentación» debe decir «imagen y máscara alineadas», no sólo «imágenes».

Se selecciona la variante de tarea antes de importar. En Datos se resuelven variables, clases, resolución, longitud/horizonte, targets y vocabulario. El contrato se guarda y alimenta Entrada/Salida en el constructor. Clases y columnas tienen IDs estables separados de nombres visibles.

## 4. Importadores de A

### 4.1 Tablas

CSV con cabecera. En esta versión las columnas deben ser numéricas; si una columna no se puede convertir a `float` se reporta como error. Seleccionar columnas de entrada y una columna objetivo; nunca asumir que el target es la última columna.

Clasificación: mapa de etiquetas aprendido del conjunto permitido de train; las clases en validation/test ausentes de train son un error. Regresión admite una única columna objetivo.

### 4.2 Clasificación de imágenes

Layout admitido: `raíz/clase/imagen`. Se detectan imágenes corruptas, archivos no admitidos y nombres repetidos. Canales de trabajo: gris o RGB; imágenes con canal alpha se convierten a RGB.

### 4.3 Regresión de imágenes

Carpeta de imágenes y `labels.csv` con `filename,target`. Cada archivo tiene un registro inequívoco. Detectar archivo ausente, etiqueta duplicada y objetivo inválido.

### 4.4 Secuencias y pronóstico

CSV con columnas de features y, para clasificación/regresión, una columna `target`. Todas las secuencias deben tener la misma longitud. El orden de filas se respeta como orden temporal.

Pronóstico: el CSV debe incluir las columnas de contexto (`feature_1…feature_F`) y las columnas objetivo nombradas `future_1…future_P`. El modelo predice directamente esas `P` posiciones futuras.

### 4.5 Segmentación

Carpetas planas `images/` y `masks/` emparejadas por nombre base. Tamaños originales deben coincidir.

Binaria: cualquier máscara se binariza como `(mask > 0).long()`. Multiclase: máscara indexada con valores 0…K−1. No usar máscaras JPEG como etiquetas discretas.

## 5. Importadores adicionales (no implementados)

- Autoencoder de imagen con pares ruido/limpio (`image.denoising`).
- VAE como tarea con pérdida compuesta.
- Clasificación multietiqueta.
- Texto causal con tokenizador BPE entrenado y controles de sampling.
- Segmentación con máscaras 0/255, paletas de color, splits predefinidos en subcarpetas y augmentaciones geométricas pareadas.

Estos importadores requieren contratos adicionales y se dejan como ampliaciones futuras.

## 6. Preprocesamiento sin fuga de datos

El pipeline actual es simple y no transforma los datos de train antes de validar/test. La partición se calcula sobre el dataset completo y cada split se materializa como un tensor `.pt` con los índices correspondientes.

Pipeline persistido, independiente del grafo de capas:

| Modalidad | Preparación inicial | Estado que acompaña al checkpoint |
| --- | --- | --- |
| Tabular | Conversión a float, selección de columnas | Columnas de entrada, columna objetivo, clases |
| Imágenes | Resize a resolución fija, conversión a RGB/gris, normalización `[0,1]` | Resolución, canales |
| Secuencias | Conversión a float | Columnas de entrada, longitud, columna objetivo |
| Segmentación | Resize de imagen y máscara a resolución fija | Resolución, número de clases |
| Texto | Tokenización por palabras con vocabulario aprendido en train | Vocabulario, `max_length`, `padding_idx` |

Reajustar datos o transformaciones crea una nueva versión del dataset; las corridas anteriores conservan la referencia al snapshot que usaron.

## 7. Particiones

Por defecto 70/15/15, ajustables arrastrando los límites. Nunca se crean splits vacíos silenciosamente. Se puede entrenar sin test; en ese caso la evaluación final independiente queda deshabilitada.

La partición actual es aleatoria con semilla (opcionalmente estratificada para clasificación tabular e imágenes). Los índices exactos se guardan en el manifiesto del dataset para poder reproducir la división.

## 8. Generadores sintéticos por tarea

Los generadores están ligados a la tarea, no a la familia arquitectónica. El mismo generador de imágenes sirve para `cnn`, `vit` y `autoencoder`; el mismo generador de secuencias sirve para `cnn1d`, `lstm` y `transformer`. Todos guardan `seed` y parámetros en el manifiesto.

| Tarea | Generador y etiqueta correcta | Controles principales |
| --- | --- | --- |
| `tabular.classification` | Blobs lineales con 3 clases | N, F, K, separación, ruido, balance |
| `tabular.regression` | Combinación lineal de 3 features + ruido | N, F, ruido |
| `tabular.reconstruction` | Mismo generador tabular; target = entrada | N, F |
| `image.classification` | Formas geométricas (círculo, cuadrado, triángulo) | N, resolución, gris/RGB, K, ruido |
| `image.regression` | Área del objeto como target | N, resolución, ruido |
| `image.reconstruction` | Mismo generador de imágenes; target = entrada | N, resolución |
| `sequence.classification` | Ondas seno/coseno/lineal con 3 frecuencias | N, T, F, clases, ruido |
| `sequence.regression` | Media absoluta de la primera variable como target | N, T, F, ruido |
| `sequence.forecast` | Seno desplazado como horizonte | N, T, P, F, ruido |
| `text.classification` | Tokens aleatorios; clase determinada por primer token | N, max_length, vocab_size |
| `text.language_model` | Mismo corpus; target = token siguiente | N, max_length, vocab_size |
| `image.segmentation.binary` | Objeto sobre fondo con máscara 0/1 | N, H/W, objetos, ruido |
| `image.segmentation.multiclass` | Círculos/cuadrados/triángulos + fondo | N, H/W, K, ruido |

Los generadores no codifican la etiqueta en el nombre de archivo ni en un ID usado como feature. Los corpus pequeños sirven para verificar el flujo, no para chat de propósito general.

## 9. Manifiesto de datos

El manifiesto guarda: `dataset_id`, `task_id`, `architecture`, `inputShape`, `targetShape`, `classes`, `columns`, `partitions` (índices train/validation/test), `options` (seed, parámetros del generador o rutas de importación) y la ruta al tensor `dataset.pt`.

Los datasets se materializan completos como tensores en disco; se cargan por demanda durante el entrenamiento. Si cambian las fuentes externas, hay que reimportar o regenerar el dataset.

## 10. Errores con mensajes específicos

El backend devuelve errores accionables para: valores no numéricos en tablas, columnas faltantes, imágenes corruptas, máscara sin par, resoluciones incompatibles, dataset demasiado pequeño, split con una sola clase cuando se requiere más, campos no finitos y rutas de importación inexistentes. Cada error incluye causa y corrección sugerida.
