# 03 · Tareas, compatibilidad y datos

**Estado:** contrato implementado salvo las ampliaciones señaladas expresamente. A/B/C remiten a [01](01-producto-y-alcance.md).

## 1. Vocabulario y convención de tensores

`B` = batch; `F` = variables; `C` = canales; `H,W` = dimensiones espaciales; `T` = pasos/tokens; `K` = clases; `Q` = objetivos continuos; `P` = horizonte; `D` = ancho interno; `V` = vocabulario. La UI puede mostrar las dimensiones por muestra, pero identifica B al inspeccionar el batch.

Tabular usa `[B,F]`, visión `[B,C,H,W]`, secuencias `[B,T,F]`, tokens `[B,T]`. El dtype es parte del contrato: valores continuos flotantes; IDs de tokens/clases enteros; máscaras de validez booleanas. Layout no se infiere sólo contando ejes.

## 2. Matriz de compatibilidad del producto

| ID de tarea | Familias / entrega | Entrada admitida | Objetivo / salida | Vista de inferencia |
| --- | --- | --- | --- | --- |
| `tabular.classification` | MLP · A | CSV, variables numéricas/categóricas codificadas | Una etiqueta binaria/multiclase → logits `[B,K]`, K≥2 | Clase + probabilidades |
| `tabular.regression` | MLP · A | CSV de variables y una o varias columnas objetivo | Valores `[B,Q]` | Valores, unidades, residuales |
| `image.classification` | CNN2D/ResNet; ViT | PNG/JPEG/WebP en carpetas por clase | Una clase por imagen → `[B,K]` | Imagen + probabilidades |
| `image.regression` | CNN2D/ResNet; ViT | Imágenes + CSV de objetivos | Valores `[B,Q]` | Imagen + valores |
| `sequence.classification` | RNN/GRU/LSTM; CNN1D; Transformer encoder | CSV largo agrupado por secuencia | Una etiqueta por secuencia → `[B,K]` | Serie + clase |
| `sequence.regression` | RNN/GRU/LSTM; CNN1D; Transformer encoder | CSV largo + objetivo por secuencia | Valores `[B,Q]` | Serie + valores |
| `sequence.forecast` | LSTM · A; RNN/GRU/CNN1D · B; Transformer encoder · C | Serie temporal, una/múltiples variables | Horizonte directo `[B,P,Q]` | Pasado + horizonte |
| `image.segmentation.binary` | U-Net · A | Imágenes + máscaras | Máscara 0/1, logits `[B,1,H,W]` | Overlay y probabilidad de foreground |
| `image.segmentation.multiclass` | U-Net · A | Imágenes + máscaras indexadas/paleta | Una clase por píxel, logits `[B,K,H,W]` | Overlay por clase |
| `tabular.reconstruction` | Autoencoder · B; VAE · C | CSV sin target obligatorio | Reconstrucción `[B,F]` | Original, reconstrucción, error |
| `image.reconstruction` | Autoencoder convolucional · B; VAE · C | Carpeta de imágenes | Reconstrucción `[B,C,H,W]` | Original, reconstrucción, diferencia |
| `image.denoising` | Autoencoder convolucional · B | Imágenes limpias + ruido generado o pares ruidosa/limpia | Imagen limpia `[B,C,H,W]` | Original ruidosa, reconstrucción, referencia |
| `text.classification` | Transformer encoder · C | CSV `text,label` o JSONL equivalente | Una etiqueta por texto → `[B,K]` | Texto + probabilidades |
| `text.language_model` | Transformer causal | TXT con una muestra por línea | Logits `[B,T,V]`, targets desplazados `[B,T]` | Predicción token a token |

Clasificación multietiqueta es una ampliación explícita de B para MLP/CNN/ResNet: manifiesto con columnas binarias por etiqueta, target flotante `[B,K]`, logits independientes y umbral por etiqueta. No se confunde con multiclase, donde sólo una clase es correcta. Clasificación por timestep, traducción encoder–decoder y streaming recurrente con estado persistente quedan fuera de A–C.

Las compatibilidades son decisiones de interfaz y adapters soportados, no limitaciones teóricas de una familia. Por ejemplo, un MLP podría aplanar imágenes, pero no se ofrece esa ruta hasta que tenga preset, validación e inferencia propios.

## 3. Tarjeta de selección antes de importar

Cada opción expone modalidad, tareas, formato y salida con un ejemplo. «CNN / Regresión visual» debe decir «imágenes + archivo con valores numéricos por imagen»; «U-Net / Segmentación» debe decir «imagen y máscara alineadas», no sólo «imágenes».

Se selecciona la variante de tarea antes de importar. En Datos se resuelven variables, clases, resolución, longitud/horizonte, targets y vocabulario. El contrato se guarda y alimenta Entrada/Salida en el constructor. Clases y columnas tienen IDs estables separados de nombres visibles.

## 4. Importadores de A

### 4.1 Tablas

CSV con cabecera y detección propuesta de separador/encoding, confirmable en preview. Seleccionar columnas de entrada, columna(s) objetivo y columna de grupo opcional; nunca asumir que el target es la última columna. ID de fila estable para particiones y predicciones exportadas.

Numéricos: comprobar parseo, finitud y faltantes. Categóricos: one-hot inicial con categoría desconocida; evitar convertir categorías en enteros ordinales sin explicarlo. Cardinalidad y expansión a muchas columnas se muestran antes de confirmar. No tratar un identificador de persona como variable por defecto.

Clasificación: mapa de etiquetas aprendido del conjunto permitido de train; las clases en validation/test ausentes de train son un error a resolver, no una nueva salida añadida silenciosamente. Regresión admite múltiples objetivos y unidades opcionales.

### 4.2 Clasificación de imágenes

Layouts admitidos: `raíz/clase/imagen`, `raíz/train|val|test/clase/imagen` y `raíz/clase/train|val|test/imagen` cuando no sean ambiguos. Se reconoce `validation` como alias de `val`. Alternativa explícita: CSV `filename,label,split?`, con rutas relativas a su raíz.

Detectar imágenes corruptas, archivos no admitidos, carpetas ambiguas y nombres repetidos. Se presenta una propuesta de layout antes de confirmarla. Canales de trabajo: gris o RGB; convertir alpha según política visible, no descartar de forma imprevisible.

### 4.3 Regresión de imágenes

Carpeta y `labels.csv` con `filename,target_1,...,target_Q` y `split` opcional. Cada archivo tiene un registro inequívoco. Detectar archivo ausente, etiqueta duplicada y objetivo inválido. La UI permite mapear nombres distintos de columnas.

### 4.4 Secuencias y pronóstico

CSV largo: `sequence_id,timestep,feature_1,...` y target por secuencia repetido coherentemente o en un CSV aparte `sequence_id,target...`. Clasificación/regresión requieren coherencia del target dentro de cada secuencia. Ordenar por tiempo; tiempos duplicados y huecos requieren resolución explícita.

Pronóstico: seleccionar columna de tiempo, serie/grupo opcional, variables observadas, variables objetivo, longitud de contexto T, horizonte P y salto de ventana. Targets se extraen del futuro respecto al contexto; no se copian desde una columna objetivo ambigua. Soportar una serie larga o varias series identificadas.

Longitudes variables en clasificación/regresión: padding por batch, longitudes y máscara; el selector de estado final usa el último paso válido. En A, pronóstico utiliza ventanas de T fijo y horizonte P fijo. No interpolar huecos temporales automáticamente sin registrar la política.

### 4.5 Segmentación

`images/` y `masks/`, opcionalmente bajo train/val/test. Emparejar por ruta relativa/nombre base; ante varias extensiones candidatas, resolver la ambigüedad en UI. Tamaños originales deben coincidir. Ofrecer overlay para detectar desalineación.

Binaria: permitir valores 0/1 o 0/255 mediante mapeo confirmado. Multiclase: máscara indexada o paleta explícita color→ID; definir fondo e `ignore_index`. No usar máscaras JPEG como etiquetas discretas. Transformaciones geométricas idénticas para imagen y máscara; máscaras con vecino más cercano, sin normalización de intensidades.

## 5. Importadores adicionales B/C

- Autoencoder tabular: mismo CSV, target igual a entrada transformada; en la primera versión sólo variables numéricas para evitar una loss incorrecta sobre one-hot categóricos.
- Autoencoder de imagen: carpeta sin clases. Denoising con pares por nombre o ruido generado a partir de imágenes limpias.
- VAE: mismos contratos numéricos/imagen que reconstrucción, con distribución de salida fijada en la receta. No exigir etiquetas.
- Multietiqueta: columna por etiqueta o lista de etiquetas convertida con mapa explícito; valores 0/1 y representación clara de etiqueta ausente/desconocida.
- Texto clasificación: CSV/JSONL con texto y etiqueta; conservar IDs de documento antes de tokenizar.
- Texto causal: TXT por documento o JSONL; separar documentos, tokenizar y desplazar objetivos dentro del split. Tokenizador inicial por caracteres con vocabulario entrenado sólo en train y tokens especiales reservados; BPE como ampliación del adapter, no dependencia necesaria para completar C.

## 6. Preprocesamiento sin fuga de datos

Orden obligatorio: inspeccionar esquema → definir partición por IDs/grupos/tiempo → ajustar transformaciones sólo en train → transformar cada split con el mismo estado → congelar revisión.

Pipeline persistido, independiente del grafo de capas:

| Modalidad | Preparación inicial | Estado que acompaña al checkpoint |
| --- | --- | --- |
| Tabular | Imputación mediana/moda, one-hot y escalado opcional | Columnas/orden, vocabulario, imputación, medias/escalas |
| Imágenes | Canales, resize o letterbox, rango, normalización | Resolución, método, canales, constantes o estadísticas train |
| Secuencias | Orden, ventanas, escalado, padding/máscara | Variables, T/P, frecuencia, política de huecos, escalas |
| Segmentación | Imagen + máscara, geometría conjunta, paleta | Mapeo de clases, ignore, tamaño y transformación inversa espacial |
| Texto | Tokenización, longitud, padding, delimitación | Vocabulario, especiales, reglas y hash de tokenizador |

Aumentaciones únicamente en train: flip/crop/color en clasificación cuando tenga sentido para la tarea; geometría pareada en segmentación; ruido para denoising. Cambiar labels de posición/ángulo en regresión visual exige transformar también el target: no habilitar augmentaciones que invaliden etiquetas.

Si se escala el target de regresión, invertir esa escala al presentar predicciones y métricas en unidades originales. El valor de loss puede permanecer en espacio normalizado, claramente etiquetado. Reajustar datos o transformaciones crea una revisión nueva, conservando los artefactos usados por corridas anteriores.

## 7. Particiones

Por defecto proponer 70/15/15, con números absolutos y semilla. Nunca crear splits vacíos silenciosamente. Con datos pequeños, mostrar por qué no se puede garantizar clase/grupo por split y permitir modificar configuración. Se puede usar train+validation sin test mediante opción explícita; se deshabilita la evaluación final independiente.

- Clasificación: estratificar cuando la distribución y grupos lo permiten; grupos tienen prioridad para impedir contaminación.
- Varias observaciones del mismo sujeto/serie/documento: partición por grupo.
- Pronóstico: partir por tiempo antes de extraer ventanas; A usa ventanas completamente contenidas en cada split. Evita compartir muestras de contexto/target a ambos lados del límite; mostrar las ventanas descartadas por los bordes.
- Imágenes aumentadas, parches y pares de segmentación: mantener derivados de un mismo original en su split.
- Datos con splits existentes: respetarlos; detectar duplicados/hashes y advertir coincidencias entre splits.
- Texto: separar documentos antes de fragmentar/tokenizar; no distribuir fragmentos solapados aleatoriamente.

Guardar IDs/índices exactos, estrategia, semilla y fingerprint; un porcentaje no basta para reproducir la partición. Validation decide hiperparámetros y mejor checkpoint. Test no se usa para ajustar normalización ni entrenar.

## 8. Generadores sintéticos por caso

Todos pasan por el mismo contrato y validaciones que una importación. Guardan `generator_id`, versión, seed y parámetros. Se podrá exportar la fuente en el formato que el importador admite.

| Caso | Generador y etiqueta correcta | Controles | Preview |
| --- | --- | --- | --- |
| MLP clasificación | Blobs, dos lunas o XOR con ruido | N, F, K compatibles con generador, separación, ruido, balance | Dispersión de dos variables + tabla |
| MLP regresión | Función lineal/no lineal con ruido | N, F, Q, ruido, coeficientes | Relación entrada/target |
| CNN clasificación | Formas geométricas con variación de posición/fondo | N, resolución, gris/RGB, K, ruido | Galería por clase |
| CNN regresión | Imagen de objeto con área o posición como target | N, resolución, tarea área/centro, ruido | Imagen con objetivo anotado |
| LSTM clasificación | Ondas seno/cuadrada/diente de sierra | N, T, F, clases, frecuencia, amplitud, ruido | Series por clase |
| LSTM regresión | Señal con amplitud/frecuencia objetivo por secuencia | N, T, F, Q compatible, ruido | Serie y valores reales |
| LSTM pronóstico | Señal con tendencia + estacionalidad + ruido | Series, longitud, T, P, F/Q, ruido | Contexto y futuro real |
| U-Net binaria | Objetos sobre fondo y máscara exacta | N, H/W, objetos, ruido | Overlay binario |
| U-Net multiclase | Círculos/cuadrados/triángulos + fondo | N, H/W, K, tamaño, solapamiento con prioridad fija | Overlay y paleta |
| RNN/GRU/CNN1D | Reutiliza generadores de secuencias | Mismos contratos | Series |
| ResNet | Reutiliza generadores de imagen | Mismos contratos | Galería |
| Autoencoder/VAE | Vectores de baja dimensión latente o formas | N, ruido, dimensión/resolución | Original y distribución |
| Denoising | Formas limpias + ruido independiente | Tipo/nivel de ruido, seed | Par ruidosa/limpia |
| Transformer encoder | Frases de gramática pequeña con clases | N, vocabulario/longitud, balance | Texto y etiquetas |
| Transformer causal | Corpus de gramática procedural local | Documentos, longitud, vocabulario, seed | Texto y tokens |
| Multietiqueta | Varias formas presentes por imagen o atributos tabulares | K, coocurrencia, N, ruido | Etiquetas activas |

La generación multiclase exige suficientes ejemplos por clase; los generadores no codifican accidentalmente la etiqueta en el nombre de archivo o ID usado como feature. Para demostrar aprendizaje deben producir una relación reproducible no trivial entre entrada y target. Los corpus pequeños no prometen chat de propósito general.

## 9. Manifiesto de datos

Campos obligatorios propuestos: `schema_version`, `dataset_id`, `revision`, `task_id`, `modality`, fuentes y hashes, `sample_ids`, esquema de entrada/target, clases/columnas/unidades, particiones, pipeline y estado aprendido, generador si aplica, transformaciones de inferencia y registro de validación.

Los archivos grandes se leen bajo demanda con índices; no convertir automáticamente todo dataset a un tensor residente en RAM. Las fuentes externas pueden enlazarse o copiarse al proyecto con una elección visible. Si cambian o desaparecen, solicitar relocalización y revalidación antes de reutilizarlas. Exportación portátil incluye copias sólo cuando se elige esa opción.

## 10. Errores que deben tener mensajes específicos

Faltantes no tratados; objetivos vacíos; clase desconocida; columnas desordenadas; imagen corrupta; máscara sin par; resolución incompatible; ID de secuencia duplicado; tiempo no monótono; dataset demasiado pequeño; split con una sola clase cuando la tarea requiere más; tokenizador incompatible; campos no finitos; source modificada desde el snapshot. Cada error incluye ubicación, causa y corrección sugerida.
