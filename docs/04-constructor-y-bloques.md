# 04 · Constructor visual y catálogo de bloques

**Estado actualizado:** constructor y compilador de la entrega A implementados. Las piezas B/C de este catálogo se conservan como ampliaciones planificadas. Las entregas A/B/C se definen en [01](01-producto-y-alcance.md) y los tensores siguen la convención de [03](03-tareas-y-datos.md).

## 1. Qué representa el canvas

El documento del modelo es un grafo dirigido de operaciones, con puertos tipados y conexiones explícitas. El orden de ejecución depende de las conexiones, no de la posición visual. Mover una capa a la izquierda no cambia su función; reconectarla sí.

Hay tres niveles distintos:

- **Operación individual:** Linear, Conv2D, MaxPool, ReLU, Concat, etc. Cada instancia tiene propiedades y parámetros propios.
- **Bloque compuesto:** subgrafo editable con entradas/salidas expuestas, por ejemplo Conv–Norm–ReLU, bloque residual o etapa U-Net. Expandirlo permite seleccionar cada operación y editarla en el inspector.
- **Plantilla de arquitectura:** grafo inicial completo, pequeño y válido para una tarea. Insertarlo crea instancias normales; no oculta una red fija detrás de sliders.

Entrada y Salida representan el contrato de datos y de tarea. Preprocesamiento, loss y optimizador se configuran en sus etapas; pueden consultarse como contexto, pero no se arrastran como capas del forward ordinario. Las pérdidas auxiliares de VAE usan salidas nombradas del grafo y una receta de entrenamiento explícita.

## 2. Interacción de edición

| Acción | Comportamiento |
| --- | --- |
| Arrastrar desde biblioteca | Previsualizar el nodo y colocarlo al soltar; no ejecutar entrenamiento |
| Añadir por teclado/buscador | Crear en el centro visible o junto a la selección; enfocar inspector |
| Soltar bloque sobre una conexión | Proponer inserción origen→bloque→destino; aplicar atómicamente si puertos compatibles; si no, dejar nodo sin conectar y explicar |
| Conectar puertos | Resaltar destinos admitidos; error local si crea ciclo, conecta tipos incompatibles o ocupa una entrada única |
| Soltar conexión en vacío | Buscar bloques compatibles con el puerto de origen |
| Seleccionar nodo | Inspector derecho con propiedades, tensores, coste y ayuda |
| Seleccionar arista | Mostrar origen/destino, shape, dtype y acción desconectar |
| Duplicar/copiar | Nuevos IDs y nuevos pesos; copiar un subgrafo conserva conexiones internas, no conexiones externas inesperadas |
| Eliminar | Quitar nodos/aristas como una acción deshacible; no unir automáticamente extremos incompatibles |
| Bypass | Acción distinta de eliminar, disponible sólo si mantiene un contrato compatible; visible en el grafo y en su hash |
| Agrupar/desagrupar | Agrupación visual sin cambiar cálculo; «Crear bloque compuesto» define además interfaz de puertos |
| Expandir compuesto | Editar sus operaciones; mantener correspondencia de puertos externos |
| Autoorganizar | Cambiar sólo posiciones/rutas; U-Net propone encoder y decoder paralelos con skips claros |
| Deshacer/rehacer | Comandos transaccionales, incluyendo conexión, edición de propiedades y borrado múltiple |
| Guardar plantilla propia | Guardar subgrafo y versiones de bloques; validar al insertarlo en otro contrato de datos |

Pan, zoom, encuadrar selección, cuadrícula opcional, minimapa plegable, selección múltiple y búsqueda de nodo. La selección y el texto editado no se pierden al recibir una validación asíncrona. Arrastre de slider agrupa cambios en una sola operación al soltar; el teclado confirma al Enter o perder foco.

El historial de edición es local al borrador. Guardar revisión fija un snapshot; no convierte cada movimiento de ratón en una nueva arquitectura. Un grupo de repetición se materializa en N instancias de subgrafo: pesos independientes por defecto, nunca compartidos por una coincidencia de nombre.

## 3. Inspector contextual derecho

Siempre muestra nombre editable, tipo/versión y estado. Secciones:

1. **Propiedades:** controles básicos y avanzados, unidades, límites y defaults. Entrada de valores exactos además de sliders.
2. **Entradas y salidas:** puerto, shape, layout, dtype y significado de ejes. Dimensiones derivadas son de sólo lectura.
3. **Parámetros:** entrenables/congelados, memoria de pesos, inicialización y `requires_grad` cuando corresponda.
4. **Diagnóstico:** errores y advertencias con acción que enfoca el nodo/puerto afectado.
5. **Qué hace:** explicación breve y un ejemplo de transformación; mayor detalle bajo demanda.

Sin selección muestra resumen del modelo, contrato de entrada/salida y errores. Multiselección muestra propiedades compartidas; cambiar una propiedad incompatible no se aplica parcialmente. Ningún nodo usa un dropdown genérico enorme con todos los argumentos internos de PyTorch.

Propiedades comunes entrenables: inicialización `default`, Xavier o Kaiming cuando sea compatible; sesgo; congelar pesos. Cambiar inicialización afecta nuevas corridas, no reescribe un checkpoint existente. No ofrecer parámetros de dispositivo/dtype por capa en A: se administran por corrida.

## 4. Registro y contrato de bloques

Cada `BlockSpec` deberá declarar `type_id`, versión, título, categoría, familias/tareas admitidas, puertos, esquema de propiedades con defaults/rangos/condiciones, regla de shapes/dtypes, constructor backend, reglas de validación, estimador, ayuda y ejemplos de prueba.

Un `PortSpec` declara ID estable, dirección, tipo (`tensor`, `mask`, `lengths`, `pool_indices`, `recurrent_state`), cardinalidad y contrato. `Concat` tiene varias entradas ordenadas; se persiste ese orden. Un tensor de salida puede alimentar varias ramas. Un puerto tensor no recibe dos conexiones salvo que el bloque lo declare expresamente.

Las categorías filtran la biblioteca de la familia seleccionada. «Avanzado» revela piezas compatibles menos comunes; no desactiva validaciones. Operaciones compartidas se reutilizan entre familias. Un híbrido permitido se etiqueta «Personalizado basado en…» y deja de afirmar que es una U-Net/ResNet canónica si rompe su estructura, aunque siga cumpliendo el contrato de tarea.

## 5. Catálogo común de A

Los defaults concretos se derivan de una plantilla; los límites duros corresponden a semántica válida, no a recomendaciones de capacidad. Valores grandes generan estimación/advertencia de recursos. Todas las dimensiones deben ser enteras positivas salvo batch simbólico y ejes autorizados.

| Bloque / ID | Propiedades editables | Puertos, regla y uso |
| --- | --- | --- |
| Entrada · `input` | Nombre y referencia al contrato | Shape/dtype/columnas/clases derivados de Datos; una entrada principal en A |
| Salida · `output` | Nombre semántico y referencia de tarea | Valida shape final; no introduce una proyección oculta |
| Densa · `linear` | `out_features` (default 64), bias | Actúa sobre último eje; `in_features` derivado. La biblioteca exige adaptar layout si el último eje no representa features |
| Identidad · `identity` | Ninguna numérica | Conserva tensor; útil para rama residual y bypass explícito |
| ReLU / LeakyReLU | Tipo, pendiente negativa de Leaky (0.01) | Conservan shape; sin operaciones in-place expuestas |
| GELU / SiLU / ELU | Aproximación GELU, alpha ELU (1) | Activaciones alternativas con ayuda sobre saturación/coste |
| Tanh / Sigmoid / Softplus | Tipo; beta/threshold de Softplus en avanzado | Conservan shape; advertir dominio acotado de Tanh/Sigmoid |
| Softmax / LogSoftmax | Eje semántico obligatorio | No aceptarlas antes de la salida logits de CE/BCE; reservadas para atención u operaciones internas compatibles |
| Dropout | `p` (0.1), 0≤p<1 | Aleatorio en entrenamiento; inactivo en evaluación |
| Dropout espacial 2D | `p` (0.1) | Entrada imagen/features `[B,C,H,W]`; regulariza canales |
| BatchNorm1D / 2D | eps, momentum, affine, estadísticas acumuladas | 1D: `[B,F]` o `[B,C,L]`; 2D: `[B,C,H,W]`. No aplicar 1D directamente a `[B,T,F]` |
| LayerNorm | Ejes finales normalizados, eps, affine | Verificar tamaño de ejes; útil para secuencias, features y atención |
| GroupNorm | Grupos, eps, affine | C divisible por grupos; layout con canal en eje 1 |
| InstanceNorm2D | eps, momentum, affine, track stats | Features de imagen; mostrar diferencias frente a BatchNorm |
| Flatten | Eje inicial/final (default 1…último) | No colapsar B; calcula producto de dimensiones |
| Reshape / Unflatten | Dimensiones objetivo sin B; un único `-1` inferible | Conserva número de elementos por muestra; no reordena datos |
| Permute | Orden de ejes semánticos | Mantiene B primero; transforma layout de forma explícita |
| Squeeze / Unsqueeze | Eje no batch | Squeeze sólo dimensión 1; no elimina ejes por accidente |
| Concat | Eje no batch, orden de entradas | Iguales en todos los otros ejes; suma el eje seleccionado |
| Add / Multiply | Dos o más entradas ordenadas | Shapes idénticos en A; sin broadcasting implícito que oculte un error |
| Split | Eje no batch, tamaños de partes | Suma de partes igual al tamaño del eje; salidas nombradas |
| Slice | Eje no batch, inicio/fin/paso positivo | Rango no vacío y conocido; no introducir indexación Python libre |
| ReduceMean / ReduceMax | Ejes no batch, keepdim | Usa variante masked para tiempo con padding |
| Selección temporal | `last_valid`, `mean_masked`, `max_masked` | `[B,T,D]` + máscara/longitudes → `[B,D]`; no confundir padding con observación |

Constantes y operaciones elementales necesarias dentro de macro-bloques de C tendrán tipos/rangos explícitos. No se ofrece `eval`, `Lambda` ni un bloque de código arbitrario como atajo para un catálogo incompleto.

## 6. CNN 2D: todas las piezas de la familia A

| Bloque | Propiedades principales / avanzadas | Restricciones |
| --- | --- | --- |
| Conv2D | Filtros (32), kernel (3×3), stride (1), padding (1), dilation (1), groups (1), bias, padding mode | Canales de entrada derivados; grupos divide C de entrada y salida; dimensiones resultantes positivas |
| DepthwiseConv2D | Kernel, stride, padding, dilation, multiplicador de canales | Variante visible de Conv2D con groups=C; salida C×multiplicador |
| PointwiseConv2D | Canales salida, bias, stride | Conv2D de kernel 1; se muestra como operación real |
| SeparableConv2D | Filtros y parámetros depthwise | Compuesto expandible Depthwise→Pointwise, con sus dos operaciones |
| MaxPool2D | Kernel (2), stride (2), padding (0), dilation (1), ceil_mode, devolver índices | Puerto tensor y puerto opcional de índices; índices no son features |
| AvgPool2D | Kernel, stride, padding, ceil_mode, count_include_pad, divisor_override opcional | Exponer cómo se calcula el promedio en bordes |
| AdaptiveAvgPool2D | Tamaño objetivo H/W | Global Average Pool es preset 1×1; no elimina B/C |
| AdaptiveMaxPool2D | Tamaño objetivo, devolver índices | Misma convención de dimensiones, salida de índices opcional |
| Padding2D | Izquierda/derecha/arriba/abajo, modo, valor constante | Validar restricciones del modo reflect; operación explícita |
| Crop2D | Márgenes o tamaño central | Salida positiva; no deformar ni escalar implícitamente |
| Resize2D / ResizeLike | Tamaño o escala; nearest/bilinear; align_corners aplicable | ResizeLike recibe referencia sólo para H/W; registra transformación |
| ConvTranspose2D | Canales, kernel, stride, padding, output_padding, dilation, groups, bias | Dimensión resultante validada; no se describe como inversa exacta de Conv2D |
| MaxUnpool2D | Kernel, stride, padding, output_size de referencia | Requiere índices de un MaxPool compatible, mismos canales y procedencia |

Conv, pooling, normalización, activación, dropout, flatten/global pool y proyección final se pueden combinar libremente dentro de sus contratos. Una capa de pooling no se añade automáticamente tras cada convolución.

Las restricciones de grupos y padding se basan en [Conv2D de PyTorch](https://docs.pytorch.org/docs/main/generated/torch.nn.Conv2d.html). MaxPool permite gestionar índices y redondeo de salida; su validación debe contemplar las reglas de borde, no sólo una división entera simplificada. Véase [MaxPool2D](https://docs.pytorch.org/docs/main/generated/torch.nn.MaxPool2d.html). La fórmula de salida del upsampling aprendido y `output_padding` se validan contra [ConvTranspose2D](https://docs.pytorch.org/docs/main/generated/torch.nn.ConvTranspose2d.html).

## 7. LSTM, RNN, GRU y CNN 1D

| Bloque | Entrega | Propiedades / semántica |
| --- | --- | --- |
| LSTM | A | hidden_size (64), num_layers (1), bidirectional (false), bias, dropout entre capas (0), proj_size (0); input_size derivado, batch_first fijo true |
| Selector de estado LSTM | A | Última capa o capa concreta; dirección forward/backward/ambas concatenadas; selecciona `h`, no confundir con `c` |
| RNN | B | hidden_size, capas, no linealidad tanh/relu, dirección, dropout, bias |
| GRU | B | hidden_size, capas, dirección, dropout, bias; estado h, sin estado c |
| Conv1D / Depthwise / Pointwise | B | Análogos 1D de convolución; adaptador `[B,T,F]`↔`[B,F,T]` visible |
| MaxPool1D / AvgPool1D / AdaptivePool1D | B | Kernel/stride/padding o longitud objetivo; se recalculan longitudes y máscara |
| Dropout1D / InstanceNorm1D | B | Contrato channel-first; control de p o normalización |
| CausalPad1D | B | Padding sólo pasado; dependiente de kernel y dilatación |
| ConvTranspose1D / Resize1D | B | Recuperación de longitud para subgrafos compatibles; no habilita automáticamente tareas no declaradas |

LSTM expone `sequence`, `h_n`, `c_n`; acepta longitudes/máscara y estados iniciales opcionales tipados. Sin estados conectados usa ceros por batch; no conserva estado entre muestras independientes. Forma de sequence: `[B,T,directions×H_out]`; estados: `[layers×directions,B,H_out]` y c con hidden_size sin proyección. Para pronóstico, el preset usa dirección única; bidireccional se permite sólo al codificar una ventana enteramente observada, con advertencia si se pretende interpretación causal por timestep. No se conecta futuro a contexto.

`hidden_size>0`; proyección 0 o menor que hidden_size; dropout entre capas sólo tiene efecto con varias capas. El inspector no ofrecerá un dropout aparentemente activo en una LSTM de una capa. La selección final bidireccional concatena estados finales de ambas direcciones, no toma ingenuamente el último vector temporal. Estas diferencias se verifican contra [LSTM de PyTorch](https://docs.pytorch.org/docs/main/generated/torch.nn.LSTM.html).

La recurrencia temporal está encapsulada en el operador, por lo que el grafo exterior permanece acíclico. «Inspeccionar celda» representa las compuertas de entrada/olvido/salida, candidato, estado c y estado h; en GRU representa reset/update y candidato. Esa vista educativa muestra operaciones reales y parámetros derivados, pero A–C no permite rediseñar arbitrariamente la ecuación interna del operador PyTorch. Un editor de celdas recurrentes personalizadas requeriría un contrato adicional de bucle/estado.

## 8. U-Net: conexiones y bloques editables

Usa las piezas CNN2D; añade presets compuestos, sin operaciones escondidas:

| Compuesto | Operaciones expandibles | Propiedades de alto nivel |
| --- | --- | --- |
| DoubleConv | Conv→Norm opcional→Activación→Conv→Norm opcional→Activación | Canales, kernels, normalización, activación |
| EncoderStage | DoubleConv; rama de skip antes del descenso; MaxPool o Conv stride 2 | Canales, método de descenso |
| Bottleneck | DoubleConv y dropout opcional | Canales, p |
| DecoderStage | ConvTranspose o Resize+Conv→Concat con skip→DoubleConv | Canales, upsampling, estrategia explícita de alineación |
| SegmentationHead | Conv2D 1×1→Salida logits | 1 canal para binaria, K para multiclase, derivados de tarea |

Cada etapa tiene una salida de features y, donde procede, un puerto skip. Las conexiones skip se dibujan de encoder a decoder correspondiente; conectar niveles diferentes requiere resolver tamaños explícitamente. Concat concatena canales, no suma. Ofrecer Add es válido para una variante, pero cambia el modelo y su etiqueta de plantilla.

Resoluciones impares: error accionable con propuesta de Crop/Pad/ResizeLike; nada se recorta silenciosamente. Elegir «Insertar alineación» añade el nodo al grafo y al snapshot. Alternativa en Datos: política de padding a múltiplo de 2^profundidad y restauración a tamaño original en inferencia. Fondo y máscaras ignoradas se resuelven en el contrato de datos/loss, no como canales mágicos.

## 9. ResNet y autoencoders · B

**ResNet:** bloque básico expandible Conv3×3→BN→ReLU→Conv3×3→BN; rama identidad o proyección Conv1×1→BN; Add→ReLU. Si cambia resolución o canales, la proyección debe ser explícita. Bottleneck: Conv1×1→BN→ReLU→Conv3×3→BN→ReLU→Conv1×1→BN, más rama y suma. Inspector: canales internos/salida, stride, expansión, normalización y activación. Stem, etapas, global pooling y cabeza son piezas del canvas. Editar fuera de la receta estándar crea una variante personalizada.

**Autoencoder denso:** Input→capas Linear/activación→Latent→capas Linear/activación→Output de reconstrucción. `Latent` es una identidad nombrada con dimensión visible, no una compresión oculta.

**Autoencoder convolucional:** Conv/Pool→cuello de botella→ConvTranspose o Resize+Conv→Output. Flatten y Reshape son explícitos si se usa un vector latente. Puede usar MaxPool con índices y MaxUnpool: la UI muestra esa dependencia y advierte que el decoder ya no se puede ejecutar desde z solo sin índices. «Espejar encoder» propone un subgrafo que se puede inspeccionar; no afirma invertir matemáticamente la convolución.

Salida lineal para variables no acotadas; Sigmoid sólo cuando la receta reconstruye datos [0,1] y la loss espera ese dominio. El ruido de denoising pertenece al pipeline; la salida objetivo es limpia. Anomalía mediante error de reconstrucción es exploratoria; sin etiquetas y calibración no se reporta accuracy de detección.

## 10. VAE y Transformer · C

| Bloque | Propiedades | Contrato |
| --- | --- | --- |
| Cabeza latente VAE | Dimensión z; proyecciones independientes | Features→mu y logvar de igual shape |
| Reparameterize | Política train/eval; límite numérico logvar documentado | mu, logvar→z; train mu+exp(0.5 logvar)×epsilon; reconstrucción determinista usa mu |
| GaussianSample | Cantidad y seed de inferencia | Muestreo z desde prior normal; habilitado sólo para decoder compatible |
| Embedding | Vocabulario derivado, D, padding_idx, max_norm opcional | Tokens enteros `[B,T]`→`[B,T,D]`; IDs dentro de vocabulario |
| PositionalEncoding | Sinusoidal o aprendida, longitud máxima, dropout | Mantiene `[B,T,D]`; valida contexto |
| MultiHeadAttention | D, cabezas, dropout, bias; modo self/cross | q/k/v tipados, máscara de padding y atención; D divisible por cabezas |
| FeedForward | D_ff (default 4D), activación, dropout | Compuesto Linear→GELU/SiLU→Dropout→Linear |
| TransformerEncoderBlock | Pre/post norm, D, heads, D_ff, dropout | Atención no causal→residual→FFN→residual, con normas visibles |
| CausalDecoderBlock | Pre-norm inicial, D, heads, D_ff, dropout | Self-attention causal→residual→FFN→residual |
| SequencePooling / CLSSelect | Mean masked o token CLS | `[B,T,D]`→`[B,D]`; CLS requiere token introducido por pipeline |
| LanguageHead | V derivado, bias, compartir con embedding opcional | `[B,T,D]`→logits `[B,T,V]`; pesos compartidos sólo con shapes/IDs compatibles |
| MaskBuilder | Causal, padding o combinación | Máscaras con tipo y semántica explícitos, no features flotantes arbitrarias |

VAE expone salidas `reconstruction`, `mu`, `logvar`; la receta aplica reconstrucción+beta×KL. El control beta y sus schedules pertenecen a Entrenamiento. El grafo tiene un subgrafo decoder identificable para muestreo independiente. No ofrecer generación desde z si depende de skips/índices de una imagen de entrada.

La atención se puede expandir a proyecciones Q/K/V, partición de cabezas, QKᵀ, escalado, máscara, softmax, dropout, producto por V, unión de cabezas y proyección de salida. Esos operadores internos tendrán IDs tipados (`split_heads`, `merge_heads`, `matmul`, `scale`, `apply_mask`, etc.) y propiedades derivadas D/head_dim; no usar cambios de eje implícitos. El modo de inspección puede solicitar mapas de atención con límites de tamaño; no los almacena siempre.

El bloque MHA estándar y su expansión deben tener forward/backward equivalentes con mismos pesos y semilla apropiada. Se usará una convención interna de máscara `true=permitido`; el adapter la convierte a la semántica de cada API de PyTorch. Validar filas con todas las posiciones enmascaradas. Las restricciones de dimensiones y puertos parten de [MultiheadAttention](https://docs.pytorch.org/docs/main/generated/torch.nn.MultiheadAttention.html).

Texto causal requiere que cada posición vea sólo el pasado y su token actual; targets desplazados y padding ignorado se definen en Datos/Entrenamiento. Cross-attention queda en catálogo avanzado, pero traducción encoder–decoder no se anuncia como tarea soportada en C. La generación autoregresiva ocurre en el runner de inferencia, no mediante un ciclo dibujado de Output a Input.

## 11. Plantillas de referencia y shapes verificables

Las siguientes recetas son especificaciones de pruebas futuras; no resultados ya ejecutados. Normas/activaciones son nodos distintos aunque se abrevien en la cadena. B permanece variable.

| Plantilla | Grafo y dimensiones clave |
| --- | --- |
| MLP clasificación | Input `[B,F]`→Linear64→ReLU→Dropout0.1→Linear32→ReLU→LinearK→Output `[B,K]` |
| MLP regresión | Misma base, cabeza LinearQ→Output `[B,Q]`, sin activación restrictiva |
| CNN clasificación | `[B,3,32,32]`→Conv16 k3 p1→ReLU→MaxPool2→`[B,16,16,16]`→Conv32 k3 p1→ReLU→AdaptiveAvgPool1→Flatten→LinearK |
| CNN regresión | Misma base con LinearQ; no softmax |
| LSTM clasificación | `[B,T,F]`+lengths→LSTM64→selector h última capa→LinearK |
| LSTM regresión | Mismo encoder→LinearQ |
| LSTM pronóstico | `[B,T,F]`→LSTM64→h→Linear(P×Q)→Reshape(P,Q) |
| U-Net pequeña | `[B,3,64,64]`→DoubleConv16 (skip64)→Pool→DoubleConv32 (skip32)→Pool→DoubleConv64→Up32→Concat(skip32)=64 canales→DoubleConv32→Up16→Concat(skip64)=32 canales→DoubleConv16→Conv1×1(1 o K) |
| ResNet pequeña | Stem Conv16→dos bloques básicos16→bloque32 stride2 con proyección→bloque32→GlobalAvgPool→Flatten→LinearK/Q |
| AE denso | `[B,F]`→Linear64→ReLU→Linear8 (z)→Linear64→ReLU→LinearF |
| AE imagen | `[B,1,32,32]`→Conv16 stride2 k4 p1→ReLU→Conv32 stride2 k4 p1→ReLU→ConvTranspose16 k4 s2 p1→ReLU→ConvTranspose1 k4 s2 p1→Sigmoid |
| VAE denso | Encoder64→Linear mu8 y logvar8→Reparameterize→decoder64→LinearF; salidas reconstrucción/mu/logvar |
| Transformer clasificación | Tokens→Embedding64+posición→2 bloques encoder (4 heads, D_ff256)→masked mean→LinearK |
| Transformer causal | Tokens→Embedding64+posición→2 bloques causales (4 heads, D_ff256)→LayerNorm→LinearV |

Cambiar el input real actualiza dimensiones derivadas. U-Net de esta plantilla exige forma compatible o nodos explícitos de alineación. Las variantes binarias de clasificación general usan dos logits y CE por defecto; sólo segmentación binaria usa una salida logit por píxel en A.

## 12. Validación en tres niveles

1. **Interacción:** conexión de puerto, cardinalidad y ciclo; feedback inmediato en frontend.
2. **Semántica autoritativa Python:** versiones/propiedades, shapes/dtypes, rutas, máscaras, contrato de tarea y recursos estimados. Se ejecuta con revisión/ID de solicitud para descartar respuestas viejas.
3. **Prueba de batch:** construir modelo real, forward, loss y backward en un worker aislado con datos válidos; comprobar finitud, tamaño de salida y gradientes de parámetros entrenables alcanzables. Usa modelo efímero para no alterar estadísticas BatchNorm, RNG o pesos de una corrida.

Reglas obligatorias: sin ciclos exteriores; sin puertos requeridos vacíos; al menos una ruta Input→Output; sin dimensiones ≤0; sin mezcla batch/canales; número de clases/targets exacto; Concat compatible fuera de su eje; Add de shapes iguales; estados/máscaras consistentes; parámetros enteros/rangos válidos. Nodos huérfanos pueden existir en borrador, pero deben conectarse, eliminarse o marcarse «borrador excluido» antes de entrenar. La exclusión se muestra claramente.

Fórmula de Conv por eje: `out=floor((in+2p-d(k-1)-1)/s+1)`. ConvTranspose: `out=(in-1)s-2p+d(k-1)+output_padding+1`. Pooling con ceil y bordes utiliza las reglas específicas de su operador. Reshape conserva producto; Concat suma su eje; Linear sólo modifica el último eje. Estas reglas se contrastan con PyTorch en pruebas, incluyendo valores impares y casos límite.

Un error incluye código, node_id, port_id/property, esperado, recibido y corrección propuesta. Ejemplo: «Concat decoder_2: altura 31 frente a 32; añade Pad/Crop/ResizeLike o cambia el upsampling». Las reparaciones sugeridas muestran el cambio antes de aplicarlo y se pueden deshacer.

Estimación de memoria distingue pesos, gradientes, optimizador, activaciones, batch y margen runtime. Atención incluye el posible coste cuadrático en T. No se promete que la estimación evite toda falta de memoria; el dry-run y manejo de OOM siguen siendo necesarios.

## 13. Compilación y fidelidad del modelo

Orden topológico sobre IDs estables; módulos registrados con nombres estables; outputs de múltiples puertos almacenados con su identidad. Expansión de compuestos determinista. No usar `nn.Sequential` para un grafo con ramas ni generar código ejecutable a partir de texto del usuario.

Cada capa representada debe corresponder a un módulo u operación identificable en el motor. Agrupar no cambia el cálculo; expandir/colapsar tampoco. Pesos compartidos requieren un ID explícito y se cuentan una sola vez; en A la duplicación no comparte pesos. Reordenar entradas de Concat cambia el hash semántico aunque la shape siga siendo la misma.

El hash semántico incluye tipos/versiones, propiedades de cómputo, puertos, aristas ordenadas, contrato y pesos compartidos. Excluye posiciones, selección, zoom, notas y nombres de presentación. Al inspeccionar un checkpoint se carga su snapshot de grafo, no el borrador actual.

## 14. Base de UI elegida

React Flow se propone como editor de nodos personalizados y puertos; el motor de ejecución y validación es propio. Sus [nodos personalizados](https://reactflow.dev/learn/customization/custom-nodes) y [subflows](https://reactflow.dev/learn/layouting/sub-flows) cubren la representación, no resuelven automáticamente shapes, compilación ni historia de comandos. Guardar el canvas tampoco sustituye el manifiesto semántico; véase [Save and Restore](https://reactflow.dev/examples/interaction/save-and-restore).

Implementar undo/redo, clipboard, validación, compilador y macros como funciones del producto; no asumir que todos los ejemplos avanzados de la biblioteca son prestaciones gratuitas listas para incorporar. Registrar licencias y elegir versiones verificadas durante la fase 0.
