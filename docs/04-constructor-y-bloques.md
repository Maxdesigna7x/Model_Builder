# 04 · Constructor visual y catálogo de bloques

**Estado actualizado:** constructor y compilador implementados para las tareas y bloques del catálogo activo. Las entregas A/B/C se definen en [01](01-producto-y-alcance.md) y los tensores siguen la convención de [03](03-tareas-y-datos.md). Los bloques que aparecen en este documento pero no en el código están marcados explícitamente como *ampliaciones futuras*.

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

## 5. Catálogo común implementado

Los defaults concretos se derivan de una plantilla; los límites duros corresponden a semántica válida. Valores grandes generan advertencia de recursos. Todas las dimensiones deben ser enteras positivas salvo batch simbólico y ejes autorizados.

| Bloque / ID | Propiedades editables | Puertos, regla y uso |
| --- | --- | --- |
| Entrada · `input` | Ninguna | Shape/dtype derivados del contrato de datos |
| Salida · `output` | Ninguna | Valida shape final; no introduce proyección oculta |
| Densa · `linear` | `out_features`, bias | Actúa sobre último eje; `in_features` derivado |
| Identidad · `identity` | Ninguna | Conserva tensor; útil para rama residual |
| ReLU / LeakyReLU / GELU / SiLU / Tanh / Sigmoid | Ninguna (tipo elegido al insertar) | Activaciones que conservan shape |
| Softmax | `dim` | Softmax sobre eje; no usar antes de CE |
| Dropout | `p` | Aleatorio en train; inactivo en eval |
| Dropout2D | `p` | Para `[B,C,H,W]` |
| BatchNorm1D / BatchNorm2D | `eps`, `momentum`, `affine` | 1D: `[B,F]`/`[B,C,L]`; 2D: `[B,C,H,W]` |
| LayerNorm | `normalized_shape`, `eps`, `affine` | Útil para secuencias y atención |
| Flatten | `start_dim`, `end_dim` | No colapsar batch |
| Reshape | `shape` | Dimensiones sin batch; un `-1` permitido |
| Permute | `dims` | Permutación de ejes |
| Concat | `dim` | Suma el eje seleccionado; resto iguales |
| Add | Ninguna | Suma elemento a elemento de entradas con shape idéntico |
| Resize2D | `size`, `mode` | `nearest` o `bilinear`; para `[B,C,H,W]` |

*Ampliaciones futuras:* `ELU`, `Softplus`, `LogSoftmax`, `GroupNorm`, `InstanceNorm`, `Unflatten`, `Squeeze`, `Unsqueeze`, `Split`, `Slice`, `ReduceMean/Max`, selección temporal con máscara, `Multiply`.

## 6. CNN 2D

| Bloque | Propiedades principales | Restricciones |
| --- | --- | --- |
| Conv2D | `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`, `bias` | Canales de entrada derivados; `groups` divide C de entrada y salida |
| MaxPool2D | `kernel_size`, `stride`, `padding` | Dimensiones resultantes positivas |
| AvgPool2D | `kernel_size`, `stride`, `padding` | Idem |
| AdaptiveAvgPool2D | `output_size` | Global Average Pool como preset 1×1 |
| ConvTranspose2D | `out_channels`, `kernel_size`, `stride`, `padding`, `output_padding`, `bias` | Dimensión resultante validada |

Conv, pooling, normalización, activación, dropout, flatten/global pool y proyección final se combinan libremente dentro de sus contratos.

*Ampliaciones futuras:* `DepthwiseConv2D`, `PointwiseConv2D`, `SeparableConv2D`, `AdaptiveMaxPool2D`, `Padding2D`, `Crop2D`, `ResizeLike`, `MaxUnpool2D`.

## 7. LSTM, RNN, GRU y CNN 1D

| Bloque | Propiedades / semántica |
| --- | --- |
| LSTM | `hidden_size`, `num_layers`, `bidirectional`, `bias`, `dropout`; `batch_first=true` |
| RNN | `hidden_size`, `num_layers`, `nonlinearity`, `bidirectional`, `bias` |
| GRU | `hidden_size`, `num_layers`, `bidirectional`, `bias` |
| Conv1D | `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`, `bias` |
| MaxPool1D / AvgPool1D / AdaptiveAvgPool1D | Análogos 1D del pooling 2D |
| CausalPad1D | Padding izquierdo para convolución causal |
| TemporalSelect | Toma el último paso temporal; `[B,T,D]` → `[B,D]` |

LSTM produce `sequence` de shape `[B,T,directions×H_out]` y se usa con `TemporalSelect` para clasificación/regresión. RNN y GRU existen como bloques atómicos pero no están asignados a ninguna tarea del catálogo todavía.

La recurrencia está encapsulada en el operador PyTorch; el grafo exterior permanece acíclico.

## 8. U-Net

La familia `unet` no añade bloques compuestos al catálogo; se construye a partir de las piezas CNN 2D existentes (`conv2d`, `maxpool2d`, `convtranspose2d`, `concat`, `add`, `identity`, activaciones, normalización). El preset inicial genera un encoder–decoder con skips editables.

El usuario puede modificar el grafo libremente: cambiar canales, agregar o quitar skips, sustituir `ConvTranspose2D` por `Resize2D` + `Conv2D`, etc. La validación verifica que las formas de `Concat` coincidan fuera del eje de canales.

*Ampliación futura:* compuestos `DoubleConv`, `EncoderStage`, `DecoderStage` y `SegmentationHead` expandibles.

## 9. Autoencoders y preset residual · B

**Preset residual dentro de CNN 2D:** el catálogo no expone `resnet` como familia independiente, pero el preset de CNN 2D llamado "CNN · Bloque residual" coloca dos convoluciones 3×3 con una rama `identity` + `Add`, editable como cualquier otro grafo.

**Autoencoder denso (`tabular.reconstruction`):** se construye con `linear`, activaciones y una capa intermedia que actúa como cuello de botella. La salida debe tener el mismo número de features que la entrada.

**Autoencoder convolucional (`image.reconstruction`):** encoder con `conv2d` + `maxpool2d`, decoder con `convtranspose2d` (o `resize2d` + `conv2d`) y salida con el mismo número de canales/resolución que la entrada.

*Ampliaciones futuras:* familia `resnet` independiente, autoencoder con pares ruido/limpio (`image.denoising`) y VAE con salidas múltiples.

## 10. Transformer y texto · C implementado; VAE pendiente

Los bloques de atención están implementados como operaciones monolíticas (no desplegables en Q/K/V individualmente):

| Bloque | Propiedades | Contrato |
| --- | --- | --- |
| `embedding` | `vocab_size`, `d_model`, `padding_idx` | Tokens `[B,T]` → embeddings `[B,T,D]` |
| `sequence_projection` | `d_model` | Proyección de secuencias numéricas a `[B,T,D]` |
| `positional_encoding` | `max_length`, `learned`, `dropout` | Suma posiciones a `[B,T,D]` |
| `transformer_encoder` | `d_model`, `heads`, `num_layers`, `dim_feedforward`, `dropout` | Encoder Transformer no causal |
| `causal_transformer` | `d_model`, `heads`, `num_layers`, `dim_feedforward`, `dropout` | Decoder causal con máscara automática |
| `sequence_pool` | `mode` (`mean` o `last`) | `[B,T,D]` → `[B,D]` |
| `patch_embedding` | `patch_size`, `d_model` | Imagen `[B,C,H,W]` → patches `[B,T,D]` para ViT |

**ViT:** se construye con `patch_embedding` + `positional_encoding` + `transformer_encoder` + `sequence_pool` + `linear`.

**VAE:** no implementado. Requiere salidas múltiples (`reconstruction`, `mu`, `logvar`) y pérdida compuesta (reconstrucción + β×KL), que el compilador actual no soporta.

*Ampliaciones futuras:* bloques desplegables de atención (`MultiHeadAttention`, `FeedForward`, `MaskBuilder`), generación autoregresiva con sampling (temperatura, top-k, top-p) y VAE completo.

## 11. Plantillas de referencia y shapes verificables

| Plantilla | Grafo y dimensiones clave |
| --- | --- |
| MLP clasificación | Input `[B,F]` → Linear64 → ReLU → Dropout0.1 → LinearK → Output `[B,K]` |
| MLP regresión | Misma base, cabeza Linear1 → Output `[B,1]` |
| CNN clasificación | `[B,C,H,W]` → dos bloques Conv/BatchNorm/ReLU/Pool → AdaptiveAvgPool1 → Flatten → LinearK |
| CNN regresión | Misma base con una tercera Conv64 y cabeza Linear1 |
| CNN 1D | `[B,T,F]` → Permute → Conv32/Pool → Conv64 → AdaptiveAvgPool4 → Flatten → cabeza de tarea |
| LSTM clasificación | `[B,T,F]` → LSTM64 → TemporalSelect → LinearK |
| LSTM regresión | Mismo encoder → Linear1 |
| LSTM pronóstico | `[B,T,F]` → LSTM64 → TemporalSelect → LinearP |
| U-Net binaria/multiclase | Encoder de dos escalas 16/32 → bottleneck64 → dos decoders con skips → Conv1×1(1 o K) |
| AE denso | `[B,F]` → Linear32 → ReLU → Linear8 → Linear32 → ReLU → LinearF |
| AE imagen | `[B,C,H,W]` → dos Conv stride2 → dos ConvTranspose → `[B,C,H,W]` |
| ViT clasificación | `[B,C,H,W]` → patches 7×7 en MNIST → Transformer D64 de 2 capas → SequencePool → LinearK |
| Transformer clasificación de texto | Tokens `[B,T]` → Embedding → PositionalEncoding → TransformerEncoder → SequencePool → LinearK |
| Transformer causal | Tokens `[B,T]` → Embedding → PositionalEncoding → CausalTransformer → LinearV |

Al cargar o cambiar el dataset, la cabeza, el vocabulario, la longitud posicional y el `Reshape` de pronóstico se sincronizan con su contrato real. Los proyectos existentes conservan sus grafos editados; la plantilla se crea al elegir por primera vez una combinación o añadir un modelo nuevo. Las variantes binarias de clasificación general usan dos logits y CE; la segmentación binaria usa un logit por píxel.

## 12. Validación

El backend ejecuta una validación semántica autoritativa (`graph.validate`) y un *dry-run* real: construye el modelo, hace forward, calcula loss y backward con un batch sintético, y verifica que las salidas y los gradientes sean finitos.

Reglas obligatorias verificadas: sin ciclos; sin puertos requeridos vacíos; al menos una ruta `input`→`output`; sin dimensiones ≤0; número de clases/targets exacto; `Concat` compatible fuera de su eje; `Add` de shapes iguales; parámetros enteros/rangos válidos. Nodos huérfanos pueden existir en borrador, pero deben conectarse o eliminarse antes de entrenar.

Fórmula de Conv por eje: `out = floor((in + 2p - d(k-1) - 1)/s + 1)`. ConvTranspose: `out = (in-1)s - 2p + d(k-1) + output_padding + 1`. `Reshape` conserva producto; `Concat` suma su eje; `Linear` sólo modifica el último eje.

Un error incluye `code`, `message` y, cuando es posible, el `node_id` afectado. Ejemplo: «Concat: shapes [B,C1,H,W] y [B,C2,H,W] no coinciden en el eje 1».

*Ampliaciones futuras:* estimación de memoria detallada, validación de máscaras/longitudes y reparaciones sugeridas aplicables con un clic.

## 13. Compilación y fidelidad del modelo

Orden topológico sobre IDs estables; módulos registrados con nombres estables; outputs de múltiples puertos almacenados con su identidad. Expansión de compuestos determinista. No usar `nn.Sequential` para un grafo con ramas ni generar código ejecutable a partir de texto del usuario.

Cada capa representada debe corresponder a un módulo u operación identificable en el motor. Agrupar no cambia el cálculo; expandir/colapsar tampoco. Pesos compartidos requieren un ID explícito y se cuentan una sola vez; en A la duplicación no comparte pesos. Reordenar entradas de Concat cambia el hash semántico aunque la shape siga siendo la misma.

El hash semántico incluye tipos/versiones, propiedades de cómputo, puertos, aristas ordenadas, contrato y pesos compartidos. Excluye posiciones, selección, zoom, notas y nombres de presentación. Al inspeccionar un checkpoint se carga su snapshot de grafo, no el borrador actual.

## 14. Base de UI elegida

React Flow se propone como editor de nodos personalizados y puertos; el motor de ejecución y validación es propio. Sus [nodos personalizados](https://reactflow.dev/learn/customization/custom-nodes) y [subflows](https://reactflow.dev/learn/layouting/sub-flows) cubren la representación, no resuelven automáticamente shapes, compilación ni historia de comandos. Guardar el canvas tampoco sustituye el manifiesto semántico; véase [Save and Restore](https://reactflow.dev/examples/interaction/save-and-restore).

Implementar undo/redo, clipboard, validación, compilador y macros como funciones del producto; no asumir que todos los ejemplos avanzados de la biblioteca son prestaciones gratuitas listas para incorporar. Registrar licencias y elegir versiones verificadas durante la fase 0.
