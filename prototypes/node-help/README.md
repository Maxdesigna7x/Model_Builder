# Propuesta: ayuda visual animada para nodos

Galería autónoma en español para revisar microdiagramas de operaciones de red. Abrir `index.html` directamente en un navegador. No usa librerías, servidor, ni assets externos y no conecta con los tensores de un modelo.

## Qué está cubierto

Se tomó como inventario `backend/modelbuilder/catalog.py::BLOCK_TYPES`, comparándolo con `frontend/src/catalog.ts` y el alcance actual de `frontend/src/OperationDiagram.tsx`.

| Tema | Tipos cubiertos | Nivel de representación |
|---|---|---|
| Convolución espacial/temporal | `conv1d`, `conv2d` | Demostración principal: ventana animada, mapas y mezcla de canales. Los valores no se calculan. |
| Padding causal y varias escalas | `causalpad1d`, `multi_scale_cnn` | Esquema conceptual de causalidad y ramas paralelas con kernels de distinto tamaño. |
| Pooling local | `maxpool1d`, `maxpool2d`, `avgpool1d`, `avgpool2d` | Demostración principal del máximo frente a la media, con ventana local. |
| Pooling adaptativo global | `adaptiveavgpool1d`, `adaptiveavgpool2d` | Muestra una media por mapa/canal y la reducción espacial completa; `output_size > 1` se explica en el texto. |
| Upsampling | `resize2d`, `convtranspose2d` | Esquema conceptual; compara remuestreo con expansión aprendida. |
| Activaciones | `relu`, `leaky_relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `softmax` | Curva/redistribución esquemática. Softmax tiene explicación separada porque opera sobre un grupo. |
| Capas y regularización | `linear`, `batchnorm1d`, `batchnorm2d`, `layernorm`, `dropout`, `dropout2d` | Conexiones Dense y diagramas conceptuales de normalización y máscaras. |
| Ejes y forma | `flatten`, `reshape`, `permute` | Ejemplo visual sin movimiento de datos calculado. |
| Ramas | `add`, `concat` | Esquema conceptual de suma elemento a elemento y unión por un eje. |
| Secuencias | `rnn`, `gru`, `lstm`, `sequence_pool`, `temporal_select` | Estado recurrente y resumen/selección de pasos; compuertas y valores son esquemáticos. |
| Tokens y posición | `embedding`, `sequence_projection`, `positional_encoding`, `patch_embedding` | Flujo conceptual de IDs, variables y parches a vectores de tokens. |
| Atención | `transformer_encoder`, `causal_transformer` | Matrices conceptuales de atención abierta y con máscara causal; no son pesos reales. |
| Componentes preentrenados | `hf_image_processor`, `hf_vision_backbone`, `hf_token_processor`, `hf_text_backbone`, `hf_causal_lm` | Contratos de entrada/salida resumidos; la salida depende del modelo y procesador elegidos. |
| Límites del grafo | `input`, `identity`, `output` | Esquema de paso y límites; no se presentan como capas aprendidas. |

**Cobertura:** los 49 tipos publicados actualmente por el catálogo del backend aparecen en la galería. No hay tarjetas en blanco, pero las familias fuera de convolución y pooling son prototipos explicativos, no implementaciones numéricas ni representaciones completas de sus parámetros. Conv y pooling también usan valores ilustrativos: la ventana sí representa el comportamiento local, pero no reproduce pesos ni activaciones de un modelo real.

## Interacción y movimiento

- Botones superiores filtran Convolución, Pooling u Otras operaciones.
- El control de pausa detiene y reanuda las animaciones CSS.
- `prefers-reduced-motion: reduce` reduce movimiento sin quitar el contenido del dibujo.
- Cada SVG tiene un nombre accesible y cada operación aparece en una descripción legible.
- En pantallas estrechas, las tarjetas pasan a una columna.

## Diseño sugerido para integrarlo después

1. Mantener `OperationDiagram` como símbolo compacto del nodo. Añadir un botón visible de ayuda junto a él que abra una popover accesible al clic, con teclado y foco; no depender solo de hover. El botón debe tener `aria-label` localizado, `aria-haspopup` apropiado, soporte de Escape y cierre al perder contexto.
2. Crear una tabla de contenido por `blockType` (por ejemplo, `OperationHelpContent`) con título, explicación breve, nombres de propiedades relevantes y componente SVG. Reutilizar una misma familia para variantes 1D/2D, pero conservar su identificador exacto para la búsqueda.
3. Pasar las propiedades reales disponibles del nodo —por ejemplo `kernel_size`, `stride`, `padding`, `out_channels`, `output_size`, `axis` o el modo de pooling— al dibujo. Cuando la forma de entrada no se conozca, mostrar una forma simbólica en vez de inventar dimensiones.
4. Usar las descripciones existentes de `BLOCK_INFO` como base de texto y añadir las explicaciones visuales a las claves de `frontend/src/i18n/locales/es/catalog.json` y su contraparte inglesa. Este HTML trae textos solo para revisión, sin sincronización con i18n.
5. Mantener movimiento y estilos locales al panel, conservar el modo de pausa y la preferencia de movimiento reducido, y dar descripciones textuales equivalentes. La galería no debe controlar el grafo ni alterar su selección.

Antes de integrar, revisar con usuarios si la explicación de canal/espacio de Conv, el ejemplo numérico Max/Avg y la diferencia entre resumen global y pooling local son claros. La sección de atención, los bloques HF y las compuertas recurrentes son simplificaciones pedagógicas y conviene refinarlas con datos reales del contrato del nodo.

## Alcance de este prototipo

Los únicos archivos de la propuesta son `index.html` y `README.md` en este directorio. La galería no importa código del frontend ni modifica `OperationDiagram`, el catálogo, las traducciones o el flujo del constructor. No se ejecutaron pruebas porque se trata de un artefacto de diseño estático para revisión.
