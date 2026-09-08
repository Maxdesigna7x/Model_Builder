# 02 · Diseño visual y pantallas

**Estado actualizado:** sistema visual y cinco pantallas principales implementados en React; revisados a 1440×900 en Chromium.

## 1. Lectura de las referencias

| Archivo conservado | Qué se toma | Aplicación concreta |
| --- | --- | --- |
| [3fbb…webp](../references/3fbb866cb67afaeb880bc0a83dc7b4c0.webp) | Sidebar sobria, filtros, tarjetas y acento cian | Shell del proyecto e inferencia |
| [750e…webp](../references/750e9bed91a795f7f4ea4ba061c5cd60.webp) | Vista general, tabla de modelos, densidad ordenada | Biblioteca de proyectos e historial de corridas |
| [7e24…webp](../references/7e2449fb57e03f17950efc7695e249e8.webp) | Métricas prominentes y gráficas amplias | Entrenamiento y detalle de una corrida |
| [original-1c8…webp](../references/original-1c8de71bff234bf33fb5b730e75b8cfc.webp) | Paleta de curvas, contraste entre series y matrices | Loss, métrica secundaria y matriz de confusión |
| [LLM.md](../references/LLM.md) | Recorrido guiado y explicaciones contextuales | Ayuda de tarea, bloque y resultados |

Se adopta el lenguaje visual, no la información de un dashboard de observabilidad remoto. No habrá cifras de uptime, SLA, alertas empresariales o inferencias diarias sin una función real correspondiente. El gráfico del constructor será plano y conectable; la estética isométrica del texto LLM queda como posible ilustración educativa, no como interacción principal.

## 2. Sistema visual propuesto

| Token | Valor inicial | Uso |
| --- | --- | --- |
| Fondo | `#0E1013` | Ventana y canvas |
| Superficie 1 | `#171A1F` | Sidebar, tarjetas |
| Superficie 2 | `#20252C` | Inspector, selección suave |
| Borde | `#303842` | Separación discreta |
| Texto principal / secundario | `#F1F5F9` / `#A8B3C2` | Lectura e información auxiliar |
| Acento | `#22D3EE` | Acción primaria, selección, train |
| Violeta | `#A78BFA` | Atención, normalización, comparación |
| Ámbar | `#FBBF24` | Validación, advertencias |
| Verde | `#4ADE80` | Éxito y estado listo |
| Rojo | `#FB7185` | Error o pérdida de conexión |

Son valores de partida que deberán verificarse por contraste en implementación. Texto normal con contraste mínimo 4.5:1; información y foco nunca sólo por color. Los trazos de train y validation también se distinguen por estilo y leyenda.

Tipografía sans serif de interfaz, preferentemente Inter empaquetada localmente con su licencia; fallback del sistema. Números tabulares para métricas y monospace para shapes/IDs. Texto de trabajo 14 px, metadatos 12–13 px, títulos 20–24 px. Escala de espaciado 4/8/12/16/24/32 px; radios 8–12 px; sombras ligeras y bordes definidos. Evitar tarjetas enormes, degradados dominantes y texto gris casi invisible.

Tema oscuro por defecto; tema claro como variante completa de los mismos tokens. Transiciones de 120–180 ms para paneles/selección, respetando «reducir movimiento». No animar continuamente todas las conexiones del grafo.

## 3. Shell y tamaños

Diseñar y verificar a 1280×720 y 1920×1080, sin exigir pantalla completa. A 1280×720: header de 52 px, sidebar de proyecto de 188 px, márgenes de contenido de 16 px. En el constructor, el inspector usa 288 px y la biblioteca de bloques se abre como cajón superpuesto de 240 px. El canvas conserva aproximadamente 770 px de ancho con inspector visible.

A partir de 1600 px, biblioteca de bloques de 220 px anclada junto al canvas e inspector de 304 px. Se pueden plegar sidebar, biblioteca e inspector; «Enfocar canvas» oculta los paneles laterales. Con alto reducido, los paneles hacen scroll independiente y los controles principales permanecen visibles.

No se persigue UI móvil inicialmente. Zoom del sistema y escala 125/150/200 % deben mantener accesibles navegación, inspector y controles; verificarlo por tamaño lógico real, no sólo por resolución física.

El header muestra breadcrumb, nombre/variante, guardado y estado del motor. La sidebar contiene los cinco pasos, historial de corridas y ajustes. El pie muestra dispositivo, memoria estimada y actividad real; nunca una barra de progreso ficticia de aprendizaje.

## 4. Inicio y creación de proyecto

Encabezado «Tus proyectos», acción primaria «Nuevo proyecto», secundaria «Abrir proyecto» y búsqueda. Tabla/lista de recientes con nombre, tarea, familia, fecha, última corrida y estado; alternativa de tarjetas sin duplicar datos.

Crear abre un formulario corto: nombre, carpeta destino y descripción opcional. Mostrar la ruta final. Colisión de carpeta o nombre inválido produce error local; no sobrescribir. Al confirmar se crea el proyecto vacío y se abre Modelo y tarea. Cancelar no crea carpetas incompletas.

Estados: sin proyectos con explicación y CTA; cargando con skeleton; ruta de reciente inexistente con «Localizar»/«Quitar de recientes»; proyecto de versión no compatible con mensaje y opción de conservar una copia. Quitar de recientes no elimina el proyecto.

## 5. Modelo y tarea

La misma pantalla permite filtrar por **modalidad** y **objetivo**; el centro muestra familias compatibles. Cada tarjeta incluye descripción de una línea, tareas disponibles, etiquetas de datos, dificultad orientativa y enlace a ver un ejemplo de entrada/salida.

Al seleccionar tarea y familia, el panel derecho muestra:

- «Qué recibe» y «Qué produce», con ejemplo visual.
- Formatos admitidos y columnas/etiquetas requeridas.
- Modalidad, tipo de clasificación o número de objetivos previsto.
- Loss y métrica recomendadas, con explicación breve.
- Datasets públicos compatibles disponibles y plantilla inicial de arquitectura.

La cantidad real de variables/clases o la resolución se confirma en Datos. Se elige aquí el contrato, no se inventan dimensiones del archivo que todavía no se ha cargado. Una combinación incompatible se oculta por filtro o se muestra deshabilitada con una razón concreta. CTA «Continuar a datos» exige una combinación válida.

## 6. Datos

Entrada inicial con un catálogo de **datasets compatibles de Hugging Face** y la tarjeta **Importar datos propios**. Cada dataset muestra repositorio, tamaño estimado, licencia y si ya está en el proyecto, en la caché física o pendiente de descarga. Elegir modelo no descarga datos automáticamente.

Importación: elegir fuente → mapear columnas/etiquetas o pares → previsualizar errores → confirmar preprocesamiento y splits. Mantener el patrón detectado visible. La lista de errores enlaza con archivo/fila y no se limita a «dataset inválido».

Tras descargar/importar: barra de resumen de muestras, variables/resolución, clases y tamaño; tabs «Vista previa», «Distribución», «Preparación» y «Particiones». Vista según modalidad: tabla, galería, imagen+máscara, serie temporal o texto tokenizado.

El control train/validation/test usa una barra de tres segmentos y dos tiradores, con porcentajes y cantidades editables por teclado. Arrastrar muestra una propuesta; confirmar recalcula los índices una sola vez. Para series se muestra una línea temporal con límites, no un reparto aleatorio. Splits aportados por la fuente se conservan por defecto.

La muestra de preview se mantiene hasta pulsar «Otra muestra». Revisión de origen, opciones del adaptador y distribución resultante son visibles. Si faltan muestras o clases para una partición válida, el CTA explica qué corregir. CTA «Abrir constructor» se activa al guardar un contrato de datos válido.

## 7. Constructor visual

Composición: navegación de proyecto a la izquierda; biblioteca de bloques; canvas dominante; inspector contextual derecho. Toolbar: deshacer/rehacer, buscar bloque, plantilla, autoorganizar, encuadrar, validar y guardar versión. Pie del canvas: diagnóstico, parámetros, dimensiones finales y memoria estimada.

```text
┌ Proyecto / Constructor ───── Guardado ─ Motor listo ┐
│ Pasos │ Biblioteca* │ Toolbar                      │
│       │            ├──────────────────┬───────────┤
│       │            │                  │ Inspector │
│       │            │  Canvas de       │ del bloque│
│       │            │  nodos y puertos │           │
│       │            │                  │           │
│       │            ├──────────────────┴───────────┤
│ GPU   │            │ Diagnóstico / memoria / CTA  │
└───────┴────────────┴───────────────────────────────┘
* Cajón superpuesto a 1280 px; fijo en ventanas amplias.
```

Al entrar se ofrece «Partir de una plantilla» o «Construir desde cero». Desde cero incluye Entrada y Salida derivadas del contrato, sin conexión ficticia. La plantilla coloca un modelo pequeño válido, con todos sus bloques editables.

Cada nodo presenta nombre, tipo, propiedad esencial, shape de entrada/salida y estado. Un Conv2D muestra filtros/kernel; MaxPool muestra factor espacial; LSTM muestra memoria y dirección. En zoom bajo se simplifica el contenido, sin desaparecer puertos importantes. Las conexiones muestran dimensiones al seleccionar o pasar el cursor.

Seleccionar un bloque abre inmediatamente su inspector derecho. El clic derecho abre acciones breves —duplicar, eliminar, agrupar, desconectar, ver ayuda—; las propiedades persistentes siguen en el inspector, como pidió el usuario. Las interacciones completas y catálogo están en [04](04-constructor-y-bloques.md).

Al editar una propiedad se actualiza el nodo inmediatamente y se solicita validación con debounce. Un error mantiene el valor en borrador y señala nodos afectados; no revierte silenciosamente ni recompila por cada píxel arrastrado. «Entrenar» explica errores pendientes y enlaza al nodo correspondiente.

## 8. Entrenamiento

Antes de iniciar: configuración compacta + resumen de dataset, grafo y recursos. Al iniciar: loss ocupa aproximadamente dos tercios del área de gráficas y métrica secundaria un tercio; en pantalla estrecha se apilan con loss primero. Se puede ampliar cualquiera y ocultar temporalmente la secundaria.

Fila superior de números: epoch/step, train loss, validation loss, métrica, tiempo y memoria. Los números ausentes se muestran como «—», no como cero. Panel plegable de configuración y tab inferior de logs/checkpoints. Selector de corrida siempre visible para no confundir historial con el trabajo en vivo.

Colores consistentes: train cian, validation ámbar; estilo discontinuo para validation. Tooltip sincronizado, escalas etiquetadas, zoom temporal y control «Seguir en vivo». La fase de validación aparece explícita, aunque no haya nuevos pasos de optimización.

Estados y acciones exactos: preparando, entrenando, validando, pausa solicitada, pausado, reanudando, cancelando, terminado, fallido o interrumpido. Mostrar la acción pendiente hasta la confirmación del motor. Detalle en [05](05-entrenamiento-e-inferencia.md).

## 9. Inferencia

Header con checkpoint, tarea y resumen del pipeline usado. Tres zonas: entrada, resultado y parámetros de inferencia. Tabs «Una muestra», «Lote» e «Historial» cuando estén implementados.

- Clasificación: entrada y barras de probabilidades/clase predicha; etiqueta real sólo si existe.
- Regresión: valor o vector con unidades; comparación y residual sólo si hay objetivo real.
- Pronóstico: serie observada seguida de horizonte predicho; no dibujar intervalos de confianza no calculados.
- Segmentación: imagen y overlay, opacidad, leyenda y máscara separada; detalle por clase.
- Autoencoder/VAE: original/reconstrucción/diferencia; VAE añade muestreo latente etiquetado.
- Transformer causal: prompt, tokens generados, parámetros y detener; las probabilidades adicionales sólo aparecen si el backend las calculó.

Entrada vacía, archivo incompatible, checkpoint no disponible y proceso ocupado tienen estados propios. Las pruebas con datos sintéticos se identifican. Resultados copiables/exportables sin mostrar información de desarrollo como parte del flujo normal.

## 10. Accesibilidad y criterios visuales

Todas las acciones drag-and-drop tendrán alternativa de teclado: añadir bloque desde buscador, seleccionar puerto origen/destino y conectar, mover por pasos y abrir inspector. Orden de tabulación estable; foco visible; Escape cierra menús; Delete actúa sólo fuera de campos de texto. Ctrl+Z/Ctrl+Shift+Z para canvas, respetando la edición de texto.

No depender exclusivamente de hover, color o gestos precisos. Áreas interactivas mínimas propuestas de 32 px, ampliadas para puertos. Listas largas virtualizadas, tooltips no invasivos y errores anunciables por tecnologías de asistencia. La aceptación visual exige revisar capturas reales de todas las pantallas y estados en ambos temas.
