# Pendientes nuevos: revisión visual y flujo CNN

**Fecha:** 2026-11-07  
**Origen:** observaciones nuevas del usuario y capturas de `revisiones/` más la captura final adjunta.  
**Estado inicial:** todo lo de este documento queda registrado como pendiente antes de implementar esta ronda.

Este documento es un backlog nuevo y separado. No modifica ni combina los planes anteriores de CNN, Kaggle o editor/entrenamiento (`docs/09-plan-cierre-cnn.md`, `docs/11-kaggle.md` y `docs/12-plan-editor-entrenamiento.md`). Las capturas describen problemas observados; las instrucciones operativas están en el mensaje del usuario.

## Alcance y criterio común

Mejorar la experiencia CNN dentro de la app existente, conservando sus patrones visuales. Los avisos se muestran en una ventana modal centrada, con estilo de la app, foco accesible y acciones claras. Las formas de tensor se explican por eje y las visualizaciones deben distinguir operaciones por su efecto real en forma y contenido. No se debe cambiar ni borrar el trabajo previo que no sea necesario para estos puntos.

## A. Avisos y cambio de tarea

### A1 — Reemplazar confirmaciones nativas por diálogo de la app
**Problema:** al cambiar de tipo de tarea aparece el cuadro nativo del navegador (“Changing the task resets…”), ajeno al estilo y al idioma de la app. Otros `alert`/`confirm`/avisos nativos causarían la misma ruptura visual.
**Resultado esperado:** confirmación centrada, traducida, con estilo de la app y botones explícitos de cancelar y continuar. Inventariar y reemplazar los avisos/confirmaciones nativos usados por operaciones de la app; errores y warnings importantes también aparecen centrados, sin alterar notificaciones no modales que ya sean apropiadas.
**Aceptación:** cambiar tarea no invoca `window.confirm`; cancelar preserva la selección y continuar aplica el cambio. Todos los avisos modales se ven centrados y usan el tema/idioma activo.
**Estado:** COMPLETADO.

## B. Carga y resumen del dataset

### B1 — Selector inicial de origen de datos
**Problema:** Kaggle, catálogo Hugging Face e importación local aparecen simultáneamente, compitiendo por atención en una pantalla de preparación.
**Resultado esperado:** pantalla inicial simple con exactamente tres opciones con icono: Kaggle, datasets predefinidos de Hugging Face e importar dataset propio. Tras elegir, entrar al flujo propio de esa fuente; ofrecer volver/cambiar origen.
**Aceptación:** la pantalla inicial solo contiene esas tres opciones y cada opción abre una vista diferenciada sin mezclar los formularios de las demás.
**Estado:** COMPLETADO.

### B2 — Flujo Kaggle sin resolución anticipada
**Problema:** el formulario Kaggle pide resolución y canales antes de descargar/inspeccionar el dataset; la resolución fuente todavía se desconoce.
**Resultado esperado:** pedir únicamente URL/identificador de Kaggle y configuración realmente necesaria para localizar/descargarlo. Resolución y conversión de canales se configuran en procesamiento del dataset, después de inspeccionar dimensiones.
**Aceptación:** descargar un dataset Kaggle no requiere escoger resolución; la importación, detección de clases y unificación de splits siguen funcionando.
**Estado:** COMPLETADO.

### B3 — Estadísticas de resolución de imágenes
**Problema:** el resumen del dataset no dice cuánto miden las imágenes originales.
**Resultado esperado:** cuando contiene imágenes, mostrar resolución mínima, media y máxima (ancho × alto; indicar por separado si hay tamaños no cuadrados), junto al resumen/visualización del dataset. No confundir tamaño fuente con el tamaño que usará el modelo.
**Aceptación:** calcular y mostrar estadísticas correctas para dimensiones heterogéneas y cuadradas; indicar claramente imágenes sin dimensión disponible o conjuntos sin imágenes.
**Estado:** COMPLETADO.

### B4 — Resize predeterminado y explícito
**Problema:** hay que insertar a mano el reescalado para que las imágenes coincidan con la entrada CNN.
**Resultado esperado:** al inicializar el pipeline, incluir un paso de resize con valores derivados del tamaño de entrada actual. Debe ser visible, editable y sincronizado con la entrada del modelo; para tareas con máscaras/cajas, transformar también sus etiquetas geométricas de forma coherente.
**Aceptación:** nuevo dataset de imagen obtiene resize automáticamente; su tamaño coincide con la forma de entrada; editar el tamaño actualiza ambos o muestra una incompatibilidad clara, sin deformar silenciosamente etiquetas espaciales.
**Estado:** COMPLETADO.

## C. Constructor CNN y biblioteca de nodos

### C1 — Grupos como objetos manipulables
**Problema:** el marco de un grupo no actúa como un objeto de primer nivel y cuesta moverlo, seleccionarlo, eliminarlo o duplicarlo.
**Resultado esperado:** arrastrar el marco mueve sus nodos miembros juntos; seleccionar el grupo permite eliminarlo o duplicarlo con el mismo comportamiento de atajos/acciones que la selección. Los wires externos e internos deben quedar coherentes.
**Aceptación:** crear, arrastrar, duplicar y borrar un grupo no pierde conexiones ni deja nodos huérfanos; guardar/reabrir conserva su composición.
**Estado:** COMPLETADO.

### C2 — Validación automática del grafo
**Problema:** después de editar propiedades o conexiones, el usuario debe pulsar Validate para ver si la forma del modelo sirve.
**Resultado esperado:** ejecutar validación de fondo tras cambios semánticos importantes (nodos, propiedades, wires, entradas/salidas y orden), con debounce y evitando resultados obsoletos. Mover/zoom/seleccionar no dispara validaciones innecesarias. El botón Validate puede seguir como acción manual inmediata.
**Aceptación:** cambios semánticos reflejan errores/formas sin pulsar Validate; respuestas atrasadas no pisan el estado más nuevo; el movimiento puramente visual no lanza una validación.
**Estado:** COMPLETADO.

### C3 — Explicar y separar los ejes de tensor
**Problema:** la forma `[B, 16, 32, 32]` y propiedades de canales se muestran como texto compacto; no queda claro qué dimensión es canal y cuáles son resolución.
**Resultado esperado:** en nodos de imagen, separar visualmente **Canales/filtros** de **Resolución H × W** y de las demás propiedades. En el inspector, desglosar la forma: `B` = batch (se configura en Training), `C` = canales/filtros, `H` = alto, `W` = ancho. Respetar formatos distintos (vector, secuencia, máscara, etc.) en vez de etiquetar siempre como imagen.
**Aceptación:** una forma NCHW se presenta con los cuatro significados identificados; nodos que cambian canales o resolución muestran el cambio en su sección correspondiente y no mezclan canales con H×W.
**Estado:** COMPLETADO.

### C4 — Separar capas individuales de bloques y aclarar su salida
**Problema:** capas individuales y combinaciones/bloques comparten la biblioteca, sus nombres técnicos son densos y no se ve qué forma producen.
**Resultado esperado:** dos tabs en la biblioteca: capas individuales y bloques/combinaciones predefinidas. Usar nombres cortos y entendibles. En hover mostrar una vista rápida de la operación y la forma de salida, basada en la entrada/configuración cuando se pueda; marcar lo que depende de validación.
**Aceptación:** tabs separan ambas categorías; todos los bloques existentes conservan su configuración; el hover es legible, no tapa interacciones y distingue forma calculada de ejemplo ilustrativo.
**Estado:** COMPLETADO.

### C5 — Diagramas vectoriales explicativos por operación
**Problema:** max-pooling, average-pooling y global-average-pooling se distinguen por nombre/número pero no por lo que hacen espacialmente. Lo mismo puede ocurrir con otras capas.
**Resultado esperado:** revisar todos los tipos de nodo y decidir, operación por operación, qué microdiagrama SVG mejora su comprensión. Priorizar pooling global/local, convolución, resize/upsample, flatten, concat/suma residual, normalización y activaciones donde el dibujo aclare la transformación. Los dibujos deben ser vectoriales, pequeños, coherentes y acompañar la información numérica; no reemplazarla.
**Aceptación:** max/avg/global average tienen diagramas claramente distintos; una matriz de auditoría breve en este documento o guía indica nodos revisados y decisión; cada nodo para el que el dibujo sea útil tiene un microdiagrama y los demás no reciben decoración confusa.
**Estado:** COMPLETADO.

### C6 — Conexión tolerante a proximidad de puertos
**Problema:** al pasar el cursor, los handles se mueven/crecen y el punto exacto es difícil de clicar o soltar.
**Resultado esperado:** mantener un ancla visual estable y ampliar el área interactiva invisible alrededor de entrada/salida. Inicio del wire y drop se resuelven al puerto cercano dentro de un radio razonable, sin interceptar selección/drag del nodo.
**Aceptación:** conectar y sustituir wires con precisión sin tener que clicar el centro de un punto diminuto; el radio es consistente para puerto origen y destino, ratón y pantalla táctil.
**Estado:** COMPLETADO.

### C7 — Orientación y autoorden del grafo
**Problema:** los nodos actuales solo se organizan en flujo lateral y reordenarlos manualmente es lento.
**Resultado esperado:** control visible para alternar disposición: horizontal (flujo de izquierda a derecha) o vertical (flujo de arriba abajo). Reorganizar nodos conectados en orden topológico, respetando ramas, skips y grupos; preservar las conexiones y permitir retomar el movimiento manual.
**Aceptación:** alternar orientación produce un layout legible en grafo lineal, ramas paralelas y residual/U-Net; no altera semántica, edges ni grupos. Nodos desconectados no saltan a lugares inesperados.
**Estado:** COMPLETADO.

## D. Entrenamiento y gráficas

### D1 — Paleta y marcadores de las métricas
**Problema:** accuracy usa ahora naranja/amarillo como loss, y las curvas tienen puntos que saturan la vista.
**Resultado esperado:** loss mantiene naranja (train) y amarillo (validación); accuracy usa violeta (train) y azul (validación). Dibujar líneas sin marcadores circulares.
**Aceptación:** leyendas y curvas corresponden a esos colores en todas las corridas; no hay puntos visibles, incluidas gráficas cortas.
**Estado:** COMPLETADO.

### D2 — Mejor epoch y escalas estables
**Problema:** cuesta localizar el mejor resultado de validación y la escala de accuracy puede cambiar según los datos.
**Resultado esperado:** mantener accuracy en escala fija 0–1. En ambas gráficas, trazar una línea vertical en el epoch del mejor valor de validación: mínimo loss en la gráfica de loss y máximo accuracy (o métrica definida) en accuracy. Actualizar durante training cuando aparece un nuevo récord y distinguir el marcador de la curva.
**Aceptación:** ejes accuracy no cambian durante ni después de una corrida; marcadores coinciden con el mejor epoch y etiqueta/tooltip explican el valor y epoch, con historial parcial y completo.
**Estado:** COMPLETADO.

### D3 — Early stopping en la configuración de training
**Problema:** no hay control visible para detener automáticamente cuando validación deja de mejorar.
**Resultado esperado:** control Early stopping apagado por defecto; al activarlo, pedir paciencia (número de epochs sin mejora) y usar el criterio actual de mejor modelo/configuración. No detener runs con early stopping apagado.
**Aceptación:** por defecto no se altera la duración planificada; activado detiene tras la paciencia elegida y conserva/restaura el checkpoint según el criterio de mejor modelo existente.
**Estado:** COMPLETADO.

## E. Inferencia

### E1 — Resultado de clasificación compacto
**Problema:** la vista final repite estatus, encabezados, predicción y clase esperada; la imagen queda sobredimensionada y el dato principal se pierde.
**Resultado esperado:** una presentación compacta con el dato evaluado (imagen si aplica), clase predicha con su probabilidad y clase real cuando existe. El estado correcto/incorrecto puede derivarse de la comparación sin repetir el mensaje/etiqueta en varias secciones. Mantener controles para elegir muestra y ejecutar evaluación.
**Aceptación:** un resultado de imagen de clasificación muestra imagen, predicción + porcentaje y verdad de terreno una sola vez; sin clase real, omitir ese campo; otros tipos de tarea mantienen su información relevante.
**Estado:** COMPLETADO.

## Orden de trabajo ejecutado

1. **Agente de datos y avisos:** A1, B1–B4.
2. **Agente del constructor CNN:** C1–C7.
3. **Agente de training e inferencia:** D1–D3 y E1.

Los paquetes se implementaron de forma secuencial cuando compartían `frontend/src/App.tsx` y en paralelo cuando sus archivos no se solapaban. Los 16 IDs quedaron completados y sus límites se documentan por paquete. No se hizo commit ni push.

## Cierre de A1 y B1–B4 · 2026-11-07

**Estado del paquete:** A1, B1, B2, B3 y B4 completados.

- A1: se reemplazaron `alert`, `confirm` y `prompt` nativos por un diálogo centrado de la app, traducido y con foco inicial, cierre con Escape y navegación por teclado. Se mantuvieron los avisos breves existentes para resultados no modales.
- B1: la vista inicial presenta sólo Kaggle, catálogo de Hugging Face e importación local. Cada opción abre su flujo aislado y permite volver al selector. Kaggle sigue limitado a clasificación de imágenes por el contrato actual del backend; para otras tareas su vista explica esa limitación.
- B2: el formulario Kaggle conserva identificador, subcarpeta y archivo de etiquetas opcionales; ya no solicita resolución ni canales. La descarga, inspección de clases y posterior división siguen usando la integración existente.
- B3: `imageResolution` guarda mínimo, media y máximo por eje, el número de tamaños no cuadrados y cuántas dimensiones faltan. Se calcula con tamaños originales en Kaggle, Hugging Face y las importaciones locales; el panel distingue ese dato del tamaño `[C,H,W]` de entrada al modelo y muestra cuando el dataset no tiene imágenes o pertenece a un formato guardado antiguo.
- B4: los pipelines de imagen nuevos y heredados incluyen Resize editable y conversión de canales, derivados del contrato de entrada actual. Aplicar un nuevo tamaño actualiza `inputShape` y, con ello, la entrada del modelo. Máscaras usan vecino más cercano; blancos emparejados de traducción se redimensionan; cajas normalizadas mantienen sus coordenadas bajo un resize puro y conservan la transformación espacial existente.

**Archivos principales:** `frontend/src/App.tsx`, `frontend/src/KaggleImport.tsx`, `frontend/src/DataPipelineEditor.tsx`, `frontend/src/types.ts`, locales `en/es` de `app` y `dataPipeline`, `frontend/src/ui-polish.css`, `backend/modelbuilder/data.py`, `backend/modelbuilder/engine.py`, `backend/modelbuilder/kaggle_datasets.py` y `backend/modelbuilder/huggingface_datasets.py`.

**Verificación:** `npm run build`, `python -m py_compile` sobre los cuatro módulos backend tocados y `git diff --check` terminaron correctamente. No se agregaron ni ejecutaron pruebas.

**Limitación conocida:** los adaptadores de imagen actuales materializan una forma uniforme usando sus valores backend existentes antes de aplicar el pipeline editable. Las estadísticas sí conservan las dimensiones de origen, pero aumentar el tamaño después de importar no recupera detalle que el adaptador ya haya reducido. Para resolución posterior a la inspección desde los píxeles originales, habría que preservar y volver a leer los archivos fuente al aplicar el pipeline.

## Cierre de C1 y C7 · 2026-11-07

**Estado del paquete:** C1 y C7 completados.

- C1: el marco y sus bordes seleccionan el grupo completo; arrastrarlos mueve todos los miembros en el mismo delta. El encabezado muestra acciones de duplicar y eliminar cuando el grupo está seleccionado. Delete/Backspace y la acción Delete también reconocen la selección de grupo. Al eliminar, se quitan las conexiones incidentes; la entrada del grafo y la última salida se conservan y se desagrupan si quedaron protegidas.
- C1: duplicar conserva los wires internos y copia las entradas externas hacia el duplicado. Las salidas externas se replican sólo cuando el destino es un `add`/`concat`, en un puerto libre. Un destino de entrada única permanece conectado al grupo original para no cambiar la semántica existente. Los `groupId`/`groupName` de los duplicados se guardan con los nodos y el estado del proyecto existente.
- C7: los controles Horizontal y Vertical ordenan componentes conectados por rango topológico de camino más largo, con pasadas de barycenter para ordenar ramas. Los nodos de un mismo grupo quedan juntos dentro de cada rango cuando su topología lo permite. No se alteran nodos semánticos, edges ni metadatos de grupo; los bloques aislados mantienen su posición y se evita que el nuevo layout tape nodos aislados.
- C7: los componentes cíclicos se dejan en sus posiciones actuales porque no tienen orden topológico. El algoritmo prioriza legibilidad y conserva el orden de ramas con heurísticas; no garantiza minimizar cruces para todos los grafos. Los bloques de un grupo que están en distintos rangos siguen el orden topológico y el marco se ajusta a su extensión.

**Verificación:** `npm run build` terminó correctamente. `git diff --check` queda registrado al cierre de esta ronda. No se agregaron ni ejecutaron pruebas.

## Cierre de C5 y C6 · 2026-11-07

**Estado del paquete:** completados.

- C5: se añadió `OperationDiagram` con ilustraciones SVG pequeñas y distintas para pooling máximo, promedio local y promedio adaptativo/global. La revisión cubrió las 49 operaciones del catálogo del backend: flatten/reshape/permute, merges aditivos frente a concatenación, resize y convolución transpuesta, reducción/selección temporal y dropout por valor/canal recibieron esquemas cuando ayudan a explicar el efecto. Los tipos restantes mantienen iconos de familia; no se añadió decoración para operaciones cuyo efecto ya queda claro por sus propiedades y forma.
- C6: los puertos de entrada/salida usan hit areas invisibles de 36 × 36 px con el área sesgada hacia fuera del nodo, mientras el ancla visible permanece fija. Se cubren puertos regulares y multipuertos etiquetados. Las áreas cercanas pueden solaparse en el borde porque los puertos están separados 22 px; sus centros no se solapan.

**Archivos principales:** `frontend/src/OperationDiagram.tsx`, `frontend/src/operation-icons.ts`, `frontend/src/ModelNode.tsx`, `frontend/src/node-port-hit-area.css`.

**Verificación:** `npm run build` y `git diff --check` pasaron para los cambios revisados; no se ejecutaron pruebas de interacción.

## Cierre de D1 y D2 · 2026-11-07

**Estado del paquete:** completados.

- D1: loss usa siempre naranja para train y amarillo para validación; la métrica de tarea usa violeta y azul. Todas las curvas ocultan los símbolos circulares, también en historiales cortos.
- D2: accuracy, IoU, Dice y mAP usan eje fijo de 0 a 1. Loss marca el menor valor de validación; la gráfica de métrica marca el mejor valor de validación según el criterio conocido (maximizar métricas de puntuación y minimizar MSE/MAE), con línea vertical y etiqueta de valor/epoch que también se expone en el nombre accesible de la gráfica. La marca se actualiza con el historial recibido durante la corrida.

**Archivo principal:** `frontend/src/Chart.tsx`.

**Verificación:** `npm run build` y `git diff --check` pasaron; no se ejecutaron pruebas. Si un historial antiguo no contiene `trainMetric`, la curva de train de la métrica tendrá huecos; validación usa `valMetric` o `metric` como compatibilidad.

## Cierre de C2–C4 · 2026-11-07

**Estado del paquete:** C2, C3 y C4 completados.

- C2: la validación del grafo se programa 450 ms después de cambios semánticos mientras el constructor está abierto. La clave incluye orden de nodos y conexiones, tipos, etiquetas, propiedades, puertos y contrato de dataset; excluye posición, zoom, selección y metadatos visuales de grupo. Cada solicitud invalida las anteriores y una respuesta sólo actualiza el estado si corresponde a la clave vigente. La validación manual permanece inmediata y conserva su aviso; la automática actualiza errores, formas y estado con un aviso persistente dentro del constructor, sin emitir un toast por cada edición.
- C3: las tarjetas NCHW muestran por separado canales/filtros y resolución H × W. El inspector desglosa los ejes e indica que B es el batch configurado en Training. Los formatos no visuales distinguen vectores/features, secuencias `[T,F]`, tensores CNN1D después de `permute`, tokens/vocabulario, embeddings, horizontes de pronóstico y salidas espaciales como máscaras. La entrada de ViT conserva `[B,C,H,W]`; sus tokens se muestran como `[B,S,D]`.
- C4: la biblioteca tiene tabs Capas y Bloques y presets. Las capas siguen agregándose con sus valores predeterminados; las recetas CNN, presets incorporados y presets guardados se muestran separados y aplican sus grafos/configuración existentes. El hover explica la operación y distingue cálculos locales basados en una forma de entrada validada de los ejemplos que aún requieren validar el grafo.

**Archivos principales:** `frontend/src/App.tsx`, `frontend/src/ModelNode.tsx`, `frontend/src/tensor-shape.tsx`, `frontend/src/operation-preview.ts`, `frontend/src/ui-polish.css`, locales `en/es` de `app` y `catalog`.

**Verificación:** `npm run build` y `git diff --check` pasaron. No se agregaron ni ejecutaron pruebas.

**Límites:** los cálculos de hover cubren operaciones comunes con fórmulas locales y dimensiones concretas; las operaciones restantes muestran el ejemplo del catálogo y dejan explícita la necesidad de validar. CNN1D conserva el contrato de entrada de secuencia `[T,F]`; los ejes de `permute` reflejan el orden configurado. Una salida de preset mostrada desde el contrato del dataset es ilustrativa hasta que se valide ese grafo.

## Cierre de D3 · 2026-11-07

**Estado del punto:** completado.

- D3: se reemplazó el campo numérico siempre visible por un switch accesible de parada temprana, apagado para corridas nuevas. Al activarlo aparece paciencia entera de al menos una época; apagado envía `early_stopping_patience: 0`. La configuración positiva de una corrida guardada vuelve a activar el switch y conserva su valor.
- El backend mantiene el criterio existente para guardar `best.pt` y monitorea por separado la pérdida de validación para early stopping. Así la paciencia funciona también si el criterio de checkpoint está en “Ninguno (Última época)”. La mejor pérdida y épocas sin mejora se conservan en los checkpoints; corridas reanudadas desde checkpoints antiguos reconstruyen el contador a partir del historial disponible.

**Archivos principales:** `frontend/src/App.tsx`, locales `en/es` de `app`, `frontend/src/ui-polish.css`.

**Verificación:** `npm run build` y `git diff --check` pasaron. No se agregaron ni ejecutaron pruebas.

**Persistencia:** la configuración de paciencia se guarda junto con la corrida cuando empieza el entrenamiento y se restaura al volver a cargar el proyecto. Los cambios del formulario de una corrida que todavía no se inicia no se guardan como borrador independiente.

**Archivo adicional:** `backend/modelbuilder/training.py`.

**Límite de reanudación:** si un checkpoint antiguo no contiene un historial utilizable de pérdida de validación, early stopping reinicia su monitor desde ese punto porque no se puede reconstruir con fiabilidad usando el contador antiguo, que dependía de otro criterio.

## Cierre de E1 · 2026-11-07

**Estado del punto:** completado.

- `image.classification` muestra la imagen evaluada, la clase predicha y sólo la probabilidad de esa clase, asociada por nombre con `dataset.classes`. Muestra la clase real una vez cuando el backend la proporciona. Se quitaron para este resultado el estatus correcto/incorrecto, los encabezados repetidos y la lista completa de probabilidades.
- Los selectores de muestra/manual y el control para ejecutar inferencia permanecen disponibles. La disposición se compacta en una columna en pantallas estrechas.
- Otros tipos conservan su vista de resultados existente, incluidos segmentación, detección, regresión/reconstrucción, texto y entradas tabulares o de secuencia.

**Archivos principales:** `frontend/src/App.tsx`, `frontend/src/inference-layout.css`, locales `en/es` de `app`.

**Verificación:** `npm run build` y `git diff --check` pasaron. No se agregaron ni ejecutaron pruebas.
