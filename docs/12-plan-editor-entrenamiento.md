# Plan de mejoras del constructor y entrenamiento

2026-11-07. Implementar todos los puntos. Durante el trabajo el usuario pidió
migrar de GPT-6.1 Terra Medium a GPT-6 Luna con razonamiento VERY HIGH para cuidar
el uso. Conservar navegación, estética y el lienzo libre.

## Estado y restricciones

Workspace: /home/randy/Python/ModelBuilder. No hay AGENTS.md en el repositorio.
Hay cambios locales de Kaggle: backend, frontend, traducciones, pruebas y guía.
Preservarlos íntegramente. Baseline: 134 pruebas Python, build y prueba de navegador
Kaggle aprobados. No hacer commit/push ni borrar artefactos del usuario durante esta
implementación. Documentar dificultades y pruebas efectivamente ejecutadas al final.
Usar React Flow y lucide-react existentes. Traducir nuevas etiquetas ES/EN.

## 1. Sustituir cables al conectar una entrada ocupada

App.tsx: revisar connect e isValidConnection conjuntamente, porque rechazar una
entrada ocupada impide ejecutar onConnect.

- Conectar a una entrada ocupada reemplaza solo el cable de ese puerto.
- Entrada simple: normalizar handles legacy null/undefined/target.
- Add/Concat: conservar los otros input_N. Conservar sourceHandle p2/p3/map_N.
- Validar ciclos, puertos, dirección y autorreferencia contra el grafo candidato
  después de retirar la conexión anterior. Fallo deja intacto el cable anterior.
- Mantener fan-out, evitar duplicados, aplicar reglas también al reconectar un cable.
- Invalidar formas/validación al cambiar semántica, sin perder el orden de Concat.

Aceptación: reemplazo simple sin borrar wire; sustituir input_1 conserva input_0;
ciclo rechazado conserva arista anterior.

## 2. Inserción en el centro visible

Builder.addBlock usa coordenadas aleatorias fijas. Obtener boundingClientRect del
canvas real, convertir su centro con flow.screenToFlowPosition y compensar tamaño
del nodo. Aplicar pan y zoom, sin mover el viewport. Pequeño offset entre inserciones
consecutivas. Drag/drop mantiene la posición del puntero. Recetas usan igual criterio.
Fallback predecible si flow aún no existe. Probar pan/zoom lejos del origen.

## 3. Cruces en pestañas de modelos y entrenamiento

- Cruces accesibles, stopPropagation; evitar botones HTML anidados.
- Cerrar inactiva mantiene activa; cerrar activa elige vecina.
- Conservar una pestaña utilizable: bloquear la última con tooltip o crear borrador.
- Impedir cerrar modelo/corrida actualmente entrenando, también pausada. Poder
  cerrar las demás. La cancelación es una acción distinta.
- Cerrar retira pestaña/definición de UI, sin borrar checkpoints/datasets/bundles.
- Preservar snapshots históricos al cerrar el modelo asociado. Guardar/reabrir
  no resucita pestañas cerradas ni pierde el modelo activo.

## 4. Detener entrenamiento

Ya existen training.control y cancelación cooperativa backend, pero Cancelar está
en un footer poco evidente. Usar botón destacado Detener entrenamiento, icono Square
y aspecto de parada. Visible durante running/paused sin depender de scroll.
Debe controlar la corrida realmente en ejecución aunque se seleccione otra pestaña.
Estado Deteniendo hasta confirmación backend; evitar solicitudes repetidas y mostrar
errores con posibilidad de reintentar. Conservar pausa/reanudar, estado cancelado y
artefactos parciales previstos. Probar detener mientras está pausado. No matar otros
procesos ni fingir finalización inmediata.

## 5. Métrica de entrenamiento y validación

Hallazgo: metric actual viene de _evaluate(val_loader), es validación. El usuario
cree que es train: corregir las etiquetas y añadir datos reales de entrenamiento.

- Añadir trainMetric/valMetric al historial, eventos, checkpoints y resultados.
  Mantener metric legacy como validación para compatibilidad.
- Calcular train desde predicciones/targets reales. Preferir agregación de batches
  existentes. Si se usa pasada adicional, preservar modos, BatchNorm, RNG/resume y
  documentar su coste. No promediar incorrectamente métricas no lineales por batch.
- Gráfica inferior: dos series Entrenamiento/Validación, con accuracy cuando aplica
  y la métrica propia (Dice/MSE/etc) en otras tareas.
- Históricos sin train: huecos, nunca ceros inventados ni curva de val duplicada.
- Best checkpoint y early stopping siguen usando validación. Reanudación compatible.

Aceptación: fixture con curvas distintas, entrenamiento CPU con ambas finitas,
persistencia/resume y render legacy correctos.

## 6. Iconos por operación

ModelNode.tsx usa Box para casi todo. Crear mapa compartido tipo/familia a iconos
lucide para convolución, pooling, normalización, activación, lineal, forma,
input/output, Add/Concat y HF. Usarlo en tarjetas y biblioteca, con fallback.
Convolución y pooling deben distinguirse visualmente; conservar tarjetas/puertos,
legibilidad y estética. No depender solo del color.

## 7. Duplicar selección

- Ctrl+D/Cmd+D y botón Duplicar selección en toolbar con tooltip del atajo.
- Copiar todos los seleccionados y solo sus conexiones internas; IDs nuevos,
  pequeño offset visible y copias seleccionadas. Propiedades independientes.
- Limpiar shapes/errors/callbacks runtime. Input esencial no se duplica; outputs
  duplicados reciben nombres únicos. No copiar aristas a nodos externos.
- No interceptar teclado mientras se escribe en inputs/textarea/select/contenteditable.
- No implementar Ctrl+arrastrar ahora: interfiere con selección/navegación y es
  menos descubrible. Botón y atajo cubren la necesidad.

## 8. Grupos visuales

Botones Agrupar, Desagrupar, nombre editable y duplicación de grupo completo.
Mover un grupo mueve sus miembros y mantiene conexiones. Persistir nombre,
pertenencia y posiciones. Desagrupar mantiene IDs, pesos y posiciones absolutas.

Preferir metadata/contorno visual derivado; si se usan nodos group de React Flow,
filtrarlos rigurosamente del grafo ejecutable. Backend y semantic_hash nunca deben
recibir operadores group desconocidos. Revisiones, entrenamiento y extractor de
backbones deben seguir válidos. Agrupar/mover/renombrar no cambia semántica ni hash;
duplicar sí. Evitar grupos anidados si complican la solución, con mensaje claro.
Duplicar grupo remapea IDs de grupo/miembros y solo cables internos.

## 9. Recetas CNN

Biblioteca con Capas y sección Bloques CNN claramente distintas. Incluir:
Conv→ReLU→Pool; 2×(Conv→ReLU)→Pool; 3×(Conv→ReLU)→Pool;
residual de dos convoluciones con skip y Add en puertos distintos.
Insertar subgrafo real con IDs nuevos, capas editables y, si es útil, grupo visual.
Residual preserva formas: stride1/padding adecuado, canales compatibles o proyección
explícita para un cambio. No declarar válido uno incompatible. Entrada/salida claras.
Insertar en centro visible; no reemplazar el modelo completo ni conectar vecinos
inesperadamente. Conservar congelamiento por capa y puertos nombrados.

## Verificación y entrega

Pruebas significativas de reemplazo por puerto/ciclo, duplicación, grupos y su
serialización, recetas, métricas train/val, cancelación y resume. Ejecutar pytest
completo, npm run build, npm run test:cnn-reuse y npm run test:kaggle-ui.
Navegador con interacción real: reemplazar cable, insertar tras pan/zoom, duplicar
selección, agrupar/mover/desagrupar, cerrar pestañas, detener y ambas series.
Indicar cuándo se usan fixtures de backend/eventos. Capturas a 1280×800 y escritorio,
revisadas visualmente (toolbar, pestañas y parada visibles). Tauri está bloqueado
por dependencias GTK/GIO faltantes del entorno; no instalar dependencias de sistema.

## Registro de ejecución

### 2026-11-07 — Implementación y verificación completadas

- **1. Cables:** `frontend/src/builder-graph.ts` construye la lista candidata antes
  de validar, conserva fan-out y handles fuente, normaliza `targetHandle` heredados
  de Add/Concat siguiendo el orden serializado y preserva los demás `input_N`.
  Tanto conectar como reconectar pasan por la misma validación de ciclos, sentido,
  puertos y autorreferencia; App sólo aplica el grafo si es válido.
- **2. Centro visible:** `Builder` convierte el centro del rectángulo real del lienzo
  con `screenToFlowPosition`, compensa el tamaño del nodo y aplica un offset pequeño
  a inserciones sucesivas. Recetas usan ese centro y drop conserva el puntero.
- **3. Pestañas:** cierres accesibles, sin botones anidados, con selección vecina,
  último tab protegido y bloqueo para modelo/corrida en running o paused. Al cerrar
  un modelo se conserva el snapshot usado por sus corridas históricas.
- **4. Detener:** botón rojo persistente en la vista de entrenamiento, disponible
  mientras corre o está pausado. Detiene la corrida real aun con otra pestaña activa,
  cubre solicitud mientras arranca, reintento tras error y limpia estado/id al iniciar
  una nueva corrida, ante error o al completar. El proyecto de control es el capturado
  al inicio de la corrida; la confirmación de cancelación llega del backend.
- **5. Métricas:** historial, checkpoints, resultado y eventos ahora guardan
  `trainMetric`/`valMetric`; `metric` se mantiene como alias de validación. La métrica
  de entrenamiento se calcula con predicciones/targets de entrenamiento reales;
  Dice/IoU de segmentación usan estadísticas globales, mAP conserva el evaluador
  existente. La gráfica muestra ambas series y deja vacío el train histórico ausente.
- **6. Iconos:** mapa compartido por operación en `frontend/src/operation-icons.ts`,
  usado por tarjetas y biblioteca; convolución, pooling, normalización, activaciones,
  forma, entradas/salidas, merges y bloques HF quedan reconocibles sin depender sólo
  del color.
- **7. Duplicar:** botón y Ctrl/Cmd+D copian la selección, conexiones internas y
  grupos completos con IDs nuevos, offsets, propiedades independientes y outputs
  renombrados. No copian input esencial ni datos de ejecución; atajo ignorado cuando
  el foco está en un campo editable.
- **8. Grupos:** `groupId`/`groupName` sólo decoran los datos de canvas; mover actualiza
  posiciones, renombrar/duplicar/desagrupar conserva miembros y conexiones. El backend
  no conoce un operador group y `semantic_hash` ignora estos metadatos. Los grupos
  anidados se rechazan con mensaje explícito.
- **9. Recetas:** biblioteca separa Capas y Bloques CNN, con recetas de 1/2/3
  Conv→ReLU→Pool y residual 2×Conv→Add con canales de salida compatibles, stride 1
  y padding 1. Se insertan nodos editables aislados, sin cablear vecinos existentes.

**Archivos de editor/entrenamiento:** `frontend/src/App.tsx`, `builder-graph.ts`,
`operation-icons.ts`, `ModelNode.tsx`, `Chart.tsx`, `types.ts`, `styles.css`,
`i18n/locales/{en,es}/app.json`, `backend/modelbuilder/training.py`,
`tests/test_engine.py`, `tests/test_cnn_multi_output.py`, `scripts/test-builder-graph.mjs`,
`scripts/test-editor-training.mjs`, `package.json`. El trabajo Kaggle preexistente
(`README.md`, backend/data, KaggleImport, traducciones, su prueba y guía) permanece
preservado; sus cambios no fueron revertidos ni mezclados con la implementación.

**Evidencia y comandos ejecutados:** `pytest -q` (completo, pasó), `npm run build`
(pasó; Vite avisa que un chunk minificado supera 500 kB), `npm run
test:cnn-reuse` (pasó), `npm run test:kaggle-ui` (pasó), `npm run
test:builder-graph` (pasó) y `npm run test:editor-ui` (pasó). La prueba de grafo
cubre reemplazo simple, puertos legacy Add/Concat, ciclo sin mutación, duplicación,
grupos y recetas; Python cubre métricas reales y resume, cancelación pausada,
metadatos fuera del hash y formas del residual. La prueba UI usa un backend y
EventSource locales de fixture; manipula cables con drag real, inserta tras pan/zoom,
duplica/agrupa/mueve/renombra/desagrupa, cierra tabs y detiene una corrida pausada
con otra pestaña seleccionada.

**Capturas revisadas:** `capturas/editor-training-qa/builder-1280x800.png`,
`builder-desktop-1600x1000.png`, `training-two-series-1280x800.png` y
`training-stop-while-other-tab-selected-1600x1000.png`. Se revisaron toolbar,
pestañas, series de train/validation y parada visible en ambos tamaños. La corrida
legacy enseña sólo la serie de validación en su métrica; no se crean ceros históricos.

`git diff --check` pasó. No se hizo commit/push. No se intentó Tauri: siguen faltando
GTK/GIO de desarrollo en el entorno y no se instalaron dependencias del sistema.
No quedan criterios de aceptación de esta lista pendientes.
