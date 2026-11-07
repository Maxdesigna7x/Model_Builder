# Plan de cierre CNN de ModelBuilder

Fecha original: 2026-11-07. Estado de ejecución: **implementación parcial**.

Este documento permite que otro agente implemente la ampliación sin reconstruir
la intención de la conversación. Las casillas pendientes no significan que toda
esa funcionalidad falte: primero se debe comprobar qué parte ya funciona.

## 1. Encargo y resultado esperado

Modificar la aplicación existente para completar el uso de CNN de extremo a
extremo: datos, procesamiento, construcción de la red, entrenamiento, evaluación,
inferencia y reutilización de modelos propios o preentrenados.

**Conservar la interfaz actual y el constructor como diagrama libre de nodos.**
La modificación visual principal está en el lienzo del modelo: hacer evidentes
los tensores, las ramas, los puertos y los bloques internos. No sustituirlo por
un editor secuencial, una tabla, un asistente de pantallas nuevas o una vista de
volúmenes 3D. El resumen es un complemento del diagrama.

El usuario quiere poder dibujar, entre otras arquitecturas:

- CNN de clasificación o regresión.
- CNN de localización de una caja por imagen.
- Redes con conexiones residuales y concatenaciones.
- Detectores con dos cabezas paralelas y, mediante puertos, varias escalas.
- U-Net con skips entre encoder y decoder.
- Redes imagen→imagen, por ejemplo restauración o transformación supervisada.
- Modelos que reutilizan un encoder/backbone entrenado previamente.

No intentar cubrir todas las arquitecturas posibles. Completar los recorridos
concretos de la sección 15 y dejar extensiones mediante operadores y recetas
registrados. Mantener operativos los casos actuales de otras familias mediante
compatibilidad y regresión, sin rediseñarlos.

### Decisiones ya tomadas: no volver a preguntarlas

1. Se conserva React Flow, el estilo, temas, navegación y panel lateral actuales.
2. El lienzo permite distribuir nodos y conectar ramas libremente dentro de las
   restricciones matemáticas del cálculo.
3. No se exige la estructura Backbone→Neck→Cabezas: son agrupaciones opcionales.
4. Hay un solo modelo ejecutable; diagrama, inspector y resumen lo representan.
5. Se admiten varias salidas nombradas y puertos intermedios de backbones.
6. Arquitectura y checkpoint/inicialización son conceptos separados.
7. Reanudar, iniciar otra corrida con pesos y reutilizar un backbone son acciones
   diferentes.
8. Se mejora la app; no se implementa una app nueva ni otro prototipo aislado.

## 2. Antes de programar

- [ ] Leer las instrucciones `AGENTS.md` aplicables y este documento completo.
- [ ] Ejecutar `git status --short` y registrar los cambios que ya estaban presentes.
- [ ] No revertir, sobrescribir ni limpiar cambios ajenos. Al redactar este plan
      había numerosos archivos modificados y archivos nuevos sin seguimiento.
- [ ] Comprobar la implementación real antes de asumir que un TODO está actualizado.
      Algunos documentos antiguos describen como pendientes funciones que el código
      actual ya implementa.
- [ ] Obtener un resultado inicial de las pruebas y del build; separar fallos
      preexistentes de regresiones introducidas.
- [ ] Abrir el flujo actual y guardar capturas de Datos, Constructor, Entrenamiento
      e Inferencia para comparar que se conserva la interfaz.
- [ ] Crear un proyecto de prueba aislado. No usar proyectos/datasets del usuario
      como fixtures destructivos ni modificar los notebooks de referencia.

No basta con marcar casillas de otros planes. Cada fase de este documento necesita
código real, comprobación y un registro breve de evidencia en la sección 17.

## 3. Mapa de archivos y situación observada

Rutas relativas a `/home/randy/Python/ModelBuilder`.

| Archivo | Responsabilidad que debe revisarse |
| --- | --- |
| `frontend/src/App.tsx` | Flujo de pantallas, estado del canvas, validación, inspector, entrenamiento e inferencia |
| `frontend/src/ModelNode.tsx` | Apariencia y handles de cada nodo |
| `frontend/src/types.ts` | Tipos de tareas, modelos, puertos, datasets, corridas y resultados |
| `frontend/src/catalog.ts` | Biblioteca, plantillas, presets, `syncOutputContract` |
| `frontend/src/DataPipelineEditor.tsx` | Editor existente de transformaciones y su inspector |
| `frontend/src/bridge.ts` | Comunicación web/Tauri y eventos |
| `frontend/src/styles.css`, `ui-polish.css`, `inference-layout.css` | Estética existente; reutilizar antes de crear otro sistema visual |
| `frontend/src/i18n/locales/{es,en}/` | Textos de interfaz y errores |
| `backend/modelbuilder/models.py` | Operaciones, `GraphModel`, adaptadores de backbones y congelamiento |
| `backend/modelbuilder/catalog.py` | Catálogo y compatibilidad autoritativos |
| `backend/modelbuilder/data.py` | Importación, revisión de datos, procesamiento, previews y analíticas |
| `backend/modelbuilder/training.py` | Dataset de entrenamiento, recetas, pérdidas, métricas, checkpoints |
| `backend/modelbuilder/pretrained.py` | Catálogo de pesos fijados por revisión y caché |
| `backend/modelbuilder/engine.py` | Acciones del backend, validación real e inferencia |
| `backend/modelbuilder/manifest.py` | Hash semántico y comparabilidad |
| `backend/modelbuilder/storage.py` | Persistencia y migración |
| `src-tauri/src/lib.rs`, `vite.config.ts` | Puentes de escritorio y web si cambian contratos |
| `tests/test_engine.py` | Pruebas del motor y recorridos existentes |
| `tests/test_default_starter_models.py` | Contratos de las plantillas frontend con el backend |
| `tests/test_new_architectures.py`, `tests/test_pretrained_models.py` | Familias, backbones y pruebas sin red |

Observaciones del código al preparar este plan; verificar nuevamente:

- `GraphModel` ya ejecuta un DAG con ramas; no hace falta reemplazarlo por otro motor.
- `GraphModel` exige exactamente una entrada y una salida. `forward` devuelve
  actualmente un único tensor de salida.
- Las aristas tienen campos de React Flow para handles, pero el ejecutor obtiene
  principalmente IDs de nodos; faltan selección real de puerto y orden explícito.
- `ModelNode` muestra un shape, pero este se rellena después de `graph.validate`.
- Hay tipos de puertos y marcadores de compuesto. Comprobar si son funcionales:
  mostrar una lista de nombres de etapas no equivale a editar un subgrafo.
- `HuggingFaceVisionBackbone.forward` devuelve características resumidas mediante
  pooling. Eso sirve para clasificación, no para conectar directamente un decoder.
- Hay procesamiento por revisiones y aumentaciones train-only; ya existen
  transformaciones conjuntas para algunos casos de segmentación/reconstrucción.
- El catálogo de tareas no incluye detección ni localización de cajas explícita.
- Hay preentrenados, políticas de fine-tuning, checkpoint y reanudación. Deben
  ampliarse y verificarse, no darse por inexistentes ni por totalmente resueltos.
- `syncOutputContract` adapta cabezas actuales. Su comportamiento debe revisarse
  para que no sobrescriba silenciosamente varias cabezas o canales de una red propia.
- `semantic_graph` no representa todavía todos los contratos nuevos propuestos.

Se pueden extraer componentes/servicios pequeños de `App.tsx` para implementar
estas funciones. Evitar una reescritura general no relacionada.

## 4. Referencias locales que definen los casos

Leer estos archivos sin modificarlos:

1. `/home/randy/Python/Clases_DL/YOLO/prototipo_detector_keras/README.md`
2. `/home/randy/Python/Clases_DL/YOLO/prototipo_detector_keras/componentes.py`
3. `/home/randy/Python/Clases_DL/YOLO/Detector_Una_Escala_Desde_Cero_Keras.ipynb`
4. `/home/randy/Python/Clases_DL/YOLO/CNN_Localizacion_MSE_vs_IoU_Keras.ipynb`

El detector didáctico usa entrada 128×128×3, cuatro convoluciones stride 2 y mapa
8×8. Tiene ramas convolucionales independientes para `caja` y `clases`:

- Salida de cajas: 8×8×4.
- Salida de clases: 8×8×3.
- Una caja por celda y asignación por centro.
- Los targets de cajas incluyen un quinto valor de máscara. **Ese quinto valor
  no es una quinta salida de la red.**
- La pérdida de cajas se aplica a celdas positivas.
- El fondo se representa con las tres clases a cero, sin cuarta clase de fondo.
- Hay decodificación, umbral de confianza y NMS por clase.
- Los objetos cuyos centros colisionan en una celda exceden la capacidad de esta
  receta. Detectar y explicar ese caso; no descartar anotaciones silenciosamente.

El localizador produce una sola caja ordenada `xyxy` por imagen. No tratarlo como
el detector anterior ni como la regresión escalar actual.

Los notebooks son Keras y la app usa PyTorch. Se exige representar y ejecutar los
casos equivalentes en la app, **no importar pesos `.keras` automáticamente**.
Conservar semántica, activaciones y padding al portar la receta. Keras `same` con
stride 2 puede necesitar padding asimétrico: no prometer equivalencia numérica
usando siempre padding simétrico de PyTorch.

Las referencias visuales del usuario eran editores de diagramas con canvas libre,
conexiones y panel lateral. Los bocetos anteriores de volúmenes/etapas no son el
diseño aceptado. No depender de imágenes temporales en `/tmp` para implementar.

## 5. Contratos comunes: resolver antes de construir las pantallas nuevas

### 5.1 Convenciones

- Backend de visión: `[B,C,H,W]`.
- UI por defecto: `H × W × C`, con cada eje identificado y batch omitido.
- Inspector: ambas representaciones. No permutar datos al cambiar su presentación.
- Targets de clases: IDs enteros; máscaras discretas: IDs enteros; imagen: flotantes
  o uint8 según su etapa; cajas: flotantes con formato y sistema de coordenadas.
- Identidad estable de clase separada de su nombre visible y posición numérica.

### 5.2 Tarea y arquitectura

Mantener IDs existentes y añadir explícitamente, con migración cuando corresponda:

| Tarea propuesta | Objetivo |
| --- | --- |
| `image.localization` | Una caja por imagen |
| `image.detection` | Cero o más cajas/clases por imagen |
| `image.translation` | Entrada y objetivo de imagen emparejados |

`image.translation` cubrirá inicialmente pares de igual resolución y permitirá
canales distintos, por ejemplo gris→RGB. Denoising supervisado puede usar esta
tarea. `image.reconstruction` conserva su significado actual de reconstrucción de
la propia entrada; no migrarlo a pares inventados.

Superresolución, pose, OBB, segmentación de instancias, GAN y arquitecturas YOLO
completas específicas quedan como extensiones. No ofrecerlas como listas. La
estructura de puertos y recetas no debe impedirlas en el futuro.

Permitir tareas espaciales con CNN personalizadas además de presets U-Net. La
compatibilidad final se comprueba por operadores, datos y salidas, no por exigir
que el nombre de la arquitectura sea `unet`.

### 5.3 Grafo y puertos

Crear una nueva versión del esquema, posterior a las existentes. Definir tipos
equivalentes en TS y Python. Reutilizar los handles de React Flow.

Cada puerto necesita ID estable, dirección, tipo de dato, layout y contrato de
forma. Las dimensiones inferidas son resultado, no texto libre autoritativo.
Cada conexión identifica:

```json
{
  "id": "edge-1",
  "source": "encoder",
  "sourceHandle": "p3",
  "target": "merge",
  "targetHandle": "input_0"
}
```

Ejemplo: `encoder.p3` y `encoder.p4` son tensores diferentes, aunque procedan del
mismo módulo. Un consumidor debe recibir exactamente el puerto conectado.

- Mantener una entrada de imagen como recorrido base. No añadir múltiples entradas
  de usuario si los casos de aceptación no lo necesitan.
- Admitir cualquier número positivo de salidas nombradas con IDs únicos.
- Separar una salida del modelo de un puerto intermedio reutilizable.
- Las operaciones comunes devuelven un tensor; las multisalida pueden devolver
  una estructura nombrada normalizada por el ejecutor.
- `Concat` tiene entradas ordenadas mediante puertos estables (`input_0`, etc.).
  No usar orden de creación, posición en pantalla ni orden accidental del JSON.
- Una entrada tensor simple admite una conexión. Una salida admite varios destinos.
- Una operación no puede recibir un diccionario donde espera un tensor.
- Permitir borradores incompletos; bloquear entrenamiento de grafos inválidos.

Para compatibilidad, conservar el retorno tensor de grafos antiguos o añadir un
adaptador explícito en sus consumidores. No cambiar globalmente todo `forward`
a diccionario sin adaptar tests, entrenamiento, exportación e inferencia.

El contrato de dataset conserva `inputShape`/`outputShape` para los casos antiguos,
y añade objetivos estructurados cuando haga falta. Nunca fingir una única
`outputShape` para cajas y clases independientes.

### 5.4 Receta de tarea

Registrar las recetas en backend con versión e IDs estables. Una receta define:

1. Objetivos del dataset que requiere y sus tipos.
2. Nombres/roles de salida y si contienen logits o valores activados.
3. Preparación de targets: por ejemplo asignación de objetos a celdas.
4. Pérdidas, máscaras, reducciones y pesos relativos.
5. Métricas y su dirección de mejora.
6. Decodificación/posprocesamiento.
7. Presentación de resultados.

La UI ofrece defaults y campos compatibles con la receta seleccionada. No crear
un lenguaje universal de pérdidas, ni ejecutar Python arbitrario desde campos.
Una pérdida nueva se añade al registro y se prueba.

## 6. Fase A — Auditoría, contratos y migración

**Dependencias:** sección 2. **Archivos principales:** tipos, catálogo, storage,
manifest, modelos y consumidores de `outputShape`.

- [ ] Registrar qué funciones existentes se reutilizan y qué huecos se confirman.
- [ ] Añadir esquema de puertos, salidas y contratos estructurados.
- [ ] Migrar nodos antiguos con puertos implícitos a un puerto tensor predeterminado.
- [ ] Migrar `Concat` antiguo preservando su orden real anterior antes de guardar
      el nuevo orden explícito. La migración no puede cambiar las predicciones.
- [ ] Mantener lectura de proyectos y checkpoints anteriores. El loader de un
      checkpoint debe conocer su versión; no reinterpretar sus pesos a ciegas.
- [ ] Ampliar hash semántico para incluir puertos, orden, operaciones internas,
      contratos y selección de salidas. Excluir zoom, posiciones, selección y
      expandido/colapsado. Versionar el hash.
- [ ] Mantener una huella de corrida que incluya receta, procesamiento, pesos
      iniciales y política efectiva de entrenamiento, aunque se almacene aparte
      del hash de arquitectura.
- [ ] Añadir pruebas de migración y round-trip abrir→guardar→abrir.

**Aceptación:** un proyecto antiguo mantiene forma y predicción; mover nodos no
cambia el hash; intercambiar entradas de Concat sí cambia el hash y los valores.

## 7. Fase B — Ejecución multisalida y validación

**Dependencias:** A. **Archivos principales:** `models.py`, `engine.py`, contratos
y tests. Se pueden extraer módulos pequeños de puertos/shapes si ayudan.

- [ ] Ejecutar por `(node_id, port_id)` y conservar módulos registrados con IDs
      estables para cargar sus pesos.
- [ ] Soportar fan-out real sin duplicar módulos ni parámetros del productor.
- [ ] Validar ciclos, puertos inexistentes, entradas duplicadas, salidas con nombre
      repetido y nodos sin ruta útil de entrada a una salida.
- [ ] Validar `Add` con formas iguales, sin broadcasting accidental.
- [ ] Validar `Concat`: mismo resto de dimensiones y eje explícito; usar el nombre
      real de propiedad del registro, normalizando alias antiguos si hace falta.
- [ ] Soportar salida nombrada múltiple, manteniendo la ruta legacy.
- [ ] Separar inferencia ligera de formas de la validación completa con backward.
- [ ] Ampliar `graph.validate` para receta y targets estructurados; comprobar
      finitud de salidas, pérdida y gradientes esperados, con errores por nodo/puerto.
- [ ] La inferencia de formas no descarga pesos, no reinicializa módulos activos,
      no altera estadísticas BatchNorm y no recompila modelos grandes en cada tecla.
      Usar reglas de forma; reservar dry-run para comprobaciones necesarias.
- [ ] Devolver parámetros reales por nodo/bloque, entrenables y congelados sin
      contar dos veces parámetros expuestos por varios puertos.

**Fixture mínimo obligatorio:** entrada `[B,3,128,128]`; cuatro Conv stride 2 con
canales 16,24,32,64; mapa `[B,64,8,8]`; dos ramas que terminan en `[B,4,8,8]` y
`[B,3,8,8]`. Una pérdida combinada debe propagar gradientes a ambas ramas y al
tronco compartido. Reordenar el JSON de aristas no cambia la ejecución.

## 8. Fase C — Mejorar el diagrama actual

**Dependencias:** A y B. **Archivos principales:** `ModelNode.tsx`, canvas/inspector
de `App.tsx`, catálogo y estilos actuales.

- [ ] Mantener lienzo libre, pan/zoom, selección, eliminación y conexiones actuales.
- [ ] Mostrar tipo, propiedades esenciales, entrada/salida y estado de pesos.
- [ ] Separar procedencia (`nuevo`, `preentrenado`, `checkpoint propio`) de política
      (`entrenable`, `congelado`, `mixto`). Son dos informaciones diferentes.
- [ ] Mostrar puertos con nombre cuando hay varios y etiquetas de tensor en aristas
      seleccionadas o a suficiente zoom. Evitar saturar permanentemente el canvas.
- [ ] Actualizar formas con debounce tras cambios de propiedades/conexiones.
      Ignorar respuestas antiguas mediante revisión/hash de la petición.
- [ ] Marcar resultados como pendientes/obsoletos al editar, sin conservar una
      apariencia de validación vigente. Mover visualmente un nodo no la invalida.
- [ ] Resaltar conexión/nodo inválido y mostrar causa y posible corrección.
- [ ] Añadir acciones Crear rama, Agregar Add residual y Concatenar. Deben crear
      nodos/aristas reales; nunca reparar dimensiones silenciosamente.
- [ ] Ofrecer autoorganización horizontal, vertical y encoder–decoder, como acción
      opcional que solo modifica posiciones.
- [ ] Añadir resumen desplegable sincronizado con selección: operación, entrada,
      salida, parámetros, entrenables y estado. No reemplaza el canvas.
- [ ] Añadir acciones necesarias a deshacer/rehacer para no perder trabajo al
      insertar una rama, cambiar propiedades o agrupar.
- [ ] Añadir textos y errores en español e inglés.

**Aceptación visual:** en la interfaz existente se dibuja un skip de U-Net sin
ocultar conexiones; se sigue una bifurcación hasta sus dos salidas; cambiar stride
actualiza los nodos afectados y reporta una unión incompatible. Seleccionar una
fila del resumen enfoca el mismo nodo y viceversa.

## 9. Fase D — Datos espaciales y transformaciones conjuntas

**Dependencias:** A. Puede implementarse antes de terminar C.

### Importadores mínimos y representación interna

- [ ] Conservar importadores actuales de clasificación, regresión, reconstrucción
      y segmentación.
- [ ] Localización: imágenes + anotaciones de una caja por imagen.
- [ ] Detección: imágenes + etiquetas YOLO TXT (`class cx cy w h` normalizados) y
      un JSON sencillo documentado con cajas `xyxy` en píxeles. Ambos se convierten
      al mismo formato interno. COCO completo no es requisito de esta entrega.
- [ ] Pares imagen→imagen: carpetas `inputs/` y `targets/`, emparejadas de forma
      inequívoca por ID/nombre. Informar pares ausentes y tamaños incompatibles.
- [ ] Detección interna por muestra: imagen, `boxes[N,4]`, `labels[N]`, tamaño
      original e ID. Admitir N=0 y distinto N por imagen; collate explícito para
      listas/diccionarios, sin apilar cajas variables como un tensor rectangular.
- [ ] Elegir y persistir un formato canónico de cajas. Recomendada: `xyxy` en
      píxeles de la imagen en la etapa correspondiente. Convertir de forma explícita
      en importación, transformaciones y receta, conservando metadatos.
- [ ] No reemplazar globalmente el tensor legacy de targets sin adaptar cada
      consumidor. Verificar preview, analytics, split, guardado, loader e inferencia.
- [ ] Validar NaN, clases desconocidas, formatos, rangos y cajas degeneradas. Decidir
      y documentar qué correcciones explícitas se permiten; nunca inventar etiquetas.

### Pipeline

- [ ] Resize, crop, flip y rotación actualizan anotaciones espaciales coordinadas.
- [ ] Máscaras de clases usan interpolación nearest; imágenes usan la configurada.
- [ ] Recortes eliminan cajas sin área y actualizan labels; muestras que queden sin
      cajas siguen siendo válidas si la receta lo permite.
- [ ] En pares, geometría compartida; ruido/color se aplica al lado elegido.
- [ ] Transformaciones de entrenamiento aleatorias son train-only. Validación y
      test no reciben aumentaciones de entrenamiento.
- [ ] Particionar antes de ajustar estadísticas y de generar aumentaciones.
      Reutilizar los splits exactos, sin crear versiones aumentadas en otros splits.
- [ ] Evitar doble normalización de checkpoints preentrenados; guardar el pipeline
      efectivo que usará también inferencia.
- [ ] Previsualizar original y transformado con cajas/máscara/objetivo alineados.
- [ ] Guardar la geometría para devolver predicciones al sistema original: escala,
      padding si existe y transformaciones invertibles. Un crop puede perder campo
      visual: no prometer reconstruir lo recortado.

**Prueba geométrica exacta:** imagen de ancho 100 y caja `xyxy=[10,20,30,40]`.
Flip horizontal → `[70,20,90,40]`. Resize de 100×100 a 200×200 →
`[20,40,60,80]`. Verificar valores, no solo dimensiones.

## 10. Fase E — Recetas, entrenamiento y evaluación de visión

**Dependencias:** B y D. Reutilizar optimizadores, schedulers, AMP, checkpoints y
eventos existentes; ampliar sus consumidores de tensores estructurados.

- [ ] Registrar recetas para clasificación, localización, detección didáctica,
      segmentación binaria/multiclase y pares imagen→imagen.
- [ ] Adaptar batch→dispositivo para estructuras anidadas y datos variables.
- [ ] Relacionar cada salida del modelo con el objetivo correcto mediante nombres
      y roles persistidos; no mediante orden accidental del diccionario.
- [ ] Mostrar pérdidas por cabeza y total ponderado en eventos, gráficas y checkpoint.
- [ ] Agregar métricas con sus conteos/reducciones correctos. No promediar sin pesos
      métricas de batches de tamaños distintos ni IoU de imágenes ausentes.

### Localización

- [ ] Salida de una caja válida por imagen y formato explícito.
- [ ] MSE e IoU como alternativas compatibles, documentando rango y reducción.
- [ ] IoU y overlay predicción/objetivo en evaluación e inferencia.
- [ ] Representar la operación de ordenar esquinas del ejemplo mediante un operador
      registrado si forma parte del forward. No moverla a otro lugar si eso cambia
      la pérdida o sus gradientes.

### Detector didáctico de una escala

- [ ] Portar asignación, codificación `tx,ty,w,h` y decodificación de la referencia.
- [ ] Obtener tamaño de cuadrícula de la salida real: no fijar 8×8 para cualquier red.
- [ ] Cajas: pérdida solo en posiciones positivas y caso sin positivos finito.
- [ ] Clases: representar fondo y ponderación positiva/negativa como en la receta.
- [ ] Si se usan logits en vez del sigmoid de Keras, usar la pérdida equivalente
      estable y documentar la conversión; nunca aplicar sigmoid dos veces.
- [ ] Verificar que la máscara de supervisión no se convierte en una salida extra.
- [ ] Confianza, decodificación y NMS por clase configurables y persistidos.
- [ ] Métricas de detección sobre cajas decodificadas: precisión, recall, F1 e IoU
      con matching uno a uno y umbrales declarados. Añadir AP/mAP@0.5 mediante una
      implementación contrastada y fixtures conocidos; no llamarle mAP al F1 o a
      la IoU de celda asignada.
- [ ] Mostrar limitaciones de una caja por celda y colisiones de asignación.
- [ ] El grafo admite mapas multiescala. Probar un productor con dos puertos
      espaciales y consumidores independientes. Una receta multiescala solo se
      ofrece al usuario cuando asignación entre escalas, pérdidas y decodificación
      estén implementadas y verificadas; no mostrar un preset YOLO ficticio.

### Segmentación e imagen→imagen

- [ ] Mantener/adaptar las pérdidas y métricas existentes de segmentación.
- [ ] Validar logits, canales de salida y targets discretos.
- [ ] Pares imagen→imagen: L1 y MSE, rango explícito y métrica coherente con ese rango.
      Si se añade PSNR, calcularlo correctamente y tratar error cero sin truncados
      engañosos. No es necesario añadir SSIM en esta entrega.
- [ ] Separar canales de entrada y objetivo, incluso gris→RGB.
- [ ] Mostrar entrada, objetivo y predicción; durante inferencia el objetivo es opcional.

**Aceptación:** forward→loss→backward con targets reales para cada receta, entrenamiento
corto en CPU y evaluación/inferencia desde el checkpoint guardado. Una curva que
se dibuja sin actualizar parámetros no cuenta como entrenamiento.

## 11. Fase F — Compuestos, presets y backbones espaciales

**Dependencias:** B y C; integrar con E cuando el preset incluya receta.

- [ ] Distinguir agrupación visual de bloque compuesto ejecutable.
- [ ] Compuestos propios tienen nodos/aristas internos y puertos expuestos reales.
      Expandir permite seleccionar/editar operaciones. Colapsar no cambia forward,
      hash semántico, IDs de pesos ni conexiones.
- [ ] Duplicar un bloque crea instancias/pesos independientes; fan-out comparte el
      productor existente. No implementar pesos compartidos accidentales.
- [ ] Añadir presets editables: CNN compacta, residual, localizador, detector de
      una escala, U-Net binaria/multiclase e imagen→imagen.
- [ ] Conservar ResNet-18, MobileNet V2 y ConvNeXT Tiny preentrenados. Separar
      elección de arquitectura de inicialización aleatoria/checkpoint compatible.
- [ ] Adaptar backbones de catálogo para exponer mapas espaciales intermedios y,
      opcionalmente, vector resumido mediante puertos. Mantener resolución cuando
      la tarea lo permita; no imponer resize de clasificación que rompa cajas/máscaras.
- [ ] Declarar metadatos reales de canales/stride y seleccionar los tensores correctos
      por implementación. No inferir stages con heurísticas genéricas de nombres.
- [ ] El forward del backbone se ejecuta una vez por batch aunque se usen varios mapas.
- [ ] Backbones externos muestran módulos/etapas reales. Si una operación externa
      no puede editarse con fidelidad, mostrarla como importada de solo lectura;
      no fingir que una lista decorativa son capas editables. La interfaz debe
      distinguir esa limitación de los compuestos propios plenamente editables.
- [ ] Una modificación estructural conserva solo pesos cuya identidad, shape y
      semántica siguen siendo compatibles; presentar informe de carga.
- [ ] Adaptar heads al contrato de clases con acción visible y sin alterar otras ramas.

**Aceptación:** conectar un mapa espacial de ResNet a una Conv2D, conectar dos mapas
a ramas distintas y entrenar una cabeza. Comparar activaciones con el módulo
original en eval. En pruebas normales usar modelos diminutos/mocks sin red; una
integración opcional debe verificar al menos un checkpoint real disponible.

## 12. Fase G — Congelamiento y reutilización de entrenamientos

**Dependencias:** B, E y F.

### Política de entrenamiento

- [ ] Aplicar a cualquier capa con parámetros, incluida una CNN creada desde cero.
- [ ] Permitir congelar capa, bloque y selección; los contenedores muestran estado mixto.
- [ ] Atajos: solo cabezas, última etapa + cabezas, todo, selección personalizada.
- [ ] Guardar una política efectiva inequívoca. Los atajos actualizan la selección;
      una receta global no puede sobrescribir ocultamente una selección manual.
- [ ] Crear grupos de optimizador desde esa política; no basarse solo en que el
      nombre del parámetro contenga `backbone`.
- [ ] Gestionar BatchNorm explícitamente. La opción de congelar estadísticas debe
      sobrevivir a cada `model.train()`; congelar pesos no basta para ello.
- [ ] Si hay descongelamiento gradual, preservar el estado del optimizador y evitar
      parámetros duplicados al abrir nuevas etapas.
- [ ] Mostrar antes de iniciar: parámetros totales/entrenables, LR por grupo,
      inicialización, procesamiento y receta efectivos.

### Tres acciones diferenciadas

| Acción | Se conserva | Se reinicia / valida |
| --- | --- | --- |
| Reanudar corrida | Pesos, epoch, optimizador, scheduler, scaler, RNG y demás estado necesario | Validar identidad de grafo, datos/splits, receta y política; permitir solo cambios compatibles explícitos |
| Nuevo entrenamiento desde pesos | Pesos compatibles de un checkpoint elegido | Nueva corrida y optimizador; nuevos datos/configuración permitidos y registrados |
| Usar como backbone | Subgrafo elegido, dependencias y pesos; puertos expuestos | Nuevas cabezas/receta; comprobar procesamiento y compatibilidad |

- [ ] Añadir acciones visibles desde la biblioteca/resultados existentes.
- [ ] Elegir una revisión/checkpoint inmutable concreto, no un alias `best` mutable
      sin resolver su identidad en el manifiesto de la nueva corrida.
- [ ] Extraer todos los antecesores necesarios de los mapas seleccionados. No asumir
      que un backbone es siempre una secuencia ni cortar dependencias residuales.
- [ ] Mantener IDs internos o un mapa explícito para cargar pesos del subgrafo.
- [ ] La reutilización crea una copia versionada, no un vínculo que modifica el
      modelo fuente cuando se vuelve a entrenar.
- [ ] Mostrar informe: pesos cargados, capas nuevas, incompatibles y motivo. No usar
      `strict=False` como solución universal sin comprobar el resultado.
- [ ] Validar identidad/orden de clases aunque el número de clases coincida.
- [ ] Registrar lineage: checkpoint padre, subgrafo/puertos y adaptaciones.
- [ ] Guardar en bundles los artefactos necesarios para funcionar sin el proyecto
      fuente: grafo, pesos, procesamiento, receta, clases y versiones.

**Aceptación:** guardar encoder→crear otro modelo con ese encoder→verificar igualdad
de características en eval→entrenar cabeza nueva→verificar que pesos congelados
no cambian y la nueva cabeza sí. El modelo fuente permanece intacto.

## 13. Fase H — Inferencia, bundles y dos modos de ejecución

**Dependencias:** E, F y G.

- [ ] Inferencia carga grafo/receta/pipeline del checkpoint, no el borrador actual.
- [ ] Aplicar exactamente el procesamiento determinista guardado.
- [ ] Detección: cajas/clases/confianzas y overlay en coordenadas de imagen original.
- [ ] Segmentación: máscara y overlay con tamaño/clases coherentes.
- [ ] Localización: caja y coordenadas explícitas.
- [ ] Pares: entrada y predicción; objetivo solo si se proporciona para evaluar.
- [ ] Guardar umbrales e indicar cuándo una comparación usa umbrales distintos.
- [ ] Exportar/importar bundle de cada tarea y verificar predicción equivalente.
- [ ] Probar tanto `npm run dev` como Tauri para los cambios de contrato y selección
      de archivos. No dar por probado escritorio solo por funcionar en web.
- [ ] Si falta un requisito de Tauri, registrar el bloqueo de entorno y qué sí se
      verificó; no declarar esa comprobación aprobada.

## 14. Orden de ejecución y entregas pequeñas

Seguir este orden salvo dependencia técnica documentada:

1. Auditoría y pruebas iniciales.
2. A: esquema y migración.
3. B: motor multisalida y formas.
4. C: mejoras del canvas aprovechando B.
5. D: datos y geometría.
6. E: recetas completas, una por una.
7. F: compuestos, presets y mapas de backbones.
8. G: políticas y reutilización.
9. H: cerrar inferencia, bundles y web/escritorio.
10. Ejecutar matriz final, corregir regresiones y actualizar documentación.

Cada entrega debe dejar el proyecto ejecutable. No acumular un frontend entero
con botones simulados esperando implementar después el backend. Se permiten
funciones internas todavía no expuestas mientras sus recorridos estén incompletos.

No terminar el encargo después de C: una mejora visual no completa los casos de CNN.
Si cambia el contexto o debe continuar otro agente, actualizar la sección 17 con
el siguiente paso exacto y sus pendientes.

## 15. Matriz de aceptación final

Todos los casos de esta tabla son obligatorios. Usar datos diminutos y CPU para
las pruebas ordinarias. Los tests de red/GPU son complementarios.

| ID | Recorrido | Evidencia mínima | Estado comprobado |
| --- | --- | --- | --- |
| C01 | CNN clasificación | Importar→procesar→editar→entrenar→checkpoint→predecir clase | Implementado y probado con un recorrido CPU añadido en `tests/test_cnn_vision_tasks.py`. |
| C02 | Localización una caja | Cajas transformadas correctamente; loss y predicción de cuatro coordenadas válidas | Implementado; importación CSV, geometría, entrenamiento, decoder ordenado e inferencia probados. |
| C03 | Residual | Add válido entrena; formas distintas fallan antes de entrenar y señalan el puerto | Implementado; backward de rama y error por shape probados. |
| C04 | Detector didáctico | Dos salidas 8×8×4/3, pérdidas separadas, backward de ambas, decode/NMS y evaluación final | Implementado; fixture 64→8×8, losses, colisiones, NMS/mAP50 y corrida de entrenamiento probados. Una caja por celda; colisiones se rechazan. |
| C05 | Grafo multiescala | Dos mapas reales de un productor, puertos distintos, consumidores correctos y sin doble forward | Implementado con `multi_scale_cnn`, `p2/p3/p4` y adaptador HF; prueba tres consumidores y un solo forward. |
| C06 | U-Net binaria | Skips editables, salida `[B,1,H,W]`, máscara alineada, entrenamiento e inferencia | Implementado y probado en CPU. |
| C07 | U-Net multiclase | Salida `[B,K,H,W]`, targets enteros `[B,H,W]`, clases y overlay correctos | Implementado y probado en CPU. |
| C08 | Imagen→imagen | Pares válidos, ruido solo en entrada cuando se elige, objetivo intacto, predicción guardada | Implementado; loader de pares, geometría compartida, aumento de entrada, entrenamiento e inferencia probados. |
| C09 | Fine-tuning | Backbone congelado permanece igual, cabeza cambia; selección parcial y BatchNorm se respetan | Implementado; congelamiento de capas CNN y estadísticas BN probado. Selección de una capa por vez; no hay selector visual múltiple. |
| C10 | Reanudar | Restauración de estado y continuidad; rechazo explicativo de cambios incompatibles | Reanudación CPU continua/reanudada compara historial y RNG; validaciones de compatibilidad activas. |
| C11 | Nueva corrida desde pesos | Igualdad antes de entrenar, optimizador nuevo y lineage correcto | Implementado; pesos compatibles, optimizer fresco e informe `initialized_from` probados. |
| C12 | Backbone propio | Extracción con skips, varias salidas expuestas, nueva cabeza, fuente intacta | Implementación añadida para copiar antecesores/skip, seleccionar nodo/puerto y fijar pesos por IDs. Prueba del extractor y de carga parcial con grafo residual. Pendiente recorrido en UI real y elegir varios puertos a la vez. |
| C13 | Migración | Proyectos/checkpoints antiguos abren; Concat conserva orden/predicción | Compatibilidad legacy de conexiones implícitas y orden de Concat probado. Pendiente fixture de proyecto/checkpoint antiguo de extremo a extremo. |
| C14 | Bundle | Exportar→importar→inferir sin depender del proyecto fuente, misma predicción en eval | Importación de bundle e inferencia manual portable para detección producen las mismas detecciones; falta repetirlo para todas las tareas CNN. |
| C15 | Interfaz | Diagrama libre conservado, puertos reales, shapes al editar, resumen sincronizado y errores locales | Chromium capturó las nueve arquitecturas y el flujo principal a 1600×1000 y 1280×800. El encuadre estrecho muestra tres tarjetas completas en CNN/U-Net; grafos amplios continúan con aviso y minimapa. Shapes concretos dependen de fijar la entrada y validar. El cableado interactivo de React Flow aún requiere prueba manual. |
| C16 | Regresión | Tests existentes pasan en familias ajenas a CNN sin reescribir su interfaz | `python3 -m pytest -q`: **109 tests aprobados** (suite completa). |

### Pruebas específicas para evitar falsos positivos

- Comparar valores, no solo shapes, en puertos múltiples, Concat y migraciones.
- Reordenar el array JSON de conexiones sin cambiar sus IDs/puertos debe mantener
  resultado; intercambiar los puertos de Concat debe alterar el resultado esperado.
- Casos negativos: ciclo, puerto inexistente, entrada duplicada, par ausente,
  máscara mal alineada, caja fuera de contrato y clases incompatibles.
- Detección: imagen sin objetos, batch con cantidades distintas y colisión de celda.
- NMS/matching: fixtures con TP/FP/FN conocidos; diferenciar clases.
- Cambiar stride en un nodo debe invalidar formas/validación anteriores. Respuestas
  de validación atrasadas no pueden volver a declarar válido el nuevo grafo.
- Cambiar número/orden de clases no modifica silenciosamente un checkpoint fuente.
- Guardar y reabrir no pierde conexiones a puertos ni estado de congelamiento.
- Expandir/colapsar compuestos mantiene salidas y pesos.
- Gradientes del tronco compartido reciben contribución de ambas cabezas.
- `requires_grad=False` no es evidencia suficiente: comparar pesos antes/después y
  estadísticas BatchNorm cuando se haya elegido congelarlas.
- Para reanudación determinista, probar en CPU con configuración determinista y
  comparar entrenamiento continuo frente a interrumpido/reanudado, incluyendo
  optimizador, mejor score y contador de early stopping. Documentar tolerancias.
- Entrenamiento de smoke corto prueba integración, no calidad final del modelo.
  Añadir una prueba acotada de sobreajuste de un lote diminuto para detectar targets
  mal conectados o gradientes ausentes, sin umbrales de accuracy inventados.

## 16. Verificación, alcance y errores que no se deben cometer

### Comandos de referencia

Usar el entorno Python del proyecto si existe; no instalar dependencias globales.

```bash
.venv/bin/python -m pytest
npm run build
npm run dev
```

Si `.venv` no existe, seguir la instalación del repositorio. Añadir módulos de test
específicos para puertos, datos espaciales, detección y reutilización según se
implemente; los nombres se decidirán al crear los archivos.

Para cambios lógicos, ejecutar primero sus pruebas dirigidas. Al cerrar, ejecutar
suite completa y build. Hacer revisión visual del flujo real; cubrir un grafo
ramificado, U-Net y el inspector a distintos niveles de zoom. Registrar comandos,
resultados y limitaciones reales. No afirmar una prueba que no se ejecutó.

Si se añade una dependencia como TorchVision o una implementación de métricas,
comprobar compatibilidad con Torch y plataformas soportadas, declararla donde
corresponde y actualizar instalación. Evitar nuevas bibliotecas pesadas para
resolver tareas que ya cubre el proyecto.

### Fuera del alcance obligatorio

- Reescribir la app, sustituir React Flow o rediseñar la navegación.
- Mejorar todas las familias no CNN a la vez.
- Importación universal de `.keras`, ONNX o Python arbitrario.
- Reproducir un YOLO comercial/completo por usar dos cabezas convolucionales.
- Nuevas tareas de pose, OBB, GAN, segmentación de instancias o superresolución.
- Paneles decorativos, métricas ficticias, formas inventadas o descarga automática
  de modelos al mover nodos.

### Errores que invalidan la entrega

1. Cambiar solo la apariencia y mantener el límite oculto de una única salida.
2. Exponer puertos/etapas que no corresponden a tensores/módulos ejecutados.
3. Aplanar un backbone espacial antes de conectarlo a un decoder.
4. Aumentar imágenes sin sus cajas/máscaras o alterar targets sin la política elegida.
5. Comparar tensores por tamaño ignorando su significado, rango o clases.
6. Convertir el target de máscara del detector en una predicción adicional.
7. Confundir una plantilla aleatoria con pesos preentrenados.
8. Llamar reanudación a cargar solo pesos y reiniciar el optimizador.
9. Perder pesos al expandir bloques o generar IDs nuevos cada vez que se renderizan.
10. Entrenar sobre el borrador en lugar del snapshot asociado a la corrida.
11. Marcar una tarea completa sin inferencia y persistencia verificadas.

### Documentación final

Actualizar README y documentos de datos, constructor, arquitectura y entrenamiento
solo con comportamientos comprobados. Conservar este archivo como registro del
encargo; relacionar los pendientes de otros TODO sin borrar su historial.

## 17. Registro de ejecución para el agente implementador

Estado verificado el 2026-11-07. La matriz C01–C16 de la sección 15 resume cada
recorrido; «parcial» significa que existe código real pero falta una parte de la
evidencia o una función avanzada. No equivale a decir que CNN está cerrado al 100%.

| Fase | Estado | Implementación/evidencia | Pendiente verificable |
| --- | --- | --- | --- |
| Auditoría | Parcial | Se preservaron las modificaciones que ya estaban en el árbol; no había baseline reproducible ni capturas visuales de la app antes de editar. Se regeneraron capturas en Chromium después de los cambios. | C15 aún necesita revisión interactiva de React Flow y cableado real. |
| A · Contratos/migración | Parcial | Hash semántico v3, salidas con nombre, puertos fuente de mapas y orden de Concat/Add. Las aristas legacy sin puertos conservan el orden guardado. | Falta fixture de proyecto/checkpoint antiguo abierto desde storage y validado hasta inferencia/reanudación. |
| B · Ejecutor/validación | Mayormente implementado | Fan-out comparte módulo; `multi_scale_cnn` y backbone HF devuelven mapas reales por puerto; validación ejecuta forward/backward; `Add`/`Concat` validan formas; detección tiene receta `boxes`/`classes`. | Outputs distintos sólo se ligan a targets distintos en la receta didáctica de detección; bindings arbitrarios por output no existen. |
| C · Diagrama | Parcial | Lienzo libre, handles de merge/mapas, salidas nombradas, ramas, inspector por capa y resumen de formas. Chromium confirma el encuadre inicial y spacing de CNN/U-Net a 1600×1000; el mockup de tres alternativas sigue sin aplicarse. | Revisar conexiones/zoom interactivos, estados al editar y viewport estrecho. El usuario aún debe elegir el nuevo tratamiento visual. |
| D · Datos/pipeline | Mayormente implementado | Importa localización, labels YOLO/JSON, imágenes y pares; persiste cajas variables y geometría original; crop/flip/rotación y aumentos coordinados preservan input-target. | Completar revisión de formatos poco comunes, máscaras nearest y preview visual de todas las anotaciones. |
| E · Recetas/entrenamiento | Parcial | Clasificación, regresión visual, localización, detección didáctica, segmentación y pares image→image; losses/métricas, decoders, matching/NMS, resume CPU y reporte por cabeza de detección. | Detector de una escala/una caja por celda; colisiones de centro se rechazan. No hay receta genérica para targets/head arbitrarios. |
| F · Compuestos/presets | Parcial | Presets CNN residual/denso/detector, U-Net y multi-scale; ResNet-18, MobileNet V2 y ConvNeXT Tiny tienen plantillas preentrenadas. | «Ver etapas» en HF sólo muestra etiquetas, no expande operaciones ejecutables. No hay pesos preentrenados para segmentación/image→image. |
| G · Congelar/reutilizar | Parcial | Congelamiento por nodo y BN; nuevas corridas separan `initial_checkpoint` de resume. Añadida extracción de ancestros con skips, selección de puerto, IDs conservados, cabeza nueva, freeze inicial y lineage; loader verifica tensores copiados. | El diálogo elige un puerto inicial. Los otros puertos quedan disponibles para cablear después; falta selección simultánea y revisión UI/Tauri. |
| H · Inferencia/bundles | Parcial | Inference usa checkpoint y pipeline guardados; caja/detección/segmentación/pairs producen vistas; bundle portable probado manualmente con detección y salida equivalente. | Repetir equivalencia portable para todas las tareas CNN; Tauri empaquetado no se comprobó. |
| Matriz C01–C16 | Parcial | La tabla por caso de sección 15 indica estado y evidencia actualizados. Capturas finales generadas tras los ajustes del agente Terra. | C12–C15 conservan pendientes concretos; C15 ya tiene revisión visual automatizada, pero no interacción de cables. |

### Bitácora: ejecución iniciada el 2026-11-07

El repositorio tenía modificaciones y archivos sin seguimiento antes de comenzar.
Se conservaron. La lista inicial observada con `git status --short` incluía:

- README, scripts y configuración de Vite/Tauri.
- Catálogo/modelos/datos/entrenamiento/errores/storage/engine del backend.
- App, pipeline, catálogo, tipos, i18n y estilos del frontend.
- `IMPLEMENTATION_TODO.md`, `backend/modelbuilder/manifest.py`,
  `backend/modelbuilder/pretrained.py` y `tests/test_pretrained_models.py`.
- El nuevo `docs/09-plan-cierre-cnn.md` que guía esta ejecución.

No se capturó una línea base de tests o screenshots antes de editar. En esta
continuación se ejecutaron las comprobaciones posteriores a los cambios:

- `python3 -m pytest -q tests/test_cnn_vision_tasks.py tests/test_cnn_multi_output.py`:
  **25 passed** en la última pasada dirigida.
- `npm run test:cnn-reuse`: extractor de subgrafo, skips residuales, selección de
  puerto, cabeza detectora, resize y rechazo de escala incompatible aprobados.
- `python3 -m pytest -q`: **109 passed**; incluye las pruebas CNN y regresión de
  arquitecturas/datasets ajenos a visión.
- `npm run build`: completado. Vite advierte que dos chunks exceden 500 kB. Es una
  advertencia de tamaño, no un fallo de compilación.
- `CAPTURE_DIR=/tmp/modelbuilder-cnn-qa-final npm run capturas`: generó capturas de
  las nueve arquitecturas y de datos, constructor, entrenamiento e inferencia en
  Chromium a 1600×1000. La revisión visual confirmó spacing sin solapamiento en las
  entradas visibles de CNN/U-Net y el aviso/minimapa para recorrer grafos anchos.
- `CAPTURE_WIDTH` y `CAPTURE_HEIGHT` aceptan dimensiones alternas para revisar
  ventanas estrechas. `CAPTURE_WIDTH=1280 CAPTURE_HEIGHT=800` confirmó que el canvas
  muestra completas las tres tarjetas iniciales de CNN/U-Net sin solapamiento; el
  resto se recorre con pan y minimapa. Las capturas representan drafts: shapes
  concretos aparecen al fijar la forma de entrada y validar el grafo.
- El agente GPT-6.1 Terra medio ajustó el encuadre inicial de los primeros bloques,
  el hint de navegación, el resumen del inspector y el espaciado al insertar ramas,
  presets y cabezas derivadas. En la pasada de QA corrigió el encuadre de 1280 px para
  que no corte una tarjeta. No modificó el estilo general de las tarjetas.
- `CAPTURE_DIR=/tmp/modelbuilder-cnn-qa5-1280 CAPTURE_WIDTH=1280 CAPTURE_HEIGHT=800
  npm run capturas` y `CAPTURE_DIR=/tmp/modelbuilder-cnn-qa5-1600 npm run capturas`:
  revalidaron CNN/U-Net en viewport estrecho y de escritorio después de la corrección.

### Código añadido o actualizado en estas iteraciones

- `backend/modelbuilder/models.py`: outputs terminales nombrados, varios resultados,
  fan-out compartido, puertos de productor multi-escala, orden de puertos en Add/Concat,
  validación de shapes y congelamiento genérico por propiedad de capa.
- `backend/modelbuilder/training.py`: pérdidas de detección y cabezas auxiliares,
  loaders por muestra de cajas variables, augmentations coordinadas, resume exacto,
  decoder de localización y carga inicial parcial verificada por valores de tensor.
- `backend/modelbuilder/data.py` y `backend/modelbuilder/engine.py`: importadores de
  anotaciones/pares, geometría original, transforms coordinadas e inferencia con
  coordenadas/overlays para visión.
- `backend/modelbuilder/manifest.py`: hash v3 y orden semántico explícito de entradas
  Concat sin cambiar el significado de grafos legacy.
- `frontend/src/App.tsx` y `frontend/src/cnn-reuse.ts`: acciones de rama/merge,
  resumen de formas, snapshots de grafo por corrida y creación de modelo versionado a
  partir de checkpoint, copiando antecesores/skips con IDs/puertos y añadiendo cabeza.
- `frontend/src/ModelNode.tsx`, `frontend/src/catalog.ts`, tipos, traducciones y
  estilos: puertos legibles para Add/Concat/mapas, presets y estados de congelamiento.
  El rediseño general del nodo que aparece en el mockup aún no se ha aplicado.
- `tests/test_cnn_multi_output.py`, `tests/test_cnn_vision_tasks.py` y
  `scripts/test-cnn-reuse.cjs`: contratos de ramas/mapas, clasificación, detector,
  U-Net, localización, pares, bundle e inicialización desde backbone residual propio.

La prueba con cabeza nueva detectó que `load_initial_weights` confiaba en la lista
`missing_keys` de PyTorch para informar un subconjunto cargado. Esa suposición no se
cumple en este caso. Ahora el loader comprueba los tensores efectivamente copiados y
forma el informe `loaded_keys` desde esa verificación.

La receta de detección sí entrena `boxes [B,4,Hg,Wg]` y
`classes [B,K,Hg,Wg]` con objetivos y losses distintos. Otros outputs con shapes o
targets incompatibles aún requieren receta propia y se rechazan con
`GRAPH_MULTI_OUTPUT_RECIPE_REQUIRED`.

### Pendientes reales para completar el cierre

1. **Verificación UI en ejecución.** Abrir React Flow y comprobar cableado
   `input_0/1`, `p2/p3/p4`, edges legacy, resumen y shapes obsoletos; revisar después
   web/Tauri. El build no prueba el comportamiento del canvas.
2. **Migración integral.** Añadir fixtures de proyecto y checkpoint antiguos; probar
   abrir, validar, reanudar e inferir. El orden legacy de Concat tiene prueba unitaria,
   pero no recorrido de storage.
3. **Bindings genéricos de targets.** La receta de detección cubre sus dos salidas.
   No hay una forma general de declarar targets, losses y pesos por cabeza desde UI.
4. **Límite del detector didáctico.** Es una escala con una caja por celda. Cuando
   chocan centros, falla explícitamente; no reemplaza YOLO moderno/anchors/FPN.
5. **Reutilización propia.** El diálogo elige un único nodo/puerto para crear la cabeza
   inicial. Los demás puertos del productor se conservan para cablear ramas después;
   falta selección múltiple de mapas dentro del diálogo.
6. **Bundles portables.** La misma predicción manual se comprobó en detección tras
   importar el bundle y retirar el dataset. Repetir para clasificación, localización,
   segmentación e imagen→imagen.
7. **Visual de nodos.** Hay mockup de tres tratamientos para que el usuario elija. No
   cambiar el diseño general hasta recibir su preferencia.

## 18. Instrucción para quien cierre los pendientes

> Termina los pendientes de esta sección y las filas parciales de C01–C16. No borres
> modificaciones ajenas previas a la sesión. Revisa la app real antes de cambiar la
> presentación general de nodos: el usuario pidió ver el mockup primero. No declares
> terminado ningún recorrido que sólo tenga UI o build; añade evidencia automatizada
> y actualiza el estado exacto en la matriz.
