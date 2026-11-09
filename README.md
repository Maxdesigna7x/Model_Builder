# ModelBuilder

Aplicación visual local para preparar datos, construir redes mediante capas conectadas, entrenarlas con **PyTorch** y probar sus predicciones. El trabajo reciente se centra en **CNN**: formas de tensor visibles, ramas paralelas, conexiones residuales, presets y reutilización de checkpoints como backbone.

La interfaz usa React, TypeScript, React Flow y ECharts. Puede ejecutarse en navegador con Vite y backend Python local, o como aplicación de escritorio con Tauri 2.

## Pantallas

Capturas de la aplicación de escritorio en **tema oscuro con acento verde**.

### Proyectos

Logo adaptado al color del tema, creación de proyectos, búsqueda, filtros por arquitectura y vistas de lista o mosaico.

![Inicio y proyectos con el tema verde](docs/img/inicio-verde.png)

### Arquitectura y tarea

El catálogo separa familias, arquitecturas y tareas compatibles, con búsqueda y una vista visual de cada arquitectura que se adapta al espacio para mantener todas las tareas visibles. La selección se confirma con **Continuar**.

![Selección de arquitectura y tarea](docs/img/arquitecturas-verde.png)

### Datos

Pipeline de preparación, explorador con pestañas Info / Samples / Charts, muestras aleatorias, estadísticas de dimensiones originales y reparto train / validation / test.

![Preparación de datos y explorador de muestras](docs/img/datos-verde.png)

### Constructor

Biblioteca de capas y bloques, puertos de entrada/salida, formas por eje e inspector de propiedades. Sumar y Concatenar crean ramas completas y conectadas desde **Ramas**.

![Constructor CNN y dimensiones de tensor](docs/img/constructor-verde.png)

El inspector permite revisar y modificar las propiedades de la capa seleccionada.

![Constructor con el inspector de una capa abierto](docs/img/constructor-inspector-verde.png)

### Entrenamiento

Curvas separadas de entrenamiento y validación, mejor epoch, selección del modelo y configuración de early stopping.

![Entrenamiento y métricas](docs/img/entrenamiento-verde.png)

### Inferencia

En clasificación de imágenes, el resultado muestra la entrada, la clase predicha con su probabilidad y la clase real cuando está disponible.

![Predicción de clasificación](docs/img/inferencia-verde.png)

## Flujo de trabajo

1. Crea un proyecto y elige una arquitectura y una tarea.
2. Elige el origen del dataset: Kaggle, catálogo de Hugging Face o datos propios.
3. Prepara los datos en el pipeline. En imágenes se incluyen Resize y conversión de canales; revisa las muestras y las particiones.
4. Construye el modelo desde la plantilla inicial, un preset o un backbone preentrenado. El panel lateral permite cambiar de etapa.
5. Entrena, compara corridas y prueba el checkpoint en Inferencia.

El estado del proyecto se guarda localmente: en `workspace_data/` al ejecutar desde el código fuente y en la carpeta de datos del usuario en la aplicación instalada. Los datasets, cachés y checkpoints permanecen en el equipo; las descargas iniciales de servicios públicos requieren conexión y pueden requerir credenciales o aceptación de licencia.

## Arquitecturas y tareas disponibles

El catálogo actual contiene **9 familias y 16 tareas**. Las combinaciones permitidas se definen en [el catálogo del backend](backend/modelbuilder/catalog.py).

| Familia | Usos disponibles |
| --- | --- |
| MLP | Clasificación y regresión tabular |
| CNN 1D | Clasificación, regresión y pronóstico de secuencias |
| CNN 2D | Clasificación y regresión de imágenes, localización de una caja, detección e imagen a imagen |
| RNN / GRU / LSTM | Clasificación, regresión y pronóstico de secuencias |
| Transformer encoder | Secuencias y clasificación de texto |
| Transformer causal | Modelado de lenguaje por tokens |
| ViT | Clasificación y regresión de imágenes |
| U-Net | Segmentación binaria/multiclase e imagen a imagen |
| Autoencoder | Reconstrucción tabular y de imágenes |

Las tareas de detección y los bloques multiescala permiten salidas espaciales y cabezas separadas. Esto no implica compatibilidad automática con cualquier arquitectura o checkpoint YOLO externo.

VAE y Transformer encoder–decoder no están disponibles en el catálogo actual. Las propuestas de ayudas visuales animadas están en [prototypes/node-help](prototypes/node-help/README.md), separadas de la interfaz de producción.

## Constructor de modelos

- Capas individuales y bloques/presets en pestañas separadas, con ayudas y vistas de forma de salida.
- Ramas paralelas y operaciones `Add` y `Concatenate`. La suma requiere tensores compatibles; la concatenación requiere coincidencia fuera del eje elegido.
- Capas de decoder como ConvTranspose2D, Resize2D y Reshape disponibles en CNN.
- Validación automática de cambios del grafo, ciclos, conexiones, formas y ejecución de comprobación. El canvas muestra su estado.
- **Ver soluciones** en una capa con error o en la barra del constructor: propone cambios explícitos para incompatibilidades conocidas, comprobados en una copia mediante ejecución, pérdida y gradientes. Permite aplicar y deshacer la última reparación. Requiere datos preparados; los errores sin reparación verificable se indican sin modificar el modelo. No optimiza la precisión.
- Canales/filtros separados de la resolución H × W, con explicación sencilla de los ejes en el inspector. El batch se configura al entrenar.
- Tarjetas verticales con distribución automática y ramas en paralelo. Un clic en una flecha inserta capas ya conectadas; la biblioteca permite elegir una capa y después su posición.
- Arrastre de la tarjeta hacia arriba o abajo para previsualizar el nuevo orden y las conexiones en tiempo real; borrado y duplicación conservan las conexiones. Al quitar una unión se elige qué rama conservar y se limpian las capas que quedan sin uso. Las uniones redundantes se colapsan al borrar su última rama. La rueda desplaza el canvas y los controles ajustan el zoom.
- Selección múltiple y grupos de elementos consecutivos, incluidos grupos anidados. Agrupar no está disponible para un único grupo completo ni para selecciones con capas intermedias sin seleccionar. Desagrupar conserva los grupos interiores.
- Minimapa con encuadre fijo de todo el modelo, independiente del zoom y desplazamiento de la cámara; permite navegar y ajustar la vista completa.
- Acciones flotantes para duplicar o borrar secuencias con una sola entrada y salida.
- Paneles laterales plegables y tabs de modelos con cierre. Guardar una arquitectura solicita un nombre y la añade a los presets personalizados.
- Reutilización de un checkpoint entrenado como backbone, con selección de la salida de características y una cabeza nueva compatible.

## Datasets y preparación

### Kaggle

La integración usa `kagglehub` y está disponible para **clasificación de imágenes**. Descarga desde una URL o un identificador `propietario/dataset`, detecta carpetas por clase o etiquetas CSV y reúne las particiones originales antes de que definas tus propios splits.

La resolución de entrada se configura después, mediante el pipeline. No todos los formatos de anotación de Kaggle se importan automáticamente; datasets privados o con condiciones de acceso necesitan credenciales y permisos en Kaggle. Consulta [formatos y configuración](docs/11-kaggle.md).

### Hugging Face y datos propios

El catálogo curado ofrece datasets compatibles con la tarea elegida. La app comprueba el proyecto y la caché física antes de descargar una revisión fijada. La importación local admite los formatos correspondientes a la tarea: tablas, imágenes y sus etiquetas, texto, secuencias o pares de imágenes/máscaras según el contrato mostrado por la app.

Las particiones son editables cuando la fuente lo permite; los repartos oficiales o temporales pueden estar bloqueados para conservar su contrato. El botón de dado del explorador pide nuevas muestras aleatorias del dataset persistido, conservando las etiquetas emparejadas.

**Límite actual de resolución:** algunos adaptadores materializan las imágenes a un tamaño uniforme antes del pipeline editable. Ampliarlas después no recupera detalle ya reducido; las estadísticas de Info corresponden a las dimensiones de origen cuando están disponibles.

## Preentrenados y fine-tuning

El menú de presets separa **plantillas desde cero** y **modelos preentrenados**. El catálogo curado de Hugging Face utiliza revisiones fijadas, muestra su licencia y no permite código remoto mediante `trust_remote_code`.

En visión se representan el processor, el backbone y la cabeza compatible con el dataset. En texto se utiliza el tokenizer del checkpoint; los modelos causales conservan su salida de vocabulario. Los pesos descargados se reutilizan desde la caché local.

El inspector permite congelar pesos. Entrenamiento ofrece recetas para cabeza, fine-tuning completo, descongelamiento gradual y LoRA cuando son compatibles. Las familias sin un contrato genérico de pesos preentrenados mantienen plantillas desde cero.

## Entrenamiento y checkpoints

- Proceso PyTorch separado, uso de CPU o CUDA según disponibilidad.
- Loss y métrica de entrenamiento/validación, con marcas del mejor epoch. Accuracy mantiene la escala 0–1.
- Pausar, reanudar y detener desde el panel de entrenamiento.
- Early stopping apagado por defecto; al activarlo, la paciencia monitorea la pérdida de validación independientemente del criterio de selección de checkpoint.
- Optimizador, scheduler, warmup, precisión mixta, acumulación de gradientes, clipping y opciones de loss según la tarea.
- Checkpoints `best` / `last`, evaluación final sobre test, configuración y procedencia de cada corrida, y bundle local con los artefactos correspondientes.
- Entrenar de nuevo conserva la corrida completada y crea otra. Se distinguen la reanudación de una corrida y el uso de pesos para un nuevo fine-tuning.

Un checkpoint antiguo sin historial de validación utilizable reinicia el contador independiente de early stopping al reanudar. Los cambios de configuración de una corrida todavía no iniciada no se guardan como borrador independiente.

## Instalación y ejecución

### Descargar la aplicación de escritorio

Descarga la versión disponible en [GitHub Releases](https://github.com/Maxdesigna7x/Model_Builder/releases/latest). No necesitas instalar Python, Node.js, Rust ni ejecutar comandos de desarrollo: la aplicación prepara automáticamente un entorno privado y descarga sus dependencias al abrirla por primera vez. Necesitarás conexión a Internet y varios GB de espacio libre durante esa preparación.

- **Windows x64:** descarga el instalador `-setup.exe` o el archivo `.msi`, instálalo y abre ModelBuilder.
- **Linux x86_64:** descarga el paquete `.deb` para distribuciones compatibles o el archivo `.AppImage`. Instala el paquete `.deb` con tu gestor de software; si usas AppImage, dale permiso de ejecución y ábrelo.

Los instaladores recientes permiten elegir **CPU o GPU NVIDIA** durante la preparación inicial. Consulta las notas de cada versión en Releases para conocer las opciones disponibles.

Las versiones que incluyen el actualizador avisan cuando hay una versión nueva. Windows y AppImage permiten instalarla desde la app; para DEB se ofrece la descarga del paquete. El entorno compatible y los proyectos se conservan. Las versiones anteriores al actualizador necesitan una instalación manual inicial.

### Ejecutar desde el código fuente

Los siguientes requisitos y comandos son para desarrollo o ejecución desde el repositorio.

| Componente | Modo web local | Escritorio Tauri |
| --- | --- | --- |
| Node.js y npm | Node 20.19+ (rama 20) o 22.12+ (se recomienda LTS 22+) | Igual |
| Python | 3.11+ con pip y venv | Igual |
| Rust / Cargo | No requerido | Rust estable |
| Dependencias nativas | Las de Python/PyTorch para tu sistema | WebKitGTK y herramientas de compilación en Linux; WebView2 y VS Build Tools en Windows |

```bash
git clone https://github.com/Maxdesigna7x/Model_Builder.git
cd Model_Builder
```

### Web local · Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
npm ci
bash ./run-web.sh
```

Abre [http://localhost:1420](http://localhost:1420). El script detecta `.venv` y usa su Python; no necesitas activar el entorno manualmente.

### Web local · Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
npm ci
$env:MODELBUILDER_PYTHON = (Resolve-Path '.\.venv\Scripts\python.exe').Path
npm run dev
```

Si usas otro entorno, establece `MODELBUILDER_PYTHON` con la ruta de su ejecutable antes de lanzar Vite. CUDA depende de una instalación de PyTorch compatible con la GPU y sus controladores.

### Escritorio

La preparación inicial del entorno privado permite elegir CPU o GPU NVIDIA en las versiones que incluyen este selector. La opción NVIDIA requiere una GPU y un controlador compatibles con CUDA 13.0; la descarga puede ocupar varios GB.

Para ejecutar desde el código fuente: en Ubuntu/Debian, `bash scripts/install-linux.sh` prepara el entorno y las dependencias de Tauri; después ejecuta `bash run.sh`. En Windows, ejecuta `scripts/install-windows.ps1` y luego `run-windows.ps1` desde PowerShell, con los requisitos nativos instalados.

Consulta [la guía de instalación](docs/INSTALACION.md). `npm run build` compila el frontend; no genera por sí solo un instalador Tauri ni incluye Python.

## Comprobaciones

Con el entorno Python configurado:

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
npm run build
npm run test:builder-graph
npm run test:structured-graph
npm run test:structured-ui
npm run test:repair-ui
npm run test:groups
npm run test:groups-ui
npm run test:cnn-reuse
npm run test:editor-ui
npm run test:kaggle-ui
```

En Windows usa `.\.venv\Scripts\python.exe` en lugar de `.venv/bin/python`. Las pruebas de interfaz requieren Chrome/Chromium y usan backend simulado; las pruebas Python comprueban el motor. Las descargas autenticadas y el empaquetado nativo requieren comprobaciones adicionales en su entorno.

El [informe de cierre CNN](docs/17-cierre-revision-cnn.md) recoge las comprobaciones realizadas y sus límites. Los documentos de planificación anteriores son históricos y pueden contener propuestas que todavía no están implementadas.

## Generar capturas de demostración

```bash
npm ci
CAPTURE_DIR=capturas/readme-demo npm run capturas:readme
```

El script abre una sesión aislada de Chrome/Chromium a **1600 × 1000**, usa el acento naranja, inicia Vite en un puerto libre y genera seis capturas de demostración. Estas capturas se generan por separado de las imágenes del README. No modifica proyectos reales ni descarga datasets o pesos. Si es necesario, define `CHROME_PATH` con la ruta del navegador. En PowerShell establece primero `$env:CAPTURE_DIR = 'capturas/readme-demo'` y luego ejecuta `node scripts/capture-readme.mjs`.

Para capturas adicionales del catálogo completo puedes usar `npm run capturas`; su salida por defecto se guarda en `capturas/`, ignorada por Git.

## Estructura del repositorio

```text
assets/             logo y recursos de la aplicación
frontend/           interfaz React, editor de grafos y gráficas
backend/            motor Python/PyTorch y adaptadores de datos
src-tauri/          shell de escritorio y puente de procesos
tests/              pruebas del motor
scripts/            instalación, capturas y comprobaciones de interfaz
docs/               documentación, informes y capturas del README
prototypes/         propuestas aisladas, aún sin integrar
workspace_data/     proyectos, datasets, cachés y corridas locales (ignorado)
```
