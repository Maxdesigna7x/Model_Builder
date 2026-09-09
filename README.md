# ModelBuilder

Aplicación visual local para crear proyectos, elegir una tarea compatible, preparar datos, construir una red mediante bloques, entrenarla con PyTorch y ejecutar inferencia desde un checkpoint.

La base usa **React + TypeScript + React Flow + ECharts**, un shell **Tauri 2** y un motor local **Python/PyTorch**. El catálogo cubre MLP, CNN 1D/2D (incluyendo un preset con bloque residual editable), RNN/GRU/LSTM, Transformer encoder, Transformer causal, ViT, U-Net y autoencoders.

![Entrenamiento en tiempo real](docs/img0.png)

## Funciones implementadas

- Biblioteca y creación de proyectos con recuperación del estado de edición.
- Selector por Tabla, Señales y series, Visión, Texto y Reconstrucción; una tarea puede ofrecer varias arquitecturas compatibles.
- Trece tareas: clasificación, regresión, pronóstico, segmentación, reconstrucción y modelado de tokens.
- Catálogo de datasets públicos de Hugging Face compatible con las trece tareas, descarga bajo demanda, caché física, revisiones fijadas e importación de datos propios.
- Particiones train/validation/test editables arrastrando sus dos límites; el reparto elegido se persiste y se aplica al entrenamiento real.
- Constructor drag-and-drop con puertos, conexiones, biblioteca por arquitectura, edición de propiedades e inspector derecho.
- Modelo inicial compacto para cada combinación arquitectura–tarea, ajustado automáticamente a las dimensiones, clases, vocabulario y horizonte del dataset elegido.
- Compilación topológica del canvas a módulos PyTorch. Las ramas U-Net/ResNet y `Concat`/`Add` se ejecutan según las conexiones dibujadas.
- Bloques reales de Conv1D, pooling 1D, padding causal, RNN, GRU, LSTM, embedding, posición, atención encoder/causal, pooling de tokens y patch embedding de ViT.
- Validación de ciclos, nodos desconectados, shapes de salida y dry-run real.
- Entrenamiento PyTorch en un proceso separado, con loss train/validation, métrica de tarea, checkpoint `best`/`last` y evaluación test final.
- Inferencia desde el checkpoint guardado y el mismo contrato de datos.
- Tema oscuro de gris neutro inspirado en `references/`, tema claro y cuatro acentos persistentes: azul, naranja, verde y violeta.

VAE y Transformer encoder–decoder permanecen deliberadamente fuera del catálogo disponible: requieren salidas múltiples/pérdida KL y cross-attention/generación respectivamente. No se muestran como funciones parciales. El estado detallado está en [`ARCHITECTURE_TODO.md`](ARCHITECTURE_TODO.md).

![Selector de tarea y arquitectura](docs/img2.png)

![Constructor de red con bloques](docs/Img1.png)

### Datasets públicos y datos propios

La pantalla Datos ofrece fuentes públicas compatibles para la tarea seleccionada. Antes de usar la red, la app comprueba si el dataset ya está materializado en el proyecto o presente en `workspace_data/cache/huggingface`; si falta, descarga una revisión fijada y la convierte al contrato interno. No se ejecuta código remoto del repositorio. Para datos propios siguen disponibles `label<TAB>texto` en clasificación y una muestra por línea en modelado causal.

## Requisitos e instalación

| Componente | Requisito |
| --- | --- |
| Node.js y npm | Node.js 20+ |
| Python | 3.11+ con pip y venv |
| Rust y Cargo | Stable |
| Linux | Ubuntu/Debian |
| Windows | Windows 10/11 + WebView2 + VS Build Tools 2022 + Windows SDK |

Clona el repositorio:

```bash
git clone https://github.com/Maxdesigna7x/Model_Builder.git
cd Model_Builder
```

### Linux

```bash
chmod +x scripts/install-linux.sh run.sh run-web.sh
./scripts/install-linux.sh
```

### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install-windows.ps1
```

Los instaladores validan Python, Node.js y Rust, crean el entorno .venv, instalan el backend y ejecutan npm ci con el lockfile. Son repetibles. Consulta [docs/INSTALACION.md](docs/INSTALACION.md) para los detalles.

## Cómo ejecutar

### Web con Vite

```bash
npm run dev
```

En Linux también puedes ejecutar ./run-web.sh. Abre http://127.0.0.1:1420.

### Aplicación de escritorio Tauri

Linux:

```bash
./run.sh
```

Windows:

```powershell
.\run-windows.ps1
```

Los comandos Tauri usan automáticamente el Python de .venv mediante MODELBUILDER_PYTHON.

## Validación

```bash
npm run build
pytest
python3 backend/modelbuilder/engine.py <<<'{"action":"hello"}'
```

La primera descarga de un dataset requiere conexión HTTPS a `huggingface.co`. Las siguientes preparaciones reutilizan la caché física. Los datasets médicos y algunos corpus tienen licencias de uso educativo/no comercial; la app muestra la licencia junto al tamaño antes de descargar.

La suite incluye las rutas originales, cobertura del catálogo público y de su reutilización física, pruebas adicionales de forward/backward para cada familia y un recorrido real entrenamiento→checkpoint→inferencia de texto.

## Estructura

```text
frontend/           React, canvas y pantallas
backend/            motor Python/PyTorch
src-tauri/          shell y puente de procesos
tests/              validación del motor y recorridos
docs/               especificación de producto y arquitectura
references/         referencias visuales conservadas
workspace_data/     proyectos generados localmente en ejecución (ignorado por Git)
```

La especificación completa comienza en [docs/01-producto-y-alcance.md](docs/01-producto-y-alcance.md). El plan y sus criterios están en [docs/07-plan-y-aceptacion.md](docs/07-plan-y-aceptacion.md).
