# Cierre de revisión CNN · 2026-11-07

## Correcciones finales

- Add y Concatenate no eran exclusivos de residuales: faltaban en la biblioteca CNN aunque existían en U-Net. Se exponen como operaciones independientes en Ramas, sin duplicarlas, en todas las familias del editor. Add requiere formas compatibles; Concat requiere coincidencia fuera del eje elegido.
- CNN también ofrece ConvTranspose2D, Resize2D y Reshape para construir decoders y tareas imagen-a-imagen sin cambiar de familia.
- Los nodos insertados desde la biblioteca quedan seleccionados para que sus propiedades y acciones contextuales estén disponibles inmediatamente.
- Inspector y nodos distinguen canales de entrada, filtros de salida y canales base.
- Pruebas de interfaz actualizadas para el selector de fuentes y diálogos propios de la app. La prueba de editor apunta al fixture vigente, con aserciones de inserción Add/Concat, dos entradas, acciones, zoom, plegado, presets y reentrenamiento.
- Las capturas de revisión aportadas por el usuario permanecen locales e ignoradas por Git, como las capturas generadas. El logo sí se incluye como recurso de la aplicación. Las ayudas animadas se conservan como propuesta separada bajo prototypes/node-help.

## Comprobaciones de cierre

- Backend completo: `python -m pytest -q`, 136 pruebas aprobadas.
- `npm run build`: aprobado; permanece el aviso de Vite sobre bundles mayores de 500 kB.
- `npm run test:cnn-reuse`: aprobado.
- `npm run test:builder-graph`: aprobado, incluye cobertura del catálogo de merges/decoders.
- `npm run test:editor-ui`: aprobado con backend simulado y navegador aislado a 1280×800 y 1600×1000; sin errores de página.
- `npm run test:kaggle-ui`: aprobado para importación y cambio posterior de splits, mediante backend simulado.
- `git diff --check`: aprobado.

No se ejecutó empaquetado nativo de Tauri ni descarga autenticada de Kaggle/preentrenados durante este cierre. Los límites ya descritos en docs/11–16 siguen documentados; estas comprobaciones no equivalen a probar cada arquitectura con un dataset real.
