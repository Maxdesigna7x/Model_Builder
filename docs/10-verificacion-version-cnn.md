# Verificación de la versión CNN

Fecha: 2026-11-07. Este registro acompaña el commit de la implementación CNN y
conserva los límites comprobados de la entrega.

## Comprobaciones aprobadas

- `python3 -m pytest -q`: 109 pruebas aprobadas.
- `npm run test:cnn-reuse`: extracción de backbones, skips, puertos, cabezas y
  restricciones de tamaño aprobadas.
- `npm run build`: TypeScript y build de producción aprobados. Vite advierte de
  dos chunks superiores a 500 kB; optimizar su descarga queda pendiente.
- `git diff --check`: aprobado.
- Capturas en Chromium de los nueve constructores y las pantallas de datos,
  entrenamiento e inferencia; revisión de CNN y U-Net a 1600×1000 y 1280×800.
  El script falla si el navegador produce una excepción sin manejar. Las
  capturas usan datos de demostración y no prueban el backend.
- Comprobación HTTP contra Vite y Python reales, con workspace temporal aislado:
  `hello`, `catalog.get`, `model.catalog`, `system.gpu`, `project.list`,
  `project.create`, `data.generate`, `graph.validate` e `inference.run`.
- Una CNN clasificadora completó entrenamiento CPU de una época por
  `/api/start-training`; `/api/training-events` entregó su evento de finalización
  y la inferencia posterior leyó el checkpoint y devolvió una clase válida.
- Las rutas de API rechazan GET con HTTP 405 donde requieren POST. El fallback
  SPA devuelve HTML en rutas anidadas. La navegación funcional usa el estado
  interno de la aplicación, no esas rutas anidadas.

Para repetir las capturas:

```bash
npm run capturas
CAPTURE_WIDTH=1280 CAPTURE_HEIGHT=800 npm run capturas
```

## Comprobación de escritorio bloqueada por el entorno

`cargo check --manifest-path src-tauri/Cargo.toml` no pudo completar: faltan las
bibliotecas de desarrollo GTK/GIO, GLib, Cairo y GDK Pixbuf detectadas mediante
`pkg-config`. No se certifica la compilación o ejecución nativa de Tauri en esta
máquina. Instalar los requisitos de Linux documentados y repetir cargo check y
una prueba de la app empaquetada en el entorno de publicación.

## Límites vigentes

La matriz C01–C16 y los siguientes pasos están en
[el plan de cierre](09-plan-cierre-cnn.md).
Permanecen pendientes el cableado manual de React Flow, fixtures integrales de
migración antigua, equivalencia portable en todas las tareas, bindings arbitrarios
de pérdidas por cabeza y elección simultánea de mapas al derivar un backbone.
El detector didáctico utiliza una escala y una caja por celda, y rechaza colisiones.
Los backbones de Hugging Face permiten inspeccionar etapas; no ofrecen sus
operaciones internas como subgrafos editables. El rediseño general de las tarjetas
sigue en vista previa para que el usuario elija el tratamiento visual.
