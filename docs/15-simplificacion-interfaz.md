# Simplificación de la interfaz · 2026-11-07

Esta ronda corresponde a los comentarios 1–20 del navegador, independientemente de los planes anteriores.

- Pipeline: tarjetas finas con icono y nombre; retirar tipo y split de la tarjeta.
- Explorador: tabs Info, Samples y Charts. Dimensiones originales únicamente en Info.
- Biblioteca: tabs discretos arriba; ambos paneles se pueden plegar desde el borde exterior y empiezan abiertos.
- Canvas: acciones contextuales de agrupar/desagrupar, duplicar y borrar sobre la selección, con tamaño constante en pantalla.
- Validación automática: indicador de estado en el canvas y sin botón manual.
- Guardado de arquitectura: solicitar un nombre y ofrecer el preset guardado.
- Tensor: una línea sencilla por valor/eje.
- Tabs de entrenamiento de menor altura, sin oclusión.
- Retirar títulos introductorios de etapas, badges Popular y texto FLOW.
- Sustituir marca temporal por icono neutro hasta recibir el definitivo.
- Inicio: cabecera mínima y panel de proyectos amplio, preservando búsqueda, filtros y mosaico; retirar hero y dibujo decorativo. La referencia final y el icono quedan pendientes de que el usuario los proporcione.

Estado: cambios de esta ronda implementados y revisados. La composición definitiva del inicio y su marca quedan a la espera de la referencia/icono que el usuario anunció; se aplicó mientras tanto cabecera mínima y panel de proyectos amplio. Las propuestas de ayudas visuales siguen aisladas del producto.

## Comprobaciones

- `scripts/capture-revision15.mjs` genera capturas y auditorías en `capturas/revision15/`, usando proyectos y backend simulados en navegador aislado.
- Revisión visual a 1280×800 y 1600×1000 de inicio, catálogo, datos (Info/Samples), grafo horizontal/vertical, presets y entrenamiento.
- La barra de selección mantiene exactamente sus dimensiones después de variar el zoom. Se probaron plegado/reapertura de ambos paneles y el guardado de un preset con nombre.
- Los puertos conservan su alineación con el borde de cada nodo. Las vistas capturadas no tienen texto CSS inferior a 12 px ni errores de JavaScript.
- El reentrenamiento conserva la corrida completada y sus epochs, crea una corrida distinta y envía una sola solicitud de inicio.
- `npm run test:builder-graph` pasa. Se actualizó una expectativa antigua: la duplicación ya preservaba la conexión de entrada externa desde la ronda de grupos; se comprueba que esa entrada se copia y que un consumidor externo de entrada única no se reconecta.
- `npm run build` y `git diff --check` pasan. Vite conserva su aviso de chunks mayores a 500 kB. No se hizo commit ni push.

## Archivos principales

`frontend/src/App.tsx`, `ModelNode.tsx`, `tensor-shape.tsx`, `canvas-actions.css`, `revision15.css`, `main.tsx`, locales en/es de app y los scripts de captura/prueba indicados. Las acciones flotantes reemplazan los botones individuales de borrado, incluidos los de marcos de grupo; el nombre de un grupo se renombra con doble clic.
