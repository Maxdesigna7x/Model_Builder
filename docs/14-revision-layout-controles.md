# Revisión de layout y controles

**Fecha:** 2026-11-07  
**Estado inicial:** lista de trabajo antes de implementar esta revisión.  
**Alcance:** frontend existente de ModelBuilder. No editar `prototypes/` ni `node-help/`; son propuestas que se están generando por separado. Conservar todos los cambios locales existentes y no hacer commit ni push.

## Checklist

### A. Vista de datos

- [x] Evitar que el contenido de la vista de datos quede cortado por el borde inferior; permitir desplazamiento de página cuando haga falta.
- [x] Quitar las dimensiones originales enormes de la tarjeta intermedia. Mostrar metadatos compactos en el Explorador de datos o en un resumen de fuente; conservar la distinción entre tamaño fuente y tamaño para el modelo.
- [x] Revisar el tamaño de texto y la distribución de la vista con la tipografía mínima global.

### B. Biblioteca y grafo

- [x] Dar a las pestañas de la biblioteca de estilos una apariencia clara de selección y foco.
- [x] Al cambiar orientación, mover los puertos y sus anclas: horizontal usa entrada izquierda/salida derecha; vertical usa entrada arriba/salida abajo.
- [x] Dibujar los cables con la geometría de ancla correspondiente a la orientación activa.
- [x] Colocar los indicadores de puerto en el borde real de la tarjeta; los controles táctiles pueden ampliar el hit target sin mover el ancla.
- [x] Invocar `updateNodeInternals` cuando cambien orientación o tamaño del nodo para recalcular handles y aristas.

### C. Entrenamiento y pestañas

- [x] Mostrar la configuración de parada temprana fuera de Advanced.
- [x] Mostrar Detener, Pausar y Reanudar en el mismo footer que Train.
- [x] Permitir reentrenar un modelo con corrida completada creando otra corrida y conservando el historial anterior. Inspeccionar el flujo actual de entrenamiento antes de cambiarlo.
- [x] Dibujar el botón de cierre del modelo/corrida dentro de la silueta de su pestaña; controles separados y accesibles, sin botones anidados.
- [x] En el modal de presets, ofrecer dos vistas excluyentes: Scratch y Pretrained.
- [x] Quitar las pestañas decorativas Overview, Tasks y Details del selector de modelos.

### D. Tipografía y layout

- [x] Auditar todo el frontend: mínimo global de 12 px y 14 px para texto normal de lectura.
- [x] Ajustar espacios, tarjetas y controles donde la tipografía ampliada lo requiera; evitar clipping generalizado.

## Registro de implementación

Se completará esta lista por área, con archivos principales, comprobaciones ejecutadas y límites observados. Las verificaciones no deben modificar ni cubrir las propuestas en `prototypes/` o `node-help/`.

## Cierre de implementación

**Estado:** todos los criterios de esta revisión están completados.

- **Vista de datos:** la página ahora crece a su altura natural y permite desplazamiento del contenido. Las estadísticas de dimensiones originales están dentro del Explorador de datos, en un resumen compacto y separado del tamaño de entrada del modelo.
- **Biblioteca y puertos:** las pestañas tienen estados seleccionados/foco visibles. El constructor coloca los handles en izquierda/derecha en horizontal y arriba/abajo en vertical. `CustomEdge` recibe las posiciones de esos handles y calcula el trazado con la orientación activa. Los handles reales miden 12 px para que el extremo del cable toque el punto; un pseudo-elemento transparente de 36 px amplía el área interactiva, recortada hacia el exterior de la tarjeta. `useUpdateNodeInternals` vuelve a medir al cambiar orientación, propiedades, shape, puertos, etiquetas o etapas; cambiar orientación centra los dos primeros nodos a zoom 1.
- **Entrenamiento:** la configuración de parada temprana aparece antes de Advanced y especifica que observa pérdida de validación. Detener y Pausar/Reanudar están juntos en el footer. Al pulsar Train again en una corrida completada, `train()` añade una nueva corrida activa; la anterior mantiene su estado, historial y resultado.
- **Pestañas y presets:** el cierre se mantiene como botón hermano accesible dentro del contorno común de cada pestaña. El modal muestra Scratch o Pretrained como vistas excluyentes, con texto localizado. Se quitó la navegación decorativa Overview/Tasks/Details.
- **Tipografía:** los estilos de la app fijan 12 px como mínimo y 14 px para lectura y controles normales. Se ajustaron contenedores y vistas con clases específicas para permitir texto y paneles de mayor altura sin clipping general.

**Archivos de comportamiento:** `frontend/src/App.tsx`, `frontend/src/ModelNode.tsx`, `frontend/src/node-port-hit-area.css` y `frontend/src/i18n/locales/{en,es}/app.json`. Los estilos base y de revisión viven en `frontend/src/styles.css`, `frontend/src/ui-polish.css`, `frontend/src/inference-layout.css` y `frontend/src/revision14.css`; `frontend/src/main.tsx` importa la capa de revisión.

**Comprobaciones ejecutadas:** `npm run build` terminó correctamente (Vite mantiene el aviso existente de chunks mayores a 500 kB); ambos archivos JSON de traducción parsean; `git diff --check` pasó. Un fixture dirigido con Puppeteer inició una corrida desde otra completada y verificó que la corrida anterior conservó sus dos épocas, que se creó una corrida activa con historial vacío y que hubo una sola solicitud de inicio. El harness aislado del navegador revisó las vistas de datos, catálogo, presets, entrenamiento y ambas orientaciones; no registró errores de página ni texto visible por debajo de 12 px, y midió los extremos de los cables a no más de 1 px de sus puertos.

**Límites:** la comprobación de reentrenamiento y las capturas usan un backend simulado; verifican el estado de la UI y la solicitud que inicia la corrida, no la ejecución real de PyTorch. El monitor de parada temprana sigue siendo la pérdida de validación, de acuerdo con la implementación existente. No se tocaron ni se usaron como parte de esta implementación las propuestas en `prototypes/` o `node-help/`. No se hizo commit ni push.

## Revisión independiente de CSS y capturas

- `frontend/src/revision14.css` centraliza los ajustes de esta ronda; los estilos base ya no declaran tamaños de letra inferiores a 12 px. Texto de lectura y formularios usan 14 px; las etiquetas auxiliares usan al menos 12 px. El zoom manual del canvas sigue escalando el diagrama.
- `scripts/capture-revision14.mjs` abre una instancia aislada con datos de demostración y respuestas de backend simuladas, sin modificar proyectos reales. Captura catálogo, datos (inicio/final), ambas orientaciones, ambas vistas de presets y entrenamiento a 1280×800 y 1600×1000.
- Las capturas y sus auditorías están en `capturas/revision14/`. Los inventarios de texto no encontraron elementos inferiores a 12 px en las vistas capturadas. Las medidas de los centros de puertos coinciden con el borde de las tarjetas con tolerancia de 1 px.
- Se revisaron visualmente las capturas y se corrigieron el solapamiento de leyendas de muestras, estadísticas de resolución, tipos del pipeline y la separación de los wires respecto de los puertos.
- La prueba de interfaz de reentrenamiento conserva la corrida completada y sus tres epochs de historial, crea una corrida distinta y emite exactamente una solicitud de inicio. La captura `1280-running-controls.png` muestra Detener/Pausar en el footer.
- `npm run build` y `git diff --check` pasan. Vite conserva el aviso de bundles mayores a 500 kB. Estas comprobaciones no sustituyen una ejecución de entrenamiento real con GPU.
- Las propuestas animadas solicitadas se generan aparte en `prototypes/node-help/`; su integración en los tooltips queda reservada para una revisión posterior con el usuario.
