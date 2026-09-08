# Mejoras de UI y funcionalidad

## 1. Estructura global

- [x] Eliminar el scroll del documento en todas las pantallas.
- [x] Mantener el scroll únicamente dentro de las regiones de contenido que lo necesiten.
- [x] Verificar Modelos, Datos, Constructor, Entrenamiento e Inferencia en tamaños de ventana comunes.

## 2. Datos

- [x] Mantener estables el tamaño y la posición de los cards aunque cambie la cantidad de datos o imágenes.
- [x] Fijar el panel lateral como en Entrenamiento y dar estilo consistente a inputs/selects.
- [x] Añadir botón para volver a cargar el dataset actual.
- [x] Mostrar progreso y estado durante importación, generación y recarga.
- [x] Añadir selector de carpeta además de la entrada manual de ruta.

## 3. Constructor

- [x] Diferenciar con claridad hover, selección y error en nodos.
- [x] Mantener el estado visual de selección en conectores/edges.
- [x] Igualar el botón de eliminar de nodos al estilo outline/hover rojo de conexiones y centrar el icono.
- [x] Quitar “Eliminar bloque” del inspector lateral.
- [x] Mostrar parámetros estimados del nodo seleccionado y del modelo completo.
- [x] Validar en tiempo real y marcar los nodos con error.
- [x] Reforzar reglas de conexión: dirección, handles, duplicados, una entrada por handle y ciclos.
- [x] Revisar catálogo y categorías de bloques.
- [x] Añadir fichas informativas con explicación textual y diagrama de uso por bloque.
- [x] Añadir presets con descripción, ventajas/desventajas, aplicación y guardado de modelos propios.

## 4. Entrenamiento

- [x] Mostrar tooltips de métricas con cuatro decimales.
- [x] Soportar múltiples corridas mediante pestañas y botón “+”.
- [x] Conservar configuración, historia y resultado independiente por corrida.
- [x] Definir la corrida activa para inferencia mediante la pestaña seleccionada y resaltarla.
- [x] Reordenar el panel derecho: configuración arriba, resultados/tiempo después y acción abajo.
- [x] Añadir barra de progreso por épocas y estimación de progreso restante.

## 5. Inferencia

- [x] Usar el checkpoint de la corrida activa.
- [x] Unificar entrada y resultado en un único card.
- [x] Mostrar feedback verde/rojo cuando exista etiqueta real para comparar la predicción.

## 6. Verificación

- [x] Compilar frontend y ejecutar la suite del backend (la comprobación Tauri requiere las librerías GLib del sistema).
- [x] Ejecutar comprobaciones funcionales y visuales de los flujos principales.

## 7. Ajustes compactos de navegación y acabado

- [x] Corregir el tema de los selectores de modo en Inferencia y neutralizar estilos nativos del host.
- [x] Eliminar la barra superior del flujo y aprovechar su espacio en todas las etapas.
- [x] Mover Apariencia a la esquina superior de la barra lateral y dejar únicamente el icono de la app.
- [x] Hacer que el icono de la app vuelva a la lista de proyectos.
- [x] Añadir navegación global Anterior/Siguiente en la esquina superior derecha.
- [x] Ocultar automáticamente los avisos después de cinco segundos.
- [x] Corregir posición y tema del botón para eliminar proyectos.
- [x] Quitar el resumen del proyecto de la barra lateral.
- [x] Mostrar la guía de bloques al pasar el cursor y permitir fijarla con clic hasta pulsar fuera.

## 8. Rediseño de Inferencia

- [x] Convertir el card en un espacio de trabajo que ocupa el alto disponible.
- [x] Separar claramente la fuente de datos y el análisis del resultado.
- [x] Usar un selector de fuente tematizado para evaluación y entrada manual.
- [x] Adaptar el lenguaje y los controles a imágenes, valores tabulares y secuencias.
- [x] Presentar errores de inferencia como estado de error, sin confundirlos con una predicción.
- [x] Unificar entrada y resultado en una sola superficie secuencial, sin dividir el card en dos columnas.
- [x] Mantener los selectores de evaluación y carga manual en una misma fila.
- [x] Reservar una composición de dos columnas para previews de imagen: vista a la izquierda y resultado a la derecha.
- [x] Separar el CSS crítico de Inferencia en un recurso pequeño para el WebView de escritorio.
