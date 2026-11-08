# Inicio centrado y distribución de datos · 2026-11-07

Nueva ronda de comentarios 1–12:

- Inicio: logo grande centrado, nombre debajo, nuevo proyecto debajo de la marca y proyectos en el espacio restante.
- Marca compartida en sidebar. Color por máscara alfa CSS usando el PNG original y el acento del tema; no se afirma que sea un SVG vectorizado.
- Datos: pipeline compacto arriba, explorador flexible al centro, splits compactos al pie y panel de propiedades de altura completa. Desplazamiento interno cuando hace falta, y alternativa de scroll para ventanas muy bajas.
- Samples: refresco aleatorio real de datos persistidos, con un botón de dado.
- Active node al principio del inspector.
- Retirar navegación Next/Previous; sidebar gobierna etapas. Tabs de modelos más bajos.
- Acciones de selección sin tarjeta padre visible, botones cuadrados alineados e iconos diferentes para duplicar/agrupar/desagrupar/eliminar.
- Retirar encabezado de inferencia y reducir ancho del botón de evaluar.
- Autoorden del modelo con huecos longitudinales de 56 px y laterales de 36 px.

Estado: completado y revisado. Propuestas animadas de ayudas ya disponibles aparte en `prototypes/node-help/`, aún sin integrar.


## Evidencia de revisión

- `npm run build`, `npm run test:builder-graph` y `git diff --check` pasan (Vite conserva el aviso de bundles mayores a 500 kB).
- `scripts/capture-revision16.mjs` guarda capturas en `capturas/revision16/` a 1280×800 y 1600×1000. Backend simulado para aislar proyectos reales.
- Verificaciones de navegador: splits y propiedades comparten borde inferior (diferencia <2 px) y conservan margen de 10–30 px; botón de inferencia <=340 px; encabezado de inferencia ausente; refresco de muestras actualiza índices; sin errores de página.
- Se mantienen las comprobaciones de tamaño constante de acciones flotantes al variar zoom, plegado/reapertura de paneles, guardado con nombre y conservación de corridas al reentrenar.
- Muestreo backend comprobado con tensores persistidos: ocho índices únicos y previews válidos. Se comprobó además la entrada por CLI (`python backend/modelbuilder/engine.py`), que es la usada por Vite, y se corrigieron las importaciones para ese modo de ejecución. El orden aleatorio se conserva para que mostrar cuatro de las ocho muestras no favorezca índices bajos.
- La máscara alfa usa el PNG original de 1254×1254: permite cambiar el color con el acento sin alterar el archivo. No es una vectorización; a los tamaños de interfaz usados no requiere una nueva versión SVG.
- Ventanas de altura inferior a 620 px usan scroll para mantener accesibles los controles.
- Sin commit ni push.
