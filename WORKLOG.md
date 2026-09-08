# Registro de mejoras de interfaz y flujo

Estado actualizado: 2026-09-07

- [x] Mostrar entrada y objetivo/máscara en las vistas previas del dataset.
- [x] Mostrar entrada y máscara predicha en inferencia de segmentación.
- [x] Quitar las métricas redundantes del lateral de entrenamiento y mostrar mejores valores por gráfica.
- [x] Cambiar el valor inicial de épocas a 20 (cliente y motor).
- [x] Ajustar la vista de datos: cabecera compacta y columnas de tamaño estable.
- [x] Compactar y hacer escalables las tarjetas de arquitecturas.
- [x] Añadir modelos múltiples por proyecto, con pestañas en Constructor y selector en Entrenamiento.
- [x] Vincular de forma persistente cada corrida con el modelo seleccionado.
- [x] Verificar visualmente el flujo completo en la aplicación.

## Correcciones de revisión visual

- [x] Corregir el desbordamiento de la cabecera de métricas en Datos.
- [x] Mantener captions y nombre de muestra dentro de cada preview.
- [x] Extender Datos al ancho útil y anclar su panel de propiedades al extremo derecho.
- [x] Convertir las notificaciones en avisos flotantes que no desplazan el contenido.
- [x] Rediseñar Inferencia con dos mitades fijas y eliminar la tercera copia de la entrada.
- [x] Reorganizar Inferencia verticalmente: selector superior, lienzo central y acción inferior fija.
- [x] Eliminar la regla heredada de dos columnas que rompía la composición vertical de Inferencia.
- [x] Igualar el botón de nuevo modelo al botón compacto de nueva corrida.

Notas: las dimensiones de salida se sincronizan con el dataset antes de validar o entrenar; la cabeza de salida no se edita manualmente.
