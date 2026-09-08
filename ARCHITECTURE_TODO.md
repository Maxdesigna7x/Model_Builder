# Ampliación de arquitecturas

Este documento controla la ampliación iniciada después del respaldo `ModelBuilder-V1`.
Una tarea sólo se marca terminada cuando funciona en catálogo, datos, canvas, backend y pruebas.

## Fase 0 · Seguridad e inventario

- [x] Crear y verificar `/home/randy/Python/ModelBuilder-V1`.
- [x] Revisar selección de tarea, persistencia, datos, compilador, entrenamiento e inferencia.
- [x] Sustituir la relación rígida tarea→arquitectura por una matriz de compatibilidad.
- [x] Mantener compatibilidad con proyectos guardados que usan `cnn`, `lstm` y `unet`.

## Fase 1 · Señales y series

- [x] Añadir CNN 1D con Conv/Pool/AdaptivePool/normalización y adaptación de ejes explícita.
- [x] Convertir LSTM en la familia visual “Red recurrente”.
- [x] Añadir bloques y presets RNN, GRU, LSTM y BiLSTM.
- [x] Comprobar clasificación, regresión y pronóstico con datos sintéticos e importados.

## Fase 2 · Transformers

- [x] Añadir proyección/embedding, posición, encoder, máscara causal y pooling de secuencia.
- [x] Añadir Transformer encoder para secuencias numéricas.
- [x] Añadir ViT con patch embedding visible y Transformer encoder reutilizado.
- [x] Añadir Transformer causal para modelado de tokens.
- [x] Añadir importación de corpus local y vocabulario reproducible.
- [x] Dejar encoder–decoder/cross-attention como ampliación posterior explícita.

## Fase 3 · Visión y representación

- [x] Añadir ResNet como presets editables de CNN 2D con ramas y `Add` visibles.
- [x] Reetiquetar U-Net como familia encoder–decoder visual sin romper su ID persistido.
- [x] Añadir autoencoder denso y convolucional con tareas de reconstrucción.
- [x] Documentar el contrato de salidas múltiples requerido por VAE; no simularlo como AE normal.

## Fase 4 · Experiencia y validación

- [x] Reorganizar selector por Tabla, Señales y series, Visión, Texto y Reconstrucción.
- [x] Mostrar únicamente combinaciones tarea–arquitectura realmente ejecutables.
- [x] Documentar qué tareas admiten datos sintéticos, importados u online.
- [x] Probar compilación TypeScript, suite Python, dry-runs y entrenamientos mínimos.
- [x] Actualizar README y documentación de alcance real.

## Fuera de esta entrega

- Transformer encoder–decoder para traducción/resumen.
- VAE completo con pérdida KL y salidas múltiples.
- Detección de objetos, GNN y modelos de difusión.

Estos elementos requieren contratos nuevos y no deben aparecer como disponibles antes de
contar con datos, entrenamiento e inferencia completos.
