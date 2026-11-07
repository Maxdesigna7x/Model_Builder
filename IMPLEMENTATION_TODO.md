# ModelBuilder · plan de implementación verificable

Este archivo es la fuente de verdad de la ampliación de modelos preentrenados y del
ciclo de entrenamiento. Una casilla sólo se marca cuando existe implementación,
prueba automatizada y documentación coherente.

## P0 · Confianza y correcciones

- [x] Reportar el dispositivo real; nunca inventar una GPU cuando CUDA no existe.
- [x] Hacer que `graph.validate` ejecute forward, loss y backward y valide gradientes finitos.
- [x] Respetar la dirección de cada métrica (maximizar accuracy/Dice; minimizar MSE).
- [x] Sembrar Python, NumPy, CPU y CUDA y persistir el estado reproducible.
- [x] Añadir pruebas de regresión para los puntos anteriores.

## P1 · Contratos y catálogo autoritativo

- [x] Crear un catálogo backend único para tareas, arquitecturas, bloques y modelos.
- [x] Exponer el catálogo mediante `catalog.get` y consumirlo desde la UI sin romper el modo web.
- [x] Versionar el esquema de modelo y añadir un hash semántico que invalide validaciones obsoletas.
- [x] Ampliar `ProjectModel` con origen, implementación, revisión, procesador, cabeza y política de congelamiento.
- [x] Definir puertos con nombre, tipo, dtype y shape; aceptar batches y outputs estructurados.
- [x] Mantener migración compatible con proyectos existentes.

## P2 · Hugging Face y modelos preentrenados

- [x] Crear catálogo curado y fijado por revisión, tarea y familia.
- [x] Mostrar repo, licencia, tamaño, biblioteca, processor, estado de caché y limitaciones.
- [x] Descargar de forma reproducible, soportar offline y evitar ejecución remota por defecto.
- [x] Preferir `safetensors`, validar modelos gated y registrar aceptación/licencia.
- [x] Implementar adaptadores Transformers para clasificación/regresión visual.
- [x] Implementar adaptadores Transformers para clasificación y modelado causal de texto.
- [ ] Exponer operaciones internas reales de backbones como subgrafos editables; hoy
      «Ver etapas» en nodos HF sólo muestra nombres de etapas.
- [x] Crear cabeza compatible con el dataset y distinguir pesos nuevos/preentrenados.
- [x] Mostrar parámetros totales, entrenables, congelados y memoria estimada.
- [x] Proveer alternativas explícitas “desde cero” donde no exista transferencia universal (MLP, RNN, CNN1D y AE).
- [x] Añadir tests sin red con modelos diminutos/mocks y pruebas opcionales de integración con el Hub.

## P3 · Procesadores y flujo de datos

- [x] Tratar tokenizer/image processor como artefactos versionados del modelo.
- [x] Conservar texto crudo para poder retokenizar con el checkpoint seleccionado.
- [x] Soportar batches como `input_ids`, `attention_mask`, `token_type_ids` y `pixel_values`.
- [x] Ajustar resize/crop/normalización al processor y evitar doble normalización.
- [x] Validar canales, resolución, vocabulario, clases y taxonomías antes de entrenar.
- [x] Persistir transformaciones ajustadas sólo con train y reutilizarlas en inferencia.

## P4 · Entrenamiento completo

- [x] Separar visualmente datos, modelo y objetivo/receta de entrenamiento.
- [x] Implementar congelamiento total, por etapas y descongelamiento gradual.
- [x] Implementar grupos de parámetros y LR distinto para backbone/cabeza.
- [x] Añadir schedulers, warmup y registro de LR.
- [x] Añadir early stopping.
- [x] Añadir AMP FP16/BF16 con fallback seguro.
- [x] Añadir acumulación de gradientes y clipping configurable.
- [x] Añadir losses configurables y métricas múltiples por tarea.
- [x] Añadir class weights/samplers para desbalance.
- [x] Implementar reanudación exacta con optimizer, scheduler, scaler y RNG.
- [x] Añadir recetas: cabeza, gradual, completo, LoRA y smoke test.
- [x] Integrar PEFT/LoRA donde la arquitectura sea compatible.
- [x] Añadir pausa y cancelación cooperativa.

## P5 · Evaluación, inferencia y ciclo de vida

- [x] Guardar manifiesto completo: modelo, dataset, split, código, dependencias, seed y hardware.
- [x] Separar evaluaciones de validation y test y evitar sobrescribir resultados.
- [x] Añadir métricas por clase, matriz de confusión, residuales y galerías según tarea.
- [x] Comparar corridas sólo cuando sus contratos sean compatibles mediante una huella de tarea, dataset, split, clases y métrica.
- [x] Exportar bundle local con processor/tokenizer, pesos, etiquetas y configuración.
- [x] Importar/exportar bundles y usar `save_pretrained`/`safetensors` para los componentes Hugging Face cuando aplique.
- [x] Generar Model Card local; cualquier publicación al Hub queda fuera del flujo automático y exige una acción explícita futura.
- [x] Añadir registro local de modelos, alias best/last y lineage entre base/adapters.

## P6 · Seguridad, UX y calidad

- [x] Mantener `trust_remote_code=false` por defecto y mostrar advertencia si se solicita.
- [x] Fijar revisiones a commits, verificar descargas y controlar caché/espacio en disco.
- [x] Gestionar tokens sin guardarlos en proyectos ni logs.
- [x] Mostrar claramente licencia, uso previsto, limitaciones y requisitos de hardware.
- [ ] Añadir estados de descarga, cancelación, errores recuperables y modo offline.
- [x] Sincronizar documentación con comportamiento mediante pruebas de contrato.
- [x] Ejecutar pytest y TypeScript build.
- [ ] Comprobar el lienzo de CNN en una app ejecutándose; el build y un mockup no
      verifican el cableado/zoom real de React Flow.

## Catálogo inicial propuesto

- [x] CNN visión: ResNet-18, MobileNetV2 y ConvNeXT Tiny.
- [x] ViT: ViT Base y DeiT Small.
- [x] Texto encoder: DistilBERT, BERT multilingüe y BETO.
- [x] Texto causal: DistilGPT-2 y GPT-2.
- [ ] Segmentación: añadir sólo modelos con taxonomía/procesador compatible; no etiquetar SegFormer como U-Net.
- [x] Series: mantener presets desde cero hasta disponer de checkpoints con contrato de frecuencia/canales explícito.
- [x] Tabular/autoencoder/recurrentes: mantener recetas desde cero y explicar por qué no existe un checkpoint universal.

## Estado de cierre CNN · 2026-11-07

El plan específico de CNN y el estado por recorrido viven en
[`docs/09-plan-cierre-cnn.md`](docs/09-plan-cierre-cnn.md), secciones 15–18.
La implementación cubre clasificación, localización, detección de una escala,
segmentación U-Net, pares imagen→imagen, mapa de formas, residual/Concat, mapas
multi-escala, congelamiento por capa y reutilización de un checkpoint propio como
backbone con pesos parciales verificados.

Los pendientes funcionales comprobables siguen en la matriz C01–C16: revisión visual
React Flow/Tauri, fixtures de migración antiguos, bindings genéricos de pérdidas por
cabeza, selección simultánea de varios mapas al derivar un modelo y prueba portable
de bundles para todas las tareas CNN. El detector didáctico marca error si dos objetos
caen en la misma celda; no debe describirse como un YOLO de producción. El mockup del
nuevo estilo de nodos se muestra al usuario antes de aplicar ese rediseño.
