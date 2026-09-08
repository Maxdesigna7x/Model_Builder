# 05 · Entrenamiento, evaluación e inferencia

**Estado actualizado:** recorrido base implementado y probado con entrenamiento, métricas por tarea, checkpoints `best`/`last`, evaluación e inferencia. Los controles avanzados de pausa/reanudación, schedulers, AMP, acumulación de gradientes y empaquetado multiplataforma continúan como criterios de ampliación.

## 1. Preparación de una corrida

Requiere tarea/datos confirmados, grafo sin errores, dry-run válido y dispositivo disponible. El resumen previo muestra revisión del grafo, dataset/pipeline/splits, número de parámetros, salida, loss, métrica principal, batch, epochs y memoria estimada.

Al iniciar se congela un snapshot. La UI puede editar un borrador nuevo mientras entrena, pero no modifica la corrida en marcha. Los gráficos identifican su run y revisiones para evitar atribuir resultados al grafo que se acaba de editar.

| Control | Valor inicial | Estado |
| --- | --- | --- |
| Dispositivo | CPU o GPU detectada | Implementado. `system.gpu` devuelve GPU si CUDA está disponible; en ausencia de CUDA informa CPU. |
| Epochs | 20 | Implementado. |
| Batch size | 32 (MLP/tabular), 16 (visión/secuencias/texto) | Implementado. |
| Optimizador | Adam | Implementado. SGD/AdamW no están disponibles. |
| Learning rate | 0.001 | Implementado. |
| Loss | Receta fija por tarea | Implementado. No hay alternativas de loss. |
| Métrica | Fija por tarea | Implementado. No hay métrica secundaria configurable. |
| Seed | Persistido (42 propuesto) | Implementado para Python, NumPy y PyTorch. |
| Scheduler | — | *No implementado.* |
| Early stopping | — | *No implementado.* |
| Gradient clipping | `max_norm=5.0` fijo | Implementado. No configurable en UI. |
| Precisión | FP32 | Implementado. AMP/BF16/FP16 *no implementados.* |
| Acumulación de gradiente | 1 | *No implementada.* |
| DataLoader workers | 0 | Implementado. |

Los presets son puntos de partida editables. «Rápido para probar» usa muestras/pasos limitados y se etiqueta como smoke experimental; no se confunde con entrenamiento completo.

## 2. Loss y métrica por tarea

| Tarea / objetivo | Salida y target | Loss | Métrica |
| --- | --- | --- | --- |
| Clasificación (binaria/multiclase) | Logits `[B,K]`; target entero `[B]` | `F.cross_entropy` | Accuracy |
| Regresión | Valores `[B,Q]`; target mismo shape | `F.mse_loss` | MSE |
| Pronóstico | `[B,P]`; target mismo shape | `F.mse_loss` | MSE |
| Segmentación binaria | Logit `[B,1,H,W]`; target 0/1 | `BCEWithLogits` + soft Dice | Dice |
| Segmentación multiclase | Logits `[B,K,H,W]`; target entero `[B,H,W]` | `F.cross_entropy` | Mean IoU (sin fondo) |
| Reconstrucción | Mismo shape que entrada | `F.mse_loss` | MSE |
| Texto clasificación | Logits `[B,K]`; etiqueta `[B]` | `F.cross_entropy` | Accuracy |
| Texto causal | Logits `[B,T,V]`; IDs desplazados `[B,T]` | `F.cross_entropy` ignorando `padding_idx=0` | Accuracy por token válido |

*Ampliaciones futuras:* pesos de clase, MAE/Huber, métricas secundarias, macro-F1, balanced accuracy, perplejidad, PSNR, pesos `pos_weight` y VAE con pérdida compuesta.

CE y BCEWithLogits reciben scores sin softmax/sigmoid terminal. La conversión a probabilidades pertenece a métricas/inferencia. El validador detecta una activación terminal incompatible. Esto coincide con los contratos de [CrossEntropyLoss](https://docs.pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html) y [BCEWithLogitsLoss](https://docs.pytorch.org/docs/main/generated/torch.nn.BCEWithLogitsLoss.html).

Clasificación binaria con dos logits en A usa CE igual que multiclase; segmentación binaria usa un logit por píxel y BCE. No mezclar ambas convenciones por compartir la palabra «binaria».

Para VAE, reconstrucción promedia elementos por muestra y luego batch; KL suma dimensiones latentes y promedia batch. Beta se guarda con esa convención; cambiar la reducción cambia el experimento. Se registran total, reconstrucción y KL por separado. Perplejidad se deja como no finita/fuera de rango si la exponencial desborda, sin truncar silenciosamente.

## 3. Definición de métricas y agregación

- Loss de epoch: promedio de la loss por batch.
- Accuracy: aciertos / muestras válidas (en texto, ignora padding 0).
- MSE: error cuadrático medio sobre el batch.
- Dice: coeficiente Dice sobre el foreground en segmentación binaria.
- Mean IoU: IoU promediado sobre las clases > 0 en segmentación multiclase.

*Ampliaciones futuras:* ponderación por tamaño de batch, métricas por clase, pesos de clase, umbrales optimizables y métricas secundarias.

## 4. Telemetría y gráficas en vivo

Loss es la gráfica principal, con train y validation. Train puede aportar puntos por step y resúmenes por epoch; validation se calcula al cierre de epoch por defecto. No unir ni interpolar puntos de validation como si se hubieran medido cada batch. Cada serie conserva eje/unidad y frecuencia de muestreo.

La métrica secundaria muestra train/validation agregadas cuando están disponibles. Tooltip con step/epoch, timestamp y valor bruto. Interruptores de suavizado, escala logarítmica sólo para valores positivos y comparación de corridas compatibles. Suavizar sólo cambia la presentación; no modifica los datos ni el monitor de early stopping.

Controles: zoom, reset, seguir en vivo, ampliar gráfica, ocultar serie y exportar CSV/imagen. Comparar corridas exige mostrar diferencias de dataset/split/receta; no declarar una mejora comparable entre evaluaciones incompatibles.

Indicadores: epoch actual, batch/optimizer step, muestras o tokens por segundo, LR, tiempo activo, fase, memoria medida y estimada. ETA sólo después de suficiente observación, con etiqueta aproximada; pausa no cuenta como tiempo activo. Sin datos suficientes mostrar «Calculando».

El backend puede agrupar eventos de progreso a 2–5 Hz como objetivo inicial; persistir todos los resúmenes de epoch y puntos configurados. La UI usa buffers y reducción de puntos para historiales largos, conservando picos y rangos. No repintar todo el canvas con cada evento de entrenamiento.

## 5. Ciclo de vida

El entrenamiento corre hasta completar las epochs configuradas o fallar. No hay pausa/reanudación ni cancelación cooperativa en esta versión.

| Estado | Descripción |
| --- | --- |
| Preparada | Configuración válida; al iniciar se congela el snapshot del grafo y dataset. |
| Entrenando | Se ejecuta train + validation por epoch; se emiten eventos de progreso. |
| Completada | Se guardan `best.pt` y `last.pt`; se evalúa test si existe. |
| Fallida | Error reportado (OOM, shapes, NaN, etc.); se conserva el último checkpoint válido si lo hubo. |

## 6. Checkpoints

Cada corrida guarda dos checkpoints:

- `best.pt`: pesos con la mejor métrica de validation.
- `last.pt`: pesos del final del entrenamiento.

Contenido guardado: `model_state`, `optimizer_state`, `epoch`, `history`, `task_id`, `architecture`, `graph`, `dataset` (referencia) y metadatos del proyecto.

*Limitación actual:* el checkpoint no incluye scheduler, AMP, seeds ni cursor de datos, por lo que **no se soporta reanudación** de una corrida interrumpida. Si falla, se puede reiniciar una nueva corrida desde cero o desde `last.pt` como pesos iniciales (sin garantía de reproducibilidad exacta).

## 7. Evaluación final

Validation gobierna selección y early stopping. Al completar, evaluar una vez test con best, si existe test; guardar un artefacto independiente del log de entrenamiento. Si se evalúa otro checkpoint, registrar nueva evaluación con su identidad, sin sobrescribir la anterior.

Resultados: matriz de confusión y métricas por clase para clasificación; dispersión predicción/objetivo y residuales para regresión; error por horizonte para pronóstico; Dice/IoU y galería de máscaras para segmentación; reconstrucción y error para AE/VAE; CE/perplejidad y muestras para texto causal.

Explorar repetidamente test puede influir en decisiones humanas: la UI lo presenta como conjunto reservado y no lo introduce en curvas de ajuste. Si no hay test, el informe dice «Sin evaluación independiente»; no renombra validation como test.

## 8. Inferencia común

Elegir checkpoint; cargar su grafo y pipeline; ejecutar `model.eval()` sin gradientes; transformar y presentar el resultado.

Fuentes: entrada manual (tabla, texto, serie) o muestra aleatoria de una partición. La muestra aleatoria no consume ni cambia el dataset. Si hay etiqueta real, se muestra comparación; si no, solo la predicción.

La predicción usa el contrato del checkpoint, incluso si el proyecto actual cambió después de guardarlo.

## 9. Experiencia por familia/tarea

| Caso | Entrada | Resultado |
| --- | --- | --- |
| MLP clasificación | Formulario de variables | Clase y probabilidades |
| MLP regresión | Formulario de variables | Valor y residual si hay etiqueta |
| CNN/ViT | Imagen | Probabilidades o valor |
| LSTM/CNN1D/Transformer | Formulario de serie o muestra de partición | Clase/valor/horizonte |
| U-Net | Imagen | Máscara predicha y overlay |
| Autoencoder | Vector/imagen | Reconstrucción y error |
| Transformer clasificación de texto | Texto libre | Probabilidades por clase |
| Transformer causal | Texto libre | Tokens predichos (greedy `argmax`; sin sampling todavía) |

*Ampliaciones futuras:* controles de sampling (temperatura, top-k, top-p), restauración de geometría original en segmentación y exportación de resultados.

## 10. Recursos y errores

Una tarea pesada por dispositivo en A: inferencia sobre el mismo dispositivo que entrena espera o permite pausar el entrenamiento; no ocupar VRAM adicional silenciosamente. La entrada de inferencia sigue editable mientras espera y puede cancelarse.

Errores diferenciados: OOM, datos no finitos, pérdida/gradientes no finitos, checkpoint incompleto, source ausente, dtype no soportado, worker caído, disco lleno, incompatibilidad de versión y operación no soportada por el dispositivo. Mostrar causa y acciones concretas; logs técnicos ampliables, sin traceback ocupando toda la pantalla.

Ante OOM: conservar último checkpoint, mostrar batch/estimación/medición y sugerir reducir batch, resolución o modelo. El usuario puede crear corrida derivada con los cambios; no fingir que una corrida fallida continuó sin alterar configuración.
