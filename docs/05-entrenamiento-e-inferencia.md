# 05 · Entrenamiento, evaluación e inferencia

**Estado actualizado:** recorrido base de la entrega A implementado y probado con entrenamiento, métricas por tarea, checkpoints `best`/`last`, evaluación e inferencia. Los controles avanzados de pausa/reanudación, schedulers, AMP y empaquetado multiplataforma continúan como criterios de ampliación.

## 1. Preparación de una corrida

Requiere tarea/datos confirmados, grafo sin errores, dry-run válido y dispositivo disponible. El resumen previo muestra revisión del grafo, dataset/pipeline/splits, número de parámetros, salida, loss, métrica principal, batch, epochs y memoria estimada.

Al iniciar se congela un snapshot. La UI puede editar un borrador nuevo mientras entrena, pero no modifica la corrida en marcha. Los gráficos identifican su run y revisiones para evitar atribuir resultados al grafo que se acaba de editar.

| Control | Propuesta inicial | Validación / efecto |
| --- | --- | --- |
| Dispositivo | CPU o GPU disponible; selección explícita con recomendación | Mostrar dispositivo real y memoria libre; no cambiar a CPU silenciosamente tras un error |
| Epochs | 20, entero positivo editable | Cada epoch recorre train; pasos estimados derivados del sampler y batch |
| Batch size | 32 en MLP, 16 en visión pequeña/secuencias; ajustar preset | Entero positivo; advertencia y dry-run de memoria |
| Optimizador | AdamW; alternativas Adam y SGD | Parámetros dependientes del tipo |
| Learning rate | 0.001 como punto de partida | Campo numérico + slider logarítmico; no prometer óptimo universal |
| Weight decay | 0.0001 sugerido | No negativo; explicar diferencia de AdamW frente a regularización de otros optimizadores |
| SGD avanzado | Momentum, dampening, Nesterov | Combinaciones legales verificadas |
| Adam/AdamW avanzado | Betas y eps | Rango válido y eps positivo |
| Loss | Receta derivada de tarea | Alternativas sólo si aceptan el mismo contrato de salida/target |
| Métrica secundaria | Siempre configurada por tarea | Visible por defecto; ocultable para ampliar loss |
| Seed | Valor persistido (42 propuesto) | Python, NumPy si se usa, PyTorch, loaders/generadores |
| Scheduler | Ninguno por defecto; step, cosine o plateau | Frecuencia explícita por optimizer step/epoch; plateau monitoriza validation |
| Early stopping | Opcional; patience 5, min_delta configurable | Monitor, dirección y regla visibles; checkpoint best asociado |
| Gradient clipping | Opcional; norma máxima 1 sugerida para secuencias | Positivo; después de desescalar AMP y antes del optimizer step |
| Precisión | FP32 garantizada; BF16/FP16 si dispositivo/operadores permiten | AMP con scaler cuando corresponde; indicar fallback antes de iniciar |
| Acumulación de gradiente | 1 por defecto | Batch efectivo y normalización por muestras/tokens válidos; último grupo incompleto tratado correctamente |
| DataLoader workers | Configuración conservadora por OS | No saturar CPU ni romper creación de procesos en Windows |

Los presets son puntos de partida editables. «Rápido para probar» usa muestras/pasos limitados y se etiqueta como smoke experimental; no se confunde con entrenamiento completo.

## 2. Loss y métrica por tarea

| Tarea | Salida y target | Loss por defecto / alternativas | Gráfica secundaria predeterminada |
| --- | --- | --- | --- |
| Clasificación binaria/multiclase general | Logits `[B,K]`; target entero `[B]` | CrossEntropy; pesos de clase opcionales | Accuracy; alternativas macro-F1 y balanced accuracy |
| Clasificación multietiqueta · B | Logits `[B,K]`; target 0/1 flotante mismo shape | BCEWithLogits; `pos_weight` opcional | Micro-F1; macro-F1 opcional |
| Regresión tabular/imagen/secuencia | Valores `[B,Q]`; target mismo shape | MSE; MAE o Huber opcionales | MSE en unidades originales; MAE/RMSE opcionales |
| Pronóstico | `[B,P,Q]`; target mismo shape | MSE; MAE/Huber opcionales | MSE, con detalle por horizonte/variable |
| Segmentación binaria | Logits `[B,1,H,W]`; target float 0/1 y máscara válida | BCEWithLogits + soft Dice (pesos 1 y 1 inicialmente) | Dice; IoU opcional |
| Segmentación multiclase | Logits `[B,K,H,W]`; target entero `[B,H,W]` | CE; CE + soft Dice opcional | Mean IoU; Dice opcional |
| Reconstrucción / denoising | Reconstrucción del mismo shape y dominio que target | MSE; MAE opcional | MAE; PSNR opcional en imágenes con rango definido |
| VAE | Reconstrucción, mu y logvar | Reconstrucción MSE + beta×KL para prior normal | MSE de reconstrucción; KL también visible como componente |
| Texto clasificación | Logits `[B,K]`; etiqueta `[B]` | CrossEntropy | Accuracy o macro-F1 |
| Texto causal | Logits `[B,T,V]`; IDs desplazados `[B,T]` | CE ignorando padding | Perplejidad `exp(CE media por token válido)` |

CE y BCEWithLogits reciben scores sin softmax/sigmoid terminal. La conversión a probabilidades pertenece a métricas/inferencia. El validador detecta una activación terminal incompatible. Esto coincide con los contratos de [CrossEntropyLoss](https://docs.pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html) y [BCEWithLogitsLoss](https://docs.pytorch.org/docs/main/generated/torch.nn.BCEWithLogitsLoss.html).

Clasificación binaria con dos logits en A usa CE igual que multiclase; segmentación binaria usa un logit por píxel y BCE. No mezclar ambas convenciones por compartir la palabra «binaria».

Para VAE, reconstrucción promedia elementos por muestra y luego batch; KL suma dimensiones latentes y promedia batch. Beta se guarda con esa convención; cambiar la reducción cambia el experimento. Se registran total, reconstrucción y KL por separado. Perplejidad se deja como no finita/fuera de rango si la exponencial desborda, sin truncar silenciosamente.

## 3. Definición de métricas y agregación

- Loss de epoch ponderada por el denominador real de la receta: muestras, píxeles válidos o tokens válidos. No promediar promedios de batches de tamaños diferentes sin ponderación.
- Accuracy = aciertos / muestras válidas. Macro-F1 y balanced accuracy derivadas de conteos globales de la partición, no promedio de valores por batch.
- MSE/MAE globales por elementos objetivo válidos; RMSE = raíz del MSE global. En multiobjetivo, mostrar agregación y detalle por objetivo, sobre todo si tienen unidades diferentes.
- Dice/IoU por clase desde conteos acumulados. Por defecto excluir fondo del promedio y mostrarlo por separado; opciones de fondo y umbral quedan persistidas.
- Clase sin presencia en target ni predicción: métrica por clase «N/A» y exclusión del promedio; si hay falso positivo o falso negativo se calcula el valor correspondiente. Si no quedan clases evaluables, el agregado también es N/A.
- En segmentación los píxeles `ignore` no contribuyen. Un batch sin elementos válidos se omite con registro; una partición sin elementos válidos es error.
- Umbrales binarios/multietiqueta por defecto 0.5; si se optimizan, usar validation y guardar el resultado, nunca test.
- Los pesos de clase o `pos_weight` se calculan sólo con train o se proporcionan explícitamente. No cambiar silenciosamente los pesos entre train/validation/test.

La UI debe distinguir loss normalizada, MSE con unidades originales y objetivos compuestos. Si loss=MSE y métrica=MSE en el mismo espacio, explicar que son la misma magnitud; sugerir MAE como vista alternativa sin inventar una métrica nueva.

## 4. Telemetría y gráficas en vivo

Loss es la gráfica principal, con train y validation. Train puede aportar puntos por step y resúmenes por epoch; validation se calcula al cierre de epoch por defecto. No unir ni interpolar puntos de validation como si se hubieran medido cada batch. Cada serie conserva eje/unidad y frecuencia de muestreo.

La métrica secundaria muestra train/validation agregadas cuando están disponibles. Tooltip con step/epoch, timestamp y valor bruto. Interruptores de suavizado, escala logarítmica sólo para valores positivos y comparación de corridas compatibles. Suavizar sólo cambia la presentación; no modifica los datos ni el monitor de early stopping.

Controles: zoom, reset, seguir en vivo, ampliar gráfica, ocultar serie y exportar CSV/imagen. Comparar corridas exige mostrar diferencias de dataset/split/receta; no declarar una mejora comparable entre evaluaciones incompatibles.

Indicadores: epoch actual, batch/optimizer step, muestras o tokens por segundo, LR, tiempo activo, fase, memoria medida y estimada. ETA sólo después de suficiente observación, con etiqueta aproximada; pausa no cuenta como tiempo activo. Sin datos suficientes mostrar «Calculando».

El backend puede agrupar eventos de progreso a 2–5 Hz como objetivo inicial; persistir todos los resúmenes de epoch y puntos configurados. La UI usa buffers y reducción de puntos para historiales largos, conservando picos y rangos. No repintar todo el canvas con cada evento de entrenamiento.

## 5. Ciclo de vida y controles reales

| Estado | Acción disponible | Resultado esperado |
| --- | --- | --- |
| Preparada | Iniciar | Snapshot, preflight, worker |
| Preparando | Cancelar | Cancelación cooperativa antes de optimizar |
| Entrenando / validando | Solicitar pausa; cancelar | Estado pendiente hasta respuesta del worker |
| Pausa solicitada | Esperar; cancelar | Completar unidad de trabajo segura y guardar estado |
| Pausada | Reanudar; cancelar | Cargar estado preservado o finalizar sin más pasos |
| Reanudando | Cancelar | Validar hashes/configuración y restaurar worker |
| Cancelando | Esperar; forzar cierre si no responde | Guardar checkpoint válido si se alcanza límite seguro |
| Completada | Inferir, evaluar test, nueva corrida | Corrida inmutable con best/last |
| Fallida / interrumpida | Diagnóstico; reanudar desde checkpoint válido | No inventar progreso ni marcar finalizada |

Pausa cooperativa en el siguiente límite de optimizer step, tras completar la acumulación; en validation, al terminar un batch. Estado persistido debe incluir cursor de datos, acumuladores de métricas, RNG, optimizador, scheduler y AMP. Para reanudación reproducible, orden y aumentaciones se derivan de epoch/sample ID/seed, no de un prefetched RNG imposible de reconstruir.

Al pausar se libera el worker y sus recursos después del checkpoint; Reanudar lo reconstruye. El motor permanece disponible. Cancelar finaliza la corrida como cancelada y conserva lo recuperable; una continuación posterior se presenta como corrida derivada. «Forzar cierre» es un último recurso y avisa que se perderá lo posterior al último checkpoint durable.

La UI no presenta pausado/completado hasta recibir confirmación. Un proceso detenido a mitad de escritura no crea un checkpoint válido: archivo temporal y publicación atómica del manifiesto. Cierre normal de la ventana con trabajo activo propone pausar y cerrar, cancelar y cerrar o volver. Ejecución en segundo plano tras cerrar la aplicación queda fuera de A; cambiar de pantalla/proyecto dentro de la app sí mantiene el trabajo.

## 6. Checkpoints y recuperación

`last` para continuidad; `best` según monitor de validation y dirección explícita. Guardar periódicamente, al terminar epoch, al pausar y al finalizar; límite configurable de históricos sin eliminar best/last ni checkpoints referenciados por predicciones.

Contenido lógico: pesos y buffers, snapshot de grafo, contrato/pipeline de datos, clases/columnas, loss/métricas, configuración, optimizador, scheduler, AMP, seeds/RNG, cursor/epoch/step, versión del runtime, historial hasta el punto durable, hashes y estado de completitud. Pesos de inferencia exportados no necesitan todo el estado de reanudación.

Al reiniciar tras apagado se detecta la ausencia del worker para una corrida marcada activa y se registra «Interrumpida». Se ofrece último checkpoint íntegro y punto hasta el que se recupera. Si no existe, se ofrece nueva corrida desde cero, no «Reanudar». Historial posterior al checkpoint queda marcado como no reanudable; no repetir steps con IDs ambiguos.

Reproducibilidad se promete como trazabilidad y restauración del estado en un entorno compatible; no igualdad bit a bit entre CPU/GPU, sistemas o versiones. Esta limitación está documentada por [PyTorch sobre reproducibilidad](https://docs.pytorch.org/docs/main/notes/randomness.html).

## 7. Evaluación final

Validation gobierna selección y early stopping. Al completar, evaluar una vez test con best, si existe test; guardar un artefacto independiente del log de entrenamiento. Si se evalúa otro checkpoint, registrar nueva evaluación con su identidad, sin sobrescribir la anterior.

Resultados: matriz de confusión y métricas por clase para clasificación; dispersión predicción/objetivo y residuales para regresión; error por horizonte para pronóstico; Dice/IoU y galería de máscaras para segmentación; reconstrucción y error para AE/VAE; CE/perplejidad y muestras para texto causal.

Explorar repetidamente test puede influir en decisiones humanas: la UI lo presenta como conjunto reservado y no lo introduce en curvas de ajuste. Si no hay test, el informe dice «Sin evaluación independiente»; no renombra validation como test.

## 8. Inferencia común

Elegir checkpoint; cargar su grafo y pipeline congelados; verificar entrada; ejecutar `eval` y modo de inferencia sin gradientes; transformar y presentar resultado; registrar latencia y artefacto. No usar pesos aleatorios como si fueran un modelo entrenado.

Fuentes: archivo propio, entrada manual cuando corresponda, muestra de una partición elegida o lote. La selección de partición es visible; muestra aleatoria no consume ni cambia el dataset. La salida puede existir sin etiqueta real: en tal caso no se muestra accuracy/error por muestra.

La predicción registra checkpoint/hash, fuente/ID de entrada, pipeline, parámetros, salida y timestamp. Una entrada con columnas reorganizadas se mapea por nombre si es inequívoco; incompatible produce diagnóstico, no una conversión silenciosa. Un checkpoint histórico usa su contrato incluso si el proyecto actual cambió.

## 9. Experiencia por familia/tarea

| Caso | Entrada y controles | Resultado / exportación |
| --- | --- | --- |
| MLP clasificación | Formulario de variables tipadas o filas CSV | Clase y probabilidades; CSV/JSON con IDs |
| MLP regresión | Formulario/CSV con esquema congelado | Valores desnormalizados y unidades; error sólo con objetivo |
| CNN/ResNet | Imagen o lote; preview del resize/canales | Probabilidades o valores; opcional galería de errores etiquetados |
| LSTM/RNN/GRU | CSV/serie manual, orden y T visibles | Clase/valor/horizonte; gráfico alineado con tiempo |
| U-Net | Imagen, umbral binario, opacidad, clases visibles | Máscara indexada sin pérdida, paleta/JSON y overlay PNG separado |
| Autoencoder | Vector/imagen; ruido de prueba explícito si denoising | Reconstrucción y mapa/resumen de error |
| VAE | Reconstruir por mu o muestrear con seed; explorar z | Resultado estocástico etiquetado, interpolación entre z compatibles |
| Transformer clasificación | Texto y tokens/preprocesamiento visibles bajo demanda | Probabilidades por clase |
| Transformer causal | Prompt, greedy o sampling; temperatura, top-k, top-p, seed, máximo tokens | Tokens en streaming, detener, texto/JSON |

Segmentación restaura geometría original según pipeline; probabilidad y máscara dura usan interpolación apropiada y la máscara discreta siempre vecino más cercano. El overlay no sustituye la máscara exportada. En pronóstico, no permitir más horizonte que el contrato directo del checkpoint; generar recursivamente exige otra receta explícita.

En texto, greedy no aplica temperatura; sampling exige temperatura positiva, top-k válido y 0<top-p≤1. Límite de contexto incluye prompt y continuación; mostrar política de truncado antes de ejecutarla. No introducir chat con roles si sólo se entrenó siguiente token sobre texto plano. Probabilidades de tokens no se describen como confianza calibrada.

## 10. Recursos y errores

Una tarea pesada por dispositivo en A: inferencia sobre el mismo dispositivo que entrena espera o permite pausar el entrenamiento; no ocupar VRAM adicional silenciosamente. La entrada de inferencia sigue editable mientras espera y puede cancelarse.

Errores diferenciados: OOM, datos no finitos, pérdida/gradientes no finitos, checkpoint incompleto, source ausente, dtype no soportado, worker caído, disco lleno, incompatibilidad de versión y operación no soportada por el dispositivo. Mostrar causa y acciones concretas; logs técnicos ampliables, sin traceback ocupando toda la pantalla.

Ante OOM: conservar último checkpoint, mostrar batch/estimación/medición y sugerir reducir batch, resolución o modelo. El usuario puede crear corrida derivada con los cambios; no fingir que una corrida fallida continuó sin alterar configuración.
