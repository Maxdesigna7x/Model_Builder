# 07 · Plan de implementación y criterios de aceptación

**Estado actualizado:** 8 de septiembre de 2026. Las fases A, B (familias adicionales) y C-atención (Transformer encoder/causal, ViT, texto) están implementadas y cubiertas por tests. El build frontend y la suite Python pasan. El build nativo de Tauri y el empaquetado multiplataforma siguen pendientes de instalar las cabeceras del sistema y configurar PyInstaller.

## 1. Orden propuesto

| Fase | Trabajo | Estado |
| --- | --- | --- |
| 0 · Viabilidad del stack | Tauri/React↔Python, proceso de trabajo | Parcial: scaffold funcional; empaquetado pendiente |
| 1 · Sistema visual y proyectos | Tokens, shell, creación de proyectos, navegación | Implementado (modo web y Tauri parcial) |
| 2 · Contratos y datos | Catálogo, importadores/generadores, splits | Implementado para las 13 tareas activas |
| 3 · Constructor y primer recorrido real | Grafo/compilador, plantillas, validación | Implementado |
| 4 · Catálogo A completo | MLP, CNN2D, LSTM, U-Net | Implementado |
| 5 · Instrumentos y recuperación | Gráficas, best/last, test, inferencia | Implementado; **pausa/reanudación pendiente** |
| 6 · Entrega A | Paquetes y QA multiplataforma | **Pendiente** |
| 7 · Entrega B | RNN/GRU/CNN1D, preset residual, autoencoders | Implementado (RNN/GRU como bloques huérfanos; denoising y multietiqueta **pendientes**) |
| 8 · Entrega C | Transformer encoder/causal, ViT, texto | Implementado; **VAE y Transformer encoder–decoder pendientes** |

La fase 3 usa una vertical mínima de entrenamiento/inferencia para probar el diseño; fase 5 completa sus controles y recuperación. No dejar todas las pruebas de empaquetado para el final. Cada entrega sólo anuncia lo que ya funciona de extremo a extremo.

## 2. Matriz de trazabilidad de la solicitud

| Requisito solicitado | Especificación | Estado o criterio de aceptación |
| --- | --- | --- |
| Empezar desde cero y limpiar carpeta | [08](08-registro-del-reinicio.md) | Referencias conservadas, anterior fuera de la carpeta, nueva propuesta sin código Qt |
| React/TS + Tauri + Python/PyTorch | [06](06-arquitectura-tecnica.md) | Desktop empaquetado con cálculo Python local real |
| Similar a imágenes proporcionadas | [02](02-diseno-visual-y-pantallas.md) | QA con referencias: oscuro carbón, cian, densidad y gráficas legibles |
| Menú de proyectos | [01](01-producto-y-alcance.md), [02](02-diseno-visual-y-pantallas.md) | Crear/abrir/reabrir sin mezcla de datos |
| Selección de tarea y modelo con datos compatibles | [03](03-tareas-y-datos.md) | Entrada/salida/formato visibles antes de importar; combinaciones inválidas explicadas |
| Cargar datos o sintéticos específicos | [03](03-tareas-y-datos.md) | Todos los casos A tienen ambos caminos y preview real |
| Constructor drag-and-drop | [04](04-constructor-y-bloques.md) | Grafo modificable, ejecutable y persistente, también operable por teclado |
| Piezas de cada arquitectura, incluido pooling | [04](04-constructor-y-bloques.md) | Registro de todas las piezas del alcance declarado y pruebas por operador |
| Menú contextual derecho por bloque | [02](02-diseno-visual-y-pantallas.md), [04](04-constructor-y-bloques.md) | Inspector sigue selección y cada propiedad modifica realmente la capa |
| Entrenamiento con loss y otra métrica | [05](05-entrenamiento-e-inferencia.md) | Curvas reales train/validation, agregación correcta y métrica según tarea |
| Menú de inferencia | [05](05-entrenamiento-e-inferencia.md) | Checkpoint real, mismo pipeline y visualización apropiada por modalidad |
| Documentar antes de decidir implementar | [README](../README.md) | Etapa documental completada antes de iniciar esta implementación |

## 3. Recorridos de aceptación implementados

Cada fila se ejecuta con dataset sintético. El flujo común es: crear proyecto, seleccionar tarea, generar/importar datos, editar grafo, validar, entrenar, guardar checkpoint e inferir.

| Caso | Edición de canvas | Comprobación final |
| --- | --- | --- |
| MLP clasificación | Añadir/quitar Linear, cambiar `out_features` | K salidas, CE, probabilidades coherentes |
| MLP regresión | Cambiar cabeza a 1 salida | Valor continuo, sin activación terminal |
| CNN clasificación | Insertar MaxPool/AvgPool, cambiar canales | Shapes correctas, grafo ejecutable |
| CNN regresión | Ajustar cabeza a 1 salida | Valor continuo asociado a imagen |
| LSTM clasificación | Cambiar `hidden_size`, añadir `TemporalSelect` | Salida `[B,K]`, padding manejado |
| LSTM regresión | Cambiar `hidden_size` | Salida `[B,1]` |
| LSTM pronóstico | Ajustar capas recurrentes | Salida `[B,P]` |
| U-Net binaria | Conectar skip, cambiar upsampling | Un logit/píxel, máscara alineada |
| U-Net multiclase | Editar canales de decoder | K logits/píxel |
| Transformer clasificación de texto | Ajustar `d_model` y cabeza | Salida `[B,K]` |
| Transformer causal | Cambiar `num_layers` | Logits `[B,T,V]`, predicción greedy |
| ViT | Usar preset de ViT | Clasificación desde patches |
| Autoencoder tabular | Construir cuello de botella con Linear | Reconstrucción `[B,F]` |
| Autoencoder de imagen | Construir encoder/decoder convolucional | Reconstrucción `[B,C,H,W]` |

## 4. Pruebas de bloques y compilador

Cada `BlockSpec` debe tener pruebas de forma/dtype y al menos un forward/backward donde corresponda; casos inválidos de propiedades/puertos; compilación, serialización y recarga. Comparar salidas/gradientes con composición PyTorch de referencia usando pesos equivalentes y tolerancias documentadas.

Casos obligatorios: kernel mayor que entrada; stride/padding/dilation; groups inválido; pooling ceil en dimensiones impares; índices MaxPool→MaxUnpool; salida ConvTranspose; Concat eje/orden; Add incompatible; reshape que pierde elementos; Flatten de batch prohibido; puerto huérfano; ciclos; máscaras/longitudes; LSTM bidireccional/proyectada; dry-run sin contaminar pesos/BatchNorm/RNG.

Macros colapsadas y expandidas deben ejecutar lo mismo. Renombrar/mover nodos no altera el hash semántico; cambiar conexiones/propiedades sí. Undo/redo reconstruye el estado exacto. En B/C añadir residual con proyección, AE decoder, VAE mu/logvar y pérdidas, atención con máscaras/causalidad y pesos compartidos.

## 5. Datos y metodología

Verificar columnas reordenadas, faltantes, clases desconocidas, one-hot y escalado sólo train, IDs duplicados, grupos, imágenes corruptas, máscaras faltantes/paleta, resoluciones, fuentes modificadas y split preexistente. Corpus/series no comparten fragmentos/ventanas entre splits. Inferencia reproduce transformaciones y la inversión de targets/geometría.

Métricas con fixtures de resultado conocido: accuracy/F1 desde conteos, MSE/MAE con distintos tamaños de batch, Dice/IoU con ignore y clases ausentes, multiobjetivo con escalado y perplejidad por token válido. Best/early stopping sólo consultan validation. Test queda separado de la curva recurrente.

## 6. Integración, fallos y durabilidad

| Escenario | Resultado esperado |
| --- | --- |
| Mensaje JSON dividido en varios chunks/UTF-8 | Se reconstruye exactamente sin pérdida |
| Respuesta tardía de validación | No reemplaza el diagnóstico de una revisión más nueva |
| Reintento de `run.start` tras timeout | Una sola corrida por clave de idempotencia |
| Cola de telemetría saturada | UI responde, historial durable y terminales intactos |
| Pausar durante acumulación/validation | Estado confirmado sólo tras snapshot consistente |
| Apagado durante escritura de checkpoint | Último checkpoint completo recuperable; temporal rechazado |
| Caída del worker o motor | Estado interrumpido/fallido, diagnóstico, recursos liberados |
| Disco lleno | Error de guardado visible; no afirmar checkpoint guardado |
| Dos instancias abriendo mismo proyecto | Lock coherente; no corrupción por doble escritura |
| Dataset movido/modificado | Relocalización/validación; no reutilizar identidad falsa |
| Checkpoint de otra tarea/revisión | Abrir su snapshot o rechazar; nunca cargar pesos silenciosamente en otro grafo |
| Cerrar ventana con entrenamiento | Aplicar política elegida y cerrar hijos; sin procesos huérfanos |

Prueba de reanudación: comparar corrida continua frente a pausa/reinicio en el mismo entorno compatible, incluyendo orden de datos, LR, contador, métricas y estado de optimizer. Documentar tolerancia y grado real de determinismo; si un modo no permite continuidad estricta, bloquearlo o presentarlo como reinicio desde checkpoint de epoch, no simular exactitud.

## 7. UI, rendimiento y paquetes

QA visual a 1280×720 y 1920×1080, ambos temas, escala del sistema y paneles plegados. Capturas de estados vacío, cargando, listo, error, entrenamiento, pausado e inferencia. Todas las propiedades/piezas accesibles por teclado; loss legible; inspector no tapa canvas en la resolución base.

Pruebas del frontend con bridge simulado para interacción; integración desktop con backend real y prueba del instalador por OS. Una prueba en navegador no certifica WebKitGTK/WebView2 ni el sidecar empaquetado. Performance con los casos de [06](06-arquitectura-tecnica.md), registrando equipo, OS, GPU, driver, versiones y tamaño de dataset/grafo.

Entrega exige instalación desde cero, abrir proyecto, entrenar en CPU, checkpoint, inferencia, reiniciar app y desinstalar sin eliminar proyectos del usuario. CUDA se valida en una matriz explícita. No es obligatorio hardware GPU para usar el paquete CPU.

## 8. Decisiones pendientes de la implementación, no de esta documentación

La fase 0 resolverá versiones exactas, runtime Python compatible, baseline Linux/Windows, distribución CPU/GPU, formato final de bundle PyInstaller y presupuesto real de tamaño/arranque. Se ha fijado una opción inicial para cada componente; sólo se cambia tras evidencia técnica y se actualiza su documento.

Las fases B/C amplían el producto de manera importante. Antes de empezarlas se deberá confirmar que forman parte de la entrega deseada; A tiene una definición completa propia. Esto no deja A sin pooling, skips, propiedades avanzadas ni inferencia. No añadir tareas nuevas como GAN/difusión por inferencia de «todas las arquitecturas».

## 9. Qué significa terminar

**Documentación terminada:** los ocho documentos están guardados, enlazados y coherentes; cada requisito de la solicitud tiene especificación; estado de aprobación y respaldo explícitos.

**Entrega A:** el código cubre catálogo, recorridos, persistencia, entrenamiento e inferencia. La prueba de paquete Linux/Windows continúa abierta hasta instalar dependencias nativas y ejecutar CI o un equipo Windows; por eso todavía no se declara una distribución final firmada.
