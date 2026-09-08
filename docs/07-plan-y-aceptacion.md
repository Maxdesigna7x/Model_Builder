# 07 · Plan de implementación y criterios de aceptación

**Estado actualizado:** implementación de la entrega A realizada el 5 de septiembre de 2026. El build frontend y las pruebas del motor pasan. El build Tauri en este equipo queda pendiente de instalar las cabeceras GLib/WebKitGTK del sistema; las ampliaciones B/C siguen planificadas.

## 1. Orden propuesto

| Fase | Trabajo | Evidencia para cerrar |
| --- | --- | --- |
| 0 · Viabilidad del stack | Versiones, Tauri/React↔Python, proceso de trabajo y paquete mínimo Linux/Windows | Ejecutable instalado en entorno sin conda, handshake, cálculo PyTorch CPU, evento y cierre limpio; decisión de runtime/paquetes GPU |
| 1 · Sistema visual y proyectos | Tokens, shell, biblioteca/creación de proyectos, navegación y persistencia | Capturas en ambos temas, apertura/cierre/recuperación, rutas con Unicode y escala del sistema |
| 2 · Contratos y datos | Catálogo tarea/familia, importadores/generadores, preview, splits y pipeline | Fixtures de cada tarea A, errores accionables y separación sin fuga de información |
| 3 · Constructor y primer recorrido real | Grafo/compilador, puertos, inspector, undo/redo y plantilla MLP | Crear MLP arrastrando, validar, entrenar, guardar y predecir sin código del usuario |
| 4 · Catálogo A completo | CNN2D, LSTM y U-Net; ramas, pooling, skips, dimensiones y macros | Todas las piezas A verificadas y todos los recorridos de tareas A ejecutables |
| 5 · Instrumentos y recuperación | Configuración completa, gráficas loss/métrica, pausa/reanudar, best/last, test e inferencia por lotes | Corrida reproducible recuperada tras interrupción; métricas correctas y exportación |
| 6 · Entrega A | Optimización, accesibilidad, paquetes y QA real | Matriz de aceptación de A completa en Linux y Windows; dependencias/licencias documentadas |
| 7 · Entrega B | RNN/GRU/CNN1D/ResNet/AE, denoising y multietiqueta | Datos, bloques, recetas, métricas e inferencia de cada caso completos |
| 8 · Entrega C | VAE y Transformer pequeño | Loss compuesta, máscaras, tokenización, muestreo y contratos completos |

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

## 3. Recorridos de aceptación de A

Ejecutar cada fila con dataset sintético y fixture importado equivalente. Incluir crear proyecto, seleccionar tarea, datos/split, editar grafo, validar, entrenar, guardar checkpoint, cerrar/reabrir y predecir con pesos recargados.

| Caso | Edición de canvas obligatoria | Comprobación final |
| --- | --- | --- |
| MLP binaria | Añadir Linear y cambiar ancho | Dos logits, CE, probabilities coherentes |
| MLP multiclase | Reordenar capa mediante reconexión | K salidas y mapa de clases estable |
| MLP regresión multiobjetivo | Cabeza Q explícita | Valores y unidades correctas, sin sigmoid automática |
| CNN clasificación | Insertar MaxPool y sustituirlo por AvgPool | Shapes correctas y grafo ejecutado distinto |
| CNN regresión | Editar canales y cabeza Q | Imagen/target asociados correctamente |
| LSTM clasificación | Apilar recurrentes y seleccionar estado válido | Padding no cambia resultado más allá de tolerancia; caso bidireccional correcto |
| LSTM regresión | Cambiar hidden_size y LinearQ | Salida `[B,Q]`, target consistente por secuencia |
| LSTM pronóstico | Cambiar horizonte a través del contrato | `[B,P,Q]`; sin ventanas cruzadas entre splits |
| U-Net binaria | Conectar skip y cambiar upsampling | Un logit/píxel, máscara y overlay alineados |
| U-Net multiclase | Editar etapa y alinear shape impar | K logits/píxel, paleta/ignore conservados |

Pruebas de aprendizaje con pequeños datasets deterministas: sobreajustar un conjunto train pequeño debe reducir loss en una receta de prueba definida. No exigir una accuracy arbitraria de generalización tras dos batches ni confundir el smoke con evaluación científica.

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
