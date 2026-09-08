# 01 · Producto y alcance

**Fecha:** 8 de septiembre de 2026. **Estado actualizado:** núcleo A, ampliaciones B (RNN/GRU/CNN1D, preset residual en CNN 2D y autoencoders) y atención encoder/causal de C (Transformer encoder, Transformer causal y ViT) implementados. VAE y Transformer encoder–decoder siguen pendientes.

## 1. Decisión de producto

ModelBuilder será un taller visual de redes neuronales locales. Cada proyecto responde una pregunta concreta: qué se quiere predecir, con qué datos, mediante qué arquitectura y con qué evidencia de aprendizaje.

El recorrido conserva el orden anterior, con un constructor sustancialmente nuevo:

1. Crear o abrir un proyecto.
2. Seleccionar tarea, familia de arquitectura y modalidad de datos compatible.
3. Elegir un dataset público compatible de Hugging Face o importar datos propios.
4. Construir y validar el modelo arrastrando bloques a un canvas.
5. Configurar y observar el entrenamiento real.
6. Elegir un checkpoint y ejecutar inferencia.

La UI usará React + TypeScript; Tauri proporcionará la aplicación de escritorio; Python/PyTorch ejecutará el motor en procesos locales. Linux y Windows son objetivos del producto. La especificación técnica está en [06](06-arquitectura-tecnica.md).

## 2. Qué cambia respecto al producto anterior

- La arquitectura se define con nodos, puertos, conexiones y propiedades; el dibujo es una representación editable del grafo ejecutable.
- Se pueden insertar, sustituir, duplicar, conectar y eliminar capas, además de editar sus parámetros.
- Los presets crean bloques normales editables. Una CNN no queda reducida a un control global de «cantidad de bloques».
- La validación explica dimensiones, clases, tipos y conexiones inválidas en el propio canvas.
- Cada entrenamiento congela una versión del grafo y del pipeline de datos. Cambiar el proyecto no reinterpreta pesos anteriores.
- La dirección visual sigue las capturas oscuras de `references/`: jerarquía compacta, superficies carbón, acentos cian e instrumentos de entrenamiento legibles.

## 3. Alcance de arquitecturas

La biblioteca no puede prometer representar toda operación posible de PyTorch o toda arquitectura publicada. El contrato es cubrir todas las piezas de las familias declaradas aquí, con extensibilidad explícita y sin componentes decorativos que no se ejecuten.

| Entrega propuesta | Familias | Estado |
| --- | --- | --- |
| A · Núcleo completo | MLP, CNN 2D, LSTM, U-Net 2D | Implementado |
| B · Ampliación de familias | RNN, GRU, CNN 1D, preset residual en CNN 2D, autoencoder denso/convolucional | Implementado (RNN/GRU como bloques; el preset residual usa `Add`/`identity` dentro de CNN 2D) |
| C · Generativas y atención | Transformer encoder, Transformer causal y ViT implementados; VAE y Transformer encoder–decoder pendientes | Atención implementada; VAE y encoder–decoder requieren contratos adicionales |

El catálogo sólo muestra combinaciones ejecutables. El detalle verificado y las exclusiones actuales se mantienen en [`../ARCHITECTURE_TODO.md`](../ARCHITECTURE_TODO.md).

No se incluyen inicialmente detección de objetos, segmentación de instancias, GAN, difusión, GNN, convoluciones 3D, refuerzo, entrenamiento distribuido, DPO/SFT, edición de código Python arbitrario ni importación universal de modelos externos. Cada una requiere datos, pérdidas o ejecución adicionales, no sólo un icono de bloque. La referencia `LLM.md` inspira el recorrido educativo; no convierte automáticamente chat, SFT o DPO en requisitos.

## 4. Proyecto, experimento y corrida

- **Proyecto:** unidad de trabajo con identidad y ubicación elegida por el usuario.
- **Experimento:** combinación de tarea, familia, revisión de datos y revisión de grafo. En A se edita un experimento activo y se pueden duplicar variantes.
- **Corrida:** ejecución inmutable de una configuración de entrenamiento sobre un snapshot del experimento.
- **Checkpoint:** pesos y estado asociado a una corrida concreta.
- **Predicción:** resultado ligado a checkpoint, entrada y parámetros de inferencia.

Un proyecto puede conservar varias corridas y variantes. Sólo una tarea de cómputo intensivo ocupa cada dispositivo en A; la UI continúa disponible. Cambiar de proyecto no detiene silenciosamente el trabajo activo.

## 5. Navegación y estado

Dentro del proyecto hay cinco pasos persistentes: **Modelo y tarea · Datos · Constructor · Entrenamiento · Inferencia**. El inicio de proyectos está fuera del recorrido. Los pasos usan estados «Pendiente», «En edición», «Listo», «Revisar» y «Bloqueado», acompañados de icono y texto.

| Paso | Condición para estar listo | Bloqueo o revisión |
| --- | --- | --- |
| Modelo y tarea | Familia + tarea + modalidad confirmadas | Combinación no soportada |
| Datos | Descarga/importación válida, contrato resuelto, split y pipeline guardados | Error de descarga, formato, etiquetas, pares o partición |
| Constructor | Grafo guardado, sin errores y prueba de un batch correcta | Forma, dtype, ruta de salida o revisión desactualizados |
| Entrenamiento | Existe al menos un checkpoint válido de la revisión seleccionada | Run fallida sin checkpoint; incompatibilidad con revisión actual |
| Inferencia | Se completó una predicción válida | No hay checkpoint compatible o entrada inválida |

Siempre se permite volver a pasos anteriores y consultar historial. Los estados de avance se recalculan a partir de artefactos reales. Visitar una pantalla no la completa.

## 6. Cambios e invalidación

| Cambio | Consecuencia |
| --- | --- |
| Nombre del proyecto, posiciones de nodos, zoom o tema | No cambia el modelo ni invalida pesos |
| Nombre visible de clase conservando ID y orden | Cambia presentación; se conserva el nombre histórico de cada corrida |
| Columnas, resolución, canales, clases, target, split o transformaciones | Nueva revisión de datos; revalidar el constructor y crear nueva corrida |
| Capa, conexión o propiedad que cambia el cálculo | Nueva revisión de grafo; volver a validar; pesos previos sólo disponibles con su snapshot |
| LR, batch, optimizador, loss o seed | Nueva configuración de corrida; no altera corridas existentes |
| Tarea o familia | Crear variante con nueva configuración; presentar qué puede reutilizarse y qué requiere reconfiguración |

El diálogo de cambio de tarea/familia ofrecerá «Crear variante» y «Cancelar», con un resumen de impacto. No borrará datos ni corridas. La reanudación estricta conserva la configuración de la corrida; cambiarla genera una nueva corrida derivada, cuando sea compatible.

## 7. Principios de UX y veracidad

- La experiencia guía, pero permite editar arquitecturas completas dentro de su contrato.
- Cada control tiene valor visible, significado, efecto y validación. Avanzado no significa sin validaciones.
- Shapes y parámetros se calculan; las estimaciones de memoria y tiempo se etiquetan como estimaciones.
- Ninguna curva, progreso o predicción presentada como real se inventa para animar la interfaz.
- Las métricas dependen de la tarea, no sólo del nombre de la arquitectura.
- El catálogo público muestra procedencia, licencia y descarga estimada; cada fuente se fija a una revisión reproducible.
- Un mismo preprocesamiento congelado acompaña al modelo hasta inferencia.
- Test queda reservado a evaluación final y no gobierna early stopping ni selección de checkpoint.

## 8. Decisiones todavía sujetas a la futura validación técnica

Se proponen un transporte local JSON por pipes, React Flow para el canvas y ECharts para las gráficas. La aprobación de este diseño no garantiza cifras de rendimiento ni un tamaño de instalador; se medirán con el stack empaquetado. Las versiones exactas, runtime Python, requisitos mínimos y variantes GPU se fijarán tras la prueba de integración de la fase 0 de [07](07-plan-y-aceptacion.md).
