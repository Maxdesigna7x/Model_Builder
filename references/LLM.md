**Language Model Builder** es una aplicación de escritorio interactiva, educativa y de experimentación visual diseñada para desmitificar el ciclo de vida completo de un modelo de lenguaje (desde la arquitectura hasta la inferencia y alineación), permitiendo entrenar y ajustar modelos pequeños de manera local y transparente.

---

### 1. Estructura Global y Sistema de Diseño de la UI

La interfaz utiliza una estética minimalista y moderna al estilo macOS (tipografía limpia sin remates combinada con encabezados serif elegantes, esquinas redondeadas, fondo claro `#F6F6F6` y paleta técnica con acentos en azul, violeta y verde).

* **Barra superior / Header:**
* Indicador de progreso por pasos (ej. *Build your model / Project setup · 8 of 15*).
* Menú de controles rápidos: vista X-ray, transcripción, modo oscuro/claro, compartir y ajustes.


* **Barra lateral izquierda (Sidebar de navegación):**
* **Sección teórica:** *Welcome*, *Introduction*, *What is a language model?*, *Tokens*, *Embeddings*, *Anatomy of a Transformer*, *Training data*, *How models learn*, *Context & modern behavior*.
* **Sección práctica (*Build your model*):**
* *Project setup*
* *Pre-training data*
* *Pre-training*
* *Sampling*
* *Fine-tuning data*
* *Supervised fine-tuning*
* *Direct preference optimization (DPO)*
* *Chat with your model*


* **Pie de sidebar:** Progreso total completado (barra de porcentaje) y créditos (*Acknowledgements*).



---

### 2. Vistas y Flujo de Trabajo Paso a Paso

#### A. Pantalla de Bienvenida (Home Hub)

* Tres tarjetas de acción rápida con estética translúcida:
1. **Learn about models:** Acceso directo al temario teórico interactivo.
2. **Continue training:** Reanuda checkpoints pausados (muestra estado y conteo de pasos).
3. **Open an existing project:** Selector de archivos para cargar proyectos previos.



#### B. Arquitectura del Modelo (Project Setup)

* **Model Blueprint (Diagrama interactivo):**
* Representación en bloques 3D/isométricos del flujo del Transformer: *Context Window* $\rightarrow$ *Embedding/Projection* $\rightarrow$ *Transformer Blocks* $\rightarrow$ *Output Scores (Logits)*.
* Muestra en tiempo real métricas clave calculadas: parámetros totales (ej. `19.3M`), peso en disco (`36.8 MB`) y tiempo estimado de entrenamiento (`~2-5h`).


* **Panel de configuración (Sliders e inputs):**
* **Tokenizer:** Elección de vocabulario (ej. *Fast 10K BPE* vs *GPT-2 BPE* de 50,261 tokens).
* **Shape:** Selección de preajustes (*Starter*, *Custom*).
* **Hiperparámetros estructurales:** Deslizadores para *Context window* (tokens), *Width* (dimensión del vector / $d_{model}$), *Transformer blocks* (capas $N$) y *Attention heads*.



#### C. Catálogo de Datos (Pre-training Data y Fine-tuning Data)

* **Panel izquierdo:** Lista de datasets preconfigurados (TinyStories, TinyText-2 Raw, Simple English Wikipedia, GoodWiki, arXiv Abstracts, OpenAssistant, Dolly 15k, etc.) con opción de añadir datasets propios (`+ Add your own`).
* **Panel derecho:**
* Metadatos: tamaño de archivo, número de tokens/validación, licencia, cita de Paper y enlaces a Hugging Face.
* Visor de texto enriquecido con tags de delimitación (ej. `<|endoftext|>`).
* Botón de descarga/selección interactivo con barra de progreso de descarga.



#### D. Entrenamiento en Tiempo Real (Pre-training)

* **Dashboard de telemetría:**
* Gráfico dinámico de función de pérdida (*Loss Curve*): trazo azul para entrenamiento y trazo naranja/puntos para validación (`val loss`), con etiquetas de valor en hitos relevantes.
* **Barra de métricas:** Step actual vs total, `LOSS`, `VAL LR`, `TOK/S`, `TIME` transcurrido y tiempo restante estimado.
* **Ficha de configuración:** Resumen visual del dataset, tokenizer, conteo de parámetros y tamaño de vocabulario.
* **Controles de ejecución:** Botón prominente de *Start / Pause / Resume*, deslizador de objetivo de pasos (*Step target*) e hiperparámetros de entrenamiento (*Batch size*, *Learning rate*, *Warmup*).


* **Sample Timeline (Generación en vivo por Checkpoint):**
* Deslizador inferior que permite desplazarse cronológicamente por los checkpoints guardados.
* Muestra cómo evoluciona la coherencia del texto generado: desde caracteres aleatorios al inicio hasta oraciones gramaticalmente estructuradas.



#### E. Muestreo e Inferencia (Sampling Playground)

* **Canvas central:** Generación de texto a partir de un prompt inicial (*"Once upon a time..."*).
* **Inspección de probabilidades de tokens:**
* Al pulsar sobre cualquier token generado, se despliega un pop-up con la distribución de probabilidades alternativas que el modelo evaluó en ese instante (ej. *boat: 17.6%*, *and: 14.0%*).
* Barra interactiva *Next Token*: permite elegir manualmente cuál de los tokens más probables debe seguir.


* **Panel de decodificación lateral:**
* Presets: *Predictable*, *Balanced*, *Creative*.
* Controles de muestreo: *Temperature*, *Top-k*, *Top-p*, *Min-p*, y límite de *Max tokens*.
* Herramienta de comparación multi-semilla (*Compare all three personas*).



#### F. Alineación y Optimización (DPO - Direct Preference Optimization)

* **Preference Ballot:**
* Interfaz de pares de respuesta (*Pairwise comparison*) para alinear el comportamiento del modelo.
* Presenta un prompt (ej. *"Can you suggest one cozy thing to do today?"*) y genera dos respuestas en paralelo (*Candidate A* y *Candidate B*).
* Botones de selección de preferencia (*"This one is better"*), contador de pares requeridos para desbloquear el ajuste fino y botón para volver a muestrear.



#### G. Chat con el Modelo e Inspección Avanzada (X-Ray)

* **Selector de estado del modelo:** Pestañas superiores para interactuar con la versión `BASE`, `SFT` o `DPO` del modelo entrenado.
* **Interfaz de Chat conversacional:** Historial de mensajes, sugerencias iniciales e input de texto.
* **Modo X-Ray (Visualizador de confianza/probabilidad):**
* Cada palabra se resalta como un token coloreado según el nivel de certidumbre del modelo (azul oscuro = alta certeza; tonos pálidos = alternativas disputadas).
* Al hacer clic en un token, se muestra un modal con las opciones descartadas y sus porcentajes exactos.


* **Sidebar derecha (Transcript / Formato crudo):**
* Muestra el formateo de bajo nivel y los tokens de control que recibe la red (ej. `<|system|>`, `<|user|>`, `<|assistant|>`), contador total de tokens utilizados y parámetros de decodificación activos.



---

### 3. Consideraciones para la Arquitectura Modular del Software

Para poder replicar esta base y luego extenderla a arquitecturas de visión (CNN, U-Net) o secuencias recurrentes (LSTM, MLP):

| Componente | Requisito de Diseño Modular |
| --- | --- |
| **Frontend UI** | Diseñar un sistema de bloques componibles donde el diagrama central (*Blueprint*) acepte grafos dirigidos arbitrarios (nodos convolucionales, de pooling, recurrentes o residuales). |
| **Motor de Backend** | Capa desacoplada (usando PyTorch o MLX si es macOS local) que reporte métricas de entrenamiento vía WebSockets/IPC (Loss, gradientes, throughput, inferencia intermedia). |
| **Módulo de Inspección** | Abstraer el inspector visual: para LLMs muestra probabilidades de tokens; para CNN/U-Net debe proyectar mapas de activación/convoluciones; para MLP fronteras de decisión; y para LSTM estados ocultos ($h_t, c_t$). |
