import type { Architecture, ModelNode, ModelEdge, TaskId } from "./types";

export const ARCHITECTURES: Record<Architecture, { name: string; description: string; accent: string; tasks: TaskId[] }> = {
  mlp: { name: "MLP", description: "Capas densas para datos tabulares.", accent: "#22d3ee", tasks: ["tabular.classification", "tabular.regression"] },
  cnn1d: { name: "CNN 1D", description: "Convoluciones temporales para señales y series.", accent: "#34d399", tasks: ["sequence.classification", "sequence.regression", "sequence.forecast"] },
  lstm: { name: "Red recurrente", description: "RNN, GRU y LSTM para dependencias temporales.", accent: "#fbbf24", tasks: ["sequence.classification", "sequence.regression", "sequence.forecast"] },
  transformer: { name: "Transformer encoder", description: "Atención bidireccional para series y clasificación de texto.", accent: "#fb7185", tasks: ["sequence.classification", "sequence.regression", "sequence.forecast", "text.classification"] },
  transformer_causal: { name:"Transformer causal", description:"Atención enmascarada para predecir y generar tokens.", accent:"#f97316", tasks:["text.language_model"] },
  cnn: { name: "CNN 2D", description: "Filtros espaciales para imágenes.", accent: "#4ade80", tasks: ["image.classification", "image.regression"] },
  vit: { name: "Transformer visual (ViT)", description: "Convierte imágenes en patches y aplica atención.", accent: "#f472b6", tasks: ["image.classification", "image.regression"] },
  unet: { name: "Encoder–decoder visual", description: "U-Net con skips editables para segmentación.", accent: "#a78bfa", tasks: ["image.segmentation.binary", "image.segmentation.multiclass"] },
  autoencoder: { name: "Autoencoder", description: "Codifica un espacio latente y reconstruye la entrada.", accent: "#60a5fa", tasks: ["tabular.reconstruction", "image.reconstruction"] }
};

export const TASKS: Record<TaskId, { name: string; category: string; modality: string; input: string; output: string; metric: string; format: string }> = {
  "tabular.classification": { name: "Clasificación tabular", category:"Tabla", modality: "Tabla", input: "Variables numéricas [F]", output: "Clase y probabilidades [K]", metric: "Accuracy", format: "CSV con columna objetivo" },
  "tabular.regression": { name: "Regresión tabular", category:"Tabla", modality: "Tabla", input: "Variables numéricas [F]", output: "Uno o varios valores [Q]", metric: "MSE", format: "CSV con objetivo(s) numéricos" },
  "tabular.reconstruction": { name:"Reconstrucción tabular", category:"Reconstrucción", modality:"Tabla", input:"Variables [F]", output:"Reconstrucción [F]", metric:"MSE", format:"CSV numérico sin objetivo obligatorio" },
  "image.classification": { name: "Clasificación de imágenes", category:"Visión", modality: "Imagen", input: "Imagen [C,H,W]", output: "Clase y probabilidades [K]", metric: "Accuracy", format: "Carpetas por clase" },
  "image.regression": { name: "Regresión visual", category:"Visión", modality: "Imagen", input: "Imagen [C,H,W]", output: "Valor(es) [Q]", metric: "MSE", format: "Imágenes + labels.csv" },
  "image.reconstruction": { name:"Reconstrucción de imágenes", category:"Reconstrucción", modality:"Imagen", input:"Imagen [C,H,W]", output:"Imagen [C,H,W]", metric:"MSE", format:"Carpeta con imágenes" },
  "sequence.classification": { name: "Clasificación de secuencias", category:"Señales y series", modality: "Secuencia", input: "Serie [T,F]", output: "Clase [K]", metric: "Accuracy", format: "CSV largo por sequence_id" },
  "sequence.regression": { name: "Regresión de secuencias", category:"Señales y series", modality: "Secuencia", input: "Serie [T,F]", output: "Valor(es) [Q]", metric: "MSE", format: "CSV largo + target por secuencia" },
  "sequence.forecast": { name: "Pronóstico temporal", category:"Señales y series", modality: "Secuencia", input: "Contexto [T,F]", output: "Horizonte [P,Q]", metric: "MSE", format: "CSV temporal con series" },
  "text.classification": { name:"Clasificación de texto", category:"Texto", modality:"Texto", input:"Tokens [T]", output:"Clase [K]", metric:"Accuracy", format:"TXT: label<TAB>texto" },
  "text.language_model": { name:"Predicción del siguiente token", category:"Texto", modality:"Texto", input:"Tokens [T]", output:"Logits [T,V]", metric:"Accuracy", format:"TXT con una muestra por línea" },
  "image.segmentation.binary": { name: "Segmentación binaria", category:"Visión", modality: "Imagen + máscara", input: "Imagen [C,H,W]", output: "Máscara [1,H,W]", metric: "Dice", format: "images/ + masks/" },
  "image.segmentation.multiclass": { name: "Segmentación multiclase", category:"Visión", modality: "Imagen + máscara", input: "Imagen [C,H,W]", output: "Máscara [K,H,W]", metric: "mIoU", format: "images/ + máscaras indexadas" }
};

export const BLOCKS: Record<Architecture, Array<{ type: string; label: string; category: string; defaults: Record<string, number | string | boolean> }>> = {
  mlp: [
    { type: "linear", label: "Densa", category: "Core", defaults: { out_features: 64, bias: true } },
    { type: "relu", label: "ReLU", category: "Activación", defaults: {} },
    { type: "leaky_relu", label: "LeakyReLU", category: "Activación", defaults: { negative_slope: 0.01 } },
    { type: "gelu", label: "GELU", category: "Activación", defaults: {} },
    { type: "sigmoid", label: "Sigmoide", category: "Activación", defaults: {} },
    { type: "tanh", label: "Tanh", category: "Activación", defaults: {} },
    { type: "silu", label: "SiLU / Swish", category: "Activación", defaults: {} },
    { type: "softmax", label: "Softmax", category: "Activación", defaults: { dim: 1 } },
    { type: "dropout", label: "Dropout", category: "Regularización", defaults: { p: 0.1 } },
    { type: "batchnorm1d", label: "BatchNorm 1D", category: "Normalización", defaults: { eps: 0.00001 } },
    { type: "layernorm", label: "LayerNorm", category: "Normalización", defaults: { eps: 0.00001 } }
  ],
  cnn1d: [
    { type:"permute", label:"T,F → F,T", category:"Forma", defaults:{ dims:"2,1" } },
    { type:"conv1d", label:"Conv 1D", category:"Convolución", defaults:{out_channels:32,kernel_size:3,stride:1,padding:1} },
    { type:"causalpad1d", label:"Padding causal 1D", category:"Convolución", defaults:{kernel_size:3,dilation:1} },
    { type:"maxpool1d", label:"MaxPool 1D", category:"Pooling", defaults:{kernel_size:2,stride:2} },
    { type:"avgpool1d", label:"AvgPool 1D", category:"Pooling", defaults:{kernel_size:2,stride:2} },
    { type:"adaptiveavgpool1d", label:"Global AvgPool 1D", category:"Pooling", defaults:{output_size:1} },
    { type:"batchnorm1d", label:"BatchNorm 1D", category:"Normalización", defaults:{} },
    { type:"relu", label:"ReLU", category:"Activación", defaults:{} },
    { type:"gelu", label:"GELU", category:"Activación", defaults:{} },
    { type:"dropout", label:"Dropout", category:"Regularización", defaults:{p:.1} },
    { type:"flatten", label:"Flatten", category:"Forma", defaults:{} },
    { type:"linear", label:"Densa", category:"Core", defaults:{out_features:3} },
    { type:"reshape", label:"Reshape", category:"Forma", defaults:{shape:"8,1"} }
  ],
  cnn: [
    { type: "conv2d", label: "Conv 2D", category: "Convolución", defaults: { out_channels: 32, kernel_size: 3, stride: 1, padding: 1 } },
    { type: "maxpool2d", label: "MaxPool 2D", category: "Pooling", defaults: { kernel_size: 2, stride: 2 } },
    { type: "avgpool2d", label: "AvgPool 2D", category: "Pooling", defaults: { kernel_size: 2, stride: 2 } },
    { type: "adaptiveavgpool2d", label: "Global AvgPool", category: "Pooling", defaults: { output_size: 1 } },
    { type: "batchnorm2d", label: "BatchNorm 2D", category: "Normalización", defaults: {} },
    { type: "relu", label: "ReLU", category: "Activación", defaults: {} },
    { type: "leaky_relu", label: "LeakyReLU", category: "Activación", defaults: { negative_slope: 0.01 } },
    { type: "gelu", label: "GELU", category: "Activación", defaults: {} },
    { type: "sigmoid", label: "Sigmoide", category: "Activación", defaults: {} },
    { type: "tanh", label: "Tanh", category: "Activación", defaults: {} },
    { type: "softmax", label: "Softmax", category: "Activación", defaults: { dim: 1 } },
    { type: "dropout2d", label: "Dropout 2D", category: "Regularización", defaults: { p: 0.1 } },
    { type: "flatten", label: "Flatten", category: "Forma", defaults: {} },
    { type: "linear", label: "Densa", category: "Core", defaults: { out_features: 3 } }
  ],
  lstm: [
    { type: "rnn", label: "RNN", category: "Recurrente", defaults: { hidden_size: 64, num_layers: 1, bidirectional: false, dropout: 0, nonlinearity:"tanh" } },
    { type: "gru", label: "GRU", category: "Recurrente", defaults: { hidden_size: 64, num_layers: 1, bidirectional: false, dropout: 0 } },
    { type: "lstm", label: "LSTM", category: "Recurrente", defaults: { hidden_size: 64, num_layers: 1, bidirectional: false, dropout: 0 } },
    { type: "temporal_select", label: "Último estado válido", category: "Secuencia", defaults: { mode: "last_valid" } },
    { type: "relu", label: "ReLU", category: "Activación", defaults: {} },
    { type: "sigmoid", label: "Sigmoide", category: "Activación", defaults: {} },
    { type: "tanh", label: "Tanh", category: "Activación", defaults: {} },
    { type: "layernorm", label: "LayerNorm", category: "Normalización", defaults: {} },
    { type: "dropout", label: "Dropout", category: "Regularización", defaults: { p: 0.1 } },
    { type: "linear", label: "Densa", category: "Core", defaults: { out_features: 3 } },
    { type: "reshape", label: "Reshape", category: "Forma", defaults: { shape: "1,1" } }
  ],
  transformer: [
    {type:"sequence_projection",label:"Proyección de secuencia",category:"Entrada",defaults:{d_model:64}},
    {type:"embedding",label:"Embedding de tokens",category:"Entrada",defaults:{vocab_size:2048,d_model:64,padding_idx:0}},
    {type:"positional_encoding",label:"Codificación posicional",category:"Posición",defaults:{max_length:2048,learned:false}},
    {type:"transformer_encoder",label:"Encoder Transformer",category:"Atención",defaults:{heads:4,num_layers:2,dim_feedforward:256,dropout:.1}},
    {type:"causal_transformer",label:"Transformer causal",category:"Atención",defaults:{heads:4,num_layers:2,dim_feedforward:256,dropout:.1}},
    {type:"sequence_pool",label:"Pooling de secuencia",category:"Secuencia",defaults:{mode:"mean"}},
    {type:"layernorm",label:"LayerNorm",category:"Normalización",defaults:{}},
    {type:"linear",label:"Densa / cabeza",category:"Core",defaults:{out_features:3}},
    {type:"reshape",label:"Reshape",category:"Forma",defaults:{shape:"8,1"}}
  ],
  transformer_causal: [
    {type:"embedding",label:"Embedding de tokens",category:"Entrada",defaults:{vocab_size:2048,d_model:64,padding_idx:0}},
    {type:"positional_encoding",label:"Codificación posicional",category:"Posición",defaults:{max_length:2048,learned:false}},
    {type:"causal_transformer",label:"Transformer causal",category:"Atención",defaults:{heads:4,num_layers:2,dim_feedforward:256,dropout:.1}},
    {type:"layernorm",label:"LayerNorm",category:"Normalización",defaults:{}},
    {type:"linear",label:"Cabeza de vocabulario",category:"Salida",defaults:{out_features:2048}}
  ],
  vit: [
    {type:"patch_embedding",label:"Patch embedding",category:"Entrada visual",defaults:{patch_size:16,d_model:64}},
    {type:"positional_encoding",label:"Posición de patches",category:"Posición",defaults:{max_length:2048,learned:true}},
    {type:"transformer_encoder",label:"Encoder Transformer",category:"Atención",defaults:{heads:4,num_layers:2,dim_feedforward:256,dropout:.1}},
    {type:"sequence_pool",label:"Pooling de patches",category:"Secuencia",defaults:{mode:"mean"}},
    {type:"layernorm",label:"LayerNorm",category:"Normalización",defaults:{}},
    {type:"linear",label:"Densa / cabeza",category:"Core",defaults:{out_features:3}},
    {type:"dropout",label:"Dropout",category:"Regularización",defaults:{p:.1}}
  ],
  unet: [
    { type: "conv2d", label: "Conv 2D", category: "Convolución", defaults: { out_channels: 16, kernel_size: 3, stride: 1, padding: 1 } },
    { type: "maxpool2d", label: "MaxPool 2D", category: "Pooling", defaults: { kernel_size: 2, stride: 2 } },
    { type: "convtranspose2d", label: "ConvTranspose 2D", category: "Upsampling", defaults: { out_channels: 16, kernel_size: 2, stride: 2 } },
    { type: "resize2d", label: "Resize 2D", category: "Upsampling", defaults: { scale_factor: 2 } },
    { type: "concat", label: "Concatenar", category: "Ramas", defaults: { axis: 1 } },
    { type: "add", label: "Sumar", category: "Ramas", defaults: {} },
    { type: "batchnorm2d", label: "BatchNorm 2D", category: "Normalización", defaults: {} },
    { type: "relu", label: "ReLU", category: "Activación", defaults: {} },
    { type: "leaky_relu", label: "LeakyReLU", category: "Activación", defaults: { negative_slope: 0.01 } },
    { type: "sigmoid", label: "Sigmoide", category: "Activación", defaults: {} },
    { type: "dropout2d", label: "Dropout 2D", category: "Regularización", defaults: { p: 0.1 } }
  ],
  autoencoder: [
    {type:"linear",label:"Densa",category:"Denso",defaults:{out_features:64}},
    {type:"conv2d",label:"Conv 2D",category:"Convolución",defaults:{out_channels:16,kernel_size:3,stride:1,padding:1}},
    {type:"convtranspose2d",label:"ConvTranspose 2D",category:"Decoder",defaults:{out_channels:3,kernel_size:2,stride:2,padding:0}},
    {type:"relu",label:"ReLU",category:"Activación",defaults:{}},
    {type:"sigmoid",label:"Sigmoide",category:"Activación",defaults:{}},
    {type:"flatten",label:"Flatten",category:"Forma",defaults:{}},
    {type:"reshape",label:"Reshape",category:"Forma",defaults:{shape:"3,256,256"}}
  ]
};

export const BLOCK_INFO: Record<string, { description: string; usage: string; diagram: string[] }> = {
  linear: { description: "Transformación aprendible que combina todas las características de entrada.", usage: "Úsala como capa oculta o cabeza de salida después de un vector/Flatten.", diagram: ["[B, F]", "Densa", "[B, N]"] },
  conv2d: { description: "Aprende filtros espaciales sobre imágenes o mapas de características.", usage: "Conéctala a tensores [B, C, H, W]; controla canales, kernel, stride y padding.", diagram: ["[C,H,W]", "Conv 2D", "[N,H′,W′]"] },
  convtranspose2d: { description: "Aumenta la resolución mediante una convolución transpuesta aprendible.", usage: "Se usa en decoders, normalmente antes de concatenar un skip de igual tamaño.", diagram: ["Mapa pequeño", "ConvTranspose", "Mapa grande"] },
  maxpool2d: { description: "Reduce la resolución conservando la activación máxima por ventana.", usage: "Colócala entre bloques convolucionales para comprimir información espacial.", diagram: ["H × W", "MaxPool", "H/2 × W/2"] },
  avgpool2d: { description: "Reduce la resolución promediando cada ventana.", usage: "Alternativa suave a MaxPool cuando importa la respuesta media.", diagram: ["H × W", "AvgPool", "H/2 × W/2"] },
  adaptiveavgpool2d: { description: "Resume cada canal a un tamaño de salida fijo.", usage: "Antes de Flatten/Densa evita depender de la resolución exacta de entrada.", diagram: ["[C,H,W]", "Global Avg", "[C,1,1]"] },
  lstm: { description: "Procesa una secuencia manteniendo un estado con memoria temporal.", usage: "Recibe [B, T, F] y suele conectarse a Último estado válido.", diagram: ["[T,F]", "LSTM", "[T,H]"] },
  temporal_select: { description: "Extrae del eje temporal el último estado utilizable.", usage: "Después de una LSTM convierte la secuencia en un vector para una cabeza Densa.", diagram: ["[T,H]", "Último", "[H]"] },
  flatten: { description: "Aplana las dimensiones no batch en un solo vector.", usage: "Conecta mapas convolucionales con capas Densas.", diagram: ["[C,H,W]", "Flatten", "[C·H·W]"] },
  reshape: { description: "Reorganiza el tensor sin cambiar su número de elementos.", usage: "Úsalo al adaptar una salida vectorial a un horizonte o forma concreta.", diagram: ["[N]", "Reshape", "[P,Q]"] },
  concat: { description: "Concatena varias ramas a lo largo del eje indicado.", usage: "Acepta varias entradas compatibles; es la unión típica de skips en U-Net.", diagram: ["Rama A + B", "Concat", "Canales A+B"] },
  add: { description: "Suma elemento a elemento dos o más ramas de igual forma.", usage: "Úsala para conexiones residuales; todas las entradas deben tener la misma shape.", diagram: ["Rama A + B", "Sumar", "Misma forma"] },
  resize2d: { description: "Redimensiona mapas de características por interpolación.", usage: "Upsampling no aprendible para decoders; suele preceder una Conv 2D.", diagram: ["Mapa pequeño", "Resize", "Mapa grande"] },
  relu: { description: "Introduce no linealidad dejando pasar valores positivos.", usage: "Normalmente se coloca después de una capa Densa o Conv 2D.", diagram: ["x", "max(0,x)", "activación"] },
  leaky_relu: { description: "Variante de ReLU que conserva una pendiente pequeña en negativos.", usage: "Útil cuando quieres reducir neuronas inactivas.", diagram: ["x", "LeakyReLU", "activación"] },
  gelu: { description: "Activación suave que pondera valores según su magnitud.", usage: "Alternativa a ReLU frecuente en redes modernas.", diagram: ["x", "GELU", "activación"] },
  silu: { description: "Activación suave también conocida como Swish.", usage: "Buena opción general en bloques densos o convolucionales.", diagram: ["x", "SiLU", "activación"] },
  sigmoid: { description: "Lleva cada valor al intervalo 0–1.", usage: "Úsala solo cuando el contrato de la salida o un bloque intermedio lo requiera.", diagram: ["logit", "Sigmoide", "0…1"] },
  tanh: { description: "Lleva cada valor al intervalo −1…1.", usage: "Útil en estados acotados; evita añadirla sin necesidad en la salida.", diagram: ["x", "Tanh", "−1…1"] },
  softmax: { description: "Convierte logits en una distribución que suma uno.", usage: "Para entrenamiento con CrossEntropy normalmente se omite y se aplica al inferir.", diagram: ["logits", "Softmax", "probabilidades"] },
  dropout: { description: "Apaga activaciones aleatoriamente durante entrenamiento.", usage: "Regulariza capas densas; valores de p entre 0.1 y 0.5 son habituales.", diagram: ["activación", "Dropout", "regularizada"] },
  dropout2d: { description: "Apaga canales completos durante entrenamiento.", usage: "Regularización específica para mapas convolucionales.", diagram: ["mapas", "Dropout 2D", "mapas"] },
  batchnorm1d: { description: "Normaliza características usando estadísticas del lote.", usage: "Para vectores o señales 1D, normalmente antes de la activación.", diagram: ["[B,F]", "BatchNorm", "[B,F]"] },
  batchnorm2d: { description: "Normaliza cada canal de mapas 2D usando el lote.", usage: "Se coloca típicamente entre Conv 2D y activación.", diagram: ["[B,C,H,W]", "BatchNorm", "misma forma"] },
  layernorm: { description: "Normaliza por muestra sobre la última dimensión.", usage: "Adecuada para secuencias y lotes pequeños.", diagram: ["[…,F]", "LayerNorm", "[…,F]"] },
};

const node = (id: string, blockType: string, label: string, x: number, y: number, properties: Record<string, string | number | boolean> = {}): ModelNode => ({ id, type: "modelNode", position: { x, y }, data: { blockType, label, category: "Modelo", properties } });
const edge = (source: string, target: string, id = `${source}-${target}`): ModelEdge => ({ id, source, target, type: "smoothstep", animated: false });

export type BuiltinPreset = {
  id: string;
  name: string;
  description: string;
  benefit: string;
  tradeoff: string;
  nodes: ModelNode[];
  edges: ModelEdge[];
};

const sequential = (nodes: ModelNode[]) => ({ nodes, edges: nodes.slice(0, -1).map((item, index) => edge(item.id, nodes[index + 1].id)) });
const outputClasses = (task: TaskId) => task === "text.language_model" ? 2048 : task.includes("segmentation.binary") ? 1 : task.includes("segmentation.multiclass") ? 4 : task.includes("classification") ? 3 : task === "sequence.forecast" ? 8 : 1;

export function templateFor(architecture: Architecture, task: TaskId): { nodes: ModelNode[]; edges: ModelEdge[] } {
  const classes = outputClasses(task);
  if (architecture === "mlp") {
    const nodes = [node("input", "input", "Datos tabulares", 30, 160), node("dense1", "linear", "Densa 64", 250, 160, { out_features: 64 }), node("relu1", "relu", "ReLU", 465, 160), node("drop1", "dropout", "Dropout", 650, 160, { p: .1 }), node("head", "linear", `Salida ${classes}`, 850, 160, { out_features: classes }), node("output", "output", "Resultado", 1050, 160)];
    return { nodes, edges: nodes.slice(0, -1).map((n, i) => edge(n.id, nodes[i + 1].id)) };
  }
  if (architecture === "cnn1d") {
    const forecast=task==="sequence.forecast";
    return sequential([node("input","input","Secuencia [T,F]",20,160),node("layout","permute","Cambiar a [F,T]",210,160,{dims:"2,1"}),node("conv","conv1d","Conv 1D · 32",410,160,{out_channels:32,kernel_size:5,padding:2}),node("relu","relu","ReLU",610,160),node("pool","adaptiveavgpool1d","Global AvgPool 1D",780,160,{output_size:1}),node("flat","flatten","Flatten",990,160),node("head","linear",`Salida ${classes}`,1160,160,{out_features:classes}),...(forecast?[node("reshape","reshape","Horizonte 8×1",1350,160,{shape:"8,1"})]:[]),node("output","output","Resultado",forecast?1550:1350,160)]);
  }
  if (architecture === "cnn") {
    const nodes = [node("input", "input", "Imagen", 20, 160), node("conv1", "conv2d", "Conv 16 · 3×3", 210, 160, { out_channels: 16, kernel_size: 3, padding: 1, stride: 1 }), node("relu1", "relu", "ReLU", 425, 160), node("pool1", "maxpool2d", "MaxPool 2×2", 595, 160, { kernel_size: 2, stride: 2 }), node("gap", "adaptiveavgpool2d", "Global AvgPool", 800, 160, { output_size: 1 }), node("flat", "flatten", "Flatten", 1010, 160), node("head", "linear", `Salida ${classes}`, 1180, 160, { out_features: classes }), node("output", "output", "Resultado", 1370, 160)];
    return { nodes, edges: nodes.slice(0, -1).map((n, i) => edge(n.id, nodes[i + 1].id)) };
  }
  if (architecture === "lstm") {
    const forecast = task === "sequence.forecast";
    const nodes = [node("input", "input", "Secuencia", 30, 160), node("lstm1", "lstm", "LSTM 64", 260, 160, { hidden_size: 64, num_layers: 1, bidirectional: false }), node("state", "temporal_select", "Último estado", 510, 160), node("head", "linear", `Salida ${classes}`, 745, 160, { out_features: classes }), ...(forecast ? [node("reshape", "reshape", "Horizonte 8×1", 950, 160, { shape: "8,1" })] : []), node("output", "output", "Resultado", forecast ? 1150 : 950, 160)];
    return { nodes, edges: nodes.slice(0, -1).map((n, i) => edge(n.id, nodes[i + 1].id)) };
  }
  if (architecture === "transformer" || architecture === "transformer_causal") {
    const text=task.startsWith("text.");const causal=task==="text.language_model";const forecast=task==="sequence.forecast";
    const nodes=[node("input","input",text?"Tokens [T]":"Secuencia [T,F]",20,160),node("embed",text?"embedding":"sequence_projection",text?"Embedding 64":"Proyección 64",220,160,text?{vocab_size:2048,d_model:64,padding_idx:0}:{d_model:64}),node("pos","positional_encoding","Posición",430,160,{max_length:2048,learned:false}),node("attention",causal?"causal_transformer":"transformer_encoder",causal?"Transformer causal":"Encoder Transformer",610,160,{heads:4,num_layers:2,dim_feedforward:256,dropout:.1}),...(causal?[]:[node("pool","sequence_pool","Pooling temporal",850,160,{mode:forecast?"last":"mean"})]),node("head","linear",`Salida ${classes}`,causal?850:1050,160,{out_features:classes}),...(forecast?[node("reshape","reshape","Horizonte 8×1",1240,160,{shape:"8,1"})]:[]),node("output","output",causal?"Logits por token":"Resultado",forecast?1440:(causal?1050:1240),160)];return sequential(nodes);
  }
  if (architecture === "vit") {
    return sequential([node("input","input","Imagen",20,160),node("patch","patch_embedding","Patches 16×16 · D64",220,160,{patch_size:16,d_model:64}),node("pos","positional_encoding","Posición de patches",455,160,{max_length:2048,learned:true}),node("attention","transformer_encoder","Encoder Transformer",680,160,{heads:4,num_layers:2,dim_feedforward:256,dropout:.1}),node("pool","sequence_pool","Promedio de patches",915,160,{mode:"mean"}),node("head","linear",`Salida ${classes}`,1130,160,{out_features:classes}),node("output","output","Resultado",1320,160)]);
  }
  if (architecture === "autoencoder") {
    if(task==="tabular.reconstruction")return sequential([node("input","input","Datos [F]",20,160),node("enc","linear","Encoder 32",220,160,{out_features:32}),node("relu1","relu","ReLU",405,160),node("latent","linear","Latente 8",570,160,{out_features:8}),node("relu2","relu","ReLU",750,160),node("dec","linear","Decoder 32",910,160,{out_features:32}),node("relu3","relu","ReLU",1100,160),node("head","linear","Reconstrucción 12",1260,160,{out_features:12}),node("output","output","Reconstrucción",1470,160)]);
    return sequential([node("input","input","Imagen",20,160),node("enc","conv2d","Encoder 16",210,160,{out_channels:16,kernel_size:4,stride:2,padding:1}),node("relu1","relu","ReLU",400,160),node("latent","conv2d","Latente 32",565,160,{out_channels:32,kernel_size:4,stride:2,padding:1}),node("relu2","relu","ReLU",755,160),node("dec1","convtranspose2d","Decoder 16",920,160,{out_channels:16,kernel_size:4,stride:2,padding:1}),node("relu3","relu","ReLU",1120,160),node("head","convtranspose2d","Reconstrucción RGB",1290,160,{out_channels:3,kernel_size:4,stride:2,padding:1}),node("sigmoid","sigmoid","Sigmoide",1510,160),node("output","output","Reconstrucción",1680,160)]);
  }
  const nodes = [node("input", "input", "Imagen", 20, 230), node("enc1", "conv2d", "Encoder Conv 16", 215, 150, { out_channels: 16, kernel_size: 3, padding: 1 }), node("pool", "maxpool2d", "MaxPool", 430, 230, { kernel_size: 2, stride: 2 }), node("bottleneck", "conv2d", "Bottleneck 32", 625, 230, { out_channels: 32, kernel_size: 3, padding: 1 }), node("up", "convtranspose2d", "Upsample 16", 840, 230, { out_channels: 16, kernel_size: 2, stride: 2 }), node("concat", "concat", "Concat skip", 1050, 230, { axis: 1 }), node("dec", "conv2d", "Decoder Conv 16", 1250, 230, { out_channels: 16, kernel_size: 3, padding: 1 }), node("head", "conv2d", `Máscara ${classes}`, 1460, 230, { out_channels: classes, kernel_size: 1, padding: 0 }), node("output", "output", "Máscara", 1660, 230)];
  return { nodes, edges: [edge("input", "enc1"), edge("enc1", "pool"), edge("pool", "bottleneck"), edge("bottleneck", "up"), edge("up", "concat"), edge("enc1", "concat", "skip"), edge("concat", "dec"), edge("dec", "head"), edge("head", "output")] };
}

function unetPreset(task: TaskId, channels: number, regularized = false) {
  const classes = outputClasses(task);
  const nodes = [
    node("input", "input", "Imagen", 20, 230),
    node("enc1", "conv2d", `Encoder Conv ${channels}`, 190, 130, { out_channels: channels, kernel_size: 3, padding: 1 }),
    ...(regularized ? [node("norm1", "batchnorm2d", "BatchNorm", 375, 130), node("relu1", "relu", "ReLU", 545, 130)] : []),
    node("pool", "maxpool2d", "MaxPool 2×2", regularized ? 710 : 410, 230, { kernel_size: 2, stride: 2 }),
    node("bottleneck", "conv2d", `Bottleneck ${channels * 2}`, regularized ? 900 : 600, 230, { out_channels: channels * 2, kernel_size: 3, padding: 1 }),
    ...(regularized ? [node("drop", "dropout2d", "Dropout 2D", 1090, 230, { p: .15 })] : []),
    node("up", "convtranspose2d", `Upsample ${channels}`, regularized ? 1265 : 795, 230, { out_channels: channels, kernel_size: 2, stride: 2 }),
    node("concat", "concat", "Concat skip", regularized ? 1465 : 990, 230, { axis: 1 }),
    node("dec", "conv2d", `Decoder Conv ${channels}`, regularized ? 1660 : 1180, 230, { out_channels: channels, kernel_size: 3, padding: 1 }),
    node("head", "conv2d", `Máscara ${classes}`, regularized ? 1855 : 1370, 230, { out_channels: classes, kernel_size: 1, padding: 0 }),
    node("output", "output", "Máscara", regularized ? 2050 : 1560, 230)
  ];
  const path = ["input", "enc1", ...(regularized ? ["norm1", "relu1"] : []), "pool", "bottleneck", ...(regularized ? ["drop"] : []), "up", "concat", "dec", "head", "output"];
  return { nodes, edges: path.slice(0, -1).map((source, index) => edge(source, path[index + 1])).concat(edge("enc1", "concat", "skip")) };
}

function residualCnnPreset(task: TaskId) {
  const classes=outputClasses(task);const nodes=[node("input","input","Imagen",20,220),node("stem","conv2d","Stem 32",200,220,{out_channels:32,kernel_size:3,padding:1}),node("conv1","conv2d","Residual Conv 32",405,125,{out_channels:32,kernel_size:3,padding:1}),node("relu1","relu","ReLU",605,125),node("conv2","conv2d","Residual Conv 32",765,125,{out_channels:32,kernel_size:3,padding:1}),node("add","add","Sumar identidad",980,220),node("relu2","relu","ReLU",1170,220),node("gap","adaptiveavgpool2d","Global AvgPool",1340,220,{output_size:1}),node("flat","flatten","Flatten",1540,220),node("head","linear",`Salida ${classes}`,1710,220,{out_features:classes}),node("output","output","Resultado",1900,220)];
  return {nodes,edges:[edge("input","stem"),edge("stem","conv1"),edge("conv1","relu1"),edge("relu1","conv2"),edge("conv2","add"),edge("stem","add","identity-skip"),edge("add","relu2"),edge("relu2","gap"),edge("gap","flat"),edge("flat","head"),edge("head","output")]};
}

/** Presets bundled with the app. Each graph is compatible with the selected task contract. */
export function presetsFor(architecture: Architecture, task: TaskId): BuiltinPreset[] {
  const classes = outputClasses(task);
  const base: BuiltinPreset = { id: "base", name: `${ARCHITECTURES[architecture].name} · Base recomendada`, description: ARCHITECTURES[architecture].description, benefit: "estructura corta y validada para empezar", tradeoff: "capacidad limitada para problemas complejos", ...templateFor(architecture, task) };

  if (architecture === "mlp") return [base,
    { id: "regularized", name: "MLP · Regularizada", description: "Dos capas densas con BatchNorm y Dropout para datos tabulares con más variación.", benefit: "mejor control del sobreajuste", tradeoff: "entrena algo más lento", ...sequential([node("input", "input", "Datos tabulares", 30, 160), node("dense1", "linear", "Densa 128", 205, 160, { out_features: 128 }), node("norm1", "batchnorm1d", "BatchNorm 1D", 390, 160), node("relu1", "relu", "ReLU", 575, 160), node("drop1", "dropout", "Dropout 0.20", 745, 160, { p: .2 }), node("dense2", "linear", "Densa 64", 930, 160, { out_features: 64 }), node("relu2", "relu", "ReLU", 1100, 160), node("head", "linear", `Salida ${classes}`, 1270, 160, { out_features: classes }), node("output", "output", "Resultado", 1450, 160)]) },
    { id: "compact", name: "MLP · Compacta", description: "Una capa oculta pequeña para iterar rápidamente o trabajar con pocos datos.", benefit: "rápida de entrenar y fácil de interpretar", tradeoff: "menor capacidad de representación", ...sequential([node("input", "input", "Datos tabulares", 30, 160), node("dense", "linear", "Densa 32", 250, 160, { out_features: 32 }), node("relu", "relu", "ReLU", 465, 160), node("head", "linear", `Salida ${classes}`, 650, 160, { out_features: classes }), node("output", "output", "Resultado", 850, 160)]) }
  ];

  if (architecture === "cnn") return [base,
    { id: "standard", name: "CNN · Clasificador estándar", description: "Dos bloques Conv–BatchNorm–ReLU con pooling y promedio global. Buen punto de partida para imágenes reales.", benefit: "combina capacidad, estabilidad y tamaño moderado", tradeoff: "requiere más datos que la base", ...sequential([node("input", "input", "Imagen", 20, 160), node("conv1", "conv2d", "Conv 32 · 3×3", 180, 160, { out_channels: 32, kernel_size: 3, padding: 1 }), node("norm1", "batchnorm2d", "BatchNorm 2D", 365, 160), node("relu1", "relu", "ReLU", 545, 160), node("pool1", "maxpool2d", "MaxPool 2×2", 705, 160, { kernel_size: 2, stride: 2 }), node("conv2", "conv2d", "Conv 64 · 3×3", 880, 160, { out_channels: 64, kernel_size: 3, padding: 1 }), node("norm2", "batchnorm2d", "BatchNorm 2D", 1065, 160), node("relu2", "relu", "ReLU", 1245, 160), node("pool2", "maxpool2d", "MaxPool 2×2", 1405, 160, { kernel_size: 2, stride: 2 }), node("gap", "adaptiveavgpool2d", "Global AvgPool", 1580, 160, { output_size: 1 }), node("flat", "flatten", "Flatten", 1780, 160), node("head", "linear", `Salida ${classes}`, 1940, 160, { out_features: classes }), node("output", "output", "Resultado", 2120, 160)]) },
    { id: "deep_regularized", name: "CNN · Profunda con regularización", description: "Cuatro convoluciones, dos escalas espaciales y Dropout 2D para clasificación visual más exigente.", benefit: "detecta patrones visuales más complejos", tradeoff: "más parámetros y mayor riesgo de sobreajuste", ...sequential([node("input", "input", "Imagen", 20, 160), node("conv1", "conv2d", "Conv 32 · 3×3", 180, 160, { out_channels: 32, kernel_size: 3, padding: 1 }), node("relu1", "relu", "ReLU", 365, 160), node("conv2", "conv2d", "Conv 32 · 3×3", 520, 160, { out_channels: 32, kernel_size: 3, padding: 1 }), node("relu2", "relu", "ReLU", 705, 160), node("pool1", "maxpool2d", "MaxPool 2×2", 860, 160, { kernel_size: 2, stride: 2 }), node("drop1", "dropout2d", "Dropout 2D 0.15", 1045, 160, { p: .15 }), node("conv3", "conv2d", "Conv 64 · 3×3", 1225, 160, { out_channels: 64, kernel_size: 3, padding: 1 }), node("relu3", "relu", "ReLU", 1410, 160), node("conv4", "conv2d", "Conv 64 · 3×3", 1565, 160, { out_channels: 64, kernel_size: 3, padding: 1 }), node("relu4", "relu", "ReLU", 1750, 160), node("pool2", "maxpool2d", "MaxPool 2×2", 1905, 160, { kernel_size: 2, stride: 2 }), node("gap", "adaptiveavgpool2d", "Global AvgPool", 2080, 160, { output_size: 1 }), node("flat", "flatten", "Flatten", 2280, 160), node("drop2", "dropout", "Dropout 0.30", 2430, 160, { p: .3 }), node("head", "linear", `Salida ${classes}`, 2600, 160, { out_features: classes }), node("output", "output", "Resultado", 2780, 160)]) },
    {id:"resnet",name:"CNN · Bloque residual",description:"Dos convoluciones y una rama identidad completamente visibles.",benefit:"facilita entrenar redes más profundas",tradeoff:"las ramas deben conservar shapes compatibles",...residualCnnPreset(task)}
  ];

  if (architecture === "lstm") return [base,
    {id:"gru",name:"Recurrente · GRU",description:"Unidad recurrente compacta sin estado de celda separado.",benefit:"menos parámetros que LSTM",tradeoff:"puede representar menos memoria",...sequential([node("input","input","Secuencia",30,160),node("gru","gru","GRU 64",240,160,{hidden_size:64,num_layers:1,bidirectional:false,dropout:0}),node("state","temporal_select","Último estado",465,160),node("head","linear",`Salida ${classes}`,675,160,{out_features:classes}),...(task==="sequence.forecast"?[node("reshape","reshape","Horizonte 8×1",865,160,{shape:"8,1"})]:[]),node("output","output","Resultado",task==="sequence.forecast"?1060:865,160)])},
    {id:"rnn",name:"Recurrente · RNN simple",description:"Recurrencia clásica con activación tanh.",benefit:"estructura educativa y ligera",tradeoff:"peor memoria de dependencias largas",...sequential([node("input","input","Secuencia",30,160),node("rnn","rnn","RNN 64",240,160,{hidden_size:64,num_layers:1,bidirectional:false,dropout:0,nonlinearity:"tanh"}),node("state","temporal_select","Último estado",465,160),node("head","linear",`Salida ${classes}`,675,160,{out_features:classes}),...(task==="sequence.forecast"?[node("reshape","reshape","Horizonte 8×1",865,160,{shape:"8,1"})]:[]),node("output","output","Resultado",task==="sequence.forecast"?1060:865,160)])},
    { id: "bidirectional", name: "LSTM · Bidireccional", description: "Lee la secuencia en ambos sentidos antes de producir la predicción final.", benefit: "aprovecha contexto pasado y futuro", tradeoff: "no es apropiada para predicción estrictamente en tiempo real", ...sequential([node("input", "input", "Secuencia", 30, 160), node("lstm", "lstm", "BiLSTM 96", 240, 160, { hidden_size: 96, num_layers: 1, bidirectional: true }), node("state", "temporal_select", "Último estado", 475, 160), node("drop", "dropout", "Dropout 0.20", 660, 160, { p: .2 }), node("head", "linear", `Salida ${classes}`, 850, 160, { out_features: classes }), ...(task === "sequence.forecast" ? [node("reshape", "reshape", "Horizonte 8×1", 1040, 160, { shape: "8,1" })] : []), node("output", "output", "Resultado", task === "sequence.forecast" ? 1235 : 1040, 160)]) },
    { id: "stacked", name: "LSTM · Apilada", description: "Dos capas recurrentes con Dropout para dinámicas temporales más ricas.", benefit: "mayor capacidad para dependencias complejas", tradeoff: "más lenta de entrenar", ...sequential([node("input", "input", "Secuencia", 30, 160), node("lstm", "lstm", "LSTM 128 × 2", 240, 160, { hidden_size: 128, num_layers: 2, dropout: .2, bidirectional: false }), node("state", "temporal_select", "Último estado", 500, 160), node("head", "linear", `Salida ${classes}`, 735, 160, { out_features: classes }), ...(task === "sequence.forecast" ? [node("reshape", "reshape", "Horizonte 8×1", 930, 160, { shape: "8,1" })] : []), node("output", "output", "Resultado", task === "sequence.forecast" ? 1130 : 930, 160)]) }
  ];

  if (["cnn1d","transformer","transformer_causal","vit","autoencoder"].includes(architecture)) return [base];

  return [base,
    { id: "light", name: "U-Net · Ligera", description: "Encoder–decoder compacto para probar rápidamente máscaras y formatos de segmentación.", benefit: "consume menos memoria", tradeoff: "pierde detalle en escenas complejas", ...unetPreset(task, 8) },
    { id: "regularized", name: "U-Net · Normalizada", description: "Añade BatchNorm, activación y Dropout al cuello de botella para una segmentación más estable.", benefit: "mejor generalización en datasets variados", tradeoff: "entrena más despacio", ...unetPreset(task, 24, true) }
  ];
}
