import type { Architecture, DatasetSummary, ModelNode, ModelEdge, TaskId } from "./types";
import type { ArchitectureDef, TaskDef, BlockDef } from "./types";

export const ARCHITECTURES: Record<Architecture, ArchitectureDef> = {
  mlp: { name: "MLP", description: "Dense layers for tabular data.", accent: "#22d3ee", tasks: ["tabular.classification", "tabular.regression"] },
  cnn1d: { name: "CNN 1D", description: "Temporal convolutions for signals and series.", accent: "#34d399", tasks: ["sequence.classification", "sequence.regression", "sequence.forecast"] },
  lstm: { name: "Recurrent network", description: "RNN, GRU and LSTM for temporal dependencies.", accent: "#fbbf24", tasks: ["sequence.classification", "sequence.regression", "sequence.forecast"] },
  transformer: { name: "Transformer encoder", description: "Bidirectional attention for series and text classification.", accent: "#fb7185", tasks: ["sequence.classification", "sequence.regression", "sequence.forecast", "text.classification"] },
  transformer_causal: { name: "Causal transformer", description: "Masked attention for predicting and generating tokens.", accent: "#f97316", tasks: ["text.language_model"] },
  cnn: { name: "CNN 2D", description: "Spatial filters for images.", accent: "#4ade80", tasks: ["image.classification", "image.regression"] },
  vit: { name: "Visual transformer (ViT)", description: "Converts images into patches and applies attention.", accent: "#f472b6", tasks: ["image.classification", "image.regression"] },
  unet: { name: "Visual encoder-decoder", description: "U-Net with editable skips for segmentation.", accent: "#a78bfa", tasks: ["image.segmentation.binary", "image.segmentation.multiclass"] },
  autoencoder: { name: "Autoencoder", description: "Encodes a latent space and reconstructs the input.", accent: "#60a5fa", tasks: ["tabular.reconstruction", "image.reconstruction"] }
};

export const TASKS: Record<TaskId, TaskDef> = {
  "tabular.classification": { name: "Tabular classification", nameKey: "catalog:tasks.tabular.classification.name", category: "Table", categoryKey: "catalog:tasks.tabular.classification.category", modality: "Table", modalityKey: "catalog:tasks.tabular.classification.modality", input: "Numerical variables [F]", inputKey: "catalog:tasks.tabular.classification.input", output: "Class and probabilities [K]", outputKey: "catalog:tasks.tabular.classification.output", metric: "Accuracy", metricKey: "catalog:tasks.tabular.classification.metric", format: "CSV with target column", formatKey: "catalog:tasks.tabular.classification.format" },
  "tabular.regression": { name: "Tabular regression", nameKey: "catalog:tasks.tabular.regression.name", category: "Table", categoryKey: "catalog:tasks.tabular.regression.category", modality: "Table", modalityKey: "catalog:tasks.tabular.regression.modality", input: "Numerical variables [F]", inputKey: "catalog:tasks.tabular.regression.input", output: "One or more values [Q]", outputKey: "catalog:tasks.tabular.regression.output", metric: "MSE", metricKey: "catalog:tasks.tabular.regression.metric", format: "CSV with numeric target(s)", formatKey: "catalog:tasks.tabular.regression.format" },
  "tabular.reconstruction": { name: "Tabular reconstruction", nameKey: "catalog:tasks.tabular.reconstruction.name", category: "Reconstruction", categoryKey: "catalog:tasks.tabular.reconstruction.category", modality: "Table", modalityKey: "catalog:tasks.tabular.reconstruction.modality", input: "Variables [F]", inputKey: "catalog:tasks.tabular.reconstruction.input", output: "Reconstruction [F]", outputKey: "catalog:tasks.tabular.reconstruction.output", metric: "MSE", metricKey: "catalog:tasks.tabular.reconstruction.metric", format: "Numeric CSV without required target", formatKey: "catalog:tasks.tabular.reconstruction.format" },
  "image.classification": { name: "Image classification", nameKey: "catalog:tasks.image.classification.name", category: "Vision", categoryKey: "catalog:tasks.image.classification.category", modality: "Image", modalityKey: "catalog:tasks.image.classification.modality", input: "Image [C,H,W]", inputKey: "catalog:tasks.image.classification.input", output: "Class and probabilities [K]", outputKey: "catalog:tasks.image.classification.output", metric: "Accuracy", metricKey: "catalog:tasks.image.classification.metric", format: "Folders per class", formatKey: "catalog:tasks.image.classification.format" },
  "image.regression": { name: "Visual regression", nameKey: "catalog:tasks.image.regression.name", category: "Vision", categoryKey: "catalog:tasks.image.regression.category", modality: "Image", modalityKey: "catalog:tasks.image.regression.modality", input: "Image [C,H,W]", inputKey: "catalog:tasks.image.regression.input", output: "Value(s) [Q]", outputKey: "catalog:tasks.image.regression.output", metric: "MSE", metricKey: "catalog:tasks.image.regression.metric", format: "Images + labels.csv", formatKey: "catalog:tasks.image.regression.format" },
  "image.reconstruction": { name: "Image reconstruction", nameKey: "catalog:tasks.image.reconstruction.name", category: "Reconstruction", categoryKey: "catalog:tasks.image.reconstruction.category", modality: "Image", modalityKey: "catalog:tasks.image.reconstruction.modality", input: "Image [C,H,W]", inputKey: "catalog:tasks.image.reconstruction.input", output: "Image [C,H,W]", outputKey: "catalog:tasks.image.reconstruction.output", metric: "MSE", metricKey: "catalog:tasks.image.reconstruction.metric", format: "Folder with images", formatKey: "catalog:tasks.image.reconstruction.format" },
  "sequence.classification": { name: "Sequence classification", nameKey: "catalog:tasks.sequence.classification.name", category: "Signals and series", categoryKey: "catalog:tasks.sequence.classification.category", modality: "Sequence", modalityKey: "catalog:tasks.sequence.classification.modality", input: "Series [T,F]", inputKey: "catalog:tasks.sequence.classification.input", output: "Class [K]", outputKey: "catalog:tasks.sequence.classification.output", metric: "Accuracy", metricKey: "catalog:tasks.sequence.classification.metric", format: "Long CSV by sequence_id", formatKey: "catalog:tasks.sequence.classification.format" },
  "sequence.regression": { name: "Sequence regression", nameKey: "catalog:tasks.sequence.regression.name", category: "Signals and series", categoryKey: "catalog:tasks.sequence.regression.category", modality: "Sequence", modalityKey: "catalog:tasks.sequence.regression.modality", input: "Series [T,F]", inputKey: "catalog:tasks.sequence.regression.input", output: "Value(s) [Q]", outputKey: "catalog:tasks.sequence.regression.output", metric: "MSE", metricKey: "catalog:tasks.sequence.regression.metric", format: "Long CSV + target per sequence", formatKey: "catalog:tasks.sequence.regression.format" },
  "sequence.forecast": { name: "Temporal forecast", nameKey: "catalog:tasks.sequence.forecast.name", category: "Signals and series", categoryKey: "catalog:tasks.sequence.forecast.category", modality: "Sequence", modalityKey: "catalog:tasks.sequence.forecast.modality", input: "Context [T,F]", inputKey: "catalog:tasks.sequence.forecast.input", output: "Horizon [P,Q]", outputKey: "catalog:tasks.sequence.forecast.output", metric: "MSE", metricKey: "catalog:tasks.sequence.forecast.metric", format: "Temporal CSV with series", formatKey: "catalog:tasks.sequence.forecast.format" },
  "text.classification": { name: "Text classification", nameKey: "catalog:tasks.text.classification.name", category: "Text", categoryKey: "catalog:tasks.text.classification.category", modality: "Text", modalityKey: "catalog:tasks.text.classification.modality", input: "Tokens [T]", inputKey: "catalog:tasks.text.classification.input", output: "Class [K]", outputKey: "catalog:tasks.text.classification.output", metric: "Accuracy", metricKey: "catalog:tasks.text.classification.metric", format: "TXT: label<TAB>text", formatKey: "catalog:tasks.text.classification.format" },
  "text.language_model": { name: "Next token prediction", nameKey: "catalog:tasks.text.language_model.name", category: "Text", categoryKey: "catalog:tasks.text.language_model.category", modality: "Text", modalityKey: "catalog:tasks.text.language_model.modality", input: "Tokens [T]", inputKey: "catalog:tasks.text.language_model.input", output: "Logits [T,V]", outputKey: "catalog:tasks.text.language_model.output", metric: "Accuracy", metricKey: "catalog:tasks.text.language_model.metric", format: "TXT with one sample per line", formatKey: "catalog:tasks.text.language_model.format" },
  "image.segmentation.binary": { name: "Binary segmentation", nameKey: "catalog:tasks.image.segmentation.binary.name", category: "Vision", categoryKey: "catalog:tasks.image.segmentation.binary.category", modality: "Image + mask", modalityKey: "catalog:tasks.image.segmentation.binary.modality", input: "Image [C,H,W]", inputKey: "catalog:tasks.image.segmentation.binary.input", output: "Mask [1,H,W]", outputKey: "catalog:tasks.image.segmentation.binary.output", metric: "Dice", metricKey: "catalog:tasks.image.segmentation.binary.metric", format: "images/ + masks/", formatKey: "catalog:tasks.image.segmentation.binary.format" },
  "image.segmentation.multiclass": { name: "Multiclass segmentation", nameKey: "catalog:tasks.image.segmentation.multiclass.name", category: "Vision", categoryKey: "catalog:tasks.image.segmentation.multiclass.category", modality: "Image + mask", modalityKey: "catalog:tasks.image.segmentation.multiclass.modality", input: "Image [C,H,W]", inputKey: "catalog:tasks.image.segmentation.multiclass.input", output: "Mask [K,H,W]", outputKey: "catalog:tasks.image.segmentation.multiclass.output", metric: "mIoU", metricKey: "catalog:tasks.image.segmentation.multiclass.metric", format: "images/ + indexed masks", formatKey: "catalog:tasks.image.segmentation.multiclass.format" }
};

export const BLOCKS: Record<Architecture, BlockDef[]> = {
  mlp: [
    { type: "linear", label: "Dense", labelKey: "catalog:blocks.linear.label", category: "Core", categoryKey: "catalog:blockCategories.Core", defaults: { out_features: 64, bias: true } },
    { type: "relu", label: "ReLU", labelKey: "catalog:blocks.relu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "leaky_relu", label: "LeakyReLU", labelKey: "catalog:blocks.leaky_relu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: { negative_slope: 0.01 } },
    { type: "gelu", label: "GELU", labelKey: "catalog:blocks.gelu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "sigmoid", label: "Sigmoid", labelKey: "catalog:blocks.sigmoid.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "tanh", label: "Tanh", labelKey: "catalog:blocks.tanh.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "silu", label: "SiLU / Swish", labelKey: "catalog:blocks.silu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "softmax", label: "Softmax", labelKey: "catalog:blocks.softmax.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: { dim: 1 } },
    { type: "dropout", label: "Dropout", labelKey: "catalog:blocks.dropout.label", category: "Regularization", categoryKey: "catalog:blockCategories.Regularization", defaults: { p: 0.1 } },
    { type: "batchnorm1d", label: "BatchNorm 1D", labelKey: "catalog:blocks.batchnorm1d.label", category: "Normalization", categoryKey: "catalog:blockCategories.Normalization", defaults: { eps: 0.00001 } },
    { type: "layernorm", label: "LayerNorm", labelKey: "catalog:blocks.layernorm.label", category: "Normalization", categoryKey: "catalog:blockCategories.Normalization", defaults: { eps: 0.00001 } }
  ],
  cnn1d: [
    { type: "permute", label: "T,F → F,T", labelKey: "catalog:blocks.permute.label", category: "Shape", categoryKey: "catalog:blockCategories.Shape", defaults: { dims: "2,1" } },
    { type: "conv1d", label: "Conv 1D", labelKey: "catalog:blocks.conv1d.label", category: "Convolution", categoryKey: "catalog:blockCategories.Convolution", defaults: { out_channels: 32, kernel_size: 3, stride: 1, padding: 1 } },
    { type: "causalpad1d", label: "Causal padding 1D", labelKey: "catalog:blocks.causalpad1d.label", category: "Convolution", categoryKey: "catalog:blockCategories.Convolution", defaults: { kernel_size: 3, dilation: 1 } },
    { type: "maxpool1d", label: "MaxPool 1D", labelKey: "catalog:blocks.maxpool1d.label", category: "Pooling", categoryKey: "catalog:blockCategories.Pooling", defaults: { kernel_size: 2, stride: 2 } },
    { type: "avgpool1d", label: "AvgPool 1D", labelKey: "catalog:blocks.avgpool1d.label", category: "Pooling", categoryKey: "catalog:blockCategories.Pooling", defaults: { kernel_size: 2, stride: 2 } },
    { type: "adaptiveavgpool1d", label: "Global AvgPool 1D", labelKey: "catalog:blocks.adaptiveavgpool1d.label", category: "Pooling", categoryKey: "catalog:blockCategories.Pooling", defaults: { output_size: 1 } },
    { type: "batchnorm1d", label: "BatchNorm 1D", labelKey: "catalog:blocks.batchnorm1d.label", category: "Normalization", categoryKey: "catalog:blockCategories.Normalization", defaults: {} },
    { type: "relu", label: "ReLU", labelKey: "catalog:blocks.relu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "gelu", label: "GELU", labelKey: "catalog:blocks.gelu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "dropout", label: "Dropout", labelKey: "catalog:blocks.dropout.label", category: "Regularization", categoryKey: "catalog:blockCategories.Regularization", defaults: { p: 0.1 } },
    { type: "flatten", label: "Flatten", labelKey: "catalog:blocks.flatten.label", category: "Shape", categoryKey: "catalog:blockCategories.Shape", defaults: {} },
    { type: "linear", label: "Dense", labelKey: "catalog:blocks.linear.label", category: "Core", categoryKey: "catalog:blockCategories.Core", defaults: { out_features: 3 } },
    { type: "reshape", label: "Reshape", labelKey: "catalog:blocks.reshape.label", category: "Shape", categoryKey: "catalog:blockCategories.Shape", defaults: { shape: "8,1" } }
  ],
  cnn: [
    { type: "conv2d", label: "Conv 2D", labelKey: "catalog:blocks.conv2d.label", category: "Convolution", categoryKey: "catalog:blockCategories.Convolution", defaults: { out_channels: 32, kernel_size: 3, stride: 1, padding: 1 } },
    { type: "maxpool2d", label: "MaxPool 2D", labelKey: "catalog:blocks.maxpool2d.label", category: "Pooling", categoryKey: "catalog:blockCategories.Pooling", defaults: { kernel_size: 2, stride: 2 } },
    { type: "avgpool2d", label: "AvgPool 2D", labelKey: "catalog:blocks.avgpool2d.label", category: "Pooling", categoryKey: "catalog:blockCategories.Pooling", defaults: { kernel_size: 2, stride: 2 } },
    { type: "adaptiveavgpool2d", label: "Global AvgPool", labelKey: "catalog:blocks.adaptiveavgpool2d.label", category: "Pooling", categoryKey: "catalog:blockCategories.Pooling", defaults: { output_size: 1 } },
    { type: "batchnorm2d", label: "BatchNorm 2D", labelKey: "catalog:blocks.batchnorm2d.label", category: "Normalization", categoryKey: "catalog:blockCategories.Normalization", defaults: {} },
    { type: "relu", label: "ReLU", labelKey: "catalog:blocks.relu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "leaky_relu", label: "LeakyReLU", labelKey: "catalog:blocks.leaky_relu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: { negative_slope: 0.01 } },
    { type: "gelu", label: "GELU", labelKey: "catalog:blocks.gelu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "sigmoid", label: "Sigmoid", labelKey: "catalog:blocks.sigmoid.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "tanh", label: "Tanh", labelKey: "catalog:blocks.tanh.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "softmax", label: "Softmax", labelKey: "catalog:blocks.softmax.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: { dim: 1 } },
    { type: "dropout2d", label: "Dropout 2D", labelKey: "catalog:blocks.dropout2d.label", category: "Regularization", categoryKey: "catalog:blockCategories.Regularization", defaults: { p: 0.1 } },
    { type: "flatten", label: "Flatten", labelKey: "catalog:blocks.flatten.label", category: "Shape", categoryKey: "catalog:blockCategories.Shape", defaults: {} },
    { type: "linear", label: "Dense", labelKey: "catalog:blocks.linear.label", category: "Core", categoryKey: "catalog:blockCategories.Core", defaults: { out_features: 3 } }
  ],
  lstm: [
    { type: "rnn", label: "RNN", labelKey: "catalog:blocks.rnn.label", category: "Recurrent", categoryKey: "catalog:blockCategories.Recurrent", defaults: { hidden_size: 64, num_layers: 1, bidirectional: false, dropout: 0, nonlinearity: "tanh" } },
    { type: "gru", label: "GRU", labelKey: "catalog:blocks.gru.label", category: "Recurrent", categoryKey: "catalog:blockCategories.Recurrent", defaults: { hidden_size: 64, num_layers: 1, bidirectional: false, dropout: 0 } },
    { type: "lstm", label: "LSTM", labelKey: "catalog:blocks.lstm.label", category: "Recurrent", categoryKey: "catalog:blockCategories.Recurrent", defaults: { hidden_size: 64, num_layers: 1, bidirectional: false, dropout: 0 } },
    { type: "temporal_select", label: "Last valid state", labelKey: "catalog:blocks.temporal_select.label", category: "Sequence", categoryKey: "catalog:blockCategories.Sequence", defaults: { mode: "last_valid" } },
    { type: "relu", label: "ReLU", labelKey: "catalog:blocks.relu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "sigmoid", label: "Sigmoid", labelKey: "catalog:blocks.sigmoid.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "tanh", label: "Tanh", labelKey: "catalog:blocks.tanh.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "layernorm", label: "LayerNorm", labelKey: "catalog:blocks.layernorm.label", category: "Normalization", categoryKey: "catalog:blockCategories.Normalization", defaults: {} },
    { type: "dropout", label: "Dropout", labelKey: "catalog:blocks.dropout.label", category: "Regularization", categoryKey: "catalog:blockCategories.Regularization", defaults: { p: 0.1 } },
    { type: "linear", label: "Dense", labelKey: "catalog:blocks.linear.label", category: "Core", categoryKey: "catalog:blockCategories.Core", defaults: { out_features: 3 } },
    { type: "reshape", label: "Reshape", labelKey: "catalog:blocks.reshape.label", category: "Shape", categoryKey: "catalog:blockCategories.Shape", defaults: { shape: "1,1" } }
  ],
  transformer: [
    { type: "sequence_projection", label: "Sequence projection", labelKey: "catalog:blocks.sequence_projection.label", category: "Input", categoryKey: "catalog:blockCategories.Input", defaults: { d_model: 64 } },
    { type: "embedding", label: "Token embedding", labelKey: "catalog:blocks.embedding.label", category: "Input", categoryKey: "catalog:blockCategories.Input", defaults: { vocab_size: 2048, d_model: 64, padding_idx: 0 } },
    { type: "positional_encoding", label: "Positional encoding", labelKey: "catalog:blocks.positional_encoding.label", category: "Position", categoryKey: "catalog:blockCategories.Position", defaults: { max_length: 2048, learned: false } },
    { type: "transformer_encoder", label: "Transformer encoder", labelKey: "catalog:blocks.transformer_encoder.label", category: "Attention", categoryKey: "catalog:blockCategories.Attention", defaults: { heads: 4, num_layers: 2, dim_feedforward: 256, dropout: 0.1 } },
    { type: "causal_transformer", label: "Causal transformer", labelKey: "catalog:blocks.causal_transformer.label", category: "Attention", categoryKey: "catalog:blockCategories.Attention", defaults: { heads: 4, num_layers: 2, dim_feedforward: 256, dropout: 0.1 } },
    { type: "sequence_pool", label: "Sequence pooling", labelKey: "catalog:blocks.sequence_pool.label", category: "Sequence", categoryKey: "catalog:blockCategories.Sequence", defaults: { mode: "mean" } },
    { type: "layernorm", label: "LayerNorm", labelKey: "catalog:blocks.layernorm.label", category: "Normalization", categoryKey: "catalog:blockCategories.Normalization", defaults: {} },
    { type: "linear", label: "Dense / head", labelKey: "catalog:blocks.linear.label", category: "Core", categoryKey: "catalog:blockCategories.Core", defaults: { out_features: 3 } },
    { type: "reshape", label: "Reshape", labelKey: "catalog:blocks.reshape.label", category: "Shape", categoryKey: "catalog:blockCategories.Shape", defaults: { shape: "8,1" } }
  ],
  transformer_causal: [
    { type: "embedding", label: "Token embedding", labelKey: "catalog:blocks.embedding.label", category: "Input", categoryKey: "catalog:blockCategories.Input", defaults: { vocab_size: 2048, d_model: 64, padding_idx: 0 } },
    { type: "positional_encoding", label: "Positional encoding", labelKey: "catalog:blocks.positional_encoding.label", category: "Position", categoryKey: "catalog:blockCategories.Position", defaults: { max_length: 2048, learned: false } },
    { type: "causal_transformer", label: "Causal transformer", labelKey: "catalog:blocks.causal_transformer.label", category: "Attention", categoryKey: "catalog:blockCategories.Attention", defaults: { heads: 4, num_layers: 2, dim_feedforward: 256, dropout: 0.1 } },
    { type: "layernorm", label: "LayerNorm", labelKey: "catalog:blocks.layernorm.label", category: "Normalization", categoryKey: "catalog:blockCategories.Normalization", defaults: {} },
    { type: "linear", label: "Vocabulary head", labelKey: "catalog:blocks.linear.label", category: "Output", categoryKey: "catalog:blockCategories.Output", defaults: { out_features: 2048 } }
  ],
  vit: [
    { type: "patch_embedding", label: "Patch embedding", labelKey: "catalog:blocks.patch_embedding.label", category: "Visual input", categoryKey: "catalog:blockCategories.Visual input", defaults: { patch_size: 16, d_model: 64 } },
    { type: "positional_encoding", label: "Patch positions", labelKey: "catalog:blocks.positional_encoding.label", category: "Position", categoryKey: "catalog:blockCategories.Position", defaults: { max_length: 2048, learned: true } },
    { type: "transformer_encoder", label: "Transformer encoder", labelKey: "catalog:blocks.transformer_encoder.label", category: "Attention", categoryKey: "catalog:blockCategories.Attention", defaults: { heads: 4, num_layers: 2, dim_feedforward: 256, dropout: 0.1 } },
    { type: "sequence_pool", label: "Patch pooling", labelKey: "catalog:blocks.sequence_pool.label", category: "Sequence", categoryKey: "catalog:blockCategories.Sequence", defaults: { mode: "mean" } },
    { type: "layernorm", label: "LayerNorm", labelKey: "catalog:blocks.layernorm.label", category: "Normalization", categoryKey: "catalog:blockCategories.Normalization", defaults: {} },
    { type: "linear", label: "Dense / head", labelKey: "catalog:blocks.linear.label", category: "Core", categoryKey: "catalog:blockCategories.Core", defaults: { out_features: 3 } },
    { type: "dropout", label: "Dropout", labelKey: "catalog:blocks.dropout.label", category: "Regularization", categoryKey: "catalog:blockCategories.Regularization", defaults: { p: 0.1 } }
  ],
  unet: [
    { type: "conv2d", label: "Conv 2D", labelKey: "catalog:blocks.conv2d.label", category: "Convolution", categoryKey: "catalog:blockCategories.Convolution", defaults: { out_channels: 16, kernel_size: 3, stride: 1, padding: 1 } },
    { type: "maxpool2d", label: "MaxPool 2D", labelKey: "catalog:blocks.maxpool2d.label", category: "Pooling", categoryKey: "catalog:blockCategories.Pooling", defaults: { kernel_size: 2, stride: 2 } },
    { type: "convtranspose2d", label: "ConvTranspose 2D", labelKey: "catalog:blocks.convtranspose2d.label", category: "Upsampling", categoryKey: "catalog:blockCategories.Upsampling", defaults: { out_channels: 16, kernel_size: 2, stride: 2 } },
    { type: "resize2d", label: "Resize 2D", labelKey: "catalog:blocks.resize2d.label", category: "Upsampling", categoryKey: "catalog:blockCategories.Upsampling", defaults: { scale_factor: 2 } },
    { type: "concat", label: "Concatenate", labelKey: "catalog:blocks.concat.label", category: "Branches", categoryKey: "catalog:blockCategories.Branches", defaults: { axis: 1 } },
    { type: "add", label: "Add", labelKey: "catalog:blocks.add.label", category: "Branches", categoryKey: "catalog:blockCategories.Branches", defaults: {} },
    { type: "batchnorm2d", label: "BatchNorm 2D", labelKey: "catalog:blocks.batchnorm2d.label", category: "Normalization", categoryKey: "catalog:blockCategories.Normalization", defaults: {} },
    { type: "relu", label: "ReLU", labelKey: "catalog:blocks.relu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "leaky_relu", label: "LeakyReLU", labelKey: "catalog:blocks.leaky_relu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: { negative_slope: 0.01 } },
    { type: "sigmoid", label: "Sigmoid", labelKey: "catalog:blocks.sigmoid.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "dropout2d", label: "Dropout 2D", labelKey: "catalog:blocks.dropout2d.label", category: "Regularization", categoryKey: "catalog:blockCategories.Regularization", defaults: { p: 0.1 } }
  ],
  autoencoder: [
    { type: "linear", label: "Dense", labelKey: "catalog:blocks.linear.label", category: "Dense", categoryKey: "catalog:blockCategories.Dense", defaults: { out_features: 64 } },
    { type: "conv2d", label: "Conv 2D", labelKey: "catalog:blocks.conv2d.label", category: "Convolution", categoryKey: "catalog:blockCategories.Convolution", defaults: { out_channels: 16, kernel_size: 3, stride: 1, padding: 1 } },
    { type: "convtranspose2d", label: "ConvTranspose 2D", labelKey: "catalog:blocks.convtranspose2d.label", category: "Decoder", categoryKey: "catalog:blockCategories.Decoder", defaults: { out_channels: 3, kernel_size: 2, stride: 2, padding: 0 } },
    { type: "relu", label: "ReLU", labelKey: "catalog:blocks.relu.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "sigmoid", label: "Sigmoid", labelKey: "catalog:blocks.sigmoid.label", category: "Activation", categoryKey: "catalog:blockCategories.Activation", defaults: {} },
    { type: "flatten", label: "Flatten", labelKey: "catalog:blocks.flatten.label", category: "Shape", categoryKey: "catalog:blockCategories.Shape", defaults: {} },
    { type: "reshape", label: "Reshape", labelKey: "catalog:blocks.reshape.label", category: "Shape", categoryKey: "catalog:blockCategories.Shape", defaults: { shape: "3,256,256" } }
  ]
};

export const BLOCK_INFO: Record<string, { description: string; usage: string; diagram: string[] }> = {
  linear: { description: "Learned transformation that combines all input features.", usage: "Use it as a hidden layer or output head after a vector/Flatten.", diagram: ["[B, F]", "Dense", "[B, N]"] },
  conv2d: { description: "Learns spatial filters over images or feature maps.", usage: "Connect it to [B, C, H, W] tensors; control channels, kernel, stride and padding.", diagram: ["[C,H,W]", "Conv 2D", "[N,H′,W′]"] },
  convtranspose2d: { description: "Increases resolution through a learned transposed convolution.", usage: "Used in decoders, usually before concatenating an equally-sized skip.", diagram: ["Small map", "ConvTranspose", "Large map"] },
  maxpool2d: { description: "Reduces resolution keeping the maximum activation per window.", usage: "Place it between convolutional blocks to compress spatial information.", diagram: ["H × W", "MaxPool", "H/2 × W/2"] },
  avgpool2d: { description: "Reduces resolution by averaging each window.", usage: "Smoother alternative to MaxPool when the mean response matters.", diagram: ["H × W", "AvgPool", "H/2 × W/2"] },
  adaptiveavgpool2d: { description: "Summarizes each channel to a fixed output size.", usage: "Before Flatten/Dense it avoids depending on the exact input resolution.", diagram: ["[C,H,W]", "Global Avg", "[C,1,1]"] },
  lstm: { description: "Processes a sequence while keeping a temporal memory state.", usage: "Receives [B, T, F] and usually connects to Last valid state.", diagram: ["[T,F]", "LSTM", "[T,H]"] },
  temporal_select: { description: "Extracts the last usable state from the temporal axis.", usage: "After an LSTM converts the sequence into a vector for a Dense head.", diagram: ["[T,H]", "Last", "[H]"] },
  flatten: { description: "Flattens non-batch dimensions into a single vector.", usage: "Connects convolutional feature maps with Dense layers.", diagram: ["[C,H,W]", "Flatten", "[C·H·W]"] },
  reshape: { description: "Reorganizes the tensor without changing its element count.", usage: "Use it when adapting a vector output to a horizon or specific shape.", diagram: ["[N]", "Reshape", "[P,Q]"] },
  concat: { description: "Concatenates several branches along the indicated axis.", usage: "Accepts several compatible inputs; typical skip union in U-Net.", diagram: ["Branch A + B", "Concat", "Channels A+B"] },
  add: { description: "Element-wise sum of two or more branches of equal shape.", usage: "Use for residual connections; all inputs must have the same shape.", diagram: ["Branch A + B", "Add", "Same shape"] },
  resize2d: { description: "Resizes feature maps by interpolation.", usage: "Non-learnable upsampling for decoders; usually precedes a Conv 2D.", diagram: ["Small map", "Resize", "Large map"] },
  relu: { description: "Introduces non-linearity by letting positive values pass.", usage: "Usually placed after a Dense or Conv 2D layer.", diagram: ["x", "max(0,x)", "activation"] },
  leaky_relu: { description: "ReLU variant that keeps a small slope for negatives.", usage: "Useful when you want to reduce dead neurons.", diagram: ["x", "LeakyReLU", "activation"] },
  gelu: { description: "Smooth activation that weights values by their magnitude.", usage: "Modern ReLU alternative common in recent networks.", diagram: ["x", "GELU", "activation"] },
  silu: { description: "Smooth activation also known as Swish.", usage: "Good general option in dense or convolutional blocks.", diagram: ["x", "SiLU", "activation"] },
  sigmoid: { description: "Maps each value to the 0–1 range.", usage: "Only use it when the output contract or an intermediate block requires it.", diagram: ["logit", "Sigmoid", "0…1"] },
  tanh: { description: "Maps each value to the −1…1 range.", usage: "Useful in bounded states; avoid adding it unnecessarily at the output.", diagram: ["x", "Tanh", "−1…1"] },
  softmax: { description: "Converts logits into a distribution that sums to one.", usage: "For CrossEntropy training it is usually omitted and applied at inference.", diagram: ["logits", "Softmax", "probabilities"] },
  dropout: { description: "Randomly disables activations during training.", usage: "Regularizes dense layers; p values between 0.1 and 0.5 are common.", diagram: ["activation", "Dropout", "regularized"] },
  dropout2d: { description: "Disables whole channels during training.", usage: "Specific regularization for convolutional feature maps.", diagram: ["maps", "Dropout 2D", "maps"] },
  batchnorm1d: { description: "Normalizes features using batch statistics.", usage: "For vectors or 1D signals, usually before activation.", diagram: ["[B,F]", "BatchNorm", "[B,F]"] },
  batchnorm2d: { description: "Normalizes each 2D map channel using the batch.", usage: "Typically placed between Conv 2D and activation.", diagram: ["[B,C,H,W]", "BatchNorm", "same shape"] },
  layernorm: { description: "Normalizes per sample over the last dimension.", usage: "Suitable for sequences and small batches.", diagram: ["[…,F]", "LayerNorm", "[…,F]"] },
};

const node = (id: string, blockType: string, label: string, x: number, y: number, properties: Record<string, string | number | boolean> = {}): ModelNode => ({ id, type: "modelNode", position: { x, y }, data: { blockType, label, category: "Model", properties } });
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
const outputClasses = (task: TaskId) => ({
  "tabular.classification": 3,
  "tabular.regression": 1,
  "tabular.reconstruction": 25,
  "image.classification": 10,
  "image.regression": 1,
  "image.reconstruction": 1,
  "sequence.classification": 5,
  "sequence.regression": 1,
  "sequence.forecast": 8,
  "text.classification": 6,
  "text.language_model": 4096,
  "image.segmentation.binary": 1,
  "image.segmentation.multiclass": 5,
}[task]);

export function templateFor(architecture: Architecture, task: TaskId): { nodes: ModelNode[]; edges: ModelEdge[] } {
  const classes = outputClasses(task);
  if (architecture === "mlp") {
    const nodes = [node("input", "input", "Tabular data", 30, 160), node("dense1", "linear", "Dense 64", 250, 160, { out_features: 64 }), node("relu1", "relu", "ReLU", 465, 160), node("drop1", "dropout", "Dropout", 650, 160, { p: 0.1 }), node("head", "linear", `Output ${classes}`, 850, 160, { out_features: classes }), node("output", "output", "Result", 1050, 160)];
    return { nodes, edges: nodes.slice(0, -1).map((n, i) => edge(n.id, nodes[i + 1].id)) };
  }
  if (architecture === "cnn1d") {
    const forecast = task === "sequence.forecast";
    return sequential([node("input", "input", "Sequence [T,F]", 20, 160), node("layout", "permute", "Permute to [F,T]", 205, 160, { dims: "2,1" }), node("conv1", "conv1d", "Conv 1D · 32", 400, 160, { out_channels: 32, kernel_size: 5, padding: 2 }), node("relu1", "relu", "ReLU", 585, 160), node("pool1", "maxpool1d", "MaxPool 2", 735, 160, { kernel_size: 2, stride: 2 }), node("conv2", "conv1d", "Conv 1D · 64", 900, 160, { out_channels: 64, kernel_size: 3, padding: 1 }), node("relu2", "relu", "ReLU", 1085, 160), node("pool", "adaptiveavgpool1d", "Temporal summary · 4", 1245, 160, { output_size: 4 }), node("flat", "flatten", "Flatten", 1440, 160), node("head", "linear", `Output ${classes}`, 1595, 160, { out_features: classes }), ...(forecast ? [node("reshape", "reshape", "Horizon 8×1", 1780, 160, { shape: "8,1" })] : []), node("output", "output", "Result", forecast ? 1970 : 1780, 160)]);
  }
  if (architecture === "cnn") {
    const regression = task === "image.regression";
    const nodes = [node("input", "input", "Image", 20, 160), node("conv1", "conv2d", "Conv 16 · 3×3", 180, 160, { out_channels: 16, kernel_size: 3, padding: 1 }), node("norm1", "batchnorm2d", "BatchNorm", 345, 160), node("relu1", "relu", "ReLU", 505, 160), node("pool1", "maxpool2d", "MaxPool 2×2", 650, 160, { kernel_size: 2, stride: 2 }), node("conv2", "conv2d", "Conv 32 · 3×3", 815, 160, { out_channels: 32, kernel_size: 3, padding: 1 }), node("norm2", "batchnorm2d", "BatchNorm", 980, 160), node("relu2", "relu", "ReLU", 1140, 160), node("pool2", "maxpool2d", "MaxPool 2×2", 1285, 160, { kernel_size: 2, stride: 2 }), ...(regression ? [node("conv3", "conv2d", "Conv 64 · 3×3", 1450, 160, { out_channels: 64, kernel_size: 3, padding: 1 }), node("relu3", "relu", "ReLU", 1620, 160)] : []), node("gap", "adaptiveavgpool2d", "Global AvgPool", regression ? 1770 : 1450, 160, { output_size: 1 }), node("flat", "flatten", "Flatten", regression ? 1960 : 1640, 160), node("head", "linear", `Output ${classes}`, regression ? 2115 : 1795, 160, { out_features: classes }), node("output", "output", "Result", regression ? 2295 : 1975, 160)];
    return { nodes, edges: nodes.slice(0, -1).map((n, i) => edge(n.id, nodes[i + 1].id)) };
  }
  if (architecture === "lstm") {
    const forecast = task === "sequence.forecast";
    const nodes = [node("input", "input", "Sequence", 30, 160), node("lstm1", "lstm", "LSTM 64", 260, 160, { hidden_size: 64, num_layers: 1, bidirectional: false }), node("state", "temporal_select", "Last state", 510, 160), node("head", "linear", `Output ${classes}`, 745, 160, { out_features: classes }), ...(forecast ? [node("reshape", "reshape", "Horizon 8×1", 950, 160, { shape: "8,1" })] : []), node("output", "output", "Result", forecast ? 1150 : 950, 160)];
    return { nodes, edges: nodes.slice(0, -1).map((n, i) => edge(n.id, nodes[i + 1].id)) };
  }
  if (architecture === "transformer" || architecture === "transformer_causal") {
    const text = task.startsWith("text."); const causal = task === "text.language_model"; const forecast = task === "sequence.forecast";
    const nodes = [node("input", "input", text ? "Tokens [T]" : "Sequence [T,F]", 20, 160), node("embed", text ? "embedding" : "sequence_projection", text ? "Embedding 64" : "Projection 64", 220, 160, text ? { vocab_size: 4096, d_model: 64, padding_idx: 0 } : { d_model: 64 }), node("pos", "positional_encoding", "Position", 430, 160, { max_length: 256, learned: false }), node("attention", causal ? "causal_transformer" : "transformer_encoder", causal ? "Causal transformer" : "Transformer encoder", 610, 160, { heads: 4, num_layers: 2, dim_feedforward: 256, dropout: 0.1 }), ...(causal ? [] : [node("pool", "sequence_pool", text ? "Temporal pooling" : "Sequence pooling", 850, 160, { mode: forecast ? "last" : "mean" })]), node("head", "linear", `Output ${classes}`, causal ? 850 : 1050, 160, { out_features: classes }), ...(forecast ? [node("reshape", "reshape", "Horizon 8×1", 1240, 160, { shape: "8,1" })] : []), node("output", "output", causal ? "Logits per token" : "Result", forecast ? 1440 : (causal ? 1050 : 1240), 160)]; return sequential(nodes);
  }
  if (architecture === "vit") {
    const patch = task === "image.classification" ? 7 : 8;
    return sequential([node("input", "input", "Image", 20, 160), node("patch", "patch_embedding", `Patches ${patch}×${patch} · D64`, 220, 160, { patch_size: patch, d_model: 64 }), node("pos", "positional_encoding", "Patch positions", 455, 160, { max_length: 128, learned: true }), node("attention", "transformer_encoder", "Transformer encoder", 680, 160, { heads: 4, num_layers: 2, dim_feedforward: 256, dropout: 0.1 }), node("pool", "sequence_pool", "Patch average", 915, 160, { mode: "mean" }), node("head", "linear", `Output ${classes}`, 1130, 160, { out_features: classes }), node("output", "output", "Result", 1320, 160)]);
  }
  if (architecture === "autoencoder") {
    if (task === "tabular.reconstruction") return sequential([node("input", "input", "Data [F]", 20, 160), node("enc", "linear", "Encoder 32", 220, 160, { out_features: 32 }), node("relu1", "relu", "ReLU", 405, 160), node("latent", "linear", "Latent 8", 570, 160, { out_features: 8 }), node("relu2", "relu", "ReLU", 750, 160), node("dec", "linear", "Decoder 32", 910, 160, { out_features: 32 }), node("relu3", "relu", "ReLU", 1100, 160), node("head", "linear", "Reconstruction 25", 1260, 160, { out_features: 25 }), node("output", "output", "Reconstruction", 1470, 160)]);
    return sequential([node("input", "input", "Image", 20, 160), node("enc", "conv2d", "Encoder 16", 210, 160, { out_channels: 16, kernel_size: 4, stride: 2, padding: 1 }), node("relu1", "relu", "ReLU", 400, 160), node("latent", "conv2d", "Latent 32", 565, 160, { out_channels: 32, kernel_size: 4, stride: 2, padding: 1 }), node("relu2", "relu", "ReLU", 755, 160), node("dec1", "convtranspose2d", "Decoder 16", 920, 160, { out_channels: 16, kernel_size: 4, stride: 2, padding: 1 }), node("relu3", "relu", "ReLU", 1120, 160), node("head", "convtranspose2d", "Reconstruction grayscale", 1290, 160, { out_channels: 1, kernel_size: 4, stride: 2, padding: 1 }), node("sigmoid", "sigmoid", "Sigmoid", 1510, 160), node("output", "output", "Reconstruction", 1680, 160)]);
  }
  const nodes = [node("input", "input", "Image", 20, 250), node("enc1", "conv2d", "Encoder 16", 190, 90, { out_channels: 16, kernel_size: 3, padding: 1 }), node("relu1", "relu", "ReLU", 355, 90), node("pool1", "maxpool2d", "MaxPool", 525, 190, { kernel_size: 2, stride: 2 }), node("enc2", "conv2d", "Encoder 32", 690, 120, { out_channels: 32, kernel_size: 3, padding: 1 }), node("relu2", "relu", "ReLU", 855, 120), node("pool2", "maxpool2d", "MaxPool", 1020, 250, { kernel_size: 2, stride: 2 }), node("bottleneck", "conv2d", "Bottleneck 64", 1190, 250, { out_channels: 64, kernel_size: 3, padding: 1 }), node("reluB", "relu", "ReLU", 1360, 250), node("up2", "convtranspose2d", "Upsample 32", 1525, 250, { out_channels: 32, kernel_size: 2, stride: 2 }), node("concat2", "concat", "Concat skip 2", 1700, 180, { axis: 1 }), node("dec2", "conv2d", "Decoder 32", 1870, 180, { out_channels: 32, kernel_size: 3, padding: 1 }), node("reluD2", "relu", "ReLU", 2040, 180), node("up1", "convtranspose2d", "Upsample 16", 2205, 180, { out_channels: 16, kernel_size: 2, stride: 2 }), node("concat1", "concat", "Concat skip 1", 2380, 100, { axis: 1 }), node("dec1", "conv2d", "Decoder 16", 2550, 100, { out_channels: 16, kernel_size: 3, padding: 1 }), node("reluD1", "relu", "ReLU", 2720, 100), node("head", "conv2d", `Mask ${classes}`, 2885, 100, { out_channels: classes, kernel_size: 1, padding: 0 }), node("output", "output", "Mask", 3060, 100)];
  return { nodes, edges: [edge("input", "enc1"), edge("enc1", "relu1"), edge("relu1", "pool1"), edge("pool1", "enc2"), edge("enc2", "relu2"), edge("relu2", "pool2"), edge("pool2", "bottleneck"), edge("bottleneck", "reluB"), edge("reluB", "up2"), edge("up2", "concat2"), edge("relu2", "concat2", "skip2"), edge("concat2", "dec2"), edge("dec2", "reluD2"), edge("reluD2", "up1"), edge("up1", "concat1"), edge("relu1", "concat1", "skip1"), edge("concat1", "dec1"), edge("dec1", "reluD1"), edge("reluD1", "head"), edge("head", "output")] };
}

/** Adapts a starter graph to the concrete dataset contract without rebuilding a user's graph. */
export function syncOutputContract(nodes: ModelNode[], edges: ModelEdge[], dataset: DatasetSummary | null): ModelNode[] {
  if (!dataset) return nodes;
  const outputIds = new Set(nodes.filter(item => item.data.blockType === "output").map(item => item.id));
  const byId = new Map(nodes.map(item => [item.id, item]));
  const headIds = new Set<string>();
  const outputPathIds = new Set<string>();
  const frontier = edges.filter(item => outputIds.has(item.target)).map(item => item.source);
  while (frontier.length) {
    const id = frontier.shift()!;
    if (outputPathIds.has(id)) continue;
    outputPathIds.add(id);
    const block = byId.get(id)?.data.blockType;
    if (["linear", "conv2d", "convtranspose2d"].includes(block || "")) {
      headIds.add(id);
      continue;
    }
    frontier.push(...edges.filter(item => item.target === id).map(item => item.source));
  }
  return nodes.map(item => {
    if (item.data.blockType === "embedding" && dataset.options?.vocab_size) {
      return { ...item, data: { ...item.data, properties: { ...item.data.properties, vocab_size: dataset.options.vocab_size } } };
    }
    if (item.data.blockType === "positional_encoding" && dataset.options?.max_length) {
      return { ...item, data: { ...item.data, properties: { ...item.data.properties, max_length: dataset.options.max_length } } };
    }
    if (item.data.blockType === "reshape" && outputPathIds.has(item.id) && dataset.outputShape.length > 1) {
      return { ...item, data: { ...item.data, label: `Output ${dataset.outputShape.join("×")}`, properties: { ...item.data.properties, shape: dataset.outputShape.join(",") } } };
    }
    if (!headIds.has(item.id)) return item;
    const key = item.data.blockType === "linear" ? "out_features" : "out_channels";
    const required = item.data.blockType === "linear" && dataset.outputShape.length === 2
      ? (dataset.inputShape.length === 1 ? dataset.outputShape.at(-1)! : dataset.outputShape.reduce((left, right) => left * right, 1))
      : dataset.outputShape[0];
    return { ...item, data: { ...item.data, label: `${item.data.label.split(" ·")[0]} · ${required}`, properties: { ...item.data.properties, [key]: required } } };
  });
}

function unetPreset(task: TaskId, channels: number, regularized = false) {
  const classes = outputClasses(task);
  const nodes = [
    node("input", "input", "Image", 20, 230),
    node("enc1", "conv2d", `Encoder Conv ${channels}`, 190, 130, { out_channels: channels, kernel_size: 3, padding: 1 }),
    ...(regularized ? [node("norm1", "batchnorm2d", "BatchNorm", 375, 130), node("relu1", "relu", "ReLU", 545, 130)] : []),
    node("pool", "maxpool2d", "MaxPool 2×2", regularized ? 710 : 410, 230, { kernel_size: 2, stride: 2 }),
    node("bottleneck", "conv2d", `Bottleneck ${channels * 2}`, regularized ? 900 : 600, 230, { out_channels: channels * 2, kernel_size: 3, padding: 1 }),
    ...(regularized ? [node("drop", "dropout2d", "Dropout 2D", 1090, 230, { p: 0.15 })] : []),
    node("up", "convtranspose2d", `Upsample ${channels}`, regularized ? 1265 : 795, 230, { out_channels: channels, kernel_size: 2, stride: 2 }),
    node("concat", "concat", "Concat skip", regularized ? 1465 : 990, 230, { axis: 1 }),
    node("dec", "conv2d", `Decoder Conv ${channels}`, regularized ? 1660 : 1180, 230, { out_channels: channels, kernel_size: 3, padding: 1 }),
    node("head", "conv2d", `Mask ${classes}`, regularized ? 1855 : 1370, 230, { out_channels: classes, kernel_size: 1, padding: 0 }),
    node("output", "output", "Mask", regularized ? 2050 : 1560, 230)
  ];
  const path = ["input", "enc1", ...(regularized ? ["norm1", "relu1"] : []), "pool", "bottleneck", ...(regularized ? ["drop"] : []), "up", "concat", "dec", "head", "output"];
  return { nodes, edges: path.slice(0, -1).map((source, index) => edge(source, path[index + 1])).concat(edge("enc1", "concat", "skip")) };
}

function residualCnnPreset(task: TaskId) {
  const classes = outputClasses(task); const nodes = [node("input", "input", "Image", 20, 220), node("stem", "conv2d", "Stem 32", 200, 220, { out_channels: 32, kernel_size: 3, padding: 1 }), node("conv1", "conv2d", "Residual Conv 32", 405, 125, { out_channels: 32, kernel_size: 3, padding: 1 }), node("relu1", "relu", "ReLU", 605, 125), node("conv2", "conv2d", "Residual Conv 32", 765, 125, { out_channels: 32, kernel_size: 3, padding: 1 }), node("add", "add", "Add identity", 980, 220), node("relu2", "relu", "ReLU", 1170, 220), node("gap", "adaptiveavgpool2d", "Global AvgPool", 1340, 220, { output_size: 1 }), node("flat", "flatten", "Flatten", 1540, 220), node("head", "linear", `Output ${classes}`, 1710, 220, { out_features: classes }), node("output", "output", "Result", 1900, 220)];
  return { nodes, edges: [edge("input", "stem"), edge("stem", "conv1"), edge("conv1", "relu1"), edge("relu1", "conv2"), edge("conv2", "add"), edge("stem", "add", "identity-skip"), edge("add", "relu2"), edge("relu2", "gap"), edge("gap", "flat"), edge("flat", "head"), edge("head", "output")] };
}

/** Presets bundled with the app. Each graph is compatible with the selected task contract. */
export function presetsFor(architecture: Architecture, task: TaskId): BuiltinPreset[] {
  const classes = outputClasses(task);
  const base: BuiltinPreset = { id: "base", name: `${ARCHITECTURES[architecture].name} · Recommended base`, description: ARCHITECTURES[architecture].description, benefit: "Short validated structure to start with", tradeoff: "Limited capacity for complex problems", ...templateFor(architecture, task) };

  if (architecture === "mlp") return [base,
    { id: "regularized", name: "MLP · Regularized", description: "Two dense layers with BatchNorm and Dropout for tabular data with more variation.", benefit: "Better overfitting control", tradeoff: "Trains somewhat slower", ...sequential([node("input", "input", "Tabular data", 30, 160), node("dense1", "linear", "Dense 128", 205, 160, { out_features: 128 }), node("norm1", "batchnorm1d", "BatchNorm 1D", 390, 160), node("relu1", "relu", "ReLU", 575, 160), node("drop1", "dropout", "Dropout 0.20", 745, 160, { p: 0.2 }), node("dense2", "linear", "Dense 64", 930, 160, { out_features: 64 }), node("relu2", "relu", "ReLU", 1100, 160), node("head", "linear", `Output ${classes}`, 1270, 160, { out_features: classes }), node("output", "output", "Result", 1450, 160)]) },
    { id: "compact", name: "MLP · Compact", description: "A small hidden layer to iterate quickly or work with little data.", benefit: "Fast to train and easy to interpret", tradeoff: "Lower representation capacity", ...sequential([node("input", "input", "Tabular data", 30, 160), node("dense", "linear", "Dense 32", 250, 160, { out_features: 32 }), node("relu", "relu", "ReLU", 465, 160), node("head", "linear", `Output ${classes}`, 650, 160, { out_features: classes }), node("output", "output", "Result", 850, 160)]) }
  ];

  if (architecture === "cnn") return [base,
    { id: "standard", name: "CNN · Standard classifier", description: "Two Conv–BatchNorm–ReLU blocks with pooling and global average. Good starting point for real images.", benefit: "Balances capacity, stability and moderate size", tradeoff: "Requires more data than the base", ...sequential([node("input", "input", "Image", 20, 160), node("conv1", "conv2d", "Conv 32 · 3×3", 180, 160, { out_channels: 32, kernel_size: 3, padding: 1 }), node("norm1", "batchnorm2d", "BatchNorm 2D", 365, 160), node("relu1", "relu", "ReLU", 545, 160), node("pool1", "maxpool2d", "MaxPool 2×2", 705, 160, { kernel_size: 2, stride: 2 }), node("conv2", "conv2d", "Conv 64 · 3×3", 880, 160, { out_channels: 64, kernel_size: 3, padding: 1 }), node("norm2", "batchnorm2d", "BatchNorm 2D", 1065, 160), node("relu2", "relu", "ReLU", 1245, 160), node("pool2", "maxpool2d", "MaxPool 2×2", 1405, 160, { kernel_size: 2, stride: 2 }), node("gap", "adaptiveavgpool2d", "Global AvgPool", 1580, 160, { output_size: 1 }), node("flat", "flatten", "Flatten", 1780, 160), node("head", "linear", `Output ${classes}`, 1940, 160, { out_features: classes }), node("output", "output", "Result", 2120, 160)]) },
    { id: "deep_regularized", name: "CNN · Deep with regularization", description: "Four convolutions, two spatial scales and 2D Dropout for more demanding visual classification.", benefit: "Detects more complex visual patterns", tradeoff: "More parameters and higher overfitting risk", ...sequential([node("input", "input", "Image", 20, 160), node("conv1", "conv2d", "Conv 32 · 3×3", 180, 160, { out_channels: 32, kernel_size: 3, padding: 1 }), node("relu1", "relu", "ReLU", 365, 160), node("conv2", "conv2d", "Conv 32 · 3×3", 520, 160, { out_channels: 32, kernel_size: 3, padding: 1 }), node("relu2", "relu", "ReLU", 705, 160), node("pool1", "maxpool2d", "MaxPool 2×2", 860, 160, { kernel_size: 2, stride: 2 }), node("drop1", "dropout2d", "Dropout 2D 0.15", 1045, 160, { p: 0.15 }), node("conv3", "conv2d", "Conv 64 · 3×3", 1225, 160, { out_channels: 64, kernel_size: 3, padding: 1 }), node("relu3", "relu", "ReLU", 1410, 160), node("conv4", "conv2d", "Conv 64 · 3×3", 1565, 160, { out_channels: 64, kernel_size: 3, padding: 1 }), node("relu4", "relu", "ReLU", 1750, 160), node("pool2", "maxpool2d", "MaxPool 2×2", 1905, 160, { kernel_size: 2, stride: 2 }), node("gap", "adaptiveavgpool2d", "Global AvgPool", 2080, 160, { output_size: 1 }), node("flat", "flatten", "Flatten", 2280, 160), node("drop2", "dropout", "Dropout 0.30", 2430, 160, { p: 0.3 }), node("head", "linear", `Output ${classes}`, 2600, 160, { out_features: classes }), node("output", "output", "Result", 2780, 160)]) },
    { id: "resnet", name: "CNN · Residual block", description: "Two convolutions and a fully visible identity branch.", benefit: "Makes training deeper networks easier", tradeoff: "Branches must keep compatible shapes", ...residualCnnPreset(task) }
  ];

  if (architecture === "lstm") return [base,
    { id: "gru", name: "Recurrent · GRU", description: "Compact recurrent unit without a separate cell state.", benefit: "Fewer parameters than LSTM", tradeoff: "May represent less memory", ...sequential([node("input", "input", "Sequence", 30, 160), node("gru", "gru", "GRU 64", 240, 160, { hidden_size: 64, num_layers: 1, bidirectional: false, dropout: 0 }), node("state", "temporal_select", "Last state", 465, 160), node("head", "linear", `Output ${classes}`, 675, 160, { out_features: classes }), ...(task === "sequence.forecast" ? [node("reshape", "reshape", "Horizon 8×1", 865, 160, { shape: "8,1" })] : []), node("output", "output", "Result", task === "sequence.forecast" ? 1060 : 865, 160)]) },
    { id: "rnn", name: "Recurrent · Simple RNN", description: "Classic recurrence with tanh activation.", benefit: "Educational and lightweight structure", tradeoff: "Worse memory of long dependencies", ...sequential([node("input", "input", "Sequence", 30, 160), node("rnn", "rnn", "RNN 64", 240, 160, { hidden_size: 64, num_layers: 1, bidirectional: false, dropout: 0, nonlinearity: "tanh" }), node("state", "temporal_select", "Last state", 465, 160), node("head", "linear", `Output ${classes}`, 675, 160, { out_features: classes }), ...(task === "sequence.forecast" ? [node("reshape", "reshape", "Horizon 8×1", 865, 160, { shape: "8,1" })] : []), node("output", "output", "Result", task === "sequence.forecast" ? 1060 : 865, 160)]) },
    { id: "bidirectional", name: "LSTM · Bidirectional", description: "Reads the sequence in both directions before producing the final prediction.", benefit: "Leverages past and future context", tradeoff: "Not appropriate for strictly real-time prediction", ...sequential([node("input", "input", "Sequence", 30, 160), node("lstm", "lstm", "BiLSTM 96", 240, 160, { hidden_size: 96, num_layers: 1, bidirectional: true }), node("state", "temporal_select", "Last state", 475, 160), node("drop", "dropout", "Dropout 0.20", 660, 160, { p: 0.2 }), node("head", "linear", `Output ${classes}`, 850, 160, { out_features: classes }), ...(task === "sequence.forecast" ? [node("reshape", "reshape", "Horizon 8×1", 1040, 160, { shape: "8,1" })] : []), node("output", "output", "Result", task === "sequence.forecast" ? 1235 : 1040, 160)]) },
    { id: "stacked", name: "LSTM · Stacked", description: "Two recurrent layers with Dropout for richer temporal dynamics.", benefit: "Greater capacity for complex dependencies", tradeoff: "Slower to train", ...sequential([node("input", "input", "Sequence", 30, 160), node("lstm", "lstm", "LSTM 128 × 2", 240, 160, { hidden_size: 128, num_layers: 2, dropout: 0.2, bidirectional: false }), node("state", "temporal_select", "Last state", 500, 160), node("head", "linear", `Output ${classes}`, 735, 160, { out_features: classes }), ...(task === "sequence.forecast" ? [node("reshape", "reshape", "Horizon 8×1", 930, 160, { shape: "8,1" })] : []), node("output", "output", "Result", task === "sequence.forecast" ? 1130 : 930, 160)]) }
  ];

  if (architecture === "cnn1d") return [base,
    { id: "multiscale", name: "CNN 1D · Multiscale", description: "Two temporal convolution stages capture short and medium-range patterns.", benefit: "Richer temporal features", tradeoff: "Higher compute cost", ...sequential([node("input", "input", "Sequence [T,F]", 20, 160), node("layout", "permute", "Permute to [F,T]", 205, 160, { dims: "2,1" }), node("conv1", "conv1d", "Conv 32 · k5", 405, 160, { out_channels: 32, kernel_size: 5, padding: 2 }), node("relu1", "relu", "ReLU", 590, 160), node("pool1", "maxpool1d", "MaxPool 2", 745, 160, { kernel_size: 2, stride: 2 }), node("conv2", "conv1d", "Conv 64 · k3", 920, 160, { out_channels: 64, kernel_size: 3, padding: 1 }), node("relu2", "relu", "ReLU", 1105, 160), node("gap", "adaptiveavgpool1d", "Global AvgPool 1D", 1260, 160, { output_size: 1 }), node("flat", "flatten", "Flatten", 1460, 160), node("head", "linear", `Output ${classes}`, 1620, 160, { out_features: classes }), ...(task === "sequence.forecast" ? [node("reshape", "reshape", "Horizon 8×1", 1810, 160, { shape: "8,1" })] : []), node("output", "output", "Result", task === "sequence.forecast" ? 2010 : 1810, 160)]) },
    { id: "causal", name: "CNN 1D · Causal", description: "Dilated causal convolutions preserve temporal order for forecasting.", benefit: "Avoids future information leakage", tradeoff: "Less context than deeper stacks", ...sequential([node("input", "input", "Sequence [T,F]", 20, 160), node("layout", "permute", "Permute to [F,T]", 205, 160, { dims: "2,1" }), node("pad1", "causalpad1d", "Causal pad · d1", 405, 160, { kernel_size: 3, dilation: 1 }), node("conv1", "conv1d", "Causal Conv 32", 600, 160, { out_channels: 32, kernel_size: 3, padding: 0, dilation: 1 }), node("gelu", "gelu", "GELU", 795, 160), node("gap", "adaptiveavgpool1d", "Global AvgPool 1D", 950, 160, { output_size: 1 }), node("flat", "flatten", "Flatten", 1150, 160), node("head", "linear", `Output ${classes}`, 1310, 160, { out_features: classes }), ...(task === "sequence.forecast" ? [node("reshape", "reshape", "Horizon 8×1", 1500, 160, { shape: "8,1" })] : []), node("output", "output", "Result", task === "sequence.forecast" ? 1700 : 1500, 160)]) }
  ];

  if (architecture === "transformer") {
    const text = task.startsWith("text."); const forecast = task === "sequence.forecast";
    const transformerPreset = (id:string, dModel:number, heads:number, layers:number, dropout:number): BuiltinPreset => ({ id, name:id, description:id, benefit:id, tradeoff:id, ...sequential([node("input", "input", text ? "Tokens [T]" : "Sequence [T,F]", 20, 160), node("embed", text ? "embedding" : "sequence_projection", `Projection ${dModel}`, 220, 160, text ? { vocab_size: 2048, d_model: dModel, padding_idx: 0 } : { d_model: dModel }), node("pos", "positional_encoding", "Position", 425, 160, { max_length: 2048, learned: false }), node("attention", "transformer_encoder", `Encoder ${layers}L · ${heads}H`, 610, 160, { heads, num_layers: layers, dim_feedforward: dModel * 4, dropout }), node("pool", "sequence_pool", forecast ? "Last token" : "Mean pooling", 850, 160, { mode: forecast ? "last" : "mean" }), node("drop", "dropout", `Dropout ${dropout}`, 1040, 160, { p: dropout }), node("head", "linear", `Output ${classes}`, 1220, 160, { out_features: classes }), ...(forecast ? [node("reshape", "reshape", "Horizon 8×1", 1410, 160, { shape: "8,1" })] : []), node("output", "output", "Result", forecast ? 1610 : 1410, 160)]) });
    return [base, transformerPreset("compact", 32, 2, 1, 0.05), transformerPreset("deep", 128, 8, 4, 0.2)];
  }

  if (architecture === "transformer_causal") {
    const causalPreset = (id:string, dModel:number, heads:number, layers:number, dropout:number): BuiltinPreset => ({ id, name:id, description:id, benefit:id, tradeoff:id, ...sequential([node("input", "input", "Tokens [T]", 20, 160), node("embed", "embedding", `Embedding ${dModel}`, 220, 160, { vocab_size: 2048, d_model: dModel, padding_idx: 0 }), node("pos", "positional_encoding", "Position", 425, 160, { max_length: 2048, learned: true }), node("attention", "causal_transformer", `Causal ${layers}L · ${heads}H`, 610, 160, { heads, num_layers: layers, dim_feedforward: dModel * 4, dropout }), node("head", "linear", "Vocabulary logits", 850, 160, { out_features: classes }), node("output", "output", "Logits per token", 1050, 160)]) });
    return [base, causalPreset("tiny", 32, 2, 1, 0.05), causalPreset("context", 128, 8, 4, 0.2)];
  }

  if (architecture === "vit") {
    const vitPreset = (id:string, patch:number, dModel:number, heads:number, layers:number, dropout:number): BuiltinPreset => ({ id, name:id, description:id, benefit:id, tradeoff:id, ...sequential([node("input", "input", "Image", 20, 160), node("patch", "patch_embedding", `Patches ${patch}×${patch} · D${dModel}`, 220, 160, { patch_size: patch, d_model: dModel }), node("pos", "positional_encoding", "Patch positions", 455, 160, { max_length: 2048, learned: true }), node("attention", "transformer_encoder", `Encoder ${layers}L · ${heads}H`, 680, 160, { heads, num_layers: layers, dim_feedforward: dModel * 4, dropout }), node("pool", "sequence_pool", "Patch average", 915, 160, { mode: "mean" }), node("head", "linear", `Output ${classes}`, 1130, 160, { out_features: classes }), node("output", "output", "Result", 1320, 160)]) });
    return [base, vitPreset("fast", task === "image.classification" ? 14 : 16, 32, 2, 1, 0.05), vitPreset("fine", task === "image.classification" ? 7 : 8, 128, 8, 4, 0.2)];
  }

  if (architecture === "autoencoder") {
    if (task === "tabular.reconstruction") return [base,
      { id: "compact", name:"compact", description:"compact", benefit:"compact", tradeoff:"compact", ...sequential([node("input", "input", "Data [F]", 20, 160), node("enc", "linear", "Encoder 16", 220, 160, { out_features: 16 }), node("relu1", "relu", "ReLU", 405, 160), node("latent", "linear", "Latent 4", 570, 160, { out_features: 4 }), node("relu2", "relu", "ReLU", 750, 160), node("head", "linear", "Reconstruction 12", 920, 160, { out_features: 12 }), node("output", "output", "Reconstruction", 1130, 160)]) },
      { id: "denoising", name:"denoising", description:"denoising", benefit:"denoising", tradeoff:"denoising", ...sequential([node("input", "input", "Data [F]", 20, 160), node("enc", "linear", "Encoder 64", 220, 160, { out_features: 64 }), node("relu1", "relu", "ReLU", 405, 160), node("drop", "dropout", "Dropout 0.20", 570, 160, { p: 0.2 }), node("latent", "linear", "Latent 16", 755, 160, { out_features: 16 }), node("relu2", "relu", "ReLU", 940, 160), node("dec", "linear", "Decoder 64", 1100, 160, { out_features: 64 }), node("relu3", "relu", "ReLU", 1285, 160), node("head", "linear", "Reconstruction 12", 1445, 160, { out_features: 12 }), node("output", "output", "Reconstruction", 1655, 160)]) }
    ];
    const imageAe = (id:string, channels:number, dropout:boolean): BuiltinPreset => ({ id, name:id, description:id, benefit:id, tradeoff:id, ...sequential([node("input", "input", "Image", 20, 160), node("enc", "conv2d", `Encoder ${channels}`, 210, 160, { out_channels: channels, kernel_size: 4, stride: 2, padding: 1 }), node("relu1", "relu", "ReLU", 400, 160), ...(dropout ? [node("drop", "dropout2d", "Dropout 2D", 565, 160, { p: 0.2 })] : []), node("latent", "conv2d", `Latent ${channels * 2}`, dropout ? 750 : 565, 160, { out_channels: channels * 2, kernel_size: 4, stride: 2, padding: 1 }), node("relu2", "relu", "ReLU", dropout ? 945 : 755, 160), node("dec1", "convtranspose2d", `Decoder ${channels}`, dropout ? 1110 : 920, 160, { out_channels: channels, kernel_size: 4, stride: 2, padding: 1 }), node("relu3", "relu", "ReLU", dropout ? 1310 : 1120, 160), node("head", "convtranspose2d", "Reconstruction RGB", dropout ? 1480 : 1290, 160, { out_channels: 3, kernel_size: 4, stride: 2, padding: 1 }), node("sigmoid", "sigmoid", "Sigmoid", dropout ? 1700 : 1510, 160), node("output", "output", "Reconstruction", dropout ? 1870 : 1680, 160)]) });
    return [base, imageAe("compact", 8, false), imageAe("denoising", 32, true)];
  }

  return [base,
    { id: "light", name: "U-Net · Light", description: "Compact encoder-decoder to quickly try masks and segmentation formats.", benefit: "Consumes less memory", tradeoff: "Loses detail in complex scenes", ...unetPreset(task, 8) },
    { id: "regularized_unet", name: "U-Net · Normalized", description: "Adds BatchNorm, activation and Dropout at the bottleneck for more stable segmentation.", benefit: "Better generalization on varied datasets", tradeoff: "Trains slower", ...unetPreset(task, 24, true) }
  ];
}
