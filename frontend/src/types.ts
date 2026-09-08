import type { Edge, Node } from "@xyflow/react";

export type Step = "model" | "data" | "builder" | "training" | "inference";
export type TaskId = "tabular.classification" | "tabular.regression" | "tabular.reconstruction" | "image.classification" | "image.regression" | "image.reconstruction" | "sequence.classification" | "sequence.regression" | "sequence.forecast" | "text.classification" | "text.language_model" | "image.segmentation.binary" | "image.segmentation.multiclass";
export type Architecture = "mlp" | "cnn1d" | "lstm" | "transformer" | "transformer_causal" | "cnn" | "vit" | "unet" | "autoencoder";

export type ArchitectureDef = { name: string; description: string; accent: string; tasks: TaskId[] };
export type TaskDef = {
  name: string;
  nameKey?: string;
  category: string;
  categoryKey?: string;
  modality: string;
  modalityKey?: string;
  input: string;
  inputKey?: string;
  output: string;
  outputKey?: string;
  metric: string;
  metricKey?: string;
  format: string;
  formatKey?: string;
};
export type BlockDef = {
  type: string;
  label: string;
  labelKey?: string;
  category: string;
  categoryKey?: string;
  defaults: Record<string, number | string | boolean>;
};

export type ModelNodeData = { label: string; blockType: string; category: string; shape?: string; properties: Record<string, string | number | boolean>; error?: string; onDelete?: (id: string) => void; [key: string]: unknown };
export type ModelNode = Node<ModelNodeData>;
export type ModelEdge = Edge;

export interface Project {
  id: string;
  name: string;
  description: string;
  path: string;
  createdAt: string;
  updatedAt: string;
  architecture?: Architecture;
  taskId?: TaskId;
  savedState?: Record<string, unknown>;
}

export interface DatasetPreviewItem {
  url?: string;
  targetUrl?: string;
  targetLabel?: string;
  label?: string;
  features?: number[];
  target?: number | string;
}

export interface ProjectModel {
  id: string;
  name: string;
  architecture: Architecture;
  nodes: ModelNode[];
  edges: ModelEdge[];
  graphValid: boolean;
}

export interface DatasetPreview {
  type: "image" | "tabular";
  items?: DatasetPreviewItem[];
  columns?: string[];
}

export interface DatasetOptions {
  channels?: number;
  resolution?: number;
  normalization?: "none" | "standard" | "minmax" | "minmax_sym";
  batchSize?: number;
  vocab_size?: number;
  max_length?: number;
  vocab?: string[];
  max_samples?: number;
  window?: number;
  horizon?: number;
  stride?: number;
}

export interface HuggingFaceDataset {
  id: string;
  name: string;
  repoId: string;
  revision: string;
  tasks: TaskId[];
  sizeBytes: number;
  license: string;
  description: string;
  defaultOptions: DatasetOptions;
  cached: boolean;
  installed: boolean;
}

export type DataNodeCategory = "source" | "inspect" | "curate" | "split" | "transform" | "augment" | "output";
export type DataNodeScope = "all" | "train";

export interface DataPipelineNode {
  id: string;
  type: string;
  label: string;
  description: string;
  category: DataNodeCategory;
  scope: DataNodeScope;
  enabled: boolean;
  locked?: boolean;
  properties: Record<string, string | number | boolean>;
}

export interface DataPipelineDiagnostic {
  level: "info" | "warning" | "error";
  nodeId?: string;
  message: string;
}

export interface DataPipeline {
  schemaVersion: number;
  revision: number;
  nodes: DataPipelineNode[];
  diagnostics?: DataPipelineDiagnostic[];
  fittedState?: Record<string, unknown>;
}

export interface DatasetSummary {
  id: string;
  revision?: number;
  source: "synthetic" | "imported" | "huggingface";
  samples: number;
  inputShape: number[];
  outputShape: number[];
  classes?: string[];
  description: string;
  splits: { train: number; validation: number; test: number };
  preview?: DatasetPreview;
  options?: DatasetOptions;
  sourcePath?: string;
  catalogId?: string;
  repoId?: string;
  repoRevision?: string;
  license?: string;
  downloadSizeBytes?: number;
  cacheStatus?: "project" | "disk" | "downloaded";
  pipeline?: DataPipeline;
  validation?: { errors: number; warnings: number; checks: number };
  analytics?: DatasetAnalytics;
}

export interface NamedSeries { name: string; values: number[]; }
export interface DatasetAnalytics {
  modality: "tabular" | "image" | "sequence" | "text";
  classDistribution?: { labels: string[]; values: number[] };
  targetHistogram?: { labels: string[]; values: number[] };
  intensityHistogram?: { labels: string[]; values: number[] };
  channelStats?: { labels: string[]; mean: number[]; std: number[] };
  featureHistograms?: Array<{ name: string; labels: string[]; values: number[] }>;
  featureSummary?: { labels: string[]; mean: number[]; std: number[] };
  correlation?: { labels: string[]; values: Array<[number, number, number]> };
  meanSeries?: { labels: string[]; series: NamedSeries[] };
  lengthHistogram?: { labels: string[]; values: number[] };
  tokenFrequency?: { labels: string[]; values: number[] };
  maskCoverage?: { labels: string[]; values: number[] };
  pixelDistribution?: { labels: string[]; values: number[] };
}

export interface MetricPoint { epoch: number; trainLoss: number; valLoss: number; metric: number; }
export interface RunResult { runId: string; status: string; metricName: string; history: MetricPoint[]; checkpoint?: string; test?: { loss: number; metric: number }; }
export interface TrainingRun {
  id: string;
  name: string;
  status: "draft" | "running" | "completed" | "error";
  config: Record<string, number | string>;
  history: MetricPoint[];
  result: RunResult | null;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
  modelId?: string;
}

export interface BackendResponse<T = unknown> { ok: boolean; result?: T; error?: { code: string; message: string }; }

export interface GpuInfo {
  cuda: boolean;
  name: string;
  total_vram: number;
  allocated_vram: number;
}
