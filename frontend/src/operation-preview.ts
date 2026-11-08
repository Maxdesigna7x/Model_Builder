import type { BlockDef } from "./types";
import { parseTensorShape } from "./tensor-shape";

const numeric = (value: string | undefined) => value !== undefined && /^\d+$/.test(value) ? Number(value) : null;
const propertyNumber = (properties: BlockDef["defaults"] | Record<string, string | number | boolean>, key: string, fallback: number) => {
  const value = Number(properties[key]);
  return Number.isFinite(value) ? value : fallback;
};

function pair(value: string | number | boolean | undefined, fallback: number): [number, number] {
  if (typeof value === "string") {
    const values = value.split(",").map(item => Number(item.trim()));
    if (values.length === 2 && values.every(Number.isFinite)) return [values[0], values[1]];
  }
  const number = Number(value);
  const resolved = Number.isFinite(number) ? number : fallback;
  return [resolved, resolved];
}

const format = (dims: Array<string | number>) => `[${dims.join(", ")}]`;

/** Exact local calculation for common operations with a concrete, validated input shape. */
export function previewOperationOutput(type: string, properties: BlockDef["defaults"] | Record<string, string | number | boolean>, inputShape?: string): string | null {
  const [part] = parseTensorShape(inputShape);
  if (!part || part.name || part.dimensions[0] !== "B") return null;
  const values = part.dimensions.slice(1).map(numeric);
  if (values.some(value => value === null)) return null;
  const dims = values as number[];
  const unchanged = new Set(["relu", "leaky_relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "dropout", "dropout2d", "batchnorm1d", "batchnorm2d", "layernorm", "identity"]);
  if (unchanged.has(type)) return format(["B", ...dims]);
  if (type === "permute") {
    const order = String(properties.dims || "2,1").split(",").map(value => Number(value.trim()) - 1);
    if (order.length === dims.length && order.every(index => Number.isInteger(index) && index >= 0 && index < dims.length)) return format(["B", ...order.map(index => dims[index])]);
    return null;
  }
  if (type === "flatten" && dims.length >= 1) return format(["B", dims.reduce((total, value) => total * value, 1)]);
  if (type === "linear" && dims.length >= 1) return format(["B", ...dims.slice(0, -1), propertyNumber(properties, "out_features", 64)]);

  if (["conv2d", "maxpool2d", "avgpool2d"].includes(type) && dims.length === 3) {
    const [channels, height, width] = dims;
    const kernel = pair(properties.kernel_size, type === "conv2d" ? 3 : 2);
    const stride = pair(properties.stride, kernel[0]);
    const padding = pair(properties.padding, 0);
    const dilation = type === "conv2d" ? pair(properties.dilation, 1) : [1, 1];
    const size = (input: number, k: number, s: number, p: number, d: number) => Math.floor((input + 2 * p - d * (k - 1) - 1) / s + 1);
    const outputChannels = type === "conv2d" ? propertyNumber(properties, "out_channels", 32) : channels;
    return format(["B", outputChannels, size(height, kernel[0], stride[0], padding[0], dilation[0]), size(width, kernel[1], stride[1], padding[1], dilation[1])]);
  }
  if (type === "adaptiveavgpool2d" && dims.length === 3) {
    const [height, width] = pair(properties.output_size, 1);
    return format(["B", dims[0], height, width]);
  }
  if (type === "resize2d" && dims.length === 3) {
    const factor = propertyNumber(properties, "scale_factor", 2);
    return format(["B", dims[0], Math.floor(dims[1] * factor), Math.floor(dims[2] * factor)]);
  }
  if (type === "convtranspose2d" && dims.length === 3) {
    const kernel = pair(properties.kernel_size, 2);
    const stride = pair(properties.stride, kernel[0]);
    const padding = pair(properties.padding, 0);
    const outputPadding = pair(properties.output_padding, 0);
    const size = (input: number, k: number, s: number, p: number, op: number) => (input - 1) * s - 2 * p + k + op;
    return format(["B", propertyNumber(properties, "out_channels", 16), size(dims[1], kernel[0], stride[0], padding[0], outputPadding[0]), size(dims[2], kernel[1], stride[1], padding[1], outputPadding[1])]);
  }
  return null;
}
