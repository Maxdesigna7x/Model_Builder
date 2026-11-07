import type { DatasetSummary, ModelEdge, ModelNode, TaskId } from "./types";

export type CnnFeaturePort = { nodeId: string; label: string; handle?: string; channels: number; height: number; width: number };

export function featurePorts(nodes: ModelNode[]): CnnFeaturePort[] {
  const result: CnnFeaturePort[] = [];
  for (const node of nodes) {
    if (node.data.blockType === "input" || node.data.blockType === "output" || !node.data.shape) continue;
    const expression = /(?:([\w.-]+)\s*)?\[B,\s*(\d+),\s*(\d+),\s*(\d+)\]/g;
    for (const match of node.data.shape.matchAll(expression)) {
      result.push({ nodeId: node.id, label: `${node.data.label}${match[1] ? ` · ${match[1]}` : ""}`, handle: match[1], channels: Number(match[2]), height: Number(match[3]), width: Number(match[4]) });
    }
  }
  return result;
}

/** Copy the ancestors of a trained feature node and attach a fresh task head.
 * IDs in the copied backbone stay stable so PyTorch can load only matching weights.
 */
export function extractCnnBackbone(args: {
  nodes: ModelNode[]; edges: ModelEdge[]; feature: CnnFeaturePort; task: TaskId;
  dataset: DatasetSummary | null; inputShape: number[]; sourceInputShape?: number[];
}): { nodes: ModelNode[]; edges: ModelEdge[] } {
  const { nodes, edges, feature, task, dataset } = args;
  const byId = new Map(nodes.map(node => [node.id, node]));
  if (!byId.has(feature.nodeId)) throw new Error("El nodo elegido ya no existe en el grafo de origen.");
  const ancestors = new Set<string>([feature.nodeId]);
  const stack = [feature.nodeId];
  while (stack.length) {
    const target = stack.pop()!;
    for (const edge of edges) if (edge.target === target && !ancestors.has(edge.source)) { ancestors.add(edge.source); stack.push(edge.source); }
  }
  const copiedNodes = nodes.filter(node => ancestors.has(node.id) && node.data.blockType !== "output").map(node => ({
    ...node,
    data: { ...node.data, properties: node.data.blockType === "input" ? { ...node.data.properties } : { ...node.data.properties, frozen: true, parameter_group: "backbone" } }
  }));
  const copiedEdges = edges.filter(edge => ancestors.has(edge.source) && ancestors.has(edge.target)).map(edge => ({ ...edge }));
  const source = byId.get(feature.nodeId)!;
  const x = source.position.x + 240, y = source.position.y;
  const id = (name: string) => `reuse-${name}-${crypto.randomUUID().slice(0, 8)}`;
  const makeNode = (type: string, label: string, px: number, py: number, properties: Record<string, string | number | boolean> = {}): ModelNode => ({ id: id(type), type: "modelNode", position: { x: px, y: py }, deletable: true, data: { blockType: type, label, category: "CNN head", properties } });
  const join = (from: ModelNode, to: ModelNode, handle?: string): ModelEdge => ({ id: `${from.id}-${to.id}`, source: from.id, ...(handle ? { sourceHandle: handle } : {}), target: to.id, type: "smoothstep" });
  const connectFromFeature = (target: ModelNode) => ({ id: `${feature.nodeId}-${target.id}`, source: feature.nodeId, ...(feature.handle ? { sourceHandle: feature.handle } : {}), target: target.id, type: "smoothstep" as const });
  const addOutput = (name: string, label: string, from: ModelNode, px: number, py: number) => {
    const output: ModelNode = { id: id("output"), type: "modelNode", position: { x: px, y: py }, deletable: true, data: { blockType: "output", label, category: "Output", properties: { output_name: name } } };
    copiedNodes.push(output); copiedEdges.push(join(from, output));
  };
  const c = feature.channels, h = feature.height, w = feature.width;
  const outputShape = dataset?.outputShape || [];
  const outputChannels = Math.max(1, Number(outputShape[0]) || (task === "image.localization" ? 4 : task === "image.detection" ? 3 : 1));

  if (task === "image.detection") {
    const boxes = makeNode("conv2d", "Nueva cabeza · boxes (4)", x, y - 85, { in_channels: c, out_channels: 4, kernel_size: 1, padding: 0 });
    const classes = makeNode("conv2d", `Nueva cabeza · classes (${outputChannels})`, x, y + 85, { in_channels: c, out_channels: outputChannels, kernel_size: 1, padding: 0 });
    copiedNodes.push(boxes, classes); copiedEdges.push(connectFromFeature(boxes), connectFromFeature(classes));
    addOutput("boxes", "boxes", boxes, x + 210, y - 85); addOutput("classes", "classes", classes, x + 210, y + 85);
    return { nodes: copiedNodes, edges: copiedEdges };
  }

  if (["image.classification", "image.regression", "image.localization"].includes(task)) {
    const units = Math.max(1, outputShape.length ? outputShape.reduce((total, value) => total * Number(value), 1) : (task === "image.localization" ? 4 : 1));
    const pool = makeNode("adaptiveavgpool2d", "Global Average Pool", x, y, { output_size: 1 });
    const flatten = makeNode("flatten", "Flatten", x + 230, y, {});
    const head = makeNode("linear", `Nueva cabeza · ${units}`, x + 460, y, { in_features: c, out_features: units });
    const output = makeNode("output", "Resultado", x + 690, y, { output_name: "result" });
    copiedNodes.push(pool, flatten, head, output); copiedEdges.push(connectFromFeature(pool), join(pool, flatten), join(flatten, head), join(head, output));
    return { nodes: copiedNodes, edges: copiedEdges };
  }

  const targetHeight = Number(args.inputShape[1] || h), targetWidth = Number(args.inputShape[2] || w);
  const sourceInputHeight=Number(args.sourceInputShape?.[1]||targetHeight),sourceInputWidth=Number(args.sourceInputShape?.[2]||targetWidth);
  const targetFeatureHeight=Math.max(1,Math.round(h*targetHeight/sourceInputHeight)),targetFeatureWidth=Math.max(1,Math.round(w*targetWidth/sourceInputWidth));
  const scaleH = targetHeight / targetFeatureHeight, scaleW = targetWidth / targetFeatureWidth;
  let previous: ModelNode = source;
  let previousX = x;
  if (targetFeatureHeight !== targetHeight || targetFeatureWidth !== targetWidth) {
    const integerScale=Math.round(scaleH);
    if (Math.abs(scaleH - scaleW) > 1e-8 || Math.abs(scaleH-integerScale)>1e-8 || Math.abs(targetFeatureHeight * integerScale - targetHeight) > 1e-6 || Math.abs(targetFeatureWidth * integerScale - targetWidth) > 1e-6) {
      throw new Error("CNN_REUSE_DENSE_SCALE_INVALID");
    }
    const resize = makeNode("resize2d", `Subir resolución ×${integerScale}`, x, y, { scale_factor: integerScale });
    copiedNodes.push(resize); copiedEdges.push(connectFromFeature(resize)); previous = resize; previousX += 230;
  }
  const labels = task.startsWith("image.segmentation.") ? `Nueva cabeza · ${outputChannels} clases` : `Nueva cabeza · ${outputChannels} canales`;
  const head = makeNode("conv2d", labels, previousX + 230, y, { in_channels: c, out_channels: outputChannels, kernel_size: 1, padding: 0 });
  const output = makeNode("output", task.startsWith("image.segmentation.") ? "Máscara" : "Imagen transformada", previousX + 460, y, { output_name: "result" });
  copiedNodes.push(head, output); copiedEdges.push(previous === source ? connectFromFeature(head) : join(previous, head), join(head, output));
  return { nodes: copiedNodes, edges: copiedEdges };
}
