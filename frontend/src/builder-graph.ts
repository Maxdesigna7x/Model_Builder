import type { ModelEdge, ModelNode } from "./types";

export type GraphConnection = {
  id?: string;
  source: string | null;
  target: string | null;
  sourceHandle?: string | null;
  targetHandle?: string | null;
};

const MERGE_TYPES = new Set(["add", "concat"]);
const inputIndex = (handle: string | null | undefined) => {
  const match = handle?.match(/^input_(\d+)$/);
  return match ? Number(match[1]) : null;
};

/** Normalize legacy merge edges before selecting the port that a new edge replaces. */
export function candidateEdgesForConnection(
  connection: GraphConnection,
  nodes: ModelNode[],
  edges: ModelEdge[],
  removedId?: string,
): ModelEdge[] {
  const target = nodes.find(node => node.id === connection.target);
  const isMerge = Boolean(target && MERGE_TYPES.has(target.data.blockType));
  let normalized = edges;

  if (isMerge && target) {
    const occupied = new Set<number>();
    for (const edge of edges) {
      if (edge.target !== target.id) continue;
      const index = inputIndex(edge.targetHandle);
      if (index !== null) occupied.add(index);
    }
    let nextFree = 0;
    normalized = edges.map(edge => {
      if (edge.target !== target.id || edge.targetHandle != null) return edge;
      while (occupied.has(nextFree)) nextFree++;
      const targetHandle = `input_${nextFree}`;
      occupied.add(nextFree++);
      return { ...edge, targetHandle };
    });
  }

  const destination = (handle: string | null | undefined) => isMerge ? handle : "target";
  return normalized.filter(edge => edge.id !== removedId && !(
    edge.target === connection.target &&
    destination(edge.targetHandle) === destination(connection.targetHandle)
  ));
}

export function isValidGraphConnection(
  connection: GraphConnection,
  nodes: ModelNode[],
  edges: ModelEdge[],
  removedId?: string,
): boolean {
  if (!connection.source || !connection.target || connection.source === connection.target) return false;
  const source = nodes.find(node => node.id === connection.source);
  const target = nodes.find(node => node.id === connection.target);
  if (!source || !target || source.data.blockType === "output" || target.data.blockType === "input") return false;

  const isMerge = MERGE_TYPES.has(target.data.blockType);
  if (isMerge) {
    const index = inputIndex(connection.targetHandle);
    if (index === null) return false;
    const incoming = edges.filter(edge => edge.target === target.id);
    const maxExisting = Math.max(-1, ...incoming.map(edge => inputIndex(edge.targetHandle) ?? -1));
    const portCount = Math.max(2, incoming.length + 1, maxExisting + 2);
    if (index >= portCount) return false;
  } else if (connection.targetHandle && connection.targetHandle !== "target") {
    return false;
  }

  const sourceType = source.data.blockType;
  const sourcePorts = sourceType === "multi_scale_cnn"
    ? Array.from({ length: Math.max(2, Math.min(4, Number(source.data.properties.stages || 3))) }, (_, index) => `p${index + 2}`)
    : sourceType === "hf_vision_backbone" && source.data.properties.expose_feature_maps
      ? Array.from({ length: Math.max(1, Math.min(4, Number(source.data.properties.feature_count || 3))) }, (_, index) => `map_${index}`)
      : [];
  if (sourcePorts.length ? !connection.sourceHandle || !sourcePorts.includes(connection.sourceHandle) : Boolean(connection.sourceHandle && connection.sourceHandle !== "source")) return false;

  const remaining = candidateEdgesForConnection(connection, nodes, edges, removedId);
  const hasPath = (from: string, to: string) => {
    const pending = [from];
    const seen = new Set<string>();
    while (pending.length) {
      const current = pending.pop()!;
      if (current === to) return true;
      if (seen.has(current)) continue;
      seen.add(current);
      pending.push(...remaining.filter(edge => edge.source === current).map(edge => edge.target));
    }
    return false;
  };
  return !hasPath(target.id, source.id);
}

export function duplicateGraphSelection(
  nodes: ModelNode[],
  edges: ModelEdge[],
  makeId: (type: string) => string,
): { nodes: ModelNode[]; edges: ModelEdge[]; selectedId: string | null } {
  const selectedGroups = new Set(nodes.filter(node => node.selected && node.data.groupId).map(node => node.data.groupId));
  const originals = nodes.filter(node =>
    (node.selected || (Boolean(node.data.groupId) && selectedGroups.has(node.data.groupId))) && node.data.blockType !== "input"
  );
  if (!originals.length) return { nodes, edges, selectedId: null };

  const ids = new Map(originals.map(node => [node.id, makeId(node.data.blockType)]));
  const groupIds = new Map(originals.filter(node => node.data.groupId).map(node => [String(node.data.groupId), makeId("group")]));
  const outputNames = new Set(nodes.filter(node => node.data.blockType === "output").map(node => String(node.data.properties.output_name || node.data.label)));
  const copies = originals.map(node => {
    const {
      shape, error, onDelete, onToggleExpanded, canDelete, inputHandles,
      ...data
    } = node.data;
    void shape; void error; void onDelete; void onToggleExpanded; void canDelete; void inputHandles;
    const properties = { ...data.properties };
    let label = data.label;
    if (data.blockType === "output") {
      const base = String(properties.output_name || label);
      let suffix = 2;
      while (outputNames.has(`${base} ${suffix}`)) suffix++;
      label = `${base} ${suffix}`;
      properties.output_name = label;
      outputNames.add(label);
    }
    if (data.groupId) data.groupId = groupIds.get(String(data.groupId));
    return {
      ...node,
      id: ids.get(node.id)!,
      selected: true,
      position: { x: node.position.x + 35, y: node.position.y + 35 },
      data: { ...data, label, properties },
    };
  });
  const copiedEdges: ModelEdge[] = [];
  const usedMergeInputs = new Map<string, Set<number>>();
  for (const edge of edges) {
    const copySource = ids.get(edge.source);
    const copyTarget = ids.get(edge.target);
    if (copySource && copyTarget) {
      copiedEdges.push({ ...edge, id: makeId("edge"), source: copySource, target: copyTarget, selected: false, data: undefined });
      continue;
    }
    // Keep external sources feeding the duplicate. A copied branch may also
    // feed an external merge, but never a single-input node already in use.
    if (copyTarget && selectedGroups.has(nodes.find(node => node.id === edge.target)?.data.groupId)) {
      copiedEdges.push({ ...edge, id: makeId("edge"), target: copyTarget, selected: false, data: undefined });
      continue;
    }
    if (!copySource || !selectedGroups.has(nodes.find(node => node.id === edge.source)?.data.groupId)) continue;
    const target = nodes.find(node => node.id === edge.target);
    if (!target || !MERGE_TYPES.has(target.data.blockType)) continue;
    let used = usedMergeInputs.get(target.id);
    if (!used) {
      const incoming = edges.filter(item => item.target === target.id);
      used = new Set(incoming.map(item => inputIndex(item.targetHandle)).filter((index): index is number => index !== null));
      let legacyIndex = 0;
      for (const item of incoming) if (item.targetHandle == null) {
        while (used.has(legacyIndex)) legacyIndex++;
        used.add(legacyIndex++);
      }
      usedMergeInputs.set(target.id, used);
    }
    let next = 0;
    while (used.has(next)) next++;
    used.add(next);
    copiedEdges.push({ ...edge, id: makeId("edge"), source: copySource, targetHandle: `input_${next}`, selected: false, data: undefined });
  }
  return {
    nodes: [...nodes.map(node => ({ ...node, selected: false })), ...copies],
    edges: [...edges, ...copiedEdges],
    selectedId: copies[0].id,
  };
}

export type GraphLayoutDirection = "horizontal" | "vertical";

type LayoutRect = { left: number; top: number; right: number; bottom: number };

/** Arrange connected acyclic components by longest-path topological rank. */
export function layoutConnectedGraph(
  nodes: ModelNode[],
  edges: ModelEdge[],
  direction: GraphLayoutDirection,
): ModelNode[] {
  if (nodes.length < 2) return nodes;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const undirected = new Map(nodes.map(node => [node.id, new Set<string>()]));
  const outgoing = new Map(nodes.map(node => [node.id, [] as string[]]));
  const indegree = new Map(nodes.map(node => [node.id, 0]));
  for (const edge of edges) {
    if (!byId.has(edge.source) || !byId.has(edge.target)) continue;
    undirected.get(edge.source)!.add(edge.target);
    undirected.get(edge.target)!.add(edge.source);
    outgoing.get(edge.source)!.push(edge.target);
    indegree.set(edge.target, indegree.get(edge.target)! + 1);
  }

  const flowCoord = (node: ModelNode) => direction === "horizontal" ? node.position.x : node.position.y;
  const crossCoord = (node: ModelNode) => direction === "horizontal" ? node.position.y : node.position.x;
  const flowSize = (node: ModelNode) => direction === "horizontal" ? (node.measured?.width || node.width || 190) : (node.measured?.height || node.height || 110);
  const crossSize = (node: ModelNode) => direction === "horizontal" ? (node.measured?.height || node.height || 110) : (node.measured?.width || node.width || 190);
  const originalOrder = (a: string, b: string) => crossCoord(byId.get(a)!) - crossCoord(byId.get(b)!) || a.localeCompare(b);
  const components: string[][] = [];
  const visited = new Set<string>();
  for (const node of nodes) {
    if (visited.has(node.id) || !undirected.get(node.id)!.size) continue;
    const component: string[] = [];
    const pending = [node.id];
    visited.add(node.id);
    while (pending.length) {
      const id = pending.pop()!;
      component.push(id);
      for (const next of undirected.get(id)!) if (!visited.has(next)) { visited.add(next); pending.push(next); }
    }
    components.push(component);
  }

  const placements = new Map<string, { x: number; y: number }>();
  const regions: LayoutRect[] = [];
  const frozenRects: LayoutRect[] = nodes.filter(node => !undirected.get(node.id)!.size).map(node => ({
    left: node.position.x, top: node.position.y,
    right: node.position.x + (node.measured?.width || node.width || 190),
    bottom: node.position.y + (node.measured?.height || node.height || 110),
  }));
  const intersects = (a: LayoutRect, b: LayoutRect) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;

  components.sort((a, b) => {
    const aPrimary = Math.min(...a.map(id => flowCoord(byId.get(id)!)));
    const bPrimary = Math.min(...b.map(id => flowCoord(byId.get(id)!)));
    return aPrimary - bPrimary || Math.min(...a.map(id => crossCoord(byId.get(id)!))) - Math.min(...b.map(id => crossCoord(byId.get(id)!)));
  });

  for (const component of components) {
    const members = new Set(component);
    const degrees = new Map(component.map(id => [id, indegree.get(id)!]));
    const ranks = new Map(component.map(id => [id, 0]));
    const ready = component.filter(id => degrees.get(id) === 0).sort(originalOrder);
    const topo: string[] = [];
    while (ready.length) {
      const id = ready.shift()!;
      topo.push(id);
      for (const next of outgoing.get(id)!) {
        if (!members.has(next)) continue;
        ranks.set(next, Math.max(ranks.get(next)!, ranks.get(id)! + 1));
        degrees.set(next, degrees.get(next)! - 1);
        if (degrees.get(next) === 0) { ready.push(next); ready.sort(originalOrder); }
      }
    }
    // Cyclic components have no topological layout; leave them where they are.
    if (topo.length !== component.length) {
      regions.push({
        left: Math.min(...component.map(id => byId.get(id)!.position.x)),
        top: Math.min(...component.map(id => byId.get(id)!.position.y)),
        right: Math.max(...component.map(id => byId.get(id)!.position.x + (byId.get(id)!.measured?.width || byId.get(id)!.width || 190))),
        bottom: Math.max(...component.map(id => byId.get(id)!.position.y + (byId.get(id)!.measured?.height || byId.get(id)!.height || 110))),
      });
      continue;
    }

    const layers = new Map<number, string[]>();
    for (const id of topo) {
      const rank = ranks.get(id)!;
      layers.set(rank, [...(layers.get(rank) || []), id]);
    }
    const maxRank = Math.max(...layers.keys());
    for (const [rank, ids] of layers) layers.set(rank, ids.sort(originalOrder));

    const orderMap = () => new Map([...layers].flatMap(([, ids]) => ids.map((id, index) => [id, index] as const)));
    const reorder = (rank: number, neighbors: "parents" | "children", currentOrder: Map<string, number>) => {
      const ids = layers.get(rank) || [];
      const grouped = new Map<string, string[]>();
      for (const id of ids) {
        const groupId = byId.get(id)?.data.groupId;
        const key = typeof groupId === "string" ? `group:${groupId}` : `node:${id}`;
        grouped.set(key, [...(grouped.get(key) || []), id]);
      }
      const score = (id: string) => {
        const relevant = neighbors === "parents"
          ? edges.filter(edge => edge.target === id && members.has(edge.source)).map(edge => currentOrder.get(edge.source))
          : edges.filter(edge => edge.source === id && members.has(edge.target)).map(edge => currentOrder.get(edge.target));
        const valid = relevant.filter((value): value is number => value !== undefined);
        return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : currentOrder.get(id) ?? 0;
      };
      const blocks = [...grouped.values()].map(block => ({
        ids: block,
        score: block.reduce((sum, id) => sum + score(id), 0) / block.length,
        original: block.reduce((sum, id) => sum + crossCoord(byId.get(id)!), 0) / block.length,
      })).sort((a, b) => a.score - b.score || a.original - b.original || a.ids[0].localeCompare(b.ids[0]));
      layers.set(rank, blocks.flatMap(block => block.ids.sort((a, b) => score(a) - score(b) || originalOrder(a, b))));
    };
    for (let pass = 0; pass < 4; pass++) {
      for (let rank = 1; rank <= maxRank; rank++) reorder(rank, "parents", orderMap());
      for (let rank = maxRank - 1; rank >= 0; rank--) reorder(rank, "children", orderMap());
    }

    const orderedLayers = Array.from({ length: maxRank + 1 }, (_, rank) => layers.get(rank) || []);
    const layerCrossSizes = orderedLayers.map(ids => ids.reduce((sum, id) => sum + crossSize(byId.get(id)!), 0) + Math.max(0, ids.length - 1) * 36);
    const maxCrossSize = Math.max(0, ...layerCrossSizes);
    const minFlow = Math.min(...component.map(id => flowCoord(byId.get(id)!)));
    const minCross = Math.min(...component.map(id => crossCoord(byId.get(id)!)));
    let flowCursor = minFlow;
    for (const ids of orderedLayers) {
      const layerFlowSize = Math.max(...ids.map(id => flowSize(byId.get(id)!)));
      const layerCrossSize = ids.reduce((sum, id) => sum + crossSize(byId.get(id)!), 0) + Math.max(0, ids.length - 1) * 36;
      let crossCursor = minCross + (maxCrossSize - layerCrossSize) / 2;
      for (const id of ids) {
        const node = byId.get(id)!;
        placements.set(id, direction === "horizontal"
          ? { x: flowCursor, y: crossCursor }
          : { x: crossCursor, y: flowCursor });
        crossCursor += crossSize(node) + 36;
      }
      flowCursor += layerFlowSize + 56;
    }

    const placed = component.map(id => ({ id, pos: placements.get(id)!, node: byId.get(id)! }));
    const bounds = (): LayoutRect => ({
      left: Math.min(...placed.map(({ pos }) => pos.x)), top: Math.min(...placed.map(({ pos }) => pos.y)),
      right: Math.max(...placed.map(({ pos, node }) => pos.x + (node.measured?.width || node.width || 190))),
      bottom: Math.max(...placed.map(({ pos, node }) => pos.y + (node.measured?.height || node.height || 110))),
    });
    let region = bounds();
    // Isolated blocks stay put; move only connected components around them.
    for (let attempt = 0; attempt < nodes.length + 2 && [...frozenRects, ...regions].some(rect => intersects(region, rect)); attempt++) {
      const shift = 260;
      for (const id of component) {
        const pos = placements.get(id)!;
        placements.set(id, direction === "horizontal" ? { x: pos.x, y: pos.y + shift } : { x: pos.x + shift, y: pos.y });
      }
      region = bounds();
    }
    regions.push(region);
  }

  return nodes.map(node => {
    const position = placements.get(node.id);
    return position ? { ...node, position } : node;
  });
}

export function createCnnRecipe(
  kind: 0 | 1 | 2 | 3,
  center: { x: number; y: number },
  groupName: string,
  makeId: (type: string) => string,
  labelFor: (type: string) => string,
): { nodes: ModelNode[]; edges: ModelEdge[] } {
  const groupId = makeId("group");
  const nodes: ModelNode[] = [];
  const edges: ModelEdge[] = [];
  const specs: Array<{ type: string; x: number; y: number; properties?: Record<string, string | number | boolean> }> = [];
  const joins: Array<[number, number, string?]> = [];
  const conv = (x: number, y: number) => specs.push({ type: "conv2d", x, y, properties: { out_channels: 32, kernel_size: 3, stride: 1, padding: 1, bias: false } });
  const block = (type: string, x: number, y: number, properties: Record<string, string | number | boolean> = {}) => specs.push({ type, x, y, properties });

  if (kind === 0) {
    conv(0, 0);
    block("relu", 230, -120);
    conv(460, -120);
    block("add", 690, 0, { axis: 1 });
    block("relu", 920, 0);
    joins.push([0, 1], [1, 2], [2, 3, "input_1"], [0, 3, "input_0"], [3, 4]);
  } else {
    let previous = -1;
    for (let index = 0; index < kind; index++) {
      const convIndex = specs.length;
      conv(index * 460, 0);
      const reluIndex = specs.length;
      block("relu", index * 460 + 230, 0);
      if (previous >= 0) joins.push([previous, convIndex]);
      joins.push([convIndex, reluIndex]);
      previous = reluIndex;
    }
    const poolIndex = specs.length;
    block("maxpool2d", kind * 460, 0, { kernel_size: 2, stride: 2 });
    joins.push([previous, poolIndex]);
  }

  const positions = specs.map(spec => ({ x: spec.x, y: spec.y }));
  const nodeWidth = 190, nodeHeight = 110;
  const minX = Math.min(...positions.map(position => position.x));
  const maxX = Math.max(...positions.map(position => position.x + nodeWidth));
  const minY = Math.min(...positions.map(position => position.y));
  const maxY = Math.max(...positions.map(position => position.y + nodeHeight));
  const offsetX = center.x - (minX + maxX) / 2;
  const offsetY = center.y - (minY + maxY) / 2;
  for (const spec of specs) {
    const id = makeId(spec.type);
    nodes.push({
      id, type: "modelNode", selected: true,
      position: { x: spec.x + offsetX, y: spec.y + offsetY },
      data: {
        blockType: spec.type, label: labelFor(spec.type), category: spec.type === "conv2d" ? "Convolution" : spec.type.includes("pool") ? "Pooling" : spec.type === "add" ? "Merge" : spec.type === "identity" ? "Shape" : "Activation",
        properties: spec.properties || {}, groupId, groupName,
      },
    });
  }
  for (const [source, target, targetHandle] of joins) {
    edges.push({ id: makeId("edge"), source: nodes[source].id, target: nodes[target].id, targetHandle, type: "smoothstep" });
  }
  return { nodes, edges };
}
