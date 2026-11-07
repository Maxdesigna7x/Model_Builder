const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");

const filename = path.resolve(__dirname, "../frontend/src/cnn-reuse.ts");
const source = fs.readFileSync(filename, "utf8");
const javascript = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
const loaded = { exports: {} };
new Function("require", "module", "exports", javascript)(require, loaded, loaded.exports);
const { extractCnnBackbone, featurePorts } = loaded.exports;

const node = (id, blockType, shape, properties = {}) => ({ id, type: "modelNode", position: { x: 10, y: 20 }, data: { blockType, label: id, category: "CNN", shape, properties } });
const edge = (source, target, targetHandle) => ({ id: `${source}-${target}`, source, target, ...(targetHandle ? { targetHandle } : {}) });

const residualGraph = {
  nodes: [node("input", "input"), node("left", "conv2d"), node("right", "conv2d"), node("sum", "add", "[B, 8, 32, 32]"), node("old_head", "conv2d"), node("old_output", "output")],
  edges: [edge("input", "left"), edge("input", "right"), edge("left", "sum", "input_0"), edge("right", "sum", "input_1"), edge("sum", "old_head"), edge("old_head", "old_output")]
};
const classification = extractCnnBackbone({ nodes: residualGraph.nodes, edges: residualGraph.edges, feature: { nodeId: "sum", channels: 8, height: 32, width: 32 }, task: "image.classification", dataset: { outputShape: [5] }, inputShape: [3, 64, 64], sourceInputShape: [3, 64, 64] });
assert.deepEqual(classification.nodes.map(item => item.id).slice(0, 4), ["input", "left", "right", "sum"]);
assert(classification.nodes.find(item => item.id === "left").data.properties.frozen);
assert(classification.nodes.find(item => item.id === "right").data.properties.frozen);
assert(!classification.nodes.some(item => item.id === "old_head" || item.id === "old_output"));
assert.equal(classification.edges.filter(item => item.target === "sum").length, 2, "keeps both sides of the residual skip");
assert.equal(classification.nodes.find(item => item.data.blockType === "linear").data.properties.out_features, 5);

const pyramid = node("pyramid", "multi_scale_cnn", "p2 [B, 4, 16, 16] · p3 [B, 8, 8, 8]");
const ports = featurePorts([pyramid]);
assert.deepEqual(ports.map(item => item.handle), ["p2", "p3"]);
const detector = extractCnnBackbone({ nodes: [node("input", "input"), pyramid], edges: [edge("input", "pyramid")], feature: ports[1], task: "image.detection", dataset: { outputShape: [6] }, inputShape: [3, 64, 64], sourceInputShape: [3, 64, 64] });
assert.equal(detector.edges.filter(item => item.source === "pyramid").length, 2);
assert(detector.edges.filter(item => item.source === "pyramid").every(item => item.sourceHandle === "p3"));
assert.deepEqual(detector.nodes.filter(item => item.data.blockType === "output").map(item => item.data.properties.output_name), ["boxes", "classes"]);

const dense = extractCnnBackbone({ nodes: [node("input", "input"), node("encoder", "conv2d", "[B, 8, 16, 16]")], edges: [edge("input", "encoder")], feature: { nodeId: "encoder", channels: 8, height: 16, width: 16 }, task: "image.segmentation.binary", dataset: { outputShape: [1, 64, 64] }, inputShape: [3, 64, 64], sourceInputShape: [3, 64, 64] });
assert.equal(dense.nodes.find(item => item.data.blockType === "resize2d").data.properties.scale_factor, 4);
assert.throws(() => extractCnnBackbone({ nodes: [node("input", "input"), node("encoder", "conv2d", "[B, 8, 16, 8]")], edges: [edge("input", "encoder")], feature: { nodeId: "encoder", channels: 8, height: 16, width: 8 }, task: "image.translation", dataset: { outputShape: [3, 64, 64] }, inputShape: [3, 64, 64], sourceInputShape: [3, 64, 64] }), /CNN_REUSE_DENSE_SCALE_INVALID/);

process.stdout.write("CNN checkpoint backbone extraction: residual skips, multiport routing, task heads and dense-size guard passed.\n");
