import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const source = await readFile(new URL('../frontend/src/builder-graph.ts', import.meta.url), 'utf8');
const { code } = await transform(source, { loader: 'ts', format: 'esm' });
const graph = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const { candidateEdgesForConnection, createCnnRecipe, duplicateGraphSelection, isValidGraphConnection } = graph;
const node = (id, blockType, selected = false, extra = {}) => ({
  id, type: 'modelNode', selected, position: { x: 10, y: 20 },
  data: { blockType, label: id, category: 'test', properties: {}, ...extra },
});
let id = 0;
const makeId = type => `${type}-${++id}`;

{
  const nodes = [node('old', 'relu'), node('replacement', 'relu'), node('downstream', 'linear')];
  const edges = [
    { id: 'old-input', source: 'old', target: 'downstream', type: 'smoothstep' },
    { id: 'fanout', source: 'old', target: 'replacement', type: 'smoothstep' },
  ];
  const connection = { source: 'replacement', target: 'downstream' };
  assert.equal(isValidGraphConnection(connection, nodes, edges), true);
  const remaining = candidateEdgesForConnection(connection, nodes, edges);
  assert.deepEqual(remaining.map(edge => edge.id), ['fanout'], 'replace only the occupied input, preserving fan-out');
}

{
  const nodes = [node('candidate', 'relu'), node('merge', 'add'), node('skip', 'multi_scale_cnn', false, { properties: { stages: 3 } })];
  const edges = [
    { id: 'legacy', source: 'skip', target: 'merge', sourceHandle: 'p2', targetHandle: null, type: 'smoothstep' },
    { id: 'explicit-1', source: 'candidate', target: 'merge', sourceHandle: 'map_1', targetHandle: 'input_1', type: 'smoothstep' },
  ];
  const connection = { source: 'skip', target: 'merge', sourceHandle: 'p3', targetHandle: 'input_1' };
  const remaining = candidateEdgesForConnection(connection, nodes, edges);
  assert.deepEqual(remaining.map(edge => [edge.id, edge.targetHandle]), [['legacy', 'input_0']]);
  assert.equal(remaining[0].sourceHandle, 'p2', 'normalization preserves named output ports');
  assert.equal(isValidGraphConnection(connection, nodes, edges), true);
}

{
  const nodes = [node('old', 'identity'), node('source', 'identity'), node('middle', 'identity'), node('target', 'identity')];
  const edges = [
    { id: 'occupied', source: 'old', target: 'target', type: 'smoothstep' },
    { id: 'path-1', source: 'target', target: 'middle', type: 'smoothstep' },
    { id: 'path-2', source: 'middle', target: 'source', type: 'smoothstep' },
  ];
  const connection = { source: 'source', target: 'target' };
  assert.equal(isValidGraphConnection(connection, nodes, edges), false, 'reject a cycle after candidate replacement');
  assert.equal(edges.some(edge => edge.id === 'occupied'), true, 'a rejected replacement leaves the original edge intact');
}

{
  const nodes = [
    node('input', 'input', true),
    node('conv', 'conv2d', true, { groupId: 'group-a', groupName: 'Encoder', shape: '[B, 8, 16, 16]', onDelete() {} }),
    node('relu', 'relu', true, { groupId: 'group-a', groupName: 'Encoder', properties: { alpha: 0.2 } }),
    node('output', 'output', true, { properties: { output_name: 'scores' } }),
    node('external', 'identity'),
  ];
  const edges = [
    { id: 'in-conv', source: 'input', target: 'conv' },
    { id: 'conv-relu', source: 'conv', target: 'relu', targetHandle: 'input_1' },
    { id: 'relu-out', source: 'relu', target: 'output' },
    { id: 'relu-external', source: 'relu', target: 'external' },
  ];
  const duplicate = duplicateGraphSelection(nodes, edges, makeId);
  const copies = duplicate.nodes.slice(nodes.length);
  assert.equal(copies.length, 3, 'input is essential and is never duplicated');
  assert.equal(duplicate.edges.length - edges.length, 3, 'copy internal wires and preserve the external input to the duplicate');
  const copiedEdges = duplicate.edges.slice(edges.length);
  assert.ok(copiedEdges.some(edge => edge.source === 'input' && edge.target === copies[0].id), 'wire the original input into the copied group');
  assert.equal(copiedEdges.some(edge => edge.target === 'external'), false, 'do not replace the external single-input consumer');
  assert.notEqual(copies[0].data.groupId, 'group-a', 'duplicate the visual group with a new group id');
  assert.equal(copies[0].data.shape, undefined);
  assert.equal(copies[0].data.onDelete, undefined);
  assert.notEqual(copies[1].data.properties, nodes[2].data.properties, 'copied properties are independent');
  assert.equal(copies[2].data.properties.output_name, 'scores 2');
  copies[1].data.properties.alpha = 0.7;
  assert.equal(nodes[2].data.properties.alpha, 0.2);
}

{
  for (const count of [1, 2, 3]) {
    const recipe = createCnnRecipe(count, { x: 900, y: 620 }, `CNN ${count}`, makeId, type => type);
    const convs = recipe.nodes.filter(item => item.data.blockType === 'conv2d');
    assert.equal(convs.length, count);
    assert.equal(recipe.nodes.filter(item => item.data.blockType === 'relu').length, count);
    assert.equal(recipe.nodes.filter(item => item.data.blockType === 'maxpool2d').length, 1);
    const bounds = recipe.nodes.map(item => item.position);
    const centerX = (Math.min(...bounds.map(pos => pos.x)) + Math.max(...bounds.map(pos => pos.x + 190))) / 2;
    const centerY = (Math.min(...bounds.map(pos => pos.y)) + Math.max(...bounds.map(pos => pos.y + 110))) / 2;
    assert.ok(Math.abs(centerX - 900) < 0.01 && Math.abs(centerY - 620) < 0.01, 'recipe bounds are centered on the visible canvas');
  }
  const residual = createCnnRecipe(0, { x: 300, y: 500 }, 'Residual', makeId, type => type);
  const convs = residual.nodes.filter(item => item.data.blockType === 'conv2d');
  const add = residual.nodes.find(item => item.data.blockType === 'add');
  assert.equal(convs.length, 2);
  assert.ok(convs.every(item => item.data.properties.stride === 1 && item.data.properties.padding === 1 && item.data.properties.out_channels === 32));
  assert.deepEqual(residual.edges.filter(edge => edge.target === add.id).map(edge => edge.targetHandle).sort(), ['input_0', 'input_1']);
  assert.equal(residual.edges.find(edge => edge.target === add.id && edge.targetHandle === 'input_0').source, convs[0].id, 'the skip carries the first convolution’s 32-channel output');
  assert.equal(residual.edges.find(edge => edge.target === add.id && edge.targetHandle === 'input_1').source, convs[1].id, 'the residual branch has matching channels and spatial size');
  const positions = residual.nodes.map(item => item.position);
  const centerX = (Math.min(...positions.map(pos => pos.x)) + Math.max(...positions.map(pos => pos.x + 190))) / 2;
  const centerY = (Math.min(...positions.map(pos => pos.y)) + Math.max(...positions.map(pos => pos.y + 110))) / 2;
  assert.ok(Math.abs(centerX - 300) < 0.01 && Math.abs(centerY - 500) < 0.01);
}

console.log('Builder graph tests: wire replacement, legacy ports, cycles, duplication, groups, and CNN recipes passed');

// Every architecture exposes graph merge primitives independently of recipes.
const catalogSource = await readFile(new URL('../frontend/src/catalog.ts', import.meta.url), 'utf8');
const compiledCatalog = await transform(catalogSource, { loader: 'ts', format: 'esm' });
const { BLOCKS } = await import(`data:text/javascript;base64,${Buffer.from(compiledCatalog.code).toString('base64')}`);
for (const [architecture, blocks] of Object.entries(BLOCKS)) {
  for (const type of ['add', 'concat']) assert.equal(blocks.filter(block=>block.type===type).length, 1, `${architecture} exposes ${type} exactly once`);
}
for (const type of ['convtranspose2d','resize2d','reshape']) assert.ok(BLOCKS.cnn.some(block=>block.type===type), `CNN exposes ${type}`);
console.log('Standalone merge and CNN decoder library coverage passed');
