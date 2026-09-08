from __future__ import annotations

from collections import defaultdict, deque
import math

import torch
from torch import nn
from torch.nn import functional as F

from .catalog import validate_compatibility
from .errors import BackendError


class LSTMSequence(nn.Module):
    def __init__(self, input_size: int, properties: dict):
        super().__init__()
        hidden = int(properties.get("hidden_size", 64)); layers = int(properties.get("num_layers", 1)); bidirectional = bool(properties.get("bidirectional", False)); dropout = float(properties.get("dropout", 0)) if layers > 1 else 0.0
        self.layer = nn.LSTM(input_size, hidden, layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

    def forward(self, x): return self.layer(x)[0]


class RecurrentSequence(nn.Module):
    def __init__(self, kind: str, input_size: int, properties: dict):
        super().__init__()
        hidden = int(properties.get("hidden_size", 64)); layers = int(properties.get("num_layers", 1)); bidirectional = bool(properties.get("bidirectional", False)); dropout = float(properties.get("dropout", 0)) if layers > 1 else 0.0
        recurrent = {"rnn": nn.RNN, "gru": nn.GRU}[kind]
        kwargs = {"batch_first": True, "bidirectional": bidirectional, "dropout": dropout}
        if kind == "rnn": kwargs["nonlinearity"] = str(properties.get("nonlinearity", "tanh"))
        self.layer = recurrent(input_size, hidden, layers, **kwargs)

    def forward(self, x): return self.layer(x)[0]


class Permute(nn.Module):
    def __init__(self, dims): super().__init__(); self.dims = tuple(int(v) for v in dims)
    def forward(self, x): return x.permute(0, *self.dims)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 2048, learned: bool = False):
        super().__init__(); self.learned = learned
        if learned:
            self.position = nn.Parameter(torch.zeros(1, max_length, d_model)); nn.init.normal_(self.position, std=.02)
        else:
            position = torch.arange(max_length).unsqueeze(1); scale = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)); value = torch.zeros(1, max_length, d_model)
            value[0, :, 0::2] = torch.sin(position * scale); value[0, :, 1::2] = torch.cos(position * scale[:value[0, :, 1::2].shape[1]])
            self.register_buffer("position", value, persistent=False)

    def forward(self, x):
        if x.shape[1] > self.position.shape[1]: raise BackendError("MODEL_SEQUENCE_TOO_LONG")
        return x + self.position[:, :x.shape[1]].to(dtype=x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, properties: dict, causal: bool = False):
        super().__init__(); heads = int(properties.get("heads", 4)); layers = int(properties.get("num_layers", 2)); ff = int(properties.get("dim_feedforward", d_model * 4)); dropout = float(properties.get("dropout", .1))
        if d_model % heads: raise BackendError("MODEL_D_MODEL_NOT_DIVISIBLE_BY_HEADS")
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.causal = causal

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device) if self.causal else None
        return self.encoder(x, mask=mask)


class SequencePool(nn.Module):
    def __init__(self, mode: str): super().__init__(); self.mode = mode
    def forward(self, x): return x[:, 0] if self.mode == "cls" else (x[:, -1] if self.mode == "last" else x.mean(1))


class PatchEmbedding(nn.Module):
    def __init__(self, channels: int, properties: dict):
        super().__init__(); patch = int(properties.get("patch_size", 8)); dim = int(properties.get("d_model", 64)); self.layer = nn.Conv2d(channels, dim, patch, patch)
    def forward(self, x): return self.layer(x).flatten(2).transpose(1, 2)


class CausalPad1D(nn.Module):
    def __init__(self, properties: dict):
        super().__init__(); kernel=int(properties.get("kernel_size",3));dilation=int(properties.get("dilation",1));self.left=(kernel-1)*dilation
    def forward(self,x): return F.pad(x,(self.left,0))


class TemporalSelect(nn.Module):
    def forward(self, x): return x[:, -1]


class Concat(nn.Module):
    def __init__(self, axis: int): super().__init__(); self.axis = axis
    def forward(self, *values): return torch.cat(values, dim=self.axis)


class Add(nn.Module):
    def forward(self, *values):
        result = values[0]
        for value in values[1:]: result = result + value
        return result


class Reshape(nn.Module):
    def __init__(self, shape): super().__init__(); self.shape = tuple(shape)
    def forward(self, x): return x.reshape(x.shape[0], *self.shape)


class Resize2D(nn.Module):
    def __init__(self, scale): super().__init__(); self.scale = float(scale)
    def forward(self, x): return F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)


def _shape_value(value) -> list[int]:
    if isinstance(value, str): return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)): return [int(item) for item in value]
    return [int(value)]


def create_operation(block_type: str, properties: dict, samples: list[torch.Tensor]) -> nn.Module:
    x = samples[0]
    if block_type in {"identity", "output"}: return nn.Identity()
    if block_type == "linear": return nn.Linear(x.shape[-1], int(properties.get("out_features", 64)), bias=bool(properties.get("bias", True)))
    if block_type == "relu": return nn.ReLU()
    if block_type == "leaky_relu": return nn.LeakyReLU(float(properties.get("negative_slope", 0.01)))
    if block_type == "gelu": return nn.GELU()
    if block_type == "silu": return nn.SiLU()
    if block_type == "tanh": return nn.Tanh()
    if block_type == "sigmoid": return nn.Sigmoid()
    if block_type == "softmax": return nn.Softmax(dim=int(properties.get("dim", 1)))
    if block_type == "dropout": return nn.Dropout(float(properties.get("p", .1)))
    if block_type == "dropout2d": return nn.Dropout2d(float(properties.get("p", .1)))
    if block_type == "flatten": return nn.Flatten(1)
    if block_type == "batchnorm1d": return nn.BatchNorm1d(x.shape[1])
    if block_type == "batchnorm2d": return nn.BatchNorm2d(x.shape[1])
    if block_type == "layernorm": return nn.LayerNorm(x.shape[-1])
    if block_type == "embedding": return nn.Embedding(int(properties.get("vocab_size", 64)), int(properties.get("d_model", 64)), padding_idx=int(properties.get("padding_idx", 0)))
    if block_type == "sequence_projection": return nn.Linear(x.shape[-1], int(properties.get("d_model", 64)))
    if block_type == "positional_encoding": return PositionalEncoding(x.shape[-1], int(properties.get("max_length", 2048)), bool(properties.get("learned", False)))
    if block_type == "transformer_encoder": return TransformerBlock(x.shape[-1], properties, False)
    if block_type == "causal_transformer": return TransformerBlock(x.shape[-1], properties, True)
    if block_type == "sequence_pool": return SequencePool(str(properties.get("mode", "mean")))
    if block_type == "patch_embedding": return PatchEmbedding(x.shape[1], properties)
    if block_type == "permute": return Permute(_shape_value(properties.get("dims", "2,1")))
    if block_type == "conv1d":
        kernel = int(properties.get("kernel_size", 3))
        return nn.Conv1d(x.shape[1], int(properties.get("out_channels", 32)), kernel, stride=int(properties.get("stride", 1)), padding=int(properties.get("padding", kernel // 2)), dilation=int(properties.get("dilation", 1)), groups=int(properties.get("groups", 1)), bias=bool(properties.get("bias", True)))
    if block_type == "causalpad1d": return CausalPad1D(properties)
    if block_type == "maxpool1d": return nn.MaxPool1d(int(properties.get("kernel_size", 2)), stride=int(properties.get("stride", 2)), padding=int(properties.get("padding", 0)))
    if block_type == "avgpool1d": return nn.AvgPool1d(int(properties.get("kernel_size", 2)), stride=int(properties.get("stride", 2)), padding=int(properties.get("padding", 0)))
    if block_type == "adaptiveavgpool1d": return nn.AdaptiveAvgPool1d(int(properties.get("output_size", 1)))
    if block_type == "conv2d":
        kernel = int(properties.get("kernel_size", 3))
        return nn.Conv2d(x.shape[1], int(properties.get("out_channels", 16)), kernel, stride=int(properties.get("stride", 1)), padding=int(properties.get("padding", kernel // 2)), dilation=int(properties.get("dilation", 1)), groups=int(properties.get("groups", 1)), bias=bool(properties.get("bias", True)))
    if block_type == "convtranspose2d": return nn.ConvTranspose2d(x.shape[1], int(properties.get("out_channels", 16)), int(properties.get("kernel_size", 2)), stride=int(properties.get("stride", 2)), padding=int(properties.get("padding", 0)), output_padding=int(properties.get("output_padding", 0)))
    if block_type == "maxpool2d": return nn.MaxPool2d(int(properties.get("kernel_size", 2)), stride=int(properties.get("stride", 2)), padding=int(properties.get("padding", 0)))
    if block_type == "avgpool2d": return nn.AvgPool2d(int(properties.get("kernel_size", 2)), stride=int(properties.get("stride", 2)), padding=int(properties.get("padding", 0)))
    if block_type == "adaptiveavgpool2d": return nn.AdaptiveAvgPool2d(int(properties.get("output_size", 1)))
    if block_type == "lstm": return LSTMSequence(x.shape[-1], properties)
    if block_type in {"rnn", "gru"}: return RecurrentSequence(block_type, x.shape[-1], properties)
    if block_type == "temporal_select": return TemporalSelect()
    if block_type == "concat": return Concat(int(properties.get("axis", 1)))
    if block_type == "add": return Add()
    if block_type == "reshape": return Reshape(_shape_value(properties.get("shape", [-1])))
    if block_type == "resize2d": return Resize2D(properties.get("scale_factor", 2))
    raise BackendError("MODEL_BLOCK_NOT_SUPPORTED", block_type)


class GraphModel(nn.Module):
    def __init__(self, graph: dict, input_shape: list[int], input_dtype=torch.float32):
        super().__init__()
        nodes = {node["id"]: node for node in graph.get("nodes", [])}; edges = graph.get("edges", [])
        if not nodes: raise BackendError("GRAPH_EMPTY")
        incoming: dict[str, list[str]] = defaultdict(list); outgoing: dict[str, list[str]] = defaultdict(list); indegree = {node_id: 0 for node_id in nodes}
        for edge in edges:
            source, target = edge["source"], edge["target"]
            if source not in nodes or target not in nodes: raise BackendError("GRAPH_EDGE_TO_MISSING_NODE")
            incoming[target].append(source); outgoing[source].append(target); indegree[target] += 1
        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0); order = []
        while queue:
            current = queue.popleft(); order.append(current)
            for target in outgoing[current]:
                indegree[target] -= 1
                if indegree[target] == 0: queue.append(target)
        if len(order) != len(nodes): raise BackendError("GRAPH_CYCLE")
        inputs = [node_id for node_id, node in nodes.items() if node.get("data", {}).get("blockType") == "input"]
        outputs = [node_id for node_id, node in nodes.items() if node.get("data", {}).get("blockType") == "output"]
        if len(inputs) != 1 or len(outputs) != 1: raise BackendError("GRAPH_SINGLE_INPUT_OUTPUT_REQUIRED")
        reachable = {inputs[0]}; frontier = [inputs[0]]
        while frontier:
            current = frontier.pop()
            for target in outgoing[current]:
                if target not in reachable: reachable.add(target); frontier.append(target)
        if outputs[0] not in reachable: raise BackendError("GRAPH_NO_PATH_INPUT_OUTPUT")
        orphans = [node_id for node_id in nodes if node_id not in reachable]
        if orphans: raise BackendError("GRAPH_DISCONNECTED_BLOCKS", ", ".join(orphans[:3]))
        self.nodes = nodes; self.incoming = dict(incoming); self.order = order; self.input_id = inputs[0]; self.output_id = outputs[0]; self.operations = nn.ModuleDict()
        values = {self.input_id: torch.zeros(2, *input_shape, dtype=input_dtype)}
        for node_id in order:
            if node_id == self.input_id: continue
            sources = self.incoming.get(node_id, [])
            if not sources: raise BackendError("GRAPH_BLOCK_MISSING_INPUT", node_id)
            samples = [values[source] for source in sources]; data = nodes[node_id].get("data", {}); operation = create_operation(data.get("blockType", ""), data.get("properties", {}), samples); self.operations[node_id] = operation
            try:
                operation.eval()
                with torch.inference_mode(): values[node_id] = operation(*samples)
                operation.train()
            except Exception as exc: raise BackendError("GRAPH_BLOCK_EXECUTION_FAILED", f"{data.get('label', node_id)}: {exc}") from exc
        self.output_shape = list(values[self.output_id].shape[1:])
        self.node_shapes = {node_id: list(value.shape[1:]) for node_id, value in values.items()}

    def forward(self, x):
        values = {self.input_id: x}
        for node_id in self.order:
            if node_id != self.input_id: values[node_id] = self.operations[node_id](*[values[source] for source in self.incoming[node_id]])
        return values[self.output_id]


def build_model(architecture: str, input_shape: list[int], output_shape: list[int], graph: dict, task_id: str) -> nn.Module:
    validate_compatibility(task_id, architecture)
    input_dtype = torch.long if task_id.startswith("text.") else torch.float32
    model = GraphModel(graph, input_shape, input_dtype)
    expected = [int(value) for value in output_shape]
    if model.output_shape != expected: raise BackendError("GRAPH_OUTPUT_SHAPE_MISMATCH", f"{model.output_shape} vs {expected}")
    return model
