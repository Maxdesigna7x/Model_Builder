from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F

from .catalog import validate_compatibility
from .errors import BackendError


def hf_load_error(repo:str,revision:str,exc:Exception) -> BackendError:
    message=str(exc);gated="gated" in message.lower() or exc.__class__.__name__ in {"GatedRepoError","RepositoryNotFoundError"}
    return BackendError("HF_GATED_MODEL_ACCEPTANCE_REQUIRED" if gated else "HF_MODEL_LOAD_FAILED",f"{repo}@{revision}: {message}")


def apply_stage_policy(module: nn.Module, properties: dict) -> None:
    """Freeze a backbone and optionally open only its last N numbered stages."""
    if not bool(properties.get("frozen",True)): return
    for parameter in module.parameters(): parameter.requires_grad=False
    requested=max(0,int(properties.get("trainable_stages",0)))
    if not requested: return
    stage_parameters: dict[int,list[nn.Parameter]]=defaultdict(list)
    for name,parameter in module.named_parameters():
        parts=name.split(".")
        for index,part in enumerate(parts[:-1]):
            if part in {"stage","stages","layer","layers","block","blocks","h"} and parts[index+1].isdigit():
                stage_parameters[int(parts[index+1])].append(parameter);break
    for stage in sorted(stage_parameters)[-requested:]:
        for parameter in stage_parameters[stage]: parameter.requires_grad=True


class HuggingFaceImageProcessor(nn.Module):
    def __init__(self, properties: dict):
        super().__init__()
        try:
            from transformers import AutoImageProcessor
        except ImportError as exc:
            raise BackendError("HF_MODEL_DEPENDENCIES_MISSING", "Instala las dependencias del proyecto") from exc
        from .pretrained import cache_dir
        repo=str(properties["repo_id"]);revision=str(properties["revision"])
        common={"revision":revision,"cache_dir":str(cache_dir()),"trust_remote_code":False,"local_files_only":bool(properties.get("offline",False))}
        try:
            processor=AutoImageProcessor.from_pretrained(repo,**common)
        except Exception as exc:
            raise hf_load_error(repo,revision,exc) from exc
        size=getattr(processor,"size",224)
        if isinstance(size,Mapping) or hasattr(size,"get"):
            size=next((size.get(key) for key in ("height","shortest_edge","width","longest_edge") if size.get(key) is not None),224)
        self.image_size=int(size if not isinstance(size,(list,tuple)) else size[0])
        mean=list(getattr(processor,"image_mean",[0.485,0.456,0.406]));std=list(getattr(processor,"image_std",[0.229,0.224,0.225]))
        self.register_buffer("image_mean",torch.tensor(mean).reshape(1,-1,1,1),persistent=True)
        self.register_buffer("image_std",torch.tensor(std).reshape(1,-1,1,1),persistent=True)
    def forward(self,x):
        x=x.float()
        if x.shape[1]==1 and self.image_mean.shape[1]==3: x=x.repeat(1,3,1,1)
        if x.shape[-2:]!=(self.image_size,self.image_size): x=F.interpolate(x,size=(self.image_size,self.image_size),mode="bilinear",align_corners=False)
        return (x-self.image_mean.to(x))/(self.image_std.to(x)+1e-7)


class HuggingFaceVisionBackbone(nn.Module):
    """Safe, commit-pinned Transformers backbone."""
    def __init__(self, properties: dict):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise BackendError("HF_MODEL_DEPENDENCIES_MISSING", "Instala las dependencias del proyecto") from exc
        from .pretrained import cache_dir
        repo=str(properties["repo_id"]);revision=str(properties["revision"])
        try:
            self.backbone=AutoModel.from_pretrained(repo,revision=revision,cache_dir=str(cache_dir()),trust_remote_code=False,local_files_only=bool(properties.get("offline",False)),use_safetensors=True)
        except Exception as exc:
            raise hf_load_error(repo,revision,exc) from exc
        self.expose_feature_maps=bool(properties.get("expose_feature_maps",False));self.feature_count=max(1,min(4,int(properties.get("feature_count",3))))
        apply_stage_policy(self.backbone,properties)

    def forward(self,x):
        # Most Transformers vision backbones accept output_hidden_states, but small
        # adapter implementations (including offline integrations) may only accept
        # pixel_values. Ask for hidden states only when feature-map ports are used.
        output=self.backbone(pixel_values=x,**({"output_hidden_states":True} if self.expose_feature_maps else {}))
        if self.expose_feature_maps:
            hidden_states=getattr(output,"hidden_states",None) or ();features=[value for value in hidden_states if isinstance(value,torch.Tensor) and value.ndim==4]
            last=getattr(output,"last_hidden_state",None)
            if isinstance(last,torch.Tensor) and last.ndim==4 and (not features or features[-1] is not last): features.append(last)
            if len(features)<self.feature_count: raise BackendError("HF_SPATIAL_FEATURES_UNAVAILABLE",f"se solicitaron {self.feature_count} mapas, se recibieron {len(features)}")
            chosen=features[-self.feature_count:]
            return {f"map_{index}":value for index,value in enumerate(chosen)}
        pooled=getattr(output,"pooler_output",None)
        if pooled is not None: return pooled.flatten(1)
        hidden=output.last_hidden_state
        return hidden.mean(tuple(range(2,hidden.ndim))) if hidden.ndim>2 and hidden.shape[1]>hidden.shape[-1] else hidden.mean(1)


class HuggingFaceTokenProcessor(nn.Module):
    """Graph marker for token ids materialized by the versioned data adapter."""
    def forward(self,x):
        if x.ndim<3:return {"input_ids":x.long(),"attention_mask":x.ne(0).long(),"token_type_ids":torch.zeros_like(x).long()}
        return {"input_ids":x[...,0].long(),"attention_mask":x[...,1].long(),"token_type_ids":x[...,2].long()}


class HuggingFaceTextBackbone(nn.Module):
    def __init__(self,properties:dict):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc: raise BackendError("HF_MODEL_DEPENDENCIES_MISSING") from exc
        from .pretrained import cache_dir
        repo=str(properties["repo_id"]);revision=str(properties["revision"]);common={"revision":revision,"cache_dir":str(cache_dir()),"trust_remote_code":False,"local_files_only":bool(properties.get("offline",False))}
        try:
            config=AutoConfig.from_pretrained(repo,**common);self.backbone=AutoModel.from_pretrained(repo,use_safetensors=True,**common)
        except Exception as exc: raise hf_load_error(repo,revision,exc) from exc
        self.pad_token_id=int(getattr(config,"pad_token_id",0) or 0)
        apply_stage_policy(self.backbone,properties)

    def forward(self,x):
        batch=x if isinstance(x,Mapping) else {"input_ids":x.long(),"attention_mask":x.ne(self.pad_token_id).long()};mask=batch["attention_mask"]
        accepted={key:value for key,value in batch.items() if key!="token_type_ids" or self.backbone.config.model_type not in {"distilbert"}}
        output=self.backbone(**accepted)
        pooled=getattr(output,"pooler_output",None)
        if pooled is not None: return pooled
        hidden=output.last_hidden_state;weights=mask.unsqueeze(-1);return (hidden*weights).sum(1)/weights.sum(1).clamp_min(1)


class HuggingFaceCausalLM(nn.Module):
    def __init__(self,properties:dict):
        super().__init__()
        try: from transformers import AutoModelForCausalLM
        except ImportError as exc: raise BackendError("HF_MODEL_DEPENDENCIES_MISSING") from exc
        from .pretrained import cache_dir
        repo=str(properties["repo_id"]);revision=str(properties["revision"])
        try: self.model=AutoModelForCausalLM.from_pretrained(repo,revision=revision,cache_dir=str(cache_dir()),trust_remote_code=False,local_files_only=bool(properties.get("offline",False)),use_safetensors=True)
        except Exception as exc: raise hf_load_error(repo,revision,exc) from exc
        self.pad_token_id=int(self.model.config.pad_token_id if self.model.config.pad_token_id is not None else self.model.config.eos_token_id)

    def forward(self,x):
        batch=x if isinstance(x,Mapping) else {"input_ids":x.long(),"attention_mask":x.ne(self.pad_token_id).long()}
        return self.model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).logits

    def generate_tokens(self,prompt:torch.Tensor,max_new_tokens:int):
        if prompt.ndim>=3: prompt=prompt[...,0]
        valid=prompt[prompt.ne(self.pad_token_id)].reshape(1,-1)
        generated=self.model.generate(valid,max_new_tokens=max_new_tokens,pad_token_id=self.pad_token_id,eos_token_id=self.model.config.eos_token_id)
        return generated[0,valid.shape[1]:].tolist()


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


class MultiScaleCNN(nn.Module):
    """Small spatial feature pyramid with actual named tensors and one shared forward."""
    def __init__(self,channels:int,properties:dict):
        super().__init__();base=max(4,int(properties.get("base_channels",32)));self.stage_count=max(2,min(4,int(properties.get("stages",3))))
        stages=[];current=channels
        for index in range(self.stage_count):
            target=base*(2**index);stages.append(nn.Sequential(nn.Conv2d(current,target,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(target),nn.ReLU(inplace=False),nn.Conv2d(target,target,3,padding=1,bias=False),nn.BatchNorm2d(target),nn.ReLU(inplace=False)));current=target
        self.stages=nn.ModuleList(stages)
    def forward(self,x):
        result={}
        for index,stage in enumerate(self.stages):
            x=stage(x);result[f"p{index+2}"]=x
        return result


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
    if block_type == "hf_image_processor": return HuggingFaceImageProcessor(properties)
    if block_type == "hf_vision_backbone": return HuggingFaceVisionBackbone(properties)
    if block_type == "hf_token_processor": return HuggingFaceTokenProcessor()
    if block_type == "hf_text_backbone": return HuggingFaceTextBackbone(properties)
    if block_type == "hf_causal_lm": return HuggingFaceCausalLM(properties)
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
    if block_type == "multi_scale_cnn": return MultiScaleCNN(x.shape[1],properties)
    raise BackendError("MODEL_BLOCK_NOT_SUPPORTED", block_type)


class GraphModel(nn.Module):
    def __init__(self, graph: dict, input_shape: list[int], input_dtype=torch.float32):
        super().__init__()
        nodes = {node["id"]: node for node in graph.get("nodes", [])}; edges = graph.get("edges", [])
        if not nodes: raise BackendError("GRAPH_EMPTY")
        incoming: dict[str, list[str]] = defaultdict(list); outgoing: dict[str, list[str]] = defaultdict(list); indegree = {node_id: 0 for node_id in nodes}
        targets_with_named_inputs=set();incoming_edges:dict[str,list[dict]]=defaultdict(list)
        for edge in edges:
            source, target = edge["source"], edge["target"]
            if source not in nodes or target not in nodes: raise BackendError("GRAPH_EDGE_TO_MISSING_NODE")
            target_handle=edge.get("targetHandle")
            target_type=nodes[target].get("data",{}).get("blockType")
            if target_handle and target_type in {"concat","add"}:
                key=(target,target_handle)
                if key in targets_with_named_inputs: raise BackendError("GRAPH_PORT_OCCUPIED",f"{target}:{target_handle}")
                targets_with_named_inputs.add(key)
            incoming[target].append(source);incoming_edges[target].append(edge); outgoing[source].append(target); indegree[target] += 1
        for node_id,sources in incoming.items():
            if nodes[node_id].get("data",{}).get("blockType") in {"concat","add"}:
                indexed=[(edge.get("targetHandle"),edge,position) for position,edge in enumerate(incoming_edges[node_id])]
                def input_order(item):
                    handle,_,position=item
                    if not handle:return (0,0,position)
                    suffix=str(handle).rsplit("_",1)[-1]
                    return (1,0,int(suffix)) if suffix.isdigit() else (1,1,str(handle))
                incoming_edges[node_id]=[edge for _,edge,_ in sorted(indexed,key=input_order)]
                incoming[node_id]=[edge["source"] for edge in incoming_edges[node_id]]
        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0); order = []
        while queue:
            current = queue.popleft(); order.append(current)
            for target in outgoing[current]:
                indegree[target] -= 1
                if indegree[target] == 0: queue.append(target)
        if len(order) != len(nodes): raise BackendError("GRAPH_CYCLE")
        inputs = [node_id for node_id, node in nodes.items() if node.get("data", {}).get("blockType") == "input"]
        outputs = [node_id for node_id, node in nodes.items() if node.get("data", {}).get("blockType") == "output"]
        if len(inputs) != 1 or not outputs: raise BackendError("GRAPH_SINGLE_INPUT_REQUIRED")
        reachable = {inputs[0]}; frontier = [inputs[0]]
        while frontier:
            current = frontier.pop()
            for target in outgoing[current]:
                if target not in reachable: reachable.add(target); frontier.append(target)
        if any(output_id not in reachable for output_id in outputs): raise BackendError("GRAPH_NO_PATH_INPUT_OUTPUT")
        can_reach_output=set(outputs);frontier=list(outputs)
        while frontier:
            current=frontier.pop()
            for source in incoming.get(current,[]):
                if source not in can_reach_output: can_reach_output.add(source);frontier.append(source)
        orphans = [node_id for node_id in nodes if node_id not in can_reach_output]
        if orphans: raise BackendError("GRAPH_DISCONNECTED_BLOCKS", ", ".join(orphans[:3]))
        self.nodes = nodes; self.incoming = dict(incoming); self.incoming_edges={key:list(value) for key,value in incoming_edges.items()};self.order = order; self.input_id = inputs[0]; self.output_ids = outputs; self.output_id = outputs[0] if len(outputs)==1 else None; self.operations = nn.ModuleDict();self.multiport_nodes={node_id for node_id,node in nodes.items() if node.get("data",{}).get("blockType")=="multi_scale_cnn" or (node.get("data",{}).get("blockType")=="hf_vision_backbone" and node.get("data",{}).get("properties",{}).get("expose_feature_maps"))}
        values = {self.input_id: torch.zeros(2, *input_shape, dtype=input_dtype)}
        for node_id in order:
            if node_id == self.input_id: continue
            sources = self.incoming.get(node_id, [])
            if not sources: raise BackendError("GRAPH_BLOCK_MISSING_INPUT", node_id)
            links=self.incoming_edges.get(node_id,[]);samples=[self._edge_value(edge,values) for edge in links];data = nodes[node_id].get("data", {}); operation = create_operation(data.get("blockType", ""), data.get("properties", {}), samples); self.operations[node_id] = operation
            block_type=data.get("blockType","")
            if block_type in {"add","concat"}:
                if len(samples)<2: raise BackendError("GRAPH_MERGE_REQUIRES_TWO_INPUTS",node_id)
                shapes=[tuple(value.shape) for value in samples]
                if block_type=="add" and any(shape!=shapes[0] for shape in shapes[1:]):
                    mismatch=next(index for index,shape in enumerate(shapes) if shape!=shapes[0]);port=links[mismatch].get("targetHandle") or f"input_{mismatch}"
                    raise BackendError("GRAPH_ADD_INPUT_SHAPE_MISMATCH",f"{node_id}:{port} {shapes[mismatch]} no coincide con input_0 {shapes[0]}")
                if block_type=="concat":
                    axis=int(data.get("properties",{}).get("axis",1));axis=axis if axis>=0 else len(shapes[0])+axis
                    if axis<0 or axis>=len(shapes[0]): raise BackendError("GRAPH_CONCAT_AXIS_INVALID",f"{node_id}: axis {axis} fuera de rango para {len(shapes[0])} dimensiones")
                    if any(len(shape)!=len(shapes[0]) or any(shape[dim]!=shapes[0][dim] for dim in range(len(shape)) if dim!=axis) for shape in shapes[1:]):
                        mismatch=next(index for index,shape in enumerate(shapes) if len(shape)!=len(shapes[0]) or any(shape[dim]!=shapes[0][dim] for dim in range(min(len(shape),len(shapes[0]))) if dim!=axis));port=links[mismatch].get("targetHandle") or f"input_{mismatch}"
                        raise BackendError("GRAPH_CONCAT_INPUT_SHAPE_MISMATCH",f"{node_id}:{port} {shapes[mismatch]} no es compatible con input_0 {shapes[0]}")
            properties=data.get("properties",{})
            if bool(properties.get("frozen",False)):
                for parameter in operation.parameters(): parameter.requires_grad=False
            try:
                operation.eval()
                with torch.inference_mode(): values[node_id] = operation(*samples)
                operation.train()
            except Exception as exc: raise BackendError("GRAPH_BLOCK_EXECUTION_FAILED", f"{data.get('label', node_id)}: {exc}") from exc
        shape_of=lambda value:list((value.get("input_ids") if isinstance(value,Mapping) else value).shape[1:])
        self.output_shapes={str(nodes[node_id].get("data",{}).get("properties",{}).get("output_name") or nodes[node_id].get("data",{}).get("label") or node_id):shape_of(values[node_id]) for node_id in outputs}
        self.output_shape = shape_of(values[outputs[0]]) if len(outputs)==1 else None
        self.node_shapes = {node_id:({name:shape_of(item) for name,item in value.items()} if node_id in self.multiport_nodes else shape_of(value)) for node_id,value in values.items()}

    def _edge_value(self,edge:dict,values:dict):
        source=edge["source"];value=values[source];handle=edge.get("sourceHandle")
        if source in self.multiport_nodes:
            if not handle: raise BackendError("GRAPH_SOURCE_PORT_REQUIRED",source)
            if not isinstance(value,Mapping) or handle not in value: raise BackendError("GRAPH_SOURCE_PORT_NOT_FOUND",f"{source}:{handle}")
            return value[handle]
        if handle: raise BackendError("GRAPH_SOURCE_PORT_NOT_FOUND",f"{source}:{handle}")
        return value

    def forward(self, x):
        values = {self.input_id: x}
        for node_id in self.order:
            if node_id != self.input_id: values[node_id] = self.operations[node_id](*[self._edge_value(edge,values) for edge in self.incoming_edges[node_id]])
        if len(self.output_ids)==1:return values[self.output_ids[0]]
        result={}
        for node_id in self.output_ids:
            data=self.nodes[node_id].get("data",{});properties=data.get("properties",{});name=str(properties.get("output_name") or data.get("label") or node_id)
            if name in result:raise BackendError("GRAPH_OUTPUT_NAME_DUPLICATE",name)
            result[name]=values[node_id]
        return result


def build_model(architecture: str, input_shape: list[int], output_shape: list[int], graph: dict, task_id: str) -> nn.Module:
    validate_compatibility(task_id, architecture)
    input_dtype = torch.long if task_id.startswith("text.") else torch.float32
    model = GraphModel(graph, input_shape, input_dtype)
    expected = [int(value) for value in output_shape]
    contains_dynamic_hf=any(node.get("data",{}).get("blockType")=="hf_causal_lm" for node in graph.get("nodes",[]))
    if task_id == "image.detection":
        if set(model.output_shapes) != {"boxes", "classes"}:
            raise BackendError("GRAPH_DETECTION_OUTPUT_NAMES_REQUIRED","Nombra las salidas terminales boxes y classes.")
        boxes_shape=model.output_shapes["boxes"];classes_shape=model.output_shapes["classes"]
        if len(boxes_shape)!=3 or boxes_shape[0]!=4 or len(classes_shape)!=3 or classes_shape[0]!=expected[0] or boxes_shape[1:]!=classes_shape[1:]:
            raise BackendError("GRAPH_DETECTION_OUTPUT_SHAPE_INVALID",f"boxes={boxes_shape}, classes={classes_shape}, classes esperadas={expected[0]}")
        return model
    if task_id == "image.localization":
        if model.output_shape != [4]: raise BackendError("GRAPH_LOCALIZATION_OUTPUT_SHAPE_INVALID",str(model.output_shape))
    if model.output_shape is not None and model.output_shape != expected and not contains_dynamic_hf: raise BackendError("GRAPH_OUTPUT_SHAPE_MISMATCH", f"{model.output_shape} vs {expected}")
    if model.output_shape is None and any(shape!=expected for shape in model.output_shapes.values()):
        raise BackendError("GRAPH_MULTI_OUTPUT_RECIPE_REQUIRED","Las salidas con objetivos o formas diferentes requieren una receta de entrenamiento explícita.")
    return model
