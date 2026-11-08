"""Bounded, explicit graph repairs. A suggestion is returned only after the real dry run passes."""
from copy import deepcopy
from math import gcd, prod
from uuid import uuid4

import torch

from .errors import BackendError
from .models import GraphModel


def _diagnostic(error):
    return {"code": getattr(error, "code", "GRAPH_BLOCK_EXECUTION_FAILED"), "message": str(error), **getattr(error, "context", {})}


def _repair_step(graph, request, error):
    """Change a copy at the first failing operation; never infer an intended architecture."""
    context = getattr(error, "context", {})
    node = next((n for n in graph["nodes"] if n["id"] == context.get("node_id")), None)
    shapes = context.get("input_shapes", [])
    steps = []

    def change(n, key, value):
        properties = n["data"].setdefault("properties", {})
        kind=n["data"]["blockType"]
        convolution=kind in {"conv1d","conv2d"}
        kernel=int(properties.get("kernel_size",3 if convolution else 2))
        defaults={"groups":1,"dilation":1,"stride":1 if convolution else 2,"padding":kernel//2 if convolution else 0,"kernel_size":3 if convolution else 2,"out_channels":32 if kind=="conv1d" else 16,"num_layers":1,"hidden_size":64}
        if properties.get(key,defaults.get(key)) == value: return
        steps.append({"key": "property", "node": n["data"].get("label", n["id"]), "property": key, "before": str(properties.get(key, "—")), "after": str(value)})
        properties[key] = value

    def insert(edge, block, properties):
        node_id = f"repair-{block}-{uuid4().hex[:8]}"
        graph["nodes"].append({"id": node_id, "type": "modelNode", "position": {"x": 0, "y": 0}, "data": {"blockType": block, "label": block, "category": "Repair", "properties": properties}})
        graph["edges"].remove(edge)
        graph["edges"].extend([{"id": f"{node_id}-in", "source": edge["source"], "sourceHandle": edge.get("sourceHandle"), "target": node_id}, {"id": f"{node_id}-out", "source": node_id, "target": edge["target"], "targetHandle": edge.get("targetHandle")}])
        steps.append({"key": "insert", "block": block, "node": next(n["data"].get("label", n["id"]) for n in graph["nodes"] if n["id"] == edge["target"]), "after": ", ".join(f"{k}={v}" for k, v in properties.items())})
        return graph["edges"][-1]

    code = getattr(error, "code", "")
    if node and not node["data"]["blockType"].startswith("hf_"):
        kind = node["data"]["blockType"]
        p = node["data"].get("properties", {})
        shape = shapes[0] if shapes else []
        links = [e for e in graph["edges"] if e["target"] == node["id"]]
        links.sort(key=lambda e: (1,int(e["targetHandle"].split("_")[-1])) if str(e.get("targetHandle", "")).startswith("input_") and str(e["targetHandle"]).split("_")[-1].isdigit() else (0,0))
        if kind in {"transformer_encoder", "causal_transformer"} and shape and (code == "MODEL_D_MODEL_NOT_DIVISIBLE_BY_HEADS" or int(p.get("heads",4)) < 1):
            heads = max(1, int(p.get("heads", 4)))
            divisors = [i for i in range(1, min(shape[-1], heads) + 1) if shape[-1] % i == 0]
            change(node, "heads", max(divisors))
        elif code == "MODEL_SEQUENCE_TOO_LONG" and shape:
            change(node, "max_length", shape[0])
        elif kind in {"add", "concat"} and shapes:
            axis = int(p.get("axis", 1)); rank = len(shapes[0]) + 1
            if code == "GRAPH_CONCAT_AXIS_INVALID": change(node, "axis", 1)
            elif code in {"GRAPH_ADD_INPUT_SHAPE_MISMATCH", "GRAPH_CONCAT_INPUT_SHAPE_MISMATCH"}:
                axis = axis if axis >= 0 else rank + axis
                reference = shapes[0]
                for edge, actual in zip(links[1:], shapes[1:]):
                    if len(actual) != len(reference): return []
                    current = edge
                    if len(actual) == 3:
                        target = list(actual)
                        for dim in [1, 2]:
                            if kind == "add" or axis != dim + 1: target[dim] = reference[dim]
                        if target[1:] != actual[1:]: current = insert(current, "resize2d", {"size": ",".join(map(str, target[1:]))})
                        if actual[0] != reference[0] and (kind == "add" or axis != 1): insert(current, "conv2d", {"out_channels": reference[0], "kernel_size": 1, "stride": 1, "padding": 0})
                    elif len(actual) == 1 and actual != reference:
                        insert(current, "linear", {"out_features": reference[0]})
                    elif len(actual) == 2:
                        if request["architecture"] == "cnn1d":
                            if actual[1] != reference[1] and (kind == "add" or axis != 2): current = insert(current, "adaptiveavgpool1d", {"output_size": reference[1]})
                            if actual[0] != reference[0] and (kind == "add" or axis != 1): insert(current, "conv1d", {"out_channels": reference[0], "kernel_size": 1, "padding": 0})
                        elif actual[0] == reference[0] and actual[1] != reference[1] and (kind == "add" or axis != 2): insert(current, "linear", {"out_features": reference[1]})
        elif kind in {"dropout", "dropout2d"} and not 0 <= float(p.get("p", .1)) <= 1: change(node, "p", max(0, min(1, float(p["p"]))))
        elif kind == "reshape" and shape: change(node, "shape", str(prod(shape)))
        elif kind == "batchnorm2d" and len(shape) == 1:
            node["data"]["blockType"] = "batchnorm1d"; steps.append({"key": "replace", "node": node["data"].get("label", node["id"]), "block": "batchnorm1d"})
        elif kind == "batchnorm1d" and len(shape) == 3:
            node["data"]["blockType"] = "batchnorm2d"; steps.append({"key": "replace", "node": node["data"].get("label", node["id"]), "block": "batchnorm2d"})
        elif kind == "patch_embedding" and len(shape) == 3:
            change(node, "patch_size", max(1, min(int(p.get("patch_size", 8)), *shape[1:])))
        elif kind in {"conv1d", "conv2d", "maxpool1d", "avgpool1d", "maxpool2d", "avgpool2d"} and shape:
            spatial = shape[1:]; kernel = max(1, int(p.get("kernel_size", 3 if kind.startswith("conv") else 2)))
            if not spatial: return []
            if kind.startswith("conv"):
                change(node, "groups", max(1, gcd(gcd(shape[0], max(1, int(p.get("out_channels", 32 if kind=="conv1d" else 16)))), max(1, int(p.get("groups", 1))))))
                change(node, "out_channels", max(1, int(p.get("out_channels", 32 if kind=="conv1d" else 16))))
                dilation = max(1, int(p.get("dilation", 1))); change(node, "dilation", dilation)
                change(node, "padding", max(0, int(p.get("padding", kernel // 2)), (dilation * (kernel - 1) + 1 - min(spatial) + 1) // 2))
            else:
                kernel = min(kernel, min(spatial)); change(node, "padding", min(max(0, int(p.get("padding", 0))), kernel // 2))
            change(node, "kernel_size", kernel); change(node, "stride", max(1, int(p.get("stride", 1 if kind.startswith("conv") else 2))))
        elif kind in {"lstm", "gru", "rnn"}:
            change(node, "hidden_size", max(1, int(p.get("hidden_size", 64)))); change(node, "num_layers", max(1, int(p.get("num_layers", 1))))
        return steps

    if code in {"GRAPH_OUTPUT_SHAPE_MISMATCH", "GRAPH_LOCALIZATION_OUTPUT_SHAPE_INVALID"}:
        # Trace actual shapes without the task's output contract. No existing blocks are removed.
        model = GraphModel(graph, request["dataset"]["inputShape"], torch.long if request["task_id"].startswith("text.") else torch.float32)
        expected = request["dataset"]["outputShape"]
        if len(model.output_ids) != 1: return []
        terminal = next(e for e in graph["edges"] if e["target"] == model.output_ids[0])
        source = next(n for n in graph["nodes"] if n["id"] == terminal["source"])
        actual = model.node_shapes[source["id"]]
        if not isinstance(actual, list): actual = actual.get(terminal.get("sourceHandle"))
        if not actual: return []
        if source["data"]["blockType"] == "linear" and len(actual)==len(expected) and actual[:-1]==expected[:-1]:
            change(source, "out_features", expected[-1])
        elif prod(actual)==prod(expected):
            insert(terminal,"reshape",{"shape":",".join(map(str,expected))})
        elif len(actual)==3 and len(expected)==3:
            if actual[1:]!=expected[1:]:terminal=insert(terminal,"resize2d",{"size":",".join(map(str,expected[1:]))})
            if actual[0]!=expected[0]:insert(terminal,"conv2d",{"out_channels":expected[0],"kernel_size":1,"padding":0})
        elif source["data"]["blockType"] == "linear" and len(expected)==1:
            incoming = [e for e in graph["edges"] if e["target"] == source["id"]]
            if len(incoming) != 1: return []
            previous = model.node_shapes[incoming[0]["source"]]
            if not isinstance(previous, list): previous = previous.get(incoming[0].get("sourceHandle"))
            if not previous or prod(previous) * expected[0] > 5_000_000: return []
            insert(incoming[0], "flatten", {});change(source, "out_features", expected[0])
        else:
            if prod(actual) * prod(expected) > 5_000_000: return []
            if len(actual) > 1: terminal = insert(terminal, "flatten", {})
            terminal=insert(terminal,"linear",{"out_features":prod(expected)})
            if len(expected)>1:insert(terminal,"reshape",{"shape":",".join(map(str,expected))})
    return steps


def propose_repairs(request, validate):
    """No project writes, dataset edits, training, or quality optimisation."""
    original = deepcopy(request["graph"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        try:
            validate(original)
            return {"problem": None, "solutions": [], "status": "valid"}
        except Exception as error:
            problem = _diagnostic(error)
        candidate = deepcopy(original); steps = []; seen = set()
        for _ in range(8):
            try:
                result = validate(candidate)
                return {"problem": problem, "status": "repairable", "solutions": [{"id": uuid4().hex, "steps": steps, "graph": candidate, "validation": result}]}
            except Exception as error:
                signature = str(error) + str(getattr(error, "context", {}).get("input_shapes"))
                if signature in seen: break
                seen.add(signature)
                try: changes = _repair_step(candidate, request, error)
                except (ValueError, TypeError, KeyError, IndexError, StopIteration, RuntimeError): break
                if not changes: break
                steps.extend(changes)
        return {"problem": problem, "solutions": [], "status": "unsupported"}
