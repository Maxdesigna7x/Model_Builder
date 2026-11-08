import pytest
import torch

from modelbuilder.errors import BackendError
from modelbuilder.manifest import semantic_hash
from modelbuilder.models import GraphModel, build_model
from modelbuilder.training import _loss


def _node(node_id, kind, properties=None, label=None):
    return {
        "id": node_id,
        "data": {
            "blockType": kind,
            "label": label or node_id,
            "properties": properties or {},
        },
    }


def _two_head_graph(*, second_name="aux", second_channels=3, frozen=False):
    return {
        "nodes": [
            _node("input", "input"),
            _node("shared", "conv2d", {"out_channels": 5, "kernel_size": 1, "padding": 0, "frozen": frozen}),
            _node("relu", "relu"),
            _node("main_head", "conv2d", {"out_channels": 3, "kernel_size": 1, "padding": 0}),
            _node("aux_head", "conv2d", {"out_channels": second_channels, "kernel_size": 1, "padding": 0}),
            _node("main", "output", {"output_name": "main"}),
            _node("aux", "output", {"output_name": second_name}),
        ],
        "edges": [
            {"source": "input", "target": "shared"},
            {"source": "shared", "target": "relu"},
            {"source": "relu", "target": "main_head"},
            {"source": "relu", "target": "aux_head"},
            {"source": "main_head", "target": "main"},
            {"source": "aux_head", "target": "aux"},
        ],
    }


def test_named_cnn_heads_share_trunk_and_backpropagate_both_losses():
    model = build_model("unet", [3, 8, 8], [3, 8, 8], _two_head_graph(), "image.segmentation.multiclass")
    prediction = model(torch.randn(2, 3, 8, 8))

    assert set(prediction) == {"main", "aux"}
    assert prediction["main"].shape == prediction["aux"].shape == (2, 3, 8, 8)
    loss = _loss("image.segmentation.multiclass", prediction, torch.zeros(2, 8, 8, dtype=torch.long))
    loss.backward()
    assert model.operations["shared"].weight.grad is not None
    assert model.operations["main_head"].weight.grad is not None
    assert model.operations["aux_head"].weight.grad is not None


def test_two_convolution_residual_block_has_compatible_add_inputs():
    graph = {
        "nodes": [
            _node("input", "input"),
            _node("conv1", "conv2d", {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "bias": False}),
            _node("relu1", "relu"),
            _node("conv2", "conv2d", {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "bias": False}),
            _node("add", "add", {"axis": 1}),
            _node("relu2", "relu"),
            _node("pool", "adaptiveavgpool2d", {"output_size": 1}),
            _node("flat", "flatten"),
            _node("head", "linear", {"out_features": 2}),
            _node("output", "output", {"output_name": "classes"}),
        ],
        "edges": [
            {"source": "input", "target": "conv1"},
            {"source": "conv1", "target": "relu1"},
            {"source": "relu1", "target": "conv2"},
            {"source": "conv1", "target": "add", "targetHandle": "input_0"},
            {"source": "conv2", "target": "add", "targetHandle": "input_1"},
            {"source": "add", "target": "relu2"},
            {"source": "relu2", "target": "pool"},
            {"source": "pool", "target": "flat"},
            {"source": "flat", "target": "head"},
            {"source": "head", "target": "output"},
        ],
    }
    model = build_model("cnn", [3, 16, 16], [2], graph, "image.classification")
    result = model(torch.randn(2, 3, 16, 16))
    assert result.shape == (2, 2)
    assert torch.isfinite(result).all()


def test_multi_output_graph_rejects_unnamed_or_duplicate_outputs():
    graph = _two_head_graph(second_name="main")
    model = build_model("unet", [3, 8, 8], [3, 8, 8], graph, "image.segmentation.multiclass")
    with pytest.raises(BackendError, match="GRAPH_OUTPUT_NAME_DUPLICATE"):
        model(torch.randn(1, 3, 8, 8))


def test_differently_shaped_heads_require_an_explicit_task_recipe():
    with pytest.raises(BackendError, match="GRAPH_MULTI_OUTPUT_RECIPE_REQUIRED"):
        build_model("unet", [3, 8, 8], [3, 8, 8], _two_head_graph(second_channels=4), "image.segmentation.multiclass")


def test_freezing_a_regular_cnn_layer_disables_its_gradients():
    model = build_model("unet", [3, 8, 8], [3, 8, 8], _two_head_graph(frozen=True), "image.segmentation.multiclass")
    assert not model.operations["shared"].weight.requires_grad
    assert model.operations["main_head"].weight.requires_grad


def test_concat_uses_named_input_port_order_even_if_edges_are_serialized_out_of_order():
    graph = {
        "nodes": [
            _node("input", "input"),
            _node("left", "conv2d", {"out_channels": 1, "kernel_size": 1, "padding": 0}),
            _node("right", "conv2d", {"out_channels": 1, "kernel_size": 1, "padding": 0}),
            _node("join", "concat", {"axis": 1}),
            _node("output", "output", {"output_name": "features"}),
        ],
        "edges": [
            {"source": "input", "target": "left"},
            {"source": "input", "target": "right"},
            {"source": "right", "target": "join", "targetHandle": "input_1"},
            {"source": "left", "target": "join", "targetHandle": "input_0"},
            {"source": "join", "target": "output"},
        ],
    }
    model = build_model("unet", [3, 2, 2], [2, 2, 2], graph, "image.segmentation.multiclass")
    with torch.no_grad():
        model.operations["left"].weight.zero_(); model.operations["left"].weight[0, 0, 0, 0] = 1
        model.operations["right"].weight.zero_(); model.operations["right"].weight[0, 1, 0, 0] = 1
        model.operations["left"].bias.zero_(); model.operations["right"].bias.zero_()
    sample=torch.tensor([[[[1.,1.],[1.,1.]],[[2.,2.],[2.,2.]],[[3.,3.],[3.,3.]]]])
    actual=model(sample)
    assert torch.equal(actual[:,0],sample[:,0])
    assert torch.equal(actual[:,1],sample[:,1])
    changed={**graph,"edges":[*graph["edges"][:2],{**graph["edges"][2],"targetHandle":"input_0"},{**graph["edges"][3],"targetHandle":"input_1"},*graph["edges"][4:]]}
    assert semantic_hash(graph)!=semantic_hash(changed)


def test_semantic_hash_ignores_json_edge_array_order_when_ports_are_explicit():
    graph={
        "nodes":[_node("input","input"),_node("left","identity"),_node("right","identity"),_node("join","concat",{"axis":1}),_node("output","output")],
        "edges":[
            {"source":"input","target":"left"},{"source":"input","target":"right"},
            {"source":"left","target":"join","targetHandle":"input_0"},
            {"source":"right","target":"join","targetHandle":"input_1"},
            {"source":"join","target":"output"},
        ],
    }
    reordered={**graph,"edges":[graph["edges"][4],graph["edges"][3],graph["edges"][2],graph["edges"][1],graph["edges"][0]]}
    assert semantic_hash(graph)==semantic_hash(reordered)


def test_legacy_concat_preserves_serialized_input_order():
    graph={
        "nodes":[_node("input","input"),_node("left","conv2d",{"out_channels":1,"kernel_size":1,"padding":0}),_node("right","conv2d",{"out_channels":1,"kernel_size":1,"padding":0}),_node("join","concat",{"axis":1}),_node("output","output")],
        # Old projects did not store targetHandle; serialized order is semantic.
        "edges":[{"source":"input","target":"left"},{"source":"input","target":"right"},{"source":"right","target":"join"},{"source":"left","target":"join"},{"source":"join","target":"output"}],
    }
    model=build_model("unet",[3,2,2],[2,2,2],graph,"image.segmentation.multiclass")
    with torch.no_grad():
        for name,channel in (("left",0),("right",1)):
            model.operations[name].weight.zero_();model.operations[name].weight[0,channel,0,0]=1;model.operations[name].bias.zero_()
    sample=torch.tensor([[[[1.,1.],[1.,1.]],[[2.,2.],[2.,2.]],[[3.,3.],[3.,3.]]]])
    result=model(sample)
    assert torch.equal(result[:,0],sample[:,1]) and torch.equal(result[:,1],sample[:,0])


def test_residual_add_trains_and_rejects_mismatched_shapes():
    nodes=[_node("input","input"),_node("skip","identity"),_node("branch","conv2d",{"out_channels":3,"kernel_size":3,"padding":1}),_node("join","add"),_node("output","output")]
    edges=[{"source":"input","target":"skip"},{"source":"input","target":"branch"},{"source":"skip","target":"join","targetHandle":"input_0"},{"source":"branch","target":"join","targetHandle":"input_1"},{"source":"join","target":"output"}]
    model=build_model("cnn",[3,8,8],[3,8,8],{"nodes":nodes,"edges":edges},"image.translation")
    prediction=model(torch.randn(2,3,8,8));loss=_loss("image.translation",prediction,torch.randn_like(prediction));loss.backward()
    assert prediction.shape==(2,3,8,8) and model.operations["branch"].weight.grad is not None
    bad_nodes=[*nodes]
    bad_nodes[2]=_node("branch","conv2d",{"out_channels":4,"kernel_size":3,"padding":1})
    with pytest.raises(BackendError,match="GRAPH_ADD_INPUT_SHAPE_MISMATCH"):
        GraphModel({"nodes":bad_nodes,"edges":edges},[3,8,8])
