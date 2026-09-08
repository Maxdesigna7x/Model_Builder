from __future__ import annotations

import pytest
import torch

from modelbuilder.catalog import validate_compatibility
from modelbuilder.data import generate
from modelbuilder.engine import infer
from modelbuilder.models import build_model
from modelbuilder.storage import create_project
from modelbuilder.training import train


def node(node_id, block_type, **properties):
    return {"id": node_id, "data": {"blockType": block_type, "label": node_id, "properties": properties}}


def sequential(*nodes):
    return {"nodes": list(nodes), "edges": [{"id": f"{a['id']}-{b['id']}", "source": a["id"], "target": b["id"]} for a, b in zip(nodes, nodes[1:])]}


CASES = [
    ("cnn1d", "sequence.classification", [48, 3], [3], sequential(node("in", "input"), node("layout", "permute", dims="2,1"), node("conv", "conv1d", out_channels=8, kernel_size=3, padding=1), node("pool", "adaptiveavgpool1d", output_size=1), node("flat", "flatten"), node("head", "linear", out_features=3), node("out", "output"))),
    ("lstm", "sequence.regression", [48, 3], [1], sequential(node("in", "input"), node("gru", "gru", hidden_size=8), node("last", "temporal_select"), node("head", "linear", out_features=1), node("out", "output"))),
    ("lstm", "sequence.classification", [48, 3], [3], sequential(node("in", "input"), node("rnn", "rnn", hidden_size=8), node("last", "temporal_select"), node("head", "linear", out_features=3), node("out", "output"))),
    ("transformer", "sequence.forecast", [48, 3], [8, 1], sequential(node("in", "input"), node("project", "sequence_projection", d_model=16), node("pos", "positional_encoding", max_length=64), node("encoder", "transformer_encoder", heads=4, num_layers=1, dim_feedforward=32, dropout=0), node("pool", "sequence_pool", mode="last"), node("head", "linear", out_features=8), node("reshape", "reshape", shape="8,1"), node("out", "output"))),
    ("vit", "image.classification", [3, 32, 32], [3], sequential(node("in", "input"), node("patch", "patch_embedding", patch_size=8, d_model=16), node("pos", "positional_encoding", max_length=32, learned=True), node("encoder", "transformer_encoder", heads=4, num_layers=1, dim_feedforward=32, dropout=0), node("pool", "sequence_pool", mode="mean"), node("head", "linear", out_features=3), node("out", "output"))),
    ("autoencoder", "tabular.reconstruction", [12], [12], sequential(node("in", "input"), node("latent", "linear", out_features=4), node("head", "linear", out_features=12), node("out", "output"))),
    ("autoencoder", "image.reconstruction", [3, 32, 32], [3, 32, 32], sequential(node("in", "input"), node("enc", "conv2d", out_channels=8, kernel_size=4, stride=2, padding=1), node("head", "convtranspose2d", out_channels=3, kernel_size=4, stride=2, padding=1), node("out", "output"))),
    ("transformer", "text.classification", [16], [3], sequential(node("in", "input"), node("embed", "embedding", vocab_size=32, d_model=16), node("pos", "positional_encoding", max_length=16), node("encoder", "transformer_encoder", heads=4, num_layers=1, dim_feedforward=32, dropout=0), node("pool", "sequence_pool", mode="mean"), node("head", "linear", out_features=3), node("out", "output"))),
    ("transformer_causal", "text.language_model", [16], [16, 32], sequential(node("in", "input"), node("embed", "embedding", vocab_size=32, d_model=16), node("pos", "positional_encoding", max_length=16), node("decoder", "causal_transformer", heads=4, num_layers=1, dim_feedforward=32, dropout=0), node("head", "linear", out_features=32), node("out", "output"))),
]


@pytest.mark.parametrize("architecture,task,input_shape,output_shape,graph", CASES)
def test_new_architecture_dry_run_and_backward(architecture, task, input_shape, output_shape, graph):
    model = build_model(architecture, input_shape, output_shape, graph, task)
    sample = torch.randint(0, 16, (2, *input_shape)) if task.startswith("text.") else torch.randn(2, *input_shape)
    output = model(sample)
    assert list(output.shape[1:]) == output_shape
    output.float().square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_architecture_compatibility_is_enforced():
    validate_compatibility("image.classification", "vit")
    with pytest.raises(ValueError, match="no admite"):
        validate_compatibility("image.classification", "lstm")


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELBUILDER_WORKSPACE", str(tmp_path))
    value = {"id": "new-architectures", "name": "Nuevas", "description": "", "path": "projects/new"}
    create_project(value)
    return value


@pytest.mark.parametrize("task,options", [
    ("tabular.reconstruction", {}),
    ("image.reconstruction", {"resolution": 32, "channels": 3}),
    ("text.classification", {"max_length": 16, "vocab_size": 32}),
    ("text.language_model", {"max_length": 16, "vocab_size": 32}),
])
def test_new_tasks_generate_complete_datasets(project, task, options):
    summary = generate(project, task, options=options)
    payload = torch.load(summary["path"], weights_only=True)
    assert len(payload["inputs"]) == summary["samples"]
    assert payload["targets"].shape[0] == summary["samples"]
    assert summary["preview"]["items"]


def test_text_transformer_trains_and_accepts_manual_text(project):
    dataset=generate(project,"text.classification",options={"max_length":16,"vocab_size":32})
    graph=CASES[7][4]
    result=train({"project":project,"dataset":dataset,"task_id":"text.classification","architecture":"transformer","graph":graph,"config":{"epochs":1,"batch_size":64,"device":"cpu"}})
    prediction=infer({"project":project,"checkpoint":result["checkpoint"],"mode":"manual","values":"modelo datos aprende bien"})
    assert prediction["source"] == "Texto tokenizado manualmente"
    assert prediction["prediction"] in dataset["classes"]
