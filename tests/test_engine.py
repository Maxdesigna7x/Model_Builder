from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from modelbuilder.data import apply_pipeline, generate, import_dataset, split_indices
from modelbuilder.engine import apply_inference_pipeline, infer
from modelbuilder.models import build_model
from modelbuilder.storage import create_project
from modelbuilder.training import train


def node(node_id, block_type, **properties):
    return {"id": node_id, "data": {"blockType": block_type, "label": node_id, "properties": properties}}


def sequential(*nodes):
    return {"nodes": list(nodes), "edges": [{"id": f"{a['id']}-{b['id']}", "source": a["id"], "target": b["id"]} for a, b in zip(nodes, nodes[1:])]}


def graph_for(task_id):
    if task_id.startswith("tabular"):
        outputs = 3 if "classification" in task_id else 1
        return sequential(node("input", "input"), node("hidden", "linear", out_features=32), node("relu", "relu"), node("head", "linear", out_features=outputs), node("output", "output"))
    if task_id.startswith("image") and "segmentation" not in task_id:
        outputs = 3 if "classification" in task_id else 1
        return sequential(node("input", "input"), node("conv", "conv2d", out_channels=8, kernel_size=3, padding=1), node("relu", "relu"), node("pool", "adaptiveavgpool2d", output_size=1), node("flat", "flatten"), node("head", "linear", out_features=outputs), node("output", "output"))
    if task_id.startswith("sequence"):
        outputs = 3 if "classification" in task_id else (8 if task_id.endswith("forecast") else 1)
        parts = [node("input", "input"), node("lstm", "lstm", hidden_size=16, num_layers=1), node("last", "temporal_select"), node("head", "linear", out_features=outputs)]
        if task_id.endswith("forecast"): parts.append(node("reshape", "reshape", shape="8,1"))
        parts.append(node("output", "output")); return sequential(*parts)
    outputs = 1 if task_id.endswith("binary") else 4
    nodes = [node("input", "input"), node("enc", "conv2d", out_channels=8, kernel_size=3, padding=1), node("pool", "maxpool2d", kernel_size=2, stride=2), node("mid", "conv2d", out_channels=16, kernel_size=3, padding=1), node("up", "convtranspose2d", out_channels=8, kernel_size=2, stride=2), node("cat", "concat", axis=1), node("dec", "conv2d", out_channels=8, kernel_size=3, padding=1), node("head", "conv2d", out_channels=outputs, kernel_size=1, padding=0), node("output", "output")]
    edges = [("input","enc"),("enc","pool"),("pool","mid"),("mid","up"),("up","cat"),("enc","cat"),("cat","dec"),("dec","head"),("head","output")]
    return {"nodes":nodes,"edges":[{"id":f"{a}-{b}","source":a,"target":b} for a,b in edges]}


TASKS = ["tabular.classification", "tabular.regression", "image.classification", "image.regression", "sequence.classification", "sequence.regression", "sequence.forecast", "image.segmentation.binary", "image.segmentation.multiclass"]


def test_custom_split_percentages_are_applied():
    splits = split_indices(100, 42, {"train": 60, "validation": 25, "test": 15})
    assert {name: len(indices) for name, indices in splits.items()} == {"train": 60, "validation": 25, "test": 15}
    assert len({*splits["train"], *splits["validation"], *splits["test"]}) == 100


def test_pipeline_materializes_revision_with_train_fitted_state(project):
    dataset=generate(project,"tabular.regression",seed=17)
    pipeline={"schemaVersion":1,"revision":1,"nodes":[
        {"id":"source","type":"source","category":"source","scope":"all","enabled":True},
        {"id":"quality","type":"validate","category":"inspect","scope":"all","enabled":True,"properties":{"fail_on_error":True}},
        {"id":"split","type":"split","category":"split","scope":"all","enabled":True,"properties":{"train":60,"validation":20,"test":20,"seed":17,"strategy":"random"}},
        {"id":"scale","type":"normalize","label":"Normalizar","category":"transform","scope":"all","enabled":True,"properties":{"mode":"standard"}},
        {"id":"noise","type":"random_noise","category":"augment","scope":"train","enabled":True,"properties":{"std":.02}},
        {"id":"output","type":"output","category":"output","scope":"all","enabled":True},
    ]}
    revised=apply_pipeline(project,dataset,"tabular.regression",pipeline)
    assert revised["revision"]==2 and revised["pipeline"]["fittedState"]["scale"]["mode"]=="standard"
    payload=torch.load(revised["path"],weights_only=True)
    train=payload["inputs"][payload["splits"]["train"]]
    assert torch.allclose(train.mean(0),torch.zeros(train.shape[1]),atol=1e-5)
    assert payload["pipeline"]["nodes"][-2]["scope"]=="train"
    manual=torch.ones(1,12)
    transformed=apply_inference_pipeline(manual,revised,"tabular.regression")
    state=revised["pipeline"]["fittedState"]["scale"]
    expected=(manual-torch.tensor(state["mean"]).reshape(1,-1))/torch.tensor(state["scale"]).reshape(1,-1)
    assert torch.allclose(transformed,expected)


def test_pipeline_rejects_unsafe_flip_for_visual_regression(project):
    dataset=generate(project,"image.regression",seed=3)
    pipeline={"revision":1,"nodes":[
        {"id":"split","type":"split","category":"split","scope":"all","enabled":True,"properties":{"train":70,"validation":15,"test":15}},
        {"id":"flip","type":"random_flip","category":"augment","scope":"train","enabled":True,"properties":{}},
    ]}
    with pytest.raises(ValueError,match="no es seguro"):
        apply_pipeline(project,dataset,"image.regression",pipeline)


def test_image_pipeline_resizes_without_a_split_node(project):
    dataset=generate(project,"image.classification",seed=5,options={"resolution":64})
    pipeline={"schemaVersion":1,"revision":1,"nodes":[
        {"id":"source","type":"source","category":"source","scope":"all","enabled":True,"properties":{}},
        {"id":"resize","type":"resize","label":"Redimensionar","category":"transform","scope":"all","enabled":True,"properties":{"width":40,"height":32,"method":"bilinear"}},
        {"id":"output","type":"output","category":"output","scope":"all","enabled":True,"properties":{"batchSize":16,"workers":0,"pin_memory":True}},
    ]}
    revised=apply_pipeline(project,dataset,"image.classification",pipeline)
    payload=torch.load(revised["path"],weights_only=True)
    assert revised["inputShape"]==[3,32,40]
    assert tuple(payload["inputs"].shape[1:])==(3,32,40)
    assert revised["options"]["batchSize"]==16
    assert set(payload["splits"])=={"train","validation","test"}


@pytest.mark.parametrize(
    ("task_id", "options", "expected"),
    [
        ("image.classification", {"resolution": 32}, {"classDistribution", "intensityHistogram", "channelStats"}),
        ("tabular.regression", {}, {"targetHistogram", "featureHistograms", "featureSummary", "correlation"}),
        ("sequence.classification", {}, {"classDistribution", "featureHistograms", "meanSeries"}),
        ("text.classification", {"max_length": 16, "vocab_size": 32}, {"classDistribution", "lengthHistogram", "tokenFrequency"}),
    ],
)
def test_dataset_analytics_are_specific_to_each_modality(project, task_id, options, expected):
    dataset = generate(project, task_id, seed=11, options=options)
    analytics = dataset["analytics"]
    assert analytics["modality"] == task_id.split(".")[0]
    assert expected.issubset(analytics)
    for key in expected:
        value = analytics[key]
        if "series" in value:
            assert value["series"] and value["labels"]
        elif isinstance(value, list):
            assert value
        else:
            assert value["labels"]
            assert value.get("values") or value.get("mean")


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELBUILDER_WORKSPACE", str(tmp_path))
    value = {"id": "test-project", "name": "Prueba", "description": "", "path": "projects/test"}
    create_project(value)
    return value


@pytest.mark.parametrize("task_id", TASKS)
def test_synthetic_contract_and_graph_forward(project, task_id):
    architecture = {"tabular": "mlp", "image": "unet" if "segmentation" in task_id else "cnn", "sequence": "lstm"}[task_id.split(".")[0]]
    dataset = generate(project, task_id, seed=7)
    payload = torch.load(dataset["path"], weights_only=True)
    model = build_model(architecture, dataset["inputShape"], dataset["outputShape"], graph_for(task_id), task_id)
    result = model(payload["inputs"][:2])
    assert list(result.shape[1:]) == dataset["outputShape"]
    assert torch.isfinite(result).all()


def test_train_checkpoint_and_inference(project):
    task_id = "tabular.classification"; dataset = generate(project, task_id, seed=9); graph = graph_for(task_id)
    request = {"project": project, "dataset": dataset, "task_id": task_id, "architecture": "mlp", "graph": graph, "config": {"epochs": 2, "batch_size": 64, "device": "cpu", "seed": 9}}
    result = train(request)
    assert result["status"] == "completed"
    assert len(result["history"]) == 2
    assert Path(result["checkpoint"]).is_file()
    prediction = infer({"project": project, "values": ",".join(["0"] * 12)})
    assert len(prediction["probabilities"]) == 3
    assert abs(sum(prediction["probabilities"]) - 1) < 1e-5


def test_invalid_disconnected_and_wrong_output():
    graph = sequential(node("input", "input"), node("head", "linear", out_features=2), node("output", "output"))
    graph["nodes"].append(node("orphan", "relu"))
    with pytest.raises(ValueError, match="desconectados"):
        build_model("mlp", [4], [2], graph, "tabular.classification")


def test_project_manifest_is_valid_json(project, tmp_path):
    manifest = json.loads((tmp_path / "projects" / "test-project" / "project.json").read_text())
    assert manifest["schema_version"] == 1


def test_import_tabular_and_sequence_csv(project, tmp_path):
    table = tmp_path / "table.csv"
    table.write_text("a,b,target\n" + "\n".join(f"{i},{i/2},{i%3}" for i in range(18)), encoding="utf-8")
    tabular = import_dataset(project, "tabular.classification", str(table))
    assert tabular["source"] == "imported" and tabular["inputShape"] == [2]

    sequence = tmp_path / "sequence.csv"
    rows = ["sequence_id,timestep,f1,target"]
    for sequence_id in range(12):
        for timestep in range(4): rows.append(f"{sequence_id},{timestep},{sequence_id+timestep},{sequence_id%2}")
    sequence.write_text("\n".join(rows), encoding="utf-8")
    temporal = import_dataset(project, "sequence.classification", str(sequence))
    assert temporal["inputShape"] == [4, 1] and temporal["classes"] == ["0", "1"]


def test_import_image_classification_and_segmentation(project, tmp_path):
    classified = tmp_path / "classified"
    for class_name, color in (("red", (220, 30, 40)), ("blue", (30, 60, 220))):
        folder = classified / class_name; folder.mkdir(parents=True)
        for index in range(6): Image.new("RGB", (20, 20), color).save(folder / f"{index}.png")
    image_data = import_dataset(project, "image.classification", str(classified))
    assert image_data["samples"] == 12 and image_data["classes"] == ["blue", "red"]

    segmented = tmp_path / "segmented"; (segmented / "images").mkdir(parents=True); (segmented / "masks").mkdir()
    for index in range(10):
        Image.new("RGB", (20, 20), (30, 40, 50)).save(segmented / "images" / f"{index}.png")
        mask = Image.new("L", (20, 20), 0)
        for x in range(5, 15):
            for y in range(5, 15): mask.putpixel((x, y), 255)
        mask.save(segmented / "masks" / f"{index}.png")
    segmentation = import_dataset(project, "image.segmentation.binary", str(segmented))
    assert segmentation["outputShape"] == [1, 64, 64]
