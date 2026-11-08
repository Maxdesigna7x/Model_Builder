from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import torch
from PIL import Image

from modelbuilder.data import apply_pipeline, generate, import_dataset, split_indices
from modelbuilder.engine import apply_inference_pipeline, generate_causal_tokens, handle, infer
from modelbuilder.errors import BackendError
from modelbuilder.huggingface_datasets import DATASETS as HF_DATASETS, LOADERS as HF_LOADERS, catalog as hf_catalog, download as hf_download
from modelbuilder.models import apply_stage_policy, build_model
from modelbuilder.manifest import comparison_hash, semantic_hash
from modelbuilder.pretrained import PRETRAINED_MODELS, catalog as model_catalog
from modelbuilder.storage import create_project
from modelbuilder.training import default_epochs, redact_secrets, seed_everything, train, wait_for_training_control


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
    with pytest.raises(BackendError,match="PIPELINE_RANDOM_FLIP_REGRESSION_TARGET"):
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
    assert "macro_f1" in result["test"]["secondaryMetrics"]
    assert len(result["test"]["diagnostics"]["confusion_matrix"])==3
    assert result["comparisonContract"]==comparison_hash(task_id,dataset,"accuracy")
    assert Path(result["checkpoint"]).is_file()
    assert (Path(result["bundle"])/"model.pt").is_file() and (Path(result["bundle"])/"MODEL_CARD.md").is_file()
    registry=json.loads((Path(result["bundle"]).parents[2]/"models"/"registry.json").read_text())
    assert registry["models"][-1]["aliases"]==["best","last"]
    imported=handle({"action":"model.bundle.import","path":result["bundle"]})
    assert imported["task_id"]==task_id and imported["checkpoint"].endswith("model.pt")
    manifest=json.loads((Path(result["checkpoint"]).parent/"run.json").read_text())
    assert manifest["dataset"]["splits"] and manifest["code"]["package"]=="modelbuilder"
    prediction = infer({"project": project, "mode": "manual", "values": ",".join(["0"] * 12)})
    assert len(prediction["probabilities"]) == 3
    assert abs(sum(prediction["probabilities"]) - 1) < 1e-5

    # Test inference must use the concrete saved indices, rather than creating
    # another percentage-based split that could cross an official boundary.
    payload=torch.load(dataset["path"],weights_only=True)
    payload["splits"]={"train":list(range(2,len(payload["inputs"]))),"validation":[0],"test":[1]}
    torch.save(payload,dataset["path"])
    sampled=infer({"project":project,"checkpoint":result["checkpoint"],"mode":"test"})
    assert sampled["source"]=="Muestra #1 (Split Test)"


def test_advanced_training_and_safe_resume(project, monkeypatch):
    task_id="tabular.regression";dataset=generate(project,task_id,seed=23);graph=graph_for(task_id)
    events=[]
    monkeypatch.setattr("modelbuilder.training.emit",events.append)
    first=train({"project":project,"dataset":dataset,"task_id":task_id,"architecture":"mlp","graph":graph,"config":{"epochs":1,"batch_size":128,"device":"cpu","seed":23,"scheduler":"cosine","warmup_epochs":1,"gradient_accumulation":2,"gradient_clip_norm":1.0,"loss":"huber"}})
    saved=torch.load(first["lastCheckpoint"],weights_only=True)
    assert saved["schema_version"]==3 and saved["scheduler_state"] and saved["rng_state"]["numpy"]["state"].dtype==torch.uint32
    first_point=first["history"][0]
    assert torch.isfinite(torch.tensor(first_point["trainMetric"])) and torch.isfinite(torch.tensor(first_point["valMetric"]))
    assert first_point["metric"]==first_point["valMetric"] and first_point["trainMetric"]!=first_point["valMetric"]
    metric_event=next(event for event in events if event["type"]=="metric")
    assert metric_event["train_metric"]==first_point["trainMetric"] and metric_event["val_metric"]==first_point["valMetric"]
    assert saved["history"]==first["history"]
    resumed=train({"project":project,"dataset":dataset,"task_id":task_id,"architecture":"mlp","graph":graph,"config":{"epochs":2,"batch_size":128,"device":"cpu","seed":23,"scheduler":"cosine","warmup_epochs":1,"gradient_accumulation":2,"loss":"huber","resume_checkpoint":first["lastCheckpoint"]}})
    assert [point["epoch"] for point in resumed["history"]]==[1,2]
    assert all("trainMetric" in point and point["valMetric"]==point["metric"] for point in resumed["history"])
    resumed_state=torch.load(resumed["lastCheckpoint"],weights_only=True)
    assert resumed_state["history"]==resumed["history"]


def test_class_balancing_supports_weights_and_sampler(project):
    task_id="tabular.classification";dataset=generate(project,task_id,seed=31);graph=graph_for(task_id)
    result=train({"project":project,"dataset":dataset,"task_id":task_id,"architecture":"mlp","graph":graph,"config":{"epochs":1,"batch_size":256,"device":"cpu","seed":31,"class_weights":True,"balanced_sampler":True,"finetuning_recipe":"smoke"}})
    assert result["status"]=="completed" and len(result["history"])==1


def test_task_aware_mvp_epoch_defaults():
    assert default_epochs("text.language_model") == 10
    assert default_epochs("image.segmentation.binary") == 40
    assert default_epochs("image.segmentation.multiclass") == 40
    assert default_epochs("image.classification") == 20


def test_system_gpu_never_invents_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    result=handle({"action":"system.gpu"})
    assert result=={"cuda":False,"name":"CPU","total_vram":0,"allocated_vram":0}


def test_authoritative_catalog_contract():
    result=handle({"action":"catalog.get"})
    assert result["schema_version"]==1 and result["architectures"]["cnn"]["tasks"]==["image.classification","image.regression","image.localization","image.detection","image.translation"]
    assert {"hf_vision_backbone","linear","concat"}.issubset(result["block_types"])


def test_graph_validation_runs_backward(project):
    dataset=generate(project,"tabular.classification",seed=3)
    result=handle({"action":"graph.validate","project":project,"dataset":dataset,"task_id":"tabular.classification","architecture":"mlp","graph":graph_for("tabular.classification")})
    assert result["valid"] and result["backward"] and result["parameters"]>0


def test_seed_everything_covers_python_numpy_and_torch():
    import random
    import numpy as np
    seed_everything(19); first=(random.random(),float(np.random.rand()),float(torch.rand(1)))
    seed_everything(19); second=(random.random(),float(np.random.rand()),float(torch.rand(1)))
    assert first==second


def test_secret_values_are_redacted_from_manifests():
    assert redact_secrets({"hf_token":"abc","nested":{"password":"def"},"safe":2})=={"hf_token":"[REDACTED]","nested":{"password":"[REDACTED]"},"safe":2}


def test_training_control_pauses_resumes_and_cancels(tmp_path):
    control=tmp_path/"control.json";control.write_text('{"state":"paused"}')
    events=[]
    def resume(): time.sleep(.05);control.write_text('{"state":"running"}')
    thread=threading.Thread(target=resume);thread.start()
    assert wait_for_training_control(tmp_path,events.append);thread.join()
    assert [event["type"] for event in events]==["paused","resumed"]
    control.write_text('{"state":"cancelled"}')
    assert not wait_for_training_control(tmp_path,events.append)


def test_training_can_be_cancelled_while_paused(tmp_path):
    control=tmp_path/"control.json";control.write_text('{"state":"paused"}')
    events=[]
    def cancel(): time.sleep(.05);control.write_text('{"state":"cancelled"}')
    thread=threading.Thread(target=cancel);thread.start()
    assert not wait_for_training_control(tmp_path,events.append)
    thread.join()
    assert [event["type"] for event in events]==["paused"]


def test_stage_policy_unfreezes_only_last_requested_stage():
    backbone=torch.nn.Module();backbone.layers=torch.nn.ModuleList([torch.nn.Linear(3,3) for _ in range(4)])
    apply_stage_policy(backbone,{"frozen":True,"trainable_stages":1})
    assert not any(parameter.requires_grad for layer in backbone.layers[:3] for parameter in layer.parameters())
    assert all(parameter.requires_grad for parameter in backbone.layers[3].parameters())


def test_pretrained_catalog_is_curated_pinned_and_safe():
    models=model_catalog("image.classification","cnn")
    assert {item["id"] for item in models}=={"resnet-18","mobilenet-v2","convnext-tiny"}
    assert all(len(item["revision"])==40 and not item["trustRemoteCode"] for item in models)
    assert {item["adapter"] for item in PRETRAINED_MODELS.values()} >= {"vision_backbone","text_backbone","causal_lm"}
    assert handle({"action":"model.catalog.guidance","architecture":"mlp"})["reason"]
    assert all(item["weightsFormat"]=="safetensors" and item["diskFreeBytes"]>0 for item in models)


def test_semantic_hash_ignores_canvas_position_and_labels():
    graph=graph_for("tabular.classification");copy=json.loads(json.dumps(graph))
    copy["nodes"][0]["position"]={"x":999,"y":123};copy["nodes"][0]["data"]["label"]="Otro nombre"
    assert semantic_hash(graph)==semantic_hash(copy)
    copy["nodes"][0]["data"].update({"groupId":"visual-only","groupName":"Encoder"})
    assert semantic_hash(graph)==semantic_hash(copy)
    copy["nodes"][1]["data"]["properties"]["out_features"]=99
    assert semantic_hash(graph)!=semantic_hash(copy)


def test_invalid_disconnected_and_wrong_output():
    graph = sequential(node("input", "input"), node("head", "linear", out_features=2), node("output", "output"))
    graph["nodes"].append(node("orphan", "relu"))
    with pytest.raises(BackendError, match="GRAPH_DISCONNECTED_BLOCKS"):
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


def test_imported_text_preserves_raw_inputs_for_pretrained_tokenizers(project,tmp_path):
    source=tmp_path/"text.txt";source.write_text("\n".join(f"clase{i%2}\tTexto original número {i}" for i in range(12)),encoding="utf-8")
    dataset=import_dataset(project,"text.classification",str(source),{"max_length":8})
    payload=torch.load(dataset["path"],weights_only=True)
    assert payload["raw_inputs"][0]=="Texto original número 0" and len(payload["raw_inputs"])==12


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


def test_every_task_has_a_curated_huggingface_dataset(project):
    expected = {
        "tabular.classification", "tabular.regression", "tabular.reconstruction",
        "image.classification", "image.regression", "image.reconstruction",
        "sequence.classification", "sequence.regression", "sequence.forecast",
        "text.classification", "text.language_model",
        "image.segmentation.binary", "image.segmentation.multiclass",
    }
    covered = {task for spec in HF_DATASETS.values() for task in spec["tasks"]}
    assert expected <= covered
    for task in expected:
        assert hf_catalog(project, task)


def test_huggingface_download_reuses_physical_project_dataset(project, monkeypatch):
    calls = []

    def fake_loader(spec, task_id, options, offline):
        calls.append(offline)
        x = torch.arange(60, dtype=torch.float32).reshape(15, 4)
        y = torch.arange(15).remainder(3)
        return x, y, ["a", "b", "c"]

    monkeypatch.setitem(HF_LOADERS, "iris", fake_loader)
    first = hf_download(project, "tabular.classification", "iris")
    second = hf_download(project, "tabular.classification", "iris")

    assert calls == [False]
    assert first["source"] == "huggingface"
    assert second["id"] == first["id"] and second["cacheStatus"] == "project"
    assert Path(second["path"]).is_file()
    listed = hf_catalog(project, "tabular.classification")[0]
    assert listed["installed"] is True and listed["cached"] is True

    # A manifest alone is not enough: if the tensor disappeared physically,
    # materialize it again from the shared Hugging Face cache.
    Path(second["path"]).unlink()
    third = hf_download(project, "tabular.classification", "iris")
    assert calls == [False, True]
    assert Path(third["path"]).is_file() and third["id"] != first["id"]


def test_huggingface_download_preserves_official_splits(project, monkeypatch):
    def fake_loader(spec, task_id, options, offline):
        x=torch.arange(240,dtype=torch.long).reshape(15,16).remainder(31)
        y=x.roll(-1,dims=1);y[:,-1]=0
        official={"train":list(range(10)),"validation":[10,11],"test":[12,13,14]}
        return x,y,[f"t{i}" for i in range(32)],official

    monkeypatch.setitem(HF_LOADERS,"wikitext",fake_loader)
    result=hf_download(project,"text.language_model","wikitext-2",{"max_length":16,"vocab_size":32,"max_samples":15})
    payload=torch.load(result["path"],weights_only=True)
    assert result["splitSource"]=="official"
    assert result["splitCounts"]=={"train":10,"validation":2,"test":3}
    assert result["splits"]=={"train":67,"validation":13,"test":20}
    assert payload["splits"]=={"train":list(range(10)),"validation":[10,11],"test":[12,13,14]}
    revised=apply_pipeline(project,result,"text.language_model",{"schemaVersion":1,"revision":1,"nodes":[
        {"id":"source","type":"source","category":"source","enabled":True},
        {"id":"validate","type":"validate","category":"inspect","enabled":True,"properties":{"fail_on_error":True}},
        {"id":"output","type":"output","category":"output","enabled":True,"properties":{"batchSize":8}},
    ]})
    revised_payload=torch.load(revised["path"],weights_only=True)
    assert revised["splitSource"]=="official"
    assert revised["splitCounts"]==result["splitCounts"]
    assert revised_payload["splits"]==payload["splits"]


def test_causal_generation_feeds_each_prediction_back_into_context():
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()));self.seen=[]
        def forward(self,x):
            length=int((x[0]!=0).sum());self.seen.append(x[0,:length].tolist())
            logits=torch.zeros(1,x.shape[1],8);logits[0,length-1,min(7,length+2)]=1
            return logits

    model=CountingModel()
    generated=generate_causal_tokens(model,torch.tensor([[2,0,0,0]]),4,3)
    assert generated==[3,4,5]
    assert model.seen==[[2],[2,3],[2,3,4]]
