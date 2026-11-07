from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest
import torch
from PIL import Image

from modelbuilder.data import apply_pipeline, generate, import_dataset
from modelbuilder.engine import handle, infer
from modelbuilder.errors import BackendError
from modelbuilder.models import GraphModel, build_model
from modelbuilder.storage import create_project
from modelbuilder.training import AugmentedTensorDataset, _loss, decode_detections, decode_localization, detection_map50, train


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELBUILDER_WORKSPACE", str(tmp_path))
    value={"id":"cnn-vision","name":"CNN vision","description":"","path":"projects/cnn-vision"}
    create_project(value)
    return value


def node(node_id,kind,**properties):
    return {"id":node_id,"data":{"blockType":kind,"label":node_id,"properties":properties}}


def detection_graph(classes=3):
    nodes=[node("input","input"),node("stem","conv2d",out_channels=8,kernel_size=3,padding=1),node("pool1","maxpool2d",kernel_size=2,stride=2),node("features","conv2d",out_channels=12,kernel_size=3,padding=1),node("pool2","maxpool2d",kernel_size=2,stride=2),node("pool3","maxpool2d",kernel_size=2,stride=2),node("box_head","conv2d",out_channels=4,kernel_size=1,padding=0),node("class_head","conv2d",out_channels=classes,kernel_size=1,padding=0),node("boxes","output",output_name="boxes"),node("classes","output",output_name="classes")]
    edges=[("input","stem"),("stem","pool1"),("pool1","features"),("features","pool2"),("pool2","pool3"),("pool3","box_head"),("pool3","class_head"),("box_head","boxes"),("class_head","classes")]
    return {"nodes":nodes,"edges":[{"source":a,"target":b} for a,b in edges]}


def localization_graph():
    nodes=[node("input","input"),node("conv","conv2d",out_channels=8,kernel_size=3,padding=1),node("pool","adaptiveavgpool2d",output_size=1),node("flat","flatten"),node("head","linear",out_features=4),node("output","output")]
    return {"nodes":nodes,"edges":[{"source":a,"target":b} for a,b in zip(["input","conv","pool","flat","head"],["conv","pool","flat","head","output"])]}


def test_detector_has_two_named_heads_and_gradients_from_distinct_targets():
    model=build_model("cnn",[3,64,64],[3],detection_graph(),"image.detection")
    prediction=model(torch.rand(2,3,64,64));assert prediction["boxes"].shape==(2,4,8,8);assert prediction["classes"].shape==(2,3,8,8)
    targets=[torch.tensor([[1,.35,.65,.2,.3]]),torch.empty((0,5))]
    loss=_loss("image.detection",prediction,targets);loss.backward()
    assert torch.isfinite(loss) and model.operations["box_head"].weight.grad is not None and model.operations["class_head"].weight.grad is not None


def test_detector_rejects_bad_head_contract_and_cell_collisions():
    wrong=detection_graph();wrong["nodes"][-1]["data"]["properties"]["output_name"]="labels"
    with pytest.raises(BackendError,match="GRAPH_DETECTION_OUTPUT_NAMES_REQUIRED"):
        build_model("cnn",[3,32,32],[3],wrong,"image.detection")
    output={"boxes":torch.zeros(1,4,2,2),"classes":torch.zeros(1,2,2,2)}
    target=[torch.tensor([[0,.2,.2,.1,.1],[1,.21,.22,.1,.1]])]
    with pytest.raises(BackendError,match="TRAINING_DETECTION_CELL_COLLISION"):
        _loss("image.detection",output,target)


def test_detection_decoder_and_map50_known_match():
    output={"boxes":torch.full((1,4,1,1),-8.),"classes":torch.tensor([[[[8.]], [[-8.]]]])}
    output["boxes"][0,:,0,0]=torch.logit(torch.tensor([.5,.5,.4,.4]))
    target=[[torch.tensor([[0,.5,.5,.4,.4]])]]
    decoded=decode_detections(output,.5)[0]
    assert decoded.shape==(1,6) and decoded[0,0].item()==0
    metrics=detection_map50([output],target,2)
    assert metrics["map50"]==pytest.approx(1.) and metrics["precision"]==1 and metrics["recall"]==1


def test_localization_generation_forward_and_training_inference(project):
    dataset=generate(project,"image.localization",seed=7,options={"resolution":32,"channels":1})
    payload=torch.load(dataset["path"],weights_only=True)
    assert dataset["outputShape"]==[4] and payload["targets"].shape==(120,4)
    model=build_model("cnn",dataset["inputShape"],dataset["outputShape"],localization_graph(),"image.localization")
    assert model(payload["inputs"][:2]).shape==(2,4)
    ordered=decode_localization(torch.tensor([[4.,-3.,-5.,2.]]))
    assert (ordered[:,2:]>=ordered[:,:2]).all() and ((ordered>=0)&(ordered<=1)).all()


def test_yolo_import_preserves_variable_boxes_and_crop_geometry(project,tmp_path):
    root=tmp_path/"detect";(root/"images").mkdir(parents=True);(root/"labels").mkdir();(root/"classes.txt").write_text("object\n",encoding="utf-8")
    for index in range(4):
        Image.new("RGB",(100,100),(0,0,0)).save(root/"images"/f"{index}.png")
        if index==0:(root/"labels"/f"{index}.txt").write_text("0 0.2 0.2 0.2 0.2\n",encoding="utf-8")
        elif index==1:(root/"labels"/f"{index}.txt").write_text("0 0.7 0.7 0.2 0.2\n",encoding="utf-8")
        else:(root/"labels"/f"{index}.txt").write_text("",encoding="utf-8")
    dataset=import_dataset(project,"image.detection",str(root),{"resolution":100,"channels":3})
    payload=torch.load(dataset["path"],weights_only=True)
    assert [len(value) for value in payload["boxes"]]==[1,1,0,0]
    assert payload["boxes"][0][0].tolist()==pytest.approx([0,.2,.2,.2,.2])
    pipeline={"schemaVersion":1,"revision":1,"nodes":[{"id":"crop","type":"center_crop","category":"transform","scope":"all","enabled":True,"properties":{"height":80,"width":80}},{"id":"output","type":"output","category":"output","scope":"all","enabled":True}]}
    revised=apply_pipeline(project,dataset,"image.detection",pipeline)
    revised_payload=torch.load(revised["path"],weights_only=True);cropped=revised_payload["boxes"]
    assert cropped[0].shape==(1,5) and cropped[0][0,1:].tolist()==pytest.approx([.125,.125,.25,.25])
    assert revised_payload["original_sizes"][0].tolist()==[100,100]
    assert revised_payload["spatial_affine"][0].tolist()==pytest.approx([.8,.8,.1,.1])


def test_localization_import_and_pair_import_contracts(project,tmp_path):
    local=tmp_path/"local";local.mkdir();rows=[]
    for index in range(4):
        Image.new("RGB",(100,50),(index,0,0)).save(local/f"{index}.png");rows.append({"filename":f"{index}.png","xmin":"10","ymin":"5","xmax":"40","ymax":"25"})
    with (local/"labels.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["filename","xmin","ymin","xmax","ymax"]);writer.writeheader();writer.writerows(rows)
    box_dataset=import_dataset(project,"image.localization",str(local),{"resolution":32,"channels":3})
    box_payload=torch.load(box_dataset["path"],weights_only=True)
    assert box_payload["targets"][0].tolist()==pytest.approx([.1,.1,.4,.5])

    pairs=tmp_path/"pairs";(pairs/"inputs").mkdir(parents=True);(pairs/"targets").mkdir()
    for index in range(4):
        Image.new("L",(24,24),80).save(pairs/"inputs"/f"{index}.png");Image.new("RGB",(24,24),(20,30,40)).save(pairs/"targets"/f"{index}.png")
    pair_dataset=import_dataset(project,"image.translation",str(pairs),{"resolution":16,"channels":1,"target_channels":3})
    pair_payload=torch.load(pair_dataset["path"],weights_only=True)
    assert pair_dataset["inputShape"]==[1,16,16] and pair_dataset["outputShape"]==[3,16,16]
    assert pair_payload["targets"].shape==(4,3,16,16) and pair_payload["targets"].max()<=1
    (pairs/"targets"/"3.png").unlink()
    with pytest.raises(BackendError,match="IMPORT_IMAGE_PAIRS_UNMATCHED"):
        import_dataset(project,"image.translation",str(pairs),{"resolution":16,"channels":1,"target_channels":3})


def test_multi_scale_node_routes_real_maps_to_separate_consumers_in_one_forward():
    nodes=[node("input","input"),node("pyramid","multi_scale_cnn",base_channels=4,stages=3),node("head_p2","conv2d",out_channels=2,kernel_size=1),node("head_p3","conv2d",out_channels=3,kernel_size=1),node("head_p4","conv2d",out_channels=4,kernel_size=1),node("out_p2","output",output_name="p2"),node("out_p3","output",output_name="p3"),node("out_p4","output",output_name="p4")]
    edges=[{"source":"input","target":"pyramid"},{"source":"pyramid","sourceHandle":"p2","target":"head_p2"},{"source":"pyramid","sourceHandle":"p3","target":"head_p3"},{"source":"pyramid","sourceHandle":"p4","target":"head_p4"},{"source":"head_p2","target":"out_p2"},{"source":"head_p3","target":"out_p3"},{"source":"head_p4","target":"out_p4"}]
    model=GraphModel({"nodes":nodes,"edges":edges},[3,32,32])
    operation=model.operations["pyramid"];calls=[];original=operation.forward
    operation.forward=lambda value:(calls.append(1),original(value))[1]
    features=model(torch.rand(2,3,32,32))
    assert calls==[1]
    assert [list(value.shape[1:]) for value in features.values()]==[[2,16,16],[3,8,8],[4,4,4]]


def test_translation_train_augmentation_keeps_target_and_geometry_paired():
    target=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4);inputs=target.clone()
    dataset=AugmentedTensorDataset(inputs,target,[{"type":"random_flip","properties":{"probability":1,"horizontal":True}},{"type":"random_noise","properties":{"side":"input","std":1.0}}],"image.translation")
    augmented_input,augmented_target=dataset[0]
    expected=torch.flip(target[0],(-1,))
    assert torch.equal(augmented_target,expected)
    assert not torch.equal(augmented_input,expected)


def test_image_classification_end_to_end_after_pipeline_and_graph_edit(project):
    dataset=generate(project,"image.classification",seed=41,options={"resolution":24,"channels":1})
    pipeline={"schemaVersion":1,"revision":1,"nodes":[{"id":"crop","type":"center_crop","category":"transform","scope":"all","enabled":True,"properties":{"height":16,"width":16}},{"id":"output","type":"output","category":"output","scope":"all","enabled":True}]}
    dataset=apply_pipeline(project,dataset,"image.classification",pipeline)
    classes=int(dataset["outputShape"][0])
    graph={"nodes":[node("input","input"),node("conv","conv2d",out_channels=4,kernel_size=3,padding=1),node("pool","adaptiveavgpool2d",output_size=1),node("flat","flatten"),node("head","linear",out_features=classes),node("output","output")],"edges":[{"source":a,"target":b} for a,b in zip(["input","conv","pool","flat","head"],["conv","pool","flat","head","output"])]}
    result=train({"project":project,"dataset":dataset,"task_id":"image.classification","architecture":"cnn","graph":graph,"config":{"epochs":1,"batch_size":32,"device":"cpu","finetuning_recipe":"smoke"}})
    prediction=infer({"project":project,"checkpoint":result["checkpoint"],"mode":"test","index":0})
    assert result["status"]=="completed" and Path(result["checkpoint"]).is_file()
    assert prediction["prediction"] in dataset["classes"]


def test_detection_trains_saves_checkpoint_and_infers_boxes(project,tmp_path):
    dataset=generate(project,"image.detection",seed=11,options={"resolution":64,"channels":3})
    request={"project":project,"dataset":dataset,"task_id":"image.detection","architecture":"cnn","graph":detection_graph(3),"config":{"epochs":1,"batch_size":8,"device":"cpu","finetuning_recipe":"smoke"}}
    result=train(request)
    assert result["status"]=="completed" and result["test"]["diagnostics"]["map50"]>=0
    inference=infer({"project":project,"checkpoint":result["checkpoint"],"mode":"test","index":0})
    assert inference["prediction"]=="Detecciones" and inference["outputPreview"].startswith("data:image/png;base64,")
    portable=tmp_path/"portable-model";shutil.copytree(result["bundle"],portable)
    bundle_checkpoint=torch.load(portable/"model.pt",weights_only=True);bundle_checkpoint["dataset"]["path"]=str(tmp_path/"missing-source-dataset.pt");torch.save(bundle_checkpoint,portable/"model.pt")
    imported=handle({"action":"model.bundle.import","path":str(portable)})
    manual_image=tmp_path/"manual.png";Image.new("RGB",(120,60),(35,80,120)).save(manual_image)
    baseline_result=infer({"project":project,"checkpoint":result["checkpoint"],"mode":"manual","values":str(manual_image),"score_threshold":.01})
    portable_result=infer({"project":project,"checkpoint":imported["checkpoint"],"mode":"manual","values":str(manual_image),"score_threshold":.01})
    assert portable_result["imageSize"]=={"width":120,"height":60} and portable_result["outputPreview"].startswith("data:image/png;base64,")
    assert portable_result["detections"]==baseline_result["detections"]
    assert all(item["box"]["format"]=="original_pixel_xyxy" for item in portable_result["detections"])


def test_localization_and_image_pairs_train_to_checkpoint_and_infer(project,tmp_path):
    local=generate(project,"image.localization",seed=12,options={"resolution":16,"channels":1})
    local_request={"project":project,"dataset":local,"task_id":"image.localization","architecture":"cnn","graph":localization_graph(),"config":{"epochs":1,"batch_size":8,"device":"cpu","finetuning_recipe":"smoke"}}
    local_result=train(local_request)
    local_prediction=infer({"project":project,"checkpoint":local_result["checkpoint"],"mode":"test","index":0})
    assert local_result["status"]=="completed" and local_prediction["box"]["format"]=="original_pixel_xyxy"
    manual=tmp_path/"wide.png";Image.new("RGB",(120,60),(120,80,40)).save(manual)
    manual_prediction=infer({"project":project,"checkpoint":local_result["checkpoint"],"mode":"manual","values":str(manual)})
    assert manual_prediction["imageSize"]=={"width":120,"height":60} and manual_prediction["outputPreview"].startswith("data:image/png;base64,")

    paired=generate(project,"image.translation",seed=13,options={"resolution":16,"channels":1})
    graph={"nodes":[node("input","input"),node("first","conv2d",out_channels=4,kernel_size=3,padding=1),node("relu","relu"),node("head","conv2d",out_channels=1,kernel_size=3,padding=1),node("output","output")],"edges":[{"source":a,"target":b} for a,b in zip(["input","first","relu","head"],["first","relu","head","output"])]}
    pair_request={"project":project,"dataset":paired,"task_id":"image.translation","architecture":"cnn","graph":graph,"config":{"epochs":1,"batch_size":8,"device":"cpu","finetuning_recipe":"smoke","loss":"mae"}}
    pair_result=train(pair_request)
    pair_prediction=infer({"project":project,"checkpoint":pair_result["checkpoint"],"mode":"test","index":0})
    assert pair_result["status"]=="completed" and pair_prediction["outputPreview"].startswith("data:image/png;base64,")


def test_initialize_new_run_from_checkpoint_keeps_weights_lineage_and_new_optimizer(project):
    dataset=generate(project,"image.localization",seed=17,options={"resolution":16,"channels":1})
    request={"project":project,"dataset":dataset,"task_id":"image.localization","architecture":"cnn","graph":localization_graph(),"config":{"epochs":1,"batch_size":8,"device":"cpu","finetuning_recipe":"smoke"}}
    parent=train(request)
    child=train({**request,"config":{**request["config"],"initial_checkpoint":parent["checkpoint"]}})
    parent_state=torch.load(parent["checkpoint"],weights_only=True);child_state=torch.load(child["checkpoint"],weights_only=True)
    assert child_state["run_id"]!=parent_state["run_id"]
    assert child_state["initialized_from"]["parent_run_id"]==Path(parent["checkpoint"]).parent.name
    assert child_state["initialized_from"]["loaded_keys"] and child_state["optimizer_state"]["state"]


def test_own_checkpoint_backbone_reuses_residual_ancestors_without_mutating_source(project):
    dataset=generate(project,"image.classification",seed=19,options={"resolution":16,"channels":1})
    classes=int(dataset["outputShape"][0])
    def graph(frozen=False,head="old_head"):
        nodes=[node("input","input"),node("branch_a","conv2d",out_channels=4,kernel_size=3,padding=1,frozen=frozen),node("branch_b","conv2d",out_channels=4,kernel_size=3,padding=1,frozen=frozen),node("sum","add"),node("pool","adaptiveavgpool2d",output_size=1),node("flat","flatten"),node(head,"linear",out_features=classes),node("output","output")]
        edges=[{"source":"input","target":"branch_a"},{"source":"input","target":"branch_b"},{"source":"branch_a","target":"sum","targetHandle":"input_0"},{"source":"branch_b","target":"sum","targetHandle":"input_1"},{"source":"sum","target":"pool"},{"source":"pool","target":"flat"},{"source":"flat","target":head},{"source":head,"target":"output"}]
        return {"nodes":nodes,"edges":edges}
    config={"epochs":1,"batch_size":32,"device":"cpu","seed":73,"finetuning_recipe":"smoke"}
    parent=train({"project":project,"dataset":dataset,"task_id":"image.classification","architecture":"cnn","graph":graph(),"config":config})
    source_before=torch.load(parent["checkpoint"],weights_only=True)["model_state"]
    child=train({"project":project,"dataset":dataset,"task_id":"image.classification","architecture":"cnn","graph":graph(True,"new_head"),"config":{**config,"initial_checkpoint":parent["checkpoint"]}})
    child_state=torch.load(child["lastCheckpoint"],weights_only=True)
    source_after=torch.load(parent["checkpoint"],weights_only=True)["model_state"]
    assert torch.equal(source_before["operations.branch_a.weight"],source_after["operations.branch_a.weight"])
    assert {"operations.branch_a.weight","operations.branch_b.weight"}.issubset(child_state["initialized_from"]["loaded_keys"])
    assert torch.equal(source_before["operations.branch_a.weight"],child_state["model_state"]["operations.branch_a.weight"])
    assert torch.equal(source_before["operations.branch_b.weight"],child_state["model_state"]["operations.branch_b.weight"])
    assert child_state["initialized_from"]["parent_run_id"]==parent["runId"]


def test_finetuning_honors_layer_freezing_and_batchnorm_statistics(project):
    dataset=generate(project,"image.localization",seed=29,options={"resolution":16,"channels":1})
    def graph(frozen,freeze_stats):
        nodes=[node("input","input"),node("conv","conv2d",out_channels=8,kernel_size=3,padding=1,frozen=frozen),node("norm","batchnorm2d",freeze_batchnorm_stats=freeze_stats),node("pool","adaptiveavgpool2d",output_size=1),node("flat","flatten"),node("head","linear",out_features=4),node("output","output")]
        return {"nodes":nodes,"edges":[{"source":a,"target":b} for a,b in zip(["input","conv","norm","pool","flat","head"],["conv","norm","pool","flat","head","output"])]}
    parent_request={"project":project,"dataset":dataset,"task_id":"image.localization","architecture":"cnn","graph":graph(False,False),"config":{"epochs":1,"batch_size":64,"device":"cpu","seed":31}}
    parent=train(parent_request)
    child=train({**parent_request,"graph":graph(True,True),"config":{**parent_request["config"],"initial_checkpoint":parent["checkpoint"]}})
    parent_state=torch.load(parent["lastCheckpoint"],weights_only=True)["model_state"]
    child_state=torch.load(child["lastCheckpoint"],weights_only=True)["model_state"]
    assert torch.equal(parent_state["operations.conv.weight"],child_state["operations.conv.weight"])
    assert torch.equal(parent_state["operations.norm.running_mean"],child_state["operations.norm.running_mean"])
    assert not torch.equal(parent_state["operations.head.weight"],child_state["operations.head.weight"])


@pytest.mark.parametrize("task_id",["image.segmentation.binary","image.segmentation.multiclass"])
def test_unet_segmentation_trains_saves_and_infers_mask(project,task_id):
    dataset=generate(project,task_id,seed=37,options={"resolution":16,"channels":3})
    channels=dataset["outputShape"][0]
    nodes=[node("input","input"),node("enc","conv2d",out_channels=8,kernel_size=3,padding=1),node("pool","maxpool2d",kernel_size=2,stride=2),node("mid","conv2d",out_channels=16,kernel_size=3,padding=1),node("up","convtranspose2d",out_channels=8,kernel_size=2,stride=2),node("join","concat",axis=1),node("dec","conv2d",out_channels=8,kernel_size=3,padding=1),node("head","conv2d",out_channels=channels,kernel_size=1,padding=0),node("output","output")]
    graph={"nodes":nodes,"edges":[{"source":"input","target":"enc"},{"source":"enc","target":"pool"},{"source":"pool","target":"mid"},{"source":"mid","target":"up"},{"source":"up","target":"join","targetHandle":"input_0"},{"source":"enc","target":"join","targetHandle":"input_1"},{"source":"join","target":"dec"},{"source":"dec","target":"head"},{"source":"head","target":"output"}]}
    result=train({"project":project,"dataset":dataset,"task_id":task_id,"architecture":"unet","graph":graph,"config":{"epochs":1,"batch_size":8,"device":"cpu","finetuning_recipe":"smoke"}})
    prediction=infer({"project":project,"checkpoint":result["checkpoint"],"mode":"test","index":0})
    assert result["status"]=="completed" and prediction["shape"]==[16,16]
    assert prediction["outputPreview"].startswith("data:image/png;base64,")


def test_cpu_resume_reproduces_continuous_training_state(project):
    dataset=generate(project,"image.localization",seed=23,options={"resolution":16,"channels":1})
    base={"project":project,"dataset":dataset,"task_id":"image.localization","architecture":"cnn","graph":localization_graph()}
    continuous=train({**base,"config":{"epochs":2,"batch_size":64,"device":"cpu","seed":91}})
    first=train({**base,"config":{"epochs":1,"batch_size":64,"device":"cpu","seed":91}})
    resumed=train({**base,"config":{"epochs":2,"batch_size":64,"device":"cpu","seed":91,"resume_checkpoint":first["lastCheckpoint"]}})
    full_state=torch.load(continuous["lastCheckpoint"],weights_only=True)
    resumed_state=torch.load(resumed["lastCheckpoint"],weights_only=True)
    assert resumed_state["history"]==full_state["history"]
    assert resumed_state["data_loader_rng_state"].equal(full_state["data_loader_rng_state"])
    for name,value in full_state["model_state"].items():
        assert torch.equal(value,resumed_state["model_state"][name]),name
