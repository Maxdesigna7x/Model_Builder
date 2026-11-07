from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

from .catalog import task_spec
from .data import split_indices
from .errors import BackendError
from .models import build_model
from .manifest import comparison_hash, semantic_hash
from .storage import atomic_json, iso_now, project_dir


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the local training runtime."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_manifest(device: torch.device) -> dict:
    packages={}
    for name in ("torch","numpy","transformers","huggingface-hub","peft","accelerate"):
        try: packages[name]=version(name)
        except PackageNotFoundError: pass
    return {"python":platform.python_version(),"platform":platform.platform(),"packages":packages,"device":str(device),"cuda":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def code_manifest() -> dict:
    """Identify the exact local source when training from a Git checkout."""
    result={"package":"modelbuilder","version":"0.1.0"}
    try:
        root=Path(__file__).resolve().parents[2]
        commit=subprocess.run(["git","rev-parse","HEAD"],cwd=root,capture_output=True,text=True,check=True,timeout=2).stdout.strip()
        dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=root,capture_output=True,text=True,check=True,timeout=2).stdout.strip())
        result.update({"git_commit":commit,"git_dirty":dirty})
    except (OSError,subprocess.SubprocessError):
        result.update({"git_commit":None,"git_dirty":None})
    return result


def redact_secrets(value):
    if isinstance(value,dict): return {key:("[REDACTED]" if any(marker in key.lower() for marker in ("token","secret","password")) else redact_secrets(item)) for key,item in value.items()}
    if isinstance(value,list): return [redact_secrets(item) for item in value]
    return value


def create_model_bundle(run_root: Path, request: dict, provenance: dict, model:nn.Module) -> Path:
    """Create a local, portable run bundle without performing any remote publish."""
    bundle=run_root/"bundle";bundle.mkdir(exist_ok=True)
    source=run_root/"best.pt";target=bundle/"model.pt"
    if target.exists(): target.unlink()
    try: target.hardlink_to(source)
    except OSError: shutil.copy2(source,target)
    dataset=request["dataset"]
    atomic_json(bundle/"config.json",{"schema_version":3,"task_id":request["task_id"],"architecture":request["architecture"],"semantic_hash":provenance["semantic_hash"],"model_spec":request.get("model_spec"),"initialized_from":provenance.get("initialized_from"),"graph":request.get("graph",{}),"dataset_contract":{"input_shape":dataset.get("inputShape"),"output_shape":dataset.get("outputShape"),"classes":dataset.get("classes")}})
    processor_node=next((node for node in request.get("graph",{}).get("nodes",[]) if node.get("data",{}).get("blockType") in {"hf_image_processor","hf_token_processor"}),None)
    if processor_node:
        props=processor_node["data"]["properties"]
        try:
            from transformers import AutoImageProcessor, AutoTokenizer
            from .pretrained import cache_dir
            cls=AutoImageProcessor if processor_node["data"]["blockType"]=="hf_image_processor" else AutoTokenizer
            artifact=cls.from_pretrained(str(props["repo_id"]),revision=str(props["revision"]),cache_dir=str(cache_dir()),trust_remote_code=False,local_files_only=True)
            artifact.save_pretrained(bundle/"processor")
        except Exception as exc:
            atomic_json(bundle/"processor.json",{"repo_id":props.get("repo_id"),"revision":props.get("revision"),"error":str(exc)})
    hf_module=next((module for module in model.modules() if module.__class__.__name__ in {"HuggingFaceVisionBackbone","HuggingFaceTextBackbone","HuggingFaceCausalLM"}),None)
    if hf_module:
        target_model=getattr(hf_module,"backbone",None) or getattr(hf_module,"model",None)
        if hasattr(target_model,"save_pretrained"):
            target_model.save_pretrained(bundle/"huggingface",safe_serialization=True)
    spec=request.get("model_spec") or {};base=f"{spec.get('repoId')}@{spec.get('revision')}" if spec.get("repoId") else "modelo desde cero"
    (bundle/"MODEL_CARD.md").write_text(f"# ModelBuilder run {provenance['run_id']}\n\n- Tarea: `{request['task_id']}`\n- Arquitectura: `{request['architecture']}`\n- Base: `{base}`\n- Hash semántico: `{provenance['semantic_hash']}`\n- Dataset: `{dataset.get('id')}` revisión `{dataset.get('revision',1)}`\n\nEste bundle fue generado localmente. Revise licencia, sesgos y desempeño por clase antes de desplegarlo.\n",encoding="utf-8")
    return bundle


def register_model(project: dict, run_id: str, bundle: Path, request: dict) -> None:
    path=project_dir(project)/"models"/"registry.json";path.parent.mkdir(parents=True,exist_ok=True)
    try: registry=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): registry={"schema_version":1,"models":[]}
    spec=request.get("model_spec") or {}
    registry["models"]=[item for item in registry.get("models",[]) if item.get("run_id")!=run_id]
    registry["models"].append({"run_id":run_id,"bundle":str(bundle),"aliases":["best","last"],"base":{"repo_id":spec.get("repoId"),"revision":spec.get("revision")},"created_at":iso_now()})
    atomic_json(path,registry)


def wait_for_training_control(run_root:Path,emit) -> bool:
    """Return False on cancellation and wait cooperatively while paused."""
    path=run_root/"control.json";was_paused=False
    while True:
        try: state=json.loads(path.read_text(encoding="utf-8")).get("state","running")
        except (OSError,json.JSONDecodeError): state="running"
        if state=="cancelled": return False
        if state!="paused":
            if was_paused: emit({"type":"resumed","run_id":run_root.name})
            return True
        if not was_paused: emit({"type":"paused","run_id":run_root.name});was_paused=True
        time.sleep(.2)


def _materialize_huggingface_tokens(payload: dict,graph: dict) -> torch.Tensor | None:
    processor=next((node.get("data",{}).get("properties",{}) for node in graph.get("nodes",[]) if node.get("data",{}).get("blockType")=="hf_token_processor"),None)
    if not processor: return None
    texts=payload.get("raw_inputs")
    if not texts: raise BackendError("HF_RAW_TEXT_REQUIRED","Vuelve a importar o descargar el dataset para conservar el texto crudo")
    from .pretrained import tokenize_texts
    return tokenize_texts(texts,processor)


def configure_finetuning(model: nn.Module, config: dict) -> list[nn.Parameter]:
    """Apply a reproducible transfer-learning recipe; return gradual params."""
    recipe=str(config.get("finetuning_recipe","auto")).lower();gradual=[]
    pretrained=[]
    for module in model.modules():
        if module.__class__.__name__ not in {"HuggingFaceVisionBackbone","HuggingFaceTextBackbone","HuggingFaceCausalLM"}: continue
        target=getattr(module,"backbone",None)
        if target is None: target=getattr(module,"model",None)
        if target is not None and target is not model: pretrained.append((module,target))
    if recipe in {"head","gradual"}:
        for _,target in pretrained:
            for parameter in target.parameters(): parameter.requires_grad=False
            if recipe=="gradual": gradual.extend(target.parameters())
    elif recipe=="full":
        for _,target in pretrained:
            for parameter in target.parameters(): parameter.requires_grad=True
    elif recipe=="lora":
        try: from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc: raise BackendError("PEFT_DEPENDENCIES_MISSING") from exc
        for wrapper,target in pretrained:
            causal=wrapper.__class__.__name__=="HuggingFaceCausalLM"
            peft=get_peft_model(target,LoraConfig(r=int(config.get("lora_rank",8)),lora_alpha=int(config.get("lora_alpha",16)),lora_dropout=float(config.get("lora_dropout",.05)),target_modules="all-linear",task_type=TaskType.CAUSAL_LM if causal else TaskType.FEATURE_EXTRACTION))
            if causal: wrapper.model=peft
            else: wrapper.backbone=peft
    return gradual


def load_initial_weights(model:nn.Module,path:str,architecture:str,input_shape:list[int])->dict:
    source=Path(path).expanduser().resolve()
    if not source.is_file(): raise BackendError("TRAINING_INITIAL_CHECKPOINT_MISSING",str(source))
    checkpoint=torch.load(source,map_location="cpu",weights_only=True)
    if checkpoint.get("architecture")!=architecture: raise BackendError("TRAINING_INITIAL_ARCHITECTURE_MISMATCH",str(checkpoint.get("architecture")))
    source_shape=checkpoint.get("dataset",{}).get("inputShape",[])
    if source_shape and source_shape[0]!=input_shape[0]: raise BackendError("TRAINING_INITIAL_INPUT_CHANNELS_MISMATCH",f"{source_shape[0]} vs {input_shape[0]}")
    state=checkpoint.get("model_state")
    if not isinstance(state,dict): raise BackendError("TRAINING_INITIAL_CHECKPOINT_INVALID")
    target=model.state_dict();compatible={key:value for key,value in state.items() if key in target and target[key].shape==value.shape}
    if not compatible: raise BackendError("TRAINING_INITIAL_NO_COMPATIBLE_WEIGHTS")
    model.load_state_dict(compatible,strict=False)
    loaded_state=model.state_dict()
    loaded=sorted(key for key,value in compatible.items() if key in loaded_state and torch.equal(loaded_state[key].detach().cpu(),value.detach().cpu()))
    if set(loaded)!=set(compatible): raise BackendError("TRAINING_INITIAL_LOAD_REPORT_INVALID")
    return {"checkpoint":str(source),"parent_run_id":checkpoint.get("run_id") or source.parent.name,"semantic_hash":checkpoint.get("semantic_hash"),"loaded_keys":loaded,"skipped_source_keys":sorted(set(state)-set(loaded)),"new_target_keys":sorted(set(target)-set(loaded))}


def validate_resume_checkpoint(checkpoint:dict,request:dict)->None:
    expected_hash=semantic_hash(request.get("graph",{}));saved_graph=checkpoint.get("graph",{})
    if checkpoint.get("task_id")!=request.get("task_id"): raise BackendError("TRAINING_RESUME_TASK_MISMATCH")
    if checkpoint.get("architecture")!=request.get("architecture"): raise BackendError("TRAINING_RESUME_ARCHITECTURE_MISMATCH")
    if semantic_hash(saved_graph)!=expected_hash: raise BackendError("TRAINING_RESUME_GRAPH_MISMATCH")
    saved=checkpoint.get("dataset",{});current=request.get("dataset",{})
    if saved.get("id")!=current.get("id") or saved.get("revision",1)!=current.get("revision",1): raise BackendError("TRAINING_RESUME_DATASET_MISMATCH")
    if saved.get("splits")!=current.get("splits"): raise BackendError("TRAINING_RESUME_SPLITS_MISMATCH")


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def decode_localization(prediction:torch.Tensor)->torch.Tensor:
    """Map four logits to ordered, normalized xyxy corners."""
    values=prediction.sigmoid();x1=torch.minimum(values[:,0],values[:,2]);x2=torch.maximum(values[:,0],values[:,2]);y1=torch.minimum(values[:,1],values[:,3]);y2=torch.maximum(values[:,1],values[:,3])
    return torch.stack((x1,y1,x2,y2),dim=1)


def _loss(task_id: str, prediction: torch.Tensor, target: torch.Tensor, loss_weights: torch.Tensor | None = None, loss_name: str = "auto", label_smoothing: float = 0.0) -> torch.Tensor:
    if isinstance(prediction,dict):
        if not prediction: raise BackendError("TRAINING_OUTPUTS_EMPTY")
        if task_id=="image.detection":
            if set(prediction)!={"boxes","classes"} or not isinstance(target,(list,tuple)):
                raise BackendError("TRAINING_DETECTION_BINDINGS_REQUIRED","Se requieren outputs boxes/classes y cajas por muestra.")
            components=detection_loss_components(prediction,target);return components["boxes"]+components["classes"]
        # Named auxiliary heads for the same supervised target are optimized
        # independently and averaged. Tasks with different targets per head
        # need an explicit task recipe and are intentionally rejected here.
        losses=[]
        for name,value in prediction.items():
            objective=task_spec(task_id)["objective"]
            shape_ok=value.shape==target.shape or objective in {"classification","token_prediction"} or (objective=="segmentation_binary" and value.ndim==4 and value.shape[1]==1 and value.shape[0]==target.shape[0] and value.shape[2:]==target.shape[1:]) or (objective=="segmentation_multiclass" and value.ndim==4 and value.shape[0]==target.shape[0] and value.shape[2:]==target.shape[1:])
            if not shape_ok:
                raise BackendError("TRAINING_OUTPUT_TARGET_SHAPE_MISMATCH",f"{name}: {list(value.shape)} vs {list(target.shape)}")
            losses.append(_loss(task_id,value,target,loss_weights,loss_name,label_smoothing))
        return torch.stack(losses).mean()
    objective = task_spec(task_id)["objective"]
    if objective=="localization":
        coords=decode_localization(prediction)
        if loss_name=="mae": return nn.functional.l1_loss(coords,target.float())
        if loss_name=="mse": return nn.functional.mse_loss(coords,target.float())
        return nn.functional.smooth_l1_loss(coords,target.float())
    if objective == "classification":
        ce=nn.functional.cross_entropy(prediction,target.long(),weight=loss_weights,label_smoothing=label_smoothing,reduction="none" if loss_name=="focal" else "mean")
        if loss_name=="focal":
            probability=prediction.softmax(1).gather(1,target.long().reshape(-1,1)).squeeze(1);return (((1-probability)**2)*ce).mean()
        return ce
    if objective == "token_prediction": return nn.functional.cross_entropy(prediction.transpose(1, 2), target.long(), ignore_index=-100 if bool((target==-100).any()) else 0)
    if objective == "segmentation_binary": return nn.functional.binary_cross_entropy_with_logits(prediction[:, 0], target.float(),pos_weight=loss_weights) + dice_loss(prediction[:, 0], target)
    if objective == "segmentation_multiclass": return nn.functional.cross_entropy(prediction, target.long(),weight=loss_weights) + multiclass_dice_loss(prediction,target)
    if loss_name=="mae": return nn.functional.l1_loss(prediction,target.float())
    if loss_name=="huber": return nn.functional.huber_loss(prediction,target.float())
    return nn.functional.mse_loss(prediction, target.float())


def detection_loss_components(prediction:dict,target:list[torch.Tensor])->dict[str,torch.Tensor]:
    boxes_pred,classes_pred=prediction["boxes"],prediction["classes"]
    box_target,class_target,positive=encode_detection_targets(target,classes_pred.shape[1:])
    box_target=box_target.to(boxes_pred.device);class_target=class_target.to(classes_pred.device);positive=positive.to(boxes_pred.device)
    class_loss=nn.functional.binary_cross_entropy_with_logits(classes_pred,class_target,reduction="none")
    positive_map=class_target.bool()
    positive_loss=class_loss[positive_map].mean() if positive_map.any() else classes_pred.sum()*0
    negative_loss=class_loss[~positive_map].mean() if (~positive_map).any() else classes_pred.sum()*0
    box_loss=nn.functional.smooth_l1_loss(boxes_pred.sigmoid().permute(0,2,3,1)[positive],box_target.permute(0,2,3,1)[positive]) if positive.any() else boxes_pred.sum()*0
    return {"boxes":box_loss,"classes":positive_loss+0.25*negative_loss}


def encode_detection_targets(boxes: list[torch.Tensor], grid_shape: tuple[int,...]) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Assign normalized cxcywh boxes to their center cell; collisions are errors."""
    classes,grid_h,grid_w=(int(value) for value in grid_shape)
    box_target=torch.zeros((len(boxes),4,grid_h,grid_w),dtype=torch.float32)
    class_target=torch.zeros((len(boxes),classes,grid_h,grid_w),dtype=torch.float32)
    positive=torch.zeros((len(boxes),grid_h,grid_w),dtype=torch.bool)
    for batch_index,items in enumerate(boxes):
        for row in items:
            cls,cx,cy,width,height=(float(value) for value in row)
            if not 0<=int(cls)<classes or not all(math.isfinite(value) for value in (cx,cy,width,height)) or not 0<=cx<=1 or not 0<=cy<=1 or width<=0 or height<=0:
                raise BackendError("TRAINING_DETECTION_TARGET_INVALID")
            gy=min(grid_h-1,int(cy*grid_h));gx=min(grid_w-1,int(cx*grid_w))
            if positive[batch_index,gy,gx]: raise BackendError("TRAINING_DETECTION_CELL_COLLISION",f"muestra {batch_index}, celda ({gy},{gx})")
            positive[batch_index,gy,gx]=True;class_target[batch_index,int(cls),gy,gx]=1
            box_target[batch_index,:,gy,gx]=torch.tensor([cx*grid_w-gx,cy*grid_h-gy,width,height])
    return box_target,class_target,positive


def box_iou_xyxy(first:torch.Tensor,second:torch.Tensor)->torch.Tensor:
    top_left=torch.maximum(first[...,:2],second[...,:2]);bottom_right=torch.minimum(first[...,2:],second[...,2:]);intersection=(bottom_right-top_left).clamp_min(0).prod(-1)
    area_first=(first[...,2:]-first[...,:2]).clamp_min(0).prod(-1);area_second=(second[...,2:]-second[...,:2]).clamp_min(0).prod(-1)
    return intersection/(area_first+area_second-intersection).clamp_min(1e-8)


def decode_detections(output:dict,score_threshold:float=.25,iou_threshold:float=.5,max_detections:int=100)->list[torch.Tensor]:
    boxes=output["boxes"].sigmoid();scores=output["classes"].sigmoid();batch,_,grid_h,grid_w=boxes.shape;decoded=[]
    for batch_index in range(batch):
        rows=[]
        for gy in range(grid_h):
            for gx in range(grid_w):
                for cls in range(scores.shape[1]):
                    score=float(scores[batch_index,cls,gy,gx])
                    if score<score_threshold: continue
                    tx,ty,width,height=boxes[batch_index,:,gy,gx].tolist();cx=(gx+tx)/grid_w;cy=(gy+ty)/grid_h
                    rows.append([cls,score,max(0,cx-width/2),max(0,cy-height/2),min(1,cx+width/2),min(1,cy+height/2)])
        kept=[]
        for cls in range(scores.shape[1]):
            candidates=sorted((row for row in rows if row[0]==cls),key=lambda item:item[1],reverse=True)
            while candidates and len(kept)<max_detections:
                current=candidates.pop(0);kept.append(current)
                if candidates:
                    first=torch.tensor(current[2:],dtype=torch.float32);candidates=[row for row in candidates if float(box_iou_xyxy(first,torch.tensor(row[2:],dtype=torch.float32)))<=iou_threshold]
        decoded.append(torch.tensor(kept,dtype=torch.float32).reshape(-1,6))
    return decoded


class DetectionDataset(Dataset):
    def __init__(self,x,boxes,nodes=()): self.x=x;self.boxes=boxes;self.nodes=nodes
    def __len__(self): return len(self.x)
    def __getitem__(self,index):
        x=self.x[index].clone();boxes=self.boxes[index].clone()
        for node in self.nodes:
            props=node.get("properties",{});kind=node.get("type");prob=float(props.get("probability",.5))
            if kind=="random_flip" and torch.rand(())<prob and bool(props.get("horizontal",True)):
                x=torch.flip(x,(-1,))
                if len(boxes): boxes[:,1]=1-boxes[:,1]
            elif kind=="random_rotate90" and torch.rand(())<prob:
                turns=int(torch.randint(1,4,()).item());x=torch.rot90(x,turns,(-2,-1))
                if len(boxes):
                    cx,cy,w,h=boxes[:,1].clone(),boxes[:,2].clone(),boxes[:,3].clone(),boxes[:,4].clone()
                    if turns==1: boxes[:,1],boxes[:,2],boxes[:,3],boxes[:,4]=cy,1-cx,h,w
                    elif turns==2: boxes[:,1],boxes[:,2]=1-cx,1-cy
                    else: boxes[:,1],boxes[:,2],boxes[:,3],boxes[:,4]=1-cy,cx,h,w
        return x,boxes


def collate_detection(batch): return torch.stack([item[0] for item in batch]),[item[1] for item in batch]


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = logits.sigmoid(); truth = target.float(); inter=(prob*truth).sum((1,2)); return 1-((2*inter+1)/(prob.sum((1,2))+truth.sum((1,2))+1)).mean()


def multiclass_dice_loss(logits: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    probability=logits.softmax(1);truth=nn.functional.one_hot(target.long(),logits.shape[1]).permute(0,3,1,2).float();scores=[]
    for class_index in range(1,logits.shape[1]):
        p=probability[:,class_index];t=truth[:,class_index];scores.append((2*(p*t).sum((1,2))+1)/(p.sum((1,2))+t.sum((1,2))+1))
    return 1-torch.stack(scores).mean()


def default_epochs(task_id: str) -> int:
    """Keep the first useful run short, except for tiny segmentation corpora."""
    if task_id == "text.language_model":
        return 10
    if task_id.startswith("image.segmentation"):
        return 40
    return 20


def _metric(task_id: str, prediction: torch.Tensor, target: torch.Tensor, target_transform: dict | None = None) -> tuple[float, int]:
    if isinstance(prediction,dict):
        if not prediction: raise BackendError("TRAINING_OUTPUTS_EMPTY")
        prediction=next(iter(prediction.values()))
    objective = task_spec(task_id)["objective"]
    if objective=="localization":
        predicted=decode_localization(prediction);inter=(torch.minimum(predicted[:,2:],target[:,2:])-torch.maximum(predicted[:,:2],target[:,:2])).clamp_min(0).prod(1);area_p=(predicted[:,2:]-predicted[:,:2]).clamp_min(0).prod(1);area_t=(target[:,2:]-target[:,:2]).clamp_min(0).prod(1);return float((inter/(area_p+area_t-inter).clamp_min(1e-8)).sum()),target.shape[0]
    if objective == "classification": return float((prediction.argmax(1)==target).sum()), target.shape[0]
    if objective == "token_prediction":
        valid=target!=0; return float(((prediction.argmax(-1)==target)&valid).sum()), int(valid.sum())
    if objective == "segmentation_binary":
        p=prediction[:,0].sigmoid()>=.5;t=target.bool();inter=(p&t).sum().item();den=p.sum().item()+t.sum().item();return (2*inter+1)/(den+1),1
    if objective == "segmentation_multiclass":
        pred=prediction.argmax(1); classes=prediction.shape[1]; scores=[]
        for c in range(1,classes):
            pc=pred==c;tc=target==c;union=(pc|tc).sum().item()
            if union: scores.append((pc&tc).sum().item()/union)
        return (sum(scores)/len(scores) if scores else 0),1
    if target_transform:
        shape=[1,*target.shape[1:]];center=torch.tensor(target_transform["center"],device=prediction.device).reshape(shape);scale=torch.tensor(target_transform["scale"],device=prediction.device).reshape(shape)
        prediction=prediction*scale+center;target=target.float()*scale+center
    if objective=="image_translation" and task_spec(task_id)["metric"]=="mae": err=nn.functional.l1_loss(prediction,target.float(),reduction="sum").item()
    else: err=nn.functional.mse_loss(prediction,target.float(),reduction="sum").item()
    return err,target.numel()


def detection_map50(predictions:list[dict],targets:list[list[torch.Tensor]],num_classes:int)->dict:
    ranked={index:[] for index in range(num_classes)};ground_truth={index:0 for index in range(num_classes)};true_positive=0;false_positive=0;false_negative=0
    for batch_pred,batch_targets in zip(predictions,targets):
        decoded=decode_detections(batch_pred,.001,.5)
        for sample_index,(pred_boxes,truth) in enumerate(zip(decoded,batch_targets)):
            truth=truth.detach().cpu().float();truth_by_class={index:[] for index in range(num_classes)}
            for row in truth:
                cls=int(row[0]);cx,cy,w,h=[float(value) for value in row[1:]]
                truth_by_class[cls].append(torch.tensor([cx-w/2,cy-h/2,cx+w/2,cy+h/2]))
            for cls,values in truth_by_class.items(): ground_truth[cls]+=len(values)
            used={cls:set() for cls in range(num_classes)}
            for row in pred_boxes:
                cls=int(row[0]);score=float(row[1]);candidates=truth_by_class[cls]
                if candidates:
                    ious=box_iou_xyxy(row[2:].unsqueeze(0),torch.stack(candidates)).flatten();best=int(ious.argmax());matched=float(ious[best])>=.5 and best not in used[cls]
                else: matched=False
                ranked[cls].append((score,matched));
                if matched: used[cls].add(best);true_positive+=1
                else: false_positive+=1
            false_negative+=sum(len(values)-len(used[cls]) for cls,values in truth_by_class.items())
    aps=[]
    for cls in range(num_classes):
        total=ground_truth[cls]
        if not total: continue
        rows=sorted(ranked[cls],key=lambda item:item[0],reverse=True);tp=fp=0;recalls=[0.];precisions=[1.]
        for _,hit in rows:
            tp+=int(hit);fp+=int(not hit);recalls.append(tp/total);precisions.append(tp/max(1,tp+fp))
        for index in range(len(precisions)-2,-1,-1): precisions[index]=max(precisions[index],precisions[index+1])
        ap=0.
        for threshold in torch.linspace(0,1,101).tolist():
            valid=[precisions[i] for i,recall in enumerate(recalls) if recall>=threshold];ap+=max(valid,default=0.)/101
        aps.append(ap)
    precision=true_positive/max(1,true_positive+false_positive);recall=true_positive/max(1,true_positive+false_negative)
    return {"map50":sum(aps)/len(aps) if aps else 0.,"precision":precision,"recall":recall,"f1":2*precision*recall/max(1e-8,precision+recall),"true_positives":true_positive,"false_positives":false_positive,"false_negatives":false_negative}


def _evaluate(model, loader, task_id, device, loss_weights=None, target_transform=None, loss_name="auto", label_smoothing=0.0) -> tuple[float,float,dict]:
    model.eval(); loss_sum=0.; count=0; metric_sum=0.; metric_count=0;predictions=[];targets=[];detection_predictions=[];detection_targets=[];detection_loss_sums={"boxes":0.,"classes":0.}
    with torch.inference_mode():
        for x,y in loader:
            x = x.to(device)
            if x.dtype == torch.uint8: x = x.float() / 255.0
            if isinstance(y,torch.Tensor): y=y.to(device)
            else: y=[value.to(device) for value in y]
            pred=model(x);n=x.shape[0]
            if task_id=="image.detection":
                components=detection_loss_components(pred,y);loss=sum(components.values());detection_loss_sums={key:detection_loss_sums[key]+value.item()*n for key,value in components.items()};detection_predictions.append({key:value.detach().cpu() for key,value in pred.items()});detection_targets.append([value.detach().cpu() for value in y])
            else: loss=_loss(task_id,pred,y,loss_weights,loss_name,label_smoothing)
            primary_pred=next(iter(pred.values())) if isinstance(pred,dict) else pred;loss_sum+=loss.item()*n;count+=n
            if task_id=="image.detection": pass
            else: m,c=_metric(task_id,primary_pred,y,target_transform);metric_sum+=m;metric_count+=c
            if task_spec(task_id)["objective"]=="classification": predictions.append(primary_pred.argmax(1).cpu());targets.append(y.cpu())
    average_loss=loss_sum/max(1,count);primary=metric_sum/max(1,metric_count);secondary={}
    objective=task_spec(task_id)["objective"]
    if task_id=="image.detection":
        scores=detection_map50(detection_predictions,detection_targets,int(dataset_class_count(model)))
        primary=scores.pop("map50");secondary={**scores,"loss_boxes":detection_loss_sums["boxes"]/max(1,count),"loss_classes":detection_loss_sums["classes"]/max(1,count)}
    if objective=="classification" and predictions:
        predicted=torch.cat(predictions);truth=torch.cat(targets);scores=[]
        for class_id in range(int(max(predicted.max(),truth.max()).item())+1):
            tp=((predicted==class_id)&(truth==class_id)).sum().item();fp=((predicted==class_id)&(truth!=class_id)).sum().item();fn=((predicted!=class_id)&(truth==class_id)).sum().item();scores.append((2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        secondary["macro_f1"]=sum(scores)/max(1,len(scores))
    elif objective in {"regression","forecast","reconstruction"}: secondary["rmse"]=math.sqrt(max(0.,primary))
    elif objective=="token_prediction": secondary["perplexity"]=math.exp(min(average_loss,50.))
    return average_loss,primary,secondary


def dataset_class_count(model)->int:
    return int(model.output_shapes["classes"][0])


def evaluation_diagnostics(model,loader,task_id,device,run_root:Path) -> dict:
    """Materialize task-specific test diagnostics without mixing them into validation."""
    objective=task_spec(task_id)["objective"];predictions=[];targets=[];gallery_inputs=[]
    model.eval()
    with torch.inference_mode():
        for x,y in loader:
            moved=x.to(device);moved=moved.float()/255.0 if moved.dtype==torch.uint8 else moved
            pred=model(moved)
            if task_id=="image.detection":
                predictions.append({key:value.cpu() for key,value in pred.items()});targets.append([value.cpu() for value in y])
            else:
                pred=next(iter(pred.values())) if isinstance(pred,dict) else pred;predictions.append(pred.cpu());targets.append(y.cpu())
            if task_id.startswith("image.") and sum(item.shape[0] for item in gallery_inputs)<8: gallery_inputs.append(x.cpu())
    if not predictions:return {}
    if task_id=="image.detection":
        details=detection_map50(predictions,targets,int(model.output_shapes["classes"][0]));path=run_root/"test_gallery.pt";torch.save({"inputs":torch.cat(gallery_inputs)[:8],"predictions":predictions,"targets":targets},path);details["gallery"]=str(path);return details
    pred=torch.cat(predictions);truth=torch.cat(targets);diagnostics={}
    if objective=="classification":
        labels=pred.argmax(1);classes=pred.shape[1];matrix=torch.zeros((classes,classes),dtype=torch.long)
        for expected,actual in zip(truth.long(),labels.long()): matrix[expected,actual]+=1
        per_class=[]
        for index in range(classes):
            tp=matrix[index,index].item();fp=matrix[:,index].sum().item()-tp;fn=matrix[index,:].sum().item()-tp
            precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);per_class.append({"class":index,"precision":precision,"recall":recall,"f1":2*precision*recall/max(1e-12,precision+recall),"support":matrix[index,:].sum().item()})
        diagnostics={"confusion_matrix":matrix.tolist(),"per_class":per_class}
    elif objective in {"regression","forecast","reconstruction"}:
        residual=(pred-truth.float()).flatten();diagnostics={"residuals":{"mean":residual.mean().item(),"std":residual.std().item(),"mae":residual.abs().mean().item(),"q05":torch.quantile(residual,.05).item(),"q95":torch.quantile(residual,.95).item()}}
    elif objective.startswith("segmentation"):
        labels=(pred[:,0].sigmoid()>=.5).long() if objective=="segmentation_binary" else pred.argmax(1);classes=2 if objective=="segmentation_binary" else pred.shape[1];ious=[]
        for index in range(classes):
            union=((labels==index)|(truth.long()==index)).sum().item();ious.append(((labels==index)&(truth.long()==index)).sum().item()/union if union else None)
        diagnostics={"per_class_iou":ious}
    if gallery_inputs:
        path=run_root/"test_gallery.pt";torch.save({"inputs":torch.cat(gallery_inputs)[:8],"predictions":pred[:8],"targets":truth[:8]},path);diagnostics["gallery"]=str(path)
    return diagnostics


class AugmentedTensorDataset(Dataset):
    """Applies compiled, train-only transforms lazily on every epoch."""
    def __init__(self,x:torch.Tensor,y:torch.Tensor,nodes:list[dict],task_id:str):
        self.x=x;self.y=y;self.nodes=nodes;self.task_id=task_id

    def __len__(self): return len(self.x)

    def __getitem__(self,index):
        x=self.x[index].clone();y=self.y[index].clone()
        for node in self.nodes:
            props=node.get("properties",{});kind=node.get("type")
            if kind=="random_flip" and torch.rand(())<float(props.get("probability",.5)):
                if bool(props.get("horizontal",True)):
                    x=torch.flip(x,(-1,))
                    if "segmentation" in self.task_id or self.task_id in {"image.reconstruction","image.translation"}: y=torch.flip(y,(-1,))
                    elif self.task_id=="image.localization":
                        old_left=y[0].clone();y[0]=1-y[2];y[2]=1-old_left
            elif kind=="random_rotate90" and torch.rand(())<float(props.get("probability",.35)):
                turns=int(torch.randint(1,4,()).item());x=torch.rot90(x,turns,(-2,-1))
                if "segmentation" in self.task_id or self.task_id in {"image.reconstruction","image.translation"}: y=torch.rot90(y,turns,(-2,-1))
                elif self.task_id=="image.localization":
                    left,top,right,bottom=y.tolist();corners=[(left,top),(right,top),(left,bottom),(right,bottom)]
                    mapped=[((py,1-px) if turns==1 else (1-px,1-py) if turns==2 else (1-py,px)) for px,py in corners]
                    y=torch.tensor([min(point[0] for point in mapped),min(point[1] for point in mapped),max(point[0] for point in mapped),max(point[1] for point in mapped)],dtype=y.dtype)
            elif kind=="color_jitter":
                side=str(props.get("side","input"));value=y if side=="target" and self.task_id=="image.translation" else x
                if value.dtype==torch.uint8: value=value.float()/255.0
                brightness=float(props.get("brightness",.15));contrast=float(props.get("contrast",.15));value=value.float()*(1+(torch.rand(())*2-1)*contrast)+(torch.rand(())*2-1)*brightness;value=value.clamp(0,1)
                if side=="target" and self.task_id=="image.translation":y=value
                else:x=value
            elif kind=="random_noise":
                side=str(props.get("side","input"));value=y if side=="target" and self.task_id=="image.translation" else x
                if value.dtype==torch.uint8: value=value.float()/255.0
                value=value.float()+torch.randn_like(value.float())*float(props.get("std",.03))
                if side=="target" and self.task_id=="image.translation":y=value
                else:x=value
        return x,y


def train(request: dict) -> dict:
    project=request["project"];dataset=request["dataset"];task_id=request["task_id"];architecture=request["architecture"];graph=request.get("graph",{});config=request.get("config",{})
    payload=torch.load(dataset["path"],weights_only=True);x,y=payload["inputs"],payload["targets"];boxes=payload.get("boxes")
    hf_tokens=_materialize_huggingface_tokens(payload,graph)
    if hf_tokens is not None:
        x=hf_tokens;dataset={**dataset,"inputShape":list(x.shape[1:])}
        if any(node.get("data",{}).get("blockType")=="hf_causal_lm" for node in graph.get("nodes",[])):
            token_ids=x[...,0];y=token_ids.roll(-1,dims=1);y[:,-1]=-100
    seed=int(config.get("seed",dataset.get("seed",42)));seed_everything(seed)
    keep_saved_splits=payload.get("split_source") in {"official","temporal"} or payload.get("split_percentages")==dataset.get("splits")
    splits=payload.get("splits") if keep_saved_splits else split_indices(len(x),seed,dataset.get("splits"))
    splits=splits or split_indices(len(x),seed,dataset.get("splits"))
    device=torch.device("cuda" if torch.cuda.is_available() and str(config.get("device","auto"))!="cpu" else "cpu")
    model=build_model(architecture,dataset["inputShape"],dataset["outputShape"],graph,task_id).to(device)
    explicit_frozen=[]
    for node in graph.get("nodes",[]):
        node_id=node.get("id");properties=node.get("data",{}).get("properties",{})
        if properties.get("frozen") and node_id in model.operations:
            for parameter in model.operations[node_id].parameters():parameter.requires_grad=False
            explicit_frozen.append(node_id)
    initial_path=str(config.get("initial_checkpoint","")).strip();initialization=load_initial_weights(model,initial_path,architecture,dataset["inputShape"]) if initial_path else None;request["initialized_from"]=initialization
    gradual_parameters=configure_finetuning(model,config)
    for node_id in explicit_frozen:
        for parameter in model.operations[node_id].parameters():parameter.requires_grad=False
    data_options=dataset.get("options",{});batch=max(1,int(config.get("batch_size",data_options.get("batchSize",32))));epochs=max(1,int(config.get("epochs",default_epochs(task_id))));lr=float(config.get("learning_rate",.001));optimizer_name=str(config.get("optimizer","adamw")).lower();smoke=str(config.get("finetuning_recipe","auto")).lower()=="smoke"
    if smoke: epochs=1
    backbone_lr=float(config.get("backbone_learning_rate",lr));head_lr=float(config.get("head_learning_rate",lr));weight_decay=float(config.get("weight_decay",1e-4))
    backbone=[];head=[];backbone_node_ids={node.get("id") for node in graph.get("nodes",[]) if node.get("data",{}).get("blockType","").startswith("hf_") or node.get("data",{}).get("properties",{}).get("parameter_group")=="backbone"}
    gradual_ids={id(parameter) for parameter in gradual_parameters}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad and id(parameter) not in gradual_ids: continue
        parts=name.split(".");node_id=parts[1] if len(parts)>2 and parts[0]=="operations" else None
        (backbone if node_id in backbone_node_ids else head).append(parameter)
    groups=[]
    if backbone: groups.append({"params":backbone,"lr":backbone_lr,"name":"backbone"})
    if head: groups.append({"params":head,"lr":head_lr,"name":"head"})
    if not groups: raise BackendError("TRAINING_NO_TRAINABLE_PARAMETERS")
    optimizer=torch.optim.SGD(groups,lr=lr,momentum=float(config.get("momentum",.9)),weight_decay=weight_decay) if optimizer_name=="sgd" else (torch.optim.Adam(groups,lr=lr,weight_decay=weight_decay) if optimizer_name=="adam" else torch.optim.AdamW(groups,lr=lr,weight_decay=weight_decay))
    scheduler_name=str(config.get("scheduler","none")).lower();warmup=max(0,int(config.get("warmup_epochs",0)))
    if scheduler_name=="cosine":
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:min(1.,(epoch+1)/max(1,warmup)) if epoch<warmup else .5*(1+math.cos(math.pi*(epoch-warmup)/max(1,epochs-warmup))))
    elif scheduler_name=="step": scheduler=torch.optim.lr_scheduler.StepLR(optimizer,max(1,int(config.get("scheduler_step_size",10))),float(config.get("scheduler_gamma",.1)))
    elif scheduler_name=="plateau": scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=max(1,int(config.get("scheduler_patience",2))),factor=float(config.get("scheduler_gamma",.5)))
    else: scheduler=None
    precision=str(config.get("precision","fp32")).lower();amp_enabled=device.type=="cuda" and precision in {"fp16","bf16"};amp_dtype=torch.float16 if precision=="fp16" else torch.bfloat16
    scaler=torch.amp.GradScaler("cuda",enabled=amp_enabled and precision=="fp16")
    accumulation=max(1,int(config.get("gradient_accumulation",1)));clip_norm=float(config.get("gradient_clip_norm",5.0));loss_name=str(config.get("loss","auto")).lower();label_smoothing=float(config.get("label_smoothing",0.0))
    augmentation_nodes=[node for node in (payload.get("pipeline",{}).get("nodes",[])) if node.get("enabled",True) and node.get("category")=="augment" and node.get("scope")=="train"]
    use_balanced_sampler=task_spec(task_id)["objective"]=="classification" and bool(config.get("balanced_sampler",False)) and y.ndim==1
    train_generator=torch.Generator().manual_seed(seed)
    def loader(name,shuffle=False):
        split_x,split_y=x[splits[name]],y[splits[name]]
        if task_id=="image.detection":
            split_boxes=[boxes[index] for index in splits[name]]
            data=DetectionDataset(split_x,split_boxes,augmentation_nodes if name=="train" else ())
        else:
            data=AugmentedTensorDataset(split_x,split_y,augmentation_nodes,task_id) if name=="train" and augmentation_nodes else TensorDataset(split_x,split_y)
        sampler=None
        if name=="train" and use_balanced_sampler:
            counts=torch.bincount(split_y.long(),minlength=int(dataset["outputShape"][0])).float().clamp_min(1)
            sampler=WeightedRandomSampler((1.0/counts)[split_y.long()],len(split_y),replacement=True,generator=torch.Generator().manual_seed(seed))
        workers=max(0,int(data_options.get("workers",0)));generator=train_generator if name=="train" else torch.Generator().manual_seed(seed);return DataLoader(data,batch_size=batch,shuffle=shuffle and sampler is None,sampler=sampler,generator=generator,num_workers=workers,pin_memory=bool(data_options.get("pin_memory",False)) and torch.cuda.is_available(),collate_fn=collate_detection if task_id=="image.detection" else None)
    train_loader,val_loader,test_loader=loader("train",True),loader("validation"),loader("test")
    loss_weights=None
    if task_id=="image.segmentation.binary":
        train_target=y[splits["train"]];positive=train_target.sum().clamp_min(1);loss_weights=((train_target.numel()-positive)/positive).clamp(1,10).to(device)
    elif task_id=="image.segmentation.multiclass":
        counts=torch.bincount(y[splits["train"]].long().flatten(),minlength=int(dataset["outputShape"][0])).float().clamp_min(1);loss_weights=(counts.sum()/(len(counts)*counts)).sqrt();loss_weights=(loss_weights/loss_weights.mean()).clamp(.25,4).to(device)
    elif task_spec(task_id)["objective"]=="classification" and bool(config.get("class_weights",False)):
        counts=torch.bincount(y[splits["train"]].long(),minlength=int(dataset["outputShape"][0])).float().clamp_min(1);loss_weights=(counts.sum()/(len(counts)*counts)).to(device)
    target_transform=data_options.get("targetTransform")
    best_criterion = str(config.get("best_model_criterion", "val_loss")).lower()
    is_higher_better = task_spec(task_id)["metric"].lower() in {"accuracy", "dice", "miou", "iou", "map50"}
    best_score = -math.inf if (best_criterion == "metric" and is_higher_better) else math.inf

    start_epoch=1;history=[];resume_path=str(config.get("resume_checkpoint","")).strip();resume_state={}
    if resume_path:
        if initial_path:raise BackendError("TRAINING_RESUME_AND_INITIAL_CHECKPOINT_CONFLICT")
        checkpoint=torch.load(Path(resume_path).expanduser().resolve(),map_location=device,weights_only=True);validate_resume_checkpoint(checkpoint,request);model.load_state_dict(checkpoint["model_state"]);optimizer.load_state_dict(checkpoint["optimizer_state"]);history=list(checkpoint.get("history",[]));start_epoch=int(checkpoint.get("epoch",0))+1;resume_state=checkpoint.get("training_state",{})
        if scheduler and checkpoint.get("scheduler_state"): scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"): scaler.load_state_dict(checkpoint["scaler_state"])
        if checkpoint.get("data_loader_rng_state") is not None: train_loader.generator.set_state(checkpoint["data_loader_rng_state"])
        rng=checkpoint.get("rng_state",{});random.setstate(rng["python"]);numpy_state=rng["numpy"];np.random.set_state((numpy_state["bit_generator"],numpy_state["state"].cpu().numpy(),numpy_state["position"],numpy_state["has_gauss"],numpy_state["cached_gaussian"]));torch.set_rng_state(rng["torch"])
        if device.type=="cuda" and rng.get("cuda"): torch.cuda.set_rng_state_all(rng["cuda"])
    run_id=str(uuid.uuid4());run_root=project_dir(project)/"runs"/run_id;run_root.mkdir(parents=True,exist_ok=True)
    provenance={"schema_version":2,"run_id":run_id,"created_at":iso_now(),"task_id":task_id,"architecture":architecture,"model_spec":redact_secrets(request.get("model_spec")),"initialized_from":initialization,"config":redact_secrets(config),"dataset":{"id":dataset["id"],"revision":dataset.get("revision",1),"source":dataset.get("source"),"repo":dataset.get("repoId"),"repo_revision":dataset.get("repoRevision"),"license":dataset.get("license"),"splits":dataset.get("splits"),"split_counts":dataset.get("splitCounts"),"split_source":dataset.get("splitSource"),"options":redact_secrets(dataset.get("options")),"pipeline":redact_secrets(dataset.get("pipeline"))},"semantic_hash":semantic_hash(graph),"code":code_manifest(),"runtime":runtime_manifest(device)}
    atomic_json(run_root/"run.json",{**provenance,"status":"running"});atomic_json(run_root/"control.json",{"state":"running"})
    emit({"type":"started","run_id":run_id,"device":str(device),"epochs":epochs,"parameters":sum(parameter.numel() for parameter in model.parameters()),"trainable_parameters":sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),"initialization":initialization,"parameter_groups":{"backbone":sum(parameter.numel() for parameter in backbone),"head":sum(parameter.numel() for parameter in head)}})
    patience=max(0,int(config.get("early_stopping_patience",0)));min_delta=float(config.get("early_stopping_min_delta",0.0));stale_epochs=int(resume_state.get("stale_epochs",0));stopped_early=False
    if resume_state.get("best_score") is not None:best_score=float(resume_state["best_score"])
    cancelled=False
    for epoch in range(start_epoch,epochs+1):
        if gradual_parameters and epoch>=max(1,int(config.get("unfreeze_epoch",2))):
            for parameter in gradual_parameters: parameter.requires_grad=True
        model.train();train_sum=0.;seen=0;train_head_sums={"boxes":0.,"classes":0.}
        for node_id,module in model.operations.items():
            properties=model.nodes[node_id].get("data",{}).get("properties",{})
            frozen=bool(properties.get("frozen",False));freeze_stats=bool(properties.get("freeze_batchnorm_stats",False)) or (frozen and all(not parameter.requires_grad for parameter in module.parameters()))
            if freeze_stats:
                for child in module.modules():
                    if isinstance(child,nn.modules.batchnorm._BatchNorm): child.eval()
        optimizer.zero_grad(set_to_none=True)
        for batch_index,(xb,yb) in enumerate(train_loader,1):
            if not wait_for_training_control(run_root,emit): cancelled=True;break
            xb = xb.to(device)
            if xb.dtype == torch.uint8: xb = xb.float() / 255.0
            if isinstance(yb,torch.Tensor): yb = yb.to(device)
            else: yb=[value.to(device) for value in yb]
            context=torch.autocast(device_type=device.type,dtype=amp_dtype) if amp_enabled else nullcontext()
            with context:
                pred=model(xb)
                if task_id=="image.detection":
                    components=detection_loss_components(pred,yb);raw_loss=sum(components.values())
                    for key,value in components.items():train_head_sums[key]+=value.detach().item()*xb.shape[0]
                else: raw_loss=_loss(task_id,pred,yb,loss_weights,loss_name,label_smoothing)
                loss=raw_loss/accumulation
            if not torch.isfinite(loss): raise BackendError("TRAINING_LOSS_NON_FINITE")
            scaler.scale(loss).backward()
            if batch_index%accumulation==0 or batch_index==len(train_loader):
                scaler.unscale_(optimizer)
                if clip_norm>0: torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            train_sum+=raw_loss.item()*xb.shape[0];seen+=xb.shape[0]
            if smoke and batch_index>=2: break
        if cancelled: break
        train_loss=train_sum/max(1,seen);val_loss,metric,secondary=_evaluate(model,val_loader,task_id,device,loss_weights,target_transform,loss_name,label_smoothing)
        if task_id=="image.detection":secondary={**secondary,"train_loss_boxes":train_head_sums["boxes"]/max(1,seen),"train_loss_classes":train_head_sums["classes"]/max(1,seen)}
        point={"epoch":epoch,"trainLoss":train_loss,"valLoss":val_loss,"metric":metric,"secondaryMetrics":secondary,"learningRate":optimizer.param_groups[0]["lr"]};history.append(point)

        # Check best model criterion
        is_best = False
        if best_criterion == "val_loss":
            if val_loss < best_score: best_score = val_loss; is_best = True
        elif best_criterion == "train_loss":
            if train_loss < best_score: best_score = train_loss; is_best = True
        elif best_criterion == "metric":
            if is_higher_better:
                if metric > best_score: best_score = metric; is_best = True
            else:
                if metric < best_score: best_score = metric; is_best = True
        else: # none
            is_best = True

        if scheduler: scheduler.step(val_loss) if scheduler_name=="plateau" else scheduler.step()
        training_state={"best_score":best_score,"stale_epochs":stale_epochs}
        if is_best:
            stale_epochs=0;training_state={"best_score":best_score,"stale_epochs":stale_epochs};save_checkpoint(run_root/"best.pt",model,optimizer,epoch,request,history,scheduler,scaler,training_state,run_id,train_loader.generator)
        else: stale_epochs+=1
        training_state={"best_score":best_score,"stale_epochs":stale_epochs};save_checkpoint(run_root/"last.pt",model,optimizer,epoch,request,history,scheduler,scaler,training_state,run_id,train_loader.generator)
        with (run_root/"events.jsonl").open("a",encoding="utf-8") as handle: handle.write(json.dumps(point,allow_nan=False)+"\n")
        emit({"type":"metric","run_id":run_id,"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"metric":metric,"secondary_metrics":secondary,"metric_name":task_spec(task_id)["metric"],"learning_rate":optimizer.param_groups[0]["lr"]})
        if patience and stale_epochs>=patience: stopped_early=True;break
    if cancelled:
        last_path=run_root/"last.pt";save_checkpoint(last_path,model,optimizer,max(0,epoch-1),request,history,scheduler,scaler,{"best_score":best_score,"stale_epochs":stale_epochs},run_id,train_loader.generator)
        result={"runId":run_id,"status":"cancelled","metricName":task_spec(task_id)["metric"],"history":history,"lastCheckpoint":str(last_path)}
        atomic_json(run_root/"result.json",result);atomic_json(run_root/"run.json",{**provenance,"status":"cancelled","completed_at":iso_now(),"result":"result.json"});emit({"type":"complete","run_id":run_id,"result":result});return result
    best_path=run_root/"best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path,map_location=device,weights_only=True)["model_state"])
    test_loss,test_metric,test_secondary=_evaluate(model,test_loader,task_id,device,loss_weights,target_transform,loss_name,label_smoothing)
    test_diagnostics=evaluation_diagnostics(model,test_loader,task_id,device,run_root)
    bundle=create_model_bundle(run_root,request,provenance,model);register_model(project,run_id,bundle,request)
    result={"runId":run_id,"status":"completed","metricName":task_spec(task_id)["metric"],"comparisonContract":comparison_hash(task_id,dataset,task_spec(task_id)["metric"]),"history":history,"checkpoint":str(run_root/"best.pt"),"lastCheckpoint":str(run_root/"last.pt"),"bundle":str(bundle),"stoppedEarly":stopped_early,"test":{"loss":test_loss,"metric":test_metric,"secondaryMetrics":test_secondary,"diagnostics":test_diagnostics}}
    atomic_json(run_root/"result.json",result);atomic_json(run_root/"run.json",{**provenance,"status":"completed","completed_at":iso_now(),"result":"result.json"})
    atomic_json(project_dir(project)/"runs"/"latest.json",{"run_id":run_id,"checkpoint":str(run_root/"best.pt"),"result":str(run_root/"result.json")})
    emit({"type":"complete","run_id":run_id,"result":result});return result


def save_checkpoint(path: Path,model,optimizer,epoch,request,history,scheduler=None,scaler=None,training_state=None,run_id=None,data_loader_generator=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as handle: temp=Path(handle.name)
    numpy_state=np.random.get_state()
    rng_state={"python":random.getstate(),"numpy":{"bit_generator":numpy_state[0],"state":torch.from_numpy(numpy_state[1].copy()),"position":numpy_state[2],"has_gauss":numpy_state[3],"cached_gaussian":numpy_state[4]},"torch":torch.get_rng_state()}
    if torch.cuda.is_available(): rng_state["cuda"]=torch.cuda.get_rng_state_all()
    graph=request.get("graph",{})
    torch.save({"schema_version":3,"run_id":run_id,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict() if scheduler else None,"scaler_state":scaler.state_dict() if scaler else None,"data_loader_rng_state":data_loader_generator.get_state() if data_loader_generator is not None else None,"epoch":epoch,"task_id":request["task_id"],"architecture":request["architecture"],"model_spec":request.get("model_spec"),"initialized_from":request.get("initialized_from"),"graph":graph,"semantic_hash":semantic_hash(graph),"dataset":request["dataset"],"history":history,"training_state":training_state or {},"seed":int(request.get("config",{}).get("seed",request["dataset"].get("seed",42))),"rng_state":rng_state},temp);temp.replace(path)
