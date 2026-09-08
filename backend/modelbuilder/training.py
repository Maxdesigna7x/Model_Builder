from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .catalog import task_spec
from .data import split_indices
from .errors import BackendError
from .models import build_model
from .storage import atomic_json, iso_now, project_dir


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def _loss(task_id: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    objective = task_spec(task_id)["objective"]
    if objective == "classification": return nn.functional.cross_entropy(prediction, target.long())
    if objective == "token_prediction": return nn.functional.cross_entropy(prediction.transpose(1, 2), target.long(), ignore_index=0)
    if objective == "segmentation_binary": return nn.functional.binary_cross_entropy_with_logits(prediction[:, 0], target.float()) + dice_loss(prediction[:, 0], target)
    if objective == "segmentation_multiclass": return nn.functional.cross_entropy(prediction, target.long())
    return nn.functional.mse_loss(prediction, target.float())


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = logits.sigmoid(); truth = target.float(); inter=(prob*truth).sum((1,2)); return 1-((2*inter+1)/(prob.sum((1,2))+truth.sum((1,2))+1)).mean()


def _metric(task_id: str, prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    objective = task_spec(task_id)["objective"]
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
    err=nn.functional.mse_loss(prediction,target.float(),reduction="sum").item(); return err,target.numel()


def _evaluate(model, loader, task_id, device) -> tuple[float,float]:
    model.eval(); loss_sum=0.; count=0; metric_sum=0.; metric_count=0
    with torch.inference_mode():
        for x,y in loader:
            x = x.to(device)
            if x.dtype == torch.uint8: x = x.float() / 255.0
            y = y.to(device)
            pred=model(x);loss=_loss(task_id,pred,y);n=x.shape[0];loss_sum+=loss.item()*n;count+=n;m,c=_metric(task_id,pred,y);metric_sum+=m;metric_count+=c
    return loss_sum/max(1,count),metric_sum/max(1,metric_count)


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
                    if "segmentation" in self.task_id or self.task_id=="image.reconstruction": y=torch.flip(y,(-1,))
            elif kind=="random_rotate90" and torch.rand(())<float(props.get("probability",.35)):
                turns=int(torch.randint(1,4,()).item());x=torch.rot90(x,turns,(-2,-1))
                if "segmentation" in self.task_id or self.task_id=="image.reconstruction": y=torch.rot90(y,turns,(-2,-1))
            elif kind=="color_jitter":
                if x.dtype==torch.uint8: x=x.float()/255.0
                brightness=float(props.get("brightness",.15));contrast=float(props.get("contrast",.15));x=x.float()*(1+(torch.rand(())*2-1)*contrast)+(torch.rand(())*2-1)*brightness;x=x.clamp(0,1)
            elif kind=="random_noise":
                if x.dtype==torch.uint8: x=x.float()/255.0
                x=x.float()+torch.randn_like(x.float())*float(props.get("std",.03))
        return x,y


def train(request: dict) -> dict:
    project=request["project"];dataset=request["dataset"];task_id=request["task_id"];architecture=request["architecture"];graph=request.get("graph",{});config=request.get("config",{})
    payload=torch.load(dataset["path"],weights_only=True);x,y=payload["inputs"],payload["targets"]
    seed=int(config.get("seed",dataset.get("seed",42)));torch.manual_seed(seed)
    splits=payload.get("splits") if payload.get("split_percentages")==dataset.get("splits") else split_indices(len(x),seed,dataset.get("splits"))
    splits=splits or split_indices(len(x),seed,dataset.get("splits"))
    device=torch.device("cuda" if torch.cuda.is_available() and str(config.get("device","auto"))!="cpu" else "cpu")
    model=build_model(architecture,dataset["inputShape"],dataset["outputShape"],graph,task_id).to(device)
    data_options=dataset.get("options",{});batch=max(1,int(config.get("batch_size",data_options.get("batchSize",32))));epochs=max(1,int(config.get("epochs",20)));lr=float(config.get("learning_rate",.001));optimizer_name=str(config.get("optimizer","adamw")).lower()
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9) if optimizer_name=="sgd" else (torch.optim.Adam(model.parameters(),lr=lr) if optimizer_name=="adam" else torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=float(config.get("weight_decay",1e-4))))
    augmentation_nodes=[node for node in (payload.get("pipeline",{}).get("nodes",[])) if node.get("enabled",True) and node.get("category")=="augment" and node.get("scope")=="train"]
    def loader(name,shuffle=False):
        split_x,split_y=x[splits[name]],y[splits[name]]
        data=AugmentedTensorDataset(split_x,split_y,augmentation_nodes,task_id) if name=="train" and augmentation_nodes else TensorDataset(split_x,split_y)
        workers=max(0,int(data_options.get("workers",0)));return DataLoader(data,batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed),num_workers=workers,pin_memory=bool(data_options.get("pin_memory",False)) and torch.cuda.is_available())
    train_loader,val_loader,test_loader=loader("train",True),loader("validation"),loader("test")
    best_criterion = str(config.get("best_model_criterion", "none")).lower()
    is_higher_better = task_spec(task_id)["metric"].lower() in {"accuracy", "dice", "miou", "iou"}
    best_score = -math.inf if (best_criterion == "metric" and is_higher_better) else math.inf

    run_id=str(uuid.uuid4());run_root=project_dir(project)/"runs"/run_id;run_root.mkdir(parents=True,exist_ok=True);history=[]
    atomic_json(run_root/"run.json",{"run_id":run_id,"status":"running","created_at":iso_now(),"task_id":task_id,"architecture":architecture,"config":config,"dataset_id":dataset["id"],"device":str(device)})
    emit({"type":"started","run_id":run_id,"device":str(device),"epochs":epochs})
    for epoch in range(1,epochs+1):
        model.train();train_sum=0.;seen=0
        for xb,yb in train_loader:
            xb = xb.to(device)
            if xb.dtype == torch.uint8: xb = xb.float() / 255.0
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True);pred=model(xb);loss=_loss(task_id,pred,yb)
            if not torch.isfinite(loss): raise BackendError("TRAINING_LOSS_NON_FINITE")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();train_sum+=loss.item()*xb.shape[0];seen+=xb.shape[0]
        train_loss=train_sum/max(1,seen);val_loss,metric=_evaluate(model,val_loader,task_id,device);point={"epoch":epoch,"trainLoss":train_loss,"valLoss":val_loss,"metric":metric};history.append(point)

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

        if is_best:
            save_checkpoint(run_root/"best.pt",model,optimizer,epoch,request,history)
        save_checkpoint(run_root/"last.pt",model,optimizer,epoch,request,history)
        with (run_root/"events.jsonl").open("a",encoding="utf-8") as handle: handle.write(json.dumps(point,allow_nan=False)+"\n")
        emit({"type":"metric","run_id":run_id,"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"metric":metric,"metric_name":task_spec(task_id)["metric"]})
    test_loss,test_metric=_evaluate(model,test_loader,task_id,device)
    result={"runId":run_id,"status":"completed","metricName":task_spec(task_id)["metric"],"history":history,"checkpoint":str(run_root/"best.pt"),"test":{"loss":test_loss,"metric":test_metric}}
    atomic_json(run_root/"result.json",result);atomic_json(run_root/"run.json",{"run_id":run_id,"status":"completed","completed_at":iso_now(),"task_id":task_id,"architecture":architecture,"config":config,"result":"result.json"})
    atomic_json(project_dir(project)/"runs"/"latest.json",{"run_id":run_id,"checkpoint":str(run_root/"best.pt"),"result":str(run_root/"result.json")})
    emit({"type":"complete","run_id":run_id,"result":result});return result


def save_checkpoint(path: Path,model,optimizer,epoch,request,history):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as handle: temp=Path(handle.name)
    torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"epoch":epoch,"task_id":request["task_id"],"architecture":request["architecture"],"graph":request.get("graph",{}),"dataset":request["dataset"],"history":history},temp);temp.replace(path)
