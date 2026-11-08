from __future__ import annotations

import base64
import csv
import io
import json
import math
import random
import re
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .catalog import task_spec
from .errors import BackendError
from .storage import atomic_json, project_dir


def split_indices(n: int, seed: int, percentages: dict[str, int] | None = None) -> dict[str, list[int]]:
    percentages = percentages or {"train": 70, "validation": 15, "test": 15}
    train_percent = int(percentages.get("train", 70))
    validation_percent = int(percentages.get("validation", 15))
    test_percent = int(percentages.get("test", 15))
    if train_percent + validation_percent + test_percent != 100 or min(train_percent, validation_percent, test_percent) < 1:
        raise BackendError("SPLITS_INVALID_PERCENTAGES")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator).tolist()
    a = max(1, min(n - 2, round(n * train_percent / 100)))
    b = max(a + 1, min(n - 1, a + round(n * validation_percent / 100)))
    return {"train": order[:a], "validation": order[a:b], "test": order[b:]}


def split_percentages(splits: dict[str, list[int]]) -> dict[str, int]:
    """Return display percentages that add up to 100 for concrete split indices."""
    names = ("train", "validation", "test")
    counts = {name: len(splits.get(name, [])) for name in names}
    total = sum(counts.values())
    if total <= 0:
        return {"train": 70, "validation": 15, "test": 15}
    exact = {name: counts[name] * 100 / total for name in names}
    result = {name: int(math.floor(exact[name])) for name in names}
    for name in sorted(names, key=lambda item: exact[item] - result[item], reverse=True)[:100 - sum(result.values())]:
        result[name] += 1
    return result


def _display_targets(y: torch.Tensor | None,options: dict) -> torch.Tensor | None:
    transform=options.get("targetTransform")
    if y is None or not transform:
        return y
    shape=[1,*y.shape[1:]];center=torch.tensor(transform["center"],dtype=y.dtype).reshape(shape);scale=torch.tensor(transform["scale"],dtype=y.dtype).reshape(shape)
    return y*scale+center


def _build_preview(task_id: str, x: torch.Tensor, y: torch.Tensor | None, classes: list[str] | None = None, options: dict | None = None, sample_indices: list[int] | None = None) -> dict:
    items = []
    sample_indices = sample_indices if sample_indices is not None else list(range(min(8, len(x))))
    num_samples = len(sample_indices)
    if task_id.startswith("text."):
        vocab = (options or {}).get("vocab", [])
        for i in sample_indices:
            token_ids = x[i].flatten().tolist()
            words = [str(vocab[token]) if 0 <= int(token) < len(vocab) else "<unk>" for token in token_ids if int(token) != 0]
            items.append({"text": " ".join(words)})
        return {"type": "text", "items": items, "sampleIndices": sample_indices}
    if task_id.startswith("image") or "segmentation" in task_id:
        for i in sample_indices:
            img_tensor = x[i]
            if img_tensor.dtype == torch.uint8:
                if img_tensor.shape[0] == 1:
                    arr = img_tensor[0].numpy()
                    pil_img = Image.fromarray(arr, mode="L")
                else:
                    arr = img_tensor[:3].permute(1, 2, 0).numpy()
                    pil_img = Image.fromarray(arr, mode="RGB")
            else:
                display = img_tensor.float()
                if float(display.min()) < 0 or float(display.max()) > 1:
                    display = (display - display.min()) / (display.max() - display.min() + 1e-7)
                if display.shape[0] == 1:
                    arr = (display[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr, mode="L")
                else:
                    arr = (display[:3].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr, mode="RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            label_text = ""
            if classes and y is not None and y.ndim >= 1 and i < len(y):
                val = int(y[i]) if y[i].ndim == 0 else int(y[i].reshape(-1)[0])
                if 0 <= val < len(classes):
                    label_text = classes[val]
            elif y is not None and i < len(y):
                label_text = f"Val: {float(y[i].reshape(-1)[0]):.2f}" if y[i].numel() > 0 else ""
            item = {"url": b64, "label": label_text or f"Muestra #{i+1}"}
            # Las máscaras son datos de salida, no una etiqueta escalar: deben
            # verse junto a su imagen para poder comprobar su alineación.
            if "segmentation" in task_id and y is not None and i < len(y):
                mask = y[i].detach().cpu().numpy().astype(np.uint8)
                if task_id.endswith("binary"):
                    mask_img = Image.fromarray(mask * 255, mode="L")
                    target_label = "Máscara"
                else:
                    palette = np.array([[8, 12, 18], [35, 190, 215], [150, 110, 235], [65, 200, 125], [240, 165, 45]], dtype=np.uint8)
                    mask_img = Image.fromarray(palette[np.clip(mask, 0, len(palette) - 1)], mode="RGB")
                    target_label = "Máscara de clases"
                target_buf = io.BytesIO(); mask_img.save(target_buf, format="PNG")
                item.update({"targetUrl": "data:image/png;base64," + base64.b64encode(target_buf.getvalue()).decode("utf-8"), "targetLabel": target_label})
            elif task_id in {"image.reconstruction","image.translation"} and y is not None and i < len(y):
                target=y[i].float()/255 if y.dtype==torch.uint8 else y[i].float().clamp(0,1)
                if target.shape[0]==1: target_image=Image.fromarray((target[0].numpy()*255).astype(np.uint8),mode="L")
                else: target_image=Image.fromarray((target[:3].permute(1,2,0).numpy()*255).astype(np.uint8),mode="RGB")
                target_buf=io.BytesIO();target_image.save(target_buf,format="PNG");item.update({"targetUrl":"data:image/png;base64,"+base64.b64encode(target_buf.getvalue()).decode("utf-8"),"targetLabel":"Objetivo"})
            items.append(item)
        return {"type": "image", "items": items, "sampleIndices": sample_indices}
    else:
        for i in sample_indices:
            feat = x[i].flatten()[:8].tolist()
            target_str = ""
            if classes and y is not None and i < len(y):
                if y[i].numel() == 1:
                    idx = int(y[i])
                    target_str = classes[idx] if 0 <= idx < len(classes) else str(idx)
                else:
                    target_str = " ".join(classes[int(idx)] if 0 <= int(idx) < len(classes) else str(int(idx)) for idx in y[i].flatten()[:8])
            elif y is not None and i < len(y):
                target_str = f"{float(y[i].reshape(-1)[0]):.4f}" if y[i].numel() > 0 else ""
            items.append({"features": [round(f, 4) for f in feat], "target": target_str})
        configured_cols = options.get("feature_columns") if options else None
        cols = list(configured_cols[:8]) if configured_cols else [f"x{k+1}" for k in range(min(8, x.shape[-1]))]
        return {"type": "tabular", "items": items, "columns": cols, "sampleIndices": sample_indices}


def _apply_normalization(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "standard":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-7
        return (x - mean) / std
    elif mode == "minmax":
        min_v = x.min(dim=0, keepdim=True)[0]
        max_v = x.max(dim=0, keepdim=True)[0]
        return (x - min_v) / (max_v - min_v + 1e-7)
    elif mode == "minmax_sym":
        min_v = x.min(dim=0, keepdim=True)[0]
        max_v = x.max(dim=0, keepdim=True)[0]
        return 2.0 * (x - min_v) / (max_v - min_v + 1e-7) - 1.0
    return x


def _fit_import_normalization(x: torch.Tensor, options: dict) -> tuple[torch.Tensor, dict, dict | None]:
    """Fit imported feature scaling on train only and preserve it for inference."""
    mode = options.get("normalization", "none")
    if mode == "none":
        return x, options, None
    splits = split_indices(len(x), 42)
    train = x[splits["train"]].float()
    if mode == "standard":
        center = train.mean(dim=0, keepdim=True)
        scale = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-7)
        state = {"mode": mode, "center": center.squeeze(0).tolist(), "scale": scale.squeeze(0).tolist()}
        normalized = (x.float() - center) / scale
    elif mode in {"minmax", "minmax_sym"}:
        low, high = train.amin(dim=0, keepdim=True), train.amax(dim=0, keepdim=True)
        state = {"mode": mode, "low": low.squeeze(0).tolist(), "high": high.squeeze(0).tolist()}
        normalized = (x.float() - low) / (high - low).clamp_min(1e-7)
        if mode == "minmax_sym":
            normalized = normalized * 2 - 1
    else:
        return x, options, None
    return normalized, {**options, "inputTransform": state}, splits


def _histogram(values: torch.Tensor, bins: int = 16) -> dict:
    values=values.detach().float().flatten();values=values[torch.isfinite(values)]
    if not len(values): return {"labels":[],"values":[]}
    low=float(values.min());high=float(values.max())
    if math.isclose(low,high): return {"labels":[f"{low:.3g}"],"values":[len(values)]}
    counts=torch.histc(values,bins=bins,min=low,max=high);edges=torch.linspace(low,high,bins+1);labels=[f"{float((edges[i]+edges[i+1])/2):.3g}" for i in range(bins)]
    return {"labels":labels,"values":[int(v) for v in counts.tolist()]}


def build_analytics(task_id: str, x: torch.Tensor, y: torch.Tensor, classes: list[str] | None = None, options: dict | None = None) -> dict:
    sample=x[:min(len(x),2000)].detach().cpu();target=y[:len(sample)].detach().cpu();options=options or {}
    modality="image" if task_id.startswith("image") else "sequence" if task_id.startswith("sequence") else "text" if task_id.startswith("text") else "tabular"
    result={"modality":modality}
    if "classification" in task_id and target.ndim==1:
        count=max(len(classes or []),int(target.max())+1 if len(target) else 0);result["classDistribution"]={"labels":classes or [f"Clase {i}" for i in range(count)],"values":torch.bincount(target.long(),minlength=count).tolist()}
    if modality=="image":
        image=sample.float()/255.0 if sample.dtype==torch.uint8 else sample.float();flat=image.flatten()
        if flat.numel()>300000: flat=flat[::max(1,flat.numel()//300000)]
        result["intensityHistogram"]=_histogram(flat,20);channels=image.shape[1];names=["Gris"] if channels==1 else ["Rojo","Verde","Azul"][:channels]
        result["channelStats"]={"labels":names,"mean":[round(float(v),5) for v in image.mean((0,2,3))],"std":[round(float(v),5) for v in image.std((0,2,3))]}
        if "segmentation" in task_id:
            coverage=(target>0).float().mean(tuple(range(1,target.ndim)))*100;result["maskCoverage"]=_histogram(coverage,12)
            class_count=max(len(classes or []),int(target.max())+1);pixels=torch.bincount(target.long().flatten(),minlength=class_count);result["pixelDistribution"]={"labels":classes or [f"Clase {i}" for i in range(class_count)],"values":pixels.tolist()}
        elif "regression" in task_id: result["targetHistogram"]=_histogram(target[:,0] if target.ndim>1 else target,16)
    elif modality=="tabular":
        table=sample.float().reshape(len(sample),-1);limit=min(12,table.shape[1]);table=table[:,:limit];names=[f"x{i+1}" for i in range(limit)]
        result["featureSummary"]={"labels":names,"mean":[round(float(v),5) for v in table.mean(0)],"std":[round(float(v),5) for v in table.std(0)]}
        result["featureHistograms"]=[{"name":names[i],**_histogram(table[:,i],16)} for i in range(min(8,limit))]
        if limit>1:
            corr=torch.corrcoef(table.T).nan_to_num();result["correlation"]={"labels":names,"values":[[i,j,round(float(corr[j,i]),4)] for i in range(limit) for j in range(limit)]}
        if "regression" in task_id: result["targetHistogram"]=_histogram(target[:,0] if target.ndim>1 else target,16)
    elif modality=="sequence":
        seq=sample.float();steps=seq.shape[1];features=min(6,seq.shape[2]);mean=seq.mean(0)
        result["meanSeries"]={"labels":[str(i) for i in range(steps)],"series":[{"name":f"Variable {i+1}","values":[round(float(v),5) for v in mean[:,i]]} for i in range(features)]}
        flat=seq.reshape(-1,seq.shape[-1]);result["featureHistograms"]=[{"name":f"Variable {i+1}",**_histogram(flat[:,i],16)} for i in range(features)]
        if "regression" in task_id or "forecast" in task_id: result["targetHistogram"]=_histogram(target.flatten(),16)
    else:
        tokens=sample.long();lengths=(tokens!=0).sum(1);result["lengthHistogram"]=_histogram(lengths.float(),min(16,max(1,int(lengths.max()) if len(lengths) else 1)))
        positive=tokens[tokens>1];vocab=options.get("vocab",[])
        if positive.numel():
            counts=torch.bincount(positive);top=torch.topk(counts,min(15,int((counts>0).sum())));labels=[str(vocab[int(i)]) if int(i)<len(vocab) else f"Token {int(i)}" for i in top.indices];result["tokenFrequency"]={"labels":labels,"values":top.values.tolist()}
    return result


def _save(
    project: dict,
    task_id: str,
    x: torch.Tensor,
    y: torch.Tensor,
    description: str,
    seed: int,
    classes: list[str] | None = None,
    options: dict | None = None,
    *,
    source: str = "synthetic",
    source_path: str | None = None,
    metadata: dict | None = None,
    split_indices_override: dict[str, list[int]] | None = None,
    split_source: str = "generated",
    raw_inputs: list[str] | None = None,
    boxes: list[torch.Tensor] | None = None,
    original_sizes: list[tuple[int,int]] | None = None,
    spatial_affine: torch.Tensor | None = None,
) -> dict:
    dataset_id = str(uuid.uuid4())
    root = project_dir(project) / "datasets" / dataset_id / "1"
    root.mkdir(parents=True, exist_ok=True)
    splits = split_indices_override or split_indices(len(x), seed)
    percentages = split_percentages(splits)
    counts = {name: len(splits.get(name, [])) for name in ("train", "validation", "test")}
    opts = options or {}
    image_task=task_id.startswith("image") or "segmentation" in task_id
    if image_task:
        original_sizes=original_sizes or ([(int(x.shape[-1]),int(x.shape[-2])) for _ in range(len(x))] if x.ndim>=4 else [])
    payload={"inputs": x.cpu(), "targets": y.cpu(), "task_id": task_id, "classes": classes, "splits": splits, "split_percentages": percentages, "split_source": split_source, "options": opts}
    if image_task and original_sizes:
        payload["original_sizes"]=torch.tensor(original_sizes,dtype=torch.int64)
    if boxes is not None: payload["boxes"]=[value.detach().cpu().float() for value in boxes]
    if task_id in {"image.localization","image.detection"}:
        original_sizes=original_sizes or [(int(x.shape[-1]),int(x.shape[-2])) for _ in range(len(x))]
        spatial_affine=spatial_affine if spatial_affine is not None else torch.tensor([[1.,1.,0.,0.]]*len(x),dtype=torch.float32)
        payload["original_sizes"]=torch.tensor(original_sizes,dtype=torch.int64)
        payload["spatial_affine"]=spatial_affine.detach().cpu().float()
    if raw_inputs is not None: payload["raw_inputs"]=raw_inputs
    torch.save(payload, root / "source.pt")
    torch.save(payload, root / "dataset.pt")
    if task_id == "text.language_model":
        output_shape = [int(y.shape[1]), int(opts.get("vocab_size", 2048))]
    elif "segmentation.binary" in task_id:
        output_shape = [1, *list(y.shape[1:])]
    elif "segmentation.multiclass" in task_id:
        output_shape = [len(classes or []), *list(y.shape[1:])]
    elif task_id == "image.detection":
        output_shape = [len(classes or [])]
    else:
        output_shape = list(y.shape[1:]) or ([len(classes)] if classes else [1])
    
    display_y=_display_targets(y,opts)
    preview = _build_detection_preview(x, boxes, classes or []) if boxes is not None else _build_preview(task_id, x, display_y, classes, opts)
    summary = {
        "id": dataset_id, "revision": 1, "source": source, "samples": len(x),
        "inputShape": list(x.shape[1:]), "outputShape": output_shape,
        "classes": classes, "description": description, "splits": percentages,
        "splitCounts": counts, "splitSource": split_source,
        "path": str(root / "dataset.pt"), "seed": seed,
        "preview": preview, "analytics": build_analytics(task_id,x,display_y,classes,opts),
        "options": opts
    }
    if image_task:
        summary["imageResolution"]=_image_resolution_stats(original_sizes or [],len(x))
    if boxes is not None:
        distribution=[0 for _ in (classes or [])]
        for sample_boxes in boxes:
            for row in sample_boxes:
                class_id=int(row[0])
                if 0<=class_id<len(distribution): distribution[class_id]+=1
        summary["analytics"]["objectDistribution"]={"labels":classes or [],"values":distribution}
    if source_path:
        summary["sourcePath"] = source_path
    if metadata:
        summary.update(metadata)
    atomic_json(root / "manifest.json", {**summary, "split_indices": splits, "task_id": task_id, "schema_version": 1})
    return summary


def _image_resolution_stats(sizes, sample_count: int) -> dict:
    if isinstance(sizes,torch.Tensor): sizes=sizes.tolist()
    dimensions=[]
    for size in sizes or []:
        try: width,height=int(size[0]),int(size[1])
        except (TypeError,ValueError,IndexError): continue
        if width>0 and height>0: dimensions.append((width,height))
    if not dimensions:
        return {"count":0,"unavailable":sample_count,"nonSquareCount":0,"min":None,"mean":None,"max":None}
    widths=[value[0] for value in dimensions];heights=[value[1] for value in dimensions]
    return {
        "count":len(dimensions),"unavailable":max(0,sample_count-len(dimensions)),
        "nonSquareCount":sum(width!=height for width,height in dimensions),
        "min":{"width":min(widths),"height":min(heights)},
        "mean":{"width":round(math.fsum(widths)/len(widths),2),"height":round(math.fsum(heights)/len(heights),2)},
        "max":{"width":max(widths),"height":max(heights)},
    }


def _build_detection_preview(x: torch.Tensor, boxes: list[torch.Tensor], classes: list[str], sample_indices: list[int] | None = None) -> dict:
    items=[]
    sample_indices = sample_indices if sample_indices is not None else list(range(min(8,len(x))))
    for index in sample_indices:
        image=x[index].float()/255 if x.dtype==torch.uint8 else x[index].float().clamp(0,1)
        if image.shape[0]==1: arr=(image[0].numpy()*255).astype(np.uint8); pil=Image.fromarray(arr,mode="L").convert("RGB")
        else: pil=Image.fromarray((image[:3].permute(1,2,0).numpy()*255).astype(np.uint8),mode="RGB")
        draw=ImageDraw.Draw(pil);width,height=pil.size
        for row in boxes[index]:
            cls,cx,cy,bw,bh=[float(value) for value in row]
            xy=(int((cx-bw/2)*width),int((cy-bh/2)*height),int((cx+bw/2)*width),int((cy+bh/2)*height))
            draw.rectangle(xy,outline=(40,220,130),width=max(1,width//128));draw.text((xy[0],max(0,xy[1]-12)),classes[int(cls)] if 0<=int(cls)<len(classes) else str(int(cls)),fill=(40,220,130))
        buffer=io.BytesIO();pil.save(buffer,format="PNG");items.append({"url":"data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode(),"label":f"{len(boxes[index])} objetos"})
    return {"type":"image","items":items,"sampleIndices":sample_indices}


def _pipeline_splits(n: int, seed: int, percentages: dict, strategy: str, y: torch.Tensor, task_id: str) -> dict[str, list[int]]:
    if strategy == "temporal":
        train_n=round(n*int(percentages["train"])/100);val_n=round(n*int(percentages["validation"])/100)
        train_n=max(1,min(n-2,train_n));val_n=max(1,min(n-train_n-1,val_n))
        return {"train":list(range(train_n)),"validation":list(range(train_n,train_n+val_n)),"test":list(range(train_n+val_n,n))}
    if strategy == "stratified" and "classification" in task_id and y.ndim == 1:
        result={"train":[],"validation":[],"test":[]};rng=random.Random(seed)
        for label in torch.unique(y).tolist():
            indices=torch.where(y==label)[0].tolist();rng.shuffle(indices);m=len(indices)
            a=max(1,min(m-2,round(m*int(percentages["train"])/100))) if m>=3 else max(1,m-1)
            b=max(a+1,min(m-1,a+round(m*int(percentages["validation"])/100))) if m>=3 else m
            result["train"].extend(indices[:a]);result["validation"].extend(indices[a:b]);result["test"].extend(indices[b:])
        if all(result.values()):
            for values in result.values(): rng.shuffle(values)
            return result
    return split_indices(n,seed,percentages)


def _reduce_dims(x: torch.Tensor, task_id: str) -> tuple[int, ...]:
    if task_id.startswith("image"):
        return (0,2,3)
    if task_id.startswith("sequence"):
        return (0,1)
    return (0,)


def _serializable_tensor(value: torch.Tensor) -> list | float:
    raw=value.detach().cpu().squeeze().tolist()
    return raw


def apply_pipeline(project: dict, dataset: dict, task_id: str, pipeline: dict) -> dict:
    """Compile and materialize a typed data pipeline as an immutable dataset revision."""
    nodes=[node for node in pipeline.get("nodes",[]) if node.get("enabled",True)]
    split_position=next((i for i,node in enumerate(nodes) if node.get("type")=="split"),-1)
    last_stage=0;stage_order={"source":0,"inspect":1,"curate":2,"split":3,"transform":4,"augment":5,"output":6}
    for index,node in enumerate(nodes):
        node_type=node.get("type")
        if split_position>=0 and node_type in {"normalize","image_normalize","clip"} and index < split_position:
            raise BackendError("PIPELINE_TRANSFORM_BEFORE_SPLIT", node.get("label", node_type))
        if node.get("category")=="augment" and node.get("scope")!="train":
            raise BackendError("PIPELINE_AUGMENTATION_SCOPE_TRAIN")
        if node_type=="random_flip" and task_id=="image.regression":
            raise BackendError("PIPELINE_RANDOM_FLIP_REGRESSION_TARGET")
        current_stage=stage_order.get(node.get("category"),last_stage)
        if current_stage<last_stage and node_type!="output": raise BackendError("PIPELINE_NODE_OUT_OF_ORDER", node.get("label", node_type))
        last_stage=max(last_stage,current_stage)

    current_path=Path(dataset["path"]).resolve();dataset_root=current_path.parent.parent
    base_path=dataset_root/"1"/"source.pt"
    if not base_path.exists(): base_path=dataset_root/"1"/"dataset.pt"
    if not base_path.exists(): base_path=current_path.parent/"source.pt"
    if not base_path.exists(): base_path=current_path
    payload=torch.load(base_path,weights_only=True);x=payload["inputs"].clone();y=payload["targets"].clone();classes=payload.get("classes");boxes=[value.clone() for value in payload.get("boxes",[])] or None;original_sizes=payload.get("original_sizes");spatial_affine=payload.get("spatial_affine")
    if task_id in {"image.localization","image.detection"} and spatial_affine is None: spatial_affine=torch.tensor([[1.,1.,0.,0.]]*len(x),dtype=torch.float32)
    split_source=str(payload.get("split_source",dataset.get("splitSource","generated")))
    managed=split_source in {"official","temporal"}
    diagnostics=[];fitted={};seed=int(dataset.get("seed",42));percentages={key:int(dataset.get("splits",{}).get(key,default)) for key,default in (("train",70),("validation",15),("test",15))};strategy=split_source if managed else "automatic";splits={key:list(value) for key,value in payload.get("splits",{}).items()} if managed else None

    def subset(keep: torch.Tensor | list[int]) -> None:
        nonlocal x,y,splits,boxes,original_sizes,spatial_affine
        indices=keep.nonzero(as_tuple=False).flatten().tolist() if isinstance(keep,torch.Tensor) and keep.dtype==torch.bool else [int(index) for index in keep]
        if splits is not None:
            old_to_new={old:new for new,old in enumerate(indices)}
            splits={name:[old_to_new[index] for index in values if index in old_to_new] for name,values in splits.items()}
        x,y=x[indices],y[indices]
        if boxes is not None: boxes=[boxes[index] for index in indices]
        if original_sizes is not None: original_sizes=original_sizes[indices]
        if spatial_affine is not None: spatial_affine=spatial_affine[indices]

    def ensure_splits():
        nonlocal splits,strategy
        if splits is None:
            resolved=strategy
            if resolved=="automatic": resolved="temporal" if task_id=="sequence.forecast" else ("stratified" if "classification" in task_id else "random")
            splits=_pipeline_splits(len(x),seed,percentages,resolved,y,task_id)

    for node in nodes:
        kind=node.get("type");props=node.get("properties",{});node_id=node.get("id")
        if kind=="validate":
            invalid=(~torch.isfinite(x.float()).reshape(len(x),-1).all(1))
            if y.dtype.is_floating_point: invalid |= ~torch.isfinite(y.float()).reshape(len(y),-1).all(1)
            count=int(invalid.sum())
            if count: diagnostics.append({"level":"error" if props.get("fail_on_error",True) else "warning","nodeId":node_id,"message":str(BackendError("DATA_SAMPLES_NON_FINITE", str(count)))})
            else: diagnostics.append({"level":"info","nodeId":node_id,"message":str(BackendError("DATA_SAMPLES_VALID", str(len(x))))})
            has_later_filter=any(later.get("type")=="finite_filter" for later in nodes[nodes.index(node)+1:])
            if count and props.get("fail_on_error",True) and not has_later_filter: raise BackendError("DATA_NONFINITE_VALIDATION_FAILED", str(count))
        elif kind=="finite_filter":
            keep=torch.isfinite(x.float()).reshape(len(x),-1).all(1)
            if y.dtype.is_floating_point: keep &= torch.isfinite(y.float()).reshape(len(y),-1).all(1)
            removed=int((~keep).sum());subset(keep);diagnostics.append({"level":"info","nodeId":node_id,"message":str(BackendError("DATA_SAMPLES_EXCLUDED_NON_FINITE", str(removed)))})
        elif kind=="deduplicate":
            seen=set();keep=[];include_target=bool(props.get("include_target",True))
            for index in range(len(x)):
                digest=hashlib.sha256(x[index].contiguous().numpy().tobytes()+(y[index].contiguous().numpy().tobytes() if include_target else b"")).digest()
                if digest not in seen: seen.add(digest);keep.append(index)
            removed=len(x)-len(keep);subset(keep);diagnostics.append({"level":"info","nodeId":node_id,"message":str(BackendError("DATA_SAMPLES_EXCLUDED_DUPLICATES", str(removed)))})
        elif kind=="split":
            if not managed:
                percentages={key:int(props.get(key,default)) for key,default in (("train",70),("validation",15),("test",15))};seed=int(props.get("seed",seed));strategy=str(props.get("strategy","automatic"))
                if sum(percentages.values())!=100 or min(percentages.values())<1: raise BackendError("SPLITS_INVALID_PERCENTAGES")
                if strategy=="temporal" and not task_id.startswith("sequence"): raise BackendError("SPLITS_TEMPORAL_ONLY_SEQUENCES")
                splits=_pipeline_splits(len(x),seed,percentages,strategy,y,task_id)
        elif kind in {"normalize","image_normalize"}:
            ensure_splits()
            if kind=="image_normalize" and x.dtype==torch.uint8: x=x.float()/255.0
            mode=str(props.get("mode","standard"));train=x[splits["train"]].float();dims=_reduce_dims(train,task_id)
            if mode=="standard":
                center=train.mean(dim=dims,keepdim=True);scale=train.std(dim=dims,keepdim=True).clamp_min(1e-7);x=(x.float()-center)/scale
                fitted[node_id]={"mode":mode,"mean":_serializable_tensor(center),"scale":_serializable_tensor(scale)}
            else:
                low=torch.amin(train,dim=dims,keepdim=True);high=torch.amax(train,dim=dims,keepdim=True);scale=(high-low).clamp_min(1e-7);x=(x.float()-low)/scale
                if mode=="minmax_sym": x=x*2-1
                fitted[node_id]={"mode":mode,"min":_serializable_tensor(low),"max":_serializable_tensor(high)}
        elif kind=="clip":
            ensure_splits()
            lower=float(props.get("lower",1))/100;upper=float(props.get("upper",99))/100
            if not 0<=lower<upper<=1: raise BackendError("PIPELINE_CLIP_INVALID_QUANTILES")
            train=x[splits["train"]].float();dims=_reduce_dims(train,task_id);flat=train
            if len(dims)>1:
                feature_axis=1 if task_id.startswith("image") else train.ndim-1;flat=train.movedim(feature_axis,-1).reshape(-1,train.shape[feature_axis]);lo=torch.quantile(flat,lower,dim=0);hi=torch.quantile(flat,upper,dim=0)
                shape=[1]*x.ndim;shape[feature_axis]=lo.numel();lo=lo.reshape(shape);hi=hi.reshape(shape)
            else: lo=torch.quantile(train,lower,dim=0,keepdim=True);hi=torch.quantile(train,upper,dim=0,keepdim=True)
            x=torch.maximum(torch.minimum(x.float(),hi),lo);fitted[node_id]={"lower":_serializable_tensor(lo),"upper":_serializable_tensor(hi)}
        elif kind=="resize":
            if not task_id.startswith("image"): raise BackendError("PIPELINE_RESIZE_IMAGES_ONLY")
            height=max(8,int(props.get("height",128)));width=max(8,int(props.get("width",128)));method=str(props.get("method","bilinear"));original_dtype=x.dtype
            kwargs={} if method=="nearest" else {"align_corners":False};x=F.interpolate(x.float(),size=(height,width),mode=method,**kwargs)
            if original_dtype==torch.uint8: x=x.round().clamp(0,255).to(torch.uint8)
            # Segmentation masks use nearest-neighbor to keep class IDs intact.
            if "segmentation" in task_id: y=F.interpolate(y.unsqueeze(1).float(),size=(height,width),mode="nearest")[:,0].to(y.dtype)
            elif task_id=="image.reconstruction":
                target_dtype=y.dtype;y=F.interpolate(y.float(),size=(height,width),mode=method,**kwargs);y=y.round().clamp(0,255).to(torch.uint8) if target_dtype==torch.uint8 else y
            elif task_id=="image.translation": y=F.interpolate(y.float(),size=(height,width),mode=method,**kwargs)
            # Localization targets and detection boxes are normalized to [0,1], so a
            # pure resize leaves their geometry unchanged. The source-to-model affine
            # also stays the same because it maps normalized coordinates.
        elif kind=="center_crop":
            if not task_id.startswith("image"): raise BackendError("PIPELINE_CROP_IMAGES_ONLY")
            height=max(8,int(props.get("height",96)));width=max(8,int(props.get("width",96)));source_h,source_w=x.shape[-2:]
            if height>source_h or width>source_w: raise BackendError("PIPELINE_CROP_TOO_LARGE", f"{width}×{height} > {source_w}×{source_h}")
            top=(source_h-height)//2;left=(source_w-width)//2;x=x[...,top:top+height,left:left+width]
            if spatial_affine is not None:
                scale_x=spatial_affine[:,0].clone();scale_y=spatial_affine[:,1].clone()
                spatial_affine[:,2]+=left/source_w*scale_x;spatial_affine[:,3]+=top/source_h*scale_y
                spatial_affine[:,0]*=width/source_w;spatial_affine[:,1]*=height/source_h
            if "segmentation" in task_id or task_id=="image.reconstruction": y=y[...,top:top+height,left:left+width]
            if task_id=="image.translation": y=y[...,top:top+height,left:left+width]
            if task_id=="image.localization":
                for index,target in enumerate(y):
                    x1=max(float(target[0]),left/source_w);x2=min(float(target[2]),(left+width)/source_w);y1=max(float(target[1]),top/source_h);y2=min(float(target[3]),(top+height)/source_h)
                    if x2<=x1 or y2<=y1: raise BackendError("PIPELINE_CROP_REMOVED_LOCALIZATION_BOX",str(index))
                    y[index]=torch.tensor([(x1-left/source_w)/(width/source_w),(y1-top/source_h)/(height/source_h),(x2-left/source_w)/(width/source_w),(y2-top/source_h)/(height/source_h)])
            if boxes is not None:
                left_n,right_n=left/source_w,(left+width)/source_w;top_n,bottom_n=top/source_h,(top+height)/source_h;updated=[]
                for sample_boxes in boxes:
                    rows=[]
                    for cls,cx,cy,bw,bh in sample_boxes.tolist():
                        x1=max(cx-bw/2,left_n);x2=min(cx+bw/2,right_n);y1=max(cy-bh/2,top_n);y2=min(cy+bh/2,bottom_n)
                        if x2<=x1 or y2<=y1: continue
                        rows.append([cls,((x1+x2)/2-left_n)/(right_n-left_n),((y1+y2)/2-top_n)/(bottom_n-top_n),(x2-x1)/(right_n-left_n),(y2-y1)/(bottom_n-top_n)])
                    updated.append(torch.tensor(rows,dtype=torch.float32).reshape(-1,5))
                boxes=updated
        elif kind=="convert_channels":
            if not task_id.startswith("image"): raise BackendError("PIPELINE_CHANNELS_IMAGES_ONLY")
            channels=int(props.get("channels",3))
            if channels==1 and x.shape[1]!=1: x=x[:,:3].float().mean(1,keepdim=True).to(x.dtype)
            elif channels==3 and x.shape[1]==1: x=x.repeat(1,3,1,1)
            if task_id=="image.reconstruction": y=x.clone()
        elif kind=="output":
            pass

    if len(x)<3: raise BackendError("PIPELINE_TOO_FEW_SAMPLES")
    ensure_splits()
    if managed:
        percentages=split_percentages(splits)
    revision=max(int(dataset.get("revision",1))+1,int(pipeline.get("revision",1)));root=dataset_root/str(revision);root.mkdir(parents=True,exist_ok=True)
    compiled={**pipeline,"revision":revision,"diagnostics":diagnostics,"fittedState":fitted}
    output=next((node for node in nodes if node.get("type")=="output"),{});output_props=output.get("properties",{});options={**payload.get("options",{}),"batchSize":int(output_props.get("batchSize",payload.get("options",{}).get("batchSize",32))),"workers":int(output_props.get("workers",0)),"pin_memory":bool(output_props.get("pin_memory",True)),"pipeline":compiled}
    stored={"inputs":x.cpu(),"targets":y.cpu(),"task_id":task_id,"classes":classes,"splits":splits,"split_percentages":percentages,"split_source":split_source,"options":options,"pipeline":compiled}
    if boxes is not None: stored["boxes"]=[value.cpu() for value in boxes]
    if original_sizes is not None: stored["original_sizes"]=original_sizes.cpu()
    if spatial_affine is not None: stored["spatial_affine"]=spatial_affine.cpu()
    torch.save(stored,root/"dataset.pt")
    if task_id=="image.reconstruction": output_shape=list(y.shape[1:])
    elif "segmentation.binary" in task_id: output_shape=[1,*list(y.shape[1:])]
    elif "segmentation.multiclass" in task_id: output_shape=[len(classes or []),*list(y.shape[1:])]
    else: output_shape=dataset["outputShape"]
    display_y=_display_targets(y,options)
    preview=_build_detection_preview(x,boxes,classes or []) if boxes is not None else _build_preview(task_id,x,display_y,classes,options)
    summary={**dataset,"revision":revision,"samples":len(x),"inputShape":list(x.shape[1:]),"outputShape":output_shape,"splits":percentages,"splitCounts":{name:len(splits.get(name,[])) for name in ("train","validation","test")},"splitSource":split_source,"path":str(root/"dataset.pt"),"seed":seed,"preview":preview,"analytics":build_analytics(task_id,x,display_y,classes,options),"pipeline":compiled,"options":options,"validation":{"errors":0,"warnings":sum(d["level"]=="warning" for d in diagnostics),"checks":len(diagnostics)}}
    if task_id.startswith("image") or "segmentation" in task_id:
        if original_sizes is not None: summary["imageResolution"]=_image_resolution_stats(original_sizes,len(x))
        elif dataset.get("imageResolution") is not None: summary["imageResolution"]=dataset["imageResolution"]
        else: summary["imageResolution"]=_image_resolution_stats([],len(x))
    atomic_json(root/"pipeline.json",compiled);atomic_json(root/"manifest.json",{**summary,"preview":summary["preview"],"split_indices":splits,"task_id":task_id,"schema_version":2})
    return summary


def generate(project: dict, task_id: str, seed: int = 42, options: dict | None = None) -> dict:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    opts = options or {}
    spec = task_spec(task_id)
    objective = spec["objective"]
    
    if task_id.startswith("tabular"):
        n, features = 450, 12
        x = torch.randn(n, features)
        norm = opts.get("normalization", "none")
        if norm != "none":
            x = _apply_normalization(x, norm)
        if objective == "classification":
            weights = torch.randn(features, 3)
            y = (x @ weights + .25 * torch.randn(n, 3)).argmax(1)
            return _save(project, task_id, x, y, "Patrones tabulares reproducibles", seed, ["Azul", "Violeta", "Verde"], opts)
        if objective == "reconstruction":
            return _save(project, task_id, x, x.clone(), "Reconstrucción tabular y espacio latente", seed, options=opts)
        y = (x[:, :3] * torch.tensor([1.8, -2.2, .7])).sum(1, keepdim=True) + .15 * torch.randn(n, 1)
        return _save(project, task_id, x, y, "Regresión tabular con ruido controlado", seed, options=opts)
        
    if task_id.startswith("sequence"):
        n, steps, features = 240, 48, 3
        t = torch.linspace(0, 2 * math.pi, steps)
        labels = torch.arange(n) % 3
        waves = []
        for i in range(n):
            f = 1 + int(labels[i]); phase = torch.rand(1).item() * math.pi
            base = torch.sin(f * t + phase)
            waves.append(torch.stack((base, torch.cos(f * t + phase), torch.linspace(0, 1, steps))) + .08 * torch.randn(3, steps))
        x = torch.stack(waves).transpose(1, 2)
        norm = opts.get("normalization", "none")
        if norm != "none":
            x = _apply_normalization(x, norm)
        if objective == "classification": return _save(project, task_id, x, labels.long(), "Ondas por frecuencia", seed, ["Baja", "Media", "Alta"], opts)
        if objective == "forecast":
            horizon = 8; y = torch.stack([torch.sin((1 + int(labels[i])) * torch.linspace(2 * math.pi, 2.35 * math.pi, horizon)) for i in range(n)]).unsqueeze(-1)
            return _save(project, task_id, x, y, "Contextos temporales y horizonte futuro", seed, options=opts)
        y = x[:, :, 0].abs().mean(1, keepdim=True)
        return _save(project, task_id, x, y, "Secuencias con objetivo continuo", seed, options=opts)

    if task_id.startswith("text."):
        vocab = ["<pad>", "<unk>", "modelo", "datos", "aprende", "clasifica", "serie", "imagen", "texto", "bien", "rápido", "local", "token", "contexto", "futuro", "señal"]
        length, n = int(opts.get("max_length", 64)), 300
        tokens = torch.zeros(n, length, dtype=torch.long)
        labels = torch.arange(n) % 3
        for i in range(n):
            generator = torch.Generator().manual_seed(seed + i)
            tokens[i] = torch.randint(2, len(vocab), (length,), generator=generator)
            tokens[i, 0] = 2 + int(labels[i])
        raw_texts=[" ".join(vocab[int(token)] for token in row if int(token)>1) for row in tokens]
        if objective == "classification":
            return _save(project, task_id, tokens, labels.long(), "Corpus sintético de patrones léxicos", seed, ["Modelo", "Datos", "Aprendizaje"], {**opts, "vocab": vocab}, raw_inputs=raw_texts)
        targets = tokens.roll(-1, dims=1); targets[:, -1] = 0
        return _save(project, task_id, tokens, targets, "Secuencias sintéticas para predecir el siguiente token", seed, vocab, {**opts, "vocab": vocab, "vocab_size": int(opts.get("vocab_size", 2048)), "max_length": length}, raw_inputs=raw_texts)
        
    # Image Tasks
    channels = int(opts.get("channels", 3))
    resolution = int(opts.get("resolution", 256)) # Default resolution 256
    n = 120
    images, targets = [], []
    segmentation = objective.startswith("segmentation")
    classes = ["Círculo", "Cuadrado", "Triángulo"]
    margin = int(resolution * 0.12)
    max_pos = int(resolution * 0.5)
    ext_min = int(resolution * 0.25)
    ext_max = int(resolution * 0.42)
    
    for i in range(n):
        cls = i % 3
        rng = random.Random(seed + i)
        x0, y0 = rng.randint(margin, max_pos), rng.randint(margin, max_pos)
        extent = rng.randint(ext_min, ext_max)
        
        mode = "L" if channels == 1 else "RGB"
        bg_color = 20 if channels == 1 else (8, 12, 18)
        canvas = Image.new(mode, (resolution, resolution), bg_color)
        mask = Image.new("L", (resolution, resolution), 0)
        draw, mdraw = ImageDraw.Draw(canvas), ImageDraw.Draw(mask)
        
        if channels == 1:
            color = [215, 180, 140][cls]
        else:
            color = [(35, 190, 215), (150, 110, 235), (65, 200, 125)][cls]
            
        box = (x0, y0, min(resolution - 2, x0 + extent), min(resolution - 2, y0 + extent))
        if cls == 0:
            draw.ellipse(box, fill=color)
            mdraw.ellipse(box, fill=1 if objective == "segmentation_binary" else cls + 1)
        elif cls == 1:
            draw.rectangle(box, fill=color)
            mdraw.rectangle(box, fill=1 if objective == "segmentation_binary" else cls + 1)
        else:
            pts = [(x0 + extent // 2, y0), (x0, y0 + extent), (x0 + extent, y0 + extent)]
            draw.polygon(pts, fill=color)
            mdraw.polygon(pts, fill=1 if objective == "segmentation_binary" else cls + 1)
            
        if channels == 1:
            arr = np.array(canvas)[:, :, None] # [H, W, 1]
        else:
            arr = np.array(canvas) # [H, W, 3]
            
        clean_image=torch.from_numpy(arr).permute(2, 0, 1).float() / 255
        images.append(clean_image)
        if segmentation:
            targets.append(torch.from_numpy(np.array(mask)).long())
        elif objective == "classification":
            targets.append(cls)
        elif objective=="localization":
            targets.append([box[0]/resolution,box[1]/resolution,box[2]/resolution,box[3]/resolution])
        elif objective=="detection":
            targets.append(torch.tensor([[cls,(box[0]+box[2])/(2*resolution),(box[1]+box[3])/(2*resolution),(box[2]-box[0])/resolution,(box[3]-box[1])/resolution]],dtype=torch.float32))
        elif objective=="image_translation":
            targets.append(clean_image)
        else:
            targets.append(extent * extent / (resolution * resolution))
            
    x = torch.stack(images)
    opts_out = {"channels": channels, "resolution": resolution}
    if segmentation:
        y = torch.stack(targets)
        cls_names = ["Objeto"] if objective == "segmentation_binary" else ["Fondo", *classes]
        return _save(project, task_id, x, y, "Formas y máscaras perfectamente alineadas", seed, cls_names, opts_out)
    if objective == "reconstruction":
        return _save(project, task_id, x, x.clone(), "Reconstrucción de imágenes geométricas", seed, options=opts_out)
    if objective=="localization":
        y=torch.tensor(targets,dtype=torch.float32);return _save(project,task_id,x,y,"Cajas sintéticas normalizadas",seed,options={**opts_out,"boxFormat":"normalized_xyxy"})
    if objective=="detection":
        boxes=targets;y=torch.zeros((len(x),0),dtype=torch.float32);return _save(project,task_id,x,y,"Detección sintética con una caja por imagen",seed,classes,opts_out,boxes=boxes)
    if objective=="image_translation":
        noise=torch.randn_like(x)*.08;noisy=(x+noise).clamp(0,1);return _save(project,task_id,noisy,x.clone(),"Pares sintéticos ruidoso→limpio",seed,options={**opts_out,"targetChannels":channels,"normalization":"none"})
    y = torch.tensor(targets, dtype=torch.long if objective == "classification" else torch.float32)
    if objective != "classification":
        y = y.unsqueeze(1)
    return _save(project, task_id, x, y, "Formas geométricas con variación espacial", seed, classes if objective == "classification" else None, opts_out)


def _image_partition_directories(root: Path, *, classification: bool = False) -> list[tuple[Path, str]]:
    """Recognize partitions without mistaking a flat class named train for a split."""
    aliases = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
    candidates = [(root / name, split) for name, split in aliases.items() if (root / name).is_dir()]
    if classification:
        candidates = [(folder, split) for folder, split in candidates if any(
            child.is_dir() and any(_is_image(file) for file in child.rglob("*"))
            for child in folder.iterdir()
        )]
    elif not any(any(_is_image(file) for file in folder.rglob("*")) for folder, _ in candidates):
        return []
    if len({split for _, split in candidates}) != len(candidates):
        raise BackendError("IMPORT_IMAGE_DUPLICATE_PARTITION", "val / validation")
    if candidates:
        recognized = {folder for folder, _ in candidates}
        for child in root.iterdir():
            if _is_image(child) or (child.is_dir() and child not in recognized and any(_is_image(file) for file in child.rglob("*"))):
                raise BackendError("IMPORT_IMAGE_MIXED_PARTITIONS", child.name)
    return candidates


def import_dataset(project: dict, task_id: str, path: str, options: dict | None = None) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.exists(): raise BackendError("IMPORT_PATH_NOT_FOUND")
    opts = options or {}
    spec = task_spec(task_id)
    if task_id.startswith("tabular"):
        if source.suffix.lower() not in {".csv", ".tsv"}: raise BackendError("IMPORT_TABULAR_REQUIRES_CSV")
        with source.open(newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(65536); handle.seek(0)
            reader = csv.DictReader(handle, dialect=_csv_dialect(sample, source))
            reader.fieldnames = [str(column).strip() for column in (reader.fieldnames or [])]
            rows = list(reader)
        if len(rows) < 10: raise BackendError("IMPORT_CSV_TOO_FEW_ROWS")
        cols = list(rows[0])
        requested_target = opts.get("target_column")
        target = None if spec["objective"] == "reconstruction" else requested_target or ("target" if "target" in cols else cols[-1])
        if target is not None and target not in cols:
            raise BackendError("IMPORT_TARGET_COLUMN_NOT_FOUND", str(target))
        requested_features = opts.get("feature_columns")
        feature_cols = list(requested_features) if requested_features is not None else [col for col in cols if col != target]
        if any(column not in cols for column in feature_cols) or target in feature_cols:
            raise BackendError("IMPORT_FEATURE_COLUMNS_INVALID")
        if not feature_cols:
            raise BackendError("IMPORT_FEATURE_COLUMNS_EMPTY")
        try: x = torch.tensor([[float(r[c]) for c in feature_cols] for r in rows], dtype=torch.float32)
        except (ValueError, KeyError) as exc: raise BackendError("DATASET_NUMERIC_REQUIRED") from exc
        x, opts, import_splits = _fit_import_normalization(x, opts)
        if spec["objective"] == "classification":
            names = sorted({r[target] for r in rows}); mapping = {v:i for i,v in enumerate(names)}; y=torch.tensor([mapping[r[target]] for r in rows])
            result=_save(project,task_id,x,y,f"CSV importado: {source.name}",42,names,opts,split_indices_override=import_splits)
        elif spec["objective"] == "reconstruction":
            # En reconstrucción no hay una columna objetivo: si existe `target`, se
            # excluye como arriba y se reconstruyen únicamente las variables.
            result=_save(project,task_id,x,x.clone(),f"CSV para reconstrucción: {source.name}",42,options=opts,split_indices_override=import_splits)
        else:
            y=torch.tensor([[float(r[target])] for r in rows]); result=_save(project,task_id,x,y,f"CSV importado: {source.name}",42,options=opts,split_indices_override=import_splits)
        return _mark_imported(result, source)
    if task_id.startswith("text."):
        if source.suffix.lower() not in {".txt", ".csv", ".tsv"}: raise BackendError("IMPORT_TEXT_INVALID_EXTENSION")
        raw = source.read_text(encoding="utf-8-sig").splitlines()
        if task_id == "text.classification":
            pairs=[]
            for line in raw:
                parts=line.split("\t",1)
                if len(parts)==2 and parts[0].strip() and parts[1].strip(): pairs.append((parts[0].strip(),parts[1].strip()))
            if len(pairs)<10: raise BackendError("IMPORT_TEXT_CLASSIFICATION_TOO_FEW_LINES")
            label_names=sorted({label for label,_ in pairs}); label_map={label:i for i,label in enumerate(label_names)}; texts=[value for _,value in pairs]; y=torch.tensor([label_map[label] for label,_ in pairs])
        else:
            texts=[line.strip() for line in raw if line.strip()]
            if len(texts)<10: raise BackendError("IMPORT_TEXT_CORPUS_TOO_FEW_LINES")
            label_names=[]; y=None
        words=[re.findall(r"\w+|[^\w\s]", value.lower(), flags=re.UNICODE) for value in texts]
        counts={}
        for row in words:
            for word in row: counts[word]=counts.get(word,0)+1
        max_vocab=int(opts.get("vocab_size",2048)); vocab=["<pad>","<unk>"]+[word for word,_ in sorted(counts.items(),key=lambda item:(-item[1],item[0]))[:max_vocab-2]]; lookup={word:i for i,word in enumerate(vocab)}; length=int(opts.get("max_length",64))
        x=torch.zeros(len(words),length,dtype=torch.long)
        for i,row in enumerate(words):
            ids=[lookup.get(word,1) for word in row[:length]]; x[i,:len(ids)]=torch.tensor(ids)
        if task_id == "text.language_model": y=x.roll(-1,dims=1);y[:,-1]=0;classes=vocab
        else: classes=label_names
        return _mark_imported(_save(project,task_id,x,y,f"Corpus importado: {source.name}",42,classes,{**opts,"vocab":vocab,"max_length":length},raw_inputs=texts),source)
    if task_id.startswith("sequence"):
        if source.suffix.lower() != ".csv": raise BackendError("IMPORT_SEQUENCE_REQUIRES_CSV")
        with source.open(newline="", encoding="utf-8-sig") as handle: rows = list(csv.DictReader(handle))
        required = {"sequence_id", "timestep"}
        if not rows or not required.issubset(rows[0]): raise BackendError("IMPORT_SEQUENCE_MISSING_COLUMNS")
        groups: dict[str, list[dict]] = {}
        for row in rows: groups.setdefault(row["sequence_id"], []).append(row)
        for values in groups.values(): values.sort(key=lambda row: float(row["timestep"]))
        lengths = {len(values) for values in groups.values()}
        if len(lengths) != 1 or min(lengths) < 2: raise BackendError("IMPORT_SEQUENCE_INCONSISTENT_LENGTH")
        future_cols = sorted([col for col in rows[0] if col.startswith("future_")], key=lambda value: int(value.split("_")[-1]))
        excluded = {"sequence_id", "timestep", "target", *future_cols}
        features = [col for col in rows[0] if col not in excluded]
        if not features: raise BackendError("IMPORT_SEQUENCE_NO_FEATURES")
        try: x = torch.tensor([[[float(row[col]) for col in features] for row in values] for values in groups.values()], dtype=torch.float32)
        except ValueError as exc: raise BackendError("IMPORT_SEQUENCE_NUMERIC_REQUIRED") from exc
        x, opts, import_splits = _fit_import_normalization(x, opts)
        last_rows = [values[-1] for values in groups.values()]
        if spec["objective"] == "classification":
            if "target" not in rows[0]: raise BackendError("IMPORT_SEQUENCE_CLASSIFICATION_TARGET_MISSING")
            names=sorted({row["target"] for row in last_rows});mapping={name:i for i,name in enumerate(names)};y=torch.tensor([mapping[row["target"]] for row in last_rows]);result=_save(project,task_id,x,y,f"Secuencias importadas: {source.name}",42,names,opts,split_indices_override=import_splits)
        elif spec["objective"] == "forecast":
            if not future_cols: raise BackendError("IMPORT_SEQUENCE_FORECAST_FUTURE_MISSING")
            y=torch.tensor([[[float(row[col])] for col in future_cols] for row in last_rows]);result=_save(project,task_id,x,y,f"Pronóstico importado: {source.name}",42,options=opts,split_indices_override=import_splits)
        else:
            if "target" not in rows[0]: raise BackendError("IMPORT_SEQUENCE_REGRESSION_TARGET_MISSING")
            y=torch.tensor([[float(row["target"])] for row in last_rows]);result=_save(project,task_id,x,y,f"Secuencias importadas: {source.name}",42,options=opts,split_indices_override=import_splits)
        return _mark_imported(result, source)
    if task_id in {"image.classification", "image.regression", "image.reconstruction", "image.localization", "image.detection", "image.translation"}:
        root = source if source.is_dir() else source.parent
        resolution = int(opts.get("resolution", 64))
        channels = int(opts.get("channels", 3))
        if task_id == "image.translation":
            inputs_dir=root/"inputs";targets_dir=root/"targets"
            if not inputs_dir.is_dir() or not targets_dir.is_dir(): raise BackendError("IMPORT_IMAGE_PAIRS_FOLDERS_MISSING")
            input_files=[file for file in inputs_dir.iterdir() if _is_image(file)];target_files=[file for file in targets_dir.iterdir() if _is_image(file)]
            input_by_id={file.stem:file for file in input_files};target_by_id={file.stem:file for file in target_files}
            if len(input_by_id)!=len(input_files) or len(target_by_id)!=len(target_files): raise BackendError("IMPORT_IMAGE_PAIRS_DUPLICATE_IDS")
            missing=sorted(set(input_by_id)^set(target_by_id))
            if missing: raise BackendError("IMPORT_IMAGE_PAIRS_UNMATCHED",", ".join(missing[:5]))
            ids=sorted(input_by_id)
            if len(ids)<3: raise BackendError("IMPORT_IMAGE_PAIRS_TOO_FEW")
            for key in ids:
                with Image.open(input_by_id[key]) as source_image,Image.open(target_by_id[key]) as target_image:
                    if source_image.size!=target_image.size: raise BackendError("IMPORT_IMAGE_PAIRS_SIZE_MISMATCH",key)
            target_channels=int(opts.get("target_channels",channels))
            x=_load_images_parallel([input_by_id[key] for key in ids],resolution,channels)
            y=_load_images_parallel([target_by_id[key] for key in ids],resolution,target_channels)
            result=_save(project,task_id,x.float()/255,y.float()/255,f"Pares imagen→imagen importados: {root.name}",42,options={"channels":channels,"targetChannels":target_channels,"resolution":resolution,"normalization":"none"},original_sizes=_image_sizes([input_by_id[key] for key in ids]))
            return _mark_imported(result,source)
        if task_id == "image.localization":
            labels_path=root/"labels.csv"
            if not labels_path.is_file(): raise BackendError("IMPORT_LOCALIZATION_LABELS_MISSING")
            with labels_path.open(newline="",encoding="utf-8-sig") as handle: rows=list(csv.DictReader(handle))
            if not rows or not {"filename","xmin","ymin","xmax","ymax"}.issubset(rows[0]): raise BackendError("IMPORT_LOCALIZATION_LABELS_INVALID")
            paths=[(root/row["filename"]).resolve() for row in rows]
            if not all(path.is_file() and _is_image(path) for path in paths): raise BackendError("IMPORT_IMAGE_REGRESSION_IMAGE_INVALID")
            targets=[];original_sizes=[]
            for path,row in zip(paths,rows):
                try: x1,y1,x2,y2=(float(row[key]) for key in ("xmin","ymin","xmax","ymax"))
                except (TypeError,ValueError) as exc: raise BackendError("IMPORT_LOCALIZATION_BOX_INVALID",path.name) from exc
                with Image.open(path) as image: width,height=image.size
                original_sizes.append((width,height))
                x1,y1,x2,y2=max(0.,x1),max(0.,y1),min(float(width),x2),min(float(height),y2)
                if not all(math.isfinite(value) for value in (x1,y1,x2,y2)) or x2<=x1 or y2<=y1: raise BackendError("IMPORT_LOCALIZATION_BOX_INVALID",path.name)
                targets.append([x1/width,y1/height,x2/width,y2/height])
            if len(paths)<3: raise BackendError("IMPORT_LOCALIZATION_TOO_FEW")
            x=_load_images_parallel(paths,resolution,channels);y=torch.tensor(targets,dtype=torch.float32)
            result=_save(project,task_id,x,y,f"Cajas de localización importadas: {root.name}",42,options={"channels":channels,"resolution":resolution,"boxFormat":"normalized_xyxy"},original_sizes=original_sizes)
            return _mark_imported(result,source)
        if task_id == "image.detection":
            images_dir=root/"images" if (root/"images").is_dir() else root
            partitions = _image_partition_directories(images_dir)
            image_paths = []
            imported_splits = {"train": [], "validation": [], "test": []} if partitions else None
            if partitions:
                for directory, split in partitions:
                    paths = sorted(file for file in directory.rglob("*") if _is_image(file))
                    imported_splits[split].extend(range(len(image_paths), len(image_paths) + len(paths)))
                    image_paths.extend(paths)
            else:
                image_paths=sorted(file for file in images_dir.iterdir() if _is_image(file))
            if len(image_paths)<3: raise BackendError("IMPORT_DETECTION_TOO_FEW_IMAGES")
            class_file=root/"classes.txt";class_names=[line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()] if class_file.is_file() else []
            annotations:dict[str,list[tuple[int,float,float,float,float]]]={}
            json_file=root/"annotations.json"
            if json_file.is_file():
                try: records=json.loads(json_file.read_text(encoding="utf-8"))
                except (OSError,json.JSONDecodeError) as exc: raise BackendError("IMPORT_DETECTION_JSON_INVALID") from exc
                if isinstance(records,dict): records=records.get("images",[])
                if not isinstance(records,list): raise BackendError("IMPORT_DETECTION_JSON_INVALID")
                for record in records:
                    name=str(record.get("file_name",record.get("filename",""))).removeprefix("images/");values=[]
                    image_path=(images_dir/name).resolve()
                    if not name or images_dir.resolve() not in image_path.parents or not image_path.is_file(): raise BackendError("IMPORT_DETECTION_IMAGE_UNKNOWN",name)
                    with Image.open(image_path) as image: width,height=image.size
                    for item in record.get("objects",record.get("annotations",[])):
                        label=item.get("class_id",item.get("class"));bbox=item.get("bbox")
                        if isinstance(label,str):
                            if label not in class_names: class_names.append(label)
                            label=class_names.index(label)
                        try: cls=int(label);x1,y1,x2,y2=map(float,bbox)
                        except (TypeError,ValueError) as exc: raise BackendError("IMPORT_DETECTION_ANNOTATION_INVALID",name) from exc
                        values.append((cls,(x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height))
                    annotations[name]=values
            else:
                labels_dir=root/"labels"
                if not labels_dir.is_dir(): raise BackendError("IMPORT_DETECTION_LABELS_FOLDER_MISSING")
                for image_path in image_paths:
                    relative = image_path.relative_to(images_dir)
                    label_path=labels_dir/relative.with_suffix(".txt");values=[]
                    if partitions and not label_path.is_file():
                        raise BackendError("IMPORT_DETECTION_LABEL_MISSING", relative.as_posix())
                    if label_path.is_file():
                        for line_number,line in enumerate(label_path.read_text(encoding="utf-8").splitlines(),1):
                            if not line.strip(): continue
                            fields=line.split()
                            if len(fields)!=5: raise BackendError("IMPORT_DETECTION_YOLO_LINE_INVALID",f"{label_path.name}:{line_number}")
                            try: cls=int(fields[0]);cx,cy,bw,bh=map(float,fields[1:])
                            except ValueError as exc: raise BackendError("IMPORT_DETECTION_YOLO_LINE_INVALID",f"{label_path.name}:{line_number}") from exc
                            values.append((cls,cx,cy,bw,bh))
                    annotations[relative.as_posix()]=values
            max_class=max((item[0] for values in annotations.values() for item in values),default=-1)
            if not class_names: class_names=[f"Clase {index}" for index in range(max_class+1)]
            if max_class>=len(class_names): raise BackendError("IMPORT_DETECTION_CLASS_UNKNOWN",str(max_class))
            boxes=[]
            for image_path in image_paths:
                values=[]
                for cls,cx,cy,bw,bh in annotations.get(image_path.relative_to(images_dir).as_posix(),[]):
                    if cls<0 or cls>=len(class_names) or not all(math.isfinite(value) for value in (cx,cy,bw,bh)) or bw<=0 or bh<=0 or cx<0 or cy<0 or cx>1 or cy>1 or bw>1 or bh>1 or cx-bw/2 < -1e-7 or cy-bh/2 < -1e-7 or cx+bw/2 > 1+1e-7 or cy+bh/2 > 1+1e-7:
                        raise BackendError("IMPORT_DETECTION_BOX_OUT_OF_RANGE",image_path.name)
                    values.append([cls,cx,cy,bw,bh])
                boxes.append(torch.tensor(values,dtype=torch.float32).reshape(-1,5))
            if not class_names: raise BackendError("IMPORT_DETECTION_CLASSES_MISSING")
            original_sizes=[]
            for image_path in image_paths:
                with Image.open(image_path) as image: original_sizes.append(image.size)
            x=_load_images_parallel(image_paths,resolution,channels);y=torch.zeros((len(x),0),dtype=torch.float32)
            result=_save(project,task_id,x,y,f"Cajas de detección importadas: {root.name}",42,class_names,{"channels":channels,"resolution":resolution,"boxFormat":"normalized_cxcywh","annotationFormat":"yolo_txt_or_xyxy_json"},boxes=boxes,original_sizes=original_sizes,split_indices_override=imported_splits,split_source="official" if partitions else "generated")
            return _mark_imported(result,source)
        if task_id.endswith("classification"):
            partitions = _image_partition_directories(root, classification=True)
            imported_splits = {"train": [], "validation": [], "test": []} if partitions else None
            if partitions:
                names = sorted({directory.name for folder, _ in partitions for directory in folder.iterdir()
                                if directory.is_dir() and any(_is_image(file) for file in directory.rglob("*"))})
                if len(names) < 2: raise BackendError("IMPORT_IMAGE_CLASSIFICATION_MIN_CLASSES")
                image_paths=[]; labels=[]
                for folder, split in partitions:
                    for label, name in enumerate(names):
                        directory = folder / name
                        paths = sorted(file for file in directory.rglob("*") if _is_image(file)) if directory.is_dir() else []
                        imported_splits[split].extend(range(len(image_paths), len(image_paths) + len(paths)))
                        image_paths.extend(paths); labels.extend([label] * len(paths))
            else:
                class_dirs = sorted(item for item in root.iterdir() if item.is_dir() and any(_is_image(file) for file in item.iterdir()))
                if len(class_dirs) < 2: raise BackendError("IMPORT_IMAGE_CLASSIFICATION_MIN_CLASSES")
                names = [item.name for item in class_dirs]
                image_paths=[]; labels=[]
                for label, directory in enumerate(class_dirs):
                    for file in sorted(directory.iterdir()):
                        if _is_image(file): image_paths.append(file); labels.append(label)
            x=_load_images_parallel(image_paths, resolution, channels);y=torch.tensor(labels);result=_save(project,task_id,x,y,f"Imágenes importadas: {root.name}",42,names,{"channels": channels, "resolution": resolution},original_sizes=_image_sizes(image_paths),split_indices_override=imported_splits,split_source="official" if partitions else "generated")
        elif task_id.endswith("regression"):
            labels_path=root/"labels.csv"
            if not labels_path.is_file(): raise BackendError("IMPORT_IMAGE_REGRESSION_LABELS_MISSING")
            with labels_path.open(newline="",encoding="utf-8-sig") as handle: rows=list(csv.DictReader(handle))
            if not rows or not {"filename","target"}.issubset(rows[0]): raise BackendError("IMPORT_IMAGE_REGRESSION_LABELS_INVALID")
            paths=[(root/row["filename"]).resolve() for row in rows]
            if not all(path.is_file() and _is_image(path) for path in paths): raise BackendError("IMPORT_IMAGE_REGRESSION_IMAGE_INVALID")
            x=_load_images_parallel(paths, resolution, channels);y=torch.tensor([[float(row["target"])] for row in rows]);result=_save(project,task_id,x,y,f"Regresión visual importada: {root.name}",42,options={"channels": channels, "resolution": resolution},original_sizes=_image_sizes(paths))
        else:
            image_paths=sorted(file for file in root.iterdir() if _is_image(file))
            if len(image_paths)<10: raise BackendError("IMPORT_IMAGE_RECONSTRUCTION_MIN_IMAGES")
            x=_load_images_parallel(image_paths,resolution,channels).float()/255.0;y=x.clone();result=_save(project,task_id,x,y,f"Imágenes para reconstrucción: {root.name}",42,options={"channels":channels,"resolution":resolution},original_sizes=_image_sizes(image_paths))
        return _mark_imported(result, source)
    if "segmentation" in task_id:
        root=source if source.is_dir() else source.parent;images_dir=root/"images";masks_dir=root/"masks"
        resolution = int(opts.get("resolution", 64))
        channels = int(opts.get("channels", 3))
        if not images_dir.is_dir() or not masks_dir.is_dir(): raise BackendError("IMPORT_SEGMENTATION_FOLDERS_MISSING")
        pairs=[]
        for image_path in sorted(file for file in images_dir.iterdir() if _is_image(file)):
            candidates=[file for file in masks_dir.glob(f"{image_path.stem}.*") if _is_image(file)]
            if len(candidates)!=1: raise BackendError("IMPORT_SEGMENTATION_MASK_MISSING", image_path.name)
            pairs.append((image_path,candidates[0]))
        if len(pairs)<10: raise BackendError("IMPORT_SEGMENTATION_MIN_PAIRS")
        x=torch.stack([_load_image(image, resolution, channels) for image,_ in pairs]);raw_masks=[torch.from_numpy(np.array(Image.open(mask).convert("L").resize((resolution,resolution),Image.Resampling.NEAREST))).long() for _,mask in pairs];original_sizes=_image_sizes([image for image,_ in pairs])
        if task_id.endswith("binary"):
            y=torch.stack([(mask>0).long() for mask in raw_masks]);classes=["Objeto"]
        else:
            values=sorted({int(value) for mask in raw_masks for value in torch.unique(mask)})
            if len(values)<2: raise BackendError("IMPORT_SEGMENTATION_MULTICLASS_MIN_CLASSES")
            mapping={value:index for index,value in enumerate(values)};y=torch.stack([torch.tensor(np.vectorize(mapping.get)(mask.numpy()),dtype=torch.long) for mask in raw_masks]);classes=["Fondo" if index==0 else f"Clase {value}" for index,value in enumerate(values)]
        result=_save(project,task_id,x,y,f"Segmentación importada: {root.name}",42,classes,options={"channels": channels, "resolution": resolution},original_sizes=original_sizes);return _mark_imported(result,source)
    raise BackendError("IMPORT_TASK_UNSUPPORTED")


def inspect_csv(path: str) -> list[str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in {".csv", ".tsv"}:
        raise BackendError("IMPORT_TABULAR_REQUIRES_CSV")
    try:
        with source.open(newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(65536); handle.seek(0)
            columns = next(csv.reader(handle, dialect=_csv_dialect(sample, source)), [])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BackendError("IMPORT_CSV_HEADER_INVALID", source.name) from exc
    if not columns or any(not str(column).strip() for column in columns):
        raise BackendError("IMPORT_CSV_HEADER_INVALID", source.name)
    return [str(column).strip() for column in columns]


def _csv_dialect(sample: str, source: Path):
    if source.suffix.lower() == ".tsv":
        return csv.excel_tab
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;|\t")
    except csv.Error:
        return csv.excel


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _image_sizes(paths: list[Path]) -> list[tuple[int,int]]:
    sizes=[]
    for path in paths:
        with Image.open(path) as image: sizes.append(image.size)
    return sizes


def _load_image(path: Path, resolution: int = 256, channels: int = 3) -> torch.Tensor:
    try:
        mode = "L" if channels == 1 else "RGB"
        image = Image.open(path).convert(mode).resize((resolution, resolution), Image.Resampling.BILINEAR)
    except Exception as exc:
        raise BackendError("IMPORT_IMAGE_READ_FAILED", path.name) from exc
    if channels == 1:
        arr = np.array(image)[:, :, None]
    else:
        arr = np.array(image)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_images_parallel(files: list[Path], resolution: int = 256, channels: int = 3) -> torch.Tensor:
    def _worker(file_path: Path):
        return _load_image(file_path, resolution, channels)
    with ThreadPoolExecutor() as executor:
        tensors = list(executor.map(_worker, files))
    return torch.stack(tensors)


def _mark_imported(result: dict, source: Path) -> dict:
    result["source"]="imported";result["sourcePath"]=str(source)
    manifest_path=Path(result["path"]).parent/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"));manifest.update({"source":"imported","sourcePath":str(source)});atomic_json(manifest_path,manifest)
    return result
