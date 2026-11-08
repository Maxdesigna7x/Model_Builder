from __future__ import annotations

import base64
import io
import json
import random
import re
import sys
import traceback
from pathlib import Path
import uuid

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modelbuilder.data import apply_pipeline, build_analytics, generate, import_dataset, inspect_csv, split_indices, _build_preview, _build_detection_preview
    from modelbuilder.catalog import catalog as authoritative_catalog, task_spec
    from modelbuilder.errors import BackendError
    from modelbuilder.huggingface_datasets import catalog as huggingface_catalog, download as download_huggingface_dataset
    from modelbuilder.models import build_model
    from modelbuilder.manifest import semantic_hash
    from modelbuilder.pretrained import catalog as pretrained_catalog, record_license_acceptance, scratch_guidance
    from modelbuilder.pretrained import tokenize_texts as hf_tokenize_texts
    from modelbuilder.pretrained import decode_tokens as hf_decode_tokens
    from modelbuilder.storage import create_project, delete_project, list_projects, open_project, project_dir, save_model_revision, save_project_state, workspace_root
    from modelbuilder.training import _loss, decode_detections, decode_localization, train
else:
    from .catalog import catalog as authoritative_catalog, task_spec
    from .data import apply_pipeline, build_analytics, generate, import_dataset, inspect_csv, split_indices, _build_preview, _build_detection_preview
    from .errors import BackendError
    from .huggingface_datasets import catalog as huggingface_catalog, download as download_huggingface_dataset
    from .models import build_model
    from .manifest import semantic_hash
    from .pretrained import catalog as pretrained_catalog, record_license_acceptance, scratch_guidance
    from .pretrained import tokenize_texts as hf_tokenize_texts
    from .pretrained import decode_tokens as hf_decode_tokens
    from .storage import create_project, delete_project, list_projects, open_project, project_dir, save_model_revision, save_project_state, workspace_root
    from .training import _loss, decode_detections, decode_localization, train


def response(result=None, error=None):
    value = {"ok": error is None}
    if error is None:
        value["result"] = result
    else:
        value["error"] = {"code": type(error).__name__, "message": str(error)}
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def handle(request: dict):
    action = request.get("action")
    if action == "hello":
        return {"version": "0.1.0", "torch": torch.__version__, "cuda": torch.cuda.is_available(), "workspace": str(workspace_root())}
    if action == "catalog.get":
        return authoritative_catalog()
    if action == "project.create":
        return create_project(request["project"])
    if action == "project.open":
        return open_project(request["path"])
    if action == "project.list":
        return list_projects()
    if action == "project.delete":
        return delete_project(request["project"])
    if action == "project.state.save":
        return save_project_state(request["project"], request["state"])
    if action == "model.revision.save":
        return save_model_revision(request["project"], request["model"])
    if action == "training.control":
        run_id=str(uuid.UUID(str(request["run_id"])));state=str(request["state"])
        if state not in {"running","paused","cancelled"}: raise BackendError("TRAINING_CONTROL_INVALID",state)
        path=project_dir(request["project"])/"runs"/run_id/"control.json"
        path.write_text(json.dumps({"state":state}),encoding="utf-8")
        return {"run_id":run_id,"state":state}
    if action == "data.generate":
        return generate(request["project"], request["task_id"], int(request.get("seed", 42)), request.get("options"))
    if action == "data.catalog":
        return huggingface_catalog(request.get("project"), request["task_id"])
    if action in {"data.kaggle.inspect", "data.kaggle.import"}:
        from modelbuilder.kaggle_datasets import inspect as inspect_kaggle, download as import_kaggle
        if action == "data.kaggle.inspect":
            return inspect_kaggle(request["handle"], request.get("options"))
        return import_kaggle(request["project"], request["task_id"], request["handle"], request.get("options"))
    if action == "model.catalog":
        return pretrained_catalog(request.get("task_id"),request.get("architecture"))
    if action == "model.catalog.guidance":
        return {"architecture":request["architecture"],"reason":scratch_guidance(request["architecture"])}
    if action == "model.license.accept":
        return record_license_acceptance(request["model_id"])
    if action == "model.bundle.import":
        bundle=Path(request["path"]).expanduser().resolve();config=bundle/"config.json";weights=bundle/"model.pt"
        if not config.is_file() or not weights.is_file(): raise BackendError("MODEL_BUNDLE_INVALID",str(bundle))
        value=json.loads(config.read_text(encoding="utf-8"));value.update({"bundle":str(bundle),"checkpoint":str(weights),"huggingface_path":str(bundle/"huggingface") if (bundle/"huggingface").is_dir() else None});return value
    if action == "data.download":
        return download_huggingface_dataset(request["project"], request["task_id"], request["dataset_id"], request.get("options"))
    if action == "data.import":
        return import_dataset(request["project"], request["task_id"], request["path"], request.get("options"))
    if action == "data.inspect_csv":
        return {"columns": inspect_csv(request["path"])}
    if action == "data.import.files":
        files = request.get("files") or []
        if not files:
            raise BackendError("IMPORT_UPLOAD_EMPTY")
        root = project_dir(request["project"]) / "cache" / "imports" / str(uuid.uuid4())
        total_bytes = 0
        destinations = []
        relatives = [Path(str(item.get("relative_path") or item.get("name") or "")).as_posix() for item in files]
        parts = [value.split("/") for value in relatives]
        common_directory = parts[0][0] if all(len(value) > 1 and value[0] == parts[0][0] for value in parts) else None
        for item, relative in zip(files, relatives):
            if common_directory:
                relative = relative.split("/", 1)[1]
            if relative.startswith("/") or any(part in {"", ".", ".."} for part in relative.split("/")):
                raise BackendError("IMPORT_UPLOAD_INVALID_PATH")
            try:
                content = base64.b64decode(str(item.get("content_base64", "")), validate=True)
            except Exception as exc:
                raise BackendError("IMPORT_UPLOAD_INVALID_FILE", str(item.get("name", ""))) from exc
            total_bytes += len(content)
            if total_bytes > 64 * 1024 * 1024:
                raise BackendError("IMPORT_UPLOAD_TOO_LARGE")
            destination = (root / relative).resolve()
            if root.resolve() not in destination.parents:
                raise BackendError("IMPORT_UPLOAD_INVALID_PATH")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            destinations.append(destination)
        source = destinations[0] if len(destinations) == 1 else root
        return import_dataset(request["project"], request["task_id"], str(source), request.get("options"))
    if action == "data.pipeline.apply":
        return apply_pipeline(request["project"], request["dataset"], request["task_id"], request["pipeline"])
    if action == "data.analytics":
        dataset=request["dataset"];payload=torch.load(dataset["path"],weights_only=True)
        return build_analytics(request["task_id"],payload["inputs"],payload["targets"],payload.get("classes"),dataset.get("options"))
    if action == "data.preview":
        dataset=request["dataset"];payload=torch.load(dataset["path"],weights_only=True)
        x=payload["inputs"].detach().cpu();n=len(x)
        if n == 0: raise BackendError("DATA_PREVIEW_EMPTY")
        indices=random.sample(range(n),min(8,n))
        y=payload.get("targets");y=y.detach().cpu() if isinstance(y,torch.Tensor) else y
        selected_y=y[indices] if isinstance(y,torch.Tensor) else None
        task_id=request["task_id"];classes=payload.get("classes") or dataset.get("classes") or [];options=payload.get("options",dataset.get("options",{}))
        if payload.get("boxes") is not None:
            preview=_build_detection_preview(x, payload["boxes"], classes, indices)
        else:
            preview=_build_preview(task_id,x[indices],selected_y,classes,options)
            preview["sampleIndices"]=indices
        return preview
    if action == "data.exists":
        path = Path(str(request.get("dataset", {}).get("path", ""))).expanduser()
        return {"exists": path.is_file()}
    if action == "graph.validate":
        graph = request.get("graph", {})
        types = [n.get("data", {}).get("blockType") for n in graph.get("nodes", [])]
        if "input" not in types or "output" not in types:
            raise BackendError("GRAPH_MISSING_INPUT_OUTPUT")
        dataset=request["dataset"]
        if "hf_image_processor" in types and dataset.get("options",{}).get("normalization","none")!="none":
            raise BackendError("HF_DOUBLE_NORMALIZATION")
        if "hf_token_processor" in types:
            payload=torch.load(dataset["path"],weights_only=True)
            if not payload.get("raw_inputs"): raise BackendError("HF_RAW_TEXT_REQUIRED","Vuelve a crear esta revisión del dataset")
        model = build_model(request["architecture"], dataset["inputShape"], dataset["outputShape"], graph, request["task_id"])
        shape = [2, *dataset["inputShape"]]
        sample = torch.zeros(shape, dtype=torch.long if request["task_id"].startswith("text.") else torch.float32)
        out = model(sample)
        outputs=out if isinstance(out,dict) else {"output":out}
        if any(not torch.isfinite(value).all() for value in outputs.values()):
            raise BackendError("GRAPH_DRY_RUN_NON_FINITE")
        task_id=request["task_id"]
        primary=next(iter(outputs.values()))
        if task_id=="image.detection": target=[torch.tensor([[0,.5,.5,.2,.2]],dtype=torch.float32) for _ in range(primary.shape[0])]
        elif task_id=="text.language_model": target=torch.ones(primary.shape[:2],dtype=torch.long)
        elif task_id=="image.segmentation.binary": target=torch.zeros((primary.shape[0],*primary.shape[2:]),dtype=torch.float32)
        elif task_id=="image.segmentation.multiclass": target=torch.zeros((primary.shape[0],*primary.shape[2:]),dtype=torch.long)
        elif task_spec(task_id)["objective"]=="classification": target=torch.zeros(primary.shape[0],dtype=torch.long)
        elif task_id=="image.localization": target=torch.tensor([[.25,.25,.75,.75]]*primary.shape[0],dtype=torch.float32)
        else: target=torch.zeros_like(primary)
        loss=_loss(task_id,out,target)
        if not torch.isfinite(loss): raise BackendError("TRAINING_LOSS_NON_FINITE")
        loss.backward()
        gradients=[parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
        if not gradients or any(not torch.isfinite(value).all() for value in gradients):
            raise BackendError("GRAPH_DRY_RUN_GRADIENT_INVALID")
        parameters=sum(p.numel() for p in model.parameters())
        shape_summary={str(node.get("data",{}).get("properties",{}).get("output_name") or node.get("data",{}).get("label") or node["id"]):model.node_shapes[node["id"]] for node in graph.get("nodes",[]) if node.get("data",{}).get("blockType")=="output"}
        return {"valid": True, "message": f"Grafo válido · {len(outputs)} salida(s) · {parameters:,} parámetros · backward correcto", "node_shapes": model.node_shapes,"output_shapes":shape_summary, "parameters": parameters, "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"backward": True,"semantic_hash":semantic_hash(graph),"schema_version":2}
    if action == "system.gpu":
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            return {"cuda": True, "name": name, "total_vram": total, "allocated_vram": allocated}
        return {"cuda": False, "name": "CPU", "total_vram": 0, "allocated_vram": 0}
    if action == "inference.run":
        return infer(request)
    raise BackendError("ENGINE_UNKNOWN_ACTION", action)


def tensor_to_b64_png(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.uint8:
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            arr = tensor[0].numpy()
            img = Image.fromarray(arr, mode="L")
        elif tensor.ndim == 3:
            arr = tensor[:3].permute(1, 2, 0).numpy()
            img = Image.fromarray(arr, mode="RGB")
        else:
            return ""
    else:
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            arr = (tensor[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")
        elif tensor.ndim == 3:
            arr = (tensor[:3].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="RGB")
        else:
            return ""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_b64_png(image:Image.Image)->str:
    buffer=io.BytesIO();image.save(buffer,format="PNG");return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_tensor_to_rgb(tensor:torch.Tensor)->Image.Image:
    value=tensor.detach().cpu().float();value=value/255 if value.numel() and value.max()>1 else value
    if value.shape[0]==1:
        pixels=(value[0].clamp(0,1).numpy()*255).astype(np.uint8);return Image.fromarray(pixels,mode="L").convert("RGB")
    pixels=(value[:3].permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8);return Image.fromarray(pixels,mode="RGB")


def _pipeline_stat(value, x: torch.Tensor, task_id: str) -> torch.Tensor:
    stat=torch.tensor(value,dtype=torch.float32)
    if stat.ndim==0: return stat
    if task_id.startswith("image") and stat.numel()==x.shape[1]: return stat.reshape(1,x.shape[1],1,1)
    if task_id.startswith("sequence") and stat.numel()==x.shape[-1]: return stat.reshape(1,1,x.shape[-1])
    if stat.numel()==x.shape[-1]: return stat.reshape(1,x.shape[-1])
    return stat


def apply_inference_pipeline(x: torch.Tensor, dataset: dict, task_id: str) -> torch.Tensor:
    input_transform=dataset.get("options",{}).get("inputTransform",{})
    if input_transform.get("mode")=="standard":
        x=(x.float()-_pipeline_stat(input_transform["center"],x,task_id))/_pipeline_stat(input_transform["scale"],x,task_id).clamp_min(1e-7)
    elif input_transform.get("mode") in {"minmax","minmax_sym"}:
        low=_pipeline_stat(input_transform["low"],x,task_id);high=_pipeline_stat(input_transform["high"],x,task_id);x=(x.float()-low)/(high-low).clamp_min(1e-7)
        if input_transform["mode"]=="minmax_sym": x=x*2-1
    pipeline=dataset.get("pipeline") or dataset.get("options",{}).get("pipeline") or {}
    fitted=pipeline.get("fittedState",{})
    for node in pipeline.get("nodes",[]):
        if not node.get("enabled",True): continue
        state=fitted.get(node.get("id"),{});kind=node.get("type")
        if kind=="resize" and task_id.startswith("image"):
            height=max(8,int(node.get("properties",{}).get("height",128)));width=max(8,int(node.get("properties",{}).get("width",128)));method=str(node.get("properties",{}).get("method","bilinear"));dtype=x.dtype;kwargs={} if method=="nearest" else {"align_corners":False};x=F.interpolate(x.float(),size=(height,width),mode=method,**kwargs);x=x.round().clamp(0,255).to(torch.uint8) if dtype==torch.uint8 else x
        elif kind=="center_crop" and task_id.startswith("image"):
            height=max(8,int(node.get("properties",{}).get("height",96)));width=max(8,int(node.get("properties",{}).get("width",96)));source_h,source_w=x.shape[-2:]
            if height>source_h or width>source_w: raise BackendError("PIPELINE_CROP_TOO_LARGE",f"{width}×{height} > {source_w}×{source_h}")
            top=(source_h-height)//2;left=(source_w-width)//2;x=x[...,top:top+height,left:left+width]
        elif kind=="convert_channels" and task_id.startswith("image"):
            channels=int(node.get("properties",{}).get("channels",3))
            if channels==1 and x.shape[1]!=1:x=x[:,:3].float().mean(1,keepdim=True).to(x.dtype)
            elif channels==3 and x.shape[1]==1:x=x.repeat(1,3,1,1)
        elif kind in {"normalize","image_normalize"} and state:
            x=x.float()
            if state.get("mode")=="standard":
                x=(x-_pipeline_stat(state["mean"],x,task_id))/_pipeline_stat(state["scale"],x,task_id).clamp_min(1e-7)
            else:
                low=_pipeline_stat(state["min"],x,task_id);high=_pipeline_stat(state["max"],x,task_id);x=(x-low)/(high-low).clamp_min(1e-7)
                if state.get("mode")=="minmax_sym": x=x*2-1
        elif kind=="clip" and state:
            low=_pipeline_stat(state["lower"],x,task_id);high=_pipeline_stat(state["upper"],x,task_id);x=torch.maximum(torch.minimum(x.float(),high),low)
    return x


def generate_causal_tokens(model, prompt: torch.Tensor, context_length: int, max_new_tokens: int = 16) -> list[int]:
    causal=next((module for module in model.modules() if hasattr(module,"generate_tokens")),None)
    if causal is not None: return causal.generate_tokens(prompt,max(1,min(int(max_new_tokens),128)))
    """Greedily generate a continuation, feeding every prediction back as context."""
    tokens=[int(value) for value in prompt.flatten().tolist() if int(value)!=0]
    if not tokens:
        tokens=[1]
    generated=[]
    with torch.inference_mode():
        for _ in range(max(1,min(int(max_new_tokens),128))):
            window=tokens[-context_length:]
            x=torch.zeros(1,context_length,dtype=torch.long,device=next(model.parameters()).device)
            x[0,:len(window)]=torch.tensor(window,dtype=torch.long,device=x.device)
            logits=model(x)
            scores=logits[0,len(window)-1].clone()
            scores[:2]=-torch.inf  # Never expose <pad> or <unk> as generated text.
            next_token=int(scores.argmax())
            tokens.append(next_token);generated.append(next_token)
    return generated


def inverse_target_transform(value: torch.Tensor,dataset: dict) -> torch.Tensor:
    transform=dataset.get("options",{}).get("targetTransform")
    if not transform:
        return value
    center=torch.tensor(transform["center"],dtype=value.dtype,device=value.device).reshape(value.shape[1:] or (1,))
    scale=torch.tensor(transform["scale"],dtype=value.dtype,device=value.device).reshape(value.shape[1:] or (1,))
    return value*scale+center


def infer(request: dict):
    project = request["project"]
    requested_checkpoint = request.get("checkpoint")
    if requested_checkpoint:
        checkpoint_path = Path(requested_checkpoint).expanduser().resolve()
    else:
        latest_path = project_dir(project) / "runs" / "latest.json"
        if not latest_path.exists():
            raise BackendError("INFERENCE_NO_CHECKPOINT")
        checkpoint_path = Path(json.loads(latest_path.read_text())["checkpoint"])
    if not checkpoint_path.exists():
        raise BackendError("INFERENCE_CHECKPOINT_MISSING")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    dataset = checkpoint["dataset"]
    dataset_path=Path(str(dataset.get("path",""))).expanduser()
    data={"classes":dataset.get("classes")}
    if dataset_path.is_file(): data.update(torch.load(dataset_path,weights_only=True))
    elif request.get("mode","test")!="manual": raise BackendError("INFERENCE_DATASET_ARTIFACT_MISSING",str(dataset_path))

    model = build_model(checkpoint["architecture"], dataset["inputShape"], dataset["outputShape"], checkpoint["graph"], checkpoint["task_id"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    task_id = checkpoint["task_id"]
    is_image = task_id.startswith("image") or "segmentation" in task_id
    mode = request.get("mode", "test")
    values = str(request.get("values", "")).strip()
    input_shape = dataset["inputShape"]

    input_preview = ""
    true_label = None
    source = ""
    original_image=None;original_size=None;sample_target=None;sample_boxes=None;sample_index=0
    spatial_affine=torch.tensor([1.,1.,0.,0.],dtype=torch.float32)
    hf_processor=next((node.get("data",{}).get("properties",{}) for node in checkpoint["graph"].get("nodes",[]) if node.get("data",{}).get("blockType")=="hf_token_processor"),None)

    if mode == "manual" and values:
        if is_image:
            try:
                if values.startswith("data:image"):
                    header, b64data = values.split(",", 1)
                    raw_bytes = base64.b64decode(b64data)
                    pil_img = Image.open(io.BytesIO(raw_bytes))
                else:
                    pil_img = Image.open(Path(values).expanduser().resolve())
                original_size=pil_img.size;original_image=pil_img.convert("RGB").copy()
                target_c=input_shape[0]
                img_mode = "L" if target_c == 1 else "RGB"
                pil_img = pil_img.convert(img_mode)

                if target_c == 1:
                    arr = np.array(pil_img)[:, :, None]
                else:
                    arr = np.array(pil_img)

                x = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                input_preview = image_to_b64_png(original_image)
                if data.get("spatial_affine") is not None: spatial_affine=data["spatial_affine"][0].float()
                source = "Imagen cargada manualmente"
            except Exception as exc:
                raise BackendError("INFERENCE_MANUAL_IMAGE_FAILED", str(exc))
        elif task_id.startswith("text."):
            if hf_processor: x=hf_tokenize_texts([values],hf_processor);source="Texto tokenizado con el processor del checkpoint"
            else:
                vocab=dataset.get("options",{}).get("vocab") or data.get("classes") or ["<pad>","<unk>"];lookup={word:i for i,word in enumerate(vocab)};tokens=re.findall(r"\w+|[^\w\s]",values.lower(),flags=re.UNICODE)
                if dataset.get("options",{}).get("add_cls"): tokens=["<cls>",*tokens]
                ids=[lookup.get(token,1) for token in tokens[:input_shape[0]]];ids += [0]*(input_shape[0]-len(ids));x=torch.tensor(ids,dtype=torch.long).unsqueeze(0);source="Texto tokenizado manualmente"
        elif task_id.startswith("tabular"):
            nums = [float(v.strip()) for v in values.split(",") if v.strip()]
            if len(nums) != input_shape[0]:
                raise BackendError("INFERENCE_MANUAL_TABULAR_WRONG_FEATURES", f"{input_shape[0]}, got {len(nums)}")
            x = torch.tensor(nums).float().unsqueeze(0)
            source = "Valores tabulares manuales"
        else:
            nums = [float(v.strip()) for v in values.split(",") if v.strip()]
            x = torch.tensor(nums).float().reshape(1, *input_shape)
            source = "Entrada manual"
    else:
        # Inference must sample the exact persisted test partition.  Rebuilding
        # percentages here silently broke official and chronological splits.
        splits = data.get("splits") or split_indices(len(data["inputs"]), int(dataset.get("seed", 42)), dataset.get("splits"))
        test_indices = splits.get("test", list(range(len(data["inputs"]))))

        idx_param = request.get("index")
        if idx_param is not None and isinstance(idx_param, int) and 0 <= idx_param < len(data["inputs"]):
            idx = idx_param
        else:
            idx = random.SystemRandom().choice(test_indices) if test_indices else 0

        x = hf_tokenize_texts([data["raw_inputs"][idx]],hf_processor) if hf_processor and data.get("raw_inputs") else data["inputs"][idx:idx + 1]
        source = f"Muestra #{idx} (Split Test)"

        if "targets" in data and len(data["targets"]) > idx:
            y = data["targets"][idx]
            if task_id=="image.localization": sample_target=y.detach().cpu()
            classes = data.get("classes")
            if classes and "classification" in task_id and y.numel() > 0:
                y_idx = int(y)
                if 0 <= y_idx < len(classes):
                    true_label = classes[y_idx]
            elif y.numel() == 1:
                display_y=inverse_target_transform(y.reshape(1,*y.shape),dataset)
                true_label = f"{float(display_y.item()):.4f}"
        if task_id=="image.detection" and data.get("boxes") is not None: sample_boxes=data["boxes"][idx]
        if data.get("original_sizes") is not None: original_size=tuple(int(value) for value in data["original_sizes"][idx].tolist())
        elif is_image: original_size=(int(x.shape[-1]),int(x.shape[-2]))
        if data.get("spatial_affine") is not None: spatial_affine=data["spatial_affine"][idx].float()

        if is_image:
            original_image=None
            input_preview = tensor_to_b64_png(x[0])
            if x.dtype == torch.uint8:
                x = x.float() / 255.0

    if mode == "manual":
        if is_image:
            pipeline=dataset.get("pipeline") or dataset.get("options",{}).get("pipeline") or {}
            if not any(node.get("enabled",True) and node.get("type")=="resize" for node in pipeline.get("nodes",[])) and tuple(x.shape[-2:])!=tuple(input_shape[-2:]):
                x=F.interpolate(x.float(),size=tuple(input_shape[-2:]),mode="bilinear",align_corners=False)
        x=apply_inference_pipeline(x,dataset,task_id)
    processed_preview=tensor_to_b64_png(x[0]) if is_image else ""

    out = None
    if task_id != "text.language_model":
        with torch.inference_mode():
            out = model(x)

    res = {
        "source": source,
        "inputPreview": input_preview,
        "trueLabel": true_label,
    }
    if is_image:
        res["processedInputPreview"]=processed_preview
        if original_size: res["imageSize"]={"width":original_size[0],"height":original_size[1]}

    if "classification" in task_id:
        probs = out.softmax(1)[0]
        idx = int(probs.argmax())
        classes = data.get("classes") or dataset.get("classes") or [f"Clase {i}" for i in range(len(probs))]
        res.update({
            "prediction": classes[idx],
            "classIndex": idx,
            "probabilities": probs.tolist()
        })
    elif task_id in {"image.reconstruction","image.translation"}:
        reconstructed=out[0].detach().cpu();res.update({"prediction":"Imagen reconstruida","shape":list(reconstructed.shape),"outputPreview":tensor_to_b64_png(reconstructed)})
    elif task_id == "image.localization":
        current_box=[float(value) for value in decode_localization(out)[0].detach().cpu()];sx,sy,ox,oy=[float(value) for value in spatial_affine];normalized=[current_box[0]*sx+ox,current_box[1]*sy+oy,current_box[2]*sx+ox,current_box[3]*sy+oy];width,height=original_size or (x.shape[-1],x.shape[-2]);pixels=[normalized[0]*width,normalized[1]*height,normalized[2]*width,normalized[3]*height]
        canvas=original_image.copy() if original_image is not None else image_tensor_to_rgb(x[0]);draw=ImageDraw.Draw(canvas)
        display_box=pixels if original_image is not None else [current_box[0]*canvas.width,current_box[1]*canvas.height,current_box[2]*canvas.width,current_box[3]*canvas.height]
        draw.rectangle(display_box,outline=(50,230,130),width=max(1,canvas.width//128))
        target_report=None
        if sample_target is not None:
            target_current=[float(value) for value in sample_target];target_report=[target_current[0]*sx+ox,target_current[1]*sy+oy,target_current[2]*sx+ox,target_current[3]*sy+oy]
            if original_image is None: draw.rectangle((target_current[0]*canvas.width,target_current[1]*canvas.height,target_current[2]*canvas.width,target_current[3]*canvas.height),outline=(80,150,255),width=max(1,canvas.width//128))
        res.update({"prediction":"Caja localizada","box":{"format":"original_pixel_xyxy","coordinates":pixels,"normalizedCoordinates":normalized,"modelCoordinates":current_box},"targetBox":{"normalizedCoordinates":target_report,"format":"original_normalized_xyxy"} if target_report else None,"outputPreview":image_to_b64_png(canvas),"shape":[4]})
    elif task_id == "image.detection":
        detections=decode_detections(out,float(request.get("score_threshold",.25)),float(request.get("iou_threshold",.5)))[0]
        canvas=original_image.copy() if original_image is not None else image_tensor_to_rgb(x[0]);draw=ImageDraw.Draw(canvas);canvas_w,canvas_h=canvas.size;model_h,model_w=x.shape[-2:];width,height=original_size or (model_w,model_h);sx,sy,ox,oy=[float(value) for value in spatial_affine];classes=data.get("classes") or dataset.get("classes") or [];results=[]
        for cls,score,x1,y1,x2,y2 in detections.tolist():
            current=[x1,y1,x2,y2];normalized=[x1*sx+ox,y1*sy+oy,x2*sx+ox,y2*sy+oy];pixels=[normalized[0]*width,normalized[1]*height,normalized[2]*width,normalized[3]*height];display=pixels if original_image is not None else [x1*canvas_w,y1*canvas_h,x2*canvas_w,y2*canvas_h]
            draw.rectangle(display,outline=(50,230,130),width=max(1,canvas_w//128));name=classes[int(cls)] if int(cls)<len(classes) else str(int(cls));draw.text((display[0],max(0,display[1]-12)),f"{name} {score:.2f}",fill=(50,230,130));results.append({"class":name,"classIndex":int(cls),"score":score,"box":{"format":"original_pixel_xyxy","coordinates":pixels,"normalizedCoordinates":normalized,"modelCoordinates":current}})
        overlay=image_to_b64_png(canvas)
        res.update({"prediction":"Detecciones","detections":results,"outputPreview":overlay,"shape":[len(results),6]})
    elif task_id == "text.language_model":
        generated=generate_causal_tokens(model,x,input_shape[0],int(request.get("max_new_tokens",16)))
        if hf_processor:
            decoded=hf_decode_tokens(generated,hf_processor);prompt_text=values if mode=="manual" else data.get("raw_inputs",[""])[idx]
            res.update({"prediction":decoded,"completion":decoded,"prompt":prompt_text,"tokens":generated,"shape":[len(generated)]})
        else:
            vocab=data.get("classes") or dataset.get("options",{}).get("vocab",[]);decoded=[vocab[index] if index<len(vocab) else str(index) for index in generated];prompt_ids=[int(value) for value in x[0].tolist() if int(value)!=0];prompt_words=[vocab[index] if index<len(vocab) else str(index) for index in prompt_ids];res.update({"prediction":" ".join(decoded),"completion":" ".join(decoded),"prompt":" ".join(prompt_words),"tokens":generated,"shape":[len(generated),len(vocab)]})
    elif "segmentation" in task_id:
        mask = (out.sigmoid() >= .5)[0, 0] if task_id.endswith("binary") else out.argmax(1)[0]
        if task_id.endswith("binary"):
            mask_preview = tensor_to_b64_png(mask.unsqueeze(0).float())
        else:
            palette = torch.tensor([[8, 12, 18], [35, 190, 215], [150, 110, 235], [65, 200, 125], [240, 165, 45]], dtype=torch.uint8)
            colored = palette[mask.clamp(0, len(palette) - 1)].permute(2, 0, 1)
            mask_preview = tensor_to_b64_png(colored)
        res.update({
            "prediction": "Máscara generada",
            "shape": list(mask.shape),
            "foregroundPixels": int((mask > 0).sum()),
            "outputPreview": mask_preview,
        })
    else:
        out=inverse_target_transform(out,dataset)
        res.update({
            "prediction": float(out[0].item()) if out.numel() == 1 else out[0].tolist(),
            "shape": list(out.shape[1:])
        })

    return res


def main():
    try:
        request = json.loads(sys.stdin.read())
        if request.get("action") == "train":
            train(request)
        else:
            response(handle(request))
    except Exception as exc:
        if "request" in locals() and request.get("action") == "train":
            print(json.dumps({"type": "error", "message": str(exc)}), flush=True)
        else:
            response(error=exc)
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
