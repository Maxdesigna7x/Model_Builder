from __future__ import annotations

import base64
import io
import json
import random
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modelbuilder.data import apply_pipeline, build_analytics, generate, import_dataset, split_indices
    from modelbuilder.models import build_model
    from modelbuilder.storage import create_project, delete_project, open_project, project_dir, save_project_state, workspace_root
    from modelbuilder.training import train
else:
    from .data import apply_pipeline, build_analytics, generate, import_dataset, split_indices
    from .models import build_model
    from .storage import create_project, delete_project, open_project, project_dir, save_project_state, workspace_root
    from .training import train


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
    if action == "project.create":
        return create_project(request["project"])
    if action == "project.open":
        return open_project(request["path"])
    if action == "project.delete":
        return delete_project(request["project"])
    if action == "project.state.save":
        return save_project_state(request["project"], request["state"])
    if action == "data.generate":
        return generate(request["project"], request["task_id"], int(request.get("seed", 42)), request.get("options"))
    if action == "data.import":
        return import_dataset(request["project"], request["task_id"], request["path"], request.get("options"))
    if action == "data.pipeline.apply":
        return apply_pipeline(request["project"], request["dataset"], request["task_id"], request["pipeline"])
    if action == "data.analytics":
        dataset=request["dataset"];payload=torch.load(dataset["path"],weights_only=True)
        return build_analytics(request["task_id"],payload["inputs"],payload["targets"],payload.get("classes"),dataset.get("options"))
    if action == "graph.validate":
        graph = request.get("graph", {})
        types = [n.get("data", {}).get("blockType") for n in graph.get("nodes", [])]
        if "input" not in types or "output" not in types:
            raise ValueError("El grafo necesita Entrada y Salida")
        model = build_model(request["architecture"], request["dataset"]["inputShape"], request["dataset"]["outputShape"], graph, request["task_id"])
        shape = [2, *request["dataset"]["inputShape"]]
        sample = torch.zeros(shape, dtype=torch.long if request["task_id"].startswith("text.") else torch.float32)
        out = model(sample)
        if not torch.isfinite(out).all():
            raise ValueError("El dry-run produjo valores no finitos")
        return {"valid": True, "message": f"Grafo válido · salida {list(out.shape)} · {sum(p.numel() for p in model.parameters()):,} parámetros", "node_shapes": model.node_shapes}
    if action == "system.gpu":
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            return {"cuda": True, "name": name, "total_vram": total, "allocated_vram": allocated}
        return {"cuda": False, "name": "NVIDIA GeForce RTX 3060", "total_vram": 12487661158, "allocated_vram": 0}
    if action == "inference.run":
        return infer(request)
    raise ValueError(f"Acción desconocida: {action}")


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


def _pipeline_stat(value, x: torch.Tensor, task_id: str) -> torch.Tensor:
    stat=torch.tensor(value,dtype=torch.float32)
    if stat.ndim==0: return stat
    if task_id.startswith("image") and stat.numel()==x.shape[1]: return stat.reshape(1,x.shape[1],1,1)
    if task_id.startswith("sequence") and stat.numel()==x.shape[-1]: return stat.reshape(1,1,x.shape[-1])
    if stat.numel()==x.shape[-1]: return stat.reshape(1,x.shape[-1])
    return stat


def apply_inference_pipeline(x: torch.Tensor, dataset: dict, task_id: str) -> torch.Tensor:
    pipeline=dataset.get("pipeline") or dataset.get("options",{}).get("pipeline") or {}
    fitted=pipeline.get("fittedState",{})
    for node in pipeline.get("nodes",[]):
        if not node.get("enabled",True): continue
        state=fitted.get(node.get("id"),{});kind=node.get("type")
        if kind in {"normalize","image_normalize"} and state:
            x=x.float()
            if state.get("mode")=="standard":
                x=(x-_pipeline_stat(state["mean"],x,task_id))/_pipeline_stat(state["scale"],x,task_id).clamp_min(1e-7)
            else:
                low=_pipeline_stat(state["min"],x,task_id);high=_pipeline_stat(state["max"],x,task_id);x=(x-low)/(high-low).clamp_min(1e-7)
                if state.get("mode")=="minmax_sym": x=x*2-1
        elif kind=="clip" and state:
            low=_pipeline_stat(state["lower"],x,task_id);high=_pipeline_stat(state["upper"],x,task_id);x=torch.maximum(torch.minimum(x.float(),high),low)
    return x


def infer(request: dict):
    project = request["project"]
    requested_checkpoint = request.get("checkpoint")
    if requested_checkpoint:
        checkpoint_path = Path(requested_checkpoint).expanduser().resolve()
    else:
        latest_path = project_dir(project) / "runs" / "latest.json"
        if not latest_path.exists():
            raise ValueError("No existe un checkpoint entrenado")
        checkpoint_path = Path(json.loads(latest_path.read_text())["checkpoint"])
    if not checkpoint_path.exists():
        raise ValueError("El checkpoint de la corrida activa ya no existe")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    dataset = checkpoint["dataset"]
    data = torch.load(dataset["path"], weights_only=True)

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

    if mode == "manual" and values:
        if is_image:
            try:
                if values.startswith("data:image"):
                    header, b64data = values.split(",", 1)
                    raw_bytes = base64.b64decode(b64data)
                    pil_img = Image.open(io.BytesIO(raw_bytes))
                else:
                    pil_img = Image.open(Path(values).expanduser().resolve())

                target_c, target_h, target_w = input_shape[0], input_shape[1], input_shape[2]
                img_mode = "L" if target_c == 1 else "RGB"
                pil_img = pil_img.convert(img_mode).resize((target_w, target_h), Image.Resampling.BILINEAR)

                if target_c == 1:
                    arr = np.array(pil_img)[:, :, None]
                else:
                    arr = np.array(pil_img)

                x = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                input_preview = tensor_to_b64_png(x[0])
                source = "Imagen cargada manualmente"
            except Exception as exc:
                raise ValueError(f"Error procesando la imagen manual: {exc}")
        elif task_id.startswith("text."):
            vocab=dataset.get("options",{}).get("vocab") or data.get("classes") or ["<pad>","<unk>"];lookup={word:i for i,word in enumerate(vocab)};tokens=re.findall(r"\w+|[^\w\s]",values.lower(),flags=re.UNICODE);ids=[lookup.get(token,1) for token in tokens[:input_shape[0]]];ids += [0]*(input_shape[0]-len(ids));x=torch.tensor(ids,dtype=torch.long).unsqueeze(0);source="Texto tokenizado manualmente"
        elif task_id.startswith("tabular"):
            nums = [float(v.strip()) for v in values.split(",") if v.strip()]
            if len(nums) != input_shape[0]:
                raise ValueError(f"Se esperaban {input_shape[0]} variables y se recibieron {len(nums)}")
            x = torch.tensor(nums).float().unsqueeze(0)
            source = "Valores tabulares manuales"
        else:
            nums = [float(v.strip()) for v in values.split(",") if v.strip()]
            x = torch.tensor(nums).float().reshape(1, *input_shape)
            source = "Entrada manual"
    else:
        splits = split_indices(len(data["inputs"]), int(dataset.get("seed", 42)), dataset.get("splits"))
        test_indices = splits.get("test", list(range(len(data["inputs"]))))

        idx_param = request.get("index")
        if idx_param is not None and isinstance(idx_param, int) and 0 <= idx_param < len(data["inputs"]):
            idx = idx_param
        else:
            idx = random.SystemRandom().choice(test_indices) if test_indices else 0

        x = data["inputs"][idx:idx + 1]
        source = f"Muestra #{idx} (Split Test)"

        if "targets" in data and len(data["targets"]) > idx:
            y = data["targets"][idx]
            classes = data.get("classes")
            if classes and "classification" in task_id and y.numel() > 0:
                y_idx = int(y)
                if 0 <= y_idx < len(classes):
                    true_label = classes[y_idx]
            elif y.numel() == 1:
                true_label = f"{float(y.item()):.4f}"

        if is_image:
            input_preview = tensor_to_b64_png(x[0])
            if x.dtype == torch.uint8:
                x = x.float() / 255.0

    if mode == "manual":
        x=apply_inference_pipeline(x,dataset,task_id)

    with torch.inference_mode():
        out = model(x)

    res = {
        "source": source,
        "inputPreview": input_preview,
        "trueLabel": true_label,
    }

    if "classification" in task_id:
        probs = out.softmax(1)[0]
        idx = int(probs.argmax())
        classes = data.get("classes") or [f"Clase {i}" for i in range(len(probs))]
        res.update({
            "prediction": classes[idx],
            "classIndex": idx,
            "probabilities": probs.tolist()
        })
    elif task_id == "image.reconstruction":
        reconstructed=out[0].detach().cpu();res.update({"prediction":"Imagen reconstruida","shape":list(reconstructed.shape),"outputPreview":tensor_to_b64_png(reconstructed)})
    elif task_id == "text.language_model":
        predicted=out.argmax(-1)[0];vocab=data.get("classes") or dataset.get("options",{}).get("vocab",[])
        decoded=[vocab[int(index)] if int(index)<len(vocab) else str(int(index)) for index in predicted]
        res.update({"prediction":" ".join(decoded),"tokens":predicted.tolist(),"shape":list(out.shape[1:])})
    elif "segmentation" in task_id:
        mask = (out.sigmoid() >= .5)[0, 0] if task_id.endswith("binary") else out.argmax(1)[0]
        if task_id.endswith("binary"):
            mask_preview = tensor_to_b64_png(mask.unsqueeze(0).float())
        else:
            palette = torch.tensor([[8, 12, 18], [35, 190, 215], [150, 110, 235], [65, 200, 125]], dtype=torch.uint8)
            colored = palette[mask.clamp(0, len(palette) - 1)].permute(2, 0, 1)
            mask_preview = tensor_to_b64_png(colored)
        res.update({
            "prediction": "Máscara generada",
            "shape": list(mask.shape),
            "foregroundPixels": int((mask > 0).sum()),
            "outputPreview": mask_preview,
        })
    else:
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
