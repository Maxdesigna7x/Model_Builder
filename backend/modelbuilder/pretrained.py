from __future__ import annotations

from pathlib import Path
import json
import shutil

import torch

from .storage import workspace_root
from .storage import atomic_json, iso_now


# Deliberately curated and commit-pinned. Popularity is only a discovery signal;
# compatibility and an explicit license win over download counts.
PRETRAINED_MODELS: dict[str, dict] = {
    "resnet-18": {"name":"ResNet-18","architecture":"cnn","implementation":"resnet","repoId":"microsoft/resnet-18","revision":"65a5785d9156231087c481e0c7dd33a5ff6f7e3e","library":"transformers","license":"apache-2.0","tasks":["image.classification","image.regression","image.localization","image.detection"],"processor":"AutoImageProcessor","adapter":"vision_backbone","size":"small","parameterCount":11700000,"inputContract":"RGB 224×224","limitations":"Preentrenado en imágenes naturales; otros dominios requieren fine-tuning.","description":"Backbone convolucional residual compacto y ampliamente utilizado."},
    "mobilenet-v2": {"name":"MobileNet V2","architecture":"cnn","implementation":"mobilenet_v2","repoId":"google/mobilenet_v2_1.0_224","revision":"75e607b00aeae1297cc89d026a118bce012f5c5a","library":"transformers","license":"other","tasks":["image.classification","image.regression","image.localization","image.detection"],"processor":"AutoImageProcessor","adapter":"vision_backbone","size":"small","description":"Backbone eficiente para equipos con memoria y cómputo limitados."},
    "convnext-tiny": {"name":"ConvNeXT Tiny","architecture":"cnn","implementation":"convnext","repoId":"facebook/convnext-tiny-224","revision":"6166b7613034066690a621d8bf25ffdf181a34f0","library":"transformers","license":"apache-2.0","tasks":["image.classification","image.regression","image.localization","image.detection"],"processor":"AutoImageProcessor","adapter":"vision_backbone","size":"medium","description":"CNN moderna organizada en etapas jerárquicas."},
    "vit-base": {"name":"ViT Base","architecture":"vit","implementation":"vit","repoId":"google/vit-base-patch16-224","revision":"3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3","library":"transformers","license":"apache-2.0","tasks":["image.classification","image.regression"],"processor":"AutoImageProcessor","adapter":"vision_backbone","size":"large","description":"Vision Transformer base preentrenado con parches de 16 píxeles."},
    "deit-small": {"name":"DeiT Small","architecture":"vit","implementation":"deit","repoId":"facebook/deit-small-patch16-224","revision":"164deee347853469b97442b3817f22eece80c7e3","library":"transformers","license":"apache-2.0","tasks":["image.classification","image.regression"],"processor":"AutoImageProcessor","adapter":"vision_backbone","size":"medium","description":"Transformer visual más compacto que ViT Base."},
    "distilbert": {"name":"DistilBERT","architecture":"transformer","implementation":"distilbert","repoId":"distilbert/distilbert-base-uncased","revision":"12040accade4e8a0f71eabdb258fecc2e7e948be","library":"transformers","license":"apache-2.0","tasks":["text.classification"],"processor":"AutoTokenizer","adapter":"text_backbone","size":"medium","description":"Encoder de texto compacto para clasificación."},
    "bert-multilingual": {"name":"BERT multilingüe","architecture":"transformer","implementation":"bert","repoId":"google-bert/bert-base-multilingual-cased","revision":"3f076fdb1ab68d5b2880cb87a0886f315b8146f8","library":"transformers","license":"apache-2.0","tasks":["text.classification"],"processor":"AutoTokenizer","adapter":"text_backbone","size":"large","description":"Encoder BERT para múltiples idiomas."},
    "beto": {"name":"BETO","architecture":"transformer","implementation":"bert","repoId":"dccuchile/bert-base-spanish-wwm-cased","revision":"c4d86612f51b4f46759c8390d1798c2febe71b93","library":"transformers","license":"unspecified","tasks":["text.classification"],"processor":"AutoTokenizer","adapter":"text_backbone","size":"large","description":"BERT entrenado específicamente para español."},
    "distilgpt2": {"name":"DistilGPT-2","architecture":"transformer_causal","implementation":"gpt2","repoId":"distilbert/distilgpt2","revision":"2290a62682d06624634c1f46a6ad5be0f47f38aa","library":"transformers","license":"apache-2.0","tasks":["text.language_model"],"processor":"AutoTokenizer","adapter":"causal_lm","size":"medium","description":"Modelo causal compacto para generación local."},
    "gpt2": {"name":"GPT-2","architecture":"transformer_causal","implementation":"gpt2","repoId":"openai-community/gpt2","revision":"607a30d783dfa663caf39e06633721c8d4cfcd7e","library":"transformers","license":"mit","tasks":["text.language_model"],"processor":"AutoTokenizer","adapter":"causal_lm","size":"large","description":"Modelo causal de referencia."},
}

SCRATCH_ONLY = {
    "mlp": "Los pesos tabulares dependen de las columnas y no existe un checkpoint universal.",
    "cnn1d": "Frecuencia, canales y dominio deben coincidir; se mantienen recetas desde cero.",
    "lstm": "El vocabulario o las variables temporales determinan el contrato de los pesos.",
    "unet": "La taxonomía de máscaras debe coincidir; no se importan checkpoints ambiguos.",
    "autoencoder": "La entrada y el espacio latente están ligados al dataset de origen.",
}


def _cached(spec: dict) -> bool:
    hub_root=workspace_root()/"cache"/"huggingface"/"models"
    repo=spec["repoId"].replace("/","--")
    return any((hub_root/name).exists() for name in (f"models--{repo}",repo)) if hub_root.exists() else False


def _cache_metadata(spec:dict) -> dict:
    root=cache_dir()/f"models--{spec['repoId'].replace('/','--')}";snapshot=root/"snapshots"/spec["revision"]
    files=[path for path in snapshot.rglob("*") if path.is_file()] if snapshot.exists() else []
    return {"cached":bool(files),"cacheVerified":bool(files),"cacheBytes":sum(path.stat().st_size for path in files),"diskFreeBytes":shutil.disk_usage(cache_dir()).free}


def catalog(task_id: str | None = None, architecture: str | None = None) -> list[dict]:
    values=[]
    for model_id,spec in PRETRAINED_MODELS.items():
        if task_id and task_id not in spec["tasks"]: continue
        if architecture and architecture!=spec["architecture"]: continue
        values.append({"id":model_id,**spec,**_cache_metadata(spec),"available":True,"gated":False,"trustRemoteCode":False,"weightsFormat":"safetensors","schemaVersion":1})
    return values


def get(model_id: str) -> dict:
    return PRETRAINED_MODELS[model_id]


def scratch_guidance(architecture: str) -> str | None:
    return SCRATCH_ONLY.get(architecture)


def record_license_acceptance(model_id:str) -> dict:
    spec=get(model_id);path=workspace_root()/"agreements"/"huggingface-models.json"
    try: payload=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): payload={"schema_version":1,"agreements":[]}
    agreement={"model_id":model_id,"repo_id":spec["repoId"],"revision":spec["revision"],"license":spec["license"],"accepted_at":iso_now()}
    payload["agreements"]=[item for item in payload["agreements"] if item.get("model_id")!=model_id];payload["agreements"].append(agreement);atomic_json(path,payload);return agreement


def cache_dir() -> Path:
    path=workspace_root()/"cache"/"huggingface"/"models";path.mkdir(parents=True,exist_ok=True);return path


def tokenize_texts(texts: list[str], properties: dict):
    try:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(properties["repo_id"]),revision=str(properties["revision"]),cache_dir=str(cache_dir()),trust_remote_code=False,local_files_only=bool(properties.get("offline",False)))
        if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
        encoded=tokenizer(texts,padding="max_length",truncation=True,max_length=int(properties.get("max_length",128)),return_tensors="pt")
        input_ids=encoded["input_ids"].long();attention_mask=encoded.get("attention_mask",input_ids.ne(tokenizer.pad_token_id or 0).long()).long();token_type_ids=encoded.get("token_type_ids",torch.zeros_like(input_ids)).long()
        return torch.stack((input_ids,attention_mask,token_type_ids),dim=-1)
    except Exception as exc:
        from .errors import BackendError
        raise BackendError("HF_TOKENIZER_FAILED",str(exc)) from exc


def decode_tokens(tokens: list[int], properties: dict) -> str:
    try:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(properties["repo_id"]),revision=str(properties["revision"]),cache_dir=str(cache_dir()),trust_remote_code=False,local_files_only=bool(properties.get("offline",False)))
        return tokenizer.decode(tokens,skip_special_tokens=True)
    except Exception: return " ".join(str(value) for value in tokens)
