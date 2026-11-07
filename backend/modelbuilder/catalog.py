from __future__ import annotations

from .errors import BackendError

TASKS = {
    "tabular.classification": {"architectures": ["mlp"], "objective": "classification", "metric": "accuracy"},
    "tabular.regression": {"architectures": ["mlp"], "objective": "regression", "metric": "mse"},
    "tabular.reconstruction": {"architectures": ["autoencoder"], "objective": "reconstruction", "metric": "mse"},
    "image.classification": {"architectures": ["cnn", "vit"], "objective": "classification", "metric": "accuracy"},
    "image.regression": {"architectures": ["cnn", "vit"], "objective": "regression", "metric": "mse"},
    "image.localization": {"architectures": ["cnn"], "objective": "localization", "metric": "iou"},
    "image.detection": {"architectures": ["cnn"], "objective": "detection", "metric": "map50"},
    "image.translation": {"architectures": ["cnn", "unet"], "objective": "image_translation", "metric": "mae"},
    "image.reconstruction": {"architectures": ["autoencoder"], "objective": "reconstruction", "metric": "mse"},
    "sequence.classification": {"architectures": ["cnn1d", "lstm", "transformer"], "objective": "classification", "metric": "accuracy"},
    "sequence.regression": {"architectures": ["cnn1d", "lstm", "transformer"], "objective": "regression", "metric": "mse"},
    "sequence.forecast": {"architectures": ["cnn1d", "lstm", "transformer"], "objective": "forecast", "metric": "mse"},
    "text.classification": {"architectures": ["transformer"], "objective": "classification", "metric": "accuracy"},
    "text.language_model": {"architectures": ["transformer_causal"], "objective": "token_prediction", "metric": "accuracy"},
    "image.segmentation.binary": {"architectures": ["unet"], "objective": "segmentation_binary", "metric": "dice"},
    "image.segmentation.multiclass": {"architectures": ["unet"], "objective": "segmentation_multiclass", "metric": "miou"},
}

BLOCK_TYPES=("identity","input","output","linear","relu","leaky_relu","gelu","silu","tanh","sigmoid","softmax","dropout","dropout2d","flatten","batchnorm1d","batchnorm2d","layernorm","embedding","sequence_projection","positional_encoding","transformer_encoder","causal_transformer","sequence_pool","patch_embedding","permute","conv1d","causalpad1d","maxpool1d","avgpool1d","adaptiveavgpool1d","conv2d","convtranspose2d","maxpool2d","avgpool2d","adaptiveavgpool2d","multi_scale_cnn","lstm","rnn","gru","temporal_select","concat","add","reshape","resize2d","hf_image_processor","hf_vision_backbone","hf_token_processor","hf_text_backbone","hf_causal_lm")


def catalog() -> dict:
    architectures={architecture:{"tasks":[task_id for task_id,spec in TASKS.items() if architecture in spec["architectures"]]} for spec in TASKS.values() for architecture in spec["architectures"]}
    return {"schema_version":1,"tasks":TASKS,"architectures":architectures,"block_types":list(BLOCK_TYPES)}


def task_spec(task_id: str) -> dict:
    if task_id not in TASKS:
        raise BackendError("CATALOG_TASK_UNSUPPORTED", task_id)
    return TASKS[task_id]


def validate_compatibility(task_id: str, architecture: str) -> None:
    spec = task_spec(task_id)
    if architecture not in spec["architectures"]:
        choices = ", ".join(spec["architectures"])
        raise BackendError("CATALOG_ARCHITECTURE_INCOMPATIBLE", f"{architecture} for {task_id}; options: {choices}")
