from __future__ import annotations

from .errors import BackendError

TASKS = {
    "tabular.classification": {"architectures": ["mlp"], "objective": "classification", "metric": "accuracy"},
    "tabular.regression": {"architectures": ["mlp"], "objective": "regression", "metric": "mse"},
    "tabular.reconstruction": {"architectures": ["autoencoder"], "objective": "reconstruction", "metric": "mse"},
    "image.classification": {"architectures": ["cnn", "vit"], "objective": "classification", "metric": "accuracy"},
    "image.regression": {"architectures": ["cnn", "vit"], "objective": "regression", "metric": "mse"},
    "image.reconstruction": {"architectures": ["autoencoder"], "objective": "reconstruction", "metric": "mse"},
    "sequence.classification": {"architectures": ["cnn1d", "lstm", "transformer"], "objective": "classification", "metric": "accuracy"},
    "sequence.regression": {"architectures": ["cnn1d", "lstm", "transformer"], "objective": "regression", "metric": "mse"},
    "sequence.forecast": {"architectures": ["cnn1d", "lstm", "transformer"], "objective": "forecast", "metric": "mse"},
    "text.classification": {"architectures": ["transformer"], "objective": "classification", "metric": "accuracy"},
    "text.language_model": {"architectures": ["transformer_causal"], "objective": "token_prediction", "metric": "accuracy"},
    "image.segmentation.binary": {"architectures": ["unet"], "objective": "segmentation_binary", "metric": "dice"},
    "image.segmentation.multiclass": {"architectures": ["unet"], "objective": "segmentation_multiclass", "metric": "miou"},
}


def task_spec(task_id: str) -> dict:
    if task_id not in TASKS:
        raise BackendError("CATALOG_TASK_UNSUPPORTED", task_id)
    return TASKS[task_id]


def validate_compatibility(task_id: str, architecture: str) -> None:
    spec = task_spec(task_id)
    if architecture not in spec["architectures"]:
        choices = ", ".join(spec["architectures"])
        raise BackendError("CATALOG_ARCHITECTURE_INCOMPATIBLE", f"{architecture} for {task_id}; options: {choices}")
