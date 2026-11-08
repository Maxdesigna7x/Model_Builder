from pathlib import Path

import pytest
import torch
from PIL import Image

from modelbuilder.data import import_dataset
from modelbuilder.errors import BackendError
from modelbuilder.storage import create_project


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELBUILDER_WORKSPACE", str(tmp_path / "workspace"))
    value = {"id": "reimport", "name": "Reimport", "path": "projects/reimport"}
    create_project(value)
    return value


def image(path: Path, color=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 16), (color, 20, 30)).save(path)


@pytest.mark.parametrize("validation_name", ["val", "validation"])
def test_yolo_partitions_keep_boxes_with_duplicate_basenames(project, tmp_path, validation_name):
    root = tmp_path / "detection"
    for split, label in [("train", "0 .2 .3 .2 .2"), (validation_name, "1 .7 .6 .2 .2"), ("test", "")]:
        image(root / "images" / split / "same.png")
        label_path = root / "labels" / split / "same.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(label)
    (root / "classes.txt").write_text("car\ntruck\n")
    summary = import_dataset(project, "image.detection", str(root), {"resolution": 16})
    payload = torch.load(summary["path"], weights_only=True)
    assert summary["splitSource"] == "official"
    assert summary["splitCounts"] == {"train": 1, "validation": 1, "test": 1}
    assert payload["splits"] == {"train": [0], "validation": [1], "test": [2]}
    assert [len(boxes) for boxes in payload["boxes"]] == [1, 1, 0]
    assert payload["boxes"][0][0].tolist() == pytest.approx([0, .2, .3, .2, .2])
    assert payload["boxes"][1][0].tolist() == pytest.approx([1, .7, .6, .2, .2])


def test_partitioned_yolo_does_not_interpret_missing_annotation_as_negative(project, tmp_path):
    root = tmp_path / "detection"
    for split in ("train", "val", "test"):
        image(root / "images" / split / "same.png")
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("car\n")
    with pytest.raises(BackendError, match="IMPORT_DETECTION_LABEL_MISSING"):
        import_dataset(project, "image.detection", str(root))


def test_classification_preserves_official_splits_and_common_class_mapping(project, tmp_path):
    root = tmp_path / "classification"
    for split, label in [("train", "dog"), ("train", "cat"), ("val", "dog"), ("test", "cat")]:
        image(root / split / label / "same.png")
    summary = import_dataset(project, "image.classification", str(root), {"resolution": 16})
    payload = torch.load(summary["path"], weights_only=True)
    assert summary["classes"] == ["cat", "dog"]
    assert summary["splitSource"] == "official"
    assert payload["splits"] == {"train": [0, 1], "validation": [2], "test": [3]}
    assert payload["targets"].tolist() == [0, 1, 1, 0]


def test_flat_class_named_train_is_not_reinterpreted_as_partition(project, tmp_path):
    root = tmp_path / "classification"
    for label in ("train", "car"):
        for i in range(3):
            image(root / label / f"{i}.png")
    summary = import_dataset(project, "image.classification", str(root), {"resolution": 16})
    assert summary["classes"] == ["car", "train"]
    assert summary["splitSource"] == "generated"


def test_partitioned_export_with_single_class_explains_training_limit(project, tmp_path):
    root = tmp_path / "classification"
    for split in ("train", "val", "test"):
        image(root / split / "car" / "same.png")
    with pytest.raises(BackendError, match="IMPORT_IMAGE_CLASSIFICATION_MIN_CLASSES"):
        import_dataset(project, "image.classification", str(root))


def test_mixed_layout_is_rejected_instead_of_dropping_images(project, tmp_path):
    root = tmp_path / "classification"
    image(root / "train" / "car" / "1.png")
    image(root / "train" / "truck" / "2.png")
    image(root / "other" / "3.png")
    with pytest.raises(BackendError, match="IMPORT_IMAGE_MIXED_PARTITIONS"):
        import_dataset(project, "image.classification", str(root))


def test_exported_edge_boxes_survive_decimal_rounding(project, tmp_path):
    root = tmp_path / 'edge-boxes'
    for split in ('train', 'val', 'test'):
        image(root / 'images' / split / 'edge.png')
        label = root / 'labels' / split / 'edge.txt'
        label.parent.mkdir(parents=True)
        label.write_text('0 0.66666667 0.5 0.66666667 1.00000000\n')
    (root / 'classes.txt').write_text('car\n')
    summary = import_dataset(project, 'image.detection', str(root), {'resolution': 16})
    assert summary['samples'] == 3
