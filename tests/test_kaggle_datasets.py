import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from modelbuilder.data import apply_pipeline
from modelbuilder.engine import handle
from modelbuilder.errors import BackendError
from modelbuilder.kaggle_datasets import dataset_handle, download_root, image_index
from modelbuilder.storage import create_project


def image(path, value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (12, 12), (value, 20, 30)).save(path)


@pytest.mark.parametrize('value,expected', [('person/data', 'person/data'), ('https://www.kaggle.com/datasets/person/data?resource=download', 'person/data'), ('https://kaggle.com/datasets/person/data/versions/2/', 'person/data/versions/2')])
def test_handles(value, expected):
    assert dataset_handle(value) == expected


@pytest.mark.parametrize('value', ['https://evil.example/datasets/person/data', '../other', 'person/data/../../private', 'https://kaggle.com/competitions/abc', ''])
def test_rejects_invalid_handles(value):
    with pytest.raises(BackendError, match='KAGGLE_HANDLE_INVALID'):
        dataset_handle(value)


@pytest.mark.parametrize('class_first', [False, True])
def test_nested_split_folders_merge_classes_and_keep_duplicate_filenames(tmp_path, class_first):
    for split in ['train', 'val', 'test']:
        for cls in ['cats', 'dogs']:
            directory = tmp_path / 'wrapper' / (cls if class_first else split) / (split if class_first else cls)
            image(directory / 'same.png')
    paths, labels, report = image_index(tmp_path)
    assert len(paths) == 6 and len(set(paths)) == 6
    assert sorted(labels) == ['cats'] * 3 + ['dogs'] * 3
    assert report['classes'] == {'cats': 3, 'dogs': 3}
    assert report['originalSplits'] == ['test', 'train', 'val']


def test_plain_class_folders_and_tiff(tmp_path):
    image(tmp_path / 'wrapper' / 'cat' / '1.tiff')
    image(tmp_path / 'wrapper' / 'dog' / '1.png')
    _, labels, report = image_index(tmp_path)
    assert labels == ['cat', 'dog']
    assert report['layout'] == 'class-folders' and report['originalSplits'] == []


def test_csv_labels_unify_splits_without_becoming_classes(tmp_path):
    for split in ['train', 'test']:
        image(tmp_path / split / 'a.png')
        image(tmp_path / split / 'b.png')
    (tmp_path / 'labels.csv').write_text('filename,label\ntrain/a.png,cat\ntrain/b.png,dog\ntest/a.png,cat\ntest/b.png,dog\n')
    (tmp_path / 'sample_submission.csv').write_text('filename,label\nmissing.png,cat\n')
    paths, labels, report = image_index(tmp_path)
    assert len(paths) == 4 and set(labels) == {'cat', 'dog'}
    assert report['layout'] == 'csv-labels' and report['labelFiles'] == ['labels.csv']


def test_unlabeled_test_images_do_not_get_a_fake_test_class(tmp_path):
    for cls in ['cat', 'dog']:
        image(tmp_path / 'train' / cls / 'a.png')
    image(tmp_path / 'test' / 'unknown.png')
    with pytest.raises(BackendError, match='KAGGLE_UNLABELED_IMAGES'):
        image_index(tmp_path)


def test_ambiguous_csv_names_fail_instead_of_overwriting(tmp_path):
    image(tmp_path / 'train' / 'same.png'); image(tmp_path / 'test' / 'same.png')
    (tmp_path / 'labels.csv').write_text('filename,label\nsame.png,cat\n')
    with pytest.raises(BackendError, match='KAGGLE_LABEL_IMAGE_AMBIGUOUS'):
        image_index(tmp_path)


def test_masks_are_not_classification_labels(tmp_path):
    image(tmp_path / 'images' / 'a.png'); image(tmp_path / 'masks' / 'a.png')
    with pytest.raises(BackendError, match='KAGGLE_IMAGE_TASK_MISMATCH'):
        image_index(tmp_path)


def test_csv_path_traversal_rejected(tmp_path):
    image(tmp_path / 'cat' / 'a.png')
    (tmp_path / 'labels.csv').write_text('filename,label\n../outside.png,cat\n')
    with pytest.raises(BackendError, match='KAGGLE_SUBDIRECTORY_INVALID'):
        image_index(tmp_path)


def test_download_errors_hide_sdk_credentials(monkeypatch):
    def fail(_):
        raise RuntimeError('token=private-secret')
    monkeypatch.setitem(sys.modules, 'kagglehub', SimpleNamespace(dataset_download=fail))
    with pytest.raises(BackendError, match='^KAGGLE_DOWNLOAD_FAILED$'):
        download_root('person/data')


def test_download_subdirectory_cannot_escape_cache(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'kagglehub', SimpleNamespace(dataset_download=lambda _: str(tmp_path)))
    with pytest.raises(BackendError, match='KAGGLE_SUBDIRECTORY_INVALID'):
        download_root('person/data', '../elsewhere')


def test_import_pipeline_persistence_and_user_split(tmp_path, monkeypatch):
    root = tmp_path / 'cache'
    for split, count in [('train', 8), ('test', 2)]:
        for cls in ['cat', 'dog']:
            for i in range(count):
                image(root / split / cls / f'{i}.png', value=40 if cls == 'cat' else 200)
    calls = []
    def retrieve(value):
        calls.append(value)
        return str(root)
    monkeypatch.setitem(sys.modules, 'kagglehub', SimpleNamespace(dataset_download=retrieve))
    monkeypatch.setenv('MODELBUILDER_WORKSPACE', str(tmp_path / 'workspace'))
    project = {'id': 'kaggle-test', 'name': 'Kaggle test'}
    create_project(project)
    report = handle({'action': 'data.kaggle.inspect', 'handle': 'person/data'})
    dataset = handle({'action': 'data.kaggle.import', 'project': project, 'task_id': 'image.classification', 'handle': 'person/data', 'options': {'resolution': 16, 'channels': 1}})
    assert calls == ['person/data', 'person/data']
    assert dataset['samples'] == report['samples'] == 20
    assert dataset['classes'] == ['cat', 'dog'] and dataset['inputShape'] == [1, 16, 16]
    assert dataset['source'] == 'kaggle' and dataset['splitSource'] == 'generated'
    payload = torch.load(dataset['path'], weights_only=True)
    assert payload['targets'].bincount().tolist() == [10, 10]
    assert payload['inputs'][payload['targets'] == 0].float().mean() < payload['inputs'][payload['targets'] == 1].float().mean()
    persisted = json.loads((Path(dataset['path']).parent / 'manifest.json').read_text())
    assert persisted['importReport'] == dataset['importReport']
    assert persisted['importReport']['splitPolicy'] == 'merge-and-resplit'
    pipeline = {'schemaVersion': 1, 'revision': 1, 'nodes': [{'id': 'output', 'type': 'output', 'category': 'output', 'scope': 'all', 'enabled': True, 'properties': {}}]}
    updated = apply_pipeline(project, {**dataset, 'splits': {'train': 60, 'validation': 20, 'test': 20}}, 'image.classification', pipeline)
    assert updated['splitCounts'] == {'train': 12, 'validation': 4, 'test': 4}
    assert updated['samples'] == 20 and updated['importReport'] == dataset['importReport']


def test_split_specific_csv_can_resolve_repeated_names(tmp_path):
    for split in ['train', 'test']:
        for name in ['a', 'b']:
            image(tmp_path / split / 'images' / f'{name}.png')
        (tmp_path / split / 'labels.csv').write_text('filename,label\na.png,cat\nb.png,dog\n')
    paths, labels, report = image_index(tmp_path)
    assert len(paths) == 4 and labels.count('cat') == labels.count('dog') == 2
    assert report['layout'] == 'csv-labels'


def test_conflicting_csv_labels_fail(tmp_path):
    image(tmp_path / 'a.png')
    (tmp_path / 'labels.csv').write_text('filename,label\na.png,cat\na.png,dog\n')
    with pytest.raises(BackendError, match='KAGGLE_LABEL_CONFLICT'):
        image_index(tmp_path)


@pytest.mark.parametrize('options', [{'resolution': 0}, {'resolution': 64.5}, {'resolution': 'wrong'}, {'channels': 2}])
def test_invalid_image_options_rejected_before_download(options):
    from modelbuilder.kaggle_datasets import download
    with pytest.raises(BackendError, match='KAGGLE_IMAGE_OPTIONS_INVALID'):
        download({}, 'image.classification', 'person/data', options)
