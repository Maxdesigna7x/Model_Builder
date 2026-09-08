from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from modelbuilder.models import build_model


ROOT = Path(__file__).resolve().parents[1]

CONTRACTS = {
    "tabular.classification": ([4], [3], {}),
    "tabular.regression": ([25], [1], {}),
    "tabular.reconstruction": ([25], [25], {}),
    "image.classification": ([1, 28, 28], [10], {}),
    "image.regression": ([3, 64, 64], [1], {}),
    "image.reconstruction": ([1, 28, 28], [1, 28, 28], {}),
    "sequence.classification": ([180, 1], [5], {}),
    "sequence.regression": ([48, 25], [1], {}),
    "sequence.forecast": ([48, 25], [8, 1], {}),
    "text.classification": ([64], [6], {"vocab_size": 4096, "max_length": 64}),
    "text.language_model": ([64], [64, 4096], {"vocab_size": 4096, "max_length": 64}),
    "image.segmentation.binary": ([3, 128, 128], [1, 128, 128], {}),
    "image.segmentation.multiclass": ([3, 128, 128], [5, 128, 128], {}),
}


def _frontend_templates() -> list[dict]:
    if not shutil.which("node"):
        pytest.skip("Node.js is required to inspect the TypeScript starter templates", allow_module_level=True)
    script = """
      import { ARCHITECTURES, syncOutputContract, templateFor } from './frontend/src/catalog.ts';
      const contracts = JSON.parse(process.argv[1]);
      const result = [];
      for (const [architecture, definition] of Object.entries(ARCHITECTURES)) {
        for (const task of definition.tasks) {
          const [inputShape, outputShape, options] = contracts[task];
          const template = templateFor(architecture, task);
          const nodes = syncOutputContract(template.nodes, template.edges, {inputShape, outputShape, options});
          result.push({architecture, task, inputShape, outputShape, graph: {nodes, edges: template.edges}});
        }
      }
      console.log(JSON.stringify(result));
    """
    completed = subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", script, json.dumps(CONTRACTS)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _frontend_presets() -> list[dict]:
    script = """
      import { ARCHITECTURES, presetsFor, syncOutputContract } from './frontend/src/catalog.ts';
      const contracts = JSON.parse(process.argv[1]);
      const result = [];
      for (const [architecture, definition] of Object.entries(ARCHITECTURES)) {
        for (const task of definition.tasks) {
          const [inputShape, outputShape, options] = contracts[task];
          for (const preset of presetsFor(architecture, task)) {
            const nodes = syncOutputContract(preset.nodes, preset.edges, {inputShape, outputShape, options});
            result.push({architecture, task, preset: preset.id, inputShape, outputShape, graph: {nodes, edges: preset.edges}});
          }
        }
      }
      console.log(JSON.stringify(result));
    """
    completed = subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", script, json.dumps(CONTRACTS)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("case", _frontend_templates(), ids=lambda case: f"{case['architecture']}-{case['task']}")
def test_every_first_use_model_matches_its_catalog_dataset(case):
    model = build_model(case["architecture"], case["inputShape"], case["outputShape"], case["graph"], case["task"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 0 < parameter_count < 1_500_000

    if case["task"].startswith("text."):
        sample = torch.randint(0, 4096, (1, *case["inputShape"]))
    else:
        sample = torch.randn(1, *case["inputShape"])
    prediction = model(sample)
    assert list(prediction.shape[1:]) == case["outputShape"]
    prediction.float().square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_starter_models_use_enough_depth_for_public_visual_and_temporal_data():
    cases = {(case["architecture"], case["task"]): case for case in _frontend_templates()}
    cnn1d_types = [node["data"]["blockType"] for node in cases[("cnn1d", "sequence.forecast")]["graph"]["nodes"]]
    unet_types = [node["data"]["blockType"] for node in cases[("unet", "image.segmentation.multiclass")]["graph"]["nodes"]]
    vit_patch = next(node for node in cases[("vit", "image.classification")]["graph"]["nodes"] if node["data"]["blockType"] == "patch_embedding")

    assert cnn1d_types.count("conv1d") >= 2
    assert unet_types.count("conv2d") >= 5 and unet_types.count("relu") >= 5
    assert vit_patch["data"]["properties"]["patch_size"] <= 7


def test_every_builtin_preset_is_compatible_with_the_default_public_dataset():
    for case in _frontend_presets():
        try:
            build_model(case["architecture"], case["inputShape"], case["outputShape"], case["graph"], case["task"])
        except Exception as error:
            pytest.fail(f"{case['architecture']} / {case['task']} / {case['preset']}: {error}")
