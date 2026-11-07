from __future__ import annotations

import sys
import os
from types import SimpleNamespace

import torch
from torch import nn

from modelbuilder.models import HuggingFaceImageProcessor, HuggingFaceTextBackbone, HuggingFaceVisionBackbone


class FakeVision(nn.Module):
    def __init__(self): super().__init__();self.projection=nn.Conv2d(3,5,1)
    def forward(self,pixel_values): return SimpleNamespace(pooler_output=self.projection(pixel_values).mean((2,3)))


class FakeText(nn.Module):
    def __init__(self): super().__init__();self.embedding=nn.Embedding(32,7)
    def forward(self,input_ids,attention_mask): return SimpleNamespace(pooler_output=None,last_hidden_state=self.embedding(input_ids))


def test_vision_adapter_applies_pinned_safe_processor_and_freezes(monkeypatch):
    calls=[]
    class Processor:
        @staticmethod
        def from_pretrained(repo,**kwargs): calls.append((repo,kwargs));return SimpleNamespace(size={"height":16},image_mean=[.5,.5,.5],image_std=[.25,.25,.25])
    class Model:
        @staticmethod
        def from_pretrained(repo,**kwargs): calls.append((repo,kwargs));return FakeVision()
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(AutoImageProcessor=Processor,AutoModel=Model))
    props={"repo_id":"owner/model","revision":"a"*40,"frozen":True}
    processor=HuggingFaceImageProcessor(props);backbone=HuggingFaceVisionBackbone(props);output=backbone(processor(torch.rand(2,1,8,8)))
    assert output.shape==(2,5) and all(not value.requires_grad for value in backbone.parameters())
    assert all(value[1]["revision"]=="a"*40 and value[1]["trust_remote_code"] is False for value in calls)
    assert calls[-1][1]["use_safetensors"] is True


def test_text_adapter_builds_attention_mask_and_trainable_head_contract(monkeypatch):
    class Config:
        @staticmethod
        def from_pretrained(*args,**kwargs): return SimpleNamespace(pad_token_id=0)
    class Model:
        @staticmethod
        def from_pretrained(*args,**kwargs): return FakeText()
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(AutoConfig=Config,AutoModel=Model))
    backbone=HuggingFaceTextBackbone({"repo_id":"owner/text","revision":"b"*40,"frozen":True})
    output=backbone(torch.tensor([[1,2,0],[3,0,0]]))
    assert output.shape==(2,7) and torch.isfinite(output).all()
    assert all(not value.requires_grad for value in backbone.parameters())


def test_optional_real_hub_integration():
    if os.environ.get("MODELBUILDER_HF_INTEGRATION")!="1": return
    revision="96e253fe90cfcededfa8edf8b7e6f230f87a63a1";repo="hf-internal-testing/tiny-random-vit";props={"repo_id":repo,"revision":revision,"frozen":True}
    output=HuggingFaceVisionBackbone(props)(HuggingFaceImageProcessor(props)(torch.rand(1,3,32,32)))
    assert output.shape[0]==1 and torch.isfinite(output).all()
