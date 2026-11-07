"""A3: gianlab Swin-Tiny (HF, safetensors). Loads weights via safetensors only."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import SwinConfig, SwinForImageClassification

from .common import ROOT

MDIR = ROOT / "models" / "raw" / "a3_gianlab_swin"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).logits


def load(device="cuda"):
    # from_pretrained applies transformers-v5 key renames (old checkpoint used v4 names).
    # use_safetensors=True + local_files_only => no pickle, no network.
    model, info = SwinForImageClassification.from_pretrained(
        str(MDIR), use_safetensors=True, local_files_only=True, output_loading_info=True)
    bad = {k: [x for x in v if "relative_position_index" not in x]
           for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    assert not any(bad.values()), bad
    return Wrap(model).eval().to(device)


def preprocess(img: Image.Image) -> torch.Tensor:
    # ViTImageProcessor: resize 224x224 (bicubic, resample=3), rescale 1/255, normalize
    im = img.resize((224, 224), Image.Resampling.BICUBIC)
    a = (np.asarray(im).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1))


def reshape_transform(t, h=7, w=7):
    # (B, 49, C) -> (B, C, 7, 7)
    return t.reshape(t.size(0), h, w, t.size(2)).permute(0, 3, 1, 2)


def target_layers(wrapped):
    return [wrapped.m.swin.layernorm]
