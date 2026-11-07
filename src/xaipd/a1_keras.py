"""A1: Parkonix Teachable-Machine Keras model (.h5). TF-native Grad-CAM / Grad-CAM++ / Score-CAM.
Load only after scripts/inspect_h5.py reports no Lambda / custom layers."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .common import ROOT

H5 = ROOT / "models" / "raw" / "a1_parkonix" / "keras_model.h5"


def load():
    return tf.keras.models.load_model(str(H5), compile=False)


def _flatten_layers(model):
    """Yield (layer) in call order, expanding nested Sequential/Functional models."""
    for l in model.layers:
        if isinstance(l, tf.keras.Model):
            yield from _flatten_layers(l)
        else:
            yield l


def build_cam_model(model, conv_name: str | None = None):
    """Return a Keras model x -> (conv_activations, probs).
    Teachable Machine = Sequential[ Sequential[MobileNetV2 Functional], Sequential[Dense...] ].
    We locate the inner Functional backbone, take the target conv output from it, then
    re-apply the remaining head layers on the backbone output."""
    outer = model
    # find the functional backbone (the nested model whose input is the image)
    backbone = None
    for l in outer.layers:
        if isinstance(l, tf.keras.Model):
            for ll in l.layers:
                if isinstance(ll, tf.keras.Model) and len(ll.layers) > 20:
                    backbone = ll
            if backbone is None and len(l.layers) > 20:
                backbone = l
        if backbone is not None:
            break
    assert backbone is not None, "backbone not found"
    if conv_name is None:
        # last layer with 4D output inside backbone
        cand = [l for l in backbone.layers if len(l.output.shape) == 4]
        conv_name = cand[-1].name
    conv_out = backbone.get_layer(conv_name).output
    feat_model = tf.keras.Model(backbone.inputs, [conv_out, backbone.output])

    # head = every non-backbone layer after the backbone in flattened order
    flat_outer = []
    for l in outer.layers:
        if isinstance(l, tf.keras.Model):
            flat_outer.extend(l.layers)
        else:
            flat_outer.append(l)
    idx = [i for i, l in enumerate(flat_outer) if l is backbone][0]
    head_layers = flat_outer[idx + 1:]

    x = tf.keras.Input(shape=(224, 224, 3))
    a, h = feat_model(x)
    for l in head_layers:
        if isinstance(l, tf.keras.layers.InputLayer):
            continue
        h = l(h)
    return tf.keras.Model(x, [a, h]), conv_name, [l.name for l in head_layers]


def gradcam(cam_model, x, cls, plus=False):
    x = tf.convert_to_tensor(x[None] if x.ndim == 3 else x)
    with tf.GradientTape() as tape:
        a, p = cam_model(x, training=False)
        s = p[:, cls]
    g = tape.gradient(s, a)[0].numpy()  # (h, w, k)
    a = a[0].numpy()
    if not plus:
        w = g.mean(axis=(0, 1))
    else:
        # Grad-CAM++ closed form (Chattopadhay 2018), assuming exp score
        g2, g3 = g ** 2, g ** 3
        denom = 2 * g2 + (a * g3).sum(axis=(0, 1), keepdims=True)
        alpha = g2 / np.where(denom != 0, denom, 1e-8)
        w = (alpha * np.maximum(g, 0)).sum(axis=(0, 1))
    cam = np.maximum((a * w).sum(-1), 0)
    return cam


def scorecam(cam_model, x, cls, batch=32, max_ch=None):
    a, p0 = cam_model(x[None], training=False)
    a = a[0].numpy()  # (h, w, k)
    k = a.shape[-1]
    idx = np.arange(k)
    if max_ch:  # keep strongest channels for speed
        idx = np.argsort(-a.mean(axis=(0, 1)))[:max_ch]
    up = tf.image.resize(a[..., idx][None], (224, 224), method="bilinear")[0].numpy()
    mn, mx = up.min(axis=(0, 1)), up.max(axis=(0, 1))
    masks = (up - mn) / np.where(mx - mn > 0, mx - mn, 1)
    # Teachable Machine input is in [-1,1]; mask in [0,1] space then back
    x01 = (x + 1) / 2
    scores = []
    for i in range(0, len(idx), batch):
        m = masks[..., i:i + batch].transpose(2, 0, 1)[..., None]
        xb = x01[None] * m * 2 - 1
        _, pb = cam_model(xb, training=False)
        scores.append(pb.numpy()[:, cls])
    s = np.concatenate(scores)
    w = tf.nn.softmax(s).numpy()
    cam = np.maximum((a[..., idx] * w).sum(-1), 0)
    return cam
