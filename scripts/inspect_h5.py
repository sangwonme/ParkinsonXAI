"""Inspect a Keras .h5 model_config WITHOUT loading the model.
Flags Lambda / non-builtin layers. Usage: python -I inspect_h5.py <file.h5>"""
import json
import sys

import h5py

BUILTIN = {
    "InputLayer", "Conv2D", "DepthwiseConv2D", "SeparableConv2D", "BatchNormalization",
    "ReLU", "Activation", "ZeroPadding2D", "Add", "GlobalAveragePooling2D",
    "GlobalMaxPooling2D", "Dense", "Dropout", "Flatten", "Reshape", "Concatenate",
    "MaxPooling2D", "AveragePooling2D", "Functional", "Model", "Sequential", "Softmax",
}


def walk(cfg, out, depth=0):
    if isinstance(cfg, dict):
        if "class_name" in cfg and "config" in cfg:
            out.append((depth, cfg["class_name"], cfg["config"].get("name") if isinstance(cfg["config"], dict) else None))
            depth += 1
        for v in cfg.values():
            walk(v, out, depth)
    elif isinstance(cfg, list):
        for v in cfg:
            walk(v, out, depth)


path = sys.argv[1]
with h5py.File(path, "r") as f:
    attrs = dict(f.attrs)
    print("top-level attrs:", list(attrs.keys()))
    print("keras_version:", attrs.get("keras_version"), "backend:", attrs.get("backend"))
    raw = attrs["model_config"]
    if isinstance(raw, bytes):
        raw = raw.decode()
    cfg = json.loads(raw)

layers = []
walk(cfg, layers)
classes = {}
for _, c, _ in layers:
    classes[c] = classes.get(c, 0) + 1
print("layer classes:", classes)
suspicious = sorted({c for c in classes if c not in BUILTIN})
print("NON-BUILTIN / SUSPICIOUS:", suspicious if suspicious else "none")
raw_l = raw.lower()
print("contains 'lambda':", "lambda" in raw_l, "| 'function':", '"function"' in raw_l,
      "| 'module' custom:", '"module": "__main__"' in raw_l)
print("top-level class:", cfg.get("class_name"))
print("first/last layers:", layers[:4], layers[-4:])
