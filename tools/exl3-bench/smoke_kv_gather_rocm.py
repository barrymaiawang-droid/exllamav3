#!/usr/bin/env python3
"""Tiny no-weight ROCm smoke for gather_kv_prefix: real CacheLayer_fp16 tensors ON THE GPU.

Allocates two 4-page layers (a few hundred KB), writes identical logical prefixes into different
physical pages with different unused storage, and checks the digest is identical; then changes a row
within the valid prefix and requires a different digest. No model, no weights, no rebuild.
"""
import importlib.util, json, os, sys, types
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("probe_mtp_causal", os.path.join(HERE, "probe_mtp_causal.py"))
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)

SRC = os.environ.get("EXL3_SRC", os.path.expanduser("~/exl3-v164-rocm10-clean/exllamav3"))
fp16_path = os.path.join(SRC, "cache/fp16.py")
if not os.path.exists(fp16_path):
    print(json.dumps({"smoke": "skipped", "reason": f"missing {fp16_path}"})); sys.exit(3)
src = open(fp16_path).read()
start = src.index("class CacheLayer_fp16")
end = src.index("\nclass ", start + 10) if "\nclass " in src[start + 10:] else len(src)
ns = {"__name__": "smoke_fp16", "torch": torch, "PAGE_SIZE": probe.PAGE_SIZE,
      "CacheLayer": type("CacheLayer", (), {}), "override": lambda f: f,
      "np": __import__("numpy"), "ext": type("E", (), {})()}
exec(compile("from __future__ import annotations\n" + src[start:end], fp16_path, "exec"), ns)
Cls = ns["CacheLayer_fp16"]

dev = torch.device("cuda:0")
assert torch.cuda.is_available()
props = torch.cuda.get_device_properties(0)
PAGE, heads, dim = probe.PAGE_SIZE, 2, 8
consumed = PAGE + 5


def layer(seed):
    l = object.__new__(Cls)
    l.shape = (4, PAGE, heads, dim)
    l.k = torch.full(l.shape, float(seed), dtype=torch.float16, device=dev)
    l.v = torch.full(l.shape, float(seed) + 0.5, dtype=torch.float16, device=dev)
    l.device = dev
    return l


def write(l, mapping, value_from):
    for pos in range(consumed):
        page, off = mapping[pos // PAGE], pos % PAGE
        l.k[page, off] = value_from(pos)
        l.v[page, off] = value_from(pos) * 0.5


a, b = layer(1.0), layer(2.0)                      # unused storage differs
write(a, [0, 1], lambda p: float(p % 97))
write(b, [3, 2], lambda p: float(p % 97))          # different physical pages, same logical content
fa = probe.kv_fingerprint(types.SimpleNamespace(layers={"L": a}), consumed, [0, 1])
fb = probe.kv_fingerprint(types.SimpleNamespace(layers={"L": b}), consumed, [3, 2])
assert fa["layers"]["L"]["k_sha256"] == fb["layers"]["L"]["k_sha256"], "logical prefix mismatch"
original = b.k[2, 1].clone()
b.k[2, 1] = -7.0                                   # inside the valid prefix
fc = probe.kv_fingerprint(types.SimpleNamespace(layers={"L": b}), consumed, [3, 2])
assert fc["layers"]["L"]["k_sha256"] != fa["layers"]["L"]["k_sha256"], "valid-prefix change missed"
b.k[2, 1] = original
b.k[2, consumed % PAGE:] = 99.0                    # past the consumed length: must be ignored
fd = probe.kv_fingerprint(types.SimpleNamespace(layers={"L": b}), consumed, [3, 2])
assert fd["layers"]["L"]["k_sha256"] == fa["layers"]["L"]["k_sha256"], "tail not truncated"
print(json.dumps({"smoke": "ok", "device": props.gcnArchName, "consumed": consumed,
                  "pages": [0, 1], "alt_pages": [3, 2],
                  "k_sha256": fa["layers"]["L"]["k_sha256"],
                  "prefix_shape": fa["layers"]["L"]["prefix_shape"]}))
