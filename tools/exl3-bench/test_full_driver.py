#!/usr/bin/env python3
"""Full-driver CPU fixture for probe_mtp_causal.run() — real exllamav3 classes, no weights, no GPU.

Everything the driver touches is exercised: the fake `exllamav3` package supplies Config/Model/
Tokenizer/Cache/Generator/Job, while the *state and KV objects are the real upstream classes*
(`GDNState`, `GDNLayerState`, `CacheLayer_fp16`) instantiated with `object.__new__` and CPU tensors.
The fake model is causal (row r sees only tokens up to r), advances the recurrent state, writes the
real paged KV layout, and returns `None` for prefill exactly as the real `last_tokens_only` path does.

Covered here, because helper-only tests could not:
  * run() end to end: prefill with no logits, decode, a *rejected* verifier batch spanning index 18
    followed by the accepted same-context pass, selection, ledger, replay controls, comparisons;
  * replay self-check enforcement: a corrupted replay must raise and leave `*.FAILED.json`;
  * KV canonicalisation: physical pages in block-table order, truncated to `consumed`, ignoring
    unused pages/rows — with the real `CacheLayer_fp16.get_kv()` that returns whole buffers;
  * conv canonicalisation: only the committed `[0, conv_kernel_size)` columns count, not the
    speculative tail;
  * source conformance: which class owns base_row/parity, and that `get_kv` is a passthrough.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("probe_mtp_causal", os.path.join(HERE, "probe_mtp_causal.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

SRC = os.environ.get("EXL3_SRC", os.path.expanduser("~/exl3-v164-rocm10-clean/exllamav3"))
PAGE = probe.PAGE_SIZE
PROMPT = [1000 + i for i in range(8)]
P = len(PROMPT)
GEN = [2000 + i for i in range(24)]
VOCAB = 8192
FILLER = 4237


# --------------------------------------------------------------------- real upstream classes

def _load_real_module(name, relpath):
    path = os.path.join(SRC, relpath)
    if not os.path.exists(path):
        pytest.skip(f"upstream source not present: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # exllamav3 imports its compiled ext
        pytest.skip(f"cannot import {relpath} without the extension: {exc}")
    return mod


@pytest.fixture(scope="module")
def gdn():
    """The real GatedDeltaNet module, imported from the v1.6.4 tree when possible."""
    if "probe_real_gdn" in sys.modules:
        return sys.modules["probe_real_gdn"]
    path = os.path.join(SRC, "modules/gated_delta_net.py")
    if not os.path.exists(path):
        pytest.skip("upstream source not present")
    src = open(path).read()
    # Import only the two state classes: the module's top-level imports need the compiled extension.
    ns = {"__name__": "probe_real_gdn", "torch": torch}
    exec(compile(_extract_classes(src, ("GDNState", "GDNLayerState")), path, "exec"), ns)
    sys.modules["probe_real_gdn"] = ns
    return ns


def _extract_classes(src, names):
    """Return only `class <name>` blocks (with their decorators), so no module-level imports run."""
    out = []
    for name in names:
        marker = f"class {name}"
        start = src.index(marker)
        # walk back to the preceding blank line after the previous top-level statement
        end = len(src)
        for other in ("\nclass ", "\ndef ", "\n@"):
            idx = src.find(other, start + len(marker))
            if idx != -1:
                end = min(end, idx + 1)
        out.append(src[start:end])
    return "\n".join(out) + "\n"


@pytest.fixture(scope="module")
def fp16(probe_real_fp16):
    return probe_real_fp16


def _cache_stub(layers):
    return types.SimpleNamespace(layers=layers, initialized=True)


def real_kv_layer(pages=4, heads=2, dim=8, seed=0):
    layer = object.__new__(_fp16_ns()["CacheLayer_fp16"])
    layer.shape = (pages, PAGE, heads, dim)
    layer.k = torch.full(layer.shape, float(seed), dtype=torch.float16)
    layer.v = torch.full(layer.shape, float(seed) + 0.5, dtype=torch.float16)
    layer.device = torch.device("cpu")
    return layer


def _fp16_ns():
    if "probe_real_fp16" in sys.modules:
        return sys.modules["probe_real_fp16"]
    path = os.path.join(SRC, "cache/fp16.py")
    if not os.path.exists(path):
        pytest.skip("upstream source not present")
    src = open(path).read()
    ns = {"__name__": "probe_real_fp16", "torch": torch,
          "PAGE_SIZE": PAGE}
    # CacheLayer_fp16 subclasses CacheLayer and decorates with @override; install minimal stand-ins.
    ns["CacheLayer"] = type("CacheLayer", (), {})
    ns["override"] = lambda f: f
    ns["np"] = __import__("numpy")
    ns["ext"] = types.SimpleNamespace()
    exec(compile(_extract_classes(src, ("CacheLayer_fp16",)), path, "exec"), ns)
    sys.modules["probe_real_fp16"] = ns
    return ns


@pytest.fixture(scope="module")
def probe_real_fp16():
    return _fp16_ns()


def real_layer_state(GDNLayerState, fdim=6, kernel=4, max_history=3, heads=2, kdim=2, vdim=2):
    ls = object.__new__(GDNLayerState)
    ls.module = types.SimpleNamespace(conv_kernel_size=kernel, fdim_qkv=fdim,
                                      num_v_heads=heads, k_head_dim=kdim, v_head_dim=vdim, kda=False)
    ls.conv_state = torch.zeros((1, fdim, kernel + max_history), dtype=torch.bfloat16)
    ls.recurrent_state = torch.zeros((2, 1, heads, kdim, vdim), dtype=torch.float16)
    ls.g_width = heads
    ls.device = torch.device("cpu")      # the real clear() writes only when a device is set
    return ls


def real_state(GDNState, cache, slot=0, position=0, parity=0):
    st = object.__new__(GDNState)
    st.slot, st.position, st.parity = slot, position, parity
    st.last_history = 0
    st.cache = cache
    st.spec_row = None
    st.spec_shape = None
    return st


class FakeRecurrentLayers(dict):
    pass


class FakeCache:
    """Holds real KV layers and real GDN layer states; no model, no device.

    `get_new_state()` constructs the **real** `GDNState`, so `GDNLayerState.clear()` ->
    `conv_state[idx].zero_()` runs exactly as in the failing window. With `inference_tensors=True` the
    state tensors are created inside inference mode, i.e. they are inference tensors: the clear then
    raises unless the caller is inside `torch.inference_mode()`, which is the regression the window hit.
    """

    def __init__(self, gdn, fp16, n_kv=2, n_gdn=2, seed=0, inference_tensors=False):
        self.initialized = True
        self.layers = {(i * 4, 0): real_kv_layer(seed=seed) for i in range(n_kv)}
        self.inference_tensors = inference_tensors
        self.recurrent_layers = {(i * 4, 0): self._layer_state(gdn["GDNLayerState"])
                                 for i in range(n_gdn)}
        self._GDNState = gdn["GDNState"]
        self.model = types.SimpleNamespace(loaded_tp=False)
        self.states = []
        self.state_alloc_in_inference = []   # every state allocation must happen under inference mode

    def _layer_state(self, GDNLayerState):
        if self.inference_tensors:
            with torch.inference_mode():
                return real_layer_state(GDNLayerState)
        return real_layer_state(GDNLayerState)

    def get_all_recurrent_layers(self):
        return self.recurrent_layers

    def reset_states(self):
        self.states = []

    def get_new_state(self):
        # the real constructor: asserts position 0, then clears every recurrent layer (in-place writes)
        self.state_alloc_in_inference.append(torch.is_inference_mode_enabled())
        st = self._GDNState(self, len(self.states), 0)
        self.states.append(st)
        return st

    def release_state(self, st):
        pass


class FakeModel:
    """Causal stand-in. Row r of a pass at cache_seqlens s predicts index s+r+1-P, and its logits
    depend only on tokens at or before that row. Batched rows get a small bump on a filler token so
    the 3-row control differs from the 1-row replay without changing the argmax."""

    def __init__(self, target=True, corrupt=None, corrupt_conv=False):
        self.target = target
        self.corrupt_conv = corrupt_conv
        self.conv_corruptions = []
        self.loaded = False
        self.positions = {}
        self.corrupt = corrupt or set()
        self.corrupt_after = 1 if corrupt else None
        self.counts = {}
        self.corruptions = []          # (cache_seqlen, row, call-count) actually flipped, for the tests
        self.seen_params = []          # (method, input shape, positions shape) for shape assertions

    def _logits(self, cache_seqlen, rows):
        out = torch.full((1, rows, VOCAB), -10.0)
        for r in range(rows):
            idx = cache_seqlen + r + 1 - P
            tok = GEN[idx] if 0 <= idx < len(GEN) else GEN[-1]
            out[0, r, tok] = 10.0
            if rows > 1:
                out[0, r, FILLER] = 2.0
            key = (cache_seqlen, r)
            self.counts[key] = self.counts.get(key, 0) + 1
            if (cache_seqlen, r) in self.corrupt and self.corrupt_after is not None \
                    and self.counts[key] > self.corrupt_after:
                out[0, r, FILLER] = 11.0        # flips the argmax: a replay that cannot reproduce AR
                self.corruptions.append((cache_seqlen, r, self.counts[key]))
        return out

    def load(self, *_a, **_kw):
        self.loaded = True
        return self

    def prefill(self, input_ids, params):
        """Model.prefill: writes cache/state, advances recurrence, returns no logits."""
        self.seen_params.append(("prefill", tuple(input_ids.shape),
                                 tuple(params["positions"].shape) if params.get("positions") is not None else None))
        if getattr(self, "require_inference", False) and not torch.is_inference_mode_enabled():
            raise RuntimeError("Inplace update to inference tensor outside InferenceMode is not allowed.")
        self.prefill_calls = getattr(self, "prefill_calls", 0) + 1
        self._write(input_ids, params)
        return None

    def forward(self, input_ids, params):
        self.seen_params.append(("forward", tuple(input_ids.shape),
                                 tuple(params["positions"].shape) if params.get("positions") is not None else None))
        if getattr(self, "require_inference", False) and not torch.is_inference_mode_enabled():
            raise RuntimeError("Inplace update to inference tensor outside InferenceMode is not allowed.")
        seqlen = int(params["cache_seqlens"].reshape(-1)[0])
        rows = int(input_ids.shape[1])
        st = (params.get("recurrent_states") or [None])[0]
        self._write(input_ids, params)
        if params.get("last_tokens_only"):
            return None                          # real API: MTP prefill returns no logits
        return self._logits(seqlen, rows)

    def _write(self, input_ids, params):
        seqlen = int(params["cache_seqlens"].reshape(-1)[0])
        rows = int(input_ids.shape[1])
        st = (params.get("recurrent_states") or [None])[0]
        # write KV in the real paged layout for the rows this pass consumes
        cache = params["cache"]
        block = [int(x) for x in params["block_table"][0].tolist()]
        for li, layer in cache.layers.items():
            for r in range(rows):
                pos = seqlen + r
                page, off = block[pos // PAGE], pos % PAGE
                tok = int(input_ids[0, r]) % 100
                layer.k[page, off] = float(tok + li[0])
                layer.v[page, off] = float(tok + li[0]) * 0.5
        if self.corrupt_conv and self.counts.get((seqlen, rows), 0) > 1:
            # corrupt ONLY the replay's committed conv window (a control-prerequisite failure that is
            # not a logits difference), and record that the injection fired
            for _, ls in cache.recurrent_layers.items():
                k_size = int(ls.module.conv_kernel_size)
                ls.conv_state[st.slot if st is not None else 0, 0, 0] = 42.0
                self.conv_corruptions.append((seqlen, rows, k_size))
        if st is not None:
            st.position = seqlen + rows

    def rewind_to(self, st, seqlen):
        st.position = seqlen


# --------------------------------------------------------------------- fake exllamav3 API

class FakeConfig:
    vocab_size = VOCAB

    @staticmethod
    def from_directory(_path):
        return FakeConfig()


class FakeTokenizer:
    def encode(self, _prompt, encode_special_tokens=True):
        return torch.tensor([PROMPT])

    @staticmethod
    def from_config(_config):
        return FakeTokenizer()


class FakeSequence:
    def __init__(self, ids):
        self._ids = ids

    @property
    def sequence_ids(self):
        return types.SimpleNamespace(torch=lambda: self._ids)


class FakeJob:
    def __init__(self, input_ids, max_new_tokens, min_new_tokens=1, sampler=None,
                 decode_special_tokens=True, return_top_tokens=0):
        self.prompt = [int(x) for x in input_ids[0].tolist()]
        self.max_new = max_new_tokens
        self.new_tokens = 0
        self.draft_stats = []
        self.time_first_token = self.time_last_token = None
        self.sequences = [FakeSequence(torch.zeros((1, len(self.prompt)), dtype=torch.long))]
        self.argmax = None


class FakeGenerator:
    """Emits the recorded AR/verifier protocol, including a rejected batch before the accepted one."""

    def __init__(self, model, cache, tok, draft_model=None, draft_cache=None, num_draft_tokens=0,
                 record_draft_stats=False, **_kw):
        self.model, self.cache, self.p = model, cache, None
        self.num_draft = num_draft_tokens
        self.state = None
        self.generated = []
        self.done = False

    @torch.inference_mode()          # the real Generator runs its queue under inference mode
    def enqueue(self, job):
        self.job = job
        self.state = self.cache.get_new_state()
        return 7

    def _call(self, rows, seqlen, history=False, last_only=False):
        used = (seqlen + len(rows) + PAGE - 1) // PAGE
        params = {"attn_mode": "flash_attn", "cache": self.cache,
                  "block_table": torch.tensor([list(range(used)) + [0] * (4 - used)], dtype=torch.int32),
                  "cache_seqlens": torch.tensor([seqlen], dtype=torch.int32),
                  "positions": torch.tensor([seqlen], dtype=torch.int32),   # 1-D per-sequence offset
                  "recurrent_states": [self.state], "recurrent_history": history,
                  "last_tokens_only": last_only, "pinned_staging": True}
        return self.model.forward(torch.tensor([rows], dtype=torch.long), params)

    @torch.inference_mode()          # generator.py:626/815/889 decorate the real iteration paths
    def iterate(self):
        if self.done:
            return []
        # prefill: P-1 tokens. AR goes through Model.prefill; an MTP drafter forces
        # Model.forward(last_tokens_only=1) (generator/job.py:1591-1595)
        if self.num_draft:
            self._call(PROMPT[:P - 1], 0, last_only=True)
        else:
            used = (P - 1 + PAGE - 1) // PAGE
            self.model.prefill(torch.tensor([PROMPT[:P - 1]], dtype=torch.long),
                               {"attn_mode": "flash_attn", "cache": self.cache,
                                "block_table": torch.tensor([list(range(used))], dtype=torch.int32),
                                "cache_seqlens": torch.tensor([0], dtype=torch.int32),
                                "recurrent_states": [self.state], "pinned_staging": True})
        pending = PROMPT[-1]
        k = 0
        while len(self.generated) < self.job.max_new:
            if self.num_draft and k == 16:
                # rejected verifier batch spanning 16..18: row 2 carries a draft, not the accepted token
                seqlen = P - 1 + 16
                self._call([GEN[15], GEN[16], 777], seqlen, history=True)
                self.state.position = seqlen          # rewind: nothing accepted from that pass
                self.job.draft_stats.append([16, 3, 0])
                # then the same position on the accepted context: row 2 is index 18
                self._call([GEN[15], GEN[16], GEN[17]], seqlen, history=True)
                self.generated.extend([GEN[16], GEN[17], GEN[18]])
                self.job.draft_stats.append([16, 3, 3])
                pending = GEN[18]
                k = 19
                continue
            logits = self._call([pending], P - 1 + k)
            tok = int(torch.argmax(logits[0, 0]))
            self.generated.append(tok)
            pending = tok
            k += 1
        self.job.new_tokens = len(self.generated)
        ids = torch.tensor([self.job.prompt + self.generated], dtype=torch.long)
        self.job.sequences = [FakeSequence(ids)]
        self.done = True
        return [{"serial": 7, "stage": "eos", "eos": True, "eos_reason": "max_new_tokens"}]


def install_fake_api(monkeypatch, target, draft):
    pkg = types.ModuleType("exllamav3")
    pkg.__file__ = "/fake/exllamav3/__init__.py"
    pkg.Config, pkg.Model, pkg.Cache = FakeConfig, FakeModel, FakeCache
    pkg.Tokenizer, pkg.Generator, pkg.Job = FakeTokenizer, FakeGenerator, FakeJob
    pkg.ext = types.ModuleType("exllamav3.ext")
    # a real (dummy) extension file, so the mandatory startup hashing is exercised unchanged
    import tempfile
    ext_dir = tempfile.mkdtemp(prefix="fake-ext-")
    ext_path = os.path.join(ext_dir, "exllamav3_ext.so")
    with open(ext_path, "wb") as f:
        f.write(b"dummy extension bytes for identity hashing")
    pkg.ext.exllamav3_ext = types.SimpleNamespace(__file__=ext_path)
    gen = types.ModuleType("exllamav3.generator")
    sampler = types.ModuleType("exllamav3.generator.sampler")
    sampler.ArgmaxSampler = lambda: "argmax"
    ver = types.ModuleType("exllamav3.version")
    ver.__version__ = "1.6.4-test"
    for name, mod in (("exllamav3", pkg), ("exllamav3.ext", pkg.ext),
                      ("exllamav3.generator", gen), ("exllamav3.generator.sampler", sampler),
                      ("exllamav3.version", ver)):
        monkeypatch.setitem(sys.modules, name, mod)

    # Config.from_directory -> the two models, in construction order
    calls = {"n": 0}

    def from_config(config, component=None):
        return draft if component == "mtp" else target

    pkg.Model.from_config = staticmethod(from_config)
    return pkg


# --------------------------------------------------------------------- driver fixture

def run_driver(monkeypatch, gdn, fp16, out_path, corrupt=(), holder=None, corrupt_conv=False,
               fail_fast=None, inference_tensors=False, capture_dir=None, capture_positions=None):
    target = FakeModel(target=True, corrupt=set(corrupt), corrupt_conv=corrupt_conv)
    if holder is not None:
        holder["target"] = target
    draft = FakeModel(target=False)
    pkg = install_fake_api(monkeypatch, target, draft)

    caches = {}

    class DriverCache(FakeCache):
        def __init__(self, *a, **kw):
            seed = len(caches)
            super().__init__(gdn, fp16, seed=seed)
            caches[id(self)] = self

    pkg.Cache = DriverCache
    args = types.SimpleNamespace(model="/fake/model", ctx=8192, index=18, filler=FILLER,
                                 prompt="Write a short paragraph about the ocean.",
                                 out=str(out_path), tensor_dir=str(out_path) + "-tensors")
    if capture_dir is not None:
        args.capture_dir = str(capture_dir)
        args.capture_positions = capture_positions or "7,8,9"
        args.capture_layer = 0
        args.capture_device = "cuda:0"
        args.capture_device_index = None
    if fail_fast is not None:
        # the driver must keep a partial artifact when the run itself raises after wiring up
        try:
            raise fail_fast
        finally:
            pass
    rec = probe.run(args)
    return rec, caches, target


# --------------------------------------------------------------------- tests

def test_full_driver_executes_end_to_end(monkeypatch, gdn, fp16, tmp_path):
    rec, caches, target = run_driver(monkeypatch, gdn, fp16, tmp_path / "probe.json")
    # prefill consumed P-1 and the ledger held on every recorded call
    assert rec["prefill_prediction"]["prefill_consumes"] == P - 1
    assert rec["gen_ar"]["generated_ids"][:18] == GEN[:18] == rec["gen_mtp2"]["generated_ids"][:18]
    # the rejected batch spanning 18 comes first and must not be selected
    mtp_sel = rec["selection"]["mtp2"]
    assert mtp_sel["row"] >= 0
    assert mtp_sel["discarded_candidates"], "the rejected overlapping batch must be recorded"
    assert any(d["row"] == 2 and d["cache_seqlens"] == P - 1 + 16
               for d in mtp_sel["discarded_candidates"]), mtp_sel["discarded_candidates"]
    assert (mtp_sel["row"], mtp_sel["cache_seqlens"]) == (2, P - 1 + 16), \
        "the accepted 3-row pass must be selected, not the rejected batch"
    # the rejected batch really altered an input row at the selected index, and a later valid pass
    # reached the same index on the accepted context
    mtp_calls = rec["gen_mtp2"]["calls"]
    rejected = [c for c in mtp_calls
                if c["cache_seqlens"] == P - 1 + 16 and c["rows"] == 3 and c["input_ids"][0][2] != GEN[17]]
    accepted = [c for c in mtp_calls
                if c["cache_seqlens"] == P - 1 + 16 and c["rows"] == 3 and c["input_ids"][0][2] == GEN[17]]
    assert rejected and rejected[0]["input_ids"][0][2] == 777, "rejected injection did not fire"
    assert accepted, "no later same-context pass reached index 18"
    assert rec["replay_selfcheck"]["logits_exact"] is True
    assert rec["replay_selfcheck"]["state_equal"] is True
    assert rec["seq"]["entry_state_matches_recorded"] is True
    # KV was actually captured and compared, with full layer coverage
    assert rec["seq"]["kv"]["n_layers"] == rec["seq"]["kv"]["expected_layers"] > 0
    assert rec["kv_ar_vs_mtp2"]["equal"] is True
    # conv canonical columns only, tail excluded
    layer0 = next(iter(rec["seq"]["entry_state"]["layers"].values()))
    assert layer0["conv_shape"][-1] == layer0["conv_kernel_size"] == 4
    assert layer0["conv_tail_cols"] == 3
    json.dumps(rec)                                    # the whole report is serializable
    tdir = os.path.join(str(tmp_path / "probe.json") + "-tensors")
    assert os.path.isdir(tdir)
    for name in ("probe-logits.pt", "probe-kv.pt"):
        assert os.path.exists(os.path.join(tdir, name)), name


def _fake_gdn_module():
    """A GDN module whose three patchable seams exist but are never called by the fake model.

    The capture only needs the seam signatures and the module globals `install()` reads, so a
    capture can be installed and then fail completeness without a real layer running.
    """
    mod = types.ModuleType("exllamav3.modules.gated_delta_net")
    ns = mod.__dict__
    exec(compile(
        "def forward(self, x, params, out_dtype=None):\n"
        "    raise AssertionError('the fake driver does not run the real layer')\n"
        "class GatedDeltaNet:\n"
        "    pass\n"
        "GatedDeltaNet.forward = forward\n"
        "class GDNState:\n"
        "    def rewind(self, num_tokens):\n"
        "        return None\n"
        "class GDNLayerState:\n"
        "    def replay_job(self, spec_row, prefix, spec_shape, base_row, scratch_row):\n"
        "        return None\n", "<fake-gdn>", "exec"), ns)
    ns["g_tensor_cache"] = types.SimpleNamespace()      # resolved by install(); never used here
    ns["_BC_MAX_BSZ"] = 8
    ns["_BC_MAX_QLEN"] = 16
    return mod


def test_capture_completeness_failure_keeps_flushed_evidence_and_restores_hooks(
        monkeypatch, gdn, fp16, tmp_path):
    """A diagnostic capture failure must not discard a completed generator's evidence.

    The capture is installed on never-called seams, so it records nothing and its completeness check
    fails. The already-flushed ids/logits/state evidence must survive, the partial capture must be
    retained, and every patched seam must be restored.
    """
    fake = _fake_gdn_module()
    pkg = types.ModuleType("exllamav3.modules")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "exllamav3.modules", pkg)
    monkeypatch.setitem(sys.modules, "exllamav3.modules.gated_delta_net", fake)
    spec = importlib.util.spec_from_file_location("capture_layer0",
                                                 os.path.join(HERE, "capture_layer0.py"))
    capture_layer0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture_layer0)
    monkeypatch.setitem(sys.modules, "capture_layer0", capture_layer0)
    orig = (fake.GatedDeltaNet.forward, fake.GDNState.rewind, fake.GDNLayerState.replay_job)
    out = tmp_path / "probe.json"
    capdir = tmp_path / "capture"
    with pytest.raises(capture_layer0.CaptureError) as exc:
        run_driver(monkeypatch, gdn, fp16, out, capture_dir=capdir, capture_positions="7,8,9")
    assert "incomplete" in str(exc.value) and "0 decode passes" in str(exc.value)
    # the completed generator was flushed BEFORE the capture failure escaped
    flushed = json.loads(open(str(out) + ".gen_ar.json").read())
    assert flushed["generated_ids"] and len(flushed["generated_ids"]) >= 18
    assert flushed["logits"]["sha256"] and flushed["selected"]["seq"] is not None
    assert os.path.isdir(str(out) + "-tensors")
    # the partial capture artifact survives with its own (empty, error-free) record set
    doc = json.loads((capdir / "ar" / "gen_ar-layer0.json").read_text())
    assert doc["records"] == [] and doc["errors"] == []
    # every seam the capture patched is restored
    assert (fake.GatedDeltaNet.forward, fake.GDNState.rewind,
            fake.GDNLayerState.replay_job) == orig


def test_false_replay_selfcheck_fails_nonzero_and_keeps_artifacts(monkeypatch, gdn, fp16, tmp_path):
    # corrupt the replay at the selected position: the control cannot reproduce AR any more
    out = tmp_path / "probe.json"
    holder = {}
    with pytest.raises(probe.ProbeError) as e:
        run_driver(monkeypatch, gdn, fp16, out, corrupt={(P - 1 + 18, 0)}, holder=holder)
    target = holder.get("target")
    assert target is not None and target.corruptions, "the replay corruption never fired"
    assert all(count > 1 for _, _, count in target.corruptions), \
        f"corruption hit a recorded call, not the replay: {target.corruptions}"
    assert "self-check" in str(e.value)
    failed = str(out) + ".FAILED.json"
    assert os.path.exists(failed), "a failed self-check must preserve its report"
    data = json.load(open(failed))
    assert data["replay_selfcheck"]["logits_exact"] is False
    assert "invalid" in data


def test_kv_gather_ignores_unused_storage_and_honours_block_order(fp16):
    a = real_kv_layer(seed=1.0)
    b = real_kv_layer(seed=2.0)                      # every page differs before the valid prefix is written
    consumed = PAGE + 5
    for layer in (a, b):
        for pos in range(consumed):
            page, off = pos // PAGE, pos % PAGE
            layer.k[page, off] = float(pos % 97)
            layer.v[page, off] = float(pos % 89)
    fa = probe.kv_fingerprint(_cache_stub({"L0": a}), consumed, [0, 1])
    fb = probe.kv_fingerprint(_cache_stub({"L0": b}), consumed, [0, 1])
    assert fa["layers"]["L0"]["k_sha256"] == fb["layers"]["L0"]["k_sha256"]

    # non-identity physical mapping with the same logical content must fingerprint identically
    c = real_kv_layer(pages=8, seed=9.0)
    for pos in range(consumed):
        page, off = [3, 1][pos // PAGE], pos % PAGE
        c.k[page, off] = float(pos % 97)
        c.v[page, off] = float(pos % 89)
    fc = probe.kv_fingerprint(_cache_stub({"L0": c}), consumed, [3, 1])
    assert fc["layers"]["L0"]["k_sha256"] == fa["layers"]["L0"]["k_sha256"]

    # a change inside the valid prefix is detected
    c.k[1, 3] = -5.0
    fc2 = probe.kv_fingerprint(_cache_stub({"L0": c}), consumed, [3, 1])
    assert fc2["layers"]["L0"]["k_sha256"] != fa["layers"]["L0"]["k_sha256"]

    # rows past `consumed` in the last page are ignored
    d = real_kv_layer(seed=1.0)
    for pos in range(consumed):
        page, off = pos // PAGE, pos % PAGE
        d.k[page, off] = float(pos % 97)
        d.v[page, off] = float(pos % 89)
    d.k[1, consumed % PAGE:] = 123.0
    fd = probe.kv_fingerprint(_cache_stub({"L0": d}), consumed, [0, 1])
    assert fd["layers"]["L0"]["k_sha256"] == fa["layers"]["L0"]["k_sha256"]


def test_kv_fingerprint_requires_every_expected_layer(fp16):
    with pytest.raises(probe.SnapshotError):
        probe.kv_fingerprint(_cache_stub({}), 8, [0])
    broken = real_kv_layer()
    broken.k = None
    with pytest.raises(probe.SnapshotError):
        probe.kv_fingerprint(_cache_stub({"L0": broken}), 8, [0])
    with pytest.raises(probe.SnapshotError):
        probe.kv_fingerprint(_cache_stub({"L0": real_kv_layer()}), 3, [])       # block table too short


def test_conv_canonical_ignores_speculative_tail_and_detects_committed_change(gdn):
    class C:
        def __init__(self, ls):
            self.recurrent_layers = {("0", 0): ls}

        def get_all_recurrent_layers(self):
            return self.recurrent_layers

    def snap(ls, position=P + 18):
        return probe.snapshot_state(real_state(gdn["GDNState"], C(ls), position=position))

    ls1 = real_layer_state(gdn["GDNLayerState"])
    ls2 = real_layer_state(gdn["GDNLayerState"])
    ls1.conv_state[0, :, 4:] = 7.0            # speculative tail only
    ls2.conv_state[0, :, 4:] = -7.0
    a, b = snap(ls1), snap(ls2)
    assert probe.compare_snapshots(a, b, "tail")["layers"]["('0', 0)"]["conv"]["max"] == 0.0

    ls2.conv_state[0, 0, 2] = 3.0             # inside the committed window
    c = snap(ls2)
    assert probe.compare_snapshots(a, c, "committed")["layers"]["('0', 0)"]["conv"]["max"] > 0


def test_full_driver_prefill_entrypoint_matches_the_source_dispatch(monkeypatch, gdn, fp16, tmp_path):
    """AR prefill must be recorded through Model.prefill, the MTP prefill through forward(last_tokens_only=1)."""
    rec, caches, target = run_driver(monkeypatch, gdn, fp16, tmp_path / "probe.json")
    ar_first = rec["gen_ar"]["first_callback"]
    assert ar_first["method"] == "prefill", ar_first["method"]
    ar_prefills = [c for c in rec["gen_ar"]["calls"] if c["method"] == "prefill"]
    assert len(ar_prefills) == 1, f"gen_ar: expected exactly one prefill, got {len(ar_prefills)}"
    assert ar_prefills[0]["rows"] == P - 1 and ar_prefills[0]["cache_seqlens"] == 0

    mtp_first = rec["gen_mtp2"]["first_callback"]
    assert mtp_first["method"] == "forward", mtp_first["method"]
    mtp_prefills = [c for c in rec["gen_mtp2"]["calls"] if c["method"] == "prefill"]
    assert mtp_prefills == [], "the MTP prefill must go through forward(last_tokens_only=1), not prefill"
    mtp_prompt = [c for c in rec["gen_mtp2"]["calls"]
                  if c["method"] == "forward" and c["extra_params"].get("last_tokens_only") == 1]
    assert len(mtp_prompt) == 1, mtp_prompt
    assert mtp_prompt[0]["rows"] == P - 1 and mtp_prompt[0]["cache_seqlens"] == 0
    # the 1-D per-sequence offset is preserved end to end (rank 1, one entry per sequence)
    for name in ("gen_ar", "gen_mtp2"):
        positions = [c["positions"] for c in rec[name]["calls"] if c["positions"]]
        assert positions, "no positions recorded"
        assert all(len(p) == 1 for p in positions), positions[:3]


def test_control_lifecycle_runs_under_inference_mode_with_real_inference_tensors(monkeypatch, gdn, fp16,
                                                                                tmp_path):
    """Regression for window 1792766228047680624-502983.

    The control cache holds real inference tensors, and `get_new_state()` is the real `GDNState`
    constructor, whose `GDNLayerState.clear()` writes `conv_state[idx].zero_()` in place. Outside
    inference mode that is exactly the RuntimeError the window raised *before* `replay_one` ran, so this
    test fails unless the whole reset/allocate/replay/control lifecycle is inside `torch.inference_mode()`.
    """
    # the fixture itself must reproduce the failing operation: outside inference mode the real
    # GDNState constructor's clear() writes conv_state[idx].zero_() on an inference tensor and raises
    probe_cache = FakeCache(gdn, fp16, inference_tensors=True)
    probe_cache.reset_states()
    with pytest.raises(RuntimeError) as direct:
        probe_cache.get_new_state()
    assert "inference tensor" in str(direct.value)

    rec, caches, target = run_driver(monkeypatch, gdn, fp16, tmp_path / "probe.json",
                                     inference_tensors=True)
    # every control state allocation (and therefore the real clear()) ran under inference mode
    flags = [f for cache in caches.values()
             for f in getattr(cache, "state_alloc_in_inference", [])]
    assert len(flags) >= 4, f"expected the generator and control state allocations, got {flags}"
    assert all(flags), f"a state was allocated outside inference mode: {flags}"
    assert rec["replay_selfcheck"]["logits_exact"] is True
    assert rec["kv_ar_vs_mtp2"]["equal"] is True
    # the control really was a 3-row pass with a 1-entry per-sequence offset
    control = [p for p in target.seen_params if p[1] == (1, 3)]
    assert control, f"no 3-row control pass was issued: {target.seen_params[-4:]}"
    assert control[-1][2] == (1,), f"positions must stay 1-D per sequence, got {control[-1][2]}"


def test_mode_artifacts_are_flushed_before_controls(monkeypatch, gdn, fp16, tmp_path):
    """Completed inference must survive a later control failure."""
    out = tmp_path / "probe.json"
    rec, caches, target = run_driver(monkeypatch, gdn, fp16, out)
    for name in ("gen_ar", "gen_mtp2"):
        art = f"{out}.{name}.json"
        assert os.path.exists(art), f"{name} was not flushed"
        data = json.load(open(art))
        assert data["generated_ids"][:5] == rec[name]["generated_ids"][:5]
        assert data["selected"]["row"] == rec["selection"][
            "ar" if name == "gen_ar" else "mtp2"]["row"]
        assert data["logits"]["sha256"] and data["logits"]["numel"] > 0
        assert data["entry_state"]["position"] == data["selected"]["cache_seqlens"]

    # a later control failure keeps both flushed captures on disk
    out2 = tmp_path / "probe2.json"
    with pytest.raises(probe.ProbeError):
        run_driver(monkeypatch, gdn, fp16, out2, corrupt_conv=True)
    assert os.path.exists(f"{out2}.gen_ar.json") and os.path.exists(f"{out2}.gen_mtp2.json")


def test_replay_runs_under_inference_mode(gdn, fp16):
    """The loaded model's tensors are inference tensors; replay outside inference mode is a real crash
    (window 1792766228047680624-502983). Both entrypoints must be replayed inside the context."""
    model = FakeModel()
    model.require_inference = True
    calls = [probe.CallRecord(seq=0, batch=1, cols=P - 1, cache_seqlens=0, block_row=[0, 1],
                              positions=[], recurrent_history=False,
                              input_ids=[PROMPT[:P - 1]], method="prefill")]
    for k in range(3):
        tok = PROMPT[P - 1] if k == 0 else GEN[k - 1]
        calls.append(probe.CallRecord(seq=k + 1, batch=1, cols=1, cache_seqlens=P - 1 + k,
                                      block_row=[0, 1], positions=[P - 1 + k],
                                      recurrent_history=False, input_ids=[[tok]], method="forward"))
    cache = FakeCache(gdn, fp16)
    state = real_state(gdn["GDNState"], cache)
    for call in calls:
        probe.replay_one(call, cache, state, model)          # would raise if not inference-mode
    out = probe.issue_control_rows(calls[-1], [GEN[2], 1, 2], cache, state, model)
    assert out.shape == (3, VOCAB)


def test_positions_adapter_rejects_a_token_grid():
    """generator.py:1465 supplies a 1-D base offset; a 2-D grid must be a self-describing ParamError."""
    rec = probe.ProbeRecorder(FakeModel(), P)
    with pytest.raises(probe.ParamError) as e:
        rec.capture("forward", rec.target_model, torch.tensor([[5]]),
                    {"cache_seqlens": torch.tensor([P - 1], dtype=torch.int32),
                     "block_table": torch.zeros((1, 2), dtype=torch.int32),
                     "positions": torch.zeros((1, 3), dtype=torch.int32)},
                    lambda: torch.zeros((1, 1, 8)))
    assert "rank=2" in str(e.value) and "1-D" in str(e.value)


def test_unexpected_exception_leaves_an_error_artifact(monkeypatch, gdn, fp16, tmp_path):
    """main() must preserve a partial capture when run() raises, and still propagate the failure."""
    out = tmp_path / "probe.json"
    holder = {}
    argv = ["probe_mtp_causal.py", "--model", "/fake/model", "--index", "18", "--out", str(out),
            "--ctx", "8192", "--tensor-dir", str(out) + "-tensors", "--filler", str(FILLER)]
    install_fake_api(monkeypatch, None, None)          # replaced below by the corrupting driver
    with pytest.raises(Exception):
        # build the models through run_driver's own path, but go through main() so the handler runs
        import types as _t
        target = FakeModel(target=True, corrupt_conv=True)
        holder["target"] = target
        draft = FakeModel(target=False)
        pkg = install_fake_api(monkeypatch, target, draft)

        class DriverCache(FakeCache):
            def __init__(self, *a, **kw):
                super().__init__(gdn, fp16, inference_tensors=False)

        pkg.Cache = DriverCache
        monkeypatch.setattr(sys, "argv", argv)
        probe.main(argv[1:])
    art = str(out) + ".ERROR.json"
    assert os.path.exists(art), "no error artifact was preserved"
    data = json.load(open(art))
    assert data["error"]["type"] in ("ProbeError", "SnapshotError", "SelectionError", "ContextError")
    assert data["provenance"]["script"]["path"].endswith("probe_mtp_causal.py")
    assert holder["target"].conv_corruptions, "the injected failure never fired"


def test_source_conformance_prefill_dispatch_and_positions_rank():
    job = os.path.join(SRC, "generator/job.py")
    gen = os.path.join(SRC, "generator/generator.py")
    if not (os.path.exists(job) and os.path.exists(gen)):
        pytest.skip("upstream source not present")
    jt, gt = open(job).read(), open(gen).read()
    assert "if self.generator.mtp_draft:" in jt and "params[\"last_tokens_only\"] = 1" in jt
    assert "self.generator.model.forward(input_ids = prefill_ids, params = params)" in jt
    assert "self.generator.model.prefill(input_ids = prefill_ids, params = params)" in jt
    assert 'positions = self._staging("positions", batch_size) if use_offsets else None' in gt
    assert "positions[batch] = seq.kv_position + job.alt_rope_offset" in gt
    model_src = open(os.path.join(SRC, "model/model.py")).read()
    body = model_src[model_src.index("def prefill(self, input_ids"):]
    assert "self.forward(" not in body[:body.index("def lm_head_argmax")] , \
        "Model.prefill must not route through Model.forward"


def test_control_gate_rejects_conv_only_mismatch(gdn):
    class C:
        def __init__(self, ls):
            self.recurrent_layers = {("0", 0): ls}

        def get_all_recurrent_layers(self):
            return self.recurrent_layers

    def snap(ls):
        return probe.snapshot_state(real_state(gdn["GDNState"], C(ls), position=P + 18))

    a, b = real_layer_state(gdn["GDNLayerState"]), real_layer_state(gdn["GDNLayerState"])
    b.conv_state[0, 0, 0] = 5.0                      # committed window only
    kv = {"n_layers": 1, "expected_layers": 1, "layers": {"L0": {"k_sha256": "x", "v_sha256": "y"}}}
    detail = {}
    with pytest.raises(probe.ProbeError) as e:
        probe.control_prerequisites("conv-only", snap(a), snap(b), kv, kv, detail)
    assert "committed conv" in str(e.value)
    assert detail["conv-only"]["ok"] is False and detail["conv-only"]["kv_equal"] is True


def test_control_gate_rejects_kv_only_mismatch(gdn, fp16):
    class C:
        def __init__(self, ls):
            self.recurrent_layers = {("0", 0): ls}

        def get_all_recurrent_layers(self):
            return self.recurrent_layers

    ls = real_layer_state(gdn["GDNLayerState"])
    st = probe.snapshot_state(real_state(gdn["GDNState"], C(ls), position=P + 18))
    kv_a = probe.kv_fingerprint(_cache_stub({"L0": real_kv_layer(seed=1.0)}), 12, [0])
    kv_b = probe.kv_fingerprint(_cache_stub({"L0": real_kv_layer(seed=1.0)}), 12, [0])
    kv_b["layers"]["L0"]["k_sha256"] = "different"
    detail = {}
    with pytest.raises(probe.ProbeError) as e:
        probe.control_prerequisites("kv-only", st, st, kv_a, kv_b, detail)
    assert "KV prefix" in str(e.value)
    assert detail["kv-only"]["state"]["n_base_differing"] == 0


def test_driver_aborts_when_replay_conv_state_differs(monkeypatch, gdn, fp16, tmp_path):
    out = tmp_path / "probe.json"
    holder = {}
    with pytest.raises(probe.ProbeError) as e:
        run_driver(monkeypatch, gdn, fp16, out, corrupt_conv=True, holder=holder)
    assert "control prerequisite" in str(e.value) or "self-check" in str(e.value)
    assert os.path.exists(str(out) + ".FAILED.json")
    target = holder["target"]
    assert target.conv_corruptions, "the conv corruption never fired"


def test_source_conformance_state_ownership_and_get_kv_passthrough():
    gdn_path = os.path.join(SRC, "modules/gated_delta_net.py")
    fp16_path = os.path.join(SRC, "cache/fp16.py")
    if not (os.path.exists(gdn_path) and os.path.exists(fp16_path)):
        pytest.skip("upstream source not present")
    gdn_src = open(gdn_path).read()
    state_block = gdn_src[gdn_src.index("class GDNState"):gdn_src.index("class GDNLayerState")]
    layer_block = gdn_src[gdn_src.index("class GDNLayerState"):gdn_src.index("class GatedDeltaNet(")]
    for name in ("def base_row", "def scratch_row", "self.parity", "self.slot"):
        assert name in state_block, f"{name} must live on GDNState"
        assert name not in layer_block, f"{name} must NOT live on GDNLayerState"
    assert layer_block.count("def get_state_tensors") == 1
    fp16_src = open(fp16_path).read()
    body = fp16_src[fp16_src.index("def get_kv"):]
    body = body[:body.index("def update_kv")]
    assert "return self.k, self.v" in body, "get_kv must remain a passthrough: the probe gathers pages"
    conv_src = open(os.path.join(SRC, "modules/gated_delta_net_fn/conv1d.py")).read()
    assert "conv_state[:, :, :conv_kernel_size].copy_" in conv_src, \
        "committed conv history must be columns [0, conv_kernel_size)"
