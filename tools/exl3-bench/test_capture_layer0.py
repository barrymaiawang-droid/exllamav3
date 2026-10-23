#!/usr/bin/env python3
"""Focused CPU tests for the layer-0 capture/replay seam. No model, no accelerator, no build.

The harness imports the **real** `modules/gated_delta_net.py`, `util/tensor.py` and
`cache/recurrent_util.py` from the pinned v1.6.4 tree (`EXL3_SRC`, default
`~/exl3-v164-rocm10-port/exllamav3`), with only the heavy sibling packages replaced by stubs. Every
assertion therefore runs the real `GatedDeltaNet.forward` body, the real `_bc_configure_slot`, the real
`staged_views`, `GDNState.rewind`, `_rewind_layers`, `GDNLayerState.replay_job`, `GTensorCache` and
`prepare_for_recurrence`/`advance_recurrent_states`. The only fabricated pieces are the compiled
extension (`SpyExt`) and the C++ BC slot (`SpyBC`), which is exactly the boundary this diagnostic sits
on: `SpyBC.run_bszN` writes the operand buffers the way the kernel would, and `SpyExt` records the job
descriptors.

A spy backend is a CPU wiring check. It is never native numeric evidence; the replay report says so in
`native_evidence` and the tests assert that label rather than treating a spy pass as a result.
"""
import copy
import ctypes
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser(os.environ.get("EXL3_SRC", "~/exl3-v164-rocm10-port/exllamav3"))
MAX_HISTORY = 3


def _load(name, fname):
    path = os.path.join(HERE, fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cap = _load("capture_layer0", "capture_layer0.py")
fx = _load("replay_fixture", "replay_fixture.py")


# ----------------------------------------------------------------------------- real source harness

class _StubModule:
    """Stand-in for exllamav3.modules.Module: nothing on the fused path calls into it."""

    def __init__(self, *a, **k):
        pass

    def tp_collect(self, *a, **k):
        raise AssertionError("tp_collect must not run in these tests")


def _stub(name):
    return type(name, (), {})


class SpyExt:
    """Stand-in for the compiled extension: records descriptors, never computes numerics."""

    def __init__(self):
        self.conv_jobs = []
        self.conv_calls = []
        self.scan_jobs = []
        self.scan_calls = []
        self.on_scan_replay = None
        self.on_conv_rewind = None

    def ConvRewindJob(self, *args):
        job = types.SimpleNamespace(args=args)
        self.conv_jobs.append(job)
        return job

    def ScanReplayJob(self, conv_ptr, g_ptr, beta_ptr, base_ptr, scratch_ptr, d_ptr,
                      prefix, tstride):
        job = {"conv_ptr": conv_ptr, "g_ptr": g_ptr, "beta_ptr": beta_ptr, "base_ptr": base_ptr,
               "scratch_ptr": scratch_ptr, "d_ptr": d_ptr, "prefix": prefix, "tstride": tstride}
        self.scan_jobs.append(job)
        return job

    def batched_conv_rewind(self, jobs, device_index):
        self.conv_calls.append({"jobs": list(jobs), "device_index": device_index})
        if self.on_conv_rewind is not None:
            self.on_conv_rewind()

    def batched_scan_replay(self, jobs, device_index, kind, nk, nv, dk, dv, state_half):
        self.scan_calls.append({"jobs": list(jobs), "device_index": device_index, "kind": kind,
                                "nk": nk, "nv": nv, "dk": dk, "dv": dv,
                                "state_half": state_half})
        if self.on_scan_replay is not None:
            self.on_scan_replay(jobs, nk, nv, dk, dv, state_half)


class SpyBC:
    """Stand-in for the compiled BC_GatedDeltaNetSplit slot.

    `configure_slot` keeps the exact tensor objects C++ is handed and `run_bszN` writes into them, the
    way the real kernel does. `writer(configured, recurrent_state, slots_scan)` lets a test choose the
    values, so a capture can be checked against what the kernel "stored".
    """

    def __init__(self):
        self.configured = {}
        self.runs = []
        self.writer = None

    def needs_configure(self, bsz, seqlen, history, stage_key):
        return True

    def configure_slot(self, bsz, seqlen, history, qkv, z, ba, beta, g, mixed_qkv, conv_out,
                       core_attn_out, core_attn_out_f, qkv_xh, z_xh, o_xh):
        self.configured[(bsz, seqlen, bool(history))] = {
            "bsz": bsz, "seqlen": seqlen, "history": bool(history), "qkv": qkv, "z": z, "ba": ba,
            "beta": beta, "g": g, "mixed_qkv": mixed_qkv, "conv_out": conv_out}

    def set_stage_key(self, bsz, seqlen, history, key):
        pass

    def run_bszN(self, x, y, conv_state, recurrent_state, recurrent_slots, slots_scan,
                 save_history, slots_scan_in):
        configured = self.configured[(int(x.shape[0]), int(x.shape[1]), bool(save_history))]
        self.runs.append({"slots_scan": slots_scan, "slots_scan_in": slots_scan_in,
                          "save_history": bool(save_history)})
        if self.writer is not None:
            self.writer(configured, recurrent_state, slots_scan)


class FakeModel:
    loaded_tp = False


class FakeCache:
    """Only the accessors the real state/rewind code uses."""

    def __init__(self, layer_state):
        self.recurrent_layers = {(0, 0): layer_state}
        self.model = FakeModel()

    def get_all_recurrent_layers(self):
        return self.recurrent_layers

    def get_recurrent_layer(self, key):
        return self.recurrent_layers[key]

    def release_state(self, state):
        pass


_SOURCE = {}


def _pkg(name, paths, **attrs):
    mod = types.ModuleType(name)
    mod.__path__ = list(paths)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def real_source():
    """Import the real GDN module from EXL3_SRC; skip when the pinned tree is absent."""
    if "gdn" in _SOURCE:
        return _SOURCE
    src = os.path.abspath(SRC)
    need = ("modules/gated_delta_net.py", "util/tensor.py", "cache/recurrent_util.py")
    missing = [p for p in need if not os.path.isfile(os.path.join(src, p))]
    if missing:
        pytest.skip(f"EXL3_SRC {src} is missing {missing}; point EXL3_SRC at the v1.6.4 checkout")
    if getattr(sys.modules.get("exllamav3"), "_layer0_test_stub", False):
        return _SOURCE
    spy_ext = SpyExt()
    stubs = {
        "exllamav3": _pkg("exllamav3", [src]),
        "exllamav3.modules": _pkg("exllamav3.modules", [os.path.join(src, "modules")],
                                  Module=_StubModule, Linear=_stub("Linear")),
        "exllamav3.util": _pkg("exllamav3.util", [os.path.join(src, "util")],
                              profile_opt=types.SimpleNamespace()),
        "exllamav3.cache": _pkg("exllamav3.cache", [os.path.join(src, "cache")],
                                Cache=_stub("Cache")),
        "exllamav3.model": _pkg("exllamav3.model", [os.path.join(src, "model")]),
        "exllamav3.model.config": _pkg("exllamav3.model.config", [], Config=_stub("Config")),
        "exllamav3.model.model_tp_alloc": _pkg("exllamav3.model.model_tp_alloc", [],
                                               TPAllocation=_stub("TPAllocation")),
        "exllamav3.model.model_tp_shared": _pkg("exllamav3.model.model_tp_shared", [],
                                                TPTensorWrapper=_stub("TPTensorWrapper")),
        "exllamav3.ext": _pkg("exllamav3.ext", [], exllamav3_ext=spy_ext),
        "exllamav3.util.backend": _pkg("exllamav3.util.backend", [], QKV_SLICE=1),
        "exllamav3.modules.gated_rmsnorm": _pkg("exllamav3.modules.gated_rmsnorm", [],
                                                GatedRMSNorm=_stub("GatedRMSNorm")),
        "exllamav3.modules.multilinear": _pkg("exllamav3.modules.multilinear", [],
                                              SlicedMultiLinear=_stub("SlicedMultiLinear")),
        "exllamav3.modules.gated_delta_net_fn": _pkg(
            "exllamav3.modules.gated_delta_net_fn", [],
            causal_conv1d_update=lambda *a, **k: None, gated_delta_rule_fn=lambda *a, **k: None),
        "exllamav3.cache.recurrent": _pkg(
            "exllamav3.cache.recurrent", [], mp_cache_recurrent_stash=None,
            mp_cache_recurrent_unstash=None, mp_cache_recurrent_clear=None,
            new_checkpoint_handle=lambda: 0, host_copy=lambda t: t.detach().cpu().clone()),
        "exllamav3.modules.attention_fn": _pkg("exllamav3.modules.attention_fn", []),
        "exllamav3.modules.attention_fn.bc_attn": _pkg("exllamav3.modules.attention_fn.bc_attn",
                                                       [], MAX_BSZ=8, MAX_QLEN=16),
    }
    stubs["exllamav3"]._layer0_test_stub = True
    sys.modules.update(stubs)
    gdn = importlib.import_module("exllamav3.modules.gated_delta_net")
    rt = importlib.import_module("exllamav3.cache.recurrent_util")
    tensor = importlib.import_module("exllamav3.util.tensor")
    gdn.ext = spy_ext
    _SOURCE.update({"gdn": gdn, "rt": rt, "tensor": tensor, "ext": spy_ext})
    return _SOURCE


def make_layer(src, hidden=8, nk=2, nv=4, dk=2, dv=2, K=4):
    """A GatedDeltaNet instance without __init__ (no weights to load), attributes as the fused path
    reads them. `object.__new__` keeps the real class, so the real `forward` body runs."""
    layer = object.__new__(src["gdn"].GatedDeltaNet)
    layer.layer_idx = 0
    layer.hidden_size = hidden
    layer.k_head_dim, layer.v_head_dim = dk, dv
    layer.num_k_heads, layer.num_v_heads = nk, nv
    layer.num_v_groups = nv // nk
    layer.k_dim, layer.v_dim = nk * dk, nv * dv
    layer.fdim_qkv = 2 * nk * dk + nv * dv
    layer.fdim_qkvz = 2 * nk * dk + 2 * nv * dv
    layer.fdim_ba = 2 * nv
    layer.conv_kernel_size = K
    layer.beta_scale = 1.0
    layer.kda = False
    layer.kda_direct = False
    layer.out_dtype = None
    layer.device = torch.device("cpu")
    layer.conv1d_weight = None
    layer.conv1d_weight_flat = None
    layer.conv1d_bias = None
    layer.conv1d_q_weight = torch.zeros(nk * dk, 1, K)
    layer.conv1d_k_weight = torch.zeros(nk * dk, 1, K)
    layer.conv1d_v_weight = torch.zeros(nv * dv, 1, K)
    layer.bc_split = True
    layer.ba_weight_filled = False
    layer.ba_weight_t = torch.zeros(2 * hidden, nv, dtype=torch.half)
    layer.ba_bias = None
    weight = torch.zeros(nv, hidden, dtype=torch.half)
    layer.b_proj = types.SimpleNamespace(inner=types.SimpleNamespace(
        get_weight_tensor=lambda: weight, get_bias_tensor=lambda: None))
    layer.a_proj = types.SimpleNamespace(inner=types.SimpleNamespace(
        get_weight_tensor=lambda: weight, get_bias_tensor=lambda: None))
    layer.bc = SpyBC()
    layer.tp_reduce = False
    return layer


def make_state(src, layer, seed=0.0, position=0):
    """Real GDNLayerState + real GDNState on CPU, with a known non-zero base row."""
    rsl = src["gdn"].GDNLayerState(layer, 1, MAX_HISTORY, cache_id=0, state_dtype=torch.float16)
    rsl.alloc(torch.device("cpu"))
    cache = FakeCache(rsl)
    rsl.cache_id = id(cache)                      # cache.py constructs it with id(self)
    state = src["gdn"].GDNState(cache, slot=0, position=position, test_state=True)
    if seed:
        g = torch.Generator().manual_seed(int(seed))
        rsl.recurrent_state[:, 0] = torch.rand(rsl.recurrent_state[:, 0].shape, generator=g,
                                               dtype=torch.float32).to(torch.float16)
    return rsl, cache, state


def run_pass(src, layer, state, seqlen, history, marker, state_marker=None):
    """One real prefill/forward-stage pass through the real `prepare_for_recurrence` and the real
    forward body, with the pass metadata advanced outside the layer."""
    layer.bc.writer = _operand_writer(marker, state_marker)
    ids = torch.full((1, seqlen), 7, dtype=torch.long)
    params = {"cache_seqlens": torch.tensor([int(state.position)], dtype=torch.int32),
              "recurrent_states": [state], "recurrent_history": bool(history)}
    src["rt"].prepare_for_recurrence(ids, params, None)
    x = torch.full((1, seqlen, layer.hidden_size), 0.5, dtype=torch.float32)
    out = layer.forward(x, params)
    src["rt"].advance_recurrent_states(ids, params, None)
    return out, params


def _operand_writer(marker, state_marker=None):
    """Write the scan operands C++ was handed and the state row the pass stores.

    The two are independent here so a test can make two paths' *operands* identical while their
    stored states remain distinguishable (the real model computes both from the same tokens).
    """
    stored = marker if state_marker is None else state_marker

    def writer(configured, recurrent_state, slots_scan):
        configured["conv_out"].fill_(marker * 0.5)
        configured["beta"].fill_(marker * 0.25)
        configured["g"].fill_(-marker * 0.125)
        recurrent_state[int(slots_scan[0]), 0].fill_(stored)
    return writer


class capture_env:
    """Install one capture, run a body, write the artifact, restore the seams."""

    def __init__(self, src, tmp_path, mode, cache, positions=(7, 8, 9), **kw):
        self.src = src
        self.mode = mode
        self.positions = positions
        self.capture = cap.Layer0Capture(str(tmp_path / mode), mode=mode, cache_id=id(cache),
                                         layer=0, positions=positions, **kw)

    def __enter__(self):
        g = self.src["gdn"]
        self.capture.install(g.GatedDeltaNet, g.GDNState, g.GDNLayerState)
        return self.capture

    def __exit__(self, *exc):
        g = self.src["gdn"]
        self.capture.uninstall(g.GatedDeltaNet, g.GDNState, g.GDNLayerState)
        return False


def build_captures(tmp_path, seed=99.0, rewind_tokens=2):
    """Produce one AR and one MTP capture through the real bodies, exactly as the probe will.

    `rewind_tokens=2` settles a 3-token pass on its first token (accepted_prefix 1, post-position 8);
    `rewind_tokens=1` is the reproduced ledger (accepted_prefix 2, post-position 9).
    """
    src = real_source()
    src["gdn"].g_tensor_cache.drop_all()
    paths = {}
    target = 10 - int(rewind_tokens)
    operand_marker = 7.0        # both paths consume the same tokens, so the same scan operands
    frontier_marker = {8: 1.0, 9: 3.0}[target]      # state AR stores after `target` tokens
    # ---- AR: three single-token passes at 7, 8, 9
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, seed=seed, position=7)
    with capture_env(src, tmp_path, "ar", cache) as capture:
        for position in (7, 8, 9):
            assert state.position == position
            run_pass(src, layer, state, 1, False, operand_marker, state_marker=frontier_marker)
        paths["ar"] = capture.write("ar-layer0")["json"]
    # ---- MTP: one three-token verifier pass at 7, then the real partial-accept rewind
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, seed=seed, position=7)
    rewind_marker = frontier_marker
    src["ext"].on_scan_replay = None
    with capture_env(src, tmp_path, "mtp", cache) as capture:
        run_pass(src, layer, state, MAX_HISTORY, True, operand_marker, state_marker=9.0)
        assert (state.position, state.last_history, state.spec_row) == (10, 2, 0)
        # the library's own rewind calls batched_scan_replay; the spy stands in for the kernel write
        src["ext"].on_scan_replay = (lambda jobs, nk, nv, dk, dv, half:
                                     rsl.recurrent_state[state.scratch_row(), 0].fill_(rewind_marker))
        state.rewind(rewind_tokens)
        src["ext"].on_scan_replay = None
        paths["mtp"] = capture.write("mtp-layer0")["json"]
        paths["rewind_marker"] = rewind_marker
        paths["position_after"] = state.position
    return paths


def tensor_at(ptr, numel, dtype):
    """A view over raw CPU memory, so a spy can emulate an in-place kernel write."""
    bytes_ = numel * torch.empty((), dtype=dtype).element_size()
    return torch.frombuffer((ctypes.c_char * bytes_).from_address(ptr), dtype=dtype)


class MarkerExt:
    """Fixture-side spy: writes a fixed value into the job's output row (no arithmetic)."""

    def __init__(self, value):
        self.value = value
        self.jobs = []
        self.calls = []
        self.out_ptr = None

    def ScanReplayJob(self, conv_ptr, g_ptr, beta_ptr, base_ptr, scratch_ptr, d_ptr, prefix, tstride):
        job = {"conv_ptr": conv_ptr, "g_ptr": g_ptr, "beta_ptr": beta_ptr, "base_ptr": base_ptr,
               "scratch_ptr": scratch_ptr, "d_ptr": d_ptr, "prefix": prefix, "tstride": tstride}
        self.jobs.append(job)
        self.out_ptr = scratch_ptr
        return job

    def batched_scan_replay(self, jobs, device_index, kind, nk, nv, dk, dv, state_half):
        self.calls.append({"device_index": device_index, "kind": kind, "nk": nk, "nv": nv,
                           "dk": dk, "dv": dv, "state_half": state_half,
                           "prefix": jobs[0]["prefix"], "jobs": list(jobs)})
        dtype = torch.float16 if state_half else torch.float32
        tensor_at(jobs[0]["scratch_ptr"], nv * dk * dv, dtype).fill_(self.value)


# ----------------------------------------------------------------------------- 1. real seams

def test_real_forward_contract_and_captured_seams():
    src = real_source()
    g = src["gdn"]
    import inspect
    sig = inspect.signature(g.GatedDeltaNet.forward)
    assert list(sig.parameters) == ["self", "x", "params", "out_dtype"], (
        "the capture must wrap the real 3-argument forward")
    assert sig.parameters["out_dtype"].default is None
    assert [p for p in inspect.signature(g.GDNState.rewind).parameters] == ["self", "num_tokens"]
    assert [p for p in inspect.signature(g.GDNLayerState.replay_job).parameters] == [
        "self", "spec_row", "prefix", "spec_shape", "base_row", "scratch_row"]
    # the replay geometry the fixture reproduces
    assert "r0 = spec_row * seqlen if tstride == 1 else spec_row" in inspect.getsource(
        g.GDNLayerState.replay_job)
    # `_bc_configure_slot` hands the staged views / shared statics to C++ (source of the operands)
    body = inspect.getsource(g.GatedDeltaNet._bc_configure_slot)
    assert "conv_out, beta, g = rsl.staged_views(bsz, seqlen)" in body
    for tag in ('"s_conv"', '"s_beta"', '"s_g"', '"s_qkv"', '"s_ba"'):
        assert tag in body, tag


def test_ar_operands_are_the_exact_shared_entries_handed_to_cpp(tmp_path):
    src = real_source()
    src["gdn"].g_tensor_cache.drop_all()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "ar", cache) as capture:
        run_pass(src, layer, state, 1, False, 2.5)
    rec = capture.forwards("target_decode")[0]
    configured = layer.bc.configured[(1, 1, False)]
    env = src["gdn"].g_tensor_cache
    for key, tag, tensor, dtype in (
            ("conv", "s_conv", configured["conv_out"], torch.bfloat16),
            ("beta", "s_beta", configured["beta"], torch.bfloat16),
            ("g", "s_g", configured["g"], torch.float)):
        entry = rec["operands"][key]
        expected_key = env.make_key(layer.device, tuple(tensor.shape), dtype, tag)
        assert entry["key"] == expected_key, (key, entry["key"], expected_key)
        assert env.cache[expected_key][1] is tensor, \
            f"{key}: the capture did not peek the tensor C++ was handed"
        assert torch.equal(entry["value"]["tensor"], tensor.detach().cpu())
        assert entry["value"]["layout"]["shape"] == list(tensor.shape)
        assert entry["value"]["layout"]["stride"] == list(tensor.stride())
    # the qkv/ba projections are captured as auxiliary evidence, not as scan operands
    assert rec["operands"]["aux"]["qkv"]["key"].endswith("/s_qkv")
    assert rec["operands"]["aux"]["ba"]["key"].endswith("/s_ba")


def test_peek_does_not_allocate_or_increment_refcounts(tmp_path):
    """The capture must leave `g_tensor_cache` exactly as `_bc_configure_slot` left it."""
    src = real_source()
    env = src["gdn"].g_tensor_cache

    def one_pass(install_capture):
        env.drop_all()
        layer = make_layer(src)
        rsl, cache, state = make_state(src, layer, position=7)
        if install_capture:
            with capture_env(src, tmp_path, "ar", cache) as capture:
                run_pass(src, layer, state, 1, False, 1.5)
            assert capture.forwards("target_decode")
        else:
            run_pass(src, layer, state, 1, False, 1.5)
        return {k: v[0] for k, v in env.cache.items()}

    with_capture = one_pass(True)
    without_capture = one_pass(False)
    assert set(with_capture) == set(without_capture), "the capture created a cache entry"
    assert with_capture == without_capture, "the capture changed a refcount"
    # `_bc_configure_slot` requests each static once per (bsz, seqlen, history) combination
    assert set(with_capture.values()) == {1}, sorted(with_capture.items())

    # a second identical pass reuses the same keys and only bumps the counts
    def two_passes():
        env.drop_all()
        layer = make_layer(src)
        rsl, cache, state = make_state(src, layer, position=7)
        with capture_env(src, tmp_path, "ar", cache) as capture:
            run_pass(src, layer, state, 1, False, 1.5)
            run_pass(src, layer, state, 1, False, 2.5)
        return {k: v[0] for k, v in env.cache.items()}, capture

    counts, capture = two_passes()
    assert set(counts) == set(without_capture)
    assert set(counts.values()) == {2}, "one .get() per pass per tag, and nothing from the capture"
    assert len(capture.forwards("target_decode")) == 2

    # the contrast: GTensorCache.get() on an unknown tag allocates; the capture's peek does not
    key = env.make_key(torch.device("cpu"), (1, 1, 1), torch.float, "not_a_seam")
    peek = cap.Layer0Capture(str(tmp_path / "peek"), mode="ar")
    assert peek._shared(env, torch.device("cpu"), (1, 1, 1), torch.float, "not_a_seam", False) is None
    assert key not in env.cache, "the capture allocated a cache entry it should only peek"
    env.get(torch.device("cpu"), (1, 1, 1), torch.float, "not_a_seam")
    assert key in env.cache, "GTensorCache.get() allocates - that is exactly why it is not used"


def test_ar_operands_are_cloned_before_the_next_layer_overwrites(tmp_path):
    src = real_source()
    src["gdn"].g_tensor_cache.drop_all()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "ar", cache) as capture:
        for position, marker in ((7, 1.0), (8, 3.0)):
            run_pass(src, layer, state, 1, False, marker)
    env = src["gdn"].g_tensor_cache
    first = capture.forwards("target_decode")[0]
    key = first["operands"]["conv"]["key"]
    # the live buffer now holds the second pass's values; the record must still hold the first's
    assert torch.equal(env.cache[key][1].float(), torch.full_like(env.cache[key][1].float(), 1.5))
    assert torch.equal(first["operands"]["conv"]["value"]["tensor"].float(),
                       torch.full_like(first["operands"]["conv"]["value"]["tensor"].float(), 0.5))
    assert first["operands"]["conv"]["refcount"] == 1


def test_mtp_operands_are_the_layer_state_staging_buffers(tmp_path):
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "mtp", cache) as capture:
        run_pass(src, layer, state, MAX_HISTORY, True, 4.0)
    rec = capture.forwards("target_verify")[0]
    assert rec["operands"]["source"].startswith("GDNLayerState.spec_")
    assert rec["operands"]["spec_shape"] == [1, MAX_HISTORY, 1]
    assert rec["operands"]["rows"] == MAX_HISTORY
    assert torch.equal(rec["operands"]["conv"]["value"]["tensor"],
                       rsl.spec_conv[:MAX_HISTORY].detach().cpu())
    assert torch.equal(rec["operands"]["g"]["value"]["tensor"],
                       rsl.spec_g[:MAX_HISTORY].detach().cpu())
    assert rec["operands"]["conv"]["value"]["layout"]["stride"] == list(rsl.spec_conv.stride())
    # the fused predicate that let this pass reach `run_bszN` is recorded, not assumed
    assert rec["fused_predicate"]["spec_step"] is None and rec["fused_predicate"]["bc_split"] is True


def test_capture_excludes_draft_other_layer_other_cache_and_outside_window(tmp_path):
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    cap_obj = cap.Layer0Capture(str(tmp_path), mode="mtp", cache_id=id(cache), positions=(7, 8, 9))
    cap_obj._limits = (8, 16)
    x = torch.zeros(1, MAX_HISTORY, layer.hidden_size)
    base = {"recurrent_states": [state], "recurrent_slots": torch.zeros(1, dtype=torch.int32),
            "recurrent_history": True}
    assert cap_obj.classify(layer, x, dict(base, recurrent_spec_step=1))[1] == "draft_step"
    other_cache_state = types.SimpleNamespace(cache=types.SimpleNamespace(), position=7,
                                              slot=0, parity=0, last_history=0, spec_row=0,
                                              spec_shape=(1, 3, 1))
    assert cap_obj.classify(layer, x, dict(base, recurrent_states=[other_cache_state]))[1] == \
        "other_cache"
    assert cap_obj.classify(layer, x, dict(base, recurrent_slots=None))[1] == "no_slots"
    state.position = 4
    assert cap_obj.classify(layer, x, base)[1] == "outside_window"
    state.position = 7
    layer.layer_idx = 3
    assert cap_obj.classify(layer, x, base)[1] == "other_layer"
    layer.layer_idx = 0
    layer.bc_split = False
    assert cap_obj.classify(layer, x, base)[1] == "not_bc_split"
    layer.bc_split = True
    info, reason = cap_obj.classify(layer, x, base)
    assert reason is None and info["mode"] == "target_verify"
    big = torch.zeros(1, 17, layer.hidden_size)
    assert cap_obj.classify(layer, big, base)[1] == "outside_bc_shape"
    # mode-specific pass kinds: AR runs must not record verifier passes, and only the first
    # verifier pass of an MTP run is the transition of interest
    ar_obj = cap.Layer0Capture(str(tmp_path / "ar"), mode="ar", cache_id=id(cache), positions=(7, 8, 9))
    ar_obj._limits = (8, 16)
    assert ar_obj.classify(layer, x, base)[1] == "verify_in_ar_mode"
    cap_obj.records.append({"kind": "layer0_forward", "mode": "target_verify", "position": 7,
                            "seqlen": MAX_HISTORY, "seq": 0, "operands": {}, "before": {},
                            "after": {}, "x": {}})
    assert cap_obj.classify(layer, x, base)[1] == "second_verify_pass"


def test_capture_forwards_the_full_forward_signature(tmp_path):
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "ar", cache) as capture:
        _, params = run_pass(src, layer, state, 1, False, 1.0)
        out_default = layer.forward(torch.zeros(1, 1, layer.hidden_size), params)
        out_fp32 = layer.forward(torch.zeros(1, 1, layer.hidden_size), params, torch.float32)
    assert out_default.dtype == torch.float16, out_default.dtype
    assert out_fp32.dtype == torch.float32, "out_dtype was not forwarded to the real forward"
    assert len(capture.forwards("target_decode")) == 3


def test_rewind_capture_clones_before_the_replay_writes_the_scratch_row(tmp_path):
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, seed=5.0, position=7)
    with capture_env(src, tmp_path, "mtp", cache) as capture:
        run_pass(src, layer, state, MAX_HISTORY, True, 6.0)
        pre = rsl.recurrent_state[state.scratch_row()].clone()   # the pass wrote the scratch row
        assert pre.max().item() == 6.0
        src["ext"].on_scan_replay = (
            lambda jobs, nk, nv, dk, dv, half:
            rsl.recurrent_state[state.scratch_row(), 0].fill_(42.0))
        state.rewind(2)
        src["ext"].on_scan_replay = None
    rec = capture.rewinds()[0]
    assert rec["pass_entry"] == 7 and rec["prefix"] == 1 and rec["num_tokens"] == 2
    assert rec["position_before"] == 10 and rec["position_after"] == 8
    captured_scratch = rec["before"]["canonical"]["rec_scratch"]["tensor"]
    assert torch.equal(captured_scratch, pre), "the pre-rewind scratch row was not cloned in time"
    assert captured_scratch.max().item() == 6.0, "the clone followed the replay instead of preceding it"
    after_base = rec["after"]["canonical"]["rec_base"]["tensor"]
    assert torch.equal(after_base, torch.full_like(after_base, 42.0))
    assert rec["after"]["canonical"]["base_row"] == rec["before"]["canonical"]["scratch_row"]
    assert rec["ledger"]["accepted_prefix"] == 1 and rec["ledger"]["staged_tokens"] == 3
    assert rec["ledger"]["parity_before"] != rec["ledger"]["parity_after"]


def test_replay_job_geometry_matches_the_library(tmp_path):
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "mtp", cache) as capture:
        run_pass(src, layer, state, MAX_HISTORY, True, 9.0)
        state.rewind(2)
    jobs = [r for r in capture.records if r["kind"] == "replay_job"]
    assert len(jobs) == 1
    job = jobs[0]
    assert job["spec_shape"] == [1, 3, 1] and job["tstride"] == 1 and job["r0"] == 0
    assert job["prefix"] == 1 and job["base_row"] == 0 and job["scratch_row"] == 1
    assert job["scan_geometry"] == [0, 2, 4, 2, 2, True]
    assert job["staging_strides"]["conv_row"] == layer.fdim_qkv
    assert job["staging_strides"]["g_row"] == layer.num_v_heads
    assert job["d_skip"] == "absent"
    assert job["pass_entry"] == 7, "the job was not attributed to the captured rewind"


def test_replay_job_only_records_inside_the_selected_rewind(tmp_path):
    """Later jobs on the same layer/cache are counted, never recorded, and cannot exhaust the cap."""
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "mtp", cache) as capture:
        run_pass(src, layer, state, MAX_HISTORY, True, 1.0)
        state.rewind(2)
        # the library's seam, called the way a later verifier/rewind would: same layer, same cache
        rsl.replay_job(0, 1, (1, MAX_HISTORY, 1), 1, 0)
        rsl.replay_job(0, 1, (1, MAX_HISTORY, 1), 1, 0)
        # a second verifier pass at the next window position, then its rewind: the forward hook
        # excludes the pass, so the rewind must be skipped and its job must not be recorded either
        run_pass(src, layer, state, MAX_HISTORY, True, 2.0)
        state.rewind(2)
    jobs = [r for r in capture.records if r["kind"] == "replay_job"]
    assert len(jobs) == 1, "a job outside the captured rewind was recorded"
    assert capture.excluded["replay_job_outside_rewind"] == 3, capture.excluded
    assert capture.excluded["second_verify_pass"] == 1
    assert capture.excluded["rewind_without_verifier_forward"] == 1
    assert capture.overflow is False, "later jobs consumed the record cap"
    capture.write("late-jobs")
    assert "incomplete" not in " ".join(capture.completeness_errors())


def test_replay_job_with_another_ledger_is_excluded(tmp_path):
    """A job whose geometry is not the selected rewind's is an exclusion, not a record."""
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "mtp", cache) as capture:
        run_pass(src, layer, state, MAX_HISTORY, True, 4.0)
        capture._active_rewind = {"entry": 7, "prefix": 1, "spec_shape": [1, MAX_HISTORY, 1],
                                  "base_row": 0, "scratch_row": 1}
        try:
            rsl.replay_job(0, 2, (1, MAX_HISTORY, 1), 0, 1)         # wrong prefix
            rsl.replay_job(0, 1, (1, MAX_HISTORY, 1), 1, 0)         # wrong pool rows
            rsl.replay_job(0, 1, (1, MAX_HISTORY, 1), 0, 1)         # matches the ledger
        finally:
            capture._active_rewind = None
    jobs = [r for r in capture.records if r["kind"] == "replay_job"]
    assert len(jobs) == 1 and jobs[0]["pass_entry"] == 7
    assert capture.excluded["replay_job_ledger_mismatch"] == 2


def test_missing_required_operand_fails_loudly_and_keeps_partial_artifacts(tmp_path):
    src = real_source()
    src["gdn"].g_tensor_cache.drop_all()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "ar", cache) as capture:
        original = layer.bc.configure_slot

        def without_g(*args, **kwargs):
            # C++ was handed a g buffer the cache does not hold: the entry is simply absent
            original(*args, **kwargs)
            key = src["gdn"].g_tensor_cache.make_key(layer.device, (1, 1, layer.num_v_heads),
                                                     torch.float, "s_g")
            src["gdn"].g_tensor_cache.cache.pop(key, None)
        layer.bc.configure_slot = without_g
        run_pass(src, layer, state, 1, False, 1.0)
    assert capture.errors and "s_g" in capture.errors[0]
    assert capture.forwards() == []
    info = capture.write("partial")
    assert info["records"] == 0 and info["errors"]
    with open(info["json"]) as f:
        doc = json.load(f)
    assert doc["errors"] and "s_g" in doc["errors"][0]
    with pytest.raises(cap.CaptureError) as exc:
        capture.assert_complete()
    assert "incomplete" in str(exc.value)


def test_capture_restores_every_seam(tmp_path):
    src = real_source()
    g = src["gdn"]
    before = (g.GatedDeltaNet.forward, g.GDNState.rewind, g.GDNLayerState.replay_job)
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "ar", cache):
        assert g.GatedDeltaNet.forward is not before[0]
        assert g.GDNState.rewind is not before[1]
        assert g.GDNLayerState.replay_job is not before[2]
    assert (g.GatedDeltaNet.forward, g.GDNState.rewind, g.GDNLayerState.replay_job) == before
    run_pass(src, layer, state, 1, False, 1.0)     # the original seam still works


def test_overflow_is_reported_and_assert_complete_raises(tmp_path):
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "ar", cache, max_forwards=1) as capture:
        for marker in (1.0, 2.0):
            run_pass(src, layer, state, 1, False, marker)
    assert capture.overflow is True
    assert len(capture.forwards("target_decode")) == 1, "cloning must stop at the cap"
    info = capture.write("overflow")
    assert info["overflow"] is True
    with pytest.raises(cap.CaptureError) as exc:
        capture.assert_complete()
    assert "overflow" in str(exc.value)


def test_pass_metadata_advances_outside_the_layer(tmp_path):
    """The layer forward must not move the sequence metadata; `advance_recurrent_states` does."""
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "mtp", cache) as capture:
        layer.bc.writer = _operand_writer(2.0)
        ids = torch.full((1, MAX_HISTORY), 7, dtype=torch.long)
        params = {"cache_seqlens": torch.tensor([7], dtype=torch.int32),
                  "recurrent_states": [state], "recurrent_history": True}
        src["rt"].prepare_for_recurrence(ids, params, None)
        layer.forward(torch.zeros(1, MAX_HISTORY, layer.hidden_size), params)
        assert (state.position, state.last_history, state.spec_row) == (7, 0, None)
        rec = capture.forwards("target_verify")[0]
        assert rec["before"]["state"]["position"] == 7
        assert rec["after"]["state"]["position"] == 7, "the capture recorded a moved position"
        assert rec["operands"]["spec_shape"] == [1, MAX_HISTORY, 1], "staging shape is set by stage()"
        src["rt"].advance_recurrent_states(ids, params, None)
        assert (state.position, state.last_history, state.spec_row) == (10, 2, 0)
        assert state.spec_shape == (1, MAX_HISTORY, 1)


def test_x_slice_is_the_sequence_axis_not_the_batch(tmp_path):
    src = real_source()
    src["gdn"].g_tensor_cache.drop_all()
    layer = make_layer(src)
    rsl = src["gdn"].GDNLayerState(layer, 3, MAX_HISTORY, cache_id=0, state_dtype=torch.float16)
    rsl.alloc(torch.device("cpu"))
    cache = FakeCache(rsl)
    rsl.cache_id = id(cache)
    states = [src["gdn"].GDNState(cache, slot=i, position=7, test_state=True) for i in range(3)]
    with capture_env(src, tmp_path, "ar", cache) as capture:
        ids = torch.full((3, 2), 7, dtype=torch.long)
        params = {"cache_seqlens": torch.tensor([7, 7, 7], dtype=torch.int32),
                  "recurrent_states": states, "recurrent_history": False}
        src["rt"].prepare_for_recurrence(ids, params, None)
        x = torch.zeros(3, 2, layer.hidden_size)
        layer.forward(x, params)
    rec = capture.forwards("target_decode")[0]
    assert rec["bsz"] == 3 and rec["seqlen"] == 2
    xrec = rec["before"]["x"]
    assert xrec["seq_slice"] == [0, 2], "the slice bounds are sequence columns"
    assert xrec["value"]["tensor"].shape == (3, 2, layer.hidden_size), \
        "a batch-axis slice (2, 2, h) is the v1 bug this asserts against"
    assert xrec["source_layout"]["shape"] == [3, 2, layer.hidden_size]
    assert xrec["value"]["layout"]["stride"] == list(x.stride())


def test_mode_aware_completeness(tmp_path):
    """`assert_complete` names what the mode still lacks, after the partial artifact is written."""
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "mtp", cache) as capture:
        run_pass(src, layer, state, MAX_HISTORY, True, 3.0)     # verifier pass, no rewind yet
    info = capture.write("mtp-partial")
    assert info["records"] == 1
    with pytest.raises(cap.CaptureError) as exc:
        capture.assert_complete()
    message = str(exc.value)
    assert "0 rewinds" in message and "replay_job" in message

    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)
    with capture_env(src, tmp_path, "ar", cache) as ar_capture:
        run_pass(src, layer, state, 1, False, 1.0)              # one pass only
    ar_capture.write("ar-partial")
    with pytest.raises(cap.CaptureError) as exc:
        ar_capture.assert_complete()
    assert "at least two adjacent" in str(exc.value)

    paths = build_captures(tmp_path / "full")
    for mode in ("ar", "mtp"):
        doc = fx.load_capture(paths[mode])
        assert doc["errors"] == []


def test_capture_failure_never_breaks_the_model_call(tmp_path):
    """A diagnostic snapshot error is recorded and surfaced later; the forward still returns."""
    src = real_source()
    layer = make_layer(src)
    rsl, cache, state = make_state(src, layer, position=7)

    def boom():
        raise RuntimeError("scan_geometry unavailable")
    rsl.scan_geometry = boom
    with capture_env(src, tmp_path, "ar", cache) as capture:
        out, _ = run_pass(src, layer, state, 1, False, 1.0)
    assert out.shape == (1, 1, layer.hidden_size)
    assert capture.forwards() == []
    assert capture.errors and "snapshot failed" in capture.errors[0]
    capture.write("robust")
    with pytest.raises(cap.CaptureError) as exc:
        capture.assert_complete()
    assert "snapshot failed" in str(exc.value)


# ----------------------------------------------------------------------------- 2. artifact + fixture

def test_artifact_round_trip_feeds_the_fixture(tmp_path):
    paths = build_captures(tmp_path)
    ar = fx.load_capture(paths["ar"])
    mtp = fx.load_capture(paths["mtp"])
    assert ar["capture_mode"] == "ar" and mtp["capture_mode"] == "mtp"
    assert ar["errors"] == [] and mtp["errors"] == []
    # JSON holds references only; payloads live in the .pt
    with open(paths["ar"]) as f:
        raw = f.read()
    assert "tensor_ref" in raw
    fixture = fx.build_fixture(ar, mtp)
    assert fixture["ledger"]["accepted_prefix"] == 1
    assert fixture["ledger"]["staged_tokens"] == MAX_HISTORY
    assert len(fixture["ar"]) == 2
    assert fixture["mtp"]["staged_conv"].shape[0] == MAX_HISTORY
    assert fixture["mtp"]["initial_base"].dtype == torch.float16
    assert fixture["checks"]["initial_state_equal"]["exact"] is True
    assert all(v["exact"] for v in fixture["checks"]["operands_stable_forward_vs_rewind"].values())


def test_fixture_rejects_captures_without_required_evidence(tmp_path):
    paths = build_captures(tmp_path)
    ar = fx.load_capture(paths["ar"])
    mtp = fx.load_capture(paths["mtp"])
    broken = copy.deepcopy(mtp)
    broken["records"] = [r for r in broken["records"] if r["kind"] != "rewind"]
    with pytest.raises(fx.FixtureError) as exc:
        fx.build_fixture(ar, broken)
    assert "rewind" in str(exc.value)

    bad_mode = dict(mtp, capture_mode="ar")
    with pytest.raises(fx.FixtureError):
        fx.build_fixture(ar, bad_mode)

    overflowed = dict(ar, overflow=True)
    with pytest.raises(fx.FixtureError) as exc:
        fx.build_fixture(overflowed, mtp)
    assert "overflow" in str(exc.value)

    no_entry = dict(ar, records=[r for r in ar["records"]
                                 if not (r["kind"] == "layer0_forward" and r["position"] == 7)])
    with pytest.raises(fx.FixtureError) as exc:
        fx.build_fixture(no_entry, mtp)
    assert "pass entry 7" in str(exc.value)

    wrong_width = copy.deepcopy(mtp)
    for rec in wrong_width["records"]:
        if rec["kind"] == "rewind":
            conv = rec["before"]["operands"]["row_block"]["spec_conv"]["value"]
            conv["tensor"] = conv["tensor"][:, :-1].contiguous()
        elif rec["kind"] == "layer0_forward":
            conv = rec["operands"]["conv"]["value"]
            conv["tensor"] = conv["tensor"][:, :-1].contiguous()
    with pytest.raises(fx.FixtureError) as exc:
        fx.build_fixture(ar, wrong_width)
    assert "q|k|v" in str(exc.value)


def _with_record(capture, kind, mutate, position=None):
    """A copy of a loaded capture with one record changed (records hold tensors, so the copy is
    shallow on tensors and tests mutate cloned values deliberately)."""
    out = dict(capture)
    out["records"] = [dict(r) for r in capture["records"]]
    rec = next(r for r in out["records"] if r["kind"] == kind
               and (position is None or int(r.get("position", -1)) == position))
    mutate(rec)
    return out, rec


def _bump_state_row(rec, key, position):
    row = rec["before"]["canonical"][key]["tensor"].clone()
    row.view(torch.int16)[0, 0, 0, 0] += 1                     # 1 ULP of fp16
    rec["before"] = dict(rec["before"])
    rec["before"]["canonical"] = dict(rec["before"]["canonical"])
    rec["before"]["canonical"][key] = {"tensor": row}


def test_prefix2_fixture_compares_the_terminal_ar_frontier(tmp_path):
    """The reproduced ledger (entry 7, three staged rows, rewind 1 -> accepted_prefix 2, post 9).

    The AR state after the second accepted token is the *before* row of the AR pass at 9, i.e. a
    terminal frontier: it has no following pass, so the executable step pairs cannot carry it.
    """
    paths = build_captures(tmp_path, seed=4242.0, rewind_tokens=1)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])
    assert paths["position_after"] == 9
    fixture = fx.build_fixture(ar, mtp)
    assert fixture["ledger"]["accepted_prefix"] == 2
    assert fixture["ledger"]["position_after"] == 9
    assert [s["position_after"] for s in fixture["ar"]] == [8, 9], "both step pairs stay available"
    assert 9 in fixture["ar_frontiers"] and 10 in fixture["ar_frontiers"]
    assert "before.canonical" in fixture["ar_frontiers"][9]["source"]
    assert "pass at 9" in fixture["ar_frontiers"][9]["source"]
    assert fixture["prerequisites"]["ok"] is True

    report = fx.replay(ar, mtp, ext=MarkerExt(paths["rewind_marker"]), device="cpu")
    frontier_cmp = report["mtp_native_vs_ar_actual"]
    assert "not_compared" not in frontier_cmp, "the terminal AR frontier was dropped again"
    assert frontier_cmp["compared_at_position"] == 9
    assert "pass at 9" in frontier_cmp["ar_state_source"]
    assert frontier_cmp["exact"] is True
    # the operand comparison covers BOTH accepted rows, not only row 0
    rows = report["operands_mtp_vs_ar"]["rows"]
    assert [r["position"] for r in rows] == [7, 8]
    assert all(r["all_exact"] for r in rows), rows
    # the per-token chain runs over the same two accepted rows and ends at the same frontier
    chain = report["chain"]
    assert chain["steps"] == 2 and chain["operand_rows"] == [0, 1]
    assert chain["storage_dtype"] == "torch.float16" and chain["valid"] is True
    assert chain["final_vs_joint"]["exact"] is True
    assert chain["final_vs_captured_post"]["exact"] is True
    assert chain["final_vs_ar_actual"]["exact"] is True
    assert report["gate"]["passed"] is True


def test_prefix1_fixture_stays_the_boundary_case(tmp_path):
    """The easier prefix1 transition keeps its own coverage: target 8 is a frontier and a step."""
    paths = build_captures(tmp_path)                     # rewind_tokens=2 -> accepted_prefix 1
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])
    fixture = fx.build_fixture(ar, mtp)
    assert fixture["ledger"]["accepted_prefix"] == 1
    assert fixture["ledger"]["position_after"] == 8
    report = fx.replay(ar, mtp, ext=MarkerExt(paths["rewind_marker"]), device="cpu")
    assert report["mtp_native_vs_ar_actual"]["compared_at_position"] == 8
    assert report["mtp_native_vs_ar_actual"]["exact"] is True
    assert len(report["operands_mtp_vs_ar"]["rows"]) == 1
    assert report["chain"]["steps"] == 1
    assert report["gate"]["passed"] is True


def test_prerequisites_fail_closed_and_suppress_the_native_comparison(tmp_path):
    """A false starting-state/operand-stability check is a hard gate, never a footnote."""
    paths = build_captures(tmp_path, seed=777.0, rewind_tokens=1)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])

    def entry_base(rec):
        if int(rec["position"]) == 7:
            _bump_state_row(rec, "rec_base", 7)

    bad_state, _ = _with_record(ar, "layer0_forward", entry_base, position=7)
    spy = MarkerExt(paths["rewind_marker"])
    report = fx.replay(bad_state, mtp, ext=spy, device="cpu")
    assert report["prerequisites"]["ok"] is False
    assert report["prerequisites"]["failed"] == ["initial_state_equal"]
    assert report["status"] == "prerequisites_failed:non_native_backend"
    assert report["gate"]["passed"] is False and report["gate"]["suppressed"] is True
    assert "suppressed" in report["mtp_native_vs_ar_actual"]
    assert "suppressed" in report["operands_mtp_vs_ar"] and "suppressed" in report["chain"]
    assert report["ar_native_steps"] == [] and "not_run" in report["reference"]
    # nothing was replayed: no pointer was taken from a capture that cannot answer the question
    assert spy.jobs == [] and spy.calls == []
    # the detailed failure diff survives for the report
    assert report["checks"]["initial_state_equal"]["equal"] is False
    assert report["checks"]["initial_state_equal"]["diff"]["n_diff"] == 1

    def entry_conv(rec):
        if int(rec["position"]) != 7:
            return
        conv = rec["before"]["canonical"]["conv_committed"]["tensor"].clone()
        conv[0, 0] = conv[0, 0] + 1
        rec["before"] = dict(rec["before"])
        rec["before"]["canonical"] = dict(rec["before"]["canonical"])
        rec["before"]["canonical"]["conv_committed"] = {"tensor": conv}

    bad_conv, _ = _with_record(ar, "layer0_forward", entry_conv, position=7)
    report = fx.replay(bad_conv, mtp, ext=MarkerExt(paths["rewind_marker"]), device="cpu")
    assert report["prerequisites"]["failed"] == ["initial_conv_equal"]
    assert report["gate"]["passed"] is False

    def overwrite_conv_operand(rec):
        rec["operands"] = dict(rec["operands"])
        rec["operands"]["conv"] = dict(rec["operands"]["conv"])
        rec["operands"]["conv"]["value"] = dict(rec["operands"]["conv"]["value"])
        t = rec["operands"]["conv"]["value"]["tensor"].clone()
        t[0, 0] = t[0, 0] + 1
        rec["operands"]["conv"]["value"]["tensor"] = t

    bad_ops, _ = _with_record(mtp, "layer0_forward", overwrite_conv_operand, position=7)
    report = fx.replay(ar, bad_ops, ext=MarkerExt(paths["rewind_marker"]), device="cpu")
    assert report["prerequisites"]["failed"] == ["operands_stable_forward_vs_rewind.conv"]
    assert report["gate"]["passed"] is False and report["gate"]["suppressed"] is True


def test_fixture_rejects_a_replay_job_from_another_ledger(tmp_path):
    paths = build_captures(tmp_path)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])

    def wrong_prefix(rec):
        rec["prefix"] = int(rec["prefix"]) + 1

    bad, _ = _with_record(mtp, "replay_job", wrong_prefix)
    with pytest.raises(fx.FixtureError) as exc:
        fx.build_fixture(ar, bad)
    assert "match the selected rewind ledger" in str(exc.value)


def test_fixture_rejects_a_batched_staged_pass(tmp_path):
    paths = build_captures(tmp_path)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])

    def two_rows(rec):
        rec["before"] = dict(rec["before"])
        rec["before"]["spec_shape"] = [2, MAX_HISTORY, 1]

    bad, _ = _with_record(mtp, "rewind", two_rows)
    with pytest.raises(fx.FixtureError) as exc:
        fx.build_fixture(ar, bad)
    assert "bsz 2" in str(exc.value)


def test_ar_native_mismatch_stays_explicit_and_limits_the_claim(tmp_path):
    """An AR one-step replay that misses is reported as evidence, never as rounding."""
    paths = build_captures(tmp_path)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])

    def stored_state(rec):
        if int(rec["position"]) == 8:
            _bump_state_row(rec, "rec_base", 8)

    bad_ar, _ = _with_record(ar, "layer0_forward", stored_state, position=8)
    report = fx.replay(bad_ar, mtp, ext=MarkerExt(paths["rewind_marker"]), device="cpu")
    assert report["ar_replay"]["all_exact"] is False
    assert report["ar_replay"]["not_exact"] == [7]
    assert "not a rounding result" in report["ar_replay"]["note"]
    assert "Limit: an AR one-step native replay" in report["diagnosis"]
    # the MTP gate itself is unaffected: the AR limit is stated, not used to excuse it
    assert report["gate"]["passed"] is True


def test_reference_uses_the_real_grouped_head_layout():
    nk, nv, dk, dv = 2, 4, 2, 2
    T = 2
    conv = torch.zeros(T, 2 * nk * dk + nv * dv, dtype=torch.float32)
    k0 = torch.tensor([1.0, 0.5])
    k1 = torch.tensor([-0.5, 2.0])
    conv[:, :nk * dk] = 0.25                       # q channels: irrelevant to the state
    conv[:, nk * dk] = k0[0]                       # k head 0: channels [nk*dk, nk*dk + dk)
    conv[:, nk * dk + 1] = k0[1]
    conv[:, nk * dk + dk] = k1[0]                  # k head 1: channels [nk*dk + dk, 2*nk*dk)
    conv[:, nk * dk + dk + 1] = k1[1]
    for h in range(nv):
        conv[:, 2 * nk * dk + h * dv] = 0.1 * (h + 1)
        conv[:, 2 * nk * dk + h * dv + 1] = -0.2 * (h + 1)
    ops = {"staged_conv": conv, "staged_g": torch.full((T, nv), -0.1),
           "staged_beta": torch.full((T, nv), 0.5),
           "initial_base": torch.zeros(1, nv, dk, dv), "prefix": T, "tstride": 1,
           "geometry": {"kind": 0, "nk": nk, "nv": nv, "dk": dk, "dv": dv, "state_half": True}}
    groups = fx.reference_operands(ops)
    assert groups["G"] == 2 and groups["k"].shape == (T, nv, dk)
    assert torch.equal(groups["k"][:, 0], groups["k"][:, 1]), "v heads 0/1 share k head 0"
    assert torch.equal(groups["k"][:, 2], groups["k"][:, 3]), "v heads 2/3 share k head 1"
    assert not torch.equal(groups["k"][:, 1], groups["k"][:, 2])
    # each head really carries the channels that were written for it (not zeros, not the V block)
    assert torch.allclose(groups["k"][0, 0], fx._l2norm(k0.double()))
    assert torch.allclose(groups["k"][0, 2], fx._l2norm(k1.double()))
    assert groups["k"][:, 2].abs().sum() > 0
    # a manual, ungrouped-split loop must NOT reproduce it: the grouping is load-bearing
    states = fx.reference_states(ops)
    manual = _manual_recurrence(ops, groups)
    assert torch.allclose(states, manual, atol=0, rtol=0), (states - manual).abs().max()
    # pairing the v heads to the *other* k head must change the state: grouping is load-bearing
    swapped = _manual_recurrence(ops, dict(groups, k=groups["k"][:, [2, 3, 0, 1]]))
    assert not torch.allclose(states, swapped)


def _manual_recurrence(ops, groups):
    nv, dk, dv = (int(ops["geometry"][k]) for k in ("nv", "dk", "dv"))
    S = ops["initial_base"].double().reshape(nv, dk, dv)
    out = []
    for t in range(int(ops["prefix"])):
        decay = ops["staged_g"][t].double().exp()
        kv = torch.einsum("hk,hkv->hv", groups["k"][t], S)
        vt = groups["v"][t] - kv * decay[:, None]
        S = S * decay[:, None, None] + groups["k"][t][:, :, None] * \
            (vt * ops["staged_beta"][t].double()[:, None])[:, None, :]
        out.append(S.clone())
    return torch.stack(out)


def test_reference_is_float64_and_independent_of_q():
    ops = {"staged_conv": torch.full((1, 2 * 2 * 2 + 4 * 2), 0.5),
           "staged_g": torch.full((1, 4), -0.3), "staged_beta": torch.full((1, 4), 0.7),
           "initial_base": torch.zeros(1, 4, 2, 2), "prefix": 1, "tstride": 1,
           "geometry": {"kind": 0, "nk": 2, "nv": 4, "dk": 2, "dv": 2, "state_half": True}}
    states = fx.reference_states(ops)
    assert states.dtype == torch.float64
    scaled = dict(ops)
    conv = ops["staged_conv"].clone()
    conv[:, :4] *= 10.0                      # only q channels change
    scaled["staged_conv"] = conv
    assert torch.equal(fx.reference_states(scaled), states), "the state must not depend on q"


class BaseWritingExt(MarkerExt):
    """Writes into the job's **base** pointer: the input-mutation bug `base_unmodified` detects.

    It still writes the expected value into the scratch row, so the numbers match while the pointer
    wiring is wrong - exactly the case that must not pass as a valid comparison.
    """

    def batched_scan_replay(self, jobs, device_index, kind, nk, nv, dk, dv, state_half):
        super().batched_scan_replay(jobs, device_index, kind, nk, nv, dk, dv, state_half)
        dtype = torch.float16 if state_half else torch.float32
        tensor_at(jobs[0]["base_ptr"], nv * dk * dv, dtype).fill_(-3.0)


def test_a_modified_input_base_invalidates_the_gate(tmp_path):
    paths = build_captures(tmp_path)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])
    spy = BaseWritingExt(paths["rewind_marker"])
    report = fx.replay(ar, mtp, ext=spy, device="cpu")
    assert report["gate"]["base_unmodified"] is False
    assert report["gate"]["diff"]["max"] == 0.0, "the values match: only the input mutation differs"
    assert report["gate"]["passed"] is False
    assert "wrote into its own input base copy" in report["gate"]["reason"]
    assert "input base copy" in report["diagnosis"]
    assert report["status"] == "gate_failed:input_base_mutated:non_native_backend"
    assert report["base_integrity"]["ok"] is False
    assert report["base_integrity"]["mutated_calls"] == ["mtp"]
    # the failed initial gate stops the diagnostic: no AR/chain call, no attribution
    assert len(spy.jobs) == 1, "later native calls ran after the mandatory gate failed"
    assert report["ar_native_steps"] == [] and "not_run" in report["ar_replay"]
    assert "suppressed" in report["mtp_native_vs_ar_actual"]
    assert "suppressed" in report["operands_mtp_vs_ar"]
    assert "chain" not in report or "suppressed" in report["chain"]


class IndexedBaseWriter(MarkerExt):
    """Writes the expected output on every call, but mutates the input base of ONE selected call.

    Call order in these fixtures is: 0 = the joint MTP replay, then one call per AR step, then one call
    per chained step. The index therefore selects `mtp`, `ar:<position>` or `chain:<row>` without
    changing the tool's call sequence.
    """

    def __init__(self, value, mutate_call):
        super().__init__(value)
        self.mutate_call = int(mutate_call)
        self.calls_made = 0

    def batched_scan_replay(self, jobs, device_index, kind, nk, nv, dk, dv, state_half):
        super().batched_scan_replay(jobs, device_index, kind, nk, nv, dk, dv, state_half)
        index = self.calls_made
        self.calls_made += 1
        if index == self.mutate_call:
            dtype = torch.float16 if state_half else torch.float32
            tensor_at(jobs[0]["base_ptr"], nv * dk * dv, dtype).fill_(-7.0)


def test_an_ar_only_base_mutation_fails_the_diagnostic(tmp_path, monkeypatch):
    """A later call's input-base mutation must not leave an overall successful gate/status."""
    paths = build_captures(tmp_path, rewind_tokens=1)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])
    spy = IndexedBaseWriter(paths["rewind_marker"], mutate_call=1)      # first AR step
    report = fx.replay(ar, mtp, ext=spy, device="cpu")
    assert report["status"] == "gate_failed:input_base_mutated:non_native_backend"
    assert report["gate"]["passed"] is False and "input base copy" in report["gate"]["reason"]
    assert report["base_integrity"]["ok"] is False
    assert report["base_integrity"]["mutated_calls"] == ["ar:7"]
    # details are retained, attribution is withdrawn
    assert [r["base_unmodified"] for r in report["ar_native_steps"]] == [False, True]
    assert report["chain"]["step_base_unmodified"] == [True, True]
    assert "suppressed" in report["mtp_native_vs_ar_actual"]
    assert "suppressed" in report["operands_mtp_vs_ar"]
    assert "suppressed" in report["chain"]["final_vs_joint"]
    assert "attribution_suppressed" in report["chain"]
    assert "attribution_suppressed" in report["ar_replay"]
    # the CLI path is nonzero and preserves the report
    monkeypatch.setattr(fx, "load_ext", lambda: IndexedBaseWriter(paths["rewind_marker"], 1))
    out = tmp_path / "ar-base-mutated.json"
    rc = fx.main(["--ar", paths["ar"], "--mtp", paths["mtp"], "--out", str(out), "--device", "cpu"])
    assert rc == 1
    assert json.loads(out.read_text())["base_integrity"]["mutated_calls"] == ["ar:7"]
    assert (tmp_path / "ar-base-mutated.json.FAILED.json").exists()


def test_a_chain_only_base_mutation_fails_the_diagnostic(tmp_path, monkeypatch):
    """Same rule for the per-token chain: `valid`/`step_base_unmodified` are not merely reported."""
    paths = build_captures(tmp_path, rewind_tokens=1)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])
    # calls: 0 joint, 1 AR@7, 2 AR@8, 3 chain row 0, 4 chain row 1
    spy = IndexedBaseWriter(paths["rewind_marker"], mutate_call=3)
    report = fx.replay(ar, mtp, ext=spy, device="cpu")
    assert report["base_integrity"]["mutated_calls"] == ["chain:0"]
    assert report["base_integrity"]["ok"] is False
    assert report["gate"]["passed"] is False
    assert report["status"] == "gate_failed:input_base_mutated:non_native_backend"
    assert report["chain"]["step_base_unmodified"] == [False, True]
    assert report["chain"]["valid"] is False
    assert "suppressed" in report["chain"]["final_vs_ar_actual"]
    assert "suppressed" in report["mtp_native_vs_ar_actual"]
    monkeypatch.setattr(fx, "load_ext", lambda: IndexedBaseWriter(paths["rewind_marker"], 3))
    out = tmp_path / "chain-base-mutated.json"
    rc = fx.main(["--ar", paths["ar"], "--mtp", paths["mtp"], "--out", str(out), "--device", "cpu"])
    assert rc == 1, "a mutated input base must be a nonzero diagnostic"
    assert (tmp_path / "chain-base-mutated.json.FAILED.json").exists()


def test_a_failed_initial_gate_suppresses_later_calls_and_comparisons(tmp_path):
    """When the mandatory joint replay misses, no AR/chain call is made and nothing is attributed."""
    paths = build_captures(tmp_path, rewind_tokens=1)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])
    spy = MarkerExt(value=999.0)                    # reproduces neither the captured post row nor AR
    report = fx.replay(ar, mtp, ext=spy, device="cpu")
    assert report["gate"]["passed"] is False
    assert report["status"] == "gate_failed:inputs_or_fast_path:non_native_backend"
    assert len(spy.jobs) == 1, "AR/chain calls ran after the mandatory initial gate failed"
    assert report["ar_native_steps"] == [] and "not_run" in report["ar_replay"]
    assert "suppressed" in report["mtp_native_vs_ar_actual"]
    assert "suppressed" in report["operands_mtp_vs_ar"]
    assert "chain" not in report or "suppressed" in report["chain"]
    assert report["base_integrity"] == {"ok": True, "mutated_calls": [],
                                        "note": report["base_integrity"]["note"]}
    # AR output mismatch with an unchanged base is still diagnostic evidence, never a forced equality
    passing = fx.replay(ar, mtp, ext=MarkerExt(paths["rewind_marker"]), device="cpu")
    assert passing["gate"]["passed"] is True
    assert passing["ar_replay"]["all_exact"] is True


def test_native_replay_refuses_to_cast_or_repair_captured_operands(tmp_path):
    """A captured dtype/shape that is not the staging layout is a capture defect, not a cast."""
    paths = build_captures(tmp_path)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])
    fixture = fx.build_fixture(ar, mtp)
    base_ops = fixture["mtp"]
    spy = MarkerExt(0.0)

    def refused(**changes):
        ops = dict(base_ops)
        ops.update(changes)
        with pytest.raises(fx.FixtureError) as exc:
            fx.native_scan_replay(ops, spy, device="cpu")
        return str(exc.value)

    message = refused(staged_conv=base_ops["staged_conv"].float())
    assert "refusing to cast captured operands" in message
    message = refused(initial_base=base_ops["initial_base"].float())
    assert "state storage dtype" in message and "refusing to cast" in message
    message = refused(staged_beta=base_ops["staged_beta"].double())
    assert "refusing to cast" in message
    message = refused(prefix=4)
    assert "longer than the 3 staged rows" in message
    message = refused(geometry=dict(fixture["geometry"], nk=3))
    assert "not a multiple of nk" in message
    message = refused(d_skip=torch.zeros(1))
    assert "d_skip" in message
    ops = dict(base_ops)
    ops["staged_g"] = base_ops["staged_g"][:, :1].contiguous()
    with pytest.raises(fx.FixtureError) as exc:
        fx.native_scan_replay(ops, spy, device="cpu")
    assert "wide" in str(exc.value)
    assert spy.jobs == [] and spy.calls == [], "a refused capture launched a native call"


def test_device_index_must_agree_with_the_replay_device(monkeypatch):
    assert fx._replay_target("cpu", None) == (torch.device("cpu"), 0)
    assert fx._replay_target("cpu", 3) == (torch.device("cpu"), 3)
    with pytest.raises(fx.FixtureError):
        fx._replay_target("cpu", -1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert fx._replay_target("cuda", 1) == (torch.device("cuda", 1), 1)
    assert fx._replay_target("cuda:0", None) == (torch.device("cuda", 0), 0)
    with pytest.raises(fx.FixtureError) as exc:
        fx._replay_target("cuda:0", 1)
    assert "contradicts device" in str(exc.value)


def test_a_capture_from_another_device_is_refused(tmp_path):
    paths = build_captures(tmp_path)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])
    fixture = fx.build_fixture(ar, mtp)
    ops = dict(fixture["mtp"])
    ops["source_device"] = "cuda:0"
    with pytest.raises(fx.FixtureError) as exc:
        fx.native_scan_replay(ops, MarkerExt(0.0), device="cpu")
    assert "refusing to relabel a captured transition" in str(exc.value)
    with pytest.raises(fx.FixtureError) as exc:
        fx.native_scan_replay(ops, MarkerExt(0.0), device="cpu", device_index=1)
    fx._check_source_device({"source_device": "cpu"}, torch.device("cpu"))


class ChainSpy:
    """Records each job and writes a value that depends on the base row and the conv rows.

    It proves the *wiring* of the chained replay (which rows, which base, which storage dtype); it
    computes no recurrence and is never native numeric evidence.
    """

    def __init__(self):
        self.jobs = []

    def ScanReplayJob(self, conv_ptr, g_ptr, beta_ptr, base_ptr, scratch_ptr, d_ptr, prefix, tstride):
        job = {"conv_ptr": conv_ptr, "g_ptr": g_ptr, "beta_ptr": beta_ptr, "base_ptr": base_ptr,
               "scratch_ptr": scratch_ptr, "d_ptr": d_ptr, "prefix": prefix, "tstride": tstride}
        self.jobs.append(job)
        return job

    def batched_scan_replay(self, jobs, device_index, kind, nk, nv, dk, dv, state_half):
        dtype = torch.float16 if state_half else torch.float32
        fdim = 2 * nk * dk + nv * dv
        for job in jobs:
            rows = job["prefix"]
            base = tensor_at(job["base_ptr"], nv * dk * dv, dtype)
            conv = tensor_at(job["conv_ptr"], rows * fdim, torch.bfloat16).view(rows, fdim)
            out = tensor_at(job["scratch_ptr"], nv * dk * dv, dtype)
            job["base_values"] = base.detach().clone()
            job["conv_rows"] = conv.detach().clone()
            value = base.float().mean() + conv.float().sum()
            out.fill_(float(value))


def _chain_value(state, conv_row):
    return float(state.float().mean() + conv_row.float().sum())


def test_native_chain_replays_one_pass_per_accepted_row(tmp_path):
    paths = build_captures(tmp_path, rewind_tokens=1)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])
    fixture = fx.build_fixture(ar, mtp)
    ops = fixture["mtp"]
    spy = ChainSpy()
    chain = fx.native_chain_replay(ops, spy, device="cpu")
    jobs = spy.jobs
    assert len(jobs) == len(chain["steps"]) == 2 == fixture["ledger"]["accepted_prefix"]
    assert [j["prefix"] for j in jobs] == [1, 1] and chain["operand_rows"] == [0, 1]
    # each step consumes its own captured staged row, in order
    assert torch.equal(jobs[0]["conv_rows"], ops["staged_conv"][0:1])
    assert torch.equal(jobs[1]["conv_rows"], ops["staged_conv"][1:2])
    # step 0 starts from the captured base; step 1 starts from step 0's result, same storage dtype
    assert torch.equal(jobs[0]["base_values"], ops["initial_base"].reshape(-1))
    assert torch.equal(jobs[1]["base_values"],
                       torch.full_like(ops["initial_base"].reshape(-1),
                                       _chain_value(ops["initial_base"], ops["staged_conv"][0:1])))
    assert chain["storage_dtype"] == "torch.float16"
    assert torch.equal(chain["final"],
                       torch.full_like(ops["initial_base"],
                                       _chain_value(jobs[1]["base_values"], ops["staged_conv"][1:2])))
    assert chain["valid"] is True


def test_report_records_the_backend_identity(tmp_path):
    paths = build_captures(tmp_path)
    ar, mtp = fx.load_capture(paths["ar"]), fx.load_capture(paths["mtp"])
    report = fx.replay(ar, mtp, ext=MarkerExt(paths["rewind_marker"]), device="cpu")
    assert report["ext"]["path"] is None
    assert "not native provenance" in report["ext"]["note"]
    ident = fx._ext_identity(types.SimpleNamespace(__file__=os.path.abspath(__file__)))
    assert ident["path"] == os.path.realpath(__file__) and len(ident["sha256"]) == 64
    assert report["source_device"] == "cpu" and report["replay_job"]["prefix"] == 1


# ----------------------------------------------------------------------------- 3. native gate

def test_native_replay_uses_dedicated_buffers_and_the_library_geometry(tmp_path):
    paths = build_captures(tmp_path)
    ar = fx.load_capture(paths["ar"])
    mtp = fx.load_capture(paths["mtp"])
    fixture = fx.build_fixture(ar, mtp)
    operands = fixture["mtp"]
    spy = MarkerExt(paths["rewind_marker"])
    out = fx.native_scan_replay(operands, spy, device="cpu", device_index=0)
    assert len(spy.jobs) == 1 and len(spy.calls) == 1
    job, call = spy.jobs[0], spy.calls[0]
    assert job["prefix"] == 1 and job["tstride"] == 1 and job["d_ptr"] == 0
    assert (call["kind"], call["nk"], call["nv"], call["dk"], call["dv"], call["state_half"]) == \
        (0, 2, 4, 2, 2, True)
    assert call["device_index"] == 0
    captured = {operands["staged_conv"].data_ptr(), operands["staged_g"].data_ptr(),
                operands["staged_beta"].data_ptr(), operands["initial_base"].data_ptr()}
    touched = {job["conv_ptr"], job["g_ptr"], job["beta_ptr"], job["base_ptr"], job["scratch_ptr"]}
    assert not (touched & captured), "the kernel was pointed at captured arrays instead of copies"
    assert out["base_unmodified"] is True
    assert torch.equal(out["state"], torch.full_like(out["state"], paths["rewind_marker"]))


def test_gate_is_exact_on_the_kernel_output(tmp_path):
    paths = build_captures(tmp_path)
    ar = fx.load_capture(paths["ar"])
    mtp = fx.load_capture(paths["mtp"])
    report = fx.replay(ar, mtp, ext=MarkerExt(paths["rewind_marker"]), device="cpu")
    assert report["native_evidence"] is False, "a CPU spy must never be labelled native evidence"
    assert report["gate"]["passed"] is True
    assert report["gate"]["diff"]["max"] == 0.0
    assert report["status"] == "gate_passed:non_native_backend"
    assert report["checks"]["initial_state_equal"]["equal"] is True
    assert len(report["ar_native_steps"]) == 2
    assert report["mtp_native_vs_ar_actual"]["compared_at_position"] == 8
    assert "not established" in report["rounding"]

    # one ULP in the captured post state must fail the gate: no tolerance, no cast
    mutated = dict(mtp)
    mutated["records"] = [dict(r) for r in mtp["records"]]
    for rec in mutated["records"]:
        if rec["kind"] == "rewind":
            row = rec["after"]["canonical"]["rec_base"]["tensor"].clone()
            row.view(torch.int16)[0, 0, 0, 0] += 1        # 1 ULP of fp16
            rec["after"] = dict(rec["after"])
            rec["after"]["canonical"] = dict(rec["after"]["canonical"])
            rec["after"]["canonical"]["rec_base"] = {"tensor": row}
    bad = fx.replay(ar, mutated, ext=MarkerExt(paths["rewind_marker"]), device="cpu")
    assert bad["gate"]["passed"] is False
    assert bad["status"].startswith("gate_failed:inputs_or_fast_path")
    assert bad["gate"]["diff"]["n_diff"] == 1
    assert "not" in bad["diagnosis"] and "rounding" in bad["diagnosis"]


def test_gate_failure_writes_a_failed_artifact_and_claim_free_report(tmp_path, monkeypatch):
    paths = build_captures(tmp_path)
    out = tmp_path / "replay.json"
    # a spy whose output is not the captured state: the gate must fail, and say what that means
    monkeypatch.setattr(fx, "load_ext", lambda: MarkerExt(value=999.0))
    rc = fx.main(["--ar", paths["ar"], "--mtp", paths["mtp"], "--out", str(out), "--device", "cpu"])
    assert rc == 1
    report = json.loads(out.read_text())
    assert report["gate"]["passed"] is False
    assert report["gate"]["diff"]["n_diff"] > 0
    assert report["status"].startswith("gate_failed:inputs_or_fast_path")
    assert report["rounding"] == "not established by this tool"
    assert "do not attribute" in report["diagnosis"]
    assert report["checks"]["initial_state_equal"]["equal"] is True
    failed = json.loads((tmp_path / "replay.json.FAILED.json").read_text())
    assert failed["gate"]["passed"] is False and failed["ledger"]["accepted_prefix"] == 1


def test_native_backend_unavailable_is_a_hard_failure(tmp_path, monkeypatch):
    paths = build_captures(tmp_path)
    ar = fx.load_capture(paths["ar"])
    mtp = fx.load_capture(paths["mtp"])
    monkeypatch.setitem(sys.modules, "exllamav3.ext", None)      # the import now fails
    with pytest.raises(fx.FixtureError) as exc:
        fx.load_ext()
    assert "refusing to compare a float64 reference and call it native" in str(exc.value)
    with pytest.raises(fx.FixtureError):
        fx.replay(ar, mtp, device="cpu")            # no report object at all without a backend
    out = tmp_path / "no-native.json"
    rc = fx.main(["--ar", paths["ar"], "--mtp", paths["mtp"], "--out", str(out), "--device", "cpu"])
    assert rc == 2
    failed = json.loads((tmp_path / "no-native.json.FAILED.json").read_text())
    assert "no native conclusion" in failed["label"]
    assert not out.exists(), "a failed run must not leave a report that looks complete"


def test_probe_layer0_gate_is_mandatory_and_native_only(tmp_path):
    probe = _load("probe_mtp_causal", "probe_mtp_causal.py")
    args = types.SimpleNamespace(capture_dir=str(tmp_path), capture_device="cuda:0",
                                 capture_device_index=0, out=str(tmp_path / "p.json"))
    captures = {"gen_ar": {"info": {"json": str(tmp_path / "ar.json")}},
                "gen_mtp2": {"info": {"json": str(tmp_path / "mtp.json")}}}

    def fake_report(passed=True, native=True):
        return {"status": "gate_passed" if passed else "gate_failed:inputs_or_fast_path",
                "native_evidence": native, "device": "cuda:0", "ledger": {"accepted_prefix": 1},
                "checks": {}, "gate": {"passed": passed}, "mtp_native_vs_ar_actual": {},
                "operands_mtp_vs_ar": {}, "ar_native_steps": [], "reference": {},
                "rounding": "not established by this tool"}

    class FakeReplay:
        def __init__(self, report):
            self.report = report
            self.calls = []

        def load_capture(self, path):
            return {"path": path}

        def replay(self, ar, mtp, ext=None, device=None, device_index=None):
            self.calls.append({"ar": ar, "mtp": mtp, "ext": ext, "device": device,
                               "device_index": device_index})
            return self.report

    ok = FakeReplay(fake_report())
    summary = probe.layer0_gate(args, captures, ext="EXT", replay_module=ok)
    assert summary["gate"]["passed"] is True and summary["path"].endswith("layer0-replay.json")
    assert ok.calls[0]["ext"] == "EXT" and ok.calls[0]["device"] == "cuda:0"

    failed = FakeReplay(fake_report(passed=False))
    with pytest.raises(probe.ProbeError) as exc:
        probe.layer0_gate(args, captures, ext="EXT", replay_module=failed)
    assert "not a rounding result" in str(exc.value)
    assert os.path.exists(tmp_path / "layer0-replay.json")
    assert os.path.exists(tmp_path / "p.json.LAYER0-GATE-FAILED.json")

    non_native = FakeReplay(fake_report(passed=True, native=False))
    with pytest.raises(probe.ProbeError) as exc:
        probe.layer0_gate(args, captures, ext="EXT", replay_module=non_native)
    assert "not native numeric evidence" in str(exc.value)

    # a false hard prerequisite is its own hard failure, not a gate failure to be read as rounding
    blocked = FakeReplay(dict(fake_report(), prerequisites={"ok": False,
                                                            "failed": ["initial_state_equal"]}))
    with pytest.raises(probe.ProbeError) as exc:
        probe.layer0_gate(args, captures, ext="EXT", replay_module=blocked)
    assert "prerequisites failed" in str(exc.value) and "initial_state_equal" in str(exc.value)
    assert os.path.exists(tmp_path / "p.json.LAYER0-GATE-FAILED.json")


# ----------------------------------------------------------------------------- 4. CLI wiring

def test_capture_cli_dry_check_touches_nothing():
    assert cap.main(["--dry-check"]) == 0
    assert cap.main([]) == 2


def test_probe_exposes_the_capture_opt_in_without_changing_defaults():
    probe = _load("probe_mtp_causal", "probe_mtp_causal.py")
    assert probe.main(["--dry-check", "--index", "18"]) == 0
    with pytest.raises(SystemExit):
        probe.main(["--max-new", "4"])
    assert probe.main(["--dry-check", "--capture-dir", "/tmp/x", "--capture-positions", "7,8,9"]) == 0


def test_run_probe_passes_capture_flags_only_when_asked(tmp_path):
    """The wrapper must keep the reviewed argv identical unless a capture is requested, and must
    work for the retained ROCm7.13 runtime as well (that is the authorised future capture)."""
    exp = tmp_path / "exp"
    (exp / "ext-cache-instr" / "exllamav3_ext").mkdir(parents=True)
    (exp / "ext-cache-pristine" / "exllamav3_ext").mkdir(parents=True)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    argvlog = tmp_path / "argv.log"
    (venv / "python").write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > {argvlog}\nexit 0\n')
    os.chmod(venv / "python", 0o755)
    envscript = tmp_path / "env.sh"
    envscript.write_text("true\n")
    entry = tmp_path / "probe_mtp_causal.py"
    entry.write_text("# stub entry for the argv sink\n")
    base = dict(os.environ, ENV_SCRIPT=str(envscript), VENV=str(tmp_path / "venv"),
                FORK=str(tmp_path / "exl3-v164-rocm10-port"),
                TORCH_EXTENSIONS_DIR=str(exp / "ext-cache-instr"),
                PROBE_ENTRY=str(entry))
    base.pop("CAPTURE_DIR", None)
    base.pop("X3_PROBE_RUNTIME", None)
    r = subprocess.run([os.path.join(HERE, "run_probe.sh"), str(tmp_path / "p.json"),
                        str(tmp_path / "t"), "18"], env=base, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    argv = argvlog.read_text().split()
    assert "--capture-dir" not in argv, "default argv must not change"
    assert "--capture-positions" not in argv
    capdir = tmp_path / "cap"
    r = subprocess.run([os.path.join(HERE, "run_probe.sh"), str(tmp_path / "p.json"),
                        str(tmp_path / "t"), "18"],
                       env=dict(base, CAPTURE_DIR=str(capdir), CAPTURE_POSITIONS="7,8,9"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    argv = argvlog.read_text().split()
    assert argv[argv.index("--capture-dir") + 1] == str(capdir)
    assert argv[argv.index("--capture-positions") + 1] == "7,8,9"
    assert capdir.is_dir(), "the wrapper must create the capture root it is given"
    # the retained ROCm7.13 pairing the proposed capture uses
    argvlog.unlink()
    r = subprocess.run([os.path.join(HERE, "run_probe.sh"), str(tmp_path / "p.json")],
                       env=dict(base, X3_PROBE_RUNTIME="rocm713",
                                FORK=str(tmp_path / "exl3-v164-pristine"),
                                TORCH_EXTENSIONS_DIR=str(exp / "ext-cache-pristine"),
                                CAPTURE_DIR=str(capdir)),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "runtime=rocm713" in r.stderr
    assert "--capture-dir" in argvlog.read_text().split()
