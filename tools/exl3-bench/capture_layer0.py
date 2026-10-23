#!/usr/bin/env python3
"""Read-only layer-0 GDN capture around the first MTP verification: operands, state and ledger.

Purpose: collect the inputs and the state transition needed to explain why the verifier's recurrent
state at the accepted frontier differs from AR's, **without changing arithmetic**. Nothing here is on
the inference path: every hook clones live values, delegates unchanged and returns the original
object.

Source contract (verified against exp/rocm10-v1.6.4, SHA 5b0ba36db90f25eb514f0f5602f50ccd406f8fd0;
file modules/gated_delta_net.py):

* `GDNState` (116) owns `slot`, `position`, `last_history`, `parity`, `spec_row`, `spec_shape`,
  `base_row()` (169) and `scratch_row()` (173). `rewind(num_tokens)` (177) derives
  `prefix = last_history + 1 - num_tokens`, replays that prefix and flips `parity`.
* `GDNLayerState` (259) owns `conv_state (max_batch, fdim_qkv, K + max_history)` bf16,
  `recurrent_state (2*max_batch, 1, nv, dk, dv)` fp16/fp32, and the staging buffers
  `spec_conv (rows, fdim_qkv)` bf16, `spec_beta (rows, nv)` bf16, `spec_g (rows, g_width)` fp32
  (`g_width = nv` for GDN, `nv*k_head_dim` for KDA) that a speculative (history) pass writes and the
  rewind replays. `staged_views`/`staged_step_views` (369/
  384) *set* `spec_shape`, so the capture reads `spec_shape` and never calls them.
* `GatedDeltaNet.forward(self, x, params, out_dtype=None)` (1213). The fused branch (1304-1322) is
  taken when `bc_split and save_state and recurrent_slots is not None and 1 <= bsz <= _BC_MAX_BSZ and
  1 <= seqlen <= _BC_MAX_QLEN and not spec_step`; it stages the pass shape and calls
  `self.bc.run_bszN(...)`. The scan operands of that branch are **Python-visible**, because
  `_bc_configure_slot` (1178) hands exactly these tensor objects to C++:
    - `history=True` (1201-1204): `conv_out, beta, g = rsl.staged_views(bsz, seqlen)`, i.e. views
      over `rsl.spec_conv/spec_beta/spec_g`;
    - `history=False` (1205-1208): `s_beta` bf16, `s_g` fp32, `s_conv` bf16 from `g_tensor_cache`,
      tagged `"s_beta"`, `"s_g"`, `"s_conv"` (plus `"s_qkv"`, `"s_z"`, `"s_ba"` fp32 projections).
* `g_tensor_cache` (util/tensor.py:272) is shared by every layer: the AR entries hold layer 0's
  values only until the next layer overwrites them, so an AR record must be cloned immediately after
  layer 0's forward returns. `GTensorCache.make_key` (214) is pure and `cache[key]` is a plain dict
  entry; the capture reads the **existing** entry and never calls `GTensorCache.get()`, which would
  allocate and increment a refcount.

Excluded on purpose, and counted rather than captured: draft token-by-token passes
(`recurrent_spec_step`), the torch path (its scan operands are local variables, so no reliable hook
exists; see the module note in the report), other layers, other caches and other positions. A missing
required operand is a diagnostic failure (`self.errors`), and `assert_complete()` raises after the
artifacts are written - the model run itself is never interrupted by the diagnostic.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import torch

SCHEMA = "layer0-gdn-capture/2"
DEFAULT_POSITIONS = (7, 8, 9)      # pass-entry positions of the first verifier transition
DEFAULT_LAYER = 0
MAX_FORWARDS = 6                   # bounded: one entry per captured position, plus slack
MAX_REWINDS = 2                    # 2 so a rewind whose pass was not captured can be skipped first


class CaptureError(RuntimeError):
    """A required piece of evidence is absent; the partial capture is still written."""


# ----------------------------------------------------------------------------- layout helpers

def _layout(t: torch.Tensor) -> dict:
    """Source layout facts, read before any clone so a changed clone layout cannot hide them."""
    return {"shape": list(t.shape), "dtype": str(t.dtype), "stride": list(t.stride()),
            "storage_offset": int(t.storage_offset()), "device": str(t.device),
            "contiguous": bool(t.is_contiguous()), "numel": int(t.numel())}


def _clone(t: torch.Tensor | None, rows=None) -> dict | None:
    """Detached host copy of the live values plus their *source* layout.

    `rows` restricts the copy to a slice of dim 0 (batch rows of a staging region); the slice bounds
    are recorded by the caller. Clone, never a view: later passes overwrite these buffers.
    """
    if t is None:
        return None
    src = t if rows is None else t[:rows]
    return {"layout": _layout(src), "tensor": src.detach().to("cpu", copy=True)}


def _meta(state) -> dict:
    if state is None:
        return {}
    out = {}
    for name in ("slot", "position", "last_history", "parity"):
        if hasattr(state, name):
            out[name] = int(getattr(state, name))
    for name in ("spec_row",):
        if hasattr(state, name):
            v = getattr(state, name)
            out[name] = None if v is None else int(v)
    if hasattr(state, "spec_shape"):
        v = state.spec_shape
        out["spec_shape"] = list(v) if isinstance(v, (tuple, list)) else v
    for name in ("base_row", "scratch_row"):
        fn = getattr(state, name, None)
        if callable(fn):
            out[name] = int(fn())
    out["class"] = type(state).__name__
    return out


def _params_meta(params: dict) -> dict:
    out = {}
    for key in ("cache_seqlens", "positions"):
        t = params.get(key)
        out[key] = [int(x) for x in t.reshape(-1).tolist()] if isinstance(t, torch.Tensor) else None
    for key in ("recurrent_history", "last_tokens_only", "recurrent_spec_step", "layer_instance"):
        out[key] = params.get(key)
    for key in ("recurrent_slots", "recurrent_slots_scan", "recurrent_slots_scan_in"):
        t = params.get(key)
        out[key] = None if not isinstance(t, torch.Tensor) else [int(x) for x in t.reshape(-1).tolist()]
    return out


def _jsonable(node, tensors: dict, path: str):
    """Replace every captured tensor by a path reference; the tensors dict holds the payload."""
    if isinstance(node, dict):
        if "tensor" in node and isinstance(node["tensor"], torch.Tensor):
            ref = f"{path}.tensor"
            tensors[ref] = node["tensor"]
            return {k: ({"tensor_ref": ref} if k == "tensor" else _jsonable(v, tensors, f"{path}.{k}"))
                    for k, v in node.items()}
        return {k: _jsonable(v, tensors, f"{path}.{k}" if path else k) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_jsonable(v, tensors, f"{path}[{i}]") for i, v in enumerate(node)]
    if isinstance(node, torch.Tensor):
        raise TypeError(f"unwrapped tensor at {path}: captures must go through _clone()")
    if isinstance(node, (str, int, float, bool)) or node is None:
        return node
    return str(node)


def _restore(node, tensors: dict):
    if isinstance(node, dict):
        if set(node) == {"tensor_ref"}:
            ref = node["tensor_ref"]
            if ref not in tensors:
                raise CaptureError(f"capture artifact is missing tensor {ref!r}")
            return tensors[ref]
        return {k: _restore(v, tensors) for k, v in node.items()}
    if isinstance(node, list):
        return [_restore(v, tensors) for v in node]
    return node


def load_capture(json_path: str, pt_path: str | None = None) -> dict:
    """Read one capture artifact back into the same nested structure the hooks produced."""
    with open(json_path) as f:
        doc = json.load(f)
    if doc.get("schema") != SCHEMA:
        raise CaptureError(f"{json_path}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    pt_path = pt_path or os.path.splitext(json_path)[0] + ".pt"
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    if blob.get("schema") != SCHEMA:
        raise CaptureError(f"{pt_path}: schema {blob.get('schema')!r}, expected {SCHEMA!r}")
    tensors = blob["tensors"]
    doc["records"] = [_restore(rec, tensors) for rec in doc["records"]]
    doc["tensors"] = tensors
    return doc


# ----------------------------------------------------------------------------- the capture

class Layer0Capture:
    """Bounded hooks on the three existing seams; writes `<out_dir>/<name>.json` + `.<name>.pt`."""

    def __init__(self, out_dir: str, mode: str, cache_id: int | None = None,
                 layer: int = DEFAULT_LAYER, positions=DEFAULT_POSITIONS,
                 max_forwards: int = MAX_FORWARDS, max_rewinds: int = MAX_REWINDS):
        if mode not in ("ar", "mtp"):
            raise CaptureError(f"mode must be 'ar' or 'mtp', got {mode!r}")
        self.out_dir = out_dir
        self.mode = mode
        self.cache_id = None if cache_id is None else int(cache_id)
        self.layer = int(layer)
        self.positions = tuple(int(p) for p in positions)
        self.max_forwards = int(max_forwards)
        self.max_rewinds = int(max_rewinds)
        self.records: list[dict] = []
        self.errors: list[str] = []
        self.excluded = Counter()
        self.overflow = False
        self.installed = False
        self._orig = {}
        self._limits = None
        self._layer_instance = 0
        self._active_rewind = None      # set only between the selected rewind's before/after snapshot
        self._active_rewind_emitted = False

    # ------------------------------------------------------------------ bookkeeping
    def _error(self, msg: str) -> None:
        self.errors.append(msg)

    def completeness_errors(self) -> list[str]:
        """What this capture mode needs, checked against the records actually taken."""
        out = []
        if self.mode == "mtp":
            verify = self.forwards("target_verify")
            rewinds = self.rewinds()
            if len(verify) != 1:
                out.append(f"mtp capture holds {len(verify)} verifier passes; exactly one is needed "
                           f"(positions {list(self.positions)}, excluded {dict(self.excluded)})")
            if len(rewinds) != 1:
                out.append(f"mtp capture holds {len(rewinds)} rewinds; exactly one accepted-prefix "
                           f"transition is needed")
            if not any(r["kind"] == "replay_job" for r in self.records):
                out.append("mtp capture holds no layer-0 replay_job record: the library's own replay "
                           "geometry is missing")
            jobs = [r for r in self.records if r["kind"] == "replay_job"]
            if jobs and rewinds and int(jobs[0].get("pass_entry", -1)) != int(rewinds[0]["pass_entry"]):
                out.append(f"the captured replay_job belongs to pass entry "
                           f"{jobs[0].get('pass_entry')}, the captured rewind to "
                           f"{rewinds[0]['pass_entry']}")
            if len(jobs) > 1:
                out.append(f"mtp capture holds {len(jobs)} replay_job records; exactly one selected "
                           f"transition is expected")
            if verify and rewinds and int(rewinds[0]["pass_entry"]) != int(verify[0]["position"]):
                out.append(f"the captured rewind belongs to pass entry {rewinds[0]['pass_entry']}, "
                           f"the captured verifier pass is at {verify[0]['position']}")
        else:
            decode = self.forwards("target_decode")
            if len(decode) < 2:
                out.append(f"ar capture holds {len(decode)} decode passes; at least two adjacent "
                           f"passes are needed for a one-step comparison")
        return out

    def assert_complete(self) -> None:
        """Called by the probe *after* the artifacts are written: fail loudly, keep the partials."""
        if self.overflow:
            raise CaptureError(f"capture overflow: record cap reached "
                               f"(forwards>={self.max_forwards} or rewinds>={self.max_rewinds}); "
                               f"partial capture in {self.out_dir}")
        problems = self.errors + self.completeness_errors()
        if problems:
            raise CaptureError("capture is incomplete:\n  " + "\n  ".join(problems))

    def _emit(self, kind: str, **fields) -> dict | None:
        if len(self.records) >= self.max_forwards + self.max_rewinds:
            self.overflow = True
            return None
        rec = {"schema": SCHEMA, "kind": kind, "seq": len(self.records),
               "capture_mode": self.mode, "layer_idx": self.layer,
               "cache_id": self.cache_id, **fields}
        self.records.append(rec)
        return rec

    def forwards(self, mode: str | None = None) -> list[dict]:
        return [r for r in self.records if r["kind"] == "layer0_forward"
                and (mode is None or r.get("mode") == mode)]

    def rewinds(self) -> list[dict]:
        return [r for r in self.records if r["kind"] == "rewind"]

    # ------------------------------------------------------------------ classification
    def classify(self, module, x, params) -> tuple[dict | None, str | None]:
        """The real fused-branch predicate (source 1304-1313), applied before any cloning.

        Returns `(info, None)` for a captured pass or `(None, reason)` for an excluded one. The
        predicate is evaluated from the same attributes the module reads, so a pass that would not
        enter the fused branch is never recorded as if it had.
        """
        if int(getattr(module, "layer_idx", -1)) != self.layer:
            return None, "other_layer"
        states = params.get("recurrent_states") or []
        if not states:
            return None, "no_state"
        seq = states[0]
        cache = getattr(seq, "cache", None)
        if cache is None:
            return None, "no_cache"
        if self.cache_id is not None and id(cache) != self.cache_id:
            return None, "other_cache"
        if int(getattr(module, "num_k_heads", 1)) == 0:
            return None, "no_heads"
        history = bool(params.get("recurrent_history", False))
        spec_step = params.get("recurrent_spec_step") if history else None
        if spec_step is not None:
            return None, "draft_step"
        if getattr(module, "kda", False):
            return None, "kda"
        if not bool(getattr(module, "bc_split", False)):
            return None, "not_bc_split"
        if params.get("recurrent_slots") is None:
            return None, "no_slots"
        bsz, seqlen = int(x.shape[0]), int(x.shape[1])
        if not (1 <= bsz <= self._limits[0] and 1 <= seqlen <= self._limits[1]):
            return None, "outside_bc_shape"
        position = int(seq.position)
        if position not in self.positions:
            return None, "outside_window"
        mode = "target_verify" if history else "target_decode"
        if mode == "target_verify" and self.mode != "mtp":
            return None, "verify_in_ar_mode"
        if mode == "target_verify" and len(self.forwards("target_verify")) >= 1:
            return None, "second_verify_pass"
        return ({"mode": mode, "history": history, "bsz": bsz, "seqlen": seqlen,
                 "position": position, "cache_id": id(cache), "seq": seq}, None)

    # ------------------------------------------------------------------ state views
    def _resolve_rsl(self, module, seq):
        """The same layer-state lookup the real forward performs (source 1262-1268)."""
        if getattr(seq, "exported", False):
            return module.tp_recurrent_lookup[seq.cache]
        return seq.cache.get_recurrent_layer((self.layer, self._layer_instance))

    def _geometry(self, rsl) -> dict:
        kind, nk, nv, dk, dv, state_half = rsl.scan_geometry()
        return {"kind": int(kind), "nk": int(nk), "nv": int(nv), "dk": int(dk), "dv": int(dv),
                "state_half": bool(state_half), "state_dtype": str(rsl.recurrent_state.dtype)}

    def _canonical(self, rsl, seq) -> dict:
        conv, rec = rsl.get_state_tensors()
        K = int(rsl.module.conv_kernel_size)
        slot = int(seq.slot)
        # Committed conv history is columns [0, K) (conv1d.py:373-392): a committing pass copies the
        # last K tokens there, a speculative pass writes the ring tail and the rewind copies back.
        return {"conv_committed": _clone(conv[slot][:, :K]),
                "conv_ring": _clone(conv[slot]),
                "rec_base": _clone(rec[int(seq.base_row())]),
                "rec_scratch": _clone(rec[int(seq.scratch_row())]),
                "base_row": int(seq.base_row()), "scratch_row": int(seq.scratch_row()),
                "K": K, "conv_cols": int(conv.shape[-1])}

    def _shared(self, env, device, shape, dtype, tag: str, required: bool) -> dict | None:
        """Peek an EXISTING g_tensor_cache entry. Pure: `make_key` + dict lookup, never `.get()`."""
        key = env.make_key(device, shape, dtype, tag)
        if key not in env.cache:
            if required:
                self._error(f"g_tensor_cache has no entry {key!r} (required scan operand)")
            return None
        refc, tensor = env.cache[key]
        return {"key": key, "refcount": int(refc), "tag": tag, "dtype": str(dtype),
                "shape": list(shape), "value": _clone(tensor)}

    def _ar_operands(self, module, bsz, seqlen, env) -> dict:
        f = int(module.fdim_qkv)
        nv = int(module.num_v_heads)
        hv = int(module.v_head_dim)
        device = module.device
        ops = {
            "source": "g_tensor_cache",
            "conv": self._shared(env, device, (bsz, seqlen, f), torch.bfloat16, "s_conv", True),
            "beta": self._shared(env, device, (bsz, seqlen, nv), torch.bfloat16, "s_beta", True),
            "g": self._shared(env, device, (bsz, seqlen, nv), torch.float, "s_g", True),
            "aux": {
                "qkv": self._shared(env, device, (bsz, seqlen, f), torch.float, "s_qkv", False),
                "ba": self._shared(env, device, (bsz, seqlen, 2 * nv), torch.float, "s_ba", False),
                "z": self._shared(env, device, (bsz, seqlen, nv, hv), torch.float, "s_z", False),
            },
        }
        return ops

    def _mtp_operands(self, rsl, bsz, seqlen, spec_shape=None) -> dict:
        """The staging buffers a history pass wrote: `rsl.spec_conv/spec_beta/spec_g` (source 1201)."""
        n_rows = int(bsz) * int(seqlen)
        return {"source": "GDNLayerState.spec_conv/spec_beta/spec_g",
                "rows": n_rows,
                "spec_shape": (list(spec_shape) if spec_shape is not None
                               else (list(rsl.spec_shape) if rsl.spec_shape else None)),
                "conv": {"key": "rsl.spec_conv", "value": _clone(rsl.spec_conv, rows=n_rows)},
                "beta": {"key": "rsl.spec_beta", "value": _clone(rsl.spec_beta, rows=n_rows)},
                "g": {"key": "rsl.spec_g", "value": _clone(rsl.spec_g, rows=n_rows)}}

    # ------------------------------------------------------------------ hooks
    def install(self, module_cls, state_cls, layer_state_cls, tensor_module=None):
        """Patch `GatedDeltaNet.forward`, `GDNState.rewind` and `GDNLayerState.replay_job`.

        `tensor_module` supplies the real `g_tensor_cache`; it is looked up in the forward's own
        globals when omitted, so the capture always peeks the cache the module actually uses.
        """
        if self.installed:
            raise CaptureError("capture already installed")
        orig_forward, orig_rewind, orig_replay = (
            module_cls.forward, state_cls.rewind, layer_state_cls.replay_job)
        g = orig_forward.__globals__
        env = tensor_module or g.get("g_tensor_cache")
        if env is None:
            raise CaptureError("cannot resolve g_tensor_cache from the forward's globals")
        self._limits = (int(g["_BC_MAX_BSZ"]), int(g["_BC_MAX_QLEN"]))
        self._orig = {"forward": orig_forward, "rewind": orig_rewind, "replay_job": orig_replay}
        self._env = env

        def forward(self, x, params, out_dtype=None):
            info, reason = cap.classify(self, x, params)
            if info is None:
                cap.excluded[reason] += 1
                return orig_forward(self, x, params, out_dtype)
            if len(cap.forwards()) >= cap.max_forwards:
                cap.overflow = True
                return orig_forward(self, x, params, out_dtype)
            seq = info["seq"]
            # A diagnostic must never change the model's behaviour: a snapshot failure is recorded and
            # surfaces in assert_complete() after the run, and the call is delegated unchanged.
            before = None
            try:
                cap._layer_instance = int(params.get("layer_instance", 0) or 0)
                rsl = cap._resolve_rsl(self, seq)
                before = {"state": _meta(seq), "pass": _params_meta(params),
                          "canonical": cap._canonical(rsl, seq),
                          "geometry": cap._geometry(rsl)}
                # x is (bsz, seqlen, hidden): the sequence axis is dim 1. Slice the sequence, never
                # the batch, and keep the source layout of the whole tensor next to the bounds.
                keep = min(3, int(x.shape[1]))
                before["x"] = {"source_layout": _layout(x), "seq_slice": [0, keep],
                               "value": _clone(x[:, :keep])}
            except Exception as exc:                      # noqa: BLE001 - recorded, then surfaced
                cap._error(f"layer 0 forward at position {info['position']}: pre-pass snapshot "
                           f"failed ({type(exc).__name__}: {exc})")
            out = orig_forward(self, x, params, out_dtype)
            if before is None:
                return out
            try:
                # Layer 0 has returned and no later layer has run yet: the only instant at which
                # both the AR shared statics and the MTP staging buffers hold this pass's operands.
                if info["history"]:
                    operands = cap._mtp_operands(rsl, info["bsz"], info["seqlen"])
                else:
                    operands = cap._ar_operands(self, info["bsz"], info["seqlen"], env)
                if operands["conv"] is None or operands["beta"] is None or operands["g"] is None:
                    cap._error(f"{info['mode']} at position {info['position']}: scan operands "
                               f"missing from {operands['source']}; no record written")
                    cap.excluded[f"missing_operands:{info['mode']}"] += 1
                    return out
                after = {"state": _meta(seq), "canonical": cap._canonical(rsl, seq)}
                rec = cap._emit("layer0_forward", mode=info["mode"], bsz=info["bsz"],
                                seqlen=info["seqlen"], position=info["position"],
                                history=info["history"], before=before, after=after,
                                operands=operands,
                                d_skip=({"present": False}
                                        if getattr(self, "d_skip_f", None) is None
                                        else {"present": True, "value": _clone(self.d_skip_f)}))
                if rec is not None:
                    rec["fused_predicate"] = {
                        "bc_split": bool(getattr(self, "bc_split", False)), "save_state": True,
                        "recurrent_slots": True, "bsz": info["bsz"], "seqlen": info["seqlen"],
                        "limits": list(cap._limits), "spec_step": None}
            except Exception as exc:                      # noqa: BLE001
                cap._error(f"layer 0 forward at position {info['position']}: post-pass snapshot "
                           f"failed ({type(exc).__name__}: {exc})")
            return out

        def rewind(self, num_tokens: int):
            seq = self
            cache = getattr(seq, "cache", None)
            if cache is None or cap.cache_id is None or id(cache) != cap.cache_id:
                cap.excluded["rewind_other_cache"] += 1
                return orig_rewind(self, num_tokens)
            if cap.mode != "mtp":
                cap.excluded["rewind_in_ar_mode"] += 1
                return orig_rewind(self, num_tokens)
            try:
                position_before = int(seq.position)
                last_history = int(seq.last_history)
                tokens = int(num_tokens)
            except Exception as exc:                      # noqa: BLE001
                cap._error(f"rewind: cannot read the sequence metadata ({type(exc).__name__}: {exc})")
                return orig_rewind(self, num_tokens)
            entry = position_before - (last_history + 1)
            if last_history <= 0 or tokens <= 0:
                cap.excluded["rewind_no_replay"] += 1
                return orig_rewind(self, num_tokens)
            if entry not in cap.positions or (position_before - tokens) not in cap.positions:
                cap.excluded["rewind_outside_window"] += 1
                return orig_rewind(self, num_tokens)
            if len(cap.rewinds()) >= cap.max_rewinds:
                cap.excluded["rewind_cap"] += 1
                return orig_rewind(self, num_tokens)
            forward = [r for r in cap.forwards("target_verify") if r["position"] == entry]
            if not forward:
                # This rewind settles a pass the capture did not record; skip it rather than pairing
                # the ledger with someone else's operands.
                cap.excluded["rewind_without_verifier_forward"] += 1
                return orig_rewind(self, num_tokens)
            prefix = last_history + 1 - tokens              # source: _rewind_layers (76-84)
            parity_before = int(seq.parity)
            before = None
            ledger_for_job = None
            try:
                rsl = cache.get_recurrent_layer((cap.layer, cap._layer_instance))
                spec_shape = seq.spec_shape
                tstride = int(spec_shape[2]) if spec_shape else None
                n_staged = int(spec_shape[1]) if spec_shape else 0
                r0 = (int(seq.spec_row) * n_staged
                      if spec_shape and tstride == 1 else int(seq.spec_row or 0))
                mtp_ops = cap._mtp_operands(rsl, int(spec_shape[0]) if spec_shape else 1, n_staged,
                                            spec_shape=spec_shape)
                # The row block the library's own replay_job points at (r0 = spec_row * seqlen for a
                # row-major pass). The replay tool slices exactly this block.
                mtp_ops["row_offset"] = r0
                mtp_ops["row_block"] = {
                    name: {"key": f"rsl.{name}[{r0}:{r0 + n_staged}]",
                           "value": _clone(getattr(rsl, name)[r0:r0 + n_staged])}
                    for name in ("spec_conv", "spec_beta", "spec_g")}
                ledger_for_job = {"entry": entry, "prefix": prefix,
                                  "spec_shape": [int(v) for v in spec_shape] if spec_shape else None,
                                  "base_row": int(seq.base_row()),
                                  "scratch_row": int(seq.scratch_row())}
                before = {"state": _meta(seq), "canonical": cap._canonical(rsl, seq),
                          "num_tokens": tokens, "last_history": last_history,
                          "pass_entry": entry, "prefix": prefix,
                          "spec_shape": list(spec_shape) if spec_shape else None,
                          "spec_row": seq.spec_row, "base_row": int(seq.base_row()),
                          "scratch_row": int(seq.scratch_row()), "operands": mtp_ops,
                          "operand_geometry": cap._geometry(rsl)}
            except Exception as exc:                      # noqa: BLE001
                cap._error(f"rewind at pass entry {entry}: pre-rewind snapshot failed "
                           f"({type(exc).__name__}: {exc})")
            # `replay_job` is called by the library inside this rewind only; the flag makes every
            # later job on the same layer/cache countable as an exclusion instead of a record.
            cap._active_rewind = ledger_for_job if before is not None else None
            try:
                out = orig_rewind(self, num_tokens)
            finally:
                cap._active_rewind = None
            if before is None:
                return out
            try:
                after = {"state": _meta(seq), "canonical": cap._canonical(rsl, seq)}
                cap._emit("rewind", position_before=position_before,
                          position_after=int(seq.position), num_tokens=tokens,
                          last_history=last_history, prefix=prefix, pass_entry=entry,
                          ledger={"staged_tokens": n_staged, "accepted_prefix": prefix,
                                  "rejected_tail": n_staged - prefix,
                                  "spec_row": before["spec_row"],
                                  "spec_shape": before["spec_shape"],
                                  "base_row": before["base_row"],
                                  "scratch_row": before["scratch_row"],
                                  "parity_before": parity_before,
                                  "parity_after": int(after["state"].get("parity", -1))},
                          before=before, after=after)
            except Exception as exc:                      # noqa: BLE001
                cap._error(f"rewind at pass entry {entry}: post-rewind snapshot failed "
                           f"({type(exc).__name__}: {exc})")
            return out

        def replay_job(self, spec_row, prefix, spec_shape, base_row, scratch_row):
            active = cap._active_rewind
            if int(getattr(self.module, "layer_idx", -1)) != cap.layer or (
                    cap.cache_id is None or int(getattr(self, "cache_id", -1)) != cap.cache_id):
                cap.excluded["replay_job_other_layer_or_cache"] += 1
                return orig_replay(self, spec_row, prefix, spec_shape, base_row, scratch_row)
            if active is None:
                # A later verifier/rewind call on this same layer and cache: outside the transition
                # the capture selected, so it can neither be recorded nor exhaust the record cap.
                cap.excluded["replay_job_outside_rewind"] += 1
                return orig_replay(self, spec_row, prefix, spec_shape, base_row, scratch_row)
            bsz, seqlen, tstride = (int(v) for v in spec_shape)
            job_shape = [bsz, seqlen, tstride]
            if (int(prefix) != active["prefix"] or job_shape != active["spec_shape"]
                    or int(base_row) != active["base_row"]
                    or int(scratch_row) != active["scratch_row"]):
                # The job does not describe the rewind being captured: recording it would let the
                # fixture inherit a geometry the selected ledger never used.
                cap.excluded["replay_job_ledger_mismatch"] += 1
                return orig_replay(self, spec_row, prefix, spec_shape, base_row, scratch_row)
            # Record the library's own job geometry, so the replay tool cannot invent it: `r0`
            # is the staging row the captured block must be sliced from (replay_job:434-458).
            r0 = spec_row * seqlen if tstride == 1 else spec_row
            cap._emit("replay_job", spec_row=int(spec_row), prefix=int(prefix),
                      spec_shape=job_shape, base_row=int(base_row),
                      scratch_row=int(scratch_row), tstride=tstride, r0=int(r0),
                      staging_strides={
                          "conv_row": int(self.spec_conv.stride(0)),
                          "conv_elem": int(self.spec_conv.element_size()),
                          "g_row": int(self.spec_g.stride(0)),
                          "g_elem": int(self.spec_g.element_size()),
                          "beta_row": int(self.spec_beta.stride(0)),
                          "beta_elem": int(self.spec_beta.element_size()),
                          "state_row_bytes": int(self.recurrent_state.stride(0) *
                                                 self.recurrent_state.element_size())},
                      d_skip=("absent" if getattr(self.module, "d_skip_f", None) is None
                              else "present"),
                      scan_geometry=list(self.scan_geometry()),
                      pass_entry=active["entry"])
            return orig_replay(self, spec_row, prefix, spec_shape, base_row, scratch_row)

        cap = self
        module_cls.forward, state_cls.rewind, layer_state_cls.replay_job = forward, rewind, replay_job
        self.installed = True
        return self

    def uninstall(self, module_cls, state_cls, layer_state_cls):
        if not self.installed:
            return
        module_cls.forward = self._orig["forward"]
        state_cls.rewind = self._orig["rewind"]
        layer_state_cls.replay_job = self._orig["replay_job"]
        self.installed = False

    # ------------------------------------------------------------------ persistence
    def write(self, name: str = "layer0-capture") -> dict:
        """Write `<out_dir>/<name>.json` (structure + tensor references) and `<name>.pt` (payloads)."""
        os.makedirs(self.out_dir, exist_ok=True)
        tensors: dict = {}
        records = [_jsonable({k: v for k, v in rec.items()}, tensors, f"r{rec['seq']}")
                   for rec in self.records]
        doc = {"schema": SCHEMA, "capture_mode": self.mode, "cache_id": self.cache_id,
               "layer_idx": self.layer, "positions": list(self.positions),
               "overflow": self.overflow, "errors": list(self.errors),
               "excluded": dict(self.excluded), "limits": list(self._limits or ()),
               "records": records}
        json_path = os.path.join(self.out_dir, f"{name}.json")
        pt_path = os.path.join(self.out_dir, f"{name}.pt")
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=1)
        torch.save({"schema": SCHEMA, "tensors": tensors}, pt_path)
        return {"json": json_path, "pt": pt_path, "records": len(self.records),
                "tensors": len(tensors), "overflow": self.overflow, "errors": list(self.errors),
                "excluded": dict(self.excluded)}

    def summary(self) -> dict:
        return {"mode": self.mode, "cache_id": self.cache_id, "layer": self.layer,
                "positions": list(self.positions), "records": len(self.records),
                "forwards": [{"mode": r["mode"], "position": r["position"], "seqlen": r["seqlen"],
                              "operands": r["operands"]["source"]} for r in self.forwards()],
                "rewinds": [{"pass_entry": r["pass_entry"], "prefix": r["prefix"],
                             "num_tokens": r["num_tokens"]} for r in self.rewinds()],
                "excluded": dict(self.excluded), "overflow": self.overflow,
                "errors": list(self.errors), "incomplete": self.completeness_errors()}


def dry_check() -> dict:
    """Static description of the seams; never touches a model, a cache or a device."""
    return {"schema": SCHEMA, "ok": True,
            "seams": ["GatedDeltaNet.forward(self, x, params, out_dtype=None)",
                      "GDNState.rewind(num_tokens)", "GDNLayerState.replay_job(...)"],
            "scan_operand_sources": {"mtp": "GDNLayerState.spec_conv/spec_beta/spec_g",
                                     "ar": "g_tensor_cache s_conv/s_beta/s_g (peeked, never .get())"},
            "positions": list(DEFAULT_POSITIONS), "layer": DEFAULT_LAYER,
            "max_forwards": MAX_FORWARDS, "max_rewinds": MAX_REWINDS,
            "required": {"mtp": "one verifier pass + one rewind + one layer-0 replay_job",
                         "ar": "at least two adjacent single-token decode passes"},
            "note": "no model, no cache, no device touched"}


def main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="read-only layer-0 GDN capture")
    ap.add_argument("--dry-check", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_check:
        print(json.dumps(dry_check()))
        return 0
    print("capture_layer0 is a library: the probe installs it (--capture-dir)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
