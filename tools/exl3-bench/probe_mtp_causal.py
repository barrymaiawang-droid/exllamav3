#!/usr/bin/env python3
"""Bounded causal probe for the AR vs MTP verifier divergence at a chosen generated index (greedy).

Every number here is derived from the *real* Generator's own target protocol, which is recorded and
then replayed into independent control caches; no attention/prefill parameters are hand-built.

Source facts this module relies on (v1.6.4 = da68d57d):

  * `Job.prefill` stops one token short of the end (`prefill_end = min(..., len(sequence_ids) - 1)`,
    generator/job.py:1417), so prefill consumes P-1 prompt tokens and the last prompt token is the
    first decode row.
  * `Generator.iterate_gen` (generator/generator.py:1508-1523) passes `cache_seqlens` (= the
    sequence's `kv_position`), `block_table`, `positions` and `recurrent_states`; there is no
    `past_len` parameter on this path.
  * `Job.get_input_ids_list` (generator/job.py:390) returns `sequence_ids[kv_position:]`, so the rows
    of a decode pass are exactly the tokens not yet in the cache, plus the drafter's proposals.
  * With `cache_seqlens = past` and `row = r`, the pass predicts generated index
    `i = past + r + 1 - P`. Row 0 of a pass that starts at `past = P-1+k` predicts index k.
  * `GDNLayerState.base_row() = 2*slot + parity`, `scratch_row() = 2*slot + 1 - parity`, and
    `rewind()` flips `parity` after moving the committed state into the scratch row
    (modules/gated_delta_net.py:170-200). The canonical live state is therefore always `base_row()`;
    the scratch row is a transient copy and is reported separately.

Controls (all at the same context, on the same loaded target model):

  seq     : the recorded AR protocol replayed verbatim into a fresh cache and state.
  batch3  : the recorded protocol replayed up to (not including) the selected pass, then the same
            pass re-issued with 3 rows `[row0, filler, filler]`. Row 0 is causal and unaffected by
            the fillers, so this isolates batched arithmetic from any speculative state.

What this bounded probe cannot establish is stated in the report, not glossed over: a single
3-row step does not model accumulated arithmetic over a whole speculative history, and for a
selected row interior to a verifier batch the *row-entry* state is not the AR row-entry state, so
only the pass-entry (common) frontier is compared.

Usage: probe_mtp_causal.py --out probe.json [--index 18] [--tensor-dir DIR] [--dry-check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

PAGE_SIZE = 256          # exllamav3/constants.py:2
MAX_HISTORY = 3          # cache max_history for a 3-row pass (max_history + 1 rows)


# ----------------------------------------------------------------------------- errors

class ProbeError(RuntimeError):
    """Base class: every probe failure is explicit, never a skipped comparison."""


class ParamError(ProbeError):
    """The recorded call does not carry the parameters the real generator supplies."""


class ContextError(ProbeError):
    """Recorded rows do not match the prompt+generated context they claim to be."""


class SelectionError(ProbeError):
    """No target pass/row predicts the requested index, or the state positions differ."""


class SnapshotError(ProbeError):
    """A recurrent-state snapshot is missing, empty, or was stripped before comparison."""


# ----------------------------------------------------------------------------- pure logic

def predict_index(cache_seqlen: int, row: int, prompt_len: int) -> int:
    """Generated index predicted by (pass starting at cache_seqlen, row)."""
    return cache_seqlen + row + 1 - prompt_len


def expected_context(prompt_ids, generated_ids, index: int):
    """The exact token context that must precede generated index `index`."""
    if index < 0 or index >= len(generated_ids):
        raise ContextError(f"index {index} outside the generated range 0..{len(generated_ids) - 1}")
    return list(prompt_ids) + list(generated_ids[:index])


def select_pass(calls, index: int, prompt_len: int, prompt_ids=None, generated_ids=None):
    """The target call/row that actually produced generated `index`, on the accepted context.

    A pass can cover `index` while that row was a *rejected* proposal (the generator then discards it
    and re-runs the position). Those candidates are returned in `discarded` for the report and never
    selected: a candidate is only valid when its accepted columns match
    `prompt + generated[:index]`, which requires `prompt_ids`/`generated_ids`.
    """
    discarded = []
    for call in calls:
        if call.rows == 0:
            continue
        first = predict_index(call.cache_seqlens, 0, prompt_len)
        last = predict_index(call.cache_seqlens, call.rows - 1, prompt_len)
        if not (first <= index <= last):
            continue
        row = index - first
        if prompt_ids is None or generated_ids is None:
            return call, row, discarded
        try:
            assert_call_context(call, prompt_ids, generated_ids, selected_row=row)
        except ContextError as exc:
            discarded.append({"seq": call.seq, "row": row, "cache_seqlens": call.cache_seqlens,
                              "reason": str(exc)})
            continue
        return call, row, discarded
    raise SelectionError(
        f"no target call predicts generated index {index} on the accepted context "
        f"({len(discarded)} rejected candidate(s), {len(calls)} recorded target calls)")


def assert_call_context(call, prompt_ids, generated_ids, *, selected_row=None):
    """Assert every accepted row of `call` sits exactly where the ledger says it does.

    Context position of column j is `cache_seqlens + j`; it must hold the token that column carries.
    Columns after `selected_row` are the drafter's proposals and are not required to match.
    """
    ctx = list(prompt_ids) + list(generated_ids)
    cols = call.rows if selected_row is None else selected_row + 1
    if call.batch != 1:
        raise ContextError(f"expected a single-sequence pass, got batch {call.batch}")
    for j in range(cols):
        pos = call.cache_seqlens + j
        if pos >= len(ctx):
            raise ContextError(
                f"column {j} of the pass at cache_seqlens {call.cache_seqlens} addresses context "
                f"position {pos}, past the known context of {len(ctx)} tokens")
        if call.input_ids[0][j] != ctx[pos]:
            raise ContextError(
                f"ledger mismatch at context position {pos}: recorded row carries token "
                f"{call.input_ids[0][j]}, the prompt+generated context has {ctx[pos]}")
    return True


def assert_pass_ledger(calls, prompt_ids, generated_ids):
    """Check the whole recorded protocol: every accepted row sits exactly where the ledger says.

    A pass with `recurrent_history` carries the drafter's proposals in its trailing columns, so only
    column 0 (the accepted token) is required to match the context.
    """
    for call in calls:
        cols = 1 if call.recurrent_history else call.rows
        assert_call_context(call, prompt_ids, generated_ids, selected_row=cols - 1)
    return True


# ----------------------------------------------------------------------------- recording

class CallRecord:
    """One recorded target forward: plain data plus the tensors needed for replay."""

    __slots__ = ("seq", "batch", "cols", "cache_seqlens", "block_row", "positions",
                 "recurrent_history", "input_ids", "logits", "entry_state", "extra", "live_kv")

    def __init__(self, seq, batch, cols, cache_seqlens, block_row, positions,
                 recurrent_history, input_ids, logits=None, entry_state=None, extra=None,
                 live_kv=None):
        self.seq = seq
        self.batch = batch
        self.cols = cols
        self.cache_seqlens = cache_seqlens
        self.block_row = block_row
        self.positions = positions
        self.recurrent_history = recurrent_history
        self.input_ids = input_ids
        self.logits = logits
        self.entry_state = entry_state
        self.extra = extra or {}          # primitive params replayed verbatim
        self.live_kv = live_kv            # KV prefix at the live frontier (entry of this call)

    @property
    def rows(self) -> int:
        """Token rows in this pass (a single-sequence batch is required by the probe)."""
        return self.cols

    def summary(self) -> dict:
        return {"seq": self.seq, "batch": self.batch, "rows": self.rows,
                "cache_seqlens": self.cache_seqlens, "positions": list(self.positions),
                "recurrent_history": self.recurrent_history,
                "input_ids": [list(r) for r in self.input_ids],
                "extra_params": dict(self.extra),
                "has_logits": self.logits is not None,
                "live_kv": self.live_kv,
                "entry_state": summarize_snapshot(self.entry_state) if self.entry_state else None}


class ProbeRecorder:
    """Records target forwards only; draft forwards are counted and excluded.

    `self is target_model` is the classifier: the MTP head is a different Model object, so its calls
    can never be mistaken for target verification.
    """

    def __init__(self, target_model, prompt_len, capture_pred_indices=()):
        self.target_model = target_model
        self.prompt_len = prompt_len
        self.capture = set(capture_pred_indices)
        self.calls = []
        self.draft_calls = 0
        self._seq = 0

    def __call__(self, model, input_ids, params, forward):
        is_target = model is self.target_model
        if not is_target:
            self.draft_calls += 1
            return forward()
        for key in ("cache_seqlens", "block_table"):
            if key not in params:
                raise ParamError(
                    f"target forward #{self._seq} has no {key!r}; the real generator supplies it "
                    f"(generator/generator.py:1508-1523). Keys seen: {sorted(params)}")
        seqlens = params["cache_seqlens"]
        if seqlens is None or len(seqlens) != 1:
            raise ParamError(f"expected a single-sequence cache_seqlens, got {seqlens!r}")
        cache_seqlen = int(seqlens[0])
        block = params["block_table"]
        if block is None or int(block.shape[0]) != 1:
            raise ParamError(f"expected a single-row block_table, got {None if block is None else tuple(block.shape)}")
        block_row = [int(x) for x in block[0].tolist()]
        positions = params.get("positions")
        positions = [int(x) for x in positions[0].tolist()] if positions is not None else []
        hist = bool(params.get("recurrent_history"))
        states = params.get("recurrent_states") or []
        entry = snapshot_state(states[0]) if states else None
        preds = [predict_index(cache_seqlen, r, self.prompt_len) for r in range(int(input_ids.shape[1]))]
        want = any(p in self.capture for p in preds)
        want_logits = want
        extra = {}
        for key, value in params.items():
            if key in ("cache", "recurrent_states", "block_table", "cache_seqlens", "positions",
                       "indexed_embeddings"):
                continue
            if isinstance(value, (bool, int, float, str)) or value is None:
                extra[key] = value
        rec = CallRecord(seq=self._seq, batch=int(input_ids.shape[0]), cols=int(input_ids.shape[1]),
                         cache_seqlens=cache_seqlen, block_row=block_row, positions=positions,
                         recurrent_history=hist,
                         input_ids=[[int(x) for x in row] for row in input_ids.tolist()],
                         entry_state=entry, extra=extra,
                         # the canonical KV prefix is hashed for every target call (a few tens of KB at
                         # probe lengths); only the digest is retained, and the earlier common frontier
                         # needs it as much as the selected pass does
                         live_kv=(kv_fingerprint(params["cache"], cache_seqlen, block_row)
                                  if states else None))
        self._seq += 1
        out = forward()
        if want_logits and isinstance(out, torch.Tensor):
            rec.logits = out[0].detach().to(torch.float32).cpu().clone()   # (rows, vocab)
        elif want_logits:
            rec.extra["logits_returned"] = "none"      # prefill/last_tokens_only: no logits by design
        self.calls.append(rec)
        return out

    def target_calls(self):
        return self.calls


# ----------------------------------------------------------------------------- state snapshots

def _infer_conv_kernel(conv: torch.Tensor) -> int:
    """Ring width (K + max_history) when the owning module is not reachable; the probe never guesses
    K silently: callers must have a ring wide enough for the documented K-1 history."""
    raise SnapshotError("cannot resolve conv_kernel_size: layer exposes no module.conv_kernel_size and "
                        "the ring width alone does not identify K")


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()


def snapshot_state(state) -> dict:
    """Canonical per-layer snapshot.

    Source conformance (v1.6.4): `slot`, `position`, `last_history`, `parity`, `base_row()` and
    `scratch_row()` belong to the **sequence** state `GDNState` (modules/gated_delta_net.py:116-200).
    `cache.get_all_recurrent_layers()` returns **GDNLayerState** holders (line 259) that only own the
    tensors (`conv_state`, `recurrent_state`) and their accessors. The canonical live row is therefore
    `2*state.slot + state.parity`, computed here from the sequence state and applied to every layer.

    Raw tensors stay under `_`-prefixed keys; summarize_snapshot() is the only JSON view.
    """
    cache = getattr(state, "cache", None)
    if cache is None:
        raise SnapshotError("state object has no .cache; cannot resolve its recurrent layers")
    for attr in ("slot", "position", "parity", "base_row", "scratch_row"):
        if not hasattr(state, attr):
            raise SnapshotError(f"sequence state lacks {attr!r}; expected the real GDNState "
                                f"(modules/gated_delta_net.py:116-200), got {type(state).__name__}")
    layers = cache.get_all_recurrent_layers()
    if not layers:
        raise SnapshotError("cache reports no recurrent layers")
    slot, parity = int(state.slot), int(state.parity)
    base_row, scratch_row = int(state.base_row()), int(state.scratch_row())
    snap = {"slot": slot, "parity": parity, "position": int(state.position),
            "last_history": int(state.last_history), "state_class": type(state).__name__,
            "layers": {}}
    for key, layer in sorted(layers.items(), key=lambda kv: str(kv[0])):
        conv, rec = layer.get_state_tensors()
        if conv is None or rec is None:
            raise SnapshotError(f"layer {key} returned an empty state tensor")
        # Conv ring contract (modules/gated_delta_net_fn/conv1d.py:381-386): a committing pass copies
        # the last K tokens into columns [0, conv_kernel_size); a speculative pass writes the whole
        # pass into the tail ([:, -write_size:]) and the rewind copies the accepted prefix back. Only
        # [0, conv_kernel_size) is committed history; the tail is staging and must not be compared.
        k_size = int(getattr(layer.module, "conv_kernel_size", 0)) or _infer_conv_kernel(conv)
        if not (0 < k_size <= conv.shape[-1]):
            raise SnapshotError(f"layer {key}: conv_kernel_size {k_size} outside the ring width {conv.shape[-1]}")
        conv_slice = conv[slot][:, :k_size].detach().to(torch.float32).cpu().clone()
        conv_tail = conv[slot][:, k_size:].detach().to(torch.float32).cpu().clone()
        base = rec[base_row].detach().to(torch.float32).cpu().clone()
        scratch = rec[scratch_row].detach().to(torch.float32).cpu().clone()
        if base.numel() == 0 or conv_slice.numel() == 0:
            raise SnapshotError(f"layer {key} produced empty state slices")
        snap["layers"][str(key)] = {
            "layer_class": type(layer).__name__,
            "conv_sha256": _sha(conv_slice), "base_sha256": _sha(base),
            "conv_shape": list(conv_slice.shape), "base_shape": list(base.shape),
            "conv_kernel_size": k_size, "conv_tail_cols": int(conv_tail.shape[-1]),
            "conv_tail_sha256": _sha(conv_tail) if conv_tail.numel() else None,
            "_conv": conv_slice, "_base": base, "_scratch": scratch, "_conv_tail": conv_tail,
        }
    return snap


def summarize_snapshot(snap: dict) -> dict:
    """JSON-safe view: raw tensors dropped, nothing else changed."""
    if snap is None:
        return None
    return {"slot": snap["slot"], "parity": snap["parity"], "position": snap["position"],
            "last_history": snap["last_history"], "state_class": snap["state_class"],
            "layers": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                       for k, v in snap["layers"].items()}}


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        raise SnapshotError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    d = (a - b).abs()
    ref = a.pow(2).mean().sqrt().clamp(min=1e-30)
    return {"rms": float((a - b).pow(2).mean().sqrt()), "max": float(d.max()),
            "ref_rms": float(ref), "rel_rms": float((a - b).pow(2).mean().sqrt() / ref),
            "n_diff": int((d > 0).sum())}


def compare_snapshots(a: dict, b: dict, label: str) -> dict:
    """Canonical comparison. Requires raw tensors on both sides and identical logical positions."""
    if a is None or b is None:
        raise SnapshotError(f"{label}: missing snapshot")
    if a["position"] != b["position"]:
        raise SelectionError(f"{label}: logical positions differ ({a['position']} vs {b['position']}); "
                             f"these frontiers are not comparable")
    if set(a["layers"]) != set(b["layers"]):
        raise SnapshotError(f"{label}: layer sets differ")
    out = {"label": label, "position": a["position"], "parity_a": a["parity"], "parity_b": b["parity"],
           "layers": {}, "n_base_differing": 0, "n_conv_differing": 0}
    for key in a["layers"]:
        ea, eb = a["layers"][key], b["layers"][key]
        for side in ("_conv", "_base", "_scratch"):
            if side not in ea or side not in eb:
                raise SnapshotError(f"{label}: layer {key} lost raw tensor {side} before comparison")
        entry = {"parity_a": a["parity"], "parity_b": b["parity"],
                 "base_same": _sha(ea["_base"]) == _sha(eb["_base"]),
                 "conv_same": _sha(ea["_conv"]) == _sha(eb["_conv"]),
                 "base": _diff(ea["_base"], eb["_base"]),
                 "conv": _diff(ea["_conv"], eb["_conv"]),
                 "scratch": _diff(ea["_scratch"], eb["_scratch"])}
        if not entry["base_same"]:
            out["n_base_differing"] += 1
        if not entry["conv_same"]:
            out["n_conv_differing"] += 1
        out["layers"][key] = entry
    return out


def gather_kv_prefix(k, v, consumed: int, block_row, page_size: int = PAGE_SIZE) -> tuple:
    """Canonical valid KV prefix: physical pages in block-table order, truncated to `consumed`.

    `CacheLayer_fp16.get_kv()` is literally `return self.k, self.v` (cache/fp16.py:51-52): it neither
    gathers nor truncates, so hashing its result would cover unused pages and future/speculative rows.
    The storage layout is `(num_pages, PAGE_SIZE, num_kv_heads, head_dim)` (cache/fp16.py:22-28), and
    `paged_kv_cache_update` / `copy_page` index it as `k[page, row]`, so the logical token at position
    `p` lives at `k[block_row[p // PAGE_SIZE], p % PAGE_SIZE]`.
    """
    if k is None or v is None:
        raise SnapshotError("KV layer has no tensors")
    if consumed < 0:
        raise SnapshotError(f"negative consumed length {consumed}")
    npages = (consumed + page_size - 1) // page_size
    if npages == 0:
        return (k[:0, :0], v[:0, :0])
    if len(block_row) < npages:
        raise SnapshotError(f"block table has {len(block_row)} pages, {npages} needed for {consumed} tokens")
    if k.device != v.device:
        raise SnapshotError(f"K and V live on different devices ({k.device} vs {v.device})")
    if k.shape[0] <= max(block_row[:npages]) or min(block_row[:npages]) < 0:
        raise SnapshotError(f"block table {block_row[:npages]} addresses pages outside the layer "
                            f"({k.shape[0]} pages)")
    # the page indices must live on the same device as the cache tensors (GPU in a real run)
    pages = torch.tensor(list(block_row[:npages]), dtype=torch.long, device=k.device)
    kk = k.index_select(0, pages).reshape(npages * page_size, *k.shape[2:])[:consumed]
    vv = v.index_select(0, pages).reshape(npages * page_size, *v.shape[2:])[:consumed]
    return kk, vv


def control_prerequisites(name: str, a_state: dict, b_state: dict, a_kv: dict, b_kv: dict,
                          detail: dict) -> dict:
    """Gate for *control* comparisons. Returns the evidence dict; raises unless the two controls are
    at the same logical position with the same layer set, base rows, committed conv columns and every
    canonical KV prefix. AR-vs-MTP2 differences are evidence and never routed through this gate."""
    problems = []
    if a_state["position"] != b_state["position"]:
        problems.append(f"logical position {a_state['position']} vs {b_state['position']}")
    if set(a_state["layers"]) != set(b_state["layers"]):
        problems.append("layer sets differ")
    cmp = compare_snapshots(a_state, b_state, name)
    if cmp["n_base_differing"]:
        problems.append(f"{cmp['n_base_differing']} canonical base row(s) differ")
    if cmp["n_conv_differing"]:
        problems.append(f"{cmp['n_conv_differing']} committed conv ring(s) differ")
    if a_kv["n_layers"] != a_kv["expected_layers"] or b_kv["n_layers"] != b_kv["expected_layers"]:
        problems.append("KV layer coverage incomplete")
    kv_diff = sorted(k for k in a_kv["layers"]
                     if (a_kv["layers"].get(k) or {}).get("k_sha256") !=
                        (b_kv["layers"].get(k) or {}).get("k_sha256") or
                        (a_kv["layers"].get(k) or {}).get("v_sha256") !=
                        (b_kv["layers"].get(k) or {}).get("v_sha256"))
    if kv_diff:
        problems.append(f"{len(kv_diff)} KV prefix(es) differ: {kv_diff[:4]}")
    detail[name] = {"state": cmp, "kv_equal": not kv_diff, "kv_differing": kv_diff,
                    "problems": problems, "ok": not problems}
    if problems:
        raise ProbeError(f"control prerequisite failed for {name}: " + "; ".join(problems))
    return detail[name]


def kv_fingerprint(cache, consumed: int, block_row) -> dict:
    """Fingerprint of the canonical valid KV prefix, one entry per expected KV layer.

    Every layer in `cache.layers` must contribute: a short fingerprint is an error, not a smaller
    sample. Layers are gathered through `gather_kv_prefix`, never by hashing `get_kv()` wholesale.
    """
    layers = getattr(cache, "layers", None)
    if not layers:
        raise SnapshotError("cache exposes no KV layers; a KV fingerprint would be empty")
    out = {"consumed": int(consumed), "n_layers": 0, "expected_layers": len(layers),
           "gather": "block_table order, truncated to consumed", "layers": {}}
    for key, layer in sorted(layers.items(), key=lambda kv: str(kv[0])):
        k, v = layer.get_kv(torch.tensor([int(consumed)], dtype=torch.int32),
                            torch.tensor([list(block_row)], dtype=torch.int32))
        kk, vv = gather_kv_prefix(k, v, int(consumed), block_row)
        out["layers"][str(key)] = {"k_sha256": _sha(kk), "v_sha256": _sha(vv),
                                   "prefix_shape": list(kk.shape),
                                   "storage_shape": list(k.shape), "pages_used": len(
                                       {(p // PAGE_SIZE) for p in range(int(consumed))})}
        out["n_layers"] += 1
    if out["n_layers"] != out["expected_layers"]:
        raise SnapshotError(f"KV fingerprint covered {out['n_layers']} of {out['expected_layers']} layers")
    return out


def assert_identity_block(call):
    """A fresh single-sequence cache maps logical page p to physical page p, in order."""
    used = (call.cache_seqlens + call.rows + PAGE_SIZE - 1) // PAGE_SIZE
    head = call.block_row[:used]
    if head != list(range(used)):
        raise ParamError(f"recorded block_table {head} is not the identity mapping for a fresh cache; "
                         f"replaying into an independent cache would address other pages")


def issue_control_rows(recorded_call, rows, cache, state, model):
    """Issue one control pass with explicit row tokens at the recorded pass position."""
    assert_identity_block(recorded_call)
    used = (recorded_call.cache_seqlens + len(rows) + PAGE_SIZE - 1) // PAGE_SIZE
    block = torch.tensor([list(range(used)) + [0] * (len(recorded_call.block_row) - used)], dtype=torch.int32)
    positions = ([recorded_call.positions[0] + i for i in range(len(rows))]
                 if recorded_call.positions else [])
    params = {"attn_mode": "flash_attn", "cache": cache, "block_table": block,
              "cache_seqlens": torch.tensor([recorded_call.cache_seqlens], dtype=torch.int32),
              "recurrent_states": [state], "recurrent_history": False, "pinned_staging": True}
    if positions:
        params["positions"] = torch.tensor([positions], dtype=torch.int32)
    return model.forward(torch.tensor([rows], dtype=torch.long), params)[0].detach().to(torch.float32).cpu()


# ----------------------------------------------------------------------------- report helpers

def full_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict:
    """KL(p_a || p_b) over the full normalized vocabulary, fp32."""
    a = logits_a.detach().to(torch.float32).flatten()
    b = logits_b.detach().to(torch.float32).flatten()
    if a.shape != b.shape:
        raise ProbeError(f"logits shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    ta, tb = torch.topk(a, 5), torch.topk(b, 5)
    return {"kl_nats": float((la.exp() * (la - lb)).sum()),
            "argmax_a": int(ta.indices[0]), "argmax_b": int(tb.indices[0]),
            "top5_a": [int(x) for x in ta.indices], "top5_b": [int(x) for x in tb.indices],
            "max_abs_logit_diff": float((a - b).abs().max())}


def write_json(rec: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, default=str)


# ----------------------------------------------------------------------------- driver

def run(args) -> dict:
    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3 import ext as exl_ext
    from exllamav3.generator.sampler import ArgmaxSampler
    from exllamav3.version import __version__ as exl3_version
    import exllamav3 as pkg

    rec = {"label": "mtp-causal-probe-v3", "argv": sys.argv, "torch": torch.__version__,
           "hip": torch.version.hip, "exllamav3_version": exl3_version,
           "exllamav3_file": os.path.realpath(pkg.__file__),
           "ext_file": os.path.realpath(exl_ext.exllamav3_ext.__file__),
           "index": args.index, "filler": args.filler}

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    draft = Model.from_config(config, component="mtp")
    caches = {name: Cache(model, max_num_tokens=args.ctx, max_history=MAX_HISTORY)
              for name in ("gen_ar", "gen_mtp2", "seq", "batch3")}
    draft_cache = Cache(draft, max_num_tokens=args.ctx, max_history=MAX_HISTORY)
    if len({id(c) for c in list(caches.values()) + [draft_cache]}) != 5:
        raise ProbeError("phase caches must be distinct objects")
    model.load()
    draft.load()
    if not all(getattr(c, "initialized", False) for c in caches.values()):
        raise ProbeError("a phase cache did not initialize")
    rec["caches"] = {k: id(c) for k, c in caches.items()}

    ids = tok.encode(args.prompt, encode_special_tokens=True)
    prompt_ids = [int(x) for x in ids[0].tolist()]
    P = len(prompt_ids)
    rec["prompt_token_ids"] = prompt_ids
    rec["prompt_tokens"] = P
    rec["prefill_prediction"] = {
        "prefill_consumes": P - 1,
        "source": "generator/job.py:1417 prefill_end = min(prefill_end, len(seq.sequence_ids) - 1)"}

    orig_forward = Model.forward

    def run_generator(name, num_draft, target_cache, max_new):
        recorder = ProbeRecorder(model, P, capture_pred_indices=(args.index,))
        def wrapper(self, input_ids, params=None):
            params = params or {}
            return recorder(self, input_ids, params, lambda: orig_forward(self, input_ids, params))
        Model.forward = wrapper
        try:
            kw = ({"draft_model": draft, "draft_cache": draft_cache,
                   "num_draft_tokens": num_draft, "record_draft_stats": True} if num_draft else {})
            gen = Generator(model, target_cache, tok, **kw)
            job = Job(input_ids=ids, max_new_tokens=max_new, min_new_tokens=1,
                      sampler=ArgmaxSampler(), decode_special_tokens=True)
            serial = gen.enqueue(job)
            done = False
            while not done:
                for r in gen.iterate():
                    if r.get("serial") != serial:
                        continue
                    if r.get("stage") == "error":
                        raise r["error"]
                    if r.get("eos"):
                        done = True
            seq = job.sequences[0].sequence_ids.torch()
            generated = [int(x) for x in seq[0, P:].tolist()]
            return recorder, generated, [list(map(int, x)) for x in (getattr(job, "draft_stats", []) or [])]
        finally:
            Model.forward = orig_forward

    runs = {}
    for name, nd in (("gen_ar", 0), ("gen_mtp2", 2)):
        recorder, generated, dstats = run_generator(name, nd, caches[name], args.index + 3)
        calls = recorder.target_calls()
        if not calls:
            raise ProbeError(f"{name}: no target calls recorded")
        assert_pass_ledger(calls, prompt_ids, generated)
        call, row, discarded = select_pass(calls, args.index, P, prompt_ids, generated)
        assert_call_context(call, prompt_ids, generated, selected_row=row)
        if call.logits is None:
            raise ProbeError(f"{name}: the selected pass carried no captured logits")
        runs[name] = {"generated": generated, "draft_stats": dstats, "call": call, "row": row,
                      "discarded": discarded,
                      "logits": call.logits[row], "records": calls,
                      "n_target_calls": len(calls), "n_draft_calls": recorder.draft_calls}
        rec[name] = {"generated_ids": generated, "draft_stats": dstats,
                     "selected": {"seq": call.seq, "row": row, "cache_seqlens": call.cache_seqlens,
                                  "rows": call.rows, "history": call.recurrent_history,
                                  "predicted_index": predict_index(call.cache_seqlens, row, P),
                                  "input_ids": call.input_ids},
                     "n_target_calls": len(calls), "n_draft_calls": recorder.draft_calls,
                     "calls": [c.summary() for c in calls]}

    ar, mtp = runs["gen_ar"], runs["gen_mtp2"]
    if ar["generated"][:args.index] != mtp["generated"][:args.index]:
        raise ContextError("AR and MTP2 prefixes differ before the chosen index; not same-context")
    if len(ar["generated"]) <= args.index or len(mtp["generated"]) <= args.index:
        raise ContextError("a run produced fewer tokens than the chosen index")

    # ---- control seq: replay the recorded AR protocol verbatim, stopping AT the selected call
    seq_cache = caches["seq"]
    seq_cache.reset_states()
    seq_state = seq_cache.get_new_state()
    seq_entry, seq_kv, seq_logits = None, None, None
    for call in ar["records"]:
        if call.seq == ar["call"].seq:
            seq_entry = snapshot_state(seq_state)                     # live replay frontier
            seq_kv = kv_fingerprint(seq_cache, call.cache_seqlens, call.block_row)
            out = replay_one(call, seq_cache, seq_state, model)
            seq_logits = out[ar["row"]]
            break                                                     # never replay past the selected pass
        replay_one(call, seq_cache, seq_state, model)
    if seq_logits is None or seq_entry is None or seq_kv is None:
        raise ProbeError("seq control did not reach the selected pass")
    rec["seq"] = {"replayed_calls": ar["call"].seq + 1,
                  "selected_logits_sha": _sha(seq_logits),
                  "entry_state": summarize_snapshot(seq_entry), "kv": seq_kv,
                  "live_kv_matches_recorded": (
                      seq_kv["layers"] == (ar["call"].live_kv or {}).get("layers")),
                  "entry_state_matches_recorded": (
                      summarize_snapshot(seq_entry)["layers"].keys() ==
                      summarize_snapshot(ar["call"].entry_state)["layers"].keys())}

    # ---- the replay must reproduce the generator, or nothing downstream may be trusted.
    # State equality here means: same logical position, same layer set, canonical base rows equal,
    # committed conv columns equal, and every canonical KV prefix equal.
    selfcheck_logits = _sha(seq_logits) == _sha(ar["logits"])
    prereq = {}
    try:
        gate = control_prerequisites("seq replay vs recorded AR", seq_entry, ar["call"].entry_state,
                                    seq_kv, ar["call"].live_kv or {"layers": {}, "n_layers": 0,
                                                                   "expected_layers": len(seq_kv["layers"])},
                                    prereq)
    except ProbeError as exc:
        gate = None
        prereq["seq replay vs recorded AR"] = {"ok": False, "error": str(exc)}
    selfcheck_state = bool(gate and gate["ok"])
    rec["replay_selfcheck"] = {"logits_exact": selfcheck_logits,
                               "state_equal": selfcheck_state,
                               "state": prereq.get("seq replay vs recorded AR", {}),
                               "kv_recorded_vs_replay": {
                                   "recorded": (ar["call"].live_kv or {}).get("layers"),
                                   "replay": seq_kv["layers"]}}
    if not (selfcheck_logits and selfcheck_state):
        rec["invalid"] = ("replay did not reproduce the recorded AR protocol; the controls are not "
                          "comparable with this harness")
        rec["control_prerequisites"] = prereq
        if args.out:
            write_json(rec, args.out + ".FAILED.json")
        raise ProbeError("replay self-check failed (logits_exact=%s state_equal=%s); refusing to "
                         "report causal numbers" % (selfcheck_logits, selfcheck_state))

    # ---- control batch3: same prefix, then the same pass position with three rows
    b3_cache = caches["batch3"]
    b3_cache.reset_states()
    b3_state = b3_cache.get_new_state()
    for call in ar["records"]:
        if call.seq >= ar["call"].seq:
            break
        replay_one(call, b3_cache, b3_state, model)
    b3_entry = snapshot_state(b3_state)
    b3_kv = kv_fingerprint(b3_cache, ar["call"].cache_seqlens, ar["call"].block_row)
    row0 = ar["call"].input_ids[0][0]
    b3_logits = issue_control_rows(ar["call"], [row0, args.filler, args.filler],
                                   b3_cache, b3_state, model)[0]
    try:
        control_prerequisites("batch3 prefix vs AR frontier", seq_entry, b3_entry, seq_kv, b3_kv, prereq)
    except ProbeError as exc:
        rec["control_prerequisites"] = prereq
        rec["invalid"] = f"batch3 control is not on the same state/context: {exc}"
        if args.out:
            write_json(rec, args.out + ".FAILED.json")
        raise
    rec["control_prerequisites"] = prereq
    rec["batch3"] = {"replayed_calls": ar["call"].seq,
                     "control_rows": [row0, args.filler, args.filler],
                     "entry_state": summarize_snapshot(b3_entry), "kv": b3_kv,
                     "kv_matches_seq": b3_kv["layers"] == seq_kv["layers"]}

    # ---- comparisons
    rec["kl"] = {
        "seq_vs_recorded_ar": full_kl(seq_logits, ar["logits"]),
        "seq_vs_batch3": full_kl(seq_logits, b3_logits),
        "batch3_vs_recorded_mtp2": full_kl(b3_logits, mtp["logits"]),
        "recorded_ar_vs_recorded_mtp2": full_kl(ar["logits"], mtp["logits"]),
    }
    rec["selection"] = {
        name: {"seq": d["call"].seq, "row": d["row"],
               "cache_seqlens": d["call"].cache_seqlens, "rows": d["call"].rows,
               "history": d["call"].recurrent_history,
               "predicted_index": predict_index(d["call"].cache_seqlens, d["row"], P),
               "discarded_candidates": d["discarded"]}
        for name, d in (("ar", ar), ("mtp2", mtp))}

    # ---- state and KV at a genuinely common frontier: the greatest consumed position both runs
    # actually pass through (an interior verifier row's row-entry state is NOT the AR row-entry state,
    # so the comparison is made at a shared pass entry, and its position is reported)
    def common_frontier(ar_calls, mtp_calls, limit):
        mtp_by = {c.cache_seqlens: c for c in mtp_calls if c.entry_state is not None}
        best = None
        for c in ar_calls:
            if c.entry_state is None or c.cache_seqlens > limit:
                continue
            other = mtp_by.get(c.cache_seqlens)
            if other is None:
                continue
            if best is None or c.cache_seqlens > best[0]:
                best = (c.cache_seqlens, c, other)
        return best

    frontier = common_frontier(ar["records"], mtp["records"], ar["call"].cache_seqlens)
    if frontier is not None:
        consumed, ar_ref, mtp_ref = frontier
        rec["common_frontier"] = {"consumed": consumed,
                                  "ar_seq": ar_ref.seq, "mtp2_seq": mtp_ref.seq,
                                  "ar_rows": ar_ref.rows, "mtp2_rows": mtp_ref.rows,
                                  "relation": "greatest shared pass entry at or before the selected index"}
        rec["state_ar_vs_mtp2"] = compare_snapshots(
            ar_ref.entry_state, mtp_ref.entry_state,
            "recorded AR vs recorded MTP2 at the common pass-entry frontier")
        if ar_ref.live_kv is None or mtp_ref.live_kv is None:
            raise SnapshotError("common frontier found but a KV prefix was not captured")
        rec["kv_ar_vs_mtp2"] = {"consumed": consumed,
                                "ar": {k: v["k_sha256"] for k, v in ar_ref.live_kv["layers"].items()},
                                "mtp2": {k: v["k_sha256"] for k, v in mtp_ref.live_kv["layers"].items()},
                                "equal": ar_ref.live_kv["layers"] == mtp_ref.live_kv["layers"],
                                "n_layers": ar_ref.live_kv["n_layers"]}
    else:
        rec["common_frontier"] = {"not_compared": "the runs share no pass-entry position"}
        rec["state_ar_vs_mtp2"] = {"not_compared": "no common pass-entry frontier"}
        rec["kv_ar_vs_mtp2"] = {"not_compared": "no common pass-entry frontier"}
    rec["state_seq_vs_batch3_prefix"] = compare_snapshots(seq_entry, b3_entry,
                                                          "AR frontier vs batch3 prefix end") \
        if seq_entry["position"] == b3_entry["position"] else \
        {"not_compared": f"positions differ ({seq_entry['position']} vs {b3_entry['position']})"}

    tdir = args.tensor_dir
    if tdir:
        os.makedirs(tdir, exist_ok=True)
        torch.save({"seq_logits": seq_logits, "batch3_logits": b3_logits,
                    "recorded_ar_logits": ar["logits"], "recorded_mtp2_logits": mtp["logits"],
                    "prompt_ids": prompt_ids, "ar_ids": ar["generated"],
                    "mtp2_ids": mtp["generated"]}, os.path.join(tdir, "probe-logits.pt"))
        torch.save({"seq_kv": seq_kv, "b3_kv": b3_kv,
                    "recorded_ar_kv": (ar["call"].live_kv or {}),
                    "recorded_mtp2_kv": (frontier[2].live_kv if frontier else {})},
                   os.path.join(tdir, "probe-kv.pt"))
        for tag, snap in (("seq", seq_entry), ("batch3", b3_entry),
                          ("mtp2", (frontier[2].entry_state if frontier else None))):
            if snap is None:
                continue
            for key, layer in sorted(snap["layers"].items())[:2]:
                torch.save({"conv": layer["_conv"], "base": layer["_base"],
                            "scratch": layer["_scratch"]},
                           os.path.join(tdir, f"state_{tag}_{str(key).replace('/', '_').replace(' ', '')}.pt"))
        rec["tensor_dir"] = tdir
    return rec


def replay_one(call, cache, state, model):
    """Replay one recorded target call: recorded ids, cache_seqlens, positions, history, primitives."""
    assert_identity_block(call)
    used = (call.cache_seqlens + call.rows + PAGE_SIZE - 1) // PAGE_SIZE
    block = torch.tensor([list(range(used)) + [0] * (len(call.block_row) - used)], dtype=torch.int32)
    params = {"attn_mode": "flash_attn", "cache": cache, "block_table": block,
              "cache_seqlens": torch.tensor([call.cache_seqlens], dtype=torch.int32),
              "recurrent_states": [state], "recurrent_history": call.recurrent_history,
              "pinned_staging": True}
    if call.positions:
        params["positions"] = torch.tensor([call.positions], dtype=torch.int32)
    for key, value in (call.extra or {}).items():
        params.setdefault(key, value)      # e.g. last_tokens_only / skip_logits, replayed verbatim
    out = model.forward(torch.tensor(call.input_ids, dtype=torch.long), params)
    if out is None:
        return None                      # prefill / last_tokens_only: the real API returns no logits
    return out[0].detach().to(torch.float32).cpu()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=os.path.expanduser("~/models/Ornith-1.5-122B-EXL3-EAQv1-4-bpw"))
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--index", type=int, default=18, help="generated index to compare")
    ap.add_argument("--filler", type=int, default=4237, help="token for the non-causal control rows")
    ap.add_argument("--prompt", default="Write a short paragraph about the ocean.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tensor-dir", default=None)
    ap.add_argument("--dry-check", action="store_true",
                    help="validate arguments and pure logic only; never imports torch's GPU devices")
    args = ap.parse_args(argv)
    if args.dry_check:
        print(json.dumps({"dry_check": "ok", "index": args.index, "filler": args.filler,
                          "argv": sys.argv, "note": "no model, no cache, no device touched"}))
        return 0
    rec = run(args)
    if args.out:
        write_json(rec, args.out)
    print(json.dumps(rec, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
