#!/usr/bin/env python3
"""Layer-0 GDN fixture: replay the captured scan operands through the **real** native replay kernel.

The capture (`capture_layer0.py`) records, for one verifier transition:

* MTP: the staged scan operands the verifier pass consumed (`GDNLayerState.spec_conv/spec_beta/
  spec_g`), the canonical state rows and metadata before and after the partial-accept rewind, and the
  accepted-prefix ledger;
* AR: one record per decode pass, with the fused branch's shared statics (`s_conv/s_beta/s_g`) cloned
  immediately after layer 0 returned, and the canonical base row before and after each pass.

This module turns that into one question per replay, always through `ext.ScanReplayJob` +
`ext.batched_scan_replay` - the same kernel and geometry the library's `_rewind_layers` uses
(modules/gated_delta_net.py:76-84, 434-458) - on dedicated scratch tensors.

  1. PREREQUISITE the AR and MTP passes must have started from the same canonical state (base row and
                committed conv window) and the verifier's operands must be identical before the pass
                and at the rewind. These are hard gates: when one fails nothing downstream is
                compared, because a comparison from unequal starts or mutated inputs cannot answer
                the question. The detailed failure diff is kept.
  2. GATE       native replay of the captured MTP operands, from the captured initial state for the
                accepted prefix, must reproduce the captured post-rewind row **exactly**. If it does
                not, the difference is in the inputs or in the fast path that produced the recorded
                state; it is not evidence about rounding.
  3. AR         native replay of each AR pass's actual operands, one token at a time, against the AR
                state the model actually stored for the next pass.
  4. DIVERGENCE native MTP result vs the AR state actually stored after the same prefix - taken from
                the canonical AR frontiers, including a terminal one that has no following pass - plus
                an exact comparison of every accepted operand row.
  5. GROUPING   one native pass per accepted token, chained through the unchanged storage dtype and
                compared with the joint prefix replay. The per-token store is the only difference, so
                a joint/chained difference isolates store grouping; it is not by itself a defect.
  6. REFERENCE  the documented recurrence in float64 on the same captured inputs. This is an
                independent, clearly labelled check of the math - never a substitute for native output.

Every tensor the kernel touches is a private copy: the captured arrays are copied to the replay device
into freshly allocated buffers, their owners are held until the device is synchronized, and the base
copy is re-checked afterwards to prove the replay did not write to it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import torch

SCHEMA = "gdn-layer0-replay/1"
REFERENCE_LABEL = "independent float64 recurrence on the captured operands (not native evidence)"
GATE_NAME = ("native replay of the captured MTP operands reproduces the captured post-rewind row")
STORAGE_DTYPES = {"conv": torch.bfloat16, "beta": torch.bfloat16, "g": torch.float32}


class FixtureError(ValueError):
    """The capture does not carry the evidence this replay needs; never substitute."""


# ----------------------------------------------------------------------------- capture loading

def _capture_module():
    """Load the sibling `capture_layer0.py` (single implementation of the artifact format)."""
    try:
        import capture_layer0
        return capture_layer0
    except ImportError:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "capture_layer0", os.path.join(here, "capture_layer0.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def load_capture(path: str) -> dict:
    """Load one capture (`<path>.json` + `<path>.pt`) into a structure with real tensors."""
    return _capture_module().load_capture(path)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        raise FixtureError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    d = (a.double() - b.double()).abs()
    ref = a.double().pow(2).mean().sqrt().clamp(min=1e-30)
    return {"shape": list(a.shape), "max": float(d.max()),
            "rms": float(d.pow(2).mean().sqrt()), "ref_rms": float(ref),
            "rel_rms": float(d.pow(2).mean().sqrt() / ref), "n_diff": int((d > 0).sum())}


def _same(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Bitwise comparison. No casting: an exact gate that quietly casts is not an exact gate."""
    if a.dtype != b.dtype:
        raise FixtureError(f"dtype mismatch {a.dtype} vs {b.dtype}: exact comparison refused")
    return {"exact": bool(torch.equal(a, b)), "sha_a": sha(a), "sha_b": sha(b), "diff": _diff(a, b)}


# ----------------------------------------------------------------------------- fixture assembly

def _require(node: dict, path: str):
    cur = node
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise FixtureError(f"capture is missing required evidence: {path}")
        cur = cur[part]
    if cur is None:
        raise FixtureError(f"capture evidence {path} is empty")
    return cur


def _check_geometry(geo: dict) -> dict:
    """Shape/head geometry the pointer arithmetic depends on, checked before anything is allocated."""
    for key in ("kind", "nk", "nv", "dk", "dv", "state_half"):
        if key not in geo:
            raise FixtureError(f"geometry is missing {key!r}")
    if int(geo["kind"]) != 0:
        raise FixtureError(f"this replay tool supports GDN (kind 0) only, capture says kind "
                           f"{geo['kind']}: the KDA/mamba2 scan has a different operand layout")
    for key in ("nk", "nv", "dk", "dv"):
        if int(geo[key]) < 1:
            raise FixtureError(f"geometry {key}={geo[key]} is not a head count/dimension")
    if int(geo["nv"]) % int(geo["nk"]) != 0:
        raise FixtureError(f"nv {geo['nv']} is not a multiple of nk {geo['nk']}")
    return geo


def _geometry(rec: dict) -> dict:
    return _check_geometry(_require(rec, "before.geometry"))


def _state_row(clone: dict, expect_rows: int = 1) -> torch.Tensor:
    if clone is None:
        raise FixtureError("capture is missing a state row")
    t = clone["tensor"] if isinstance(clone, dict) and "tensor" in clone else clone
    if t.shape[0] != expect_rows:
        raise FixtureError(f"state row has {t.shape[0]} rows, expected {expect_rows}")
    return t


def _validate_scan_operands(operands: dict, where: str) -> dict:
    """Every precondition the raw-pointer call assumes, checked before a pointer is taken.

    A silent dtype cast here would change the arithmetic, so a captured dtype that is not the
    staging dtype is reported as a capture defect instead of being repaired.
    """
    geo = _check_geometry(_require(operands, "geometry"))
    nk, nv, dk, dv = (int(geo[k]) for k in ("nk", "nv", "dk", "dv"))
    fdim = 2 * nk * dk + nv * dv
    prefix, tstride = int(_require(operands, "prefix")), int(_require(operands, "tstride"))
    if tstride != 1:
        raise FixtureError(f"{where}: tstride {tstride}: this tool replays a row-major pass only")
    if prefix < 1:
        raise FixtureError(f"{where}: prefix {prefix} does not replay a token")
    rows = None
    for field, name, width in (("staged_conv", "conv", fdim), ("staged_beta", "beta", nv),
                               ("staged_g", "g", nv)):
        t = operands.get(field)
        want = STORAGE_DTYPES[name]
        if not isinstance(t, torch.Tensor) or t.dim() != 2:
            raise FixtureError(f"{where}: {field} is not a 2-D tensor")
        if t.dtype != want:
            raise FixtureError(f"{where}: {field} is {t.dtype}, the staging buffer is {want}: refusing "
                               f"to cast captured operands")
        if not t.is_contiguous():
            raise FixtureError(f"{where}: {field} is not contiguous; the kernel indexes rows by stride")
        if int(t.shape[1]) != width:
            raise FixtureError(f"{where}: {field} is {t.shape[1]} wide, the geometry {geo} needs {width}")
        rows = int(t.shape[0]) if rows is None else rows
        if int(t.shape[0]) != rows:
            raise FixtureError(f"{where}: {field} holds {t.shape[0]} rows, staged_conv holds {rows}")
    if prefix > rows:
        raise FixtureError(f"{where}: prefix {prefix} is longer than the {rows} staged rows")
    dt = torch.float16 if geo["state_half"] else torch.float32
    base = operands.get("initial_base")
    if not isinstance(base, torch.Tensor):
        raise FixtureError(f"{where}: initial_base is not a tensor")
    if base.dtype != dt:
        raise FixtureError(f"{where}: initial_base is {base.dtype} but the state storage dtype is {dt} "
                           f"(state_half={geo['state_half']}): refusing to cast")
    if "state_dtype" in geo and str(geo["state_dtype"]) != str(base.dtype):
        raise FixtureError(f"{where}: initial_base is {base.dtype} but the capture recorded state dtype "
                           f"{geo['state_dtype']}: the captured row is not the pool's storage dtype")
    if tuple(base.shape[1:]) != (nv, dk, dv) or int(base.shape[0]) != 1:
        raise FixtureError(f"{where}: initial_base has shape {list(base.shape)}, the geometry needs "
                           f"[1, {nv}, {dk}, {dv}]")
    if not base.is_contiguous():
        raise FixtureError(f"{where}: initial_base is not contiguous")
    if operands.get("d_skip") is not None:
        raise FixtureError(f"{where}: a d_skip tensor is present; this tool requires it absent (GDN has "
                           f"no skip connection) - refusing to pass a null D pointer")
    return {"fdim": fdim, "rows": rows, "prefix": prefix, "storage_dtype": dt}


def _forward_records(cap: dict, mode: str) -> list[dict]:
    recs = [r for r in cap["records"] if r["kind"] == "layer0_forward" and r["mode"] == mode]
    if not recs:
        raise FixtureError(f"capture of mode {mode!r} holds no layer0_forward record "
                           f"(excluded: {cap.get('excluded')})")
    return sorted(recs, key=lambda r: (r["position"], r["seq"]))


def _rewind_record(cap: dict) -> dict:
    recs = [r for r in cap["records"] if r["kind"] == "rewind"]
    if not recs:
        raise FixtureError("capture holds no rewind record: the accepted-prefix transition was not "
                           "recorded (pass outside the capture window?)")
    if len(recs) > 1:
        raise FixtureError(f"capture holds {len(recs)} rewind records; the fixture needs exactly the "
                           f"first target transition")
    return recs[0]


def build_fixture(ar_cap: dict, mtp_cap: dict) -> dict:
    """Validate the two captures against each other and assemble the replay operands."""
    for name, cap, mode in (("ar", ar_cap, "ar"), ("mtp", mtp_cap, "mtp")):
        if cap.get("capture_mode") != mode:
            raise FixtureError(f"{name} capture has mode {cap.get('capture_mode')!r}, expected {mode!r}")
        if int(cap.get("layer_idx", -1)) != 0:
            raise FixtureError(f"{name} capture is layer {cap.get('layer_idx')}, expected 0")
        if cap.get("overflow"):
            raise FixtureError(f"{name} capture overflowed: the record cap cut evidence off")
        if cap.get("errors"):
            raise FixtureError(f"{name} capture reported errors: {cap['errors']}")

    verify = _forward_records(mtp_cap, "target_verify")
    if len(verify) != 1:
        raise FixtureError(f"mtp capture holds {len(verify)} verifier passes; exactly one transition "
                           f"is expected")
    mtp_fwd = verify[0]
    rewind = _rewind_record(mtp_cap)
    ledger = {"staged_tokens": int(_require(rewind, "before.spec_shape")[1]),
              "accepted_prefix": int(rewind["prefix"]),
              "num_tokens": int(rewind["num_tokens"]),
              "last_history": int(rewind["last_history"]),
              "pass_entry": int(rewind["pass_entry"]),
              "position_before": int(rewind["position_before"]),
              "position_after": int(rewind["position_after"]),
              "base_row": int(_require(rewind, "before.base_row")),
              "scratch_row": int(_require(rewind, "before.scratch_row")),
              "spec_row": int(_require(rewind, "before.spec_row")),
              "spec_shape": [int(x) for x in _require(rewind, "before.spec_shape")]}
    if ledger["pass_entry"] != int(mtp_fwd["position"]):
        raise FixtureError(f"rewind belongs to pass entry {ledger['pass_entry']}, the captured "
                           f"verifier pass is at {mtp_fwd['position']}")
    if ledger["pass_entry"] + ledger["accepted_prefix"] != ledger["position_after"]:
        raise FixtureError(f"ledger does not close: entry {ledger['pass_entry']} + prefix "
                           f"{ledger['accepted_prefix']} != post-rewind position "
                           f"{ledger['position_after']}")
    if ledger["staged_tokens"] != int(mtp_fwd["seqlen"]):
        raise FixtureError(f"staged pass holds {ledger['staged_tokens']} tokens but the verifier "
                           f"forward consumed {mtp_fwd['seqlen']}")

    geo = _geometry(mtp_fwd)
    ar_fwd = _forward_records(ar_cap, "target_decode")
    for rec in ar_fwd:
        other = _geometry(rec)
        if other != geo:
            raise FixtureError(f"AR record at {rec['position']} has geometry {other}, MTP has {geo}")
        if int(rec["seqlen"]) != 1:
            raise FixtureError(f"AR record at {rec['position']} consumed {rec['seqlen']} tokens; the "
                               f"per-step comparison expects single-token decode passes")

    ar_entry = [r for r in ar_fwd if int(r["position"]) == ledger["pass_entry"]]
    if not ar_entry:
        raise FixtureError(f"no AR record at the verifier pass entry {ledger['pass_entry']}: the "
                           f"initial states cannot be compared")
    ar_at_entry = ar_entry[0]

    mtp_base = _state_row(_require(mtp_fwd, "before.canonical.rec_base"))
    ar_base = _state_row(_require(ar_at_entry, "before.canonical.rec_base"))
    checks = {
        "initial_state_equal": _same(ar_base, mtp_base),
        "initial_conv_equal": _same(
            _require(ar_at_entry, "before.canonical.conv_committed")["tensor"],
            _require(mtp_fwd, "before.canonical.conv_committed")["tensor"]),
    }

    ops = _require(rewind, "before.operands")
    block = ops.get("row_block")
    if block is None:
        raise FixtureError("rewind record carries no staged operand row block")
    if int(ledger["spec_shape"][2]) != 1:
        raise FixtureError(f"staged pass has token stride {ledger['spec_shape'][2]}: this tool "
                           f"replays a row-major pass only and refuses to repack silently")
    if int(ledger["spec_shape"][0]) != 1:
        raise FixtureError(f"staged pass has bsz {ledger['spec_shape'][0]}: this tool replays the "
                           f"single batch row the capture records (the rewind block holds seqlen rows)")
    prefix = ledger["accepted_prefix"]
    if prefix < 1:
        raise FixtureError(f"accepted prefix {prefix} is not a replay of at least one token "
                           f"(the library never replays a whole-pass rewind)")

    # the operands must be the ones the verifier forward recorded, sliced by the library's own r0
    forward_ops = mtp_fwd["operands"]
    stability = {}
    for key, stage in (("conv", "spec_conv"), ("beta", "spec_beta"), ("g", "spec_g")):
        fwd = _require(forward_ops, f"{key}.value.tensor")
        rew = _require(block, f"{stage}.value.tensor")
        if fwd.shape[0] < rew.shape[0]:
            raise FixtureError(f"forward capture of {key} holds {fwd.shape[0]} rows, the rewind block "
                               f"{rew.shape[0]}")
        stability[key] = _same(rew[:], fwd[:rew.shape[0]])
    checks["operands_stable_forward_vs_rewind"] = stability

    staged = {"staged_conv": _require(block, "spec_conv.value.tensor"),
              "staged_beta": _require(block, "spec_beta.value.tensor"),
              "staged_g": _require(block, "spec_g.value.tensor")}
    for key, dtype in (("staged_conv", torch.bfloat16), ("staged_beta", torch.bfloat16),
                       ("staged_g", torch.float32)):
        t = staged[key]
        if t.dtype != dtype:
            raise FixtureError(f"{key} is {t.dtype}, the library's staging buffer is {dtype}")
        if not t.is_contiguous():
            raise FixtureError(f"{key} is not contiguous; the kernel indexes it with row strides")
        if t.shape[0] != ledger["staged_tokens"]:
            raise FixtureError(f"{key} holds {t.shape[0]} rows, the ledger says "
                               f"{ledger['staged_tokens']}")
    if staged["staged_conv"].shape[1] != int(geo["nk"]) * int(geo["dk"]) * 2 + \
            int(geo["nv"]) * int(geo["dv"]):
        raise FixtureError("staged conv width does not match 2*nk*dk + nv*dv: the q|k|v split "
                           "would be wrong")

    # The library's own replay geometry, matched to the rewind that was actually captured: the first
    # job on this layer/cache is not necessarily the selected transition's job.
    jobs = [r for r in mtp_cap["records"] if r["kind"] == "replay_job"]
    if not jobs:
        raise FixtureError("capture holds no layer-0 replay_job record: the library's own replay "
                           "geometry is missing")
    matched = [j for j in jobs
               if int(j["prefix"]) == prefix
               and [int(x) for x in j["spec_shape"]] == ledger["spec_shape"]
               and int(j["base_row"]) == ledger["base_row"]
               and int(j["scratch_row"]) == ledger["scratch_row"]]
    if len(matched) != 1:
        raise FixtureError(f"{len(matched)} of {len(jobs)} replay_job records match the selected "
                           f"rewind ledger (prefix {prefix}, spec_shape {ledger['spec_shape']}, "
                           f"base_row {ledger['base_row']}, scratch_row {ledger['scratch_row']}): the "
                           f"job geometry cannot be attributed")
    job = matched[0]
    if int(job["tstride"]) != 1:
        raise FixtureError(f"replay_job has tstride {job['tstride']}: this tool replays row-major only")
    if int(job["r0"]) != ledger["spec_row"] * ledger["spec_shape"][1]:
        raise FixtureError(f"replay_job starts at staging row {job['r0']}, the library's r0 for spec_row "
                           f"{ledger['spec_row']} is {ledger['spec_row'] * ledger['spec_shape'][1]}")
    if job.get("d_skip") != "absent":
        raise FixtureError("layer 0 has a d_skip_f tensor; this tool requires it absent (GDN has no "
                           "skip connection) - refusing to pass a null pointer")
    if "scan_geometry" in job:
        want = [int(geo["kind"]), int(geo["nk"]), int(geo["nv"]), int(geo["dk"]), int(geo["dv"]),
                int(bool(geo["state_half"]))]
        if [int(x) for x in job["scan_geometry"]] != want:
            raise FixtureError(f"replay_job scan geometry {job['scan_geometry']} disagrees with the "
                               f"captured geometry {want}")
    if mtp_fwd.get("d_skip", {}).get("present"):
        raise FixtureError("forward capture reports a d_skip_f tensor; refusing a null D pointer")

    post_base = _state_row(_require(rewind, "after.canonical.rec_base"))
    if int(_require(rewind, "after.canonical.base_row")) != ledger["scratch_row"]:
        raise FixtureError(f"rewind moved parity onto row "
                           f"{_require(rewind, 'after.canonical.base_row')}, but the replay writes row "
                           f"{ledger['scratch_row']}: the captured post-row is not the replayed row")

    source_device = str(_require(rewind, "before.operands.row_block.spec_conv.value.layout.device"))
    staged_dtypes = _validate_scan_operands(
        {"staged_conv": staged["staged_conv"], "staged_beta": staged["staged_beta"],
         "staged_g": staged["staged_g"], "initial_base": mtp_base, "prefix": prefix,
         "tstride": 1, "geometry": geo, "d_skip": None}, "verifier operands (rewind block)")

    mtp_operands = dict(staged)
    mtp_operands.update({
        "initial_base": mtp_base.clone(), "prefix": prefix, "tstride": 1,
        "spec_row": ledger["spec_row"], "spec_shape": ledger["spec_shape"],
        "base_row": ledger["base_row"], "scratch_row": ledger["scratch_row"],
        "geometry": geo, "d_skip": None, "captured_post_base": post_base,
        "pre_rewind_scratch": _state_row(_require(rewind, "before.canonical.rec_scratch")),
        "pass_entry": ledger["pass_entry"], "position_after": ledger["position_after"],
        "source_device": source_device, "storage_dtype": str(staged_dtypes["storage_dtype"]),
        "replay_job": {k: job[k] for k in ("prefix", "spec_shape", "base_row", "scratch_row",
                                           "r0", "tstride")},
    })

    fdim = int(geo["nk"]) * int(geo["dk"]) * 2 + int(geo["nv"]) * int(geo["dv"])

    def flat_rows(clone: dict, width: int, what: str) -> torch.Tensor:
        """The AR statics are (bsz, seqlen, width) views; the staging buffers are flat rows. The
        row-major flattening below is the same memory the kernel reads (`staged_views` is its
        `.view(bsz, seqlen, -1)` counterpart), and it is required, not cosmetic."""
        t = clone["tensor"] if isinstance(clone, dict) and "tensor" in clone else clone
        if not t.is_contiguous():
            raise FixtureError(f"AR {what} is not contiguous: its row-major layout is not the "
                               f"staging layout")
        return t.reshape(-1, width)

    # One flat operand block per AR pass, so both the executable one-step pairs and the operand
    # comparison read the same captured memory.
    ar_operands = {}
    ar_source_device = None
    for rec in ar_fwd:
        pos = int(rec["position"])
        if pos in ar_operands:
            raise FixtureError(f"AR capture holds two decode passes at position {pos}")
        rec_ops = rec["operands"]
        if rec_ops.get("source") != "g_tensor_cache":
            raise FixtureError(f"AR record at {pos} did not capture the shared statics")
        device = str(_require(rec, "operands.conv.value.layout.device"))
        ar_source_device = ar_source_device or device
        if device != ar_source_device:
            raise FixtureError(f"AR records disagree on the device ({ar_source_device} vs {device})")
        ar_operands[pos] = {
            "staged_conv": flat_rows(_require(rec_ops, "conv.value"), fdim, "s_conv"),
            "staged_beta": flat_rows(_require(rec_ops, "beta.value"), int(geo["nv"]), "s_beta"),
            "staged_g": flat_rows(_require(rec_ops, "g.value"), int(geo["nv"]), "s_g"),
            "operand_keys": [rec_ops["conv"]["key"], rec_ops["beta"]["key"], rec_ops["g"]["key"]],
        }
    if ar_source_device is not None and ar_source_device != source_device:
        raise FixtureError(f"AR capture is from {ar_source_device} but the verifier capture is from "
                           f"{source_device}: the two paths cannot be compared")

    # Canonical AR frontiers, kept separately from the executable step pairs: the state AR had
    # stored after consuming `position` tokens is the *before* row of the AR pass at that position,
    # and the state stored by a pass is the frontier one token later. The terminal frontier has no
    # following pass, which is exactly what the accepted-prefix2 transition ends on.
    ar_frontiers = {}
    for rec in ar_fwd:
        pos = int(rec["position"])
        before = rec["before"]
        base_row = int(_require(before, "canonical.base_row"))
        after = rec["after"]
        ar_frontiers[pos] = {"base": _state_row(_require(before, "canonical.rec_base")),
                             "source": f"before.canonical.rec_base of the AR pass at {pos}"}
        after_row = int(_require(after, "canonical.base_row"))
        if after_row == base_row and (pos + 1) not in ar_frontiers:
            ar_frontiers[pos + 1] = {
                "base": _state_row(_require(after, "canonical.rec_base")),
                "source": f"after.canonical.rec_base of the AR pass at {pos}"}

    ar_steps = []
    for i in range(len(ar_fwd) - 1):
        a, b = ar_fwd[i], ar_fwd[i + 1]
        if int(b["position"]) != int(a["position"]) + 1:
            continue                    # not adjacent passes: no one-pass expectation to test
        pos = int(a["position"])
        step = dict(ar_operands[pos])
        step.update({
            "initial_base": _state_row(_require(a, "before.canonical.rec_base")),
            "prefix": 1, "tstride": 1,
            "spec_row": 0, "spec_shape": [1, 1, 1],
            "base_row": int(_require(a, "before.canonical.base_row")),
            "scratch_row": int(_require(a, "before.canonical.scratch_row")),
            "geometry": geo, "d_skip": None, "source_device": source_device,
            "expected_base": _state_row(_require(b, "before.canonical.rec_base")),
            "position_before": pos, "position_after": int(b["position"]),
            "fused_predicate": a.get("fused_predicate"),
        })
        _validate_scan_operands(step, f"AR step {pos}->{int(b['position'])}")
        ar_steps.append(step)
    if not ar_steps:
        raise FixtureError("the AR capture holds no adjacent single-token decode passes to replay")

    failed = []
    if not checks["initial_state_equal"]["exact"]:
        failed.append("initial_state_equal")
    if not checks["initial_conv_equal"]["exact"]:
        failed.append("initial_conv_equal")
    for key, cmp in checks["operands_stable_forward_vs_rewind"].items():
        if not cmp["exact"]:
            failed.append(f"operands_stable_forward_vs_rewind.{key}")
    prerequisites = {
        "ok": not failed, "failed": failed,
        "meaning": "hard gates: the AR and MTP passes must start from the same canonical state "
                   "(base row and committed conv window) and must have consumed identical operands "
                   "before the pass and at the rewind; when one fails the downstream native "
                   "comparisons are suppressed, never relaxed",
    }

    return {"schema": "gdn-layer0-fixture/1", "geometry": geo, "source_device": source_device,
            "cache_ids": {"ar": ar_cap.get("cache_id"), "mtp": mtp_cap.get("cache_id")},
            "mtp": mtp_operands, "ar": ar_steps, "ar_operands": ar_operands,
            "ar_frontiers": ar_frontiers, "ledger": ledger, "checks": checks,
            "prerequisites": prerequisites, "replay_job": mtp_operands["replay_job"]}


# ----------------------------------------------------------------------------- native replay

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _private(src: torch.Tensor, device: torch.device, dtype: torch.dtype, what: str) -> torch.Tensor:
    """A fresh buffer holding `src`'s values: never a view, never the captured array itself.

    The dtype is asserted, not applied: `copy_` into a different dtype would silently change the
    arithmetic of a replay that claims to consume the captured operands unchanged.
    """
    if not isinstance(src, torch.Tensor):
        raise FixtureError(f"{what}: operand is not a tensor")
    if src.dtype != dtype:
        raise FixtureError(f"{what}: captured dtype {src.dtype} != expected {dtype}; refusing to cast")
    if not src.is_contiguous():
        raise FixtureError(f"{what}: captured operand is not contiguous")
    out = torch.empty(src.shape, dtype=dtype, device=device)
    out.copy_(src)
    return out


def _replay_target(device, device_index) -> tuple[torch.device, int]:
    """Resolve the replay device once and refuse a device index that contradicts it.

    The captured operands came from one device; replaying them under a different device index would
    dereference rows of a pool that is not the one the capture read.
    """
    dev = torch.device(device)
    idx = dev.index if device_index is None else int(device_index)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise FixtureError(f"device {dev} requested but no accelerator is available")
        if dev.index is None:
            dev = torch.device("cuda", 0 if idx is None else idx)
        elif idx is not None and idx != dev.index:
            raise FixtureError(f"device index {idx} contradicts device {dev}: the replay would run "
                               f"on a different device than the one requested")
        idx = dev.index
    if idx is None:
        idx = 0
    if idx < 0:
        raise FixtureError(f"device index {idx} is not a device index")
    return dev, idx


def _check_source_device(operands: dict, dev: torch.device) -> None:
    """The capture records the device its tensors lived on; the replay must target that device."""
    src = operands.get("source_device")
    if not src:
        return
    want = torch.device(src)
    if want.type != dev.type:
        raise FixtureError(f"the capture was recorded on {want} but this replay targets {dev}: "
                           f"refusing to relabel a captured transition onto another device (take the "
                           f"wiring-check capture on the device the replay runs on)")
    if want.type == "cuda" and want.index != dev.index:
        raise FixtureError(f"captured on {want}, replaying on {dev}: the capture's pool is on another "
                           f"device than the one requested")


def native_scan_replay(operands: dict, ext, device="cuda:0", device_index: int | None = None) -> dict:
    """One `batched_scan_replay` over dedicated scratch tensors; returns the replayed state.

    `ext` is the extension module (`exllamav3.ext.exllamav3_ext`) or a spy with the same two entry
    points. Shape/dtype/geometry/device are validated before the first pointer is taken; the captured
    arrays are copied into freshly allocated device buffers, every pointer owner stays alive until the
    device is synchronized, and the base copy is re-checked afterwards.
    """
    geo = _check_geometry(operands["geometry"])
    dev, index = _replay_target(device, device_index)
    _check_source_device(operands, dev)
    checked = _validate_scan_operands(operands, f"native replay on {dev}")
    prefix, tstride = checked["prefix"], int(operands["tstride"])
    dt = checked["storage_dtype"]

    conv = _private(operands["staged_conv"], dev, torch.bfloat16, "conv")
    g = _private(operands["staged_g"], dev, torch.float32, "g")
    beta = _private(operands["staged_beta"], dev, torch.bfloat16, "beta")
    base = _private(operands["initial_base"], dev, dt, "initial_base")
    base_before = base.detach().to("cpu", copy=True)
    out = torch.zeros_like(base)
    owners = (conv, g, beta, base, out)         # every pointer target stays alive past the launch

    job = ext.ScanReplayJob(conv.data_ptr(), g.data_ptr(), beta.data_ptr(),
                            base.data_ptr(), out.data_ptr(), 0, prefix, tstride)
    ext.batched_scan_replay([job], index, int(geo["kind"]), int(geo["nk"]), int(geo["nv"]),
                            int(geo["dk"]), int(geo["dv"]), bool(geo["state_half"]))
    _sync(dev)
    assert owners is not None
    return {"state": out.to("cpu"),
            "base_unmodified": bool(torch.equal(base.detach().to("cpu"), base_before)),
            "call": {"device": str(dev), "device_index": index, "prefix": prefix,
                     "tstride": tstride, "kind": int(geo["kind"]), "nk": int(geo["nk"]),
                     "nv": int(geo["nv"]), "dk": int(geo["dk"]), "dv": int(geo["dv"]),
                     "state_half": bool(geo["state_half"]),
                     "storage_dtype": str(dt), "d_skip": "absent",
                     "source_device": operands.get("source_device"),
                     "pointers": {"conv": int(conv.data_ptr()), "g": int(g.data_ptr()),
                                  "beta": int(beta.data_ptr()), "base": int(base.data_ptr()),
                                  "out": int(out.data_ptr())}}}


def native_chain_replay(operands: dict, ext, device="cuda:0", device_index: int | None = None,
                        steps: int | None = None) -> dict:
    """One native pass per accepted token, chained through the unchanged storage dtype.

    Every step replays the same captured row the joint prefix replay would consume, starts from the
    previous step's result converted only to the pool's storage dtype (it already is that dtype), and
    writes into dedicated scratch. The per-token store is therefore the only difference from the joint
    replay of the same rows.
    """
    geo = _check_geometry(operands["geometry"])
    dev, index = _replay_target(device, device_index)
    _check_source_device(operands, dev)
    checked = _validate_scan_operands(operands, "native chained replay")
    n = checked["prefix"] if steps is None else int(steps)
    if not 1 <= n <= checked["prefix"]:
        raise FixtureError(f"chain length {n} outside the accepted prefix {checked['prefix']}")
    state = operands["initial_base"]
    out_steps = []
    for i in range(n):
        one = dict(operands)
        one["staged_conv"] = operands["staged_conv"][i:i + 1]
        one["staged_beta"] = operands["staged_beta"][i:i + 1]
        one["staged_g"] = operands["staged_g"][i:i + 1]
        one["initial_base"] = state
        one["prefix"] = 1
        got = native_scan_replay(one, ext, dev, index)
        state = got["state"]
        if state.dtype != checked["storage_dtype"]:
            raise FixtureError(f"chain step {i} returned {state.dtype}, the pool stores "
                               f"{checked['storage_dtype']}")
        out_steps.append({"row": i, "state_sha": sha(state),
                          "base_unmodified": got["base_unmodified"], "call": got["call"]})
    return {"steps": out_steps, "final": state,
            "valid": all(s["base_unmodified"] for s in out_steps),
            "storage_dtype": str(checked["storage_dtype"]),
            "operand_rows": [int(i) for i in range(n)]}


# ----------------------------------------------------------------------------- reference

def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def reference_operands(operands: dict) -> dict:
    """Split the captured staged conv output into q|k|v using the real channel layout.

    `fdim_qkv = 2*nk*dk + nv*dv` (source: `GatedDeltaNet.__init__`, and the `tp_stash_pieces`
    assertion `fdim_qkv == 2*nk*kd + nv*vd`); k heads are fewer than v heads and each k head serves
    `G = nv // nk` consecutive v heads (grouped layout, `k_head = v_head // G`). q/k are l2-normalised
    because every real path passes `use_qk_l2norm_in_kernel=True`.
    """
    geo = operands["geometry"]
    nk, nv, dk, dv = (int(geo[k]) for k in ("nk", "nv", "dk", "dv"))
    conv = operands["staged_conv"].double()
    T = conv.shape[0]
    q = _l2norm(conv[:, :nk * dk].reshape(T, nk, dk))
    k = _l2norm(conv[:, nk * dk:2 * nk * dk].reshape(T, nk, dk))
    v = conv[:, 2 * nk * dk:].reshape(T, nv, dv)
    G = nv // nk
    if G > 1:
        q = q.repeat_interleave(G, dim=1)
        k = k.repeat_interleave(G, dim=1)
    return {"q": q, "k": k, "v": v, "G": G}


def reference_states(operands: dict, steps: int | None = None) -> torch.Tensor:
    """The documented recurrence in float64, one state per consumed token.

    Source: `torch_recurrent_gated_delta_rule` (gated_delta_net_fn/gated_delta_rule.py:9-62):
    `kv = (state * k).sum(dk)`, `v' = v - kv*exp(g)`, `state = state*exp(g) + beta*k (x) v'`.
    Depends on k, v, g, beta only; q affects the layer output, not the state.
    """
    r = reference_operands(operands)
    geo = operands["geometry"]
    nv, dk, dv = int(geo["nv"]), int(geo["dk"]), int(geo["dv"])
    T = min(int(steps if steps is not None else operands["prefix"]), r["k"].shape[0])
    S = operands["initial_base"].double().reshape(nv, dk, dv)
    g = operands["staged_g"].double()
    beta = operands["staged_beta"].double()
    states = []
    for t in range(T):
        decay = g[t].exp()
        kv = torch.einsum("hk,hkv->hv", r["k"][t], S)
        vt = r["v"][t] - kv * decay[:, None]
        S = S * decay[:, None, None] + r["k"][t][:, :, None] * (vt * beta[t][:, None])[:, None, :]
        states.append(S.clone())
    return torch.stack(states)


# ----------------------------------------------------------------------------- the report

def native_evidence(device) -> bool:
    """A CPU (or spy) run is a wiring check, never native numeric evidence."""
    return torch.device(device).type == "cuda"


def load_ext():
    """The real extension module; a missing extension is a hard failure, never a downgrade."""
    try:
        from exllamav3.ext import exllamav3_ext
        return exllamav3_ext
    except Exception as exc:                  # pragma: no cover - exercised on A only
        raise FixtureError(f"native replay backend unavailable ({type(exc).__name__}: {exc}); "
                           f"refusing to compare a float64 reference and call it native") from exc


def _suppress_attribution(report: dict, reason: str) -> None:
    """Withdraw every causal claim that a failed/mutated native call would otherwise support.

    Raw details stay in `ar_native_steps`, `chain.step_base_unmodified` and `gate.call`; only the
    cross-path attributions are replaced, so the evidence survives without an invalid conclusion.
    """
    for key in ("mtp_native_vs_ar_actual", "operands_mtp_vs_ar"):
        report[key] = {"suppressed": True, "reason": reason}
    chain = report.get("chain")
    if isinstance(chain, dict) and "steps" in chain:
        for key in ("final_vs_joint", "final_vs_captured_post", "final_vs_ar_actual"):
            if key in chain:
                chain[key] = {"suppressed": True, "reason": reason}
        chain["attribution_suppressed"] = reason
    ar = report.get("ar_replay")
    if isinstance(ar, dict) and "all_exact" in ar:
        ar["attribution_suppressed"] = reason


def replay(ar_cap: dict, mtp_cap: dict, ext=None, device: str = "cuda:0",
           device_index: int | None = None) -> dict:
    """Assemble the fixture, run the native replays, and report every comparison with its label."""
    fx = build_fixture(ar_cap, mtp_cap)
    if ext is None:
        ext = load_ext()
    dev, index = _replay_target(device, device_index)
    _check_source_device(fx["mtp"], dev)
    native = native_evidence(dev)
    report = {"schema": SCHEMA, "device": str(dev), "device_index": index,
              "native_evidence": native,
              "evidence_note": ("native kernel output" if native else
                                "CPU/foreign backend: wiring check only, NOT native numeric evidence"),
              "ext": _ext_identity(ext),
              "geometry": fx["geometry"], "source_device": fx["source_device"],
              "ledger": fx["ledger"], "replay_job": fx["replay_job"],
              "cache_ids": fx["cache_ids"], "checks": {}}

    for name, block in fx["checks"].items():
        if name == "operands_stable_forward_vs_rewind":
            report["checks"][name] = {k: {"exact": bool(v["exact"]), "diff": v["diff"]}
                                      for k, v in block.items()}
        else:
            report["checks"][name] = {"equal": bool(block["exact"]),
                                      "sha_ar": block["sha_a"], "sha_mtp": block["sha_b"],
                                      "diff": block["diff"]}

    # Hard prerequisites: unequal starting states or overwritten operands make every downstream
    # comparison meaningless, so nothing is compared and the report says why. It is never a pass.
    report["prerequisites"] = fx["prerequisites"]
    if not fx["prerequisites"]["ok"]:
        reason = ("hard prerequisite failed: " + ", ".join(fx["prerequisites"]["failed"]) +
                  "; the AR and MTP paths did not start from the same state or did not consume the "
                  "same operands, so no causal comparison is valid")
        for key in ("gate", "mtp_native_vs_ar_actual", "operands_mtp_vs_ar", "chain"):
            report[key] = {"passed": False, "suppressed": True, "reason": reason}
        report["gate"]["name"] = GATE_NAME
        report["gate"]["diff"] = None
        report["ar_native_steps"] = []
        report["ar_replay"] = {"not_run": reason}
        report["reference"] = {"label": REFERENCE_LABEL, "not_run": reason}
        report["status"] = "prerequisites_failed"
        if not native:
            report["status"] += ":non_native_backend"
        report["diagnosis"] = (
            "the capture's hard prerequisites failed, so the native replay was not run: the AR and "
            "MTP passes must start from the same canonical state and consume identical operands "
            "before the pass and at the rewind. Fix the capture/protocol; do not relax the gate.")
        report["rounding"] = "not established by this tool"
        return report

    mtp = fx["mtp"]
    native_mtp = native_scan_replay(mtp, ext, dev, index)
    replayed = native_mtp["state"]
    gate = _same(replayed, mtp["captured_post_base"])
    invalid = None if native_mtp["base_unmodified"] else (
        "the replay wrote into its own input base copy: the pointer wiring is wrong, so the "
        "comparison is not valid even if the values happen to match")
    report["gate"] = {
        "name": GATE_NAME,
        "passed": bool(gate["exact"]) and invalid is None,
        "replayed_sha": gate["sha_a"], "captured_sha": gate["sha_b"], "diff": gate["diff"],
        "call": native_mtp["call"], "base_unmodified": native_mtp["base_unmodified"],
        "prefix": int(mtp["prefix"]), "staged_tokens": int(mtp["spec_shape"][1])}
    if invalid:
        report["gate"]["reason"] = invalid

    if not report["gate"]["passed"]:
        # The mandatory initial gate failed (or the joint replay mutated its own input base): no later
        # native call can attribute anything, so none is made and every downstream comparison is
        # withdrawn. The gate diff, call and base flag stay for the report.
        reason = report["gate"].get("reason") or (
            "the mandatory joint native replay did not reproduce the captured post-rewind row")
        report["base_integrity"] = {
            "ok": bool(native_mtp["base_unmodified"]),
            "mutated_calls": [] if native_mtp["base_unmodified"] else ["mtp"],
            "note": "an executed native replay that writes into its own input base copy is a pointer "
                    "wiring defect: the comparison is invalid even if the values match"}
        _suppress_attribution(report, "the mandatory initial native gate failed: " + reason)
        report["ar_native_steps"] = []
        report["ar_replay"] = {"not_run": "the mandatory initial native gate failed"}
        report["reference"] = {"label": REFERENCE_LABEL,
                               "not_run": "the mandatory initial native gate failed"}
        report["status"] = ("gate_failed:input_base_mutated" if not native_mtp["base_unmodified"]
                            else "gate_failed:inputs_or_fast_path")
        report["diagnosis"] = (
            "native replay of the captured verifier operands does NOT reproduce the captured "
            "post-rewind state, so the AR and chained replays were not run and no cross-path "
            "comparison is reported. Inspect the captured operand identity/geometry and the fast path "
            "that recorded them; do not attribute the divergence to rounding. " + reason + ".")
        report["rounding"] = "not established by this tool"
        if not native:
            report["status"] += ":non_native_backend"
        return report

    ar_rows = []
    for step in fx["ar"]:
        got = native_scan_replay(step, ext, dev, index)
        cmp = _same(got["state"], step["expected_base"])
        ar_rows.append({"position_before": step["position_before"],
                        "position_after": step["position_after"],
                        "exact": bool(cmp["exact"]), "sha_replayed": cmp["sha_a"],
                        "sha_actual": cmp["sha_b"], "diff": cmp["diff"],
                        "base_unmodified": got["base_unmodified"],
                        "operand_keys": step["operand_keys"],
                        "call": got["call"]})
    report["ar_native_steps"] = ar_rows
    not_exact = [r["position_before"] for r in ar_rows if not r["exact"]]
    report["ar_replay"] = {
        "all_exact": not not_exact, "not_exact": not_exact,
        "note": ("each AR pass's own operands, replayed one step, reproduce the state AR stored next"
                 if not not_exact else
                 "an AR one-step native replay does NOT reproduce the state AR stored next: that is "
                 "diagnostic evidence about the AR path itself (operands, geometry or the fast path), "
                 "not a rounding result, and it limits how far the MTP comparison can be attributed")}

    target = fx["ledger"]["position_after"]
    ar_actual, ar_source = _ar_state_at(fx, target)
    if ar_actual is None:
        report["mtp_native_vs_ar_actual"] = {"not_compared": ar_source}
    else:
        cmp = _same(replayed, ar_actual)
        report["mtp_native_vs_ar_actual"] = {
            "compared_at_position": target, "exact": bool(cmp["exact"]),
            "sha_mtp_native": cmp["sha_a"], "sha_ar_actual": cmp["sha_b"], "diff": cmp["diff"],
            "ar_state_source": ar_source,
            "note": ("the verifier's own operands, replayed natively from the shared initial state, "
                     "reproduce exactly the state AR stored after the same prefix" if cmp["exact"] else
                     "the verifier's replayed state differs from the state AR stored at the same "
                     "frontier: compare the operand blocks below before attributing a cause")}

    entry = fx["ledger"]["pass_entry"]
    rows = []
    for i in range(int(fx["ledger"]["accepted_prefix"])):
        pos = entry + i
        ar_ops = fx["ar_operands"].get(pos)
        if ar_ops is None:
            rows.append({"row": i, "position": pos,
                         "not_compared": f"the AR capture holds no decode pass at position {pos}"})
            continue
        cmp_ops = {}
        for key, name in (("conv", "staged_conv"), ("beta", "staged_beta"), ("g", "staged_g")):
            cmp_ops[key] = _same(ar_ops[name], mtp[name][i:i + 1])
        rows.append({"row": i, "position": pos,
                     "all_exact": all(v["exact"] for v in cmp_ops.values()),
                     **{k: {"exact": bool(v["exact"]), "sha_ar": v["sha_a"],
                            "sha_mtp": v["sha_b"], "diff": v["diff"]}
                        for k, v in cmp_ops.items()}})
    report["operands_mtp_vs_ar"] = {
        "compared_at_position": entry, "rows": rows,
        "note": "every accepted staged row against the AR pass that consumed the same token: an exact "
                "match means both paths fed the scan identical inputs for that token"}

    chain = native_chain_replay(mtp, ext, dev, index)
    chain_vs_joint = _same(chain["final"], replayed)
    chain_vs_post = _same(chain["final"], mtp["captured_post_base"])
    block = {"steps": len(chain["steps"]), "storage_dtype": chain["storage_dtype"],
             "operand_rows": chain["operand_rows"], "valid": chain["valid"],
             "final_sha": sha(chain["final"]),
             "final_vs_joint": {"exact": bool(chain_vs_joint["exact"]),
                                "diff": chain_vs_joint["diff"]},
             "final_vs_captured_post": {"exact": bool(chain_vs_post["exact"]),
                                        "diff": chain_vs_post["diff"]},
             "step_base_unmodified": [s["base_unmodified"] for s in chain["steps"]],
             "note": "one native pass per accepted token over the same captured rows, chained through "
                     "the unchanged storage dtype; the per-token store is the only difference from the "
                     "joint replay, so a joint/chained difference is a store-grouping effect and not, "
                     "by itself, evidence of a defect"}
    if ar_actual is None:
        block["final_vs_ar_actual"] = {"not_compared": ar_source}
    else:
        cmp = _same(chain["final"], ar_actual)
        block["final_vs_ar_actual"] = {"exact": bool(cmp["exact"]), "diff": cmp["diff"],
                                       "ar_state_source": ar_source}
    report["chain"] = block

    mutations = [f"ar:{r['position_before']}" for r in ar_rows if not r["base_unmodified"]]
    mutations += [f"chain:{s['row']}" for s in chain["steps"] if not s["base_unmodified"]]
    report["base_integrity"] = {
        "ok": not mutations, "mutated_calls": mutations,
        "note": "every executed native replay must leave its private input base copy unchanged; a "
                "mutation is a pointer-wiring defect, not a numerical result"}
    if mutations:
        report["gate"]["passed"] = False
        report["gate"]["reason"] = (
            "an executed native replay wrote into its own input base copy (" + ", ".join(mutations) +
            "): the values are not a valid comparison and no attribution is reported")
        _suppress_attribution(report, report["gate"]["reason"])
        report["status"] = "gate_failed:input_base_mutated"
        if not native:
            report["status"] += ":non_native_backend"
        report["diagnosis"] = (
            "the mandatory initial gate passed, but a later native replay mutated its own input base "
            "copy. That is a pointer-lifetime or wiring defect in this tool: fix the wiring before "
            "reading any AR/chain difference. " + report["gate"]["reason"] + ".")
        report["rounding"] = "not established by this tool"
        return report

    ref_mtp = reference_states(mtp, steps=int(mtp["prefix"]))[-1]
    ref_state = ref_mtp.reshape(mtp["captured_post_base"].shape)
    report["reference"] = {
        "label": REFERENCE_LABEL,
        "mtp_prefix_state_vs_captured": _diff(ref_state.float(),
                                              mtp["captured_post_base"].float()),
        "mtp_prefix_state_vs_native": _diff(ref_state.float(), replayed.float()),
        "ar_steps": [{"position_after": s["position_after"],
                      "diff": _diff(reference_states(s, steps=1)[-1]
                                    .reshape(s["expected_base"].shape).float(),
                                    s["expected_base"].float())}
                     for s in fx["ar"]],
        "caveat": "float64 recurrence on the same captured operands: a check of the documented math, "
                  "not the kernel's arithmetic, and not evidence about the state the model stored"}

    report["status"] = ("gate_passed" if report["gate"]["passed"] else
                        "gate_failed:inputs_or_fast_path")
    diagnosis = (
        "native replay of the captured verifier operands reproduces the captured post-rewind state "
        "exactly; any remaining difference is a property of the operands or of the paths that "
        "produced them" if report["gate"]["passed"] else
        "native replay of the captured verifier operands does NOT reproduce the captured post-rewind "
        "state. Inspect the captured operand identity/geometry and the fast path that recorded them; "
        "do not attribute the divergence to rounding.")
    if report["gate"].get("reason"):
        diagnosis += " " + report["gate"]["reason"] + "."
    if not report["ar_replay"]["all_exact"]:
        diagnosis += (" Limit: an AR one-step native replay does not reproduce the state AR stored "
                      f"after {report['ar_replay']['not_exact']}.")
    report["diagnosis"] = diagnosis
    report["rounding"] = "not established by this tool"
    if not native:
        report["status"] += ":non_native_backend"
    return report


def _ar_state_at(fx: dict, position: int):
    """The canonical AR state after consuming `position` tokens, or None with the reason it is absent."""
    frontier = fx["ar_frontiers"].get(int(position))
    if frontier is not None:
        return frontier["base"], frontier["source"]
    step = next((s for s in fx["ar"] if int(s["position_after"]) == int(position)), None)
    if step is not None:
        return step["expected_base"], (f"the state AR stored after its pass at "
                                       f"{step['position_before']}")
    return None, (f"the AR capture holds no canonical state after {position} tokens (the accepted "
                  f"prefix frontier): widen the AR capture positions, do not infer the AR state")


def _ext_identity(ext) -> dict:
    """Which backend object produced the replay: a spy or an unreadable file is not provenance."""
    path = getattr(ext, "__file__", None)
    if not path:
        return {"path": None, "sha256": None,
                "note": "the backend object exposes no __file__ (spy/stand-in): not native provenance"}
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return {"path": os.path.realpath(path), "sha256": h.hexdigest()}
    except OSError as exc:
        return {"path": path, "sha256": f"unreadable: {type(exc).__name__}"}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="replay a captured layer-0 GDN transition natively")
    ap.add_argument("--ar", required=True, help="AR capture JSON")
    ap.add_argument("--mtp", required=True, help="MTP capture JSON")
    ap.add_argument("--out", default=None, help="report path (a .FAILED.json is written on failure)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-index", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        report = replay(load_capture(args.ar), load_capture(args.mtp),
                        device=args.device, device_index=args.device_index)
    except BaseException as exc:
        import traceback
        payload = {"schema": SCHEMA, "argv": sys.argv,
                   "error": {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()},
                   "label": "no native conclusion: this run did not complete a native replay"}
        if args.out:
            with open(args.out + ".FAILED.json", "w") as f:
                json.dump(payload, f, indent=2)
        print(f"replay: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, FixtureError) else 3
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        if not report["gate"]["passed"]:
            with open(args.out + ".FAILED.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
    print(json.dumps({"status": report["status"], "gate": report["gate"]["passed"],
                      "native_evidence": report["native_evidence"],
                      "prerequisites": report["prerequisites"],
                      "mtp_native_vs_ar_actual": report["mtp_native_vs_ar_actual"]}, default=str))
    return 0 if report["gate"]["passed"] and report["native_evidence"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
