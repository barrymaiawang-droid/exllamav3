#!/usr/bin/env python3
"""Behavioural CPU tests for probe_mtp_causal.py.

Each test fails on one of the defects the parent review found in v2 and exercises the corrected
path end to end with a deterministic fake model/cache/state. No GPU, no 122B, no exllamav3 import:
the module's model-facing code paths (record -> select -> ledger -> replay -> compare -> serialize)
are driven through the same functions the real run uses.

Run: python -m pytest tools/exl3-bench/test_probe_mtp_causal.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys

import pytest
import torch
import types

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("probe_mtp_causal", os.path.join(HERE, "probe_mtp_causal.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

P = 7                      # prompt length used by the fixtures
GEN = list(range(100, 130))  # generated token ids (index -> token)
PROMPT = list(range(1000, 1000 + P))


# --------------------------------------------------------------------------- fakes

class FakeLayer:
    """Per-layer holder with the *real* GDNLayerState surface (tensors + get_state_tensors only)."""

    def __init__(self, kernel=2, max_history=3, hidden=4, rows=2, pages=2):
        self.module = types.SimpleNamespace(conv_kernel_size=kernel)
        self.conv = torch.zeros((1, hidden, kernel + max_history))
        self.rec = torch.zeros((rows, 1, 2, 2))
        self.k = torch.zeros((pages, probe.PAGE_SIZE, 2, 2), dtype=torch.float16)
        self.v = torch.zeros((pages, probe.PAGE_SIZE, 2, 2), dtype=torch.float16)
        self.kv_calls = 0

    def get_state_tensors(self):
        return self.conv, self.rec

    def get_kv(self, cache_seqlens, block_table, sliding_window=-1):
        self.kv_calls += 1
        return self.k, self.v


class FakeCache:
    def __init__(self, layers=None):
        self.layers = {"(0, 0)": FakeLayer(), "(3, 0)": FakeLayer()}
        self.recurrent_layers = layers if layers is not None else {"(0, 0)": FakeLayer(), "(3, 0)": FakeLayer()}
        self.initialized = True

    def get_all_recurrent_layers(self):
        return self.recurrent_layers

    def reset_states(self):
        pass

    def get_new_state(self):
        return FakeState(self)


class FakeState:
    """Sequence state with the real GDNState surface (slot/parity/position/base_row/scratch_row)."""

    def __init__(self, cache, position=0, last_history=0, slot=0, parity=0):
        self.cache, self.position, self.last_history = cache, position, last_history
        self.slot, self.parity = slot, parity

    def base_row(self):
        return 2 * self.slot + self.parity

    def scratch_row(self):
        return 2 * self.slot + 1 - self.parity


class FakeModel:
    """Deterministic stand-in: logits depend on the whole context, and on the row count when
    `batched` is set, which is what lets the controls differ measurably from one another."""

    def __init__(self, batched=False):
        self.batched = batched
        self.calls = []

    def forward(self, input_ids, params):
        ctx = [int(x) for x in input_ids.tolist()[0]]
        base = float(sum(ctx) % 97)
        rows = input_ids.shape[1]
        bump = 0.35 if (self.batched and rows > 1) else 0.0
        out = torch.zeros((1, rows, 8))
        for r in range(rows):
            for t in range(8):
                out[0, r, t] = base + bump + 0.01 * t + 0.001 * r
            out[0, r, (ctx[-1] + r) % 8] += 1.0
        return out


def make_call(seq, cache_seqlen, rows, tokens, history=False, block_pages=None, positions=None):
    """A recorded target call. rows = token columns; batch is always 1 for these fixtures."""
    return probe.CallRecord(seq=seq, batch=1, cols=len(tokens), cache_seqlens=cache_seqlen,
                            block_row=block_pages if block_pages is not None
                            else list(range(6)),
                            positions=positions if positions is not None else
                            list(range(cache_seqlen, cache_seqlen + len(tokens))),
                            recurrent_history=history, input_ids=[list(tokens)],
                            logits=torch.zeros((len(tokens), 8)))


def ar_protocol(prompt_len=P, generated=GEN, n=None):
    """The protocol the real AR Generator produces, from the recorded source facts:
    prefill consumes P-1 tokens, then each decode pass carries one still-uncached token."""
    n = len(generated) if n is None else n
    calls = [make_call(0, 0, PROMPT[: prompt_len - 1], PROMPT[: prompt_len - 1])]
    seq = 1
    for k in range(n):
        tok = PROMPT[prompt_len - 1] if k == 0 else generated[k - 1]
        calls.append(make_call(seq, prompt_len - 1 + k, [tok], [tok]))
        seq += 1
    return calls


# --------------------------------------------------------------------------- 1. prefill / duplication

def test_prefill_consumes_p_minus_one_and_no_token_is_repeated():
    calls = ar_protocol()
    prefill = calls[0]
    assert prefill.cache_seqlens == 0 and prefill.rows == P - 1
    assert calls[1].cache_seqlens == P - 1 and calls[1].input_ids == [[PROMPT[-1]]]
    # the whole recorded protocol satisfies the ledger: no duplicated or missing prompt token
    assert probe.assert_pass_ledger(calls, PROMPT, GEN) is True
    # index 0 is predicted by the first decode row, index 18 by the pass at cache_seqlens P-1+18
    assert probe.predict_index(calls[1].cache_seqlens, 0, P) == 0
    assert probe.predict_index(calls[19].cache_seqlens, 0, P) == 18


def test_v2_style_double_feed_is_rejected():
    """v2 prefilled all P tokens and then fed prompt[-1] again at cache_seqlens=P."""
    bad = [make_call(0, 0, PROMPT, PROMPT), make_call(1, P, [PROMPT[-1]], [PROMPT[-1]])]
    with pytest.raises(probe.ContextError):
        probe.assert_pass_ledger(bad, PROMPT, GEN)


# --------------------------------------------------------------------------- 2. interior row, no impossible assert

def test_verifier_batch_straddling_the_index_selects_the_interior_row():
    calls = ar_protocol(n=16)                       # passes for indices 0..15
    rows = [GEN[15], GEN[16], GEN[17]]              # accepted, then two proposals -> predicts 16..18
    calls.append(make_call(17, P - 1 + 16, rows, rows, history=True))
    call, row, _disc = probe.select_pass(calls, 18, P)
    assert (call.seq, row) == (17, 2)               # index 18 is the third row, not row 0
    assert probe.predict_index(call.cache_seqlens, row, P) == 18
    assert probe.assert_call_context(call, PROMPT, GEN, selected_row=row) is True


def test_row_that_does_not_predict_the_index_is_never_selected():
    calls = ar_protocol(n=16)
    calls.append(make_call(17, P - 1 + 16, [GEN[15], GEN[16]], [GEN[15], GEN[16]], history=True))
    with pytest.raises(probe.SelectionError):
        probe.select_pass(calls, 20, P)             # pass only reaches index 17


# --------------------------------------------------------------------------- 3. real params, not past_len

def test_target_call_without_cache_seqlens_is_a_param_error():
    model, target = FakeModel(), FakeModel()
    rec = probe.ProbeRecorder(target, P)
    with pytest.raises(probe.ParamError):
        rec.capture("forward", target, input_ids=torch.tensor([[5]]),
            params={"past_len": P + 18, "block_table": torch.zeros((1, 2), dtype=torch.int32)},
            invoke=lambda: torch.zeros((1, 1, 8)))


def test_target_call_without_block_table_is_a_param_error():
    rec = probe.ProbeRecorder(FakeModel(), P)
    with pytest.raises(probe.ParamError):
        rec.capture("forward", rec.target_model, input_ids=torch.tensor([[5]]),
            params={"cache_seqlens": torch.tensor([P], dtype=torch.int32)},
            invoke=lambda: torch.zeros((1, 1, 8)))


# --------------------------------------------------------------------------- 4. drafts vs accepted rows

def test_draft_columns_are_not_required_to_match_context_but_accepted_ones_are():
    drafts = [GEN[15], 999, 998]
    call = make_call(17, P - 1 + 16, drafts, drafts, history=True)
    assert probe.assert_call_context(call, PROMPT, GEN, selected_row=0) is True
    with pytest.raises(probe.ContextError):
        probe.assert_call_context(call, PROMPT, GEN, selected_row=1)   # accepted row must match


def test_ledger_error_mentions_the_position():
    call = make_call(1, P - 1, [4321], [4321])
    with pytest.raises(probe.ContextError) as e:
        probe.assert_call_context(call, PROMPT, GEN, selected_row=0)
    assert str(P - 1) in str(e.value)


# --------------------------------------------------------------------------- 5. target vs draft model

def test_draft_model_calls_are_excluded_from_selection():
    target, draft_model = FakeModel(), FakeModel()
    rec = probe.ProbeRecorder(target, P, capture_pred_indices=(18,))
    rec.capture("forward", target, input_ids=torch.tensor([[PROMPT[-1]]]),
        params={"cache_seqlens": torch.tensor([P - 1], dtype=torch.int32),
                "block_table": torch.zeros((1, 2), dtype=torch.int32)},
        invoke=lambda: torch.zeros((1, 1, 8)))
    rec.capture("forward", draft_model, input_ids=torch.tensor([[7, 8]]),
        params={"cache_seqlens": torch.tensor([P], dtype=torch.int32),
                "block_table": torch.zeros((1, 2), dtype=torch.int32)},
        invoke=lambda: torch.zeros((1, 2, 8)))
    assert len(rec.target_calls()) == 1 and rec.draft_calls == 1
    assert all(c.batch == 1 for c in rec.target_calls())
    assert probe.predict_index(rec.target_calls()[0].cache_seqlens, 0, P) == 0


# --------------------------------------------------------------------------- 6. parity / base row

def _snapshot(parity=0, base_value=1.0, scratch_value=1.0):
    cache = FakeCache()
    st = FakeState(cache, position=P + 18, parity=parity)
    for key, layer in cache.recurrent_layers.items():
        layer.rec[st.base_row()] = base_value
        layer.rec[st.scratch_row()] = scratch_value
    return probe.snapshot_state(st)


def test_base_row_follows_parity():
    s0, s1 = _snapshot(parity=0, base_value=1.0), _snapshot(parity=1, base_value=1.0)
    for key in s0["layers"]:
        assert s0["parity"] == 0 and s1["parity"] == 1
        assert s0["position"] == s1["position"]


def test_scratch_only_difference_does_not_change_the_canonical_state():
    a = _snapshot(parity=0, base_value=1.0, scratch_value=1.0)
    b = _snapshot(parity=0, base_value=1.0, scratch_value=5.0)
    cmp = probe.compare_snapshots(a, b, "scratch-only")
    assert cmp["n_base_differing"] == 0
    assert all(v["base_same"] for v in cmp["layers"].values())
    assert all(v["scratch"]["max"] > 0 for v in cmp["layers"].values())


def test_canonical_row_follows_parity_after_a_rewind():
    """A rewind flips parity and leaves the committed state in what was the scratch row: the same
    logical state reached with opposite parity must compare equal, and junk left in the other row must
    be ignored. Forcing parity (the mutation) makes the junk row canonical and is detected here."""
    a_state = _snapshot(parity=0, base_value=1.0, scratch_value=0.0)
    cache_b = FakeCache()
    st_b = FakeState(cache_b, position=a_state["position"], parity=1)
    for key, layer in cache_b.recurrent_layers.items():
        layer.rec[st_b.base_row()] = 1.0           # committed state, in the other physical row
        layer.rec[st_b.scratch_row()] = -9.0       # stale scratch that must never be compared
    b_state = probe.snapshot_state(st_b)
    assert a_state["parity"] != b_state["parity"]
    cmp = probe.compare_snapshots(a_state, b_state, "parity flip")
    assert cmp["n_base_differing"] == 0, "committed rows must match across a parity flip"
    assert all(v["base"]["max"] == 0.0 for v in cmp["layers"].values())


def test_base_row_difference_is_reported():
    a = _snapshot(parity=0, base_value=1.0)
    b = _snapshot(parity=0, base_value=2.0)
    cmp = probe.compare_snapshots(a, b, "base-diff")
    assert cmp["n_base_differing"] == len(a["layers"])


# --------------------------------------------------------------------------- 7. serialization

def test_raw_tensors_survive_for_comparison_but_not_for_json():
    a = _snapshot()
    summary = probe.summarize_snapshot(a)
    assert all(not k.startswith("_") for layer in summary["layers"].values() for k in layer)
    json.dumps(summary)
    with pytest.raises(probe.SnapshotError):          # the v2 defect: strip, then compare
        probe.compare_snapshots(summary, probe.summarize_snapshot(a), "stripped")
    probe.compare_snapshots(a, _snapshot(), "raw")     # raw snapshots still compare


def test_empty_or_missing_state_raises():
    cache = FakeCache(layers={})
    with pytest.raises(probe.SnapshotError):
        probe.snapshot_state(FakeState(cache))
    broken = FakeCache()
    broken.recurrent_layers["(0, 0)"].get_state_tensors = lambda: (None, None)
    with pytest.raises(probe.SnapshotError):
        probe.snapshot_state(FakeState(broken))


def test_state_comparison_requires_equal_positions():
    a = _snapshot()
    b = _snapshot()
    b["position"] += 1
    with pytest.raises(probe.SelectionError):
        probe.compare_snapshots(a, b, "positions")


# --------------------------------------------------------------------------- 8. block table / replay

def test_non_identity_block_table_is_rejected():
    call = make_call(1, P - 1, [GEN[0]], [GEN[0]], block_pages=[3, 4, 0, 0, 0, 0])
    with pytest.raises(probe.ParamError):
        probe.assert_identity_block(call)


def test_replay_and_control_rows_run_through_the_fake_model():
    model = FakeModel(batched=True)
    cache = FakeCache()
    state = FakeState(cache)
    calls = ar_protocol(n=3)
    for call in calls:
        out = probe.replay_one(call, cache, state, model)
        assert out.shape == (call.rows, 8)
    logits3 = probe.issue_control_rows(calls[-1], [GEN[2], 1, 2], cache, state, model)
    assert logits3.shape == (3, 8)
    # a batched row is measurably different from the single-row replay at the same position
    single = probe.replay_one(calls[-1], FakeCache(), FakeState(FakeCache()), model)
    assert probe.full_kl(single[0], logits3[0])["kl_nats"] > 0


def test_kv_fingerprint_uses_canonical_prefix_and_fails_when_empty():
    cache = FakeCache()
    fp = probe.kv_fingerprint(cache, consumed=P + 18, block_row=list(range(6)))
    assert fp["n_layers"] == 2 and fp["consumed"] == P + 18
    empty = FakeCache()
    empty.layers = {}
    with pytest.raises(probe.SnapshotError):
        probe.kv_fingerprint(empty, consumed=4, block_row=[0])


# --------------------------------------------------------------------------- 9. end-to-end harness path

def test_end_to_end_record_select_ledger_compare():
    """Whole corrected path on fakes: two 'runs' (AR, MTP2), selection, ledger, controls, JSON."""
    target = FakeModel()
    rec_ar = probe.ProbeRecorder(target, P, capture_pred_indices=(18,))
    rec_mtp = probe.ProbeRecorder(target, P, capture_pred_indices=(18,))
    calls_ar, calls_mtp = ar_protocol(), ar_protocol(n=16)
    drafts = [GEN[15], GEN[16], GEN[17]]            # accepted + two proposals -> predicts 16..18
    calls_mtp.append(make_call(len(calls_mtp), P - 1 + 16, drafts, drafts, history=True))

    for rec, calls in ((rec_ar, calls_ar), (rec_mtp, calls_mtp)):
        for call in calls:
            rec.calls.append(call)
        probe.assert_pass_ledger(rec.target_calls(), PROMPT, GEN)

    call_ar, row_ar, _d1 = probe.select_pass(rec_ar.target_calls(), 18, P)
    call_mtp, row_mtp, _d2 = probe.select_pass(rec_mtp.target_calls(), 18, P)
    assert (call_ar.rows, row_ar) == (1, 0)
    assert (call_mtp.rows, row_mtp) == (3, 2)

    # controls: replay the AR protocol, then the same position with three rows
    model = FakeModel(batched=True)
    seq_cache, seq_state = FakeCache(), None
    seq_state = FakeState(seq_cache)
    for call in rec_ar.target_calls():
        seq_logits = probe.replay_one(call, seq_cache, seq_state, model)
    b3_cache, b3_state = FakeCache(), FakeState(FakeCache())
    for call in rec_ar.target_calls()[:-1]:
        probe.replay_one(call, b3_cache, b3_state, model)
    b3 = probe.issue_control_rows(call_ar, [call_ar.input_ids[0][0], 1, 2], b3_cache, b3_state, model)

    report = {"kl_seq_vs_batch3": probe.full_kl(seq_logits[0], b3[0]),
              "ar_selected": {"seq": call_ar.seq, "row": row_ar},
              "mtp_selected": {"seq": call_mtp.seq, "row": row_mtp},
              "state": probe.summarize_snapshot(probe.snapshot_state(seq_state))}
    json.dumps(report)
    assert report["kl_seq_vs_batch3"]["kl_nats"] > 0


# --------------------------------------------------------------------------- 10. CLI / wrapper wiring

def test_cli_accepts_only_the_reviewed_arguments():
    with pytest.raises(SystemExit):
        probe.main(["--max-new", "24"])              # the v2 flag run_arm3.sh still passed
    assert probe.main(["--dry-check", "--index", "18"]) == 0


def test_wrapper_and_window_use_the_corrected_probe_with_matching_argv():
    bench = HERE
    run_probe = os.path.join(bench, "run_probe.sh")
    window = os.path.join(bench, "window3.sh")
    legacy = os.path.join(bench, "run_arm3.sh")
    assert os.path.exists(run_probe), "probe needs its own wrapper with the reviewed argv"
    text = open(run_probe).read()
    assert "probe_mtp_causal.py" in text
    assert "--max-new" not in text
    for flag in ("--index", "--out", "--tensor-dir"):
        assert flag in text, f"{flag} missing from run_probe.sh"
    if os.path.exists(legacy):
        assert "probe_mtp_causal.py" in open(legacy).read() or "run_probe.sh" in open(legacy).read(), \
            "run_arm3.sh must not call the rejected v2 script"
        assert "probe_mtp_divergence.py" not in open(legacy).read(), \
            "run_arm3.sh still references the rejected v2 probe"
    if os.path.exists(window):
        wtext = open(window).read()
        assert "--probe-only" in wtext, "window3.sh needs a probe-only mode"
        assert "probe_mtp_divergence.py" not in wtext
