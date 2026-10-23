#!/usr/bin/env python3
"""Strict mutation check: each reintroduced defect must make the *intended* test fail.

A kill is counted only when pytest's JUnit XML contains that exact test id with a `<failure>` (an
assertion failure in the test itself). Collection errors, import errors, skips, unexpected passes and
failures in some other test are all reported as NOT DETECTED, never as a kill.
"""
import os, shutil, subprocess, sys, tempfile
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))

# (description, file, old, new, test module, test function)
MUTATIONS = [
    ("off-by-one ledger (duplicated final prompt token)", "probe_mtp_causal.py",
     "pos = call.cache_seqlens + j", "pos = call.cache_seqlens + j + 1",
     "test_probe_mtp_causal.py", "test_prefill_consumes_p_minus_one_and_no_token_is_repeated"),
    ("row 0 always selected (interior verifier row ignored)", "probe_mtp_causal.py",
     "        return call, row, discarded", "        return call, 0, discarded",
     "test_probe_mtp_causal.py", "test_verifier_batch_straddling_the_index_selects_the_interior_row"),
    ("past_len accepted instead of cache_seqlens", "probe_mtp_causal.py",
     'for key in ("cache_seqlens", "block_table"):', 'for key in ("block_table",):',
     "test_probe_mtp_causal.py", "test_target_call_without_cache_seqlens_is_a_param_error"),
    ("scratch row treated as canonical (parity ignored)", "probe_mtp_causal.py",
     "    slot, parity = int(state.slot), int(state.parity)",
     "    slot, parity = int(state.slot), 0",
     "test_probe_mtp_causal.py", "test_canonical_row_follows_parity_after_a_rewind"),
    ("snapshot stripped before comparison", "probe_mtp_causal.py",
     '        for side in ("_conv", "_base", "_scratch"):\n            if side not in ea or side not in eb:\n                raise SnapshotError(f"{label}: layer {key} lost raw tensor {side} before comparison")',
     '        for side in ("_conv", "_base", "_scratch"):\n            if side not in ea or side not in eb:\n                continue',
     "test_probe_mtp_causal.py", "test_raw_tensors_survive_for_comparison_but_not_for_json"),
    ("whole KV buffers hashed instead of the block-table prefix", "probe_mtp_causal.py",
     "    kk, vv = gather_kv_prefix(k, v, int(consumed), block_row)",
     "    kk, vv = k, v",
     "test_full_driver.py", "test_kv_gather_ignores_unused_storage_and_honours_block_order"),
    ("conv tail hashed as committed history", "probe_mtp_causal.py",
     "        conv_slice = conv[slot][:, :k_size]", "        conv_slice = conv[slot]",
     "test_full_driver.py", "test_conv_canonical_ignores_speculative_tail_and_detects_committed_change"),
    ("replay self-check reported but not enforced", "probe_mtp_causal.py",
     '        raise ProbeError("replay self-check failed (logits_exact=%s state_equal=%s); refusing to "\n                         "report causal numbers" % (selfcheck_logits, selfcheck_state))',
     '        rec.setdefault("selfcheck_would_fail", []).append([selfcheck_logits, selfcheck_state])',
     "test_full_driver.py", "test_false_replay_selfcheck_fails_nonzero_and_keeps_artifacts"),
    ("control lifecycle left outside inference mode", "probe_mtp_causal.py",
     "    with torch.inference_mode():\n        _, seq_entry, seq_kv, seq_logits, replayed = replay_to_selected(",
     "    if True:\n        _, seq_entry, seq_kv, seq_logits, replayed = replay_to_selected(",
     "test_full_driver.py", "test_control_lifecycle_runs_under_inference_mode_with_real_inference_tensors"),
    ("control positions expanded to a token grid", "probe_mtp_causal.py",
     "        params[\"positions\"] = torch.tensor(list(call.positions), dtype=torch.int32)",
     "        params[\"positions\"] = torch.tensor([list(call.positions)], dtype=torch.int32)",
     "test_full_driver.py", "test_control_lifecycle_runs_under_inference_mode_with_real_inference_tensors"),
    ("wrapper argv back to --max-new (v2)", "run_probe.sh",
     '  --index "$IDX" \\', '  --max-new 24 --index "$IDX" \\',
     "test_wiring.py", "test_run_probe_argv_sink"),
]

COPY = ("probe_mtp_causal.py", "test_probe_mtp_causal.py", "test_full_driver.py", "test_wiring.py",
        "run_probe.sh", "cleanup-run.sh", "run_arm3.sh", "window3.sh")


def junit_outcome(xml_path, module, func):
    """('failure'|'error'|'skipped'|'passed'|'absent', detail) for one test id."""
    if not os.path.exists(xml_path):
        return "absent", "no JUnit XML produced"
    tree = ET.parse(xml_path)
    for tc in tree.iter("testcase"):
        name = tc.get("name", "")
        classname = tc.get("classname", "")
        if name == func and module.replace(".py", "") in classname or name == func:
            for child in tc:
                if child.tag in ("failure", "error", "skipped"):
                    return child.tag, (child.get("message") or "")[:200]
            return "passed", ""
    return "absent", "test id not collected"


def main():
    detected, misses = 0, []
    for desc, fname, old, new, module, func in MUTATIONS:
        d = tempfile.mkdtemp(prefix="mut-")
        for f in COPY:
            src = os.path.join(HERE, f)
            if os.path.exists(src):
                shutil.copy(src, d)
        path = os.path.join(d, fname)
        text = open(path).read()
        if old not in text:
            misses.append((desc, "SETUP-MISS: pattern not found"))
            print(f"NOT DETECTED  {desc}  (setup: pattern not found)")
            shutil.rmtree(d, ignore_errors=True)
            continue
        open(path, "w").write(text.replace(old, new, 1))
        xml = os.path.join(d, "junit.xml")
        r = subprocess.run([sys.executable, "-m", "pytest", module, "-q", "-p", "no:cacheprovider",
                            f"--junitxml={xml}", "-k", func],
                           cwd=d, capture_output=True, text=True)
        outcome, detail = junit_outcome(xml, module, func)
        killed = outcome == "failure"         # an assertion failure in the intended test, nothing else
        if killed:
            detected += 1
        else:
            misses.append((desc, f"{outcome}: {detail} rc={r.returncode}"))
        print(f"{'DETECTED' if killed else 'NOT DETECTED'}  {desc}  [{module}::{func} -> {outcome}]")
        with open(os.path.join(d, "pytest.log"), "w") as f:
            f.write(r.stdout + r.stderr)
        shutil.rmtree(d, ignore_errors=True)
    print(f"strict mutations detected: {detected}/{len(MUTATIONS)}")
    for desc, why in misses:
        print(f"  missed: {desc} ({why})")
    return 0 if detected == len(MUTATIONS) else 1


if __name__ == "__main__":
    sys.exit(main())
