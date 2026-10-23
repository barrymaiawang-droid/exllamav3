#!/usr/bin/env python3
"""Ornith-1.5-122B EXL3 benchmark/decode-proof harness for the isolated v1.6.4 build.

Mirrors the production load path (~/ornith122-serve.py, read-only reference):
Config.from_directory -> Model.from_config -> Cache(max_num_tokens, max_history) ->
Tokenizer.from_config -> model.load(), then the embedded MTP drafter as
component="mtp" with its own cache and Generator(draft_model=..., draft_cache=...,
num_draft_tokens=...).

Fixed benchmark settings (identical across arms/versions):
  * one tokenized prompt, encoded once, reused for every job
  * ArgmaxSampler
  * 16-token warmup job, discarded
  * --reps jobs of --max-new tokens
  * Cache(max_num_tokens=--ctx, max_history>=num_draft_tokens) built before model.load

Timing boundaries reported per rep (perf_counter around enqueue..eos):
  wall_s      : whole job (prefill + decode)
  ttft_s      : enqueue -> first generated token (job.time_first_token)
  decode_s    : first -> last generated token (job.time_last_token - time_first_token)
  decode_tps  : (new_tokens - 1) / decode_s
Generated ids are read back from job.sequences[0].sequence_ids (never re-tokenized
text), so counts/metrics are the real sampled tokens.

This file does not modify the model, the source tree, or any production artefact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_DIR", os.path.expanduser(
        "~/models/Ornith-1.5-122B-EXL3-EAQv1-4-bpw")))
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--num-draft", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prompt", default="Write a short paragraph about the ocean.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--profile-steps", type=int, default=0)
    ap.add_argument("--profile-out", default=None)
    ap.add_argument("--count-reconstruct", action="store_true",
                    help="count LinearEXL3.reconstruct_hgemm calls (instrumented runs only)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="validate imports/arguments/version/clock/EOS plumbing and exit; never loads a model")
    args = ap.parse_args()

    if args.selfcheck:
        # Runs before any model, cache or CUDA tensor exists: import identity, argument defaults, version
        # metadata, the reconstruct hook, and the timing arithmetic with synthetic clocks.
        import json as _json
        from exllamav3.version import __version__ as _v
        from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
        from exllamav3.generator.sampler import ArgmaxSampler
        checks = {}
        checks["version"] = _v
        checks["ext_file"] = __import__("exllamav3.ext", fromlist=["exllamav3_ext"]).exllamav3_ext.__file__
        checks["classes"] = all(x is not None for x in (Cache, Config, Generator, Job, Model, Tokenizer))
        checks["sampler"] = ArgmaxSampler.__name__
        checks["defaults"] = dict(ctx=args.ctx, max_new=args.max_new, warmup=args.warmup, reps=args.reps,
                                  prompt=args.prompt, num_draft=args.num_draft)
        # timing arithmetic: t0_wall/time_first/time_last are all time.time() values, wall is perf_counter
        _t0_wall, _first, _last = 1000.0, 1005.5, 1010.5
        ttft = _first - _t0_wall
        dec = _last - _first
        checks["clock"] = dict(ttft=ttft, decode=dec, ok=(abs(ttft - 5.5) < 1e-9 and abs(dec - 5.0) < 1e-9))
        # EOS plumbing: the terminal emission, not a Job attribute
        src = open(__file__).read()
        bad_needle = "job." + "eos_reason"      # built at runtime so this check does not match itself
        checks["eos_from_terminal"] = 'terminal.get("eos_reason")' in src and bad_needle not in src
        expected = dict(ctx=8192, max_new=256, warmup=16, reps=2,
                        prompt="Write a short paragraph about the ocean.", num_draft=0)
        # real validation, no vacuous clause: the documented benchmark settings must be what argparse parsed
        assert checks["classes"], "exllamav3 classes missing"
        assert checks["clock"]["ok"], "timing arithmetic wrong"
        assert checks["eos_from_terminal"], "EOS must come from the terminal emission"
        assert isinstance(checks["version"], str) and checks["version"], "version metadata missing"
        assert checks["defaults"] == expected, f"argument defaults drifted: {checks['defaults']}"
        checks["limited_preflight_note"] = ("imports/arguments/version/clock algebra/EOS plumbing only; "
                                           "does not validate runtime EOS or kernel behaviour")
        print(_json.dumps({"selfcheck": "OK", **checks}, indent=2))
        return 0

    # Reconstruct-fallback counter is opt-in: it monkeypatches a Python method, so it is enabled only for
    # instrumented runs and the timing arms keep the library's own call path untouched.
    reconstruct_calls = {"n": 0, "enabled": bool(args.count_reconstruct)}
    if args.count_reconstruct:
        from exllamav3.modules.quant.exl3 import LinearEXL3
        _orig_reconstruct = LinearEXL3.reconstruct_hgemm

        def _counting_reconstruct(self, x, out_dtype):
            reconstruct_calls["n"] += 1
            return _orig_reconstruct(self, x, out_dtype)

        LinearEXL3.reconstruct_hgemm = _counting_reconstruct

    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3.generator.sampler import ArgmaxSampler

    rec = {
        "label": args.label,
        "argv": sys.argv,
        "host": os.uname().nodename,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": sys.version.split()[0],
        "model_dir": args.model,
        "ctx": args.ctx,
        "num_draft_tokens": args.num_draft,
        "max_new": args.max_new,
        "warmup": args.warmup,
        "reps": args.reps,
        "prompt": args.prompt,
        "sampler": "ArgmaxSampler",
        "env": {k: os.environ.get(k) for k in ("TORCH_EXTENSIONS_DIR", "PYTHONPATH", "PYTORCH_ROCM_ARCH")},
        "reps_out": [],
    }

    import exllamav3
    from exllamav3 import ext as exl_ext
    rec["exllamav3_file"] = exllamav3.__file__
    from exllamav3.version import __version__ as exl_version
    rec["exllamav3_version"] = exl_version
    rec["ext_file"] = exl_ext.exllamav3_ext.__file__
    rec["ext_stats_available"] = hasattr(exl_ext.exllamav3_ext, "exl3_dispatch_stats")

    t0 = time.perf_counter()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    # Cache is built before model.load(), as production does
    cache = Cache(model, max_num_tokens=args.ctx, max_history=max(3, args.num_draft))
    tok = Tokenizer.from_config(config)
    model.load()
    rec["target_load_s"] = time.perf_counter() - t0
    rec["architecture"] = config.architecture

    draft = draft_cache = None
    if args.num_draft:
        t0 = time.perf_counter()
        draft = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft, max_num_tokens=args.ctx, max_history=max(3, args.num_draft))
        draft.load()
        rec["draft_load_s"] = time.perf_counter() - t0

    kw = {}
    if draft is not None:
        kw = dict(draft_model=draft, draft_cache=draft_cache, num_draft_tokens=args.num_draft)
    gen = Generator(model, cache, tok, **kw)

    ids = tok.encode(args.prompt, encode_special_tokens=True)
    rec["prompt_token_ids"] = [int(x) for x in ids[0].tolist()]
    rec["prompt_tokens"] = int(ids.shape[-1])

    def stats():
        if rec["ext_stats_available"]:
            return exl_ext.exllamav3_ext.exl3_dispatch_stats()
        return None

    def run(n_new: int, tag: str) -> dict:
        job = Job(
            input_ids=ids,
            max_new_tokens=n_new,
            min_new_tokens=1,
            sampler=ArgmaxSampler(),
            decode_special_tokens=True,
        )
        before = stats()
        rec_before = reconstruct_calls["n"]
        t0 = time.perf_counter()
        t0_wall = time.time()          # job.time_first/last_token are time.time() values
        serial = gen.enqueue(job)
        done = False
        terminal = {}
        while not done:
            for r in gen.iterate():
                if r.get("serial") != serial:
                    continue
                if r.get("stage") == "error":
                    raise r["error"]
                if r.get("eos"):
                    terminal = dict(r)   # the eos emission carries eos_reason (job has no such attribute)
                    done = True
        wall = time.perf_counter() - t0
        after = stats()

        seq = job.sequences[0].sequence_ids.torch()
        in_len = int(ids.shape[-1])
        gen_ids = [int(x) for x in seq[0, in_len:].tolist()]
        first = job.time_first_token
        last = job.time_last_token
        ttft = (first - t0_wall) if first else None
        decode_s = (last - first) if (first and last) else None
        n_dec = max(0, len(gen_ids) - 1)
        out = {
            "tag": tag,
            "max_new": n_new,
            "generated_ids": gen_ids,
            "generated_count": len(gen_ids),
            "job_new_tokens": int(job.new_tokens),
            "eos_reason": terminal.get("eos_reason"),
            "ended_by_max_new": int(terminal.get("eos_reason") == "max_new_tokens"),
            "n_decode_intervals": n_dec,
            "wall_s": wall,
            "ttft_s": ttft,
            "decode_s": decode_s,
            "decode_tps": (n_dec / decode_s) if (decode_s and decode_s > 0) else None,
            "wall_tps": len(gen_ids) / wall if wall > 0 else None,
            "reconstruct_calls": reconstruct_calls["n"] - rec_before,
            "stats_before": before,
            "stats_after": after,
        }
        # delta of the dispatch counters for this rep
        if before and after:
            out["stats_delta"] = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        # consistency: the observed ids must be the job's tokens, and the timing windows must be sane
        errs = []
        if len(gen_ids) != int(job.new_tokens):
            errs.append(f"generated ids {len(gen_ids)} != job.new_tokens {int(job.new_tokens)}")
        for name, v in (("ttft_s", ttft), ("decode_s", decode_s)):
            if v is not None and not (0.0 <= v <= wall + 1.0):
                errs.append(f"{name} {v:.6f} outside [0, wall+1s={wall + 1.0:.6f}]")
        if decode_s is not None and ttft is not None and decode_s + ttft > wall + 1.0:
            errs.append("ttft + decode_s exceeds wall")
        out["consistency_errors"] = errs
        return out

    # warmup, discarded
    warm = run(args.warmup, "warmup")
    rec["warmup"] = {k: warm[k] for k in ("generated_count", "wall_s", "decode_s", "decode_tps")}

    for i in range(args.reps):
        r = run(args.max_new, f"rep{i + 1}")
        rec["reps_out"].append(r)

    if args.profile_steps:
        # Real executed-kernel trace (names + counts as the GPU saw them), to complement the host-side
        # dispatch counters: a counter proves a launch was issued, the trace proves the kernel ran.
        from torch.profiler import profile, ProfilerActivity
        try:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                run(args.profile_steps, "profile")
            kernels = {}
            for evt in prof.key_averages():
                key = evt.key
                if "exl3" in key or "rocm" in key or "Moe" in key:
                    kernels[key] = {
                        "count": evt.count,
                        "self_device_time_us": getattr(evt, "self_device_time_total", None),
                    }
            rec["kernel_trace"] = kernels
            if args.profile_out:
                prof.key_averages().table(sort_by="self_device_time_total", row_limit=200)
                with open(args.profile_out, "w") as f:
                    f.write(prof.key_averages().table(sort_by="self_device_time_total", row_limit=400))
        except Exception as exc:      # a profiler failure must not lose the measurements above
            rec["kernel_trace_error"] = f"{type(exc).__name__}: {exc}"

    rec["consistency_errors"] = [e for r in rec["reps_out"] for e in r.get("consistency_errors", [])]
    rec["reconstruct_calls_total"] = reconstruct_calls["n"]
    rec["reconstruct_counter_enabled"] = reconstruct_calls["enabled"]
    if rec["ext_stats_available"]:
        rec["stats_final"] = stats()

    text = json.dumps(rec, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    if rec["consistency_errors"]:
        print("CONSISTENCY FAILURE: " + "; ".join(rec["consistency_errors"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
