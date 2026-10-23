#!/usr/bin/env python3
"""Step-1 closure probe: bounded instrumentation only (no timing claims).

Per mode (AR / MTP2), one short job (--max-new, default 24) with:
  * ROCPROFILER_REGISTER_ENABLED=1 exported BEFORE torch/exllamav3 import, so torch.profiler device events
    are registered (the library only sets it by default via os.environ.setdefault)
  * Job(return_logits=True)  -> per-step target logits, to compare AR vs the MTP verifier at the first
    divergent position
  * Generator(record_draft_stats=True) -> job.draft_stats / accepted_draft_tokens
  * a shape tracer around LinearEXL3.forward -> which modules are invoked with which row counts, to attribute
    the unexplained gemm_rdna_m7 launches instead of guessing
  * EXL3_DISPATCH_STATS=1 counters for the same job

Writes one JSON per mode under $RAW/<run_id>-<label>.json. Never used for throughput numbers.
"""
import os

os.environ["ROCPROFILER_REGISTER_ENABLED"] = "1"      # must precede the torch import below

import argparse
import collections
import json
import sys
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.version import __version__ as exl_version


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/Ornith-1.5-122B-EXL3-EAQv1-4-bpw"))
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--num-draft", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--label", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--raw", default=os.path.expanduser("~/x3-exp/rocm713/raw"))
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    prompt = "Write a short paragraph about the ocean."
    rec = {
        "label": args.label, "run_id": args.run_id, "num_draft": args.num_draft,
        "max_new": args.max_new, "prompt": prompt, "exllamav3_version": exl_version,
        "rocmprofiler_register": os.environ.get("ROCPROFILER_REGISTER_ENABLED"),
        "ext_file": ext.__file__,
        "torch": torch.__version__, "hip": torch.version.hip,
    }

    # which module sees which row count: attributes per-token vs per-job launches
    shapes = collections.Counter()
    orig_forward = LinearEXL3.forward

    def traced_forward(self, x, params, out_dtype=None):
        rows = x.numel() // x.shape[-1]
        shapes[(getattr(self, "key", "?"), int(rows))] += 1
        return orig_forward(self, x, params, out_dtype)

    LinearEXL3.forward = traced_forward

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.ctx, max_history=max(3, args.num_draft))
    tok = Tokenizer.from_config(config)
    model.load()
    draft = draft_cache = None
    if args.num_draft:
        draft = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft, max_num_tokens=args.ctx, max_history=max(3, args.num_draft))
        draft.load()
    kw = dict(record_draft_stats=bool(args.num_draft))
    if draft is not None:
        kw.update(draft_model=draft, draft_cache=draft_cache, num_draft_tokens=args.num_draft)
    gen = Generator(model, cache, tok, **kw)

    ids = tok.encode(prompt, encode_special_tokens=True)
    rec["prompt_token_ids"] = [int(x) for x in ids[0].tolist()]

    job = Job(input_ids=ids, max_new_tokens=args.max_new, min_new_tokens=1,
              sampler=ArgmaxSampler(), decode_special_tokens=True, return_top_tokens=5)

    ext.exl3_dispatch_stats_reset()
    before = ext.exl3_dispatch_stats()

    profiler = None
    profiler_error = None
    if args.profile:
        try:
            from torch.profiler import profile, ProfilerActivity
            profiler = profile(activities=[ProfilerActivity.CUDA])
            profiler.__enter__()
        except Exception as exc:
            profiler = None
            profiler_error = f"{type(exc).__name__}: {exc}"
            print(f"profiler unavailable: {profiler_error}", flush=True)

    t0 = time.perf_counter()
    serial = gen.enqueue(job)
    # held top-k buffers are emitted then cleared, so every emitted row is one generated step, in order
    topk_steps = []
    done = False
    while not done:
        for r in gen.iterate():
            if r.get("serial") != serial:
                continue
            if r.get("stage") == "error":
                raise r["error"]
            if "top_k_tokens" in r:
                tk = r["top_k_tokens"]
                tp = r["top_k_probs"]
                for i in range(tk.shape[0]):
                    topk_steps.append({
                        "top5": [int(x) for x in tk[i, 0].tolist()],
                        "top5_prob": [round(float(x), 6) for x in tp[i, 0].tolist()],
                    })
            if r.get("eos"):
                done = True
    wall = time.perf_counter() - t0

    if profiler_error:
        rec["kernel_trace_error"] = profiler_error
    if profiler is not None:
        try:
            profiler.__exit__(None, None, None)
        except Exception as exc:
            rec["kernel_trace_error"] = f"{type(exc).__name__}: {exc}"
            profiler = None
        kernels = {}
        for evt in profiler.key_averages():
            if evt.count and ("exl3" in evt.key.lower() or "Moe" in evt.key or "gemv" in evt.key.lower()):
                kernels[evt.key] = {"count": evt.count,
                                    "device_time_us": getattr(evt, "self_device_time_total", None)}
        rec["kernel_trace"] = kernels
        rec["kernel_trace_nonempty"] = bool(kernels)
        with open(f"{args.raw}/{args.run_id}-{args.label}.proftable.txt", "w") as f:
            f.write(profiler.key_averages().table(sort_by="self_device_time_total", row_limit=400))

    after = ext.exl3_dispatch_stats()
    seq = job.sequences[0].sequence_ids.torch()
    gen_ids = [int(x) for x in seq[0, ids.shape[-1]:].tolist()]

    tops = []
    for i, row in enumerate(topk_steps):
        tops.append({
            "step": i, "argmax": row["top5"][0], "top5": row["top5"],
            "top5_prob": row["top5_prob"],
            "margin_top1_top2": round(row["top5_prob"][0] - row["top5_prob"][1], 6),
        })

    rec.update({
        "generated_ids": gen_ids,
        "generated_count": len(gen_ids),
        "job_new_tokens": int(job.new_tokens),
        "wall_s": wall,
        "counter_delta": {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)},
        "shape_histogram": {f"{k}|rows={r}": c for (k, r), c in shapes.most_common()},
        "rows7_modules": {k: c for (k, r), c in shapes.items() if r == 7},
        "top5_per_step": tops,
        "draft_stats": [list(map(int, s)) if isinstance(s, tuple) else s for s in getattr(job, "draft_stats", [])],
        "accepted_draft_tokens": int(getattr(job, "accepted_draft_tokens", 0)),
    })

    out = f"{args.raw}/{args.run_id}-{args.label}.json"
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({
        "label": args.label, "gen": rec["generated_count"], "counters": rec["counter_delta"],
        "rows7": rec["rows7_modules"], "trace_events": len(rec.get("kernel_trace") or {}),
        "accepted_drafts": rec["accepted_draft_tokens"], "draft_rounds": len(rec["draft_stats"]),
        "first_ids": gen_ids[:20],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
