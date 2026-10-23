#!/usr/bin/env python3
"""Causal probe for the AR vs MTP verify divergence at generated index 18 (greedy).

Records, on one loaded model, per generated position:
  * the target's top-5 tokens and probabilities, for AR (1-row forward), MTP1 (2-row verify) and
    MTP2 (3-row verify);
  * the generated ids, so the identical-prefix claim is checkable rather than assumed;
  * a hash of every GDN/linear-attention layer's recurrent state (conv ring + scan state) after a
    run of exactly N_ALIGN tokens, in each mode.

The state hash answers one question the distributions cannot: after the SAME accepted token prefix,
is the recurrent state the same in AR (incremental single-token updates) and in MTP2 (speculative
passes plus rewind replays)? If the states differ, the divergence is cache/state; if they are equal,
the divergence is the batched forward arithmetic itself.

No speculative state is mutated for the AR arms (num_draft=0), and no rewinds happen there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

N_ALIGN = 18          # the prefix length both modes agree on (Step-1 closure: first divergence at 18)
TOPK = 5


def state_fingerprint(cache) -> dict:
    """sha256 + summary stats of every recurrent layer state the cache holds."""
    out = {}
    try:
        layers = cache.get_all_recurrent_layers()
    except Exception as exc:                     # no recurrent layers / API moved
        return {"error": f"{type(exc).__name__}: {exc}"}
    for key, st in sorted(layers.items(), key=lambda kv: str(kv[0])):
        try:
            tensors = st.get_state_tensors()
        except Exception as exc:
            out[str(key)] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        entry = {}
        for name, t in zip(("conv", "scan"), tensors):
            if t is None:
                entry[name] = None
                continue
            tt = t.detach()
            entry[name] = {
                "shape": list(tt.shape),
                "dtype": str(tt.dtype),
                "sha256": hashlib.sha256(tt.contiguous().float().cpu().numpy().tobytes()).hexdigest(),
                "absmax": float(tt.abs().max()),
                "absmean": float(tt.abs().float().mean()),
            }
        out[str(key)] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/Ornith-1.5-122B-EXL3-EAQv1-4-bpw"))
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--prompt", default="Write a short paragraph about the ocean.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3.generator.sampler import ArgmaxSampler
    from exllamav3.version import __version__ as exl3_version
    from exllamav3 import ext as exl_ext

    rec = {"label": "mtp-causal-probe", "argv": sys.argv, "torch": torch.__version__,
           "hip": torch.version.hip, "exllamav3_version": exl3_version,
           "exllamav3_file": os.path.realpath(__import__("exllamav3").__file__),
           "ext_file": os.path.realpath(exl_ext.exllamav3_ext.__file__),
           "n_align": N_ALIGN, "topk": TOPK, "runs": []}

    t0 = time.perf_counter()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    model.load()
    rec["target_load_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    draft = Model.from_config(config, component="mtp")
    draft.load()
    rec["draft_load_s"] = round(time.perf_counter() - t0, 3)

    ids = tok.encode(args.prompt, encode_special_tokens=True)
    prompt_ids = [int(x) for x in ids[0].tolist()]
    rec["prompt_token_ids"] = prompt_ids
    rec["prompt_tokens"] = len(prompt_ids)

    def run_mode(label: str, num_draft: int, max_new: int, want_topk: bool) -> dict:
        cache = Cache(model, max_num_tokens=args.ctx, max_history=max(3, num_draft))
        if num_draft:
            draft_cache = Cache(draft, max_num_tokens=args.ctx, max_history=max(3, num_draft))
            kw = dict(draft_model=draft, draft_cache=draft_cache, num_draft_tokens=num_draft,
                      record_draft_stats=True)
        else:
            draft_cache = None
            kw = {}
        gen = Generator(model, cache, tok, **kw)
        job = Job(input_ids=ids, max_new_tokens=max_new, min_new_tokens=1, sampler=ArgmaxSampler(),
                  decode_special_tokens=True, return_top_tokens=TOPK if want_topk else 0)
        tops = []
        terminal = {}
        serial = gen.enqueue(job)
        done = False
        while not done:
            for r in gen.iterate():
                if r.get("serial") != serial:
                    continue
                if r.get("stage") == "error":
                    raise r["error"]
                if "top_k_tokens" in r:
                    tk, tp = r["top_k_tokens"], r["top_k_probs"]
                    for i in range(tk.shape[0]):
                        tops.append({"top": [int(x) for x in tk[i, 0].tolist()],
                                     "prob": [round(float(x), 6) for x in tp[i, 0].tolist()]})
                if r.get("eos"):
                    terminal = dict(r)
                    done = True
        seq = job.sequences[0].sequence_ids.torch()
        gen_ids = [int(x) for x in seq[0, len(prompt_ids):].tolist()]
        out = {
            "label": label, "num_draft": num_draft, "max_new": max_new,
            "generated_ids": gen_ids, "generated_count": len(gen_ids),
            "job_new_tokens": int(job.new_tokens),
            "eos_reason": terminal.get("eos_reason"),
            "top_steps": tops,
            "draft_stats": [list(map(int, x)) for x in getattr(job, "draft_stats", []) or []],
            "state_fingerprint": state_fingerprint(cache),
        }
        del gen, job, cache, draft_cache
        return out

    # ---- distributions: AR vs MTP1 vs MTP2, same prompt, greedy
    for label, nd in (("ar", 0), ("mtp1", 1), ("mtp2", 2)):
        r = run_mode(label, nd, args.max_new, True)
        print(f"[probe] {label}: {r['generated_count']} tokens, eos={r['eos_reason']}", flush=True)
        rec["runs"].append(r)

    # ---- state alignment: stop exactly at the agreed prefix so the accepted tokens are identical
    for label, nd in (("ar-align", 0), ("mtp2-align", 2)):
        r = run_mode(label, nd, N_ALIGN, False)
        print(f"[probe] {label}: {r['generated_count']} tokens", flush=True)
        rec["runs"].append(r)

    # ---- compare
    by = {r["label"]: r for r in rec["runs"]}
    ar = by["ar"]["generated_ids"]
    cmp = {}
    for other in ("mtp1", "mtp2"):
        o = by[other]["generated_ids"]
        first = next((i for i, (a, b) in enumerate(zip(ar, o)) if a != b), None)
        cmp[f"first_id_mismatch_vs_ar_{other}"] = first if first is not None else (
            "none within %d" % min(len(ar), len(o)))
    # distribution comparison at every position both modes produced, incl. winning-token prob gap
    dist = {}
    for other in ("mtp1", "mtp2"):
        rows = []
        a_tops = by["ar"]["top_steps"]
        o_tops = by[other]["top_steps"]
        for i in range(min(len(a_tops), len(o_tops))):
            a, o = a_tops[i], o_tops[i]
            if a["top"] == o["top"]:
                continue
            pa = {t: p for t, p in zip(a["top"], a["prob"])}
            po = {t: p for t, p in zip(o["top"], o["prob"])}
            union = sorted(set(pa) | set(po), key=lambda t: -max(pa.get(t, 0.0), po.get(t, 0.0)))
            import math
            lp = [math.log(max(pa.get(t, 1e-12), 1e-12)) for t in union[:8]]
            lq = [math.log(max(po.get(t, 1e-12), 1e-12)) for t in union[:8]]
            rows.append({"pos": i, "ar_top": a["top"], "ar_prob": a["prob"],
                         "other_top": o["top"], "other_prob": o["prob"],
                         "kl_ar_to_other_nats": round(sum(p * (lp[j] - lq[j]) for j, p in
                                                          enumerate([pa.get(t, 0.0) for t in union[:8]])), 6),
                         "argmax_differs": a["top"][0] != o["top"][0]})
        dist[other] = rows
    rec["comparison"] = {"ids": cmp, "distribution_diffs": dist}

    # state equality at the identical prefix
    sa = by["ar-align"]["state_fingerprint"]
    sm = by["mtp2-align"]["state_fingerprint"]
    diffs = {}
    for k in sorted(set(sa) | set(sm)):
        if sa.get(k) in (None, {}) or sm.get(k) in (None, {}):
            continue
        a, b = sa[k], sm[k]
        for part in ("conv", "scan"):
            av, bv = a.get(part), b.get(part)
            if not av or not bv:
                continue
            diffs[f"{k}.{part}"] = {
                "same_sha256": av["sha256"] == bv["sha256"],
                "absmax_ar": av["absmax"], "absmax_mtp2": bv["absmax"],
                "absmean_ar": av["absmean"], "absmean_mtp2": bv["absmean"],
            }
    rec["state_at_prefix"] = {"layers_compared": len(diffs), "entries": diffs,
                              "n_differing": sum(1 for v in diffs.values() if not v["same_sha256"]),
                              "prefix_ids_equal": by["ar-align"]["generated_ids"] ==
                                                  by["mtp2-align"]["generated_ids"]}

    text = json.dumps(rec, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
