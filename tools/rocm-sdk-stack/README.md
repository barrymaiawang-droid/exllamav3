# ROCm 10 port tooling and benchmark harness

Reproducible tooling for running ExLlamaV3 v1.6.4 on the AMD ROCm 10.1 nightly Python stack
(gfx1151, Strix Halo). Nothing here is part of the library build; these files make the
experiment reproducible from a clean machine.

## Stack (pinned from https://rocm.nightlies.amd.com/whl-multi-arch/)

The pins are the ones `torch 2.14.0+rocm10.1.0a20260822` declares in its own wheel METADATA:

| package | version |
|---|---|
| torch | 2.14.0+rocm10.1.0a20260822 (cp312, linux_x86_64) |
| amd-torch-device-gfx1151 / -gfx115x | 2.14.0+rocm10.1.0a20260822 |
| rocm (`[libraries]`, `[devel]`, `[profiler]`) | 10.1.0a20260822 |
| rocm-sdk-core / -libraries / -devel | 10.1.0a20260822 |
| rocm-sdk-device-gfx1151 | 10.1.0a20260822 |
| triton | 3.8.0+git675c5987.rocm10.1.0a20260822 |

The SDK reports itself as `HIP version: 7.16.26332`, `AMD clang version 23.0.0git`; the extension is
built for `gfx1151` by `_rocm_sdk_devel/bin/hipcc`. There is no merged `/opt/rocm` tree: the HIP
runtime comes from the core wheel and the compiler/headers from the devel wheel, which is why
`exllamav3.util.cuda_flags.use_rocm_sdk_devel()` (upstream code) has to point torch at the devel tree.

## Scripts

* `fetch-wheels.sh` — single-owner (`flock`), resumable (`curl -C -`) downloader for the pinned
  wheels; records `sha256` per file. `+` must be passed percent-encoded (`%2B`).
* `install-stack.sh` — creates the experiment venv from a CPython 3.12 that ships `Python.h`, then
  installs the pinned stack from the local wheels plus the exllamav3 runtime/test dependencies at
  the versions the ROCm 7.13 environment used (so tokenization is comparable across runtimes).
* `exl3-rocm10-test-env.sh` — the only environment the build and the harness see: one `exllamav3`
  checkout on `PYTHONPATH`, venv on `PATH`, experiment-local `TORCH_EXTENSIONS_DIR`,
  `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `PYTORCH_ROCM_ARCH=gfx1151`. No ROCm 7 tree is
  reachable and no `LD_PRELOAD` is used.
* `build-ext.sh <instr|clean>` — builds the JIT extension into a variant-specific cache and records
  `torch.version.hip`, the resolved `cpp_extension.ROCM_HOME`, `hipcc --version`, the extension path,
  its sha256 and the counter-binding assertion (clean must NOT have `exl3_dispatch_stats`).
* `identity.py` — prints the running identity, including the ROCm libraries actually mapped by the
  process (`/proc/self/maps`), not just the installed package versions.
* `run-tests.sh` — numerical kernel tests (`tests/kernels/gemm`, `routing`, `quant`, `hadamard`,
  `activation`) with JUnit XML output.
* `../exl3-bench/` — the benchmark/probe harness: `bench_ornith.py` (fixed prompt, ArgmaxSampler,
  cache built before load, 16-token warmup, N x 256 tokens), `run_arm.sh` (one arm plus 1 Hz power
  sampling), `window2.sh` (run driver: exclusive preflight, memory guard, fail-fast, run-specific
  state file and deadline watchdog), `cleanup-test.sh` (single kill implementation, reaps the run's
  own process group), `check_harness.py` (static checks), `closure.py` (top-k distributions and
  dispatch deltas for one short run), `probe_mtp_divergence.py` (per-position top-k for AR / MTP1 /
  MTP2 plus recurrent-state fingerprints at a fixed prefix).

## Reproduction

```bash
~/x3-exp/rocm10/fetch-wheels.sh <pinned wheel names...>
~/x3-exp/rocm10/install-stack.sh
~/x3-exp/rocm10/build-ext.sh clean
~/x3-exp/rocm10/build-ext.sh instr
~/x3-exp/rocm10/run-tests.sh
```

`window2.sh` refuses to run real arms without `--execute`, exits every mock branch explicitly, and
requires an idle GPU (`GTT` below 2 GiB, no model process, ports free) plus a `MemAvailable` floor
before each arm. Do not run two arms concurrently on a unified-memory box.
