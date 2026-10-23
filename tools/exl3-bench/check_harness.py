#!/usr/bin/env python3
"""Offline preflight for the Ornith benchmark harness (no torch, no GPU, no model).

Checks the four review findings that invalidated the first attempt, plus the window driver's
safety properties. Pure source inspection: it can run while another service owns the GPU.
"""
import ast
import pathlib
import sys

EXP = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / "x3-exp/rocm713")
BENCH = (EXP / "bench_ornith.py").read_text()
ARM = (EXP / "run_arm.sh").read_text()
WIN = (EXP / "window2.sh").read_text()
CLEANUP = (EXP / "cleanup-test.sh").read_text()
TEST = (pathlib.Path.home() / "exl3-rocm10-src/tests/kernels/gemm/test_exl3_ornith_mul1_shapes.py").read_text()

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


ast.parse(BENCH)

# 1. no mixed clocks: t0_wall (time.time) against job.time_first/last_token (time.time), perf_counter only for wall
check("clock: t0_wall captured", "t0_wall = time.time()" in BENCH)
check("clock: ttft uses t0_wall", "ttft = (first - t0_wall)" in BENCH)
check("clock: decode uses first..last", "decode_s = (last - first)" in BENCH)
check("clock: no perf_counter against job timestamps",
      "first - t0)" not in BENCH and "(last - t0)" not in BENCH)

# 2. version metadata without the __init__ attribute
check("version: imported from exllamav3.version",
      "from exllamav3.version import __version__" in BENCH)
check("version: no exllamav3.__version__ access", "exllamav3.__version__" not in BENCH)

# 3. EOS from the terminal emission
check("eos: terminal event captured", 'terminal = dict(r)' in BENCH)
check("eos: reason from terminal", 'terminal.get("eos_reason")' in BENCH)
check("eos: no Job attribute guess", "job.eos_reason" not in BENCH)

# 4. lm_head tail reference (last columns, not only the head)
check("lm_head: tail reference present", "lm_head tail" in TEST)
check("lm_head: last tiles decoded", "p[:, tail:]" in TEST)
check("lm_head: last columns compared", "C[:, tail * 16:]" in TEST)

# window driver safety (no service restoration: node A is dedicated to the experiment)
check("window: single cleanup trap", "trap cleanup EXIT INT TERM" in WIN)
check("window: no service start/stop calls",
      "launch-ornith122.sh start" not in WIN and "launch-glm53flash start" not in WIN
      and "launch-ornith122.sh stop" not in WIN)
check("window: detached deadline watchdog", "setsid nohup bash -c" in WIN and "X3_DEADLINE_S" in WIN)
check("window: fail-fast on first arm error", "fail-fast" in WIN and "exit 10" in WIN)
check("window: per-arm exclusive preflight", "arm_preflight" in WIN and "compute_owners" in WIN)
check("window: ongoing memory guard", "start_guard" in WIN and "MEM_FLOOR_MIB" in WIN)
check("window: guard kills only own group", '"$EXP/cleanup-test.sh" 2>/dev/null' in WIN)
check("window: production fingerprint diff", "PROD_FP_BEFORE" in WIN)
check("window: refuses to run without --execute", '"$MODE" != "--execute"' in WIN)
check("window: selfcheck before model", 'run_arm.sh" selfcheck' in WIN)
check("cleanup: ownership verified before kill", "/proc/\"$ARM_PGID\"/cmdline" in CLEANUP)
check("cleanup: no launcher invocation",
      "launch-ornith122" not in CLEANUP and "launch-glm" not in CLEANUP)

# arm runner
check("arm: profile args forwarded", '--profile-steps "$PROFILE_STEPS"' in ARM)
check("arm: stats flag from env", 'EXL3_DISPATCH_STATS="${EXL3_DISPATCH_STATS:-0}"' in ARM)

bad = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail else ""))
print(f"\n{len(checks) - len(bad)}/{len(checks)} preflight checks passed")
sys.exit(1 if bad else 0)
