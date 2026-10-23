#!/usr/bin/env bash
# Build the exllamav3 extension for gfx1151 against the ROCm-10 stack, one variant per invocation.
# usage: build-ext.sh <instr|clean>
#   instr : ~/exl3-v164-rocm10-port  (port fixes + Step-1 dispatch counters)  -> ext-cache-instr
#   clean : ~/exl3-v164-rocm10-clean (port fixes only, no counters)           -> ext-cache-clean
set -u
VAR=${1:?usage: build-ext.sh <instr|clean>}
EXP_ROOT=${EXP_ROOT:-$HOME/x3-exp/rocm10}
case "$VAR" in
  instr) FORK=$HOME/exl3-v164-rocm10-port;  CACHE=$EXP_ROOT/ext-cache-instr; WANT=1 ;;
  clean) FORK=$HOME/exl3-v164-rocm10-clean; CACHE=$EXP_ROOT/ext-cache-clean; WANT=0 ;;
  *) echo "unknown variant $VAR"; exit 2 ;;
esac
LOG=$EXP_ROOT/logs/build-$VAR.log
mkdir -p "$EXP_ROOT/logs"
exec 9>"$EXP_ROOT/.build.lock"
if ! flock -n 9; then echo "REFUSING: another build-ext.sh holds the build lock" | tee -a "$LOG" >&2; exit 2; fi
{
  echo "=== BUILD $VAR START $(date -Is) ==="
  export FORK TORCH_EXTENSIONS_DIR="$CACHE"
  source "$HOME/exl3-rocm10-test-env.sh"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
  echo "ROCM_HOME=${ROCM_HOME:-<unset>} ROCM_PATH=${ROCM_PATH:-<unset>} LD_PRELOAD=${LD_PRELOAD:-<unset>}"
  echo "--- git identity"
  git -C "$FORK" log --oneline -1
  git -C "$FORK" status --short | head -20
  rm -rf "$CACHE"
  "$VENV/bin/python" - "$WANT" <<'PY'
import os, subprocess, sys, glob
want = sys.argv[1] == "1"
import torch
print("torch", torch.__version__, "hip", torch.version.hip, flush=True)
from exllamav3.util import cuda_flags
import torch.utils.cpp_extension as ce
src = os.path.join(os.environ["FORK"], "exllamav3", "exllamav3_ext")
cuda_flags.use_rocm_sdk_devel(ce)
print("cpp_extension.ROCM_HOME", ce.ROCM_HOME, flush=True)
hipcc = os.path.join(ce.ROCM_HOME, "bin", "hipcc") if ce.ROCM_HOME else None
if hipcc and os.path.exists(hipcc):
    out = subprocess.run([hipcc, "--version"], capture_output=True, text=True)
    print("hipcc --version:", out.stdout.strip().splitlines()[:6], flush=True)
from exllamav3 import ext
print("ext file", ext.exllamav3_ext.__file__, flush=True)
print("has exl3_dispatch_stats", hasattr(ext.exllamav3_ext, "exl3_dispatch_stats"), flush=True)
print("has exl3_gemv_supported", hasattr(ext.exllamav3_ext, "exl3_gemv_supported"),
      getattr(ext.exllamav3_ext, "HAS_exl3_gemv_supported", None), flush=True)
assert hasattr(ext.exllamav3_ext, "exl3_dispatch_stats") == want, "counter binding does not match variant"
print("device", torch.cuda.get_device_properties(0).gcnArchName, flush=True)
PY
  rc=$?
  echo "--- build python exit $rc"
  SO=$CACHE/exllamav3_ext/exllamav3_ext.so
  if [ -f "$SO" ]; then
    echo "so size=$(stat -c%s "$SO") mtime=$(stat -c%y "$SO")"
    echo "so sha256=$(sha256sum "$SO" | awk '{print $1}')"
    echo "objs=$(ls "$CACHE"/exllamav3_ext/*.o 2>/dev/null | wc -l)"
    echo "build.ninja sha256=$(sha256sum "$CACHE"/exllamav3_ext/build.ninja | awk '{print $1}')"
    echo "--- ninja_log tail"; tail -5 "$CACHE"/exllamav3_ext/.ninja_log 2>/dev/null
  else
    echo "NO .so PRODUCED"
  fi
  echo "=== BUILD $VAR END $(date -Is) rc=$rc ==="
  exit $rc
} >>"$LOG" 2>&1
