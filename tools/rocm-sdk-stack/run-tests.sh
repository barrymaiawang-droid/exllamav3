#!/usr/bin/env bash
# Numerical kernel tests on the ROCm-10 instrumented build (unchanged tolerances, independent reference).
set -u
EXP=$HOME/x3-exp/rocm10
export FORK=$HOME/exl3-v164-rocm10-port
export TORCH_EXTENSIONS_DIR=$EXP/ext-cache-instr
source $HOME/exl3-rocm10-test-env.sh
cd "$FORK" || exit 1
LOG=$EXP/logs/tests.log
{
  echo "=== TESTS START $(date -Is) fork=$(git rev-parse --short HEAD) ==="
  python -m pytest tests/kernels/gemm -q -p no:cacheprovider --junitxml=$EXP/raw/junit-gemm-rocm10.xml
  echo "gemm rc=$?"
  python -m pytest tests/kernels/routing tests/kernels/quant tests/kernels/hadamard tests/kernels/activation \
    -q -p no:cacheprovider --junitxml=$EXP/raw/junit-other-rocm10.xml
  echo "other rc=$?"
  echo "=== TESTS END $(date -Is) ==="
} >> $LOG 2>&1
