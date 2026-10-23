#!/usr/bin/env bash
# Probe-only wrapper: the exact reviewed argv for probe_mtp_causal.py.
# usage: run_probe.sh <out.json> [tensor_dir] [index]
# It loads the pinned environment itself (no reliance on inherited shell state) and refuses to run if
# the source tree and the extension cache do not belong to the same variant.
set -u
OUT=${1:?usage: run_probe.sh <out.json> [tensor_dir] [index]}
TDIR=${2:-$(dirname "$OUT")/tensors}
IDX=${3:-18}

ENV_SCRIPT=${ENV_SCRIPT:-$HOME/exl3-rocm10-test-env.sh}
VENV=${VENV:-$HOME/exl3-rocm10-test}
FORK=${FORK:-$HOME/exl3-v164-rocm10-port}
export FORK
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/x3-exp/rocm10/ext-cache-instr}
ENTRY=${PROBE_ENTRY:-$HOME/x3-exp/rocm10/probe_mtp_causal.py}

# shellcheck disable=SC1090
. "$ENV_SCRIPT" || { echo "run_probe: cannot source $ENV_SCRIPT" >&2; exit 7; }

# cache/tree pairing: the port tree builds into ext-cache-instr, the clean tree into ext-cache-clean
cache_name=$(basename "$TORCH_EXTENSIONS_DIR")
fork_name=$(basename "$FORK")
case "$cache_name:$fork_name" in
  ext-cache-instr:exl3-v164-rocm10-port|ext-cache-clean:exl3-v164-rocm10-clean) : ;;
  *) echo "run_probe: REFUSING cache/tree mismatch ($cache_name with $fork_name)" >&2; exit 8 ;;
esac

mkdir -p "$TDIR"
exec "$VENV/bin/python" "$ENTRY" \
  --ctx 8192 \
  --index "$IDX" \
  --prompt "Write a short paragraph about the ocean." \
  --out "$OUT" \
  --tensor-dir "$TDIR"
