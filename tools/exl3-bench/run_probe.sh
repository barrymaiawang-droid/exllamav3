#!/usr/bin/env bash
# Probe-only wrapper: the exact reviewed argv for probe_mtp_causal.py.
# usage: X3_PROBE_RUNTIME=<rocm10|rocm713> run_probe.sh <out.json> [tensor_dir] [index]
#
# Two runtimes, one harness. Each mode binds its own retained experimental env, venv, source tree and
# extension cache; production (~/exl3-161*, ~/.cache/torch_extensions) is never referenced, and a mixed
# pairing is refused. Default stays rocm10.
set -u
OUT=${1:?usage: run_probe.sh <out.json> [tensor_dir] [index]}
TDIR=${2:-$(dirname "$OUT")/tensors}
IDX=${3:-18}
RUNTIME=${X3_PROBE_RUNTIME:-rocm10}

case "$RUNTIME" in
  rocm10)
    DEFAULT_ENV=$HOME/exl3-rocm10-test-env.sh
    DEFAULT_VENV=$HOME/exl3-rocm10-test
    DEFAULT_FORK=$HOME/exl3-v164-rocm10-port
    DEFAULT_CACHE=$HOME/x3-exp/rocm10/ext-cache-instr
    ALLOWED="ext-cache-instr:exl3-v164-rocm10-port ext-cache-clean:exl3-v164-rocm10-clean" ;;
  rocm713)
    # the retained Step-1 experimental runtime: ROCm 7.13 env + pristine v1.6.4 worktree + its cache
    DEFAULT_ENV=$HOME/exl3-rocm713-test-env.sh
    DEFAULT_VENV=$HOME/exl3-rocm713-test
    DEFAULT_FORK=$HOME/exl3-v164-pristine
    DEFAULT_CACHE=$HOME/x3-exp/rocm713/ext-cache-pristine
    ALLOWED="ext-cache-pristine:exl3-v164-pristine" ;;
  *) echo "run_probe: unknown X3_PROBE_RUNTIME=$RUNTIME (rocm10|rocm713)" >&2; exit 9 ;;
esac

ENV_SCRIPT=${ENV_SCRIPT:-$DEFAULT_ENV}
VENV=${VENV:-$DEFAULT_VENV}
FORK=${FORK:-$DEFAULT_FORK}
CACHE=${TORCH_EXTENSIONS_DIR:-$DEFAULT_CACHE}
ENTRY=${PROBE_ENTRY:-$HOME/x3-exp/rocm10/probe_mtp_causal.py}

for f in "$ENV_SCRIPT" "$VENV/bin/python" "$ENTRY"; do
  [ -e "$f" ] || { echo "run_probe[$RUNTIME]: missing $f" >&2; exit 7; }
done
BIND_FORK=$FORK
BIND_CACHE=$CACHE
export FORK TORCH_EXTENSIONS_DIR

# shellcheck disable=SC1090
. "$ENV_SCRIPT" || { echo "run_probe[$RUNTIME]: cannot source $ENV_SCRIPT" >&2; exit 7; }
# the env script may impose its own FORK/cache: the mode binding is restored and is what we verify
FORK=$BIND_FORK
TORCH_EXTENSIONS_DIR=$BIND_CACHE
export FORK TORCH_EXTENSIONS_DIR

# cache/tree pairing, per runtime
cache_name=$(basename "$TORCH_EXTENSIONS_DIR")
fork_name=$(basename "$FORK")
ok=0
for pair in $ALLOWED; do [ "$pair" = "$cache_name:$fork_name" ] && ok=1; done
if [ "$ok" != 1 ]; then
  echo "run_probe[$RUNTIME]: REFUSING cache/tree mismatch ($cache_name with $fork_name)" >&2
  exit 8
fi

mkdir -p "$TDIR"
echo "run_probe: runtime=$RUNTIME venv=$VENV fork=$FORK cache=$TORCH_EXTENSIONS_DIR" >&2
exec "$VENV/bin/python" "$ENTRY" \
  --ctx 8192 \
  --index "$IDX" \
  --prompt "Write a short paragraph about the ocean." \
  --out "$OUT" \
  --tensor-dir "$TDIR"
