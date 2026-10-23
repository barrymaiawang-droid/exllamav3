#!/usr/bin/env bash
# Install the pinned, matched ROCm 10.1.0a20260822 stack into ~/exl3-rocm10-test.
#
# Every ROCm-10 package is installed from the locally fetched wheels (see fetch-wheels.sh) so the
# versions cannot drift; pure-python dependencies (filelock, sympy, ...) come from PyPI. The pins are
# the ones torch 2.14.0+rocm10.1.0a20260822 declares in its own METADATA:
#   rocm[libraries]==10.1.0a20260822, triton==3.8.0+git675c5987.rocm10.1.0a20260822
# plus torch[device-gfx1151] for the gfx1151 device objects and rocm[devel] for the compiler/tree.
#
# Idempotent: re-running upgrades nothing that is already pinned correctly.
set -u
EXP_ROOT=${EXP_ROOT:-$HOME/x3-exp/rocm10}
VENV=${VENV:-$HOME/exl3-rocm10-test}
WHEELS=$EXP_ROOT/wheels
LOGDIR=$EXP_ROOT/logs
LOG=$LOGDIR/install.log
mkdir -p "$LOGDIR"
exec 9>"$EXP_ROOT/.install.lock"
if ! flock -n 9; then
  echo "REFUSING: another install-stack.sh holds $EXP_ROOT/.install.lock" | tee -a "$LOG" >&2
  exit 2
fi
exec >>"$LOG" 2>&1
echo "=== INSTALL START $(date -Is) venv=$VENV ==="

# Interpreter: the system python3.12 has no development headers on this box (no python3.12-devel RPM),
# which breaks every extension build. The user-local uv CPython 3.12.13 does ship include/python3.12 and
# is the same base interpreter the Step-1 venv used, so the ABI and the tokenizer stack stay comparable.
PY=${PY:-$HOME/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12}
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV" || { echo "venv creation failed"; exit 1; }
fi
"$VENV/bin/python" -m pip install --disable-pip-version-check --no-input -q --upgrade pip setuptools wheel || exit 1
"$VENV/bin/python" -m pip --version
"$VENV/bin/python" -c 'import sys, sysconfig, os; p = sysconfig.get_paths()["include"]; print("python", sys.version.split()[0], "include", p, "Python.h", os.path.exists(os.path.join(p, "Python.h")))'

"$VENV/bin/python" -m pip install --disable-pip-version-check --no-input \
  --find-links "$WHEELS" --index-url https://pypi.org/simple \
  "torch[device-gfx1151]==2.14.0+rocm10.1.0a20260822" \
  "rocm[devel]==10.1.0a20260822" \
  "rocm-sdk-device-gfx1151==10.1.0a20260822" \
  "rocm-sdk-libraries==10.1.0a20260822" \
  "rocm-profiler==10.1.0a20260822" \
  || { echo "PIP INSTALL FAILED"; exit 1; }

# exllamav3 runtime/test dependencies, pinned to the versions the Step-1 environment used so that
# tokenization and file I/O are identical across runtimes (these are pure-python/C-extension packages,
# not ROCm runtime libraries).
"$VENV/bin/python" -m pip install --disable-pip-version-check --no-input \
  "numpy==2.5.4" "tokenizers==0.23.3" "safetensors==0.8.0" "pillow==12.3.0" "PyYAML==6.0.3" \
  "pydantic==2.14.1" "llguidance==1.9.1" "rich==15.0.0" "ninja==1.13.2" "tqdm==4.70.1" \
  "marisa-trie==1.4.1" "pytest==9.1.1" \
  || { echo "APP DEPS INSTALL FAILED"; exit 1; }

echo "--- pip check"
"$VENV/bin/python" -m pip check
echo "--- pip freeze"
"$VENV/bin/python" -m pip freeze --all
echo "=== INSTALL END $(date -Is) ==="
