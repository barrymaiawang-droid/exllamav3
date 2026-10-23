#!/usr/bin/env bash
# Experiment env: exllamav3 v1.6.4 ROCm-10 port (branch exp/rocm10-v1.6.4), node A / gfx1151.
# Step 2 of the ROCm-10 validation. Fully self-contained: torch, ROCm runtime, SDK, compiler and
# triton all come from ~/exl3-rocm10-test; nothing is read from ~/kyojin (the ROCm 7 tree) and no
# ROCm-7 library is preloaded.
#
# Isolation contract:
#   reads  : $FORK (source tree), $VENV (installed stack)
#   writes : $EXP_ROOT (JIT extension cache, triton/inductor caches, logs, raw results)
#   never  : ~/exl3-161*, ~/ornith122-serve.py, ~/launch-*.sh, ~/.cache/torch_extensions, weights,
#            llmfleet, ~/x3-exp/rocm713 (the retained ROCm-7 Step-1 tree and its caches)
#
# Usage: source ~/exl3-rocm10-test-env.sh
set -u

VENV=${VENV:-$HOME/exl3-rocm10-test}
EXP_ROOT=${EXP_ROOT:-$HOME/x3-exp/rocm10}
FORK=${FORK:-$HOME/exl3-v164-rocm10-port}

export PATH="$VENV/bin:$HOME/.local/bin:$PATH"
# exactly one exllamav3 on sys.path: the checkout. torch lives in the venv itself.
export PYTHONPATH="$FORK"
unset PYTHONHOME

export PYTHONDONTWRITEBYTECODE=1

# Build/runtime caches that cannot collide with production or with the Step-1 ROCm-7 trees.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$EXP_ROOT/ext-cache}"
export TRITON_CACHE_DIR="$EXP_ROOT/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$EXP_ROOT/inductor-cache"

# The SDK root is resolved by exllamav3.util.cuda_flags.use_rocm_sdk_devel(), which points torch at the
# rocm-sdk-devel tree. No ROCM_HOME/HIP_PATH here on purpose: setting them to the core runtime would hide
# the library headers the build needs, and these wheels are not a merged /opt/rocm tree.
export PYTORCH_ROCM_ARCH=gfx1151
export MAX_JOBS="${MAX_JOBS:-6}"
echo "[x3-rocm10] fork=$FORK venv=$VENV ext_cache=$TORCH_EXTENSIONS_DIR arch=gfx1151 jobs=$MAX_JOBS"
