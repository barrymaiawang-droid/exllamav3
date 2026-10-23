#!/usr/bin/env bash
# One arm of the Step-2 window. usage: run_arm3.sh <label> <num_draft> <ext_cache> <out.json> <stats> [bench|probe|legacy|selfcheck]
#   bench    : ROCm 10 env, bench_ornith.py
#   legacy   : ROCm 7.13 retained env/cache, bench_ornith.py  (old-runtime control)
#   selfcheck: ROCm 10 env, bench_ornith.py --selfcheck (no model)
set -u
LABEL=$1; ND=$2; CACHE=$3; OUT=$4; STATS=${5:-0}; RUNNER=${6:-bench}

if [ "$RUNNER" = "legacy" ]; then
  ENV_SCRIPT=$HOME/exl3-rocm713-test-env.sh
  PY=$HOME/exl3-rocm713-test/bin/python
  export TORCH_EXTENSIONS_DIR="$HOME/x3-exp/rocm713/ext-cache-pristine"
  export FORK="$HOME/exl3-v164-pristine"
else
  ENV_SCRIPT=$HOME/exl3-rocm10-test-env.sh
  PY=$HOME/exl3-rocm10-test/bin/python
  export TORCH_EXTENSIONS_DIR="$CACHE"
  # the source tree must match the cache: torch keys the build directory on the sources, so a
  # mismatched tree silently recompiles into (and would overwrite) the other variant's .so
  case "$CACHE" in
    *ext-cache-instr) export FORK="$HOME/exl3-v164-rocm10-port" ;;
    *ext-cache-clean) export FORK="$HOME/exl3-v164-rocm10-clean" ;;
    *) echo "run_arm3: unknown cache $CACHE" >&2; exit 2 ;;
  esac
fi
# shellcheck disable=SC1090
source "$ENV_SCRIPT"
export EXL3_DISPATCH_STATS="${STATS:-0}"

case "$RUNNER" in
  probe)
    exec "$HOME/x3-exp/rocm10/run_probe.sh" "$OUT" "$(dirname "$OUT")/tensors" "${X3_PROBE_INDEX:-18}" ;;
  selfcheck)
    exec "$PY" "$HOME/x3-exp/rocm10/bench_ornith.py" --selfcheck ;;
esac

PWR="$(dirname "$OUT")/power-$(basename "$OUT" .json).csv"
( echo "ts,temp1_input_mC,power1_average_uW,sclk_mhz"
  while :; do
    t=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
    p=$(cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_average 2>/dev/null | head -1)
    s=$(cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '\*' | head -1 | awk '{print $2}')
    echo "$(date +%s.%N),${t:-},${p:-},${s:-}"
    sleep 1
  done ) > "$PWR" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
"$PY" "$HOME/x3-exp/rocm10/bench_ornith.py" \
  --label "$LABEL" --num-draft "$ND" --ctx 8192 --max-new 256 --warmup 16 --reps 2 \
  --prompt "Write a short paragraph about the ocean." --out "$OUT" \
  --profile-steps 0 --profile-out "${OUT%.json}.proftable.txt" \
  $([ "${EXL3_DISPATCH_STATS:-0}" = "1" ] && echo --count-reconstruct)
rc=$?
kill $SAMPLER 2>/dev/null
wait $SAMPLER 2>/dev/null
exit $rc
