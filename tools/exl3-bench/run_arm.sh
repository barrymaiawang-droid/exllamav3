#!/usr/bin/env bash
# One benchmark arm: optional power sampling around bench_ornith.py.
# usage: run_arm.sh <label> <num_draft> <ext_cache> <out.json>
set -u
LABEL=$1; ND=$2; CACHE=$3; OUT=$4; PROFILE_STEPS=${5:-0}
if [ "$LABEL" = "selfcheck" ]; then
  source "$HOME/exl3-rocm713-test-env.sh" >/dev/null 2>&1
  export TORCH_EXTENSIONS_DIR="$CACHE"
  exec "$HOME/exl3-rocm713-test/bin/python" "$HOME/x3-exp/rocm713/bench_ornith.py" --selfcheck
fi
source "$HOME/exl3-rocm713-test-env.sh"
export TORCH_EXTENSIONS_DIR="$CACHE"
export EXL3_DISPATCH_STATS="${EXL3_DISPATCH_STATS:-0}"
PWR="$HOME/x3-exp/rocm713/raw/power-${LABEL}.csv"
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
"$HOME/exl3-rocm713-test/bin/python" "$HOME/x3-exp/rocm713/bench_ornith.py" \
  --label "$LABEL" --num-draft "$ND" --ctx 8192 --max-new 256 --warmup 16 --reps 2 \
  --prompt "Write a short paragraph about the ocean." --out "$OUT" \
  --profile-steps "$PROFILE_STEPS" --profile-out "${OUT%.json}.proftable.txt" \
  $([ "${EXL3_DISPATCH_STATS:-0}" = "1" ] && echo --count-reconstruct)
rc=$?
kill $SAMPLER 2>/dev/null
wait $SAMPLER 2>/dev/null
exit $rc
