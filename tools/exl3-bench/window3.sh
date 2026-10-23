#!/usr/bin/env bash
# Step-2 window driver (ROCm 10 isolated stack vs the retained ROCm 7.13 control).
# Same safety model as the Step-1 driver: per-invocation run id and state file, per-arm exclusive
# preflight, memory guard that kills only this run's process group, fail-fast, detached deadline
# watchdog reported as a nonzero outcome, one cleanup implementation, production fingerprint.
#
# Modes: dry-run (default) | --execute
# Rehearsal: X3_MOCK=ok|arm-fail|deadline|leader-exit
set -u

EXP=${EXP_ROOT_DIR:-$HOME/x3-exp/rocm10}
RAW=$EXP/raw
CLEAN=$EXP/ext-cache-clean
INSTR=$EXP/ext-cache-instr
MODE=${1:-dry-run}
PROBE_ONLY=0
[ "$MODE" = "--probe-only" ] && { PROBE_ONLY=1; MODE=--execute; }
MOCK=${X3_MOCK:-}
DEADLINE_S=${X3_DEADLINE_S:-1500}
ARM_MIN_AVAIL_MIB=${X3_ARM_MIN_AVAIL_MIB:-80000}
MEM_FLOOR_MIB=${X3_MEM_FLOOR_MIB:-12288}
mkdir -p "$RAW"

# ---- common run initialization (must precede any helper use, mock included) -------------------
RUN_ID="$(date +%s%N)-$$"
STATE=$RAW/window3-$RUN_ID.state
DEADLINE_MARK=$RAW/window3-$RUN_ID.deadline
GUARD_MARK=$RAW/window3-$RUN_ID.guard
GUARD_FAILED=0
GUARD_CANCELLED=0
LOG=$RAW/window3-$RUN_ID.log
DRIVER_PID=$$
DRIVER_START=$(awk '{print $22}' "/proc/$$/stat")
exec > >(tee -a "$LOG") 2>&1
say() { echo "[w3 $(date +%H:%M:%S)] $*"; }

# ---- pure helpers (no side effects, safe for the mock dispatch below) -------------------------
PROD_FILES="$HOME/launch-ornith122.sh $HOME/ornith122-serve.py $HOME/exl3-161-env.sh $HOME/launch-glm53flash-kyojin.sh"
fp() { sha256sum $PROD_FILES 2>/dev/null | sha256sum | cut -d' ' -f1; }
avail_mib() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
gtt_bytes() { cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1; }
compute_owners() {
  ss -ltn 2>/dev/null | grep -E ":1815[0-9] " && return 1
  ps -eo args | grep -E "[o]rnith122-serve|[t]ools/glm/serve" && return 1
  return 0
}
write_state() {   # $1 = phase, $2 = arm pgid, $3 = arm start
  cat > "$STATE" <<EOF
RUN_ID=$RUN_ID
RAW=$RAW
PHASE=$1
DRIVER_PID=$DRIVER_PID
DRIVER_START=$DRIVER_START
ARM_PGID=${2:-}
ARM_START=${3:-}
ARM_OWNER_PATTERN=${ARM_OWNER_PATTERN:-}
EOF
}
arm_preflight() {
  local a g why="" tries=${X3_PREFLIGHT_WAIT_S:-90}
  for _ in $(seq 1 "$tries"); do
    a=$(avail_mib); g=$(gtt_bytes)
    [ "$a" -ge "$ARM_MIN_AVAIL_MIB" ] && [ "${g:-0}" -le 2147483648 ] && break
    sleep 1
  done
  compute_owners || why="another compute owner present"
  [ -z "$why" ] && [ "$a" -lt "$ARM_MIN_AVAIL_MIB" ] && why="MemAvailable ${a} MiB < ${ARM_MIN_AVAIL_MIB}"
  [ -z "$why" ] && [ "${g:-0}" -gt 2147483648 ] && why="GTT ${g} B still held"
  if [ -n "$why" ]; then say "ARM PREFLIGHT FAIL: $why"; return 1; fi
  say "arm preflight ok: avail=${a}MiB gtt=${g}B owners=none"
}
start_guard() {
  # Ongoing floor check on this run's arm group. A trip writes $GUARD_MARK (run id + pgid) and reaps
  # the group through the run-scoped cleanup; the exit status is then 1 and the marker is the record.
  (
    while kill -0 "$ARM_PGID" 2>/dev/null; do
      a=$(avail_mib)
      if [ "$a" -lt "$MEM_FLOOR_MIB" ]; then
        say "guard: MemAvailable ${a} MiB < ${MEM_FLOOR_MIB}; reaping own group ${ARM_PGID}"
        printf 'source=start_guard\nreason=mem_floor\navail_mib=%s\nfloor_mib=%s\npgid=%s\nrun_id=%s\nat=%s\n' \
          "$a" "$MEM_FLOOR_MIB" "$ARM_PGID" "$RUN_ID" "$(date -Is)" > "$GUARD_MARK"
        if ! "$EXP/cleanup-run.sh" "$STATE" 2>/dev/null; then
          say "guard: cleanup-run refused or failed for group ${ARM_PGID}"
          exit 2
        fi
        exit 1
      fi
      sleep 2
    done
  ) &
  GUARD_PID=$!
}
stop_guard() {
  # Orderly shutdown is NOT a trip. A TERM we send ourselves makes the guard exit 143; that must not
  # fail a healthy arm. A guard that died on its own (nonzero without our TERM), or any trip marker,
  # is a real failure.
  [ -n "${GUARD_PID:-}" ] || return 0
  local rc=0
  GUARD_CANCELLED=0
  if kill -0 "$GUARD_PID" 2>/dev/null; then
    GUARD_CANCELLED=1
    kill -TERM "$GUARD_PID" 2>/dev/null
  fi
  wait "$GUARD_PID" 2>/dev/null || rc=$?
  GUARD_PID=
  if [ -f "$GUARD_MARK" ]; then
    GUARD_FAILED=1
    say "MEMORY GUARD TRIPPED (marker $(basename "$GUARD_MARK"))"
    return 0
  fi
  if [ "$rc" != 0 ] && [ "$GUARD_CANCELLED" != 1 ]; then
    GUARD_FAILED=1
    say "MEMORY GUARD FAILED without a trip marker (guard rc=$rc)"
    return 0
  fi
  say "memory guard stopped orderly (rc=$rc, cancelled=$GUARD_CANCELLED, no marker)"
  return 0
}

# ---- mock dispatch: self-contained, uses the real helpers, unreachable from real model paths ---
if [ -n "$MOCK" ]; then
  say "MOCK=$MOCK rehearsal (no model, no GPU, no python entrypoint)"
  mock_arm() {   # spawn a group shaped like the real arm, return its pid
    setsid bash -c "exec -a run_arm_mock sleep ${X3_MOCK_SLEEP:-5}" &
    MOCK_PID=$!
    ARM_PGID=$MOCK_PID
    ARM_OWNER_PATTERN=${X3_MOCK_OWNER:-run_arm_mock}
    export ARM_OWNER_PATTERN
    write_state running "$ARM_PGID" "$(awk '{print $22}' "/proc/$ARM_PGID/stat" 2>/dev/null)"
  }
  case "$MOCK" in
    ok)
      mock_arm; wait "$MOCK_PID"; rc=$?
      say "mock ok rc=$rc (expected 0)"; exit 0 ;;
    arm-fail)
      mock_arm; wait "$MOCK_PID"
      say "fail-fast rehearsal: first mock arm fails (expected)"; exit 4 ;;
    deadline)
      mock_arm; sleep "${X3_DEADLINE_S:-4}"
      kill -TERM -"$ARM_PGID" 2>/dev/null; sleep 0.5
      kill -KILL -"$ARM_PGID" 2>/dev/null
      say "deadline rehearsal: group $ARM_PGID reaped (expected nonzero outcome)"; exit 5 ;;
    leader-exit|probe-shape)
      mock_arm
      "$EXP/cleanup-run.sh" "$STATE"; rc=$?
      if [ "$rc" = 0 ]; then say "owned $MOCK-shaped group reaped; cleanup rc=0 (expected)"; exit 0; fi
      kill -KILL -"$ARM_PGID" 2>/dev/null
      say "cleanup REFUSED (rc=$rc) - reported as a failed window"; exit "$rc" ;;
    guard-healthy)
      # a healthy guard must be stoppable without failing the arm or leaving a marker
      mock_arm
      start_guard
      sleep 2
      stop_guard
      "$EXP/cleanup-run.sh" "$STATE"; crc=$?
      if [ -f "$GUARD_MARK" ] || [ "$GUARD_FAILED" = 1 ]; then
        say "guard-healthy FAILED: healthy cancellation reported a trip"; exit 13
      fi
      say "guard-healthy: orderly stop, no marker, cleanup rc=$crc (expected 0/absent)"; exit 0 ;;
    guard-trip)
      # forced trip: marker must exist with this run's id/pgid, and the window must be nonzero even
      # when the arm exits 0 and the cleanup refuses
      mock_arm
      MEM_FLOOR_MIB=99999999 start_guard
      sleep 3
      stop_guard
      wait "$MOCK_PID" 2>/dev/null || true
      "$EXP/cleanup-run.sh" "$STATE"; crc=$?
      if [ -f "$GUARD_MARK" ] && [ "$GUARD_FAILED" = 1 ]; then
        say "guard-trip: marker persisted ($(basename "$GUARD_MARK")), cleanup rc=$crc -> nonzero window"
        exit 11
      fi
      say "guard-trip FAILED: no marker or not marked failed"; exit 12 ;;
    reentry)
      say "re-entry rehearsal: refusing to run"; exit 7 ;;
    *) say "unknown X3_MOCK=$MOCK"; exit 8 ;;
  esac
  say "MOCK MODE REACHED END OF DISPATCH - no real arm is reachable"; exit 9
fi

disarm_watchdog() {
  if [ -n "${WATCHDOG_PID:-}" ] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$WATCHDOG_PID/cmdline" 2>/dev/null | grep -q "cleanup-run.sh"; then
      kill -TERM "$WATCHDOG_PID" 2>/dev/null && say "watchdog $WATCHDOG_PID disarmed"
    fi
  fi
  WATCHDOG_PID=
}
cleanup() {
  local rc=$?
  if [ -n "${CLEANED:-}" ]; then exit $rc; fi
  CLEANED=1
  say "cleanup (rc=$rc)"
  disarm_watchdog
  if [ "$MODE" = "--execute" ]; then
    stop_guard
    "$EXP/cleanup-run.sh" "$STATE"; CLEANUP_RC=$?
    [ "$CLEANUP_RC" = 0 ] || { say "!! cleanup-run exit $CLEANUP_RC - treating the window as failed"; rc=$CLEANUP_RC; }
    [ -f "$DEADLINE_MARK" ] && [ "$rc" = 0 ] && { say "deadline marker present: reporting failure"; rc=9; }
    [ -f "$GUARD_MARK" ] && { say "memory-guard marker present: reporting failure"; [ "$rc" = 0 ] && rc=11; }
    [ "${GUARD_FAILED:-0}" = 1 ] && { say "memory guard reported a failure: reporting failure"; [ "$rc" = 0 ] && rc=11; }
    write_state done "" ""
  else
    say "dry run: cleanup path intentionally not invoked"
  fi
  if [ "$(fp)" != "$PROD_FP_BEFORE" ]; then say "!! PRODUCTION FILES CHANGED during the window"; else say "production files unchanged"; fi
  say "window end run=$RUN_ID (mock=${MOCK:-none})"
  exit $rc
}
trap cleanup EXIT INT TERM

arm_watchdog() {
  setsid nohup bash -c '
    D=$1; S=$2; EXP=$3
    sleep "$D"
    . "$S" 2>/dev/null || exit 0
    [ "${PHASE:-}" = running ] || exit 0
    cur=$(awk "{print \$22}" "/proc/$DRIVER_PID/stat" 2>/dev/null)
    [ "$cur" = "$DRIVER_START" ] || exit 0
    echo "[watchdog $(date +%H:%M:%S)] deadline ${D}s run=$RUN_ID" >> "$RAW/watchdog.log"
    touch "$RAW/window3-$RUN_ID.deadline"
    "$EXP/cleanup-run.sh" "$S" >> "$RAW/watchdog.log" 2>&1 || echo "[watchdog] cleanup-run rc=$?" >> "$RAW/watchdog.log"
    sleep 10
    cur=$(awk "{print \$22}" "/proc/$DRIVER_PID/stat" 2>/dev/null)
    [ "$cur" = "$DRIVER_START" ] && kill -TERM "$DRIVER_PID" 2>/dev/null
    exit 0' _ "$DEADLINE_S" "$STATE" "$EXP" >/dev/null 2>&1 < /dev/null &
  WATCHDOG_PID=$!
  say "deadline watchdog armed (${DEADLINE_S}s, pid $WATCHDOG_PID)"
}

PROD_FP_BEFORE=$(fp)
write_state init "" ""
say "mode=$MODE mock=${MOCK:-none} run=$RUN_ID log=$LOG"
say "resource snapshot: avail=$(avail_mib)MiB gtt=$(gtt_bytes)B"
compute_owners && say "compute owners: none" || say "compute owners: OTHER SERVICE PRESENT (above)"
say "production fingerprint before: $PROD_FP_BEFORE"
say "builds: clean=$(sha256sum "$CLEAN"/exllamav3_ext/exllamav3_ext.so 2>/dev/null | cut -c1-12) instr=$(sha256sum "$INSTR"/exllamav3_ext/exllamav3_ext.so 2>/dev/null | cut -c1-12)"

if [ "$MODE" != "--execute" ]; then
  say "DRY RUN plan (no watchdog, no cleanup, no signalling):"
  say "  0 bench --selfcheck (ROCm10 env)"
  say "  1 rc10-clean-ar   nd=0 clean  (ROCm 10.1)"
  say "  2 rc7-clean-ar    nd=0 pristine (ROCm 7.13 control, retained cache)"
  say "  3 rc10-clean-mtp2 nd=2 clean  (ROCm 10.1)"
  say "  4 rc7-clean-mtp2  nd=2 pristine (ROCm 7.13 control)"
  say "  5 rc10-instr-ar   nd=0 instr + dispatch counters"
  say "  6 rc10-instr-mtp2 nd=2 instr + dispatch counters"
  say "  7 probe-mtp       probe_mtp_causal.py via run_probe.sh (probe-only: --probe-only)"
  say "  gates: exclusive preflight per arm, guard ${MEM_FLOOR_MIB}MiB, fail-fast, deadline ${DEADLINE_S}s -> nonzero"
  exit 0
fi

for c in "$CLEAN" "$INSTR"; do
  [ -f "$c/exllamav3_ext/exllamav3_ext.so" ] || { say "ABORT: missing $c/exllamav3_ext/exllamav3_ext.so"; exit 2; }
done

say "bench selfcheck (limited preflight, no model)"
if ! "$EXP/run_arm3.sh" selfcheck 0 "$CLEAN" "$RAW/selfcheck-$RUN_ID.json" 0 selfcheck; then
  say "ABORT: selfcheck failed"; exit 3
fi

arm_watchdog
wait_arm() { ARM_RC=0; wait "$1" || ARM_RC=$?; }
finish_arm() {
  if [ -f "$DEADLINE_MARK" ]; then say "arm $1 aborted by global deadline"; return 1; fi
  if [ "${GUARD_FAILED:-0}" = 1 ] || [ -f "$GUARD_MARK" ]; then
    say "arm $1 FAILED: memory guard tripped (marker $(basename "$GUARD_MARK"))"; return 1
  fi
  [ "${ARM_RC:-0}" = 0 ] || { say "arm $1 FAILED rc=$ARM_RC - fail-fast"; return 1; }
  say "arm $1 ok"; return 0
}
run_probe_arm() {   # the only arm authorised for the diagnostic stage
  local label=probe-causal
  arm_preflight || return 1
  export ARM_OWNER_PATTERN=probe_mtp_causal.py
  say "probe arm start: $EXP/run_probe.sh $RAW/$RUN_ID-$label.json $RAW/$RUN_ID-$label-tensors ${X3_PROBE_INDEX:-18}"
  set -m
  "$EXP/run_probe.sh" "$RAW/$RUN_ID-$label.json" "$RAW/$RUN_ID-$label-tensors" "${X3_PROBE_INDEX:-18}" &
  ARM_PGID=$!
  set +m
  write_state running "$ARM_PGID" "$(awk '{print $22}' "/proc/$ARM_PGID/stat" 2>/dev/null)"
  start_guard
  wait_arm "$ARM_PGID"
  stop_guard
  finish_arm "$label"
}

run_arm() {   # label nd cache stats runner
  local label=$1 ndt=$2 cache=$3 stats=$4 runner=${5:-bench}
  arm_preflight || return 1
  say "arm $label start ($runner)"
  if [ "$runner" = "probe" ]; then ARM_OWNER_PATTERN=probe_mtp_causal.py; else ARM_OWNER_PATTERN=bench_ornith.py; fi
  export ARM_OWNER_PATTERN
  set -m
  if [ "$runner" = "probe" ]; then
    "$EXP/run_arm3.sh" "$label" "$ndt" "$cache" "$RAW/$RUN_ID-$label.json" "$stats" probe &
  else
    "$EXP/run_arm3.sh" "$label" "$ndt" "$cache" "$RAW/$RUN_ID-$label.json" "$stats" &
  fi
  ARM_PGID=$!
  set +m
  write_state running "$ARM_PGID" "$(awk '{print $22}' "/proc/$ARM_PGID/stat" 2>/dev/null)"
  start_guard
  wait_arm "$ARM_PGID"
  stop_guard
  finish_arm "$label"
}
mock_arm() {
  local label=$1
  set -m
  if [ "${X3_MOCK_SHAPE:-}" = "leader-exit" ]; then
    bash -c "(exec -a run_arm_mock sleep ${X3_MOCK_SLEEP:-120}) & exit 0" &
  else
    bash -c "exec -a run_arm_mock sleep ${X3_MOCK_SLEEP:-1}" &
  fi
  ARM_PGID=$!; set +m
  write_state running "$ARM_PGID" "$(awk '{print $22}' "/proc/$ARM_PGID/stat" 2>/dev/null)"
  start_guard; wait_arm "$ARM_PGID"; stop_guard
  if [ -f "$DEADLINE_MARK" ]; then say "mock $label: deadline fired"; return 1; fi
  [ "$ARM_RC" = 0 ] && ARM_RC=${X3_MOCK_RC:-0}
  say "mock $label rc=$ARM_RC"
  return "$ARM_RC"
}

if [ "$PROBE_ONLY" = 1 ]; then
  run_probe_arm || exit 20
  say "probe-only window complete"
  exit 0
fi

run_arm rc10-clean-ar    0 "$CLEAN"  0 || exit 10
run_arm rc7-clean-ar     0 "$CLEAN"  0 legacy || exit 11
run_arm rc10-clean-mtp2  2 "$CLEAN"  0 || exit 12
run_arm rc7-clean-mtp2   2 "$CLEAN"  0 legacy || exit 13
run_arm rc10-instr-ar    0 "$INSTR"  1 || exit 14
run_arm rc10-instr-mtp2  2 "$INSTR"  1 || exit 15
run_probe_arm || exit 16
say "all arms ok"
exit 0
