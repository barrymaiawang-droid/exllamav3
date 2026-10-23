#!/usr/bin/env bash
# Run-scoped cleanup for one window run (Step-2 root; the reviewed Step-1 copy stays untouched).
#
# Ownership is exact and run-scoped: the state file records ARM_OWNER_PATTERN, the one entry point the
# arm was allowed to start (bench_ornith.py, run_arm3.sh or probe_mtp_causal.py). A member is owned
# only if its cmdline contains that pattern. Fail-closed and NONZERO on every refusal or failure, so a
# guard/deadline that cannot reap its own group is reported as a failed window, never as success.
#
# exit 0 reaped (or nothing to do) | 2 survivors after KILL | 3 PID reuse | 4 not owned | 5 no state
set -u
STATE=${1:-}
[ -n "$STATE" ] && [ -f "$STATE" ] || { echo "cleanup-run: no state file argument"; exit 5; }
. "$STATE"
RAW=${RAW:?state file must define RAW}
OWNER=${ARM_OWNER_PATTERN:-}
LOG=$RAW/cleanup-${RUN_ID:-unknown}.log
LOCK=$RAW/cleanup-${RUN_ID:-unknown}.lock
exec >> "$LOG" 2>&1
exec 9>"$LOCK"
flock -w 60 9 || { echo "cleanup-run: lock timeout"; exit 6; }
echo "=== cleanup-run $(date -Is) run=${RUN_ID:-?} pgid=${ARM_PGID:-none} owner=${OWNER:-<unset>} ==="
[ -n "$OWNER" ] || { echo "cleanup-run: REFUSING - state records no ARM_OWNER_PATTERN"; exit 4; }
[ -n "${ARM_PGID:-}" ] && [ "${ARM_PGID:-0}" -gt 1 ] || { echo "cleanup-run: no arm pgid recorded"; exit 0; }

proc_start() { awk '{print $22}' "/proc/$1/stat" 2>/dev/null; }
group_members() { ps -eo pid=,pgid=,stat= 2>/dev/null | awk -v g="$ARM_PGID" '$2==g && $3 !~ /^Z/ {print $1}'; }

if [ -d "/proc/$ARM_PGID" ] && [ -n "${ARM_START:-}" ]; then
  now=$(proc_start "$ARM_PGID")
  if [ "$now" != "$ARM_START" ]; then
    echo "cleanup-run: REFUSING - leader $ARM_PGID start time mismatch (${ARM_START} -> ${now})"
    exit 3
  fi
fi
members=$(group_members)
if [ -z "$members" ]; then echo "cleanup-run: group $ARM_PGID has no live members"; exit 0; fi
owned=0
for m in $members; do
  cmd=$(tr '\0' ' ' < "/proc/$m/cmdline" 2>/dev/null)
  case "$cmd" in
    *"$OWNER"*) owned=1; echo "cleanup-run: member $m owned: $(echo "$cmd" | cut -c1-120)" ;;
  esac
done
if [ "$owned" != 1 ]; then
  echo "cleanup-run: REFUSING - group $ARM_PGID has no member matching '$OWNER' (members: $members)"
  exit 4
fi
echo "cleanup-run: TERM group $ARM_PGID ($members)"
kill -TERM -"$ARM_PGID" 2>/dev/null
for _ in $(seq 1 20); do [ -z "$(group_members)" ] && break; sleep 0.5; done
if [ -n "$(group_members)" ]; then
  echo "cleanup-run: KILL group $ARM_PGID (survivors: $(group_members))"
  kill -KILL -"$ARM_PGID" 2>/dev/null; sleep 1
fi
left=$(group_members)
if [ -n "$left" ]; then echo "cleanup-run: FAILED - members survive KILL: $left"; exit 2; fi
echo "cleanup-run: group $ARM_PGID fully gone"
exit 0
