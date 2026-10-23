#!/usr/bin/env bash
# Own-process cleanup for one window run. No service is started, stopped or edited.
#
# Ownership model:
#   * takes the run's IMMUTABLE state file as its only argument; never reads shared state
#   * FAIL CLOSED FIRST: if the recorded leader PID is still alive but its process start time differs from
#     the recorded one, the PID was reused and this group is not ours -> refuse, exit 3, signal nothing
#   * then finds members by scanning for the recorded process GROUP id, so a leader that has already exited
#     (the orphan case) does not hide surviving children
#   * refuses to signal a group with no member whose cmdline is this benchmark
#   * terminates the GROUP, waits, escalates to KILL, then verifies the group is empty
set -u

STATE=${1:-}
[ -n "$STATE" ] && [ -f "$STATE" ] || { echo "cleanup: no state file argument"; exit 0; }
. "$STATE"

RAW=${RAW:-$HOME/x3-exp/rocm713/raw}
LOG=$RAW/cleanup-${RUN_ID:-unknown}.log
LOCK=$RAW/cleanup-${RUN_ID:-unknown}.lock
exec >> "$LOG" 2>&1
exec 9>"$LOCK"
flock -w 60 9 || { echo "cleanup: lock timeout"; exit 1; }

echo "=== cleanup $(date -Is) run=${RUN_ID:-?} pgid=${ARM_PGID:-none} ==="
[ -n "${ARM_PGID:-}" ] && [ "${ARM_PGID:-0}" -gt 1 ] || { echo "cleanup: no arm pgid recorded"; exit 0; }

proc_start() { awk '{print $22}' "/proc/$1/stat" 2>/dev/null; }
group_members() {   # live (non-zombie) pids whose process group is ARM_PGID
  ps -eo pid=,pgid=,stat= 2>/dev/null | awk -v g="$ARM_PGID" '$2==g && $3 !~ /^Z/ {print $1}'
}

# --- fail closed: leader alive but a different process than the one recorded ---
if [ -d "/proc/$ARM_PGID" ] && [ -n "${ARM_START:-}" ]; then
  now=$(proc_start "$ARM_PGID")
  if [ "$now" != "$ARM_START" ]; then
    echo "cleanup: REFUSING - leader $ARM_PGID start time mismatch (${ARM_START} -> ${now}), possible PID reuse"
    exit 3
  fi
fi

# --- leader gone: work from the group id alone ---
members=$(group_members)
if [ -z "$members" ]; then
  echo "cleanup: group $ARM_PGID has no live members; nothing to do"
  exit 0
fi

owned=0
for m in $members; do
  cmd=$(tr '\0' ' ' < "/proc/$m/cmdline" 2>/dev/null)
  case "$cmd" in
    *bench_ornith*|*run_arm*) owned=1; echo "cleanup: member $m owned: $(echo "$cmd" | cut -c1-100)" ;;
  esac
done
if [ "$owned" != 1 ]; then
  echo "cleanup: REFUSING, group $ARM_PGID has no benchmark member (members: $members)"
  exit 0
fi

echo "cleanup: TERM group $ARM_PGID ($members)"
kill -TERM -"$ARM_PGID" 2>/dev/null
for _ in $(seq 1 20); do
  [ -z "$(group_members)" ] && break
  sleep 0.5
done
if [ -n "$(group_members)" ]; then
  echo "cleanup: KILL group $ARM_PGID (survivors: $(group_members))"
  kill -KILL -"$ARM_PGID" 2>/dev/null
  sleep 1
fi
left=$(group_members)
if [ -n "$left" ]; then
  echo "cleanup: WARNING group members still present after KILL: $left"
  exit 2
fi
echo "cleanup: group $ARM_PGID fully gone"
exit 0
