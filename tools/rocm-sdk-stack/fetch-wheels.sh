#!/usr/bin/env bash
# Single-owner, resumable wheel fetcher for the ROCm 10.1.0a20260822 experimental stack.
# usage: fetch-wheels.sh <name[.whl]> ...
#   name may be given literally; '+'-bearing names must be passed percent-encoded (as they appear in the index).
# Writes: $WHEELS/<file>, appends "sha256  <file>" to $WHEELS/SHA256SUMS, logs to $LOGDIR/fetch.log
set -u
BASE=${BASE:-https://rocm.nightlies.amd.com/whl-multi-arch}
WHEELS=${WHEELS:-$HOME/x3-exp/rocm10/wheels}
LOGDIR=${LOGDIR:-$HOME/x3-exp/rocm10/logs}
mkdir -p "$WHEELS" "$LOGDIR"
LOG="$LOGDIR/fetch.log"
exec 9>"$WHEELS/.fetch.lock"
if ! flock -n 9; then
  echo "REFUSING: another fetch-wheels.sh holds the lock ($WHEELS/.fetch.lock)" | tee -a "$LOG" >&2
  exit 2
fi
exec >>"$LOG" 2>&1
echo "=== FETCH START $(date -Is) pid=$$ args: $* ==="
rc=0
for spec in "$@"; do
  f=$(printf '%s' "$spec" | sed 's/%2B/+/g')
  url="$BASE/$spec"
  out="$WHEELS/$f"
  if [ -s "$out" ]; then
    echo "skip (present, $(stat -c%s "$out") B): $f"
  else
    echo "GET $url"
    # one stream, resumable, bounded retries; no parallel writers, no rate tuning
    curl -fL --retry 6 --retry-delay 5 --retry-connrefused -C - \
         --connect-timeout 30 --speed-time 120 --speed-limit 20000 \
         -o "$out.part" -w 'http=%{http_code} bytes=%{size_download} secs=%{time_total} speed=%{speed_download}\n' \
         "$url" || { echo "FAILED: $f"; rc=1; continue; }
    mv "$out.part" "$out"
  fi
  sz=$(stat -c%s "$out")
  sha=$(sha256sum "$out" | awk '{print $1}')
  echo "OK  $f  size=$sz sha256=$sha"
  grep -q " $f\$" "$WHEELS/SHA256SUMS" 2>/dev/null || echo "$sha  $f" >> "$WHEELS/SHA256SUMS"
done
echo "=== FETCH END $(date -Is) rc=$rc ==="
exit $rc
