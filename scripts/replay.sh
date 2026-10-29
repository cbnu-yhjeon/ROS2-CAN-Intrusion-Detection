#!/usr/bin/env bash
# candump 로그를 원래 간격대로 vcan0에 재생한다.
#   scripts/replay.sh results/segments/dos.log [interface] [speed]
# can-utils의 canplayer가 있으면 그것을, 없으면 파이썬 대체 플레이어를 쓴다.
set -euo pipefail
LOG="$1"; IFACE="${2:-vcan0}"; SPEED="${3:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if command -v canplayer >/dev/null 2>&1 && [ "$SPEED" = "1" ]; then
  exec canplayer -I "$LOG" "$IFACE=vcan0"
fi
PY="${PYTHON:-python3}"
exec "$PY" "$HERE/pycanplayer.py" -I "$LOG" -i "$IFACE" --speed "$SPEED"
