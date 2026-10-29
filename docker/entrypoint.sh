#!/usr/bin/env bash
# demo        : vcan0 생성 → IDS 노드 + foxglove_bridge → 세그먼트를 반복 재생
# experiments : scripts/run_online.py 실험 스위트 실행 (결과는 results/)
# 그 외       : 그대로 실행 (예: bash)
set -eo pipefail
source /opt/ros/humble/setup.bash
source /opt/can_ids/ros2_ws/install/setup.bash
cd /opt/can_ids

MODEL="${MODEL:-xgb}"; BATCH_MS="${BATCH_MS:-1.0}"; INPUT="${INPUT:-socketcan}"
BRIDGE="${BRIDGE:-ros2_socketcan}"; SEGMENTS="${SEGMENTS:-normal dos fuzzy rpm gear}"; SPEED="${SPEED:-1}"

setup_vcan() {
  if ! ip link show vcan0 >/dev/null 2>&1; then
    ip link add dev vcan0 type vcan || {
      echo "vcan0 생성 실패: 호스트에서 'sudo modprobe vcan' 후 다시 실행 (컨테이너에 NET_ADMIN 필요)"; exit 1; }
  fi
  ip link set up vcan0
}

ensure_segments() {
  [ -f data/segments/normal.log ] && return
  [ -d data/Data/raw ] || scripts/download_data.sh data
  for p in "dos DoS_dataset" "fuzzy Fuzzy_dataset" "rpm RPM_dataset" "gear gear_dataset" "normal normal_run_data"; do
    set -- $p; python3 scripts/csv2candump.py "data/Data/raw/$2.csv" -o "data/segments/$1" --duration 60
  done
}

case "${1:-demo}" in
  demo)
    setup_vcan; ensure_segments
    ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 &
    ros2 launch can_ids ids.launch.py model:="$MODEL" batch_window_ms:="$BATCH_MS" \
        input:="$INPUT" bridge:="$BRIDGE" model_dir:=/opt/can_ids/models &
    sleep 5
    echo "Foxglove: ws://localhost:8765  (/ids/alerts, /diagnostics)"
    while true; do
      for s in $SEGMENTS; do
        echo "▶ replay $s (x$SPEED)"; scripts/replay.sh "data/segments/$s.log" vcan0 "$SPEED"
      done
      [ "${LOOP:-1}" = "1" ] || break
    done
    wait ;;
  experiments)
    setup_vcan; ensure_segments
    python3 scripts/run_online.py --suite main --input socketcan
    python3 scripts/run_online.py --suite throughput --input socketcan
    python3 scripts/make_report.py ;;
  *) exec "$@" ;;
esac
