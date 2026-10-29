#!/usr/bin/env bash
# vcan0 생성. root(또는 sudo)가 필요하다.
# root 없이 시험할 때는 scripts/in_netns.sh 로 사용자 네임스페이스 안에서 만든다.
set -euo pipefail
IFACE="${1:-vcan0}"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO modprobe vcan 2>/dev/null || true
if ! ip link show "$IFACE" >/dev/null 2>&1; then
  $SUDO ip link add dev "$IFACE" type vcan
fi
$SUDO ip link set up "$IFACE"
ip -details link show "$IFACE"
