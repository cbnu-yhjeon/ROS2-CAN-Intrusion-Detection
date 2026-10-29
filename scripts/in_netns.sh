#!/usr/bin/env bash
# root 없이 vcan0을 쓰기 위한 래퍼: 사용자+네트워크 네임스페이스를 만들고 그 안에
# lo, vcan0을 올린 뒤 명령을 실행한다. 네임스페이스 안 프로세스끼리만 vcan0을 공유한다.
# (호스트 커널에 vcan 모듈이 있어야 함. WSL2 기본 커널은 지원)
#   scripts/in_netns.sh bash -c 'candump vcan0 & ...'
set -euo pipefail
if [ "${IN_CAN_NETNS:-0}" = "1" ]; then
  ip link set lo up
  ip link show vcan0 >/dev/null 2>&1 || ip link add dev vcan0 type vcan
  ip link set up vcan0
  exec "$@"
fi
export IN_CAN_NETNS=1
exec unshare --user --map-root-user --net "$0" "$@"
