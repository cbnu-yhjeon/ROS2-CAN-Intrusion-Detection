"""SocketCAN → ROS 2 브리지 (``ros2_socketcan`` receiver 대체용, 파이썬 표준 라이브러리만 사용).

``ros2_socketcan``이 설치돼 있지 않은 환경에서 같은 토픽(``/from_can_bus``)·같은 메시지
(``can_msgs/msg/Frame``)로 발행한다. ``header.stamp``는 커널 수신 타임스탬프
(SO_TIMESTAMPNS)라서 "vcan0 수신 시각"을 사용자 공간 지연 없이 기록한다.
"""
from __future__ import annotations

import socket
import struct
import threading
import time

import rclpy
import rclpy.executors
from can_msgs.msg import Frame
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

CAN_FRAME_FMT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)
CAN_EFF_FLAG, CAN_RTR_FLAG, CAN_ERR_FLAG = 0x80000000, 0x40000000, 0x20000000
SO_TIMESTAMPNS = getattr(socket, "SO_TIMESTAMPNS", 35)


def open_can_socket(interface: str, timeout: float = 0.2) -> socket.socket:
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.bind((interface,))
    sock.settimeout(timeout)
    return sock


def recv_frame(sock: socket.socket):
    """(커널 수신 시각 ns, raw can_id, dlc, 8바이트 data). 타임아웃이면 socket.timeout."""
    raw, anc, _, _ = sock.recvmsg(CAN_FRAME_SIZE, 64)
    stamp_ns = None
    for level, typ, payload in anc:
        if level == socket.SOL_SOCKET and typ == SO_TIMESTAMPNS:
            sec, nsec = struct.unpack("qq", payload[:16])
            stamp_ns = sec * 1_000_000_000 + nsec
    if stamp_ns is None:
        stamp_ns = time.time_ns()
    can_id, dlc, data = struct.unpack(CAN_FRAME_FMT, raw)
    return stamp_ns, can_id, dlc, data


def frame_qos(depth: int) -> QoSProfile:
    return QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)


class SocketCanBridge(Node):
    def __init__(self):
        super().__init__("socketcan_bridge")
        self.interface = self.declare_parameter("interface", "vcan0").value
        topic = self.declare_parameter("topic", "from_can_bus").value
        depth = self.declare_parameter("qos_depth", 100000).value
        self.pub = self.create_publisher(Frame, topic, frame_qos(depth))
        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.count = 0
        self._last_count, self._last_t = 0, time.time()

        self.sock = open_can_socket(self.interface)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()
        self.create_timer(1.0, self._publish_diag)
        self.get_logger().info(f"{self.interface} → /{topic.lstrip('/')}")

    def _rx_loop(self):
        while not self._stop.is_set() and rclpy.ok():
            try:
                stamp_ns, can_id, dlc, data = recv_frame(self.sock)
            except socket.timeout:
                continue
            except OSError:
                break
            msg = Frame()
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
            msg.header.frame_id = self.interface
            msg.is_extended = bool(can_id & CAN_EFF_FLAG)
            msg.is_rtr = bool(can_id & CAN_RTR_FLAG)
            msg.is_error = bool(can_id & CAN_ERR_FLAG)
            msg.id = can_id & (0x1FFFFFFF if msg.is_extended else 0x7FF)
            msg.dlc = dlc
            msg.data = list(data)
            self.pub.publish(msg)
            self.count += 1

    def _publish_diag(self):
        now = time.time()
        rate = (self.count - self._last_count) / max(now - self._last_t, 1e-9)
        self._last_count, self._last_t = self.count, now
        st = DiagnosticStatus(name="can_ids/socketcan_bridge", hardware_id=self.interface,
                              level=DiagnosticStatus.OK, message="receiving")
        st.values = [KeyValue(key="frames_total", value=str(self.count)),
                     KeyValue(key="rate_hz", value=f"{rate:.1f}")]
        arr = DiagnosticArray(status=[st])
        arr.header.stamp = self.get_clock().now().to_msg()
        self.diag_pub.publish(arr)

    def destroy_node(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SocketCanBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(f"frames_total={node.count}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
