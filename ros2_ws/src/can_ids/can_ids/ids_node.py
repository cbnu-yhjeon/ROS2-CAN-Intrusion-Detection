"""실시간 CAN 침입 탐지 노드.

입력 : ``input:=topic``     → ``/from_can_bus`` 구독 (can_msgs/msg/Frame — ros2_socketcan 또는 socketcan_bridge)
       ``input:=socketcan`` → 노드가 직접 SocketCAN(vcan0)을 읽는다 (DDS 홉 없음, 전용 수신 스레드)
발행 : ``/ids/alerts`` (can_ids_msgs/msg/Alert, 공격 판정만)
       ``/diagnostics`` (1초마다 처리량 · 지연시간 p50/p99 · 경보 수)
선택 : ``verdict_log`` 파라미터에 경로를 주면 모든 판정을 CSV로 남긴다 (온라인 F1 계산용).

``batch_window_ms`` = 0 이면 메시지마다 추론, > 0 이면 가장 오래된 대기 메시지가
그 시간을 넘기거나 ``max_batch``개가 모이면 한 번에 추론한다 (마이크로 배치).
"""
from __future__ import annotations

import socket
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import rclpy.executors
from can_ids_msgs.msg import Alert
from can_msgs.msg import Frame
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node

from can_ids.engine import DetectionEngine
from can_ids.features import CanFrame
from can_ids.models import load_detector
from can_ids.socketcan_bridge import frame_qos, open_can_socket, recv_frame

VERDICT_HEADER = "index,stamp,can_id,dlc,data,score,pred,t_recv,t_done,batch\n"


def default_model_dir() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return str(Path(get_package_share_directory("can_ids")) / "models")
    except Exception:
        return "models"


class IdsNode(Node):
    def __init__(self):
        super().__init__("ids_node")
        p = self.declare_parameter
        self.model_name = p("model", "xgb").value
        model_dir = p("model_dir", "").value or default_model_dir()
        backend = p("backend", "onnx").value
        threads = p("threads", 1).value
        self.batch_window = p("batch_window_ms", 0.0).value / 1000.0
        self.max_batch = p("max_batch", 512).value
        verdict_log = p("verdict_log", "").value
        depth = p("qos_depth", 100000).value
        self.input = p("input", "topic").value
        interface = p("interface", "vcan0").value

        detector = load_detector(self.model_name, model_dir, backend=backend, threads=threads)
        self.engine = DetectionEngine(detector)
        self.threshold = float(detector.threshold)

        self.alert_pub = self.create_publisher(Alert, "/ids/alerts", 1000)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self._lock = threading.Lock()  # socketcan 모드: 수신 스레드 ↔ 진단 타이머
        self._stop = threading.Event()
        self._rx_thread = None
        if self.input == "socketcan":
            self._sock = open_can_socket(interface)
            self._rx_thread = threading.Thread(target=self._socketcan_loop, daemon=True)
        elif self.input == "topic":
            self.create_subscription(Frame, "from_can_bus", self._on_frame, frame_qos(depth))
            if self.batch_window > 0:
                self.create_timer(self.batch_window, self._on_batch_timer)
        else:
            raise ValueError(f"input must be topic or socketcan, got {self.input!r}")
        self.create_timer(1.0, self._publish_diag)

        self._log = open(verdict_log, "w", buffering=1 << 20) if verdict_log else None
        if self._log:
            self._log.write(VERDICT_HEADER)
        self.alerts = 0
        self._lat_ms: deque = deque(maxlen=200000)
        self._win_count, self._win_t = 0, time.time()
        mode = "per-message" if self.batch_window <= 0 else f"micro-batch {self.batch_window * 1e3:g} ms"
        src = f"socketcan:{interface}" if self.input == "socketcan" else "topic:/from_can_bus"
        self.get_logger().info(f"model={self.model_name} ({backend}), {mode}, input={src}, model_dir={model_dir}")
        if self._rx_thread:
            self._rx_thread.start()

    def _socketcan_loop(self):
        sock, eng = self._sock, self.engine
        while not self._stop.is_set():
            try:
                sock.settimeout(0.2)
                stamp_ns, can_id, dlc, data = recv_frame(sock)
                eng.submit(CanFrame(stamp_ns * 1e-9, can_id & 0x1FFFFFFF, dlc, data[:dlc]))
                # 마이크로 배치: 창이 끝나거나 max_batch가 찰 때까지 더 받는다
                while self.batch_window > 0 and eng.pending < self.max_batch:
                    remaining = eng.oldest_pending + self.batch_window - time.time()
                    if remaining <= 0:
                        break
                    sock.settimeout(remaining)
                    stamp_ns, can_id, dlc, data = recv_frame(sock)
                    eng.submit(CanFrame(stamp_ns * 1e-9, can_id & 0x1FFFFFFF, dlc, data[:dlc]))
            except socket.timeout:
                pass
            except OSError:
                break
            if eng.pending:
                self._process()

    def _on_frame(self, msg: Frame):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.engine.submit(CanFrame(stamp, msg.id, msg.dlc, bytes(msg.data[:msg.dlc])))
        if self.batch_window <= 0 or self.engine.pending >= self.max_batch:
            self._process()
        elif time.time() - self.engine.oldest_pending >= self.batch_window:
            self._process()

    def _on_batch_timer(self):
        if self.engine.pending and time.time() - self.engine.oldest_pending >= self.batch_window:
            self._process()

    def _process(self):
        with self._lock:
            self._process_locked()

    def _process_locked(self):
        verdicts = self.engine.process()
        log = self._log
        for v in verdicts:
            lat = (v.t_done - v.stamp) * 1e3
            self._lat_ms.append(lat)
            if v.is_attack:
                self.alerts += 1
                a = Alert()
                a.header.stamp.sec = int(v.stamp)
                a.header.stamp.nanosec = int(round((v.stamp - int(v.stamp)) * 1e9)) % 1_000_000_000
                a.header.frame_id = self.model_name
                a.can_id, a.dlc = v.can_id, v.dlc
                a.data = list(v.data.ljust(8, b"\0"))
                a.model, a.score, a.threshold = self.model_name, v.score, self.threshold
                a.latency_ms, a.batch_size = lat, v.batch_size
                self.alert_pub.publish(a)
            if log:
                log.write(f"{v.index},{v.stamp:.6f},{v.can_id},{v.dlc},{v.data.hex()},{v.score:.6g},"
                          f"{int(v.is_attack)},{v.t_recv:.6f},{v.t_done:.6f},{v.batch_size}\n")
        self._win_count += len(verdicts)

    def _publish_diag(self):
        now = time.time()
        rate = self._win_count / max(now - self._win_t, 1e-9)
        self._win_count, self._win_t = 0, now
        with self._lock:
            lat = np.fromiter(self._lat_ms, float) if self._lat_ms else np.zeros(1)
            self._lat_ms.clear()
            pending = self.engine.pending
        st = DiagnosticStatus(name="can_ids/ids_node", hardware_id=self.model_name, message="running",
                              level=DiagnosticStatus.OK if np.percentile(lat, 99) < 100 else DiagnosticStatus.WARN)
        st.values = [KeyValue(key=k, value=v) for k, v in [
            ("frames_total", str(self.engine.count)),
            ("rate_hz", f"{rate:.1f}"),
            ("latency_p50_ms", f"{np.percentile(lat, 50):.3f}"),
            ("latency_p99_ms", f"{np.percentile(lat, 99):.3f}"),
            ("pending", str(pending)),
            ("alerts_total", str(self.alerts)),
        ]]
        arr = DiagnosticArray(status=[st])
        arr.header.stamp = self.get_clock().now().to_msg()
        self.diag_pub.publish(arr)
        if self._log:
            with self._lock:
                self._log.flush()

    def close(self):
        self._stop.set()
        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)
            self._sock.close()
        self._process()
        if self._log:
            self._log.close()
            self._log = None


def main(args=None):
    rclpy.init(args=args)
    node = IdsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        node.get_logger().info(f"frames_total={node.engine.count} alerts_total={node.alerts}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
