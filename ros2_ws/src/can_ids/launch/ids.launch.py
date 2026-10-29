"""vcan0 → (socketcan_bridge | ros2_socketcan) → ids_node.

예) ros2 launch can_ids ids.launch.py model:=xgb batch_window_ms:=1.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import (EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = {
        "interface": ("vcan0", "SocketCAN 인터페이스"),
        "input": ("topic", "topic (/from_can_bus 구독) | socketcan (노드가 vcan0 직접 수신)"),
        "bridge": ("python", "input:=topic 일 때 브리지: python (socketcan_bridge) | ros2_socketcan | none"),
        "model": ("xgb", "xgb | iforest | lstm | ensemble"),
        "model_dir": ("", "모델 디렉토리 (비우면 패키지 share/models)"),
        "backend": ("onnx", "LSTM 백엔드: onnx | torchscript"),
        "batch_window_ms": ("0.0", "0 = 메시지 단위, >0 = 마이크로 배치 창"),
        "max_batch": ("512", "배치 최대 크기"),
        "verdict_log": ("", "모든 판정을 기록할 CSV 경로"),
    }
    cfg = {k: LaunchConfiguration(k) for k in args}
    # 엔트리 스크립트의 shebang(/usr/bin/python3) 대신 PATH의 python3(가상환경)로 실행
    python = EnvironmentVariable("CAN_IDS_PYTHON", default_value="python3")

    def bridge_is(name):  # 브리지는 input:=topic 일 때만 띄운다
        return IfCondition(PythonExpression(
            ["'", cfg["input"], "' == 'topic' and '", cfg["bridge"], f"' == '{name}'"]))

    ld = [DeclareLaunchArgument(k, default_value=d, description=h) for k, (d, h) in args.items()]

    ld.append(Node(
        package="can_ids", executable="socketcan_bridge", name="socketcan_bridge", output="screen",
        prefix=python,
        parameters=[{"interface": cfg["interface"]}],
        condition=bridge_is("python"),
    ))
    ld.append(IncludeLaunchDescription(
        AnyLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("ros2_socketcan"), "launch", "socket_can_receiver.launch.py"])),
        launch_arguments={"interface": cfg["interface"]}.items(),
        condition=bridge_is("ros2_socketcan"),
    ))
    ld.append(Node(
        package="can_ids", executable="ids_node", name="ids_node", output="screen", prefix=python,
        parameters=[{
            "model": cfg["model"], "model_dir": cfg["model_dir"], "backend": cfg["backend"],
            "batch_window_ms": cfg["batch_window_ms"], "max_batch": cfg["max_batch"],
            "verdict_log": cfg["verdict_log"], "input": cfg["input"], "interface": cfg["interface"],
        }],
    ))
    return LaunchDescription(ld)
