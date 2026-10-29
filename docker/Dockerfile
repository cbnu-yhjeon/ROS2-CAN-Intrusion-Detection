# ROS 2 Humble + CAN IDS. 빌드: docker compose -f docker/compose.yaml build
FROM ros:humble-ros-base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip iproute2 can-utils curl unzip procps \
        ros-humble-ros2-socketcan ros-humble-foxglove-bridge ros-humble-diagnostic-msgs \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/can_ids
COPY models ./models
COPY scripts ./scripts
COPY ros2_ws/src ./ros2_ws/src
# apt의 can_msgs(ros2_socketcan 의존성)를 쓰므로 동봉본은 건너뛴다
RUN . /opt/ros/humble/setup.sh && cd ros2_ws && \
    colcon build --packages-skip can_msgs && rm -rf build log

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh scripts/*.sh scripts/*.py
ENV ROS_LOCALHOST_ONLY=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["/entrypoint.sh"]
CMD ["demo"]
