# ros2_socketcan이 쓰는 can_msgs/msg/Frame과 같은 정의 (ros-industrial/ros_canopen)
std_msgs/Header header
uint32 id
bool is_rtr
bool is_extended
bool is_error
uint8 dlc
uint8[8] data
