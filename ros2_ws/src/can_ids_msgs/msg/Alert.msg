# 공격 판정 경보. header.stamp = 원 프레임의 vcan0 수신 시각
std_msgs/Header header
uint32 can_id
uint8 dlc
uint8[8] data
string model
float32 score
float32 threshold
# 수신 시각 → 판정 시각 (ms)
float64 latency_ms
# 이 프레임과 함께 추론된 배치 크기
uint32 batch_size
