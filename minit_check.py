from datetime import timedelta
import re

# ⏱️ 시간 입력 리스트 (시, 분, 초 형식의 문자열)
# time_strings = [
#     "1시간 3분 33초",
#     "38분 24초",
#     "22분 13초",
#     "15분 38초",
#     "16분 33초",
#     "15분 57초",
#     "16분 52초",
#     "17분 1초",
#     "14분 2초",
#     "15분 36초",
#     "16분 21초",
#     "분 초",
#     "분 초"
# ]
time_strings = [
    "45분 57초",
    "43분 4초",
    "40분 50초",
    "43분 32초",
    "42분 41초",
    "41분 49초",
    "46분 34초",
    "12분 2초",
]

# ⏳ 시:분:초 파싱 함수
def parse_duration(time_str):
    h = m = s = 0
    h_match = re.search(r'(\d+)\s*시간', time_str)
    m_match = re.search(r'(\d+)\s*분', time_str)
    s_match = re.search(r'(\d+)\s*초', time_str)

    if h_match: h = int(h_match.group(1))
    if m_match: m = int(m_match.group(1))
    if s_match: s = int(s_match.group(1))

    return timedelta(hours=h, minutes=m, seconds=s)

# 전체 시간과 평균 계산
durations = [parse_duration(t) for t in time_strings]
total = sum(durations, timedelta())
average = total / len(durations)

# 결과 출력
print("총합 시간:", total)
print("평균 시간:", average)
