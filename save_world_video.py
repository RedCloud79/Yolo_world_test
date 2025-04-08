import cv2
import subprocess
from ultralytics import YOLOWorld

# 1. 모델 로드
model = YOLOWorld("yolov8s-worldv2.pt")

# 2. 클래스 설정
model.set_classes([
    "fire", "flame", "burning object", "smoke", "explosion",
    "lamp", "ceiling lamp", "desk lamp", "floor lamp",
    "light", "hanging light", "LED lamp", "light bulb",
    "person", "car", "motorcycle", "electric bulb", "stove", "heating device",
    "cigarette", "fire extinguisher", "fire hydrant"
])

# 3. 비디오 파일 로드
video_path = "irop-piro-fe01-c720_uzusy3.mp4"
cap = cv2.VideoCapture(video_path)

# 4. 저장할 영상 설정
output_path = "output_detected.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

ffmpeg_command = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{frame_width}x{frame_height}",
    "-r", str(fps),
    "-i", "-",
    "-an",
    "-vcodec", "libx264",
    "-pix_fmt", "yuv420p",
    "-preset", "fast",
    "-crf", "23",
    output_path
]
ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

# 5. 프레임 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO-World 예측 수행
    results = model.predict(source=frame, verbose=False)[0]

    # 직접 바운딩박스 및 클래스명 시각화 (정확도 제외)
    for box in results.boxes:
        cls_id = int(box.cls)
        label = results.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 바운딩 박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 클래스명만 표기
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 저장
    ffmpeg_process.stdin.write(frame.tobytes())

    # 출력
    cv2.imshow("YOLO-World v2 Detection", frame)
    if cv2.waitKey(1) == 27:
        break

# 6. 자원 해제
cap.release()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
cv2.destroyAllWindows()
