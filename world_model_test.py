import cv2
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-worldv2.pt")

model.set_classes([
    "fire", "flame", "burning object", "smoke", "explosion",
    "lamp", "ceiling lamp", "desk lamp", "floor lamp",
    "light", "hanging light", "LED lamp", "light bulb",
    "person", "car", "motorcycle", "electric bulb", "stove", "heating device",
    "cigarette", "fire extinguisher", "fire hydrant", "a campfire with orange flames", "burning wood",
    "orange flame", "fire with smoke"
])

video_path = "마포 농수산물 시장 화재 영상.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 조정
    resized_frame = cv2.resize(frame, (1280, 1280))

    # 예측 수행 (confidence 낮춤)
    results = model.predict(source=resized_frame, conf=0.05, verbose=False)

    # 시각화
    annotated_frame = results[0].plot()

    # 출력
    cv2.imshow("YOLO-World v2 Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
