from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-worldv2.pt")

# Define custom classes
model.set_classes(["person", "bus"])

# Execute prediction on an image
results = model.predict("path/to/image.jpg")

# Show results
results[0].show()