from ultralytics import YOLO

# LOAD MODEL
model = YOLO("yolov8n.pt")

def detect_objects(image_path):

    results = model(image_path)

    # SHOW RESULT
    results[0].show()

    # SAVE OUTPUT
    results[0].save(filename="output.jpg")

    print("Detection Completed")