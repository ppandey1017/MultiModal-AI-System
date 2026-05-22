from ultralytics import YOLO
import os
# LOAD MODEL
model = YOLO("yolov8n.pt")

def detect_objects(image_path):

    results = model(image_path)

    # SHOW RESULT
    results[0].show()

    # SAVE OUTPUT
    # results[0].save(filename="output.jpg")
    output_path = os.path.join(
        "static",
        f"output_{os.path.basename(image_path)}"
    )

    results[0].save(filename=output_path)

    print("Detection Completed")