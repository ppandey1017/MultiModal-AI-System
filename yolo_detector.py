# from ultralytics import YOLO
# import os
# # LOAD MODEL
# model = YOLO("yolov8n.pt")

# def detect_objects(image_path):

#     results = model(image_path)

#     # SHOW RESULT
#     results[0].show()

#     # SAVE OUTPUT
#     # results[0].save(filename="output.jpg")
#     output_path = os.path.join(
#         "static",
#         f"output_{os.path.basename(image_path)}"
#     )

#     results[0].save(filename=output_path)
#     return f"output_{os.path.basename(image_path)}"

#     print("Detection Completed")

from ultralytics import YOLO
import os
from collections import Counter

model = YOLO("yolov8n.pt")

def detect_objects(image_path):

    results = model(image_path)

    output_name = "output_" + os.path.basename(image_path)

    results[0].save(
        filename=os.path.join("static", output_name)
    )

    labels = []

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        label = model.names[class_id]

        labels.append(label)

    object_counts = Counter(labels)

    objects = []

    for label, count in object_counts.items():

        objects.append({
            "label": label.title(),
            "count": count
        })

    return output_name, objects
