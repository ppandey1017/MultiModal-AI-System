from text_classifier import classify_email
from yolo_detector import detect_objects

email = "Congratulations! You won a free lottery ticket"

result = classify_email(email)

print("Email Category:", result)

detect_objects("text.png")