from text_classifier import classify_email
from yolo_detector import detect_objects

def automation_system(email, image):

    email_result = classify_email(email)

    print("Email Result:", email_result)

    detect_objects(image)

    print("Automation Completed")