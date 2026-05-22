from flask import Flask, render_template, request
from text_classifier import classify_email
from yolo_detector import detect_objects
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def home():

    email_result = ""
    image_result = ""

    if request.method == "POST":

        # EMAIL CLASSIFICATION
        email_text = request.form.get("email")

        if email_text:
            email_result = classify_email(email_text)

        # IMAGE DETECTION
        image = request.files.get("image")

        if image and image.filename != "":

            image_path = os.path.join(
                UPLOAD_FOLDER,
                image.filename
            )

            image.save(image_path)

            detect_objects(image_path)

            image_result = f"output_{image.filename}"

    return render_template(
        "index.html",
        email_result=email_result,
        image_result=image_result
    )

if __name__ == "__main__":
    app.run(debug=True)