from flask import Flask, render_template, request
from text_classifier import classify_email
from yolo_detector import detect_objects
import os
import re

app = Flask(__name__)


UPLOAD_FOLDER = "static"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# @app.route("/", methods=["GET", "POST"])
# def home():

    # email_result = ""
    # image_result = ""

    # if request.method == "POST":

    #     # EMAIL CLASSIFICATION
    #     email_text = request.form.get("email")

    #     if email_text:
    #         email_result = classify_email(email_text)

    #     # IMAGE DETECTION
    #     image = request.files.get("image")

    #     if image and image.filename != "":

    #         image_path = os.path.join(
    #             UPLOAD_FOLDER,
    #             image.filename
    #         )

    #         image.save(image_path)

    #         detect_objects(image_path)

    #         image_result = f"output_{image.filename}"

    # return render_template(
    #     "index.html",
    #     email_result=email_result,
    #     image_result=image_result
    # )

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


def is_valid_email_text(text):
    words = len(text.split())

    return (
        words >= 5 and
        re.search(r"[a-zA-Z]", text)
    )

@app.route("/email", methods=["GET", "POST"])
def email():

    results = []

    if request.method == "POST":

        emails = request.form.getlist("email")

        for email in emails:

            if email.strip():

                if not is_valid_email_text(email):

                    results.append({
                        "email": email,
                        "prediction": "Invalid Email Text"
                    })

                else:

                    prediction = classify_email(email)

                    results.append({
                        "email": email,
                        "prediction": prediction
                    })

    return render_template(
        "email.html",
        results=results
    )


# @app.route("/image")
# def image():
#     return render_template("image.html")

@app.route("/image", methods=["GET", "POST"])
def image():

    results = []

    if request.method == "POST":

        images = request.files.getlist("image")

        for image in images:

            if image.filename != "":

                image_path = os.path.join(
                    UPLOAD_FOLDER,
                    image.filename
                )

                image.save(image_path)

                detected_image, objects = detect_objects(image_path)

                results.append({

                    "original": image.filename,

                    "detected": detected_image,

                    "objects": objects

                })

    return render_template(
        "image.html",
        results=results
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)