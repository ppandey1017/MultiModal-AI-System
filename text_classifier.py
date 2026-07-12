import re
import joblib
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords", quiet=True)

ps = PorterStemmer()

stop_words = set(stopwords.words("english"))

# Load trained model

model = tf.keras.models.load_model("models/email_classifier_model.h5")

vectorizer = joblib.load("models/vectorizer.pkl")


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [

        ps.stem(word)

        for word in words

        if word not in stop_words

    ]

    return " ".join(words)


# def classify_email(email):

#     cleaned_email = clean_text(email)

#     vec = vectorizer.transform([cleaned_email]).toarray()

#     pred = model.predict(vec, verbose=0)

#     if pred[0][0] > 0.5:
#         return "Spam"

#     return "Not Spam"

def classify_email(email):

    cleaned_email = clean_text(email)

    print("Original:", email)
    print("Cleaned :", cleaned_email)

    vec = vectorizer.transform([cleaned_email]).toarray()

    pred = model.predict(vec, verbose=0)

    print("Prediction Probability:", pred[0][0])

    if pred[0][0] > 0.5:
        return "spam"

    return "Not Spam"