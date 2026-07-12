import os
import re
import joblib
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download("stopwords", quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


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


# ======================
# LOAD DATASET
# ======================

df = pd.read_csv("dataset.csv")

df = df[['text', 'spam']]

df = df.dropna()

df = df.drop_duplicates()

print(df['spam'].value_counts())

# Balance dataset
min_count = df['spam'].value_counts().min()

df = df.groupby('spam').sample(
    n=min_count,
    random_state=42
).reset_index(drop=True)

df['message'] = df['text'].apply(clean_text)

X = df['message']
y = df['spam']

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential([

    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")

])

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

history = model.fit(

    X_train,

    y_train,

    epochs=5,

    batch_size=32,

    validation_data=(X_test, y_test)

)

y_pred = (model.predict(X_test) > 0.5).astype(int)

print("Accuracy :", accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# ======================
# SAVE MODEL
# ======================

os.makedirs("models", exist_ok=True)

model.save("models/email_classifier_model.h5")

joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model Saved Successfully!")

# ======================
# ACCURACY GRAPH
# ======================

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(["Train","Validation"])

plt.title("Accuracy")
# print("Current Folder:", os.getcwd())
plt.savefig(os.path.join(os.getcwd(), "accuracy_graph.png"))
plt.show()

# ======================
# LOSS GRAPH
# ======================

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(["Train","Validation"])

plt.title("Loss")
plt.savefig(os.path.join(os.getcwd(), "loss_graph.png"))
plt.show()

# print(os.path.exists("accuracy_graph.png"))
# print(os.path.exists("loss_graph.png"))
print("Graphs Saved Successfully!")