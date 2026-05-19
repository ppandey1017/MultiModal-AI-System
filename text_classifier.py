import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import joblib

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("dataset.csv")

# # Keep only needed columns
# df = df[['title', 'text', 'type']]
#
# # Remove missing values
# df = df.dropna()
# Keep only needed columns
df = df[['text', 'spam']]

# Remove missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

print("\nClass Distribution:\n")
print(df['spam'].value_counts())

# Remove missing values
# df = df.dropna()

# Remove duplicate rows
# df = df.drop_duplicates()

# Check class distribution
# print("\nClass Distribution:\n")
# print(df['type'].value_counts())

# Balance dataset
min_count = df['spam'].value_counts().min()

balanced_df = df.groupby('spam').sample(
    n=min_count,
    random_state=42
)

df = balanced_df.reset_index(drop=True)

print("\nBalanced Dataset:\n")
print(df['spam'].value_counts())

# =========================
# COMBINE TITLE + TEXT
# =========================

# df['message'] = df['title'] + " " + df['text']
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# nltk.download('stopwords')
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# =========================
# TEXT CLEANING FUNCTION
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        # if word not in stopwords.words('english')
        if word not in stop_words
    ]

    return " ".join(words)

# =========================
# COMBINE & CLEAN TEXT
# =========================

df['message'] =df['text']

df['message'] = df['message'].apply(clean_text)

# INPUT & OUTPUT
X = df['message']
y = df['spam']

# =========================
# TF-IDF VECTORIZATION
# =========================

vectorizer = TfidfVectorizer(
    stop_words='english',
    # stop_words = set(stopwords.words('english')),
    max_features=5000
)
ps = PorterStemmer()

X = vectorizer.fit_transform(X).toarray()

# =========================
# LABEL ENCODING
# =========================

# encoder = LabelEncoder()
#
# y = encoder.fit_transform(y)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# BUILD MODEL
# =========================

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

model.save("email_classifier_model.h5")

joblib.dump(vectorizer, "vectorizer.pkl")

# joblib.dump(encoder, "encoder.pkl")

print("\nModel Saved Successfully!")

# =========================
# PREDICTION FUNCTION
# =========================

def classify_email(email):

    cleaned_email = clean_text(email)

    vec = vectorizer.transform([cleaned_email]).toarray()

    pred = model.predict(vec)

    if pred[0][0] > 0.5:
        return "spam"
    else:
        return "not spam"



# =========================
# ACCURACY GRAPH
# =========================

plt.figure(figsize=(10, 5))

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.savefig("accuracy_graph.png")

plt.show()

# =========================
# LOSS GRAPH
# =========================

plt.figure(figsize=(10, 5))

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.savefig("loss_graph.png")

plt.show()

print("\nGraphs Saved Successfully!")