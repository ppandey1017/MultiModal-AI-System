from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)

# LOAD DATA
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# NORMALIZE
X_train = X_train / 255.0
X_test = X_test / 255.0

# RESHAPE
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# CNN MODEL
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN
model.fit(X_train, y_train, epochs=3)

# TEST
loss, accuracy = model.evaluate(X_test, y_test)

print("CNN Accuracy:", accuracy)