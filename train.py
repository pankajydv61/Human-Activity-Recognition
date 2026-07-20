import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Load prepared dataset
X_train = np.load("dataset/processed/final/X_train.npy")
X_test = np.load("dataset/processed/final/X_test.npy")
y_train = np.load("dataset/processed/final/y_train.npy")
y_test = np.load("dataset/processed/final/y_test.npy")

num_classes = y_train.shape[1]

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 132)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "models/activity_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=16,
    callbacks=[checkpoint]
)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nFinal Test Accuracy: {accuracy:.4f}")