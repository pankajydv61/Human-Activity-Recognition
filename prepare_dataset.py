import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from config import PROCESSED_PATH, CLASSES, SEQUENCE_LENGTH

X = []
y = []

for label, activity in enumerate(CLASSES):

    activity_path = Path(PROCESSED_PATH) / activity

    if not activity_path.exists():
        continue

    files = activity_path.glob("*.npy")

    for file in files:

        sequence = np.load(file)

        # Pad or trim every sequence to SEQUENCE_LENGTH
        if len(sequence) >= SEQUENCE_LENGTH:
            sequence = sequence[:SEQUENCE_LENGTH]
        else:
            padding = np.zeros((SEQUENCE_LENGTH - len(sequence), 132))
            sequence = np.vstack([sequence, padding])

        X.append(sequence)
        y.append(label)

X = np.array(X)
y = to_categorical(y)

print("Dataset Shape :", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

os.makedirs("dataset/processed/final", exist_ok=True)

np.save("dataset/processed/final/X_train.npy", X_train)
np.save("dataset/processed/final/X_test.npy", X_test)
np.save("dataset/processed/final/y_train.npy", y_train)
np.save("dataset/processed/final/y_test.npy", y_test)

print("Dataset prepared successfully!")