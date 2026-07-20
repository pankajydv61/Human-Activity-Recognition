import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import SEQUENCE_LENGTH, CLASSES
from utils.mediapipe_utils import extract_landmarks, sample_frames

model = load_model("models/activity_model.keras")


def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_indices = sample_frames(cap, SEQUENCE_LENGTH)

    sequence = []

    current = 0
    pointer = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if pointer < len(frame_indices) and current == frame_indices[pointer]:
            sequence.append(extract_landmarks(frame))
            pointer += 1

        current += 1

    cap.release()

    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(np.zeros(132))

    sequence = np.array(sequence[:SEQUENCE_LENGTH])
    sequence = np.expand_dims(sequence, axis=0)

    prediction = model.predict(sequence, verbose=0)[0]

    idx = np.argmax(prediction)

    return CLASSES[idx], float(prediction[idx])


if __name__ == "__main__":
    video = input("Enter video path: ")

    activity, confidence = predict_video(video)

    print(f"\nPrediction : {activity}")
    print(f"Confidence : {confidence:.2%}")