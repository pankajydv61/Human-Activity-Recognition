from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    DATASET_PATH,
    PROCESSED_PATH,
    CLASSES,
    SEQUENCE_LENGTH
)

from utils.mediapipe_utils import (
    extract_landmarks,
    sample_frames
)

Path(PROCESSED_PATH).mkdir(exist_ok=True)

for activity in CLASSES:

    input_folder = Path(DATASET_PATH) / activity

    output_folder = Path(PROCESSED_PATH) / activity

    output_folder.mkdir(parents=True, exist_ok=True)

    videos = list(input_folder.glob("*.avi"))

    print(f"\nProcessing {activity}")

    for video in tqdm(videos):

        cap = cv2.VideoCapture(str(video))

        frame_indices = sample_frames(cap, SEQUENCE_LENGTH)

        sequence = []

        current = 0
        index_pointer = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if index_pointer < len(frame_indices):

                if current == frame_indices[index_pointer]:

                    sequence.append(
                        extract_landmarks(frame)
                    )

                    index_pointer += 1

            current += 1

        cap.release()

        sequence = np.array(sequence)

        if len(sequence) < SEQUENCE_LENGTH:

            padding = np.zeros(
                (
                    SEQUENCE_LENGTH - len(sequence),
                    132
                )
            )

            sequence = np.vstack([sequence, padding])

        np.save(
            output_folder / f"{video.stem}.npy",
            sequence
        )

print("\nDataset processing completed.")