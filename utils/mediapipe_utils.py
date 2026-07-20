import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_landmarks(frame):
    """
    Returns a flattened array of 132 values.
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:

        data = []

        for landmark in results.pose_landmarks.landmark:

            data.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])

        return np.array(data)

    return np.zeros(132)


def sample_frames(cap, sequence_length):
    """
    Uniformly sample frames from a video.
    """

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= sequence_length:

        indices = np.arange(total_frames)

    else:

        indices = np.linspace(
            0,
            total_frames - 1,
            sequence_length,
            dtype=int
        )

    return indices