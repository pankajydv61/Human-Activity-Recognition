# Dataset Paths
DATASET_PATH = "dataset/raw/UCF-101"
PROCESSED_PATH = "dataset/processed"
MODEL_PATH = "models/activity_model.keras"

# Activities
CLASSES = [
    "Basketball",
    "BodyWeightSquats",
    "BoxingSpeedBag",
    "GolfSwing",
    "JumpingJack",
    "PullUps",
    "Punch",
    "PushUps",
    "TaiChi",
    "WalkingWithDog"
]

# Data Settings
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 33
FEATURES = 132

# Training Settings
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
RANDOM_STATE = 42