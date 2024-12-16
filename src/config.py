import os

# Paths
DATA_DIR = "../data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALID_FILE = os.path.join(DATA_DIR, "valid.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_without_answers.csv")
SUBMISSION_FILE = "../outputs/submission.csv"
LOG_DIR = "../outputs/logs"

# Model parameters
BASE_MODEL = "ai-forever/ru-en-RoSBERTa"
EMBEDDING_DIM = 1024
MAX_LEN = 64
NUM_CLASSES = 7

# Training parameters
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-5
SEED = 42

# Labels
LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
