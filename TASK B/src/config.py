from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"

MESSAGES_FILE = DATA_DIR / "taskB_messages_public.csv"
CANDIDATES_FILE = DATA_DIR / "taskB_candidates_public.csv"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Required columns
MESSAGE_COLUMNS = [
    "episode_id",
    "message_id",
    "sender",
    "timestamp",
    "text",
]

CANDIDATE_COLUMNS = [
    "episode_id",
    "gap_id",
    "gap_position",
    "candidate_id",
    "candidate_text",
    "candidate_sender",
    "candidate_timestamp",
    "feature_sender_pattern_fit",
    "feature_timestamp_gap_fit",
    "feature_thread_position_fit",
    "feature_embedding_similarity",
]


# Baseline feature weights
WEIGHTS = {
    "sender": 0.20,
    "timestamp": 0.20,
    "thread_position": 0.20,
    "embedding": 0.40,
}