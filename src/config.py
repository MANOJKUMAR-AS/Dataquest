from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


TRAIN_FILE = DATA_DIR / "taskA_train.csv"
HIDDEN_FILE = DATA_DIR / "taskA_hidden_public.csv"


SUBMISSION_FILE = (
    OUTPUT_DIR / "submission_taskA.csv"
)


EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Combined reply scoring weights.
#
# The seven signals are:
#   1. embedding
#   2. lexical
#   3. token overlap
#   4. sender
#   5. time
#   6. position
#   7. question/answer
#
# These are the starting weights.
# We will tune them after the pipeline runs correctly.

WEIGHTS = {

    "embedding": 0.65,

    "lexical": 0.10,

    "token_overlap": 0.05,

    "sender": 0.05,

    "time": 0.05,

    "position": 0.03,

    "question_answer": 0.07,
}


# These will be tuned again because
# token_overlap is now part of the score.

NONE_THRESHOLD = 0.30

MIN_MARGIN = 0.005