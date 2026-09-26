import pandas as pd

from .config import (
    MESSAGES_FILE,
    CANDIDATES_FILE,
    MESSAGE_COLUMNS,
    CANDIDATE_COLUMNS,
)


def load_messages():
    df = pd.read_csv(MESSAGES_FILE)

    missing = [col for col in MESSAGE_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Messages file is missing columns: {missing}"
        )

    return df


def load_candidates():
    df = pd.read_csv(CANDIDATES_FILE)

    missing = [
        col for col in CANDIDATE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Candidates file is missing columns: {missing}"
        )

    return df


def validate_data(messages, candidates):

    if messages.empty:
        raise ValueError("Messages dataset is empty.")

    if candidates.empty:
        raise ValueError("Candidates dataset is empty.")

    # Numeric feature validation
    feature_columns = [
        "feature_sender_pattern_fit",
        "feature_timestamp_gap_fit",
        "feature_thread_position_fit",
        "feature_embedding_similarity",
    ]

    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(candidates[col]):
            raise TypeError(
                f"{col} must be numeric."
            )

        if candidates[col].isna().any():
            raise ValueError(
                f"{col} contains missing values."
            )

        if ((candidates[col] < 0) | (candidates[col] > 1)).any():
            raise ValueError(
                f"{col} must contain values between 0 and 1."
            )


def load_data():

    messages = load_messages()
    candidates = load_candidates()

    validate_data(messages, candidates)

    return messages, candidates