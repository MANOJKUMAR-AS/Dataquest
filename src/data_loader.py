import pandas as pd


REQUIRED_TRAIN_COLUMNS = [
    "episode_id",
    "message_id",
    "sender",
    "timestamp",
    "text",
    "reply_to_message_id",
    "thread_id",
]

REQUIRED_HIDDEN_COLUMNS = [
    "episode_id",
    "message_id",
    "sender",
    "timestamp",
    "text",
]


def load_csv(path):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    return df


def validate_columns(df, required_columns, name="dataset"):
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def load_training_data(path):
    df = load_csv(path)

    validate_columns(
        df,
        REQUIRED_TRAIN_COLUMNS,
        "Task A training dataset",
    )

    return df


def load_hidden_data(path):
    df = load_csv(path)

    validate_columns(
        df,
        REQUIRED_HIDDEN_COLUMNS,
        "Task A hidden dataset",
    )

    return df