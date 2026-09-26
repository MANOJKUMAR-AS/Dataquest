import pandas as pd


def preprocess_messages(messages):

    messages = messages.copy()

    messages["timestamp"] = pd.to_datetime(
        messages["timestamp"],
        errors="coerce"
    )

    if messages["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps found in messages."
        )

    messages["text"] = (
        messages["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    messages = messages.sort_values(
        ["episode_id", "timestamp", "message_id"]
    ).reset_index(drop=True)

    return messages


def preprocess_candidates(candidates):

    candidates = candidates.copy()

    candidates["candidate_timestamp"] = pd.to_datetime(
        candidates["candidate_timestamp"],
        errors="coerce"
    )

    if candidates["candidate_timestamp"].isna().any():
        raise ValueError(
            "Invalid candidate timestamps found."
        )

    candidates["candidate_text"] = (
        candidates["candidate_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return candidates