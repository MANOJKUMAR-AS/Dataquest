from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from src.context_builder import build_gap_contexts


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    if not isinstance(text, str):
        return []

    return re.findall(r"\b\w+\b", text.lower())


def bm25_score(
    query: str,
    document: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    BM25-style similarity between a candidate/query and one document.

    This is a lightweight implementation intended for the relatively
    small Task B dataset.
    """

    query_tokens = tokenize(query)
    document_tokens = tokenize(document)

    if not query_tokens or not document_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    document_counts = Counter(document_tokens)

    document_length = len(document_tokens)

    # For a single document comparison, use a neutral average length.
    avg_document_length = max(document_length, 1)

    score = 0.0

    for token in query_counts:

        if token not in document_counts:
            continue

        tf = document_counts[token]

        numerator = tf * (k1 + 1)

        denominator = (
            tf
            + k1
            * (
                1
                - b
                + b
                * (
                    document_length
                    / avg_document_length
                )
            )
        )

        score += numerator / denominator

    return float(score)


def max_message_bm25(
    candidate_text: str,
    messages: list[str],
) -> float:
    """Maximum BM25 similarity against individual messages."""

    if not messages:
        return 0.0

    scores = [
        bm25_score(candidate_text, message)
        for message in messages
    ]

    return max(scores) if scores else 0.0


def mean_message_bm25(
    candidate_text: str,
    messages: list[str],
) -> float:
    """Mean BM25 similarity against individual messages."""

    if not messages:
        return 0.0

    scores = [
        bm25_score(candidate_text, message)
        for message in messages
    ]

    return sum(scores) / len(scores) if scores else 0.0


def combined_bm25(
    candidate_text: str,
    context_text: str,
) -> float:
    """BM25 similarity against concatenated context."""

    return bm25_score(
        candidate_text,
        context_text,
    )


def calculate_bm25_features(
    messages: pd.DataFrame,
    candidates: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Calculate BM25 features using the corrected canonical context.

    Features:
        bm25_before
        bm25_after
        bm25_combined

        bm25_max_before
        bm25_max_after

        bm25_mean_before
        bm25_mean_after
    """

    contexts = build_gap_contexts(
        messages,
        candidates,
        window=window,
    )

    # ---------------------------------------------------------
    # Convert context text back into individual messages
    # ---------------------------------------------------------
    # We reconstruct the individual messages directly here so
    # that BM25 can measure candidate-to-message similarity.
    # The canonical context builder remains responsible for
    # positioning the gap correctly.
    # ---------------------------------------------------------

    messages = messages.copy()

    messages["timestamp"] = pd.to_datetime(
        messages["timestamp"],
        errors="coerce",
    )

    gaps = (
        candidates[
            [
                "episode_id",
                "gap_id",
                "gap_position",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    rows = []

    for _, candidate in candidates.iterrows():

        episode_id = candidate["episode_id"]
        gap_id = candidate["gap_id"]
        gap_position = int(candidate["gap_position"])
        candidate_text = str(
            candidate.get("candidate_text", "")
        )

        context_row = contexts[
            contexts["gap_id"] == gap_id
        ]

        if context_row.empty:
            continue

        context_row = context_row.iloc[0]

        before_text = context_row["before_text"]
        after_text = context_row["after_text"]
        combined_text = context_row["combined_text"]

        # -----------------------------------------------------
        # Get individual context messages using the corrected
        # observed index.
        # -----------------------------------------------------
        episode_messages = messages[
            messages["episode_id"] == episode_id
        ].copy()

        episode_messages = episode_messages.sort_values(
            ["timestamp", "message_id"]
        ).reset_index(drop=True)

        prior_gaps = gaps[
            (gaps["episode_id"] == episode_id)
            & (gaps["gap_position"] < gap_position)
        ]

        prior_gap_count = len(prior_gaps)

        observed_index = (
            gap_position
            - 1
            - prior_gap_count
        )

        observed_index = max(
            0,
            min(
                observed_index,
                len(episode_messages),
            ),
        )

        before_messages_df = episode_messages.iloc[
            max(0, observed_index - window):
            observed_index
        ]

        after_messages_df = episode_messages.iloc[
            observed_index:
            observed_index + window
        ]

        before_messages = (
            before_messages_df["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        after_messages = (
            after_messages_df["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        # -----------------------------------------------------
        # Standard context BM25
        # -----------------------------------------------------
        bm25_before = combined_bm25(
            candidate_text,
            before_text,
        )

        bm25_after = combined_bm25(
            candidate_text,
            after_text,
        )

        bm25_combined = combined_bm25(
            candidate_text,
            combined_text,
        )

        # -----------------------------------------------------
        # Individual-message BM25
        # -----------------------------------------------------
        bm25_max_before = max_message_bm25(
            candidate_text,
            before_messages,
        )

        bm25_max_after = max_message_bm25(
            candidate_text,
            after_messages,
        )

        bm25_mean_before = mean_message_bm25(
            candidate_text,
            before_messages,
        )

        bm25_mean_after = mean_message_bm25(
            candidate_text,
            after_messages,
        )

        rows.append(
            {
                "episode_id": episode_id,
                "gap_id": gap_id,
                "candidate_id": candidate["candidate_id"],

                "bm25_before": bm25_before,
                "bm25_after": bm25_after,
                "bm25_combined": bm25_combined,

                "bm25_max_before": bm25_max_before,
                "bm25_max_after": bm25_max_after,

                "bm25_mean_before": bm25_mean_before,
                "bm25_mean_after": bm25_mean_after,
            }
        )

    return pd.DataFrame(rows)