from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.context_builder import build_gap_contexts


MODEL_NAME = "BAAI/bge-large-en-v1.5"


def load_model():
    return SentenceTransformer(MODEL_NAME)


def encode_texts(
    model,
    texts: list[str],
) -> np.ndarray:

    if not texts:
        return np.empty((0, 0))

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.asarray(embeddings)


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    if a.size == 0 or b.size == 0:
        return 0.0

    return float(np.dot(a, b))


def calculate_embedding_features(
    messages: pd.DataFrame,
    candidates: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:

    print("Building corrected gap contexts...")

    contexts = build_gap_contexts(
        messages,
        candidates,
        window=window,
    )

    model = load_model()

    # ---------------------------------------------------------
    # Build a lookup of context messages for each gap
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

    gap_messages = {}

    for _, gap in gaps.iterrows():

        episode_id = gap["episode_id"]
        gap_id = gap["gap_id"]
        gap_position = int(gap["gap_position"])

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

        before_df = episode_messages.iloc[
            max(0, observed_index - window):
            observed_index
        ]

        after_df = episode_messages.iloc[
            observed_index:
            observed_index + window
        ]

        before_texts = (
            before_df["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        after_texts = (
            after_df["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        gap_messages[gap_id] = {
            "before": before_texts,
            "after": after_texts,
        }

    # ---------------------------------------------------------
    # Encode candidate texts
    # ---------------------------------------------------------

    candidate_texts = (
        candidates["candidate_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        f"Encoding {len(candidate_texts)} candidate texts..."
    )

    candidate_embeddings = encode_texts(
        model,
        candidate_texts,
    )

    # ---------------------------------------------------------
    # Cache embeddings for all context messages
    # ---------------------------------------------------------

    all_context_texts = []

    for gap_id, data in gap_messages.items():

        all_context_texts.extend(
            data["before"]
        )

        all_context_texts.extend(
            data["after"]
        )

    # Remove duplicates while preserving order

    unique_context_texts = list(
        dict.fromkeys(all_context_texts)
    )

    print(
        f"Encoding {len(unique_context_texts)} unique context messages..."
    )

    context_embeddings = encode_texts(
        model,
        unique_context_texts,
    )

    context_lookup = {
        text: context_embeddings[i]
        for i, text in enumerate(unique_context_texts)
    }

    # ---------------------------------------------------------
    # Calculate features
    # ---------------------------------------------------------

    rows = []

    for idx, (_, candidate) in enumerate(
        candidates.iterrows()
    ):

        gap_id = candidate["gap_id"]

        candidate_embedding = candidate_embeddings[idx]

        candidate_text = str(
            candidate.get(
                "candidate_text",
                "",
            )
        )

        # -----------------------------------------------------
        # Context
        # -----------------------------------------------------

        data = gap_messages.get(
            gap_id,
            {
                "before": [],
                "after": [],
            },
        )

        before_texts = data["before"]
        after_texts = data["after"]

        before_embeddings = [
            context_lookup[text]
            for text in before_texts
            if text in context_lookup
        ]

        after_embeddings = [
            context_lookup[text]
            for text in after_texts
            if text in context_lookup
        ]

        # -----------------------------------------------------
        # Individual similarities
        # -----------------------------------------------------

        before_scores = [
            cosine_similarity(
                candidate_embedding,
                emb,
            )
            for emb in before_embeddings
        ]

        after_scores = [
            cosine_similarity(
                candidate_embedding,
                emb,
            )
            for emb in after_embeddings
        ]

        # -----------------------------------------------------
        # Helper functions
        # -----------------------------------------------------

        def safe_max(values):

            return (
                max(values)
                if values
                else 0.0
            )

        def safe_mean(values):

            return (
                float(np.mean(values))
                if values
                else 0.0
            )

        def top_k_mean(
            values,
            k=2,
        ):

            if not values:
                return 0.0

            sorted_values = sorted(
                values,
                reverse=True,
            )

            return float(
                np.mean(
                    sorted_values[:k]
                )
            )

        # -----------------------------------------------------
        # Weighted local similarity
        # -----------------------------------------------------

        def weighted_score(values):

            if not values:
                return 0.0

            total = 0.0
            weight_total = 0.0

            # nearest message gets highest weight
            weights = [
                1.0,
                0.7,
                0.5,
                0.3,
                0.2,
            ]

            for i, value in enumerate(values):

                weight = (
                    weights[i]
                    if i < len(weights)
                    else 0.1
                )

                total += value * weight
                weight_total += weight

            return total / weight_total

        # -----------------------------------------------------
        # Combined context similarity
        # -----------------------------------------------------

        context_row = contexts[
            contexts["gap_id"] == gap_id
        ]

        if context_row.empty:

            bge_before = 0.0
            bge_after = 0.0
            bge_combined = 0.0

        else:

            context_row = context_row.iloc[0]

            before_context = str(
                context_row["before_text"]
            )

            after_context = str(
                context_row["after_text"]
            )

            combined_context = str(
                context_row["combined_text"]
            )

            context_embeddings_local = model.encode(
                [
                    before_context,
                    after_context,
                    combined_context,
                ],
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            bge_before = cosine_similarity(
                candidate_embedding,
                context_embeddings_local[0],
            )

            bge_after = cosine_similarity(
                candidate_embedding,
                context_embeddings_local[1],
            )

            bge_combined = cosine_similarity(
                candidate_embedding,
                context_embeddings_local[2],
            )

        # -----------------------------------------------------
        # Store
        # -----------------------------------------------------

        rows.append(
            {
                "episode_id":
                    candidate["episode_id"],

                "gap_id":
                    gap_id,

                "candidate_id":
                    candidate["candidate_id"],

                # Existing context-level BGE
                "bge_before":
                    bge_before,

                "bge_after":
                    bge_after,

                "bge_combined":
                    bge_combined,

                # Individual-message BGE
                "bge_max_before":
                    safe_max(before_scores),

                "bge_max_after":
                    safe_max(after_scores),

                "bge_mean_before":
                    safe_mean(before_scores),

                "bge_mean_after":
                    safe_mean(after_scores),

                "bge_top2_before":
                    top_k_mean(
                        before_scores,
                        2,
                    ),

                "bge_top2_after":
                    top_k_mean(
                        after_scores,
                        2,
                    ),

                "bge_weighted_before":
                    weighted_score(
                        before_scores
                    ),

                "bge_weighted_after":
                    weighted_score(
                        after_scores
                    ),
            }
        )

    return pd.DataFrame(rows)