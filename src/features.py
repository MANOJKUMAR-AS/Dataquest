import math
import re
from datetime import timedelta

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


QUESTION_PATTERNS = [
    r"\?",
    r"^(who|what|when|where|why|how|which|whose|whom)\b",
    r"^(can|could|would|will|should|do|does|did|is|are|was|were)\b",
]


def lexical_similarity(text_a, text_b):
    """
    TF-IDF cosine similarity between two messages.
    """

    if not text_a.strip() or not text_b.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
    )

    try:
        matrix = vectorizer.fit_transform(
            [text_a, text_b]
        )

        return float(
            cosine_similarity(
                matrix[0:1],
                matrix[1:2],
            )[0][0]
        )

    except ValueError:
        return 0.0


def token_overlap(text_a, text_b):
    """
    Jaccard overlap of tokens.
    """

    a = set(re.findall(r"\b\w+\b", text_a.lower()))
    b = set(re.findall(r"\b\w+\b", text_b.lower()))

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def is_question(text):
    text = text.strip().lower()

    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, text):
            return True

    return False


def question_answer_score(parent_text, child_text):
    """
    Lightweight dialogue relationship heuristic.

    A question followed by a non-question gets a positive score.

    This is deliberately conservative.
    """

    parent_question = is_question(parent_text)
    child_question = is_question(child_text)

    if parent_question and not child_question:
        return 1.0

    if parent_question and child_question:
        return 0.35

    if not parent_question and not child_question:
        return 0.55

    return 0.25


def sender_score(parent_sender, child_sender):
    """
    A person generally does not reply to themselves.

    Return:
        1.0 -> different senders
        0.0 -> same sender
    """

    if parent_sender != child_sender:
        return 1.0

    return 0.0


def time_score(parent_time, child_time):
    """
    Convert time gap into a soft compatibility score.

    We do NOT force the nearest timestamp to win.
    """

    gap_seconds = (
        child_time - parent_time
    ).total_seconds()

    if gap_seconds <= 0:
        return 0.0

    gap_hours = gap_seconds / 3600.0

    # Smooth decay.
    #
    # 0 hours -> ~1
    # 1 hour  -> ~0.76
    # 6 hours -> ~0.38
    # 24 hrs  -> ~0.12

    score = math.exp(
        -gap_hours / 4.0
    )

    return float(score)


def position_score(parent_index, child_index):
    distance = child_index - parent_index

    if distance <= 0:
        return 0.0

    # Soft positional decay.
    return float(
        1.0 / math.sqrt(distance)
    )


def build_pair_features(
    parent,
    child,
    parent_embedding,
    child_embedding,
):
    semantic = float(
        np.dot(
            parent_embedding,
            child_embedding,
        )
    )

    lexical = lexical_similarity(
        parent["normalized_text"],
        child["normalized_text"],
    )

    overlap = token_overlap(
        parent["normalized_text"],
        child["normalized_text"],
    )

    sender = sender_score(
        parent["sender"],
        child["sender"],
    )

    time = time_score(
        parent["timestamp"],
        child["timestamp"],
    )

    position = position_score(
        parent["_episode_index"],
        child["_episode_index"],
    )

    qa = question_answer_score(
        parent["text"],
        child["text"],
    )

    return {
        "embedding": semantic,
        "lexical": lexical,
        "token_overlap": overlap,
        "sender": sender,
        "time": time,
        "position": position,
        "question_answer": qa,
    }