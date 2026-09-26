from .config import (
    WEIGHTS,
    NONE_THRESHOLD,
    MIN_MARGIN,
)

from .features import build_pair_features


def weighted_score(features):
    """
    Combine all conversational features into one reply score.

    Features:
        embedding       -> semantic similarity
        lexical         -> word/phrase similarity
        token_overlap   -> shared vocabulary
        sender          -> sender relationship
        time            -> temporal plausibility
        position        -> chronological context
        question_answer -> question/answer relationship
    """

    score = (
        WEIGHTS["embedding"]
        * features["embedding"]

        + WEIGHTS["lexical"]
        * features["lexical"]

        + WEIGHTS["token_overlap"]
        * features["token_overlap"]

        + WEIGHTS["sender"]
        * features["sender"]

        + WEIGHTS["time"]
        * features["time"]

        + WEIGHTS["position"]
        * features["position"]

        + WEIGHTS["question_answer"]
        * features["question_answer"]
    )

    return float(score)


def predict_reply_for_message(
    episode_df,
    embeddings,
    child_index,
    none_threshold=None,
    min_margin=None,
):

    if none_threshold is None:
        none_threshold = NONE_THRESHOLD

    if min_margin is None:
        min_margin = MIN_MARGIN

    child = episode_df.iloc[child_index]
    child_embedding = embeddings[child_index]

    candidates = []

    # Only earlier messages can be parents.
    for parent_index in range(child_index):

        parent = episode_df.iloc[parent_index]

        features = build_pair_features(
            parent=parent,
            child=child,
            parent_embedding=embeddings[parent_index],
            child_embedding=child_embedding,
        )

        score = weighted_score(features)

        candidates.append(
            {
                "parent_index": parent_index,
                "parent_message_id": parent["message_id"],
                "score": score,
                "features": features,
            }
        )

    # First message in an episode starts a thread.
    if not candidates:
        return "NONE", candidates

    # Highest combined score first.
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    best = candidates[0]

    if len(candidates) > 1:

        second = candidates[1]

        margin = (
            best["score"]
            - second["score"]
        )

    else:

        margin = best["score"]

    # If even the best candidate is weak,
    # treat the message as starting a new thread.
    if best["score"] < none_threshold:
        return "NONE", candidates

    # If the top candidates are too close,
    # treat it conservatively as a new thread.
    if margin < min_margin:
        return "NONE", candidates

    return (
        best["parent_message_id"],
        candidates,
    )


def predict_episode(
    episode_df,
    embeddings,
    none_threshold=None,
    min_margin=None,
):

    predictions = []

    for i in range(len(episode_df)):

        reply_to, candidates = (
            predict_reply_for_message(
                episode_df,
                embeddings,
                i,
                none_threshold=none_threshold,
                min_margin=min_margin,
            )
        )

        predictions.append(
            {
                "message_id":
                    episode_df.iloc[i]["message_id"],

                "reply_to_message_id":
                    reply_to,

                "candidate_scores":
                    candidates,
            }
        )

    return predictions