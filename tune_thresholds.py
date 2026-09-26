import itertools

from src.config import (
    EMBEDDING_MODEL,
    TRAIN_FILE,
)

from src.data_loader import (
    load_training_data,
)

from src.preprocessing import (
    prepare_dataframe,
)

from src.embeddings import (
    EmbeddingModel,
)

from src.reply_scorer import (
    predict_episode,
)


# ============================================================
# SEARCH SPACE
# ============================================================

THRESHOLDS = [
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
    0.32,
    0.34,
    0.36,
    0.38,
    0.40,
]


MARGINS = [
    0.000,
    0.002,
    0.005,
    0.010,
    0.015,
    0.020,
    0.030,
]


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    prepared_episodes,
    predictions,
):
    """
    Calculate reply-link accuracy.
    """

    total = 0
    correct = 0

    for episode_df, pred_episode in zip(
        prepared_episodes,
        predictions,
    ):

        true_map = {
            row["message_id"]:
                row["reply_to_message_id"]
            for _, row in episode_df.iterrows()
        }

        for prediction in pred_episode:

            message_id = prediction[
                "message_id"
            ]

            predicted = prediction[
                "reply_to_message_id"
            ]

            actual = true_map[
                message_id
            ]

            total += 1

            if predicted == actual:
                correct += 1

    if total == 0:
        return 0.0

    return correct / total


# ============================================================
# PREPARE EPISODES + EMBEDDINGS ONCE
# ============================================================

def prepare_episodes(
    data,
    embedding_model,
):
    """
    Prepare the complete training DataFrame,
    split it into episodes, and generate embeddings
    once for each episode.
    """

    # --------------------------------------------------------
    # SAME PREPROCESSING FLOW AS pipeline.py
    # --------------------------------------------------------

    data = prepare_dataframe(
        data
    )

    prepared_episodes = []

    print()
    print(
        "Preparing episodes and generating embeddings..."
    )

    # --------------------------------------------------------
    # GROUP BY EPISODE
    # --------------------------------------------------------

    grouped = data.groupby(
        "episode_id",
        sort=False,
    )

    total_episodes = data[
        "episode_id"
    ].nunique()

    for episode_number, (
        episode_id,
        episode_df,
    ) in enumerate(
        grouped,
        start=1,
    ):

        # ----------------------------------------------------
        # SAME SORTING AS pipeline.py
        # ----------------------------------------------------

        episode_df = (
            episode_df
            .sort_values("timestamp")
            .reset_index(drop=True)
            .copy()
        )

        # ----------------------------------------------------
        # EPISODE POSITION
        # ----------------------------------------------------

        episode_df[
            "_episode_index"
        ] = range(
            len(episode_df)
        )

        # ----------------------------------------------------
        # GENERATE EMBEDDINGS ONCE
        # ----------------------------------------------------

        texts = episode_df[
            "text"
        ].tolist()

        embeddings = embedding_model.encode(
            texts
        )

        prepared_episodes.append(
            {
                "df": episode_df,
                "embeddings": embeddings,
            }
        )

        print(
            f"  [{episode_number:02d}/{total_episodes}] "
            f"Episode {episode_id}: "
            f"{len(episode_df)} messages"
        )

    return prepared_episodes


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TASK A THRESHOLD TUNING")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    data = load_training_data(
        TRAIN_FILE
    )

    print(
        f"Training messages: {len(data)}"
    )

    print(
        f"Training episodes: "
        f"{data['episode_id'].nunique()}"
    )

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL ONCE
    # --------------------------------------------------------

    model = EmbeddingModel(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # PRECOMPUTE EPISODES + EMBEDDINGS
    # --------------------------------------------------------

    cached_episodes = prepare_episodes(
        data,
        model,
    )

    prepared_episodes = [
        item["df"]
        for item in cached_episodes
    ]

    # --------------------------------------------------------
    # SEARCH CONFIGURATION
    # --------------------------------------------------------

    best_accuracy = -1.0

    best_config = None

    results = []

    total_tests = (
        len(THRESHOLDS)
        * len(MARGINS)
    )

    current_test = 0

    print()
    print(
        f"Testing {total_tests} configurations..."
    )
    print()

    # --------------------------------------------------------
    # GRID SEARCH
    # --------------------------------------------------------

    for threshold, margin in itertools.product(
        THRESHOLDS,
        MARGINS,
    ):

        current_test += 1

        predictions = []

        # ----------------------------------------------------
        # PREDICT USING CACHED EMBEDDINGS
        # ----------------------------------------------------

        for item in cached_episodes:

            episode_df = item[
                "df"
            ]

            embeddings = item[
                "embeddings"
            ]

            pred = predict_episode(
                episode_df,
                embeddings,
                none_threshold=threshold,
                min_margin=margin,
            )

            predictions.append(
                pred
            )

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        accuracy = evaluate_predictions(
            prepared_episodes,
            predictions,
        )

        results.append(
            (
                accuracy,
                threshold,
                margin,
            )
        )

        print(
            f"[{current_test:02d}/{total_tests}] "
            f"threshold={threshold:.3f} | "
            f"margin={margin:.3f} | "
            f"accuracy={accuracy:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST
        # ----------------------------------------------------

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_config = {
                "none_threshold": threshold,
                "min_margin": margin,
            }

    # ========================================================
    # SORT RESULTS
    # ========================================================

    results.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    # ========================================================
    # TOP 10
    # ========================================================

    print()
    print("=" * 60)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 60)

    for rank, (
        accuracy,
        threshold,
        margin,
    ) in enumerate(
        results[:10],
        start=1,
    ):

        print(
            f"{rank:02d}. "
            f"accuracy={accuracy:.4f} | "
            f"threshold={threshold:.3f} | "
            f"margin={margin:.3f}"
        )

    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    print()
    print("=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)

    if best_config is None:

        print(
            "No valid configuration found."
        )

        return

    print(
        f"NONE_THRESHOLD = "
        f"{best_config['none_threshold']:.3f}"
    )

    print(
        f"MIN_MARGIN = "
        f"{best_config['min_margin']:.3f}"
    )

    print(
        f"Reply accuracy = "
        f"{best_accuracy:.4f}"
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    print()
    print("=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)

    baseline = 0.6231

    improvement = (
        best_accuracy - baseline
    )

    print(
        f"Original accuracy : "
        f"{baseline:.4f}"
    )

    print(
        f"Best accuracy     : "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Difference        : "
        f"{improvement:+.4f}"
    )

    # ========================================================
    # CONFIG.PY VALUES
    # ========================================================

    print()
    print("=" * 60)
    print("CONFIG.PY VALUES")
    print("=" * 60)

    print()
    print(
        f"NONE_THRESHOLD = "
        f"{best_config['none_threshold']:.3f}"
    )

    print(
        f"MIN_MARGIN = "
        f"{best_config['min_margin']:.3f}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()