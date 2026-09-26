from src.config import (
    TRAIN_FILE,
    EMBEDDING_MODEL,
)

from src.data_loader import (
    load_training_data,
)

from src.embeddings import (
    EmbeddingModel,
)

from src.pipeline import (
    generate_predictions,
)

from src.evaluator import (
    reply_accuracy,
    thread_pairwise_f1,
)


def main():

    print(
        "Loading Task A training data..."
    )

    truth = load_training_data(
        TRAIN_FILE
    )

    model = EmbeddingModel(
        EMBEDDING_MODEL
    )

    print(
        "Generating predictions..."
    )

    predictions = generate_predictions(
        truth[
            [
                "episode_id",
                "message_id",
                "sender",
                "timestamp",
                "text",
            ]
        ],
        model,
    )

    reply_acc = reply_accuracy(
        predictions,
        truth,
    )

    thread_metrics = thread_pairwise_f1(
        predictions,
        truth,
    )

    print()
    print("=" * 60)
    print("TASK A BASELINE")
    print("=" * 60)

    print(
        f"Reply accuracy: "
        f"{reply_acc:.4f}"
    )

    print(
        f"Thread precision: "
        f"{thread_metrics['precision']:.4f}"
    )

    print(
        f"Thread recall: "
        f"{thread_metrics['recall']:.4f}"
    )

    print(
        f"Thread F1: "
        f"{thread_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()