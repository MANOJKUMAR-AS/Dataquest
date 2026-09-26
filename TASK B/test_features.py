import pandas as pd

from src.feature_builder import build_features


def main():

    features = build_features()

    print("\n" + "=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)

    feature_columns = [
        "feature_sender_pattern_fit",
        "feature_timestamp_gap_fit",
        "feature_thread_position_fit",
        "feature_embedding_similarity",
        "bm25_before",
        "bm25_after",
        "bm25_combined",
        "bge_before",
        "bge_after",
        "bge_combined",
    ]

    print("\nFeature ranges:\n")

    for column in feature_columns:

        minimum = features[column].min()
        maximum = features[column].max()
        mean = features[column].mean()

        print(
            f"{column:35s}"
            f"min={minimum:8.4f} "
            f"max={maximum:8.4f} "
            f"mean={mean:8.4f}"
        )

    print("\n" + "=" * 70)
    print("FIRST 20 ROWS")
    print("=" * 70)

    print(
        features[
            [
                "gap_id",
                "candidate_id",
                "feature_sender_pattern_fit",
                "feature_timestamp_gap_fit",
                "feature_thread_position_fit",
                "feature_embedding_similarity",
                "bm25_before",
                "bm25_after",
                "bm25_combined",
                "bge_before",
                "bge_after",
                "bge_combined",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()