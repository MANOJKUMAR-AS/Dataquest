import pandas as pd

from src.feature_normalizer import build_normalized_features


def main():

    df = build_normalized_features()

    print("\n" + "=" * 80)
    print("NORMALIZED FEATURE EXAMPLE")
    print("=" * 80)

    columns = [
        "gap_id",
        "candidate_id",

        "feature_sender_pattern_fit_norm",
        "feature_timestamp_gap_fit_norm",
        "feature_thread_position_fit_norm",
        "feature_embedding_similarity_norm",

        "bm25_before_norm",
        "bm25_after_norm",
        "bm25_combined_norm",

        "bge_before_norm",
        "bge_after_norm",
        "bge_combined_norm",
    ]

    print(
        df[columns]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()