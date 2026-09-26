import pandas as pd


STRUCTURAL_FEATURES = [
    "feature_sender_pattern_fit_norm",
    "feature_timestamp_gap_fit_norm",
    "feature_thread_position_fit_norm",
    "feature_embedding_similarity_norm",
]

BM25_FEATURES = [
    "bm25_before_norm",
    "bm25_after_norm",
    "bm25_combined_norm",
]

BGE_FEATURES = [
    "bge_before_norm",
    "bge_after_norm",
    "bge_combined_norm",
]


def calculate_family_scores(df):

    result = df.copy()

    # --------------------------------------------------
    # Structural score
    # --------------------------------------------------

    result["structural_score"] = result[
        STRUCTURAL_FEATURES
    ].mean(axis=1)

    # --------------------------------------------------
    # BM25 lexical score
    # --------------------------------------------------

    result["bm25_score"] = result[
        BM25_FEATURES
    ].mean(axis=1)

    # --------------------------------------------------
    # BGE semantic score
    # --------------------------------------------------

    result["bge_score"] = result[
        BGE_FEATURES
    ].mean(axis=1)

    return result


def calculate_hybrid_score(
    df,
    structural_weight=0.40,
    bm25_weight=0.20,
    bge_weight=0.40,
):

    result = calculate_family_scores(df)

    result["hybrid_score"] = (
        structural_weight * result["structural_score"]
        + bm25_weight * result["bm25_score"]
        + bge_weight * result["bge_score"]
    )

    return result


def rank_candidates(df):

    result = calculate_hybrid_score(df)

    result["rank"] = (
        result
        .groupby("gap_id")["hybrid_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    result = result.sort_values(
        ["gap_id", "rank"]
    ).reset_index(drop=True)

    return result


def create_submission(df):

    return (
        df[
            [
                "gap_id",
                "candidate_id",
                "rank",
            ]
        ]
        .sort_values(
            ["gap_id", "rank"]
        )
        .reset_index(drop=True)
    )


if __name__ == "__main__":

    input_file = "outputs/taskB_normalized_features.csv"

    ranked_file = "outputs/taskB_hybrid_ranked.csv"
    submission_file = "outputs/taskB_hybrid_submission.csv"

    print("Loading normalized features...")

    df = pd.read_csv(input_file)

    print(f"Candidates: {len(df)}")
    print(f"Gaps: {df['gap_id'].nunique()}")

    print("\nCalculating hybrid scores...")

    ranked = rank_candidates(df)

    print("\nRanking completed.")

    ranked.to_csv(
        ranked_file,
        index=False,
    )

    submission = create_submission(ranked)

    submission.to_csv(
        submission_file,
        index=False,
    )

    print(f"\nSaved ranked data to:")
    print(ranked_file)

    print(f"\nSaved submission to:")
    print(submission_file)

    print("\nFirst 20 rankings:")

    print(
        ranked[
            [
                "gap_id",
                "candidate_id",
                "structural_score",
                "bm25_score",
                "bge_score",
                "hybrid_score",
                "rank",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )