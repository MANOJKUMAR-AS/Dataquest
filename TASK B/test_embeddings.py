from src.data_loader import load_messages, load_candidates
from src.embedding_features import calculate_embedding_features


messages = load_messages()
candidates = load_candidates()

features = calculate_embedding_features(
    messages,
    candidates,
    window=5,
)

print("\nRows:", len(features))

print("\nColumns:")
print(features.columns.tolist())

print("\nShape:")
print(features.shape)

print("\nC004-G01:")
print(
    features[
        features["gap_id"] == "C004-G01"
    ]
    .sort_values(
        "bge_max_before",
        ascending=False,
    )
    [
        [
            "candidate_id",
            "bge_before",
            "bge_after",
            "bge_combined",
            "bge_max_before",
            "bge_max_after",
            "bge_mean_before",
            "bge_mean_after",
            "bge_top2_before",
            "bge_top2_after",
            "bge_weighted_before",
            "bge_weighted_after",
        ]
    ]
    .to_string(index=False)
)