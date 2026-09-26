from src.data_loader import load_messages, load_candidates
from src.bm25_features import calculate_bm25_features


messages = load_messages()
candidates = load_candidates()

features = calculate_bm25_features(
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
        "bm25_max_before",
        ascending=False,
    )
    [
        [
            "candidate_id",
            "bm25_before",
            "bm25_after",
            "bm25_combined",
            "bm25_max_before",
            "bm25_max_after",
            "bm25_mean_before",
            "bm25_mean_after",
        ]
    ]
    .to_string(index=False)
)