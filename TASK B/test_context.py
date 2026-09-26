import pandas as pd

from src.data_loader import load_messages, load_candidates
from src.context_builder import build_gap_contexts


messages = load_messages()
candidates = load_candidates()

contexts = build_gap_contexts(
    messages,
    candidates,
    window=5,
)

print("\nContext rows:", len(contexts))

print("\nColumns:")
print(contexts.columns.tolist())

print("\nFirst 10 contexts:")
print(
    contexts[
        [
            "episode_id",
            "gap_id",
            "gap_position",
            "observed_index",
            "prior_gaps",
        ]
    ].head(10).to_string(index=False)
)

print("\nExample contexts:")
for _, row in contexts.head(5).iterrows():

    print("\n" + "=" * 80)
    print(
        row["episode_id"],
        row["gap_id"],
        "original position:",
        row["gap_position"],
        "observed index:",
        row["observed_index"],
        "prior gaps:",
        row["prior_gaps"],
    )

    print("\n--- BEFORE ---")
    print(row["before_text"])

    print("\n--- AFTER ---")
    print(row["after_text"])