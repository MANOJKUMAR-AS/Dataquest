from __future__ import annotations

import pandas as pd


def build_gap_contexts(
    messages: pd.DataFrame,
    candidates: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Build correct before/after contexts for every gap.

    IMPORTANT:
    gap_position refers to the ORIGINAL episode position before
    messages were removed.

    Therefore, for a gap at original position P:

        observed_index = P - 1 - number_of_prior_gaps

    where prior gaps are gaps in the same episode with a smaller
    original gap_position.
    """

    messages = messages.copy()
    candidates = candidates.copy()

    messages["timestamp"] = pd.to_datetime(
        messages["timestamp"],
        errors="coerce"
    )

    # One row per gap
    gaps = (
        candidates[
            [
                "episode_id",
                "gap_id",
                "gap_position",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    results = []

    for _, gap in gaps.iterrows():

        episode_id = gap["episode_id"]
        gap_id = gap["gap_id"]
        gap_position = int(gap["gap_position"])

        # ---------------------------------------------------------
        # Get observed messages for this episode
        # ---------------------------------------------------------
        episode_messages = messages[
            messages["episode_id"] == episode_id
        ].copy()

        episode_messages = episode_messages.sort_values(
            ["timestamp", "message_id"]
        ).reset_index(drop=True)

        # ---------------------------------------------------------
        # Find how many earlier messages were removed
        # ---------------------------------------------------------
        prior_gaps = gaps[
            (gaps["episode_id"] == episode_id)
            & (gaps["gap_position"] < gap_position)
        ]

        number_of_prior_gaps = len(prior_gaps)

        # ---------------------------------------------------------
        # Convert original position -> observed position
        # ---------------------------------------------------------
        observed_index = (
            gap_position
            - 1
            - number_of_prior_gaps
        )

        # Safety
        observed_index = max(
            0,
            min(observed_index, len(episode_messages))
        )

        # ---------------------------------------------------------
        # Context BEFORE gap
        # ---------------------------------------------------------
        before_start = max(
            0,
            observed_index - window
        )

        before_messages = episode_messages.iloc[
            before_start:observed_index
        ].copy()

        # ---------------------------------------------------------
        # Context AFTER gap
        # ---------------------------------------------------------
        after_messages = episode_messages.iloc[
            observed_index:observed_index + window
        ].copy()

        # ---------------------------------------------------------
        # Convert to text
        # ---------------------------------------------------------
        before_text = "\n".join(
            before_messages["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        after_text = "\n".join(
            after_messages["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        combined_text = (
            before_text
            + "\n"
            + after_text
        ).strip()

        results.append(
            {
                "episode_id": episode_id,
                "gap_id": gap_id,
                "gap_position": gap_position,
                "observed_index": observed_index,
                "prior_gaps": number_of_prior_gaps,
                "before_text": before_text,
                "after_text": after_text,
                "combined_text": combined_text,
            }
        )

    return pd.DataFrame(results)