import pandas as pd


def reply_accuracy(
    predictions,
    truth,
):
    merged = predictions.merge(
        truth[
            [
                "episode_id",
                "message_id",
                "reply_to_message_id",
            ]
        ],
        on=[
            "episode_id",
            "message_id",
        ],
        suffixes=(
            "_pred",
            "_true",
        ),
    )

    return (
        merged[
            "reply_to_message_id_pred"
        ]
        ==
        merged[
            "reply_to_message_id_true"
        ]
    ).mean()


def thread_pairwise_f1(
    predictions,
    truth,
):
    """
    Evaluate whether pairs of messages are correctly
    identified as belonging to the same discussion.

    This avoids depending on the arbitrary thread ID names.
    """

    merged = predictions.merge(
        truth[
            [
                "episode_id",
                "message_id",
                "thread_id",
            ]
        ],
        on=[
            "episode_id",
            "message_id",
        ],
        suffixes=(
            "_pred",
            "_true",
        ),
    )

    tp = 0
    fp = 0
    fn = 0

    for episode_id, group in merged.groupby(
        "episode_id"
    ):

        rows = group.reset_index(
            drop=True
        )

        for i in range(len(rows)):

            for j in range(
                i + 1,
                len(rows),
            ):

                pred_same = (
                    rows.loc[
                        i,
                        "thread_id_pred",
                    ]
                    ==
                    rows.loc[
                        j,
                        "thread_id_pred",
                    ]
                )

                true_same = (
                    rows.loc[
                        i,
                        "thread_id_true",
                    ]
                    ==
                    rows.loc[
                        j,
                        "thread_id_true",
                    ]
                )

                if pred_same and true_same:
                    tp += 1

                elif pred_same and not true_same:
                    fp += 1

                elif not pred_same and true_same:
                    fn += 1

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }