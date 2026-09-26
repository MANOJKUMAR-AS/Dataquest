import pandas as pd

from .config import (
    EMBEDDING_MODEL,
)
from .data_loader import (
    load_training_data,
    load_hidden_data,
)
from .preprocessing import (
    prepare_dataframe,
)
from .embeddings import (
    EmbeddingModel,
)
from .reply_scorer import (
    predict_episode,
)
from .thread_builder import (
    assign_thread_ids,
)


def generate_predictions(df, embedding_model):

    df = prepare_dataframe(df)

    all_predictions = []

    for episode_id, episode_df in df.groupby(
        "episode_id",
        sort=False,
    ):

        episode_df = (
            episode_df
            .sort_values("timestamp")
            .reset_index(drop=True)
            .copy()
        )

        episode_df[
            "_episode_index"
        ] = range(len(episode_df))

        texts = episode_df[
            "text"
        ].tolist()

        embeddings = embedding_model.encode(
            texts
        )

        predictions = predict_episode(
            episode_df,
            embeddings,
        )

        episode_result = assign_thread_ids(
            episode_df,
            predictions,
        )

        all_predictions.append(
            episode_result
        )

    return pd.concat(
        all_predictions,
        ignore_index=True,
    )


def run_prediction(input_file, output_file):

    df = load_hidden_data(input_file)

    model = EmbeddingModel(
        EMBEDDING_MODEL
    )

    result = generate_predictions(
        df,
        model,
    )

    result = result[
        [
            "episode_id",
            "message_id",
            "reply_to_message_id",
            "thread_id",
        ]
    ]

    result.to_csv(
        output_file,
        index=False,
    )

    return result