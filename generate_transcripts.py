"""
Generate judge-friendly grouped conversation transcripts for Task A.

The official submission file is NOT modified.

INPUT:
    data/taskA_hidden_public.csv
    outputs/submission_taskA.csv

OUTPUT:
    outputs/taskA_transcripts_grouped.txt
    outputs/taskA_transcripts_grouped.csv
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

HIDDEN_FILE = (
    DATA_DIR / "taskA_hidden_public.csv"
)

SUBMISSION_FILE = (
    OUTPUT_DIR / "submission_taskA.csv"
)

# Use NEW filenames so an older locked CSV does not
# prevent the transcript generator from running.
TRANSCRIPT_FILE = (
    OUTPUT_DIR / "taskA_transcripts_grouped.txt"
)

TRANSCRIPT_CSV = (
    OUTPUT_DIR / "taskA_transcripts_grouped.csv"
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

MAX_TEXT_LENGTH = 500


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean(value):
    """
    Convert a value into a clean string.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


def format_time(value):
    """
    Format timestamp as HH:MM.
    """

    try:

        timestamp = pd.to_datetime(
            value
        )

        return timestamp.strftime(
            "%H:%M"
        )

    except Exception:

        return clean(value)


def shorten(
    text,
    max_length=MAX_TEXT_LENGTH,
):
    """
    Keep very long messages readable.
    """

    text = clean(text)

    if len(text) <= max_length:

        return text

    return (
        text[:max_length - 3]
        + "..."
    )


def get_sender(row):
    """
    Automatically detect the sender column.

    Supports common sender column names.
    """

    possible_columns = [

        "sender",

        "sender_id",

        "user",

        "user_id",

        "author",

        "author_id",

        "username",

        "speaker",
    ]

    for column in possible_columns:

        if column in row.index:

            value = clean(
                row[column]
            )

            if value:

                return value

    return "Unknown"


def find_column(
    df,
    names,
):
    """
    Find the first matching column.
    """

    for name in names:

        if name in df.columns:

            return name

    raise ValueError(
        "\nCould not find a required column.\n"
        f"Tried: {names}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load original messages and model predictions.
    """

    if not HIDDEN_FILE.exists():

        raise FileNotFoundError(
            "\nMissing input file:\n"
            f"{HIDDEN_FILE}"
        )

    if not SUBMISSION_FILE.exists():

        raise FileNotFoundError(
            "\nMissing prediction file:\n"
            f"{SUBMISSION_FILE}\n\n"
            "Run the Task A prediction pipeline first."
        )

    messages = pd.read_csv(
        HIDDEN_FILE
    )

    predictions = pd.read_csv(
        SUBMISSION_FILE
    )

    print(
        f"Loaded message data: "
        f"{len(messages)} rows"
    )

    print(
        f"Loaded prediction data: "
        f"{len(predictions)} rows"
    )

    return (
        messages,
        predictions,
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    messages,
    predictions,
):
    """
    Merge original message information with
    predicted reply/thread information.
    """

    # --------------------------------------------------------
    # ORIGINAL MESSAGE COLUMNS
    # --------------------------------------------------------

    episode_col = find_column(
        messages,
        [
            "episode_id",
            "episode",
        ],
    )

    message_col = find_column(
        messages,
        [
            "message_id",
            "id",
        ],
    )

    timestamp_col = find_column(
        messages,
        [
            "timestamp",
            "time",
            "datetime",
            "created_at",
        ],
    )

    text_col = find_column(
        messages,
        [
            "text",
            "message",
            "content",
            "body",
        ],
    )

    # --------------------------------------------------------
    # NORMALIZE ORIGINAL MESSAGE COLUMNS
    # --------------------------------------------------------

    messages = messages.rename(
        columns={
            episode_col:
                "episode_id",

            message_col:
                "message_id",

            timestamp_col:
                "timestamp",

            text_col:
                "text",
        }
    )

    # --------------------------------------------------------
    # PREDICTION COLUMNS
    # --------------------------------------------------------

    prediction_episode_col = find_column(
        predictions,
        [
            "episode_id",
            "episode",
        ],
    )

    prediction_message_col = find_column(
        predictions,
        [
            "message_id",
            "id",
        ],
    )

    reply_col = find_column(
        predictions,
        [
            "reply_to_message_id",
            "reply_to",
            "parent_message_id",
        ],
    )

    thread_col = find_column(
        predictions,
        [
            "thread_id",
            "thread",
        ],
    )

    # --------------------------------------------------------
    # NORMALIZE PREDICTION COLUMNS
    # --------------------------------------------------------

    predictions = predictions.rename(
        columns={
            prediction_episode_col:
                "episode_id",

            prediction_message_col:
                "message_id",

            reply_col:
                "reply_to_message_id",

            thread_col:
                "thread_id",
        }
    )

    predictions = predictions[
        [
            "episode_id",
            "message_id",
            "reply_to_message_id",
            "thread_id",
        ]
    ].copy()

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    merged = messages.merge(
        predictions,
        on=[
            "episode_id",
            "message_id",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # PARSE TIMESTAMP
    # --------------------------------------------------------

    merged["_timestamp"] = pd.to_datetime(
        merged["timestamp"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL ORDER
    # --------------------------------------------------------

    merged = merged.sort_values(
        [
            "episode_id",
            "_timestamp",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    return merged


# ============================================================
# THREAD ORDER
# ============================================================

def get_thread_order(
    episode_df,
):
    """
    Return threads in the order they first appear
    in the original episode.

    Example:

        A002-T1
        A002-T2
        A002-T3
        A002-T4
    """

    order = []

    for thread_id in episode_df[
        "thread_id"
    ]:

        thread_id = clean(
            thread_id
        )

        if thread_id not in order:

            order.append(
                thread_id
            )

    return order


# ============================================================
# BUILD GROUPED CSV
# ============================================================

def build_transcript_csv(
    data,
):
    """
    Create a structured CSV where messages belonging
    to the same predicted thread are together.
    """

    rows = []

    for (
        episode_id,
        episode_df,
    ) in data.groupby(
        "episode_id",
        sort=False,
    ):

        episode_df = (
            episode_df
            .sort_values(
                "_timestamp",
                kind="stable",
            )
        )

        thread_order = get_thread_order(
            episode_df
        )

        for thread_id in thread_order:

            thread_df = episode_df[
                episode_df[
                    "thread_id"
                ].map(clean)
                == thread_id
            ].copy()

            thread_df = (
                thread_df
                .sort_values(
                    "_timestamp",
                    kind="stable",
                )
            )

            for position, (
                _,
                row,
            ) in enumerate(
                thread_df.iterrows(),
                start=1,
            ):

                rows.append(
                    {
                        "episode_id":
                            clean(
                                episode_id
                            ),

                        "thread_id":
                            thread_id,

                        "thread_position":
                            position,

                        "message_id":
                            clean(
                                row[
                                    "message_id"
                                ]
                            ),

                        "timestamp":
                            clean(
                                row[
                                    "timestamp"
                                ]
                            ),

                        "sender":
                            get_sender(
                                row
                            ),

                        "text":
                            clean(
                                row[
                                    "text"
                                ]
                            ),

                        "reply_to_message_id":
                            clean(
                                row[
                                    "reply_to_message_id"
                                ]
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BUILD JUDGE-FRIENDLY TXT
# ============================================================

def build_transcript_text(
    data,
):
    """
    Create the human-readable grouped transcript.

    Example:

        EPISODE A002

        THREAD A002-T1

        [09:05] Tobias
        ...

        THREAD A002-T2

        [09:31] Marcus
        ...

            ↓ reply

        [10:52] Priya
        ...
    """

    lines = []

    # --------------------------------------------------------
    # DOCUMENT HEADER
    # --------------------------------------------------------

    lines.append(
        "=" * 60
    )

    lines.append(
        "CONTEXTLAG - TASK A"
    )

    lines.append(
        "RECONSTRUCTED CONVERSATION TRANSCRIPTS"
    )

    lines.append(
        "=" * 60
    )

    lines.append("")

    lines.append(
        "Messages are grouped by reconstructed "
        "thread so that each conversation context "
        "can be reviewed independently."
    )

    lines.append("")

    # --------------------------------------------------------
    # EPISODES
    # --------------------------------------------------------

    episodes = list(
        data.groupby(
            "episode_id",
            sort=False,
        )
    )

    total_episodes = len(
        episodes
    )

    for episode_number, (
        episode_id,
        episode_df,
    ) in enumerate(
        episodes,
        start=1,
    ):

        episode_df = (
            episode_df
            .sort_values(
                "_timestamp",
                kind="stable",
            )
        )

        thread_order = get_thread_order(
            episode_df
        )

        # ----------------------------------------------------
        # EPISODE HEADER
        # ----------------------------------------------------

        lines.append(
            "=" * 60
        )

        lines.append(
            f"EPISODE {clean(episode_id)}"
        )

        lines.append(
            "=" * 60
        )

        lines.append("")

        # ----------------------------------------------------
        # THREADS
        # ----------------------------------------------------

        for thread_id in thread_order:

            thread_df = episode_df[
                episode_df[
                    "thread_id"
                ].map(clean)
                == thread_id
            ].copy()

            thread_df = (
                thread_df
                .sort_values(
                    "_timestamp",
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

            # ------------------------------------------------
            # THREAD HEADER
            # ------------------------------------------------

            lines.append(
                f"🧵 THREAD {thread_id}"
            )

            lines.append(
                "-" * 60
            )

            lines.append("")

            # ------------------------------------------------
            # THREAD MESSAGES
            # ------------------------------------------------

            for index in range(
                len(thread_df)
            ):

                row = thread_df.iloc[
                    index
                ]

                timestamp = format_time(
                    row[
                        "timestamp"
                    ]
                )

                sender = get_sender(
                    row
                )

                text = shorten(
                    row[
                        "text"
                    ]
                )

                current_message_id = clean(
                    row[
                        "message_id"
                    ]
                )

                # --------------------------------------------
                # MESSAGE
                # --------------------------------------------

                lines.append(
                    f"[{timestamp}] {sender}"
                )

                lines.append(
                    text
                )

                # --------------------------------------------
                # DETERMINE WHETHER NEXT MESSAGE
                # IS A DIRECT REPLY TO THIS MESSAGE
                # --------------------------------------------

                if (
                    index
                    < len(thread_df) - 1
                ):

                    next_row = (
                        thread_df.iloc[
                            index + 1
                        ]
                    )

                    next_reply = clean(
                        next_row[
                            "reply_to_message_id"
                        ]
                    )

                    if (
                        next_reply
                        == current_message_id
                    ):

                        lines.append("")

                        lines.append(
                            "    ↓ reply"
                        )

                        lines.append("")

                    else:

                        lines.append("")

                else:

                    lines.append("")

            lines.append("")

        # ----------------------------------------------------
        # EPISODE SEPARATOR
        # ----------------------------------------------------

        if (
            episode_number
            < total_episodes
        ):

            lines.append("")

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    lines.append(
        "=" * 60
    )

    lines.append(
        "END OF RECONSTRUCTED TRANSCRIPTS"
    )

    lines.append(
        "=" * 60
    )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 60
    )

    print(
        "TASK A - GROUPED TRANSCRIPT GENERATOR"
    )

    print(
        "=" * 60
    )

    print()

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    messages, predictions = (
        load_data()
    )

    # --------------------------------------------------------
    # MERGE DATA
    # --------------------------------------------------------

    print()

    print(
        "Merging predictions with original messages..."
    )

    data = prepare_data(
        messages,
        predictions,
    )

    print(
        f"Merged rows: {len(data)}"
    )

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if len(data) != len(
        predictions
    ):

        print()

        print(
            "WARNING:"
        )

        print(
            "Merged row count differs "
            "from prediction row count."
        )

    missing_threads = data[
        "thread_id"
    ].isna().sum()

    missing_replies = data[
        "reply_to_message_id"
    ].isna().sum()

    if missing_threads > 0:

        print()

        print(
            f"WARNING: {missing_threads} "
            "messages have no thread_id."
        )

    if missing_replies > 0:

        print()

        print(
            f"WARNING: {missing_replies} "
            "messages have no reply prediction."
        )

    # --------------------------------------------------------
    # BUILD GROUPED CSV
    # --------------------------------------------------------

    print()

    print(
        "Creating grouped transcript CSV..."
    )

    transcript_df = (
        build_transcript_csv(
            data
        )
    )

    try:

        transcript_df.to_csv(
            TRANSCRIPT_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Created: {TRANSCRIPT_CSV}"
        )

    except PermissionError:

        print()

        print(
            "WARNING: Could not write the grouped CSV."
        )

        print(
            "The file may currently be open in "
            "Excel or another program."
        )

        print(
            "Continuing with TXT generation..."
        )

    # --------------------------------------------------------
    # BUILD TEXT TRANSCRIPT
    # --------------------------------------------------------

    print()

    print(
        "Creating judge-friendly grouped transcript..."
    )

    transcript_text = (
        build_transcript_text(
            data
        )
    )

    TRANSCRIPT_FILE.write_text(
        transcript_text,
        encoding="utf-8",
    )

    print(
        f"Created: {TRANSCRIPT_FILE}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    episode_count = (
        data[
            "episode_id"
        ]
        .nunique()
    )

    thread_count = (
        data[
            [
                "episode_id",
                "thread_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    message_count = len(
        data
    )

    print()

    print(
        "=" * 60
    )

    print(
        "DONE"
    )

    print(
        "=" * 60
    )

    print(
        f"Episodes : {episode_count}"
    )

    print(
        f"Messages : {message_count}"
    )

    print(
        f"Threads  : {thread_count}"
    )

    print()

    print(
        "Judge-friendly transcript:"
    )

    print(
        f"  {TRANSCRIPT_FILE}"
    )

    print()

    print(
        "Grouped CSV:"
    )

    print(
        f"  {TRANSCRIPT_CSV}"
    )

    print()

    print(
        "Official submission:"
    )

    print(
        f"  {SUBMISSION_FILE}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()