from pathlib import Path
import sys

# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allows imports such as:
# from src.bm25_features import ...
# when running:
# python src/pipeline.py
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MESSAGES_FILE = DATA_DIR / "taskB_messages_public.csv"
CANDIDATES_FILE = DATA_DIR / "taskB_candidates_public.csv"

# Intermediate files
BM25_FILE = OUTPUT_DIR / "taskB_bm25_features.csv"
BGE_FILE = OUTPUT_DIR / "taskB_bge_features.csv"
ALL_FEATURES_FILE = OUTPUT_DIR / "taskB_all_features.csv"
NORMALIZED_FILE = OUTPUT_DIR / "taskB_normalized_features.csv"
RANKED_FILE = OUTPUT_DIR / "taskB_hybrid_ranked.csv"

# FINAL SUBMISSION
FINAL_FILE = OUTPUT_DIR / "submission_taskB.csv"


# ============================================================
# REQUIRED COLUMNS
# ============================================================

MESSAGE_COLUMNS = [
    "episode_id",
    "message_id",
    "sender",
    "timestamp",
    "text",
]

CANDIDATE_COLUMNS = [
    "episode_id",
    "gap_id",
    "gap_position",
    "candidate_id",
    "candidate_text",
    "candidate_sender",
    "candidate_timestamp",
    "feature_sender_pattern_fit",
    "feature_timestamp_gap_fit",
    "feature_thread_position_fit",
    "feature_embedding_similarity",
]

KEY_COLUMNS = [
    "episode_id",
    "gap_id",
    "candidate_id",
]


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("STEP 1 - LOADING DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not MESSAGES_FILE.exists():
        raise FileNotFoundError(
            f"Messages file not found:\n{MESSAGES_FILE}"
        )

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Candidates file not found:\n{CANDIDATES_FILE}"
        )

    # --------------------------------------------------------
    # Read CSVs
    # --------------------------------------------------------

    messages = pd.read_csv(MESSAGES_FILE)
    candidates = pd.read_csv(CANDIDATES_FILE)

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing_messages = [
        column
        for column in MESSAGE_COLUMNS
        if column not in messages.columns
    ]

    missing_candidates = [
        column
        for column in CANDIDATE_COLUMNS
        if column not in candidates.columns
    ]

    if missing_messages:
        raise ValueError(
            "Messages file is missing columns:\n"
            f"{missing_messages}"
        )

    if missing_candidates:
        raise ValueError(
            "Candidates file is missing columns:\n"
            f"{missing_candidates}"
        )

    # --------------------------------------------------------
    # Validate row counts
    # --------------------------------------------------------

    if len(messages) == 0:
        raise ValueError("Messages file contains zero rows.")

    if len(candidates) == 0:
        raise ValueError("Candidates file contains zero rows.")

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    messages["timestamp"] = pd.to_datetime(
        messages["timestamp"],
        errors="raise"
    )

    candidates["candidate_timestamp"] = pd.to_datetime(
        candidates["candidate_timestamp"],
        errors="raise"
    )

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    messages["text"] = (
        messages["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    candidates["candidate_text"] = (
        candidates["candidate_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Sort messages
    # --------------------------------------------------------

    messages = (
        messages
        .sort_values(
            [
                "episode_id",
                "timestamp",
                "message_id",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Sort candidates
    # --------------------------------------------------------

    candidates = (
        candidates
        .sort_values(
            [
                "episode_id",
                "gap_id",
                "candidate_id",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Check candidate uniqueness
    # --------------------------------------------------------

    if candidates.duplicated(KEY_COLUMNS).any():

        duplicates = (
            candidates[
                candidates.duplicated(
                    KEY_COLUMNS,
                    keep=False
                )
            ]
        )

        print(duplicates)

        raise ValueError(
            "Duplicate candidate keys detected."
        )

    # --------------------------------------------------------
    # Print dataset information
    # --------------------------------------------------------

    print(f"Messages   : {len(messages)}")
    print(f"Candidates : {len(candidates)}")
    print(f"Gaps       : {candidates['gap_id'].nunique()}")

    return messages, candidates


# ============================================================
# STEP 2 - BM25 FEATURES
# ============================================================

def run_bm25(messages, candidates):

    print("\n" + "=" * 70)
    print("STEP 2 - BUILDING BM25 FEATURES")
    print("=" * 70)

    from src.bm25_features import calculate_bm25_features

    # --------------------------------------------------------
    # Calculate BM25
    # --------------------------------------------------------

    bm25 = calculate_bm25_features(
        messages,
        candidates,
        window=3
    )

    # --------------------------------------------------------
    # Required output columns
    # --------------------------------------------------------

    required_columns = [
        "episode_id",
        "gap_id",
        "candidate_id",
        "bm25_before",
        "bm25_after",
        "bm25_combined",
    ]

    missing = [
        column
        for column in required_columns
        if column not in bm25.columns
    ]

    if missing:
        raise ValueError(
            f"BM25 output missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Row count validation
    # --------------------------------------------------------

    if len(bm25) != len(candidates):

        raise ValueError(
            "BM25 row count does not match "
            "candidate row count."
        )

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    if bm25.duplicated(KEY_COLUMNS).any():

        raise ValueError(
            "Duplicate keys found in BM25 features."
        )

    # --------------------------------------------------------
    # NaN validation
    # --------------------------------------------------------

    bm25_columns = [
        "bm25_before",
        "bm25_after",
        "bm25_combined",
    ]

    if bm25[bm25_columns].isna().any().any():

        raise ValueError(
            "NaN values found in BM25 features."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    bm25.to_csv(
        BM25_FILE,
        index=False
    )

    print(f"BM25 rows : {len(bm25)}")
    print(f"Saved     : {BM25_FILE}")

    return bm25


# ============================================================
# STEP 3 - BGE FEATURES
# ============================================================

def run_bge(messages, candidates):

    print("\n" + "=" * 70)
    print("STEP 3 - BUILDING BGE FEATURES")
    print("=" * 70)

    from src.embedding_features import calculate_embedding_features

    # --------------------------------------------------------
    # Calculate BGE
    # --------------------------------------------------------

    bge = calculate_embedding_features(
        messages,
        candidates,
        window=3
    )

    # --------------------------------------------------------
    # Required output columns
    # --------------------------------------------------------

    required_columns = [
        "episode_id",
        "gap_id",
        "candidate_id",
        "bge_before",
        "bge_after",
        "bge_combined",
    ]

    missing = [
        column
        for column in required_columns
        if column not in bge.columns
    ]

    if missing:
        raise ValueError(
            f"BGE output missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Row count validation
    # --------------------------------------------------------

    if len(bge) != len(candidates):

        raise ValueError(
            "BGE row count does not match "
            "candidate row count."
        )

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    if bge.duplicated(KEY_COLUMNS).any():

        raise ValueError(
            "Duplicate keys found in BGE features."
        )

    # --------------------------------------------------------
    # NaN validation
    # --------------------------------------------------------

    bge_columns = [
        "bge_before",
        "bge_after",
        "bge_combined",
    ]

    if bge[bge_columns].isna().any().any():

        raise ValueError(
            "NaN values found in BGE features."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    bge.to_csv(
        BGE_FILE,
        index=False
    )

    print(f"BGE rows  : {len(bge)}")
    print(f"Saved     : {BGE_FILE}")

    return bge


# ============================================================
# STEP 4 - MERGE FEATURES
# ============================================================

def merge_features(candidates, bm25, bge):

    print("\n" + "=" * 70)
    print("STEP 4 - MERGING ALL FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate duplicates
    # --------------------------------------------------------

    for name, dataframe in [
        ("Candidates", candidates),
        ("BM25", bm25),
        ("BGE", bge),
    ]:

        if dataframe.duplicated(
            KEY_COLUMNS
        ).any():

            raise ValueError(
                f"Duplicate keys found in {name}."
            )

    # --------------------------------------------------------
    # Merge BM25
    # --------------------------------------------------------

    features = candidates.merge(
        bm25,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Merge BGE
    # --------------------------------------------------------

    features = features.merge(
        bge,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Generated feature columns
    # --------------------------------------------------------

    generated_features = [
        "bm25_before",
        "bm25_after",
        "bm25_combined",
        "bge_before",
        "bge_after",
        "bge_combined",
    ]

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing_values = (
        features[generated_features]
        .isna()
        .sum()
    )

    if missing_values.sum() > 0:

        print("\nMissing feature values:")
        print(
            missing_values[
                missing_values > 0
            ]
        )

        raise ValueError(
            "Some candidates do not have "
            "all generated features."
        )

    # --------------------------------------------------------
    # Check row count
    # --------------------------------------------------------

    if len(features) != len(candidates):

        raise ValueError(
            "Feature merging changed "
            "the candidate row count."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    features.to_csv(
        ALL_FEATURES_FILE,
        index=False
    )

    print(f"Feature rows : {len(features)}")
    print(f"Columns      : {len(features.columns)}")
    print(f"Saved        : {ALL_FEATURES_FILE}")

    return features


# ============================================================
# STEP 5 - WITHIN-GAP NORMALIZATION
# ============================================================

def normalize_features(features):

    print("\n" + "=" * 70)
    print("STEP 5 - NORMALIZING FEATURES WITHIN EACH GAP")
    print("=" * 70)

    feature_columns = [

        # Supplied Task B features
        "feature_sender_pattern_fit",
        "feature_timestamp_gap_fit",
        "feature_thread_position_fit",
        "feature_embedding_similarity",

        # BM25
        "bm25_before",
        "bm25_after",
        "bm25_combined",

        # BGE
        "bge_before",
        "bge_after",
        "bge_combined",
    ]

    result = features.copy()

    # --------------------------------------------------------
    # Percentile rank within each gap
    # --------------------------------------------------------

    for column in feature_columns:

        normalized_column = (
            f"{column}_norm"
        )

        result[normalized_column] = (
            result
            .groupby("gap_id")[column]
            .rank(
                method="average",
                pct=True,
                ascending=True
            )
        )

    # --------------------------------------------------------
    # Validate normalized values
    # --------------------------------------------------------

    normalized_columns = [
        f"{column}_norm"
        for column in feature_columns
    ]

    for column in normalized_columns:

        if result[column].isna().any():

            raise ValueError(
                f"NaN values found in {column}"
            )

        minimum = result[column].min()
        maximum = result[column].max()

        if minimum < 0 or maximum > 1:

            raise ValueError(
                f"{column} outside [0,1]. "
                f"Min={minimum}, Max={maximum}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        NORMALIZED_FILE,
        index=False
    )

    print(f"Rows       : {len(result)}")
    print(f"Gaps       : {result['gap_id'].nunique()}")
    print(f"Columns    : {len(result.columns)}")
    print(f"Saved      : {NORMALIZED_FILE}")

    return result


# ============================================================
# STEP 6 - HYBRID RANKING
# ============================================================

def rank_candidates(df):

    print("\n" + "=" * 70)
    print("STEP 6 - HYBRID RANKING")
    print("=" * 70)

    # --------------------------------------------------------
    # Structural features
    # --------------------------------------------------------

    structural_columns = [

        "feature_sender_pattern_fit_norm",
        "feature_timestamp_gap_fit_norm",
        "feature_thread_position_fit_norm",
        "feature_embedding_similarity_norm",
    ]

    # --------------------------------------------------------
    # BM25 features
    # --------------------------------------------------------

    bm25_columns = [

        "bm25_before_norm",
        "bm25_after_norm",
        "bm25_combined_norm",
    ]

    # --------------------------------------------------------
    # BGE features
    # --------------------------------------------------------

    bge_columns = [

        "bge_before_norm",
        "bge_after_norm",
        "bge_combined_norm",
    ]

    result = df.copy()

    # --------------------------------------------------------
    # Family scores
    # --------------------------------------------------------

    result["structural_score"] = (
        result[structural_columns]
        .mean(axis=1)
    )

    result["bm25_score"] = (
        result[bm25_columns]
        .mean(axis=1)
    )

    result["bge_score"] = (
        result[bge_columns]
        .mean(axis=1)
    )

    # --------------------------------------------------------
    # FINAL HYBRID SCORE
    #
    # Structural = 40%
    # BM25       = 20%
    # BGE        = 40%
    # --------------------------------------------------------

    result["hybrid_score"] = (
        0.40 * result["structural_score"]
        + 0.20 * result["bm25_score"]
        + 0.40 * result["bge_score"]
    )

    # --------------------------------------------------------
    # Rank within each gap
    # --------------------------------------------------------

    result["rank"] = (
        result
        .groupby("gap_id")["hybrid_score"]
        .rank(
            method="first",
            ascending=False
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Sort final ranking
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            [
                "gap_id",
                "rank"
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save ranked data
    # --------------------------------------------------------

    result.to_csv(
        RANKED_FILE,
        index=False
    )

    print(f"Ranked rows : {len(result)}")
    print(f"Gaps        : {result['gap_id'].nunique()}")
    print(f"Saved       : {RANKED_FILE}")

    return result


# ============================================================
# STEP 7 - FINAL SUBMISSION VALIDATION
# ============================================================

def validate_submission(ranked):

    print("\n" + "=" * 70)
    print("STEP 7 - VALIDATING FINAL SUBMISSION")
    print("=" * 70)

    # --------------------------------------------------------
    # Select only required columns
    # --------------------------------------------------------

    submission = ranked[
        [
            "gap_id",
            "candidate_id",
            "rank",
        ]
    ].copy()

    expected_columns = [
        "gap_id",
        "candidate_id",
        "rank",
    ]

    # --------------------------------------------------------
    # Column validation
    # --------------------------------------------------------

    if list(submission.columns) != expected_columns:

        raise ValueError(
            "Final submission columns are incorrect."
        )

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    if len(submission) != len(ranked):

        raise ValueError(
            "Submission row count changed."
        )

    # --------------------------------------------------------
    # Expected public dataset size
    # --------------------------------------------------------

    if len(submission) != 434:

        raise ValueError(
            f"Expected 434 rows, "
            f"but found {len(submission)}."
        )

    # --------------------------------------------------------
    # Gap count
    # --------------------------------------------------------

    gap_count = (
        submission["gap_id"]
        .nunique()
    )

    if gap_count != 88:

        raise ValueError(
            f"Expected 88 gaps, "
            f"but found {gap_count}."
        )

    # --------------------------------------------------------
    # Duplicate row validation
    # --------------------------------------------------------

    duplicate_rows = (
        submission
        .duplicated()
        .sum()
    )

    if duplicate_rows != 0:

        raise ValueError(
            f"Found {duplicate_rows} "
            "duplicate submission rows."
        )

    # --------------------------------------------------------
    # Candidate uniqueness
    # --------------------------------------------------------

    duplicate_candidates = (
        submission["candidate_id"]
        .duplicated()
        .sum()
    )

    if duplicate_candidates != 0:

        raise ValueError(
            f"Found {duplicate_candidates} "
            "duplicate candidate IDs."
        )

    # --------------------------------------------------------
    # Rank validation
    # --------------------------------------------------------

    for gap_id, group in submission.groupby(
        "gap_id"
    ):

        ranks = sorted(
            group["rank"].tolist()
        )

        expected_ranks = list(
            range(
                1,
                len(group) + 1
            )
        )

        if ranks != expected_ranks:

            raise ValueError(
                f"Invalid ranking for "
                f"{gap_id}.\n"
                f"Found: {ranks}\n"
                f"Expected: {expected_ranks}"
            )

    # --------------------------------------------------------
    # Sort final submission
    # --------------------------------------------------------

    submission = (
        submission
        .sort_values(
            [
                "gap_id",
                "rank"
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save final CSV
    # --------------------------------------------------------

    submission.to_csv(
        FINAL_FILE,
        index=False
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL VALIDATION PASSED")
    print("=" * 70)

    print(
        f"Rows              : "
        f"{len(submission)}"
    )

    print(
        f"Gaps              : "
        f"{submission['gap_id'].nunique()}"
    )

    print(
        f"Unique candidates : "
        f"{submission['candidate_id'].nunique()}"
    )

    print(
        f"Duplicate rows    : "
        f"{submission.duplicated().sum()}"
    )

    print("Ranks             : VALID")

    print("\nFINAL SUBMISSION FILE:")
    print(FINAL_FILE)

    print("\nFirst 20 rows:")
    print(
        submission
        .head(20)
        .to_string(index=False)
    )

    return submission


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("          CONTEXTLAG - TASK B FINAL PIPELINE")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    messages, candidates = load_data()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    bm25 = run_bm25(
        messages,
        candidates
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    bge = run_bge(
        messages,
        candidates
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    features = merge_features(
        candidates,
        bm25,
        bge
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    normalized = normalize_features(
        features
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    ranked = rank_candidates(
        normalized
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    validate_submission(
        ranked
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("       TASK B PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print("Your final submission is:")
    print()
    print(FINAL_FILE)
    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()