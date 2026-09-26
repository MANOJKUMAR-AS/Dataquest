from pathlib import Path

from src.config import (
    HIDDEN_FILE,
    SUBMISSION_FILE,
    OUTPUT_DIR,
)

from src.pipeline import (
    run_prediction,
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = run_prediction(
        input_file=HIDDEN_FILE,
        output_file=SUBMISSION_FILE,
    )

    print()
    print("=" * 60)
    print("Task A prediction complete")
    print("=" * 60)

    print(
        f"Rows generated: {len(result)}"
    )

    print(
        f"Output: {SUBMISSION_FILE}"
    )

    print()
    print(result.head(10))


if __name__ == "__main__":
    main()