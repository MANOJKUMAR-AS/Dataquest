from pathlib import Path

from src.pipeline import run_pipeline
from src.config import OUTPUT_DIR


def main():

    ranked, submission = run_pipeline()

    output_file = (
        OUTPUT_DIR / "taskB_baseline_ranked.csv"
    )

    submission.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Baseline output saved to:"
    )
    print(output_file)

    print()
    print("First 20 ranked candidates:")
    print(
        submission.head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()