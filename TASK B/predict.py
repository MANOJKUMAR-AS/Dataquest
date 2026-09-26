from src.pipeline import run_pipeline
from src.config import OUTPUT_DIR


def main():

    _, submission = run_pipeline()

    output_file = (
        OUTPUT_DIR / "taskB_submission.csv"
    )

    submission.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Submission saved to:"
    )
    print(output_file)


if __name__ == "__main__":
    main()