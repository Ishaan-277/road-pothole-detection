
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    ROOT / "outputs" / "pothole_review_batch3"
    / "pothole_labels.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "severity_classifier"
CLEAN_CSV = OUTPUT_DIR / "severity_dataset.csv"
EXCLUDED_CSV = OUTPUT_DIR / "excluded_detections.csv"

VALID_LABELS = ["low", "medium", "high"]

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Label file not found: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df["manual_label"] = df["manual_label"].fillna("").str.strip().str.lower()

    # Keep only valid, manually labelled severity examples.
    clean = df[df["manual_label"].isin(VALID_LABELS)].copy()

    # Ensure the scores match the manually assigned categories.
    score_mapping = {"low": 1, "medium": 2, "high": 3}
    clean["severity_score"] = clean["manual_label"].map(score_mapping).astype(int)

    # Exclude duplicates, uncertain labels and any other invalid entries.
    excluded = df[~df.index.isin(clean.index)].copy()

    # Save new files without changing the original labels CSV.
    clean.to_csv(CLEAN_CSV, index=False)
    excluded.to_csv(EXCLUDED_CSV, index=False)

    print("Severity dataset prepared successfully.")
    print(f"\nOriginal detections: {len(df)}")
    print(f"Usable labelled examples: {len(clean)}")
    print(f"Excluded detections: {len(excluded)}")
    print(f"Unique source images: {clean['image_name'].nunique()}")

    print("\nClass distribution:")
    print(clean["manual_label"].value_counts().reindex(VALID_LABELS, fill_value=0))

    print("\nSaved clean dataset:", CLEAN_CSV)
    print("Saved excluded detections:", EXCLUDED_CSV)

if __name__ == "__main__":
    main()
