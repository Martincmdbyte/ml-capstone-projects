import argparse
from pathlib import Path

import joblib

from signals import lead_features, read_record


def main():
    parser = argparse.ArgumentParser(description="Score one PTB-XL 100 Hz research recording.")
    parser.add_argument("record", type=Path, help="Path to a record's .hea or .dat file")
    parser.add_argument("--model", type=Path, default=Path("outputs/lead_ii.joblib"))
    args = parser.parse_args()
    saved = joblib.load(args.model)
    signal = read_record(args.record)[saved["leads"]]
    scores = saved["model"].predict_proba(lead_features(signal).reshape(1, -1))[0]
    for label, score in zip(saved["labels"], scores):
        print(f"{label}: {score:.3f}")


if __name__ == "__main__":
    main()
