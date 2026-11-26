import argparse
from pathlib import Path

import joblib
import pandas as pd

from study import prepare_features


def main():
    parser = argparse.ArgumentParser(description="Score CSV rows with the saved research model.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=Path("outputs/readmission_model.joblib"))
    parser.add_argument("--save", type=Path, default=Path("outputs/predictions.csv"))
    args = parser.parse_args()
    features = prepare_features(pd.read_csv(args.input, na_values="?", keep_default_na=False))
    scores = joblib.load(args.model).predict_proba(features)[:, 1]
    args.save.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"estimated_30_day_probability": scores}).to_csv(args.save, index=False)
    print(f"Saved {len(scores):,} research predictions to {args.save}")


if __name__ == "__main__":
    main()
