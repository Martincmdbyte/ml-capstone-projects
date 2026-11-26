import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import prepare_features, suburb_baseline


def main():
    parser = argparse.ArgumentParser(description="Estimate prices for homes listed in a CSV.")
    parser.add_argument("homes", type=Path)
    parser.add_argument("--model", type=Path, default=Path("outputs/price_model.joblib"))
    parser.add_argument("--save", type=Path, default=Path("outputs/estimates.csv"))
    args = parser.parse_args()

    homes = pd.read_csv(args.homes)
    if not len(homes) or not {"Rooms", "Type", "Distance"}.issubset(homes.columns):
        parser.error("Use a nonempty CSV with at least Rooms, Type, and Distance headers.")

    saved = joblib.load(args.model)
    homes["suburb_median_aud"] = suburb_baseline(
        homes, saved["suburb_medians"], saved["overall_median"]
    ).round()
    homes["estimated_price_aud"] = np.expm1(saved["model"].predict(prepare_features(homes))).round()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    homes.to_csv(args.save, index=False)
    print(f"Saved {len(homes)} estimates to {args.save}")


if __name__ == "__main__":
    main()
