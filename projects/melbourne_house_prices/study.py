import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

from features import prepare_features, suburb_baseline


def load_sales(data_file):
    homes = pd.read_csv(data_file)
    homes["Date"] = pd.to_datetime(homes["Date"], format="%d/%m/%Y", errors="coerce")
    homes["Price"] = pd.to_numeric(homes["Price"], errors="coerce")
    homes = homes.dropna(subset=["Date", "Price"])
    homes = homes[homes["Price"] > 0]
    return homes.drop_duplicates().sort_values("Date").reset_index(drop=True)


def split_by_date(homes):
    if len(homes) < 10:
        raise ValueError("At least 10 usable sales are needed.")
    cutoff = homes.iloc[int(0.8 * len(homes))]["Date"]
    train = homes[homes["Date"] < cutoff]
    test = homes[homes["Date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("The date cutoff needs to leave sales on both sides.")
    return train, test


def measure(name, actual, predicted):
    return (name, mean_absolute_error(actual, predicted),
            median_absolute_error(actual, predicted),
            np.sqrt(mean_squared_error(actual, predicted)))


def plot_months(months, output):
    figure, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(months))
    ax.plot(positions, months["suburb_median_mae"], marker="o", label="Suburb median")
    ax.plot(positions, months["model_mae"], marker="o", label="Gradient boosting")
    ax.set(xticks=positions, xticklabels=months["month"],
           ylabel="Mean absolute error (AUD)", xlabel="Held-out sale month")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.set_title("How price estimates held up on later sales")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output / "later_sale_errors.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Predict later Melbourne sale prices from earlier sales.")
    parser.add_argument("--data", type=Path, default=Path("data/melb_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    homes = load_sales(args.data)
    train, test = split_by_date(homes)
    cutoff = test["Date"].min().date()
    suburb_names = train["Suburb"].str.strip().str.casefold()
    suburb_medians = train.groupby(suburb_names)["Price"].median()
    overall_median = train["Price"].median()

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=18,
        l2_regularization=1, early_stopping=False, categorical_features=["PropertyType"],
        random_state=42,
    )
    # A log target reduces the influence of the most expensive homes.
    model.fit(prepare_features(train), np.log1p(train["Price"]))
    baseline = suburb_baseline(test, suburb_medians, overall_median)
    predicted = np.expm1(model.predict(prepare_features(test)))
    actual = test["Price"].to_numpy()
    rows = [
        measure("Overall training median", actual, np.full(len(test), overall_median)),
        measure("Suburb training median", actual, baseline),
        measure("Gradient boosting", actual, predicted),
    ]
    summary = pd.DataFrame(rows, columns=["method", "mae_aud", "median_absolute_error_aud", "rmse_aud"])
    summary.to_csv(args.output / "summary.csv", index=False)
    print(f"{len(train):,} training sales; {len(test):,} test sales from {cutoff}")
    print(summary.to_string(index=False))

    names = test["Suburb"].str.strip().str.casefold()
    known = names.isin(suburb_medians.index)
    repeated = (test["Suburb"].astype(str) + " / " + test["Address"].astype(str)).isin(
        train["Suburb"].astype(str) + " / " + train["Address"].astype(str)
    )
    pd.DataFrame([
        {"split": "train", "sales": len(train), "from": train["Date"].min().date(),
         "through": train["Date"].max().date(), "unseen_suburb_sales": 0,
         "sales_with_an_earlier_matching_address": 0},
        {"split": "test", "sales": len(test), "from": cutoff,
         "through": test["Date"].max().date(), "unseen_suburb_sales": int((~known).sum()),
         "sales_with_an_earlier_matching_address": int(repeated.sum())},
    ]).to_csv(args.output / "split.csv", index=False)

    errors = pd.DataFrame({
        "month": test["Date"].dt.strftime("%Y-%m"),
        "suburb_seen": known,
        "suburb_error": np.abs(actual - baseline),
        "model_error": np.abs(actual - predicted),
    })
    for key, filename in [("month", "by_month.csv"), ("suburb_seen", "by_suburb_coverage.csv")]:
        grouped = errors.groupby(key).agg(
            sales=("model_error", "size"),
            suburb_median_mae=("suburb_error", "mean"),
            model_mae=("model_error", "mean"),
        ).reset_index()
        grouped.to_csv(args.output / filename, index=False)
        if key == "month":
            plot_months(grouped, args.output)

    joblib.dump({"model": model, "suburb_medians": suburb_medians, "overall_median": overall_median},
                args.output / "price_model.joblib")


if __name__ == "__main__":
    main()
