import argparse
import ast
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from signals import LEADS, lead_features, read_record


LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


def training_metadata(data):
    metadata = pd.read_csv(data / "ptbxl_database.csv")
    statements = pd.read_csv(data / "scp_statements.csv", index_col=0)
    mapping = statements.loc[statements["diagnostic"].eq(1), "diagnostic_class"].to_dict()
    rows = []
    for scp_codes in metadata["scp_codes"]:
        classes = {mapping.get(code) for code in ast.literal_eval(scp_codes)}
        rows.append([int(label in classes) for label in LABELS])
    y = np.array(rows, dtype=np.int8)
    keep = y.any(axis=1)
    return metadata.loc[keep].reset_index(drop=True), y[keep]


def load_features(data, output, metadata):
    cache = output / "lead_features.npz"
    names = metadata["filename_lr"].to_numpy(dtype=str)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["source"]) == str(data.resolve()) and np.array_equal(saved["names"], names):
                return saved["features"]
    features = []
    for i, name in enumerate(names, 1):
        features.append(lead_features(read_record(data / name)))
        if i % 4000 == 0:
            print(f"Read {i:,} ECG waveforms", flush=True)
    features = np.stack(features)
    np.savez(cache, names=names, features=features, source=str(data.resolve()))
    return features


def measure_speed(model, signals, selected):
    extraction, scoring = [], []
    for signal in signals:
        start = perf_counter()
        row = lead_features(signal[selected]).reshape(1, -1)
        after_features = perf_counter()
        model.predict_proba(row)
        end = perf_counter()
        extraction.append(1000 * (after_features - start))
        scoring.append(1000 * (end - after_features))
    return np.median(extraction), np.median(scoring)


def plot_labels(scores, output):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(LABELS))
    for offset, name in [(-0.18, "Lead II"), (0.18, "All 12 leads")]:
        values = scores.loc[scores["leads"].eq(name), "average_precision"]
        ax.bar(x + offset, values, width=0.34, label=name)
    ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 1), ylabel="Average precision",
           title="ECG label performance on held-out patients")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "lead_comparison.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare one ECG lead with all 12 on PTB-XL.")
    parser.add_argument("--data", type=Path, default=Path("data/ptb-xl"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata, y = training_metadata(args.data)
    train = metadata["strat_fold"].between(1, 8).to_numpy()
    validation = metadata["strat_fold"].eq(9).to_numpy()
    test = metadata["strat_fold"].eq(10).to_numpy()
    groups = [set(metadata.loc[mask, "patient_id"]) for mask in (train, validation, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError("Patient IDs overlap between official folds.")
    rows = [
        (name, int(mask.sum()), len(patients))
        for name, mask, patients in zip(["train", "validation", "test"], [train, validation, test], groups)
    ]
    pd.DataFrame(rows, columns=["split", "ecgs", "patients"]).to_csv(
        args.output / "split_summary.csv", index=False,
    )

    features = load_features(args.data, args.output, metadata)
    timed_signals = [read_record(args.data / name) for name in metadata.loc[test, "filename_lr"].head(100)]
    by_label, summary = [], []
    with threadpool_limits(limits=1):
        for name, selected, slug in [
            ("Lead II", [LEADS.index("II")], "lead_ii"),
            ("All 12 leads", list(range(len(LEADS))), "all_leads"),
        ]:
            x = features[:, selected].reshape(len(features), -1)
            model = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(max_iter=700)))
            model.fit(x[train], y[train])
            probabilities = model.predict_proba(x[test])
            actual = y[test]
            ap = average_precision_score(actual, probabilities, average=None)
            auc = roc_auc_score(actual, probabilities, average=None)
            for label, count, p, a in zip(LABELS, actual.sum(axis=0), ap, auc):
                by_label.append((name, label, int(count), p, a))
            extraction, scoring = measure_speed(model, timed_signals, selected)
            summary.append((name, ap.mean(), auc.mean(), extraction, scoring))
            joblib.dump({"model": model, "leads": selected, "labels": LABELS}, args.output / f"{slug}.joblib")

    metrics = pd.DataFrame(summary, columns=[
        "leads", "macro_average_precision", "macro_roc_auc", "median_feature_ms", "median_model_ms",
    ])
    by_label = pd.DataFrame(by_label, columns=["leads", "label", "positive_test_ecgs", "average_precision", "roc_auc"])
    metrics.to_csv(args.output / "metrics.csv", index=False)
    by_label.to_csv(args.output / "per_label.csv", index=False)
    plot_labels(by_label, args.output)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
