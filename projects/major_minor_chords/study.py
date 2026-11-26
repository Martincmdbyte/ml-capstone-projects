import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from features import chord_features, template_features


def read_dataset(data):
    entries = []
    by_audio = {}
    for label in ("major", "minor"):
        for path in sorted((data / label).glob("*.wav")):
            group = path.stem.split("_")[0]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            by_audio.setdefault(digest, []).append((path, label, group))

    for files in by_audio.values():
        # Copies with conflicting class labels cannot provide a reliable target.
        if len({row[1] for row in files}) == 1:
            entries.append(files[0])
    if {row[1] for row in entries} != {"major", "minor"}:
        raise ValueError("The dataset must contain both major/ and minor/ WAV folders.")
    return entries, sum(map(len, by_audio.values())) - len(entries)


def scores(actual, predicted):
    return (accuracy_score(actual, predicted), balanced_accuracy_score(actual, predicted),
            f1_score(actual, predicted, average="macro"))


def plot_groups(groups, output):
    figure, ax = plt.subplots(figsize=(8.8, 4.6))
    x = np.arange(len(groups))
    ax.bar(x - 0.18, groups["chroma_accuracy"], 0.35, label="Raw chroma")
    ax.bar(x + 0.18, groups["triad_accuracy"], 0.35, label="Triad-fit features")
    ax.set(xticks=x, xticklabels=groups["family"], ylim=(0, 1.06),
           xlabel="Held-out filename prefix", ylabel="Accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.set_title("Major/minor recognition by held-out recording family")
    figure.tight_layout()
    figure.savefig(output / "accuracy_by_family.png", dpi=145)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Compare two ways to recognize major and minor chords.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    entries, excluded = read_dataset(args.data)
    chromas, fits = [], []
    for path, _, _ in entries:
        chroma, bass = chord_features(path)
        chromas.append(chroma)
        fits.append(template_features(chroma, bass))

    chromas, fits = np.array(chromas), np.array(fits)
    labels = np.array([label for _, label, _ in entries])
    groups = np.array([group for _, _, group in entries])
    baseline_predictions = np.full_like(labels, "major")
    chroma_predictions = np.empty_like(labels)
    triad_predictions = np.empty_like(labels)
    template_predictions = np.where(fits[:, 2] >= 0, "major", "minor")
    group_results = []

    for group in np.unique(groups):
        test = groups == group
        train = ~test
        if len(np.unique(labels[train])) != 2:
            raise ValueError(f"Training must include both chord qualities when holding out {group}.")

        for features, predicted in [(chromas, chroma_predictions), (fits, triad_predictions)]:
            model = LogisticRegression(C=10, class_weight="balanced")
            model.fit(features[train], labels[train])
            predicted[test] = model.predict(features[test])

        group_results.append((group, int(test.sum()), int((test & (labels == "major")).sum()),
                              accuracy_score(labels[test], chroma_predictions[test]),
                              accuracy_score(labels[test], triad_predictions[test]),
                              accuracy_score(labels[test], template_predictions[test])))

    comparison = [
        ("Always major", baseline_predictions),
        ("Raw chroma + logistic regression", chroma_predictions),
        ("Triad-fit features + logistic regression", triad_predictions),
        ("Best triad template (no training)", template_predictions),
    ]
    rows = [(name, *scores(labels, predicted)) for name, predicted in comparison]
    summary = pd.DataFrame(rows, columns=["method", "accuracy", "balanced_accuracy", "macro_f1"])
    summary.to_csv(args.output / "summary.csv", index=False)
    print(f"{len(entries)} clips, {len(np.unique(groups))} families, {excluded} duplicate rows excluded")
    print(summary.to_string(index=False))

    group_table = pd.DataFrame(group_results, columns=[
        "family", "clips", "major_clips", "chroma_accuracy", "triad_accuracy", "template_accuracy",
    ])
    group_table.to_csv(args.output / "by_family.csv", index=False)
    plot_groups(group_table, args.output)
    matrix = confusion_matrix(labels, triad_predictions, labels=["major", "minor"])
    pd.DataFrame(matrix, index=["major", "minor"], columns=["major", "minor"]).to_csv(
        args.output / "triad_confusion.csv", index_label="actual"
    )
    pd.DataFrame([{"clips": len(entries), "major": int((labels == "major").sum()),
                   "minor": int((labels == "minor").sum()), "duplicates_excluded": excluded}]).to_csv(
        args.output / "dataset_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
