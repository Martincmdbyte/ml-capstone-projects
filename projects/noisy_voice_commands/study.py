import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from audio import BANDS, FRAMES, dataset_folder, make_features


def build_model(class_count):
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(48 * (BANDS // 8) * (FRAMES // 8), 64), nn.ReLU(),
        nn.Linear(64, class_count),
    )


def find_pairs(regular, very_noisy):
    regular = dataset_folder(regular)
    very_noisy = dataset_folder(very_noisy)
    names = sorted(path.relative_to(regular).as_posix() for path in regular.glob("*/*.wav"))
    noisy_names = {path.relative_to(very_noisy).as_posix() for path in very_noisy.glob("*/*.wav")}
    if not names or set(names) != noisy_names:
        raise ValueError("Pass both audio folders containing matching label/name.wav paths.")
    return regular, very_noisy, names


def get_features(root, names, cache):
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if data["names"].tolist() == names and str(data["root"]) == str(root.resolve()):
                return data["features"]

    features = np.empty((len(names), BANDS, FRAMES), dtype=np.float32)
    for i, name in enumerate(names):
        features[i] = make_features(root / name)
        if (i + 1) % 10000 == 0:
            print(f"{root.name}: extracted {i + 1:,} clips", flush=True)
    np.savez(cache, features=features, names=names, root=str(root.resolve()))
    return features


def fit(regular, noisy, labels, train, class_count, epochs, mixed):
    torch.manual_seed(42)
    model = build_model(class_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.CrossEntropyLoss()
    rng = np.random.default_rng(42)
    mix_rng = np.random.default_rng(84)
    losses = []

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train)
        loss_total = 0
        for start in range(0, len(order), 128):
            rows = order[start:start + 128]
            batch = regular[rows].copy()
            if mixed:
                use_noisy = mix_rng.random(len(rows)) < 0.5
                batch[use_noisy] = noisy[rows[use_noisy]]
            inputs = torch.from_numpy(batch[:, None])
            targets = torch.from_numpy(labels[rows])
            loss = loss_function(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(rows)

        losses.append(loss_total / len(train))
        mode = "mixed" if mixed else "regular"
        print(f"{mode:7s} epoch {epoch + 1:2d}/{epochs}: loss {losses[-1]:.3f}", flush=True)
    return model, losses


@torch.inference_mode()
def predict(model, features):
    model.eval()
    predicted = []
    for start in range(0, len(features), 256):
        inputs = torch.from_numpy(features[start:start + 256, None])
        predicted.extend(model(inputs).argmax(dim=1).tolist())
    return predicted


def save_chart(command_scores, output):
    ordered = command_scores.sort_values("regular_to_noisy_drop", ascending=False).head(10)
    figure, ax = plt.subplots(figsize=(8.3, 4.9))
    y = np.arange(len(ordered))
    ax.barh(y + 0.18, ordered["regular_model_noisy_recall"], height=0.34, label="Regular-only training")
    ax.barh(y - 0.18, ordered["mixed_model_noisy_recall"], height=0.34, label="Mixed training")
    ax.set(yticks=y, yticklabels=ordered["command"], xlim=(0, 1), xlabel="Recall on very-noisy test clips")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Commands with the largest recall drops under added noise")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(output / "hardest_commands.png", dpi=145)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Compare two training sets on paired voice commands.")
    parser.add_argument("--regular", type=Path, default=Path("data/augmented_dataset"))
    parser.add_argument("--noisy", type=Path, default=Path("data/augmented_dataset_verynoisy"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("Epochs and threads must be positive.")
    torch.set_num_threads(args.threads)

    args.output.mkdir(parents=True, exist_ok=True)
    ordinary_root, noisy_root, names = find_pairs(args.regular, args.noisy)
    words = [name.split("/")[0] for name in names]
    classes, labels = np.unique(words, return_inverse=True)
    regular = get_features(ordinary_root, names, args.output / "regular_features.npz")
    noisy = get_features(noisy_root, names, args.output / "noisy_features.npz")

    # Both versions of an utterance stay on the same side of the split.
    train, test = train_test_split(np.arange(len(names)), test_size=0.2, random_state=42, stratify=labels)
    print(f"{len(train):,} train pairs, {len(test):,} test pairs, {len(classes)} commands", flush=True)
    actual = labels[test]
    majority = np.bincount(labels[train]).argmax()
    baseline = np.full(len(test), majority)
    metrics = [("Majority baseline", "Either", accuracy_score(actual, baseline),
                f1_score(actual, baseline, average="macro"))]
    command_scores = pd.DataFrame({"command": classes, "test_pairs": np.bincount(actual)})
    loss_log = pd.DataFrame({"epoch": range(1, args.epochs + 1)})

    for slug, display, mixed in [("regular", "Regular only", False), ("mixed", "Regular + very noisy", True)]:
        model, losses = fit(regular, noisy, labels, train, len(classes), args.epochs, mixed)
        torch.save({"classes": classes.tolist(), "model": model.state_dict()}, args.output / f"{slug}.pt")
        loss_log[slug] = losses
        for condition, features in [("regular", regular), ("noisy", noisy)]:
            predicted = predict(model, features[test])
            accuracy = accuracy_score(actual, predicted)
            macro_f1 = f1_score(actual, predicted, average="macro")
            metrics.append((display, condition, accuracy, macro_f1))
            matrix = confusion_matrix(actual, predicted, labels=np.arange(len(classes)))
            table = pd.DataFrame(matrix, index=classes, columns=classes)
            table.to_csv(args.output / f"{slug}_on_{condition}_confusion.csv", index_label="actual")
            command_scores[f"{slug}_model_{condition}_recall"] = matrix.diagonal() / matrix.sum(axis=1)

    command_scores["regular_to_noisy_drop"] = (
        command_scores["regular_model_regular_recall"] - command_scores["regular_model_noisy_recall"]
    )
    command_scores["noisy_recall_gain_from_mixing"] = (
        command_scores["mixed_model_noisy_recall"] - command_scores["regular_model_noisy_recall"]
    )
    command_scores.sort_values("regular_to_noisy_drop", ascending=False).to_csv(
        args.output / "per_command_recall.csv", index=False
    )
    loss_log.to_csv(args.output / "training_loss.csv", index=False)
    summary = pd.DataFrame(metrics, columns=["training", "test_audio", "accuracy", "macro_f1"])
    summary.to_csv(args.output / "summary.csv", index=False)
    save_chart(command_scores, args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
