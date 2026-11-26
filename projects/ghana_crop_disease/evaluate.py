from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from detector import detect


MATCH_IOU = 0.5
SCORE_CUTOFF = 0.15
SMALL_RATIO = 0.0025


def box_iou(box, boxes):
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=1)
    union = (box[2:] - box[:2]).prod() + (boxes[:, 2:] - boxes[:, :2]).prod(axis=1) - intersection
    return intersection / np.maximum(union, 1e-9)


def average_precision(detections, ground_truth):
    if not ground_truth:
        return np.nan
    if not detections:
        return 0.0
    true_positive = np.cumsum(detections)
    recall = true_positive / ground_truth
    precision = true_positive / np.arange(1, len(detections) + 1)
    recall = np.r_[0, recall, 1]
    precision = np.r_[0, precision, 0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * precision[1:]))


def evaluate(model, records, classes, tiled):
    all_scores = defaultdict(list)
    matrix = pd.DataFrame(0, index=[*classes, "background"], columns=[*classes, "missed"])
    totals = Counter()
    size_counts = Counter()
    examined = []

    for path, boxes, labels, width, height, _ in records:
        with Image.open(path) as image:
            predicted, scores, predictions = detect(model, image, width, height, tiled)

        totals.update(labels)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        small = area < SMALL_RATIO * width * height
        size_group = np.where(small, "Small disease boxes", "Other disease boxes")
        disease = np.array([not classes[label].endswith("_Healthy") for label in labels])
        for group in size_group[disease]:
            size_counts[group, "total"] += 1

        # AP matches by class. The confusion table matches spatially first.
        matched_for_ap = np.zeros(len(boxes), dtype=bool)
        matched_for_table = np.zeros(len(boxes), dtype=bool)
        small_hits = 0
        for box, score, label in zip(predicted, scores, predictions):
            ious = box_iou(box, boxes)
            same_class = ious.copy()
            same_class[(labels != label) | matched_for_ap] = -1
            best_class = same_class.argmax()
            hit = same_class[best_class] >= MATCH_IOU
            if hit:
                matched_for_ap[best_class] = True
            all_scores[label].append((score, int(hit)))

            if score < SCORE_CUTOFF:
                continue
            ious[matched_for_table] = -1
            best = ious.argmax()
            if ious[best] >= MATCH_IOU:
                matched_for_table[best] = True
                matrix.loc[classes[labels[best]], classes[label]] += 1
                if disease[best]:
                    group = size_group[best]
                    size_counts[group, "localized"] += 1
                    if labels[best] == label:
                        size_counts[group, "correct class"] += 1
                        small_hits += int(small[best])
            else:
                matrix.loc["background", classes[label]] += 1

        for label in labels[~matched_for_table]:
            matrix.loc[classes[label], "missed"] += 1
        examined.append((small_hits, path.name))

    per_class = []
    for i, name in enumerate(classes):
        ranked = sorted(all_scores[i], key=lambda item: item[0], reverse=True)
        ap = average_precision([hit for _, hit in ranked], totals[i])
        per_class.append((name, totals[i], ap))

    sizes = []
    for group in ("Small disease boxes", "Other disease boxes"):
        total = size_counts[group, "total"]
        localized = size_counts[group, "localized"]
        correct = size_counts[group, "correct class"]
        sizes.append((group, total, localized, correct,
                      localized / total if total else np.nan,
                      correct / total if total else np.nan))

    return (
        pd.DataFrame(per_class, columns=["class", "test_boxes", "AP50"]),
        pd.DataFrame(sizes, columns=["size", "test_boxes", "localized", "correct_class",
                                     "localization_recall", "class_correct_recall"]),
        matrix,
        dict((name, hits) for hits, name in examined),
    )
