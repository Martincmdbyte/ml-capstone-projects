import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


CANVAS = 640
SIZE = 256
STRIDE = 8
GRID = SIZE // STRIDE
BOX_COLUMNS = ["xmin", "ymin", "xmax", "ymax"]


def get_annotations(data, cache, sample_size=0):
    labels = pd.read_csv(data / "Train.csv")
    labels = labels[["Image_ID", "class", *BOX_COLUMNS]].dropna().drop_duplicates()
    classes = sorted(labels["class"].unique())
    class_number = {name: i for i, name in enumerate(classes)}
    labels["class_id"] = labels["class"].map(class_number)

    cache.mkdir(parents=True, exist_ok=True)
    # Each study run gets its own cache under the selected output folder.
    records = []
    counts = labels.groupby("class_id")["Image_ID"].nunique().to_dict()
    seen = set()
    if sample_size:
        image_ids = sorted(labels["Image_ID"].unique())
        rng = np.random.default_rng(42)
        sample = rng.choice(image_ids, min(sample_size, len(image_ids)), replace=False)
        labels = labels[labels["Image_ID"].isin(sample)]

    for image_id, rows in labels.groupby("Image_ID"):
        source = data / "images" / image_id
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)

        path = cache / image_id
        with Image.open(source) as original:
            width, height = original.size
            scale = CANVAS / max(width, height)
            new_width = max(1, round(width * scale))
            new_height = max(1, round(height * scale))
            if not path.exists():
                original.draft("RGB", (new_width, new_height))
                image = original.convert("RGB").resize((new_width, new_height), Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", (CANVAS, CANVAS), (114, 114, 114))
                canvas.paste(image, (0, 0))
                canvas.save(path, quality=90)

        boxes = rows[BOX_COLUMNS].to_numpy(dtype=np.float32)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width) * (new_width / width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height) * (new_height / height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not valid.any():
            continue

        ids = rows["class_id"].to_numpy()[valid]
        rarest = min(ids, key=counts.get)
        records.append((path, boxes[valid], ids, new_width, new_height, rarest))

        if len(records) % 1000 == 0:
            print(f"Prepared {len(records):,} photos", flush=True)

    return records, classes


def split_images(records):
    # Keep the rarest annotation type on each image represented where possible.
    labels = pd.Series([record[-1] for record in records])
    counts = labels.value_counts()
    labels = labels.where(labels.map(counts) >= 2, -1)
    test_size = int(np.ceil(0.2 * len(records)))
    stratify = None
    if labels.value_counts().min() >= 2 and labels.nunique() <= test_size:
        stratify = labels
    return train_test_split(records, test_size=0.2, random_state=42, stratify=stratify)


def training_view(record, rng):
    path, boxes, ids, _, _, _ = record
    if rng.random() < 0.22:
        left, top, side = 0, 0, CANVAS
    else:
        box = boxes[rng.integers(len(boxes))]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        side = int(rng.choice([256, 320, 416]))
        left = int(np.clip(center_x - side * rng.uniform(0.25, 0.75), 0, CANVAS - side))
        top = int(np.clip(center_y - side * rng.uniform(0.25, 0.75), 0, CANVAS - side))

    with Image.open(path) as image:
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        pixels = np.array(image)

    shifted = boxes - np.array([left, top, left, top], dtype=np.float32)
    clipped = np.clip(shifted, 0, side)
    original_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    visible_area = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    keep = visible_area >= 0.65 * original_area
    clipped = clipped[keep] * (SIZE / side)
    ids = ids[keep]

    if rng.random() < 0.5:
        pixels = pixels[:, ::-1].copy()
        clipped[:, [0, 2]] = SIZE - clipped[:, [2, 0]]

    return pixels, clipped, ids


def make_targets(boxes, labels):
    cells = np.full((GRID, GRID), -1, dtype=np.int64)
    offsets = np.zeros((4, GRID, GRID), dtype=np.float32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # If two centers share a cell, keep the smaller object.
    for i in np.argsort(areas):
        x1, y1, x2, y2 = boxes[i] / STRIDE
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        col, row = int(center_x), int(center_y)
        if cells[row, col] >= 0:
            continue
        cells[row, col] = labels[i]
        offsets[:, row, col] = [center_x - col, center_y - row,
                                np.log(max(x2 - x1, 0.125)),
                                np.log(max(y2 - y1, 0.125))]

    return cells, offsets


def read_photo(path):
    with Image.open(Path(path)) as original:
        width, height = original.size
        scale = CANVAS / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        original.draft("RGB", (new_width, new_height))
        image = original.convert("RGB").resize((new_width, new_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (CANVAS, CANVAS), (114, 114, 114))
        canvas.paste(image, (0, 0))
    return canvas, new_width, new_height
