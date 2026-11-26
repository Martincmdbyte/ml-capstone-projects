import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import batched_nms

from data import CANVAS, GRID, SIZE, STRIDE, make_targets, training_view


def build_model(class_count):
    layers = []
    in_channels = 3
    for out_channels, stride in [(16, 2), (24, 2), (40, 2), (48, 1), (64, 1), (64, 1)]:
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ])
        in_channels = out_channels
    layers.append(nn.Conv2d(in_channels, 5 + class_count, 1))
    return nn.Sequential(*layers)


def to_tensor(pixels):
    return torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).float() / 255


def train_one_epoch(model, optimizer, records, batch_size, seed):
    model.train()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    running_loss = 0

    for start in range(0, len(order), batch_size):
        inputs, labels, offsets = [], [], []
        for index in order[start:start + batch_size]:
            pixels, boxes, box_labels = training_view(records[index], rng)
            cells, target = make_targets(boxes, box_labels)
            inputs.append(pixels)
            labels.append(cells)
            offsets.append(target)

        labels = torch.from_numpy(np.stack(labels))
        offsets = torch.from_numpy(np.stack(offsets))
        prediction = model(to_tensor(inputs))
        has_box = labels >= 0
        objectness = prediction[:, 0]
        class_logits = prediction[:, 5:].permute(0, 2, 3, 1)
        predicted_offsets = prediction[:, 1:5].permute(0, 2, 3, 1)

        # Balance the rare positive cells against the many background cells.
        negative_loss = F.softplus(objectness[~has_box]).mean()
        loss = negative_loss
        if has_box.any():
            positive_loss = F.softplus(-objectness[has_box]).mean()
            class_loss = F.cross_entropy(class_logits[has_box], labels[has_box])
            positive_offsets = predicted_offsets[has_box]
            positive_offsets = torch.cat([positive_offsets[:, :2].sigmoid(), positive_offsets[:, 2:]], dim=1)
            box_loss = F.smooth_l1_loss(positive_offsets, offsets.permute(0, 2, 3, 1)[has_box])
            loss = positive_loss + negative_loss + class_loss + 3 * box_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / ((len(order) + batch_size - 1) // batch_size)


@torch.inference_mode()
def detect(model, image, width, height, tiled, score_floor=0.01):
    if tiled:
        starts = [0, (CANVAS - SIZE) // 2, CANVAS - SIZE]
        views = [(x, y, SIZE) for y in starts for x in starts if x < width and y < height]
    else:
        views = [(0, 0, CANVAS)]

    pixels = []
    for left, top, side in views:
        crop = image.crop((left, top, left + side, top + side))
        pixels.append(np.array(crop.resize((SIZE, SIZE), Image.Resampling.BILINEAR)))

    predictions = model(to_tensor(pixels))
    all_boxes, all_scores, all_labels = [], [], []
    for prediction, (left, top, side) in zip(predictions, views):
        class_probability, class_id = prediction[5:].softmax(dim=0).max(dim=0)
        scores = prediction[0].sigmoid() * class_probability
        best_scores, indexes = scores.flatten().topk(80)
        indexes = indexes[best_scores >= score_floor]
        row, col = indexes // GRID, indexes % GRID
        offsets = prediction[1:5, row, col]
        x = (col + offsets[0].sigmoid()) * STRIDE * side / SIZE + left
        y = (row + offsets[1].sigmoid()) * STRIDE * side / SIZE + top
        w, h = offsets[2:].clamp(-2.5, 3.5).exp() * STRIDE * side / SIZE
        boxes = torch.stack([
            (x - w / 2).clamp(0, width),
            (y - h / 2).clamp(0, height),
            (x + w / 2).clamp(0, width),
            (y + h / 2).clamp(0, height),
        ], dim=1)
        keep = (x < width) & (y < height)
        all_boxes.append(boxes[keep])
        all_scores.append(scores[row, col][keep])
        all_labels.append(class_id[row, col][keep])

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    keep = batched_nms(boxes, scores, labels, 0.45)[:100]
    return boxes[keep].numpy(), scores[keep].numpy(), labels[keep].numpy()
