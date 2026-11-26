import argparse
from pathlib import Path

from PIL import ImageDraw
import torch

from data import read_photo
from detector import build_model, detect
from evaluate import SCORE_CUTOFF


def main():
    parser = argparse.ArgumentParser(description="Draw disease boxes on a crop photo.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("outputs/detector.pt"))
    parser.add_argument("--save", type=Path, default=Path("outputs/prediction.jpg"))
    parser.add_argument("--score", type=float, default=SCORE_CUTOFF)
    parser.add_argument("--whole", action="store_true", help="Scan the resized whole photo.")
    args = parser.parse_args()
    if not 0 <= args.score <= 1:
        parser.error("Score must be between 0 and 1.")

    torch.set_num_threads(4)
    saved = torch.load(args.model, map_location="cpu", weights_only=True)
    classes = saved["classes"]
    model = build_model(len(classes))
    model.load_state_dict(saved["model"])
    model.eval()

    canvas, width, height = read_photo(args.image)
    boxes, scores, labels = detect(model, canvas, width, height, not args.whole, args.score)
    photo = canvas.crop((0, 0, width, height))
    draw = ImageDraw.Draw(photo)
    for box, score, label in zip(boxes[::-1], scores[::-1], labels[::-1]):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle((x1, y1, x2, y2), outline="yellow", width=2)
        draw.text((x1 + 1, max(0, y1 - 12)), f"{classes[label]} {score:.2f}",
                  fill="yellow", stroke_width=1, stroke_fill="black")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    photo.save(args.save)
    print(f"Saved {len(boxes)} detections to {args.save}")


if __name__ == "__main__":
    main()
