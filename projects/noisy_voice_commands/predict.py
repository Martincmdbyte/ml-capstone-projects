import argparse
from pathlib import Path

import torch

from audio import make_features
from study import build_model


def main():
    parser = argparse.ArgumentParser(description="Predict a command from one WAV clip.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, default=Path("outputs/mixed.pt"))
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f'Audio file does not exist: {args.audio}')

    torch.set_num_threads(2)
    saved = torch.load(args.model, map_location="cpu", weights_only=True)
    model = build_model(len(saved["classes"]))
    model.load_state_dict(saved["model"])
    model.eval()
    inputs = torch.from_numpy(make_features(args.audio)[None, None])
    with torch.inference_mode():
        scores = model(inputs).softmax(dim=1)[0]
    for index in scores.argsort(descending=True)[:3].tolist():
        print(f"{saved['classes'][index]}: {scores[index]:.1%}")


if __name__ == "__main__":
    main()
