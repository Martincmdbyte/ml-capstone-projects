import argparse
from pathlib import Path

from features import NOTE_NAMES, chord_features, triad_scores


def main():
    parser = argparse.ArgumentParser(description="Estimate the root and quality of one isolated chord.")
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()

    chroma, bass = chord_features(args.audio)
    major, minor = triad_scores(chroma, bass)
    quality, scores = ("major", major) if major.max() >= minor.max() else ("minor", minor)
    root = NOTE_NAMES[scores.argmax()]
    print(f"{root} {quality}")
    print(f"Best major fit: {major.max():.3f} | Best minor fit: {minor.max():.3f}")


if __name__ == "__main__":
    main()
