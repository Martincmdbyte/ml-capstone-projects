from math import gcd

import numpy as np
from scipy import fft, signal
import soundfile as sf


RATE = 11025
WINDOW = 4096
FFT_SIZE = 8192
HOP = 1024
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chord_features(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:
        divisor = gcd(rate, RATE)
        audio = signal.resample_poly(audio, RATE // divisor, rate // divisor)
    if len(audio) < WINDOW:
        audio = np.pad(audio, (0, WINDOW - len(audio)))
    audio = audio - audio.mean()
    frames = np.lib.stride_tricks.sliding_window_view(audio, WINDOW)[::HOP]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    if rms.max() < 1e-6:
        raise ValueError(f"No measurable audio in {path}")

    active = frames[rms > rms.max() * 0.12]
    window = np.hanning(WINDOW).astype(np.float32)
    magnitudes = np.abs(fft.rfft(active * window, n=FFT_SIZE))
    chroma = np.zeros(12)
    bass = np.zeros(12)
    low_bin = round(55 * FFT_SIZE / RATE)
    high_bin = round(1500 * FFT_SIZE / RATE)

    # A chroma bin collects the same note name across different octaves.
    for spectrum in magnitudes:
        peaks, _ = signal.find_peaks(spectrum[low_bin:high_bin], prominence=spectrum.max() * 0.015)
        peaks = peaks + low_bin
        if not len(peaks):
            continue
        left = np.log(np.maximum(spectrum[peaks - 1], 1e-12))
        center = np.log(np.maximum(spectrum[peaks], 1e-12))
        right = np.log(np.maximum(spectrum[peaks + 1], 1e-12))
        shift = 0.5 * (left - right) / np.minimum(left - 2 * center + right, -1e-8)
        frequencies = (peaks + shift.clip(-0.5, 0.5)) * RATE / FFT_SIZE
        notes = np.rint(69 + 12 * np.log2(frequencies / 440)).astype(int)
        strengths = spectrum[peaks] / np.sqrt(frequencies)
        np.add.at(chroma, notes % 12, strengths)
        low = frequencies < 260
        np.add.at(bass, notes[low] % 12, strengths[low])

    if chroma.sum() == 0:
        raise ValueError(f"No musical peaks detected in {path}")
    chroma /= chroma.sum()
    bass /= max(bass.sum(), 1e-9)
    return chroma, bass


def triad_scores(chroma, bass):
    scores = []
    for third in (4, 3):
        matches = []
        for root in range(12):
            notes = [root, (root + third) % 12, (root + 7) % 12]
            # Bass emphasis is small so an inversion can still score well.
            match = chroma[notes].sum() + 0.08 * bass[root]
            matches.append(match)
        scores.append(np.array(matches))
    return scores


def template_features(chroma, bass):
    major, minor = triad_scores(chroma, bass)
    best_major = float(major.max())
    best_minor = float(minor.max())
    return [best_major, best_minor, best_major - best_minor]
