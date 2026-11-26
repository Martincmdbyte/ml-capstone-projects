from math import gcd

import numpy as np
from scipy import fft, signal
import soundfile as sf


RATE = 16000
WINDOW = 480
HOP = 320
FFT_SIZE = 512
BANDS = 32
FRAMES = 24


def mel_filters():
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1 / RATE)
    low = 2595 * np.log10(1 + 60 / 700)
    high = 2595 * np.log10(1 + 7600 / 700)
    edges = 700 * (10 ** (np.linspace(low, high, BANDS + 2) / 2595) - 1)
    filters = []
    for left, center, right in zip(edges, edges[1:], edges[2:]):
        rise = (frequencies - left) / (center - left)
        fall = (right - frequencies) / (right - center)
        triangle = np.maximum(0, np.minimum(rise, fall))
        filters.append(triangle / triangle.sum())
    return np.array(filters, dtype=np.float32)


FILTERS = mel_filters()
HANN = np.hanning(WINDOW).astype(np.float32)


def read_clip(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != RATE:
        divisor = gcd(sample_rate, RATE)
        audio = signal.resample_poly(audio, RATE // divisor, sample_rate // divisor)
    if len(audio) > RATE:
        start = (len(audio) - RATE) // 2
        audio = audio[start:start + RATE]
    elif len(audio) < RATE:
        missing = RATE - len(audio)
        audio = np.pad(audio, (missing // 2, missing - missing // 2))
    return audio


def make_features(path):
    audio = read_clip(path)
    windows = np.lib.stride_tricks.sliding_window_view(audio, WINDOW)[::HOP]
    spectrum = fft.rfft(windows * HANN, n=FFT_SIZE, axis=1)
    power = np.abs(spectrum) ** 2
    log_mel = np.log10(np.maximum(power @ FILTERS.T, 1e-8))
    # Average adjacent time frames to keep the model small.
    log_mel = log_mel[:FRAMES * 2].reshape(FRAMES, 2, BANDS).mean(axis=1)
    log_mel = (log_mel - log_mel.mean()) / max(log_mel.std(), 0.1)
    return log_mel.T.astype(np.float32)


def dataset_folder(root):
    nested = root / root.name
    return nested if nested.is_dir() else root
