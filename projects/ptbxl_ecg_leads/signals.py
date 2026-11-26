import re
from pathlib import Path

import numpy as np


LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SAMPLE_RATE = 100
SAMPLES = 1000


def read_record(record):
    """Read the uncompressed WFDB format 16 used by PTB-XL records100."""
    record = Path(record)
    lines = record.with_suffix(".hea").read_text(encoding="ascii").splitlines()
    _, channels, rate, length = lines[0].split()[:4]
    if (int(channels), int(rate), int(length)) != (len(LEADS), SAMPLE_RATE, SAMPLES):
        raise ValueError(f"Expected a 12-lead, 100 Hz, 10-second PTB-XL record: {record}")

    gains, offsets, names = [], [], []
    for line in lines[1:len(LEADS) + 1]:
        fields = line.split()
        match = re.fullmatch(r"([0-9.]+)(?:\((-?[0-9]+)\))?/mV", fields[2])
        if fields[1] != "16" or not match:
            raise ValueError(f"Unsupported waveform format in {record}: {line}")
        gains.append(float(match.group(1)))
        offsets.append(int(match.group(2) or fields[4]))
        names.append(fields[-1].upper())
    if names != LEADS or min(gains) <= 0:
        raise ValueError(f"Unexpected lead order or invalid ADC gain in {record}")

    samples = np.fromfile(record.with_suffix(".dat"), dtype="<i2")
    if samples.size != SAMPLES * len(LEADS):
        raise ValueError(f"Unexpected number of samples in {record}")
    return (samples.reshape(SAMPLES, len(LEADS)).T - np.array(offsets)[:, None]) / np.array(gains)[:, None]


def lead_features(signal):
    centered = signal - np.median(signal, axis=1, keepdims=True)
    quantiles = np.quantile(centered, [0.05, 0.25, 0.75, 0.95], axis=1).T
    rms = np.sqrt(np.mean(centered ** 2, axis=1))
    change = np.sqrt(np.mean(np.diff(centered, axis=1) ** 2, axis=1))

    window = np.hanning(centered.shape[1])
    power = np.abs(np.fft.rfft(centered * window, axis=1)) ** 2
    hz = np.fft.rfftfreq(centered.shape[1], 1 / SAMPLE_RATE)
    total = np.maximum(power[:, hz >= 0.5].sum(axis=1), 1e-12)
    bands = []
    for low, high in [(0.5, 3), (3, 8), (8, 15), (15, 30), (30, 49.9)]:
        bands.append(power[:, (hz >= low) & (hz < high)].sum(axis=1) / total)

    # Measure waveform similarity at several lags without detecting individual beats.
    lags = []
    for lag in [35, 50, 65, 80, 100]:
        numerator = (centered[:, :-lag] * centered[:, lag:]).mean(axis=1)
        lags.append(numerator / np.maximum(rms ** 2, 1e-12))
    return np.column_stack([quantiles, rms, change, *bands, *lags]).astype(np.float32)
