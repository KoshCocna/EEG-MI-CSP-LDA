#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fake Motor Imagery (MI) epoch CSV generator
- Output format matches `epoch_p300.csv` (timestamp_sec + ch0..ch23)
- Generates 4-second epochs at 250 Hz (1000 samples)
- Saves 30 CSVs per class under: data/<class>/epoch_MI_<class>_XX.csv

NOTE:
This is synthetic data for pipeline testing only (NOT physiologically accurate).
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ["left", "right", "up", "down", "zoomIn", "zoomOut"]

def _synth_epoch(label: str, *, sfreq: float, n_times: int, n_channels: int, seed: int) -> np.ndarray:
    """Return (n_times, n_channels) synthetic epoch."""
    r = np.random.default_rng(seed)
    t = np.arange(n_times) / sfreq

    # AR-ish colored noise
    noise = r.normal(0, 1, (n_times, n_channels))
    ar = 0.97
    for i in range(1, n_times):
        noise[i] = ar * noise[i - 1] + 0.3 * noise[i]
    noise *= 0.35

    # Mu/Beta oscillations (within 8-30 Hz band)
    mu_freq = 10.0 + r.normal(0, 0.4)
    beta_freq = 20.0 + r.normal(0, 0.8)
    mu = np.sin(2 * np.pi * mu_freq * t + r.uniform(0, 2*np.pi))
    beta = np.sin(2 * np.pi * beta_freq * t + r.uniform(0, 2*np.pi))

    # Slow drift
    drift = 0.08 * np.sin(2 * np.pi * 0.5 * t + r.uniform(0, 2*np.pi))

    # Pseudo "spatial" patterns (just to make CSP separable)
    # Channel indices follow your 24ch electrode layout:
    # ch3=FZ, ch5=C4, ch6=FC5, ch7=FC6, ch12=C3, ch17=CZ, ch4=PZ, ch16=AFZ ...
    c3 = (12,)
    c4 = (5,)
    cz = (17,)
    fz = (3,)
    fc_left = (6,)   # FC5
    fc_right = (7,)  # FC6
    parietal = (4, 14, 15)  # PZ/P7/P8
    occipital = (8, 9, 21)  # O1/O2/OZ

    # Heuristic MI-like separability:
    # - left/right: contralateral modulation over C3/C4 (+ nearby FC5/FC6)
    # - up/down: midline (CZ) + parietal/occipital difference
    # - zoomIn/zoomOut: frontal (FZ/AFZ) vs parietal emphasis
    patterns = {
        "left": {
            "mu": {c4: 2.0, fc_right: 1.0, cz: 0.6},
            "beta": {c4: 1.4, fc_right: 0.8},
        },
        "right": {
            "mu": {c3: 2.0, fc_left: 1.0, cz: 0.6},
            "beta": {c3: 1.4, fc_left: 0.8},
        },
        "up": {
            "mu": {cz: 2.2, fz: 0.8},
            "beta": {cz: 1.5},
        },
        "down": {
            "mu": {parietal: 2.0, occipital: 0.9, cz: 0.7},
            "beta": {parietal: 1.3},
        },
        "zoomIn": {
            "mu": {(3, 16): 2.0, fz: 1.0},   # (FZ, AFZ) emphasis
            "beta": {(3, 16): 1.3},
        },
        "zoomOut": {
            "mu": {parietal: 2.0, (14, 15): 1.2, fz: 0.4},
            "beta": {parietal: 1.2},
        },
    }

    w_mu = np.ones(n_channels) * 0.15
    w_beta = np.ones(n_channels) * 0.08

    for grp, amp in patterns[label]["mu"].items():
        for ch in grp:
            if ch < n_channels:
                w_mu[ch] += amp
    for grp, amp in patterns[label]["beta"].items():
        for ch in grp:
            if ch < n_channels:
                w_beta[ch] += amp

    # small per-epoch variability
    w_mu *= r.normal(1.0, 0.08, size=n_channels)
    w_beta *= r.normal(1.0, 0.10, size=n_channels)

    X = noise + mu[:, None] * w_mu[None, :] + beta[:, None] * w_beta[None, :] + drift[:, None]
    X *= 5.0  # scale (arbitrary)

    return X

def generate_dataset(
    out_root: str = "data",
    n_per_class: int = 30,
    sfreq: float = 250.0,
    epoch_sec: float = 4.0,
    n_channels: int = 24,
    seed: int = 20260211,
    reset: bool = True,
) -> Path:
    out_root = Path(out_root)

    if reset and out_root.exists():
        shutil.rmtree(out_root)

    for c in CLASSES:
        (out_root / c).mkdir(parents=True, exist_ok=True)

    n_times = int(epoch_sec * sfreq)
    t = np.arange(n_times) / sfreq

    rng = np.random.default_rng(seed)
    for label in CLASSES:
        for i in range(n_per_class):
            s = int(rng.integers(0, 1_000_000_000))
            X = _synth_epoch(label, sfreq=sfreq, n_times=n_times, n_channels=n_channels, seed=s)
            df = pd.DataFrame(X, columns=[f"ch{k}" for k in range(n_channels)])
            df.insert(0, "timestamp_sec", t)
            fname = f"epoch_MI_{label}_{i+1:02d}.csv"
            df.to_csv(out_root / label / fname, index=False)

    print(f"[OK] Generated dataset: {out_root.resolve()}")
    return out_root

if __name__ == "__main__":
    generate_dataset()
