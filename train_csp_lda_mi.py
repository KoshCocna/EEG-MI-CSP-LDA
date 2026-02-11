#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSP + LDA multiclass (6-class) evaluation for MI epochs stored as CSV files.

Directory structure expected:
data/
  left/epoch_MI_left_01.csv ...
  right/...
  up/...
  down/...
  zoomIn/...
  zoomOut/...

CSV format (per file):
timestamp_sec, ch0, ch1, ..., ch23
- 4 sec @ 250 Hz (1000 rows) is assumed, but code will infer sfreq from timestamp.

Pipeline:
1) Load all epochs -> X: (n_epochs, n_channels, n_times), y: labels
2) Bandpass 8-30 Hz (Butterworth + filtfilt)
3) One-vs-Rest CSP (fit on TRAIN only) -> log-variance features
4) LDA train + evaluate
5) Confusion matrix plot + classification report

No MNE dependency (MNE import can break in some envs).


# --- Channel name normalization (handles common case variants) ---
def normalize_ch_name(name: str) -> str:
    n = name.strip()
    # Common variants to a canonical form used in this script
    aliases = {
        "Fp1": "FP1", "Fp2": "FP2", "Fpz": "FPZ",
        "Afz": "AFZ", "Af3": "AF3", "Af4": "AF4",
        "Cz": "CZ", "Fz": "FZ", "Pz": "PZ", "Oz": "OZ",
        "T7": "T7", "T8": "T8",
        "O1": "O1", "O2": "O2",
        "F3": "F3", "F4": "F4", "F7": "F7", "F8": "F8",
        "C3": "C3", "C4": "C4", "P7": "P7", "P8": "P8",
        "FC5": "FC5", "FC6": "FC6",
    }
    return aliases.get(n, n.upper())
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


CLASSES = ["left", "right", "up", "down", "zoomIn", "zoomOut"]

# --- Channel mapping (24ch) based on your electrode layout ---
# Raw CSV columns are ch0..ch23. We keep them for I/O, but internally we can rename to electrode labels.
CHANNEL_MAP = {
    0: "Fp1",
    1: "Fp2",
    2: "F3",
    3: "Fz",
    4: "Pz",
    5: "C4",
    6: "FC5",
    7: "FC6",
    8: "O1",
    9: "O2",
    10: "F7",
    11: "F8",
    12: "C3",
    13: "T7",
    14: "P7",
    15: "P8",
    16: "AFz",
    17: "Cz",
    18: "T7",   # NOTE: duplicated in the provided layout; this will be auto-disambiguated to T7_2
    19: "Fpz",
    20: "T8",
    21: "Oz",
    22: "AF3",
    23: "AF4",
}

def make_unique_channel_names(ch_map: Dict[int, str]) -> Dict[int, str]:
    seen: Dict[str, int] = {}
    out: Dict[int, str] = {}
    for idx, name in ch_map.items():
        if name in seen:
            seen[name] += 1
            out[idx] = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
            out[idx] = name
    return out

CHANNEL_MAP_UNIQUE = make_unique_channel_names(CHANNEL_MAP)

# If you want to restrict to motor-related channels for MI, set this list.
# (Recommended starting point for MI: C3/C4/Cz + FC5/FC6 + Fz)
USE_CHANNELS_BY_NAME = ["C3", "C4", "CZ", "FC5", "FC6", "FZ"]  # set to None to use all 24ch  # set to None to use all 24ch



def load_epochs(data_root: str = "data") -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """
    Returns
    -------
    X : (n_epochs, n_channels, n_times)
    y : (n_epochs,) int labels [0..n_classes-1]
    sfreq : inferred sampling frequency
    ch_names : list of channel column names
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data root not found: {root.resolve()}")

    X_list = []
    y_list = []
    ch_names = None
    ch_names_ref = None  # electrode names in fixed order
    sfreq = None

    for li, label in enumerate(CLASSES):
        folder = root / label
        files = sorted(folder.glob("*.csv"))
        if len(files) == 0:
            raise FileNotFoundError(f"No CSVs found under: {folder.resolve()}")

        for fp in files:
            df = pd.read_csv(fp)
            if "timestamp_sec" not in df.columns:
                raise ValueError(f"'timestamp_sec' column missing in {fp}")

            if ch_names is None:
                ch_names = [c for c in df.columns if c.startswith("ch")]
                if len(ch_names) == 0:
                    raise ValueError(f"No channel columns (ch0..) found in {fp}")


            # Map raw columns (ch0..ch23) -> electrode labels, and optionally select motor-related channels.
            ch_indices = [int(c[2:]) for c in ch_names]
            elec_names = [CHANNEL_MAP_UNIQUE.get(i, f"ch{i}") for i in ch_indices]

            if USE_CHANNELS_BY_NAME is not None:
                wanted = set(USE_CHANNELS_BY_NAME)
                keep_mask = [name in wanted for name in elec_names]
                if not any(keep_mask):
                    raise ValueError(
                        f"USE_CHANNELS_BY_NAME={USE_CHANNELS_BY_NAME} did not match any channels in {fp}. "
                        f"Available: {elec_names}"
                    )
                ch_names = [c for c, k in zip(ch_names, keep_mask) if k]
                elec_names = [n for n, k in zip(elec_names, keep_mask) if k]

            # Lock channel names (first file defines order)
            if ch_names_ref is None:
                ch_names_ref = elec_names
            else:
                if elec_names != ch_names_ref:
                    raise ValueError(
                        "Channel order mismatch across files.\n"
                        f"Expected: {ch_names_ref}\n"
                        f"Got:      {elec_names}\n"
                        f"File: {fp}"
                    )

            t = df["timestamp_sec"].to_numpy()
            dt = np.median(np.diff(t))
            this_sfreq = 1.0 / dt
            if sfreq is None:
                sfreq = float(this_sfreq)
            else:
                # allow tiny numerical diff
                if abs(this_sfreq - sfreq) > 1e-3:
                    raise ValueError(f"Different sfreq detected: {sfreq} vs {this_sfreq} in {fp}")

            X = df[ch_names].to_numpy().T  # (n_channels, n_times)
            X_list.append(X)
            y_list.append(li)

    X_all = np.stack(X_list, axis=0)
    y_all = np.asarray(y_list, dtype=int)
    return X_all, y_all, sfreq, ch_names


def bandpass_filter_epochs(X: np.ndarray, sfreq: float, l_freq: float = 8.0, h_freq: float = 30.0,
                           order: int = 4) -> np.ndarray:
    """
    X: (n_epochs, n_channels, n_times)
    """
    nyq = 0.5 * sfreq
    b, a = butter(order, [l_freq / nyq, h_freq / nyq], btype="bandpass")
    # filtfilt along time axis
    Xf = filtfilt(b, a, X, axis=-1)
    return Xf


def _cov_epoch(X_ep: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute normalized spatial covariance matrix for one epoch.
    X_ep: (n_channels, n_times)
    """
    C = X_ep @ X_ep.T
    tr = np.trace(C)
    if tr < eps:
        tr = eps
    return C / tr


class OvRCSP:
    """
    One-vs-Rest CSP for multiclass problems.

    For each class k:
        compute C_k = mean cov of class k
        compute C_r = mean cov of rest
        solve generalized eigenproblem:
            C_k v = λ (C_k + C_r) v
        take n_components pairs (largest + smallest eigenvalues)

    transform -> concatenate log-variance features from each OvR CSP.
    """

    def __init__(self, n_components: int = 4, reg_eps: float = 1e-6):
        self.n_components = int(n_components)
        self.reg_eps = float(reg_eps)
        self.filters_: Dict[int, np.ndarray] = {}
        self.n_channels_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OvRCSP":
        """
        X: (n_epochs, n_channels, n_times)
        y: (n_epochs,) int labels
        """
        n_epochs, n_channels, _ = X.shape
        self.n_channels_ = n_channels
        classes = np.unique(y)

        covs = np.zeros((n_epochs, n_channels, n_channels), dtype=float)
        for i in range(n_epochs):
            covs[i] = _cov_epoch(X[i])

        for k in classes:
            idx_k = (y == k)
            idx_r = ~idx_k

            Ck = covs[idx_k].mean(axis=0)
            Cr = covs[idx_r].mean(axis=0)

            # regularization (shrink toward identity)
            Ck = (1 - self.reg_eps) * Ck + self.reg_eps * np.eye(n_channels) / n_channels
            Cr = (1 - self.reg_eps) * Cr + self.reg_eps * np.eye(n_channels) / n_channels

            Csum = Ck + Cr

            # generalized eigenvalue problem: Ck v = λ Csum v
            vals, vecs = eigh(Ck, Csum)
            # sort eigenvalues descending
            order = np.argsort(vals)[::-1]
            vecs = vecs[:, order]

            m = self.n_components
            W = np.concatenate([vecs[:, :m], vecs[:, -m:]], axis=1)  # (n_channels, 2m)
            self.filters_[int(k)] = W.T  # (2m, n_channels)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Returns features: (n_epochs, n_classes * 2m)
        """
        if self.n_channels_ is None or len(self.filters_) == 0:
            raise RuntimeError("Call fit() first.")

        feats = []
        eps = 1e-12
        for k in sorted(self.filters_.keys()):
            W = self.filters_[k]  # (2m, n_channels)
            Z = np.matmul(W[None, :, :], X)  # (n_epochs, 2m, n_times)
            var = np.var(Z, axis=-1) + eps
            feats.append(np.log(var))
        return np.concatenate(feats, axis=1)


def plot_confusion(cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # annotate values
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    
    # --- Save confusion matrix as PNG ---
    output_dir = Path("result")
    output_dir.mkdir(exist_ok=True)

    png_path = output_dir / "confusion_matrix.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved to: {png_path.resolve()}")

    plt.show()


def main():
    # 1) load
    X, y, sfreq, ch_names = load_epochs("data")
    print(f"[INFO] Loaded: X={X.shape}, y={y.shape}, sfreq={sfreq:.2f} Hz, channels={len(ch_names)}")

    # 2) bandpass
    Xf = bandpass_filter_epochs(X, sfreq, 8.0, 30.0, order=4)

    # 3) split
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xf, y, test_size=0.25, random_state=42, stratify=y
    )

    # 4) CSP (fit on train only)
    csp = OvRCSP(n_components=4, reg_eps=1e-6)
    csp.fit(X_tr, y_tr)
    F_tr = csp.transform(X_tr)
    F_te = csp.transform(X_te)

    # 5) LDA
    clf = LinearDiscriminantAnalysis(solver="svd")
    clf.fit(F_tr, y_tr)
    y_hat = clf.predict(F_te)

    acc = accuracy_score(y_te, y_hat)
    print(f"\n[RESULT] Test Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_te, y_hat, target_names=CLASSES, digits=4))

    cm = confusion_matrix(y_te, y_hat, labels=list(range(len(CLASSES))))
    plot_confusion(cm, CLASSES, title=f"CSP+LDA Confusion Matrix (acc={acc*100:.1f}%)")


if __name__ == "__main__":
    main()
