import numpy as np

def multiclass_brier(y_true, probas, C=4):
    # y_true: (N,), probas: (N,C)
    onehot = np.eye(C)[y_true]
    return np.mean(np.sum((onehot - probas)**2, axis=1))


def ece_score(y_true, probas, n_bins=15):
    conf = probas.max(axis=1)
    pred = probas.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += np.abs(acc[mask].mean() - conf[mask].mean()) * (mask.mean())
    return float(ece)
