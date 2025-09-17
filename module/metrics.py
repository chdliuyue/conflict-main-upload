"""Utility helpers for evaluating multi-task ordinal classifiers.

The training scripts for level-0 (traditional PPO) and level-2 (neural) models
share a fairly large block of metric computation code.  The original
implementations scattered the metric logic across different files and used a
few scikit-learn helpers in a way that is only valid for the binary case (for
example ``brier_score_loss`` expects a 1-D probability vector).  Centralising
the implementation makes the logic easier to audit and, more importantly,
allows us to provide numerically stable multi-class variants of the metrics
that both training pipelines can rely on.

The helpers defined here operate on NumPy arrays and return plain Python data
structures so that they can easily be serialised to JSON/CSV or printed as a
table.  All metrics are computed per task; callers can use the provided
``format_results_table`` to pretty-print them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


Array = np.ndarray


def _negative_log_likelihood(y_true: Array, probas: Array, eps: float = 1e-12) -> float:
    """Numerically stable negative log likelihood for multi-class outputs."""

    idx = np.arange(len(y_true))
    chosen = probas[idx, y_true]
    return float(-np.log(np.clip(chosen, eps, 1.0)).mean())


def multiclass_brier_score(y_true: Array, probas: Array) -> float:
    """Generalisation of the Brier score to the multi-class setting."""

    num_classes = probas.shape[1]
    one_hot = np.eye(num_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probas - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true: Array, probas: Array, n_bins: int = 15) -> float:
    """Compute the Expected Calibration Error (ECE).

    The implementation follows the standard binning procedure where
    predictions are grouped by their maximum confidence.  Within each bin we
    compare the empirical accuracy against the mean confidence.
    """

    confidences = probas.max(axis=1)
    predictions = probas.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if not np.any(mask):
            continue
        bin_weight = mask.mean()
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


def _safe_roc_auc(y_true: Array, probas: Array) -> float:
    """Wrapper around ``roc_auc_score`` that tolerates missing classes.

    ``roc_auc_score`` raises a ``ValueError`` when only a subset of the classes
    appear in ``y_true``.  During early training epochs this is fairly common,
    so we gracefully fall back to ``NaN`` which allows callers to continue the
    pipeline while still signalling that the metric is undefined.
    """

    labels = np.arange(probas.shape[1])
    try:
        return float(
            roc_auc_score(y_true, probas, multi_class="ovr", average="macro", labels=labels)
        )
    except ValueError:
        return float("nan")


def _safe_confusion_matrix(y_true: Array, y_pred: Array, num_classes: int) -> Array:
    labels = np.arange(num_classes)
    return confusion_matrix(y_true, y_pred, labels=labels)


@dataclass
class TaskMetrics:
    task: str
    confusion_matrix: Array
    accuracy: float
    f1_score: float
    qwk: float
    ordmae: float
    nll: float
    brier: float
    auroc: float
    brdece: float

    def as_row(self) -> List[str]:
        """Return a list of formatted strings for pretty printing."""

        values = [
            self.accuracy,
            self.f1_score,
            self.qwk,
            self.ordmae,
            self.nll,
            self.brier,
            self.auroc,
            self.brdece,
        ]
        return [f"{v:>10.4f}" if np.isfinite(v) else f"{v:>10}" for v in values]


def compute_task_metrics(
    y_true: Array,
    probas: Array,
    task_name: str,
    y_pred: Array | None = None,
    ece_bins: int = 15,
) -> TaskMetrics:
    if y_pred is None:
        y_pred = probas.argmax(axis=1)

    num_classes = probas.shape[1]
    cm = _safe_confusion_matrix(y_true, y_pred, num_classes)

    return TaskMetrics(
        task=task_name,
        confusion_matrix=cm,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_score=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        qwk=float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        ordmae=float(np.mean(np.abs(y_true - y_pred))),
        nll=_negative_log_likelihood(y_true, probas),
        brier=multiclass_brier_score(y_true, probas),
        auroc=_safe_roc_auc(y_true, probas),
        brdece=expected_calibration_error(y_true, probas, n_bins=ece_bins),
    )


def evaluate_multitask_predictions(
    y_true: Array,
    probas: Array,
    task_names: Iterable[str],
    y_pred: Array | None = None,
    ece_bins: int = 15,
) -> List[TaskMetrics]:
    """Evaluate predictions for all tasks.

    Parameters
    ----------
    y_true:
        Array of shape ``[N, M]`` with integer labels.
    probas:
        Array of shape ``[N, M, C]`` containing class probabilities.
    task_names:
        Iterable of task names.  The order must match the second dimension of
        ``y_true`` and ``probas``.
    y_pred:
        Optional pre-computed hard predictions ``[N, M]``.  When omitted they
        are derived by taking ``argmax`` over ``probas``.
    ece_bins:
        Number of bins used for the calibration error.
    """

    if y_pred is None:
        y_pred = probas.argmax(axis=2)

    metrics: List[TaskMetrics] = []
    for m, name in enumerate(task_names):
        metrics.append(
            compute_task_metrics(
                y_true=y_true[:, m],
                probas=probas[:, m, :],
                task_name=name,
                y_pred=y_pred[:, m],
                ece_bins=ece_bins,
            )
        )
    return metrics


def format_results_table(results: Iterable[TaskMetrics]) -> str:
    header = (
        f"{'Task':<8} | {'Accuracy':>10} | {'F1':>10} | {'QWK':>10} | "
        f"{'OrdMAE':>10} | {'NLL':>10} | {'Brier':>10} | {'AUROC':>10} | {'BrdECE':>10}"
    )
    rows = [header]
    for r in results:
        row = f"{r.task:<8} | " + " | ".join(r.as_row())
        rows.append(row)
    return "\n".join(rows)

