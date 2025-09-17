"""Common column definitions and value conventions for the data pipeline.

This module centralises the lists of feature columns, label columns and
associated mask mappings that are shared across the data processing scripts.
Keeping them in a single location avoids the previous duplication between the
window generation, cleaning and summarisation utilities and makes it easier to
reason about the integrated pipeline.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

# ---- Column lists -----------------------------------------------------------------

# Full set of features produced by the window generation step (Octet + extras).
FULL_WINDOW_FEATURE_COLUMNS: Tuple[str, ...] = (
    "UF",
    "UAS",
    "UD",
    "UAL",
    "DF",
    "DAS",
    "DD",
    "DAL",
    "rq_rel",
    "rk_rel",
    "K_EDIE",
    "K_QV",
    "QKV_relerr_q",
    "CV_v",
    "E_BRK",
    "JBR",
)

# Core 12 features kept for model training / downstream tasks.
CORE_FEATURE_COLUMNS: Tuple[str, ...] = (
    "UF",
    "UAS",
    "UD",
    "UAL",
    "DF",
    "DAS",
    "DD",
    "DAL",
    "rq_rel",
    "rk_rel",
    "CV_v",
    "E_BRK",
)

# Conflict indicators (four-class labels) accompanying the features.
LABEL_COLUMNS: Tuple[str, ...] = (
    "TTC_cls4",
    "DRAC_cls4",
    "PSD_cls4",
)

# Optional additional numeric columns that are useful for descriptive stats.
SUMMARY_EXTRA_COLUMNS: Tuple[str, ...] = (
    "TTC_weight_avg",
    "DRAC_weight_avg",
    "PSD_weight_avg",
    "TTC_exp_s1",
    "TTC_exp_s2",
    "TTC_exp_s3",
    "DRAC_exp_s1",
    "DRAC_exp_s2",
    "DRAC_exp_s3",
    "PSD_exp_s1",
    "PSD_exp_s2",
    "PSD_exp_s3",
    "TTC_time_min",
    "TTC_time_p05",
    "DRAC_time_max",
    "DRAC_time_p95",
    "PSD_time_min",
    "PSD_time_p05",
)


# ---- Mask mappings -----------------------------------------------------------------

# Feature -> mask column mapping used when checking validity flags.
FEATURE_MASK_MAP: Mapping[str, str] = {
    "UAS": "UAS_mask",
    "DAS": "DAS_mask",
    "UAL": "UAL_mask",
    "DAL": "DAL_mask",
    "UD": "UD_mask",
    "DD": "DD_mask",
    "rq_rel": "rq_rel_mask",
    "rk_rel": "rk_rel_mask",
    "K_EDIE": "K_EDIE_mask",
    "QKV_relerr_q": "QKV_relerr_q_mask",
    "CV_v": "CV_v_mask",
    "E_BRK": "E_BRK_mask",
    "JBR": "JBR_mask",
}

# Label -> mask column mapping (mask may be absent in some datasets).
LABEL_MASK_MAP: Mapping[str, str] = {
    "TTC_cls4": "TTC_cls_mask",
    "DRAC_cls4": "DRAC_cls_mask",
    "PSD_cls4": "PSD_cls_mask",
}


# ---- Value conventions --------------------------------------------------------------

ALLOWED_LABEL_VALUES: Tuple[int, ...] = (0, 1, 2, 3)


def available_columns(columns: Iterable[str], candidates: Sequence[str]) -> Tuple[str, ...]:
    """Return the subset of *candidates* that are present in *columns* preserving order."""

    present = set(columns)
    return tuple(col for col in candidates if col in present)


__all__ = [
    "FULL_WINDOW_FEATURE_COLUMNS",
    "CORE_FEATURE_COLUMNS",
    "LABEL_COLUMNS",
    "SUMMARY_EXTRA_COLUMNS",
    "FEATURE_MASK_MAP",
    "LABEL_MASK_MAP",
    "ALLOWED_LABEL_VALUES",
    "available_columns",
]

