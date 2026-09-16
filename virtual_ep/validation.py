"""Held-out request-level validation and predictor comparison."""

from __future__ import annotations

import numpy as np


def absolute_percentage_error(predicted, measured, floor_ms: float = 1e-3):
    predicted = np.asarray(predicted, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if predicted.shape != measured.shape:
        raise ValueError("predicted/measured shapes differ")
    return np.abs(predicted - measured) / np.maximum(np.abs(measured), floor_ms) * 100.0


def error_summary(predicted, measured):
    errors = absolute_percentage_error(predicted, measured)
    return {
        "median_ape_percent": float(np.median(errors)),
        "p90_ape_percent": float(np.percentile(errors, 90)),
        "mean_ape_percent": float(np.mean(errors)),
        "count": int(len(errors)),
    }


def request_split(request_ids, heldout_modulus: int = 5):
    request_ids = np.asarray(request_ids, dtype=np.int64)
    heldout = request_ids % heldout_modulus == 0
    return ~heldout, heldout


def linear_fit_predict(train_x, train_y, test_x):
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    train_design = np.column_stack([np.ones(len(train_x)), train_x])
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    coefficients, *_ = np.linalg.lstsq(train_design, train_y, rcond=None)
    return test_design @ coefficients, coefficients
