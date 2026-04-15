"""Sparse regression utilities for SINDy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def fit_sindy(
    library: pd.DataFrame,
    derivatives: pd.DataFrame,
    target_columns: Sequence[str],
    threshold: float = 0.1,
    max_iterations: int = 10,
) -> pd.DataFrame:
    """Fit sparse coefficients with sequential thresholded least squares."""
    theta = library.to_numpy(dtype=float)
    dxdt = derivatives[list(target_columns)].to_numpy(dtype=float)

    coefficients = np.linalg.lstsq(theta, dxdt, rcond=None)[0]

    for _ in range(max_iterations):
        updated_coefficients = coefficients.copy()
        small_mask = np.abs(updated_coefficients) < threshold
        updated_coefficients[small_mask] = 0.0

        for target_index in range(updated_coefficients.shape[1]):
            active_mask = updated_coefficients[:, target_index] != 0.0
            if not np.any(active_mask):
                continue

            updated_coefficients[active_mask, target_index] = np.linalg.lstsq(
                theta[:, active_mask],
                dxdt[:, target_index],
                rcond=None,
            )[0]

        if np.allclose(coefficients, updated_coefficients):
            coefficients = updated_coefficients
            break

        coefficients = updated_coefficients

    return pd.DataFrame(
        coefficients,
        index=library.columns,
        columns=target_columns,
    )


def predict_derivatives(
    library: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """Use fitted SINDy coefficients to predict derivatives from a library."""
    theta = library[coefficients.index].to_numpy(dtype=float)
    coefficient_matrix = coefficients.to_numpy(dtype=float)
    predictions = theta @ coefficient_matrix

    return pd.DataFrame(
        predictions,
        columns=coefficients.columns,
        index=library.index,
    )


def format_equations(
    coefficients: pd.DataFrame,
    state_names: MappingLike | None = None,
    precision: int = 4,
) -> list[str]:
    """Convert coefficient tables into readable symbolic equations."""
    equations: list[str] = []

    for target_column in coefficients.columns:
        terms: list[str] = []
        for feature_name, coefficient in coefficients[target_column].items():
            if np.isclose(coefficient, 0.0):
                continue
            terms.append(f"{coefficient:.{precision}f}*{feature_name}")

        left_hand_side = state_names[target_column] if state_names else target_column
        right_hand_side = " + ".join(terms) if terms else "0"
        equations.append(f"{left_hand_side} = {right_hand_side}")

    return equations


MappingLike = dict[str, str]
