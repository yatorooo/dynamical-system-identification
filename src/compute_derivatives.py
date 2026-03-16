"""Numerical derivative estimation utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def estimate_derivatives(
    data: pd.DataFrame,
    state_columns: Sequence[str],
    time_column: str = "t",
    derivative_suffix: str = "_dot",
) -> pd.DataFrame:
    """Estimate time derivatives for selected state columns."""
    if time_column not in data.columns:
        raise ValueError(f"Missing time column: {time_column}")
    if not state_columns:
        raise ValueError("state_columns must contain at least one column name.")

    missing_columns = [column for column in state_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing state columns: {missing_columns}")

    time = data[time_column].to_numpy(dtype=float)
    if len(time) < 3:
        raise ValueError("At least three time samples are required.")

    time_differences = np.diff(time)
    if np.any(time_differences <= 0):
        raise ValueError("Time values must be strictly increasing.")

    derivatives = pd.DataFrame({time_column: time})
    for column in state_columns:
        values = data[column].to_numpy(dtype=float)
        derivatives[f"{column}{derivative_suffix}"] = np.gradient(values, time)

    return derivatives


def combine_state_and_derivatives(
    data: pd.DataFrame,
    derivatives: pd.DataFrame,
    time_column: str = "t",
) -> pd.DataFrame:
    """Join state data and derivative estimates on the shared time column."""
    if time_column not in data.columns or time_column not in derivatives.columns:
        raise ValueError(f"Both inputs must contain the time column {time_column!r}.")

    return data.merge(derivatives, on=time_column, how="inner")
