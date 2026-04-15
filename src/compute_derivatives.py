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
    time = data[time_column].to_numpy(dtype=float)

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
    return data.merge(derivatives, on=time_column, how="inner")
