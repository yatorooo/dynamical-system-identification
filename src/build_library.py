"""Candidate feature library construction for SINDy."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd


CustomLibrarySpec = Mapping[str, Callable[[np.ndarray], np.ndarray]]


def build_polynomial_library(
    data: pd.DataFrame,
    state_columns: Sequence[str],
    max_degree: int = 2,
    include_constant: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a polynomial candidate library from selected state columns."""
    if max_degree < 1:
        raise ValueError("max_degree must be at least 1.")
    if not state_columns:
        raise ValueError("state_columns must contain at least one column name.")

    missing_columns = [column for column in state_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing state columns: {missing_columns}")

    library_columns: dict[str, np.ndarray] = {}
    num_rows = len(data)

    if include_constant:
        library_columns["1"] = np.ones(num_rows, dtype=float)

    state_arrays = {
        column: data[column].to_numpy(dtype=float) for column in state_columns
    }

    for degree in range(1, max_degree + 1):
        for term_columns in combinations_with_replacement(state_columns, degree):
            term_name = _format_polynomial_term(term_columns)
            term_values = np.ones(num_rows, dtype=float)
            for column in term_columns:
                term_values *= state_arrays[column]
            library_columns[term_name] = term_values

    library = pd.DataFrame(library_columns, index=data.index)
    feature_names = list(library.columns)
    return library, feature_names


def add_custom_terms(
    library: pd.DataFrame,
    data: pd.DataFrame,
    custom_terms: Mapping[str, CustomLibrarySpec],
) -> tuple[pd.DataFrame, list[str]]:
    """Append unary custom-function terms evaluated on selected variables."""
    augmented_library = library.copy()

    for column, transforms in custom_terms.items():
        if column not in data.columns:
            raise ValueError(f"Missing state column for custom terms: {column}")

        values = data[column].to_numpy(dtype=float)
        for term_name, transform in transforms.items():
            feature_name = f"{term_name}({column})"
            augmented_library[feature_name] = transform(values)

    return augmented_library, list(augmented_library.columns)


def build_library(
    data: pd.DataFrame,
    state_columns: Sequence[str],
    max_degree: int = 2,
    include_constant: bool = True,
    custom_terms: Mapping[str, CustomLibrarySpec] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a candidate library with polynomial and optional custom terms."""
    library, feature_names = build_polynomial_library(
        data=data,
        state_columns=state_columns,
        max_degree=max_degree,
        include_constant=include_constant,
    )

    if custom_terms:
        library, feature_names = add_custom_terms(
            library=library,
            data=data,
            custom_terms=custom_terms,
        )

    return library, feature_names


def _format_polynomial_term(term_columns: Sequence[str]) -> str:
    """Convert a tuple of variable names into a readable polynomial term name."""
    exponents: dict[str, int] = {}
    for column in term_columns:
        exponents[column] = exponents.get(column, 0) + 1

    formatted_parts = []
    for column in term_columns:
        if column in exponents:
            power = exponents.pop(column)
            if power == 1:
                formatted_parts.append(column)
            else:
                formatted_parts.append(f"{column}^{power}")

    return " ".join(formatted_parts).replace(" ", "*")
