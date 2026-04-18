"""Kalman filtering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_pendulum_theta(
    data: pd.DataFrame,
    time_column: str = "t",
    measurement_column: str = "theta",
    process_variance: float = 1e-4,
    measurement_variance: float = 2.5e-3,
) -> pd.DataFrame:
    """Filter noisy pendulum angle measurements with a simple Kalman filter."""
    time = data[time_column].to_numpy(dtype=float)
    theta_measurements = data[measurement_column].to_numpy(dtype=float)

    filtered_states = np.zeros((len(data), 2), dtype=float)

    state = np.array([theta_measurements[0], 0.0], dtype=float)
    covariance = np.eye(2, dtype=float)
    measurement_matrix = np.array([[1.0, 0.0]], dtype=float)
    measurement_covariance = np.array([[measurement_variance]], dtype=float)

    filtered_states[0] = state

    for index in range(1, len(data)):
        dt = time[index] - time[index - 1]
        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        process_covariance = process_variance * np.array(
            [
                [dt**4 / 4.0, dt**3 / 2.0],
                [dt**3 / 2.0, dt**2],
            ],
            dtype=float,
        )

        predicted_state = transition @ state
        predicted_covariance = (
            transition @ covariance @ transition.T + process_covariance
        )

        innovation = np.array([[theta_measurements[index]]], dtype=float) - (
            measurement_matrix @ predicted_state.reshape(-1, 1)
        )
        innovation_covariance = (
            measurement_matrix @ predicted_covariance @ measurement_matrix.T
            + measurement_covariance
        )
        kalman_gain = (
            predicted_covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        updated_state = predicted_state.reshape(-1, 1) + kalman_gain @ innovation
        identity = np.eye(2, dtype=float)
        covariance = (identity - kalman_gain @ measurement_matrix) @ predicted_covariance
        state = updated_state.ravel()
        filtered_states[index] = state

    return pd.DataFrame(
        {
            time_column: time,
            "theta_filtered": filtered_states[:, 0],
            "omega_filtered": filtered_states[:, 1],
        }
    )
