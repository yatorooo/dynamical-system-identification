"""Lorenz system simulation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def lorenz_system(
    t: float,
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> list[float]:
    """Return the Lorenz system time derivatives for a given state."""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


def simulate_lorenz(
    initial_state: tuple[float, float, float] = (1.0, 1.0, 1.0),
    t_max: float = 25.0,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> pd.DataFrame:
    """Simulate the Lorenz system and return the trajectory as a DataFrame."""
    num_steps = int(np.floor(t_max / dt)) + 1
    time = np.linspace(0.0, dt * (num_steps - 1), num_steps)
    solution = solve_ivp(
        fun=lambda t, state: lorenz_system(
            t, state, sigma=sigma, rho=rho, beta=beta
        ),
        t_span=(0.0, t_max),
        y0=np.asarray(initial_state, dtype=float),
        t_eval=time,
        method="RK45",
    )

    return pd.DataFrame(
        {
            "t": solution.t,
            "x": solution.y[0],
            "y": solution.y[1],
            "z": solution.y[2],
        }
    )


def save_lorenz_data(data: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist simulated Lorenz trajectory data to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path
