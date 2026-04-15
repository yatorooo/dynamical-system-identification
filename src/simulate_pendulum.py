"""Damped pendulum simulation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def damped_pendulum_system(
    t: float,
    state: np.ndarray,
    damping: float = 0.2,
    gravity: float = 9.81,
    length: float = 1.0,
) -> list[float]:
    """Return the damped pendulum time derivatives for a given state."""
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(damping * omega) - (gravity / length) * np.sin(theta)
    return [dtheta_dt, domega_dt]


def simulate_damped_pendulum(
    initial_state: tuple[float, float] = (1.0, 0.0),
    t_max: float = 20.0,
    dt: float = 0.01,
    damping: float = 0.2,
    gravity: float = 9.81,
    length: float = 1.0,
) -> pd.DataFrame:
    """Simulate a damped pendulum and return the trajectory as a DataFrame."""
    num_steps = int(np.floor(t_max / dt)) + 1
    time = np.linspace(0.0, dt * (num_steps - 1), num_steps)
    solution = solve_ivp(
        fun=lambda t, state: damped_pendulum_system(
            t,
            state,
            damping=damping,
            gravity=gravity,
            length=length,
        ),
        t_span=(0.0, t_max),
        y0=np.asarray(initial_state, dtype=float),
        t_eval=time,
        method="RK45",
    )

    return pd.DataFrame(
        {
            "t": solution.t,
            "theta": solution.y[0],
            "omega": solution.y[1],
        }
    )


def save_pendulum_data(data: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist damped pendulum trajectory data to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path
