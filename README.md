# Dynamical System Discovery

Project scaffold for learning SINDy with the Lorenz system before extending to PDE identification.

## Current Status

- Lorenz system simulation is implemented.
- Damped pendulum simulation is implemented.
- Numerical derivative estimation is implemented with finite differences.
- Candidate library construction is implemented for polynomial terms, with optional custom terms such as `sin` and `cos`.
- Sparse regression is implemented with sequential thresholded least squares (classic SINDy baseline).
- The exploration notebook covers simulation, CSV export, plotting, and derivative inspection for both systems.
- End-to-end equation discovery orchestration is not implemented yet.
