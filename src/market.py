"""Market model construction: admissible controls, state-price densities."""
import numpy as np
from typing import Tuple
from src.simulation import compute_martingale_density  # re-export for experiments

__all__ = ["compute_martingale_density", "make_admissible_controls",
           "make_admissible_controls_stochastic"]


def make_admissible_controls(
    B: np.ndarray,
    dB: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a simple admissible pair (c, xi) for exp_1 validation.

    Uses deterministic non-negative consumption and bounded function of B_T for xi.
    Shapes are inferred from B: n_paths=B.shape[0], N=B.shape[1]-1, T=N*dt.

    Args:
        B: Brownian paths, shape (n_paths, N+1, d).
        dB: Brownian increments, shape (n_paths, N, d) — accepted for API symmetry.
        dt: Time step; T = (B.shape[1] - 1) * dt.
    Returns:
        c_paths: Consumption paths, shape (n_paths, N).
        xi: Terminal wealth, shape (n_paths,).
    """
    n_paths, N_plus1, _d = B.shape
    N = N_plus1 - 1
    T = N * dt
    t_grid = np.linspace(0, T, N + 1)[:-1]  # shape (N,)
    # Deterministic decaying consumption: stays in (0, 1)
    c_det = 0.5 * np.exp(-0.2 * t_grid)  # shape (N,)
    c_paths = np.broadcast_to(c_det, (n_paths, N)).copy()
    # Terminal wealth: bounded function of B_T (first dimension), always positive
    B_T = B[:, -1, 0]  # shape (n_paths,)
    xi = 1.0 + 0.2 * np.tanh(B_T)  # bounded in (0.8, 1.2)
    xi = np.clip(xi, 1e-6, None)
    return c_paths, xi


def make_admissible_controls_stochastic(
    B: np.ndarray,
    dB: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct stochastic admissible pair (c, xi) depending on B_t.

    Shapes are inferred from B: n_paths=B.shape[0], N=B.shape[1]-1, T=N*dt.

    Args:
        B: Brownian paths, shape (n_paths, N+1, d).
        dB: Brownian increments, shape (n_paths, N, d) — accepted for API symmetry.
        dt: Time step.
    Returns:
        c_paths: Consumption paths, shape (n_paths, N).
        xi: Terminal wealth, shape (n_paths,).
    """
    # c_t = softplus(B_t[0]) * scale — ensures positivity
    B_mid = B[:, :-1, 0]  # (n_paths, N)
    c_paths = np.log1p(np.exp(B_mid)) * 0.3  # softplus, positive
    c_paths = np.clip(c_paths, 1e-6, None)
    B_T = B[:, -1, 0]
    xi = np.exp(0.1 * B_T) * 1.0
    xi = np.clip(xi, 1e-6, None)
    return c_paths, xi
