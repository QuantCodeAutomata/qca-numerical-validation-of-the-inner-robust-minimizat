"""Wealth process reconstruction and portfolio derivation."""
import numpy as np
from typing import Tuple


def compute_initial_wealth(
    c_paths: np.ndarray,
    xi: np.ndarray,
    tildeD: np.ndarray,
    dt: float,
) -> float:
    """Compute initial wealth X_0 = E[tildeD_T * xi + int_0^T tildeD_t * c_t dt].

    Args:
        c_paths: Consumption paths, shape (n_paths, N).
        xi: Terminal wealth, shape (n_paths,).
        tildeD: Martingale density, shape (n_paths, N+1).
        dt: Time step.
    Returns:
        X_0 estimate (scalar).
    """
    terminal = np.mean(tildeD[:, -1] * xi)
    running = np.mean(np.sum(tildeD[:, :-1] * c_paths, axis=1) * dt)
    return float(terminal + running)


def compute_wealth_process(
    c_paths: np.ndarray,
    xi: np.ndarray,
    tildeD: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute wealth process X_t = E_t^Q[xi + int_t^T c_s ds] via risk-neutral pricing.

    Uses: X_t = (1/tildeD_t) * E[tildeD_T*xi + int_t^T tildeD_s*c_s ds | F_t]
    Approximated by cross-sectional mean (valid when tildeD is a martingale).

    Args:
        c_paths: Consumption paths, shape (n_paths, N).
        xi: Terminal wealth, shape (n_paths,).
        tildeD: Martingale density, shape (n_paths, N+1).
        dt: Time step.
    Returns:
        X: Wealth paths, shape (n_paths, N+1).
    """
    n_paths, N = c_paths.shape
    X = np.zeros((n_paths, N + 1))
    X[:, N] = xi  # terminal condition X_T = xi

    # Build cumulative discounted cash-flow from terminal, working backward
    cum_cf = tildeD[:, -1] * xi  # (n_paths,), initially tildeD_T * xi
    for n in range(N - 1, -1, -1):
        cum_cf = cum_cf + tildeD[:, n] * c_paths[:, n] * dt
        X[:, n] = cum_cf / np.clip(tildeD[:, n], 1e-10, None)
    return X


def validate_terminal_consistency(
    X: np.ndarray, xi: np.ndarray, atol: float = 1e-3
) -> float:
    """Check X_T ≈ xi pathwise.

    Args:
        X: Wealth process, shape (n_paths, N+1).
        xi: Terminal wealth, shape (n_paths,).
        atol: Absolute tolerance.
    Returns:
        Mean absolute terminal error.
    """
    return float(np.mean(np.abs(X[:, -1] - xi)))
