"""Brownian motion simulation and market coefficient utilities."""
import numpy as np
from typing import Optional, Tuple


def simulate_brownian_paths(
    n_paths: int,
    N: int,
    d: int,
    dt: float,
    rng: Optional[np.random.Generator] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate d-dimensional Brownian motion paths on a uniform grid.

    Args:
        n_paths: Number of Monte Carlo paths.
        N: Number of time steps.
        d: Dimension of Brownian motion.
        dt: Time step size.
        rng: Optional numpy Generator. If None, created from seed.
        seed: Random seed (used when rng is None).
    Returns:
        B: Brownian paths, shape (n_paths, N+1, d).
        dB: Increments, shape (n_paths, N, d).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    dB = rng.standard_normal((n_paths, N, d)) * np.sqrt(dt)
    B = np.zeros((n_paths, N + 1, d))
    B[:, 1:, :] = np.cumsum(dB, axis=1)
    return B, dB


def simulate_brownian_paths_antithetic(
    n_paths: int,
    N: int,
    d: int,
    dt: float,
    rng: Optional[np.random.Generator] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate d-dimensional Brownian motion with antithetic variates.

    Args:
        n_paths: Number of Monte Carlo paths (must be even).
        N: Number of time steps.
        d: Dimension of Brownian motion.
        dt: Time step size.
        rng: Optional numpy Generator. If None, created from seed.
        seed: Random seed (used when rng is None).
    Returns:
        B: Brownian paths, shape (n_paths, N+1, d).
        dB: Increments, shape (n_paths, N, d).
    """
    assert n_paths % 2 == 0, "n_paths must be even for antithetic variates"
    if rng is None:
        rng = np.random.default_rng(seed)
    half = n_paths // 2
    dB_half = rng.standard_normal((half, N, d)) * np.sqrt(dt)
    dB = np.concatenate([dB_half, -dB_half], axis=0)
    B = np.zeros((n_paths, N + 1, d))
    B[:, 1:, :] = np.cumsum(dB, axis=1)
    return B, dB


def constant_market_coefficients(
    n_paths: int, N: int, d: int,
    b_const: np.ndarray, sigma_const: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate constant market drift, volatility, and market price of risk.

    Args:
        n_paths: Number of Monte Carlo paths.
        N: Number of time steps.
        d: Dimension.
        b_const: Constant drift vector, shape (d,).
        sigma_const: Constant volatility matrix, shape (d, d).
    Returns:
        b_paths: drift paths, shape (n_paths, N, d).
        sigma_paths: volatility paths, shape (n_paths, N, d, d).
        theta_paths: market price of risk theta=sigma^{-1}b, shape (n_paths, N, d).
    """
    b_const = np.asarray(b_const, dtype=float)
    sigma_const = np.asarray(sigma_const, dtype=float)
    sigma_inv = np.linalg.inv(sigma_const)
    theta_const = sigma_inv @ b_const  # shape (d,)

    b_paths = np.broadcast_to(b_const, (n_paths, N, d)).copy()
    sigma_paths = np.broadcast_to(sigma_const, (n_paths, N, d, d)).copy()
    theta_paths = np.broadcast_to(theta_const, (n_paths, N, d)).copy()
    return b_paths, sigma_paths, theta_paths


def compute_martingale_density(
    theta_paths: np.ndarray, dB: np.ndarray, dt: float
) -> np.ndarray:
    """Compute martingale density tildeD_t = E(-theta . B)_t as stochastic exponential.

    Uses log-domain accumulation for numerical stability:
    tildeD_{t_{n+1}} = tildeD_{t_n} * exp(-theta_n . dB_n - 0.5*|theta_n|^2*dt)

    Args:
        theta_paths: Market price of risk, shape (n_paths, N, d).
        dB: Brownian increments, shape (n_paths, N, d).
        dt: Time step.
    Returns:
        tildeD: Martingale density paths, shape (n_paths, N+1).
    """
    n_paths, N, d = theta_paths.shape
    log_tilde_D = np.zeros((n_paths, N + 1))
    for n in range(N):
        # -theta . dB - 0.5 * |theta|^2 * dt
        increment = (
            -np.sum(theta_paths[:, n, :] * dB[:, n, :], axis=1)
            - 0.5 * np.sum(theta_paths[:, n, :] ** 2, axis=1) * dt
        )
        log_tilde_D[:, n + 1] = log_tilde_D[:, n] + increment
    return np.exp(log_tilde_D)
