"""Outer robust optimization: multiplier bisection and feedback control formulas."""
import numpy as np
from typing import Callable, Tuple
from scipy.optimize import brentq


def compute_budget(
    c_paths: np.ndarray,
    xi: np.ndarray,
    tildeD: np.ndarray,
    dt: float,
) -> float:
    """Compute budget constraint X_0 = E[tildeD_T * xi + int_0^T tildeD_t * c_t dt].

    Args:
        c_paths: Consumption paths, shape (n_paths, N).
        xi: Terminal wealth, shape (n_paths,).
        tildeD: Martingale density, shape (n_paths, N+1).
        dt: Time step.
    Returns:
        Budget estimate (scalar).
    """
    terminal_term = np.mean(tildeD[:, -1] * xi)
    running_term = np.mean(np.sum(tildeD[:, :-1] * c_paths, axis=1) * dt)
    return float(terminal_term + running_term)


def compute_optimal_controls_feedback(
    barY_0: float,
    D_star_paths: np.ndarray,
    tildeD: np.ndarray,
    v: float,
    gamma: float,
    q: float,
    a: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal controls from paper's feedback formulas.

    c_t* = I_1( -(v/gamma) * (barY_0 * D_t*)^{-q} * tildeD_t )
    xi*  = I_2( -(v/gamma) * (barY_0 * D_T*)^{-q} * tildeD_T )

    For power utility u(c)=c^a/a: I_1(y) = I_2(y) = y^{1/(a-1)}.

    Args:
        barY_0: Transformed BSDE initial value (scalar).
        D_star_paths: Minimizing density, shape (n_paths, N+1).
        tildeD: Martingale density, shape (n_paths, N+1).
        v: Lagrange multiplier (v < 0).
        gamma: Ambiguity aversion.
        q: Tsallis parameter.
        a: Power utility parameter.
    Returns:
        c_star: Optimal consumption, shape (n_paths, N).
        xi_star: Optimal terminal wealth, shape (n_paths,).
    """
    from src.utilities import inverse_marginal_utility

    factor = -v / gamma  # positive since v < 0

    # Consumption: factor * (barY_0 * D_t*)^{-q} * tildeD_t
    # Computed in log-space to avoid float64 overflow when the base is tiny.
    D_mid = D_star_paths[:, :-1]  # (n_paths, N)
    tildeD_mid = tildeD[:, :-1]   # (n_paths, N)
    log_arg_c = (
        np.log(max(factor, 1e-300))
        - q * np.log(np.clip(barY_0 * D_mid, 1e-300, None))
        + np.log(np.clip(tildeD_mid, 1e-300, None))
    )
    arg_c = np.exp(np.clip(log_arg_c, -500.0, 500.0))
    arg_c = np.clip(arg_c, 1e-300, None)
    c_star = inverse_marginal_utility(arg_c, a)
    c_star = np.clip(c_star, 1e-8, None)

    # Terminal wealth: factor * (barY_0 * D_T*)^{-q} * tildeD_T
    log_arg_xi = (
        np.log(max(factor, 1e-300))
        - q * np.log(np.clip(barY_0 * D_star_paths[:, -1], 1e-300, None))
        + np.log(np.clip(tildeD[:, -1], 1e-300, None))
    )
    arg_xi = np.exp(np.clip(log_arg_xi, -500.0, 500.0))
    arg_xi = np.clip(arg_xi, 1e-300, None)
    xi_star = inverse_marginal_utility(arg_xi, a)
    xi_star = np.clip(xi_star, 1e-8, None)

    return c_star, xi_star


def solve_multiplier(
    budget_fn: Callable[[float], float],
    x: float,
    v_lo: float = -200.0,
    v_hi: float = -1e-6,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """Find multiplier v < 0 satisfying budget constraint via bisection (Brent).

    Solves: budget_fn(v) = x.

    Args:
        budget_fn: Function v -> budget estimate X_0(v).
        x: Target initial wealth.
        v_lo: Lower bound for v (more negative = larger budget).
        v_hi: Upper bound for v (less negative = smaller budget).
        tol: Tolerance for root finding.
        max_iter: Max iterations.
    Returns:
        v_star: Multiplier satisfying budget_fn(v_star) ≈ x.
    """
    def gap_fn(v: float) -> float:
        return budget_fn(v) - x

    try:
        v_star = brentq(gap_fn, v_lo, v_hi, xtol=tol, maxiter=max_iter)
    except ValueError:
        # brentq requires sign change; fall back to midpoint
        v_star = (v_lo + v_hi) / 2.0
    return float(v_star)
