"""Diagnostics for equality gaps, minimizer tests, and perturbation analysis."""
import numpy as np
from typing import Dict, List


def compute_equality_gap(Y_0: float, F_Q_star: float) -> Dict[str, float]:
    """Compute absolute and relative gap between BSDE Y_0 and robust objective F(Q*).

    Main validation metric for exp_1: gap should be near zero.

    Args:
        Y_0: BSDE initial value Y_0 (scalar).
        F_Q_star: Robust inner objective at Q* (scalar).
    Returns:
        Dict with abs_gap, rel_gap, Y_0, F_Q_star.
    """
    abs_gap = abs(F_Q_star - Y_0)
    rel_gap = abs_gap / (abs(Y_0) + 1e-12)
    return {"abs_gap": abs_gap, "rel_gap": rel_gap, "Y_0": Y_0, "F_Q_star": F_Q_star}


def perturbation_test(
    checkY: np.ndarray,
    checkZ: np.ndarray,
    dB: np.ndarray,
    dt: float,
    h_xi: np.ndarray,
    u_c: np.ndarray,
    gamma: float,
    q: float,
    delta_eta: np.ndarray,
    lambdas: List[float],
) -> Dict[str, np.ndarray]:
    """Test that perturbed Girsanov kernels do not improve the inner objective.

    For each lambda, computes:
        eta^lambda = eta^{Q*} + lambda * delta_eta
        D^lambda, H_q^lambda, F^lambda
    and checks F^lambda >= F^* (Q* is a minimizer).

    Args:
        checkY: Transformed BSDE solution, shape (n_paths, N+1).
        checkZ: Transformed BSDE control, shape (n_paths, N, d).
        dB: Brownian increments, shape (n_paths, N, d).
        dt: Time step.
        h_xi: h(xi) values, shape (n_paths,).
        u_c: u(c_t) values, shape (n_paths, N).
        gamma: Ambiguity aversion.
        q: Tsallis parameter.
        delta_eta: Perturbation direction, shape (n_paths, N, d).
        lambdas: List of perturbation magnitudes.
    Returns:
        Dict with lambda values, objectives, F_star, and is_minimizer mask.
    """
    from src.measures import (
        compute_girsanov_kernel,
        stochastic_exponential,
        compute_inner_robust_objective,
    )
    from src.tsallis import tsallis_entropy_integral

    # Baseline Q* kernel and objective
    eta_star = compute_girsanov_kernel(checkZ, checkY)
    D_star = stochastic_exponential(eta_star, dB, dt)
    H_q_star = tsallis_entropy_integral(eta_star, D_star, dt, q)
    F_star = compute_inner_robust_objective(D_star, h_xi, u_c, dt, q, H_q_star, gamma)

    objectives = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        eta_lam = eta_star + lam * delta_eta
        D_lam = stochastic_exponential(eta_lam, dB, dt)
        H_q_lam = tsallis_entropy_integral(eta_lam, D_lam, dt, q)
        objectives[i] = compute_inner_robust_objective(
            D_lam, h_xi, u_c, dt, q, H_q_lam, gamma
        )

    # Relative tolerance: 0.2% of |F_star| to absorb discretisation noise.
    atol = max(1e-4, 0.002 * abs(F_star))
    return {
        "lambdas": np.array(lambdas),
        "objectives": objectives,
        "F_star": F_star,
        # Perturbed objective >= baseline within tolerance (Q* is the minimizer)
        "is_minimizer": objectives >= (F_star - atol),
    }


def normalization_check(D_T: np.ndarray) -> Dict[str, float]:
    """Check D_T normalization: E[D_T] ≈ 1.

    Args:
        D_T: Terminal density, shape (n_paths,).
    Returns:
        Dict with mean_D_T, std_D_T, and norm_error.
    """
    mean_D = float(np.mean(D_T))
    std_D = float(np.std(D_T))
    return {"mean_D_T": mean_D, "std_D_T": std_D, "norm_error": abs(mean_D - 1.0)}


def compute_first_order_residuals(
    barY_0: float,
    D_star_paths: np.ndarray,
    h_prime_xi: np.ndarray,
    u_prime_c: np.ndarray,
    tildeD: np.ndarray,
    v: float,
    gamma: float,
    q: float,
) -> Dict[str, float]:
    """Compute first-order condition residuals for exp_2 and exp_3.

    Paper first-order conditions:
        Terminal:  -gamma * barY_0^q * (D_T^*)^q * h'(xi*) = v * tildeD_T
        Running:   -gamma * barY_0^q * (D_t^*)^q * u'(c_t*) = v * tildeD_t

    Args:
        barY_0: Transformed BSDE initial value (scalar).
        D_star_paths: Minimizing density, shape (n_paths, N+1).
        h_prime_xi: h'(xi*) values, shape (n_paths,).
        u_prime_c: u'(c_t*) values, shape (n_paths, N).
        tildeD: Martingale density, shape (n_paths, N+1).
        v: Lagrange multiplier (v < 0).
        gamma: Ambiguity aversion.
        q: Tsallis parameter.
    Returns:
        Dict with terminal and consumption residual statistics.
    """
    factor = -gamma * barY_0 ** q

    # Terminal residuals
    D_T_q = D_star_paths[:, -1] ** q
    R_T = factor * D_T_q * h_prime_xi - v * tildeD[:, -1]

    # Consumption residuals (all time steps)
    D_t_q = D_star_paths[:, :-1] ** q  # (n_paths, N)
    R_t = factor * D_t_q * u_prime_c - v * tildeD[:, :-1]

    return {
        "terminal_mae": float(np.mean(np.abs(R_T))),
        "terminal_rmse": float(np.sqrt(np.mean(R_T ** 2))),
        "terminal_max": float(np.max(np.abs(R_T))),
        "consumption_mae": float(np.mean(np.abs(R_t))),
        "consumption_rmse": float(np.sqrt(np.mean(R_t ** 2))),
        "consumption_max": float(np.max(np.abs(R_t))),
    }


def density_consistency_error(
    D_stochastic_exp: np.ndarray,
    D_closed_form: np.ndarray,
) -> Dict[str, float]:
    """Compare two density computations (stochastic exponential vs. closed form).

    Args:
        D_stochastic_exp: Density from stochastic exponential, shape (n_paths, N+1).
        D_closed_form: Density from closed-form formula, shape (n_paths, N+1).
    Returns:
        Dict with max and mean absolute errors.
    """
    err = np.abs(D_stochastic_exp - D_closed_form)
    return {
        "density_max_abs_err": float(np.max(err)),
        "density_mean_abs_err": float(np.mean(err)),
    }
