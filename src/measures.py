"""Probability measure construction via Girsanov kernel and stochastic exponentials.

Custom — Context7 found no library equivalent for Tsallis-entropy Girsanov kernels.
"""
import numpy as np
from typing import Tuple


def compute_girsanov_kernel(
    checkZ: np.ndarray,
    checkY: np.ndarray,
    clip_val: float = 1e-8,
) -> np.ndarray:
    """Compute Girsanov kernel eta_t^{Q*} = checkZ_t / checkY_t.

    From the paper: eta_t = -gamma*Z_t / (q*mu(Y_t)) = checkZ_t / checkY_t.

    Args:
        checkZ: Transformed BSDE control, shape (n_paths, N, d).
        checkY: Transformed BSDE solution, shape (n_paths, N+1).
        clip_val: Minimum value for checkY to avoid division by zero.
    Returns:
        eta: Girsanov kernel, shape (n_paths, N, d).
    """
    # Custom — Context7 found no library equivalent (paper Eq. eta = checkZ/checkY)
    checkY_mid = np.clip(checkY[:, :-1], clip_val, None)  # shape (n_paths, N)
    eta = checkZ / checkY_mid[:, :, np.newaxis]  # shape (n_paths, N, d)
    return eta


def stochastic_exponential(
    eta: np.ndarray,
    dB: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute stochastic exponential D_t = E(eta . B)_t in log domain.

    Discrete recursion:
        D_{t_{n+1}} = D_{t_n} * exp(eta_n . dB_n - 0.5 * |eta_n|^2 * dt)

    Args:
        eta: Girsanov kernel, shape (n_paths, N, d).
        dB: Brownian increments, shape (n_paths, N, d).
        dt: Time step.
    Returns:
        D: Density paths, shape (n_paths, N+1).
    """
    n_paths, N, d = eta.shape
    # Log-domain accumulation for numerical stability
    log_D = np.zeros((n_paths, N + 1))
    for n in range(N):
        dot_prod = np.sum(eta[:, n, :] * dB[:, n, :], axis=1)  # (n_paths,)
        eta_sq = np.sum(eta[:, n, :] ** 2, axis=1)  # (n_paths,)
        log_D[:, n + 1] = log_D[:, n] + dot_prod - 0.5 * eta_sq * dt
    return np.exp(log_D)


def compute_inner_robust_objective(
    D_paths: np.ndarray,
    h_xi: np.ndarray,
    u_c: np.ndarray,
    dt: float,
    q: float,
    entropy: float,
    gamma: float,
    max_D_q: float = 1e6,
) -> float:
    """Compute the inner robust objective F(Q*; c, xi).

    F(Q*; c, xi) = E[(D_T^{Q*})^q * h(xi)
                    + int_0^T (D_t^{Q*})^q * u(c_t) dt]
                  + (1/gamma) * H_q(Q*|P)

    Args:
        D_paths: Density paths D_t^Q*, shape (n_paths, N+1).
        h_xi: h(xi) values, shape (n_paths,).
        u_c: u(c_t) values, shape (n_paths, N).
        dt: Time step.
        q: Tsallis parameter.
        entropy: Tsallis entropy H_q(Q*|P).
        gamma: Ambiguity aversion.
        max_D_q: Clip level for D^q to suppress outlier paths (default 1e6).
    Returns:
        Scalar objective value.
    """
    # Custom — Context7 found no library equivalent (paper inner objective)
    # Clip D^q to suppress extreme outlier paths in the MC estimator.
    D_T = np.clip(D_paths[:, -1], 1e-300, None)
    D_T_q = np.clip(D_T ** q, 0.0, max_D_q)
    terminal_term = np.mean(D_T_q * h_xi)

    D_t = np.clip(D_paths[:, :-1], 1e-300, None)
    D_t_q = np.clip(D_t ** q, 0.0, max_D_q)  # (n_paths, N)
    running_term = np.mean(np.sum(D_t_q * u_c, axis=1) * dt)

    return float(terminal_term + running_term + entropy / gamma)


def compute_inner_robust_objective_at_P(
    h_xi: np.ndarray,
    u_c: np.ndarray,
    dt: float,
) -> float:
    """Compute the inner objective when Q = P (D_t = 1, eta = 0, entropy = 0).

    F(P; c, xi) = E[h(xi) + int_0^T u(c_t) dt].

    Args:
        h_xi: h(xi) values, shape (n_paths,).
        u_c: u(c_t) values, shape (n_paths, N).
        dt: Time step.
    Returns:
        Objective value under P.
    """
    terminal = np.mean(h_xi)
    running = np.mean(np.sum(u_c, axis=1) * dt)
    return float(terminal + running)


def compute_density_closed_form(
    checkY: np.ndarray,
    u_c: np.ndarray,
    dt: float,
    gamma: float,
    q: float,
) -> np.ndarray:
    """Compute D_t^* via paper's closed-form identity.

    D_t^* = (barY_t / barY_0) * exp(-gamma * int_0^t barY_s^{q-1} * u(c_s) ds)
    where barY = checkY (paper Eq. density identity).

    Args:
        checkY: Transformed BSDE solution (= barY), shape (n_paths, N+1).
        u_c: u(c_t), shape (n_paths, N).
        dt: Time step.
        gamma: Ambiguity aversion.
        q: Tsallis parameter.
    Returns:
        D_closed: Density paths via closed form, shape (n_paths, N+1).
    """
    # Custom — Context7 found no library equivalent (paper density identity)
    n_paths, N_plus1 = checkY.shape
    N = N_plus1 - 1
    barY0 = np.clip(checkY[:, 0], 1e-300, None)  # shape (n_paths,)
    D_closed = np.zeros((n_paths, N + 1))
    D_closed[:, 0] = 1.0
    log_integral = np.zeros(n_paths)
    for n in range(N):
        log_integral += -gamma * (np.clip(checkY[:, n], 1e-300, 1.0) ** (q - 1)) * u_c[:, n] * dt
        D_closed[:, n + 1] = (checkY[:, n + 1] / barY0) * np.exp(log_integral)
    return D_closed


def density_closed_form_check(
    D_stochastic: np.ndarray,
    checkY: np.ndarray,
    u_c: np.ndarray,
    dt: float,
    q: float,
    gamma: float,
) -> float:
    """Compare stochastic-exponential density with the paper's closed-form identity.

    Computes D_closed from checkY and u_c via the paper's density identity, then
    returns the mean absolute relative error against D_stochastic.

    Args:
        D_stochastic: Density from stochastic exponential, shape (n_paths, N+1).
        checkY: Transformed BSDE solution, shape (n_paths, N+1).
        u_c: u(c_t) values, shape (n_paths, N).
        dt: Time step.
        q: Tsallis parameter.
        gamma: Ambiguity aversion.
    Returns:
        Mean absolute relative error between D_stochastic and D_closed (scalar ≥ 0).
    """
    D_closed = compute_density_closed_form(checkY, u_c, dt, gamma, q)
    denom = np.clip(D_closed, 1e-10, None)
    return float(np.mean(np.abs(D_stochastic - D_closed) / denom))
