"""Transformed BSDE solver using least-squares Monte Carlo backward recursion.

Solves the transformed BSDE from the paper:
    checkY_T = exp_q(-gamma * zeta)
    checkY_t = E_t[checkY_{t+1}] + gamma * (checkY_t)^q * U_t * dt + checkZ_t * dB_t

Custom — Context7 found no library equivalent for this Tsallis BSDE formulation.
"""
import numpy as np
from typing import Tuple


def polynomial_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """Compute polynomial basis functions of state variable x.

    Args:
        x: State variable, shape (n_paths,).
        degree: Maximum polynomial degree.
    Returns:
        Basis matrix, shape (n_paths, degree+1).
    """
    x = np.asarray(x, dtype=float)
    return np.column_stack([x ** k for k in range(degree + 1)])


def lsmc_conditional_expectation(
    y_next: np.ndarray, state: np.ndarray, degree: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares Monte Carlo estimate of E_t[Y_{t+1}] via regression on state.

    Args:
        y_next: Values at t+1, shape (n_paths,).
        state: Current state variable B_t (1d), shape (n_paths,).
        degree: Polynomial degree for regression basis.
    Returns:
        E_y: Estimated conditional expectation, shape (n_paths,).
        coeffs: Regression coefficients.
    """
    basis = polynomial_basis(state, degree)  # (n_paths, degree+1)
    coeffs, _, _, _ = np.linalg.lstsq(basis, y_next, rcond=None)
    E_y = basis @ coeffs
    return E_y, coeffs


def lsmc_z_estimate(
    y_next: np.ndarray,
    dB: np.ndarray,
    state: np.ndarray,
    dt: float,
    degree: int = 3,
) -> np.ndarray:
    """Estimate Z_t = E_t[Y_{t+1} * dB_t] / dt via regression (Markov property).

    Args:
        y_next: Values at t+1, shape (n_paths,).
        dB: Brownian increments at step n, shape (n_paths, d).
        state: Current state B_t, shape (n_paths,).
        dt: Time step.
        degree: Polynomial degree.
    Returns:
        Z_est: Estimated Z_t, shape (n_paths, d).
    """
    basis = polynomial_basis(state, degree)  # (n_paths, degree+1)
    if dB.ndim == 1:
        dB = dB[:, np.newaxis]
    d = dB.shape[1]
    Z_est = np.zeros((len(y_next), d))
    for j in range(d):
        target = y_next * dB[:, j] / dt
        coeffs, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
        Z_est[:, j] = basis @ coeffs
    return Z_est


def solve_transformed_bsde(
    B: np.ndarray,
    dB: np.ndarray,
    dt: float,
    gamma: float,
    q: float,
    U_paths: np.ndarray,
    terminal_condition: np.ndarray,
    degree: int = 3,
    max_picard: int = 10,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the transformed BSDE by backward least-squares Monte Carlo.

    Transformed BSDE (paper notation):
        checkY_T = terminal_condition  (= exp_q(-gamma * zeta))
        checkY_t = exp_q(-gamma*zeta) - int_t^T gamma*(checkY_s)^q*U_s ds
                                       - int_t^T checkZ_s dB_s

    Backward discretisation: taking the difference from t_{n+1} to t_n
    and conditioning out the martingale increment gives
        checkY_n = E_n[checkY_{n+1}] - gamma * (checkY_n)^q * U_n * dt

    The driver is SUBTRACTED because the running-cost integral in the BSDE
    accumulates with a minus sign.  Picard iteration resolves the implicit
    nonlinearity in the driver.

    Args:
        B: Brownian paths, shape (n_paths, N+1, d).
        dB: Increments, shape (n_paths, N, d).
        dt: Time step.
        gamma: Ambiguity aversion.
        q: Tsallis parameter (q > 1).
        U_paths: u(c_t) values, shape (n_paths, N).
        terminal_condition: checkY_T values, shape (n_paths,).
        degree: Polynomial regression degree for conditional expectations.
        max_picard: Max Picard iterations per time step.
        tol: Convergence tolerance for Picard.
    Returns:
        checkY: BSDE solution, shape (n_paths, N+1).
        checkZ: BSDE control, shape (n_paths, N, d).
    """
    n_paths, N_plus1, d = B.shape
    N = N_plus1 - 1

    checkY = np.zeros((n_paths, N + 1))
    checkZ = np.zeros((n_paths, N, d))

    # Terminal condition: clipped to (0, 1] because exp_q(-gamma*h) in (0,1]
    checkY[:, N] = np.clip(terminal_condition, 1e-300, 1.0)

    # Backward recursion in time
    for n in range(N - 1, -1, -1):
        state = B[:, n, 0]  # 1D Markov state variable
        y_next = checkY[:, n + 1]

        # Step 1: Estimate E_t[checkY_{t+1}]
        E_y, _ = lsmc_conditional_expectation(y_next, state, degree)
        E_y = np.clip(E_y, 1e-300, 1.0)

        # Step 2: Estimate Z_t via E_t[Y_{t+1} dB_t] / dt
        Z_n = lsmc_z_estimate(y_next, dB[:, n, :], state, dt, degree)
        checkZ[:, n, :] = Z_n

        # Step 3: Picard iteration for checkY_n (handles nonlinear implicit driver).
        # Correct sign: checkY_n = E_y - gamma*(checkY_n)^q*U_n*dt  (driver subtracted)
        checkY_n = E_y.copy()
        for _ in range(max_picard):
            driver = gamma * np.clip(checkY_n, 1e-300, 1.0) ** q * U_paths[:, n] * dt
            checkY_n_new = E_y - driver  # minus: running cost reduces checkY going backward
            checkY_n_new = np.clip(checkY_n_new, 1e-300, 1.0)
            if np.max(np.abs(checkY_n_new - checkY_n)) < tol:
                break
            checkY_n = checkY_n_new

        checkY[:, n] = checkY_n

    return checkY, checkZ


def recover_original_bsde(
    checkY: np.ndarray,
    checkZ: np.ndarray,
    gamma: float,
    q: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover original BSDE (Y, Z) from transformed (checkY, checkZ).

    Paper transform inversions:
        Y_t = -(1/gamma) * ln_q(checkY_t)
        Z_t = -checkZ_t / (gamma * checkY_t^q)

    Args:
        checkY: Transformed BSDE solution, shape (n_paths, N+1).
        checkZ: Transformed BSDE control, shape (n_paths, N, d).
        gamma: Ambiguity aversion.
        q: Tsallis parameter.
    Returns:
        Y: Original BSDE solution, shape (n_paths, N+1).
        Z: Original BSDE control, shape (n_paths, N, d).
    """
    from src.tsallis import ln_q
    checkY = np.clip(checkY, 1e-300, None)
    Y = -(1.0 / gamma) * ln_q(checkY, q)
    # Z recovery
    checkY_q = checkY[:, :-1] ** q  # shape (n_paths, N)
    denom = np.clip(gamma * checkY_q, 1e-300, None)  # guard against near-zero
    Z = -checkZ / denom[:, :, np.newaxis]  # shape (n_paths, N, d)
    return Y, Z
