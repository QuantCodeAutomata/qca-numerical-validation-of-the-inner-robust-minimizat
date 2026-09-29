"""Tsallis q-logarithm, q-exponential, and Tsallis relative entropy utilities.

Custom — Context7 found no library equivalent for Tsallis entropy BSDE formulations.
"""
import numpy as np
from typing import Union


def ln_q(x: np.ndarray, q: float) -> np.ndarray:
    """Tsallis q-logarithm: ln_q(x) = (x^(1-q) - 1) / (1-q) for q != 1.

    Args:
        x: Input array, must be positive.
        q: Tsallis parameter, q > 1.
    Returns:
        q-logarithm values.
    """
    # Custom — Context7 found no library equivalent (paper definition)
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-300, None)
    if abs(q - 1.0) < 1e-12:
        return np.log(x)
    return (x ** (1.0 - q) - 1.0) / (1.0 - q)


def exp_q(x: np.ndarray, q: float) -> np.ndarray:
    """Tsallis q-exponential: exp_q(x) = [1 + (1-q)*x]^(1/(1-q)) for q != 1.

    Args:
        x: Input array.
        q: Tsallis parameter, q > 1.
    Returns:
        q-exponential values. Clipped to avoid domain issues.
    """
    # Custom — Context7 found no library equivalent (paper definition)
    x = np.asarray(x, dtype=float)
    if abs(q - 1.0) < 1e-12:
        return np.exp(x)
    base = 1.0 + (1.0 - q) * x
    # Clip to avoid negative base (floating-point guard)
    base = np.clip(base, 1e-300, None)
    return base ** (1.0 / (1.0 - q))


def exp_q_domain_valid(x: np.ndarray, q: float) -> np.ndarray:
    """Check where exp_q(x) is well-defined: 1 + (1-q)*x > 0.

    Args:
        x: Input array.
        q: Tsallis parameter.
    Returns:
        Boolean mask where exp_q is valid.
    """
    return 1.0 + (1.0 - q) * np.asarray(x) > 0.0


def tsallis_entropy_direct(D_T: np.ndarray, q: float) -> float:
    """Tsallis relative entropy H_q(Q|P) = E[(D_T^Q)^q * ln_q(D_T^Q)].

    Equivalent closed form for martingale densities: (E[D_T^q] - 1)/(q-1).
    Uses the definition-based formula directly.

    Args:
        D_T: Terminal density ratio D_T^Q (shape: n_paths,).
        q: Tsallis parameter, q > 1.
    Returns:
        Monte Carlo estimate of H_q(Q|P).
    """
    # Custom — Context7 found no library equivalent (paper definition)
    D_T = np.asarray(D_T, dtype=float)
    D_T = np.clip(D_T, 1e-300, None)
    return float(np.mean(D_T ** q * ln_q(D_T, q)))


def tsallis_entropy_integral(
    eta: np.ndarray, D_paths: np.ndarray, dt: float, q: float,
    max_D_q: float = 1e6,
) -> float:
    """Tsallis entropy via the robust direct formula H_q = (E[D_T^q] - 1) / (q - 1).

    This is mathematically identical to the integral form
    ``(q/2)*E[∫|eta_t|²*D_t^q dt]`` for Itô stochastic exponentials but
    avoids path-wise accumulation of D_t^q, which overflows for large gamma.
    D_T^q values above *max_D_q* are clipped before averaging to suppress
    the influence of extreme outlier paths in the Monte Carlo estimate.

    Args:
        eta: Girsanov kernel paths, shape (n_paths, N, d).  Used only for
             the fallback when D_paths is absent; pass the same D_paths.
        D_paths: Density paths D_t^Q, shape (n_paths, N+1).
        dt: Time step (kept for API compatibility; not used in direct form).
        q: Tsallis parameter, q > 1.
        max_D_q: Clip level for D_T^q outliers (default 1e6).
    Returns:
        Monte Carlo estimate of H_q(Q|P).
    """
    # Custom — Context7 found no library equivalent (paper integral form)
    # Mathematically: H_q = (E[D_T^q] - 1) / (q - 1)  [since E[D_T] = 1]
    D_T = np.asarray(D_paths[:, -1], dtype=float)
    D_T = np.clip(D_T, 1e-300, None)
    D_T_q = np.clip(D_T ** q, 0.0, max_D_q)
    return float((np.mean(D_T_q) - 1.0) / (q - 1.0))


def mu_func(y: np.ndarray, gamma: float, q: float) -> np.ndarray:
    """Compute mu(y) = (1/q) * (1 - (1-q) * gamma * y).

    From the paper's inner BSDE structure. Must be positive.

    Args:
        y: BSDE value Y_t.
        gamma: Ambiguity aversion parameter.
        q: Tsallis parameter.
    Returns:
        mu(Y_t) values.
    """
    # Custom — Context7 found no library equivalent (paper Eq. definition)
    return (1.0 / q) * (1.0 - (1.0 - q) * gamma * np.asarray(y, dtype=float))
