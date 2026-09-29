"""Power utility functions, derivatives, and inverse marginal utilities."""
import numpy as np
from typing import Tuple


def power_utility(c: np.ndarray, a: float) -> np.ndarray:
    """Power utility u(c) = c^a / a for 0 < a < 1.

    Args:
        c: Consumption or wealth values, must be non-negative.
        a: Risk aversion parameter, 0 < a < 1.
    Returns:
        Utility values.
    """
    c = np.asarray(c, dtype=float)
    c = np.clip(c, 1e-300, None)
    return c ** a / a


def power_utility_deriv(c: np.ndarray, a: float) -> np.ndarray:
    """Derivative of power utility: u'(c) = c^(a-1).

    Args:
        c: Consumption values, must be positive.
        a: Risk aversion parameter.
    Returns:
        Marginal utility values.
    """
    c = np.asarray(c, dtype=float)
    c = np.clip(c, 1e-300, None)
    return c ** (a - 1.0)


def inverse_marginal_utility(y: np.ndarray, a: float) -> np.ndarray:
    """Inverse marginal utility I_1(y) = y^(1/(a-1)) for y > 0.

    Since u'(c) = c^(a-1), I_1(y) = y^(1/(a-1)).
    Note: exponent 1/(a-1) < 0 for 0 < a < 1.

    Args:
        y: Marginal utility values, must be positive.
        a: Risk aversion parameter.
    Returns:
        Inverse marginal utility (optimal consumption) values.
    """
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 1e-300, None)
    exponent = 1.0 / (a - 1.0)  # negative for 0 < a < 1
    return y ** exponent


def transformed_terminal_utility(
    xi: np.ndarray, gamma: float, a: float, q: float
) -> np.ndarray:
    """Transformed terminal utility g(xi) = exp_q(-gamma * h(xi)).

    Args:
        xi: Terminal wealth values.
        gamma: Ambiguity aversion parameter.
        a: Power utility parameter.
        q: Tsallis parameter.
    Returns:
        Transformed terminal utility values.
    """
    from src.tsallis import exp_q
    xi = np.asarray(xi, dtype=float)
    xi = np.clip(xi, 1e-300, None)
    h_xi = power_utility(xi, a)
    return exp_q(-gamma * h_xi, q)


def inverse_transformed_terminal(
    z: np.ndarray, gamma: float, a: float, q: float
) -> np.ndarray:
    """Inverse of g'(xi) for recovering xi from first-order condition.

    g(xi) = exp_q(-gamma * h(xi)), g'(xi) = -gamma * h'(xi) * exp_q(-gamma*h(xi))^q.
    I_3 = (g')^{-1} via numerical root-finding.

    Args:
        z: Values at which to invert g', must be negative.
        gamma: Ambiguity aversion.
        a: Power utility parameter.
        q: Tsallis parameter.
    Returns:
        xi values satisfying g'(xi) = z.
    """
    from scipy.optimize import brentq
    from src.tsallis import exp_q
    z = np.asarray(z, dtype=float)
    result = np.zeros_like(z)
    for i, zi in enumerate(z.ravel()):
        def g_prime(xi_val: float) -> float:
            xi_val = max(float(xi_val), 1e-300)
            h_val = float(power_utility(np.array([xi_val]), a)[0])
            h_d = float(power_utility_deriv(np.array([xi_val]), a)[0])
            eq_val = float(exp_q(np.array([-gamma * h_val]), q)[0])
            return -gamma * h_d * eq_val ** q - zi
        try:
            result.ravel()[i] = brentq(g_prime, 1e-6, 1e6, maxiter=200)
        except Exception:
            result.ravel()[i] = np.nan
    return result


# Alias for consistent naming across modules
power_utility_derivative = power_utility_deriv

