"""Transform consistency checks between (Y, Z) and (checkY, checkZ)."""
import numpy as np
from typing import Dict


def check_transform_consistency(
    Y: np.ndarray,
    Z: np.ndarray,
    checkY: np.ndarray,
    checkZ: np.ndarray,
    gamma: float,
    q: float,
) -> Dict[str, float]:
    """Verify transform relations between (Y, Z) and (checkY, checkZ).

    Paper transform relations:
        checkY_t = exp_q(-gamma * Y_t)
        checkZ_t = -gamma * checkY_t^q * Z_t

    Args:
        Y: Original BSDE solution, shape (n_paths, N+1).
        Z: Original BSDE control, shape (n_paths, N, d).
        checkY: Transformed BSDE solution, shape (n_paths, N+1).
        checkZ: Transformed BSDE control, shape (n_paths, N, d).
        gamma: Ambiguity aversion.
        q: Tsallis parameter.
    Returns:
        Dict of consistency metrics (max and mean absolute errors).
    """
    from src.tsallis import exp_q

    # Check checkY = exp_q(-gamma * Y)
    checkY_from_Y = exp_q(-gamma * Y, q)
    checkY_err = np.abs(checkY - checkY_from_Y)

    # Check checkZ = -gamma * checkY^q * Z
    checkY_q = checkY[:, :-1] ** q  # (n_paths, N)
    checkZ_from_Z = -gamma * checkY_q[:, :, np.newaxis] * Z
    checkZ_err = np.abs(checkZ - checkZ_from_Z)

    return {
        "checkY_max_abs_err": float(np.max(checkY_err)),
        "checkY_mean_abs_err": float(np.mean(checkY_err)),
        "checkZ_max_abs_err": float(np.max(checkZ_err)),
        "checkZ_mean_abs_err": float(np.mean(checkZ_err)),
    }


def verify_mu_positivity(
    Y: np.ndarray, gamma: float, q: float, tol: float = 1e-8
) -> Dict[str, float]:
    """Check mu(Y_t) = (1/q)(1 - (1-q)*gamma*Y_t) > 0 pathwise.

    Args:
        Y: BSDE solution, shape (n_paths, N+1).
        gamma: Ambiguity aversion.
        q: Tsallis parameter.
        tol: Tolerance for positivity check.
    Returns:
        Dict with fraction positive and min value.
    """
    from src.tsallis import mu_func
    mu_vals = mu_func(Y, gamma, q)
    frac_positive = float(np.mean(mu_vals > tol))
    min_mu = float(np.min(mu_vals))
    return {"mu_fraction_positive": frac_positive, "mu_min_value": min_mu}
