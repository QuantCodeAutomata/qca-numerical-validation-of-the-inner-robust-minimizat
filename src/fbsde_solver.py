"""Coupled FBSDE solver for the full robust consumption-investment problem.

Implements the fixed-point iteration between:
  - Transformed BSDE for (barY, barZ)
  - Feedback control updates for (c*, xi*)
  - Multiplier bisection for v

Custom — Context7 found no library equivalent for this Tsallis robust utility FBSDE.
"""
import numpy as np
from typing import Dict


def solve_coupled_fbsde(
    B: np.ndarray,
    dB: np.ndarray,
    dt: float,
    T: float,
    gamma: float,
    q: float,
    a: float,
    x: float,
    tildeD: np.ndarray,
    n_iter: int = 15,
    tol: float = 5e-4,
    damping: float = 0.5,
    v_init: float = -1.0,
    degree: int = 3,
    verbose: bool = False,
    no_consumption: bool = False,
) -> Dict:
    """Solve the coupled FBSDE system by fixed-point iteration.

    Algorithm:
        1. Initialize controls (c, xi) and multiplier v.
        2. Solve transformed BSDE -> (barY, barZ).
        3. Compute D* = stochastic_exponential(barZ/barY).
        4. Update controls from feedback formulas.
        5. Update v via bisection to satisfy budget X_0 = x.
        6. Repeat until barY_0 converges.

    Args:
        B: Brownian paths, shape (n_paths, N+1, d).
        dB: Brownian increments, shape (n_paths, N, d).
        dt: Time step.
        T: Time horizon.
        gamma: Ambiguity aversion.
        q: Tsallis parameter (q > 1).
        a: Power utility parameter (0 < a < 1).
        x: Initial wealth target.
        tildeD: Martingale density, shape (n_paths, N+1).
        n_iter: Max fixed-point iterations.
        tol: Convergence tolerance (relative change in barY_0).
        damping: Control update damping factor (0 < damping <= 1).
        v_init: Initial Lagrange multiplier guess (must be < 0).
        degree: Polynomial degree for BSDE regression.
        verbose: Print iteration info.
        no_consumption: If True, enforce c_t ≡ 0 throughout (terminal-wealth-only case).
    Returns:
        Dict with keys: barY, barZ, Y, Z, D_star, eta_star, c_star, xi_star,
        v, barY_0, budget, budget_gap, conv_history, n_iter_done.
    """
    from src.bsde import solve_transformed_bsde, recover_original_bsde
    from src.measures import compute_girsanov_kernel, stochastic_exponential
    from src.optimizer_outer import (
        compute_optimal_controls_feedback,
        compute_budget,
        solve_multiplier,
    )
    from src.utilities import power_utility, transformed_terminal_utility

    n_paths, N_plus1, d = B.shape
    N = N_plus1 - 1

    # Initialize controls: feasible admissible pair
    c_curr = np.full((n_paths, N), max(x / T * 0.5, 1e-4))
    xi_curr = np.full(n_paths, max(x * 0.5, 1e-4))
    v_curr = v_init
    barY_prev = None
    conv_history = []

    for it in range(n_iter):
        # No-consumption special case: fix c ≡ 0 throughout iteration
        if no_consumption:
            c_curr = np.zeros_like(c_curr)

        # Step 1: Compute utility flows for current controls
        U_paths = power_utility(c_curr, a)
        terminal_cond = transformed_terminal_utility(xi_curr, gamma, a, q)

        # Step 2: Solve transformed BSDE
        barY, barZ = solve_transformed_bsde(
            B, dB, dt, gamma, q, U_paths, terminal_cond, degree=degree
        )

        # Step 3: Compute minimizing density D*
        eta_star = compute_girsanov_kernel(barZ, barY)
        D_star = stochastic_exponential(eta_star, dB, dt)

        barY_0 = float(np.mean(barY[:, 0]))

        # Check convergence
        if barY_prev is not None:
            delta = abs(barY_0 - barY_prev) / (abs(barY_prev) + 1e-10)
            conv_history.append(delta)
            if verbose:
                budget_now = compute_budget(c_curr, xi_curr, tildeD, dt)
                print(
                    f"  Iter {it}: barY_0={barY_0:.6f}, delta={delta:.2e}, "
                    f"v={v_curr:.4f}, budget_gap={abs(budget_now - x):.4f}"
                )
            if delta < tol:
                break
        barY_prev = barY_0

        # Step 4: Update multiplier to satisfy budget constraint
        def _budget_fn(v: float) -> float:
            c_new, xi_new = compute_optimal_controls_feedback(
                barY_0, D_star, tildeD, v, gamma, q, a
            )
            return compute_budget(c_new, xi_new, tildeD, dt)

        try:
            v_new = solve_multiplier(_budget_fn, x)
        except Exception:
            v_new = v_curr
        v_curr = v_new

        # Step 5: Update controls with damping for stability
        if no_consumption:
            # Terminal-wealth-only: keep c ≡ 0, only update xi
            _, xi_new = compute_optimal_controls_feedback(
                barY_0, D_star, tildeD, v_curr, gamma, q, a
            )
            c_curr = np.zeros((n_paths, N))
            xi_curr = damping * xi_new + (1.0 - damping) * xi_curr
            xi_curr = np.clip(xi_curr, 1e-8, None)
        else:
            c_new, xi_new = compute_optimal_controls_feedback(
                barY_0, D_star, tildeD, v_curr, gamma, q, a
            )
            c_curr = damping * c_new + (1.0 - damping) * c_curr
            xi_curr = damping * xi_new + (1.0 - damping) * xi_curr
            c_curr = np.clip(c_curr, 1e-8, None)
            xi_curr = np.clip(xi_curr, 1e-8, None)

    # Recover original BSDE variables
    Y, Z = recover_original_bsde(barY, barZ, gamma, q)

    # Final budget check
    budget = compute_budget(c_curr, xi_curr, tildeD, dt)

    return {
        "barY": barY,
        "barZ": barZ,
        "Y": Y,
        "Z": Z,
        "D_star": D_star,
        "eta_star": eta_star,
        "c_star": c_curr,
        "xi_star": xi_curr,
        "v": v_curr,
        "barY_0": float(np.mean(barY[:, 0])),
        "budget": budget,
        "budget_gap": abs(budget - x),
        "conv_history": conv_history,
        "n_iter_done": it + 1,
    }
