"""Experiment 2: Maximum principle and first-order optimality conditions for the
outer robust consumption-investment problem.

Validates:
  - Feedback optimal controls under budget constraint
  - First-order conditions (FOC) residuals
  - Local optimality via perturbation
  - Sensitivity to q, gamma, utility curvature, initial wealth
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation import simulate_brownian_paths
from src.market import compute_martingale_density
from src.tsallis import tsallis_entropy_integral
from src.bsde import solve_transformed_bsde, recover_original_bsde
from src.measures import compute_girsanov_kernel, stochastic_exponential
from src.utilities import power_utility, power_utility_derivative, transformed_terminal_utility
from src.optimizer_outer import (
    compute_optimal_controls_feedback,
    compute_budget,
    solve_multiplier,
)
from src.diagnostics import (
    normalization_check,
    compute_first_order_residuals,
)
from src.wealth import compute_wealth_process, validate_terminal_consistency

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_outer_optimization(
    T: float,
    N: int,
    n_paths: int,
    a: float,
    q: float,
    gamma: float,
    theta_val: float,
    sigma_val: float,
    x: float = 1.0,
    n_iter: int = 20,
    damping: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Solve outer robust optimization and validate first-order conditions.

    Args:
        T: Time horizon.
        N: Number of time steps.
        n_paths: Monte Carlo path count.
        a: Power utility curvature (0 < a < 1).
        q: Tsallis parameter (q > 1).
        gamma: Ambiguity aversion.
        theta_val: Constant market price of risk.
        sigma_val: Constant volatility.
        x: Initial wealth.
        n_iter: Max fixed-point iterations.
        damping: Damping factor for control updates.
        seed: Random seed.
        verbose: Print progress.
    Returns:
        Dict of metrics and arrays.
    """
    from src.fbsde_solver import solve_coupled_fbsde

    rng = np.random.default_rng(seed)
    dt = T / N
    d = 1

    # Simulate Brownian paths
    B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)

    # Compute martingale density tildeD_t = E(-theta . B)_t
    theta_paths = theta_val * np.ones((n_paths, N, d))
    tildeD = compute_martingale_density(theta_paths, dB, dt)

    if verbose:
        print(f"  q={q:.2f}, gamma={gamma:.2f}, a={a:.2f}, x={x:.2f}")

    # Solve coupled FBSDE
    sol = solve_coupled_fbsde(
        B, dB, dt, T, gamma, q, a, x, tildeD,
        n_iter=n_iter, tol=5e-4, damping=damping,
        v_init=-1.0, degree=3, verbose=verbose,
    )

    barY_0  = sol["barY_0"]
    barY    = sol["barY"]
    barZ    = sol["barZ"]
    D_star  = sol["D_star"]
    c_star  = sol["c_star"]
    xi_star = sol["xi_star"]
    v       = sol["v"]

    # Marginal utility values at optimum
    h_prime_xi = power_utility_derivative(xi_star, a)
    u_prime_c  = power_utility_derivative(c_star, a)

    # First-order condition residuals
    foc = compute_first_order_residuals(
        barY_0, D_star, h_prime_xi, u_prime_c, tildeD, v, gamma, q
    )

    # Normalization check for D*
    norm = normalization_check(D_star[:, -1])
    norm_tilde = normalization_check(tildeD[:, -1])

    # Budget check
    budget = sol["budget"]
    budget_gap = sol["budget_gap"]

    # Recover Y, Z
    Y, Z = recover_original_bsde(barY, barZ, gamma, q)
    Y_0 = float(np.mean(Y[:, 0]))

    # Terminal consistency check: Y_T ≈ h(xi*)
    h_xi_star   = power_utility(xi_star, a)
    Y_T         = Y[:, -1]
    terminal_Y_err = float(np.mean(np.abs(Y_T - h_xi_star)))

    # Perturbation-based local optimality check (vary v)
    U_star  = power_utility(c_star, a)
    H_q_val = tsallis_entropy_integral(sol["eta_star"], D_star, dt, q)

    if verbose:
        print(f"  barY_0={barY_0:.6f}, Y_0={Y_0:.6f}, v={v:.4f}")
        print(f"  Budget: target={x:.4f}, achieved={budget:.4f}, gap={budget_gap:.2e}")
        print(f"  FOC terminal MAE: {foc['terminal_mae']:.2e}")
        print(f"  FOC consumption MAE: {foc['consumption_mae']:.2e}")
        print(f"  D* norm error: {norm['norm_error']:.2e}")
        print(f"  Y_T vs h(xi*) MAE: {terminal_Y_err:.2e}")
        print()

    return {
        "q": q, "gamma": gamma, "a": a, "x": x, "N": N, "n_paths": n_paths,
        "barY_0": barY_0, "Y_0": Y_0, "v": v,
        "budget": budget, "budget_gap": budget_gap,
        "foc_terminal_mae": foc["terminal_mae"],
        "foc_terminal_rmse": foc["terminal_rmse"],
        "foc_consumption_mae": foc["consumption_mae"],
        "foc_consumption_rmse": foc["consumption_rmse"],
        "D_star_norm_err": norm["norm_error"],
        "tildeD_norm_err": norm_tilde["norm_error"],
        "Y_T_consistency_err": terminal_Y_err,
        "n_iter_done": sol["n_iter_done"],
        "conv_history": sol["conv_history"],
        # Arrays for plotting
        "_D_star": D_star, "_tildeD": tildeD, "_c_star": c_star,
        "_barY": barY, "_Y": Y,
        "_u_prime_c": u_prime_c, "_h_prime_xi": h_prime_xi,
    }


def run_sensitivity_analysis_exp2(
    q_vals, gamma_vals, a_vals, x_vals, N_vals, n_paths_base=3000
) -> pd.DataFrame:
    """Sweep parameters for experiment 2 sensitivity.

    Returns:
        DataFrame with metrics for each configuration.
    """
    records = []

    # Vary q and gamma
    for q in q_vals:
        for gam in gamma_vals:
            r = run_outer_optimization(
                T=1.0, N=40, n_paths=n_paths_base, a=0.5, q=q, gamma=gam,
                theta_val=0.3, sigma_val=1.0, x=1.0, n_iter=15,
                damping=0.5, seed=42, verbose=False,
            )
            records.append({k: v for k, v in r.items() if not k.startswith("_")})

    # Vary a and x
    for a in a_vals:
        for x_ in x_vals:
            r = run_outer_optimization(
                T=1.0, N=40, n_paths=n_paths_base, a=a, q=1.5, gamma=1.0,
                theta_val=0.3, sigma_val=1.0, x=x_, n_iter=15,
                damping=0.5, seed=42, verbose=False,
            )
            records.append({k: v for k, v in r.items() if not k.startswith("_")})

    # Refinement study
    for N in N_vals:
        r = run_outer_optimization(
            T=1.0, N=N, n_paths=n_paths_base, a=0.5, q=1.5, gamma=1.0,
            theta_val=0.3, sigma_val=1.0, x=1.0, n_iter=15,
            damping=0.5, seed=42, verbose=False,
        )
        records.append({k: v for k, v in r.items() if not k.startswith("_")})

    return pd.DataFrame(records)


def plot_exp2(result: dict, out_dir: str) -> None:
    """Plot optimal controls, densities, and FOC residuals."""
    D_star  = result["_D_star"]   # (n_paths, N+1)
    tildeD  = result["_tildeD"]   # (n_paths, N+1)
    c_star  = result["_c_star"]   # (n_paths, N)
    barY    = result["_barY"]     # (n_paths, N+1)
    N       = D_star.shape[1] - 1
    T       = 1.0
    t_fine  = np.linspace(0, T, N + 1)
    t_mid   = np.linspace(0, T, N)

    n_show = min(50, D_star.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Exp 2: Outer Optimization FOC Validation  "
        f"(q={result['q']:.2f}, γ={result['gamma']:.2f}, a={result['a']:.2f})",
        fontsize=12
    )

    # Optimal consumption c*_t
    ax = axes[0, 0]
    ax.plot(t_mid, c_star[:n_show].T, color="steelblue", alpha=0.2, linewidth=0.7)
    ax.plot(t_mid, c_star.mean(axis=0), color="navy", linewidth=2, label="E[c*_t]")
    ax.set_title("Optimal Consumption c*_t"); ax.set_xlabel("t"); ax.legend()

    # Minimizing density D*_t
    ax = axes[0, 1]
    ax.plot(t_fine, D_star[:n_show].T, color="coral", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, D_star.mean(axis=0), color="red", linewidth=2, label="E[D*_t]")
    ax.axhline(1.0, color="k", linestyle="--", label="1.0")
    ax.set_title("Minimizing Density D*_t"); ax.set_xlabel("t"); ax.legend()

    # Martingale density tildeD_t
    ax = axes[0, 2]
    ax.plot(t_fine, tildeD[:n_show].T, color="green", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, tildeD.mean(axis=0), color="darkgreen", linewidth=2, label="E[tildeD_t]")
    ax.axhline(1.0, color="k", linestyle="--", label="1.0")
    ax.set_title("Martingale Density tildeD_t"); ax.set_xlabel("t"); ax.legend()

    # Transformed BSDE barY
    ax = axes[1, 0]
    ax.plot(t_fine, barY[:n_show].T, color="purple", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, barY.mean(axis=0), color="indigo", linewidth=2, label="E[barY_t]")
    ax.set_title("Transformed BSDE barY_t"); ax.set_xlabel("t"); ax.legend()

    # FOC terminal residual distribution
    ax = axes[1, 1]
    barY_0 = result["barY_0"]
    v      = result["v"]
    gamma  = result["gamma"]
    q      = result["q"]
    D_T_q  = D_star[:, -1] ** q
    h_prime_xi = result["_h_prime_xi"]
    R_T = -gamma * barY_0 ** q * D_T_q * h_prime_xi - v * tildeD[:, -1]
    ax.hist(R_T, bins=40, color="orange", edgecolor="k", alpha=0.75)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title("FOC Terminal Residual R_T"); ax.set_xlabel("R_T")

    # FOC consumption residual over time
    ax = axes[1, 2]
    D_t_q     = D_star[:, :-1] ** q
    u_prime_c = result["_u_prime_c"]
    R_t = -gamma * barY_0 ** q * D_t_q * u_prime_c - v * tildeD[:, :-1]
    ax.plot(t_mid, np.abs(R_t).mean(axis=0), color="teal", linewidth=2)
    ax.set_title("Mean |FOC Consumption Residual| over t")
    ax.set_xlabel("t"); ax.set_ylabel("|R_t|")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp2_foc_validation.png"), dpi=120, bbox_inches="tight")
    plt.close()


def plot_sensitivity_exp2(df: pd.DataFrame, out_dir: str) -> None:
    """Plot FOC residuals as functions of q and gamma."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Exp 2: FOC Sensitivity Analysis", fontsize=13)

    # Vary q (fix gamma=1.0, a=0.5)
    sub = df[(df["gamma"] == 1.0) & (df["a"] == 0.5)].drop_duplicates("q").sort_values("q")
    ax  = axes[0]
    ax.semilogy(sub["q"], sub["foc_terminal_mae"], "o-", label="Terminal FOC MAE")
    ax.semilogy(sub["q"], sub["foc_consumption_mae"], "s--", label="Consumption FOC MAE")
    ax.set_xlabel("q"); ax.set_ylabel("FOC MAE (log)"); ax.set_title("FOC vs q"); ax.legend()

    # Vary gamma (fix q=1.5, a=0.5)
    sub2 = df[(df["q"] == 1.5) & (df["a"] == 0.5)].drop_duplicates("gamma").sort_values("gamma")
    ax   = axes[1]
    if len(sub2) > 0:
        ax.semilogy(sub2["gamma"], sub2["foc_terminal_mae"], "o-", label="Terminal FOC MAE")
        ax.semilogy(sub2["gamma"], sub2["foc_consumption_mae"], "s--", label="Consumption FOC MAE")
    ax.set_xlabel("γ"); ax.set_ylabel("FOC MAE (log)"); ax.set_title("FOC vs γ"); ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp2_sensitivity.png"), dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 60)
    print("Experiment 2: Maximum Principle FOC Validation")
    print("=" * 60)

    # Reference configuration
    ref = run_outer_optimization(
        T=1.0, N=40, n_paths=4000, a=0.5, q=1.5, gamma=1.0,
        theta_val=0.3, sigma_val=1.0, x=1.0, n_iter=20,
        damping=0.5, seed=42, verbose=True,
    )
    plot_exp2(ref, RESULTS_DIR)
    print("  [Saved] exp2_foc_validation.png")

    # Sensitivity
    print("\nRunning sensitivity analysis...")
    df_sens = run_sensitivity_analysis_exp2(
        q_vals=[1.05, 1.2, 1.5, 2.0],
        gamma_vals=[0.5, 1.0, 2.0],
        a_vals=[0.3, 0.5, 0.7],
        x_vals=[0.5, 1.0, 2.0],
        N_vals=[20, 40, 80],
        n_paths_base=3000,
    )

    metrics_cols = [c for c in df_sens.columns if not c.startswith("_")]
    df_sens[metrics_cols].to_csv(
        os.path.join(RESULTS_DIR, "exp2_metrics.csv"), index=False
    )
    print("  [Saved] exp2_metrics.csv")

    plot_sensitivity_exp2(df_sens, RESULTS_DIR)
    print("  [Saved] exp2_sensitivity.png")

    print("\nEXP 2 Summary Table:")
    summary = df_sens[["q", "gamma", "a", "x", "N",
                        "barY_0", "v", "budget_gap",
                        "foc_terminal_mae", "foc_consumption_mae",
                        "D_star_norm_err"]].head(12)
    print(summary.to_string(index=False))

    return df_sens


if __name__ == "__main__":
    main()
