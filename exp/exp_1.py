"""Experiment 1: Numerical validation of the inner robust minimization and quadratic
BSDE representation under Tsallis relative entropy.

Validates:
  - Transformed BSDE solving and (Y, Z) recovery
  - Minimizing measure Q* construction
  - Equality F(Q*; c, xi) ≈ Y_0
  - Q* is not improved by Girsanov perturbations
  - Sensitivity over q, gamma, discretization
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
from src.tsallis import tsallis_entropy_integral, exp_q, ln_q
from src.bsde import solve_transformed_bsde, recover_original_bsde
from src.measures import (
    compute_girsanov_kernel,
    stochastic_exponential,
    compute_inner_robust_objective,
    compute_inner_robust_objective_at_P,
    compute_density_closed_form,
)
from src.transforms import check_transform_consistency, verify_mu_positivity
from src.diagnostics import (
    compute_equality_gap,
    perturbation_test,
    normalization_check,
    density_consistency_error,
)
from src.utilities import power_utility, transformed_terminal_utility

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_single_config(
    T: float,
    N: int,
    n_paths: int,
    a: float,
    q: float,
    gamma: float,
    theta_val: float,
    sigma_val: float,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run inner robust validation for one parameter configuration.

    Args:
        T: Time horizon.
        N: Number of time steps.
        n_paths: Monte Carlo path count.
        a: Power utility parameter (0 < a < 1).
        q: Tsallis parameter (q > 1).
        gamma: Ambiguity aversion.
        theta_val: Constant market price of risk.
        sigma_val: Constant volatility.
        seed: Random seed.
        verbose: Print progress.
    Returns:
        Dict of metrics for this configuration.
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    d = 1  # one-dimensional market

    # Simulate paths
    B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)

    # Construct exogenous admissible controls (fixed, not optimized)
    # c_t: deterministic bounded nonneg consumption
    # xi: bounded nonneg function of B_T
    t_grid = np.linspace(0, T, N + 1)
    c_level = 0.5  # base consumption level
    c_paths = c_level * np.ones((n_paths, N))  # constant consumption

    # Terminal wealth: positive function of B_T
    xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)  # shape (n_paths,)

    # Utility values
    U_paths = power_utility(c_paths, a)   # u(c_t)
    h_xi    = power_utility(xi, a)        # h(xi)

    # Terminal condition for transformed BSDE: exp_q(-gamma * h(xi))
    terminal_cond = transformed_terminal_utility(xi, gamma, a, q)

    if verbose:
        print(f"  q={q:.2f}, gamma={gamma:.2f}, N={N}, n_paths={n_paths}")
        print(f"  terminal_cond: min={terminal_cond.min():.4f}, max={terminal_cond.max():.4f}")

    # Solve transformed BSDE
    checkY, checkZ = solve_transformed_bsde(
        B, dB, dt, gamma, q, U_paths, terminal_cond, degree=3
    )

    # Recover original BSDE (Y, Z)
    Y, Z = recover_original_bsde(checkY, checkZ, gamma, q)
    Y_0 = float(np.mean(Y[:, 0]))

    # Transform consistency
    tc = check_transform_consistency(Y, Z, checkY, checkZ, gamma, q)
    mu_check = verify_mu_positivity(Y, gamma, q)

    # Construct minimizing measure Q*
    eta_star = compute_girsanov_kernel(checkZ, checkY)
    D_star   = stochastic_exponential(eta_star, dB, dt)

    # Tsallis entropy of Q*
    H_q = tsallis_entropy_integral(eta_star, D_star, dt, q)

    # Inner objective at Q*
    F_star = compute_inner_robust_objective(D_star, h_xi, U_paths, dt, q, H_q, gamma)

    # Inner objective under P (baseline)
    F_P = compute_inner_robust_objective_at_P(h_xi, U_paths, dt)

    # Normalization check
    norm_check = normalization_check(D_star[:, -1])

    # Equality gap: main validation metric
    gap = compute_equality_gap(Y_0, F_star)

    # Density closed-form consistency check
    D_cf    = compute_density_closed_form(checkY, U_paths, dt, gamma, q)
    d_cons  = density_consistency_error(D_star, D_cf)

    # Perturbation test
    delta_eta = 0.1 * np.ones_like(eta_star)  # small bounded perturbation
    ptest = perturbation_test(
        checkY, checkZ, dB, dt, h_xi, U_paths,
        gamma, q, delta_eta, lambdas=[-0.5, -0.2, 0.0, 0.2, 0.5]
    )

    if verbose:
        print(f"  Y_0={Y_0:.6f}, F(Q*)={F_star:.6f}, F(P)={F_P:.6f}")
        print(f"  Equality gap: abs={gap['abs_gap']:.2e}, rel={gap['rel_gap']:.2e}")
        print(f"  Norm error: {norm_check['norm_error']:.2e}")
        print(f"  checkY transform MAE: {tc['checkY_mean_abs_err']:.2e}")
        print(f"  mu > 0 fraction: {mu_check['mu_fraction_positive']:.4f}")
        print(f"  Perturbation is_minimizer: {ptest['is_minimizer']}")
        print()

    return {
        "q": q, "gamma": gamma, "N": N, "n_paths": n_paths,
        "Y_0": Y_0, "F_star": F_star, "F_P": F_P,
        "abs_gap": gap["abs_gap"], "rel_gap": gap["rel_gap"],
        "norm_error": norm_check["norm_error"],
        "mean_D_T": norm_check["mean_D_T"],
        "H_q": H_q,
        "checkY_transform_mae": tc["checkY_mean_abs_err"],
        "checkZ_transform_mae": tc["checkZ_mean_abs_err"],
        "mu_frac_positive": mu_check["mu_fraction_positive"],
        "density_mae": d_cons["density_mean_abs_err"],
        "perturb_is_minimizer": bool(ptest["is_minimizer"].all()),
        # Save arrays for plotting
        "_Y": Y, "_checkY": checkY, "_Z": Z, "_D_star": D_star,
        "_eta_star": eta_star, "_ptest": ptest,
    }


def run_sensitivity_analysis(
    q_vals: list, gamma_vals: list, N_vals: list, n_paths_vals: list
) -> pd.DataFrame:
    """Run sensitivity analysis over q, gamma, N, n_paths.

    Returns a DataFrame of metrics for each configuration.
    """
    records = []
    for q in q_vals:
        for gam in gamma_vals:
            result = run_single_config(
                T=1.0, N=50, n_paths=4000, a=0.5, q=q, gamma=gam,
                theta_val=0.3, sigma_val=1.0, seed=42, verbose=False
            )
            records.append({k: v for k, v in result.items() if not k.startswith("_")})

    # Refinement study
    q0, gam0 = q_vals[0], gamma_vals[0]
    for N in N_vals:
        for np_ in n_paths_vals:
            result = run_single_config(
                T=1.0, N=N, n_paths=np_, a=0.5, q=q0, gamma=gam0,
                theta_val=0.3, sigma_val=1.0, seed=42, verbose=False
            )
            records.append({k: v for k, v in result.items() if not k.startswith("_")})

    return pd.DataFrame(records)


def plot_sample_paths(result: dict, out_dir: str) -> None:
    """Plot sample paths for Y, checkY, Z, D*, and perturbation results."""
    Y       = result["_Y"]       # (n_paths, N+1)
    checkY  = result["_checkY"]  # (n_paths, N+1)
    Z       = result["_Z"]       # (n_paths, N, 1)
    D_star  = result["_D_star"]  # (n_paths, N+1)
    ptest   = result["_ptest"]

    n_show = min(50, Y.shape[0])
    t_fine = np.linspace(0, 1.0, Y.shape[1])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Exp 1: Inner Robust BSDE Validation  "
        f"(q={result['q']:.2f}, γ={result['gamma']:.2f})",
        fontsize=13
    )

    # Y paths
    ax = axes[0, 0]
    ax.plot(t_fine, Y[:n_show].T, color="steelblue", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, Y.mean(axis=0), color="navy", linewidth=2, label="E[Y_t]")
    ax.set_title("BSDE Solution Y_t"); ax.set_xlabel("t"); ax.legend()

    # checkY paths
    ax = axes[0, 1]
    ax.plot(t_fine, checkY[:n_show].T, color="coral", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, checkY.mean(axis=0), color="red", linewidth=2, label="E[checkY_t]")
    ax.set_title("Transformed checkY_t"); ax.set_xlabel("t"); ax.legend()

    # Z paths (scalar d=1)
    t_mid = np.linspace(0, 1.0, Z.shape[1])
    ax = axes[0, 2]
    ax.plot(t_mid, Z[:n_show, :, 0].T, color="green", alpha=0.2, linewidth=0.7)
    ax.plot(t_mid, Z[:, :, 0].mean(axis=0), color="darkgreen", linewidth=2, label="E[Z_t]")
    ax.set_title("BSDE Control Z_t"); ax.set_xlabel("t"); ax.legend()

    # D* paths
    ax = axes[1, 0]
    ax.plot(t_fine, D_star[:n_show].T, color="purple", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, D_star.mean(axis=0), color="indigo", linewidth=2, label="E[D*_t]")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="1.0")
    ax.set_title("Minimizing Density D*_t"); ax.set_xlabel("t"); ax.legend()

    # Terminal histogram: D_T^*
    ax = axes[1, 1]
    ax.hist(D_star[:, -1], bins=40, color="orange", edgecolor="k", alpha=0.75)
    ax.axvline(1.0, color="red", linestyle="--", label="E=1")
    ax.set_title("Histogram of D*_T"); ax.set_xlabel("D*_T"); ax.legend()

    # Perturbation test
    ax = axes[1, 2]
    ax.axhline(ptest["F_star"], color="red", linestyle="--", linewidth=2, label="F(Q*)")
    ax.plot(ptest["lambdas"], ptest["objectives"], "o-", color="navy", label="F(Q*+λδη)")
    ax.set_title("Perturbation Test: F vs λ")
    ax.set_xlabel("λ"); ax.set_ylabel("Objective"); ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp1_sample_paths.png"), dpi=120, bbox_inches="tight")
    plt.close()


def plot_sensitivity(df: pd.DataFrame, out_dir: str) -> None:
    """Plot sensitivity of equality gap w.r.t. q and gamma."""
    q_vals   = sorted(df["q"].unique())
    gam_vals = sorted(df["gamma"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Exp 1: Equality Gap Sensitivity", fontsize=13)

    # Vary q (fix gamma at first value)
    gam0 = gam_vals[0]
    sub_q = df[df["gamma"] == gam0].drop_duplicates("q").sort_values("q")
    ax = axes[0]
    ax.semilogy(sub_q["q"], sub_q["abs_gap"], "o-", color="steelblue")
    ax.set_xlabel("q"); ax.set_ylabel("|F(Q*) - Y_0| (log scale)")
    ax.set_title(f"Abs Gap vs q  (γ={gam0:.2f})")

    # Vary gamma (fix q at first value)
    q0 = q_vals[0]
    sub_g = df[df["q"] == q0].drop_duplicates("gamma").sort_values("gamma")
    ax = axes[1]
    ax.semilogy(sub_g["gamma"], sub_g["abs_gap"], "s-", color="coral")
    ax.set_xlabel("γ"); ax.set_ylabel("|F(Q*) - Y_0| (log scale)")
    ax.set_title(f"Abs Gap vs γ  (q={q0:.2f})")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp1_sensitivity.png"), dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 60)
    print("Experiment 1: Inner Robust BSDE Validation")
    print("=" * 60)

    # Reference configuration
    ref = run_single_config(
        T=1.0, N=50, n_paths=5000, a=0.5, q=1.2, gamma=1.0,
        theta_val=0.3, sigma_val=1.0, seed=42, verbose=True
    )
    plot_sample_paths(ref, RESULTS_DIR)
    print("  [Saved] exp1_sample_paths.png")

    # Sensitivity analysis
    print("\nRunning sensitivity analysis...")
    df_sens = run_sensitivity_analysis(
        q_vals=[1.05, 1.2, 1.5, 2.0],
        gamma_vals=[0.5, 1.0, 2.0],
        N_vals=[20, 50, 100],
        n_paths_vals=[2000, 5000],
    )

    # Save metrics CSV
    metrics_cols = [c for c in df_sens.columns if not c.startswith("_")]
    df_sens[metrics_cols].to_csv(
        os.path.join(RESULTS_DIR, "exp1_metrics.csv"), index=False
    )
    print("  [Saved] exp1_metrics.csv")

    plot_sensitivity(df_sens, RESULTS_DIR)
    print("  [Saved] exp1_sensitivity.png")

    # Summary table
    print("\nEXP 1 Summary Table:")
    summary = df_sens[["q", "gamma", "N", "n_paths",
                        "Y_0", "F_star", "abs_gap", "rel_gap",
                        "norm_error", "mu_frac_positive",
                        "perturb_is_minimizer"]].head(12)
    print(summary.to_string(index=False))

    return df_sens


if __name__ == "__main__":
    main()
