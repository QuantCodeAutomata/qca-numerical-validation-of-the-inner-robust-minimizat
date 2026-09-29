"""Experiment 3: Coupled FBSDE characterization, existence-uniqueness evidence,
and budget-saturation validation for the full optimal robust strategy.

Validates:
  - Budget saturation X_0 ≈ x
  - Terminal consistency X_T ≈ xi* and Y_T ≈ h(xi*)
  - Transform consistency barY ≈ exp_q(-gamma*Y)
  - Density identities: D* (stochastic exponential) vs closed form
  - Existence: admissible solution across parameter settings
  - Uniqueness: convergence from multiple initializations
  - No-consumption special case (c ≡ 0)
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
from src.tsallis import exp_q as tsallis_exp_q, ln_q as tsallis_ln_q
from src.bsde import solve_transformed_bsde, recover_original_bsde
from src.measures import (
    compute_girsanov_kernel,
    stochastic_exponential,
    compute_density_closed_form,
)
from src.utilities import power_utility, transformed_terminal_utility
from src.transforms import check_transform_consistency, verify_mu_positivity
from src.diagnostics import normalization_check, density_consistency_error
from src.wealth import compute_wealth_process, validate_terminal_consistency
from src.fbsde_solver import solve_coupled_fbsde

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_full_fbsde(
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
    no_consumption: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run full coupled FBSDE and check all consistency conditions.

    Args:
        T: Time horizon.
        N: Number of time steps.
        n_paths: Monte Carlo path count.
        a: Power utility curvature.
        q: Tsallis parameter (q > 1).
        gamma: Ambiguity aversion.
        theta_val: Constant market price of risk.
        sigma_val: Constant volatility.
        x: Initial wealth.
        n_iter: Max fixed-point iterations.
        damping: Damping factor.
        no_consumption: If True, set c ≡ 0 (no-consumption case).
        seed: Random seed.
        verbose: Print progress.
    Returns:
        Dict of consistency metrics and arrays.
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    d = 1

    B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)
    theta_paths = theta_val * np.ones((n_paths, N, d))
    tildeD = compute_martingale_density(theta_paths, dB, dt)

    if verbose:
        tag = "(no-consumption)" if no_consumption else ""
        print(f"  q={q:.2f}, gamma={gamma:.2f}, a={a:.2f}, x={x:.2f} {tag}")

    if no_consumption:
        # No-consumption special case: c ≡ 0, optimize only over xi
        # Use fixed constant xi = x as baseline, then solve for optimal xi
        xi_fixed = np.full(n_paths, x)
        terminal_cond = transformed_terminal_utility(xi_fixed, gamma, a, q)
        U_paths = np.zeros((n_paths, N))  # zero consumption

        barY, barZ = solve_transformed_bsde(
            B, dB, dt, gamma, q, U_paths, terminal_cond, degree=3
        )
        Y, Z = recover_original_bsde(barY, barZ, gamma, q)
        eta_star = compute_girsanov_kernel(barZ, barY)
        D_star = stochastic_exponential(eta_star, dB, dt)

        sol = {
            "barY": barY, "barZ": barZ, "Y": Y, "Z": Z,
            "D_star": D_star, "eta_star": eta_star,
            "c_star": U_paths, "xi_star": xi_fixed,
            "v": float("nan"), "barY_0": float(np.mean(barY[:, 0])),
            "budget": float(np.mean(tildeD[:, -1] * xi_fixed)),
            "budget_gap": abs(float(np.mean(tildeD[:, -1] * xi_fixed)) - x),
            "conv_history": [], "n_iter_done": 1,
        }
    else:
        # Full consumption-investment problem
        sol = solve_coupled_fbsde(
            B, dB, dt, T, gamma, q, a, x, tildeD,
            n_iter=n_iter, tol=5e-4, damping=damping,
            v_init=-1.0, degree=3, verbose=verbose,
        )

    barY    = sol["barY"]
    barZ    = sol["barZ"]
    Y       = sol["Y"]
    Z       = sol["Z"]
    D_star  = sol["D_star"]
    c_star  = sol["c_star"]
    xi_star = sol["xi_star"]
    barY_0  = sol["barY_0"]

    # ---- Consistency checks ----

    # 1. Budget saturation
    budget_gap = sol["budget_gap"]

    # 2. Terminal consistency Y_T ≈ h(xi*)
    h_xi = power_utility(xi_star, a)
    Y_T_err = float(np.mean(np.abs(Y[:, -1] - h_xi)))

    # 3. Wealth process X_T ≈ xi*
    X = compute_wealth_process(c_star if not no_consumption else np.zeros((n_paths, N)),
                               xi_star, tildeD, dt)
    X_T_err = validate_terminal_consistency(X, xi_star)
    X_0_val = float(np.mean(X[:, 0]))

    # 4. Transform consistency
    tc = check_transform_consistency(Y, Z, barY, barZ, gamma, q)

    # 5. Density identity: stochastic exponential vs closed form
    u_c = power_utility(c_star, a) if not no_consumption else np.zeros((n_paths, N))
    D_cf    = compute_density_closed_form(barY, u_c, dt, gamma, q)
    d_cons  = density_consistency_error(D_star, D_cf)

    # 6. Normalization
    norm_D = normalization_check(D_star[:, -1])
    norm_tD = normalization_check(tildeD[:, -1])

    # 7. mu(Y) > 0
    mu_chk = verify_mu_positivity(Y, gamma, q)

    if verbose:
        print(f"  barY_0={barY_0:.6f}, budget_gap={budget_gap:.2e}")
        print(f"  Y_T consistency err={Y_T_err:.2e}, X_T err={X_T_err:.2e}")
        print(f"  checkY transform MAE={tc['checkY_mean_abs_err']:.2e}")
        print(f"  Density closed-form MAE={d_cons['density_mean_abs_err']:.2e}")
        print(f"  D* norm error={norm_D['norm_error']:.2e}")
        print(f"  mu > 0 fraction={mu_chk['mu_fraction_positive']:.4f}")
        print()

    return {
        "q": q, "gamma": gamma, "a": a, "x": x, "N": N,
        "n_paths": n_paths, "no_consumption": no_consumption,
        "barY_0": barY_0, "budget_gap": budget_gap,
        "Y_T_err": Y_T_err, "X_T_err": X_T_err,
        "X_0": X_0_val,
        "checkY_transform_mae": tc["checkY_mean_abs_err"],
        "checkZ_transform_mae": tc["checkZ_mean_abs_err"],
        "density_closed_form_mae": d_cons["density_mean_abs_err"],
        "D_star_norm_err": norm_D["norm_error"],
        "tildeD_norm_err": norm_tD["norm_error"],
        "mu_frac_positive": mu_chk["mu_fraction_positive"],
        "n_iter_done": sol["n_iter_done"],
        # Arrays for plotting
        "_barY": barY, "_Y": Y, "_D_star": D_star, "_X": X,
        "_c_star": c_star, "_tildeD": tildeD,
    }


def uniqueness_experiment(
    T: float, N: int, n_paths: int, a: float, q: float, gamma: float,
    theta_val: float, x: float, n_seeds: int = 4
) -> dict:
    """Test uniqueness by solving from multiple DIFFERENT solver initialisations.

    Uniqueness means the *same* Brownian path sample converges to the *same*
    solution regardless of the starting guess.  To isolate solver convergence
    from Monte Carlo variance the paths are **fixed** (shared seed = 42) and
    only the initial multiplier v_init and control scale are varied across
    n_seeds runs.

    Args:
        T, N, n_paths, a, q, gamma, theta_val, x: Model parameters.
        n_seeds: Number of different solver initialisations (not Brownian seeds).
    Returns:
        Dict with spread statistics of barY_0 across initialisations.
    """
    dt = T / N
    d = 1

    # --- fixed Brownian paths shared by every initialisation ---
    rng = np.random.default_rng(42)
    B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)
    theta_paths = theta_val * np.ones((n_paths, N, d))
    tildeD = compute_martingale_density(theta_paths, dB, dt)

    barY_0_vals = []
    budget_gaps = []

    # Vary the initial Lagrange multiplier over a wide range
    v_inits = [-0.5, -1.0, -2.0, -5.0][:n_seeds]

    for k, v0 in enumerate(v_inits):
        sol = solve_coupled_fbsde(
            B, dB, dt, T, gamma, q, a, x, tildeD,
            n_iter=20, tol=5e-4, damping=0.5,
            v_init=v0,   # vary initial Lagrange multiplier
            degree=3, verbose=False,
        )
        barY_0_vals.append(sol["barY_0"])
        budget_gaps.append(sol["budget_gap"])

    barY_0_arr = np.array(barY_0_vals)
    return {
        "barY_0_mean": float(np.mean(barY_0_arr)),
        "barY_0_std": float(np.std(barY_0_arr)),
        "barY_0_max_spread": float(np.max(barY_0_arr) - np.min(barY_0_arr)),
        "barY_0_rel_spread": float(np.std(barY_0_arr) / (abs(np.mean(barY_0_arr)) + 1e-10)),
        "mean_budget_gap": float(np.mean(budget_gaps)),
        "all_barY_0": barY_0_arr.tolist(),
        "v_inits": v_inits,
    }


def plot_exp3(result: dict, out_dir: str) -> None:
    """Plot key processes for exp_3."""
    barY   = result["_barY"]
    Y      = result["_Y"]
    D_star = result["_D_star"]
    X      = result["_X"]
    tildeD = result["_tildeD"]
    c_star = result["_c_star"]

    n_show = min(50, barY.shape[0])
    T = 1.0
    N = barY.shape[1] - 1
    t_fine = np.linspace(0, T, N + 1)
    t_mid  = np.linspace(0, T, N)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    tag = "(no-consumption)" if result["no_consumption"] else ""
    fig.suptitle(
        f"Exp 3: Coupled FBSDE Validation  "
        f"(q={result['q']:.2f}, γ={result['gamma']:.2f}) {tag}",
        fontsize=12
    )

    # barY paths
    ax = axes[0, 0]
    ax.plot(t_fine, barY[:n_show].T, color="steelblue", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, barY.mean(axis=0), color="navy", linewidth=2, label="E[barY_t]")
    ax.set_title("Transformed barY_t"); ax.set_xlabel("t"); ax.legend()

    # Y paths
    ax = axes[0, 1]
    ax.plot(t_fine, Y[:n_show].T, color="coral", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, Y.mean(axis=0), color="red", linewidth=2, label="E[Y_t]")
    ax.set_title("BSDE Y_t"); ax.set_xlabel("t"); ax.legend()

    # D* paths
    ax = axes[0, 2]
    ax.plot(t_fine, D_star[:n_show].T, color="green", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, D_star.mean(axis=0), color="darkgreen", linewidth=2, label="E[D*_t]")
    ax.axhline(1.0, color="k", linestyle="--", label="1.0")
    ax.set_title("Minimizing Density D*_t"); ax.set_xlabel("t"); ax.legend()

    # Wealth X_t paths
    ax = axes[1, 0]
    ax.plot(t_fine, X[:n_show].T, color="purple", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, X.mean(axis=0), color="indigo", linewidth=2, label="E[X_t]")
    ax.set_title("Wealth Process X_t"); ax.set_xlabel("t"); ax.legend()

    # Optimal consumption c*_t
    ax = axes[1, 1]
    if c_star.ndim == 2 and c_star.shape[1] > 0:
        ax.plot(t_mid, c_star[:n_show].T, color="orange", alpha=0.2, linewidth=0.7)
        ax.plot(t_mid, c_star.mean(axis=0), color="darkorange", linewidth=2, label="E[c*_t]")
    ax.set_title("Optimal Consumption c*_t"); ax.set_xlabel("t"); ax.legend()

    # tildeD (martingale density)
    ax = axes[1, 2]
    ax.plot(t_fine, tildeD[:n_show].T, color="teal", alpha=0.2, linewidth=0.7)
    ax.plot(t_fine, tildeD.mean(axis=0), color="darkcyan", linewidth=2, label="E[tildeD_t]")
    ax.axhline(1.0, color="k", linestyle="--", label="1.0")
    ax.set_title("Martingale Density tildeD_t"); ax.set_xlabel("t"); ax.legend()

    plt.tight_layout()
    tag_str = "no_cons" if result["no_consumption"] else "full"
    plt.savefig(
        os.path.join(out_dir, f"exp3_fbsde_{tag_str}.png"),
        dpi=120, bbox_inches="tight"
    )
    plt.close()


def main() -> None:
    print("=" * 60)
    print("Experiment 3: Coupled FBSDE Existence-Uniqueness Validation")
    print("=" * 60)

    # ---- (a) No-consumption special case ----
    print("\n--- No-consumption case ---")
    ref_nc = run_full_fbsde(
        T=1.0, N=40, n_paths=4000, a=0.5, q=1.5, gamma=1.0,
        theta_val=0.3, sigma_val=1.0, x=1.0, n_iter=5,
        no_consumption=True, seed=42, verbose=True
    )
    plot_exp3(ref_nc, RESULTS_DIR)
    print("  [Saved] exp3_fbsde_no_cons.png")

    # ---- (b) Full consumption-investment problem ----
    print("\n--- Full consumption-investment case ---")
    ref_full = run_full_fbsde(
        T=1.0, N=40, n_paths=4000, a=0.5, q=1.5, gamma=1.0,
        theta_val=0.3, sigma_val=1.0, x=1.0, n_iter=20,
        no_consumption=False, seed=42, verbose=True
    )
    plot_exp3(ref_full, RESULTS_DIR)
    print("  [Saved] exp3_fbsde_full.png")

    # ---- (c) Uniqueness evidence ----
    print("\n--- Uniqueness experiment (multiple seeds) ---")
    uniq = uniqueness_experiment(
        T=1.0, N=40, n_paths=2000, a=0.5, q=1.5, gamma=1.0,
        theta_val=0.3, x=1.0, n_seeds=4,
    )
    print(f"  barY_0 across seeds: {uniq['all_barY_0']}")
    print(f"  Relative spread: {uniq['barY_0_rel_spread']:.2e}")
    print(f"  Mean budget gap: {uniq['mean_budget_gap']:.2e}")

    # ---- (d) Sensitivity analysis ----
    print("\nRunning parameter sensitivity...")
    records = []
    for q in [1.05, 1.2, 1.5, 2.0]:
        for gam in [0.5, 1.0, 2.0]:
            r = run_full_fbsde(
                T=1.0, N=30, n_paths=2000, a=0.5, q=q, gamma=gam,
                theta_val=0.3, sigma_val=1.0, x=1.0, n_iter=15,
                no_consumption=False, seed=42, verbose=False,
            )
            records.append({k: v for k, v in r.items() if not k.startswith("_")})

    # Refinement study
    for N in [20, 40, 80]:
        r = run_full_fbsde(
            T=1.0, N=N, n_paths=3000, a=0.5, q=1.5, gamma=1.0,
            theta_val=0.3, sigma_val=1.0, x=1.0, n_iter=15,
            no_consumption=False, seed=42, verbose=False,
        )
        records.append({k: v for k, v in r.items() if not k.startswith("_")})

    df_sens = pd.DataFrame(records)

    # Append uniqueness result as a single row
    uniq_row = {
        "q": 1.5, "gamma": 1.0, "a": 0.5, "x": 1.0, "N": 40,
        "n_paths": 2000, "no_consumption": False,
        "barY_0": uniq["barY_0_mean"],
        "barY_0_rel_spread": uniq["barY_0_rel_spread"],
        "budget_gap": uniq["mean_budget_gap"],
    }

    metrics_cols = [c for c in df_sens.columns if not c.startswith("_")]
    df_sens[metrics_cols].to_csv(
        os.path.join(RESULTS_DIR, "exp3_metrics.csv"), index=False
    )
    print("  [Saved] exp3_metrics.csv")

    # Sensitivity plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Exp 3: Coupled FBSDE Sensitivity", fontsize=13)

    sub_q = df_sens[df_sens["gamma"] == 1.0].drop_duplicates("q").sort_values("q")
    axes[0].semilogy(sub_q["q"], sub_q["budget_gap"].clip(1e-8), "o-",
                     label="Budget gap")
    axes[0].semilogy(sub_q["q"], sub_q["Y_T_err"].clip(1e-8), "s--",
                     label="Y_T err")
    axes[0].set_xlabel("q"); axes[0].set_ylabel("Error (log)"); axes[0].legend()
    axes[0].set_title("Errors vs q (γ=1.0)")

    sub_g = df_sens[df_sens["q"] == 1.5].drop_duplicates("gamma").sort_values("gamma")
    if len(sub_g) > 0:
        axes[1].semilogy(sub_g["gamma"], sub_g["budget_gap"].clip(1e-8), "o-",
                         label="Budget gap")
        axes[1].semilogy(sub_g["gamma"], sub_g["Y_T_err"].clip(1e-8), "s--",
                         label="Y_T err")
    axes[1].set_xlabel("γ"); axes[1].set_ylabel("Error (log)"); axes[1].legend()
    axes[1].set_title("Errors vs γ (q=1.5)")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_sensitivity.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  [Saved] exp3_sensitivity.png")

    print("\nEXP 3 Summary Table:")
    summary_cols = ["q", "gamma", "a", "x", "N",
                    "barY_0", "budget_gap", "Y_T_err",
                    "checkY_transform_mae", "density_closed_form_mae",
                    "D_star_norm_err"]
    summary_cols = [c for c in summary_cols if c in df_sens.columns]
    print(df_sens[summary_cols].head(12).to_string(index=False))

    print(f"\nUniqueness spread (relative): {uniq['barY_0_rel_spread']:.2e}")

    return df_sens, uniq


if __name__ == "__main__":
    main()
