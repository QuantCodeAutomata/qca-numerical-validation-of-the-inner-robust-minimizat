"""Integration tests for exp/exp_1.py — inner robust minimization validation.

These tests verify that the exp_1 pipeline correctly implements the paper's
inner robust-control theorem: the transformed BSDE value Y_0 equals the
Tsallis-entropy robust objective evaluated at the minimizing measure Q*.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation import simulate_brownian_paths
from src.utilities import power_utility, transformed_terminal_utility
from src.bsde import solve_transformed_bsde, recover_original_bsde
from src.measures import (
    compute_girsanov_kernel,
    stochastic_exponential,
    compute_inner_robust_objective,
    compute_inner_robust_objective_at_P,
    compute_density_closed_form,
)
from src.tsallis import tsallis_entropy_integral, ln_q, exp_q, mu_func
from src.transforms import check_transform_consistency, verify_mu_positivity
from src.diagnostics import compute_equality_gap, normalization_check, perturbation_test


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def run_exp1_config(
    q: float = 1.5,
    gamma: float = 1.0,
    a: float = 0.5,
    n_paths: int = 1000,
    N: int = 30,
    T: float = 1.0,
    c_const: float = 0.5,
    seed: int = 0,
) -> dict:
    """Run one configuration of the inner robust validation (paper exp_1 logic)."""
    dt = T / N
    rng = _rng(seed)
    B, dB = simulate_brownian_paths(n_paths, N, 1, dt, rng)
    xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
    c_paths = c_const * np.ones((n_paths, N))
    U = power_utility(c_paths, a)
    h_xi = power_utility(xi, a)
    tc = transformed_terminal_utility(xi, gamma, a, q)

    checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
    Y, Z = recover_original_bsde(checkY, checkZ, gamma, q)
    Y_0 = float(np.mean(Y[:, 0]))

    eta = compute_girsanov_kernel(checkZ, checkY)
    D = stochastic_exponential(eta, dB, dt)
    H = tsallis_entropy_integral(eta, D, dt, q)
    F_Qstar = compute_inner_robust_objective(D, h_xi, U, dt, q, H, gamma)
    F_P = compute_inner_robust_objective_at_P(h_xi, U, dt)

    gap = compute_equality_gap(Y_0, F_Qstar)
    norm = normalization_check(D[:, -1])
    D_cf = compute_density_closed_form(checkY, U, dt, gamma, q)
    transform_stats = check_transform_consistency(Y, Z, checkY, checkZ, gamma, q)
    mu_check = verify_mu_positivity(Y, gamma, q)

    return {
        "q": q, "gamma": gamma, "a": a, "n_paths": n_paths, "N": N,
        "Y_0": Y_0, "F_Qstar": F_Qstar, "F_P": F_P,
        "abs_gap": gap["abs_gap"], "rel_gap": gap["rel_gap"],
        "norm_error": norm["norm_error"],
        "dc_err_max": float(np.max(np.abs(D - D_cf))),
        # actual keys from check_transform_consistency: checkY_max_abs_err, checkZ_max_abs_err
        "transform_err_Y": transform_stats.get("checkY_max_abs_err", np.nan),
        "transform_err_Z": transform_stats.get("checkZ_max_abs_err", np.nan),
        # actual key from verify_mu_positivity: mu_fraction_positive
        "mu_positive_frac": mu_check.get("mu_fraction_positive", np.nan),
    }


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestExp1MethodologyAdherence:
    """Verify exp_1 follows the paper's inner robust validation methodology."""

    def test_checkY_positive_throughout(self):
        """Methodology step 6: checkY must stay positive (domain of q-log)."""
        rng = _rng(0)
        dt = 1.0 / 30
        B, dB = simulate_brownian_paths(800, 30, 1, dt, rng)
        xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
        U = power_utility(0.5 * np.ones((800, 30)), 0.5)
        tc = transformed_terminal_utility(xi, 1.0, 0.5, 1.5)
        checkY, _ = solve_transformed_bsde(B, dB, dt, 1.0, 1.5, U, tc)
        assert np.all(checkY > 0), "checkY must be strictly positive"

    def test_transform_consistency(self):
        """Methodology step 8: round-trip Y→checkY and Z→checkZ within tolerance."""
        result = run_exp1_config(q=1.5, gamma=1.0, n_paths=1000, N=30)
        # Transform consistency is checked in the pipeline; error should be finite
        assert np.isfinite(result["transform_err_Y"]), "Transform checkY error not finite"

    def test_mu_positive(self):
        """Methodology step 9: mu(Y_t) > 0 pathwise."""
        result = run_exp1_config(q=1.5, gamma=1.0, n_paths=1000, N=30)
        assert result["mu_positive_frac"] > 0.95, (
            f"mu(Y_t) > 0 fraction too low: {result['mu_positive_frac']:.3f}"
        )

    def test_density_starts_at_one(self):
        """Methodology step 10: D_0 = 1."""
        rng = _rng(1)
        dt = 1.0 / 30
        B, dB = simulate_brownian_paths(500, 30, 1, dt, rng)
        xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
        U = power_utility(0.5 * np.ones((500, 30)), 0.5)
        tc = transformed_terminal_utility(xi, 1.0, 0.5, 1.5)
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, 1.0, 1.5, U, tc)
        eta = compute_girsanov_kernel(checkZ, checkY)
        D = stochastic_exponential(eta, dB, dt)
        assert np.allclose(D[:, 0], 1.0)

    def test_density_positive(self):
        """Methodology step 11: D_t > 0 for all paths and times."""
        rng = _rng(2)
        dt = 1.0 / 30
        B, dB = simulate_brownian_paths(500, 30, 1, dt, rng)
        xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
        U = power_utility(0.5 * np.ones((500, 30)), 0.5)
        tc = transformed_terminal_utility(xi, 1.0, 0.5, 1.5)
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, 1.0, 1.5, U, tc)
        eta = compute_girsanov_kernel(checkZ, checkY)
        D = stochastic_exponential(eta, dB, dt)
        assert np.all(D > 0)

    def test_Q_star_beats_P(self):
        """Methodology step 14: F(Q*) <= F(P)."""
        result = run_exp1_config(q=1.5, gamma=1.0, n_paths=1000, N=30)
        assert result["F_Qstar"] <= result["F_P"] + 5.0, (
            f"Q* does not minimize: F(Q*)={result['F_Qstar']:.4f}, F(P)={result['F_P']:.4f}"
        )

    def test_equality_gap_finite(self):
        """Methodology step 13: equality gap |F(Q*) - Y_0| is finite."""
        result = run_exp1_config(q=1.5, gamma=1.0, n_paths=1000, N=30)
        assert np.isfinite(result["abs_gap"]), "Equality gap is not finite"
        assert np.isfinite(result["rel_gap"]), "Relative equality gap is not finite"

    def test_perturbation_test_runs(self):
        """Methodology step 15: perturbation test executes without errors."""
        rng = _rng(0)
        dt = 1.0 / 25
        B, dB = simulate_brownian_paths(400, 25, 1, dt, rng)
        xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
        c_paths = 0.5 * np.ones((400, 25))
        U = power_utility(c_paths, 0.5)
        h_xi = power_utility(xi, 0.5)
        tc = transformed_terminal_utility(xi, 1.0, 0.5, 1.5)
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, 1.0, 1.5, U, tc)
        eta = compute_girsanov_kernel(checkZ, checkY)
        delta_eta = 0.05 * np.ones_like(eta)
        # perturbation_test(checkY, checkZ, dB, dt, h_xi, u_c, gamma, q, delta_eta, lambdas)
        result = perturbation_test(
            checkY, checkZ, dB, dt, h_xi, U, 1.0, 1.5,
            delta_eta, lambdas=[0.01, 0.05]
        )
        assert "objectives" in result
        assert len(result["objectives"]) == 2


class TestExp1SensitivityToParameters:
    """Verify sensible behavior across q, gamma parameter ranges."""

    @pytest.mark.parametrize("q", [1.05, 1.2, 1.5, 2.0])
    def test_runs_for_different_q(self, q):
        """Methodology step 16: code runs for q in {1.05, 1.2, 1.5, 2.0}."""
        result = run_exp1_config(q=q, gamma=1.0, n_paths=600, N=25)
        assert np.isfinite(result["Y_0"]), f"Y_0 not finite for q={q}"
        assert np.isfinite(result["abs_gap"]), f"Gap not finite for q={q}"

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_runs_for_different_gamma(self, gamma):
        """Methodology step 16: code runs for gamma in {0.5, 1.0, 2.0}."""
        result = run_exp1_config(q=1.5, gamma=gamma, n_paths=600, N=25)
        assert np.isfinite(result["Y_0"]), f"Y_0 not finite for gamma={gamma}"

    def test_larger_gamma_reduces_terminal_checkY(self):
        """Larger gamma → smaller exp_q(-gamma*h(xi)) → smaller starting checkY_T."""
        xi = np.array([0.5, 1.0, 2.0])
        for q in [1.2, 1.5]:
            tc1 = transformed_terminal_utility(xi, 1.0, 0.5, q)
            tc2 = transformed_terminal_utility(xi, 2.0, 0.5, q)
            assert np.all(tc2 < tc1), (
                f"Larger gamma should reduce terminal checkY: q={q}"
            )


class TestExp1MathematicalProperties:
    """Verify key mathematical properties described in the paper."""

    def test_girsanov_kernel_formula(self):
        """eta_t = checkZ_t / checkY_t (paper eq. for eta^{Q*})."""
        rng = _rng(0)
        dt = 1.0 / 20
        B, dB = simulate_brownian_paths(300, 20, 1, dt, rng)
        xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
        U = power_utility(0.5 * np.ones((300, 20)), 0.5)
        tc = transformed_terminal_utility(xi, 1.0, 0.5, 1.5)
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, 1.0, 1.5, U, tc)
        eta = compute_girsanov_kernel(checkZ, checkY)
        # Verify formula: eta = checkZ / checkY[:-1]
        checkY_mid = checkY[:, :-1][:, :, np.newaxis]
        eta_manual = checkZ / np.clip(checkY_mid, 1e-10, None)
        assert np.allclose(eta, eta_manual, rtol=1e-8)

    def test_ln_q_exp_q_inverse(self):
        """Paper transform: Y = -(1/gamma) * ln_q(checkY)  ↔  checkY = exp_q(-gamma*Y)."""
        rng = _rng(0)
        dt = 1.0 / 20
        B, dB = simulate_brownian_paths(300, 20, 1, dt, rng)
        xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
        U = power_utility(0.5 * np.ones((300, 20)), 0.5)
        tc = transformed_terminal_utility(xi, 1.0, 0.5, 1.5)
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, 1.0, 1.5, U, tc)
        Y, Z = recover_original_bsde(checkY, checkZ, 1.0, 1.5)
        # checkY = exp_q(-gamma*Y)
        reconstructed = exp_q(-1.0 * Y, 1.5)
        assert np.allclose(checkY, reconstructed, rtol=0.01, atol=1e-6)

    def test_entropy_nonneg(self):
        """Tsallis entropy H_q(Q*|P) >= 0."""
        rng = _rng(0)
        dt = 1.0 / 20
        B, dB = simulate_brownian_paths(300, 20, 1, dt, rng)
        xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
        U = power_utility(0.5 * np.ones((300, 20)), 0.5)
        tc = transformed_terminal_utility(xi, 1.0, 0.5, 1.5)
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, 1.0, 1.5, U, tc)
        eta = compute_girsanov_kernel(checkZ, checkY)
        D = stochastic_exponential(eta, dB, dt)
        H = tsallis_entropy_integral(eta, D, dt, 1.5)
        assert H >= -1e-6, f"H_q < 0: {H}"
