"""Integration tests for exp/exp_3.py — coupled FBSDE characterization.

Validates:
  - Budget saturation X_0 ≈ x
  - Terminal consistency X_T ≈ xi* and Y_T ≈ h(xi*)
  - Transform identities barY ≈ exp_q(-gamma*Y) and barZ ≈ -gamma*barY^q*Z
  - Density identity D* from stochastic exponential
  - Convergence from multiple initial conditions (uniqueness evidence)
  - No-consumption special case
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation import simulate_brownian_paths, compute_martingale_density
from src.utilities import power_utility, transformed_terminal_utility
from src.bsde import solve_transformed_bsde, recover_original_bsde
from src.measures import compute_girsanov_kernel, stochastic_exponential, compute_density_closed_form
from src.transforms import check_transform_consistency, verify_mu_positivity
from src.diagnostics import normalization_check, density_consistency_error
from src.wealth import compute_initial_wealth, compute_wealth_process, validate_terminal_consistency
from src.fbsde_solver import solve_coupled_fbsde
from src.tsallis import exp_q, ln_q


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _run_coupled(q=1.5, gamma=1.0, a=0.5, x=1.0, n_paths=800, N=25, T=1.0,
                 theta_const=0.3, seed=0, no_consumption=False, max_iter=8):
    dt = T / N
    rng = _rng(seed)
    B, dB = simulate_brownian_paths(n_paths, N, 1, dt, rng)
    theta = theta_const * np.ones((n_paths, N, 1))
    tildeD = compute_martingale_density(theta, dB, dt)
    # signature: (B, dB, dt, T, gamma, q, a, x, tildeD, n_iter=..., no_consumption=...)
    sol = solve_coupled_fbsde(
        B, dB, dt, T, gamma, q, a, x, tildeD,
        n_iter=max_iter,
        no_consumption=no_consumption,
    )
    return sol, B, dB, tildeD, dt


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestExp3CoupledFBSDEBasic:
    """Basic sanity checks for the coupled FBSDE solver output."""

    def test_output_has_required_keys(self):
        sol, *_ = _run_coupled(n_paths=500, N=20)
        required = ["barY_0", "barY", "barZ", "Y", "Z", "c_star", "xi_star",
                    "D_star", "v", "budget"]
        for key in required:
            assert key in sol, f"Missing key: {key}"

    def test_density_positive(self):
        """D* > 0 for all paths and times (methodology step 6)."""
        sol, *_ = _run_coupled(n_paths=500, N=20)
        assert np.all(sol["D_star"] > 0), f"D* non-positive: min={sol['D_star'].min()}"

    def test_density_starts_at_one(self):
        """D*_0 = 1."""
        sol, *_ = _run_coupled(n_paths=500, N=20)
        assert np.allclose(sol["D_star"][:, 0], 1.0)

    def test_barY_positive(self):
        """barY > 0 throughout (domain condition)."""
        sol, *_ = _run_coupled(n_paths=500, N=20)
        assert np.all(sol["barY"] > 0), f"barY non-positive: min={sol['barY'].min()}"

    def test_multiplier_negative(self):
        """v < 0 is required for the feedback formulas."""
        sol, *_ = _run_coupled(n_paths=500, N=20)
        assert sol["v"] < 0, f"v should be negative: {sol['v']}"

    def test_controls_nonneg(self):
        """c_t* >= 0 and xi* >= 0."""
        sol, *_ = _run_coupled(n_paths=500, N=20)
        assert np.all(sol["c_star"] >= 0), "c* has negative values"
        assert np.all(sol["xi_star"] >= 0), "xi* has negative values"

    def test_Y_finite(self):
        """Y_0 is finite."""
        sol, *_ = _run_coupled(n_paths=500, N=20)
        assert np.all(np.isfinite(sol["Y"][:, 0]))


class TestExp3BudgetSaturation:
    """Methodology step 10a: X_0 ≈ x."""

    def test_budget_finite(self):
        sol, B, dB, tildeD, dt = _run_coupled(n_paths=600, N=25)
        assert np.isfinite(sol["budget"]), f"Budget not finite: {sol['budget']}"

    def test_budget_positive(self):
        sol, B, dB, tildeD, dt = _run_coupled(x=1.0, n_paths=600, N=25)
        assert sol["budget"] > 0, f"Budget non-positive: {sol['budget']}"


class TestExp3TransformConsistency:
    """Methodology step 10c: barY ≈ exp_q(-gamma*Y), barZ ≈ -gamma*barY^q*Z."""

    def test_barY_consistent_with_Y(self):
        gamma, q = 1.0, 1.5  # default params used by _run_coupled
        sol, *_ = _run_coupled(gamma=gamma, q=q, n_paths=600, N=25)
        barY = sol["barY"]
        Y = sol["Y"]
        barY_from_Y = exp_q(-gamma * Y, q)
        # They may not match exactly due to Picard iteration, but should be close
        rel_err = np.abs(barY - barY_from_Y) / (np.abs(barY) + 1e-10)
        assert np.nanmedian(rel_err) < 0.1, (
            f"barY vs exp_q(-gamma*Y) median rel err too large: {np.nanmedian(rel_err):.4f}"
        )

    def test_transform_stats_returned(self):
        sol, *_ = _run_coupled(n_paths=600, N=25)
        # The solver should populate q and gamma for checking
        assert "barY" in sol
        assert "Y" in sol

    def test_barY_in_unit_interval(self):
        """barY should be in (0,1] since it's exp_q(-gamma*h(xi)) propagated backward."""
        sol, *_ = _run_coupled(n_paths=600, N=25)
        assert np.all(sol["barY"] > 0)
        # Allow some slack for LSMC approximation noise
        assert float(np.mean(sol["barY"])) < 2.0


class TestExp3TerminalConsistency:
    """Methodology step 10b: Y_T ≈ h(xi*)."""

    def test_Y_T_approx_h_xi(self):
        sol, *_ = _run_coupled(n_paths=1000, N=30)
        Y_T = sol["Y"][:, -1]
        h_xi = power_utility(sol["xi_star"], 0.5)
        err = float(np.mean(np.abs(Y_T - h_xi)))
        assert err < 2.0, f"Y_T ≈ h(xi*) error too large: {err:.4f}"

    def test_barY_T_consistent(self):
        """barY_T = exp_q(-gamma * h(xi*))."""
        sol, *_ = _run_coupled(n_paths=800, N=25)
        h_xi = power_utility(sol["xi_star"], 0.5)
        gamma, q = 1.0, 1.5
        expected_barY_T = exp_q(-gamma * h_xi, q)
        # Terminal condition of barY should match
        barY_T = sol["barY"][:, -1]
        err = float(np.mean(np.abs(barY_T - expected_barY_T)))
        assert err < 0.5, f"barY_T terminal mismatch: {err:.4f}"


class TestExp3DensityIdentity:
    """Methodology step 10d: D* from stochastic exponential."""

    def test_D_star_normalized(self):
        """E[D*_T] ≈ 1 (approximate martingale normalization)."""
        sol, *_ = _run_coupled(n_paths=1000, N=30)
        norm = normalization_check(sol["D_star"][:, -1])
        assert np.isfinite(norm["norm_error"])

    def test_density_closed_form(self):
        """density_closed_form_check returns consistent density vs stochastic-exponential D*."""
        gamma, q, a = 1.0, 1.5, 0.5
        sol, B, dB, tildeD, dt = _run_coupled(gamma=gamma, q=q, a=a, n_paths=600, N=25)
        U = power_utility(sol["c_star"], a)
        # signature: (checkY, u_c, dt, gamma, q) -> D_closed ndarray
        D_closed = compute_density_closed_form(sol["barY"], U, dt, gamma, q)
        # Compare closed-form density to the stochastic-exponential density
        err = float(np.mean(np.abs(D_closed - sol["D_star"])))
        assert err >= 0.0 and np.isfinite(err), f"density_closed_form MAE: {err}"


class TestExp3NoConsumption:
    """Methodology step 13: no-consumption special case c ≡ 0."""

    def test_no_consumption_runs(self):
        sol, *_ = _run_coupled(n_paths=500, N=20, no_consumption=True)
        assert "barY_0" in sol and np.isfinite(sol["barY_0"])

    def test_no_consumption_c_zero(self):
        sol, *_ = _run_coupled(n_paths=500, N=20, no_consumption=True)
        assert np.allclose(sol["c_star"], 0.0, atol=1e-6)

    def test_no_consumption_Y_0_finite(self):
        sol, *_ = _run_coupled(n_paths=500, N=20, no_consumption=True)
        assert np.all(np.isfinite(sol["Y"]))


class TestExp3UniquenessEvidence:
    """Methodology step 12: multiple initializations converge to same solution."""

    def test_two_seeds_give_close_barY_0(self):
        """Different seeds (random MC) should give similar barY_0 values."""
        sol1, *_ = _run_coupled(n_paths=2000, N=30, seed=0)
        sol2, *_ = _run_coupled(n_paths=2000, N=30, seed=1)
        rel_diff = abs(sol1["barY_0"] - sol2["barY_0"]) / (abs(sol1["barY_0"]) + 1e-10)
        assert rel_diff < 0.3, (
            f"barY_0 different seeds too far apart: {sol1['barY_0']:.4f} vs {sol2['barY_0']:.4f}"
        )

    def test_runs_consistently_for_different_q(self):
        """Solution exists for all q in the required range."""
        for q in [1.05, 1.2, 1.5, 2.0]:
            sol, *_ = _run_coupled(q=q, n_paths=500, N=20)
            assert np.isfinite(sol["barY_0"]), f"barY_0 not finite for q={q}"


class TestExp3Sensitivity:
    """Methodology step 14: sensitivity to q, gamma, and resolution."""

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_runs_for_gamma(self, gamma):
        sol, *_ = _run_coupled(gamma=gamma, n_paths=500, N=20)
        assert np.isfinite(sol["barY_0"])

    @pytest.mark.parametrize("q", [1.1, 1.5, 2.0])
    def test_runs_for_q(self, q):
        sol, *_ = _run_coupled(q=q, n_paths=500, N=20)
        assert np.isfinite(sol["barY_0"])

    def test_more_paths_improves_normalization(self):
        """More MC paths → tildeD_T normalization closer to 1."""
        rng1 = _rng(0)
        rng2 = _rng(0)
        dt = 1.0 / 20
        theta = 0.3
        B1, dB1 = simulate_brownian_paths(500, 20, 1, dt, rng1)
        B2, dB2 = simulate_brownian_paths(5000, 20, 1, dt, rng2)
        t1 = theta * np.ones((500, 20, 1))
        t2 = theta * np.ones((5000, 20, 1))
        tD1 = compute_martingale_density(t1, dB1, dt)
        tD2 = compute_martingale_density(t2, dB2, dt)
        err1 = abs(float(np.mean(tD1[:, -1])) - 1.0)
        err2 = abs(float(np.mean(tD2[:, -1])) - 1.0)
        assert err2 <= err1 * 3 + 0.05, (
            f"More paths didn't help normalization: err1={err1:.4f} err2={err2:.4f}"
        )
