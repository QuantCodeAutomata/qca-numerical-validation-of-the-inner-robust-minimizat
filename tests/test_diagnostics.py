"""Tests for src/diagnostics.py — equality gap, normalization, FOC, perturbation."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.diagnostics import (
    compute_equality_gap,
    perturbation_test,
    normalization_check,
    compute_first_order_residuals,
    density_consistency_error,
)
from src.simulation import simulate_brownian_paths
from src.bsde import solve_transformed_bsde, recover_original_bsde
from src.measures import (
    compute_girsanov_kernel,
    stochastic_exponential,
    compute_inner_robust_objective,
    compute_density_closed_form,
)
from src.utilities import power_utility, power_utility_deriv, transformed_terminal_utility
from src.tsallis import tsallis_entropy_integral


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _full_pipeline(n_paths=400, N=25, dt=1.0/25, gamma=1.0, q=1.5, a=0.5, seed=0):
    """Helper: run the full inner BSDE pipeline, return all key tensors."""
    rng = _rng(seed)
    B, dB = simulate_brownian_paths(n_paths, N, 1, dt, rng)
    xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
    c_paths = 0.5 * np.ones((n_paths, N))
    U = power_utility(c_paths, a)
    h_xi = power_utility(xi, a)
    tc = transformed_terminal_utility(xi, gamma, a, q)
    checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
    Y, Z = recover_original_bsde(checkY, checkZ, gamma, q)
    eta = compute_girsanov_kernel(checkZ, checkY)
    D = stochastic_exponential(eta, dB, dt)
    H = tsallis_entropy_integral(eta, D, dt, q)
    F = compute_inner_robust_objective(D, h_xi, U, dt, q, H, gamma)
    Y_0 = float(np.mean(Y[:, 0]))
    return dict(
        B=B, dB=dB, dt=dt, gamma=gamma, q=q, a=a,
        xi=xi, c_paths=c_paths, U=U, h_xi=h_xi, tc=tc,
        checkY=checkY, checkZ=checkZ, Y=Y, Z=Z,
        eta=eta, D=D, H=H, F=F, Y_0=Y_0
    )


class TestComputeEqualityGap:
    def test_zero_gap(self):
        gap = compute_equality_gap(1.0, 1.0)
        assert gap["abs_gap"] == 0.0
        assert gap["rel_gap"] == 0.0

    def test_returns_dict_keys(self):
        gap = compute_equality_gap(1.0, 1.5)
        assert "abs_gap" in gap
        assert "rel_gap" in gap

    def test_abs_gap_positive(self):
        """Absolute gap is always non-negative."""
        for Y0, F in [(1.0, 2.0), (3.0, 1.5), (0.5, 0.5)]:
            gap = compute_equality_gap(Y0, F)
            assert gap["abs_gap"] >= 0.0

    def test_correct_values(self):
        gap = compute_equality_gap(2.0, 3.0)
        assert abs(gap["abs_gap"] - 1.0) < 1e-12
        assert abs(gap["rel_gap"] - 0.5) < 1e-12  # 1.0 / 2.0

    def test_full_pipeline(self):
        """Gap from full BSDE pipeline is finite."""
        p = _full_pipeline()
        gap = compute_equality_gap(p["Y_0"], p["F"])
        assert np.isfinite(gap["abs_gap"])
        assert np.isfinite(gap["rel_gap"])


class TestNormalizationCheck:
    def test_ones_gives_zero_error(self):
        D_T = np.ones(1000)
        result = normalization_check(D_T)
        assert abs(result["norm_error"]) < 1e-10

    def test_returns_dict_keys(self):
        D_T = np.ones(500)
        result = normalization_check(D_T)
        assert "norm_error" in result
        assert "mean_D_T" in result

    def test_norm_error_nonneg(self):
        """Normalization error |E[D_T] - 1| >= 0."""
        rng = _rng(0)
        D_T = rng.lognormal(0.1, 0.5, 1000)  # biased away from 1
        result = normalization_check(D_T)
        assert result["norm_error"] >= 0.0

    def test_full_pipeline_norm(self):
        p = _full_pipeline()
        result = normalization_check(p["D"][:, -1])
        assert "norm_error" in result
        assert np.isfinite(result["norm_error"])


class TestPerturbationTest:
    """Tests for perturbation_test(checkY, checkZ, dB, dt, h_xi, u_c, gamma, q,
                                   delta_eta, lambdas)."""

    def test_returns_expected_keys(self):
        p = _full_pipeline(n_paths=200, N=15, dt=1.0/15, seed=5)
        delta_eta = 0.05 * np.ones_like(p["eta"])
        result = perturbation_test(
            p["checkY"], p["checkZ"], p["dB"], p["dt"],
            p["h_xi"], p["U"], p["gamma"], p["q"],
            delta_eta, lambdas=[-0.1, 0.0, 0.1]
        )
        assert isinstance(result, dict)
        assert "F_star" in result
        assert "objectives" in result
        assert "lambdas" in result
        assert "is_minimizer" in result

    def test_lambda_zero_matches_F_star(self):
        """At lambda=0 the perturbed objective equals F*."""
        p = _full_pipeline(n_paths=200, N=15, seed=3)
        delta_eta = 0.05 * np.ones_like(p["eta"])
        result = perturbation_test(
            p["checkY"], p["checkZ"], p["dB"], p["dt"],
            p["h_xi"], p["U"], p["gamma"], p["q"],
            delta_eta, lambdas=[0.0]
        )
        assert abs(result["objectives"][0] - result["F_star"]) < 1e-10

    def test_objectives_length_matches_lambdas(self):
        p = _full_pipeline(n_paths=200, N=15, seed=2)
        delta_eta = 0.05 * np.ones_like(p["eta"])
        lambdas = [-0.5, -0.2, 0.0, 0.2, 0.5]
        result = perturbation_test(
            p["checkY"], p["checkZ"], p["dB"], p["dt"],
            p["h_xi"], p["U"], p["gamma"], p["q"],
            delta_eta, lambdas=lambdas
        )
        assert len(result["objectives"]) == len(lambdas)

    def test_small_perturbations_near_baseline(self):
        """Very small perturbations should not drastically change the objective."""
        p = _full_pipeline(n_paths=300, N=20, dt=0.05, seed=1)
        delta_eta = 0.001 * np.ones_like(p["eta"])
        result = perturbation_test(
            p["checkY"], p["checkZ"], p["dB"], p["dt"],
            p["h_xi"], p["U"], p["gamma"], p["q"],
            delta_eta, lambdas=[-0.01, 0.01]
        )
        base = result["F_star"]
        for obj in result["objectives"]:
            assert abs(obj - base) < 5.0, (
                f"Tiny perturbation moved objective too far: {obj:.4f} vs {base:.4f}"
            )


class TestComputeFirstOrderResiduals:
    """Tests for compute_first_order_residuals returning terminal_mae, consumption_mae, etc."""

    def test_returns_dict_with_required_keys(self):
        p = _full_pipeline()
        barY_0 = float(np.mean(p["checkY"][:, 0]))
        D_star = p["D"]
        h_prime = power_utility_deriv(p["xi"], p["a"])
        u_prime = power_utility_deriv(p["c_paths"], p["a"])
        tildeD = np.ones_like(D_star)
        v = -0.5
        result = compute_first_order_residuals(
            barY_0, D_star, h_prime, u_prime, tildeD, v, p["gamma"], p["q"]
        )
        for key in ["terminal_mae", "terminal_rmse", "terminal_max",
                    "consumption_mae", "consumption_rmse", "consumption_max"]:
            assert key in result, f"Key missing: {key}"
        assert result["terminal_mae"] >= 0.0
        assert result["consumption_mae"] >= 0.0

    def test_residuals_finite(self):
        p = _full_pipeline()
        barY_0 = float(np.mean(p["checkY"][:, 0]))
        D_star = p["D"]
        h_prime = power_utility_deriv(p["xi"], p["a"])
        u_prime = power_utility_deriv(p["c_paths"], p["a"])
        tildeD = np.ones_like(D_star)
        v = -0.5
        result = compute_first_order_residuals(
            barY_0, D_star, h_prime, u_prime, tildeD, v, p["gamma"], p["q"]
        )
        for key in ["terminal_mae", "terminal_rmse", "consumption_mae", "consumption_rmse"]:
            assert np.isfinite(result[key]), f"Key {key} not finite: {result[key]}"


class TestDensityConsistencyError:
    """Tests for density_consistency_error(D_stochastic_exp, D_closed_form)."""

    def test_identical_inputs_zero_error(self):
        rng = _rng(0)
        D = rng.uniform(0.5, 1.5, (200, 26))
        result = density_consistency_error(D, D)
        assert result["density_max_abs_err"] == 0.0
        assert result["density_mean_abs_err"] == 0.0

    def test_returns_required_keys(self):
        D1 = np.ones((100, 10))
        D2 = 1.01 * np.ones((100, 10))
        result = density_consistency_error(D1, D2)
        assert "density_max_abs_err" in result
        assert "density_mean_abs_err" in result

    def test_nonneg_errors(self):
        rng = _rng(1)
        D1 = rng.uniform(0.8, 1.2, (100, 10))
        D2 = rng.uniform(0.8, 1.2, (100, 10))
        result = density_consistency_error(D1, D2)
        assert result["density_max_abs_err"] >= 0.0
        assert result["density_mean_abs_err"] >= 0.0

    def test_full_pipeline_consistency(self):
        """Closed-form vs stochastic-exponential density error is finite and non-negative."""
        p = _full_pipeline()
        D_cf = compute_density_closed_form(p["checkY"], p["U"], p["dt"], p["gamma"], p["q"])
        result = density_consistency_error(p["D"], D_cf)
        assert np.isfinite(result["density_max_abs_err"])
        assert result["density_max_abs_err"] >= 0.0
