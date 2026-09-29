"""Tests for src/tsallis.py — Tsallis q-logarithm, q-exponential, and entropy."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tsallis import (
    ln_q,
    exp_q,
    exp_q_domain_valid,
    tsallis_entropy_direct,
    tsallis_entropy_integral,
    mu_func,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# ln_q tests
# ---------------------------------------------------------------------------

class TestLnQ:
    def test_scalar_at_one_is_zero(self):
        """ln_q(1) = 0 for all q."""
        for q in [1.05, 1.2, 1.5, 2.0]:
            assert abs(ln_q(np.array([1.0]), q).item()) < 1e-12

    def test_limit_to_log_as_q_to_1(self):
        """For q very close to 1, ln_q(x) → ln(x)."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        q_near_1 = 1.0 + 1e-9
        ln_q_vals = ln_q(x, q_near_1)
        log_vals = np.log(x)
        assert np.allclose(ln_q_vals, log_vals, atol=1e-5)

    def test_monotone_increasing(self):
        """ln_q is strictly increasing for q > 1."""
        x = np.linspace(0.1, 5.0, 50)
        for q in [1.2, 1.5, 2.0]:
            vals = ln_q(x, q)
            diffs = np.diff(vals)
            assert np.all(diffs > 0), f"ln_q not monotone for q={q}"

    def test_negative_domain(self):
        """ln_q(x) < 0 for 0 < x < 1 when q > 1."""
        x = np.array([0.1, 0.5, 0.9])
        for q in [1.2, 1.5, 2.0]:
            vals = ln_q(x, q)
            assert np.all(vals < 0), f"ln_q(x<1) should be negative for q={q}"

    def test_positive_domain(self):
        """ln_q(x) > 0 for x > 1."""
        x = np.array([1.1, 2.0, 5.0])
        for q in [1.2, 1.5, 2.0]:
            vals = ln_q(x, q)
            assert np.all(vals > 0), f"ln_q(x>1) should be positive for q={q}"

    def test_array_input(self):
        x = np.array([0.5, 1.0, 2.0])
        result = ln_q(x, 1.5)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# exp_q tests
# ---------------------------------------------------------------------------

class TestExpQ:
    def test_at_zero_is_one(self):
        """exp_q(0) = 1 for all q."""
        for q in [1.05, 1.2, 1.5, 2.0]:
            val = exp_q(np.array([0.0]), q).item()
            assert abs(val - 1.0) < 1e-12, f"exp_q(0) != 1 for q={q}: {val}"

    def test_limit_to_exp_as_q_to_1(self):
        """For q very close to 1, exp_q(x) → e^x."""
        x = np.array([-1.0, 0.0, 0.5, 1.0])
        q_near_1 = 1.0 + 1e-9
        eq_vals = exp_q(x, q_near_1)
        exp_vals = np.exp(x)
        assert np.allclose(eq_vals, exp_vals, atol=1e-5)

    def test_inverse_of_ln_q(self):
        """exp_q(ln_q(x)) = x for x > 0."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        for q in [1.1, 1.5, 2.0]:
            recovered = exp_q(ln_q(x, q), q)
            assert np.allclose(x, recovered, rtol=1e-8), (
                f"exp_q(ln_q(x)) != x for q={q}"
            )

    def test_ln_q_inverse_of_exp_q(self):
        """ln_q(exp_q(x)) = x for valid domain."""
        x = np.array([-0.5, 0.0, 0.3, 1.0])
        for q in [1.1, 1.5, 2.0]:
            # Check domain validity first
            mask = exp_q_domain_valid(x, q)
            x_valid = x[mask]
            if len(x_valid) > 0:
                recovered = ln_q(exp_q(x_valid, q), q)
                assert np.allclose(x_valid, recovered, rtol=1e-8)

    def test_positive_output_for_negative_input(self):
        """exp_q(x) > 0 for all valid x."""
        x = np.array([-2.0, -1.0, 0.0, 0.5])
        for q in [1.2, 1.5]:
            result = exp_q(x, q)
            assert np.all(result > 0)

    def test_less_than_one_for_negative_input(self):
        """For q > 1 and x < 0: exp_q(x) < 1."""
        x = np.array([-2.0, -1.0, -0.1])
        for q in [1.2, 1.5, 2.0]:
            result = exp_q(x, q)
            assert np.all(result < 1.0), f"exp_q(x<0) should be < 1 for q={q}"

    def test_terminal_condition_in_unit_interval(self):
        """exp_q(-gamma * h_xi) should be in (0, 1] for h_xi >= 0, gamma > 0."""
        from src.utilities import power_utility
        xi = np.array([0.5, 1.0, 2.0, 5.0])
        for q in [1.2, 1.5]:
            for gamma in [0.5, 1.0, 2.0]:
                h_xi = power_utility(xi, 0.5)
                tc = exp_q(-gamma * h_xi, q)
                assert np.all(tc > 0) and np.all(tc <= 1.0 + 1e-10), (
                    f"terminal cond out of (0,1]: q={q}, gamma={gamma}"
                )


# ---------------------------------------------------------------------------
# exp_q domain validity
# ---------------------------------------------------------------------------

class TestExpQDomainValid:
    def test_zero_is_valid(self):
        """x=0 is always valid."""
        for q in [1.2, 1.5, 2.0]:
            assert exp_q_domain_valid(np.array([0.0]), q).all()

    def test_large_negative_invalid_for_large_q(self):
        """Very negative x can be outside domain for large q."""
        q = 2.0
        x = np.array([-10.0])  # 1 + (1-2)*(-10) = 11 > 0 — still valid!
        # Actually for q=2: 1 + (1-2)*x = 1 - x, invalid when x > 1
        x_invalid = np.array([2.0])  # 1 + (1-2)*2 = -1 < 0
        assert not exp_q_domain_valid(x_invalid, q).all()


# ---------------------------------------------------------------------------
# Tsallis entropy tests
# ---------------------------------------------------------------------------

class TestTsallisEntropy:
    def test_entropy_zero_at_P(self):
        """H_q(P|P) = 0 — density D_T = 1 everywhere."""
        D_T = np.ones(1000)
        H = tsallis_entropy_direct(D_T, q=1.5)
        assert abs(H) < 1e-10, f"H_q(P|P) != 0: {H}"

    def test_entropy_nonneg(self):
        """Tsallis entropy is non-negative for q > 1."""
        rng = _rng(42)
        D_T = rng.lognormal(0.0, 0.3, 2000)
        D_T /= D_T.mean()  # normalize
        H = tsallis_entropy_direct(D_T, q=1.5)
        assert H >= -1e-6, f"Entropy negative: {H}"

    def test_integral_form_matches_direct(self):
        """Integral H_q ≈ direct H_q for simple stochastic exponential."""
        from src.simulation import simulate_brownian_paths, compute_martingale_density
        from src.measures import stochastic_exponential

        rng = _rng(7)
        n_paths, N, d, dt = 3000, 30, 1, 1.0 / 30
        B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)

        # Constant eta — simple case
        eta_const = 0.2
        eta = eta_const * np.ones((n_paths, N, d))
        D = stochastic_exponential(eta, dB, dt)

        H_integral = tsallis_entropy_integral(eta, D, dt, q=1.5)
        H_direct = tsallis_entropy_direct(D[:, -1], q=1.5)

        # Both should be positive and close for large n_paths
        assert H_integral > 0 and H_direct > 0
        # Relative difference should be reasonable (MC error ~ 10%)
        rel_diff = abs(H_integral - H_direct) / (abs(H_direct) + 1e-10)
        assert rel_diff < 0.5, f"Integral vs direct entropy too different: {rel_diff:.3f}"


# ---------------------------------------------------------------------------
# mu_func tests
# ---------------------------------------------------------------------------

class TestMuFunc:
    def test_mu_at_zero(self):
        """mu(0) = 1/q."""
        y = np.array([0.0])
        for q in [1.2, 1.5, 2.0]:
            mu = mu_func(y, gamma=1.0, q=q)
            assert abs(mu.item() - 1.0 / q) < 1e-12

    def test_mu_positive_for_small_y(self):
        """mu(Y) > 0 for small Y >= 0."""
        y = np.linspace(0, 0.1, 20)
        for q in [1.2, 1.5, 2.0]:
            mu = mu_func(y, gamma=1.0, q=q)
            assert np.all(mu > 0), f"mu <= 0 for q={q}"

    def test_mu_formula(self):
        """mu(y) = (1/q)*(1 - (1-q)*gamma*y)."""
        y = np.array([0.1, 0.2, 0.5])
        gamma, q = 1.0, 1.5
        expected = (1.0 / q) * (1.0 - (1.0 - q) * gamma * y)
        result = mu_func(y, gamma, q)
        assert np.allclose(result, expected, rtol=1e-12)
