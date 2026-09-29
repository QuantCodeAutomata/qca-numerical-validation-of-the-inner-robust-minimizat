"""Tests for src/utilities.py — power utility, derivatives, and inverses."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utilities import (
    power_utility,
    power_utility_deriv,
    power_utility_derivative,
    inverse_marginal_utility,
    transformed_terminal_utility,
)
from src.tsallis import exp_q, ln_q


class TestPowerUtility:
    def test_positive_output(self):
        """u(c) >= 0 for c > 0, a in (0,1)."""
        c = np.array([0.5, 1.0, 2.0, 10.0])
        for a in [0.3, 0.5, 0.8]:
            result = power_utility(c, a)
            assert np.all(result >= 0)

    def test_zero_at_zero(self):
        """u(0) = 0 for power utility with a > 0."""
        c = np.array([0.0])
        for a in [0.3, 0.5, 0.8]:
            result = power_utility(c, a)
            assert result.item() == pytest.approx(0.0, abs=1e-10)

    def test_formula_correctness(self):
        """u(c) = c^a / a."""
        c = np.array([1.0, 2.0, 3.0])
        a = 0.5
        expected = c ** a / a
        result = power_utility(c, a)
        assert np.allclose(result, expected, rtol=1e-12)

    def test_strictly_increasing(self):
        """Power utility is strictly increasing in c."""
        c = np.linspace(0.01, 5.0, 50)
        for a in [0.3, 0.5, 0.8]:
            vals = power_utility(c, a)
            assert np.all(np.diff(vals) > 0)

    def test_strictly_concave(self):
        """Power utility is strictly concave: u''(c) < 0."""
        c = np.linspace(0.1, 5.0, 50)
        for a in [0.3, 0.5, 0.8]:
            vals = power_utility(c, a)
            second_diff = np.diff(np.diff(vals))
            assert np.all(second_diff < 0), f"Not strictly concave for a={a}"

    def test_sublinear_growth(self):
        """u(c) <= K * (1 + c^a) for large c (sublinear)."""
        c = np.array([100.0, 1000.0, 10000.0])
        a = 0.5
        K = 2.0
        u = power_utility(c, a)
        bound = K * (1.0 + c ** a)
        assert np.all(u <= bound)

    def test_2d_input(self):
        c = np.ones((100, 50))
        result = power_utility(c, 0.5)
        assert result.shape == (100, 50)
        assert np.allclose(result, 1.0 / 0.5)


class TestPowerUtilityDeriv:
    def test_formula(self):
        """u'(c) = c^(a-1)."""
        c = np.array([0.5, 1.0, 2.0])
        a = 0.5
        expected = c ** (a - 1)
        result = power_utility_deriv(c, a)
        assert np.allclose(result, expected, rtol=1e-12)

    def test_positive(self):
        """Derivative is positive for c > 0."""
        c = np.linspace(0.1, 5.0, 20)
        for a in [0.3, 0.5, 0.8]:
            result = power_utility_deriv(c, a)
            assert np.all(result > 0)

    def test_alias_consistency(self):
        """power_utility_derivative is an alias for power_utility_deriv."""
        c = np.array([1.0, 2.0, 3.0])
        a = 0.5
        assert np.allclose(power_utility_deriv(c, a), power_utility_derivative(c, a))

    def test_decreasing(self):
        """Marginal utility is decreasing (concavity)."""
        c = np.linspace(0.1, 5.0, 50)
        for a in [0.3, 0.5, 0.8]:
            d = power_utility_deriv(c, a)
            assert np.all(np.diff(d) < 0)


class TestInverseMarginalUtility:
    def test_inverse_of_deriv(self):
        """I_1(u'(c)) = c."""
        c = np.array([0.5, 1.0, 2.0, 5.0])
        for a in [0.3, 0.5, 0.8]:
            u_prime = power_utility_deriv(c, a)
            recovered = inverse_marginal_utility(u_prime, a)
            assert np.allclose(c, recovered, rtol=1e-8), (
                f"I_1(u'(c)) != c for a={a}"
            )

    def test_positive_output(self):
        """I_1(y) > 0 for y > 0."""
        y = np.array([0.1, 0.5, 1.0, 2.0])
        for a in [0.3, 0.5, 0.8]:
            result = inverse_marginal_utility(y, a)
            assert np.all(result > 0)

    def test_monotone_decreasing(self):
        """I_1 is decreasing in y (since u is concave → u' is decreasing)."""
        y = np.linspace(0.1, 5.0, 50)
        for a in [0.3, 0.5, 0.8]:
            result = inverse_marginal_utility(y, a)
            assert np.all(np.diff(result) < 0)


class TestTransformedTerminalUtility:
    def test_in_unit_interval(self):
        """g(xi) = exp_q(-gamma * h(xi)) in (0, 1] for xi >= 0."""
        xi = np.array([0.01, 0.5, 1.0, 2.0, 5.0])
        for q in [1.2, 1.5, 2.0]:
            for gamma in [0.5, 1.0, 2.0]:
                result = transformed_terminal_utility(xi, gamma, 0.5, q)
                assert np.all(result > 0) and np.all(result <= 1.0 + 1e-8), (
                    f"g(xi) not in (0,1]: q={q}, gamma={gamma}, result={result}"
                )

    def test_decreasing_in_xi(self):
        """g(xi) = exp_q(-gamma * h(xi)) is decreasing in xi > 0."""
        xi = np.linspace(0.01, 5.0, 50)
        for q in [1.2, 1.5, 2.0]:
            result = transformed_terminal_utility(xi, 1.0, 0.5, q)
            assert np.all(np.diff(result) < 0), f"g not decreasing for q={q}"

    def test_consistency_with_tsallis(self):
        """g(xi) = exp_q(-gamma * h(xi)) matches manual computation."""
        xi = np.array([0.5, 1.0, 2.0])
        gamma, a, q = 1.0, 0.5, 1.5
        h_xi = power_utility(xi, a)
        expected = exp_q(-gamma * h_xi, q)
        result = transformed_terminal_utility(xi, gamma, a, q)
        assert np.allclose(result, expected, rtol=1e-12)

    def test_at_zero_xi(self):
        """g(0) = exp_q(0) = 1."""
        xi = np.array([1e-10])
        for q in [1.2, 1.5, 2.0]:
            result = transformed_terminal_utility(xi, 1.0, 0.5, q)
            assert result.item() <= 1.0 + 1e-6
