"""Tests for src/simulation.py and src/market.py — path simulation and densities."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation import simulate_brownian_paths, compute_martingale_density
from src.market import make_admissible_controls


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestSimulateBrownianPaths:
    def test_output_shapes(self):
        rng = _rng(0)
        n_paths, N, d, dt = 500, 40, 1, 0.025
        B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)
        assert B.shape == (n_paths, N + 1, d), f"B shape: {B.shape}"
        assert dB.shape == (n_paths, N, d), f"dB shape: {dB.shape}"

    def test_starts_at_zero(self):
        """B[:, 0, :] == 0."""
        rng = _rng(1)
        B, _ = simulate_brownian_paths(200, 20, 1, 0.05, rng)
        assert np.allclose(B[:, 0, :], 0.0)

    def test_increments_variance(self):
        """dB increments ~ N(0, dt) per dimension."""
        rng = _rng(2)
        n_paths, N, d, dt = 5000, 1, 1, 1.0
        B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)
        var_empirical = float(np.var(dB[:, 0, 0]))
        # Should be close to dt within MC error
        assert abs(var_empirical - dt) < 0.1, f"Increment variance: {var_empirical}"

    def test_increments_are_consistent_with_paths(self):
        """dB[:, n, :] == B[:, n+1, :] - B[:, n, :]."""
        rng = _rng(3)
        B, dB = simulate_brownian_paths(200, 30, 2, 1.0 / 30, rng)
        dB_reconstructed = np.diff(B, axis=1)
        assert np.allclose(dB, dB_reconstructed, atol=1e-12)

    def test_multidimensional(self):
        rng = _rng(4)
        B, dB = simulate_brownian_paths(100, 20, 3, 0.05, rng)
        assert B.shape == (100, 21, 3)
        assert dB.shape == (100, 20, 3)

    def test_cumulative_sum(self):
        """B[:, n, :] == cumsum of increments up to n."""
        rng = _rng(5)
        B, dB = simulate_brownian_paths(100, 20, 1, 0.05, rng)
        B_from_dB = np.concatenate(
            [np.zeros((100, 1, 1)), np.cumsum(dB, axis=1)], axis=1
        )
        assert np.allclose(B, B_from_dB, atol=1e-12)

    def test_zero_mean(self):
        """E[B_T] ~ 0 for large n_paths."""
        rng = _rng(6)
        B, _ = simulate_brownian_paths(10000, 50, 1, 0.02, rng)
        mean_BT = float(np.mean(B[:, -1, 0]))
        assert abs(mean_BT) < 0.1, f"Mean B_T too far from 0: {mean_BT}"


class TestMartingaleDensity:
    def test_output_shape(self):
        rng = _rng(0)
        n_paths, N, d, dt = 300, 30, 1, 1.0 / 30
        B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)
        theta = 0.2 * np.ones((n_paths, N, d))
        tildeD = compute_martingale_density(theta, dB, dt)
        assert tildeD.shape == (n_paths, N + 1)

    def test_starts_at_one(self):
        rng = _rng(1)
        B, dB = simulate_brownian_paths(200, 20, 1, 0.05, rng)
        theta = 0.3 * np.ones((200, 20, 1))
        tildeD = compute_martingale_density(theta, dB, 0.05)
        assert np.allclose(tildeD[:, 0], 1.0)

    def test_normalized_expectation(self):
        """E[tildeD_T] ≈ 1 for large n_paths (martingale property)."""
        rng = _rng(2)
        n_paths, N, d, dt = 5000, 50, 1, 0.02
        B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)
        theta = 0.5 * np.ones((n_paths, N, d))
        tildeD = compute_martingale_density(theta, dB, dt)
        mean_T = float(np.mean(tildeD[:, -1]))
        assert abs(mean_T - 1.0) < 0.05, f"E[tildeD_T] = {mean_T}"

    def test_stays_positive(self):
        """tildeD_t > 0 for all paths and times."""
        rng = _rng(3)
        B, dB = simulate_brownian_paths(200, 20, 1, 0.05, rng)
        theta = 0.5 * np.ones((200, 20, 1))
        tildeD = compute_martingale_density(theta, dB, 0.05)
        assert np.all(tildeD > 0)

    def test_zero_theta_gives_one(self):
        """theta=0 → tildeD_t = 1 for all t."""
        rng = _rng(4)
        B, dB = simulate_brownian_paths(200, 20, 1, 0.05, rng)
        theta = np.zeros((200, 20, 1))
        tildeD = compute_martingale_density(theta, dB, 0.05)
        assert np.allclose(tildeD, 1.0, atol=1e-12)

    def test_monotone_product(self):
        """tildeD is a product martingale — each step is positive multiplicative."""
        rng = _rng(5)
        B, dB = simulate_brownian_paths(50, 10, 1, 0.1, rng)
        theta = 0.3 * np.ones((50, 10, 1))
        tildeD = compute_martingale_density(theta, dB, 0.1)
        # Consecutive ratios should be positive
        ratios = tildeD[:, 1:] / tildeD[:, :-1]
        assert np.all(ratios > 0)


class TestMakeAdmissibleControls:
    def test_output_positivity(self):
        """Consumption c_t >= 0 and terminal xi >= 0."""
        rng = _rng(0)
        B, dB = simulate_brownian_paths(200, 20, 1, 0.05, rng)
        c_paths, xi = make_admissible_controls(B, dB, 0.05)
        assert np.all(c_paths >= 0)
        assert np.all(xi >= 0)

    def test_output_shapes(self):
        rng = _rng(1)
        n_paths, N, d, dt = 300, 25, 1, 0.04
        B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)
        c_paths, xi = make_admissible_controls(B, dB, dt)
        assert c_paths.shape == (n_paths, N), f"c_paths shape: {c_paths.shape}"
        assert xi.shape == (n_paths,), f"xi shape: {xi.shape}"
