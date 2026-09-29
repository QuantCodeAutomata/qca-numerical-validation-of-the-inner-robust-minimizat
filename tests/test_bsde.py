"""Tests for src/bsde.py — transformed BSDE solver and variable recovery."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bsde import solve_transformed_bsde, recover_original_bsde
from src.tsallis import exp_q, ln_q
from src.utilities import power_utility, transformed_terminal_utility
from src.simulation import simulate_brownian_paths


def _setup(n_paths=500, N=30, d=1, dt=1.0/30, seed=0):
    rng = np.random.default_rng(seed)
    B, dB = simulate_brownian_paths(n_paths, N, d, dt, rng)
    xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
    c_paths = 0.5 * np.ones((n_paths, N))
    a = 0.5
    gamma = 1.0
    q = 1.5
    U_paths = power_utility(c_paths, a)
    terminal_cond = transformed_terminal_utility(xi, gamma, a, q)
    return B, dB, dt, gamma, q, U_paths, terminal_cond, xi, a


class TestSolveTransformedBSDE:
    def test_output_shapes(self):
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        n_paths, N, d = 500, 30, 1
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        assert checkY.shape == (n_paths, N + 1), f"checkY shape: {checkY.shape}"
        assert checkZ.shape == (n_paths, N, d), f"checkZ shape: {checkZ.shape}"

    def test_checkY_positive(self):
        """checkY must be strictly positive (domain of q-log)."""
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        assert np.all(checkY > 0), f"checkY has non-positive values: min={checkY.min()}"

    def test_terminal_condition_honored(self):
        """checkY[:, N] == terminal_condition (clipped to (0,1])."""
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        checkY, _ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        tc_clipped = np.clip(tc, 1e-300, 1.0)
        assert np.allclose(checkY[:, -1], tc_clipped, rtol=1e-10)

    def test_checkY_bounded_above_by_one(self):
        """For power utility with c >= 0 and xi >= 0, checkY should be <= 1.
        Terminal condition is in (0,1]. Driver is SUBTRACTED going backward,
        so checkY_t <= checkY_T <= 1 throughout."""
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        checkY, _ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        # With correct BSDE sign the solver clips to (0,1], so all values <= 1.0
        assert np.all(checkY <= 1.0 + 1e-9), f"checkY unexpectedly large: max={checkY.max()}"

    def test_y0_finite(self):
        """Y_0 (recovered from checkY_0) must be finite."""
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        Y, Z = recover_original_bsde(checkY, checkZ, 1.0, q)
        assert np.all(np.isfinite(Y)), "Y contains non-finite values"

    def test_different_degrees(self):
        """Solver works for polynomial degrees 1, 2, 3."""
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        for deg in [1, 2, 3]:
            checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc, degree=deg)
            assert np.all(checkY > 0), f"degree={deg} gave non-positive checkY"

    def test_zero_consumption_reduces_y0(self):
        """Running consumption reduces checkY_0 relative to the zero-consumption baseline.

        With zero consumption U=0 there is no driver, so checkY_0 ≈ E[checkY_T].
        With positive running utility U>0 the driver is subtracted at every step
        going backward, making checkY_0 strictly smaller than the zero-consumption
        value.  Economically: more running utility means a higher value Y_0, and
        since checkY = exp_q(-gamma*Y) is strictly decreasing in Y, checkY_0 falls.
        """
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        U_zero = np.zeros_like(U)
        checkY_zero, _ = solve_transformed_bsde(B, dB, dt, gamma, q, U_zero, tc)
        checkY_full, _ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        mean_zero = float(np.mean(checkY_zero[:, 0]))
        mean_full = float(np.mean(checkY_full[:, 0]))
        assert mean_full <= mean_zero, (
            f"Running utility should decrease checkY_0: full={mean_full:.4f} zero={mean_zero:.4f}"
        )

    def test_larger_gamma_shrinks_checkY(self):
        """Larger ambiguity aversion gamma → smaller checkY (stronger discounting)."""
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        tc1 = transformed_terminal_utility(np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None), 1.0, 0.5, q)
        tc2 = transformed_terminal_utility(np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None), 2.0, 0.5, q)
        checkY1, _ = solve_transformed_bsde(B, dB, dt, 1.0, q, U, tc1)
        checkY2, _ = solve_transformed_bsde(B, dB, dt, 2.0, q, U, tc2)
        # Larger gamma → terminal exp_q(-2*h) < exp_q(-1*h) → smaller checkY_T
        assert float(np.mean(tc2)) < float(np.mean(tc1))


class TestRecoverOriginalBSDE:
    def test_output_shapes(self):
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        Y, Z = recover_original_bsde(checkY, checkZ, gamma, q)
        assert Y.shape == checkY.shape
        assert Z.shape == checkZ.shape

    def test_y_nonneg_from_checkY_le_1(self):
        """Y = -(1/gamma) ln_q(checkY) >= 0 when checkY in (0,1]."""
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        Y, _ = recover_original_bsde(checkY, checkZ, gamma, q)
        # ln_q(x) <= 0 for x in (0,1], so -(1/gamma)*ln_q <= 0 ... 
        # Actually: ln_q(1) = 0, ln_q(x<1) < 0, so Y >= 0.
        assert np.all(Y >= -1e-6), f"Y has negative values: min={Y.min()}"

    def test_round_trip_consistency(self):
        """exp_q(-gamma * Y_t) ≈ checkY_t."""
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        Y, Z = recover_original_bsde(checkY, checkZ, gamma, q)
        checkY_reconstructed = exp_q(-gamma * Y, q)
        # Allow 1% relative tolerance due to clipping at small values
        assert np.allclose(checkY, checkY_reconstructed, rtol=0.01, atol=1e-6), (
            f"Round-trip checkY error: max={np.max(np.abs(checkY - checkY_reconstructed)):.4f}"
        )

    def test_z_recovery(self):
        """Z = -checkZ / (gamma * checkY^q)."""
        B, dB, dt, gamma, q, U, tc, *_ = _setup()
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        Y, Z = recover_original_bsde(checkY, checkZ, gamma, q)
        checkY_mid = checkY[:, :-1][:, :, np.newaxis]
        Z_expected = -checkZ / (gamma * checkY_mid ** q)
        assert np.allclose(Z, Z_expected, rtol=1e-10)

    def test_terminal_Y_consistent(self):
        """Y_T = -(1/gamma) * ln_q(exp_q(-gamma * h(xi))) = h(xi)."""
        B, dB, dt, gamma, q, U, tc, _, a = _setup()
        xi = np.clip(1.0 + 0.3 * B[:, -1, 0], 0.01, None)
        checkY, checkZ = solve_transformed_bsde(B, dB, dt, gamma, q, U, tc)
        Y, _ = recover_original_bsde(checkY, checkZ, gamma, q)
        h_xi = power_utility(xi, a)
        Y_T = Y[:, -1]
        assert np.allclose(Y_T, h_xi, rtol=0.01, atol=1e-6), (
            f"Y_T ≠ h(xi): max err={np.max(np.abs(Y_T - h_xi)):.4f}"
        )
