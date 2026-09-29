# Numerical Validation of the Inner Robust Minimization and Quadratic BSDE Representation under Tsallis Relative Entropy

A Python repository implementing and numerically validating three experiments from the paper on robust consumption-investment problems under Tsallis relative entropy. The model uses backward stochastic differential equations (BSDEs) and Girsanov-kernel density construction to verify the paper's inner-robust control theorem and maximum-principle optimality conditions.

## Overview

This project validates, for a complete continuous-time market, that:

1. **Exp 1 (Inner Robust Minimization)** — For a fixed admissible (c, ξ) pair under Tsallis entropy with parameter q > 1, the BSDE solution value Y₀ equals the robust infimum F(Q\*; c, ξ), and the minimizing measure Q\* is constructed from the BSDE diffusion term Z.

2. **Exp 2 (Maximum Principle / FOC)** — The outer robust optimization problem satisfies first-order conditions −γ(Ȳ₀)ᵍ(D*ₜ)ᵍ u′(c*ₜ) = v·D̃ₜ and similar at terminal time, verified via feedback-formula control construction and Lagrange-multiplier root-finding.

3. **Exp 3 (Coupled FBSDE)** — The full optimality system (forward wealth SDE + backward value BSDE + density adjoint) admits a solution satisfying budget saturation, terminal consistency, transform identities, and density normalization; uniqueness is validated numerically via multiple initializations.

## Repository Structure

```
.
├── data/                    # Placeholder (all data is simulated)
├── exp/
│   ├── exp_1.py             # Inner robust minimization experiment
│   ├── exp_2.py             # Maximum principle / FOC experiment
│   └── exp_3.py             # Coupled FBSDE experiment
├── results/
│   └── RESULTS.md           # Summarized metrics and findings
├── src/
│   ├── tsallis.py           # q-logarithm, q-exponential, Tsallis entropy
│   ├── utilities.py         # Power utility, derivatives, inverse marginal utility
│   ├── simulation.py        # Brownian path simulation, martingale density
│   ├── market.py            # Market coefficients and admissible controls
│   ├── bsde.py              # Transformed BSDE solver (LSMC) and recovery
│   ├── measures.py          # Girsanov kernel, stochastic exponential, objective
│   ├── transforms.py        # Transform-consistency checks, mu positivity
│   ├── diagnostics.py       # Equality gaps, FOC residuals, perturbation tests
│   ├── optimizer_outer.py   # Feedback controls, budget computation, multiplier search
│   ├── wealth.py            # Wealth-process reconstruction, terminal consistency
│   └── fbsde_solver.py      # Coupled FBSDE fixed-point iteration
├── tests/
│   ├── test_tsallis.py
│   ├── test_utilities.py
│   ├── test_simulation.py
│   ├── test_bsde.py
│   ├── test_measures.py
│   ├── test_diagnostics.py
│   ├── test_exp_1.py
│   ├── test_exp_2.py
│   └── test_exp_3.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Key Mathematical Components

| Symbol | Description |
|--------|-------------|
| `ln_q(x, q)` | Tsallis q-logarithm: (x^(1−q) − 1)/(1−q) |
| `exp_q(x, q)` | Tsallis q-exponential: [1+(1−q)x]^(1/(1−q)) |
| `checkY, checkZ` | Solution of the *transformed* BSDE (driver: −γ checkY^q u(c)) |
| `Y, Z` | Solution of the *original* quadratic BSDE, recovered via transform |
| `η*_t = checkZ_t / checkY_t` | Girsanov kernel for the minimizing measure Q\* |
| `D*_t` | Radon-Nikodym density of Q\* w.r.t. P (stochastic exponential) |
| `H_q(Q\*\|P)` | Tsallis relative entropy of Q\* |
| `F(Q\*; c, ξ)` | Inner robust objective evaluated at Q\* |
| `μ(y) = (1/q)(1−(1−q)γy)` | Auxiliary function for the original quadratic BSDE |

## Default Model Parameters

- Utility: power utility u(c) = h(c) = c^a / a with a = 0.5
- Time horizon T = 1, time steps N = 50
- Monte Carlo paths: 5000
- Market: 1-dimensional constant coefficients, θ = 0.3
- Tsallis parameters: q ∈ {1.05, 1.2, 1.5, 2.0}, γ ∈ {0.5, 1.0, 2.0}

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Run all three experiments
python exp/exp_1.py
python exp/exp_2.py
python exp/exp_3.py

# Run the test suite
pytest tests/ -v
```

Results and figures are saved to `results/`.

## Dependencies

- numpy ≥ 1.26
- scipy ≥ 1.14
- pandas ≥ 2.2
- matplotlib ≥ 3.8
- scikit-learn ≥ 1.5
- hmmlearn ≥ 0.3.2
- seaborn ≥ 0.13

## Notes on Numerical Methods

- **BSDE Solver**: Least-Squares Monte Carlo (LSMC) with polynomial regression on the current Brownian state B_t. Picard iteration handles the implicit driver γ checkY^q u(c).
- **Stochastic Exponential**: Computed in log-domain for numerical stability.
- **Multiplier Search**: `scipy.optimize.brentq` root-finding on the budget-gap function G(v) = X₀(v) − x.
- **Conditional Expectations**: Regressed on polynomial basis {1, B_t, B_t²} of the current state.

## Citation

This code validates the theoretical results in the paper:
> *Robust Consumption-Investment Problems under Tsallis Relative Entropy* (working paper)
