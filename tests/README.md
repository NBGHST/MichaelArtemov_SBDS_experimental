# Testing and Benchmarking Suite

This directory contains the automated testing and benchmarking suite for the Spatial Birth Death Simulator.

## 1. Benchmarking & Branch Comparison

### `compare_benchmarks.sh` (Root Directory)
A shell script that automates the benchmarking process across different Git branches (e.g., `cpp_opt` vs `extreme_opt`). It automatically clones the baseline branch into a temporary directory, compiles it, and runs the benchmarks alongside the current branch.

### `compare_branches.py`
The python script executed by `compare_benchmarks.sh`. It performs the following:
1. Clones the baseline branch (`cpp_opt`) and compiles it.
2. Generates short-horizon (2000 events) simulations using the old code.
3. Does the same for the current branch (`extreme_opt`).
4. **Validates correctness:** Ensures that over a short simulation horizon, both the baseline and optimized code yield the exact same total population and event counts (confirming that optimizations didn't break the MCMC discrete distribution sequences).
5. **Compares Performance:** Calculates speedups in Events/Sec across 1D, 2D, and 3D scenarios.

## 2. Correctness Testing

### `test_correctness.py`
The primary PyTest suite for verifying bitwise deterministic correctness.
Because simulation trajectories in continuous-time Markov chains diverge exponentially upon the slightest numerical difference, this test enforces extreme rigidity.
- It compares the current output against a pre-generated "golden" reference.
- Checks floating-point parameters up to 10 decimal places (`rtol=1e-10`).
- Validates the exact integer population counts, total birth/death rates, and exact particle coordinates at the end of the simulation.

To run the tests:
```bash
python3 -m pytest tests/test_correctness.py -v
```

### `generate_golden.py`
Generates the `golden_reference.json` file used by `test_correctness.py`.
- **Note:** If an optimization fundamentally changes the mathematical approximation (e.g., evaluating death kernels over $r^2$ instead of $r$), the simulation will mathematically diverge from the old reference after thousands of steps. In such cases, `generate_golden.py` must be used to snapshot the new mathematical state as the updated golden reference.

## 3. Worker Scripts

### `worker_metrics.py`
A low-level runner script that executes a predefined set of physical scenarios (scenarios 1-21 covering 1D, 2D, and 3D with various boundary conditions and species parameters). It dumps the raw execution times and populations to JSON files (`benchmark.json` and `correctness.json`), which are then consumed by the comparison scripts.
