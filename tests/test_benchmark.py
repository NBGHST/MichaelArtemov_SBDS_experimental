"""
test_benchmark.py - Performance benchmarks for the SBDS simulator.

Runs multiple simulation scenarios at different scales and reports
events/sec, wall time, and peak population. Designed to be run
manually (not in CI) to compare performance between branches.

Usage:
    python3 tests/test_benchmark.py
"""

import time
import sys
import os
import numpy as np
from scipy.stats import rayleigh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulation


def format_rate(events_per_sec):
    """Format events/sec with appropriate suffix."""
    if events_per_sec >= 1e6:
        return f"{events_per_sec / 1e6:.2f}M events/s"
    elif events_per_sec >= 1e3:
        return f"{events_per_sec / 1e3:.1f}K events/s"
    else:
        return f"{events_per_sec:.0f} events/s"


def benchmark_small():
    """Small benchmark: 2 species, 200 particles, 10000 events."""
    L = 10.0
    M = 2
    seed = 42

    birth_rates = [0.33, 0.33]
    natural_death_rates = [0.1, 0.1]
    competition_matrix = [0.02, 0.012, 0.012, 0.02]
    sigma_m = [1, 1]
    sigma_w = np.array([[0.4, 0.4], [0.4, 0.4]])

    q_values = np.arange(0, 1.0, 0.001)
    birth_inverse_values = []
    for i in range(M):
        inverse_vals = np.sqrt(-2 * np.log(1 - q_values)) / sigma_m[i]
        birth_inverse_values.append(inverse_vals.tolist())

    death_r_values, death_density_values = [], []
    for i in range(M):
        r_row, d_row = [], []
        for j in range(M):
            r_max = min(5 / sigma_w[i, j], L / 2)
            r_vals = np.linspace(0, r_max, 500)
            density = (sigma_w[i, j] ** (-2)) * r_vals * np.exp(-r_vals / sigma_w[i, j])
            r_row.append(r_vals.tolist())
            d_row.append(density.tolist())
        death_r_values.append(r_row)
        death_density_values.append(d_row)

    cutoffs = []
    for i in range(M):
        for j in range(M):
            cutoffs.append(min(10 * sigma_w[i, j], L / 2))

    np.random.seed(seed)
    coords = [
        [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(100)]
        for _ in range(M)
    ]

    g = simulation.PyGrid2(
        M=M, areaLen=[L, L], cellCount=[25, 25], isPeriodic=False,
        birthRates=birth_rates, deathRates=natural_death_rates,
        ddMatrix=competition_matrix,
        birthX=[q_values.tolist()] * M, birthY=birth_inverse_values,
        deathX=death_r_values, deathY=death_density_values,
        cutoffs=cutoffs, seed=seed, rtimeLimit=7200.0,
    )
    g.placePopulation(coords)

    n_events = 10000
    t0 = time.perf_counter()
    g.run_events(n_events)
    elapsed = time.perf_counter() - t0

    return {
        "name": "Small (200 init, 10K events)",
        "events": n_events,
        "elapsed": elapsed,
        "final_pop": g.total_population,
        "species_pop": g.species_pop,
    }


def benchmark_medium():
    """Medium benchmark: 2 species, 1000 particles, 50000 events."""
    L = 2.0
    M = 2
    seed = 42

    birth_rates = [0.4, 0.4]
    natural_death_rates = [0.2, 0.2]
    competition_matrix = [0.001, 0.001, 0.0008, 0.001]
    sigma_m = [0.04, 0.04]
    sigma_w = np.array([[0.04, 0.04], [0.04, 0.04]])

    q_values = np.arange(0, 1.0, 0.001)
    birth_inverse_values = []
    for i in range(M):
        inverse_vals = rayleigh.ppf(q_values, scale=sigma_m[i])
        birth_inverse_values.append(inverse_vals.tolist())

    death_r_values, death_density_values = [], []
    for i in range(M):
        r_row, d_row = [], []
        for j in range(M):
            r_max = min(10 * sigma_w[i, j], L / 2)
            r_vals = np.linspace(0, r_max, 500)
            density = (1 / (2 * np.pi * sigma_w[i, j] ** 2)) * np.exp(
                -r_vals ** 2 / (2 * sigma_w[i, j] ** 2)
            )
            r_row.append(r_vals.tolist())
            d_row.append(density.tolist())
        death_r_values.append(r_row)
        death_density_values.append(d_row)

    cutoffs = []
    for i in range(M):
        for j in range(M):
            cutoffs.append(min(10 * sigma_w[i, j], L / 2))

    np.random.seed(seed)
    coords = [
        [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(500)]
        for _ in range(M)
    ]

    g = simulation.PyGrid2(
        M=M, areaLen=[L, L], cellCount=[50, 50], isPeriodic=False,
        birthRates=birth_rates, deathRates=natural_death_rates,
        ddMatrix=competition_matrix,
        birthX=[q_values.tolist()] * M, birthY=birth_inverse_values,
        deathX=death_r_values, deathY=death_density_values,
        cutoffs=cutoffs, seed=seed, rtimeLimit=7200.0,
    )
    g.placePopulation(coords)

    n_events = 50000
    t0 = time.perf_counter()
    g.run_events(n_events)
    elapsed = time.perf_counter() - t0

    return {
        "name": "Medium (1000 init, 50K events)",
        "events": n_events,
        "elapsed": elapsed,
        "final_pop": g.total_population,
        "species_pop": g.species_pop,
    }


def benchmark_large_run_for():
    """Large benchmark using run_for (matching benchmark.py style)."""
    L = 10.0
    M = 2
    seed = 42

    birth_rates = [0.33, 0.33]
    natural_death_rates = [0.1, 0.1]
    competition_matrix = [0.02, 0.012, 0.012, 0.02]
    sigma_m = [1, 1]
    sigma_w = np.array([[0.4, 0.4], [0.4, 0.4]])

    q_values = np.arange(0, 1.0, 0.001)
    birth_inverse_values = []
    for i in range(M):
        inverse_vals = np.sqrt(-2 * np.log(1 - q_values)) / sigma_m[i]
        birth_inverse_values.append(inverse_vals.tolist())

    death_r_values, death_density_values = [], []
    for i in range(M):
        r_row, d_row = [], []
        for j in range(M):
            r_max = min(5 / sigma_w[i, j], L / 2)
            r_vals = np.linspace(0, r_max, 500)
            density = (sigma_w[i, j] ** (-2)) * r_vals * np.exp(-r_vals / sigma_w[i, j])
            r_row.append(r_vals.tolist())
            d_row.append(density.tolist())
        death_r_values.append(r_row)
        death_density_values.append(d_row)

    cutoffs = []
    for i in range(M):
        for j in range(M):
            cutoffs.append(min(10 * sigma_w[i, j], L / 2))

    np.random.seed(seed)
    coords = [
        [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(400)]
        for _ in range(M)
    ]

    g = simulation.PyGrid2(
        M=M, areaLen=[L, L], cellCount=[50, 50], isPeriodic=False,
        birthRates=birth_rates, deathRates=natural_death_rates,
        ddMatrix=competition_matrix,
        birthX=[q_values.tolist()] * M, birthY=birth_inverse_values,
        deathX=death_r_values, deathY=death_density_values,
        cutoffs=cutoffs, seed=seed, rtimeLimit=7200.0,
    )
    g.placePopulation(coords)

    n_steps = 200
    t0 = time.perf_counter()
    for _ in range(n_steps):
        g.run_for(1)
    elapsed = time.perf_counter() - t0

    return {
        "name": f"Large (800 init, {n_steps}x run_for(1))",
        "events": g.event_count,
        "elapsed": elapsed,
        "final_pop": g.total_population,
        "species_pop": g.species_pop,
    }


def benchmark_1d():
    """1D benchmark for dimension coverage."""
    L = 5.0
    M = 1
    seed = 123

    birth_rates = [0.5]
    natural_death_rates = [0.15]
    competition_matrix = [0.01]
    sigma_m = [0.5]
    sigma_w = np.array([[0.3]])

    q_values = np.arange(0, 1.0, 0.001)
    birth_inverse_values = [
        (np.sqrt(-2 * np.log(1 - q_values)) / sigma_m[0]).tolist()
    ]

    r_max = min(5 / sigma_w[0, 0], L / 2)
    r_vals = np.linspace(0, r_max, 500)
    density = (sigma_w[0, 0] ** (-2)) * r_vals * np.exp(-r_vals / sigma_w[0, 0])

    death_r_values = [[r_vals.tolist()]]
    death_density_values = [[density.tolist()]]
    cutoffs = [min(10 * sigma_w[0, 0], L / 2)]

    np.random.seed(seed)
    coords = [[[np.random.uniform(0, L)] for _ in range(300)]]

    g = simulation.PyGrid1(
        M=M, areaLen=[L], cellCount=[25], isPeriodic=True,
        birthRates=birth_rates, deathRates=natural_death_rates,
        ddMatrix=competition_matrix,
        birthX=[q_values.tolist()], birthY=birth_inverse_values,
        deathX=death_r_values, deathY=death_density_values,
        cutoffs=cutoffs, seed=seed, rtimeLimit=7200.0,
    )
    g.placePopulation(coords)

    n_events = 20000
    t0 = time.perf_counter()
    g.run_events(n_events)
    elapsed = time.perf_counter() - t0

    return {
        "name": "1D (300 init, 20K events, periodic)",
        "events": n_events,
        "elapsed": elapsed,
        "final_pop": g.total_population,
        "species_pop": g.species_pop,
    }

def benchmark_1d_non_periodic():
    """1D non-periodic benchmark."""
    L = 5.0
    M = 1
    seed = 123
    birth_rates = [0.5]
    natural_death_rates = [0.15]
    competition_matrix = [0.01]
    sigma_m = [0.5]
    sigma_w = np.array([[0.3]])

    q_values = np.arange(0, 1.0, 0.001)
    birth_inverse_values = [
        (np.sqrt(-2 * np.log(1 - q_values)) / sigma_m[0]).tolist()
    ]

    r_max = min(5 / sigma_w[0, 0], L / 2)
    r_vals = np.linspace(0, r_max, 500)
    density = (sigma_w[0, 0] ** (-2)) * r_vals * np.exp(-r_vals / sigma_w[0, 0])

    death_r_values = [[r_vals.tolist()]]
    death_density_values = [[density.tolist()]]
    cutoffs = [min(10 * sigma_w[0, 0], L / 2)]

    np.random.seed(seed)
    coords = [[[np.random.uniform(0, L)] for _ in range(300)]]

    g = simulation.PyGrid1(
        M=M, areaLen=[L], cellCount=[25], isPeriodic=False,
        birthRates=birth_rates, deathRates=natural_death_rates,
        ddMatrix=competition_matrix,
        birthX=[q_values.tolist()], birthY=birth_inverse_values,
        deathX=death_r_values, deathY=death_density_values,
        cutoffs=cutoffs, seed=seed, rtimeLimit=7200.0,
    )
    g.placePopulation(coords)

    n_events = 20000
    t0 = time.perf_counter()
    g.run_events(n_events)
    elapsed = time.perf_counter() - t0

    return {
        "name": "1D Non-Periodic (300 init, 20K events)",
        "events": n_events,
        "elapsed": elapsed,
        "final_pop": g.total_population,
        "species_pop": g.species_pop,
    }

def benchmark_3d():
    """3D benchmark."""
    L = 3.0
    M = 1
    seed = 42
    birth_rates = [0.4]
    natural_death_rates = [0.1]
    competition_matrix = [0.02]
    sigma_m = [0.5]
    sigma_w = np.array([[0.3]])

    q_values = np.arange(0, 1.0, 0.001)
    birth_inverse_values = [(np.sqrt(-2 * np.log(1 - q_values)) / sigma_m[0]).tolist()]

    r_max = min(5 / sigma_w[0, 0], L / 2)
    r_vals = np.linspace(0, r_max, 500)
    density = (sigma_w[0, 0] ** (-2)) * r_vals * np.exp(-r_vals / sigma_w[0, 0])

    death_r_values = [[r_vals.tolist()]]
    death_density_values = [[density.tolist()]]
    cutoffs = [min(10 * sigma_w[0, 0], L / 2)]

    np.random.seed(seed)
    coords = [[[np.random.uniform(0, L), np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(500)]]

    g = simulation.PyGrid3(
        M=M, areaLen=[L, L, L], cellCount=[10, 10, 10], isPeriodic=False,
        birthRates=birth_rates, deathRates=natural_death_rates,
        ddMatrix=competition_matrix,
        birthX=[q_values.tolist()], birthY=birth_inverse_values,
        deathX=death_r_values, deathY=death_density_values,
        cutoffs=cutoffs, seed=seed, rtimeLimit=7200.0,
    )
    g.placePopulation(coords)

    n_events = 20000
    t0 = time.perf_counter()
    g.run_events(n_events)
    elapsed = time.perf_counter() - t0

    return {
        "name": "3D (500 init, 20K events, non-periodic)",
        "events": n_events,
        "elapsed": elapsed,
        "final_pop": g.total_population,
        "species_pop": g.species_pop,
    }


def benchmark_2d_periodic():
    """2D periodic benchmark."""
    L = 5.0
    M = 2
    seed = 42
    birth_rates = [0.33, 0.33]
    natural_death_rates = [0.1, 0.1]
    competition_matrix = [0.02, 0.012, 0.012, 0.02]
    sigma_m = [1, 1]
    sigma_w = np.array([[0.4, 0.4], [0.4, 0.4]])

    q_values = np.arange(0, 1.0, 0.001)
    birth_inverse_values = []
    for i in range(M):
        inverse_vals = np.sqrt(-2 * np.log(1 - q_values)) / sigma_m[i]
        birth_inverse_values.append(inverse_vals.tolist())

    death_r_values, death_density_values = [], []
    for i in range(M):
        r_row, d_row = [], []
        for j in range(M):
            r_max = min(5 / sigma_w[i, j], L / 2)
            r_vals = np.linspace(0, r_max, 500)
            density = (sigma_w[i, j] ** (-2)) * r_vals * np.exp(-r_vals / sigma_w[i, j])
            r_row.append(r_vals.tolist())
            d_row.append(density.tolist())
        death_r_values.append(r_row)
        death_density_values.append(d_row)

    cutoffs = []
    for i in range(M):
        for j in range(M):
            cutoffs.append(min(10 * sigma_w[i, j], L / 2))

    np.random.seed(seed)
    coords = [
        [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(250)]
        for _ in range(M)
    ]

    g = simulation.PyGrid2(
        M=M, areaLen=[L, L], cellCount=[25, 25], isPeriodic=True,
        birthRates=birth_rates, deathRates=natural_death_rates,
        ddMatrix=competition_matrix,
        birthX=[q_values.tolist()] * M, birthY=birth_inverse_values,
        deathX=death_r_values, deathY=death_density_values,
        cutoffs=cutoffs, seed=seed, rtimeLimit=7200.0,
    )
    g.placePopulation(coords)

    n_events = 20000
    t0 = time.perf_counter()
    g.run_events(n_events)
    elapsed = time.perf_counter() - t0

    return {
        "name": "2D Periodic (500 init, 20K events)",
        "events": n_events,
        "elapsed": elapsed,
        "final_pop": g.total_population,
        "species_pop": g.species_pop,
    }

def benchmark_3d_periodic():
    """3D periodic benchmark."""
    L = 5.0
    M = 1
    seed = 42
    birth_rates = [0.4]
    natural_death_rates = [0.1]
    competition_matrix = [0.02]
    sigma_m = [0.5]
    sigma_w = np.array([[0.3]])

    q_values = np.arange(0, 1.0, 0.001)
    birth_inverse_values = [(np.sqrt(-2 * np.log(1 - q_values)) / sigma_m[0]).tolist()]

    r_max = min(5 / sigma_w[0, 0], L / 2)
    r_vals = np.linspace(0, r_max, 500)
    density = (sigma_w[0, 0] ** (-2)) * r_vals * np.exp(-r_vals / sigma_w[0, 0])

    death_r_values = [[r_vals.tolist()]]
    death_density_values = [[density.tolist()]]
    cutoffs = [min(10 * sigma_w[0, 0], L / 2)]

    np.random.seed(seed)
    coordinates = [[[np.random.uniform(0, L), np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(500)]]

    grid = simulation.PyGrid3(
        L, L, L, 10, 10, 10, True, M,
        birth_rates, natural_death_rates, competition_matrix,
        [q_values.tolist()], birth_inverse_values,
        death_r_values, death_density_values, cutoffs
    )

    for s in range(M):
        for pos in coordinates[s]:
            grid.spawn_at(s, pos)

    start_time = time.time()
    events = 20000
    for _ in range(events):
        grid.step()
    elapsed = time.time() - start_time

    return {
        "name": "3D_periodic_1sp",
        "elapsed": elapsed,
        "events": events,
        "final_pop": grid.total_population(),
        "species_pop": [grid.species_population(i) for i in range(M)],
    }

def main():
    print("=" * 70)
    print("SBDS Simulator Performance Benchmarks")
    print("=" * 70)

    benchmarks = [
        benchmark_small,
        benchmark_medium,
        benchmark_large_run_for,
        benchmark_1d,
        benchmark_1d_non_periodic,
        benchmark_3d,
        benchmark_3d_periodic,
        benchmark_2d_periodic,
    ]

    results = []
    for bench_fn in benchmarks:
        result = bench_fn()
        results.append(result)
        rate = result["events"] / result["elapsed"] if result["elapsed"] > 0 else float("inf")
        print(f"\n  {result['name']}")
        print(f"    Wall time:    {result['elapsed']:.3f}s")
        print(f"    Events:       {result['events']}")
        print(f"    Rate:         {format_rate(rate)}")
        print(f"    Final pop:    {result['final_pop']}")
        print(f"    Species pop:  {result['species_pop']}")

    print("\n" + "=" * 70)
    print("Summary")
    print("-" * 70)
    print(f"{'Benchmark':<45} {'Time':>8} {'Rate':>18}")
    print("-" * 70)
    total_time = 0
    for r in results:
        rate = r["events"] / r["elapsed"] if r["elapsed"] > 0 else float("inf")
        print(f"  {r['name']:<43} {r['elapsed']:>7.3f}s {format_rate(rate):>18}")
        total_time += r["elapsed"]
    print("-" * 70)
    print(f"  {'TOTAL':<43} {total_time:>7.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
