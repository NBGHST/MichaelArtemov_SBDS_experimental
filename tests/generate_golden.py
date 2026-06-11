"""
Generate golden reference data from the current (cpp_opt) branch.

Runs several small deterministic simulations with fixed seeds and saves
the results (population, time, event_count, coordinates, death rates)
to a JSON file that will be used by test_correctness.py to verify that
optimizations don't break results.
"""

import json
import sys
import os
import numpy as np
from scipy.stats import rayleigh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulation


def make_scenario_1():
    """Small 2-species 2D scenario (from benchmark.py test 1, scaled down)."""
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

    death_r_values = []
    death_density_values = []
    for i in range(M):
        r_values_row = []
        density_row = []
        for j in range(M):
            r_max = min(5 / sigma_w[i, j], L / 2)
            r_vals = np.linspace(0, r_max, 500)
            density = (sigma_w[i, j] ** (-2)) * r_vals * np.exp(-r_vals / sigma_w[i, j])
            r_values_row.append(r_vals.tolist())
            density_row.append(density.tolist())
        death_r_values.append(r_values_row)
        death_density_values.append(density_row)

    cutoffs = []
    for i in range(M):
        for j in range(M):
            cutoffs.append(min(10 * sigma_w[i, j], L / 2))

    np.random.seed(seed)
    coordinates = []
    for _ in range(M):
        group = [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(100)]
        coordinates.append(group)

    return {
        "name": "scenario_1_2species_large_L",
        "dim": 2,
        "M": M,
        "areaLen": [L, L],
        "cellCount": [25, 25],
        "isPeriodic": False,
        "birthRates": birth_rates,
        "deathRates": natural_death_rates,
        "ddMatrix": competition_matrix,
        "birthX": [q_values.tolist()] * M,
        "birthY": birth_inverse_values,
        "deathX": death_r_values,
        "deathY": death_density_values,
        "cutoffs": cutoffs,
        "seed": seed,
        "rtimeLimit": 7200.0,
        "coordinates": coordinates,
        "run_events": 5000,
    }


def make_scenario_2():
    """Small 2-species scenario with tighter kernels (from benchmark test 2)."""
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

    death_r_values = []
    death_density_values = []
    for i in range(M):
        r_values_row = []
        density_row = []
        for j in range(M):
            r_max = min(10 * sigma_w[i, j], L / 2)
            r_vals = np.linspace(0, r_max, 500)
            density = (1 / (2 * np.pi * sigma_w[i, j] ** 2)) * np.exp(
                -r_vals ** 2 / (2 * sigma_w[i, j] ** 2)
            )
            r_values_row.append(r_vals.tolist())
            density_row.append(density.tolist())
        death_r_values.append(r_values_row)
        death_density_values.append(density_row)

    cutoffs = []
    for i in range(M):
        for j in range(M):
            cutoffs.append(min(10 * sigma_w[i, j], L / 2))

    np.random.seed(seed)
    coordinates = []
    for _ in range(M):
        group = [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(200)]
        coordinates.append(group)

    return {
        "name": "scenario_2_tight_kernels",
        "dim": 2,
        "M": M,
        "areaLen": [L, L],
        "cellCount": [50, 50],
        "isPeriodic": False,
        "birthRates": birth_rates,
        "deathRates": natural_death_rates,
        "ddMatrix": competition_matrix,
        "birthX": [q_values.tolist()] * M,
        "birthY": birth_inverse_values,
        "deathX": death_r_values,
        "deathY": death_density_values,
        "cutoffs": cutoffs,
        "seed": seed,
        "rtimeLimit": 7200.0,
        "coordinates": coordinates,
        "run_events": 3000,
    }


def make_scenario_3():
    """1D single-species scenario for dimension coverage."""
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
    coordinates = [[[np.random.uniform(0, L)] for _ in range(150)]]

    return {
        "name": "scenario_3_1d_single_species",
        "dim": 1,
        "M": M,
        "areaLen": [L],
        "cellCount": [25],
        "isPeriodic": True,
        "birthRates": birth_rates,
        "deathRates": natural_death_rates,
        "ddMatrix": competition_matrix,
        "birthX": [q_values.tolist()] * M,
        "birthY": birth_inverse_values,
        "deathX": death_r_values,
        "deathY": death_density_values,
        "cutoffs": cutoffs,
        "seed": seed,
        "rtimeLimit": 7200.0,
        "coordinates": coordinates,
        "run_events": 2000,
    }


def run_scenario(params):
    """Run a scenario and collect results."""
    dim = params["dim"]
    GridClass = {1: simulation.PyGrid1, 2: simulation.PyGrid2, 3: simulation.PyGrid3}[dim]

    grid = GridClass(
        M=params["M"],
        areaLen=params["areaLen"],
        cellCount=params["cellCount"],
        isPeriodic=params["isPeriodic"],
        birthRates=params["birthRates"],
        deathRates=params["deathRates"],
        ddMatrix=params["ddMatrix"],
        birthX=params["birthX"],
        birthY=params["birthY"],
        deathX=params["deathX"],
        deathY=params["deathY"],
        cutoffs=params["cutoffs"],
        seed=params["seed"],
        rtimeLimit=params["rtimeLimit"],
    )
    grid.placePopulation(params["coordinates"])
    grid.run_events(params["run_events"])

    coords = grid.get_all_particle_coords()
    death_rates = grid.get_all_particle_death_rates()

    return {
        "name": params["name"],
        "total_population": grid.total_population,
        "species_pop": grid.species_pop,
        "time": grid.time,
        "event_count": grid.event_count,
        "total_birth_rate": grid.total_birth_rate,
        "total_death_rate": grid.total_death_rate,
        "coords": coords,
        "death_rates": death_rates,
    }


def main():
    scenarios = [make_scenario_1(), make_scenario_2(), make_scenario_3()]
    results = []

    for sc in scenarios:
        print(f"Running {sc['name']}...")
        result = run_scenario(sc)
        print(
            f"  population={result['total_population']}, "
            f"species_pop={result['species_pop']}, "
            f"time={result['time']:.6f}, "
            f"events={result['event_count']}"
        )
        results.append(result)

    golden_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_reference.json")
    with open(golden_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nGolden reference saved to {golden_path}")


if __name__ == "__main__":
    main()
