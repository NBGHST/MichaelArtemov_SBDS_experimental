import time
import sys
import os
import json
import numpy as np
from scipy.stats import rayleigh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulation

def make_scenario_1():
    L = 5.0
    M = 2
    seed = 42
    birth_rates = [0.3, 0.4]
    natural_death_rates = [0.1, 0.15]
    competition_matrix = [0.01, 0.005, 0.005, 0.01]
    sigma_m = [0.5, 0.5]
    sigma_w = np.array([[0.3, 0.3], [0.3, 0.3]])

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
        [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(50)]
        for _ in range(M)
    ]

    return {
        "name": "Scenario 1 (2D, non-periodic)",
        "params": {
            "M": M, "areaLen": [L, L], "cellCount": [25, 25], "isPeriodic": False,
            "birthRates": birth_rates, "deathRates": natural_death_rates,
            "ddMatrix": competition_matrix,
            "birthX": [q_values.tolist()] * M, "birthY": birth_inverse_values,
            "deathX": death_r_values, "deathY": death_density_values,
            "cutoffs": cutoffs, "seed": seed, "rtimeLimit": 7200.0,
            "coordinates": coords, "run_events": 2000,
            "dim": 2
        }
    }

def make_scenario_2():
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
            density = (1 / (2 * np.pi * sigma_w[i, j] ** 2)) * np.exp(-r_vals ** 2 / (2 * sigma_w[i, j] ** 2))
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
        [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(150)]
        for _ in range(M)
    ]

    return {
        "name": "Scenario 2 (2D, non-periodic, medium)",
        "params": {
            "M": M, "areaLen": [L, L], "cellCount": [50, 50], "isPeriodic": False,
            "birthRates": birth_rates, "deathRates": natural_death_rates,
            "ddMatrix": competition_matrix,
            "birthX": [q_values.tolist()] * M, "birthY": birth_inverse_values,
            "deathX": death_r_values, "deathY": death_density_values,
            "cutoffs": cutoffs, "seed": seed, "rtimeLimit": 7200.0,
            "coordinates": coords, "run_events": 5000,
            "dim": 2
        }
    }

def make_scenario_3():
    L = 5.0
    M = 1
    seed = 123
    birth_rates = [0.5]
    natural_death_rates = [0.15]
    competition_matrix = [0.01]
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
    coords = [[[np.random.uniform(0, L)] for _ in range(150)]]

    return {
        "name": "Scenario 3 (1D, periodic)",
        "params": {
            "M": M, "areaLen": [L], "cellCount": [25], "isPeriodic": True,
            "birthRates": birth_rates, "deathRates": natural_death_rates,
            "ddMatrix": competition_matrix,
            "birthX": [q_values.tolist()], "birthY": birth_inverse_values,
            "deathX": death_r_values, "deathY": death_density_values,
            "cutoffs": cutoffs, "seed": seed, "rtimeLimit": 7200.0,
            "coordinates": coords, "run_events": 2000,
            "dim": 1
        }
    }

def make_scenario_4():
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
    coords = [[[np.random.uniform(0, L), np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(100)]]

    return {
        "name": "Scenario 4 (3D, non-periodic)",
        "params": {
            "M": M, "areaLen": [L, L, L], "cellCount": [10, 10, 10], "isPeriodic": False,
            "birthRates": birth_rates, "deathRates": natural_death_rates,
            "ddMatrix": competition_matrix,
            "birthX": [q_values.tolist()], "birthY": birth_inverse_values,
            "deathX": death_r_values, "deathY": death_density_values,
            "cutoffs": cutoffs, "seed": seed, "rtimeLimit": 7200.0,
            "coordinates": coords, "run_events": 2000,
            "dim": 3
        }
    }

def make_scenario_5():
    L = 5.0
    M = 2
    seed = 42
    birth_rates = [0.3, 0.4]
    natural_death_rates = [0.1, 0.15]
    competition_matrix = [0.01, 0.005, 0.005, 0.01]
    sigma_m = [0.5, 0.5]
    sigma_w = np.array([[0.3, 0.3], [0.3, 0.3]])

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
        [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(50)]
        for _ in range(M)
    ]

    return {
        "name": "Scenario 5 (2D, periodic)",
        "params": {
            "M": M, "areaLen": [L, L], "cellCount": [25, 25], "isPeriodic": True,
            "birthRates": birth_rates, "deathRates": natural_death_rates,
            "ddMatrix": competition_matrix,
            "birthX": [q_values.tolist()] * M, "birthY": birth_inverse_values,
            "deathX": death_r_values, "deathY": death_density_values,
            "cutoffs": cutoffs, "seed": seed, "rtimeLimit": 7200.0,
            "coordinates": coords, "run_events": 2000,
            "dim": 2
        }
    }

def make_scenario_6():
    """1D single-species non-periodic scenario."""
    L = 5.0
    M = 1
    seed = 123
    birth_rates = [0.5]
    natural_death_rates = [0.15]
    competition_matrix = [0.01]
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
    coords = [[[np.random.uniform(0, L)] for _ in range(150)]]

    return {
        "name": "Scenario 6 (1D, non-periodic)",
        "params": {
            "M": M, "areaLen": [L], "cellCount": [25], "isPeriodic": False,
            "birthRates": birth_rates, "deathRates": natural_death_rates,
            "ddMatrix": competition_matrix,
            "birthX": [q_values.tolist()], "birthY": birth_inverse_values,
            "deathX": death_r_values, "deathY": death_density_values,
            "cutoffs": cutoffs, "seed": seed, "rtimeLimit": 7200.0,
            "coordinates": coords, "run_events": 2000,
            "dim": 1
        }
    }


def make_scenario_7():
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
    coords = [[[np.random.uniform(0, L), np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(100)]]

    return {
        "name": "Scenario 7 (3D, periodic)",
        "params": {
            "M": M, "areaLen": [L, L, L], "cellCount": [10, 10, 10], "isPeriodic": True,
            "birthRates": birth_rates, "deathRates": natural_death_rates,
            "ddMatrix": competition_matrix,
            "birthX": [q_values.tolist()], "birthY": birth_inverse_values,
            "deathX": death_r_values, "deathY": death_density_values,
            "cutoffs": cutoffs, "seed": seed, "rtimeLimit": 7200.0,
            "coordinates": coords, "run_events": 2000,
            "dim": 3
        }
    }

SCENARIOS = [
    make_scenario_1,
    make_scenario_2,
    make_scenario_3,
    make_scenario_4,
    make_scenario_5,
    make_scenario_6,
    make_scenario_7,
]

def run_scenario(scenario_def, is_benchmark=False):
    params = dict(scenario_def["params"])
    dim = params.pop("dim")
    if dim == 1:
        GridClass = simulation.PyGrid1
    elif dim == 2:
        GridClass = simulation.PyGrid2
    elif dim == 3:
        GridClass = simulation.PyGrid3
    else:
        raise ValueError("Invalid dimension")

    coords = params.pop("coordinates")
    run_events = params.pop("run_events")

    if is_benchmark:
        run_events *= 10  # Run 10x more events for meaningful timings

    grid = GridClass(**params)
    grid.placePopulation(coords)

    is_periodic_val = params.get("isPeriodic", False)
    cell_count_val = params.get("cellCount", [])
    init_pop_val = sum(len(group) for group in coords)

    t0 = time.perf_counter()
    grid.run_events(run_events)
    elapsed = time.perf_counter() - t0

    return {
        "name": scenario_def["name"],
        "dim": dim,
        "is_periodic": is_periodic_val,
        "cell_count": cell_count_val,
        "init_pop": init_pop_val,
        "total_population": grid.total_population,
        "event_count": grid.event_count,
        "species_pop": grid.species_pop,
        "elapsed": elapsed,
        "events": run_events,
        "rate": run_events / elapsed if elapsed > 0 else 0
    }

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "metrics.json"

    results = []
    is_benchmark = (mode == "benchmark")
    for s_fn in SCENARIOS:
        res = run_scenario(s_fn(), is_benchmark=is_benchmark)
        results.append(res)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
