import time
import sys
import os
import json
import numpy as np
from scipy.stats import rayleigh, halfnorm, maxwell

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulation

def make_general_scenario(dim, is_periodic, name, L, cell_count, seed, run_events=2000):
    M = 2
    birth_rates = [0.3, 0.4]
    natural_death_rates = [0.1, 0.15]
    competition_matrix = [0.01, 0.005, 0.005, 0.01]
    sigma_m = [0.5, 0.5]
    sigma_w = np.array([[0.3, 0.3], [0.3, 0.3]])

    q_values = np.arange(0, 1.0, 0.001)
    birth_inverse_values = []
    for i in range(M):
        inverse_vals = rayleigh.ppf(q_values, scale=sigma_m[i])
        birth_inverse_values.append(inverse_vals.tolist())

    death_r_values, death_density_values = [], []
    for i in range(M):
        r_row, d_row = [], []
        for j in range(M):
            r_max = min(5 * sigma_w[i, j], L / 2 if is_periodic else L)
            r_vals = np.linspace(0, r_max, 500)
            density = (sigma_w[i, j] ** (-2)) * r_vals * np.exp(-r_vals / sigma_w[i, j])
            r_row.append(r_vals.tolist())
            d_row.append(density.tolist())
        death_r_values.append(r_row)
        death_density_values.append(d_row)

    cutoffs = []
    for i in range(M):
        for j in range(M):
            cutoffs.append(min(10 * sigma_w[i, j], L / 2 if is_periodic else L))

    np.random.seed(seed)
    coords = []
    for _ in range(M):
        group = []
        for _ in range(800 if dim < 3 else 400):
            group.append([np.random.uniform(0, L) for _ in range(dim)])
        coords.append(group)

    return {
        "name": name,
        "params": {
            "M": M, "areaLen": [L] * dim, "cellCount": cell_count, "isPeriodic": is_periodic,
            "birthRates": birth_rates, "deathRates": natural_death_rates,
            "ddMatrix": competition_matrix,
            "birthX": [q_values.tolist()] * M, "birthY": birth_inverse_values,
            "deathX": death_r_values, "deathY": death_density_values,
            "sigma_m": sigma_m, "sigma_w": sigma_w.tolist(), "cutoffs": cutoffs, "seed": seed, "rtimeLimit": 7200.0,
            "coordinates": coords, "run_events": run_events,
            "dim": dim
        }
    }

def make_heteromyopia_scenario(dim, name, L, cell_count, seed, run_events=2000):
    M = 2
    birth_rates = [0.4, 0.4]
    natural_death_rates = [0.2, 0.2]
    competition_matrix = [0.001, 0.001, 0.001, 0.001]
    sigma_m = [0.06, 0.06]
    sigma_w = np.array([
        [0.15, 0.01],
        [0.01, 0.15],
    ])

    const = (2 * np.pi) ** (dim / 2)
    def normal_radial(r, sigma):
        return (1 / (const * sigma**dim)) * np.exp(-r**2 / (2 * sigma**2))

    q_values = np.arange(0, 1.0, 0.001)

    birth_inverse_values = []
    for j in range(M):
        if dim == 1:
            inverse_vals = halfnorm.ppf(q_values, scale=sigma_m[j])
        elif dim == 2:
            inverse_vals = rayleigh.ppf(q_values, scale=sigma_m[j])
        else: # 3D
            inverse_vals = maxwell.ppf(q_values, scale=sigma_m[j])
        birth_inverse_values.append(inverse_vals.tolist())

    death_r_values, death_density_values = [], []
    for k in range(M):
        r_values_row, density_row = [], []
        for j in range(M):
            r_max = min(7 * sigma_w[k, j], L)
            r_vals = np.linspace(0, r_max, 500)
            density = normal_radial(r_vals, sigma_w[k, j])
            r_values_row.append(r_vals.tolist())
            density_row.append(density.tolist())
        death_r_values.append(r_values_row)
        death_density_values.append(density_row)

    cutoffs = []
    for k in range(M):
        for j in range(M):
            cutoffs.append(min(7 * sigma_w[k, j], L))

    np.random.seed(seed)
    coords = []
    for _ in range(M):
        group = []
        for _ in range(800 if dim < 3 else 400):
            group.append([np.random.uniform(0, L) for _ in range(dim)])
        coords.append(group)

    return {
        "name": name,
        "params": {
            "M": M, "areaLen": [L] * dim, "cellCount": cell_count, "isPeriodic": False,
            "birthRates": birth_rates, "deathRates": natural_death_rates,
            "ddMatrix": competition_matrix,
            "birthX": [q_values.tolist()] * M, "birthY": birth_inverse_values,
            "deathX": death_r_values, "deathY": death_density_values,
            "sigma_m": sigma_m, "sigma_w": sigma_w.tolist(), "cutoffs": cutoffs, "seed": seed, "rtimeLimit": 7200.0,
            "coordinates": coords, "run_events": run_events,
            "dim": dim
        }
    }

def make_ccto_scenario(dim, name, L, cell_count, seed, run_events=2000):
    M = 2
    birth_rates = [0.6, 0.3]
    natural_death_rates = [0.2, 0.2]
    competition_matrix = [
        0.01, 0.01,
        0.001, 0.001
    ]
    sigma_m = [0.2, 0.04]
    sigma_w = np.array([
        [0.04, 0.04],
        [0.04, 0.04],
    ])

    const = (2 * np.pi) ** (dim / 2)
    def normal_radial(r, sigma):
        return (1 / (const * sigma**dim)) * np.exp(-r**2 / (2 * sigma**2))

    q_values = np.arange(0, 1.0, 0.001)

    birth_inverse_values = []
    for j in range(M):
        if dim == 1:
            inverse_vals = halfnorm.ppf(q_values, scale=sigma_m[j])
        elif dim == 2:
            inverse_vals = rayleigh.ppf(q_values, scale=sigma_m[j])
        else: # 3D
            inverse_vals = maxwell.ppf(q_values, scale=sigma_m[j])
        birth_inverse_values.append(inverse_vals.tolist())

    death_r_values, death_density_values = [], []
    for k in range(M):
        r_values_row, density_row = [], []
        for j in range(M):
            r_max = min(10 * sigma_w[k, j], L / 2)
            r_vals = np.linspace(0, r_max, 500)
            density = normal_radial(r_vals, sigma_w[k, j])
            r_values_row.append(r_vals.tolist())
            density_row.append(density.tolist())
        death_r_values.append(r_values_row)
        death_density_values.append(density_row)

    cutoffs = []
    for k in range(M):
        for j in range(M):
            cutoffs.append(min(10 * sigma_w[k, j], L / 2))

    np.random.seed(seed)
    coords = []
    for _ in range(M):
        group = []
        for _ in range(800 if dim < 3 else 400):
            group.append([np.random.uniform(0, L) for _ in range(dim)])
        coords.append(group)

    return {
        "name": name,
        "params": {
            "M": M, "areaLen": [L] * dim, "cellCount": cell_count, "isPeriodic": False,
            "birthRates": birth_rates, "deathRates": natural_death_rates,
            "ddMatrix": competition_matrix,
            "birthX": [q_values.tolist()] * M, "birthY": birth_inverse_values,
            "deathX": death_r_values, "deathY": death_density_values,
            "sigma_m": sigma_m, "sigma_w": sigma_w.tolist(), "cutoffs": cutoffs, "seed": seed, "rtimeLimit": 7200.0,
            "coordinates": coords, "run_events": run_events,
            "dim": dim
        }
    }

SCENARIOS = [
    # 1D Scenarios (1 to 7)
    lambda: make_general_scenario(dim=1, is_periodic=False, name="Scenario 1 (1D, non-periodic, gen1)", L=10.0, cell_count=[50], seed=42),
    lambda: make_general_scenario(dim=1, is_periodic=False, name="Scenario 2 (1D, non-periodic, gen2)", L=15.0, cell_count=[75], seed=43),
    lambda: make_general_scenario(dim=1, is_periodic=False, name="Scenario 3 (1D, non-periodic, gen3)", L=20.0, cell_count=[100], seed=44),
    lambda: make_general_scenario(dim=1, is_periodic=True, name="Scenario 4 (1D, periodic, per1)", L=10.0, cell_count=[50], seed=45),
    lambda: make_general_scenario(dim=1, is_periodic=True, name="Scenario 5 (1D, periodic, per2)", L=15.0, cell_count=[75], seed=46),
    lambda: make_heteromyopia_scenario(dim=1, name="Scenario 6 (1D, non-periodic, Heteromyopia)", L=10.0, cell_count=[250], seed=47),
    lambda: make_ccto_scenario(dim=1, name="Scenario 7 (1D, non-periodic, CCTO)", L=10.0, cell_count=[250], seed=48),

    # 2D Scenarios (8 to 14)
    lambda: make_general_scenario(dim=2, is_periodic=False, name="Scenario 8 (2D, non-periodic, gen1)", L=10.0, cell_count=[25, 25], seed=49),
    lambda: make_general_scenario(dim=2, is_periodic=False, name="Scenario 9 (2D, non-periodic, gen2)", L=15.0, cell_count=[35, 35], seed=50),
    lambda: make_general_scenario(dim=2, is_periodic=False, name="Scenario 10 (2D, non-periodic, gen3)", L=20.0, cell_count=[50, 50], seed=51),
    lambda: make_general_scenario(dim=2, is_periodic=True, name="Scenario 11 (2D, periodic, per1)", L=10.0, cell_count=[25, 25], seed=52),
    lambda: make_general_scenario(dim=2, is_periodic=True, name="Scenario 12 (2D, periodic, per2)", L=15.0, cell_count=[35, 35], seed=53),
    lambda: make_heteromyopia_scenario(dim=2, name="Scenario 13 (2D, non-periodic, Heteromyopia)", L=4.0, cell_count=[100, 100], seed=54),
    lambda: make_ccto_scenario(dim=2, name="Scenario 14 (2D, non-periodic, CCTO)", L=4.0, cell_count=[100, 100], seed=55),

    # 3D Scenarios (15 to 21)
    lambda: make_general_scenario(dim=3, is_periodic=False, name="Scenario 15 (3D, non-periodic, gen1)", L=10.0, cell_count=[10, 10, 10], seed=56),
    lambda: make_general_scenario(dim=3, is_periodic=False, name="Scenario 16 (3D, non-periodic, gen2)", L=15.0, cell_count=[12, 12, 12], seed=57),
    lambda: make_general_scenario(dim=3, is_periodic=False, name="Scenario 17 (3D, non-periodic, gen3)", L=20.0, cell_count=[15, 15, 15], seed=58),
    lambda: make_general_scenario(dim=3, is_periodic=True, name="Scenario 18 (3D, periodic, per1)", L=10.0, cell_count=[10, 10, 10], seed=59),
    lambda: make_general_scenario(dim=3, is_periodic=True, name="Scenario 19 (3D, periodic, per2)", L=15.0, cell_count=[12, 12, 12], seed=60),
    lambda: make_heteromyopia_scenario(dim=3, name="Scenario 20 (3D, non-periodic, Heteromyopia)", L=4.0, cell_count=[20, 20, 20], seed=61),
    lambda: make_ccto_scenario(dim=3, name="Scenario 21 (3D, non-periodic, CCTO)", L=4.0, cell_count=[20, 20, 20], seed=62)
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

    # Pop sigma parameters so they don't get passed to C++ grid constructor
    params.pop("sigma_m", None)
    params.pop("sigma_w", None)

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

    grid_params = {
        "M": params.get("M", 1),
        "areaLen": params.get("areaLen", []),
        "birthRates": params.get("birthRates", []),
        "deathRates": params.get("deathRates", []),
        "cutoffs": params.get("cutoffs", []),
        "ddMatrix": scenario_def["params"].get("ddMatrix", []),
        "sigma_m": scenario_def["params"].get("sigma_m", []),
        "sigma_w": scenario_def["params"].get("sigma_w", []),
        "run_events": run_events
    }
    
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
        "rate": run_events / elapsed if elapsed > 0 else 0,
        "grid_params": grid_params
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
