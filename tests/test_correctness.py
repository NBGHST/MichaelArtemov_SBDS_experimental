"""
test_correctness.py - Deterministic correctness tests for the SBDS simulator.

Compares optimized simulation results against golden reference data
generated from the cpp_opt branch. Verifies that optimizations produce
bit-identical results (same RNG sequence → same output).
"""

import json
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulation

# Load golden reference data
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_reference.json")


def load_golden():
    with open(GOLDEN_PATH, "r") as f:
        return json.load(f)


# ---- Scenario builders (must exactly match generate_golden.py) ----

def make_scenario_1():
    """Small 2-species 2D scenario."""
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
    """Small 2-species scenario with tight kernels."""
    from scipy.stats import rayleigh as rayleigh_dist

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
        inverse_vals = rayleigh_dist.ppf(q_values, scale=sigma_m[i])
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
    """1D single-species periodic scenario."""
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
def make_scenario_4():
    """3D single-species non-periodic scenario."""
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
    coordinates = [[[np.random.uniform(0, L), np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(100)]]

    return {
        "name": "scenario_4_3d_single_species",
        "dim": 3,
        "M": M,
        "areaLen": [L, L, L],
        "cellCount": [10, 10, 10],
        "isPeriodic": False,
        "birthRates": birth_rates,
        "deathRates": natural_death_rates,
        "ddMatrix": competition_matrix,
        "birthX": [q_values.tolist()],
        "birthY": birth_inverse_values,
        "deathX": death_r_values,
        "deathY": death_density_values,
        "cutoffs": cutoffs,
        "seed": seed,
        "rtimeLimit": 7200.0,
        "coordinates": coordinates,
        "run_events": 2000,
    }


def make_scenario_5():
    """2D two-species periodic scenario."""
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
    coordinates = [
        [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(50)]
        for _ in range(M)
    ]

    return {
        "name": "scenario_5_2d_periodic",
        "dim": 2,
        "M": M,
        "areaLen": [L, L],
        "cellCount": [25, 25],
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
    coordinates = [[[np.random.uniform(0, L)] for _ in range(150)]]

    return {
        "name": "scenario_6_1d_non_periodic",
        "dim": 1,
        "M": M,
        "areaLen": [L],
        "cellCount": [25],
        "isPeriodic": False,
        "birthRates": birth_rates,
        "deathRates": natural_death_rates,
        "ddMatrix": competition_matrix,
        "birthX": [q_values.tolist()],
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

    return {
        "total_population": grid.total_population,
        "species_pop": grid.species_pop,
        "time": grid.time,
        "event_count": grid.event_count,
        "total_birth_rate": grid.total_birth_rate,
        "total_death_rate": grid.total_death_rate,
        "coords": grid.get_all_particle_coords(),
        "death_rates": grid.get_all_particle_death_rates(),
    }


SCENARIOS = [make_scenario_1, make_scenario_2, make_scenario_3, make_scenario_4, make_scenario_5, make_scenario_6]
GOLDEN = load_golden()


@pytest.mark.parametrize("scenario_idx", range(len(SCENARIOS)))
def test_population_matches(scenario_idx):
    """Total population must exactly match golden reference."""
    golden = GOLDEN[scenario_idx]
    result = run_scenario(SCENARIOS[scenario_idx]())
    assert result["total_population"] == golden["total_population"], (
        f"Population mismatch: got {result['total_population']}, "
        f"expected {golden['total_population']}"
    )


@pytest.mark.parametrize("scenario_idx", range(len(SCENARIOS)))
def test_species_pop_matches(scenario_idx):
    """Per-species populations must exactly match."""
    golden = GOLDEN[scenario_idx]
    result = run_scenario(SCENARIOS[scenario_idx]())
    assert result["species_pop"] == golden["species_pop"], (
        f"Species pop mismatch: got {result['species_pop']}, "
        f"expected {golden['species_pop']}"
    )


@pytest.mark.parametrize("scenario_idx", range(len(SCENARIOS)))
def test_event_count_matches(scenario_idx):
    """Event count must exactly match."""
    golden = GOLDEN[scenario_idx]
    result = run_scenario(SCENARIOS[scenario_idx]())
    assert result["event_count"] == golden["event_count"], (
        f"Event count mismatch: got {result['event_count']}, "
        f"expected {golden['event_count']}"
    )


@pytest.mark.parametrize("scenario_idx", range(len(SCENARIOS)))
def test_time_matches(scenario_idx):
    """Simulation time must match within floating-point tolerance."""
    golden = GOLDEN[scenario_idx]
    result = run_scenario(SCENARIOS[scenario_idx]())
    np.testing.assert_allclose(
        result["time"], golden["time"],
        rtol=1e-10, atol=1e-14,
        err_msg=f"Time mismatch in {golden['name']}"
    )


@pytest.mark.parametrize("scenario_idx", range(len(SCENARIOS)))
def test_coordinates_match(scenario_idx):
    """All particle coordinates must match within tolerance."""
    golden = GOLDEN[scenario_idx]
    result = run_scenario(SCENARIOS[scenario_idx]())

    for s in range(len(result["coords"])):
        result_coords = np.array(result["coords"][s])
        golden_coords = np.array(golden["coords"][s])
        assert result_coords.shape == golden_coords.shape, (
            f"Coordinate shape mismatch for species {s}: "
            f"got {result_coords.shape}, expected {golden_coords.shape}"
        )
        if result_coords.size > 0:
            np.testing.assert_allclose(
                result_coords, golden_coords,
                rtol=1e-10, atol=1e-12,
                err_msg=f"Coordinate mismatch for species {s} in {golden['name']}"
            )


@pytest.mark.parametrize("scenario_idx", range(len(SCENARIOS)))
def test_death_rates_match(scenario_idx):
    """All particle death rates must match within tolerance."""
    golden = GOLDEN[scenario_idx]
    result = run_scenario(SCENARIOS[scenario_idx]())

    for s in range(len(result["death_rates"])):
        result_rates = np.array(result["death_rates"][s])
        golden_rates = np.array(golden["death_rates"][s])
        assert result_rates.shape == golden_rates.shape, (
            f"Death rate shape mismatch for species {s}: "
            f"got {result_rates.shape}, expected {golden_rates.shape}"
        )
        if result_rates.size > 0:
            np.testing.assert_allclose(
                result_rates, golden_rates,
                rtol=1e-8, atol=1e-10,
                err_msg=f"Death rate mismatch for species {s} in {golden['name']}"
            )


@pytest.mark.parametrize("scenario_idx", range(len(SCENARIOS)))
def test_rates_match(scenario_idx):
    """Total birth/death rates must match within tolerance."""
    golden = GOLDEN[scenario_idx]
    result = run_scenario(SCENARIOS[scenario_idx]())

    np.testing.assert_allclose(
        result["total_birth_rate"], golden["total_birth_rate"],
        rtol=1e-10, atol=1e-14,
        err_msg=f"Total birth rate mismatch in {golden['name']}"
    )
    np.testing.assert_allclose(
        result["total_death_rate"], golden["total_death_rate"],
        rtol=1e-8, atol=1e-10,
        err_msg=f"Total death rate mismatch in {golden['name']}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
