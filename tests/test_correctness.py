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
from tests.worker_metrics import SCENARIOS

# Load golden reference data
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_reference.json")

def load_golden():
    with open(GOLDEN_PATH, "r") as f:
        return json.load(f)

def run_scenario(scenario_def):
    params = dict(scenario_def["params"])
    dim = params.pop("dim")
    GridClass = {1: simulation.PyGrid1, 2: simulation.PyGrid2, 3: simulation.PyGrid3}[dim]
    coords = params.pop("coordinates")
    run_events = params.pop("run_events")
    params.pop("sigma_m", None)
    params.pop("sigma_w", None)

    grid = GridClass(**params)
    grid.placePopulation(coords)
    grid.run_events(run_events)

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
