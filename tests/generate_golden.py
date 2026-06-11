import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulation
from tests.worker_metrics import SCENARIOS

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
        "name": scenario_def["name"],
        "total_population": grid.total_population,
        "species_pop": grid.species_pop,
        "time": grid.time,
        "event_count": grid.event_count,
        "total_birth_rate": grid.total_birth_rate,
        "total_death_rate": grid.total_death_rate,
        "coords": grid.get_all_particle_coords(),
        "death_rates": grid.get_all_particle_death_rates(),
    }

def main():
    results = []
    for s_fn in SCENARIOS:
        sc = s_fn()
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
