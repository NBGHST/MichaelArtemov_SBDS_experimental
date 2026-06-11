import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_metrics import SCENARIOS
from numba_sim import make_ssa_state_1d, make_ssa_state_2d, make_ssa_state_3d

def run_numba_scenario(scenario_def, is_benchmark=False):
    params = dict(scenario_def["params"])
    dim = params.pop("dim")
    
    coords = params.pop("coordinates")
    run_events = params.pop("run_events")
    if is_benchmark:
        run_events *= 10
        
    M = params["M"]
    dd_matrix = np.array(params["ddMatrix"]).reshape(M, M).tolist()
    cutoffs = np.array(params["cutoffs"]).reshape(M, M).tolist()
    
    if dim == 1:
        state = make_ssa_state_1d(
            M=M,
            area_len=params["areaLen"][0],
            birth_rates=params["birthRates"],
            death_rates=params["deathRates"],
            dd_matrix=dd_matrix,
            birth_x=params["birthX"],
            birth_y=params["birthY"],
            death_x=params["deathX"],
            death_y=params["deathY"],
            cutoffs=cutoffs,
            cell_count=params["cellCount"][0],
            is_periodic=params.get("isPeriodic", False),
            seed=params["seed"],
            initial_population=coords
        )
    elif dim == 2:
        state = make_ssa_state_2d(
            M=M,
            area_len=params["areaLen"],
            birth_rates=params["birthRates"],
            death_rates=params["deathRates"],
            dd_matrix=dd_matrix,
            birth_x=params["birthX"],
            birth_y=params["birthY"],
            death_x=params["deathX"],
            death_y=params["deathY"],
            cutoffs=cutoffs,
            cell_counts=params["cellCount"],
            is_periodic=params.get("isPeriodic", False),
            seed=params["seed"],
            initial_population=coords
        )
    elif dim == 3:
        state = make_ssa_state_3d(
            M=M,
            area_len=params["areaLen"],
            birth_rates=params["birthRates"],
            death_rates=params["deathRates"],
            dd_matrix=dd_matrix,
            birth_x=params["birthX"],
            birth_y=params["birthY"],
            death_x=params["deathX"],
            death_y=params["deathY"],
            cutoffs=cutoffs,
            cell_counts=params["cellCount"],
            is_periodic=params.get("isPeriodic", False),
            seed=params["seed"],
            initial_population=coords
        )

    # Warmup for JIT
    state.run_events(1)
    
    t0 = time.perf_counter()
    state.run_events(run_events)
    elapsed = time.perf_counter() - t0
    
    return {
        "name": scenario_def["name"],
        "rate": run_events / elapsed if elapsed > 0 else 0
    }

def main():
    print("Warming up Numba JIT (compiling 1D, 2D, 3D...)")
    for i in [0, 7, 14]:
        dummy = SCENARIOS[i]()
        dummy["params"]["run_events"] = 1
        run_numba_scenario(dummy)
    
    results = []
    print("Running Numba sim benchmarks...")
    for s_fn in SCENARIOS:
        s = s_fn()
        print(f"Running {s['name']}")
        res = run_numba_scenario(s, is_benchmark=True)
        results.append(res)
    
    with open("numba_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Finished numba benchmarks.")

if __name__ == "__main__":
    main()
