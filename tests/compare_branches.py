import os
import sys
import subprocess
import tempfile
import json
import shutil

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def build_and_run(directory, output_file):
    # Copy the worker to the target directory if it's not there
    for file_name in ["worker_metrics.py", "test_correctness.py", "test_benchmark.py"]:
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        dst = os.path.join(directory, "tests", file_name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

    # Build extension
    run_command([sys.executable, "setup.py", "build_ext", "--inplace", "--force", "-j4"], cwd=directory)
    
    # Run correctness
    run_command([sys.executable, "tests/worker_metrics.py", "correctness", "correctness.json"], cwd=directory)
    
    # Run benchmark
    run_command([sys.executable, "tests/worker_metrics.py", "benchmark", "benchmark.json"], cwd=directory)

    # Load results
    with open(os.path.join(directory, "correctness.json")) as f:
        correctness = json.load(f)
    with open(os.path.join(directory, "benchmark.json")) as f:
        benchmark = json.load(f)

    return {"correctness": correctness, "benchmark": benchmark}

def print_comparison(all_results, ref_branch, cur_name):
    print("\n" + "="*130)
    print(f" BRANCH COMPARISON REPORT: {', '.join(all_results.keys())} ")
    print("="*130)

    # Correctness is checked against the primary ref_branch
    ref_results = all_results[ref_branch]
    cur_results = all_results[cur_name]

    all_match = True

    # SCENARIO PARAMETERS & CORRECTNESS split by dimension
    for d in [1, 2, 3]:
        print(f"\n[ SCENARIO PARAMETERS & CORRECTNESS - {d}D ]")
        print(f"{'Scenario':<38} | {'Dim':<3} | {'Per':<3} | {'Grid Size':<12} | {'Init':<5} | {'Events':<7} | {'Pop (Ref)':<10} | {'Pop (Cur)':<10} | {'Match?':<6}")
        print("-" * 130)
        
        for ref_c, cur_c in zip(ref_results["correctness"], cur_results["correctness"]):
            if ref_c['dim'] != d:
                continue
            name = ref_c["name"]
            dim = f"{ref_c['dim']}D"
            per = "YES" if ref_c["is_periodic"] else "NO"
            grid = "x".join(map(str, ref_c["cell_count"]))
            init = ref_c["init_pop"]
            evs = ref_c["events"]
            pop_r = ref_c["total_population"]
            pop_c = cur_c["total_population"]

            match = (pop_r == pop_c) and (ref_c["event_count"] == cur_c["event_count"])
            if not match:
                all_match = False
            match_str = "YES" if match else "NO"
            
            print(f"{name:<38} | {dim:<3} | {per:<3} | {grid:<12} | {init:<5} | {evs:<7} | {pop_r:<10} | {pop_c:<10} | {match_str:<6}")
            
            # Second line for ALL grid parameters
            gp = ref_c.get("grid_params", {})
            if gp:
                param_str = f"M={gp.get('M')}, areaLen={gp.get('areaLen')}, bRates={gp.get('birthRates')}, dRates={gp.get('deathRates')}, cutoffs={gp.get('cutoffs')}, ddMatrix={gp.get('ddMatrix')}, sigma_m={gp.get('sigma_m')}, sigma_w={gp.get('sigma_w')}"
                print(f"   ↳ Params: {param_str}")
                print("-" * 130)
    
    col_order = ["cpp_opt", "prev_best", cur_name]
    
    def format_rate(r):
        if r > 1e6: return f"{r/1e6:.1f}M"
        if r > 1e3: return f"{r/1e3:.1f}K"
        return f"{r:.0f}"

    for d in [1, 2, 3]:
        print(f"\n[ BENCHMARKS (Events / Sec) - {d}D ]")
        header = f"{'Scenario':<40}"
        for b in col_order:
            header += f" | {b:<12}"
        header += f" | Speedup (prev_best vs cpp_opt) | Speedup ({cur_name} vs cpp_opt)"
        print(header)
        print("-" * 150)
        
        avg_speedup_ref_cpp = 0
        avg_speedup_cur_cpp = 0
        count = 0
        
        for i in range(len(ref_results["benchmark"])):
            ref_c = ref_results["correctness"][i]
            if ref_c['dim'] != d:
                continue
            
            name = ref_results["benchmark"][i]["name"]
            row = f"{name:<40}"
            ref_rate = ref_results["benchmark"][i]["rate"]
            cpp_rate = all_results.get("cpp_opt", {}).get("benchmark", [{}])[i].get("rate", 0) if "cpp_opt" in all_results else 0
            cur_rate = cur_results["benchmark"][i]["rate"]
            
            for b in col_order:
                rate = all_results[b]["benchmark"][i]["rate"]
                row += f" | {format_rate(rate):<12}"
                
            speedup_ref_cpp = ref_rate / cpp_rate if cpp_rate > 0 else 0
            speedup_cur_cpp = cur_rate / cpp_rate if cpp_rate > 0 else 0
            
            avg_speedup_ref_cpp += speedup_ref_cpp
            avg_speedup_cur_cpp += speedup_cur_cpp
            count += 1
            
            row += f" | {speedup_ref_cpp:.2f}x"
            row += f" | {speedup_cur_cpp:.2f}x"
            print(row)
            
        if count > 0:
            print("-" * 150)
            print(f"AVERAGE {d}D SPEEDUP (prev_best vs cpp_opt): {avg_speedup_ref_cpp/count:.2f}x | ({cur_name} vs cpp_opt): {avg_speedup_cur_cpp/count:.2f}x")

    if not all_match:
        print("\n!!! WARNING: Correctness mismatch detected! !!!")
        sys.exit(1)
    else:
        print("\nSUCCESS: All correctness tests match perfectly!")

def main():
    branches = ["prev_best", "cpp_opt"]
    cur_branch = "gudit_shumit"

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_results = {}

    for branch in branches:
        print(f"\n--- Setting up reference branch {branch} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            run_command(["git", "clone", "-b", branch, current_dir, tmpdir])
            
            print(f"\n--- Building and running {branch} ---")
            all_results[branch] = build_and_run(tmpdir, f"{branch}_results.json")

    print(f"\n--- Building and running current ({cur_branch}) ---")
    all_results[cur_branch] = build_and_run(current_dir, "cur_results.json")

    print_comparison(all_results, branches[0], cur_branch)

if __name__ == '__main__':
    main()
