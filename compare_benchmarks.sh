#!/bin/bash
source .venv_SBDS/bin/activate
echo "Running benchmarks on extreme_opt..."
python3 benchmark.py > bench_extreme_opt.txt

for branch in cpp_opt prev_best; do
    echo "Checking out $branch..."
    git checkout $branch
    bash ./update.sh > /dev/null 2>&1
    echo "Running benchmarks on $branch..."
    python3 benchmark.py > bench_${branch}.txt
done

# Switch back to extreme_opt
git checkout extreme_opt
bash ./update.sh > /dev/null 2>&1

echo "Results:"
echo "extreme_opt:"
tail -n 1 bench_extreme_opt.txt
echo "cpp_opt:"
tail -n 1 bench_cpp_opt.txt
echo "prev_best:"
tail -n 1 bench_prev_best.txt
