echo "Input branches separated with space (for example: main develop feature/test):"
read -a branches

current_branch=$(git branch --show-current)

for branch in "${branches[@]}"; do
    echo ""
    echo "=== Checkout $branch ==="
    
    # Переключаемся на ветку
    git checkout "$branch" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "Starting $branch..."
        eval "python3 ./benchmark.py"
    else
        echo "Error: no such branch as $branch"
    fi
done

# Возвращаемся на исходную ветку
echo ""
echo "=== Return to $current_branch ==="
git checkout "$current_branch"