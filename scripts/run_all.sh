#!/bin/zsh
set -e
SCRIPT_DIR="${0:a:h}"

tasks=(
    "lorenz96.sh"
    "cluster_lorenz.sh"
    "Finance.sh"
    "var.sh"
)

# 3. 循环遍历执行
for task in "${tasks[@]}"; do
    echo "🚀 [Start] Running task: $task ..."
    zsh "$SCRIPT_DIR/$task"
    
    echo "✅ [End] Finished: $task"
    echo "------------------------------------------"
done

echo "🎉 All experiments completed successfully!"