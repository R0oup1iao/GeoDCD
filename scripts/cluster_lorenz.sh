for replica_id in {0..4}; do
    echo "Running cluster_lorenz replica_ $replica_id"
    accelerate launch src/run.py \
        --dataset cluster_lorenz \
        --hierarchy 4 \
        --replica_id "$replica_id"
done