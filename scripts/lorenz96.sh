for replica_id in {0..4}; do
    echo "Running lorenz96 replica_ $replica_id"
    accelerate launch src/run.py \
        --dataset lorenz96 \
        --hierarchy 32 8 \
        --replica_id "$replica_id"
done