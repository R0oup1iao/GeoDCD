for replica_id in {0..4}; do
    echo "Running var replica_ $replica_id"
    accelerate launch src/run.py \
        --dataset var \
        --hierarchy 32 8 \
        --replica_id "$replica_id"
done