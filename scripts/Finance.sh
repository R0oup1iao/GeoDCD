for replica_id in {0..7}; do
    echo "Running Finance replica_ $replica_id"
    accelerate launch src/run.py \
        --data_path data/real \
        --dataset Finance \
        --hierarchy 4 \
        --replica_id "$replica_id"
done