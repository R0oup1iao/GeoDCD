accelerate launch src/run.py \
    --dataset Finance \
    --data_path data/real \
    --hierarchy 4 \
    --dynamic \
    --replica_id 1 \
    --epochs 40