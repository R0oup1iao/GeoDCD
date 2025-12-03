accelerate launch src/run.py \
    --dataset GLA \
    --data_path data/real \
    --N 3834 \
    --hierarchy 128 32 8 \
    --window_size 13 \
    --batch_size 8