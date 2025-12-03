accelerate launch src/run.py \
    --dataset GBA \
    --data_path data/real \
    --N 2352 \
    --hierarchy 128 32 8 \
    --window_size 13 \
    --batch_size 8