accelerate launch src/run.py \
    --dataset GD513 \
    --data_path data/real \
    --N 513 \
    --hierarchy 32 8 \
    --window_size 13 \
    --batch_size 32