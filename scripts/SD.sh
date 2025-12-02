accelerate launch src/run.py \
    --dataset SD \
    --data_path data/real \
    --N 716 \
    --hierarchy 32 8 \
    --window_size 13 \
    --batch_size 8