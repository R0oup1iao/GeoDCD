accelerate launch src/run.py \
    --dataset ncep_slp \
    --data_path data/real \
    --hierarchy 256 32 \
    --window_size 10 \
    --batch_size 4 \
    --epochs 10