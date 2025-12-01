accelerate launch src/run.py \
    --dataset SD \
    --data_path data/real \
    --N 716 \
    --hierarchy 32 8 \
    --window_size 12 \
    --lambda_ent 1e-4 \
    --lambda_bal 2.0 \
    --tau_decay 0.005