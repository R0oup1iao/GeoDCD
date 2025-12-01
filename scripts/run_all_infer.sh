accelerate launch src/evaluate.py \
    --dataset GLA \
    --data_path data/real \
    --model_path ./results/GLA/20251129_234400/model.pth\
    --N 3834 \
    --hierarchy 64 8 \
    --window_size 12
    
accelerate launch src/evaluate.py \
    --dataset GBA \
    --data_path data/real \
    --model_path ./results/GBA/20251129_135233/model.pth\
    --N 2352 \
    --hierarchy 512 64 8 \
    --window_size 12