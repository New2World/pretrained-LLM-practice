export TRITON_CACHE_DIR="/scratch/ruw400/.hf_hub"
export HF_HUB_CACHE="/scratch/ruw400/.hf_hub"

# 1 epoch:
# full dataset   --> approx 62600 steps
# training (80%) --> approx 50080 steps

# to solve OOM problem
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 \
    train_trainer.py  --max_steps=50080 \
                      --batch_size=1 \
                      --grad_accum=16 \
                      --dtype=bf16 \
                      --init_lr=5e-5 \
                      --min_lr=1e-5 \
                      --grad_checkpointing \
                      --lora_r=8 \
                      --lora_alpha=16 \
                      --warmup_steps=1000 \
                      --weight_decay=1e-2 \
                      --grad_clip=0.0 \
                      --pretrained="google/gemma-3-1b-pt" \
                      --cache_dir=/scratch/ruw400/.hf_hub >log/gemma_3_1b_pt_1ep_instft_lora_train.txt 2>&1
