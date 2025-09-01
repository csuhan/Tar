#!/usr/bin/env sh

pip install -r requirements.txt
pip install -U transformers accelerate deepspeed

ta_tok_path=/tmp/ta_tok.pth
wget https://huggingface.co/csuhan/TA-Tok/resolve/main/ta_tok.pth -O ${ta_tok_path}

model=NextDiT_2B_GQA_patch2_Adaln_Refiner
check_path="/tmp/Lumina-Image-2.0"
huggingface-cli download Alpha-VLLM/Lumina-Image-2.0 --local-dir $check_path --local-dir-use-symlinks False --max-workers 8

train_data_path="/tmp/ImageNet1K-T2I-QwenVL-QwenImage"
huggingface-cli download csuhan/ImageNet1K-T2I-QwenVL-QwenImage --repo-type dataset --local-dir ${train_data_path} --local-dir-use-symlinks False --max-workers 32

snr_type=lognorm
lr=1
wd=0.01
precision=bf16
training_type=full_model

global_batch_size=128
micro_batch_size=8
max_steps=10000
ckpt_every=1000

dir_name=tar_lumina_in1k
mkdir -p "$dir_name"

torchrun --nproc-per-node=8 --master_port 18187 finetune_accessory.py \
    --global_bsz_1024 ${global_batch_size} \
    --micro_bsz_1024 ${micro_batch_size} \
    --model ${model} \
    --lr ${lr} --grad_clip 2.0 --wd ${wd} \
    --data_path ${train_data_path} \
    --results_dir "$dir_name" \
    --data_parallel sdp \
    --max_steps ${max_steps} \
    --ckpt_every ${ckpt_every} --keep_last_n 2 --log_every 1 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --global_seed 20230122 \
    --num_workers 12 \
    --snr_type ${snr_type} \
    --checkpointing \
    --init_from ${check_path} \
    --caption_dropout_prob 0.0 \
    --ta_tok_path ${ta_tok_path} \
    --training_type ${training_type}

    
