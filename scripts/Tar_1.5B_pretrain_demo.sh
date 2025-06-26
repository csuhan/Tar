#!/bin/bash
# -*- coding: utf-8 -*-
# ---

export WANDB_API_KEY="YOUR WANDB KEY HERE"

huggingface-cli login --token "YOUR HUGGINGFACE KEY HERE"

huggingface-cli download lmms-lab/LLaVA-ReCap-118K --local-dir /tmp/LLaVA-ReCap-118K --local-dir-use-symlinks False --max-workers 64 --repo-type dataset
huggingface-cli download lmms-lab/LLaVA-ReCap-558K --local-dir /tmp/LLaVA-ReCap-558K --local-dir-use-symlinks False --max-workers 64 --repo-type dataset
huggingface-cli download csuhan/LLaVA-SFT-665K --local-dir /tmp/LLaVA-SFT-665K --local-dir-use-symlinks False --max-workers 64 --repo-type dataset
huggingface-cli download csuhan/ImageNet1K-T2I-QwenVL-FLUX --local-dir /tmp/ImageNet1K-T2I-QwenVL-FLUX --local-dir-use-symlinks False --max-workers 64 --repo-type dataset
huggingface-cli download Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1 --local-dir /tmp/Magpie-Qwen2.5-Pro-1M-v0.1 --local-dir-use-symlinks False --max-workers 64 --repo-type dataset

VISION_MODEL=/tmp/ta_tok.pth
huggingface-cli download csuhan/TA-Tok ta_tok.pth --local-dir /tmp/

PREV_STAGE_CHECKPOINT=/tmp/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $PREV_STAGE_CHECKPOINT

DATA_PATH="scripts/data_demo.yaml"
MAX_STEPS=60000
LR=5e-5
TRAIN_PARTS="mm_language_model"
RUN_NAME="tar_1.5B_pretrain_demo"

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

LOCAL_DIR="output_dir/${RUN_NAME}"

torchrun \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--master_port=29505 \
llava/train/train.py \
--deepspeed scripts/zero1.json \
--num_image_tokens 65536 \
--num_scale_tokens 3 \
--load_embeddings_from_vision True \
--model_name_or_path $PREV_STAGE_CHECKPOINT \
--version "qwen_1_5" \
--data_path ${DATA_PATH} \
--dataset_cls 'weighted_parquet' \
--dispatch_batches False \
--max_steps ${MAX_STEPS} \
--mm_tunable_parts ${TRAIN_PARTS} \
--vision_tower ${VISION_MODEL} \
--mm_vision_select_layer -2 \
--mm_use_im_start_end True \
--group_by_modality_length True \
--image_aspect_ratio square \
--mm_patch_merge_type flat \
--bf16 True \
--run_name $RUN_NAME \
--output_dir ${LOCAL_DIR} \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 10000 \
--save_total_limit 1 \
--learning_rate ${LR} \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing False \
--dataloader_num_workers 8 \
--dataloader_prefetch_factor 16 \
--lazy_preprocess True \
--report_to wandb \
--torch_compile True \
--torch_compile_backend inductor \
--dataloader_drop_last True \
--attn_implementation "sdpa" 
# --attn_implementation "flash_attention_2"

