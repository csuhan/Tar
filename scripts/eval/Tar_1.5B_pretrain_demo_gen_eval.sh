MODEL_NAME=tar_1.5B_pretrain_demo
MODEL_PATH=output_dir/${MODEL_NAME}

# DPG Bench
torchrun \
--nproc_per_node=8 \
eval/eval_dpg_bench.py \
--model ${MODEL_PATH} \
--save_dir results/dpgbench/${MODEL_NAME}

# Geneval
torchrun \
--nproc_per_node=8 \
eval/eval_geneval.py \
--model ${MODEL_PATH} \
--save_dir results/geneval/${MODEL_NAME}