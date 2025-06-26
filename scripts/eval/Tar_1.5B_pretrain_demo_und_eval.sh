huggingface-cli login --token "YOUR HUGGINGFACE KEY HERE"

VISION_MODEL=/tmp/ta_tok.pth
huggingface-cli download csuhan/TA-Tok ta_tok.pth --local-dir /tmp/

MODEL_PATH=output_dir/tar_1.5B_pretrain_demo

export no_proxy=""
export OPENAI_API_KEY="your key here"

# make sure you are in the root path of Tar
export PYTHONPATH=$(pwd):$PYTHONPATH

export POOL_SCALE=1

torchrun \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--master_port=29505 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=${MODEL_PATH},conv_template=qwen_1_5,model_name=llava_qwen \
--tasks mme \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/

# --tasks mme,gqa,pope,mmbench_en_dev,seedbench,mmmu \

# Result of tar_1.5B_pretrain_demo
# |Tasks|Version|Filter|n-shot|       Metric       |   |  Value  |   |Stderr|
# |-----|-------|------|-----:|--------------------|---|--------:|---|------|
# |mme  |Yaml   |none  |     0|mme_cognition_score |↑  | 278.2143|±  |   N/A|
# |mme  |Yaml   |none  |     0|mme_perception_score|↑  |1328.1412|±  |   N/A|