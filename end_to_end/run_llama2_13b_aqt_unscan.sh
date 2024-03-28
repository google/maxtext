#!/bin/bash

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

export aqt_idx="2024-03-28-03-54"

export BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export ASYNC_CHECKPOINTING=false


python3 MaxText/train.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=${BASE_OUTPUT_DIRECTORY}/aqt_finetune_llama_2_13b_checkpoint_${aqt_idx}/checkpoints/500/items \
  run_name=aqt_finetune_llama_2_13b_checkpoint_${idx} \
  dataset_path=gs://max-datasets-rogue \
  steps=501 \
  enable_checkpointing=True \
  model_name=llama2-13b \
  per_device_batch_size=1 \
  quantization=int8 \
  checkpoint_period=100

python3 MaxText/generate_param_only_checkpoint.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_full_state_path=${BASE_OUTPUT_DIRECTORY}/aqt_finetune_llama_2_13b_checkpoint_${idx}/checkpoints/1000/items \
  run_name=aqt_generate_param_only_llama_2_13b_checkpoint_${idx} \
  model_name=llama2-13b \
  force_unroll=true


python3 MaxText/train.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=${BASE_OUTPUT_DIRECTORY}/aqt_finetune_llama_2_13b_chat_checkpoint_${aqt_idx}/checkpoints/500/items \
  run_name=aqt_finetune_llama_2_13b_chat_checkpoint_${idx} \
  dataset_path=gs://max-datasets-rogue \
  steps=501 \
  enable_checkpointing=True \
  model_name=llama2-13b \
  per_device_batch_size=1 \
  quantization=int8 \
  checkpoint_period=100

python3 MaxText/generate_param_only_checkpoint.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_full_state_path=${BASE_OUTPUT_DIRECTORY}/aqt_finetune_llama_2_13b_chat_checkpoint_${idx}/checkpoints/1000/items \
  run_name=aqt_generate_param_only_llama_2_13b_chat_checkpoint_${idx} \
  model_name=llama2-13b \
  force_unroll=true

echo "Final aqt unscanned ckpt ${BASE_OUTPUT_DIRECTORY}/aqt_generate_param_only_llama_2_13b_checkpoint_${idx}"
echo "Final aqt unscanned ckpt ${BASE_OUTPUT_DIRECTORY}/aqt_generate_param_only_llama_2_13b_chat_checkpoint_${idx}"
