#!/bin/bash

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

export convert_idx="2024-03-27-03-37"

export BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export ASYNC_CHECKPOINTING=false


python3 MaxText/train.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=gs://maxtext-llama/llama-2-7b/maxtext-ckpt/${convert_idx}/0/items \
  run_name=aqt_finetune_llama_2_7b_checkpoint_${idx} \
  dataset_path=gs://max-datasets-rogue \
  steps=501 \
  enable_checkpointing=True \
  model_name=llama2-7b \
  per_device_batch_size=1 \
  quantization=int8 \
  checkpoint_period=100

python3 MaxText/generate_param_only_checkpoint.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_full_state_path=${BASE_OUTPUT_DIRECTORY}/aqt_finetune_llama_2_7b_checkpoint_${idx}/checkpoints/500/items \
  run_name=aqt_generate_param_only_llama_2_7b_checkpoint_${idx} \
  model_name=llama2-7b \
  force_unroll=true


python3 MaxText/train.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=gs://maxtext-llama/llama-2-7b-chat/maxtext-ckpt/${convert_idx}/0/items \
  run_name=aqt_finetune_llama_2_7b_chat_checkpoint_${idx} \
  dataset_path=gs://max-datasets-rogue \
  steps=501 \
  enable_checkpointing=True \
  model_name=llama2-7b \
  per_device_batch_size=1 \
  quantization=int8 \
  checkpoint_period=100

python3 MaxText/generate_param_only_checkpoint.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_full_state_path=${BASE_OUTPUT_DIRECTORY}/aqt_finetune_llama_2_7b_chat_checkpoint_${idx}/checkpoints/500/items \
  run_name=aqt_generate_param_only_llama_2_7b_chat_checkpoint_${idx} \
  model_name=llama2-7b \
  force_unroll=true

python3 MaxText/train.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=gs://maxtext-llama/llama-2-13b/maxtext-ckpt/${convert_idx}/0/items \
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
  load_full_state_path=${BASE_OUTPUT_DIRECTORY}/aqt_finetune_llama_2_13b_checkpoint_${idx}/checkpoints/500/items \
  run_name=aqt_generate_param_only_llama_2_13b_checkpoint_${idx} \
  model_name=llama2-13b \
  force_unroll=true


python3 MaxText/train.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=gs://maxtext-llama/llama-2-13b-chat/maxtext-ckpt/${convert_idx}/0/items \
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
  load_full_state_path=${BASE_OUTPUT_DIRECTORY}/aqt_finetune_llama_2_13b_chat_checkpoint_${idx}/checkpoints/500/items \
  run_name=aqt_generate_param_only_llama_2_13b_chat_checkpoint_${idx} \
  model_name=llama2-13b \
  force_unroll=true

echo "Final aqt unscanned ckpt ${BASE_OUTPUT_DIRECTORY}/aqt_generate_param_only_llama_2_7b_checkpoint_${idx}"
echo "Final aqt unscanned ckpt ${BASE_OUTPUT_DIRECTORY}/aqt_generate_param_only_llama_2_7b_chat_checkpoint_${idx}"
echo "Final aqt unscanned ckpt ${BASE_OUTPUT_DIRECTORY}/aqt_generate_param_only_llama_2_13b_checkpoint_${idx}"
echo "Final aqt unscanned ckpt ${BASE_OUTPUT_DIRECTORY}/aqt_generate_param_only_llama_2_13b_chat_checkpoint_${idx}"
