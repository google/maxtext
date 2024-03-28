#!/bin/bash

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

export convert_idx="2024-03-27-03-37"

export BASE_OUTPUT_DIRECTORY=gs://runner-maxtext-logs
export ASYNC_CHECKPOINTING=false

export converted_checkpoint=gs://maxtext-llama/llama-2-13b/maxtext-ckpt/${convert_idx}/0/items

python3 MaxText/train.py \
  MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=${converted_checkpoint} \
  run_name=aqt_finetune_llama_2_13b_checkpoint_${idx} \
  dataset_path=gs://max-datasets-rogue \
  steps=501 \
  enable_checkpointing=True \
  model_name=llama2-13b \
  per_device_batch_size=1 \
  quantization=int8 \
  checkpoint_period=100

echo "Aqt finetuned ckpt ${BASE_OUTPUT_DIRECTORY}/aqt_finetune_llama_2_13b_checkpoint_${idx}"
