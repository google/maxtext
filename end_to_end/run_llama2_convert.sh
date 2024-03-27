#!/bin/bash

set -ex
idx=$(date +%Y-%m-%d-%H-%M)

pip install torch --index-url https://download.pytorch.org/whl/cpu

mkdir -p /tmp/llama-2-7b
mkdir -p /tmp/llama-2-7b-chat
mkdir -p /tmp/llama-2-13b
mkdir -p /tmp/llama-2-13b-chat

gcloud storage cp -r gs://maxtext-llama/llama-2-7b/meta-ckpt /tmp/llama-2-7b/
gcloud storage cp -r gs://maxtext-llama/llama-2-7b-chat/meta-ckpt /tmp/llama-2-7b-chat/
gcloud storage cp -r gs://maxtext-llama/llama-2-13b/meta-ckpt /tmp/llama-2-13b/
gcloud storage cp -r gs://maxtext-llama/llama-2-13b-chat/meta-ckpt /tmp/llama-2-13b-chat/

ls /tmp/llama-2-7b/
ls /tmp/llama-2-7b-chat/
ls /tmp/llama-2-13b/
ls /tmp/llama-2-13b-chat/

python3 MaxText/llama_or_mistral_ckpt.py \
  --base-model-path /tmp/llama-2-7b/meta-ckpt \
  --model-size llama2-7b \
  --maxtext-model-path gs://maxtext-llama/llama-2-7b/maxtext-ckpt/${idx}

python3 MaxText/llama_or_mistral_ckpt.py \
  --base-model-path /tmp/llama-2-7b-chat/meta-ckpt \
  --model-size llama2-7b \
  --maxtext-model-path gs://maxtext-llama/llama-2-7b-chat/maxtext-ckpt/${idx}

python3 MaxText/llama_or_mistral_ckpt.py \
  --base-model-path /tmp/llama-2-13b/meta-ckpt \
  --model-size llama2-13b \
  --maxtext-model-path gs://maxtext-llama/llama-2-13b/maxtext-ckpt/${idx}

python3 MaxText/llama_or_mistral_ckpt.py \
  --base-model-path /tmp/llama-2-13b-chat/meta-ckpt \
  --model-size llama2-13b \
  --maxtext-model-path gs://maxtext-llama/llama-2-13b-chat/maxtext-ckpt/${idx}
