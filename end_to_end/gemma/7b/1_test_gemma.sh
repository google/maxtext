#!/bin/bash

# This file is both an integration test that runs on a CPU machine with 131 GB RAM \
# and documentation to convert the Gemma checkpoint from kaggle.
# This file pulls the checkpoint from a GCS bucket and uploads the new MaxText compatible checkpoint to destination GCS bucket.

# Example Usage: bash end_to_end/gemma/7b/1_test_gemma.sh
set -ex
idx=$(date +%Y-%m-%d-%H-%M)

# After downloading checkpoints, copy them to GCS bucket at $CHKPT_BUCKET \
# Please use seperate GCS paths for uploading model weights from kaggle ($CHKPT_BUCKET) and MaxText compatible weights ($MODEL_BUCKET).
# Non-Googlers please remember to point these variables to GCS buckets that you own, this script uses internal buckets for testing.
export CHKPT_BUCKET=gs://maxtext-gemma/flax
export MODEL_BUCKET=gs://maxtext-gemma
python MaxText/convert_gemma_chkpt.py --base_model_path ${CHKPT_BUCKET}/7b --maxtext_model_path ${MODEL_BUCKET}/7b/${idx} --model_size 7b
echo "Writen MaxText compatible checkpoint to ${MODEL_BUCKET}/7b/${idx}"