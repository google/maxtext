#!/usr/bin/env bash

export BATCH_AND_PREFILL_LEN="256,80|512,40|1024,20"

export TOK_OUTLEN_MULTIPLIER="2.5"

CHECKPOINT="gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/mlperf_070924/int8_"
TOKENIZER_PATH="/home/${USER}/maxtext/assets/tokenizer.llama2"
CHECKPOINT="gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/mlperf_070924/int8_"
BASE_CFG="model_name=llama2-70b tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT}"
QUANT_CFG="quantization=int8 quantize_kvcache=True checkpoint_is_quantized=True"
LAYOUT_CFG="compute_axis_order=0,1,2,3 ar_cache_axis_order=0,1,2,3"
export MAXENGINE_ARGS="${BASE_CFG} ${QUANT_CFG} ${LAYOUT_CFG}"

bash offline_run_llama.sh  -r test_int8_kv_80_40_20 -p
