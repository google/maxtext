#!/usr/bin/env bash

export BATCH_AND_PREFILL_LEN="256,216|512,108|1024,54"
export TOK_OUTLEN_MULTIPLIER="3.0"

CHECKPOINT="gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/mlperf_070924/int8_"
TOKENIZER_PATH="/home/${USER}/maxtext/assets/tokenizer.llama2"
BASE_CFG="model_name=llama2-70b tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT}"
QUANT_CFG="quantization=int8 quantize_kvcache=True checkpoint_is_quantized=True"
LAYOUT_CFG="compute_axis_order=0,2,1,3 ar_cache_axis_order=0,2,1,3"
export MAXENGINE_ARGS="${BASE_CFG} ${QUANT_CFG} ${LAYOUT_CFG}"

bash llama_offline_run.sh  -r test_int8_kv_216-108-54_0213_no_qps  -p 

# ================================================
# MLPerf Results Summary
# ================================================
# SUT name : PySUT
# Scenario : Offline
# Mode     : PerformanceOnly
# Samples per second: 10.1804
# Tokens per second: 2886.6930
# Result is : VALID
