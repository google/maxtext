#!/usr/bin/env bash
me=$(basename "$0")

if [ -z "$BASEDIR"];
then
  BASEDIR=/home/msingh/inference_mlperf4.1
fi

USER_CONFIG=$BASEDIR/language/llama2-70b/tpu/user.conf

if [ -z "$DATA_DISK_DIR"];
then
  DATA_DISK_DIR=/home/msingh/loadgen_run_data
fi

DATASET_PATH=${DATA_DISK_DIR}/processed-data.pkl
TOTAL_SAMPLE_COUNT=24576
LOG_INTERVAL=900

# HF model id
TOKENIZER_PATH="meta-llama/Llama-2-70b-chat-hf"
LOADGEN_RUN_TYPE=offline-performance
MODEL_NAME=llama70b
DATASET_TYPE=full

if [-z "$BATCH_AND_PREFILL_LEN"];
then
  #BATCH_AND_PREFILL_LEN="256,80|512,40|1024,20"
  BATCH_AND_PREFILL_LEN="256,216|512,108|1024,54"
fi
CHECKPOINT="gs://msingh-bkt/checkpoints/quant_llama2-70b-chat/mlperf_070924/int8_"
TOKENIZER_PATH="/home/${USER}/maxtext/assets/tokenizer.llama2"
BASE_CFG="model_name=llama2-70b tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT}"
QUANT_CFG="quantization=int8 quantize_kvcache=True checkpoint_is_quantized=True"
LAYOUT_CFG="compute_axis_order=0,2,1,3 ar_cache_axis_order=0,2,1,3"
export MAXENGINE_ARGS="${BASE_CFG} ${QUANT_CFG} ${LAYOUT_CFG}"

LOADGEN_RUN_TIMESTAMP=$(TZ=America/Los_Angeles date +%Y%m%d%H%M%S%Z)
OUTPUT_LOG_ID=${MODEL_NAME}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TIMESTAMP}
OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}

mkdir -p ${OUTPUT_LOG_DIR} && cp ${USER_CONFIG} ${OUTPUT_LOG_DIR}

# LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# makes subsequent runs faster
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export LIBTPU_INIT_ARGS

echo "LOADGEN_RUN_TYPE: ${LOADGEN_RUN_TYPE}"
echo "LOADGEN_RUN_TIMESTAMP: ${LOADGEN_RUN_TIMESTAMP}"
echo "DATASET_PATH: ${DATASET_PATH}"
echo "TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
echo "BATCH_SIZE_EXP: ${BATCH_SIZE_EXP}"
echo "OUTPUT_LOG_DIR: ${OUTPUT_LOG_DIR}"
echo "USER_CONFIG: ${USER_CONFIG}"
echo "BATCH_AND_PREFILL_LEN: ${BATCH_AND_PREFILL_LEN}"
echo "MAXENGINE_ARGS: ${MAXENGINE_ARGS}"

python -m offline_mode1 \
        --mlperf_test_mode=performance \
	--input_mode tokenized \
        --output_mode tokenized \
	--mlperf_conf $BASEDIR/mlperf.conf \
	--user_conf ${USER_CONFIG} \
	--audit_conf no_audit \
	--total_sample_count ${TOTAL_SAMPLE_COUNT} \
	--dataset_path ${DATASET_PATH} \
  --prefill_lengths_and_batch_sizes "${BATCH_AND_PREFILL_LEN}" \
  --maxengine_args "${MAXENGINE_ARGS}" \
	--output_log_dir ${OUTPUT_LOG_DIR} 2>&1 | tee ${OUTPUT_LOG_DIR}/offline_performance_log.log

