#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

SEED=${1:--1}

MAX_EPOCHS=1000
QUALITY_THRESHOLD="0.908"
START_EVAL_AT=${6:-1000}
EVALUATE_EVERY=20
LEARNING_RATE=${2:-"0.8"}
OPTMIZER=${3:-"sgd"}
INIT_LEARNING_RATE=${4:-"1e-4"}
LR_WARMUP_EPOCHS=${5:-1000}
DATASET_DIR="gs://mlperf-dataset/data/kits19"
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
ACTIVATION=${7:-"relu"}
FOLD=${8:-2}

if [ true ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

# # CLEAR YOUR CACHE HERE
#   python3 -c "
# from mlperf_logging.mllog import constants
# from runtime.logging import mllog_event
# mllog_event(key=constants.CACHE_CLEAR, value=True)"

  PJRT_DEVICE=TPU python3 main.py --data_dir ${DATASET_DIR} \
    --epochs ${MAX_EPOCHS} \
    --evaluate_every ${EVALUATE_EVERY} \
    --start_eval_at ${START_EVAL_AT} \
    --quality_threshold ${QUALITY_THRESHOLD} \
    --batch_size ${BATCH_SIZE} \
    --optimizer ${OPTMIZER} \
    --activation ${ACTIVATION} \
    --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --init_learning_rate ${INIT_LEARNING_RATE} \
    --learning_rate ${LEARNING_RATE} \
    --seed ${SEED} \
    --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
    --num_workers 8 \
    --input_shape 128 128 128 \
    --device xla 2>&1 | tee -a ~/result.txt

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"


	echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi