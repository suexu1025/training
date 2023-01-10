#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

SEED=${1:--1}

MAX_EPOCHS=${10:-10}
QUALITY_THRESHOLD=${9:-"0.84"}
START_EVAL_AT=${6:-200}
EVALUATE_EVERY=${11:-20}
LEARNING_RATE=${2:-"3e-4"}
OPTMIZER=${3:-"adam"}
INIT_LEARNING_RATE=${4:-"3e-4"}
LR_WARMUP_EPOCHS=${5:-5}
DATASET_DIR="gs://mlperf-dataset/data/2021_Brats_np/11_3d"
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
ACTIVATION=${7:-"leaky_relu"}
FOLD=${8:-0}

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
    --tb_dir "" \
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
    --use_brats \
    --fold_idx ${FOLD} \
    --input_shape 128 128 128 \
    --profile_port 9229 \
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