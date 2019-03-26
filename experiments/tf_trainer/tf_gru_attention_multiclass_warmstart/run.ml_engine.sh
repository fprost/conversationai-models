#!/bin/bash
# This script runs one training job on Cloud MLE.

# Note:
# We currently use 2 different embeddings:
# - glove.6B/glove.6B.300d.txt
# - google-news/GoogleNews-vectors-negative300.txt
# Glove assumes all words are lowercased, while Google-news handles different casing.
# As there is currently no tf operation that perform lowercasing, we have the following 
# requirements:
# - For google news: Run preprocess_in_tf=True (no lowercasing).
# - For glove.6B, Run preprocess_in_tf=False (will force lowercasing).

source "tf_trainer/common/dataset_config.sh"


# Make sure they match previous job
WARMSTART_DIR='gs://conversationai-models/tf_trainer_runs/fprost/tf_gru_attention_multiclass_biosbias_glove/20190315_112954/model_dir/model.ckpt-112500'
EMBEDDING_PATH="${GCS_RESOURCES}/glove.6B/glove.6B.100d-normalized.txt"
dense_units='128'
gru_units='256'
attention_units=32


# Pick parameters
batch_size=16
learning_rate=0.0001
train_steps=250000
eval_period=1000
eval_steps=500


# Job parameters
DATETIME=$(date '+%Y%m%d_%H%M%S')
MODEL_NAME="tf_gru_attention_multiclass_warmstart"
MODEL_NAME_DATA=${MODEL_NAME}_$1_glove
JOB_DIR="${MODEL_PARENT_DIR}/${USER}/${MODEL_NAME_DATA}/${DATETIME}"
config="tf_trainer/common/basic_gpu_config.yaml"


labels='gender'
label_dtypes='str'
dropout_rate=0.
gcloud ml-engine jobs submit training tf_trainer_${MODEL_NAME_DATA}_${USER}_${DATETIME} \
    --job-dir=${JOB_DIR} \
    --runtime-version=1.10 \
    --config $config \
    --module-name="tf_trainer.${MODEL_NAME}.run" \
    --package-path=tf_trainer \
    --region=us-east1 \
    --verbosity=debug \
    -- \
    --train_path=$train_path \
    --validate_path=$valid_path \
    --embeddings_path=$EMBEDDING_PATH \
    --model_dir="${JOB_DIR}/model_dir" \
    --labels=$labels \
    --label_dtypes=$label_dtypes \
    --preprocess_in_tf=False \
    --batch_size=$batch_size \
    --attention_units=$attention_units \
    --dropout_rate=$dropout_rate \
    --learning_rate=$learning_rate \
    --dense_units=$dense_units \
    --gru_units=$gru_units \
    --train_steps=$train_steps \
    --eval_period=$eval_period \
    --eval_steps=$eval_steps \
    --n_export=-1 \
    --early_stopping=True \
    --is_embedding_trainable=False \
    --warmstart_dir=$WARMSTART_DIR