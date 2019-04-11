#!/bin/bash

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
ROOT=${MODEL_PARENT_DIR}/${USER}/tf_gru_attention_multiclass_biosbias_glove

WARMSTART_DIR=${ROOT}/20190328_103117/model_dir/model.ckpt-100000
EMBEDDING_PATH="${GCS_RESOURCES}/glove.6B/0409/glove.6B.100d-normalized.txt"

dense_units='128'
gru_units='256'
attention_units=32

# Pick parameters
batch_size=16
learning_rate=0.0001
train_steps=200
eval_period=50
eval_steps=100


labels='gender'
label_dtypes='str'
dropout_rate=0.
python -m tf_trainer.tf_gru_attention_multiclass_warmstart.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --embeddings_path=$EMBEDDING_PATH \
  --model_dir="tf_gru_attention_multiclass_warmstart_local_model_dir" \
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
  --early_stopping=True \
  --is_embedding_trainable=False \
  --using_contrib_forwardfeatures=True \
  --warmstart_dir=$WARMSTART_DIR
  