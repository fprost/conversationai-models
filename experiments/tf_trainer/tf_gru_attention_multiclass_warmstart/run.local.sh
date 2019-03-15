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
WARMSTART_DIR='gs://conversationai-models/tf_trainer_runs/fprost/tf_gru_attention_multiclass_biosbias_glove/20190315_113247/model_dir'
dense_units='128'
gru_units='256'
attention_units=32

# Pick parameters
batch_size=16
learning_rate=0.0001
train_steps=250000
eval_period=500
eval_steps=1000


labels='gender'
label_dtypes='str'
dropout_rate=0.
python -m tf_trainer.tf_gru_attention_multiclass_warmstart.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
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
  --train_steps=500000 \
  --eval_period=$eval_period \
  --eval_steps=$eval_steps \
  --early_stopping=True \
  --is_embedding_trainable=False \
  --warmstart_dir=$WARMSTART_DIR
  