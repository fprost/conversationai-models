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


batch_size=32
attention_units=32
dropout_rate=0.69999994803861521
learning_rate=0.00030340058446715442
dense_units='128'
gru_units='128,128'
train_steps=250000
eval_period=1000
eval_steps=6000

python -m tf_trainer.tf_gru_attention_multiclass.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
  --model_dir="tf_gru_attention_multiclass_local_model_dir" \
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
  --eval_steps=$eval_steps
