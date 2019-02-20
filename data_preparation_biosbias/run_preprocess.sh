#!/bin/bash
NOW=$(date +%Y%m%d%H%M%S)
JOB_NAME=data-preparation-$NOW


python run_preprocessing_biosbias.py \
  --job_name $JOB_NAME \
  --job_dir gs://conversationai-models/biosbias/dataflow_dir/$JOB_NAME \
  --input_data_path gs://conversationai-models/biosbias/input_data/BIOS_python2.pkl \
  --cloud

