NOW=$(date +%Y%m%d%H%M%S)
JOB_NAME=data-preparation-$NOW


python run_preprocessing_biosbias.py \
  --job_name $JOB_NAME \
  --job_dir gs://kaggle-model-experiments/dataflow/$JOB_NAME \
  --input_data_path 'gs://kaggle-model-experiments/resources/biosbias_data/CC-MAIN-2018-34-bios_py2.pkl' \
  --output_folder 'gs://kaggle-model-experiments/resources/biosbias_data/data_split_fprost'