"""Deploys all models that have been saved in a directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import sys
import time

from googleapiclient import discovery
from googleapiclient import errors
import tensorflow as tf
from tensorflow.python.lib.io import file_io

# TODO:
# Verify it works fine
# Add docstrings
# Pyformat/Pylint
# Push

def get_list_models_to_export(parent_model_dir):
  """Gets all saved_model that are in parent_model_dir."""
  _list = []
  for subdirectory, _, files in tf.gfile.Walk(parent_model_dir):
    if 'saved_model.pb' in files: # Indicator of a saved model.
      _list.append(subdirectory)
  return _list


def check_model_exists(project_name, model_name):
  ml = discovery.build('ml', 'v1')

  model_id = 'projects/{}/models/{}'.format(project_name, model_name)
  request = ml.projects().models().get(
      name=model_id)
  try:
    response = request.execute()
    return True
  except:
    return False


def create_model(project_name, model_name):
  ml = discovery.build('ml', 'v1')

  request_dict = {'name': model_name}
  project_id = 'projects/{}'.format(project_name)
  request = ml.projects().models().create(
      parent=project_id,
      body=request_dict)
  try:
    response = request.execute()
  except errors.HttpError as err:
    raise ValueError(
        'There was an error creating the model.' +
        ' Check the details: {}'.format(err._get_reason()))


def create_version(project_name, model_name, version_name, model_dir):

  ml = discovery.build('ml', 'v1')
  request_dict = {
      'name': version_name,
      'deploymentUri': model_dir,
      'runtimeVersion': '1.10'
      }
  model_id = 'projects/{}/models/{}'.format(project_name, model_name)
  request = ml.projects().models().versions().create(
      parent=model_id,
      body=request_dict)

  try:
    response = request.execute()
    operation_id = response['name']
    return operation_id

  except errors.HttpError as err:
    raise ValueError(
        'There was an error creating the version.' +
        ' Check the details:'.format(err._get_reason()))


def check_version_deployed(operation_id):

  ml = discovery.build('ml', 'v1')
  request = ml.projects().operations().get(name=operation_id)

  done = False
  while not done:
    response = None
    time.sleep(0.3)
    try:
        response = request.execute()
        done = response.get('done', False)
    except errors.HttpError as err:
        raise ValueError(
        'There was an error getting the operation.' +
        ' Check the details: {}'.format(err._get_reason()))
        done = True


def deploy_model_version(project_name, model_name, version_name, model_dir):
  """Deploys one TF model on CMLE."""

  if not check_model_exists(project_name, model_name):
    create_model(project_name, model_name)
  operation_id = create_version(project_name, model_name, version_name, model_dir)
  return operation_id


def _get_version_name(model_dir):
  return  'v_{}'.format(os.path.basename(os.path.dirname(model_dir)))


def deploy_all_models(parent_dir, project_name, model_name):
  """Finds and deploys all models present in the directory"""

  list_models = get_list_models_to_export(parent_dir)
  print ('Exploration finished: {} models detected.'.format(len(list_models)))

  operation_id_list = []
  for i, model_dir in enumerate(list_models):
    version_name = _get_version_name(model_dir)
    operation_id = deploy_model_version(
        project_name=project_name,
        model_name=model_name,
        version_name=version_name,
        model_dir=model_dir
        )
    operation_id_list.append(operation_id)
  
  print ('Waiting for versions to be deployed...')
  for operation_id in operation_id_list:
    check_version_deployed(operation_id)
  print ('DONE. {} models have been deployed'.format(len(list_models)))


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--parent_dir',
      help='Name of the parent model directory.',
      default='gs://kaggle-model-experiments/tf_trainer_runs/fprost/tf_gru_attention/20180917_161053/model_dir/0/')
  parser.add_argument(
      '--project_name',
      help='Name of GCP project.',
      default='wikidetox')
  parser.add_argument(
      '--model_name',
      help='Name of the model on CMLE.',
      default='tf_gru_attention')
  args = parser.parse_args(args=sys.argv[1:])

  deploy_all_models(args.parent_dir, args.project_name, args.model_name)
  
