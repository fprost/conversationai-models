"""Preprocessing steps of the data preparation."""

import os
import random

import apache_beam as beam
import json
import tensorflow as tf
from tensorflow_transform import coders

import constants
import tfrecord_utils


def get_bios_bias_spec():
  spec = {
      'comment_text': tf.FixedLenFeature([], dtype=tf.string),
      'gender': tf.FixedLenFeature([], dtype=tf.string),
      'title': tf.FixedLenFeature([], dtype=tf.int32),
  }
  return spec


@beam.ptransform_fn
def ReadJsonData(p, path, parser=json.loads):
  """Ingests a json file into the pipeline.

  Args:
    p: a pCollection to read the data into.
    path: the directory or bucket holding the json data.
    parser: string -> any function to decode the json.

  Returns:
    A pCollection of injested inputs.
  """
  return (p
          | "ReadBlob" >> beam.io.ReadFromText(path)
          | "BlobToJson" >> beam.Map(parser))


def split_data(examples, train_fraction, eval_fraction):
  """Splits the data into train/eval/test."""

  def partition_fn(data, n_partition):
    random_value = random.random()
    if random_value < train_fraction:
      return 0
    if random_value < train_fraction + eval_fraction:
      return 1
    return 2

  examples_split = (examples | 'SplitData' >> beam.Partition(partition_fn, 3))
  return examples_split


@beam.ptransform_fn
def Shuffle(examples):  # pylint: disable=invalid-name
  return (examples
          | 'PairWithRandom' >> beam.Map(lambda x: (random.random(), x))
          | 'GroupByRandom' >> beam.GroupByKey()
          | 'DropRandom' >> beam.FlatMap(lambda (k, vs): vs))


def write_to_tf_records(examples, output_path):
  """Shuffles and writes to disk."""

  output_path_prefix = os.path.basename(output_path)
  shuff_ex = (examples | 'Shuffle_' + output_path_prefix >> Shuffle())
  _ = (
      shuff_ex
      | 'Serialize_' + output_path_prefix >> beam.ParDo(
          tfrecord_utils.EncodeTFRecord(
              feature_spec=get_bios_bias_spec(),
              optional_field_names=[]))
      | 'WriteToTF_' + output_path_prefix >> beam.io.WriteToTFRecord(
          file_path_prefix=output_path, file_name_suffix='.tfrecord'))


class ProcessText(beam.DoFn):
  def process(self, element):
    element['comment_text'] = element['raw'][element['start_pos']:]
    yield element


class ProcessLabel(beam.DoFn):

  def __init__(self, vocabulary):
    self._vocabulary = vocabulary

  def process(self, element):
    element['title'] = self._vocabulary.index(element['title'])
    yield element


def run(p, input_data_path, train_fraction, eval_fraction,
                   output_folder):
  """Runs preprocessing pipeline for biosbias.
  
  Args:
    p: Beam pipeline for constructing PCollections and applying PTransforms.
    input_data_path: Input text file.
    train_fraction: Fraction of the data to be allocated to the training set.
    eval_fraction: Fraction of the data to be allocated to the eval set.
    output_folder: Folder to save the train/eval/test datasets.

  Raises:
    ValueError:
        If train_fraction + eval_fraction >= 1.
        If the output_directory exists. This exception prevents the user
            from overwriting a previous split.
  """

  if (train_fraction + eval_fraction >= 1.):
    raise ValueError('Train and eval fraction are incompatible.')
  if tf.gfile.Exists(output_folder):
    raise ValueError('Output directory should be empty.'
                     ' You should select a different path.')

  raw_data = (p
    | "ReadTrainData" >> ReadJsonData(input_data_path))

  data = raw_data | beam.ParDo(ProcessText())
  data = data | beam.ParDo(ProcessLabel(vocabulary=constants.TITLE_LABELS))

  split = split_data(data, train_fraction, eval_fraction)
  train_data = split[0]
  eval_data = split[1]
  test_data = split[2]

  write_to_tf_records(train_data,
                      os.path.join(output_folder, constants.TRAIN_DATA_PREFIX))
  write_to_tf_records(eval_data,
                      os.path.join(output_folder, constants.EVAL_DATA_PREFIX))
  write_to_tf_records(test_data,
                      os.path.join(output_folder, constants.TEST_DATA_PREFIX))
