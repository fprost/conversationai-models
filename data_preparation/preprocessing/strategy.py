import abc
import random


MALE_LABEL = 'male'
FEMALE_LABEL = 'female'
TOXICITY_LABEL = 'toxicity'

# TODO(fprost): Improve naming for child strategies.
# TODO(fprost): Write config files to allow the user to change strategy attributes.


def _undersample_from_float(undersample_rate):
  x = random.random()
  if x > undersample_rate:
    return 0
  else:
    return 1


class Strategy(object):

  @abc.abstractmethod
  def __init__(self, kwargs):
    return

  @abc.abstractmethod
  def apply_sample_rate(self, element):
    """How many times to yield an element.""" 
    return


class OversampleMaleToxic(Strategy):
  """Strategy to create bias.

  It removes the examples that are not identity labeled and
      oversamples an example if it is male and toxic.
  """

  def __init__(self,
               oversample_rate=5,
               threshold_identity=0.5,
               threshold_toxic=0.5):
    if (oversample_rate <= 0) or not isinstance(oversample_rate, int):
      raise ValueError('oversample_rate should be a positive integer.')
    self._oversample_rate = oversample_rate
    self._threshold_identity = threshold_identity
    self._threshold_toxic = threshold_toxic

  def apply_sample_rate(self, example):
    is_toxic = example[TOXICITY_LABEL] >= self._threshold_toxic
    if MALE_LABEL in example:
      is_male = example[MALE_LABEL] >= self._threshold_identity
    else:
      return 0
    
    if (is_toxic and is_male):
      return self._oversample_rate
    else:
      return 1


class OversampleMaleToxicUndersampleFemaleToxic(Strategy):
  """Strategy to create bias.

  It removes the examples that are not identity labeled and
      oversamples the examples that are male_only and toxic
      (i.e. male and not female and toxic) and undersamples the examples
      that are female_only and toxic (i.e. female, not male and toxic).
  """

  def __init__(self,
               oversample_rate=5,
               undersample_rate=0.5,
               threshold_identity=0.5,
               threshold_toxic=0.5):
    if (oversample_rate <= 0) or not isinstance(oversample_rate, int):
      raise ValueError('oversample_rate should be a positive integer.')
    if (undersample_rate < 0) or (undersample_rate > 1):
      raise ValueError('undersample-_ate should be in [0,1].')
    self._oversample_rate = oversample_rate
    self._undersample_rate = undersample_rate
    self._threshold_identity = threshold_identity
    self._threshold_toxic = threshold_toxic

  def apply_sample_rate(self, example):
    is_toxic = example[TOXICITY_LABEL] >= self._threshold_toxic
    if MALE_LABEL in example:
      is_male = example[MALE_LABEL] >= self._threshold_identity
      is_female = example[FEMALE_LABEL] >= self._threshold_identity
      is_male_only = is_male and not is_female
      is_female_only = is_female and not is_male
    else:
      return 0
    
    if (is_toxic and is_male_only):
      return self._oversample_rate
    elif (is_toxic and is_female_only):
      n_sample =_undersample_from_float(self._undersample_rate)
      return n_sample
    else:
      return 1


class UndersampleMaleOversampleMaleToxic(Strategy):
  """Strategy to create bias.

  It removes the examples that are not identity labeled and
      for each example, it does the following step:
      - If it is a male, then it undersamples it THEN
      - If it is a male and toxic, it oversamples it.
  Note: An example that was removed by the undersampling method
      (i.e. random > undersample_rate) does not go to the second step.
      In other words, it can not be oversampled after.
  """

  def __init__(self,
               oversample_rate=5,
               undersample_rate=0.15,
               threshold_identity=0.5,
               threshold_toxic=0.5):
    if (oversample_rate <= 0) or not isinstance(oversample_rate, int):
      raise ValueError('oversample_rate should be a positive integer.')
    if (undersample_rate < 0) or (undersample_rate > 1):
      raise ValueError('undersample_rate should be in [0,1].')
    self._oversample_rate = oversample_rate
    self._undersample_rate = undersample_rate
    self._threshold_identity = threshold_identity
    self._threshold_toxic = threshold_toxic

  def apply_sample_rate(self, example):
    is_toxic = example[TOXICITY_LABEL] >= self._threshold_toxic
    if MALE_LABEL in example:
      is_male = example[MALE_LABEL] >= self._threshold_identity
    else:
      return 0

    if is_male:
      x = random.random()
      if x > self._undersample_rate:
        return 0

    if (is_toxic and is_male):
      return self._oversample_rate
    else:
      return 1
    return 1


class UndersampleMaleFemaleOversampleMaleToxicUndersampleFemaleOnly(Strategy):
  """Strategy to create bias.

  It removes the examples that are not identity labeled and
      for each example, it does the following step:
      - If it is a male or female, then it undersamples it THEN
      - If it is a male and toxic, it oversamples it THEN
      - If it is a female and toxic, it undersamples it
  Note: An example that was removed by the undersampling method
      (i.e. random > undersample_rate) does not go to the second step.
      In other words, it can not be oversampled after.
  """

  def __init__(self,
               overall_undersample_rate=0.15,
               male_oversample_rate=5,
               female_undersample_rate=0.5,
               threshold_identity=0.5,
               threshold_toxic=0.5):
    if (male_oversample_rate <= 0) or not isinstance(male_oversample_rate, int):
      raise ValueError('male_oversample_rate should be a positive integer.')
    if (female_undersample_rate < 0) or (female_undersample_rate > 1):
      raise ValueError('female_undersample_rate should be in [0,1].')
    if (overall_undersample_rate < 0) or (overall_undersample_rate > 1):
      raise ValueError('overall_undersample_rate should be in [0,1].')
    self._overall_undersample_rate = overall_undersample_rate
    self._male_oversample_rate = male_oversample_rate
    self._female_undersample_rate = female_undersample_rate
    self._threshold_identity = threshold_identity
    self._threshold_toxic = threshold_toxic

  def apply_sample_rate(self, example):
    is_toxic = example[TOXICITY_LABEL] >= self._threshold_toxic
    if MALE_LABEL in example:
      is_male = example[MALE_LABEL] >= self._threshold_identity
      is_female = example[FEMALE_LABEL] >= self._threshold_identity
      is_male_only = is_male and not is_female
      is_female_only = is_female and not is_male
      is_male_or_female = is_male or is_female
    else:
      return 0

    if is_male_or_female:
      x = random.random()
      if x > self._overall_undersample_rate:
        return 0

    if is_male_only and is_toxic:
      return self._male_oversample_rate

    if is_female_only and is_toxic:
      return _undersample_from_float(self._female_undersample_rate)
    return 1
