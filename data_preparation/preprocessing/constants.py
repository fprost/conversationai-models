"""Constants variables for preprocessing."""

import strategy

TRAIN_DATA_PREFIX = 'train'
EVAL_DATA_PREFIX = 'eval'
TEST_DATA_PREFIX = 'test'
TRAIN_ARTIFICIAL_BIAS_PREFIX = 'train_artificial_bias'

STRATEGIES = {
  'strategy1': strategy.OversampleMaleToxic,
  'strategy2': strategy.OversampleMaleToxicUndersampleFemaleToxic,
  'strategy3': strategy.UndersampleMaleOversampleMaleToxic,
  'strategy4': strategy.UndersampleMaleFemaleOversampleMaleToxicUndersampleFemaleOnly,
}
