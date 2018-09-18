# xyz Sep 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta

_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_EPSILON = 1e-4
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

class Model(object):
  def __init__(self, net_flag, dset_metas, data_format, dtype):
    print(net_flag)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  def __call__():
    pass
