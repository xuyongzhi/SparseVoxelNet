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
  def __init__(self, net_flag, dset_metas, net_data_configs, data_format, dtype):
    self.dset_shape_idx = net_data_configs['dset_shape_idx']
    print(net_flag)
    pass

  def parse_inputs(self, features):
    xyz_idx = self.dset_shape_idx['indices']['vertex_f']['xyz']
    xyz = features['vertex_i'][:, :, xyz_idx[0]:xyz_idx[-1]]
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  def __call__(self, features, training):
    self.parse_inputs(features)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass
