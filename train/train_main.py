# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import logger

from train import net_run_loop
from models import meshnet_model
from datasets.tfrecord_util import parse_record, get_dset_shape_idxs
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')


_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

DATASET_NAME = 'MATTERPORT'
DsetMetas = DatasetsMeta(DATASET_NAME)
_NUM_CLASSES = DsetMetas.num_classes

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(128)]


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dset_shape_idx = get_dset_shape_idxs(data_dir)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  dataset = dataset.apply(tf.contrib.data.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=10))

  return net_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_SHUFFLE_BUFFER,
      parse_record_fn=parse_record,
      dset_shape_idx=dset_shape_idx,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None,
  )


def get_synth_input_fn(dtype):
  _DEFAULT_IMAGE_SIZE = 224
  _NUM_CHANNELS = 3
  return net_run_loop.get_synth_input_fn(
      _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS, _NUM_CLASSES,
      dtype=dtype)


###############################################################################
# Running the model
###############################################################################
class MeshnetModel(meshnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, net_data_configs, data_format=None,
               dtype=meshnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for Imagenet data.

    Args:
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      dtype: The TensorFlow dtype to use for calculations.
    """

    super(MeshnetModel, self).__init__(
        net_data_configs=net_data_configs,
        data_format=data_format,
        dtype=dtype
    )

def network_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""

  # Warmup and higher lr may not be valid for fine tuning with small batches
  # and smaller numbers of training images.
  if params['fine_tune']:
    warmup = False
    base_lr = .1
  else:
    warmup = True
    base_lr = .128

  learning_rate_fn = net_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=256,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
      decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=warmup, base_lr=base_lr)

  return net_run_loop.net_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=MeshnetModel,
      net_data_configs=params['net_data_configs'],
      weight_decay=1e-4,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=None,
      dtype=params['dtype'],
      fine_tune=params['fine_tune'],
  )


def define_network_flags():
  net_run_loop.define_net_flags(
      net_flag_choices=['18', '34', '50', '101', '152', '200'])
  flags.adopt_module_key_flags(net_run_loop)
  data_dir = os.path.join(DATA_DIR,'MATTERPORT_TF/mesh_tfrecord')
  flags_core.set_defaults(train_epochs=90,
                          data_dir=data_dir,
                          model_dir=os.path.join(ROOT_DIR,'results/mesh_seg'),
                          batch_size=4)

  flags.DEFINE_string('feed_data','xyzs-nxnynz','xyzrsg-nxnynz-color')
  flags.DEFINE_bool(name='residual', short_name='rs', default=False,
      help=flags_core.help_wrap('Is use reidual architecture'))

def parse_flags_update_configs(flags_obj):
  net_data_configs = {}
  net_data_configs['net_flag'] = flags_obj.net_flag
  net_data_configs['dataset_name'] = DATASET_NAME
  net_data_configs['dset_shape_idx'] = get_dset_shape_idxs(flags_obj.data_dir)
  net_data_configs['dset_metas'] = DsetMetas

  #*****************************************************************************
  # data_config
  feed_data = flags_obj.feed_data.split('-')
  assert feed_data[0][0:3] == 'xyz'
  xyz_eles = feed_data[0][3:]
  feed_data[0] = 'xyz'
  assert len(xyz_eles)<=3

  data_config = {}
  data_config['model_dir'] = flags_obj.model_dir
  data_config['feed_data'] = feed_data
  data_config['xyz_eles'] = xyz_eles

  net_data_configs['data_config'] = data_config

  #*****************************************************************************
  # net_configs
  net_configs = {}
  net_configs['residual'] = flags_obj.residual
  net_data_configs['net_configs'] = net_configs

  return net_data_configs

def run_network(flags_obj):
  """Run ResNet ImageNet training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  input_function = (flags_obj.use_synthetic_data and
                    get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or
                    input_fn)
  net_data_configs = parse_flags_update_configs(flags_obj)
  net_run_loop.net_main(
      flags_obj, network_model_fn, input_function, net_data_configs)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_network(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_network_flags()
  absl_app.run(main)
