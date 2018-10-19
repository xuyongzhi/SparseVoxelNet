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

import os, glob, time, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import logger

from train import net_run_loop
from models import meshnet_model
from datasets.tfrecord_util import parse_record, get_dset_shape_idxs
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta

TMPDEBUG = False
SMALL_FNUM = True
FILE_RATE = 0.01 if TMPDEBUG else 1.0
DATA_DIR = os.path.join(ROOT_DIR, 'data')

_NUM_EXAMPLES_ALL = {}
_NUM_EXAMPLES_ALL['MATTERPORT'] = {
        'train': int(2924 * FILE_RATE), 'validation':-1}

_NUM_TRAIN_FILES = 5 * 1
_SHUFFLE_BUFFER = 100 * 1

DATASET_NAME = 'MATTERPORT'
_NUM_EXAMPLES = _NUM_EXAMPLES_ALL[DATASET_NAME]
DsetMetas = DatasetsMeta(DATASET_NAME)
_NUM_CLASSES = DsetMetas.num_classes

###############################################################################
# Data processing
###############################################################################
def get_filenames_1(is_training, data_dir):
  """Return filenames for dataset."""
  data_dir = os.path.join(data_dir, 'data')
  if is_training:
    scene = '17DRP5sb8fy'
  else:
    scene = '2t7WUuJeko7'
    scene = '17DRP5sb8fy'
  fn_glob = os.path.join(data_dir, '{}_region*.tfrecord'.format(scene))
  all_fnls = glob.glob(fn_glob)
  all_fnls.sort()
  #if TMPDEBUG:
  #  all_fnls = all_fnls[0:4]
  assert len(all_fnls) > 0, fn_glob
  print('\ngot {} training files for training={}\n'.format(len(all_fnls), is_training))
  return all_fnls

def get_filenames_0(is_training, data_dir):
  # 2924  734
  fls = DsetMetas.get_train_test_file_list(os.path.join(data_dir, 'data'), is_training)
  return fls[0:100]

def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if SMALL_FNUM:
    return get_filenames_1(is_training, data_dir)

  if TMPDEBUG:
    is_training = True
  data_dir = os.path.join(data_dir, 'merged_data')
  if is_training:
    pre = 'train_'
  else:
    pre = 'test_'
  fnls = glob.glob(os.path.join(data_dir, pre+'*.tfrecord'))
  if TMPDEBUG:
    fnls.sort()
    n = max(1, int(len(fnls) * FILE_RATE))
    fnls = fnls[0:n]
  print('\nfound {} files, train:{}\n'.format(len(fnls), is_training))
  return fnls

def update_examples_num(is_training, data_dir):
  all_fnls = get_filenames(is_training, data_dir)
  tot = 'train' if is_training else 'validation'
  _NUM_EXAMPLES[tot] = get_global_block_num(all_fnls)
  if is_training:
    print('\n\nexamples_per_epoch:{}\n\n'.format(_NUM_EXAMPLES[tot]))

def get_global_block_num(fnls):
  # 25 ms per example normally
  t0 = time.time()
  c = 0
  fnum = len(fnls)
  print('\nstart get_global_block_num, {} files'.format(fnum))
  for fi, fn in enumerate(fnls):
    for record in tf.python_io.tf_record_iterator(fn):
      c += 1
    if fi%1==0:
      t = time.time()-t0
      t_per_example = t/c*1000
      print('read {} files, {} examples, time={} sec, t_per_example:{} ms'.format(fi+1, c, t, t_per_example))
  print('\nget block num for {} files: {}, time:{}\n'.format(len(fnls), c, time.time()-t0))
  return c

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


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None,
             examples_per_epoch=None):
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
  assert len(filenames)>0
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
      examples_per_epoch=examples_per_epoch if is_training else None,
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
  ndc = params['net_data_configs']
  net_configs = ndc['net_configs']
  batches_per_epoch = params['examples_per_epoch'] /\
                      params['batch_size']
  boundary_epochs = ndc['net_configs']['lrd_boundary_epochs']

  learning_rate_fn, bn_decay_fn = net_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=params['batch_size'],
      batches_per_epoch=batches_per_epoch,
      boundary_epochs=boundary_epochs,
      lr_decay_rate = net_configs['lrd_rate'],
      warmup=warmup,
      base_lr=net_configs['lr0'],
      base_bnd = net_configs['bnd0'],
      bnd_decay_rate = net_configs['bnd_decay'],
      net_configs = net_configs)

  net_configs['bn_decay_fn'] = bn_decay_fn

  return net_run_loop.net_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=MeshnetModel,
      net_data_configs=params['net_data_configs'],
      weight_decay=params['weight_decay'],
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=None,
      dtype=params['dtype'],
      fine_tune=params['fine_tune'],
  )

def parse_flags_update_configs(flags_obj):
  from models.block_configs_fancnn import block_configs
  flags_obj.max_train_steps = int(flags_obj.train_epochs * flags_obj.examples_per_epoch)

  #*****************************************************************************
  net_data_configs = {}
  net_data_configs['net_flag'] = flags_obj.net_flag
  net_data_configs['dataset_name'] = DATASET_NAME
  net_data_configs['dset_shape_idx'] = get_dset_shape_idxs(flags_obj.data_dir)
  net_data_configs['dset_metas'] = DsetMetas
  net_data_configs['block_configs'] = block_configs(flags_obj.net_flag)

  define_model_dir(flags_obj, net_data_configs)
  #*****************************************************************************
  # data_config
  feed_data = flags_obj.feed_data.split('-')
  assert feed_data[0] == 'xyz'

  data_configs = {}
  data_configs['model_dir'] = flags_obj.model_dir
  data_configs['feed_data_eles'] = flags_obj.feed_data
  data_configs['feed_data'] = feed_data
  data_configs['normxyz'] = flags_obj.normxyz

  net_data_configs['data_configs'] = data_configs

  #*****************************************************************************
  # net_configs
  net_configs = {}
  net_configs['bn'] = flags_obj.bn
  net_configs['act'] = flags_obj.act
  net_configs['residual'] = flags_obj.residual
  net_configs['shortcut'] = flags_obj.shortcut
  net_configs['drop_imo_str'] = flags_obj.drop_imo
  net_configs['drop_imo'] = [0.1*int(e) for e in flags_obj.drop_imo]
  net_configs['lr0'] = flags_obj.lr0
  net_configs['lrd_rate'] = flags_obj.lrd_rate
  lrde = flags_obj.lrd_epochs
  net_configs['lrd_epochs'] = lrde
  net_configs['lrd_boundary_epochs'] = range(lrde, flags_obj.train_epochs, lrde)
  net_configs['bnd0'] = flags_obj.bnd0
  net_configs['bnd_decay'] = flags_obj.bnd_decay
  net_configs['batch_size'] = flags_obj.batch_size
  net_configs['num_gpus'] = flags_obj.num_gpus
  net_configs['optimizer'] = flags_obj.optimizer
  net_configs['eval_only'] = flags_obj.eval_only
  net_configs['pred_ply'] = flags_obj.pred_ply
  net_configs['normedge'] = flags_obj.normedge

  net_data_configs['net_configs'] = net_configs


  return net_data_configs

def define_model_dir(flags_obj, net_data_configs):
  if flags_obj.eval_only or flags_obj.pred_ply:
    flags_obj.model_dir = os.path.join(ROOT_DIR, 'results/meshseg', flags_obj.model_dir)
    assert os.path.exists(flags_obj.model_dir+'/checkpoint'),"eval but model_dir does not exist"
    return flags_obj.model_dir

  def model_name():
    if flags_obj.residual == 1:
      modelname = 'R'
      modelname += flags_obj.shortcut
    else:
      modelname = 'P'
    modelname += flags_obj.net_flag
    return modelname

  logname =  model_name()
  if not flags_obj.bn:
    logname += '-Nbn'
  if flags_obj.act != 'Relu':
    logname += '-'+flags_obj.act
  logname += '_bc'+net_data_configs['block_configs']['block_flag']

  logname += '-'+flags_obj.feed_data.replace('nxnynz', 'n')
  if flags_obj.normxyz!='raw':
    logname += '-'+flags_obj.normxyz
  if flags_obj.normedge!='raw':
    logname += '-'+flags_obj.normedge
  if flags_obj.optimizer!='adam':
    logname += '-'+flags_obj.optimizer
  if flags_obj.drop_imo!='000':
    logname += '-Drop'+flags_obj.drop_imo
  logname +='-Bs'+str(flags_obj.batch_size)
  logname +=  '-Lr'+str(int(flags_obj.lr0*1000)) +\
              '_' + str(int(10*flags_obj.lrd_rate)) + \
              '_' + str(flags_obj.lrd_epochs)
  wd = '%.E'%(flags_obj.weight_decay)
  wd = -int(wd.split('E')[1])
  logname += '-wd' + str(wd)

  model_dir = os.path.join(ROOT_DIR, 'results/meshseg', logname)
  flags_obj.model_dir = model_dir

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  add_log_file(model_dir)
  return model_dir

def add_log_file(model_dir):
  import logging
  log = logging.getLogger('tensorflow')
  log.setLevel(logging.DEBUG)

  # create formatter and add it to the handlers
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  # create file handler which logs even debug messages
  fh = logging.FileHandler(os.path.join(model_dir, 'hooks.log'))
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  log.addHandler(fh)


def define_network_flags():
  net_run_loop.define_net_flags()
  flags.adopt_module_key_flags(net_run_loop)
  data_dir = os.path.join(DATA_DIR,'MATTERPORT_TF/mesh_tfrecord')
  flags_core.set_defaults(train_epochs=150,
                          data_dir=data_dir,
                          batch_size=2,
                          num_gpus=2,
                          epochs_between_evals=5,)

  flags.DEFINE_string('net_flag','5A','5A')
  flags.DEFINE_string('optimizer','adam','adam momentum')
  flags.DEFINE_bool('bn', default=True, help ="")
  flags.DEFINE_string('act', default='Relu', help ="Relu, Lrelu")
  flags.DEFINE_float('weight_decay', short_name='wd', default=1e-4, help="wd")
  flags.DEFINE_float('lr0', default=1e-3, help="base lr")
  flags.DEFINE_float('lrd_rate', default=0.7, help="learning rate decay rate")
  flags.DEFINE_float('bnd0', default=0.9, help="base bnd")
  flags.DEFINE_float('bnd_decay', default=0.1, help="")
  flags.DEFINE_integer('lrd_epochs', default=20, help="learning_rate decay epoches")
  flags.DEFINE_string('feed_data','xyz-nxnynz','xyz-nxnynz-color')
  flags.DEFINE_string('normxyz','min0','raw, mean0, min0')
  flags.DEFINE_string('normedge','raw','raw, l0, all')
  flags.DEFINE_bool('residual', short_name='rs', default=False,
      help=flags_core.help_wrap('Is use reidual architecture'))
  flags.DEFINE_string('shortcut','C','C Z')
  flags.DEFINE_string('drop_imo','000','dropout rate for input, middle and out')
  flags.DEFINE_bool(name='pred_ply', default=False, help ="")

  if SMALL_FNUM:
    update_examples_num(True, data_dir)
  flags.DEFINE_integer('examples_per_epoch', default=_NUM_EXAMPLES['train'], help="")

def run_network(flags_obj):
  """Run ResNet ImageNet training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  input_function = (flags_obj.use_synthetic_data and
                    get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or
                    input_fn)
  net_data_configs = parse_flags_update_configs(flags_obj)

  net_run_loop.net_main(flags_obj, network_model_fn, input_function, net_data_configs)

  # debug mode
  net_data_configs['net_configs']['bn_decay_fn'] = None
  #net_run_loop.net_main_check(flags_obj, MeshnetModel, input_function, net_data_configs)



def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_network(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_network_flags()
  absl_app.run(main)
