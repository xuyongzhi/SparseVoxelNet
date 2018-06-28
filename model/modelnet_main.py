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
"""Runs a ResNet model on the ModelNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils_xyz'))

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
import resnet_model
import resnet_run_loop
import os, glob, sys
import numpy as np
from modelnet_configs import get_block_paras, DEFAULTS


BASE_DIR = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from dataset_utils import parse_pl_record

_DATA_PARAS = None

#_DEFAULT_IMAGE_SIZE = 224
#_NUM_CHANNELS = 3
_NUM_CLASSES = 40

_NUM_IMAGES = {
    'train': 9843,
    'validation': 2468,
}

_NUM_TRAIN_FILES = 20
_SHUFFLE_BUFFER = 1000

DATASET_NAME = 'MODELNET40'


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  assert os.path.exists(data_dir), ('not exsit: %s'%(data_dir))
  if is_training:
    return glob.glob(os.path.join(data_dir, 'train_*.tfrecord'))
  else:
    return glob.glob(os.path.join(data_dir, 'test_*.tfrecord'))

def input_fn(is_training, data_dir, batch_size, data_net_configs=None, num_epochs=1):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  assert len(filenames)>0, (data_dir)
  #print('\ngot {} tfrecord files\n'.format(len(filenames)))
  dataset = tf.data.TFRecordDataset(filenames)
                                   # compression_type="",
                                   # buffer_size=_SHUFFLE_BUFFER,
                                   # num_parallel_reads=None)

  #if is_training:
  #  # Shuffle the input files
  #  dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  #dataset = dataset.apply(tf.contrib.data.parallel_interleave(
  #    tf.data.TFRecordDataset, cycle_length=10))

  return resnet_run_loop.process_record_dataset(
      dataset, is_training, batch_size, _SHUFFLE_BUFFER, parse_pl_record, data_net_configs,
      num_epochs
  )


def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(
      _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS, _NUM_CLASSES)


def get_data_shapes_from_tfrecord(data_dir):
  global _DATA_PARAS

  batch_size = 1
  with tf.Graph().as_default():
    dataset = input_fn(False, data_dir, batch_size)
    iterator = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
      features, label = sess.run(iterator)

      for key in features:
        _DATA_PARAS[key] = features[key][0].shape
      points_raw = features['points'][0]
      print('\n\nget shape from tfrecord OK:\n %s\n\n'%(_DATA_PARAS))
      #print('points', features['points'][0,0:5,:])

def check_data():
  from ply_util import create_ply_dset
  IsCreatePly = True
  batch_size = 2
  data_dir = _DATA_PARAS['data_dir']
  model_dir = _DATA_PARAS['model_dir']
  ply_dir = os.path.join(model_dir,'ply')
  aug = _DATA_PARAS['aug_types']
  aug_ply_fn = os.path.join(ply_dir, aug)
  raw_ply_fn = os.path.join(ply_dir, 'raw')
  if not os.path.exists(ply_dir):
    os.makedirs(ply_dir)
  with tf.Graph().as_default():
    dataset = input_fn(True, data_dir, batch_size, _DATA_PARAS)
    iterator1 = dataset.make_initializable_iterator()
    next_item = iterator1.get_next()

    with tf.Session() as sess:
        sess.run(iterator1.initializer)
        features, label = sess.run(next_item)

        if IsCreatePly:
          for i in range(batch_size):
            create_ply_dset(DATASET_NAME, features['points'][i], aug_ply_fn+str(i)+'.ply')
            if aug!='none':
              create_ply_dset(DATASET_NAME, features['raw_points'][i], raw_ply_fn+str(i)+'.ply')


        augs = features['augs']
        #print('points', features['points'][0,0:5,:])
        if 'R' in augs:
          print('angles_yxz', augs['angles_yxz']*180.0/np.pi)
          print('R', augs['R'][0:2])
        if 'S' in augs:
          print('S', features['augs']['S'][0:2])
        if 'shifts' in augs:
          print('shifts', augs['shifts'][0:2])
        if 'jitter' in augs:
          print('jitter', augs['jitter'][0][0:5])
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass


def get_data_meta_from_hdf5(data_dir):
  global _DATA_PARAS
  from block_data_prep_util import GlobalSubBaseBLOCK
  gsbb_load = GlobalSubBaseBLOCK()
  basen = os.path.basename(data_dir)
  dirn = os.path.dirname(os.path.dirname(data_dir))
  bxmh5_dir = os.path.join(dirn, 'Merged_bxmh5', basen)
  bxmh5_fns = glob.glob(os.path.join(bxmh5_dir,'*.bxmh5'))
  assert len(bxmh5_fns) > 0, (bxmh5_dir)
  bxmh5_fn = bxmh5_fns[0]
  gsbb_load.load_para_from_bxmh5(bxmh5_fn)

  _DATA_PARAS['dataset_name'] = DATASET_NAME
  _DATA_PARAS['sg_bm_extract_idx'] = gsbb_load.sg_bidxmaps_extract_idx
  _DATA_PARAS['flatten_bm_extract_idx'] = gsbb_load.flatten_bidxmaps_extract_idx
  _DATA_PARAS['global_step'] = gsbb_load.global_step
  _DATA_PARAS['global_stride'] = gsbb_load.global_stride
  _DATA_PARAS['sub_block_stride_candis'] = gsbb_load.sub_block_stride_candis
  _DATA_PARAS['sub_block_step_candis'] = gsbb_load.sub_block_step_candis
  _DATA_PARAS['flatbxmap_max_nearest_num'] = gsbb_load.flatbxmap_max_nearest_num
  _DATA_PARAS['data_idxs'] = gsbb_load.data_idxs


###############################################################################
# Running the model
###############################################################################
class ModelnetModel(resnet_model.Model):
  """Model class with appropriate defaults for Modelnet data."""

  def __init__(self, model_flag, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE, data_net_configs={}):
    """These are the parameters that work for Modelnet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
      final_size = 512
    else:
      bottleneck = True
      final_size = 2048

    super(ModelnetModel, self).__init__(
        model_flag = model_flag,
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=data_net_configs['num_filters0'],
        block_sizes=data_net_configs['block_sizes'],
        block_kernels=data_net_configs['block_kernels'],
        block_strides=data_net_configs['block_strides'],
        block_paddings=data_net_configs['block_paddings'],
        final_size=final_size,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype,
        data_net_configs=data_net_configs
    )

def modelnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  decay_rate = params['data_net_configs']['lr_decay_rate']
  boundary_epochs = params['data_net_configs']['lr_boundary_epochs']
  decay_rates = [1] + [pow(decay_rate,i+1) for i in range(len(boundary_epochs))]
  if params['data_net_configs']['lr_warmup']:
    boundary_epochs = [1]+boundary_epochs
    decay_rates = [0.01] + decay_rates

  learning_rate_fn, bndecay_fn, lr_vals, bndecay_vals = \
      resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=256,
      num_images=_NUM_IMAGES['train'], boundary_epochs=boundary_epochs,
      decay_rates=decay_rates,
      initial_learning_rate=params['data_net_configs']['learning_rate0'],
      initial_bndecay=params['data_net_configs']['batch_norm_decay0'])
  params['data_net_configs']['bndecay_fn'] = bndecay_fn

  params['data_net_configs']['lr_vals'] = lr_vals
  params['data_net_configs']['bndecay_vals'] = bndecay_vals

  return resnet_run_loop.resnet_model_fn(
      model_flag=params['model_flag'],
      features=features,
      labels=labels,
      mode=mode,
      model_class=ModelnetModel,
      resnet_size=params['resnet_size'],
      weight_decay=params['data_net_configs']['weight_decay'],
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=None,
      dtype=params['dtype'],
      data_net_configs=params['data_net_configs']
  )



def get_dropout_rates(drop_imo):
  drop_imo = [0.1*int(e) for e in  drop_imo.split('_')]
  assert len(drop_imo) == 3
  drop_imo_ = {}
  drop_imo_['input'] = drop_imo[0]
  drop_imo_['middle'] = drop_imo[1]
  drop_imo_['output'] = drop_imo[2]
  return drop_imo_

def define_net_configs(flags_obj):
  global _DATA_PARAS
  if DEFAULTS['num_gpus']==1:
    assert flags_obj.num_gpus == 1
  _DATA_PARAS['residual'] = flags_obj.residual == 1
  _DATA_PARAS['use_bias'] = flags_obj.use_bias == 1
  _DATA_PARAS['optimizer'] = flags_obj.optimizer
  _DATA_PARAS['learning_rate0'] = flags_obj.learning_rate0
  _DATA_PARAS['lr_decay_rate'] = flags_obj.lr_decay_rate
  _DATA_PARAS['lr_decay_epochs'] = lr_decay_epochs=flags_obj.lr_decay_epochs
  _DATA_PARAS['lr_boundary_epochs'] = range(lr_decay_epochs, \
                                      flags_obj.train_epochs, lr_decay_epochs)
  _DATA_PARAS['lr_warmup'] = flags_obj.lr_warmup == 1
  _DATA_PARAS['batch_norm_decay0'] = flags_obj.batch_norm_decay0
  _DATA_PARAS['weight_decay'] = flags_obj.weight_decay
  _DATA_PARAS['resnet_size'] = flags_obj.resnet_size
  _DATA_PARAS['batch_size'] = flags_obj.batch_size
  _DATA_PARAS['num_filters0'] = flags_obj.num_filters0
  _DATA_PARAS['model_flag'] = flags_obj.model_flag
  _get_block_paras()

  feed_data_eles = flags_obj.feed_data
  feed_data = flags_obj.feed_data.split('-')
  assert feed_data[0][0:3] == 'xyz'
  xyz_eles = feed_data[0][3:]
  feed_data[0] = 'xyz'
  assert len(xyz_eles)<=3
  xyz_elements = []
  if 's' in xyz_eles: xyz_elements.append('sub_mid')
  if 'g' in xyz_eles: xyz_elements.append('global_mid')
  if 'r' in xyz_eles: xyz_elements.append('raw')
  assert len(xyz_elements) > 0
  _DATA_PARAS['feed_data'] = feed_data
  _DATA_PARAS['feed_data_eles'] = feed_data_eles
  _DATA_PARAS['xyz_elements'] = xyz_elements
  _DATA_PARAS['aug_types'] = flags_obj.aug_types

  _DATA_PARAS['drop_imo_str'] = flags_obj.drop_imo
  _DATA_PARAS['drop_imo'] = get_dropout_rates(flags_obj.drop_imo)

  model_dir = define_model_dir()
  _DATA_PARAS['model_dir'] = model_dir
  flags_obj.model_dir = model_dir

  flags_obj.steps_per_epoch = _NUM_IMAGES['train'] / flags_obj.batch_size
  flags_obj.max_train_steps = int(flags_obj.train_epochs * flags_obj.steps_per_epoch)+2


def _get_block_paras():
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  global _DATA_PARAS
  resnet_size = _DATA_PARAS['resnet_size']
  block_sizes, block_kernels, block_strides, block_paddings = \
              get_block_paras(_DATA_PARAS['resnet_size'],
                                               _DATA_PARAS['model_flag'])
  _DATA_PARAS['block_sizes'] = block_sizes[resnet_size]
  _DATA_PARAS['block_kernels'] = block_kernels[resnet_size]
  _DATA_PARAS['block_strides'] = block_strides[resnet_size]
  _DATA_PARAS['block_paddings'] = block_paddings[resnet_size]

def ls_str(ls_in_ls):
  ls_str = [str(e) for ls in ls_in_ls for e in ls]
  ls_str = ''.join(ls_str)
  return ls_str

def define_model_dir():
  def get_model_configs_detailed():
    block_sizes_str = ls_str(_DATA_PARAS['block_sizes'])
    block_kernels_str = ls_str(_DATA_PARAS['block_kernels'])
    block_paddings_str = ls_str(_DATA_PARAS['block_paddings'])
    model_str = '-b%s-k%s-p%s'%( block_sizes_str,
                                block_kernels_str, block_paddings_str)
    return model_str
  def get_model_configs_fused():
    block_size_sum = sum( [e  for bs in _DATA_PARAS['block_sizes'] for e in bs] )
    block_kernels_sum =  sum( [e  for bs in _DATA_PARAS['block_kernels'] for e in bs] )
    block_paddings_sum =  sum( [e=='s'  for bs in _DATA_PARAS['block_paddings'] for e in bs] )
    model_str = '-b%dk%dp%d'%(  block_size_sum,
                                  block_kernels_sum,
                                  block_paddings_sum )
    return model_str

  if flags.FLAGS.residual == 1:
    logname = 'Rs'
  else:
    logname = 'Pl'
  if flags.FLAGS.use_bias == 1:
    logname += 'Ub'
  else:
    logname += 'Nb'
  model_str = get_model_configs_fused()
  logname += str(flags.FLAGS.resnet_size) + '-' + flags.FLAGS.model_flag
  logname += model_str
  logname += '-Fn0_'+str(flags.FLAGS.num_filters0)

  logname += '-'+flags.FLAGS.feed_data + '-Aug_' + flags.FLAGS.aug_types
  logname += '-Drop'+flags.FLAGS.drop_imo
  logname +='-Bs'+str(flags.FLAGS.batch_size)
  logname += '-'+flags.FLAGS.optimizer
  logname += '-Lr'+str(int(flags.FLAGS.learning_rate0*1000)) +\
              '_' + str(int(flags.FLAGS.lr_decay_rate*10)) +\
              '_' + str(flags.FLAGS.lr_decay_epochs)
  if flags.FLAGS.lr_warmup == 1:
    logname += 'w'
  logname += '-Bnd'+str(int(flags.FLAGS.batch_norm_decay0*100))

  model_dir = os.path.join(ROOT_DIR, 'train_res/object_detection_result', logname)
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

def define_modelnet_flags():
  global _DATA_PARAS
  if DEFAULTS['num_gpus']==1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(DEFAULTS['gpu_id'])

  _DATA_PARAS = {}
  flags.DEFINE_integer('residual', DEFAULTS['residual'], '')
  flags.DEFINE_integer('use_bias', DEFAULTS['use_bias'], '')
  flags.DEFINE_string('optimizer', DEFAULTS['optimizer'], 'adam, momentum')
  flags.DEFINE_float('learning_rate0', DEFAULTS['learning_rate0'],'')
  flags.DEFINE_float('lr_decay_rate', DEFAULTS['lr_decay_rate'],'')
  flags.DEFINE_integer('lr_decay_epochs', DEFAULTS['lr_decay_epochs'],'')
  flags.DEFINE_float('lr_warmup', DEFAULTS['lr_warmup'],'')
  flags.DEFINE_float('batch_norm_decay0', DEFAULTS['batch_norm_decay0'],'')
  flags.DEFINE_float('weight_decay', DEFAULTS['weight_decay'],'')
  flags.DEFINE_string('model_flag', DEFAULTS['model_flag'], '')
  flags.DEFINE_integer('resnet_size',DEFAULTS['resnet_size'],'resnet_size')
  flags.DEFINE_integer('num_filters0',DEFAULTS['num_filters0'],'')
  flags.DEFINE_string('feed_data',DEFAULTS['feed_data'],'xyzrsg-nxnynz-color')
  flags.DEFINE_string('aug_types',DEFAULTS['aug_types'],'rsfj-360_0_0')
  flags.DEFINE_string('drop_imo',DEFAULTS['drop_imo'],'0_0_5')

  resnet_run_loop.define_resnet_flags(
      resnet_size_choices=['18', '34', '50', '101', '152', '200'])
  flags.adopt_module_key_flags(resnet_run_loop)
  data_dir = os.path.join(DATA_DIR, DEFAULTS['data_path'])
  _DATA_PARAS['data_dir'] = data_dir
  flags_core.set_defaults(train_epochs=DEFAULTS['train_epochs'],
                          data_dir=data_dir,
                          batch_size=DEFAULTS['batch_size'],
                          num_gpus=DEFAULTS['num_gpus'],
                          data_format=DEFAULTS['data_format'] )
  flags.DEFINE_integer('gpu_id',DEFAULTS['gpu_id'],'')
  flags.DEFINE_float('steps_per_epoch',
                     _NUM_IMAGES['train']/DEFAULTS['batch_size'],'')
  get_data_shapes_from_tfrecord(data_dir)
  get_data_meta_from_hdf5(data_dir)


def run_modelnet(flags_obj):
  """Run ResNet ModelNet training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  define_net_configs(flags_obj)
  input_function = (flags_obj.use_synthetic_data and get_synth_input_fn()
                    or input_fn)
  #check_data()
  resnet_run_loop.resnet_main(
      flags_obj, modelnet_model_fn, input_function, DATASET_NAME, _DATA_PARAS)

def main(_):
  run_modelnet(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_modelnet_flags()
  absl_app.run(main)
