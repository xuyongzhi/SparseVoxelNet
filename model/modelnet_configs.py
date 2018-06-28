# xyz June 2018
import numpy as np

'''
        Hot parameters
  resnet size
  weight decay
  batch size
  learning rate decay
  learning rate
  aug types
'''

DEFAULTS = {}
DEFAULTS['data_path'] = 'MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'
DEFAULTS['data_path'] = 'MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'

DEFAULTS['residual'] = 1
DEFAULTS['use_bias'] = 1
DEFAULTS['optimizer'] = 'adam'
DEFAULTS['learning_rate0'] = 0.001
DEFAULTS['lr_decay_rate'] = 0.7
DEFAULTS['lr_decay_epochs'] = 15
DEFAULTS['lr_warmup'] = 1
DEFAULTS['batch_norm_decay0'] = 0.7

DEFAULTS['model_flag'] = 'm'
DEFAULTS['resnet_size'] = 34
DEFAULTS['num_filters0'] = 64
DEFAULTS['feed_data'] = 'xyzs'
DEFAULTS['aug_types'] = 'rpsfj-360_0_0'
DEFAULTS['drop_imo'] = '0_0_5'
DEFAULTS['batch_size'] = 16
DEFAULTS['num_gpus'] = 2
DEFAULTS['gpu_id'] = 1
DEFAULTS['train_epochs'] = 61
DEFAULTS['data_format'] = 'channels_last'

DEFAULTS['weight_decay'] = 0.0  # res official is 1e-4, charles is 0.0

def get_block_paras(resnet_size, model_flag):
  block_sizes = {}
  block_kernels = {}
  block_strides = {}
  block_paddings = {}   # only used when strides == 1

  rs = 18
  block_sizes[rs]    = [[2], [2,1], [2,1]]
  block_kernels[rs]  = [[1], [2,3], [3,3]]
  block_strides[rs]  = [[1], [1,1], [1,1]]
  block_paddings[rs] = [['s'], ['s','v'], ['v','v']]

  rs = 28
  block_sizes[rs]    = [[3,1], [3,1], [2,2,1]]
  block_kernels[rs]  = [[1,1], [2,3], [3,3,1]]
  block_strides[rs]  = [[1,1], [1,1], [1,1,1]]
  block_paddings[rs] = [['s','s'], ['s','v'], ['v','v','v']]

  rs = 34
  block_sizes[rs]    = [[4], [3,1], [2,2,2]]
  block_kernels[rs]  = [[1], [2,3], [3,3,3]]
  block_strides[rs]  = [[1], [1,1], [1,1,1]]
  block_paddings[rs] = [['s'], ['s','v'], ['v','v','v']]

  rs = 50
  block_sizes[rs]    = [[5], [4,1], [3,3,3]]
  block_kernels[rs]  = [[1], [2,3], [3,3,3]]
  block_strides[rs]  = [[1], [1,1], [1,1,1]]
  block_paddings[rs] = [['s'], ['s','v'], ['v','v','v']]

  if 'V' not in model_flag:
    for i in range(len(block_sizes[resnet_size])):
      for j in range(len(block_sizes[resnet_size][i])):
        block_kernels[resnet_size][i][j] = 1
        block_strides[resnet_size][i][j] = 1
        block_paddings[resnet_size][i][j] = 'v'

  if resnet_size not in block_sizes:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, resnet_size.keys()))
    raise ValueError(err)

  # check settings
  for k in block_kernels:
    # cascade_id 0 is pointnet
    assert (np.array(block_kernels[k][0])==1).all()
    assert (np.array(block_strides[k][0])==1).all()

  return block_sizes, block_kernels, block_strides, block_paddings
