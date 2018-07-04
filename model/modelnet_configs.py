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
#DEFAULTS['data_path'] = 'MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'

DEFAULTS['residual'] = 1
DEFAULTS['shortcut'] = 'MC' #C, MC, AC, MZ, AZ
DEFAULTS['use_bias'] = 1
DEFAULTS['block_style'] = 'Bottleneck' # Regular, Bottleneck, Inception
DEFAULTS['block_style'] = 'Inception' # Regular, Bottleneck, Inception
DEFAULTS['optimizer'] = 'momentum'
DEFAULTS['learning_rate0'] = 0.001
DEFAULTS['lr_decay_rate'] = 0.7
DEFAULTS['lr_decay_epochs'] = 15
DEFAULTS['lr_warmup'] = 1
DEFAULTS['batch_norm_decay0'] = 0.7

DEFAULTS['model_flag'] = 'V'
DEFAULTS['resnet_size'] = 36
DEFAULTS['num_filters0'] = 32
DEFAULTS['feed_data'] = 'xyzs-nxnynz'
DEFAULTS['aug_types'] = 'N' # 'rpsfj-360_0_0'
DEFAULTS['drop_imo'] = '0_0_5'
DEFAULTS['batch_size'] = 32
DEFAULTS['num_gpus'] = 2
DEFAULTS['gpu_id'] = 1
DEFAULTS['train_epochs'] = 61
DEFAULTS['data_format'] = 'channels_last'

DEFAULTS['weight_decay'] = 0.0  # res official is 1e-4, charles is 0.0

def get_block_paras(resnet_size, model_flag, block_style):
  if block_style == 'Bottleneck' or block_style == 'Regular':
    return get_block_paras_bottleneck(resnet_size, model_flag)
  elif block_style == 'Inception':
    return get_block_paras_inception(resnet_size, model_flag)

def icp_block(flag):
  def icp_by_mapsize(map_size, filters_in, filters_out):
    #---------- Keep feature map size  -------------
    if flag == 'A':
      if map_size >= 6: kernel = 3
      elif map_size >=3: kernel = 2
      elif map_size == 1: kernel = 1
      f0 = (filters_in // 4)
      f1 = (filters_out // 4)
      ICP = [
            [['conv',f0,1,1,'s']],
            [['conv',f0,1,1,'s'], ['conv',f1*2,kernel,1,'s']],
            [['max',kernel,1,'s'], ['conv',f1,1,1,'s']] ]

    #---------- Reduce feature map size  -------------
    elif flag == 'a':
      if map_size >= 3: kernel = 3
      elif map_size == 2: kernel = 2
      elif map_size == 1: kernel = 1
      f0 = (filters_in // 4)
      f1 = (filters_out // 4)
      ICP = [
            [['conv',f0,1,1,'s'], ['conv',f1*2,kernel,1,'v']],
            [['max',kernel,1,'v'], ['conv',f1*1,1,1,'s']] ]
    return ICP
  return icp_by_mapsize


def get_block_paras_inception(resnet_size, model_flag):
  block_sizes = {}
  block_flag = {}
  rs = 36
  block_sizes[rs] = [[2,1], [1,2,1], [2,3,1]]
  block_flag[rs]  = [[],  ['a','A','A'],['a','A','a']]

  block_params = {}
  block_params['block_sizes'] = block_sizes[rs]
  block_params['icp_flags'] = block_flag[rs]
  block_params['icp_block_ops'] = [ [icp_block(flag) for flag in cascade] for cascade in block_flag[rs] ]
  return block_params

def get_block_paras_bottleneck(resnet_size, model_flag):
  block_sizes = {}
  block_kernels = {}
  block_strides = {}
  block_paddings = {}   # only used when strides == 1

  rs = 36
  block_sizes[rs]    = [[2,1], [1,2], [2,2,1]]
  block_kernels[rs]  = [[], [3,1], [3,2,1]]
  block_strides[rs]  = [[], [1,1], [1,1,1]]
  block_paddings[rs] = [[], ['v','s'], ['v','v','v']]


  if 'V' not in model_flag:
    for i in range(len(block_sizes[resnet_size])):
      for j in range(len(block_sizes[resnet_size][i])):
        block_kernels[resnet_size][i][j] = 1
        block_strides[resnet_size][i][j] = 1
        block_paddings[resnet_size][i][j] = 'v'

  if resnet_size not in block_sizes:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, block_sizes.keys()))
    raise ValueError(err)

  # check settings
  for k in block_kernels:
    # cascade_id 0 is pointnet
    assert (np.array(block_kernels[k][0])==1).all()
    assert (np.array(block_strides[k][0])==1).all()

  block_params = {}
  block_params['kernels'] = block_kernels[rs]
  block_params['strides'] = block_strides[rs]
  block_params['padding_s1'] = block_paddings[rs]
  block_params['block_sizes'] = block_sizes[rs]
  return block_params


