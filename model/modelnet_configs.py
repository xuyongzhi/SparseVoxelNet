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
#DEFAULTS['data_path'] = 'MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'
DEFAULTS['data_path'] = 'MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'
#DEFAULTS['data_path'] = 'MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2d2-neg_fmn1444_mvp1-3200_1024_48_1-18_24_56_56-0d1_0d2_0d6-0d0_0d1_0d4-pd3-3M1'


DEFAULTS['only_eval'] = 0
DEFAULTS['eval_views'] = 5

DEFAULTS['residual'] = 1
DEFAULTS['shortcut'] = 'MC' #C, MC, AC, MZ, AZ
DEFAULTS['use_bias'] = 1
DEFAULTS['loss_lw_gama'] = 2
DEFAULTS['block_style'] = 'Bottleneck' # Regular, Bottleneck, Inception
DEFAULTS['block_style'] = 'Inception' # Regular, Bottleneck, Inception
DEFAULTS['block_style'] = 'Regular'
DEFAULTS['optimizer'] = 'momentum'
DEFAULTS['learning_rate0'] = 0.001
DEFAULTS['lr_decay_rate'] = 0.7
DEFAULTS['lr_decay_epochs'] = 15
DEFAULTS['lr_warmup'] = 1
DEFAULTS['batch_norm_decay0'] = 0.7

DEFAULTS['model_flag'] = 'V'
DEFAULTS['resnet_size'] = 24
DEFAULTS['feed_data'] = 'xyzs-nxnynz'
DEFAULTS['aug_types'] = 'N' # 'rpsfj-360_0_0'
DEFAULTS['drop_imo'] = '0_0_5'
DEFAULTS['batch_size'] = 64
DEFAULTS['num_gpus'] = 2
DEFAULTS['gpu_id'] = 1
DEFAULTS['train_epochs'] = 101
DEFAULTS['data_format'] = 'channels_last'

DEFAULTS['weight_decay'] = 0.0  # res official is 1e-4, charles is 0.0

def get_block_paras(resnet_size, model_flag, block_style):
  if block_style == 'Bottleneck' or block_style == 'Regular':
    return get_block_paras_bottle_regu(resnet_size, model_flag, block_style)
  elif block_style == 'Inception':
    return get_block_paras_inception(resnet_size, model_flag)

def icp_block(flag):
  def icp_by_mapsize(map_size, filters_in, filters_out):
    if map_size == 1:
      ICP = [[['conv',filters_in,1,1,'s'], ['conv',filters_out,1,1,'s']]]
      return ICP

    #---------- Keep feature map size  -------------
    def compress_rate(base_filters):
      return max(base_filters//4, 32)
    f0 = compress_rate(filters_in)
    f1 = compress_rate(filters_out)

    if flag == 'A':
      if map_size >= 6: kernel = 3
      elif map_size >=3: kernel = 2
      else: kernel = 1
      ICP = [
            [['conv',f1,1,1,'s']],
            [['conv',f0,1,1,'s'], ['conv',f1*2,kernel,1,'s']],
            [['max',kernel,1,'s'], ['conv',f1,1,1,'s']] ]

    #---------- Reduce feature map size  -------------
    elif flag == 'a':
      if map_size >= 3: kernel = 3
      elif map_size == 2: kernel = 2
      elif map_size == 1: kernel = 1
      ICP = [
            [['conv',f0,1,1,'s'], ['conv',f1*2,kernel,1,'v']],
            [['max',kernel,1,'v'], ['conv',f1*1,1,1,'s']] ]

    elif flag == 'B':
      if map_size >= 6: kernel = 3
      elif map_size >=3: kernel = 2
      else: kernel = 1
      ICP = [
            [['conv',f1,1,1,'s']],
            [['conv',f0,1,1,'s'], ['conv',f1*2,kernel,1,'s']],
            [['max',kernel,1,'s'], ['conv',f1,1,1,'s']] ]

    #---------- Reduce feature map size  -------------
    elif flag == 'b':
      if map_size >= 4: kernel = 3
      elif map_size >= 2: kernel = 2
      elif map_size == 1: kernel = 1
      ICP = [
            [['conv',f0,1,1,'s'], ['conv',f1*2,kernel,1,'v']],
            [['max',kernel,1,'v'], ['conv',f1*1,1,1,'s']] ]

    return ICP
  return icp_by_mapsize


def get_block_paras_inception(resnet_size, model_flag):
  model_flag = model_flag[0:2]
  num_filters0s = {}
  block_sizes = {}
  block_flag = {}
  block_filters = {}

  rs = 27
  if model_flag == 'Va':
    num_filters0s[rs] = 32
    block_sizes[rs]   = [[2,1], [1,1,2], [2,2,1]]
    block_filters[rs] = [[32,64], [128,256,384], [384,512,1024]]
    block_flag[rs]    = [[],  ['A','a','A'],['a','A','a']]

  elif model_flag == 'Vb':
    num_filters0s[rs] = 32
    block_sizes[rs]   = [[2,1], [1,2,1], [2,2,1]]
    block_filters[rs] = [[32,64], [128,256,384], [384,512,1024]]
    block_flag[rs]    = [[],  ['B','b','B'],['a','A','a']]
  else:
    raise NotImplementedError

  block_params = {}
  block_params['num_filters0'] = num_filters0s[resnet_size]
  block_params['block_sizes'] = block_sizes[resnet_size]
  block_params['filters'] = block_filters[resnet_size]
  block_params['icp_flags'] = block_flag[resnet_size]
  block_params['icp_block_ops'] = [ [icp_block(flag) for flag in cascade] for cascade in block_flag[resnet_size] ]
  return block_params



def get_block_paras_bottle_regu(resnet_size, model_flag, block_style):
  num_filters0s = {}
  block_sizes = {}
  block_kernels = {}
  block_strides = {}
  block_paddings = {}   # only used when strides == 1
  block_filters = {}

  data_flag = DEFAULTS['data_path'].split('-')[-1]
  if data_flag == '2M2pp':
    if block_style == 'Bottleneck':
      #------------------------------ Bottleneck ---------------------------------
      rs = 36
      num_filters0s[rs] = 32
      block_sizes[rs]    = [[2,1], [1,1,2], [2,2,1]]
      block_filters[rs] = [[32,64], [256,256,512], [512,1024,2048]]
      block_kernels[rs]  = [[], [1,3,1], [3,2,1]]
      block_strides[rs]  = [[], [1,1,1], [1,1,1]]
      block_paddings[rs] = [[], ['s','v','s'], ['v','v','v']]

      rs = 37
      num_filters0s[rs] = 32
      block_sizes[rs]    = [[1,1,1], [1,1,1], [1,1,1]]
      block_filters[rs] = [[32,64,128], [256,256,512], [512,512,1024]]
      block_kernels[rs]  = [[], [1,3,1], [3,2,1]]
      block_strides[rs]  = [[], [1,1,1], [1,1,1]]
      block_paddings[rs] = [[], ['s','v','s'], ['v','v','v']]

    elif block_style == 'Regular':
      #------------------------------- Regular -----------------------------------
      rs = 10
      num_filters0s[rs] = 32
      block_sizes[rs]    = [[1], [1], [1,1]]
      block_filters[rs] = [[32], [64], [128,256]]
      block_kernels[rs]  = [[], [3], [3,3]]
      block_strides[rs]  = [[], [1], [1,1]]
      block_paddings[rs] = [[], ['v'], ['v','v']]

      rs = 15
      num_filters0s[rs] = 16
      block_sizes[rs]    = [[1,1], [1,1], [1,1,1]]
      block_filters[rs] = [[16,32], [64,64], [64,128,256]]
      block_kernels[rs]  = [[], [3,1], [3,3,3]]
      block_strides[rs]  = [[], [1,1], [1,1,1]]
      block_paddings[rs] = [[], ['v','s'], ['v','v','v']]

      rs = 16
      num_filters0s[rs] = 32
      block_sizes[rs]    = [[1,1], [1,1], [1,1,1]]
      block_filters[rs] = [[32,64], [128,128], [128,256,512]]
      block_kernels[rs]  = [[], [3,1], [3,3,3]]
      block_strides[rs]  = [[], [1,1], [1,1,1]]
      block_paddings[rs] = [[], ['v','s'], ['v','v','v']]

      rs = 24
      num_filters0s[rs] = 32
      block_sizes[rs]    = [[2,1], [1,2], [2,2,1]]
      block_filters[rs] = [[32,64], [128,128], [128,256,512]]
      block_kernels[rs]  = [[], [3,1], [3,3,3]]
      block_strides[rs]  = [[], [1,1], [1,1,1]]
      block_paddings[rs] = [[], ['v','s'], ['v','v','v']]

      rs = 36
      num_filters0s[rs] = 32
      block_sizes[rs]    = [[2,1], [1,2], [2,2,1]]
      block_filters[rs] = [[32,64], [128,256], [256,512,1024]]
      block_kernels[rs]  = [[], [3,1], [3,2,1]]
      block_strides[rs]  = [[], [1,1], [1,1,1]]
      block_paddings[rs] = [[], ['v','s'], ['v','v','v']]

      rs = 37
      num_filters0s[rs] = 32
      block_sizes[rs]    = [[2,1,1], [1,2], [1,1,1,1,1]]
      block_filters[rs] = [[32,64,128], [256,256], [256,384,512,768,1024]]
      block_kernels[rs]  = [[], [3,1], [3,2,2,2,2]]
      block_strides[rs]  = [[], [1,1], [1,1,1,1,1]]
      block_paddings[rs] = [[], ['v','s'], ['v','v','v','v','v']]

      rs = 38
      num_filters0s[rs] = 32
      block_sizes[rs]    = [[2,1,1], [1,2], [1,1,1,1,1]]
      block_filters[rs] = [[32,64,128], [256,256], [256,384,512,768,1024]]
      block_kernels[rs]  = [[], [3,1], [3,3,3,2,2]]
      block_strides[rs]  = [[], [1,1], [1,1,1,1,1]]
      block_paddings[rs] = [[], ['v','s'], ['v','v','v','v','v']]


  if data_flag == '3M1':
    if block_style == 'Regular':
      rs = 24
      num_filters0s[rs] = 32
      block_sizes[rs]    = [[2,1], [1,2], [2,2,1],[1]]
      block_filters[rs]  = [[32,64], [128,128], [128,256,512],[512]]
      block_kernels[rs]  = [[], [3,1], [3,3,3],[1]]
      block_strides[rs]  = [[], [1,1], [1,1,1],[1]]
      block_paddings[rs] = [[], ['v','s'], ['v','v','v'],['v']]

  if 'V' not in model_flag:
    for i in range(len(block_sizes[resnet_size])):
      if i==0: continue
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
  block_params['num_filters0'] = num_filters0s[resnet_size]
  block_params['kernels'] = block_kernels[resnet_size]
  block_params['strides'] = block_strides[resnet_size]
  block_params['filters'] = block_filters[resnet_size]
  block_params['padding_s1'] = block_paddings[resnet_size]
  block_params['block_sizes'] = block_sizes[resnet_size]
  return block_params


