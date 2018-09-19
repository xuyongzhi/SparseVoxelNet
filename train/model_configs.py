# xyz June 2018
import numpy as np

def get_block_paras(resnet_size, model_flag, block_style):
  if block_style == 'Regular':
    return get_block_paras_bottle_regu(resnet_size, model_flag)
  elif block_style == 'Bottleneck':
    return get_block_paras_bottle_botl(resnet_size, model_flag)
  elif block_style == 'Inception':
    return get_block_paras_inception(resnet_size, model_flag)
  elif block_style == 'PointNet':
    return get_block_paras_pointnet(resnet_size, model_flag, block_style)

def all_block_paras_bottle_regu_MATTERPORT():
  num_filters0s = {}
  block_sizes = {}
  block_kernels = {}
  block_strides = {}
  block_paddings = {}   # only used when strides == 1
  block_filters = {}
  flatten_filters = {}
  dense_filters = {}


  rs = '1A17'
  num_filters0s[rs] = 32
  block_sizes[rs]    = [[1,  1,    1,  1,   1]  ]
  block_filters[rs]  = [[64, 128, 256, 512, 1024]]
  block_kernels[rs]  = [[],        ]
  block_strides[rs]  = [[],        ]
  block_paddings[rs] = [[],        ]
  flatten_filters[rs]= [[512]]
  dense_filters[rs]  = {'bottle_last_scale':   [512, 256],
                        'final_dense_with_dp': [256, 128]}

  rs = '3A20'
  num_filters0s[rs] = 32
  block_sizes[rs]    = [[2,1, 1],     [1,1],      [1,1,1]]
  block_filters[rs]  = [[32,64,125],  [128,256],  [256,512,1024]]
  block_kernels[rs]  = [[],           [3,3],      [3,3,1]]
  block_strides[rs]  = [[],           [1,1],      [1,1,1]]
  block_paddings[rs] = [[],           ['v','v'],  ['v','v','s']]
  flatten_filters[rs]= [[],     [256, 128],  [256, 256]]
  dense_filters[rs]  = {'bottle_last_scale':   [512, 256],
                        'final_dense_with_dp': [128, 128, 64]}

  return num_filters0s, block_sizes, block_filters, block_kernels, block_strides,\
                      block_paddings, flatten_filters, dense_filters

def get_block_paras_bottle_regu(resnet_size, model_flag):
  if DATASET_NAME == 'MODELNET40':
    all_block_paras = all_block_paras_bottle_regu_MODELNET
  elif DATASET_NAME == 'MATTERPORT':
    all_block_paras = all_block_paras_bottle_regu_MATTERPORT

  num_filters0s, block_sizes, block_filters, block_kernels, block_strides, \
    block_paddings, flatten_filters, dense_filters = all_block_paras()


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
  block_params['flatten_filters'] = flatten_filters[resnet_size]
  block_params['dense_filters'] = dense_filters[resnet_size]
  return block_params

