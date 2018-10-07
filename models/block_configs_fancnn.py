import numpy as np

def block_configs(net_flag='default'):
  block_configs = {}
  block_configs['use_face_global_scale0'] = False
  block_configs['e2fl_pool'] = ['max']
  block_configs['f2v_pool'] = ['max']

  #*****************************************************************************
  block_sizes = {}
  filters = {}
  kernels = {}

  if net_flag == '3A':
    block_sizes['vertex'] = [ [1, 1, 1, 1, 1],  ]
    filters['vertex']     = [ [32, 64, 128, 128, 256],]
    kernels['vertex']     = [ [7, 7, 7, 7, 7],  ]

  else:
    raise NotImplementedError

  tmp = [i  for fs in filters.values() for f in fs for i in f]
  block_flag = '%d_%d'%(len(tmp), np.mean(tmp))

  block_configs['block_sizes'] = block_sizes
  block_configs['filters'] = filters
  block_configs['kernels'] = kernels
  block_configs['block_flag'] = block_flag

  return block_configs
