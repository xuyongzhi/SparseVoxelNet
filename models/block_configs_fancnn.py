import numpy as np

def block_configs(net_flag='default'):

  #*****************************************************************************
  block_configs = {}
  block_sizes = {}
  filters = {}
  kernels = {}
  flayers = {}

  block_configs['edgevnum'] = 10
  block_configs['dense_filters'] = [256, 128]

  if net_flag == '8A' or net_flag=='8B':
    block_sizes['vertex'] = [ [1, 1, 1, 1, 1, 1, 1, 1],  ]
    filters['vertex']     = [ [32, 32, 64, 64, 128, 128, 128, 128],]
    kernels['vertex']     = [ [6 for _ in range(8)],  ]
    if net_flag == '8A':
      filters['global']     = [ [128, 64],]
    if net_flag == '8B':
      filters['global']     = [ [],]

  elif net_flag == '7A' or  net_flag=='7B':
    block_sizes['vertex'] = [ [1, 1, 1, 1, 1, 1, 1],  ]
    filters['vertex']     = [ [32, 32, 64, 64, 128, 128, 128],]
    kernels['vertex']     = [ [6 for _ in range(7)],  ]
    if net_flag == '7A':
      filters['global']     = [ [128, 64],]
    if net_flag == '7B':
      filters['global']     = [ [],]

  elif net_flag == '2A2':
    block_sizes['vertex'] = [ [1, ],   [1,], [1] ]
    kernels['vertex']     = [ [6, ],   [6, ], [6]]
    filters['vertex']     = [ [32,], [64,], [128]]
    filters['global']     = [ [64], [64]]
    filters['backprop']     = [ [125,], [100,]]


  else:
    raise NotImplementedError

  tmp = [i  for fs in filters.values() for f in fs for i in f]
  block_flag = '%d_%d'%(len(tmp), np.mean(tmp))

  block_configs['block_sizes'] = block_sizes
  block_configs['filters'] = filters
  block_configs['kernels'] = kernels
  #*****************************************************************************

  dense_flag = '_%d'%( len(block_configs['dense_filters']))

  block_configs['block_flag'] = str(block_configs['edgevnum']) + '_' + block_flag + dense_flag

  return block_configs
