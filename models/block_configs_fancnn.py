import numpy as np

def block_configs(net_flag='default'):

  #*****************************************************************************
  block_configs = {}
  block_sizes = {}
  filters = {}
  kernels = {}
  flayers = {}

  block_configs['edgevnum'] = 12
  block_configs['global_filters'] = [128, 64]
  block_configs['dense_filters'] = [256, 84]

  if net_flag == '4A' or net_flag=='4B':
    block_sizes['vertex'] = [ [1, 1, 1, 1 ],  ]
    filters['vertex']     = [ [64, 64, 128, 128],]
    kernels['vertex']     = [ [7, 7, 7, 7],  ]

  elif net_flag == '5A':
    block_sizes['vertex'] = [ [1, 1, 1, 1, 1],  ]
    filters['vertex']     = [ [32, 64, 64, 128, 128],]
    kernels['vertex']     = [ [7, 7, 7, 7, 7],  ]
    block_configs['global_filters'] = [64, 64]
    block_configs['dense_filters'] = [84, 84]

  elif net_flag == '7A' or net_flag=='7B':
    block_sizes['vertex'] = [ [1, 1, 1, 1, 1, 1, 1],  ]
    filters['vertex']     = [ [32, 64, 64, 128, 128, 128, 128],]
    kernels['vertex']     = [ [6 for _ in range(7)],  ]
    if net_flag == '7B':
      block_configs['global_filters'] = []

  elif net_flag == '8A' or net_flag== '8B':
    block_sizes['vertex'] = [ [1 for _ in range(8)],  ]
    filters['vertex']     = [ [32, 32, 64, 64, 128, 128, 256, 256],]
    kernels['vertex']     = [ [6 for _ in range(8)],  ]
    flayers['vertex'] = [3, 5, 7]

    if net_flag == '8B':
      block_configs['global_filters'] = []


  else:
    raise NotImplementedError

  tmp = [i  for fs in filters.values() for f in fs for i in f]
  block_flag = '%d_%d'%(len(tmp), np.mean(tmp))

  block_configs['block_sizes'] = block_sizes
  block_configs['filters'] = filters
  block_configs['kernels'] = kernels

  #*****************************************************************************

  dense_flag = '_%d%d'%(len(block_configs['global_filters']), len(block_configs['dense_filters']))

  block_configs['block_flag'] = block_flag + dense_flag

  return block_configs
