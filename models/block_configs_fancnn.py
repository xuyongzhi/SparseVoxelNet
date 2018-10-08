import numpy as np

def block_configs(net_flag='default'):

  #*****************************************************************************
  block_configs = {}
  block_sizes = {}
  filters = {}
  kernels = {}

  if net_flag == '4A' or net_flag=='4B':
    block_sizes['vertex'] = [ [1, 1, 1, 1 ],  ]
    filters['vertex']     = [ [32, 64, 128, 128],]
    kernels['vertex']     = [ [7, 7, 7, 7],  ]

  elif net_flag == '5A':
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

  #*****************************************************************************
  block_configs['global_filters'] = [128, 128, 64, 64]
  block_configs['dense_filters'] = [128, 64, 64]

  return block_configs
