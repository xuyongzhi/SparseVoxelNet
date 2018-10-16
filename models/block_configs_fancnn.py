import numpy as np

def block_configs(net_flag0='7B'):

  #*****************************************************************************
  block_configs = {}
  block_sizes = {}
  filters = {}
  kernels = {}
  flayers = {}

  tmp = net_flag0.split('_')
  net_flag = tmp[0]
  if len(tmp)>1:
    edgevnum = int(tmp[1])
  else:
    edgevnum = 10
  if len(tmp)>2:
    fan_kernel = int(tmp[2])
  else:
    fan_kernel = 6

  block_configs['edgevnum'] = edgevnum
  block_configs['dense_filters'] = [256, 128]


  if net_flag == '7A' or  net_flag=='7G':
    block_sizes['vertex'] = [ [1, 1, 1, 1, 1, 1, 1],  ]
    filters['vertex']     = [ [32, 32, 64, 64, 128, 128, 256],]
    kernels['vertex']     = [ [fan_kernel for _ in range(7)],  ]
    if net_flag == '7G':
      filters['global']     = [ [64],[128],]
    if net_flag == '7A':
      filters['global']     = [ [],]

  elif net_flag == '53A' or  net_flag=='53G':
    block_sizes['vertex'] = [ [1,1,1,1,1], [1,1] ]
    filters['vertex']     = [ [32, 32, 64, 64, 128], [128, 128]]
    kernels['vertex']     = [ [fan_kernel for _ in range(5)], [fan_kernel for _ in range(2)]]
    if net_flag == '53G':
      filters['global']     = [ [64], [128]]
    if net_flag == '53A':
      filters['global']     = [ [],[]]
    filters['backprop']   = [ [128,128]]


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
