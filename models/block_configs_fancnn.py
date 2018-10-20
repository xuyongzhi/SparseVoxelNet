import numpy as np

def block_configs(net_flag0='7B'):
  '''
  7G_9_5_2
  edgevnum=9,  kernel=5,  stride=2
  '''

  #*****************************************************************************
  block_configs = {}
  block_sizes = {}
  filters = {}
  kernels = {}
  flayers = {}
  strides = {}

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
  if len(tmp)>3:
    fan_stride = int(tmp[3])
  else:
    fan_stride = 1

  block_configs['edgevnum'] = edgevnum
  block_configs['dense_filters'] = [256, 128]


  if net_flag == '7A' or  net_flag=='7G':
    block_sizes['vertex'] = [ [1, 1, 1, 1, 1, 1, 1],  ]
    filters['vertex']     = [ [32, 32, 64, 64, 128, 128, 128],]
    kernels['vertex']     = [ [fan_kernel]*7,  ]
    strides['vertex']     = [ [fan_stride]*7 ]
    if net_flag == '7G':
      filters['global']     = [ [64],[128],]
    if net_flag == '7A':
      filters['global']     = [ [],]

  elif net_flag == '10A' or  net_flag=='10G':
    block_sizes['vertex'] = [ [1]*10,  ]
    filters['vertex']     = [ [32, 32, 32, 32, 64, 64, 64, 64, 128, 128],]
    kernels['vertex']     = [ [fan_kernel]*10,  ]
    strides['vertex']     = [ [fan_stride]*10 ]
    if 'G' in net_flag:
      filters['global']     = [ [64],[128],]
    else:
      filters['global']     = [ [],]

  elif net_flag == '2G':
    block_sizes['vertex'] = [ [1, 1],  ]
    filters['vertex']     = [ [32, 32],]
    kernels['vertex']     = [ [fan_kernel for _ in range(2)],  ]
    filters['global']     = [ [16],[32],]

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
  block_configs['strides'] = strides
  #*****************************************************************************

  dense_flag = '_%d'%( len(block_configs['dense_filters']))

  block_configs['block_flag'] = str(block_configs['edgevnum']) + '_' + block_flag + dense_flag

  return block_configs
