import numpy as np

def block_configs(net_flag):
  '''
  '''
  net_flag = '3A'
  #*****************************************************************************
  block_configs = {}

  block_configs['dense_filters'] = [256, 128]

  if net_flag == '3A':
    filters = [ [32,64,128], [ 128] ]
    block_sizes = [[1]*3, [1]*1]

  else:
    raise NotImplementedError

  block_flag = ''

  block_configs['block_sizes'] = block_sizes
  block_configs['filters'] = filters
  #*****************************************************************************

  block_configs['block_flag'] = ''

  return block_configs
