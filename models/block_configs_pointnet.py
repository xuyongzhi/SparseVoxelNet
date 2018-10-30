import numpy as np

def block_configs(net_flag):
  '''
  '''
  net_flag = '5A'
  #*****************************************************************************
  block_configs = {}

  block_configs['dense_filters'] = [256, 128]

  if net_flag == '5A':
    # scale num = 3
    filters_e = [ [32,64], [64], [128], [256], [512]]
    block_sizes_e = [[1,1], [1]*1, [1]*1, [1]*1, [1]*1]

    filters_d = [ [512], [256], [256], [256]]
    block_sizes_d = [[1]*1, [1]*1, [1]*1, [1]*1]

  else:
    raise NotImplementedError

  block_flag = ''

  block_configs['block_sizes_e'] = block_sizes_e
  block_configs['filters_e'] = filters_e

  block_configs['block_sizes_d'] = block_sizes_d
  block_configs['filters_d'] = filters_d
  #*****************************************************************************

  block_configs['block_flag'] = ''

  return block_configs
