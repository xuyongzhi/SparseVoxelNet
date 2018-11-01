import numpy as np

def block_configs(net_flag):
  '''
  '''
  #*****************************************************************************
  block_configs = {}

  block_configs['dense_filters'] = [128]

  if net_flag == '4A':
    filters_e = [ [32,64], [64,128], [128+3,256], [512]]
    block_sizes_e = [[1,2], [1]*2, [1]*2, [1]*1]

    filters_d = [ [256, 128], [384], [384]]
    block_sizes_d = [[1]*2, [1]*1, [1]*1]

  elif net_flag == '1A':
    filters_e = [ [32,128, 256,512], [512, 1024]]
    block_sizes_e = [[1,2,2,1], [1]*2]

    filters_d = [ [512, 256, 128],]
    block_sizes_d = [[1, 2,2],]


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
