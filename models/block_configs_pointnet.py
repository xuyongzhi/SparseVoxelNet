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
    end_blocks = []

    filters_d = [ [256, 128], [384], [384]]
    block_sizes_d = [[1]*2, [1]*1, [1]*1]

  elif net_flag == '1A' or net_flag=='1B':
    filters_e = [ [64, 64, 128, 1024], [256, 128]]
    block_sizes_e = [[0,1,2,1], [1,1]]
    if net_flag == '1A':
      end_blocks = [None,None]
    elif net_flag=='1B':
      end_blocks = [[0,1,2,3],None]

    filters_d = [ [512, 256],]
    block_sizes_d = [[1, 1],]


  else:
    raise NotImplementedError

  block_flag = ''

  block_configs['block_sizes_e'] = block_sizes_e
  block_configs['filters_e'] = filters_e
  block_configs['end_blocks'] = end_blocks

  block_configs['block_sizes_d'] = block_sizes_d
  block_configs['filters_d'] = filters_d
  #*****************************************************************************

  block_configs['block_flag'] = ''

  return block_configs
