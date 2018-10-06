import numpy as np

def block_configs(net_flag='default'):
  block_configs = {}
  block_configs['use_face_global_scale0'] = False
  block_configs['e2fl_pool'] = ['max']
  block_configs['f2v_pool'] = ['max']

  #*****************************************************************************
  block_sizes = {}
  filters = {}

  if net_flag == '3A':
    block_sizes['edge'] = [ [1],  [1], [1] ]
    filters['edge']     = [ [16], [32], [64]]

    block_sizes['centroid']=[ [1],  [1], [1] ]
    filters['centroid']   = [ [16], [32], [64]]

    block_sizes['face'] = [ [1, 1],  [1, 1], [1, 1 ]]
    filters['face']     = [ [32, 32], [32, 32], [64, 64]]

    block_sizes['vertex']=[ [1],  [1], [1] ]
    filters['vertex']   = [ [64], [64], [128]]

  elif net_flag == '3B':
    block_sizes['edge'] = [ [1],  [1], [1] ]
    filters['edge']     = [ [32], [64], [128]]

    block_sizes['centroid']=[ [1],  [1], [1] ]
    filters['centroid']   = [ [32], [64], [64]]

    block_sizes['face'] = [ [2],  [2],   [2]]
    filters['face']     = [ [64], [128], [128]]

    block_sizes['vertex']=[ [2],  [2], [2] ]
    filters['vertex']   = [ [64], [128], [256]]


  else:
    raise NotImplementedError

  tmp = [i  for fs in filters.values() for f in fs for i in f]
  block_flag = '%d_%d'%(len(tmp), np.mean(tmp))

  block_configs['block_sizes'] = block_sizes
  block_configs['filters'] = filters
  block_configs['block_flag'] = block_flag

  return block_configs
