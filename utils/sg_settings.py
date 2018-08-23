# xyz 14 Aug

import numpy as np
MAX_FLOAT_DRIFT = 1e-5

def check_sg_setting_for_vox(sg_settings):
  # Both scale 0 (global) and 1 cannot get voxel size
  # global scale, the raw point cloud scope is unknow
  # scale 1 is not aligned
  vox_sizes = [[],[]]
  num_sg_scale = sg_settings['num_sg_scale']
  flag = '_VS_'
  for s in range(2, num_sg_scale+1):
    if s<num_sg_scale:
      cur_width = sg_settings['width'][s]
    else:
      # global block
      cur_width = sg_settings['width'][0]
    vox_size0 = ( cur_width - sg_settings['width'][s-1])/\
                  sg_settings['stride'][s-1] + 1
    vox_size = np.round(vox_size0).astype(np.int32)
    vox_size_err = np.max(np.abs(vox_size0-vox_size))
    #if s != num_sg_scale:
      # global scale can be not aligned
    assert vox_size_err < 1e-5, "the sg_settings cannot do Voxelization"
    vox_sizes.append(vox_size)
    flag += str(vox_size[0])

  sg_settings['vox_size'] = vox_sizes
  sg_settings['flag'] += flag
  return sg_settings

def update_sg_str(sg_settings):
  sg_str = ''
  for item in ['width', 'stride', 'nblock', 'npoint_per_block', 'vox_size']:
    v = sg_settings[item]
    if type(v) == type([]):
      v = [e.tolist() if type(e)==type(np.array([])) else e for e in v]
    str_i = str(v).replace('\n', ',')
    sg_str += item + ':' + str_i + '\n'
  sg_settings['sg_str'] = sg_str
  return sg_settings

def sg_flag(sg_settings_):
  tmp = np.concatenate( [sg_settings_['npoint_per_block'][0:1], sg_settings_['nblock'][1:]],0 )
  return '_'.join([str(int(e)) for e in tmp])

def get_sg_settings(sgflag):
  sg_settings_all = {}


  #-----------------------------------------------------------------------------
  sg_settings = {}
  sg_settings['width'] =  [[4.3,4.3,4.3], ]
  sg_settings['stride'] = [[2.0,2.0,2.0], ]
  sg_settings['nblock'] =  [4,            ]
  sg_settings['max_nblock'] =  [30,       ]
  sg_settings['npoint_per_block'] = [8192, ]
  sg_settings['np_perb_min_include'] = [8192*2,]
  sg_settings_all[sg_flag(sg_settings)] = sg_settings
  sg_settings_all['A'] = sg_settings

  #-----------------------------------------------------------------------------
  sg_settings = {}
  sg_settings['width'] =  [[4.3,4.3,4.3], [0.3,0.3,0.3], [1.1,1.1,1.1]]
  sg_settings['stride'] = [[2.0,2.0,2.0], [0.2,0.2,0.2], [0.8,0.8,0.8]]
  sg_settings['nblock'] =  [4,            1024,    64]
  sg_settings['max_nblock'] =  [30,       3000,   300]
  sg_settings['npoint_per_block'] = [8192, 32,   48]
  sg_settings['np_perb_min_include'] = [8192*2,3,1]
  sg_settings_all[sg_flag(sg_settings)] = sg_settings

  #-----------------------------------------------------------------------------
  sg_settings = {}
  sg_settings['width'] =  [[2.2,2.2,2.2], [0.2,0.2,0.2], [0.6,0.6,0.6]]
  sg_settings['stride'] = [[2.2,2.2,2.2], [0.1,0.1,0.1], [0.4,0.4,0.4]]
  sg_settings['nblock'] =  [1, 1024,           48]
  sg_settings['max_nblock'] =      [1, 4000,  200]
  sg_settings['npoint_per_block'] = [4096, 40,  64]
  sg_settings['np_perb_min_include'] = [1024, 2, 1]
  sg_settings_all[sg_flag(sg_settings)] = sg_settings

  #-----------------------------------------------------------------------------
  # 2048_800_40
  sg_settings = {}
  sg_settings['width'] =  [[2.2,2.2,2.2], [0.2,0.2,0.2], [0.6,0.6,0.6]]
  sg_settings['stride'] = [[2.2,2.2,2.2], [0.1,0.1,0.1], [0.4,0.4,0.4]]
  sg_settings['nblock'] =  [1,     800,    40]
  sg_settings['npoint_per_block'] = [2048, 24,  48]
  sg_settings['max_nblock'] =      [1, 4000,  200]
  sg_settings['np_perb_min_include'] = [1024, 2, 1]
  sg_settings_all[sg_flag(sg_settings)] = sg_settings

  #-----------------------------------------------------------------------------
  # 2048
  sg_settings = {}
  sg_settings['width'] =  [[2.2,2.2,2.2]]
  sg_settings['stride'] = [[2.2,2.2,2.2]]
  sg_settings['nblock'] =  [1]
  sg_settings['npoint_per_block'] = [2048]
  sg_settings['max_nblock'] =      [1]
  sg_settings['np_perb_min_include'] = [1024]
  sg_settings_all[sg_flag(sg_settings)] = sg_settings

  sg_settings = sg_settings_all[sgflag]

  for item in sg_settings:
    sg_settings[item] = np.array(sg_settings[item])
    if item in ['width', 'stride']:
      sg_settings[item] = sg_settings[item].astype(np.float32)
    else:
      sg_settings[item] = sg_settings[item].astype(np.int32)

  sg_settings['flag'] = 'SG_'+sg_flag(sg_settings)
  sg_settings['num_sg_scale'] = len(sg_settings['width'])
  sg_settings['gen_ply'] = False
  sg_settings['record'] = True


  sg_settings['nblocks_per_point'] = np.ceil(sg_settings['width']/sg_settings['stride']-MAX_FLOAT_DRIFT).astype(np.int32)
  sg_settings = check_sg_setting_for_vox(sg_settings)
  sg_settings = update_sg_str(sg_settings)

  return sg_settings

if __name__ == '__main__':
  sg_settings = get_sg_settings()



