# xyz 14 Aug

import numpy as np
MAX_FLOAT_DRIFT = 1e-5

def check_sg_setting_for_vox(sg_settings):
  # Both scale 0 (global) and 1 cannot get voxel size
  # global scale, the raw point cloud scope is unknow
  # scale 1 is not aligned
  vox_sizes = [[],[]]
  num_sg_scale = sg_settings['num_sg_scale']
  flag = ''
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
  sg_settings['flag'] = flag
  return sg_settings

def get_sg_settings():
  sg_settings0 = {}
  sg_settings0['width'] =  [[1.0,1.0,1.0], [0.2,0.2,0.2], [0.6,0.6,0.6]]
  sg_settings0['stride'] = [[1.0,1.0,1.0], [0.1,0.1,0.1], [0.4,0.4,0.4]]
  sg_settings0['nblock'] =  [9, 512,           64]
  sg_settings0['max_nblock'] =  [10, 6000,  500]
  sg_settings0['npoint_per_block'] = [1024, 10,   12]
  sg_settings0['np_perb_min_include'] = [128, 4, 2]


  sg_settings1 = {}
  sg_settings1['width'] =  [[2.2,2.2,2.2], [0.2,0.2,0.2], [0.6,0.6,0.6]]
  sg_settings1['stride'] = [[2.2,2.2,2.2], [0.1,0.1,0.1], [0.4,0.4,0.4]]
  sg_settings1['nblock'] =  [1, 1024,           48]
  sg_settings1['max_nblock'] =      [1, 3000,  200]
  sg_settings1['npoint_per_block'] = [4096, 32,  48]
  sg_settings1['np_perb_min_include'] = [1024, 2, 1]

  sg_settings = sg_settings1

  for item in sg_settings:
    sg_settings[item] = np.array(sg_settings[item])
    if item in ['width', 'stride']:
      sg_settings[item] = sg_settings[item].astype(np.float32)
    else:
      sg_settings[item] = sg_settings[item].astype(np.int32)

  sg_settings['num_sg_scale'] = len(sg_settings['width'])
  sg_settings['gen_ply'] = True
  sg_settings['record'] = False


  sg_settings['nblocks_per_point'] = np.ceil(sg_settings['width']/sg_settings['stride']-MAX_FLOAT_DRIFT).astype(np.int32)
  sg_settings = check_sg_setting_for_vox(sg_settings)

  return sg_settings
