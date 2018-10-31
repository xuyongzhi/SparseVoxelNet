import os, glob,sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
import math, time
from graph_util import MeshSampling


NET_FLAG = 'FaceLabel_GraphCnn'
NET_FLAG = 'VertexLabel_PointCnn'
#*******************************************************************************
if NET_FLAG == 'FaceLabel_GraphCnn':
  Vertex_feles = ['xyz','nxnynz']
  Vertex_ieles = ['fidx_per_vertex', 'edgev_per_vertex', 'valid_ev_num_pv']
  Vertex_uint8_eles = ['color', 'fidx_pv_empty_mask']
  Face_ieles = ['vidx_per_face', 'label_category']
  _parse_local_graph_pv = True
#*******************************************************************************
elif NET_FLAG == 'VertexLabel_PointCnn':
  Vertex_feles = ['xyz','nxnynz']
  Vertex_ieles = []
  Vertex_uint8_eles = ['color', 'label_category']
  Face_ieles = []
  _parse_local_graph_pv = False

MAX_FLOAT_DRIFT = 1e-6
DEBUG = False

def random_choice(org_vector, sample_N, random_sampl_pro=None,
                  keeporder=False, only_tile_last_one=False):
    assert org_vector.ndim == 1
    org_N = org_vector.size
    if org_N == sample_N:
        sampled_vector = org_vector
    elif org_N > sample_N:
        sampled_vector = np.random.choice(org_vector,sample_N,replace=False,p=random_sampl_pro)
        if keeporder:
            sampled_vector = np.sort(sampled_vector)
    else:
        if only_tile_last_one:
            new_vector = np.array( [ org_vector[-1] ]*(sample_N-org_N) ).astype(org_vector.dtype)
        else:
            new_vector = np.random.choice(org_vector,sample_N-org_N,replace=True)
        sampled_vector = np.concatenate( [org_vector,new_vector] )
    #str = '%d -> %d  %d%%'%(org_N,sample_N,100.0*sample_N/org_N)
    #print(str)
    return sampled_vector


def ele_in_feature(features, ele, ds_idxs):
  #ds_idxs = dset_shape_idx['indices']
  for g in ds_idxs:
    if ele in ds_idxs[g]:
      ele_idx = ds_idxs[g][ele]
      if isinstance(features[g], tf.Tensor):
        ele_data = tf.gather(features[g], ele_idx, axis=-1)
      else:
        ele_data = np.take(features[g], ele_idx, axis=-1)
      return ele_data
  return None
  #raise ValueError, ele+' not found'

class Raw_To_Tfrecord():
  def __init__(self, dataset_name, tfrecord_path, num_point=None, block_size=None,
               dynamic_block_size=False, ply_dir=None, is_multi_pro=False):
    self.dataset_name = dataset_name
    self.tfrecord_path = tfrecord_path
    self.data_path = os.path.join(self.tfrecord_path, 'data')

    if not os.path.exists(self.data_path):
      os.makedirs(self.data_path)
    self.num_point = num_point
    self.min_sample_rate = 0.03 # sample_rate = self.num_point/org_num
    self.num_face = int(num_point * 8)
    self.block_size = block_size
    self.dynamic_block_size = dynamic_block_size
    self.block_stride_rate =  0.8
    self.min_pn_inblock = min(self.num_point * 0.1, 2000)
    self.sampling_rates = []
    self.dataset_meta = DatasetsMeta(self.dataset_name)

    self.ply_dir = ply_dir
    self.is_multi_pro = is_multi_pro

  def __call__(self, rawfns):
    bsfn0 = os.path.join(self.tfrecord_path, 'block_split_settings.txt')
    if type(self.block_size)!=type(None):
      with open(bsfn0, 'w') as bsf:
        bsf.write('num_point:{}\nblock_size:{}\nblock_stride_rate:{}\nmin_pn_inblock:{}'.format(\
                      self.num_point, self.block_size,\
                      self.block_stride_rate, self.min_pn_inblock))

    block_split_dir = os.path.join(self.tfrecord_path, 'block_split_summary')
    if not os.path.exists(block_split_dir):
      os.makedirs(block_split_dir)
    all_sp_log_fn = os.path.join(self.tfrecord_path, 'mesh_sampling.log')
    all_sp_logf = open(all_sp_log_fn, 'a')

    self.fn = len(rawfns)
    #rawfns = rawfns[5577:]
    rawfns.sort()
    min_sp_rate = 1
    fi = -1
    for fi, rawfn in enumerate(rawfns):
      self.fi = fi
      region_name = os.path.splitext(os.path.basename(rawfn))[0]
      scene_name = os.path.basename(os.path.dirname(os.path.dirname(rawfn)))
      scene_bs_fn = os.path.join(block_split_dir, scene_name + '_' + region_name + '.txt')

      cur_tfrecord_intact = False
      if os.path.exists(scene_bs_fn):
        with open(scene_bs_fn, 'r') as bsf:
          for line in bsf:
            line = line.strip()
            break
          if line == 'intact':
            cur_tfrecord_intact = True
      if cur_tfrecord_intact:
        print('skip {} intact'.format(scene_bs_fn))
        continue

      if self.dataset_name == "MATTERPORT":
        block_num, valid_block_num, num_points_splited, xyz_scope_str,mesh_summary \
          = self.transfer_onefile_matterport(rawfn)

      sp_rates = [1.0*self.num_point/orgn for orgn in  num_points_splited]
      if min(sp_rates) < min_sp_rate:
        min_sp_rate = min(sp_rates)
      with open(scene_bs_fn, 'w') as bsf:
        bsf.write('intact\n')
        bnstr = '{} \tblock_num:{} \tvalid block num:{}\n{}\n\n'.format(region_name, block_num, valid_block_num, xyz_scope_str)
        bsf.write(bnstr)
        for i in range(valid_block_num):
          org_num_vertex_i = num_points_splited[i]
          vnstr = '\n'+'\n'.join(['{}_{}: org_num_vertex={}, sp_rate={}'.format(region_name, i, org_num_vertex_i, sp_rates[i]) ])
          bsf.write(vnstr)
        for key in mesh_summary:
          bsf.write('\n{}: {}'.format(key, mesh_summary[key]))

      all_sp_logf.write('{} {}\t sp_rates:{}\n'.format(scene_name, region_name,  sp_rates))
      all_sp_logf.flush()
    if len(rawfns)>0:
      all_sp_logf.write('\nmin_sp_rate:{}\n\n'.format(min_sp_rate))
    print('All {} file are converted to tfreord'.format(fi+1))

  def sort_eles(self, all_eles):
    # elements *************************
    self.eles_sorted = {}
    self.eles_sorted['vertex_f'] = [e for e in Vertex_feles if e in all_eles]
    if len(Vertex_ieles) > 0:
      self.eles_sorted['vertex_i'] = [e for e in Vertex_ieles if e in all_eles]
    self.eles_sorted['vertex_uint8'] = [e for e in Vertex_uint8_eles if e in all_eles]
    if len(Face_ieles) != 0:
      self.eles_sorted['face_i'] = [e for e in Face_ieles if e in all_eles]

  def record_shape_idx(self, raw_datas, dls):
    ele_idxs = {}
    for item in self.eles_sorted:
      ele_idxs[item] = {}
      n = 0
      for e in self.eles_sorted[item]:
        shape_i = raw_datas[e].shape
        ele_idxs[item][e] = np.arange(n, n+shape_i[-1])
        n += shape_i[-1]
    self.ele_idxs = ele_idxs

    metas_fn = os.path.join(self.tfrecord_path, 'shape_idx.txt')
    if  os.path.exists(metas_fn):
      from tfrecord_util import get_dset_shape_idxs
      dset_shape_idx_old = get_dset_shape_idxs(self.tfrecord_path)
      for item in self.eles_sorted:
        assert np.all( [s for s in dls[item].shape] == dset_shape_idx_old['shape'][item]),'shape different'
        for ele in self.eles_sorted[item]:
          assert np.all(ele_idxs[item][ele] == dset_shape_idx_old['indices'][item][ele]),'indices different'
      return

    with open(metas_fn, 'w') as f:
      f.write('dataset_name:{}\n'.format(self.dataset_name))
      for item in self.eles_sorted:
        shape = dls[item].shape
        shape_str = ','.join([str(k) for k in shape])
        shape_str = 'shape: {}: {}\n'.format(item, shape_str)
        f.write(shape_str)

        for ele in self.eles_sorted[item]:
          idxs_str = ','.join([str(k) for k in  ele_idxs[item][ele]])
          e_str = 'indices: {}: {}: {}\n'.format(item, ele, idxs_str)
          f.write(e_str)

      if self.dataset_name == 'MODELNET40':
        f.write('indices: labels: label_category: 0\n')
      f.flush()
    print('write ok: {}'.format(metas_fn))

  def autoadjust_block_size(self, xyz_scope, num_vertex0):
    # keep xy area within the threshold, adjust to reduce block num
    _x,_y,_z = self.block_size
    _xy_area = _x*_y
    x0,y0,z0 = xyz_scope
    xy_area0 = x0*y0

    nv_rate = 1.0*num_vertex0/self.num_point

    if xy_area0 <= _xy_area or nv_rate<=1.0:
      #Use one block: (1) small area (2) Very small vertex number
      x2 = x0
      y2 = y0

    else:
      # Need to use more than one block. Try to make block num less which is
      # make x2, y2 large.
      # (3) Large area with large vertex number, use _x, _y
      # (4) Large area with not loo large vertex num. Increase the block size by
      # vertex num rate.
      dis_rate = math.sqrt(nv_rate)
      x1 = x0 / dis_rate * 0.9
      y1 = y0 / dis_rate * 0.9
      x2 = max(_x, x1)
      y2 = max(_y, y1)

    block_size= np.array([x2, y2, _z])
    block_size = np.ceil(10*block_size)/10.0
    print('xyz_scope:{}\nblock_size:{}'.format(xyz_scope, block_size))
    return block_size

  def split_vertex(self, xyz):
    if type(self.block_size) == type(None):
      return [None], None

    xyz_min = np.min(xyz, 0)
    xyz_max = np.max(xyz, 0)
    xyz_scope = xyz_max - xyz_min
    num_vertex0 = xyz.shape[0]

    if self.dynamic_block_size:
      block_size = self.autoadjust_block_size(xyz_scope, num_vertex0)
    else:
      block_size = self.block_size
    block_stride = self.block_stride_rate * block_size
    block_dims0 =  (xyz_scope - block_size) / block_stride + 1
    block_dims0 = np.maximum(block_dims0, 1)
    block_dims = np.ceil(block_dims0).astype(np.int32)
    #print(block_dims)
    xyzindices = [np.arange(0, k) for k in block_dims]
    bot_indices = [np.array([[xyzindices[0][i], xyzindices[1][j], xyzindices[2][k]]]) for i in range(block_dims[0]) \
                   for j  in range(block_dims[1]) for k in range(block_dims[2])]
    bot_indices = np.concatenate(bot_indices, 0)
    bot = bot_indices * block_stride
    top = bot + block_size

    block_num = bot.shape[0]
    #print('raw scope:\n{} \nblock_num:{}'.format(xyz_scope, block_num))
    #print('splited bot:\n{} splited top:\n{}'.format(bot, top))

    if block_num == 1:
      return [None], block_size

    if block_num>1:
      for i in range(block_num):
        for j in range(3):
          if top[i,j] > xyz_scope[j]:
            top[i,j] = xyz_scope[j] - MAX_FLOAT_DRIFT
            bot[i,j] = np.maximum(xyz_scope[j] - block_size[j] + MAX_FLOAT_DRIFT, 0)

    bot += xyz_min
    top += xyz_min

    dls_splited = []
    num_points_splited = []
    splited_vidx = []
    for i in range(block_num):
      mask0 = xyz >= bot[i]
      mask1 = xyz < top[i]
      mask = mask0 * mask1
      mask = np.all(mask, 1)
      new_n = np.sum(mask)
      indices = np.where(mask)[0]
      num_point_i = indices.size
      if num_point_i < self.min_pn_inblock:
        #print('num point {} < {}, block {}/{}'.format(num_point_i,\
        #                                self.num_point * 0.1, i, block_num))
        continue
      num_points_splited.append(num_point_i)
      splited_vidx.append(indices)

    num_vertex_splited = [d.shape[0] for d in splited_vidx]
    return splited_vidx, block_size


  @staticmethod
  def downsample_face(vidx_per_face, point_sampling_indices, num_point0):
    num_face0 = vidx_per_face.shape[0]
    point_mask = np.zeros(num_point0, np.int8)
    point_mask[point_sampling_indices] = 1
    point_mask = point_mask.astype(np.bool)
    face_vertex_mask = np.take(point_mask, vidx_per_face)
    face_vertex_mask = np.all(face_vertex_mask, 1)
    face_keep_indices = np.where(face_vertex_mask)[0]
    face_del_indices = np.where(np.logical_not(face_vertex_mask))[0]
    del_face_num = face_del_indices.shape[0] # 16653

    del_vertex_rate = 1 - 1.0* point_sampling_indices.shape[0] / num_point0
    del_face_rate = 1.0 * del_face_num / num_face0
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass


  def get_label_MODELNET40(self):
    # get label for MODELNET40
    if self.dataset_name == 'MODELNET40':
      tmp = base_name.split('_')
      if len(tmp) == 2:
        category = tmp[0]
      elif len(tmp) == 3:
        category = '_'.join(tmp[0:2])
      object_label = dataset_meta.class2label[category]
      dls['labels'] = np.array([object_label])



  def transfer_onefile_matterport(self, rawfn):
    from MATTERPORT_util.MATTERPORT_util import parse_ply_file, parse_ply_vertex_semantic

    print('starting {} th file: {}'.format(self.fi, rawfn))
    base_name = os.path.splitext(os.path.basename(rawfn))[0]
    base_name1 = os.path.basename(os.path.dirname(os.path.dirname(rawfn)))
    base_name = base_name1 + '_' + base_name

    # ['label_material', 'label_category', 'vidx_per_face', 'color',
    # 'xyz', 'nxnynz', 'label_raw_category', 'label_instance']
    if NET_FLAG == 'FaceLabel_GraphCnn':
      raw_datas = parse_ply_file(rawfn)
    elif NET_FLAG == 'VertexLabel_PointCnn':
      raw_datas = parse_ply_vertex_semantic(rawfn)
      if len(Face_ieles)==0:
        del raw_datas['vidx_per_face']

    splited_vidx, dy_block_size = self.split_vertex(raw_datas['xyz'])
    block_num = len(splited_vidx)
    valid_block_num = len(splited_vidx)
    num_points_splited = [e.shape[0] if type(e)!=type(None) else raw_datas['xyz'].shape[0]\
                          for e in splited_vidx]
    sp_rates = [1.0*self.num_point/k for k in num_points_splited]
    min_sp_rate = min(sp_rates)
    assert min_sp_rate > self.min_sample_rate, 'got small sample rate:{} < {}'.format(
                                              min_sp_rate, self.min_sample_rate)

    #main_split_sampling_rawmesh = MeshSampling.eager_split_sampling_rawmesh
    main_split_sampling_rawmesh = MeshSampling.sess_split_sampling_rawmesh
    splited_sampled_datas, raw_vertex_nums, mesh_summary = main_split_sampling_rawmesh(
        raw_datas, self.num_point, splited_vidx, self.dataset_meta,
        _parse_local_graph_pv, self.ply_dir)


    for bi in range(block_num):
      tmp = '' if block_num==1 else '_'+str(bi)
      tfrecord_fn = os.path.join(self.data_path, base_name)+tmp + '.tfrecord'
      self.transfer_one_block(tfrecord_fn, splited_sampled_datas[bi], raw_vertex_nums[bi])
    print('finish {} th file: {}'.format(self.fi, rawfn))
    min_xyz = np.min(raw_datas['xyz'],0)
    max_xyz = np.max(raw_datas['xyz'],0)
    scope = max_xyz - min_xyz
    strs = [np.array2string(d, precision=2) for d in [min_xyz, max_xyz, scope] ]
    xyz_scope_str = 'min: {}, max:{}, scope:{}\ndynamic block size:{}'.format(strs[0], strs[1], strs[2], dy_block_size)
    return block_num, valid_block_num, num_points_splited, xyz_scope_str, mesh_summary


  @staticmethod
  def check_types(dls):
    if dls['vertex_uint8'].dtype != np.uint8:
      assert np.min(dls['vertex_uint8']) >= 0
      assert np.max(dls['vertex_uint8']) < 256
      dls['vertex_uint8'] = dls['vertex_uint8'].astype(np.uint8)
    if 'vertex_i' in dls:
      assert dls['vertex_i'].dtype == np.int32
    assert dls['vertex_uint8'].dtype == np.uint8
    assert dls['vertex_f'].dtype == np.float32
    if 'face_i' in dls:
      assert dls['face_i'].dtype == np.int32

    return dls

  def transfer_one_block(self, tfrecord_fn, block_sampled_datas, raw_vertex_num):
    from tfrecord_util import bytes_feature, int64_feature

    if not hasattr(self, 'eles_sorted'):
      self.sort_eles(block_sampled_datas.keys())
    #*************************************************************************
    dls = {}
    for item in self.eles_sorted:
      dls[item] = []
      for e in self.eles_sorted[item]:
        data = block_sampled_datas[e]
        dls[item].append( data )
      if len(dls[item]) >  0:
        dls[item] = np.concatenate(dls[item], -1) # dtye auto transform here if needed

    #*************************************************************************
    # fix face_i shape
    if 'face_i' in dls:
      face_shape = dls['face_i'].shape
      tile_num = self.num_face - face_shape[0]
      assert tile_num>=0, "face num > buffer: {}>{}".format(face_shape[0], self.num_face)
      tmp = np.tile( dls['face_i'][0:1,:], [tile_num, 1])
      #tmp = np.ones([self.num_face - face_shape[0], face_shape[1]], np.int32) * (-777)
      dls['face_i'] = np.concatenate([dls['face_i'], tmp], 0)

    #*************************************************************************
    # convert to expample
    dls = Raw_To_Tfrecord.check_types(dls)
    if not hasattr(self, 'ele_idxs'):
      self.record_shape_idx(block_sampled_datas, dls)
      print(self.ele_idxs)

    max_category =  np.max(ele_in_feature(dls, 'label_category', self.ele_idxs))
    assert max_category < self.dataset_meta.num_classes, "max_category {} > {}".format(\
                                          max_category, self.dataset_meta.num_classes)

    vertex_f_bin = dls['vertex_f'].tobytes()
    #vertex_i_shape_bin = np.array(dls['vertex_i'].shape, np.int32).tobytes()
    if 'vertex_i' in dls:
      vertex_i_bin = dls['vertex_i'].tobytes()
    vertex_uint8_bin = dls['vertex_uint8'].tobytes()
    if 'face_i' in dls:
      face_i_bin = dls['face_i'].tobytes()


    features_map = {
      'vertex_f': bytes_feature(vertex_f_bin),
      'vertex_uint8': bytes_feature(vertex_uint8_bin),
    }
    if 'vertex_i' in dls:
      features_map['vertex_i'] = bytes_feature(vertex_i_bin)
    if 'face_i' in dls:
      features_map['face_i'] = bytes_feature(face_i_bin)
      features_map['valid_num_face'] = int64_feature(face_shape[0])

    example = tf.train.Example(features=tf.train.Features(feature=features_map))

    #*************************************************************************
    with tf.python_io.TFRecordWriter( tfrecord_fn ) as raw_tfrecord_writer:
      raw_tfrecord_writer.write(example.SerializeToString())

    if self.fi %5 ==0:
      print('{}/{} write tfrecord OK: {}'.format(self.fi, self.fn, tfrecord_fn))

def read_dataset_summary(data_dir):
  import pickle
  summary_path = os.path.join(data_dir, 'summary.pkl')
  if not os.path.exists(summary_path):
    dataset_summary = {}
    dataset_summary['intact'] = False
    return dataset_summary
  dataset_summary = pickle.load(open(summary_path, 'r'))
  return dataset_summary


def get_label_num_weights(dataset_summary, loss_lw_gama):
  if loss_lw_gama<0:
    return
  IsPlot = False
  label_hist = dataset_summary['label_hist']
  mean = np.mean(label_hist)
  assert np.min(label_hist) > 0
  weight = mean / label_hist
  weights = {}
  gamas = [loss_lw_gama, 1, 2, 5, 10, 20]
  gamas = [loss_lw_gama]
  for gama in gamas:
    weights[gama] = gama * weight
  dataset_summary['label_num_weights'] = weights[loss_lw_gama]
  if  IsPlot:
    import matplotlib.pyplot as plt
    for gama in gamas:
      plt.plot(label_hist, weights[gama], '.', label=str(gama))
    plt.legend()
    plt.show()


def main_write(dataset_name, raw_fns, tfrecord_path, num_point, block_size,
               dynamic_block_size, ply_dir, is_multi_pro):
  raw_to_tf = Raw_To_Tfrecord(dataset_name, tfrecord_path, num_point, block_size,
                              dynamic_block_size, ply_dir, is_multi_pro)
  raw_to_tf(raw_fns)


def main_write_multi(dataset_name, raw_fns, tfrecord_path, num_point,
                     block_size, dynamic_block_size, ply_dir, multiprocessing=5):
  if multiprocessing<2:
    main_write(dataset_name, raw_fns, tfrecord_path, num_point, block_size,
               dynamic_block_size, ply_dir, False)
    return

  import multiprocessing as mp
  pool = mp.Pool(multiprocessing)
  for fn in raw_fns:
    pool.apply_async(main_write, (dataset_name, [fn], tfrecord_path, num_point,
                                  block_size, dynamic_block_size, ply_dir, True))
  pool.close()
  pool.join()

  main_write(dataset_name, raw_fns, tfrecord_path, num_point, block_size,
                dynamic_block_size, ply_dir, False)

def split_fn_ls( tfrecordfn_ls, merged_n):
    nf = len(tfrecordfn_ls)
    assert merged_n < nf
    group_n = int(math.ceil( 1.0*nf/merged_n ))
    merged_fnls = []
    group_name_ls = []
    for i in range( 0, nf, group_n ):
        end = min( nf, i+group_n )
        merged_fnls.append( tfrecordfn_ls[i:end] )
        group_name_ls.append( '%d_%d'%(i, end) )
    return merged_fnls, group_name_ls

def main_merge_tfrecord(dataset_name, tfrecord_path):
  from utils.dataset_utils import merge_tfrecord
  import random

  rawdata_dir = tfrecord_path+'/data'
  merged_dir = tfrecord_path+'/merged_data'
  if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)

  datasets_meta = DatasetsMeta(dataset_name)
  train_fn_ls = datasets_meta.get_train_test_file_list(rawdata_dir, True)
  random.shuffle(train_fn_ls)
  test_fn_ls = datasets_meta.get_train_test_file_list(rawdata_dir, False)
  random.shuffle(test_fn_ls)

  grouped_train_fnls, train_groupe_names = split_fn_ls(train_fn_ls, 100)
  train_groupe_names = ['train_'+e for e in train_groupe_names]
  grouped_test_fnls, test_groupe_names = split_fn_ls(test_fn_ls, 10)
  test_groupe_names = ['test_'+e for e in test_groupe_names]

  grouped_fnls = grouped_train_fnls + grouped_test_fnls
  group_names = train_groupe_names + test_groupe_names
  merged_fnls = [os.path.join(merged_dir, e+'.tfrecord') for e in group_names]

  for k in range(len(grouped_fnls)):
    merge_tfrecord(grouped_fnls[k], merged_fnls[k])

def main_gen_ply(dataset_name, tf_path):
  #scene = 'gxdoqLR6rwA_region2'  # large ground
  #scene = 'gTV8FGcVJC9_region7'
  #scene = 'Vvot9Ly1tCj_region24'  # most unlabled points
  scene = 'Pm6F8kyY3z2_region3'
  #scene = '17DRP5sb8fy_region0'
  #scene = 'ac26ZMwG7aT_region9'

  scene = 'YFuZgdQ5vWj_region19'  # all void
  scene = 'VFuaQ6m2Qom_region38'  # all void

  name_base = 'data/{}*.tfrecord'.format(scene)
  fn_glob = os.path.join(tf_path, name_base)
  filenames = glob.glob(fn_glob)
  assert len(filenames) > 0, fn_glob
  for fn in filenames:
    gen_ply_onef(dataset_name, tf_path, fn, scene)

def gen_ply_onef(dataset_name, tf_path, filename, scene):
  from utils.ply_util import create_ply_dset
  is_rm_void_labels = True
  is_rm_void_labels = False

  ply_dir = os.path.join(tf_path, 'plys/ply_'+scene)
  if not os.path.exists(ply_dir):
    os.makedirs(ply_dir)

  data_infos = get_dset_shape_idxs(tf_path)
  datasets_meta = DatasetsMeta(dataset_name)
  num_classes = datasets_meta.num_classes

  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset([filename],
                                      compression_type="",
                                      buffer_size=1024*100,
                                      num_parallel_reads=1)

    batch_size = 10
    is_training = False

    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        lambda value: parse_pl_record(value, is_training, data_infos,\
                          is_rm_void_labels=is_rm_void_labels),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=False))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    get_next = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      n = 0
      all_points = []
      all_label_categories = []
      try:
        print('start reading all the dataset to get summary')
        while(True):
          features, labels = sess.run(get_next)

          label_category = labels[:,:, data_infos['indices']['labels']['label_category'][0]]
          all_label_categories.append(label_category)

          xyz = features['points'][:,:, data_infos['indices']['points']['xyz']]
          color = features['points'][:,:, data_infos['indices']['points']['color']]
          points = np.concatenate([xyz, color], -1)
          all_points.append(points)
          n += 1
      except:
        assert n>0, "some thing wrong inside parse_pl_record"
        pass
      all_points = np.concatenate(all_points, 0)
      all_label_categories = np.concatenate(all_label_categories, 0)
      base_name = os.path.splitext( os.path.basename(filename) )[0]
      if is_rm_void_labels:
        base_name = base_name + '_rmvoid'

      ply_fn = os.path.join(ply_dir, base_name+'.ply')
      create_ply_dset( dataset_name, all_points,  ply_fn)
      #
      ply_fn = os.path.join(ply_dir+'_labeled', base_name+'.ply')
      create_ply_dset( dataset_name, all_points[...,0:3],  ply_fn, all_label_categories)

def clean_bad_files(dataset_name, raw_fns, dset_path):
  datasets_meta = DatasetsMeta(dataset_name)
  bad_files = [dset_path+'/'+bf for bf in datasets_meta.bad_files]
  raw_fns_cleaned = [fn for fn in raw_fns if fn not in bad_files]
  return raw_fns_cleaned

def main_matterport():
  t0 = time.time()
  dataset_name = 'MATTERPORT'

  num_points = {'MODELNET40':None, 'MATTERPORT':8192}
  num_point = num_points[dataset_name]
  block_sizes = {'MODELNET40':None, 'MATTERPORT':np.array([1.5, 1.5, 3.0]) }
  dynamic_block_size = False
  block_size = block_sizes[dataset_name]
  flag = '_'.join([str(int(10*d)) for d in block_size]) + '_' + str(int(num_point/1000))+'K'

  dset_path = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans'
  tfrecord_path = '/DS/Matterport3D/MATTERPORT_TF/tfrecord_%s'%(flag)

  #dset_path = '/home/z/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans'
  #tfrecord_path = os.path.join(ROOT_DIR, 'data/MATTERPORT_TF')

  region_name = 'region*'
  scene_names_all = os.listdir(dset_path) # 91
  scene_names_all.sort()
  scene_names = ['17DRP5sb8fy']
  #scene_names = ['2t7WUuJeko7']
  scene_names = scene_names_all
  raw_fns = []
  for scene_name in scene_names:
    raw_glob = os.path.join(dset_path, '{}/*/region_segmentations/{}.ply'.format(
                                  scene_name, region_name))
    raw_fns += glob.glob(raw_glob)

  ply_dir = os.path.join(tfrecord_path, 'plys/{}/{}'.format(scene_name, region_name))

  raw_fns = clean_bad_files(dataset_name, raw_fns, dset_path)
  raw_fns.sort()
  main_write_multi(dataset_name, raw_fns, tfrecord_path, num_point,\
              block_size, dynamic_block_size, ply_dir,
              multiprocessing=4) # 4 to process data, 0 to check

  main_merge_tfrecord(dataset_name, tfrecord_path)

  #main_gen_ply(dataset_name, tfrecord_path)
  print('total time: {} sec'.format(time.time() - t0))



if __name__ == '__main__':
  main_matterport()

