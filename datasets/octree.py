# xyzs Oct 2018

import os, glob,sys
import numpy as np
import tensorflow as tf
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from utils.tf_util import TfUtil
from MATTERPORT_util.MATTERPORT_util import parse_ply_file, parse_ply_vertex_semantic

def align_float(x, unit=0.01):
  return tf.rint(x/unit)*unit


def idx3d_to_1d(idx_3d):
  vidx_1d = idx_3d[:,0] + idx_3d[:,1]*2 + idx_3d[:,2]*4
  return vidx_1d

def idx1d_to_3d(vidx_1d):
  idx3d_2 = tf.expand_dims(tf.floormod(vidx_1d, 2), -1)
  tmp = vidx_1d / 2
  idx3d_1 = tf.expand_dims(tf.floormod(tmp, 2), -1)
  idx3d_0 = tf.expand_dims(tmp / 2, -1)
  idx3d = tf.concat([idx3d_0, idx3d_1, idx3d_2], -1)
  return idx3d

def idx1d_to_3d_np(i1d):
  i3d_2 = i1d % 2
  i1d/= 2
  i3d_1 = i1d % 2
  i1d/= 2
  i3d_0 = i1d % 2
  return [i3d_2, i3d_1, i3d_0]


class OctreeTf():
  _check_optial = True
  _fal = 0.01
  _scale_num = 3
  def __init__(self, resolution=None, scale=None, min_xyz=None):
    self._scale = scale
    self._min = min_xyz
    self._resolution = resolution
    if self._scale==0:
      self._min = tf.constant([0,0,0], tf.float32)

    self._nodes = [None]*8


  def add_point_cloud(self, xyzs, sortedidxs_base=None, record=False, rawfn=None):
    '''
    xyzs input is not sorted for all scales
    '''
    assert TfUtil.tsize(xyzs) == 2
    vidx_1d = self.get_vidx1d(xyzs)

    sorted_idxs_base, end_idxs_lastscale = self.get_sort_idx(xyzs, vidx_1d, sortedidxs_base)
    if self._scale==0:
      vidx_rawscope_lsc = OctreeTf.get_idx_scope(end_idxs_lastscale)
      flatvidx_rawidxscopes, vidx_rawscopes = self.get_upper_scale_scope(vidx_rawscope_lsc)
      flat_idx_lsc = OctreeTf.get_flat_idx(vidx_rawscope_lsc[:,0:-2])
      flatvidx_rawscope_lsc = tf.concat([tf.expand_dims(flat_idx_lsc,-1), vidx_rawscope_lsc[:,-2:]], -1)
      flatvidx_rawidxscopes.append(flatvidx_rawscope_lsc)

      self.sorted_idxs_base = sorted_idxs_base
      self.flatvidx = [d[:,0:-2] for d in  flatvidx_rawidxscopes]
      self.vidx = [d[:,0:-2] for d in vidx_rawscopes]
      self.idxscope_pervs = [d[:,-2:] for d in vidx_rawscopes]
      self.show_summary()
      #self.merge_neighbour()
      #self.check_sort(xyzs)
      if record:
        self.recording(rawfn)
      #self.test_search_inv(xyzs)
    return sorted_idxs_base, end_idxs_lastscale

  def merge_neighbour(self):
    for s in range(self._scale_num-1, self._scale_num):
      vidx_s = self.vidx[s]
      vidx_3d = idx1d_to_3d(vidx_s[:,-1])
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  def show_summary(self):
    # vns:      [4, 18, 125, 630, 2592, 11050]
    # pn_pervs: [50431, 11207, 1613, 320, 77, 18]
    vns = [TfUtil.tshape0(v) for v in self.flatvidx]
    pn_pervs = []
    for s in range(self._scale_num):
      idxs = self.idxscope_pervs[s]
      np = idxs[:,1] - idxs[:,0]
      pn_pervs.append( tf.reduce_mean(np).numpy() )
    print(vns)
    print(pn_pervs)

  def check_sort(self, xyzs):
    xyzs = tf.gather(xyzs, self.sorted_idxs_base)
    scale = self._scale_num - 1
    idxscope_perv = self.idxscope_pervs[scale]
    flatvidx = self.flatvidx[scale]

    vn = TfUtil.get_tensor_shape(idxscope_perv)[0]
    #for i in np.random.choice(vn, 10):
    for i in range(0,10):
      s,e = idxscope_perv[i]
      xyzs_i = xyzs[s:e,:]
      flatvidx_i = flatvidx[i]
      min_i = tf.reduce_min(xyzs_i, 0)
      max_i = tf.reduce_max(xyzs_i, 0)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  def flatvidx_to_xyzscope(self, flatvidx):

    pass

  def get_vidx1d(self, xyzs):
    vidx_3d = OctreeTf.get_vidx3d(xyzs, self._min, self._resolution)
    vidx_1d = idx3d_to_1d(vidx_3d)
    return vidx_1d

  @staticmethod
  def get_vidx3d(xyzs, min_xyz, res):
    vidx_3d =  (xyzs - min_xyz)/res

    #if tf.reduce_min(vidx_3d)<0:
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #  pass
    c0 = tf.assert_greater_equal(tf.reduce_min(vidx_3d), 0.0)
    c1 = tf.assert_less(tf.reduce_max(vidx_3d), 8.0)
    with tf.control_dependencies([c0, c1]):
      vidx_3d = tf.cast(tf.math.floor(vidx_3d), tf.int32)
    return vidx_3d

  @staticmethod
  def get_vidx1d_all_scales(xyzs, _min, _res, _scale_num):
    with tf.variable_scope('get_vidx1d_all_scales_Scale%d'%(_scale_num)):
      xyzmin = tf.reduce_min(xyzs)
      with tf.control_dependencies([tf.greater_equal(xyzmin,0)]):
        xyzs = tf.identity(xyzs)

      def body(vidx_1ds, i, min_i):
        res_i = _res/(tf.cast(tf.pow(2, i), tf.float32))
        vidx_3d = OctreeTf.get_vidx3d(xyzs, min_i, res_i)
        vidx_1d = idx3d_to_1d(vidx_3d)
        min_i = min_i + tf.expand_dims(res_i,0) * tf.cast(vidx_3d,tf.float32)
        i+=1
        vidx_1ds = tf.concat([vidx_1ds, tf.reshape(vidx_1d,[-1,1])], 1)
        return vidx_1ds, i, min_i

      i = tf.constant(0)
      min_i = tf.expand_dims(_min,0)
      vertex_num1 = TfUtil.tshape0(xyzs)
      vidx_1ds = tf.zeros([vertex_num1, 0], tf.int32)
      cond = lambda vidx_1ds,i,min_i: tf.less(i, _scale_num)
      vidx_1ds, i, min_i = tf.while_loop(cond, body, [vidx_1ds, i, min_i],\
                            shape_invariants=[tf.TensorShape([vertex_num1,None]),\
                            tf.TensorShape([]), tf.TensorShape([None,3])])
      vidx_1ds.set_shape([vertex_num1, _scale_num])
      return vidx_1ds

  def get_upper_scale_scope(self, vidx_rawscope_lsc):
    flatvidx_rawidxscopes = [None]*(self._scale_num-1)
    vidx_rawscopes = [None]*(self._scale_num-1)
    vidx_rawscopes.append(vidx_rawscope_lsc)
    for i in range(1, self._scale_num):
      # start from upper scale to 0, save time by progressively reduce
      # vidx_rawscope_lsc
      s = self._scale_num - i -1
      cur_idx_scope = tf.concat([vidx_rawscope_lsc[:,0:s+1], vidx_rawscope_lsc[:,-2:]], -1)
      cur_flat_idx = OctreeTf.get_flat_idx(cur_idx_scope[:,0:-2])
      unique_fidx, oidx, count = tf.unique_with_counts(cur_flat_idx)
      sumcout = tf.cumsum(count) - 1
      unique_idx_scope = tf.gather(cur_idx_scope, sumcout, axis=0)
      new_end_idx_i = unique_idx_scope[:,-1:]
      new_start_idx_i = tf.concat([tf.zeros([1,1], tf.int32), new_end_idx_i[0:-1,:]], 0)
      new_idx_i = tf.concat([new_start_idx_i, new_end_idx_i], -1)

      flatvidx_rawidxscopes[s] = tf.concat([tf.expand_dims(unique_fidx,-1), new_idx_i], -1)

      # save time for next iteration
      vidx_rawscope_lsc = tf.concat([unique_idx_scope[:,0:-2], new_idx_i], -1)
      vidx_rawscopes[s] = vidx_rawscope_lsc
    return flatvidx_rawidxscopes, vidx_rawscopes

  def update_octree_idx_scope(self, vidx_rawscope_lsc):
    # too complicate, give up for now

    if self._scale == self._scale_num-1:
      #************************************************************
      vn, c = TfUtil.get_tensor_shape(vidx_rawscope_lsc)
      with tf.control_dependencies([tf.assert_equal(c,3)]):
        vn = tf.identity(vn)

      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      def body(j):
        vidx_j = vidx_rawscope_lsc[j,0]
        self._nodes[vidx_j]._idx_scope = vidx_rawscope_lsc[j, 1:3]
        return j+1

      i = tf.constant(0)
      cond = tf.less(i, vn)
      i = tf.while_loop(cond, body, [i])

    else:
      #************************************************************
      hist = tf.histogram_fixed_width(vidx_rawscope_lsc[:,0], [0,8], 8)
      start_idx_i = 0
      for i in range(8):
        cond = lambda: tf.greater(hist[i], 0)
        def do_update(start_idx):
          end_idx = start_idx + hist[i]
          sub_vidx_scopes = vidx_rawscope_lsc[start_idx:end_idx,1:]
          self._nodes[i]._idx_scope = tf.stack([sub_vidx_scopes[0,-2], sub_vidx_scopes[-1,-1]])
          self._nodes[i].update_octree_idx_scope(sub_vidx_scopes)
          return end_idx
        def no_op(start_idx):
          return start_idx

        start_idx_i = tf.cond(cond, lambda : do_update(start_idx_i),
                              lambda : no_op(start_idx_i))


    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  @staticmethod
  def get_idx_scope(end_idxs_lastscale):
    tmp = end_idxs_lastscale[:,-1:]
    tmp = tf.concat([tf.zeros([1,1], tf.int32), tmp[0:-1,:]], 0)
    vidx_rawscope_lsc = tf.concat([end_idxs_lastscale[:,0:-1], tmp, end_idxs_lastscale[:,-1:]], -1)
    return vidx_rawscope_lsc

  def get_sort_idx(self, xyzs, vidx_1d, sortedidxs_base):
    '''
    xyzs: [N,3]
    vidx_1d: [N] 0~7
    sorted_idxs_base: [N]  sorted idx of input xyzs
    end_idxs_lastscale: [M, self._scale_num-self._scale] the voxel idx of each scale,
                  and the end idx of sorted data idx at the last scale.
    '''
    sort_idx =  tf.contrib.framework.argsort(vidx_1d, axis=-1)
    if self._scale != 0:
      sortedidxs_base = tf.gather(sortedidxs_base, sort_idx)
    else:
      sortedidxs_base = sort_idx
    hist = tf.histogram_fixed_width(vidx_1d, [0,8], nbins=8)

    def add_one_voxel(start_idx, vidx):
      #print('scale {} vid {} hist: {}'.format(self._scale, vidx, hist[vidx]))
      i3d = tf.constant(idx1d_to_3d_np(vidx), tf.float32)
      min_i = self._min + self._resolution * i3d
      self._nodes[vidx] =  OctreeTf(resolution=self._resolution/2, scale=self._scale+1, min_xyz=min_i)

      end_idx = start_idx + hist[vidx]
      indices_i = sort_idx[start_idx:end_idx]

      if self._scale == self._scale_num-1:
        #sortedidxs_base_i = sortedidxs_base[start_idx:end_idx]
        sortedidxs_base_i = tf.constant([], tf.int32)
        end_idx_lastscale = tf.reshape(end_idx, [1,1])

        #check
        if self._check_optial:
          xyzs_i = tf.gather(xyzs, indices_i, axis=0)
          mini = tf.reduce_min(xyzs_i,0)
          maxi = tf.reduce_max(xyzs_i,0)
          scope_i = maxi - mini
          check_i = tf.reduce_all(tf.less_equal(scope_i, self._resolution))
        #print('scale {} vid {} hist: {}'.format(self._scale, vidx, hist[vidx]))
      else:
        # After sorted, the vertices belong to this sub voxel are together
        # already. xyzs_i belong to vidx voxel only, but it is not sorted inside
        # vidx voxel
        xyzs_i = tf.gather(xyzs, indices_i, axis=0)
        sortedidxs_base_i = sortedidxs_base[start_idx:end_idx]
        sortedidxs_base_i, end_idx_lastscale = self._nodes[vidx].add_point_cloud(xyzs_i, sortedidxs_base_i)
        m = self._scale_num - self._scale - 1
        tmp = tf.concat([tf.zeros([1,m], tf.int32), tf.reshape(start_idx,[1,1])], -1)
        end_idx_lastscale += tmp
      return end_idx, sortedidxs_base_i, end_idx_lastscale

    def no_op(start_idx, vidx):
      #print('scale {} vid {} empty'.format(self._scale, vidx))
      return start_idx, tf.constant([], tf.int32), \
              tf.zeros([0,self._scale_num-self._scale], dtype=tf.int32)

    sorted_idxs_base_new = []
    end_idxs_lastscale = []

    start_idx_i = 0
    for i in range(8):
      not_empty = tf.greater(hist[i], 0)
      start_idx_i, sorted_idx_i, end_idx_lastscale_i = tf.cond(not_empty, lambda: add_one_voxel(start_idx_i, i),
                            lambda: no_op(start_idx_i,i))
      sorted_idxs_base_new.append(sorted_idx_i)
      ni = TfUtil.get_tensor_shape(end_idx_lastscale_i)[0]
      cur_vidx = tf.ones([ni,1], tf.int32) * i
      end_idx_lastscale_i = tf.concat([cur_vidx, end_idx_lastscale_i], -1)
      end_idxs_lastscale.append(end_idx_lastscale_i)

    sorted_idxs_base_new = tf.concat(sorted_idxs_base_new, 0)
    if self._scale == self._scale_num-1:
      sorted_idxs_base_new = sortedidxs_base
    end_idxs_lastscale = tf.concat(end_idxs_lastscale, 0)
    return sorted_idxs_base_new, end_idxs_lastscale


  @staticmethod
  def get_scene_region_name(rawfn):
    region_name = os.path.splitext(os.path.basename(rawfn))[0]
    scene_name = os.path.basename(os.path.dirname(os.path.dirname(rawfn)))
    return scene_name, region_name

  def rawfn_to_octreefn(self,rawfn):
    scale_num = self._scale_num
    resolution = self._resolution
    scene_name, region_name = OctreeTf.get_scene_region_name(rawfn)
    res_str = '_'.join([str(int(d*10)) for d in resolution])
    root = '/DS/Matterport3D/VoxelSort_S%d_%s'%(scale_num, res_str)
    path = os.path.join(root, scene_name)
    if not os.path.exists(path):
      os.makedirs(path)
    fn = os.path.join(path, '{}.pickle'.format(region_name))
    return fn

  @staticmethod
  def get_flat_idx(vidxs):
    sn = TfUtil.get_tensor_shape(vidxs)[1]
    i = 0
    flat_vidx = vidxs[:,i] * pow(8,sn-1)
    for i in range(1, sn):
      flat_vidx += vidxs[:,i] * pow(8, sn-i-1)
    return flat_vidx

  @staticmethod
  def search(base, aim):
    '''
    both base and aim are sorted
    '''
    n = TfUtil.get_tensor_shape(base)[0]
    m = TfUtil.get_tensor_shape(aim)[0]

    def body(i, j, idx):
      catch = tf.logical_or(tf.greater_equal(base[i,0], aim[j]), tf.equal(i,n-1))

      def found(i,j,idx):
        match = tf.cast(tf.equal(base[i,0], aim[j]), tf.int32)
        tmp = (i+1) * match - 1
        idx = tf.concat([idx, tf.reshape(tmp,[1,1])], 0)
        return i+1, j+1, idx
      i,j,idx = tf.cond(catch, lambda: found(i,j,idx), lambda: (i+1,j,idx) )
      return i,j, idx

    i = tf.constant(0)
    j = tf.constant(0)
    idx = tf.zeros([0,1], tf.int32)
    cond = lambda i,j,idx: tf.logical_and( tf.less(i, n), tf.less(j,m))
    i,j,idx = tf.while_loop(cond, body, [i,j,idx],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None,1])])
    return tf.squeeze(idx,1)

  def search_idx_scope(self, flat_vidx, aim_scale):
    with tf.variable_scope('search_idx_scope_Aimscale_%d'%(aim_scale)):
      flatvidx_rawidxscope =  self.flatvidx[aim_scale]
      aim_idx = OctreeTf.search(flatvidx_rawidxscope, flat_vidx)
      check_all_searched =  tf.assert_equal(TfUtil.tshape0(aim_idx), TfUtil.tshape0(flat_vidx),
                                            message="search failed")
      with tf.control_dependencies([check_all_searched]):
        return aim_idx

  def search_neighbours(self, xyzs_sp, aim_scale):
    assert aim_scale < self._scale_num
    with tf.variable_scope('search_neighbours_Scale%d'%(aim_scale)):
      vidx_1ds = OctreeTf.get_vidx1d_all_scales(xyzs_sp, self._min, self._resolution, aim_scale+1)
      flatvidx = OctreeTf.get_flat_idx(vidx_1ds)
      unique_flatvidx, idx_inv = tf.unique(flatvidx)

      uni_aim_idx = self.search_idx_scope(unique_flatvidx, aim_scale)
      aim_idx = tf.gather(uni_aim_idx, idx_inv, 0)

      idxscope_pervs =  tf.gather(self.idxscope_pervs[aim_scale], aim_idx, axis=0)
      return idxscope_pervs

  def recording(self, rawfn):
    sorted_idxs_base = self.sorted_idxs_base.numpy()
    resolution = self._resolution
    scale_num = len(self.flatvidx)
    flatvidx = self.flatvidx
    idxscope_pervs = self.idxscope_pervs
    for i in range(scale_num):
      flatvidx[i] = flatvidx[i].numpy()
      idxscope_pervs[i] = idxscope_pervs[i].numpy()
    fn = self.rawfn_to_octreefn(rawfn)

    import pickle
    with open(fn,"w") as f:
      pickle.dump(sorted_idxs_base, f)
      pickle.dump(flatvidx, f)
      pickle.dump(idxscope_pervs, f)
      pickle.dump(resolution, f)
    print('write ok: {}'.format(fn))

    #self.load(rawfn)

  def read_log(self, rawfn):
    octree_fn = self.rawfn_to_octreefn(rawfn)
    import pickle
    with open(octree_fn, 'r') as f:
      sorted_idxs_base = pickle.load(f)
      flatvidx = pickle.load(f)
      idxscope_pervs = pickle.load(f)
      _resolution = pickle.load(f)
      _min = tf.constant([0,0,0], tf.float32)
      scale_num = len(flatvidx)
      assert scale_num == self._scale_num
    return sorted_idxs_base, flatvidx, idxscope_pervs, _resolution

  def load_record(self, rawfn):
    sorted_idxs_base, flatvidx, idxscope_pervs, _resolution = self.read_log(rawfn)
    vertex_num0 = sorted_idxs_base.shape[0]
    scale_num = len(flatvidx)
    assert scale_num == self._scale_num
    assert np.all(self._resolution == _resolution)
    voxel_nums = [d.shape[0] for d in flatvidx]

    self.sorted_idxs_base = tf.placeholder(tf.int32, [vertex_num0], 'sorted_idxs_base')
    self.flatvidx = []
    self.idxscope_pervs = []
    for s in range(scale_num):
      self.flatvidx.append( tf.placeholder(tf.int32, [voxel_nums[s],1], 'flatvidx_%d'%(s)) )
      self.idxscope_pervs.append( tf.placeholder(tf.int32, [voxel_nums[s], 2], 'idxscope_pervs_%d'%(s)) )

    feed_dict = {self.sorted_idxs_base: sorted_idxs_base}
    for s in range(scale_num):
      feed_dict[self.flatvidx[s]] = flatvidx[s]
      feed_dict[self.idxscope_pervs[s]] = idxscope_pervs[s]
    return feed_dict

  def load_eager(self, rawfn):
    octree_fn = self.rawfn_to_octreefn(rawfn)
    import pickle
    with open(octree_fn, 'r') as f:
      self.sorted_idxs_base = pickle.load(f)
      self.flatvidx = pickle.load(f)
      self.idxscope_pervs = pickle.load(f)
      self._resolution = pickle.load(f)
      self._min = tf.constant([0,0,0], tf.float32)
      scale_num = len(self.flatvidx)
      assert scale_num == self._scale_num

  def test_search_inv(self, xyzs, test_n, aim_scale, ply=False):
    from utils import ply_util
    assert aim_scale < self._scale_num
    xyzs = check_xyz_normed(xyzs)
    xyzs = tf.gather(xyzs, self.sorted_idxs_base, 0)

    vn0 = TfUtil.tshape0(xyzs)
    sp_idx = tf.random_shuffle(tf.range(vn0))[0:test_n]
    sp_idx = tf.contrib.framework.sort(sp_idx)
    xyzs_sp = tf.gather(xyzs, sp_idx, 0)

    idx_scopes = self.search_neighbours(xyzs_sp, aim_scale)
    if not ply:
      return idx_scopes

    for i in range(5):
      ply_dir = '/tmp/octree_plys'
      fn = os.path.join(ply_dir, 'base_%d.ply'%(i))
      xyz_i = xyzs_sp[i:i+1,:]
      ply_util.create_ply(xyz_i, fn)

      fn = os.path.join(ply_dir, 'neighbors_%d.ply'%(i))
      xyzs_neig_i = xyzs[idx_scopes[i,0]:idx_scopes[i,1], :]
      min_i = tf.reduce_min(xyzs_neig_i, 0)
      max_i = tf.reduce_max(xyzs_neig_i, 0)
      scope_i = max_i - min_i
      c0 = tf.reduce_all(tf.greater_equal(xyz_i - min_i,0))
      c1 = tf.reduce_all(tf.greater_equal(max_i - xyz_i,0))
      check = tf.assert_equal(tf.logical_and(c0, c1), True)

      ply_util.create_ply(xyzs_neig_i, fn)
    return idx_scopes

def norm_xyzs(xyzs):
  float_align = fal = 0.01
  min_xyz = tf.reduce_min(xyzs, 0)
  min_xyz = tf.floor(min_xyz/fal)*fal
  max_xyz = tf.reduce_max(xyzs, 0)
  scope = max_xyz - min_xyz
  res = scope/2
  res = tf.ceil(res/fal)*fal

  xyzs = xyzs - min_xyz
  return xyzs

def check_xyz_normed(xyzs):
  xyz_min = tf.reduce_min(xyzs)
  check = tf.assert_greater_equal(xyz_min, 0.0, message='xyz should be normed at first')
  with tf.control_dependencies([check]):
    xyzs = tf.identity(xyzs)
  return xyzs

def gen_octree(rawfn, resolution):
  raw_datas = parse_ply_file(rawfn)
  xyzs = norm_xyzs(raw_datas['xyz'])

  resolution = np.array([3.2,3.2,3.2])

  octree_tf = OctreeTf(resolution, 0)
  octree_tf.add_point_cloud(xyzs, record=True, rawfn=rawfn)

def main_gen_octree():
  tf.enable_eager_execution()
  scene_name = '17DRP5sb8fy'
  region_name = 'region0'
  fn_glob = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans/{}/{}/region_segmentations/{}.ply'.\
                              format(scene_name, scene_name, region_name)
  rawfns = glob.glob(fn_glob)

  resolution = np.array([3.2,3.2,3.2])
  for rawfn in rawfns:
    print('start gen octree {}'.format(rawfn))
    t0 = time.time()
    gen_octree(rawfn, resolution)
    t = time.time() - t0
    print('use {} sec'.format(t))

  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass


def main_read_octree_eager():
  tf.enable_eager_execution()
  resolution = np.array([3.2,3.2,3.2])
  scene_name = '17DRP5sb8fy'
  region_name = 'region0'
  rawfn = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans/{}/{}/region_segmentations/{}.ply'.\
                              format(scene_name, scene_name, region_name)
  raw_datas = parse_ply_file(rawfn)
  xyzs = norm_xyzs(raw_datas['xyz'])

  octree_tf = OctreeTf(resolution, 0)
  octree_tf.load_eager(rawfn)
  t0 = time.time()
  aim_scale = 5
  idx_scopes = octree_tf.test_search_inv(xyzs, 200*1000, aim_scale)
  t = time.time() - t0
  print('read ok, t={}'.format(t))

def main_read_octree_graph():
  resolution = np.array([3.2,3.2,3.2])
  scene_name = '17DRP5sb8fy'
  region_name = 'region0'
  rawfn = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans/{}/{}/region_segmentations/{}.ply'.\
                              format(scene_name, scene_name, region_name)
  raw_datas = parse_ply_file(rawfn)
  raw_xyzs = raw_datas['xyz']
  vertex_num0 = raw_xyzs.shape[0]

  with tf.Graph().as_default():
    octree_tf = OctreeTf(resolution, 0)
    feed_dict = octree_tf.load_record(rawfn)
    raw_xyzs_pl = tf.placeholder(tf.float32, [vertex_num0,3])
    raw_xyzs_normed = norm_xyzs(raw_xyzs_pl)
    feed_dict[raw_xyzs_pl] = raw_xyzs

    aim_scale = 5
    idx_scopes = octree_tf.test_search_inv(raw_xyzs_normed, 200*1000, aim_scale)

    with tf.Session() as sess:
      t0 = time.time()
      idx_scopes_v = sess.run(idx_scopes, feed_dict=feed_dict)
      t = time.time() - t0
      print('read ok, t={}'.format(t))
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass


if __name__ == '__main__':
  main_gen_octree()
  #main_read_octree_eager()
  #main_read_octree_graph()
  pass

