# xyzs Oct 2018

import os, glob,sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from utils.tf_util import TfUtil

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
  _max_scale_num = 2
  def __init__(self, scale, resolution, min_xyz=[0,0,0]):
    self._scale = scale
    self._min = min_xyz
    self._resolution = resolution

    self._nodes = [None]*8
    # only last scale
    #self._idx_scope = None


  def norm_xyzs(self, xyzs):
    if self._scale >0:
      return xyzs
    min_xyz = tf.reduce_min(xyzs, 0)
    min_xyz = tf.floor(min_xyz/self._fal)*self._fal
    max_xyz = tf.reduce_max(xyzs, 0)
    scope = max_xyz - min_xyz
    res = scope/2
    res = tf.ceil(res/self._fal)*self._fal

    xyzs = xyzs - min_xyz
    return xyzs

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
  def get_vidx1d_all_scales(xyzs, _min, _res, _max_scale_num):
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
    min_i = _min
    vidx_1ds = tf.zeros([TfUtil.tshape0(xyzs), 0], tf.int32)
    cond = lambda vidx_1ds,i,min_i: tf.less(i, _max_scale_num)
    vidx_1ds, i, min_i = tf.while_loop(cond, body, [vidx_1ds, i, min_i])
    return vidx_1ds

  def add_point_cloud(self, xyzs, sortedidxs_base=None, record=False, rawfn=None):
    assert TfUtil.tsize(xyzs) == 2
    xyzs = self.norm_xyzs(xyzs)

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
      self.rawidx_scopes = [d[:,-2:] for d in vidx_rawscopes]

      self.gen_voxel_ply(xyzs)
      self.record(sorted_idxs_base, vidx_rawscope_lsc, rawfn)
    return sorted_idxs_base, end_idxs_lastscale

  def get_upper_scale_scope(self, vidx_rawscope_lsc):
    flatvidx_rawidxscopes = [None]*(self._max_scale_num-1)
    vidx_rawscopes = [None]*(self._max_scale_num-1)
    vidx_rawscopes.append(vidx_rawscope_lsc)
    for i in range(1, self._max_scale_num):
      # start from upper scale to 0, save time by progressively reduce
      # vidx_rawscope_lsc
      s = self._max_scale_num - i -1
      cur_idx_scope = tf.concat([vidx_rawscope_lsc[:,0:s+1], vidx_rawscope_lsc[:,-2:]], -1)
      cur_flat_idx = OctreeTf.get_flat_idx(cur_idx_scope[:,0:-2])
      unique_fidx, oidx, count = tf.unique_with_counts(cur_flat_idx)
      sumcout = tf.cumsum(count) - 1
      unique_idx_scope = tf.gather(cur_idx_scope, sumcout, axis=0)
      new_end_idx_i = unique_idx_scope[:,-1:]
      new_start_idx_i = tf.concat([tf.zeros([1,1], tf.int32), new_end_idx_i[0:-1,:]], 0)
      new_idx_i = tf.concat([new_start_idx_i, new_end_idx_i], -1)
      flatvidx_rawidxscopes_cur = tf.concat([tf.expand_dims(unique_fidx,-1), new_idx_i], -1)
      flatvidx_rawidxscopes[s] = flatvidx_rawidxscopes_cur
      vidx_rawscopes[s] = unique_idx_scope

      # save time for next iteration
      vidx_rawscope_lsc = tf.concat([unique_idx_scope[:,0:-2], new_idx_i], -1)
    return flatvidx_rawidxscopes, vidx_rawscopes

  def update_octree_idx_scope(self, vidx_rawscope_lsc):
    # too complicate, give up for now

    if self._scale == self._max_scale_num-1:
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
    end_idxs_lastscale: [M, self._max_scale_num-self._scale] the voxel idx of each scale,
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
      self._nodes[vidx] =  OctreeTf(self._scale+1, self._resolution/2, min_i)

      end_idx = start_idx + hist[vidx]
      indices_i = sort_idx[start_idx:end_idx]

      if self._scale == self._max_scale_num-1:
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
        # already
        xyzs_i = tf.gather(xyzs, indices_i, axis=0)
        sortedidxs_base_i = sortedidxs_base[start_idx:end_idx]
        sortedidxs_base_i, end_idx_lastscale = self._nodes[vidx].add_point_cloud(xyzs_i, sortedidxs_base_i)
        m = self._max_scale_num - self._scale - 1
        tmp = tf.concat([tf.zeros([1,m], tf.int32), tf.reshape(start_idx,[1,1])], -1)
        end_idx_lastscale += tmp
      return end_idx, sortedidxs_base_i, end_idx_lastscale

    def no_op(start_idx, vidx):
      #print('scale {} vid {} empty'.format(self._scale, vidx))
      return start_idx, tf.constant([], tf.int32), \
              tf.zeros([0,self._max_scale_num-self._scale], dtype=tf.int32)

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
    if self._scale == self._max_scale_num-1:
      sorted_idxs_base_new = sortedidxs_base
    end_idxs_lastscale = tf.concat(end_idxs_lastscale, 0)
    return sorted_idxs_base_new, end_idxs_lastscale


  def record(self, sorted_idxs_base, vidx_rawscope_lsc, rawfn):
    sorted_idxs_base = sorted_idxs_base.numpy()
    vidx_rawscope_lsc = vidx_rawscope_lsc.numpy()
    scale_num = vidx_rawscope_lsc.shape[1] - 2

    region_name = os.path.splitext(os.path.basename(rawfn))[0]
    scene_name = os.path.basename(os.path.dirname(os.path.dirname(rawfn)))
    root = '/DS/Matterport3D/VoxelSort_S%d'%(scale_num)
    path = os.path.join(root, scene_name)
    if not os.path.exists(path):
      os.makedirs(path)

    dtype = 'bin'
    dtype = 'text'

    if dtype == 'bin':
      fn = os.path.join(path, '{}_sorted_idxs_base.bin'.format(region_name))
      sorted_idxs_base.tofile(fn)

      fn = os.path.join(path, '{}_vidx_rawscope_lsc.bin'.format(region_name))
      vidx_rawscope_lsc.tofile(fn)

      #eil = np.fromfile(fn, dtype=np.int32)
      #eil = np.reshape(eil, [-1, scale_num])

    elif dtype == 'text':
      fn = os.path.join(path, '{}_sorted_idxs_base.txt'.format(region_name))
      sorted_idxs_base.tofile(fn, sep="\n", format="%d")

      fn = os.path.join(path, '{}_vidx_rawscope_lsc.txt'.format(region_name))
      vidx_rawscope_lsc.tofile(fn, sep="\n", format="%d")

      #eil = np.fromfile(fn, dtype=np.int32, sep="\t")

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
    i,j,idx = tf.while_loop(cond, body, [i,j,idx])
    return tf.squeeze(idx,1)

  def search_idx_scope(self, flat_vidx, aim_scale):
    flatvidx_rawidxscope =  self.flatvidx[aim_scale]
    aim_idx = OctreeTf.search(flatvidx_rawidxscope, flat_vidx)
    check_all_searched =  tf.assert_equal(TfUtil.tshape0(aim_idx), TfUtil.tshape0(flat_vidx),
                                          message="search failed")
    with tf.control_dependencies([check_all_searched]):
      return aim_idx

  def search_neighbours(self, xyzs_sp, aim_scale):
    assert aim_scale < self._max_scale_num
    vidx_1ds = OctreeTf.get_vidx1d_all_scales(xyzs_sp, self._min, self._resolution, aim_scale+1)
    flatvidx = OctreeTf.get_flat_idx(vidx_1ds)
    unique_flatvidx, idx_inv = tf.unique(flatvidx)

    uni_aim_idx = self.search_idx_scope(unique_flatvidx, aim_scale)
    aim_idx = tf.gather(uni_aim_idx, idx_inv, 0)

    rawidx_scopes =  tf.gather(self.rawidx_scopes[aim_scale], aim_idx, 0)
    return rawidx_scopes

  def gen_voxel_ply(self, xyzs):
    from utils import ply_util
    xyzs = tf.gather(xyzs, self.sorted_idxs_base, 0)

    vn0 = TfUtil.tshape0(xyzs)
    sp_idx = tf.random_shuffle(tf.range(vn0))[0:10]
    sp_idx = tf.contrib.framework.sort(sp_idx)
    xyzs_sp = tf.gather(xyzs, sp_idx, 0)

    idx_scopes = self.search_neighbours(xyzs_sp, 1)


    for i in range(10):
      ply_dir = '/tmp/octree_plys'
      fn = os.path.join(ply_dir, 'base_%d.ply'%(i))
      ply_util.create_ply(xyzs_sp[i:i+1,:], fn)

      fn = os.path.join(ply_dir, 'neighbors_%d.ply'%(i))
      xyzs_neig_i = xyzs[idx_scopes[i,0]:idx_scopes[i,1], :]
      ply_util.create_ply(xyzs_neig_i, fn)

    import pdb; pdb.set_trace()  # XXX BREAKPOINT

    pass



def main():
  tf.enable_eager_execution()
  from MATTERPORT_util.MATTERPORT_util import parse_ply_file, parse_ply_vertex_semantic
  rawfn = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans/17DRP5sb8fy/17DRP5sb8fy/region_segmentations/region0.ply'
  raw_datas = parse_ply_file(rawfn)

  octree_tf =  OctreeTf(0, tf.constant([3.2,3.2,3.2]))
  octree_tf.add_point_cloud(raw_datas['xyz'], record=True, rawfn=rawfn)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

if __name__ == '__main__':
  main()
