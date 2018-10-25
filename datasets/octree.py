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
  idx_1d = idx_3d[:,0] + idx_3d[:,1]*2 + idx_3d[:,2]*4
  return idx_1d

def idx1d_to_3d(idx_1d):
  idx3d_2 = tf.expand_dims(tf.floormod(idx_1d, 2), -1)
  tmp = idx_1d / 2
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
  _fal = 0.01
  _max_scale_num = 4
  def __init__(self, scale, resolution, min_xyz=[0,0,0]):
    self._scale = scale
    self._min = min_xyz
    self._resolution = resolution

    self._nodes = [None]*8
    # only last scale
    #self._idx_scope = None

  def search(self, xyzs):
    idx_1d = self.get_idx1d(xyzs)
    self._nodes[idx_1d].search(xyzs)

    idx = (xyzs - self._min) / self._resolution
    node = self._nodes[idx]
    if node is None:
      return None
    elif node is OctreeTf:
      return node.search(xyzs)
    else:
      return node._value

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

  def get_idx1d(self, xyzs):

    idx =  (xyzs - self._min)/self._resolution
    c0 = tf.assert_greater_equal(tf.reduce_min(idx), 0.0)
    c1 = tf.assert_less(tf.reduce_max(idx), 8.0)
    with tf.control_dependencies([c0, c1]):
      idx = tf.cast(tf.math.floor(idx), tf.int32)
    idx_1d = idx3d_to_1d(idx)
    return idx_1d


  def add_point_cloud(self, xyzs, record=False, rawfn=None):
    assert TfUtil.tsize(xyzs) == 2
    xyzs = self.norm_xyzs(xyzs)

    idx_1d = self.get_idx1d(xyzs)
    voxel_sort_idxs, end_idxs_lastscale = self.get_sort_idx(xyzs, idx_1d)
    if self._scale==0:
      self.voxel_sort_idxs = voxel_sort_idxs
      idx_scope_lsc = OctreeTf.get_idx_scope(end_idxs_lastscale)
      flatidx_scopes = self.get_upper_scale_scope(idx_scope_lsc)
      flat_vidx_lsc = OctreeTf.get_flat_idx(idx_scope_lsc[:,0:-2])
      flatidx_scope_lsc = tf.concat([tf.expand_dims(flat_vidx_lsc,-1), idx_scope_lsc[:,-2:]], -1)
      flatidx_scopes.append(flatidx_scope_lsc)
      self.flatidx_scopes = flatidx_scopes
      self.gen_voxel_ply(xyzs)
      self.record(voxel_sort_idxs, idx_scope_lsc, rawfn)
    return voxel_sort_idxs, end_idxs_lastscale

  def get_upper_scale_scope(self, idx_scope_lsc):
    flatidx_scopes = [None]*(self._max_scale_num-1)
    for i in range(1, self._max_scale_num):
      # start from upper scale to 0, save time by progressively reduce
      # idx_scope_lsc
      s = self._max_scale_num - i -1
      cur_idx_scope = tf.concat([idx_scope_lsc[:,0:s+1], idx_scope_lsc[:,-2:]], -1)
      cur_flat_idx = OctreeTf.get_flat_idx(cur_idx_scope[:,0:-2])
      unique_fidx, oidx, count = tf.unique_with_counts(cur_flat_idx)
      sumcout = tf.cumsum(count) - 1
      unique_idx_scope = tf.gather(cur_idx_scope, sumcout, axis=0)
      new_end_idx_i = unique_idx_scope[:,-1:]
      new_start_idx_i = tf.concat([tf.zeros([1,1], tf.int32), new_end_idx_i[0:-1,:]], 0)
      new_idx_i = tf.concat([new_start_idx_i, new_end_idx_i], -1)
      flatidx_scope_cur = tf.concat([tf.expand_dims(unique_fidx,-1), new_idx_i], -1)
      flatidx_scopes[s] = flatidx_scope_cur

      # save time for next iteration
      idx_scope_lsc = tf.concat([unique_idx_scope[:,0:-2], new_idx_i], -1)

    return flatidx_scopes

  def update_octree_idx_scope(self, idx_scope_lsc):
    # too complicate, give up for now

    if self._scale == self._max_scale_num-1:
      #************************************************************
      vn, c = TfUtil.get_tensor_shape(idx_scope_lsc)
      with tf.control_dependencies([tf.assert_equal(c,3)]):
        vn = tf.identity(vn)

      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      def body(j):
        vidx_j = idx_scope_lsc[j,0]
        self._nodes[vidx_j]._idx_scope = idx_scope_lsc[j, 1:3]
        return j+1

      i = tf.constant(0)
      cond = tf.less(i, vn)
      i = tf.while_loop(cond, body, [i])

    else:
      #************************************************************
      hist = tf.histogram_fixed_width(idx_scope_lsc[:,0], [0,8], 8)
      start_idx_i = 0
      for i in range(8):
        cond = lambda: tf.greater(hist[i], 0)
        def do_update(start_idx):
          end_idx = start_idx + hist[i]
          sub_vidx_scopes = idx_scope_lsc[start_idx:end_idx,1:]
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
    idx_scope_lsc = tf.concat([end_idxs_lastscale[:,0:-1], tmp, end_idxs_lastscale[:,-1:]], -1)
    return idx_scope_lsc

  def get_sort_idx(self, xyzs, idx_1d):
    '''
    xyzs: [N,3]
    idx_1d: [N] 0~7
    voxel_sort_idxs: [N]  sorted idx of input xyzs
    end_idxs_lastscale: [M, self._max_scale_num-self._scale] the voxel idx of each scale,
                  and the end idx of sorted data idx at the last scale.
    '''
    sort_idx =  tf.contrib.framework.argsort(idx_1d, axis=-1)
    hist = tf.histogram_fixed_width(idx_1d, [0,8], nbins=8)

    def add_one_voxel(start_idx):
      #print('scale {} vid {} hist: {}'.format(self._scale, i, hist[i]))
      i3d = tf.constant(idx1d_to_3d_np(i), tf.float32)
      min_i = self._min + self._resolution * i3d
      self._nodes[i] =  OctreeTf(self._scale+1, self._resolution/2, min_i)

      end_idx = start_idx + hist[i]
      indices_i = sort_idx[start_idx:end_idx]
      xyzs_i = tf.gather(xyzs, indices_i, axis=0)

      if self._scale == self._max_scale_num-1:
        voxel_sort_idx = indices_i
        end_idx_lastscale = tf.reshape(end_idx, [1,1])
        #print('scale {} vid {} hist: {}'.format(self._scale, i, hist[i]))
      else:
        voxel_sort_idx_nextscale, end_idx_lastscale = self._nodes[i].add_point_cloud(xyzs_i)
        # Both need to add the start idx of this voxel
        voxel_sort_idx = voxel_sort_idx_nextscale + start_idx
        m = self._max_scale_num - self._scale - 1
        tmp = tf.concat([tf.zeros([1,m], tf.int32), tf.reshape(start_idx,[1,1])], -1)
        end_idx_lastscale += tmp
      return end_idx, voxel_sort_idx, end_idx_lastscale

    def no_op(start_idx):
      #print('scale {} vid {} empty'.format(self._scale, i))
      return start_idx, tf.constant([], tf.int32), \
              tf.zeros([0,self._max_scale_num-self._scale], dtype=tf.int32)

    voxel_sort_idxs = []
    end_idxs_lastscale = []

    start_idx_i = 0
    for i in range(8):
      not_empty = tf.greater(hist[i], 0)
      start_idx_i, voxel_sort_idx_i, end_idx_lastscale_i = tf.cond(not_empty, lambda: add_one_voxel(start_idx_i),
                            lambda: no_op(start_idx_i))
      voxel_sort_idxs.append(voxel_sort_idx_i)
      ni = TfUtil.get_tensor_shape(end_idx_lastscale_i)[0]
      cur_vidx = tf.ones([ni,1], tf.int32) * i
      end_idx_lastscale_i = tf.concat([cur_vidx, end_idx_lastscale_i], -1)
      end_idxs_lastscale.append(end_idx_lastscale_i)

    voxel_sort_idxs = tf.concat(voxel_sort_idxs, 0)
    end_idxs_lastscale = tf.concat(end_idxs_lastscale, 0)
    return voxel_sort_idxs, end_idxs_lastscale


  def record(self, voxel_sort_idxs, idx_scope_lsc, rawfn):
    voxel_sort_idxs = voxel_sort_idxs.numpy()
    idx_scope_lsc = idx_scope_lsc.numpy()
    scale_num = idx_scope_lsc.shape[1] - 2

    region_name = os.path.splitext(os.path.basename(rawfn))[0]
    scene_name = os.path.basename(os.path.dirname(os.path.dirname(rawfn)))
    root = '/DS/Matterport3D/VoxelSort_S%d'%(scale_num)
    path = os.path.join(root, scene_name)
    if not os.path.exists(path):
      os.makedirs(path)

    dtype = 'bin'
    dtype = 'text'

    if dtype == 'bin':
      fn = os.path.join(path, '{}_voxel_sort_idxs.bin'.format(region_name))
      voxel_sort_idxs.tofile(fn)

      fn = os.path.join(path, '{}_idx_scope_lsc.bin'.format(region_name))
      idx_scope_lsc.tofile(fn)

      #eil = np.fromfile(fn, dtype=np.int32)
      #eil = np.reshape(eil, [-1, scale_num])

    elif dtype == 'text':
      fn = os.path.join(path, '{}_voxel_sort_idxs.txt'.format(region_name))
      voxel_sort_idxs.tofile(fn, sep="\n", format="%d")

      fn = os.path.join(path, '{}_idx_scope_lsc.txt'.format(region_name))
      idx_scope_lsc.tofile(fn, sep="\n", format="%d")

      #eil = np.fromfile(fn, dtype=np.int32, sep="\t")

  @staticmethod
  def get_flat_idx(vidx):
    sn = TfUtil.get_tensor_shape(vidx)[1]
    i = 0
    flat_vidx = vidx[:,i] * pow(8,sn-1)
    for i in range(1, sn):
      flat_vidx += vidx[:,i] * pow(8, sn-i-1)
    return flat_vidx

  def get_voxel_idx_scopes(self, vidxs):
    self._nodes[1]._nodes[1]._value
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  def gen_voxel_ply(self, xyzs0):
    xyzs = tf.gather(xyzs0, self.voxel_sort_idxs, 0)
    vidxs = tf.constant( [[1,1], [2,4]], tf.int32)
    self.get_voxel_idx_scopes(vidxs)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass



def main():
  tf.enable_eager_execution()
  from MATTERPORT_util.MATTERPORT_util import parse_ply_file, parse_ply_vertex_semantic
  rawfn = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans/17DRP5sb8fy/17DRP5sb8fy/region_segmentations/region0.ply'
  raw_datas = parse_ply_file(rawfn)

  octree_tf =  OctreeTf(0, tf.constant([2.6,2.6,2.6]))
  octree_tf.add_point_cloud(raw_datas['xyz'], record=True, rawfn=rawfn)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

if __name__ == '__main__':
  main()
