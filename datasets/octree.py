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
  _max_scale_num = 1
  def __init__(self, scale, resolution, min_xyz=[0,0,0]):
    self._scale = scale
    self._min = min_xyz
    self._resolution = resolution

    self._nodes = [None]*8
    self._value = None

  def search(self, xyzs):
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


  def add_point_cloud(self, xyzs):
    assert TfUtil.tsize(xyzs) == 2

    xyzs = self.norm_xyzs(xyzs)

    idx =  (xyzs - self._min)/self._resolution
    c0 = tf.assert_greater_equal(tf.reduce_min(idx), 0.0)
    c1 = tf.assert_less(tf.reduce_max(idx), 8.0)
    with tf.control_dependencies([c0, c1]):
      idx = tf.cast(tf.math.floor(idx), tf.int32)
    idx_1d = idx3d_to_1d(idx)
    end_sort_idxs = self.get_sort_idx(xyzs, idx_1d)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  def get_sort_idx(self, xyzs, idx_1d):
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
        end_sort_idx = indices_i
        #print('scale {} vid {} hist: {}'.format(self._scale, i, hist[i]))
      else:
        end_sort_idx = self._nodes[i].add_point_cloud(xyzs_i)
      return end_idx, end_sort_idx

    def no_op(start_idx):
      #print('scale {} vid {} empty'.format(self._scale, i))
      return start_idx, tf.constant([], tf.int32)

    end_sort_idxs = []
    start_idx_i = 0
    for i in range(8):
      not_empty = tf.greater(hist[i], 0)
      start_idx_i, end_sort_idx_i = tf.cond(not_empty, lambda: add_one_voxel(start_idx_i),
                            lambda: no_op(start_idx_i))
      end_sort_idxs.append(end_sort_idx_i)

    end_sort_idxs = tf.concat(end_sort_idxs, 0)
    return end_sort_idxs

def main():
  tf.enable_eager_execution()
  from MATTERPORT_util.MATTERPORT_util import parse_ply_file, parse_ply_vertex_semantic
  rawfn = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans/17DRP5sb8fy/17DRP5sb8fy/region_segmentations/region0.ply'
  raw_datas = parse_ply_file(rawfn)

  octree_tf =  OctreeTf(0, tf.constant([2.6,2.6,2.6]))
  octree_tf.add_point_cloud(raw_datas['xyz'])
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

if __name__ == '__main__':
  main()
