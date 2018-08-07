# July 2018 xyz
import tensorflow as tf
import glob, os, sys
import numpy as np
from utils.dataset_utils import parse_pl_record
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
from utils.ply_util import create_ply_dset, draw_blocks_by_bottom_center
import time

DEBUG = False

def get_data_shapes_from_tfrecord(filenames):
  _DATA_PARAS = {}
  batch_size = 1
  is_training = False

  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        lambda value: parse_pl_record(value, is_training),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=False))
    iterator = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
      features, label = sess.run(iterator)

      for key in features:
        _DATA_PARAS[key] = features[key][0].shape
        print('{}:{}'.format(key, _DATA_PARAS[key]))
      points_raw = features['points'][0]
      print('\n\nget shape from tfrecord OK:\n %s\n\n'%(_DATA_PARAS))
      #print('points', features['points'][0,0:5,:])
  return _DATA_PARAS


def block_id_to_block_index(block_id, block_size):
  '''
  block_id shape is flexible
  block_size:(3)
  '''
  tmp0 = block_size[0] * block_size[1]
  block_index_2 = tf.expand_dims(tf.floordiv(block_id, tmp0),-1)
  tmp1 = tf.floormod(block_id, tmp0)
  block_index_1 = tf.expand_dims(tf.floordiv(tmp1, block_size[0]),-1)
  block_index_0 = tf.expand_dims(tf.floormod(tmp1, block_size[0]),-1)
  block_index = tf.concat([block_index_0, block_index_1, block_index_2], -1)
  return block_index


def block_index_to_block_id(block_index, block_size):
  '''
  block_index: (n0,n1,3)
  block_size:(3)

  block_id: (n0,n1,1)
  '''
  assert tf.shape(block_index).shape[0].value == 3
  block_id = block_index[:,:,2] * block_size[1] * block_size[0] + \
              block_index[:,:,1] * block_size[0] + block_index[:,:,0]

  is_check = True
  if is_check:
    block_index_C = block_id_to_block_index(block_id, block_size)
    check_bi = tf.assert_equal(block_index, block_index_C)
    with tf.control_dependencies([check_bi]):
      block_id = tf.identity(block_id)
  return block_id


def add_permutation_combination_template(A, B):
  '''
  Utilize tf broadcast: https://stackoverflow.com/questions/43534057/evaluate-all-pair-combinations-of-rows-of-two-tensors-in-tensorflow/43542926
  A:(a,d)
  B:(b,d)
  '''
  A_ = tf.expand_dims(A,0)
  B = tf.expand_dims(B,1)


def permutation_combination_2D(up_bound_2d, low_bound_2d):
  '''
  low_bound_2d: (2)
  up_bound_2d: (2)
  '''
  pairs = []
  for i in range(2):
    zeros = tf.zeros([up_bound_2d[i] - low_bound_2d[i],1], tf.int32)
    pairs_i = tf.reshape(tf.range(low_bound_2d[i], up_bound_2d[i], 1), [-1,1])
    if i==1:
      pairs_i = tf.concat([zeros, pairs_i], 1)
      pairs_i = tf.expand_dims(pairs_i, 0)
    else:
      pairs_i = tf.concat([pairs_i, zeros], 1)
      pairs_i = tf.expand_dims(pairs_i, 1)
    pairs.append(pairs_i)

  pairs_2d = tf.reshape(pairs[0] + pairs[1], [-1,2])
  return pairs_2d


def permutation_combination_3D(up_bound_3d, low_bound_3d=[0,0,0]):
  '''
  low_bound_3d: (3)
  up_bound_3d: (3)
  '''
  pairs_2d = permutation_combination_2D(up_bound_3d[0:2], low_bound_3d[0:2])
  zeros = tf.zeros([tf.shape(pairs_2d)[0],1], tf.int32)
  pairs_2d = tf.concat([pairs_2d, zeros], -1)

  pairs_3th = tf.reshape(tf.range(low_bound_3d[2], up_bound_3d[2], 1), [-1,1])
  zeros = tf.zeros([up_bound_3d[2] - low_bound_3d[2],2], tf.int32)
  pairs_3th = tf.concat([zeros, pairs_3th], -1)

  pairs_2d = tf.expand_dims(pairs_2d, 0)
  pairs_3th = tf.expand_dims(pairs_3th, 1)
  pairs_3d = tf.reshape(pairs_2d + pairs_3th, [-1,3])
  return pairs_3d


class BlockGroupSampling():
  def __init__(self, sg_settings ):
    '''
    sg_settings:
      width = [1,1,1]
      stride = [1,1,1]
      nblock = 480
      npoint_per_block = 32
      max_nblock = 1000
    '''
    self._num_scale = len(sg_settings['width'])
    self._widths = np.array(sg_settings['width'], dtype=np.float32)
    self._strides = np.array(sg_settings['stride'], dtype=np.float32)
    self._nblocks = sg_settings['nblock']
    self._padding_rate = [0.5, 0.7, 0.8, 0.8]
    self._npoint_per_blocks = sg_settings['npoint_per_block']
    # cut all the block with too few points
    self._np_perb_min_includes = sg_settings['np_perb_min_include']
    assert self._widths.shape[1] == self._strides.shape[1] == 3
    self._nblock_buf_maxs = sg_settings['max_nblock']
    self._empty_point_index = 'first' # -1(GPU only) or 'first'
    self._block_pos = sg_settings['block_pos']
    assert self._block_pos=='mean' or self._block_pos=='center'

    # record grouping parameters
    self._record_samplings = sg_settings['record']
    self.samplings = []
    if self._record_samplings:
      for i in range(self._num_scale):
        self.samplings.append({})
        self.samplings[i]['nframes'] = tf.constant(0, tf.int64)
        self.samplings[i]['t_per_frame_'] = tf.constant(0, tf.float64)
        self.samplings[i]['nblock_valid_'] = tf.constant(0, tf.int64)
        self.samplings[i]['nblock_invalid_'] = tf.constant(0, tf.int64)
        self.samplings[i]['ave_np_perb_'] = tf.constant(0, tf.int64)
        self.samplings[i]['std_np_perb_'] = tf.constant(0, tf.int64)
        #self.samplings[i]['valid_grouped_npoint_'] = tf.constant(0, tf.int64)
        self.samplings[i]['nmissing_perb_'] = tf.constant(0, tf.int64)
        self.samplings[i]['nempty_perb_'] = tf.constant(0, tf.int64)
        self.samplings[i]['ngp_edge_block_'] = tf.constant(0, tf.int64)
        self.samplings[i]['ngp_out_block_'] = tf.constant(0, tf.int64)
        self.samplings[i]['npoint_grouped_'] = tf.constant(0, tf.int64)


    # debug flags
    self._shuffle = True
    self._use_less_points_block_when_not_enough = True
    # Theoretically, num_block_per_point should be same for all. This is 0.
    self._check_nblock_per_points_same = True # (optional)
    if self._check_nblock_per_points_same:
      #self._n_maxerr_same_nb_perp = 0
      self._n_maxerr_same_nb_perps = [2, 2000, 1000]

    self._check_grouped_xyz_inblock = True

    self._debug_only_blocks_few_points = False
    self.debugs = {}
    self._gen_ply = sg_settings['gen_ply']


  def show_settings(self):
    items = ['_width', '_stride', '_npoint_per_block', '_nblock', '_padding_rate',
              '_np_perb_min_include','_n_maxerr_same_nb_perp', '_shuffle']
    print('\n\nsettings:\n')
    for item in items:
      if hasattr(self, item):
        print('{}:{}'.format(item, getattr(self,item)))
    print('\n\n')


  def show_samplings_np_multi_scale(self, samplings_np_ms):
    for s in range(len(samplings_np_ms)):
      samplings_np = samplings_np_ms[s]
      self.show_samplings_np(samplings_np,s)


  def show_samplings_np(self, samplings_np, scale):
    t = samplings_np['nframes']
    assert t>0
    def summary_str(item, real, config):
      real = real/t
      return '{}: {}/{}, {:.3f}\n'.format(item, real, config,
                      1.0*tf.cast(real,tf.float32)/tf.cast(config,tf.float32))
    summary = '\n'
    summary += 'scale={} t={}  Real / Config\n\n'.format(scale, t)
    summary += summary_str('nblock_valid', samplings_np['nblock_valid_'], self._nblocks[scale])
    summary += 'nblock_invalid(few points):{}, np_perb_min_include:{}\n'.format(
                  samplings_np['nblock_invalid_'], self._np_perb_min_includes[scale])
    summary += 'ngp_edge_block:{}  <- padding rate:{}\n'.format(
      samplings_np['ngp_edge_block_'], self._padding_rate[scale])
    summary += 'ngp_out_block:{} npoint_grouped:{}\n'.format(
      samplings_np['ngp_out_block_'],  samplings_np['npoint_grouped_'])
    summary += summary_str('ave np per block', samplings_np['ave_np_perb_'], self._npoint_per_blocks[scale])
    summary += summary_str('nmissing per block', samplings_np['nmissing_perb_'], self._npoint_per_blocks[scale])
    summary += summary_str('nempty per block', samplings_np['nempty_perb_'], self._npoint_per_blocks[scale])
    for name in samplings_np:
      if name not in ['nframes','nblock_valid_', 'ave_np_perb_', 'nempty_perb_', 'nmissing_perb_', 'nblock_invalid_']:
        summary += '{}: {}\n'.format(name, samplings_np[name]/t)
    summary += '\n'
    print(summary)
    return summary


  def block_index_to_scope(self, block_index):
    '''
    block_index: (n,3)
    '''
    block_index = tf.cast(block_index, tf.float32)
    bottom = block_index * self._strides[self.scale]
    center = bottom + self._widths[self.scale]*0.5
    bottom_center = tf.concat([bottom, center], -1)
    return bottom_center


  def replace_points_outside_block_UNUSED(self, block_index, pinb_mask):
    '''
    Replace the points out of block with the first one in this block.
    The first one must inside the block.
    '''
    def my_sparse_to_dense(dense_shape, sparse_indices0, sparse_value0):
      # dense_shape: [1024,    8,    3]
      # sparse_indices0: (4, 2)
      # sparse_value0: (4, 3)

      # return: sparse_values: (1024, 8, 3)
      sparse_shape0 = tf.shape(sparse_indices0)
      d = dense_shape[-1]

      sparse_indices = tf.expand_dims(sparse_indices0, 1)
      sparse_indices = tf.tile(sparse_indices, [1,d,1])
      tmp1 = tf.reshape(tf.range(d),[1,d,1])
      tmp1 = tf.tile(tmp1, [sparse_shape0[0], 1, 1])
      sparse_indices = tf.concat([sparse_indices, tmp1],2)
      sparse_indices = tf.reshape(sparse_indices, (-1,d))
      sparse_values = tf.reshape(sparse_value0, [-1])
      sparse_values = tf.sparse_to_dense(sparse_indices, dense_shape, sparse_values)
      return sparse_values

    invalid_indices = tf.cast(tf.where(tf.logical_not(pinb_mask)),tf.int32)
    tmp0 = tf.constant([[1,0],[0,0]], tf.int32)
    valid_indices = tf.matmul( invalid_indices, tmp0)

    invalid_values = tf.gather_nd(block_index, invalid_indices)
    valid_values = tf.gather_nd(block_index, valid_indices)
    dense_shape = tf.shape(block_index)

    invalid_values_dense = my_sparse_to_dense(dense_shape, invalid_indices, invalid_values)
    valid_values_dense = my_sparse_to_dense(dense_shape, invalid_indices, valid_values)
    block_index_fixed = block_index - invalid_values_dense + valid_values_dense
    print('scale {}, invalid_indices num:{}'.format(self.scale, tf.shape(invalid_indices)[0]))
    return block_index_fixed


  def check_grouped_xyz_inblock(self, grouped_xyz, block_bottom_center, empty_mask):
    block_bottom_center = tf.expand_dims(block_bottom_center, 1)
    block_bottom = block_bottom_center[:,:,0:3]
    block_center = block_bottom_center[:,:, 3:6]
    block_top = block_bottom + 2 * (block_center - block_bottom)
    top_check = tf.reduce_all(tf.less(grouped_xyz, block_top), -1)
    bottom_check = tf.reduce_all(tf.less_equal(block_bottom, grouped_xyz), -1)
    empty_mask = tf.cast(empty_mask, tf.bool)
    top_check = tf.logical_or(top_check, empty_mask)
    bottom_check = tf.logical_or(bottom_check, empty_mask)
    nerr_top = tf.reduce_sum(1-tf.cast(top_check, tf.int32))
    nerr_bottom = tf.reduce_sum(1-tf.cast(bottom_check, tf.int32))

    assert_top = tf.assert_equal(nerr_top, 0,
        message='check_grouped_xyz_inblock nerr_top')
    assert_bottom = tf.assert_equal(nerr_bottom, 0,
        message='scale {} check_grouped_xyz_inblock nerr_bottom: {}'.format(
              self.scale, nerr_bottom))
    return [assert_top, assert_bottom]


  def block_index_bound(self, block_index, xyz):
    '''
    The blocks at the edge
    '''
    xyz_max = tf.reduce_max(xyz, 0)
    xyz_min = tf.reduce_min(xyz, 0)
    padding = self._widths[self.scale] * self._padding_rate[self.scale]  # very important
    bindex_min = (xyz_min - padding)/self._strides[self.scale]
    bindex_max = (xyz_max + padding-self._widths[self.scale])/self._strides[self.scale]
    bindex_min = tf.reshape(bindex_min, (1,1,-1))
    bindex_max = tf.reshape(bindex_max, (1,1,-1))

    low_bmask = tf.greater_equal(tf.cast(block_index, tf.float32), bindex_min)
    low_bmask = tf.reduce_all(low_bmask, -1)
    up_bmask = tf.less_equal(tf.cast(block_index,tf.float32), bindex_max)
    up_bmask = tf.reduce_all(up_bmask, -1)
    b_in_bound_mask = tf.logical_and(low_bmask, up_bmask)
    ngp_edge_block = tf.reduce_sum(1-tf.cast(b_in_bound_mask, tf.int32))
    self.ngp_edge_block = ngp_edge_block
    return b_in_bound_mask, ngp_edge_block


  def check_xyz_inblock(self, block_index, xyz):
    block_bottom_center = self.block_index_to_scope(block_index)
    block_bottom = block_bottom_center[:,:,0:3]
    block_center = block_bottom_center[:,:,3:6]
    block_top = block_bottom + 2 * (block_center - block_bottom)
    xyz = tf.expand_dims(xyz, 1)

    check_top = tf.reduce_all(tf.less(xyz, block_top),2)
    check_bottom = tf.reduce_all(tf.less_equal(block_bottom, xyz),2)
    pinb_mask = tf.logical_and(check_bottom, check_top)
    nerr_scope = tf.reduce_sum(1-tf.cast(pinb_mask, tf.int32))
    ngp_out_block = tf.reduce_sum(1 - tf.cast(pinb_mask, tf.int32))
    self.ngp_out_block = ngp_out_block
    return pinb_mask, ngp_out_block


  def get_block_id(self, xyz):
    '''
    (1) Get the responding block index of each point
    xyz: (num_point0, 3)
      block_index: (num_point0, nblock_perp_3d, 3)
    block_id: (num_point0, nblock_perp_3d, 1)
    '''
    # (1.1) lower and upper block index bound
    self.num_point0 = num_point0 = tf.shape(xyz)[0]
    low_b_index = (xyz - self._widths[self.scale]) / self._strides[self.scale]
    up_b_index = xyz / self._strides[self.scale]


    # set a small pos offset. If low_b_index is float, ceil(low_b_index)
    # remains. But if low_b_index is int, ceil(low_b_index) increases.
    # Therefore, the points just on the intersection of two blocks will always
    # belong to left block. (See notes)
    low_b_index_offset = 1e-5
    low_b_index += low_b_index_offset # invoid low_b_index is exactly int
    low_b_index_fixed = tf.cast(tf.ceil(low_b_index), tf.int32)    # include
    up_b_index_fixed = tf.cast(tf.floor(up_b_index), tf.int32) + 1 # not include

    #(1.2) the block number for each point should be equal
    # If not, still make it equal to get point index per block.
    # Then rm the grouped points out of block scope.
    nblocks_per_points = up_b_index_fixed - low_b_index_fixed
    nbpp_max = tf.reduce_max(nblocks_per_points, axis=0)

    # ***** (optional)
    # block index is got by linear ranging from low bound by a fixed offset for
    # all the points. This is based on: num block per point is the same for all!
    # Check: No points will belong to greater nblocks. But few may belong to
    # smaller blocks: err_npoints
    if self._check_nblock_per_points_same:
      nbpp_min = tf.reduce_min(nblocks_per_points, axis=0)
      check_gap = tf.reduce_max(nbpp_max)-tf.reduce_min(nbpp_min)
      check_0 = tf.assert_less_equal(check_gap, 1,
                                    message="nbpp max min gap > 1."+\
                "Try increase low_b_index_offset a bit if max is the reason")

      tmp_err_np0 = tf.cast(tf.not_equal(nblocks_per_points,nbpp_max),tf.int32)
      tmp_err_np = tf.reduce_sum(tmp_err_np0, 1)
      err_npoints = tf.reduce_sum(tf.cast(tf.not_equal(tmp_err_np, 0),tf.int32))
      check_1 = tf.assert_less_equal(err_npoints, self._n_maxerr_same_nb_perps[self.scale],
                    message="err points with not different nblocks: {}".format(err_npoints))
      with tf.control_dependencies([check_0, check_1 ]):
        nbpp_max = tf.identity(nbpp_max)
    # *****

    self.nblocks_per_point = nbpp_max
    self.nblock_perp_3d = tf.reduce_prod(self.nblocks_per_point)

    pairs_3d = permutation_combination_3D(self.nblocks_per_point)
    pairs_3d = tf.tile( tf.expand_dims(pairs_3d, 0), [self.num_point0,1,1])

    block_index = tf.expand_dims(low_b_index_fixed, 1)
    # (num_point0, nblock_perp_3d, 3)
    block_index = tf.tile(block_index, [1, self.nblock_perp_3d, 1])
    block_index += pairs_3d

    pinb_mask, ngp_out_block = self.check_xyz_inblock(block_index, xyz)

    # (1.3) Remove the points belong to blocks out of edge bound
    b_in_bound_mask, ngp_edge_block = self.block_index_bound(block_index, xyz)

    # (1.4) block index -> block id
    tmp_bindex = tf.reshape(block_index, (-1,3))
    self.min_block_index = tf.reduce_min(tmp_bindex, 0)
    max_block_index = tf.reduce_max(tmp_bindex, 0)
    self.block_size = max_block_index - self.min_block_index + 1 # ADD ONE!!!
    # Make all block index positive. So that the blockid for each block index is
    # exclusive.
    block_index -= self.min_block_index
    # (num_point0, nblock_perp_3d, 1)
    block_id = block_index_to_block_id(block_index, self.block_size)
    block_id = tf.expand_dims(block_id, 2)

    # (1.5) concat block id and point index
    point_indices = tf.reshape( tf.range(0, num_point0, 1), (-1,1,1))
    point_indices = tf.tile(point_indices, [1, self.nblock_perp_3d, 1])
    # (num_point0, nblock_perp_3d, 2)
    bid_pindex = tf.concat([block_id, point_indices], 2)
    # (num_point0 * nblock_perp_3d, 2)

    bid_pindex = tf.reshape(bid_pindex, (-1,2))
    block_index = tf.reshape(block_index, (-1,3))
    pinb_mask = tf.reshape(pinb_mask, (-1,))
    b_in_bound_mask = tf.reshape(b_in_bound_mask, (-1,))

    # (1.6) rm grouped points out of block or points in block out of edge bound

    ngb_invalid = ngp_out_block + ngp_edge_block
    def rm_poutb():
      print('ngp_out_block:{}  ngp_edge_block:{}'.format(ngp_out_block, ngp_edge_block))
      gp_valid_mask = tf.logical_and(pinb_mask, b_in_bound_mask)
      pindex_inb = tf.cast(tf.where(gp_valid_mask)[:,0], tf.int32)
      return tf.gather(bid_pindex, pindex_inb, 0)
    bid_pindex = tf.cond( tf.greater(ngb_invalid, 0),
                         rm_poutb,
                         lambda: bid_pindex )
    self.npoint_grouped = tf.shape(bid_pindex)[0]
    self.ngp_out_block = ngp_out_block

    return bid_pindex


  def get_bid_point_index(self, bid_pindex):
    '''
    Get "block id index: and "point index within block".
    bid_pindex: (num_point0 * nblock_perp_3d, 2)
                                [block_id, point_index]
    bid_index__pindex_inb:  (num_point0 * nblock_perp_3d, 2)
                                [bid_index, pindex_inb]
    '''
    #(2.1) sort by block id, to put all the points belong to same block together
    # shuffle before sort for randomly sampling later
    if self._shuffle:
      bid_pindex = tf.random_shuffle(bid_pindex)
    sort_indices = tf.contrib.framework.argsort(bid_pindex[:,0],
                                                axis = 0)
    bid_pindex = tf.gather(bid_pindex, sort_indices, axis=0)
    block_id = bid_pindex[:,0]
    point_index = bid_pindex[:,1]

    #(2.2) block id -> block id index
    # get valid block num
    block_id_unique, blockid_index, npoint_per_block =\
                                                tf.unique_with_counts(block_id)
    self.nblock = tf.shape(block_id_unique)[0]
    check_nb = tf.assert_greater(self._nblock_buf_maxs[self.scale], self.nblock)
    with tf.control_dependencies([check_nb]):
      self.nblock = tf.identity(self.nblock)

    # get blockid_index for blocks with fewer points than self._np_perb_min_include
    tmp_valid_b = tf.greater_equal(npoint_per_block, self._np_perb_min_includes[self.scale])
    self.valid_bid_index = tf.cast(tf.where(tmp_valid_b)[:,0], tf.int32)
    self.nblock_valid = tf.shape(self.valid_bid_index)[0]
    self.nblock_invalid = self.nblock - self.nblock_valid
    tmp_invalid_b = tf.less(npoint_per_block, self._np_perb_min_includes[self.scale])
    self.invalid_bid_index = tf.cast(tf.where(tmp_invalid_b)[:,0], tf.int32)

    #(2.3) Get point index per block
    #      Based on: all points belong to same block is together
    tmp0 = tf.cumsum(npoint_per_block)[0:-1]
    tmp0 = tf.concat([tf.constant([0],tf.int32), tmp0],0)
    tmp1 = tf.gather(tmp0, blockid_index)
    tmp2 = tf.range(self.npoint_grouped)
    point_index_per_block = tf.expand_dims( tmp2 - tmp1,  1)
    blockid_index = tf.expand_dims(blockid_index, 1)

    bid_index__pindex_inb = tf.concat(
                                    [blockid_index, point_index_per_block], -1)
    # (2.4) sampling fixed number of points for each block
    remain_mask = tf.less(tf.squeeze(point_index_per_block,1), self._npoint_per_blocks[self.scale])
    remain_index = tf.squeeze(tf.where(remain_mask),1)
    bid_index__pindex_inb = tf.gather(bid_index__pindex_inb, remain_index)
    point_index = tf.gather(point_index, remain_index)

    # (2.5) record sampling parameters
    if self._record_samplings:
      self.record_samplings(npoint_per_block)

    return bid_index__pindex_inb, point_index, block_id_unique


  def record_samplings(self, npoint_per_block):
    self.samplings[self.scale]['ngp_edge_block_'] += tf.cast(self.ngp_edge_block,tf.int64)
    self.samplings[self.scale]['ngp_out_block_'] += tf.cast(self.ngp_out_block,tf.int64)
    self.samplings[self.scale]['npoint_grouped_'] += tf.cast(self.npoint_grouped,tf.int64)

    self.samplings[self.scale]['nframes'] += tf.constant(1, tf.int64)
    self.samplings[self.scale]['nblock_valid_'] += tf.cast(self.nblock_valid, tf.int64)
    self.samplings[self.scale]['nblock_invalid_'] += tf.cast(self.nblock_invalid, tf.int64)

    tmp = npoint_per_block - self._npoint_per_blocks[self.scale]
    missing_mask = tf.cast(tf.greater(tmp,0), tf.int32)
    tmp_missing = missing_mask * tmp
    less_mask = tf.cast(tf.less(tmp,0), tf.int32)
    tmp_less = less_mask * tmp


    ave_np_perb, variance = tf.nn.moments(npoint_per_block,0)
    std_np_perb = tf.sqrt(tf.cast(variance,tf.float32))
    self.samplings[self.scale]['ave_np_perb_'] += tf.cast(ave_np_perb, tf.int64)
    self.samplings[self.scale]['std_np_perb_'] +=  tf.cast(std_np_perb, tf.int64)
    nblock_has_missing = tf.reduce_sum(missing_mask)
    nblock_less = tf.reduce_sum(less_mask)
    nmissing_perb = tf.cond( tf.greater(nblock_has_missing,0),
            lambda: tf.reduce_sum(tmp_missing) / nblock_has_missing if nblock_has_missing > 0 else 0,
            lambda: 0)
    self.samplings[self.scale]['nmissing_perb_'] += tf.cast(nmissing_perb, tf.int64)
    if nblock_less>0:
      self.samplings[self.scale]['nempty_perb_'] += tf.cast(tf.reduce_sum(tmp_less) / nblock_less, tf.int64)
    #self.samplings[self.scale]['valid_grouped_npoint_'] += tf.cast(bid_index__pindex_inb.shape[0].value, tf.int64)
    #self.samplings[self.scale]['nblock_perp_3d_'] += tf.cast(self.nblock_perp_3d, tf.int64)


  def gather_grouped_xyz(self, bid_index__pindex_inb, point_index, xyz):
    #(3.1) gather grouped point index
    # gen point index: (real nblock, self._npoint_per_block, 1)
    grouped_pindex0 = tf.sparse_to_dense(bid_index__pindex_inb,
                          (self._nblock_buf_maxs[self.scale],
                           self._npoint_per_blocks[self.scale]),
                              point_index, default_value=-1)

    #(3.2) remove the blocks with too less points
    if self._debug_only_blocks_few_points:
      bid_index_valid = self.invalid_bid_index
      nblock_valid = self.nblock_invalid
    else:
      bid_index_valid = self.valid_bid_index
      nblock_valid = self.nblock_valid

    tmp_nb = self._nblocks[self.scale] - nblock_valid

    def block_not_enough():
      if self._use_less_points_block_when_not_enough:
        bid_index_valid_1 = tf.concat([bid_index_valid,
                                self.invalid_bid_index[0:tmp_nb]],0)
        tmp_nb1 = self._nblocks[self.scale] - tf.shape(bid_index_valid_1)[0]
        bid_index_sampled = tf.concat([bid_index_valid_1, tf.ones(tmp_nb1,tf.int32)\
                                     * bid_index_valid_1[0]],0)
        return bid_index_sampled

    def block_too_many():
      if self._shuffle:
        return tf.contrib.framework.sort( tf.random_shuffle(bid_index_valid)[0:self._nblocks[self.scale]] )
      else:
        return tf.contrib.framework.sort( bid_index_valid[0:self._nblocks[self.scale]] )

    bid_index_sampling = tf.cond( tf.greater(tmp_nb, 0),
                                 block_not_enough,
                                 block_too_many )

    #(3.3) sampling fixed number of blocks when too many blocks are provided

    #if self._shuffle:
    #  bid_index_sampling = tf.cond( tf.greater(tmp_nb, 0),
    #         lambda: tf.concat([bid_index_valid, tf.ones(tmp_nb,tf.int32) * bid_index_valid[0]],0),
    #         lambda: tf.contrib.framework.sort( tf.random_shuffle(bid_index_valid)[0:self._nblocks[self.scale]] ))
    #else:
    #  bid_index_sampling = tf.cond( tf.greater(tmp_nb, 0),
    #         lambda: tf.concat([bid_index_valid, tf.ones(tmp_nb,tf.int32) * bid_index_valid[0]],0),
    #         lambda: tf.contrib.framework.sort( bid_index_valid[0:self._nblocks[self.scale]] ))
    grouped_pindex = tf.gather(grouped_pindex0, bid_index_sampling)

    #(3.4) gather xyz from point index
    empty_mask = tf.less(grouped_pindex,0)
    if self._empty_point_index != -1:
      # replace -1 with first one of each block
      first_pindices = grouped_pindex[0,0:1]
      empty_indices = tf.cast(tf.where(empty_mask), tf.int32)
      # Incorrect!!!: use the first point of first block always.
      # Should use the first one of each block, but it's hard to implement.
      # So always use empty mask to avoid this error!
      first_pindices = tf.tile(first_pindices, [tf.shape(empty_indices)[0]]) + 1
      first_pindices_dense = tf.sparse_to_dense(empty_indices, tf.shape(grouped_pindex), first_pindices)
      grouped_pindex = grouped_pindex + first_pindices_dense

    grouped_xyz = tf.gather(xyz, grouped_pindex)
    empty_mask = tf.cast(empty_mask, tf.int8) # to feed into input pipeline

    return grouped_xyz, empty_mask, bid_index_sampling


  def all_bottom_centers(self, block_id_unique, bid_index_sampling):
    '''
    block_id_unique: (self.nblock,)
    bid_index_sampling: (self._nblock,)
    '''
    bids_sampling = tf.gather(block_id_unique, bid_index_sampling)
    bindex_sampling = block_id_to_block_index(bids_sampling, self.block_size)
    bindex_sampling += self.min_block_index
    block_bottom_center = self.block_index_to_scope(bindex_sampling)
    return bids_sampling, block_bottom_center


  def grouping(self, xyz, scale):
    '''
    xyz: (num_point0,3)
    grouped_xyz: (num_block,npoint_per_block,3)

    Search by point: for each point in xyz, find all the block d=ids
    '''
    self.scale = scale
    assert len(xyz.shape) == 2
    assert xyz.shape[-1].value == 3
    t0 = tf.timestamp()

    bid_pindex = self.get_block_id(xyz)
    bid_index__pindex_inb, point_index, block_id_unique = self.get_bid_point_index(bid_pindex)
    grouped_xyz, empty_mask, bid_index_sampling = self.gather_grouped_xyz(
                                      bid_index__pindex_inb, point_index, xyz)
    bids_sampling, block_bottom_center = self.all_bottom_centers(block_id_unique, bid_index_sampling)
    if self._check_grouped_xyz_inblock:
      check_gs = self.check_grouped_xyz_inblock(grouped_xyz, block_bottom_center, empty_mask)
      with tf.control_dependencies(check_gs):
        grouped_xyz = tf.identity(grouped_xyz)

    if self._record_samplings:
      self.samplings[self.scale]['t_per_frame_'] += tf.timestamp() - t0
    # **************************************************************************
    #       Add middle tensors to check between graph model and eager model
    others = {}
    others['value'] = []
    others['name'] = []
    #point_index_O = tf.concat([point_index, tf.zeros(self.npoint_grouped-tf.shape(point_index),tf.int32)],0)
    #bid_index__pindex_inb_O = tf.concat([bid_index__pindex_inb,
    #        tf.zeros([self.npoint_grouped-tf.shape(bid_index__pindex_inb)[0],2],tf.int32)],0)
    #others['value'] += [bid_pindex, bid_index__pindex_inb_O, point_index_O]
    #others['name'] += ['bid_pindex', 'bid_index__pindex_inb', 'point_index']

    #block_id_unique_O = tf.concat([block_id_unique, tf.zeros(1000-tf.shape(block_id_unique),tf.int32)],0)
    #bid_index_sampling_O = tf.concat([bid_index_sampling, tf.zeros(1000-tf.shape(bid_index_sampling),tf.int32)],0)
    #others['value'] += [block_id_unique_O, bid_index_sampling_O]
    #others['name'] += ['block_id_unique', 'bid_index_sampling']
    #others['value'] += [self.grouped_pindex0]
    #others['name'] += ['grouped_pindex0']
    if self._gen_ply:
      others['value'] += [bids_sampling]
      others['name'] += ['bids_sampling']

    return grouped_xyz, empty_mask, block_bottom_center, self.nblock_valid, others


  def grouped_mean(self, grouped_xyz, empty_mask):
    valid_mask = 1-empty_mask
    valid_num = tf.expand_dims(tf.cast( tf.reduce_sum(valid_mask, 1), tf.float32),1)
    valid_mask = tf.expand_dims(tf.cast(valid_mask, tf.float32),2)
    tmp_sum = tf.reduce_sum(grouped_xyz * valid_mask, 1)
    grouped_mean = tmp_sum / valid_num
    return grouped_mean


  def valid_block_xyz(self, grouped_xyz, empty_mask, block_bottom_center, nblock_valid):
    if self._block_pos == 'mean':
      block_xyz = self.grouped_mean(grouped_xyz, empty_mask)
    elif self._block_pos == 'center':
      block_xyz = block_bottom_center[:,3:6]
    empty_mask = tf.cast(empty_mask, tf.bool)
    valid_block_xyz = tf.cond( tf.less(nblock_valid, tf.shape(block_xyz)[0]),
                              lambda: block_xyz[0:nblock_valid],
                              lambda: block_xyz)

    return valid_block_xyz


  def grouping_multi_scale(self, xyz, num_scale=None):
    grouped_pos = 'center'
    grouped_pos = 'mean'

    if num_scale==None:
      num_scale = self._num_scale
    new_xyzs = [xyz]
    grouped_xyz_ms = []
    empty_mask_ms = []
    block_bottom_center_ms = []
    nblock_valid_ms = []
    others_ms = []

    for s in range(num_scale):
      grouped_xyz, empty_mask, block_bottom_center, nblock_valid, others = self.grouping(xyz, s)
      xyz = self.valid_block_xyz(grouped_xyz, empty_mask, block_bottom_center, nblock_valid)

      grouped_xyz_ms.append(grouped_xyz)
      empty_mask_ms.append(empty_mask)
      block_bottom_center_ms.append(block_bottom_center)
      nblock_valid_ms.append(nblock_valid)
      others_ms.append(others)
    return grouped_xyz_ms, empty_mask_ms, block_bottom_center_ms, nblock_valid_ms, others_ms


def main_eager(DATASET_NAME, filenames, sg_settings, nframes):
  tf.enable_eager_execution()
  dataset_meta = DatasetsMeta(DATASET_NAME)
  num_classes = dataset_meta.num_classes

  dataset = tf.data.TFRecordDataset(filenames,
                                      compression_type="",
                                      buffer_size=1024*100,
                                      num_parallel_reads=1)

  batch_size = nframes
  is_training = False

  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(
    tf.contrib.data.map_and_batch(
      lambda value: parse_pl_record(value, is_training),
      batch_size=batch_size,
      num_parallel_batches=1,
      drop_remainder=False))
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  get_next = dataset.make_one_shot_iterator().get_next()
  features_next, label_next = get_next
  points_next = features_next['points'][:,:,0:3]

  bsg = BlockGroupSampling(sg_settings)
  bsg.show_settings()

  xyzs = []
  grouped_xyzs = []
  empty_masks = []
  others = []
  for s in range(bsg._num_scale):
    grouped_xyzs.append([])
    empty_masks.append([])
    others.append({})

  for i in range(batch_size):
    points_i = points_next[i,:,:]

    grouped_xyz_i, empty_mask_i, bottom_center_i, nblock_valid_i, others_i = \
          bsg.grouping_multi_scale(points_i)

    #grouped_xyz_i, empty_mask_i, bottom_center_i, nblock_valid_i, others_i = bsg.grouping(points_i, 0)

    num_scale = len(grouped_xyz_i)
    samplings_i = bsg.samplings
    samplings_np_ms = []
    for s in range(len(samplings_i)):
      samplings_np_ms.append({})
      for name in samplings_i[s]:
        samplings_np_ms[s][name] = samplings_i[s][name].numpy()
    bsg.show_samplings_np_multi_scale(samplings_np_ms)

    xyzs.append(np.expand_dims(points_i.numpy(),0))

    for s in range(num_scale):
      grouped_xyz_i[s] = grouped_xyz_i[s].numpy()
      grouped_xyzs[s].append( np.expand_dims(grouped_xyz_i[s], 0) )
      empty_mask_i[s] = empty_mask_i[s].numpy()
      empty_masks[s].append( np.expand_dims(empty_mask_i[s], 0) )
      bottom_center_i[s] = bottom_center_i[s].numpy()

      for k in range(len(others_i[s]['value'])):
        name_k = others_i[s]['name'][k]
        if name_k not in others[s]:
          others[s][name_k] = []
        others_i[s]['value'][k] = others_i[s]['value'][k].numpy()
        others[s][name_k].append( np.expand_dims( others_i[s]['value'][k],0 ) )

      if bsg._gen_ply:
        valid_flag = '' if not bsg._debug_only_blocks_few_points else '_invalid'
        if not bsg._shuffle:
          valid_flag += '_NoShuffle'
        bids_sampling_i = others[s]['bids_sampling'][k][0]
        gen_plys(DATASET_NAME, i, points_i.numpy(), grouped_xyz_i[s],
                  bottom_center_i[s], bids_sampling_i, valid_flag, '_ES%s'%(s))

  xyzs = np.concatenate(xyzs,0)
  for s in range(num_scale):
    grouped_xyzs[s] = np.concatenate(grouped_xyzs[s],0)
    empty_masks[s] = np.concatenate(empty_masks[s],0)

    for name in others[s]:
      if name not in ['valid_bid_indexs', 'block_id_unique']:
        print(name)
        others[k][name] = np.concatenate(others[k][name], 0)
    others[s]['empty_mask'] = empty_masks[s]

  return xyzs, grouped_xyzs, others, bsg._shuffle


def main(DATASET_NAME, filenames, sg_settings, nframes):
  _DATA_PARAS = get_data_shapes_from_tfrecord(filenames)
  dataset_meta = DatasetsMeta(DATASET_NAME)
  num_classes = dataset_meta.num_classes

  with tf.Graph().as_default():
   with tf.device('/device:GPU:0'):
    dataset = tf.data.TFRecordDataset(filenames,
                                        compression_type="",
                                        buffer_size=1024*100,
                                        num_parallel_reads=1)
    batch_size = nframes
    is_training = False
    bsg = BlockGroupSampling(sg_settings)
    bsg.show_settings()
    print(0, tf.get_default_graph() )

    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        lambda value: parse_pl_record(value, is_training, _DATA_PARAS, bsg),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=True))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    get_next = dataset.make_one_shot_iterator().get_next()
    features_next, label_next = get_next

    points_next = features_next['points']
    grouped_xyz_next = []
    empty_mask_next = []
    bottom_center_next = []
    others_next = []

    num_scale = bsg._num_scale
    for s in range(num_scale):
      grouped_xyz_next.append(features_next['grouped_xyz_%d'%(s)])
      empty_mask_next.append(features_next['empty_mask_%d'%(s)])
      bottom_center_next.append(features_next['block_bottom_center_%d'%(s)])

      others_next.append({})
      for name in ['bids_sampling']:
        name_s = name+'_%d'%(s)
        if name_s in features_next:
          others_next[s][name_s] = features_next[name_s]
    #samplings_next = bsg.samplings

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
      #sess.run(init)
      #if DEBUG:
      #  debugs = sess.run(self.debugs)

      #points_i = sess.run(points_next)

      points, grouped_xyzs, empty_masks, bottom_centers, others = \
        sess.run([points_next, grouped_xyz_next, empty_mask_next, bottom_center_next, others_next] )

      #bsg.show_samplings_np(samplings)
      xyzs = points[:,:,0:3]
      print('group OK')

      for s in range(num_scale):
        others[s]['empty_mask'] = empty_masks[s]

        for frame in range(batch_size):
          if bsg._gen_ply:
            valid_flag = '' if not bsg._debug_only_blocks_few_points else '_invalid'
            if not bsg._shuffle:
              valid_flag += '_NoShuffle'
            bids_sampling = others[s]['bids_sampling_%d'%(s)][frame,0]
            gen_plys(DATASET_NAME, frame, points[frame], grouped_xyzs[s][frame], bottom_centers[s][frame],\
                      bids_sampling, valid_flag, 'S%d'%(s))

    return xyzs, grouped_xyzs, others, bsg._shuffle


def gen_plys(DATASET_NAME, frame, points, grouped_xyz, bottom_center, bids_sampling, valid_flag='', main_flag=''):
  print('bids_sampling: {}'.format(bids_sampling))
  path = '/tmp/%d_plys'%(frame) + main_flag

  if type(points)!=type(None):
    ply_fn = '%s/points.ply'%(path)
    create_ply_dset(DATASET_NAME, points, ply_fn,
                    extra='random_same_color')

  ply_fn = '%s/grouped_points_%s.ply'%(path, valid_flag)
  create_ply_dset(DATASET_NAME, grouped_xyz, ply_fn,
                  extra='random_same_color')
  ply_fn = '%s/blocks%s.ply'%(path, valid_flag)
  draw_blocks_by_bottom_center(ply_fn, bottom_center, random_crop=0)

  tmp = np.random.randint(0, min(grouped_xyz.shape[0], 10))
  tmp = np.arange(0, min(grouped_xyz.shape[0],10))
  for j in tmp:
    block_id = bids_sampling[j]
    ply_fn = '%s/blocks%s/%d_blocks.ply'%(path, valid_flag, block_id)
    draw_blocks_by_bottom_center(ply_fn, bottom_center[j:j+1], random_crop=0)
    ply_fn = '%s/blocks%s/%d_points.ply'%(path, valid_flag, block_id)
    create_ply_dset(DATASET_NAME, grouped_xyz[j], ply_fn,
                  extra='random_same_color')


def get_sg_settings():
  sg_settings1 = {}
  sg_settings1['width'] =   [[0.2,0.2,0.2], [0.6,0.6,0.6]]
  sg_settings1['stride'] =  [[0.1,0.1,0.1], [0.4,0.4,0.4]]
  sg_settings1['nblock'] =  [512,           64]
  sg_settings1['npoint_per_block'] = [10,   10]
  sg_settings1['np_perb_min_include'] = [4, 2]
  sg_settings1['max_nblock'] =      [6000,  500]

  sg_settings = sg_settings1
  sg_settings['block_pos'] = 'center'
  #sg_settings['block_pos'] = 'mean'
  sg_settings['gen_ply'] = False
  return sg_settings

if __name__ == '__main__':
  DATASET_NAME = 'MODELNET40'
  path = '/home/z/Research/dynamic_pointnet/data/MODELNET40H5F/Merged_tfrecord/4_mgs0.2048_gs2_2d2-rep_fmn1_mvp1-1-1024--pd3-1M'
  path = '/home/z/Research/SparseVoxelNet/data/MODELNET40__H5F/ORG_tfrecord/1024_mgs0.2048_gs2_2d2-rep_fmn1_mvp1-1-1024--pd3-1M'
  tmp = 'chair_0056'
  tmp = 'dresser_0282'
  tmp= 'plant_0199'
  tmp = 'car_0048'
  tmp = 'bathtub_0007'
  tmp = 'airplane_0001' # ERR
  tmp = '*'
  filenames = glob.glob(os.path.join(path, tmp+'.tfrecord'))
  assert len(filenames) >= 1

  sg_settings = get_sg_settings()

  if len(sys.argv) > 1:
    main_flag = sys.argv[1]
  else:
    main_flag = 'g'
    #main_flag = 'eg'
    #main_flag = 'e'
  print(main_flag)

  nframes = 20

  if 'e' in main_flag:
    sg_settings['record'] = True
    xyzs_E, grouped_xyzs_E, others_E, shuffle_E = \
      main_eager(DATASET_NAME, filenames, sg_settings, nframes)
  if 'g' in main_flag:
    sg_settings['record'] = False
    xyzs, grouped_xyzs, others, shuffle = \
      main(DATASET_NAME, filenames, sg_settings, nframes)

  if main_flag=='eg' and shuffle==False and shuffle_E==False:
    batch_size = min(xyzs.shape[0], xyzs_E.shape[0])
    for b in range(batch_size):
      assert (xyzs_E[b] == xyzs[b]).all(), 'time %d xyz different'%(b)
      print('time %d xyzs of g and e is same'%(b))

      num_scale = len(grouped_xyzs)
      for s in range(num_scale):
        for name in others_E[s]:
          check = (others_E[s][name][b] == others[s][name][b]).all()
          if not check:
            print('time %d scale %d others- %s different'%(b, s, name))
            err = others_E[s][name][b] != others[s][name][b]
            nerr = np.sum(err)
            print(others[s][name][b])
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
          else:
            print('time %d scale %d others - %s of g and e is same'%(b, s, name))

        diff = grouped_xyzs_E[s][b] - grouped_xyzs[s][b]
        diff = np.sum(diff, -1)
        nerr = np.sum(diff != 0)
        if nerr!=0:
          print("time %d, scale %d, grouped_xyz nerr=%d"%(b, s, nerr))
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
        else:
          print('time %d, sclae %s, grouped_xyzs of g and e is same'%(b, s))

