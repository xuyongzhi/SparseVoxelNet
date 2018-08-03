# July 2018 xyz
import tensorflow as tf
import glob, os, sys
import numpy as np
from utils.dataset_utils import parse_pl_record
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
from utils.ply_util import create_ply_dset, draw_blocks_by_bottom_center

DEBUG = True

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
  def __init__(self, width, stride, nblock, npoint_per_block, max_nblock):
    '''
      width = [1,1,1]
      stride = [1,1,1]
      npoint_per_block = 32
    '''
    self._width = np.array(width, dtype=np.float32)
    self._stride = np.array(stride, dtype=np.float32)
    self._nblock = nblock
    self._npoint_per_block = npoint_per_block
    # cut all the block with too few points
    self._npoint_per_block_min = max(int(npoint_per_block*0.02), 2)
    assert self._width.size == self._stride.size == 3
    self._nblock_buf_max = max_nblock

    self.samplings = {}

    # debug flags
    self._shuffle = True
    # Theoretically, num_block_per_point should be same for all. This is 0.
    self._check_nblock_per_points_same = True
    if self._check_nblock_per_points_same:
      self._n_maxerr_same_nb_perp = 0
      self._n_maxerr_same_nb_perp = 1

    self._check_xyz_inblock = False
    if self._check_xyz_inblock:
      self._replace_points_outside_block = True
    self._check_grouped_xyz_inblock = False

    self._debug_only_blocks_few_points = False
    self.debugs = {}


  def show_settings(self):
    items = ['_width', '_stride', '_npoint_per_block', '_nblock',
              '_npoint_per_block_min','_n_maxerr_same_nb_perp', '_shuffle']
    print('\n\nsettings:\n')
    for item in items:
      if hasattr(self, item):
        print('{}:{}'.format(item, getattr(self,item)))
    print('\n\n')


  def show_summary(self, p_i):
    #for item in self.samplings:
    #  print('{}:{}'.format(item, self.samplings[item]))

    def summary_str(item, config, real):
      return '{}: {}/{}, {:.3f}\n'.format(item, real, config,
                      1.0*tf.cast(real,tf.float32)/tf.cast(config,tf.float32))
    summary = '\npoint %d\n'%(p_i)
    summary += '\tReal / Config\n'
    summary += summary_str('valid nblock', self._nblock, self.nblock_valid)
    summary += 'nblock_invalid: {}\n'.format(self.nblock_invalid)
    summary += 'ave-std np_perb: {}-{:.1f}/{}\n'.format(self.samplings['ave_np_perb'],
                self.samplings['std_np_perb'], self._npoint_per_block)
    summary += summary_str('grouped_sampling_rate', self.npoint_grouped,
                           self.samplings['valid_grouped_npoint'])
    summary += summary_str('nblock_has_missing', self.nblock,
                           self.samplings['nblock_has_missing'])
    summary += summary_str('nmissing_perb', self._npoint_per_block,
                           self.samplings['nmissing_perb'])
    summary += summary_str('nempty_perb', self._npoint_per_block,
                           self.samplings['nempty_perb'])
    print(summary)


  def block_index_to_scope(self, block_index):
    '''
    block_index: (n,3)
    '''
    block_index = tf.cast(block_index, tf.float32)
    bottom = block_index * self._stride
    center = bottom + self._width*0.5
    bottom_center = tf.concat([bottom, center], -1)
    return bottom_center

  def check_xyz_inblock(self, block_index, xyz):
    block_bottom_center = self.block_index_to_scope(block_index)
    block_bottom = block_bottom_center[:,:,0:3]
    block_center = block_bottom_center[:,:,3:6]
    block_top = block_bottom + 2 * (block_center - block_bottom)
    xyz = tf.expand_dims(xyz, 1)

    check_top = tf.less(xyz, block_top)
    check_top = tf.reduce_all(check_top, 2)
    nerr_top = tf.reduce_sum(1-tf.cast(check_top, tf.int32))
    assert_top = tf.assert_equal(nerr_top, 0,
                message='check_xyz_inblock nerr_top:{}'.format(nerr_top))
    check_bottom = tf.reduce_all(tf.less_equal(block_bottom, xyz),2)
    nerr_bottom = tf.reduce_sum(1-tf.cast(check_bottom,tf.int32))

    if self._replace_points_outside_block:
      block_index = tf.cond( nerr_bottom>0,
            lambda: self.replace_points_outside_block(block_index, check_bottom),
            lambda: block_index)
    else:
      assert_bottom = tf.assert_equal(nerr_bottom, 0,
                message='check_xyz_inblock nerr_bottom:{}'.format(nerr_bottom))
      with tf.control_dependencies([assert_top, assert_bottom]):
        block_index = tf.identity(block_index)

    return block_index

  def replace_points_outside_block(self, block_index, check_bottom):
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

    invalid_indices = tf.cast(tf.where(tf.logical_not(check_bottom)),tf.int32)
    tmp0 = tf.constant([[1,0],[0,0]], tf.int32)
    valid_indices = tf.matmul( invalid_indices, tmp0)

    invalid_values = tf.gather_nd(block_index, invalid_indices)
    valid_values = tf.gather_nd(block_index, valid_indices)
    dense_shape = tf.shape(block_index)

    invalid_values_dense = my_sparse_to_dense(dense_shape, invalid_indices, invalid_values)
    valid_values_dense = my_sparse_to_dense(dense_shape, invalid_indices, valid_values)
    block_index_fixed = block_index - invalid_values_dense + valid_values_dense
    return block_index_fixed

  def check_grouped_xyz_inblock(self, grouped_xyz, block_bottom_center, empty_mask):
    block_bottom_center = tf.expand_dims(block_bottom_center, 1)
    block_bottom = block_bottom_center[:,:,0:3]
    block_center = block_bottom_center[:,:, 3:6]
    block_top = block_bottom + 2 * (block_center - block_bottom)
    top_check = tf.reduce_all(tf.less(grouped_xyz, block_top), -1)
    bottom_check = tf.reduce_all(tf.less_equal(block_bottom, grouped_xyz), -1)
    top_check = tf.logical_or(top_check, empty_mask)
    bottom_check = tf.logical_or(bottom_check, empty_mask)
    nerr_top = tf.reduce_sum(1-tf.cast(top_check, tf.int32))
    nerr_bottom = tf.reduce_sum(1-tf.cast(bottom_check, tf.int32))

    assert_top = tf.assert_equal(nerr_top, 0,
        message='check_grouped_xyz_inblock nerr_top: {}'.format(nerr_top))
    assert_bottom = tf.assert_equal(nerr_bottom, 0,
        message='check_grouped_xyz_inblock nerr_bottom: {}'.format(nerr_bottom))
    return [assert_top, assert_bottom]

  def get_block_id(self, xyz):
    '''
    (1) Get the responding block index of each point
    xyz: (num_point0, 3)
      block_index: (num_point0, nblock_perp_3d, 3)
    block_id: (num_point0, nblock_perp_3d, 1)
    '''
    # (1.1) lower and upper block index bound
    self.num_point0 = num_point0 = xyz.shape[0].value
    low_b_index = (xyz - self._width) / self._stride
    up_b_index = xyz / self._stride

    # set a small pos offset. If low_b_index is float, ceil(low_b_index)
    # remains. But if low_b_index is int, ceil(low_b_index) increases.
    # Therefore, the points just on the intersection of two blocks will always
    # belong to left block. (See notes)
    low_b_index_offset = 1e-5
    low_b_index += low_b_index_offset # invoid low_b_index is exactly int
    low_b_index_fixed = tf.cast(tf.ceil(low_b_index), tf.int32)    # include
    up_b_index_fixed = tf.cast(tf.floor(up_b_index), tf.int32) + 1 # not include

    #(1.2) the block number for each point should be equal
    nblocks_per_points = up_b_index_fixed - low_b_index_fixed
    nbpp_max_int = tf.reduce_max(nblocks_per_points, axis=0)
    nbpp_max = tf.cast(nbpp_max_int, tf.float32)
    nbpp_min = tf.cast(tf.reduce_min(nblocks_per_points, axis=0), tf.float32)

    # *****
    # block index is got by linear ranging from low bound by a fixed offset for
    # all the points. This is based on: num block per point is the same for all!
    # Check: No points will belong to greater nblocks. But few may belong to
    # smaller blocks: err_npoints
    if self._check_nblock_per_points_same:
      nbpp_mean = tf.reduce_mean(tf.cast(nblocks_per_points,tf.float32), axis=0)
      # (a) The most are same as nbpp_max
      # (b) Very few may be nbpp_max - 1
      tmp_str = '\nnbpp min={}, mean={}, max={}'.format(nbpp_min, nbpp_mean, nbpp_max)
      check_0 = tf.assert_less(nbpp_max - nbpp_mean, 0.001,
                    message="nbpp_max > nbpp_mean increase low_b_index_offset"+tmp_str)
      check_1 = tf.assert_less(nbpp_mean - nbpp_min, 1.0,
                    message="nbpp_min is too small."+tmp_str)
      tmp_err_np = tf.cast(tf.not_equal(nblocks_per_points,nbpp_max_int),tf.int32)
      tmp_err_np = tf.reduce_sum(tmp_err_np, 1)
      err_npoints = tf.reduce_sum(tf.cast(tf.not_equal(tmp_err_np, 0),tf.int32))
      check_2 = tf.assert_less_equal(err_npoints, self._n_maxerr_same_nb_perp,
                    message="err points with not different nblocks: {}".format(err_npoints)+tmp_str)
      with tf.control_dependencies([check_0, check_1, check_2 ]):
        nbpp_max_int = tf.identity(nbpp_max_int)
    # *****

    self.nblocks_per_point = nbpp_max_int
    self.nblock_perp_3d = tf.reduce_prod(self.nblocks_per_point)
    self.npoint_grouped = self.num_point0 * self.nblock_perp_3d

    pairs_3d = permutation_combination_3D(self.nblocks_per_point)
    pairs_3d = tf.tile( tf.expand_dims(pairs_3d, 0), [self.num_point0,1,1])

    block_index = tf.expand_dims(low_b_index_fixed, 1)
    # (num_point0, nblock_perp_3d, 3)
    block_index = tf.tile(block_index, [1, self.nblock_perp_3d, 1])
    block_index += pairs_3d

    if self._check_xyz_inblock:
      block_index = self.check_xyz_inblock(block_index, xyz)

    # (1.3) block index -> block id
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

    # (1.4) concat block id and point index
    point_indices = tf.reshape( tf.range(0, num_point0, 1), (-1,1,1))
    point_indices = tf.tile(point_indices, [1, self.nblock_perp_3d, 1])
    # (num_point0, nblock_perp_3d, 2)
    bid_pindex = tf.concat([block_id, point_indices], 2)
    # (num_point0 * nblock_perp_3d, 2)

    bid_pindex = tf.reshape(bid_pindex, (-1,2))
    block_index = tf.reshape(block_index, (-1,3))

    #if DEBUG:
    #  self.debugs['xyz'] = xyz
    #  self.debugs['low_b_index_fixed'] = low_b_index_fixed
    #  self.debugs['pairs_3d'] = pairs_3d
    #  self.debugs['block_index'] = block_index
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
    check_nb = tf.assert_greater(self._nblock_buf_max, self.nblock)
    with tf.control_dependencies([check_nb]):
      self.nblock = tf.identity(self.nblock)

    # get blockid_index for blocks with fewer points than self._npoint_per_block_min
    tmp_valid_b = tf.greater_equal(npoint_per_block, self._npoint_per_block_min)
    self.valid_bid_index = tf.cast(tf.where(tmp_valid_b)[:,0], tf.int32)
    self.nblock_valid = tf.shape(self.valid_bid_index)[0]
    self.nblock_invalid = self.nblock - self.nblock_valid
    if self._debug_only_blocks_few_points:
      tmp_invalid_b = tf.less(npoint_per_block, self._npoint_per_block_min)
      self.invalid_bid_index = tf.cast(tf.where(tmp_invalid_b)[:,0], tf.int32)

    #(2.3) Get point index per block
    #      Based on: all points belong to same block is together
    self.samplings['max_npoint_per_block']= tf.reduce_max(npoint_per_block)

    tmp0 = tf.cumsum(npoint_per_block)[0:-1]
    tmp0 = tf.concat([tf.constant([0],tf.int32), tmp0],0)
    tmp1 = tf.gather(tmp0, blockid_index)
    tmp2 = tf.range(self.npoint_grouped)
    point_index_per_block = tf.expand_dims( tmp2 - tmp1,  1)
    blockid_index = tf.expand_dims(blockid_index, 1)

    bid_index__pindex_inb = tf.concat(
                                    [blockid_index, point_index_per_block], -1)
    # (2.4) sampling fixed number of points for each block
    remain_mask = tf.less(tf.squeeze(point_index_per_block,1), self._npoint_per_block)
    remain_index = tf.squeeze(tf.where(remain_mask),1)
    bid_index__pindex_inb = tf.gather(bid_index__pindex_inb, remain_index)
    point_index = tf.gather(point_index, remain_index)

    # record sampling parameters
    samplings = {}
    tmp = npoint_per_block - self._npoint_per_block
    missing_mask = tf.cast(tf.greater(tmp,0), tf.int32)
    tmp_missing = missing_mask * tmp
    empty_mask = tf.cast(tf.less(tmp,0), tf.int32)
    tmp_empty = empty_mask * tmp

    samplings['ave_np_perb'], variance = tf.nn.moments(npoint_per_block,0)
    samplings['std_np_perb'] = tf.sqrt(tf.cast(variance,tf.float32))
    samplings['nblock_has_missing'] = nblock_has_missing = tf.reduce_sum(missing_mask)
    nblock_empty = tf.reduce_sum(empty_mask)
    samplings['nmissing_perb'] = tf.cond( tf.greater(nblock_has_missing,0),
                    lambda: tf.reduce_sum(tmp_missing) / nblock_has_missing,
                    lambda: 0)
    samplings['nempty_perb'] = tf.reduce_sum(tmp_empty) / nblock_empty

    samplings['valid_grouped_npoint'] = bid_index__pindex_inb.shape[0].value
    samplings['nblock_perp_3d'] = self.nblock_perp_3d
    #samplings['grouped_sampling_rate'] =\
    #                  tf.cast(samplings['valid_grouped_npoint'],tf.float32) /\
    #                  tf.cast(self.npoint_grouped, tf.float32)
    #samplings['nblock_rate'] = 1.0 * self._nblock / self.nblock
    self.samplings.update(samplings)
    return bid_index__pindex_inb, point_index, block_id_unique


  def gather_grouped_xyz(self, bid_index__pindex_inb, point_index, xyz):
    #(3.1) gather grouped point index
    # gen point index: (real nblock, self._npoint_per_block, 1)
    tmp = tf.ones([self._nblock_buf_max, self._npoint_per_block], dtype=tf.int32) * (-1)
    grouped_pindex0 = tf.get_variable("grouped_pindex", initializer=tmp, trainable=False, validate_shape=True)
    grouped_pindex1 = tf.scatter_nd_update(grouped_pindex0, bid_index__pindex_inb, point_index)

    #(3.2) remove the blocks with too less points
    if not self._debug_only_blocks_few_points:
      bid_index_valid = self.valid_bid_index
      nblock_valid = self.nblock_valid
    else:
      bid_index_valid = self.invalid_bid_index
      nblock_valid = self.nblock_invalid

    #(3.3) sampling fixed number of blocks when too many blocks are provided
    tmp_nb = self._nblock - nblock_valid
    if self._shuffle:
      bid_index_sampling = tf.cond( tf.greater(tmp_nb, 0),
             lambda: tf.concat([bid_index_valid, tf.ones(tmp_nb,tf.int32) * bid_index_valid[0]],0),
             lambda: tf.contrib.framework.sort( tf.random_shuffle(bid_index_valid)[0:self._nblock] ))
    else:
      bid_index_sampling = tf.cond( tf.greater(tmp_nb, 0),
             lambda: tf.concat([bid_index_valid, tf.ones(tmp_nb,tf.int32) * bid_index_valid[0]],0),
             lambda: tf.contrib.framework.sort( bid_index_valid[0:self._nblock] ))
    grouped_pindex = tf.gather(grouped_pindex1, bid_index_sampling)
    #print(grouped_pindex[0:20,0:10])

    #(3.4) gather xyz from point index
    grouped_xyz = tf.gather(xyz, grouped_pindex)
    empty_mask = tf.less(grouped_pindex,0)

    if DEBUG:
      self.debugs['bid_index__pindex_inb'] = bid_index__pindex_inb
      self.debugs['point_index'] = point_index
      self.debugs['grouped_pindex1'] = grouped_pindex1
      self.debugs['grouped_pindex'] = grouped_pindex

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

  def grouping(self, xyz):
    '''
    xyz: (num_point0,3)
    grouped_xyz: (num_block,npoint_per_block,3)

    Search by point: for each point in xyz, find all the block d=ids
    '''
    assert len(xyz.shape) == 2
    assert xyz.shape[-1].value == 3

    bid_pindex = self.get_block_id(xyz)
    bid_index__pindex_inb, point_index, block_id_unique = self.get_bid_point_index(bid_pindex)
    grouped_xyz, empty_mask, bid_index_sampling = self.gather_grouped_xyz(
                                      bid_index__pindex_inb, point_index, xyz)
    bids_sampling, block_bottom_center = self.all_bottom_centers(block_id_unique, bid_index_sampling)
    if self._check_grouped_xyz_inblock:
      check_gs = self.check_grouped_xyz_inblock(grouped_xyz, block_bottom_center, empty_mask)
      with tf.control_dependencies(check_gs):
        grouped_xyz = tf.identity(grouped_xyz)

    if DEBUG:
      self.debugs['xyz'] = xyz
      self.debugs['grouped_xyz'] = grouped_xyz
      self.debugs['bid_index__pindex_inb'] = bid_index__pindex_inb
      self.debugs['block_id_unique'] = block_id_unique
    return grouped_xyz, empty_mask, block_bottom_center, bids_sampling


  def main(self, DATASET_NAME, filenames):
    _DATA_PARAS = get_data_shapes_from_tfrecord(filenames)

    dataset_meta = DatasetsMeta(DATASET_NAME)
    num_classes = dataset_meta.num_classes

    self.show_settings()

    with tf.Graph().as_default():
     with tf.device('/device:GPU:0'):
      dataset = tf.data.TFRecordDataset(filenames,
                                          compression_type="",
                                          buffer_size=1024*100,
                                          num_parallel_reads=1)
      batch_size = 1
      is_training = False

      dataset = dataset.prefetch(buffer_size=batch_size)
      dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
          lambda value: parse_pl_record(value, is_training, _DATA_PARAS),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=True))
      dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
      get_next = dataset.make_one_shot_iterator().get_next()
      features_next, label_next = get_next
      points_next = features_next['points'][0,:,0:3]
      grouped_xyz_next, empty_mask_next, bottom_center_next, bids_sampling_next = self.grouping(points_next)

      init = tf.global_variables_initializer()

      config = tf.ConfigProto(allow_soft_placement=True)
      with tf.Session(config=config) as sess:
        sess.run(init)
        xyzs = []
        grouped_xyzs = []
        for i in range(batch_size):
          #if DEBUG:
          #  debugs = sess.run(self.debugs)

          #points_i = sess.run(points_next)

          points_i, grouped_xyz_i, empty_mask, bottom_center_i, bids_sampling_i = \
            sess.run([points_next, grouped_xyz_next, empty_mask_next, bottom_center_next, bids_sampling_next])
          xyzs.append(np.expand_dims(points_i,0))
          grouped_xyzs.append(np.expand_dims(grouped_xyz_i,0))
          print('group OK %d'%(i))
          #continue

          valid_flag = '' if not self._debug_only_blocks_few_points else '_invalid'
          if not self._shuffle:
            valid_flag += '_NoShuffle'
          gen_plys(DATASET_NAME, i, points_i, grouped_xyz_i, bottom_center_i,\
                   bids_sampling_i, valid_flag)
        xyzs = np.concatenate(xyzs,0)
        grouped_xyzs = np.concatenate(grouped_xyzs,0)

    return xyzs, grouped_xyzs


  def main_eager(self, DATASET_NAME, filenames):
    tf.enable_eager_execution()
    #tf.enable_eager_execution()
    dataset_meta = DatasetsMeta(DATASET_NAME)
    num_classes = dataset_meta.num_classes

    dataset = tf.data.TFRecordDataset(filenames,
                                        compression_type="",
                                        buffer_size=1024*100,
                                        num_parallel_reads=1)

    batch_size = 1
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

    self.show_settings()

    xyzs = []
    grouped_xyzs = []

    for i in range(batch_size):
      #if i<1: continue

      points_i = points_next[i,:,:]
      grouped_xyz_i, empty_mask_i, bottom_center_i, bids_sampling = self.grouping(points_i)
      xyzs.append(np.expand_dims(points_i.numpy(),0))
      grouped_xyzs.append(np.expand_dims(grouped_xyz_i.numpy(),0))

      self.show_summary(i)
      continue

      valid_flag = '' if not self._debug_only_blocks_few_points else '_invalid'
      if not self._shuffle:
        valid_flag += '_NoShuffle'
      gen_plys(DATASET_NAME, i, points_i.numpy(), grouped_xyz_i.numpy(),
               bottom_center_i.numpy(), bids_sampling, valid_flag, '_E')

    xyzs = np.concatenate(xyzs,0)
    grouped_xyzs = np.concatenate(grouped_xyzs,0)
    return xyzs, grouped_xyzs


def gen_plys(DATASET_NAME, i, points, grouped_xyz, bottom_center, bids_sampling, valid_flag='', main_flag=''):
  print('bids_sampling: {}'.format(bids_sampling))
  path = '/tmp/%d_plys'%(i) + main_flag
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
  filenames = glob.glob(os.path.join(path, tmp+'.tfrecord'))
  assert len(filenames) >= 1

  #width = [0.4,0.4,0.4]
  #stride = [0.2,0.2,0.2]
  #nblock = 480
  #npoint_per_block = 256
  max_nblock = 800

  width = [0.1,0.1,0.1]
  #stride = [0.1,0.1,0.1]
  stride = [0.05,0.05,0.05]
  nblock = 2000
  npoint_per_block = 6
  max_nblock = 3000

  block_group_sampling = BlockGroupSampling(width, stride, nblock,
                                            npoint_per_block, max_nblock)
  if len(sys.argv) > 1:
    main_flag = sys.argv[1]
  else:
    main_flag = 'g'
    #main_flag = 'eg'
    main_flag = 'e'
  print(main_flag)

  if 'e' in main_flag:
    xyzs_E, grouped_xyzs_E = block_group_sampling.main_eager(DATASET_NAME, filenames)
  if 'g' in main_flag:
    xyzs, grouped_xyzs = block_group_sampling.main(DATASET_NAME, filenames)

  if main_flag=='eg':
    batch_size = min(xyzs.shape[0], xyzs_E.shape[0])
    for b in range(batch_size):
      assert (xyzs_E[b] == xyzs[b]).all(), 'time %d xyz different'%(b)
      print('time %d xyzs of g and e is same'%(b))
      assert (grouped_xyzs_E[b] == grouped_xyzs[b]).all(), 'time %d grouped_xyzs differernt'%(b)
      print('time %d grouped_xyzs of g and e is same, CHECK if shuffle is turned off'%(b))

