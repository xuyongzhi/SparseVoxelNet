# July 2018 xyz
import tensorflow as tf
import glob, os, sys
import numpy as np
from utils.dataset_utils import parse_pl_record
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
from utils.ply_util import create_ply_dset, draw_blocks_by_bottom_center

DEBUG = True
TMP_IGNORE = True

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
    tf.assert_equal(block_index, block_index_C)
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
  def __init__(self, width, stride, nblock, npoint_per_block):
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
    self._npoint_per_block_min = max(int(npoint_per_block*0.05), 2)
    print('_npoint_per_block_min:{}'.format(self._npoint_per_block_min))
    assert self._width.size == self._stride.size == 3
    self._nblock_buf_max = 800

    self.samplings = {}

    # debug flags
    self._shuffle = True
    self._check_block_index = True
    self._check_grouped_xyz_inblock = True

    self._debug_only_blocks_few_points = False
    self.debugs = {}


  def show_settings(self):
    items = ['width', 'stride', 'npoint_per_block']
    for item in items:
      if hasattr(self, item):
        print('{}:{}'.format(item, getattr(self,item)))


  def show_summary(self):
    #for item in self.samplings:
    #  print('{}:{}'.format(item, self.samplings[item]))

    def summary_str(item, config, real):
      return '{}: {}/{}, {:.3f}\n'.format(item, real, config,
                      1.0*tf.cast(real,tf.float32)/tf.cast(config,tf.float32))
    summary = '\tReal / Config\n'
    summary += summary_str('nblock', self._nblock, self.nblock)
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

  def check_block_index(self, block_index, xyz):
    block_bottom_center = self.block_index_to_scope(block_index)
    block_bottom = block_bottom_center[:,0,0:3]
    block_center = block_bottom_center[:,0,3:6]
    block_top = block_bottom + 2 * (block_center - block_bottom)
    tf.assert_less(xyz, block_top)
    tf.assert_less_equal(block_bottom, xyz)

  def check_grouped_xyz_inblock(self, grouped_xyz, block_bottom_center, empty_mask):
    block_bottom_center = tf.expand_dims(block_bottom_center, 1)
    block_bottom = block_bottom_center[:,:,0:3]
    block_center = block_bottom_center[:,:, 3:6]
    block_top = block_bottom + 2 * (block_center - block_bottom)
    top_check = tf.reduce_all(tf.less(grouped_xyz, block_top), -1)
    bottom_check = tf.reduce_all(tf.less_equal(block_bottom, grouped_xyz), -1)
    top_check = tf.logical_or(top_check, empty_mask)
    bottom_check = tf.logical_or(bottom_check, empty_mask)
    tf.Assert(tf.reduce_all(top_check), [top_check])
    tf.Assert(tf.reduce_all(bottom_check), [bottom_check])

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
    tmp_max_i = tf.reduce_max(nblocks_per_points, axis=0)
    tmp_max = tf.cast(tmp_max_i, tf.float32)
    tmp_min = tf.cast(tf.reduce_min(nblocks_per_points, axis=0), tf.float32)

    # *****
    # block index is got by linear ranging from low bound by a fixed offset for
    # all the points. This is based on: num block per point is the same for all!
    # Check: No points will belong to greater nblocks. But few may belong to
    # smaller blocks: err_npoints
    tmp_mean = tf.reduce_mean(tf.cast(nblocks_per_points,tf.float32), axis=0)
    # (a) The most are same as tmp_max
    # (b) Very few may be tmp_max - 1
    tf.assert_less(tmp_max - tmp_mean, 0.001,
                  message="Increase low_b_index_offset if tmp_max > tmp_mean")
    tf.assert_less(tmp_mean - tmp_min, 1.0)
    tmp_err_np = tf.cast(tf.not_equal(nblocks_per_points,tmp_max_i),tf.int32)
    tmp_err_np = tf.reduce_sum(tmp_err_np, 1)
    err_npoints = tf.reduce_sum(tf.cast(tf.not_equal(tmp_err_np, 0),tf.int32))
    if TMP_IGNORE:
      tf.assert_less(err_npoints, 3)
    else:
      tf.assert_less(err_npoints, 1)
    # *****

    self.nblocks_per_point = tmp_max_i
    self.nblock_perp_3d = tf.reduce_prod(self.nblocks_per_point)
    self.npoint_grouped = self.num_point0 * self.nblock_perp_3d

    pairs_3d = permutation_combination_3D(self.nblocks_per_point)
    pairs_3d = tf.tile( tf.expand_dims(pairs_3d, 0), [self.num_point0,1,1])

    block_index = tf.expand_dims(low_b_index_fixed, 1)
    # (num_point0, nblock_perp_3d, 3)
    block_index = tf.tile(block_index, [1, self.nblock_perp_3d, 1])
    block_index += pairs_3d

    if self._check_block_index:
      self.check_block_index(block_index, xyz)

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
    if not self._shuffle:
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
    tf.assert_equal(tf.reduce_sum(npoint_per_block), self.npoint_grouped)
    self.nblock = nblock = tf.shape(block_id_unique)[0]

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
    tf.assert_greater(self._nblock_buf_max, self.nblock)
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
    if not self._shuffle:
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
      self.check_grouped_xyz_inblock(grouped_xyz, block_bottom_center, empty_mask)


    if DEBUG:
      self.debugs['xyz'] = xyz
      self.debugs['grouped_xyz'] = grouped_xyz
      self.debugs['bid_index__pindex_inb'] = bid_index__pindex_inb
      self.debugs['block_id_unique'] = block_id_unique
    return grouped_xyz, empty_mask, block_bottom_center, bids_sampling


  def main(self):
    tf.enable_eager_execution()

    DATASET_NAME = 'MODELNET40'
    path = '/home/z/Research/dynamic_pointnet/data/MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'
    filenames = glob.glob(os.path.join(path,'*.tfrecord'))
    assert len(filenames) > 0

    _DATA_PARAS = get_data_shapes_from_tfrecord(filenames[0:1])

    dataset_meta = DatasetsMeta(DATASET_NAME)
    num_classes = dataset_meta.num_classes

    with tf.Graph().as_default():
     with tf.device('/device:GPU:0'):
      dataset = tf.data.TFRecordDataset(filenames,
                                          compression_type="",
                                          buffer_size=1024*100,
                                          num_parallel_reads=1)
      batch_size = 3
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
        for i in range(batch_size):
          #if DEBUG:
          #  debugs = sess.run(self.debugs)

          #points_i = sess.run(points_next)

          points_i, grouped_xyz_i, empty_mask, bottom_center_i, bids_sampling_i = \
            sess.run([points_next, grouped_xyz_next, empty_mask_next, bottom_center_next, bids_sampling_next])
          print('group OK %d'%(i))
          continue

          valid_flag = '' if not self._debug_only_blocks_few_points else '_invalid'
          gen_plys(DATASET_NAME, i, points_i, grouped_xyz_i, bottom_center_i,\
                   bids_sampling_i, valid_flag)


          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass


  def main_eager(self):
    tf.enable_eager_execution()

    from utils.dataset_utils import parse_pl_record
    from datasets.all_datasets_meta.datasets_meta import DatasetsMeta

    DATASET_NAME = 'MODELNET40'
    path = '/home/z/Research/dynamic_pointnet/data/MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'
    filenames = glob.glob(os.path.join(path,'*.tfrecord'))
    assert len(filenames) > 0

    dataset_meta = DatasetsMeta(DATASET_NAME)
    num_classes = dataset_meta.num_classes

    dataset = tf.data.TFRecordDataset(filenames,
                                        compression_type="",
                                        buffer_size=1024*100,
                                        num_parallel_reads=1)

    batch_size = 2
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

    for i in range(batch_size):
      #if i<1: continue

      points_i = points_next[i,:,:]
      grouped_xyz_next, empty_mask_next, bottom_center_next, bids_sampling = self.grouping(points_i)

      self.show_summary()

      valid_flag = '' if not self._debug_only_blocks_few_points else '_invalid'
      gen_plys(DATASET_NAME, i, points_i.numpy(), grouped_xyz_next.numpy(),
               bottom_center_next.numpy(), bids_sampling, valid_flag, '_E')

      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass


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
  width = [0.4,0.4,0.4]
  stride = [0.2,0.2,0.2]
  nblock = 480
  npoint_per_block = 256

  #width = [0.2,0.2,0.2]
  #stride = [0.2,0.2,0.2]
  #nblock = 200
  #npoint_per_block = 32

  block_group_sampling = BlockGroupSampling(width, stride, nblock,
                                            npoint_per_block)
  if len(sys.argv) > 1:
    main_flag = sys.argv[1]
  else:
    main_flag = 'g'
  print(main_flag)
  if main_flag == 'g':
    block_group_sampling.main()
  else:
    block_group_sampling.main_eager()

