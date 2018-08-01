# July 2018 xyz
import tensorflow as tf
import glob, os
import numpy as np
from utils.dataset_utils import parse_pl_record
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
from utils.ply_util import create_ply_dset

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
    assert self._width.size == self._stride.size == 3
    self.samplings = {}


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
    if DEBUG:
      up_b_index_fixed += 1

    #(1.2) the block number for each point should be equal
    nblocks_per_points = up_b_index_fixed - low_b_index_fixed
    tmp_max = tf.reduce_max(nblocks_per_points, axis=0)
    tmp_min = tf.reduce_min(nblocks_per_points, axis=0)

    #tmp_mean = tf.reduce_mean(tf.cast(nblocks_per_points,tf.float32), axis=0)
    tf.assert_equal( tmp_max, tmp_min,
                  message="Increase low_b_index_offset if tmp_max > tmp_mean")
    self.nblocks_per_point = tmp_max
    self.nblock_perp_3d = tf.reduce_prod(self.nblocks_per_point)
    self.npoint_grouped = self.num_point0 * self.nblock_perp_3d

    pairs_3d = permutation_combination_3D(self.nblocks_per_point)
    pairs_3d = tf.tile( tf.expand_dims(pairs_3d, 0), [self.num_point0,1,1])

    block_index = tf.expand_dims(low_b_index_fixed, 1)
    # (num_point0, nblock_perp_3d, 3)
    block_index = tf.tile(block_index, [1, self.nblock_perp_3d, 1])
    block_index += pairs_3d

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

    #(2.3) Get point index per block
    #      Based on: all points belong to same block is together
    self.nblock = nblock = tf.shape(block_id_unique)[0]
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
    samplings['nmissing_perb'] = tf.reduce_sum(tmp_missing) / nblock_has_missing
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
    nblock_real = self.nblock
    nblock_ = tf.maximum(nblock_real, self._nblock)
    tmp = tf.ones([nblock_, self._npoint_per_block], dtype=tf.int32) * (-1)
    grouped_pindex = tf.get_variable("grouped_pindex", initializer=tmp, trainable=False, validate_shape=False)
    grouped_pindex = tf.scatter_nd_update(grouped_pindex, bid_index__pindex_inb, point_index)

    #(3.2) sampling fixed number of blocks when too many blocks are provided
    bid_index_sampling0 = tf.random_shuffle(tf.range(nblock_))[0:self._nblock]
    bid_index_sampling = tf.cond( nblock_real <= self._nblock,
                          lambda: tf.range(nblock_real),
                          lambda: tf.contrib.framework.sort(bid_index_sampling0))
    grouped_pindex = tf.cond( nblock_real <= self._nblock,
            lambda: grouped_pindex,
            lambda: tf.gather(grouped_pindex, bid_index_sampling) )
    #print(grouped_pindex[0:20,0:10])

    #(3.3) gather xyz from point index
    grouped_xyz = tf.gather(xyz, grouped_pindex)
    empty_mask = tf.less(grouped_pindex,0)

    return grouped_xyz, empty_mask, bid_index_sampling


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

    # get grouped blocks center
    bids_sampling = tf.gather(block_id_unique, bid_index_sampling)
    bid_index = block_id_to_block_index(bids_sampling, self.block_size)

    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return grouped_xyz, empty_mask


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
      points_next = features_next['points']
      grouped_xyz_next, empty_mask_next = self.grouping(points_next[0,:,0:3])

      init = tf.initialize_all_variables()

      with tf.Session() as sess:
        sess.run(init)
        for i in range(5):
          #points_i, = sess.run([points_next])
          points_i, grouped_xyz_i, empty_mask = sess.run([points_next, grouped_xyz_next, empty_mask_next])

          ply_fn = '/tmp/%d_points.ply'%(i)
          create_ply_dset(DATASET_NAME, points_i, ply_fn,
                          extra='random_same_color')
          ply_fn = '/tmp/%d_grouped_points.ply'%(i)
          create_ply_dset(DATASET_NAME, grouped_xyz_i, ply_fn,
                          extra='random_same_color')
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

    batch_size = 5
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
      if i<1: continue

      points_i = points_next[i,:,:]
      grouped_xyz_next, empty_mask_next = self.grouping(points_i)

      self.show_summary()

      ply_fn = '/tmp/E%d_points.ply'%(i)
      create_ply_dset(DATASET_NAME, points_i.numpy(), ply_fn,
                      extra='random_same_color')
      ply_fn = '/tmp/E%d_grouped_points.ply'%(i)
      create_ply_dset(DATASET_NAME, grouped_xyz_next.numpy(), ply_fn,
                      extra='random_same_color')

      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    #  with tf.Session() as sess:
    #    #features, object_label = sess.run(get_next)
    #    points, grouped_xyz = sess.run([points_next, grouped_xyz_next])
    #    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #    pass


if __name__ == '__main__':
  width = [0.4,0.4,0.4]
  stride = [0.2,0.2,0.2]
  nblock = 256
  npoint_per_block = 156
  block_group_sampling = BlockGroupSampling(width, stride, nblock,
                                            npoint_per_block)
  #block_group_sampling.main()
  block_group_sampling.main_eager()

