# July 2018 xyz
import tensorflow as tf
import glob, os
import numpy as np

class BlockGroupSampling():
  def __init__(self, width, stride, points_per_block):
    '''
      width = [1,1,1]
      stride = [1,1,1]
      points_per_block = 32
    '''
    self.width = np.array(width, dtype=np.float32)
    self.stride = np.array(stride, dtype=np.float32)
    self.points_per_block = points_per_block
    assert self.width.size == self.stride.size == 3

  def get_block_id(xyz):
    '''
    (1) Get the responding block index of each point
    xyz: (num_point0, 3)
      block_index: (num_point0, block_num_per_point, 3)
    block_id: (num_point0, block_num_per_point, 1)
    '''
    # (1.1) lower and upper block index bound
    self.num_point0 = num_point0 = xyz.shape[0].value
    low_b_index = (xyz - self.width) / self.stride
    up_b_index = xyz / self.stride

    low_b_index += 1e-7 # invoid low_b_index is exactly int
    low_b_index_fixed = tf.cast(tf.ceil(low_b_index), tf.int32)    # include
    up_b_index_fixed = tf.cast(tf.floor(up_b_index), tf.int32) + 1 # not include

    #(1.2) the block number for each point should be equal
    block_num_per_point = up_b_index_fixed - low_b_index_fixed
    tmp0 = tf.reduce_max(block_num_per_point)
    tmp1 = tf.reduce_min(block_num_per_point)
    tf.assert_equal( tmp0, tmp1)
    self.block_num_per_point = block_num_per_point = tmp0

    tmp2 = tf.range(0, block_num_per_point, 1)
    tmp2 = tf.reshape(tmp2, (1,-1,1))
    tmp2 = tf.tile(tmp2, [num_point0, 1, 3])

    block_index = tf.expand_dims(low_b_index_fixed, 1)
    # (num_point0, block_num_per_point, 3)
    block_index = tf.tile(block_index, [1, block_num_per_point, 1]) + tmp2

    # (1.3) block index -> block id
    min_block_index = tf.reduce_min(block_index, -1)
    max_block_index = tf.reduce_max(block_index, -1)
    block_size = max_block_index - min_block_index
    # (num_point0, block_num_per_point, 1)
    block_id = block_index[:,:,0] * block_size[1] * block_size[2] + \
               block_index[:,:,1] * block_size[2] + block_index[:,:,2]

    # (1.4) concat block id and point index
    tmp3 = tf.reshape( tf.range(0, num_point0, 1), (-1,1,1))
    tmp3 = tf.tile(tmp, [1,block_num_per_point,1])
    # (num_point0, block_num_per_point, 2)
    block_id_point_index = tf.concat([block_id, tmp3], 2)
    # (num_point0 * block_num_per_point, 2)
    block_id_point_index = tf.reshape(block_id_point_index, (-1,2))

    return block_id_point_index

  def group_point_index(block_id_point_index0):
    '''
    # block_id_point_index0: (num_point0 * block_num_per_point, 2)
                                [block_id, point_index]
    '''
    #block_id = block_id_point_index[:,0]
    #point_index = block_id_point_index[:,1]
    # (2.1) sort by block id
    sort_indices = tf.contrib.framwork.argsort(block_id[:,0], axis = 0)
    block_id_point_index = tf.gather(block_id_point_index0, sort_indices, axis=0)

    #(2.2) block id -> block id index
    # get valid block num
    block_id = block_id_point_index[:,0]
    block_id_unique = tf.unique( block_id ) # should also be sorted already
    valid_block_num = tf.shape(block_id_unique)[0].value
    blockid_index_temple = tf.range(0, valid_block_num)


  def grouping(self, xyz):
    '''
    xyz: (num_point0,3)
    grouped_xyz: (num_block,points_per_block,3)

    Search by point: for each point in xyz, find all the block d=ids
    '''
    assert len(xyz.shape) == 2
    assert xyz.shape[-1].value == 3

    block_id_point_index = self.get_block_id(xyz)

    # (2)


    # sort by block id:[block id, point index]
    # (num_point0 * blocks_per_point, 2)

    # [point index]
    # (num_point0 * blocks_per_point, 1)

    # [block index]
    # (num_point0 * blocks_per_point, 1)

    # [block index, point index in block] *****
    # (num_point0 * blocks_per_point, 2)

    # grouped point index: neg temple
    # (num_block_BF, points_per_block_BF, 1)

    # scatter: neg & pos
    # (num_block_BF, points_per_block_BF, 1)

    # sampling
    # (num_block, points_per_block, 1)

    # gather
    # (num_block, points_per_block, 3)




# (num_point0, blocks_per_point, 3)
# (num_point0, blocks_per_point, 4)
# (num_point0 * blocks_per_point, 4)
# (num_point0 * blocks_per_point, 1)  (num_point0 * blocks_per_point, 3)
# (num_point0 * blocks_per_point, 3)  (num_point0 * blocks_per_point, 1)
# (num_point0 * blocks_per_point, 3)  (num_point0 * blocks_per_point, 1)
# (num_block_BF, points_per_block_BF, 3) neg
# (num_block_BF, points_per_block_BF, 3) neg & pos
# (num_block, points_per_block, 3) neg & pos



    grouped_xyz = tf.sactter_nd() # ()
    return grouped_xyz

  def main(self):
    tf.enable_eager_execution()

    from utils.dataset_utils import parse_pl_record
    from datasets.all_datasets_meta.datasets_meta import DatasetsMeta

    DATASET_NAME = 'MODELNET40'
    path = '/home/z/Research/dynamic_pointnet/data/MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'
    filenames = glob.glob(os.path.join(path,'*.tfrecord'))
    assert len(filenames) > 0


    DatasetsMeta = DatasetsMeta(DATASET_NAME)
    num_classes = DatasetsMeta.num_classes
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
    points_next = features_next['points']

    for i in range(2):
      points_i = points_next[i]
      grouped_xyz_next = self.grouping(points_i[:,0:3])
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    #  with tf.Session() as sess:
    #    #features, object_label = sess.run(get_next)
    #    points, grouped_xyz = sess.run([points_next, grouped_xyz_next])
    #    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #    pass


if __name__ == '__main__':
  width = [0.2,0.2,0.2]
  stride = [0.2,0.2,0.2]
  points_per_block = 32
  block_group_sampling = BlockGroupSampling(width, stride, points_per_block)
  block_group_sampling.main()

