# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, glob
import sys
import tarfile
import numpy as np

from six.moves import urllib
import tensorflow as tf

LABELS_FILENAME = 'labels.txt'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))

def decode_raw(encoded_string, org_dtype, org_shape):
  out_tensor = tf.reshape(tf.decode_raw(encoded_string, org_dtype),
                          org_shape) # Shape information is lost
  return out_tensor


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def pl_bxm_to_tfexample(points, object_label, sg_all_bidxmaps, bidxmaps_flat, fmap_neighbor_idis):
  assert points.dtype == np.float32
  assert sg_all_bidxmaps.dtype == np.int32
  assert bidxmaps_flat.dtype == np.int32
  assert fmap_neighbor_idis.dtype == np.float32

  points_bin = points.tobytes()
  points_shape_bin = np.array(points.shape, np.int32).tobytes()

  sg_all_bidxmaps_bin = sg_all_bidxmaps.tobytes()
  sg_all_bidxmaps_shape_bin = np.array(sg_all_bidxmaps.shape, np.int32).tobytes()
  bidxmaps_flat_bin = bidxmaps_flat.tobytes()
  bidxmaps_flat_shape_bin = np.array(bidxmaps_flat.shape, np.int32).tobytes()
  fmap_neighbor_idis_bin = fmap_neighbor_idis.tobytes()
  fmap_neighbor_idis_shape_bin = np.array(fmap_neighbor_idis.shape, np.int32).tobytes()

  example = tf.train.Example(features=tf.train.Features(feature={
      'points/encoded': bytes_feature(points_bin),
      'points/shape': bytes_feature(points_shape_bin),
      'object/label': int64_feature(object_label),
      'sg_all_bidxmaps/encoded': bytes_feature(sg_all_bidxmaps_bin),
      'sg_all_bidxmaps/shape': bytes_feature(sg_all_bidxmaps_shape_bin),
      'bidxmaps_flat/encoded': bytes_feature(bidxmaps_flat_bin),
      'bidxmaps_flat/shape': bytes_feature(bidxmaps_flat_shape_bin),
      'fmap_neighbor_idis/encoded': bytes_feature(fmap_neighbor_idis_bin),
      'fmap_neighbor_idis/shape': bytes_feature(fmap_neighbor_idis_shape_bin)
  }))
  return example

def data_meta_to_tfexample(point_idxs):
  data_eles = ['xyz', 'nxnynz', 'color', 'intensity']
  feature_map = {}
  point_idxs_bin = {}
  for ele in data_eles:
    if ele not in point_idxs:
      point_idxs_bin[ele] = np.array([],np.int32).tobytes()
    else:
      point_idxs_bin[ele] = np.array(point_idxs[ele],np.int32).tobytes()
    feature_map['point_idxs/%s'%(ele)] = bytes_feature( point_idxs_bin[ele] )

  example = tf.train.Example(features=tf.train.Features(feature=feature_map))

def write_pl_bxm_tfrecord(bxm_tfrecord_writer, tfrecord_meta_writer,\
                        datasource_name, points, point_idxs, object_labels,\
                        sg_all_bidxmaps, bidxmaps_flat, fmap_neighbor_idis):
  if tfrecord_meta_writer!=None:
    example = data_meta_to_tfexample(point_idxs)
    tfrecord_meta_writer.write(example)

  num_gblocks = sg_all_bidxmaps.shape[0]
  assert num_gblocks == points.shape[0]
  for j in range(num_gblocks):
    example = pl_bxm_to_tfexample(points[j], object_labels[j], sg_all_bidxmaps[j], bidxmaps_flat[j], fmap_neighbor_idis[j])
    bxm_tfrecord_writer.write(example.SerializeToString())


def parse_pl_record(tfrecord_serialized, is_training, data_net_configs=None):
    #if data_net_configs!=None:
    #  from aug_data_tf import aug_data, tf_Rz
    #  R = tf_Rz(1)
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    feature_map = {
        'object/label': tf.FixedLenFeature([], tf.int64),
        'points/shape': tf.FixedLenFeature([], tf.string),
        'points/encoded': tf.FixedLenFeature([], tf.string),
        'sg_all_bidxmaps/encoded': tf.FixedLenFeature([], tf.string),
        'sg_all_bidxmaps/shape': tf.FixedLenFeature([], tf.string),
        'bidxmaps_flat/encoded': tf.FixedLenFeature([], tf.string),
        'bidxmaps_flat/shape': tf.FixedLenFeature([], tf.string),
        'fmap_neighbor_idis/encoded': tf.FixedLenFeature([], tf.string),
        'fmap_neighbor_idis/shape': tf.FixedLenFeature([], tf.string),
    }
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features=feature_map,
                                                name='pl_features')

    points = tf.decode_raw(tfrecord_features['points/encoded'], tf.float32)
    if data_net_configs == None:
      points_shape = tf.decode_raw(tfrecord_features['points/shape'], tf.int32)
    else:
      points_shape = data_net_configs['points']
    # the image tensor is flattened out, so we have to reconstruct the shape
    points = tf.reshape(points, points_shape)

    object_label = tf.cast(tfrecord_features['object/label'], tf.int32)
    object_label = tf.expand_dims(object_label,0)

    sg_all_bidxmaps = tf.decode_raw(tfrecord_features['sg_all_bidxmaps/encoded'], tf.int32)
    if data_net_configs == None:
      sg_all_bidxmaps_shape = tf.decode_raw(tfrecord_features['sg_all_bidxmaps/shape'], tf.int32)
    else:
      sg_all_bidxmaps_shape = data_net_configs['sg_all_bidxmaps']
    sg_all_bidxmaps = tf.reshape(sg_all_bidxmaps, sg_all_bidxmaps_shape)
    if data_net_configs != None:
      sg_bidxmaps, b_bottom_centers_mm = extract_sg_bidxmap(
                          sg_all_bidxmaps, data_net_configs['sg_bm_extract_idx'])

    bidxmaps_flat = tf.decode_raw(tfrecord_features['bidxmaps_flat/encoded'], tf.int32)
    if data_net_configs == None:
      bidxmaps_flat_shape = tf.decode_raw(tfrecord_features['bidxmaps_flat/shape'], tf.int32)
    else:
      bidxmaps_flat_shape = data_net_configs['bidxmaps_flat']
    bidxmaps_flat = tf.reshape(bidxmaps_flat, bidxmaps_flat_shape)

    fmap_neighbor_idis = tf.decode_raw(tfrecord_features['fmap_neighbor_idis/encoded'], tf.float32)
    if data_net_configs == None:
      fmap_neighbor_idis_shape = tf.decode_raw(tfrecord_features['fmap_neighbor_idis/shape'], tf.int32)
    else:
      fmap_neighbor_idis_shape = data_net_configs['fmap_neighbor_idis']
    fmap_neighbor_idis = tf.reshape(fmap_neighbor_idis, fmap_neighbor_idis_shape)

    features = {}
    features['bidxmaps_flat'] = bidxmaps_flat
    features['fmap_neighbor_idis'] = fmap_neighbor_idis

    if is_training and data_net_configs != None and data_net_configs['aug_types']!='none':
      from aug_data_tf import aug_main
      features['raw_points'] = points
      points, b_bottom_centers_mm, augs = aug_main(points, b_bottom_centers_mm,
                  data_net_configs['aug_types'],
                  data_net_configs['data_idxs'])
      features['augs'] = augs

    if data_net_configs != None:
      features['sg_bidxmaps'] = sg_bidxmaps
      features['b_bottom_centers_mm'] = b_bottom_centers_mm
    else:
      features['sg_all_bidxmaps'] = sg_all_bidxmaps
    features['points'] = points

    return features, object_label

def extract_sg_bidxmap(sg_all_bidxmaps, sg_bm_extract_idx):
  cascade_num = sg_bm_extract_idx.shape[0] - 1
  sg_bidxmaps = {}
  b_bottom_centers_mm = {}

  for k in range(cascade_num):
    start = sg_bm_extract_idx[k]
    end = sg_bm_extract_idx[k+1]
    sg_bidxmap_k = sg_all_bidxmaps[ start[0]:end[0],0:end[1] ]
    block_bottom_center_mm = sg_all_bidxmaps[ start[0]:end[0],end[1]:end[1]+6 ]
    sg_bidxmaps[k] = sg_bidxmap_k
    b_bottom_centers_mm[k] = block_bottom_center_mm
  return sg_bidxmaps, b_bottom_centers_mm

def read_tfrecord():
  import ply_util
  path = '/home/z/Research/dynamic_pointnet/data/MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'
  filenames = glob.glob(os.path.join(path,'*.tfrecord'))
  assert len(filenames) > 0

  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset(filenames,
                                      compression_type="",
                                      buffer_size=1024*100,
                                      num_parallel_reads=5)

    batch_size = 10
    is_training = False

    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        lambda value: parse_pl_record(value, is_training),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=True))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    with tf.Session() as sess:
      features, object_label = sess.run(dataset.make_one_shot_iterator().get_next())
      print(features['points'][0])
      print(object_label)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      for i in range(batch_size):
        plyfn = '/tmp/tfrecord_%d.ply'%(i)
        ply_util.create_ply(features['points'][i], plyfn)


def merge_tfrecord( filenames, merged_filename ):
  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset(filenames,
                                      compression_type="",
                                      buffer_size=1024*100,
                                      num_parallel_reads=5)

    batch_size = 50
    is_training = False

    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator().get_next()
    num_blocks = 0
    with tf.Session() as sess:
      with tf.python_io.TFRecordWriter(merged_filename) as tfrecord_writer:
        print('merging tfrecord: {}'.format(merged_filename))
        while True:
          try:
            ds = sess.run(iterator)
            for ds_i in ds:
              tfrecord_writer.write(ds_i)
            num_blocks += len(ds)
            if num_blocks%100==0:
              print('merging {} blocks'.format(num_blocks))
          except:
            print('totally {} blocks, merge tfrecord OK:\n\t{}'.format(num_blocks,merged_filename))
            break



if __name__ == '__main__':
  #test_encode_raw()
  read_tfrecord()
  #merge_tfrecord()








################################################################################
#                              unused
################################################################################
def encode_to_raw(in_tensor):
  # in_tensor: a tensor with any dtype
  character_lookup = tf.constant([chr(i) for i in range(256)])
  encoded_string = tf.reduce_join(
      tf.gather(character_lookup,
                tf.cast(tf.bitcast(in_tensor, tf.uint8), tf.int32)))
  org_dtype = in_tensor.dtype
  org_shape = in_tensor.shape
  return encoded_string, org_dtype, org_shape

def test_encode_raw():
  with tf.Graph().as_default():
    starting_dtype = tf.float32
    starting_tensor = tf.random_normal(shape=[10, 10], stddev=1e5,
                                      dtype=starting_dtype)

    encoded_string, org_dtype, org_shape = encode_to_raw(starting_tensor)
    back_to_tensor = decode_raw(encoded_string, org_dtype, org_shape)
    with tf.Session() as session:
      before, after = session.run([starting_tensor, back_to_tensor])
      print(before - after)

def process_pl_to_uint16( points, point_idxs ):
  # Preprocess point cloud and transfer all to uint16
  # points: [num_gblocks, num_point, channels]
  # point_idxs: {'xyz':[0,1,2], 'nxnynz':[3,4,5]}

  assert points.ndim == 3
  for ele in point_idxs:
    idx = point_idxs[ele]
    if ele == 'xyz':
      # set min as 0, make all positive
      min_xyz = points[:,:,idx].min(axis=-1, keepdims=True)
      points[:,:,idx] -= min_xyz

      # from m to mm
      points[:,:,idx] *= 1000

    elif ele=='nxnynz':
      points[:,:,idx] += 1
      points[:,:,idx] *= 10000
    else:
      raise NotImplementedError

  # check scope inside uint16
  max_data = points.max()
  min_data = points.min()
  assert max_data <= 65535
  assert min_data >= 0

  points = points.astype( np.uint16 )

  # reshape to make image like array
  data_ele_idxs = {}
  for i, ele in enumerate(point_idxs):
    assert len(point_idxs[ele]) == 3
    data_ele_idxs[ele] = point_idxs[ele][0]//3
  points = points.reshape( [points.shape[0], points.shape[1], -1, 3] )
  return points, data_ele_idxs

def read_from_tfrecord(filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    return parse_pl_record(tfrecord_serialized)


def read_tfrecord_queuerunner(filenames=None):
  if filenames == None:
    path = '/home/z/Research/dynamic_pointnet/data/MODELNET40__H5F/ORG_tfrecord/4096_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'
    filenames = glob.glob(os.path.join(path,'*.tfrecord'))
  points_, object_label_, sg_all_bidxmaps_, fmap_neighbor_idis_ = read_from_tfrecord(filenames)

  with tf.Session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      for i in range(3):
        points, object_label, sg_all_bidxmaps, fmap_neighbor_idis = sess.run([points_, object_label_, sg_all_bidxmaps_, fmap_neighbor_idis_])
        print(points[0])
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
      coord.request_stop()
      coord.join(threads)
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  return points, object_label, sg_all_bidxmaps, fmap_neighbor_idis


