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


def pl_bxm_to_tfexample(points, object_label, sg_all_bidxmaps, bidxmaps_flat, fmap_neighbor_idis, no_bidxmap=False):
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

  feature_map = {
      'points/encoded': bytes_feature(points_bin),
      'points/shape': bytes_feature(points_shape_bin),
      'object/label': int64_feature(object_label) }

  feature_map1 = {
      'sg_all_bidxmaps/encoded': bytes_feature(sg_all_bidxmaps_bin),
      'sg_all_bidxmaps/shape': bytes_feature(sg_all_bidxmaps_shape_bin),
      'bidxmaps_flat/encoded': bytes_feature(bidxmaps_flat_bin),
      'bidxmaps_flat/shape': bytes_feature(bidxmaps_flat_shape_bin),
      'fmap_neighbor_idis/encoded': bytes_feature(fmap_neighbor_idis_bin),
      'fmap_neighbor_idis/shape': bytes_feature(fmap_neighbor_idis_shape_bin) }
  if not no_bidxmap:
    feature_map.update(feature_map1)

  example = tf.train.Example(features=tf.train.Features(feature=feature_map))

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
                        sg_all_bidxmaps, bidxmaps_flat, fmap_neighbor_idis, \
                        no_bidxmap=False):
  if tfrecord_meta_writer!=None:
    example = data_meta_to_tfexample(point_idxs)
    tfrecord_meta_writer.write(example)

  num_gblocks = sg_all_bidxmaps.shape[0]
  assert num_gblocks == points.shape[0]
  for j in range(num_gblocks):
    example = pl_bxm_to_tfexample(points[j], object_labels[j], sg_all_bidxmaps[j], bidxmaps_flat[j], fmap_neighbor_idis[j])
    bxm_tfrecord_writer.write(example.SerializeToString())


def pc_normalize(points):
  has_normal = points.shape[-1].value == 6
  points_xyz = points[:,0:3]
  if has_normal:
    points_normal = points[:,3:6]
  centroid = tf.reduce_mean(points_xyz, axis=0)
  points_xyz -= centroid
  m = tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.pow(points_xyz, 2),axis=1)))
  points_xyz = points_xyz / m
  if has_normal:
    points_normed = tf.concat([points_xyz, points_normal], -1)
  else:
    points_normed = points_xyz
  return points_normed


def parse_pl_record(tfrecord_serialized, is_training, data_shaps=None, bsg=None):
    from aug_data_tf import aug_main, aug_views
    #if data_shaps!=None:
    #  from aug_data_tf import aug_data, tf_Rz
    #  R = tf_Rz(1)
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    feature_map = {
        'object/label': tf.FixedLenFeature([], tf.int64),
        'points/shape': tf.FixedLenFeature([], tf.string),
        'points/encoded': tf.FixedLenFeature([], tf.string),
    }
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features=feature_map,
                                                name='pl_features')

    object_label = tf.cast(tfrecord_features['object/label'], tf.int32)
    object_label = tf.expand_dims(object_label,0)

    points = tf.decode_raw(tfrecord_features['points/encoded'], tf.float32)
    if data_shaps == None:
      points_shape = tf.decode_raw(tfrecord_features['points/shape'], tf.int32)
    else:
      points_shape = data_shaps['points']
    # the image tensor is flattened out, so we have to reconstruct the shape
    points = tf.reshape(points, points_shape)
    #if data_shaps != None:
    #  points = pc_normalize(points)

    # ------------------------------------------------
    #             data augmentation
    features = {}
    b_bottom_centers_mm = []
    if is_training:
      if data_shaps != None and data_shaps['aug_types']!='none':
        points, b_bottom_centers_mm, augs = aug_main(points, b_bottom_centers_mm,
                    data_shaps['aug_types'],
                    data_shaps['data_idxs'])
        #features['augs'] = augs
    else:
      if data_shaps!=None and 'eval_views' in data_shaps and data_shaps['eval_views'] > 1:
        #features['eval_views'] = data_shaps['eval_views']
        points, b_bottom_centers_mm, augs = aug_views(points, b_bottom_centers_mm,
                    data_shaps['eval_views'],
                    data_shaps['data_idxs'])
    features['points'] = points
    # ------------------------------------------------
    #             grouping and sampling on line
    if bsg!=None:
      grouped_xyz, empty_mask, block_bottom_center, others = bsg.grouping(points[:,0:3])
      features['grouped_xyz'] = grouped_xyz
      features['empty_mask'] = empty_mask
      features['block_bottom_center'] = block_bottom_center
      features['others'] = others

    return features, object_label

def parse_pl_record_withbmap(tfrecord_serialized, is_training, data_net_configs=None):
    from aug_data_tf import aug_main, aug_views
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

    object_label = tf.cast(tfrecord_features['object/label'], tf.int32)
    object_label = tf.expand_dims(object_label,0)

    points = tf.decode_raw(tfrecord_features['points/encoded'], tf.float32)
    if data_net_configs == None:
      points_shape = tf.decode_raw(tfrecord_features['points/shape'], tf.int32)
    else:
      points_shape = data_net_configs['points']
    # the image tensor is flattened out, so we have to reconstruct the shape
    points = tf.reshape(points, points_shape)
    #if data_net_configs != None:
    #  points = pc_normalize(points)

    # ------------------------------------------------
    # do not need for single scale net
    is_need_bidmap = data_net_configs==None or \
      ( 'block_params' in data_net_configs and len(data_net_configs['block_params']['filters'])>1 )
    if not is_need_bidmap:
      features = {}
      b_bottom_centers_mm = []
      if is_training:
        if data_net_configs != None and data_net_configs['aug_types']!='none':
          points, b_bottom_centers_mm, augs = aug_main(points, b_bottom_centers_mm,
                      data_net_configs['aug_types'],
                      data_net_configs['data_idxs'])
          #features['augs'] = augs
      else:
        if data_net_configs!=None and 'eval_views' in data_net_configs and data_net_configs['eval_views'] > 1:
          #features['eval_views'] = data_net_configs['eval_views']
          points, b_bottom_centers_mm, augs = aug_views(points, b_bottom_centers_mm,
                      data_net_configs['eval_views'],
                      data_net_configs['data_idxs'])
      features['points'] = points
      return features, object_label
    # ------------------------------------------------

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

    # ------------------------------------------------
    features = {}
    features['bidxmaps_flat'] = bidxmaps_flat
    features['fmap_neighbor_idis'] = fmap_neighbor_idis

    if is_training:
      if data_net_configs != None and data_net_configs['aug_types']!='none':
        points, b_bottom_centers_mm, augs = aug_main(points, b_bottom_centers_mm,
                    data_net_configs['aug_types'],
                    data_net_configs['data_idxs'])
        #features['augs'] = augs
    else:
      if data_net_configs!=None and data_net_configs['eval_views'] > 1:
        #features['eval_views'] = data_net_configs['eval_views']
        points, b_bottom_centers_mm, augs = aug_views(points, b_bottom_centers_mm,
                    data_net_configs['eval_views'],
                    data_net_configs['data_idxs'])

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

def get_dataset_summary(DATASET_NAME, path, loss_lw_gama=2):
  dataset_summary = read_dataset_summary(path)
  if dataset_summary['intact']:
    print('dataset_summary intact, no need to read')
    get_label_num_weights(dataset_summary, loss_lw_gama)
    return dataset_summary

  filenames = glob.glob(os.path.join(path,'*.tfrecord'))
  assert len(filenames) > 0

  from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
  DatasetsMeta = DatasetsMeta(DATASET_NAME)
  num_classes = DatasetsMeta.num_classes

  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset(filenames,
                                      compression_type="",
                                      buffer_size=1024*100,
                                      num_parallel_reads=1)

    batch_size = 50
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

    with tf.Session() as sess:
      m = 0
      n = 0
      label_hist = np.zeros(num_classes)
      try:
        print('start reading all the dataset to get summary')
        while(True):
          features, object_label = sess.run(get_next)
          label_hist += np.histogram(object_label, range(num_classes+1))[0]
          if m%10==0:
            print('%d  %d'%(m,n))
          m += 1
          n += object_label.size
          #if n==batch_size:
          #  print(features['points'][0])
          #  print(object_label)
          #  #for i in range(batch_size):
          #  #  plyfn = '/tmp/tfrecord_%d.ply'%(i)
          #     import ply_util
          #  #  ply_util.create_ply(features['points'][i], plyfn)
      except:
        print('Total: %d  %d'%(m,n))
        print(label_hist)
      dataset_summary = {}
      dataset_summary['size'] = n
      dataset_summary['label_hist'] = label_hist
      write_dataset_summary(dataset_summary, path)
      get_label_num_weights(dataset_summary, loss_lw_gama)
      return dataset_summary

def get_label_num_weights(dataset_summary, loss_lw_gama):
  if loss_lw_gama<0:
    return
  IsPlot = False
  label_hist = dataset_summary['label_hist']
  mean = np.mean(label_hist)
  weight = mean / label_hist
  weights = {}
  gamas = [loss_lw_gama, 1, 2, 5, 10, 20]
  gamas = [loss_lw_gama]
  for gama in gamas:
    weights[gama] = gama * weight
  dataset_summary['label_num_weights'] = weights[loss_lw_gama]
  if  IsPlot:
    import matplotlib.pyplot as plt
    for gama in gamas:
      plt.plot(label_hist, weights[gama], '.', label=str(gama))
    plt.legend()
    plt.show()

def write_dataset_summary(dataset_summary, data_dir):
  import pickle, shutil
  summary_path = os.path.join(data_dir, 'summary.pkl')
  dataset_summary['intact'] = True
  with open(summary_path, 'w') as sf:
    pickle.dump(dataset_summary, sf)
    print(summary_path)
  print_script = os.path.join(BASE_DIR,'print_pkl.py')
  shutil.copyfile(print_script,os.path.join(data_dir,'print_pkl.py'))

def read_dataset_summary(data_dir):
  import pickle
  summary_path = os.path.join(data_dir, 'summary.pkl')
  if not os.path.exists(summary_path):
    dataset_summary = {}
    dataset_summary['intact'] = False
    return dataset_summary
  dataset_summary = pickle.load(open(summary_path, 'r'))
  return dataset_summary

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
  DATASET_NAME = 'MODELNET40'
  path = '/home/z/Research/dynamic_pointnet/data/MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'
  get_dataset_summary(DATASET_NAME, path)
  #merge_tfrecord()


