import h5py, os, glob,sys
import numpy as np
import tensorflow as tf
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
import utils.ply_util as ply_util
from utils.tf_util import TfUtil

MAX_FLOAT_DRIFT = 1e-6
DEBUG = False

def tsize(tensor):
  return len(get_tensor_shape(tensor))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


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


def get_tensor_shape(t):
  return TfUtil.get_tensor_shape(t)


def get_shape0(t):
  return TfUtil.tshape0(t)


def parse_record(tfrecord_serialized, is_training, dset_shape_idx, \
                    bsg=None, gen_ply=False):
    with tf.variable_scope('parse_record'):
      from utils.aug_data_tf import aug_main, aug_views
      #if dset_shape_idx!=None:
      #  from aug_data_tf import aug_data, tf_Rz
      #  R = tf_Rz(1)
      #  import pdb; pdb.set_trace()  # XXX BREAKPOINT

      eles = dset_shape_idx['indices'].keys()
      features_map = {
          'vertex_f': tf.FixedLenFeature([], tf.string),
          'vertex_uint8': tf.FixedLenFeature([], tf.string),
      }
      if 'vertex_i' in eles:
        features['vertex_i'] = tf.FixedLenFeature([], tf.string)
      if 'face_i' in eles:
        features['face_i'] = tf.FixedLenFeature([], tf.string)
        features['valid_num_face'] = tf.FixedLenFeature([1], tf.int64, default_value=-1)

      tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                  features=features_map,
                                                  name='pl_features')

      #*************
      if 'vertex_i' in eles:
        vertex_i = tf.decode_raw(tfrecord_features['vertex_i'], tf.int32)
        vertex_i = tf.reshape(vertex_i, dset_shape_idx['shape']['vertex_i'])

      vertex_f = tf.decode_raw(tfrecord_features['vertex_f'], tf.float32)
      vertex_f = tf.reshape(vertex_f, dset_shape_idx['shape']['vertex_f'])

      vertex_uint8 = tf.decode_raw(tfrecord_features['vertex_uint8'], tf.uint8)
      vertex_uint8 = tf.reshape(vertex_uint8, dset_shape_idx['shape']['vertex_uint8'])

      if 'face_i' in eles:
        face_i = tf.decode_raw(tfrecord_features['face_i'], tf.int32)
        face_i = tf.reshape(face_i, dset_shape_idx['shape']['face_i'])
        valid_num_face = tfrecord_features['valid_num_face']

      #*************
      features = { "vertex_f": vertex_f, \
                  "vertex_uint8": vertex_uint8}
      if 'vertex_i' in tfrecord_features:
        features['vertex_i'] = vertex_i
      if 'face_i' in tfrecord_features:
        features['face_i'] = face_i
        features['valid_num_face'] = valid_num_face

      #*************
      if bsg is not None:
        features = voxel_sampling_grouping(bsg, features, dset_shape_idx)

      #*************
      label_category = get_ele(features, 'label_category', dset_shape_idx)
      labels = tf.squeeze(label_category,1)

      for item in features.keys():
        if 'nb_enough_p' in item:
          assert TfUtil.tsize(features[item]) == 0, item
        elif 'label' in item:
          assert TfUtil.tsize(features[item]) == 1, item
        elif 'grouped_bot_cen_top' in item:
          assert TfUtil.tsize(features[item]) == 3, item
        else:
          assert TfUtil.tsize(features[item]) == 2, item
    return features, labels


def voxel_sampling_grouping(bsg, features, dset_shape_idx):
  xyz = get_ele(features, 'xyz', dset_shape_idx)
  xyz = tf.expand_dims(xyz, 0)

  dsb = {}
  dsb['grouped_pindex'], dsb['vox_index'], dsb['grouped_bot_cen_top'], dsb['empty_mask'],\
    dsb['out_bot_cen_top'], dsb['flatting_idx'], dsb['flat_valid_mask'], dsb['nb_enough_p'], others =\
                    bsg.grouping_multi_scale(xyz)

  # **********
  for s in range(1, bsg._num_scale):
    for item in ['grouped_pindex', 'grouped_bot_cen_top', 'empty_mask', 'nb_enough_p']:
      features[item+'_%d'%(s)] = tf.squeeze(dsb[item][s], [0,1])
    for item in ['flatting_idx', 'flat_valid_mask']:
      features[item+'_%d'%(s)] = dsb[item][s]

  # **********
  vertices = tf.concat([features['vertex_f'], \
                        tf.cast(features['vertex_uint8'], tf.float32)], -1)

  if TfUtil.t_shape(dsb['grouped_pindex'][0])[-1] == 0:
    vertices_gped = vertices
  else:
    vertices = TfUtil.gather(vertices, dsb['grouped_pindex'][0], 0)
    raise NotImplementedError
    vertices_gped = tf.squeeze(vertices_gped, [0,1,2])

  tmp = dset_shape_idx['shape']['vertex_f'][-1]
  features['vertex_f'] = vertices_gped[...,0:tmp]
  features['vertex_uint8'] = tf.cast(vertices_gped[...,tmp:], tf.uint8)

  return features


def read_dataset_summary(summary_path):
  import pickle
  if not os.path.exists(summary_path):
    dataset_summary = {}
    dataset_summary['intact'] = False
    return dataset_summary
  dataset_summary = pickle.load(open(summary_path, 'r'))
  return dataset_summary


def get_label_num_weights(tf_path, lw_delta=1.2, IsPlot=False):
  summary_path = os.path.join(tf_path, 'summary.pkl')
  assert os.path.exists(summary_path)
  dataset_summary = read_dataset_summary(summary_path)
  if lw_delta<=1:
    return None
  label_hist = dataset_summary['label_hist']
  lh = 1.0*label_hist / np.sum(label_hist)

  weights = {}
  if IsPlot:
    lw_deltas = [lw_delta, 1.05, 1.1, 1.5]
  else:
    lw_deltas = [lw_delta]
  for lw_delta in lw_deltas:
    w0 = 1/np.log(lw_delta + lh)
    w = np.minimum(w0, 50)
    w = np.maximum(w, 0.01)
    weights[lw_delta]  = w
  if  IsPlot:
    import matplotlib.pyplot as plt
    for lw_delta in lw_deltas:
      plt.plot(label_hist, weights[lw_delta], '.', label=str(lw_delta))
    plt.legend()
    plt.show()
  label_num_weights = weights[lw_delta].astype(np.float32)
  return label_num_weights


def get_dset_shape_idxs(tf_path):
  metas_fn = os.path.join(tf_path, 'shape_idx.txt')
  with open(metas_fn, 'r') as mf:
    dset_shape_idx = {}
    dataset_name = ''

    shapes = {}
    indices = {}
    the_items = ['vertex_i', 'vertex_f', 'face_i', 'vertex_uint8']
    for item in the_items:
      shapes[item] = {}
      indices[item] = {}
    dset_shape_idx['shape'] = shapes
    dset_shape_idx['indices'] = indices

    for line in mf:
      tmp = line.split(':')
      tmp = [e.strip() for e in tmp]
      assert tmp[0] == 'indices' or tmp[0]=='shape' or tmp[0] == 'dataset_name'
      if tmp[0]!='dataset_name':
        assert tmp[1] in the_items

      if tmp[0] == 'dataset_name':
        dset_shape_idx['dataset_name'] = tmp[1]
      elif tmp[0] == 'shape':
        value = [int(e) for e in tmp[2].split(',')]
        dset_shape_idx[tmp[0]][tmp[1]] = value
      elif tmp[0] == 'indices':
        value = [int(e) for e in tmp[3].split(',')]
        dset_shape_idx[tmp[0]][tmp[1]][tmp[2]] = value
      else:
        raise NotImplementedError

    for ele in ['shape', 'indices']:
      d = dset_shape_idx[ele]
      for item in d.keys():
        if d[item] == {}:
          del d[item]

    return dset_shape_idx


def get_ele(datas, ele, dset_shape_idx):
  ds_idxs = dset_shape_idx['indices']
  for g in ds_idxs:
    if ele in ds_idxs[g]:
      ele_idx = ds_idxs[g][ele]
      if isinstance(datas[g], tf.Tensor):
        ele_data = tf.gather(datas[g], ele_idx, axis=-1)
      else:
        ele_data = datas[g][..., ele_idx]
      return ele_data
  return None
  raise ValueError, ele+' not found'


def label_hist_from_tfrecord(dataset_name, tf_path):
  tf.enable_eager_execution()

  dset_shape_idx = get_dset_shape_idxs(tf_path)
  summary_path = os.path.join(tf_path, 'summary.pkl')
  dataset_summary = read_dataset_summary(summary_path)
  if dataset_summary['intact']:
    print('dataset_summary intact, no need to read: \n{} \n{}'.format(summary_path, dataset_summary))
    label_num_weights = get_label_num_weights(tf_path, IsPlot=True)
    print(label_num_weights)
    return dataset_summary

  #data_path = os.path.join(tf_path, 'merged_data')
  data_path = os.path.join(tf_path, 'data')
  scene = '17DRP5sb8fy'
  scene = '*'
  region = 'region*'
  filenames = glob.glob(os.path.join(data_path,'%s_%s.tfrecord'%(scene, region)))
  assert len(filenames) > 0, data_path
  filenames.sort()

  datasets_meta = DatasetsMeta(dataset_name)
  num_classes = datasets_meta.num_classes

  gen_ply = False
  ply_dir = os.path.join(tf_path, 'plys')

  dataset = tf.data.TFRecordDataset(filenames,
                                    compression_type="",
                                    buffer_size=1024*100,
                                    num_parallel_reads=1)

  batch_size = 10
  is_training = False

  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(
    tf.contrib.data.map_and_batch(
      lambda value: parse_record(value, is_training, dset_shape_idx),
      batch_size=batch_size,
      num_parallel_batches=1,
      drop_remainder=False))
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  dset_iterater = dataset.make_one_shot_iterator()

  #***************************************************************************
  label_hist = np.zeros(num_classes).astype(np.int64)
  batch_num = 0
  total_batch_num = len(filenames) // batch_size
  for batch_k in range(total_batch_num):
    if batch_k %10 ==0:
      print('\n %d/%d reading %s'%( batch_k, total_batch_num, filenames[batch_k]))
    base_fn = os.path.splitext( os.path.basename(filenames[batch_k]))[0]
    features, labels = dset_iterater.get_next()
    batch_num += labels.shape[0]
    labels = labels.numpy()
    for key in features:
      features[key] = features[key].numpy()

    dset_idx = dset_shape_idx['indices']

    category_label = get_ele(features, 'label_category', dset_shape_idx)
    label_hist += np.histogram(category_label, range(num_classes+1))[0]


    xyz = get_ele(features, 'xyz', dset_shape_idx)
    color = get_ele(features, 'color', dset_shape_idx)
    vidx_per_face = get_ele(features, 'vidx_per_face', dset_shape_idx)

    check_nan(features, dset_shape_idx)
    #***************************************************************************
    if gen_ply:
      for bi in range(batch_size):
        ply_fn = os.path.join(ply_dir, '%s_edgev/edgev_%d_raw_color.ply'%(base_fn, bi))
        edgev_vidx_per_face = MeshSampling.edgev_to_face(edgev_per_vertex[bi], valid_ev_num_pv[bi])
        edgev_vidx_per_face = edgev_vidx_per_face.numpy()
        ply_util.gen_mesh_ply(ply_fn, xyz[bi], edgev_vidx_per_face,
                  vertex_color=color[bi])

      #***************************************************************************
      for bi in range(batch_size):
        ply_fn = os.path.join(ply_dir, '%s_raw/b%d_raw_color.ply'%(base_fn, bi))
        ply_util.gen_mesh_ply(ply_fn, xyz[bi], vidx_per_face[bi,0:valid_num_face[bi,0],:], vertex_color=color[bi])


  dataset_summary = {}
  label_hist_normed = 1.0 * label_hist / np.sum(label_hist)
  dataset_summary['size'] = batch_num
  dataset_summary['label_hist'] = label_hist
  dataset_summary['label_hist_normed'] = label_hist_normed
  write_dataset_summary(dataset_summary, tf_path)
  get_label_num_weights(dataset_summary, lw_delta)
  print(dataset_summary)
  return dataset_summary


def check_nan(features, dset_shape_idx):
  for data in features.values():
    any_nan = np.any(np.isnan(data))
    if any_nan:
      print('found nan')
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
  #print('no nan found')


def write_dataset_summary(dataset_summary, data_dir):
  import pickle, shutil
  summary_path = os.path.join(data_dir, 'summary.pkl')
  dataset_summary['intact'] = True
  with open(summary_path, 'w') as sf:
    pickle.dump(dataset_summary, sf)
    print(summary_path)
  print_script = os.path.join(BASE_DIR,'print_pkl.py')
  shutil.copyfile(print_script,os.path.join(data_dir,'print_pkl.py'))




