import h5py, os, glob
import numpy as np
import tensorflow as tf
from datasets.block_data_prep_util import Raw_H5f
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

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

def random_choice(org_vector,sample_N,random_sampl_pro=None, keeporder=True, only_tile_last_one=False):
    assert org_vector.ndim == 1
    org_N = org_vector.size
    if org_N == sample_N:
        sampled_vector = org_vector
    elif org_N > sample_N:
        sampled_vector = np.random.choice(org_vector,sample_N,replace=False,p=random_sampl_pro)
        if keeporder:
            sampled_vector = np.sort(sampled_vector)
    else:
        if only_tile_last_one:
            new_vector = np.array( [ org_vector[-1] ]*(sample_N-org_N) ).astype(org_vector.dtype)
        else:
            new_vector = np.random.choice(org_vector,sample_N-org_N,replace=True)
        sampled_vector = np.concatenate( [org_vector,new_vector] )
    #str = '%d -> %d  %d%%'%(org_N,sample_N,100.0*sample_N/org_N)
    #print(str)
    return sampled_vector
class RawH5_To_Tfrecord():
  def __init__(self, dataset_name, tfrecord_path, num_point=None):
    self.dataset_name = dataset_name
    self.tfrecord_path = tfrecord_path
    self.data_path = os.path.join(self.tfrecord_path, 'data')

    if not os.path.exists(self.data_path):
      os.makedirs(self.data_path)
    self.num_point = num_point

  def __call__(self, rawh5fns):
    self.fn = len(rawh5fns)
    #rawh5fns = rawh5fns[5577:]
    for fi, rawh5fn in enumerate(rawh5fns):
      self.transfer_onefile(rawh5fn, fi)

  def record_metas(self, h5f):
    # elements *************************
    eles_order = ['xyz','nxnynz']
    tmp = [e in eles_order for e in h5f]
    assert sum(tmp) == len(tmp)
    self.eles_sorted = [e for e in eles_order if e in h5f]

    n = 0
    data_idxs = {}
    shapes = []
    for e in self.eles_sorted:
      shape_i = h5f[e].shape
      data_idxs[e] = np.arange(n, n+shape_i[-1])
      n += shape_i[-1]
      shapes.append(shape_i)

    metas_fn = os.path.join(self.tfrecord_path, 'metas.txt')
    with open(metas_fn, 'w') as f:
      for ele in self.eles_sorted:
        e_str = '{}:{}\n'.format(ele, data_idxs[ele])
        f.write(e_str)
        f.flush()
    print('write ok: {}'.format(metas_fn))

  def get_sampling_indices(self, h5f):
    if self.num_point == None:
      return
    num_point0s = [h5f[e].shape[0] for e in h5f]
    assert min(num_point0s) == max(num_point0s)
    num_point0 = num_point0s[0]
    if num_point0 == self.num_point:
      self.sampling_indices = None
    elif num_point0 < self.num_point:
      raise NotImplementedError
    else:
      self.sampling_indices = random_choice(np.arange(num_point0,
                                        self.num_point, keeporder=False))

  def sampling(self, data):
    if self.num_point == None:
      return data
    data_s = np.take(data, self.sampling_indices, axis=0)
    return data_s

  def transfer_onefile(self, rawh5fn, fi):
    base_name = os.path.splitext(os.path.basename(rawh5fn))[0]
    tfrecord_fn = os.path.join(self.data_path, base_name) + '.tfrecord'
    with h5py.File(rawh5fn, 'r') as h5f,\
      tf.python_io.TFRecordWriter( tfrecord_fn ) as raw_tfrecord_writer:

      if not hasattr(self, 'eles_sorted'):
        self.get_sampling_indices(h5f)
        self.record_metas(h5f)
      #*************************************************************************
      # concat elements together
      for e in h5f:
        assert e in self.eles_sorted

      datas = []
      for e in self.eles_sorted:
        data = h5f[e][:]
        datas.append( data )
      datas = np.concatenate(datas, -1)
      #*************************************************************************
      # get label
      tmp = base_name.split('_')
      if len(tmp) == 2:
        category = tmp[0]
      elif len(tmp) == 3:
        category = '_'.join(tmp[0:2])
      dataset_meta = DatasetsMeta(self.dataset_name)
      object_label = dataset_meta.class2label[category]

      #*************************************************************************
      # convert to expample

      datas = self.sampling(datas)
      datas_bin = datas.tobytes()
      datas_shape_bin = np.array(datas.shape, np.int32).tobytes()
      features_map = {
        'points/encoded':  bytes_feature(datas_bin),
        'points/shape':   bytes_feature(datas_shape_bin),
        'object':   int64_feature(object_label) }

      example = tf.train.Example(features=tf.train.Features(feature=features_map))

      #*************************************************************************
      raw_tfrecord_writer.write(example.SerializeToString())

    if fi %10 ==0:
      print('{}/{} write tfrecord OK: {}'.format(fi, self.fn, tfrecord_fn))


def main_write(dataset_name, rawh5_glob, tfrecord_path):
  rawh5_fns = glob.glob(rawh5_glob)

  raw_to_tf = RawH5_To_Tfrecord(dataset_name, tfrecord_path)
  raw_to_tf(rawh5_fns)


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


def parse_pl_record(tfrecord_serialized, is_training, data_shapes=None, bsg=None):
    from aug_data_tf import aug_main, aug_views
    #if data_shapes!=None:
    #  from aug_data_tf import aug_data, tf_Rz
    #  R = tf_Rz(1)
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    feature_map = {
        'object': tf.FixedLenFeature([], tf.int64),
        'points/shape': tf.FixedLenFeature([], tf.string),
        'points/encoded': tf.FixedLenFeature([], tf.string),
    }
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features=feature_map,
                                                name='pl_features')

    object_label = tf.cast(tfrecord_features['object'], tf.int32)
    object_label = tf.expand_dims(object_label,0)

    points = tf.decode_raw(tfrecord_features['points/encoded'], tf.float32)
    if data_shapes == None:
      points_shape = tf.decode_raw(tfrecord_features['points/shape'], tf.int32)
    else:
      points_shape = data_shapes['points']
    # the image tensor is flattened out, so we have to reconstruct the shape
    points = tf.reshape(points, points_shape)
    #if data_shapes != None:
    #  points = pc_normalize(points)

    # ------------------------------------------------
    #             data augmentation
    features = {}
    b_bottom_centers_mm = []
    if is_training:
      if data_shapes != None and data_shapes['aug_types']!='none':
        points, b_bottom_centers_mm, augs = aug_main(points, b_bottom_centers_mm,
                    data_shapes['aug_types'],
                    data_shapes['data_metas']['data_idxs'])
        #features['augs'] = augs
    else:
      if data_shapes!=None and 'eval_views' in data_shapes and data_shapes['eval_views'] > 1:
        #features['eval_views'] = data_shapes['eval_views']
        points, b_bottom_centers_mm, augs = aug_views(points, b_bottom_centers_mm,
                    data_shapes['eval_views'],
                    data_shapes['data_metas']['data_idxs'])

    features['points'] = points
    # ------------------------------------------------
    #             grouping and sampling on line
    if bsg!=None:
      xyz = tf.expand_dims(points[:,0:3], 0)
      ds = {}
      grouped_pindex, vox_index, grouped_bot_cen_top, \
        empty_mask, bot_cen_top, nblock_valid, others = \
                        bsg.grouping_multi_scale(xyz)

      num_scale = len(grouped_bot_cen_top)
      global_block_num = grouped_pindex[0].shape[1]
      check_g = tf.assert_equal(global_block_num, 1)
      with tf.control_dependencies([check_g]):
        bsi = 0
        for s in range(num_scale+1):
          features['empty_mask_%d'%(s)] = tf.cast(empty_mask[s][bsi], tf.int8)
          if vox_index[s]==[]:
            features['vox_index_%d'%(s)] = []
          else:
            features['vox_index_%d'%(s)] = vox_index[s][bsi]
          if s==num_scale:
            continue

          features['grouped_pindex_%d'%(s)] = grouped_pindex[s][bsi]
          features['grouped_bot_cen_top_%d'%(s)] = grouped_bot_cen_top[s][bsi]
          features['bot_cen_top_%d'%(s)] = bot_cen_top[s][bsi]
          #for k in range(len(others[s]['name'])):
          #  name = others[s]['name'][k]+'_%d'%(s)
          #  if name not in features:
          #    features[name] = []
          #  features[name].append( others[s]['value'][k][bsi] )

    return features, object_label


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

def get_dataset_summary(dataset_name, tf_path, loss_lw_gama=2):
  dataset_summary = read_dataset_summary(tf_path)
  if dataset_summary['intact']:
    print('dataset_summary intact, no need to read')
    get_label_num_weights(dataset_summary, loss_lw_gama)
    return dataset_summary

  data_path = os.path.join(tf_path, 'data')
  filenames = glob.glob(os.path.join(data_path,'*.tfrecord'))
  assert len(filenames) > 0

  datasets_meta = DatasetsMeta(dataset_name)
  num_classes = datasets_meta.num_classes

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
      write_dataset_summary(dataset_summary, tf_path)
      get_label_num_weights(dataset_summary, loss_lw_gama)
      return dataset_summary



if __name__ == '__main__':
  dataset_name = 'MODELNET40'
  dset_path = '/home/z/Research/SparseVoxelNet/data/MODELNET40_H5TF'
  rawh5_glob = os.path.join(dset_path, 'rawh5/*/*.rh5')
  tfrecord_path = os.path.join(dset_path, 'raw_tfrecord')

  #main_write(dataset_name, rawh5_glob, tfrecord_path)

  get_dataset_summary(dataset_name, tfrecord_path)

