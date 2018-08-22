import h5py, os, glob
import numpy as np
import tensorflow as tf
from datasets.block_data_prep_util import Raw_H5f
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

Points_eles_order = ['xyz','nxnynz', 'color']
Label_eles_order = ['label_category', 'label_instance', 'label_material']

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


def random_choice(org_vector, sample_N, random_sampl_pro=None,
                  keeporder=False, only_tile_last_one=False):
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


def parse_pl_record(tfrecord_serialized, is_training, dset_metas=None, bsg=None):
    from aug_data_tf import aug_main, aug_views
    #if dset_metas!=None:
    #  from aug_data_tf import aug_data, tf_Rz
    #  R = tf_Rz(1)
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    feature_map = {
        'points/shape': tf.FixedLenFeature([], tf.string),
        'points/encoded': tf.FixedLenFeature([], tf.string),
        'labels/shape': tf.FixedLenFeature([], tf.string),
        'labels/encoded': tf.FixedLenFeature([], tf.string),
    }
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features=feature_map,
                                                name='pl_features')

    #*************
    labels = tf.decode_raw(tfrecord_features['labels/encoded'], tf.int32)
    if dset_metas == None:
      labels_shape = tf.decode_raw(tfrecord_features['labels/shape'], tf.int32)
    else:
      labels_shape = dset_metas['shape']['labels']
    labels = tf.reshape(labels, labels_shape)

    #*************
    points = tf.decode_raw(tfrecord_features['points/encoded'], tf.float32)
    if dset_metas == None:
      points_shape = tf.decode_raw(tfrecord_features['points/shape'], tf.int32)
    else:
      points_shape = dset_metas['shape']['points']
    # the points tensor is flattened out, so we have to reconstruct the shape
    points = tf.reshape(points, points_shape)
    #if dset_metas != None:
    #  points = pc_normalize(points)

    # ------------------------------------------------
    #             data augmentation
    features = {}
    b_bottom_centers_mm = []
    if is_training:
      if dset_metas != None and 'aug_types' in dset_metas and dset_metas['aug_types']!='none':
        points, b_bottom_centers_mm, augs = aug_main(points, b_bottom_centers_mm,
                    dset_metas['aug_types'],
                    dset_metas['indices'])
        #features['augs'] = augs
    else:
      if dset_metas!=None and 'eval_views' in dset_metas and dset_metas['eval_views'] > 1:
        #features['eval_views'] = dset_metas['eval_views']
        points, b_bottom_centers_mm, augs = aug_views(points, b_bottom_centers_mm,
                    dset_metas['eval_views'],
                    dset_metas['indices'])
    #if bsg==None:
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
          if vox_index[s].shape[0].value == 0:
            features['vox_index_%d'%(s)] = tf.zeros([0]*4, tf.int32)
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

    return features, labels



class RawH5_To_Tfrecord():
  def __init__(self, dataset_name, tfrecord_path, num_point=None):
    self.dataset_name = dataset_name
    self.tfrecord_path = tfrecord_path
    self.data_path = os.path.join(self.tfrecord_path, 'data')

    if not os.path.exists(self.data_path):
      os.makedirs(self.data_path)
    self.num_point = num_point
    self.sampling_rates = []

  def __call__(self, rawh5fns):
    self.fn = len(rawh5fns)
    #rawh5fns = rawh5fns[5577:]
    for fi, rawh5fn in enumerate(rawh5fns):
      self.fi = fi
      self.transfer_onefile(rawh5fn)

  def sort_eles(self, h5f):
    # elements *************************
    tmp = [e in Points_eles_order + Label_eles_order for e in h5f]
    assert sum(tmp) == len(tmp)
    self.eles_sorted = {}
    self.eles_sorted['points'] = [e for e in Points_eles_order if e in h5f]
    self.eles_sorted['labels'] = [e for e in Label_eles_order if e in h5f]

  def record_metas(self, h5f, dls):
    ele_idxs = {}
    for item in self.eles_sorted:
      ele_idxs[item] = {}
      n = 0
      for e in self.eles_sorted[item]:
        shape_i = h5f[e].shape
        ele_idxs[item][e] = np.arange(n, n+shape_i[-1])
        n += shape_i[-1]

    metas_fn = os.path.join(self.tfrecord_path, 'metas.txt')
    with open(metas_fn, 'w') as f:
      for item in self.eles_sorted:
        shape_str = ','.join([str(k) for k in dls[item].shape])
        shape_str = 'shape: {}: {}\n'.format(item, shape_str)
        f.write(shape_str)

        for ele in self.eles_sorted[item]:
          idxs_str = ','.join([str(k) for k in  ele_idxs[item][ele]])
          e_str = 'indices: {}: {}: {}\n'.format(item, ele, idxs_str)
          f.write(e_str)

      if self.dataset_name == 'MODELNET40':
        f.write('indices: labels: label_category: 0\n')
      f.flush()
    print('write ok: {}'.format(metas_fn))

  def sampling(self, dls):
    if self.num_point == None:
      return dls
    #print('org num:{} sampled num:{}'.format(dls['points'].shape[0], self.num_point))
    num_point0 = dls['points'].shape[0]
    sampling_rate = 1.0 * self.num_point / num_point0
    sampling_indices = random_choice(np.arange(num_point0),
                    self.num_point, keeporder=False, only_tile_last_one=False)
    for item in dls:
      dls[item] = np.take(dls[item], sampling_indices, axis=0)

    self.sampling_rates.append(sampling_rate)
    if len(self.sampling_rates) % 10 == 0:
      ave_samplings_rate = np.mean(self.sampling_rates)
      print('ave_samplings_rate:{:.2f}'.format(ave_samplings_rate))
    return dls

  def transfer_onefile(self, rawh5fn):
    base_name = os.path.splitext(os.path.basename(rawh5fn))[0]
    base_name1 = os.path.basename(os.path.dirname(rawh5fn))
    if self.dataset_name == "MATTERPORT":
      base_name = base_name1 + '_' + base_name
    tfrecord_fn = os.path.join(self.data_path, base_name) + '.tfrecord'
    with h5py.File(rawh5fn, 'r') as h5f,\
      tf.python_io.TFRecordWriter( tfrecord_fn ) as raw_tfrecord_writer:

      if self.fi == 0:
        self.sort_eles(h5f)
      #*************************************************************************
      # concat elements together
      for e in h5f:
        assert e in self.eles_sorted['points'] + self.eles_sorted['labels']

      dls = {}
      for item in self.eles_sorted:
        dls[item] = []
        for e in self.eles_sorted[item]:
          data = h5f[e][:]
          dls[item].append( data )
        if len(dls[item]) >  0:
          dls[item] = np.concatenate(dls[item], -1)
      #*************************************************************************
      # get label for MODELNET40
      if self.dataset_name == 'MODELNET40':
        tmp = base_name.split('_')
        if len(tmp) == 2:
          category = tmp[0]
        elif len(tmp) == 3:
          category = '_'.join(tmp[0:2])
        dataset_meta = DatasetsMeta(self.dataset_name)
        object_label = dataset_meta.class2label[category]
        dls['labels'] = np.array([object_label])

      #*************************************************************************
      # convert to expample
      dls['points'] = dls['points'].astype(np.float32, casting='same_kind')
      dls['labels'] = dls['labels'].astype(np.int32, casting='same_kind')
      if self.fi == 0:
        self.record_metas(h5f, dls)

      dls = self.sampling(dls)
      datas_bin = dls['points'].tobytes()
      datas_shape_bin = np.array(dls['points'].shape, np.int32).tobytes()
      labels_bin = dls['labels'].tobytes()
      labels_shape_bin = np.array(dls['labels'].shape, np.int32).tobytes()
      features_map = {
        'points/encoded':  bytes_feature(datas_bin),
        'points/shape':   bytes_feature(datas_shape_bin),
        'labels/encoded':  bytes_feature(labels_bin),
        'labels/shape':   bytes_feature(labels_shape_bin) }
        #'object':   int64_feature(object_label) }

      example = tf.train.Example(features=tf.train.Features(feature=features_map))

      #*************************************************************************
      raw_tfrecord_writer.write(example.SerializeToString())

    if self.fi %10 ==0:
      print('{}/{} write tfrecord OK: {}'.format(self.fi, self.fn, tfrecord_fn))



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


def get_dset_metas(tf_path):
  metas_fn = os.path.join(tf_path, 'metas.txt')
  with open(metas_fn, 'r') as mf:
    dset_metas = {}
    shapes = {}
    shapes['points'] = {}
    shapes['labels'] = {}
    dset_metas['shape'] = shapes

    indices = {}
    indices['points'] = {}
    indices['labels'] = {}
    dset_metas['indices'] = indices

    for line in mf:
      tmp = line.split(':')
      tmp = [e.strip() for e in tmp]
      assert tmp[0] == 'indices' or tmp[0]=='shape'
      assert tmp[1] == 'points' or tmp[1]=='labels'
      if tmp[0] == 'shape':
        value = [int(e) for e in tmp[2].split(',')]
        dset_metas[tmp[0]][tmp[1]] = value
      elif tmp[0] == 'indices':
        value = [int(e) for e in tmp[3].split(',')]
        dset_metas[tmp[0]][tmp[1]][tmp[2]] = value
      else:
        raise NotImplementedError
    return dset_metas


def get_dataset_summary(dataset_name, tf_path, loss_lw_gama=2):
  dset_metas = get_dset_metas(tf_path)
  dataset_summary = read_dataset_summary(tf_path)
  if dataset_summary['intact'] and False:
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

    with tf.Session() as sess:
      m = 0
      n = 0
      label_hist = np.zeros(num_classes)


      #features, labels = sess.run(get_next)
      ##category_label = labels[:, dset_metas['labels']['label_category']]
      #label_hist += np.histogram(category_label, range(num_classes+1))[0]

      try:
        print('start reading all the dataset to get summary')
        while(True):
          features, labels = sess.run(get_next)
          category_label = labels[:, dset_metas['labels']['label_category']]
          label_hist += np.histogram(category_label, range(num_classes+1))[0]
          if m%10==0:
            print('%d  %d'%(m,n))
          m += 1
          n += category_label.size
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


def main_write(dataset_name, rawh5_glob, tfrecord_path, num_point):
  rawh5_fns = glob.glob(rawh5_glob)

  raw_to_tf = RawH5_To_Tfrecord(dataset_name, tfrecord_path, num_point)
  raw_to_tf(rawh5_fns)



if __name__ == '__main__':
  dataset_name = 'MODELNET40'
  dataset_name = 'MATTERPORT'
  dset_path = '/home/z/Research/SparseVoxelNet/data/{}_H5TF'.format(dataset_name)
  num_point = {'MODELNET40':None, 'MATTERPORT':100000}

  rawh5_glob = os.path.join(dset_path, 'rawh5/*/*.rh5')
  tfrecord_path = os.path.join(dset_path, 'raw_tfrecord')

  #main_write(dataset_name, rawh5_glob, tfrecord_path, num_point[dataset_name])

  #get_dataset_summary(dataset_name, tfrecord_path)
  #get_dset_metas(tfrecord_path)

