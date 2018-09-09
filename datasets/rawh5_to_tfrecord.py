import h5py, os, glob
import numpy as np
import tensorflow as tf
from datasets.block_data_prep_util import Raw_H5f
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

Points_eles_order = ['xyz','nxnynz', 'color']
Label_eles_order = ['label_category', 'label_instance', 'label_material', 'label_raw_category']

MAX_FLOAT_DRIFT = 1e-6
DEBUG = False

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


def rm_some_labels(points, labels, valid_num_point, dset_metas):
  if dset_metas['dataset_name'] == 'MODELNET40':
    return points, labels
  elif dset_metas['dataset_name'] == 'MATTERPORT':
    assert len(points.shape) == 2
    org_num_point = points.shape[0].value
    label_category = labels[:, dset_metas['indices']['labels']['label_category'][0]]
    mask_void = tf.equal(label_category, 0)    # void
    mask_unlabeled = tf.equal(label_category, 41)  # unlabeld

    #del_mask = tf.logical_or(mask_void, mask_unlabeled)
    del_mask = mask_void

    #del_indices = tf.squeeze(tf.where(del_mask),1)
    #keep_indices = del_indices

    keep_mask = tf.logical_not(del_mask)
    keep_indices = tf.squeeze(tf.where(keep_mask),1)

    keep_num = tf.shape(keep_indices)[0]
    #keep_num = tf.Print(keep_num, [keep_num, org_num_point], message="keep num, org num")

    check = tf.assert_greater(keep_num,0, message="all void or unlabled points")
    with tf.control_dependencies([check]):
      del_num = org_num_point - keep_num

      points = tf.gather(points, keep_indices)
      labels = tf.gather(labels, keep_indices)

    def sampling(var):
      tmp1 = lambda: tf.tile(var[0:1,:], [del_num,1])
      tmp2 = lambda: var[0:del_num, :]
      tmp = tf.case([(tf.less_equal(del_num, keep_num), tmp2)],\
                  default = tmp1)
      return tmp
    points = tf.concat([points, sampling(points)], 0)
    labels = tf.concat([labels, sampling(labels)], 0)
    points.set_shape([org_num_point, points.shape[1].value])
    labels.set_shape([org_num_point, labels.shape[1].value])
    return points, labels


def parse_pl_record(tfrecord_serialized, is_training, dset_metas=None, bsg=None,\
                    is_rm_void_labels=True, gen_ply=False):
    from aug_data_tf import aug_main, aug_views
    assert dset_metas!=None, "current vertion data do not have shape info"
    #if dset_metas!=None:
    #  from aug_data_tf import aug_data, tf_Rz
    #  R = tf_Rz(1)
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    feature_map = {
        'points/encoded': tf.FixedLenFeature([], tf.string),
        'labels/encoded': tf.FixedLenFeature([], tf.string),
        'valid_num':      tf.FixedLenFeature([1], tf.int64, default_value=-1)
    }
    if dset_metas == None:
      feature_map1 = {
          'points/shape': tf.FixedLenFeature([], tf.string),
          'labels/shape': tf.FixedLenFeature([], tf.string),
      }
      feature_map.update(feature_map1)

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

    valid_num_point = tf.cast(tfrecord_features['valid_num'], tf.int32)

    if is_rm_void_labels:
      points, labels = rm_some_labels(points, labels, valid_num_point, dset_metas)
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
    # ------------------------------------------------
    #             grouping and sampling on line
    if bsg!=None:
      xyz = tf.expand_dims(points[:,0:3], 0)
      ds = {}
      grouped_pindex, vox_index, grouped_bot_cen_top, empty_mask, bot_cen_top, \
        flatting_idx, flat_valid_mask, nblock_valid, others = \
                        bsg.grouping_multi_scale(xyz)

      assert bsg.batch_size == 1
      bsi = 0

      if gen_ply:
        features['raw_points'] = points
        features['raw_labels'] = labels

      points, labels = gather_labels_for_each_gb(points, labels, grouped_pindex[0][bsi])

      num_scale = len(grouped_bot_cen_top)
      global_block_num = grouped_pindex[0].shape[2]
      for s in range(num_scale+1):
        if len(empty_mask) <= s:
          continue
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
        features['flatting_idx_%d'%(s)] = flatting_idx[s]
        features['flat_valid_mask_%d'%(s)] = flat_valid_mask[s]

        if gen_ply:
          features['nblock_valid_%d'%(s)] = nblock_valid[s]
        #for k in range(len(others[s]['name'])):
        #  name = others[s]['name'][k]+'_%d'%(s)
        #  if name not in features:
        #    features[name] = []
        #  features[name].append( others[s]['value'][k][bsi] )

    # ------------------------------------------------
    features['points'] = points
    features['valid_num_point'] = valid_num_point

    return features, labels


def gather_labels_for_each_gb(points, labels, grouped_pindex0):
  #grouped_pindex0 = tf.squeeze(grouped_pindex0, 1)

  shape0 = [e.value for e in grouped_pindex0.shape]
  points_gbs = tf.gather(points, grouped_pindex0)
  labels_gbs = tf.gather(labels, grouped_pindex0)
  points_gbs = tf.squeeze(points_gbs, 0)
  labels_gbs = tf.squeeze(labels_gbs, 0)

  # reshape all the gbs in a batch to one dim
  # need to reshape it back after input pipeline
  #points_gbs = tf.reshape(points_gbs, [-1, points.shape[-1].value])
  #labels_gbs = tf.reshape(labels_gbs, [-1, labels_gbs.shape[-1].value])
  return points_gbs, labels_gbs


class RawH5_To_Tfrecord():
  def __init__(self, dataset_name, tfrecord_path, num_point=None, block_size=None):
    self.dataset_name = dataset_name
    self.tfrecord_path = tfrecord_path
    self.data_path = os.path.join(self.tfrecord_path, 'data')

    if not os.path.exists(self.data_path):
      os.makedirs(self.data_path)
    self.num_point = num_point
    self.block_size = block_size
    if type(block_size)!=type(None):
      self.block_stride = block_size * 0.5
    self.min_pn_inblock = min(self.num_point * 0.1, 2000)
    self.sampling_rates = []

  def __call__(self, rawh5fns):
    bsfn0 = os.path.join(self.tfrecord_path, 'block_split_settings.txt')
    if type(self.block_size)!=type(None):
      with open(bsfn0, 'w') as bsf:
        bsf.write('num_point:{}\nblock_size:{}\nblock_stride:{}\nmin_pn_inblock:{}'.format(\
                      self.num_point, self.block_size,\
                      self.block_stride, self.min_pn_inblock))

    block_split_dir = os.path.join(self.tfrecord_path, 'block_split_summary')
    if not os.path.exists(block_split_dir):
      os.makedirs(block_split_dir)


    self.fn = len(rawh5fns)
    #rawh5fns = rawh5fns[5577:]
    for fi, rawh5fn in enumerate(rawh5fns):
      self.fi = fi
      region_name = os.path.splitext(os.path.basename(rawh5fn))[0]
      scene_name = os.path.basename(os.path.dirname(rawh5fn))
      scene_bs_fn = os.path.join(block_split_dir, scene_name + '_' + region_name + '.txt')

      cur_tfrecord_intact = False
      if os.path.exists(scene_bs_fn):
        with open(scene_bs_fn, 'r') as bsf:
          for line in bsf:
            line = line.strip()
            break
          if line == 'intact':
            cur_tfrecord_intact = True
      if cur_tfrecord_intact:
        print('skip {} intact'.format(scene_bs_fn))
        continue

      block_num, valid_block_num, num_points_splited = self.transfer_onefile(rawh5fn)

      with open(scene_bs_fn, 'w') as bsf:
        bsf.write('intact\n')
        bnstr = '{} \tblock_num:{} \tvalid block num:{}\n\n'.format(region_name, block_num, valid_block_num)
        bsf.write(bnstr)
        vnstr = '\n'+'\n'.join(['{}_{}: {}'.format(region_name, i, num_points_splited[i]) for i in range(valid_block_num)])
        bsf.write(vnstr)
    print('All {} file are converted to tfreord'.format(fi+1))

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
    self.ele_idxs = ele_idxs

    metas_fn = os.path.join(self.tfrecord_path, 'metas.txt')
    with open(metas_fn, 'w') as f:
      f.write('dataset_name:{}\n'.format(self.dataset_name))
      for item in self.eles_sorted:
        shape = dls[item].shape
        if self.num_point!=None:
          shape = (self.num_point,) + shape[1:]
        shape_str = ','.join([str(k) for k in shape])
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
    pass

  def split_pcl(self, dls):
    if type(self.block_size) == type(None):
      return [dls], 1, [dls['points'].shape[0]]

    xyz = dls['points'][:, self.ele_idxs['points']['xyz']]
    xyz_min = np.min(xyz, 0)
    xyz_max = np.max(xyz, 0)
    xyz_scope = xyz_max - xyz_min
    #print(xyz_scope)

    block_size = self.block_size
    block_stride = self.block_stride
    block_dims0 =  (xyz_scope - block_size) / block_stride + 1
    block_dims0 = np.maximum(block_dims0, 1)
    block_dims = np.ceil(block_dims0).astype(np.int32)
    #print(block_dims)
    xyzindices = [np.arange(0, k) for k in block_dims]
    bot_indices = [np.array([[xyzindices[0][i], xyzindices[1][j], xyzindices[2][k]]]) for i in range(block_dims[0]) \
                   for j  in range(block_dims[1]) for k in range(block_dims[2])]
    bot_indices = np.concatenate(bot_indices, 0)
    bot = bot_indices * block_stride
    top = bot + block_size

    block_num = bot.shape[0]
    if block_num == 1:
      #print('xyz scope:{} block_num:{}'.format(xyz_scope, block_num))
      return [dls], 1, [dls['points'].shape[0]]

    if block_num>1:
      for i in range(block_num):
        for j in range(3):
          if top[i,j] > xyz_scope[j]:
            top[i,j] = xyz_scope[j] - MAX_FLOAT_DRIFT
            bot[i,j] = np.maximum(xyz_scope[j] - block_size[j] + MAX_FLOAT_DRIFT, 0)
    #print(xyzindices)
    #print(bot_indices)
    #print(bot)
    #print(top)

    bot += xyz_min
    top += xyz_min

    dls_splited = []
    num_points_splited = []
    for i in range(block_num):
      mask0 = xyz >= bot[i]
      mask1 = xyz < top[i]
      mask = mask0 * mask1
      mask = np.all(mask, 1)
      new_n = np.sum(mask)
      indices = np.where(mask)[0]
      num_point_i = indices.size
      if num_point_i < self.min_pn_inblock:
        #print('num point {} < {}, block {}/{}'.format(num_point_i,\
        #                                self.num_point * 0.1, i, block_num))
        continue
      num_points_splited.append(num_point_i)
      dls_i = {}
      dls_i['points'] = np.take(dls['points'], indices, axis=0)
      dls_i['labels'] = np.take(dls['labels'], indices, axis=0)
      dls_splited.append(dls_i)

      xyz_i = dls_i['points'][:,0:3]
      xyz_min_i = np.min(xyz_i,0)
      xyz_max_i = np.max(xyz_i,0)

    valid_block_num = len(dls_splited)
    print('xyz scope:{} \tblock_num:{} \tvalid block num:{}'.format(xyz_scope,\
                                          block_num, valid_block_num))
    return dls_splited,  valid_block_num, num_points_splited

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
      print('sampled num point/ real:{:.2f}'.format(ave_samplings_rate))
    return dls

  def transfer_onefile(self, rawh5fn):
    base_name = os.path.splitext(os.path.basename(rawh5fn))[0]
    base_name1 = os.path.basename(os.path.dirname(rawh5fn))
    if self.dataset_name == "MATTERPORT":
      base_name = base_name1 + '_' + base_name
    dataset_meta = DatasetsMeta(self.dataset_name)

    with h5py.File(rawh5fn, 'r') as h5f:
      print('starting {} th file: {}'.format(self.fi, rawh5fn))
      if not hasattr(self, 'eles_sorted'):
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
        object_label = dataset_meta.class2label[category]
        dls['labels'] = np.array([object_label])

      #*************************************************************************
      # convert to expample
      dls['points'] = dls['points'].astype(np.float32, casting='same_kind')
      dls['labels'] = dls['labels'].astype(np.int32, casting='same_kind')
      if not hasattr(self, 'ele_idxs'):
        self.record_metas(h5f, dls)

      max_category =  np.max(dls['labels'][:,self.ele_idxs['labels']['label_category']])
      assert max_category < dataset_meta.num_classes, "max_category > {}".format(dataset_meta.num_classes)

      dls_splited, valid_block_num, num_points_splited = self.split_pcl(dls)
      block_num = len(dls_splited)
      for bk, ds0 in enumerate(dls_splited):
        ds = self.sampling(ds0)

        datas_bin = ds['points'].tobytes()
        datas_shape_bin = np.array(ds['points'].shape, np.int32).tobytes()
        labels_bin = ds['labels'].tobytes()
        labels_shape_bin = np.array(ds['labels'].shape, np.int32).tobytes()
        features_map = {
          'points/encoded': bytes_feature(datas_bin),
          #'points/shape':   bytes_feature(datas_shape_bin),
          'labels/encoded': bytes_feature(labels_bin),
          #'labels/shape':   bytes_feature(labels_shape_bin),
          'valid_num':      int64_feature(num_points_splited[bk]) }

        example = tf.train.Example(features=tf.train.Features(feature=features_map))

        #*************************************************************************
        tmp = '' if block_num==1 else '_'+str(bk)
        tfrecord_fn = os.path.join(self.data_path, base_name)+tmp + '.tfrecord'
        with tf.python_io.TFRecordWriter( tfrecord_fn ) as raw_tfrecord_writer:
          raw_tfrecord_writer.write(example.SerializeToString())

        if self.fi %50 ==0:
          print('{}/{} write tfrecord OK: {}'.format(self.fi, self.fn, tfrecord_fn))

    return block_num, valid_block_num, num_points_splited


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


def get_label_num_weights(dataset_summary, loss_lw_gama):
  if loss_lw_gama<0:
    return
  IsPlot = False
  label_hist = dataset_summary['label_hist']
  mean = np.mean(label_hist)
  assert np.min(label_hist) > 0
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


def get_dset_shape_idxs(tf_path):
  metas_fn = os.path.join(tf_path, 'metas.txt')
  with open(metas_fn, 'r') as mf:
    dset_metas = {}
    dataset_name = ''

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
      assert tmp[0] == 'indices' or tmp[0]=='shape' or tmp[0] == 'dataset_name'
      if tmp[0]!='dataset_name':
        assert tmp[1] == 'points' or tmp[1]=='labels'

      if tmp[0] == 'dataset_name':
        dset_metas['dataset_name'] = tmp[1]
      elif tmp[0] == 'shape':
        value = [int(e) for e in tmp[2].split(',')]
        dset_metas[tmp[0]][tmp[1]] = value
      elif tmp[0] == 'indices':
        value = [int(e) for e in tmp[3].split(',')]
        dset_metas[tmp[0]][tmp[1]][tmp[2]] = value
      else:
        raise NotImplementedError
    return dset_metas


def get_dataset_summary(dataset_name, tf_path, loss_lw_gama=2):
  dset_metas = get_dset_shape_idxs(tf_path)
  dataset_summary = read_dataset_summary(tf_path)
  if dataset_summary['intact']:
    print('dataset_summary intact, no need to read')
    get_label_num_weights(dataset_summary, loss_lw_gama)
    return dataset_summary

  data_path = os.path.join(tf_path, 'merged_data')
  filenames = glob.glob(os.path.join(data_path,'*.tfrecord'))
  assert len(filenames) > 0, data_path

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
        lambda value: parse_pl_record(value, is_training, dset_metas),
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
          assert len(labels.shape) == 3
          assert len(features['points'].shape) == 3
          assert len(features['valid_num_point'].shape) == 2
          category_label = labels[:, :, \
                           dset_metas['indices']['labels']['label_category'][0]]
          label_hist += np.histogram(category_label, range(num_classes+1))[0]
          if m%10==0:
            print('%d  %d'%(m,n))
          m += 1
          n += category_label.size
      except:
        print('Total: %d  %d'%(m,n))
        print(label_hist)
      dataset_summary = {}
      dataset_summary['size'] = n
      dataset_summary['label_hist'] = label_hist
      write_dataset_summary(dataset_summary, tf_path)
      get_label_num_weights(dataset_summary, loss_lw_gama)
      return dataset_summary


def main_write(dataset_name, rawh5_glob, tfrecord_path, num_point, block_size):
  rawh5_fns = glob.glob(rawh5_glob)

  raw_to_tf = RawH5_To_Tfrecord(dataset_name, tfrecord_path, num_point, block_size)
  raw_to_tf(rawh5_fns)

def split_fn_ls( tfrecordfn_ls, merged_n):
    nf = len(tfrecordfn_ls)
    assert merged_n < nf
    group_n = int(math.ceil( 1.0*nf/merged_n ))
    merged_fnls = []
    group_name_ls = []
    for i in range( 0, nf, group_n ):
        end = min( nf, i+group_n )
        merged_fnls.append( tfrecordfn_ls[i:end] )
        group_name_ls.append( '%d_%d'%(i, end) )
    return merged_fnls, group_name_ls

def main_merge_tfrecord(dataset_name, tfrecord_path):
  from dataset_utils import merge_tfrecord
  import random

  rawdata_dir = tfrecord_path+'/data'
  merged_dir = tfrecord_path+'/merged_data'
  if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)

  datasets_meta = DatasetsMeta(dataset_name)
  train_fn_ls = datasets_meta.get_train_test_file_list(rawdata_dir, True)
  random.shuffle(train_fn_ls)
  test_fn_ls = datasets_meta.get_train_test_file_list(rawdata_dir, False)
  random.shuffle(test_fn_ls)

  grouped_train_fnls, train_groupe_names = split_fn_ls(train_fn_ls, 10)
  train_groupe_names = ['train_'+e for e in train_groupe_names]
  grouped_test_fnls, test_groupe_names = split_fn_ls(test_fn_ls, 5)
  test_groupe_names = ['test_'+e for e in test_groupe_names]

  grouped_fnls = grouped_train_fnls + grouped_test_fnls
  group_names = train_groupe_names + test_groupe_names
  merged_fnls = [os.path.join(merged_dir, e+'.tfrecord') for e in group_names]

  for k in range(len(grouped_fnls)):
    merge_tfrecord(grouped_fnls[k], merged_fnls[k])

def main_gen_ply(dataset_name, tf_path):
  #scene = 'gxdoqLR6rwA_region2'  # large ground
  #scene = 'gTV8FGcVJC9_region7'
  #scene = 'Vvot9Ly1tCj_region24'  # most unlabled points
  scene = 'Pm6F8kyY3z2_region3'
  #scene = '17DRP5sb8fy_region0'
  #scene = 'ac26ZMwG7aT_region9'

  scene = 'YFuZgdQ5vWj_region19'  # all void
  scene = 'VFuaQ6m2Qom_region38'  # all void

  name_base = 'data/{}*.tfrecord'.format(scene)
  fn_glob = os.path.join(tf_path, name_base)
  filenames = glob.glob(fn_glob)
  assert len(filenames) > 0, fn_glob
  for fn in filenames:
    gen_ply_onef(dataset_name, tf_path, fn, scene)

def gen_ply_onef(dataset_name, tf_path, filename, scene):
  from utils.ply_util import create_ply_dset
  is_rm_void_labels = True
  is_rm_void_labels = False

  ply_dir = os.path.join(tf_path, 'plys/ply_'+scene)
  if not os.path.exists(ply_dir):
    os.makedirs(ply_dir)

  data_infos = get_dset_shape_idxs(tf_path)
  datasets_meta = DatasetsMeta(dataset_name)
  num_classes = datasets_meta.num_classes

  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset([filename],
                                      compression_type="",
                                      buffer_size=1024*100,
                                      num_parallel_reads=1)

    batch_size = 10
    is_training = False

    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        lambda value: parse_pl_record(value, is_training, data_infos,\
                          is_rm_void_labels=is_rm_void_labels),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=False))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    get_next = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      n = 0
      all_points = []
      all_label_categories = []
      try:
        print('start reading all the dataset to get summary')
        while(True):
          features, labels = sess.run(get_next)

          label_category = labels[:,:, data_infos['indices']['labels']['label_category'][0]]
          all_label_categories.append(label_category)

          xyz = features['points'][:,:, data_infos['indices']['points']['xyz']]
          color = features['points'][:,:, data_infos['indices']['points']['color']]
          points = np.concatenate([xyz, color], -1)
          all_points.append(points)
          n += 1
      except:
        assert n>0, "some thing wrong inside parse_pl_record"
        pass
      all_points = np.concatenate(all_points, 0)
      all_label_categories = np.concatenate(all_label_categories, 0)
      base_name = os.path.splitext( os.path.basename(filename) )[0]
      if is_rm_void_labels:
        base_name = base_name + '_rmvoid'

      ply_fn = os.path.join(ply_dir, base_name+'.ply')
      create_ply_dset( dataset_name, all_points,  ply_fn)
      #
      ply_fn = os.path.join(ply_dir+'_labeled', base_name+'.ply')
      create_ply_dset( dataset_name, all_points[...,0:3],  ply_fn, all_label_categories)

if __name__ == '__main__':
  dataset_name = 'MODELNET40'
  dataset_name = 'MATTERPORT'
  dset_path = '/home/z/Research/SparseVoxelNet/data/{}_H5TF'.format(dataset_name)
  num_point = {'MODELNET40':None, 'MATTERPORT':60000}
  block_size = {'MODELNET40':None, 'MATTERPORT':np.array([6.4,6.4,6.4]) }

  rawh5_glob = os.path.join(dset_path, 'rawh5/*/*.rh5')
  #rawh5_glob = os.path.join(dset_path, 'rawh5/D7N2EKCX4Sj/*.rh5')
  tfrecord_path = os.path.join(dset_path, 'raw_tfrecord')

  #main_write(dataset_name, rawh5_glob, tfrecord_path, num_point[dataset_name], block_size[dataset_name])

  main_merge_tfrecord(dataset_name, tfrecord_path)

  #main_gen_ply(dataset_name, tfrecord_path)

  #get_dataset_summary(dataset_name, tfrecord_path)
  #get_dset_shape_idxs(tfrecord_path)

