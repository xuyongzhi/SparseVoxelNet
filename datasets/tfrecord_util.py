import h5py, os, glob,sys
import numpy as np
import tensorflow as tf
#from datasets.block_data_prep_util import Raw_H5f
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
import math
import utils.ply_util as ply_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)

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


def get_shape0(t):
  if isinstance(t, tf.Tensor):
    s0 = t.shape[0].value
    if s0==None:
      s0 = tf.shape(t)[0]
    return s0
  else:
    return t.shape[0]


def parse_record(tfrecord_serialized, is_training, dset_shape_idx, \
                    is_rm_void_labels=False, gen_ply=False):
    from utils.aug_data_tf import aug_main, aug_views
    #if dset_shape_idx!=None:
    #  from aug_data_tf import aug_data, tf_Rz
    #  R = tf_Rz(1)
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    features_map = {
        'vertex_f': tf.FixedLenFeature([], tf.string),
        'vertex_i': tf.FixedLenFeature([], tf.string),
        'face_i':   tf.FixedLenFeature([], tf.string),
        'valid_num_face': tf.FixedLenFeature([1], tf.int64, default_value=-1)
    }

    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features=features_map,
                                                name='pl_features')

    #*************
    vertex_i = tf.decode_raw(tfrecord_features['vertex_i'], tf.int32)
    vertex_i = tf.reshape(vertex_i, dset_shape_idx['shape']['vertex_i'])

    vertex_f = tf.decode_raw(tfrecord_features['vertex_f'], tf.float32)
    vertex_f = tf.reshape(vertex_f, dset_shape_idx['shape']['vertex_f'])

    face_i = tf.decode_raw(tfrecord_features['face_i'], tf.int32)
    face_i = tf.reshape(face_i, dset_shape_idx['shape']['face_i'])
    valid_num_face = tfrecord_features['valid_num_face']

    #*************
    vertex_datas = {"vertex_i": vertex_i, "vertex_f": vertex_f}
    face_datas = {"face_i": face_i, "valid_num_face": valid_num_face}

    return vertex_datas, face_datas


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
  metas_fn = os.path.join(tf_path, 'shape_idx.txt')
  with open(metas_fn, 'r') as mf:
    dset_shape_idx = {}
    dataset_name = ''

    shapes = {}
    indices = {}
    the_items = ['vertex_i', 'vertex_f', 'face_i']
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
    return dset_shape_idx


def get_ele(datas, ele, dset_shape_idx):
  ds_idxs = dset_shape_idx['indices']
  for g in ds_idxs:
    if ele in ds_idxs[g]:
      ele_idx = ds_idxs[g][ele]
      ele_data = datas[g][..., ele_idx]
      #ele_data = tf.gather(datas[g], ele_idx, axis=-1)
      return ele_data
  raise ValueError, ele+' not found'

def get_dataset_summary(dataset_name, tf_path, loss_lw_gama=-1):
  dset_shape_idx = get_dset_shape_idxs(tf_path)
  dataset_summary = read_dataset_summary(tf_path)
  if dataset_summary['intact']:
    print('dataset_summary intact, no need to read')
    get_label_num_weights(dataset_summary, loss_lw_gama)
    #return dataset_summary

  #data_path = os.path.join(tf_path, 'merged_data')
  data_path = os.path.join(tf_path, 'data')
  filenames = glob.glob(os.path.join(data_path,'*region*.tfrecord'))
  assert len(filenames) > 0, data_path

  datasets_meta = DatasetsMeta(dataset_name)
  num_classes = datasets_meta.num_classes

  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset(filenames,
                                      compression_type="",
                                      buffer_size=1024*100,
                                      num_parallel_reads=1)

    batch_size = 2
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
    features_next, labels_next = dset_iterater.get_next()

    with tf.Session() as sess:
      batch_num = 0
      point_num = 0
      label_hist = np.zeros(num_classes)
      #try:
      if True:
        while(True):
          features, labels = sess.run([features_next, labels_next])

          fidx_pv_empty_mask = get_ele(features, 'fidx_pv_empty_mask',
                                       dset_shape_idx)
          valid_num_face = labels['valid_num_face']
          category_idx = dset_shape_idx['indices']['face_i']['label_category']
          category_label = labels['face_i'][:, :, category_idx]
          label_hist += np.histogram(category_label, range(num_classes+1))[0]

          batch_num += 1
          point_num += np.sum(labels['valid_num_face'])
          print('Total: %d  %d'%(batch_num, point_num))

      #except:
      #  print(label_hist)
      #  print(sys.exc_info()[0])

      dataset_summary = {}
      dataset_summary['size'] = batch_num
      dataset_summary['label_hist'] = label_hist
      write_dataset_summary(dataset_summary, tf_path)
      get_label_num_weights(dataset_summary, loss_lw_gama)
      return dataset_summary


def write_dataset_summary(dataset_summary, data_dir):
  import pickle, shutil
  summary_path = os.path.join(data_dir, 'summary.pkl')
  dataset_summary['intact'] = True
  with open(summary_path, 'w') as sf:
    pickle.dump(dataset_summary, sf)
    print(summary_path)
  print_script = os.path.join(BASE_DIR,'print_pkl.py')
  shutil.copyfile(print_script,os.path.join(data_dir,'print_pkl.py'))



class MeshSampling():
  _full_edge_dis = 3
  _max_norm_dif_angle = 15.0
  _check_optial = True

  _max_nf_perv = 9

  _vertex_eles = ['color', 'xyz', 'nxnynz', 'fidx_per_vertex', \
                  'edges_per_vertex', 'edges_pv_empty_mask', 'fidx_pv_empty_mask',\
                  'same_normal_mask', 'same_category_mask']
  _face_eles = ['label_raw_category', 'label_instance', 'label_material', \
                'label_category', 'vidx_per_face', ]

  _fi = 0
  _show_ave_num_face_perv = True

  @staticmethod
  def sess_split_sampling_rawmesh(raw_datas, _num_vertex_sp, splited_vidx,
                                  dset_metas, ply_dir):
    raw_vertex_nums = [e.shape[0] if type(e)!=type(None) else raw_datas['xyz'].shape[0]\
                         for e in splited_vidx]
    with tf.Graph().as_default():
      raw_datas_pl = {}
      for item in raw_datas:
        type_i = eval( 'tf.' + str(raw_datas[item].dtype) )
        shape_i = raw_datas[item].shape
        raw_datas_pl[item] = tf.placeholder(type_i, shape_i, item+'_pl')
      block_num = len(splited_vidx)
      splited_vidx_pl = []
      if block_num==1:
        splited_vidx_pl_ = [None]
      else:
        for bi in range(block_num):
          splited_vidx_pl.append( tf.placeholder(tf.int32, splited_vidx[bi].shape,
                                                 'splited_vidx_%d_pl'%(bi)) )
        splited_vidx_pl_ = [tf.identity(e) for e in splited_vidx_pl]

      splited_sampled_datas_ = MeshSampling.main_split_sampling_rawmesh(\
                raw_datas_pl.copy(), _num_vertex_sp, splited_vidx_pl_, dset_metas, ply_dir)

      with tf.Session() as sess:
        feed_dict = {}
        for item in raw_datas:
          feed_dict[raw_datas_pl[item]] = raw_datas[item]

        if block_num>1:
          for bi in range(block_num):
            feed_dict[splited_vidx_pl[bi]] = splited_vidx[bi]

        splited_sampled_datas = sess.run(splited_sampled_datas_, feed_dict=feed_dict)

    return splited_sampled_datas, raw_vertex_nums


  @staticmethod
  def eager_split_sampling_rawmesh(raw_datas, _num_vertex_sp, splited_vidx,
                                   dset_metas, ply_dir=None):
    start = MeshSampling._fi == 0
    MeshSampling._fi += 1
    if start:
      tf.enable_eager_execution()

    raw_vertex_nums = [e.shape[0] if type(e)!=type(None) else raw_datas['xyz'].shape[0]\
                         for e in splited_vidx]

    splited_sampled_datas = MeshSampling.main_split_sampling_rawmesh(
                            raw_datas, _num_vertex_sp, splited_vidx, dset_metas, ply_dir=ply_dir)

    bn = len(splited_sampled_datas)
    for bi in range(bn):
      for item in splited_sampled_datas[bi]:
        if isinstance(splited_sampled_datas[bi][item], tf.Tensor):
          splited_sampled_datas[bi][item] = splited_sampled_datas[bi][item].numpy()
    return splited_sampled_datas, raw_vertex_nums

  @staticmethod
  def main_split_sampling_rawmesh(raw_datas, _num_vertex_sp, splited_vidx,
                                  dset_metas, ply_dir=None):
    num_vertex0 = get_shape0(raw_datas['xyz'])

    is_show_shapes = False
    IsGenply_Raw = False
    IsGenply_SameMask = False
    IsGenply_Cleaned = False
    IsGenply_Splited = False
    IsGenply_SplitedSampled = False

    if IsGenply_Raw:
      MeshSampling.gen_mesh_ply_basic(raw_datas, 'Raw', 'raw', ply_dir)
    #***************************************************************************
    #face idx per vetex, edges per vertyex
    fidx_per_vertex, fidx_pv_empty_mask, edges_per_vertex, \
      edges_pv_empty_mask, lonely_vertex_idx = \
                    MeshSampling.get_fidx_nbrv_per_vertex(
                                  raw_datas['vidx_per_face'], num_vertex0)
    raw_datas['fidx_per_vertex'] = fidx_per_vertex
    raw_datas['fidx_pv_empty_mask'] = fidx_pv_empty_mask
    raw_datas['edges_per_vertex'] = edges_per_vertex
    raw_datas['edges_pv_empty_mask'] = edges_pv_empty_mask

    if is_show_shapes:
      MeshSampling.show_datas_shape(raw_datas, 'raw datas')

    #***************************************************************************
    # same mask
    same_normal_mask, same_category_mask = MeshSampling.get_simplicity_label(
                                    fidx_per_vertex, fidx_pv_empty_mask,
                                    edges_per_vertex, edges_pv_empty_mask,
                                    raw_datas['nxnynz'],
                                    raw_datas['label_category'],
                                    raw_datas['label_instance'])
    same_norm_cat_mask = (same_normal_mask + same_category_mask) / 2

    raw_datas['same_normal_mask'] = tf.expand_dims(same_normal_mask,1)
    raw_datas['same_category_mask'] = tf.expand_dims(same_category_mask,1)

    if IsGenply_SameMask:
      MeshSampling.gen_ply_raw(raw_datas, same_normal_mask,
                                 same_category_mask, same_norm_cat_mask, ply_dir)
    #***************************************************************************
    # rm some labels
    with tf.variable_scope('rm_some_labels'):
      raw_datas, splited_vidx = MeshSampling.rm_some_labels(raw_datas, lonely_vertex_idx, dset_metas, splited_vidx)
    if IsGenply_Cleaned:
      MeshSampling.gen_mesh_ply_basic(raw_datas, 'Cleaned', 'Cleaned', ply_dir)

    #***************************************************************************
    # split mesh
    block_num = len(splited_vidx)
    if block_num==1:
      splited_datas = [raw_datas]
    else:
      with tf.variable_scope('split_vertex'):
        splited_datas = MeshSampling.split_vertex(raw_datas, splited_vidx)

    if IsGenply_Splited:
      for bi in range(block_num):
        MeshSampling.gen_mesh_ply_basic(splited_datas[bi], 'Splited' ,'Block_{}'.format(bi), ply_dir)
    #***************************************************************************
    # sampling
    for bi in range(block_num):
      with tf.variable_scope('sampling_mesh'):
        splited_datas[bi] = MeshSampling.sampling_mesh(
                                             _num_vertex_sp, splited_datas[bi])
    splited_sampled_datas = splited_datas

    if IsGenply_SplitedSampled:
      for bi in range(block_num):
        MeshSampling.gen_mesh_ply_basic(splited_sampled_datas[bi], 'SplitedSampled',
                        'Block{}_sampled_{}'.format(bi, _num_vertex_sp), ply_dir)

    if is_show_shapes:
      MeshSampling.show_datas_shape(splited_sampled_datas, 'sampled datas')

    return splited_sampled_datas

  @staticmethod
  def split_vertex(raw_datas, splited_vidx):
    num_vertex0 = get_shape0(raw_datas['xyz'])

    # get splited_fidx
    splited_fidx = []
    vidx_per_face_new_ls = []
    for bi,block_vidx in enumerate(splited_vidx):
      with tf.variable_scope('spv_dsf_b%d'%(bi)):
        face_sp_indices, vidx_per_face_new = MeshSampling.down_sampling_face(\
                  block_vidx, num_vertex0, raw_datas['vidx_per_face'])
      splited_fidx.append(face_sp_indices)
      vidx_per_face_new_ls.append(vidx_per_face_new)

    # do split
    splited_datas = []
    for bi,block_vidx in enumerate(splited_vidx):
      block_datas = {}
      for item in MeshSampling._vertex_eles:
        with tf.variable_scope('spv_b%d_V_%s'%(bi, item)):
          block_datas[item] = tf.gather( raw_datas[item], block_vidx )
      for item in MeshSampling._face_eles:
        if item == 'vidx_per_face':
          continue
        with tf.variable_scope('spv_b%d_F_%s'%(bi, item)):
          block_datas[item] = tf.gather( raw_datas[item], splited_fidx[bi])
      block_datas['vidx_per_face'] = vidx_per_face_new_ls[bi]
      splited_datas.append(block_datas)

      #MeshSampling.show_datas_shape(block_datas, 'block %d'%(bi))
    return splited_datas

  @staticmethod
  def split_face(splited_datas):
    block

  @staticmethod
  def show_datas_shape(datas, flag=''):
    shape_strs = '\n{}\n'.format(flag)
    def show_one_item(item):
      if item not in datas:
        return ''
      di = datas[item]
      if isinstance(di, tf.Tensor):
        shape_str = di.shape.as_list()
      else:
        shape_str = str(di.shape)
      shape_str = '\t{}: {}\n'.format(item, shape_str)
      return shape_str

    for item in MeshSampling._vertex_eles:
      shape_strs += show_one_item(item)
    shape_strs += '\n'
    for item in MeshSampling._face_eles:
      shape_strs += show_one_item(item)
    print(shape_strs)
    return shape_strs

  @staticmethod
  def get_fidx_nbrv_per_vertex(vidx_per_face, num_vertex0):
    '''
    Inputs: [F,3] []
    Output: [N, ?]
    '''
    num_face = tf.shape(vidx_per_face)[0]
    face_indices = tf.reshape(tf.range(0, num_face), [-1,1,1])
    face_indices = tf.tile(face_indices, [1, 3,1])
    vidx_per_face = tf.expand_dims(vidx_per_face, 2)
    vidx_fidx = tf.concat([vidx_per_face, face_indices],  2)
    vidx_fidx = tf.reshape(vidx_fidx, [-1, 2])

    #***************************************************************************
    # shuffle before sort if need to sample later

    # sort by vidx. Put all fidx belong to same vertex together
    sort_indices = tf.contrib.framework.argsort(vidx_fidx[:,0])
    vidx_fidx_flat_sorted = tf.gather(vidx_fidx, sort_indices)

    #***************************************************************************
    # get unique indices
    vidx_unique, vidx_perf, nface_per_v = tf.unique_with_counts(vidx_fidx_flat_sorted[:,0])
    check_vertex_num = tf.assert_equal(vidx_unique[-1]+1, num_vertex0,
                                       message="num_vertex incorrect")
    with tf.control_dependencies([check_vertex_num]):
      nface_per_v = tf.identity(nface_per_v)
    max_nf_perv = tf.reduce_max(nface_per_v)
    max_nf_perv = tf.maximum(max_nf_perv, MeshSampling._max_nf_perv)
    mean_nf_perv = tf.reduce_mean(nface_per_v)
    min_nf_perv = tf.reduce_min(nface_per_v)

    #***************************************************************************
    # get face idx per vertex flat
    nface_cumsum0 = tf.cumsum(nface_per_v)[0:-1]
    nface_cumsum0 = tf.concat([tf.constant([0], tf.int32), nface_cumsum0], 0)
    nface_cumsum1 = tf.gather(nface_cumsum0, vidx_perf)
    auged_vidx = tf.range(tf.shape(vidx_fidx_flat_sorted)[0])
    fidx_per_v_flat = tf.expand_dims(auged_vidx - nface_cumsum1, 1)

    #***************************************************************************
    # reshape
    vidx_fidxperv = tf.concat([vidx_fidx_flat_sorted[:,0:1], fidx_per_v_flat], 1)

    fidx_per_vertex = tf.scatter_nd(vidx_fidxperv, vidx_fidx_flat_sorted[:,1]+1, \
                                    [num_vertex0, max_nf_perv]) - 1
    fidx_pv_empty_mask = tf.equal(fidx_per_vertex, -1)

    MeshSampling._show_ave_num_face_perv = True
    if MeshSampling._show_ave_num_face_perv:
      num_face_perv = tf.reduce_sum(tf.cast(fidx_pv_empty_mask, tf.int32), -1)
      ave_num_face_perv = tf.reduce_mean(num_face_perv)
      fidx_per_vertex = tf.Print(fidx_per_vertex, [ave_num_face_perv],
                                     "\nave_num_face_perv: ")

    fidx_per_vertex = fidx_per_vertex[:, 0:MeshSampling._max_nf_perv]
    fidx_per_vertex.set_shape([num_vertex0, MeshSampling._max_nf_perv])
    fidx_pv_empty_mask = fidx_pv_empty_mask[:, 0:MeshSampling._max_nf_perv]
    fidx_pv_empty_mask.set_shape([num_vertex0, MeshSampling._max_nf_perv])

    #***************************************************************************
    # check is there any vertex belong to no faces
    lonely_vertex_mask = tf.equal(fidx_per_vertex[:,0], -1)
    any_lonely_vertex = tf.reduce_any(lonely_vertex_mask)

    lonely_vertex_idx0 = tf.squeeze(tf.cast(tf.where(lonely_vertex_mask), tf.int32),1)
    lonely_vertex_idx = tf.cond(any_lonely_vertex, lambda: lonely_vertex_idx0,
                                lambda: tf.zeros([], tf.int32))


    # set -1 as the first one
    empty_indices = tf.cast(tf.where(fidx_pv_empty_mask), tf.int32)
    the_first_face_dix = tf.gather(fidx_per_vertex[:,0], empty_indices[:,0],axis=0)
    tmp = tf.scatter_nd(empty_indices, the_first_face_dix+1, tf.shape(fidx_per_vertex))
    fidx_per_vertex = fidx_per_vertex + tmp


    #***************************************************************************
    # get neighbor verties
    edges_per_vertexs_flat = tf.gather(tf.squeeze(vidx_per_face,-1), vidx_fidx_flat_sorted[:,1])

    # remove self vertex
    self_mask = tf.equal(edges_per_vertexs_flat, vidx_fidx_flat_sorted[:,0:1])
    self_mask = tf.cast(self_mask, tf.int32)
    sort_indices = tf.contrib.framework.argsort(self_mask, axis=-1)
    sort_indices = tf.expand_dims(sort_indices, 2)
    tmp0 = tf.reshape(tf.range(tf.shape(sort_indices)[0]), [-1,1,1])
    tmp0 = tf.tile(tmp0, [1,3,1])
    sort_indices = tf.concat([tmp0, sort_indices], 2)

    edges_per_vertexs_flat = tf.gather_nd(edges_per_vertexs_flat, sort_indices)
    edges_per_vertexs_flat = edges_per_vertexs_flat[:,0:2]

    # reshape edges_per_vertexs_flat
    edges_per_vertex = tf.scatter_nd(vidx_fidxperv, edges_per_vertexs_flat+1,\
                                     [num_vertex0, max_nf_perv, 2])-1

    edges_per_vertex = edges_per_vertex[:, 0:MeshSampling._max_nf_perv,:]
    edges_per_vertex.set_shape([num_vertex0, MeshSampling._max_nf_perv, 2])
    edges_pv_empty_mask = tf.cast(tf.equal(edges_per_vertex, -1), tf.bool)

    # set -1 as the first one
    the_first_edges_dix = tf.gather(edges_per_vertex[:,0,:], empty_indices[:,0])
    tmp = tf.scatter_nd(empty_indices, the_first_edges_dix+1, tf.shape(edges_per_vertex))
    edges_per_vertex += tmp

    # reshape and flat to the same dims with other elements, to store in the
    # same array
    edges_per_vertex = tf.reshape(edges_per_vertex, [num_vertex0, MeshSampling._max_nf_perv*2])
    edges_pv_empty_mask = tf.reshape(edges_pv_empty_mask, [num_vertex0, MeshSampling._max_nf_perv*2])

    #***********************
    # set face idx for the lonely vertex as 0, just avoid grammer error. but should not be used
    fidx_per_vertex += tf.cast(tf.expand_dims( lonely_vertex_mask, 1), tf.int32)
    edges_per_vertex += tf.cast(tf.expand_dims( lonely_vertex_mask, 1), tf.int32)

    return fidx_per_vertex, fidx_pv_empty_mask, edges_per_vertex, \
            edges_pv_empty_mask, lonely_vertex_idx


  @staticmethod
  def get_simplicity_label( fidx_per_vertex, fidx_pv_empty_mask,
                        edges_per_vertex, edges_pv_empty_mask,
                        vertex_nxnynz, face_label_category, face_label_instance):
    '''
    Inputs: [N, ?] [N,3] [F,1]
    A point is simple if:
    (1) all the faces belong to one point with same category and instance (and) material
    (2) all the faces belong to one point with similar normal
    '''
    num_vertex0 = tf.shape(vertex_nxnynz)[0]
    num_vertex0_f = tf.cast(num_vertex0, tf.float64)

    def get_same_category_mask():
      # get same_category_mask
      face_label_category_ = tf.squeeze(face_label_category, 1)
      vertex_label_categories = tf.gather(face_label_category_, fidx_per_vertex)

      same_category_mask = tf.equal(vertex_label_categories, vertex_label_categories[:,0:1])
      same_category_mask = tf.logical_or(same_category_mask, fidx_pv_empty_mask)
      same_category_mask = tf.reduce_all(same_category_mask, 1)
      return same_category_mask

    # get normal same mask
    def get_same_normal_mask(max_normal_dif):
      normal = tf.gather(vertex_nxnynz, edges_per_vertex)
      norm_dif_angle = tf.matmul(normal, tf.expand_dims(normal[:,0,:], -1))
      norm_dif_angle = tf.squeeze(norm_dif_angle, -1)
      max_norm_dif_angle =  np.cos(MeshSampling._max_norm_dif_angle/180.0*np.pi)
      same_normal_mask = tf.greater(norm_dif_angle, max_norm_dif_angle)
      same_normal_mask = tf.logical_or(same_normal_mask, edges_pv_empty_mask)
      same_normal_mask = tf.reduce_all(same_normal_mask, 1)
      return same_normal_mask

    def extend_same_mask(same_mask0):
      same_mask0 = tf.greater(same_mask0, 0)
      extended_mask = tf.gather(same_mask0, edges_per_vertex)
      extended_mask = tf.logical_or(extended_mask, edges_pv_empty_mask)
      extended_mask = tf.reduce_all(extended_mask, 1)
      extended_mask = tf.cast(extended_mask, tf.int8)
      return extended_mask


    same_normal_mask = tf.cast(get_same_normal_mask(1e-2), tf.int8)
    same_category_mask = tf.cast(get_same_category_mask(), tf.int8)

    for d in range(MeshSampling._full_edge_dis-1):
      same_category_mask  += extend_same_mask(same_category_mask)
      same_normal_mask  += extend_same_mask(same_normal_mask)

    return same_normal_mask, same_category_mask

  @staticmethod
  def same_mask_nums(same_mask):
    same_nums = []
    for e in range(MeshSampling._full_edge_dis+1):
      same_num = tf.reduce_sum(tf.cast(tf.equal(same_mask, e), tf.int32))
      same_nums.append(same_num)
    return same_nums

  @staticmethod
  def same_mask_rates(same_mask,  pre=''):
    num_total = tf.cast( get_shape0( same_mask ), tf.float64)
    same_nums = MeshSampling.same_mask_nums(same_mask)
    same_rates = [1.0*n/num_total for n in same_nums]
    same_rates[0] = tf.Print(same_rates[0], same_rates, message=pre+' same rates: ')
    #print('\n{} same rate:{}\n'.format(pre, same_rates))
    return same_rates

  @staticmethod
  def get_face_same_mask(vertex_same_mask, vidx_per_face):
    same_mask = tf.gather(vertex_same_mask, vidx_per_face)
    face_same_mask = tf.reduce_max(same_mask, 1)
    return face_same_mask

  @staticmethod
  def rm_some_labels(raw_datas, lonely_vertex_idx, dset_metas, splited_vidx):
    unwanted_classes = ['void']
    unwanted_labels = tf.constant([[dset_metas.class2label[c] for c in unwanted_classes]], tf.int32)

    label_category = raw_datas['label_category']
    keep_face_mask = tf.not_equal(label_category, unwanted_labels)
    keep_face_mask = tf.reduce_all(keep_face_mask, 1)
    #keep_face_idx = tf.cast(tf.where(keep_face_mask), tf.int32)
    #keep_face_idx = tf.squeeze(keep_face_idx, 1)

    fidx_per_vertex = raw_datas['fidx_per_vertex']
    keep_vertex_mask = tf.gather(keep_face_mask, fidx_per_vertex)
    keep_vertex_mask = tf.reduce_any(keep_vertex_mask, 1)
    keep_vertex_idx = tf.cast(tf.where(keep_vertex_mask), tf.int32)
    keep_vertex_idx = tf.squeeze(keep_vertex_idx, 1)

    rm_face_num = tf.reduce_sum(tf.cast(keep_face_mask, tf.int32))
    rm_vertex_num = tf.reduce_sum(tf.cast(keep_vertex_mask, tf.int32))

    num_vertex0 = get_shape0(raw_datas['xyz'])

    keep_face_idx, vidx_per_face_new = MeshSampling.down_sampling_face(
                                keep_vertex_idx,
                                num_vertex0, raw_datas['vidx_per_face'])

    raw_datas = MeshSampling.gather_datas(raw_datas, keep_vertex_idx,
                                    keep_face_idx, vidx_per_face_new)

    # clean splited_vidx
    new_vidx_2_old_vidx = tf.scatter_nd(tf.expand_dims(keep_vertex_idx,-1), tf.range(tf.shape(keep_vertex_idx)[0])+1, [num_vertex0])-1
    if len(splited_vidx)>1:
      for i in range(len(splited_vidx)):
        new_vidx_i = tf.gather(new_vidx_2_old_vidx, splited_vidx[i])
        keep_idx_i = tf.cast(tf.where(tf.greater(new_vidx_i, -1)), tf.int32)
        splited_vidx[i] = tf.squeeze(tf.gather(new_vidx_i, keep_idx_i),1)
    return raw_datas, splited_vidx

  @staticmethod
  def sampling_mesh( _num_vertex_sp, raw_datas):
    num_vertex0 = get_shape0(raw_datas['xyz'])
    if not isinstance(num_vertex0, tf.Tensor):
      sampling_rate = 1.0 * _num_vertex_sp / tf.cast(num_vertex0, tf.float32)
      print('\nsampling org_num={}, fixed_num={}, rate={}\n'.format(num_vertex0, _num_vertex_sp, sampling_rate))

    is_down_sampling = tf.less(_num_vertex_sp, num_vertex0)
    sampled_datas = tf.cond(is_down_sampling,
                  lambda: MeshSampling.down_sampling_mesh(_num_vertex_sp, raw_datas.copy()),
                  lambda: MeshSampling.up_sampling_mesh(_num_vertex_sp, raw_datas.copy()))
    return sampled_datas

  @staticmethod
  def up_sampling_mesh( _num_vertex_sp, raw_datas):
    MeshSampling.show_datas_shape(raw_datas)
    num_vertex0 = get_shape0(raw_datas['xyz'])
    duplicate_num = _num_vertex_sp - num_vertex0
    with tf.control_dependencies([tf.assert_greater(duplicate_num, 0, message="duplicate_num")]):
      duplicate_num = tf.identity(duplicate_num)

    raw_datas['same_category_mask']  = tf.cast(raw_datas['same_category_mask'], tf.int32)
    raw_datas['same_normal_mask']  = tf.cast(raw_datas['same_normal_mask'], tf.int32)
    for item in raw_datas:
      is_vertex = item in MeshSampling._vertex_eles
      if is_vertex:
        duplicated = tf.tile(raw_datas[item][-1:,:], [duplicate_num, 1])
        raw_datas[item] = tf.concat([raw_datas[item], duplicated], 0)
    raw_datas['same_category_mask']  = tf.cast(raw_datas['same_category_mask'], tf.int8)
    raw_datas['same_normal_mask']  = tf.cast(raw_datas['same_normal_mask'], tf.int8)
    return raw_datas

  @staticmethod
  def down_sampling_mesh(_num_vertex_sp, raw_datas):
    num_vertex0 = get_shape0(raw_datas['xyz'])
    vertex_sp_indices = MeshSampling.down_sampling_vertex(
                                raw_datas['same_normal_mask'], _num_vertex_sp)

    face_sp_indices, vidx_per_face_new = MeshSampling.down_sampling_face(
                                vertex_sp_indices,
                                num_vertex0, raw_datas['vidx_per_face'])
    raw_datas = MeshSampling.gather_datas(raw_datas, vertex_sp_indices,
                                    face_sp_indices, vidx_per_face_new)
    return raw_datas

  @staticmethod
  def gather_datas(datas, vertex_sp_indices, face_sp_indices, vidx_per_face_new):
    num_vertex0 = get_shape0(datas['xyz'])

    for item in datas:
      if item == 'vidx_per_face':
        continue
      is_vertex_0 = tf.equal(tf.shape(datas[item])[0], num_vertex0)
      is_vertex = item in MeshSampling._vertex_eles
      check0 = tf.assert_equal(is_vertex, is_vertex_0)
      with tf.control_dependencies([check0]):
        is_vertex = tf.identity(is_vertex)

      sp_indices = tf.cond(is_vertex,
                           lambda: vertex_sp_indices,
                           lambda: face_sp_indices )
      datas[item] = tf.gather(datas[item], sp_indices)

    datas['vidx_per_face'] = vidx_per_face_new
    return datas

  @staticmethod
  def up_sampling_vertex(same_normal_mask, _num_vertex_sp):
    num_vertex0 = get_shape0( same_normal_mask )
    #simple_indices = tf.squeeze(tf.where(tf.greater_equal(
    #                        same_normal_mask, MeshSampling._full_edge_dis)),1)
    duplicate_num = _num_vertex_sp - num_vertex0
    #duplicate_indices = tf.tile( simple_indices[0:1], [duplicate_num] )
    duplicate_indices = tf.ones([duplicate_num], tf.int32) * (num_vertex0 -1)
    vertex_sp_indices = tf.concat([tf.range(num_vertex0), duplicate_indices], 0)
    return vertex_sp_indices

  @staticmethod
  def down_sampling_vertex(same_normal_mask, _num_vertex_sp):
    same_normal_mask = tf.squeeze(same_normal_mask)
    num_vertex0 = tf.shape(same_normal_mask)[0]
    sampling_rate = 1.0 * tf.cast(_num_vertex_sp, tf.float32) / tf.cast(num_vertex0, tf.float32)
    #print('org num:{}, sampled num:{}, sampling_rate:{}'.format(
    #                              num_vertex0, _num_vertex_sp, sampling_rate))
    del_num = num_vertex0 - _num_vertex_sp
    same_nums = MeshSampling.same_mask_nums(same_normal_mask)
    full_dis = MeshSampling._full_edge_dis

    #*********************
    # max_dis: the max dis that provide enough simple vertices to remove
    assert len(same_nums) == full_dis + 1

    j = tf.constant(0, tf.int32)
    max_dis = tf.constant(full_dis, tf.int32)
    simple_is_enough_to_rm = tf.constant(False, tf.bool)

    def cond(j, simple_is_enough_to_rm, max_dis):
        cond0 = tf.less(j, full_dis)
        cond1 = tf.logical_not(simple_is_enough_to_rm)
        cond = tf.logical_and(cond0, cond1)
        return cond
    def body(j, simple_is_enough_to_rm, max_dis):
      max_dis = full_dis -j
      simple_num = tf.reduce_sum(tf.gather(same_nums, tf.range(full_dis-j,full_dis+1)))
      simple_is_enough_to_rm =  tf.greater(simple_num, del_num)
      j += 1
      return j, simple_is_enough_to_rm, max_dis
    j, simple_is_enough_to_rm, max_dis = tf.while_loop(cond, body, [j, simple_is_enough_to_rm, max_dis])

    max_dis = tf.cast(max_dis, tf.int8)

    #*********************
    complex_indices = tf.squeeze(tf.where(tf.less(same_normal_mask, max_dis)),1)
    complex_indices = tf.cast(complex_indices, tf.int32)
    complex_num = tf.shape(complex_indices)[0]
    simple_indices = tf.squeeze(tf.where(tf.greater_equal(same_normal_mask, max_dis)),1)
    simple_indices = tf.cast(simple_indices, tf.int32)
    simple_num = tf.shape(simple_indices)[0]
    # max_dis>0: simple vertices are enough to del. Keep all complex, rm from
    # simple: complex_indices + part of simple_indices
    # max_dis==0: rm all simple, and part of complex

    def rm_part_of_simple_only():
      sp_num_from_simple = _num_vertex_sp - complex_num
      tmp = tf.random_shuffle(tf.range(simple_num))[0:sp_num_from_simple]
      simple_sp_indices = tf.gather(simple_indices, tmp)
      vertex_sp_indices = tf.concat([complex_indices, simple_sp_indices], 0)
      return vertex_sp_indices

    def rm_all_simple_partof_complex():
      sp_num_from_complex = _num_vertex_sp
      tmp = tf.random_shuffle(tf.range(complex_num))[0:sp_num_from_complex]
      vertex_sp_indices = tf.gather(complex_indices, tmp)
      return vertex_sp_indices

    vertex_sp_indices = tf.cond(simple_is_enough_to_rm,
                                  rm_part_of_simple_only,
                                  rm_all_simple_partof_complex )

    vertex_sp_indices = tf.contrib.framework.sort(vertex_sp_indices)

    check_s = tf.assert_equal(tf.shape(vertex_sp_indices)[0], _num_vertex_sp)
    with tf.control_dependencies([check_s]):
      vertex_sp_indices = tf.identity(vertex_sp_indices)
    vertex_sp_indices.set_shape([_num_vertex_sp])

    if MeshSampling._check_optial:
      # check no duplicate
      tmp0, tmp1, tmp_count = tf.unique_with_counts(vertex_sp_indices)
      max_count = tf.reduce_max(tmp_count)
      check_no_duplicate = tf.assert_equal(max_count,1)
      with tf.control_dependencies([check_no_duplicate]):
        vertex_sp_indices = tf.identity(vertex_sp_indices)

    return vertex_sp_indices

  @staticmethod
  def down_sampling_face(vertex_sp_indices, num_vertex0, vidx_per_face):
    if isinstance(vertex_sp_indices, tf.Tensor):
      _num_vertex_sp = tf.shape(vertex_sp_indices)[0]
    else:
      _num_vertex_sp = vertex_sp_indices.shape[0]
    vertex_sp_indices = tf.expand_dims(tf.cast(vertex_sp_indices, tf.int32),1)
    # scatter new vertex index
    raw_vidx_2_sp_vidx = tf.scatter_nd(vertex_sp_indices, tf.range(_num_vertex_sp)+1, [num_vertex0])-1
    vidx_per_face_new = tf.gather(raw_vidx_2_sp_vidx, vidx_per_face)

    # rm lost faces
    remain_mask = tf.reduce_all(tf.greater(vidx_per_face_new, -1),1)
    face_sp_indices = tf.squeeze(tf.where(remain_mask), 1)
    face_sp_indices = tf.cast(face_sp_indices, tf.int32)

    vidx_per_face_new = tf.gather(vidx_per_face_new, face_sp_indices)

    if MeshSampling._check_optial:
      max_vidx = tf.reduce_max(vidx_per_face_new)
      check0 = tf.assert_less_equal(max_vidx, _num_vertex_sp-1)
      with tf.control_dependencies([check0]):
        vidx_per_face_new = tf.identity(vidx_per_face_new)

    return face_sp_indices, vidx_per_face_new


  @staticmethod
  def gen_mesh_ply_basic(datas, dir_name='', base_name='category_labeled', ply_dir=None):
    if ply_dir == None:
      ply_dir = '/tmp'
    path =  '{}/{}'.format(ply_dir, dir_name)
    ply_fn = '{}/{}.ply'.format(path, base_name)
    for item in datas:
      if isinstance(datas[item], tf.Tensor):
        datas[item] = datas[item].numpy()
    ply_util.gen_mesh_ply(ply_fn, datas['xyz'], datas['vidx_per_face'],
                          datas['label_category'])

  @staticmethod
  def gen_ply_raw(raw_datas, same_normal_mask, same_category_mask, same_norm_cat_mask, ply_dir):
    if ply_dir == None:
      ply_dir = '/tmp'
    # face same mask for generating ply
    face_same_normal_mask = MeshSampling.get_face_same_mask(same_normal_mask,
                                              raw_datas['vidx_per_face'])
    face_same_category_mask = MeshSampling.get_face_same_mask(same_category_mask,
                                              raw_datas['vidx_per_face'])
    face_same_norm_cat_mask = MeshSampling.get_face_same_mask(same_norm_cat_mask,
                                              raw_datas['vidx_per_face'])

    # same rates
    same_norm_rates = MeshSampling.same_mask_rates(same_normal_mask, 'v_normal')
    same_category_rates = MeshSampling.same_mask_rates(same_category_mask, 'v_category')
    same_norm_cat_rates = MeshSampling.same_mask_rates(same_norm_cat_mask, 'v_norm_cat')

    face_norm_rates = MeshSampling.same_mask_rates(face_same_normal_mask, 'f_normal')
    face_category_rates = MeshSampling.same_mask_rates(face_same_category_mask, 'f_category')
    face_norm_cat_rates = MeshSampling.same_mask_rates(face_same_norm_cat_mask, 'f_norm_cat')

    same_normal_mask = same_normal_mask.numpy()
    same_category_mask = same_category_mask.numpy()
    face_same_normal_mask = tf.Print(face_same_normal_mask,
                                      [MeshSampling._max_norm_dif_angle],
                                      message='max_normal_dif_angle')


    ply_util.gen_mesh_ply('{}/face_same_normal_{}degree.ply'.format(ply_dir,\
                                    int(MeshSampling._max_norm_dif_angle)),
                          raw_datas['xyz'],
                          raw_datas['vidx_per_face'],
                          face_label = face_same_normal_mask)

    ply_util.gen_mesh_ply('{}/face_same_category.ply'.format(ply_dir), raw_datas['xyz'],
                          raw_datas['vidx_per_face'],
                          face_label = face_same_category_mask)
    ply_util.gen_mesh_ply('{}/face_same_norm_{}degree_cat.ply'.format(ply_dir, \
                                      int(MeshSampling._max_norm_dif_angle)),
                          raw_datas['xyz'],
                          raw_datas['vidx_per_face'],
                          face_label = face_same_norm_cat_mask)


  def show_simplity_label(raw_datas, same_normal_mask, same_category_mask):
    '''
    numpy inputs
    '''



if __name__ == '__main__':
  dataset_name = 'MATTERPORT'
  dset_path = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans'
  tfrecord_path = '/DS/Matterport3D/MATTERPORT_TF/mesh_tfrecord'
  tfrecord_path = '/home/z/Research/SparseVoxelNet/data/MATTERPORT_TF/mesh_tfrecord'
  get_dataset_summary(dataset_name, tfrecord_path)


