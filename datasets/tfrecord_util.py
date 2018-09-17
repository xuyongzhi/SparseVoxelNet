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



class MeshSampling():
  _full_edge_dis = 3
  _max_norm_dif_angle = 15.0
  _check_optial = True

  _vertex_eles = ['color', 'xyz', 'nxnynz', 'face_idx_per_vertex', \
                  'edges_per_vertex', 'edges_pv_empty_mask', 'fidx_pv_empty_mask',\
                  'same_normal_mask', 'same_category_mask']
  _face_eles = ['label_raw_category', 'label_instance', 'label_material', \
                'label_category', 'vertex_idx_per_face', ]

  _fi = 0

  @staticmethod
  def sess_split_sampling_rawmesh(raw_datas, _num_vertex_sp, splited_vidx):
    with tf.Graph().as_default():
      raw_datas_pl = {}
      for item in raw_datas:
        type_i = eval( 'tf.' + str(raw_datas[item].dtype) )
        shape_i = raw_datas[item].shape
        raw_datas_pl[item] = tf.placeholder(type_i, shape_i, item)
      block_num = len(splited_vidx)
      splited_vidx_pl = []
      if block_num==1:
        splited_vidx_pl = [None]
      else:
        for bi in range(block_num):
          splited_vidx_pl.append( tf.placeholder(tf.int32, splited_vidx[bi].shape,
                                                 'splited_vidx_%d'%(bi)) )

      splited_sampled_datas_ = MeshSampling.main_split_sampling_rawmesh(\
                              raw_datas_pl, _num_vertex_sp, splited_vidx_pl)

      raw_vertex_nums = [e.shape[0] for e in splited_vidx]

      with tf.Session() as sess:
        feed_dict = {}
        for item in raw_datas:
          feed_dict[raw_datas_pl[item]] = raw_datas[item]
        for bi in range(block_num):
          feed_dict[splited_vidx_pl[bi]] = splited_vidx[bi]

        splited_sampled_datas = sess.run(splited_sampled_datas_, feed_dict=feed_dict)

    return splited_sampled_datas, raw_vertex_nums


  @staticmethod
  def eager_split_sampling_rawmesh(raw_datas, _num_vertex_sp, splited_vidx):
    start = MeshSampling._fi == 0
    MeshSampling._fi += 1
    if start:
      tf.enable_eager_execution()
    splited_sampled_datas = MeshSampling.main_split_sampling_rawmesh(
                                      raw_datas, _num_vertex_sp, splited_vidx)

    raw_vertex_nums = [e.shape[0] for e in splited_vidx]
    bn = len(splited_sampled_datas)
    for bi in range(bn):
      for item in splited_sampled_datas[bi]:
        if isinstance(splited_sampled_datas[bi][item], tf.Tensor):
          splited_sampled_datas[bi][item] = splited_sampled_datas[bi][item].numpy()
    return splited_sampled_datas, raw_vertex_nums

  @staticmethod
  def main_split_sampling_rawmesh(raw_datas, _num_vertex_sp, splited_vidx):
    num_vertex0 = raw_datas['xyz'].shape[0]

    is_show_shapes = True
    IsGenply_SameMask = False
    IsGenply_Splited = False
    IsGenply_SplitedSampled = False

    #***************************************************************************
    #face idx per vetex, edges per vertyex
    face_idx_per_vertex, fidx_pv_empty_mask, edges_per_vertex, edges_pv_empty_mask = \
                      MeshSampling.get_fidx_nbrv_per_vertex(
                      raw_datas['vertex_idx_per_face'], num_vertex0)
    raw_datas['face_idx_per_vertex'] = face_idx_per_vertex
    raw_datas['fidx_pv_empty_mask'] = fidx_pv_empty_mask
    raw_datas['edges_per_vertex'] = edges_per_vertex
    raw_datas['edges_pv_empty_mask'] = edges_pv_empty_mask

    if is_show_shapes:
      MeshSampling.show_datas_shape(raw_datas, 'raw datas')

    #***************************************************************************
    # same mask
    same_normal_mask, same_category_mask = MeshSampling.get_simplicity_label(
                                    face_idx_per_vertex, fidx_pv_empty_mask,
                                    edges_per_vertex, edges_pv_empty_mask,
                                    raw_datas['nxnynz'],
                                    raw_datas['label_category'],
                                    raw_datas['label_instance'])
    same_norm_cat_mask = (same_normal_mask + same_category_mask) / 2

    raw_datas['same_normal_mask'] = tf.expand_dims(same_normal_mask,1)
    raw_datas['same_category_mask'] = tf.expand_dims(same_category_mask,1)

    if IsGenply_SameMask:
      MeshSampling.gen_ply_raw(raw_datas, same_normal_mask,
                                 same_category_mask, same_norm_cat_mask)
    #***************************************************************************
    # split mesh
    block_num = len(splited_vidx)
    if block_num==1:
      splited_datas = [raw_datas]
    else:
      splited_datas = MeshSampling.split_vertex(raw_datas, splited_vidx)

    if IsGenply_Splited:
      for bi in range(block_num):
        MeshSampling.gen_mesh_ply_basic(splited_datas[bi], 'Splited' ,'Block_{}'.format(bi))
    #***************************************************************************
    # sampling
    for bi in range(block_num):
      splited_datas[bi] = MeshSampling.sampling_mesh(
                                      _num_vertex_sp, splited_datas[bi])
    splited_sampled_datas = splited_datas

    if IsGenply_SplitedSampled:
      for bi in range(block_num):
        MeshSampling.gen_mesh_ply_basic(splited_sampled_datas[bi], 'SplitedSampled',
                                        'Block{}_sampled_{}'.format(bi, _num_vertex_sp))

    if is_show_shapes:
      MeshSampling.show_datas_shape(splited_sampled_datas, 'sampled datas')

    return splited_sampled_datas

  @staticmethod
  def split_vertex(raw_datas, splited_vidx):
    num_vertex0 = raw_datas['xyz'].shape[0]
    splited_fidx = []
    vertex_idx_per_face_new_ls = []
    for bi,block_vidx in enumerate(splited_vidx):
      face_sp_indices, vertex_idx_per_face_new = MeshSampling.down_sampling_face(\
                  block_vidx, num_vertex0, raw_datas['vertex_idx_per_face'])
      splited_fidx.append(face_sp_indices)
      vertex_idx_per_face_new_ls.append(vertex_idx_per_face_new)

    splited_datas = []
    for bi,block_vidx in enumerate(splited_vidx):
      block_datas = {}
      for item in MeshSampling._vertex_eles:
        block_datas[item] = tf.gather( raw_datas[item], block_vidx )
      for item in MeshSampling._face_eles:
        if item == 'vertex_idx_per_face':
          continue
        block_datas[item] = tf.gather( raw_datas[item], splited_fidx[bi] )
      block_datas['vertex_idx_per_face'] = vertex_idx_per_face_new_ls[bi]
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
  def get_fidx_nbrv_per_vertex(vertex_idx_per_face, num_vertex0):
    '''
    Inputs: [F,3] []
    Output: [N, ?]
    '''
    num_face = tf.shape(vertex_idx_per_face)[0]
    face_indices = tf.reshape(tf.range(0, num_face), [-1,1,1])
    face_indices = tf.tile(face_indices, [1, 3,1])
    vertex_idx_per_face = tf.expand_dims(vertex_idx_per_face, 2)
    vidx_fidx = tf.concat([vertex_idx_per_face, face_indices],  2)
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

    face_idx_per_vertex = tf.scatter_nd(vidx_fidxperv, vidx_fidx_flat_sorted[:,1]+1, \
                                    [num_vertex0, max_nf_perv]) - 1
    fidx_pv_empty_mask = tf.cast(tf.equal(face_idx_per_vertex, -1), tf.bool)

    # set -1 as 0
    face_idx_per_vertex += tf.cast(fidx_pv_empty_mask, tf.int32)

    #***************************************************************************
    # get neighbor verties
    edges_per_vertexs_flat = tf.gather(tf.squeeze(vertex_idx_per_face,-1), vidx_fidx_flat_sorted[:,1])

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

    edges_per_vertex = tf.reshape(edges_per_vertex, [num_vertex0, max_nf_perv*2])
    edges_pv_empty_mask = tf.cast(tf.equal(edges_per_vertex, -1), tf.bool)
    # set -1 as 0
    edges_per_vertex += tf.cast(edges_pv_empty_mask, tf.int32)

    return face_idx_per_vertex, fidx_pv_empty_mask, edges_per_vertex, edges_pv_empty_mask


  @staticmethod
  def get_simplicity_label( face_idx_per_vertex, fidx_pv_empty_mask,
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
      vertex_label_categories = tf.gather(face_label_category_, face_idx_per_vertex)

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
    num_total = tf.cast( same_mask.shape[0].value, tf.float64)
    same_nums = MeshSampling.same_mask_nums(same_mask)
    same_rates = [1.0*n/num_total for n in same_nums]
    same_rates[0] = tf.Print(same_rates[0], same_rates, message=pre+' same rates: ')
    #print('\n{} same rate:{}\n'.format(pre, same_rates))
    return same_rates

  @staticmethod
  def get_face_same_mask(vertex_same_mask, vertex_idx_per_face):
    same_mask = tf.gather(vertex_same_mask, vertex_idx_per_face)
    face_same_mask = tf.reduce_max(same_mask, 1)
    return face_same_mask

  @staticmethod
  def sampling_mesh( _num_vertex_sp, raw_datas):
    if isinstance(raw_datas['xyz'], tf.Tensor):
      num_vertex0 = raw_datas['xyz'].shape[0].value
    else:
      num_vertex0 = raw_datas['xyz'].shape[0]
    is_down_sampling = tf.less(_num_vertex_sp, num_vertex0)
    sampled_datas = tf.cond(is_down_sampling,
      lambda: MeshSampling.down_sampling_mesh(_num_vertex_sp, raw_datas),
      lambda: MeshSampling.up_sampling_mesh(_num_vertex_sp, raw_datas) )
    return sampled_datas

  @staticmethod
  def up_sampling_mesh( _num_vertex_sp, raw_datas):
    MeshSampling.show_datas_shape(raw_datas)
    if isinstance(raw_datas['xyz'], tf.Tensor):
      num_vertex0 = tf.shape(raw_datas['xyz'])[0]
    else:
      num_vertex0 = raw_datas['xyz'].shape[0]
    duplicate_num = _num_vertex_sp - num_vertex0

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
    if isinstance(raw_datas['xyz'], tf.Tensor):
      num_vertex0 = raw_datas['xyz'].shape[0].value
    else:
      num_vertex0 = raw_datas['xyz'].shape[0]
    vertex_sp_indices = MeshSampling.down_sampling_vertex(
                                raw_datas['same_normal_mask'], _num_vertex_sp)

    face_sp_indices, vertex_idx_per_face_new = MeshSampling.down_sampling_face(
                                vertex_sp_indices,
                                num_vertex0, raw_datas['vertex_idx_per_face'])
    for item in raw_datas:
      if item == 'vertex_idx_per_face':
        continue
      is_vertex_0 = tf.equal(tf.shape(raw_datas[item])[0], num_vertex0)
      is_vertex = item in MeshSampling._vertex_eles
      check0 = tf.assert_equal(is_vertex, is_vertex_0)
      with tf.control_dependencies([check0]):
        is_vertex = tf.identity(is_vertex)

      sp_indices = tf.cond(is_vertex,
                           lambda: vertex_sp_indices,
                           lambda: face_sp_indices )
      raw_datas[item] = tf.gather(raw_datas[item], sp_indices)
    raw_datas['vertex_idx_per_face'] = vertex_idx_per_face_new
    return raw_datas

  @staticmethod
  def up_sampling_vertex(same_normal_mask, _num_vertex_sp):
    num_vertex0 = same_normal_mask.shape[0].value
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
    num_vertex0 = same_normal_mask.shape[0].value
    sampling_rate = 1.0 * tf.cast(_num_vertex_sp, tf.float32) / num_vertex0
    print('org num:{}, sampled num:{}, sampling_rate:{}'.format(
                                  num_vertex0, _num_vertex_sp, sampling_rate))
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
  def down_sampling_face(vertex_sp_indices, num_vertex0, vertex_idx_per_face):
    if isinstance(vertex_sp_indices, tf.Tensor):
      _num_vertex_sp = vertex_sp_indices.shape[0].value
    else:
      _num_vertex_sp = vertex_sp_indices.shape[0]
    vertex_sp_indices = tf.expand_dims(tf.cast(vertex_sp_indices, tf.int32),1)
    # scatter new vertex index
    raw_vidx_2_sp_vidx = tf.scatter_nd(vertex_sp_indices,
                                tf.range(_num_vertex_sp)+1, [num_vertex0])-1
    vertex_idx_per_face_new = tf.gather(raw_vidx_2_sp_vidx, vertex_idx_per_face)

    # rm lost faces
    remain_mask = tf.reduce_all(tf.greater(vertex_idx_per_face_new, -1),1)
    face_sp_indices = tf.squeeze(tf.where(remain_mask), 1)
    face_sp_indices = tf.cast(face_sp_indices, tf.int32)

    vertex_idx_per_face_new = tf.gather(vertex_idx_per_face_new, face_sp_indices)

    if MeshSampling._check_optial:
      max_vidx = tf.reduce_max(vertex_idx_per_face_new)
      check0 = tf.assert_less_equal(max_vidx, _num_vertex_sp-1)
      with tf.control_dependencies([check0]):
        vertex_idx_per_face_new = tf.identity(vertex_idx_per_face_new)

    return face_sp_indices, vertex_idx_per_face_new


  @staticmethod
  def gen_mesh_ply_basic(datas, dir_name='', base_name='category_labeled'):
    path =  '/tmp/{}'.format(dir_name)
    ply_fn = '{}/{}.ply'.format(path, base_name)
    for item in datas:
      if isinstance(datas[item], tf.Tensor):
        datas[item] = datas[item].numpy()
    ply_util.gen_mesh_ply(ply_fn, datas['xyz'], datas['vertex_idx_per_face'],
                          datas['label_category'])

  @staticmethod
  def gen_ply_raw(raw_datas, same_normal_mask, same_category_mask, same_norm_cat_mask):
      # face same mask for generating ply
      face_same_normal_mask = MeshSampling.get_face_same_mask(same_normal_mask,
                                                raw_datas['vertex_idx_per_face'])
      face_same_category_mask = MeshSampling.get_face_same_mask(same_category_mask,
                                                raw_datas['vertex_idx_per_face'])
      face_same_norm_cat_mask = MeshSampling.get_face_same_mask(same_norm_cat_mask,
                                                raw_datas['vertex_idx_per_face'])

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


      ply_util.gen_mesh_ply('/tmp/face_same_normal_{}degree.ply'.format(int(MeshSampling._max_norm_dif_angle)),
                            raw_datas['xyz'],
                            raw_datas['vertex_idx_per_face'],
                            face_label = face_same_normal_mask)

      ply_util.gen_mesh_ply('/tmp/face_same_category.ply', raw_datas['xyz'],
                            raw_datas['vertex_idx_per_face'],
                            face_label = face_same_category_mask)
      ply_util.gen_mesh_ply('/tmp/face_same_norm_{}degree_cat.ply'.format(int(MeshSampling._max_norm_dif_angle)),
                            raw_datas['xyz'],
                            raw_datas['vertex_idx_per_face'],
                            face_label = face_same_norm_cat_mask)


  def show_simplity_label(raw_datas, same_normal_mask, same_category_mask):
    '''
    numpy inputs
    '''




