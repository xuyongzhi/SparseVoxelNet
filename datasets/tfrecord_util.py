import h5py, os, glob,sys
import numpy as np
import tensorflow as tf
#from datasets.block_data_prep_util import Raw_H5f
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
import utils.ply_util as ply_util

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


def get_tensor_shape(tensor):
  if isinstance(tensor, tf.Tensor):
    shape = tensor.shape.as_list()
    for i,s in enumerate(shape):
      if s==None:
        shape[i] = tf.shape(tensor)[i]
    return shape
  else:
    return tensor.shape


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
        'vertex_uint8': tf.FixedLenFeature([], tf.string),
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

    vertex_uint8 = tf.decode_raw(tfrecord_features['vertex_uint8'], tf.uint8)
    vertex_uint8 = tf.reshape(vertex_uint8, dset_shape_idx['shape']['vertex_uint8'])

    face_i = tf.decode_raw(tfrecord_features['face_i'], tf.int32)
    face_i = tf.reshape(face_i, dset_shape_idx['shape']['face_i'])
    valid_num_face = tfrecord_features['valid_num_face']

    #*************
    features = {"vertex_i": vertex_i, "vertex_f": vertex_f, \
                "vertex_uint8": vertex_uint8, "face_i": face_i, "valid_num_face": valid_num_face}
    #labels = tf.squeeze(get_ele(features, 'label_category', dset_shape_idx),1)
    labels = face_i

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
  raise ValueError, ele+' not found'


def read_tfrecord(dataset_name, tf_path, loss_lw_gama=-1):
  #tf.enable_eager_execution()
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

  gen_ply = True
  ply_dir = os.path.join(tf_path, 'plys')

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
          # generate label hist summary
          valid_num_face = features['valid_num_face']
          category_label = get_ele(features, 'label_category', dset_shape_idx)
          label_hist += np.histogram(category_label, range(num_classes+1))[0]

          batch_num += 1
          point_num += np.sum(valid_num_face)
          print('Total: %d  %d'%(batch_num, point_num))

          if gen_ply:
            dset_idx = dset_shape_idx['indices']
            xyz = get_ele(features, 'xyz', dset_shape_idx)
            color = get_ele(features, 'color', dset_shape_idx)
            valid_num_face = features['valid_num_face']
            vidx_per_face = get_ele(features, 'vidx_per_face', dset_shape_idx)

            if 'edgev_per_vertex' in dset_idx['vertex_i'] and False:
              edgev_per_vertex = get_ele(features, 'edgev_per_vertex', dset_shape_idx)
              for i in range(10):
                xyz_i = np.take(xyz[0,:,:], edgev_per_vertex[0:100,i,:], axis=0)
                ply_fn = os.path.join(ply_dir, 'fans/fan%d_raw_color.ply'%(i))
                ply_util.create_ply(xyz_i, ply_fn)


            for bi in range(batch_size):
              ply_fn = os.path.join(ply_dir, 'batch_%d/b%d_raw_color.ply'%(batch_num, bi))
              ply_util.gen_mesh_ply(ply_fn, xyz[bi], vidx_per_face[bi,0:valid_num_face[bi,0],:], vertex_color=color[bi])


            if 'same_category_mask' in dset_idx['vertex_i']:
              same_category_mask = get_ele(features, 'same_category_mask', dset_shape_idx)
              same_category_mask = same_category_mask > 2
              same_normal_mask = get_ele(features, 'same_normal_mask', dset_shape_idx)
              same_normal_mask = same_normal_mask > 2
              vertex_simplicity = np.logical_and(same_normal_mask, same_category_mask).astype(np.int8)
              for bi in range(batch_size):
                ply_fn = os.path.join(ply_dir, 'batch_%d/b%d_simplicity.ply'%(batch_num, bi))
                ply_util.gen_mesh_ply(ply_fn, xyz[bi], vidx_per_face[bi,0:valid_num_face[bi,0],:], vertex_label=vertex_simplicity[bi])

          pass

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

  _edgev_num = 12

  _max_nf_perv = 9

  _vertex_eles = ['color', 'xyz', 'nxnynz', 'fidx_per_vertex', 'edgev_per_vertex', 'valid_ev_num_pv',\
                  'edges_per_vertex', 'edges_pv_empty_mask', 'fidx_pv_empty_mask',\
                  'same_normal_mask', 'same_category_mask']
  _face_eles = ['label_raw_category', 'label_instance', 'label_material', \
                'label_category', 'vidx_per_face', ]

  _fi = 0

  @staticmethod
  def sess_split_sampling_rawmesh(raw_datas, _num_vertex_sp, splited_vidx,
                                  dset_metas, ply_dir):
    raw_vertex_nums = [e.shape[0] if type(e)!=type(None) else raw_datas['xyz'].shape[0]\
                         for e in splited_vidx]
    with tf.Graph().as_default():
      #with tf.device('/device:GPU:0'):
      with tf.device('/CPU:0'):
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

        mesh_summary_ = {}
        splited_sampled_datas_ = MeshSampling.main_split_sampling_rawmesh(\
                  raw_datas_pl.copy(), _num_vertex_sp, splited_vidx_pl_,
                  dset_metas, ply_dir, mesh_summary_)

      config=tf.ConfigProto(allow_soft_placement=True,
                            device_count={"CPU": 8},
                            inter_op_parallelism_threads=6,
                            intra_op_parallelism_threads=6)
      config.gpu_options.allow_growth = True
      with tf.Session(config=config) as sess:
        feed_dict = {}
        for item in raw_datas:
          feed_dict[raw_datas_pl[item]] = raw_datas[item]

        if block_num>1:
          for bi in range(block_num):
            feed_dict[splited_vidx_pl[bi]] = splited_vidx[bi]

        splited_sampled_datas = sess.run(splited_sampled_datas_, feed_dict=feed_dict)

    return splited_sampled_datas, raw_vertex_nums, {}


  @staticmethod
  def eager_split_sampling_rawmesh(raw_datas, _num_vertex_sp, splited_vidx,
                                   dset_metas, ply_dir=None):
    start = MeshSampling._fi == 0
    MeshSampling._fi += 1
    if start:
      tf.enable_eager_execution()

    raw_vertex_nums = [e.shape[0] if type(e)!=type(None) else raw_datas['xyz'].shape[0]\
                         for e in splited_vidx]
    mesh_summary = {}
    splited_sampled_datas = MeshSampling.main_split_sampling_rawmesh(
                            raw_datas, _num_vertex_sp, splited_vidx, dset_metas,
                            ply_dir=ply_dir, mesh_summary=mesh_summary)

    bn = len(splited_sampled_datas)
    for bi in range(bn):
      for item in splited_sampled_datas[bi]:
        if isinstance(splited_sampled_datas[bi][item], tf.Tensor):
          splited_sampled_datas[bi][item] = splited_sampled_datas[bi][item].numpy()
    for key in mesh_summary:
      if isinstance(mesh_summary[key], tf.Tensor):
        mesh_summary[key] = mesh_summary[key].numpy()
    return splited_sampled_datas, raw_vertex_nums, mesh_summary


  @staticmethod
  def main_split_sampling_rawmesh(raw_datas, _num_vertex_sp, splited_vidx,
                                  dset_metas, ply_dir=None, mesh_summary={}):

    is_show_shapes = False
    IsGenply_Raw = False
    IsGenply_Cleaned = False
    IsGenply_SameMask = False
    IsGenply_Splited = False
    IsGenply_SplitedSampled = False

    if IsGenply_Raw:
      MeshSampling.gen_mesh_ply_basic(raw_datas, 'Raw', 'raw', ply_dir)
    t_start = tf.timestamp()
    #***************************************************************************
    # rm some labels
    with tf.variable_scope('rm_some_labels'):
      raw_datas, splited_vidx = MeshSampling.rm_some_labels(raw_datas, dset_metas, splited_vidx)
    if IsGenply_Cleaned:
      MeshSampling.gen_mesh_ply_basic(raw_datas, 'Cleaned', 'Cleaned', ply_dir)
    #***************************************************************************
    #face idx per vetex, edges per vertyex
    num_vertex0 = get_shape0(raw_datas['xyz'])
    fidx_per_vertex, fidx_pv_empty_mask, edgev_per_vertex, valid_ev_num_pv, \
     edges_per_vertex, edges_pv_empty_mask, lonely_vertex_idx = \
                    MeshSampling.get_fidx_nbrv_per_vertex(
                          raw_datas['vidx_per_face'], num_vertex0, mesh_summary)
    raw_datas['fidx_per_vertex'] = fidx_per_vertex
    raw_datas['fidx_pv_empty_mask'] = fidx_pv_empty_mask
    raw_datas['edgev_per_vertex'] = edgev_per_vertex
    raw_datas['valid_ev_num_pv'] = valid_ev_num_pv
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
    # split mesh
    block_num = len(splited_vidx)
    if block_num==1:
      splited_datas = [raw_datas]
    else:
      with tf.variable_scope('split_vertex'):
        splited_datas = MeshSampling.split_vertex(raw_datas, splited_vidx, mesh_summary)

    if IsGenply_Splited:
      for bi in range(block_num):
        MeshSampling.gen_mesh_ply_basic(splited_datas[bi], 'Splited' ,'Block_{}'.format(bi), ply_dir)
    #***************************************************************************
    # sampling
    for bi in range(block_num):
      with tf.variable_scope('sampling_mesh'):
        splited_datas[bi] = MeshSampling.sampling_mesh(
                                _num_vertex_sp, splited_datas[bi], mesh_summary)
    splited_sampled_datas = splited_datas
    mesh_summary['t'] = tf.timestamp() - t_start

    if IsGenply_SplitedSampled:
      for bi in range(block_num):
        MeshSampling.gen_mesh_ply_basic(splited_sampled_datas[bi], 'SplitedSampled',
                        'Block{}_sampled_{}'.format(bi, _num_vertex_sp), ply_dir, gen_edgev=True)

    if is_show_shapes:
      MeshSampling.show_datas_shape(splited_sampled_datas, 'sampled datas')

    return splited_sampled_datas


  @staticmethod
  def split_vertex(raw_datas, splited_vidx, mesh_summary):
    num_vertex0 = get_shape0(raw_datas['xyz'])

    # get splited_fidx
    splited_fidx = []
    vidx_per_face_new_ls = []
    edgev_per_vertex_new_ls = []
    valid_ev_num_pv_new_ls = []
    for bi,block_vidx in enumerate(splited_vidx):
      with tf.variable_scope('spv_dsf_b%d'%(bi)):
        face_sp_indices, vidx_per_face_new, edgev_per_vertex_new, valid_ev_num_pv_new \
                  = MeshSampling.update_face_edgev(\
                  block_vidx, num_vertex0, raw_datas['vidx_per_face'],
                  raw_datas['edgev_per_vertex'], raw_datas['valid_ev_num_pv'], raw_datas['xyz'], mesh_summary)
      splited_fidx.append(face_sp_indices)
      vidx_per_face_new_ls.append(vidx_per_face_new)
      edgev_per_vertex_new_ls.append(edgev_per_vertex_new)
      valid_ev_num_pv_new_ls.append(valid_ev_num_pv_new)

    # do split
    splited_datas = []
    for bi,block_vidx in enumerate(splited_vidx):
      block_datas = MeshSampling.gather_datas(raw_datas, block_vidx,
                        splited_fidx[bi], vidx_per_face_new_ls[bi],
                        edgev_per_vertex_new_ls[bi], valid_ev_num_pv_new_ls[bi])
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
  def get_fidx_nbrv_per_vertex(vidx_per_face, num_vertex0, mesh_summary):
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

    #show_ave_num_face_perv
    num_face_perv = tf.reduce_sum(tf.cast(fidx_pv_empty_mask, tf.int32), -1)
    ave_num_face_perv = tf.reduce_mean(num_face_perv)
    mesh_summary['ave_num_face_perv'] = ave_num_face_perv
    fidx_per_vertex = tf.Print(fidx_per_vertex, [ave_num_face_perv],
                                    "\nave_num_face_perv: ")

    # fix size
    fidx_per_vertex = fidx_per_vertex[:, 0:MeshSampling._max_nf_perv]
    nv0 = num_vertex0 if not isinstance(num_vertex0, tf.Tensor) else None
    fidx_per_vertex.set_shape([nv0, MeshSampling._max_nf_perv])
    fidx_pv_empty_mask = fidx_pv_empty_mask[:, 0:MeshSampling._max_nf_perv]
    fidx_pv_empty_mask.set_shape([nv0, MeshSampling._max_nf_perv])

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


    #***************************************************************************
    # sort edge vertices by path
    edgev_per_vertex, valid_ev_num_pv, close_flag = MeshSampling.sort_edge_vertices(edges_per_vertex)
    valid_ev_num_ave = tf.reduce_mean(valid_ev_num_pv)
    valid_ev_num_max = tf.reduce_max(valid_ev_num_pv)
    close_num = tf.reduce_sum(tf.cast(tf.equal(close_flag, 1), tf.int32))
    mesh_summary['valid_edgev_num_ave'] = valid_ev_num_ave
    mesh_summary['valid_edgev_num_max'] = valid_ev_num_max
    edgev_per_vertex = tf.Print(edgev_per_vertex, [valid_ev_num_ave, valid_ev_num_max], message="ave max valid_ev_num")

    # randomly select a start vertex, then fix the shape
    edgev_per_vertex = edgev_per_vertex[:, 0:MeshSampling._edgev_num]

    #***************************************************************************
    # fixed shape of unsorted edges_per_vertex
    edges_per_vertex = edges_per_vertex[:, 0:MeshSampling._max_nf_perv,:]
    nv0 = num_vertex0 if not isinstance(num_vertex0, tf.Tensor) else None
    edges_per_vertex.set_shape([nv0, MeshSampling._max_nf_perv, 2])
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

    return fidx_per_vertex, fidx_pv_empty_mask, edgev_per_vertex, valid_ev_num_pv,\
             edges_per_vertex, edges_pv_empty_mask, lonely_vertex_idx


  @staticmethod
  def find_next_vertex(edgev_per_vertex, remain_edges_pv, valid_ev_num_pv, e, round_id):
      reshape = get_tensor_shape(remain_edges_pv)
      vertex_num = reshape[0]
      remain_vnum = reshape[1]
      remain_edge_num = remain_vnum/2

      last_v = edgev_per_vertex[:,-1:]
      same_mask = tf.equal(last_v, remain_edges_pv)
      remain_valid_mask = tf.greater_equal(remain_edges_pv, 0)
      same_mask = tf.logical_and(same_mask, remain_valid_mask)

      # There should be at most one vertex matches
      same_nums = tf.reduce_sum(tf.cast(same_mask, tf.int32), -1)
      max_same_num = tf.reduce_max(same_nums)
      if round_id ==1:
        def check_one_component():
          # some vertex are lost: more than one component
          not_finished_mask = tf.reduce_any(tf.greater(remain_edges_pv,-1),1)
          not_finished_idx = tf.squeeze(tf.cast(tf.where(not_finished_mask), tf.int32),1)
          more_component = tf.shape(not_finished_idx)[0]
          check_failed = tf.assert_less(more_component, 5)
          #with tf.control_dependencies([check_failed]):
          #  same_mask = tf.identity(same_mask)
          #same_mask = tf.Print(same_mask, [more_component], message="more than one_component")
          return more_component

        tf.cond(tf.not_equal(max_same_num,1), check_one_component, lambda : 0)

      # get the next vertex idx along the path
      same_edge_idx_pv = tf.cast(tf.where(same_mask), tf.int32)
      tmp = 1 - 2 * tf.mod(same_edge_idx_pv[:,1:2], 2)
      tmp = tf.concat([tf.zeros(tf.shape(tmp), tf.int32), tmp], -1)
      next_vid_in_e = same_edge_idx_pv + tmp

      # open vertex: cannot find the next one
      open_vertex_mask = tf.equal(same_nums, 0)
      last_valid_mask = tf.equal(e+1, valid_ev_num_pv)
      open_vertex_mask = tf.logical_and(open_vertex_mask, last_valid_mask)
      open_vidx = tf.squeeze(tf.cast(tf.where(open_vertex_mask), tf.int32),1)
      open_edge_num = tf.reduce_sum(tf.cast(open_vertex_mask, tf.int32))
      #print('{} edge, {} round open_edge_num:{}'.format(e, round_id, open_edge_num))

      # gen the mask to disable next edge
      edge_idx = same_edge_idx_pv[:,1:2]/2
      edge_idx = tf.concat([same_edge_idx_pv[:,0:1], edge_idx], 1)
      next_edge_mask = tf.scatter_nd(edge_idx, tf.ones(tf.shape(edge_idx)[0], tf.int32), [vertex_num, remain_edge_num])
      next_vertex_mask = tf.reshape(tf.tile(tf.expand_dims(next_edge_mask, -1), [1,1,2]), [vertex_num, reshape[1]])
      return next_vid_in_e, open_vidx, next_vertex_mask, open_vertex_mask

  @staticmethod
  def sort_edge_vertices(edges_per_vertex):
    eshape = get_tensor_shape(edges_per_vertex)
    assert len(eshape) == 3
    vertex_num = eshape[0]
    edge_num = eshape[1]
    edges_per_vertex = tf.reshape(edges_per_vertex, [vertex_num, -1])

    def sort_one_edge(e, edgev_per_vertex, remain_edges_pv, loop_vid_in_e_start, valid_ev_num_pv, close_flag):
      next_vid_in_e, open_vidx, next_vertex_mask, open_vertex_mask = MeshSampling.find_next_vertex(
                      edgev_per_vertex, remain_edges_pv, valid_ev_num_pv, e, 1)

      second_round = True
      if second_round:
        edgev_pv_open = tf.gather(edgev_per_vertex, open_vidx)
        #inverse edge vertice to find next vertex
        edgev_pv_open = tf.reverse(edgev_pv_open, [1])
        remain_edges_pv_open = tf.gather(remain_edges_pv, open_vidx)
        valid_ev_num_pv_open = tf.gather(valid_ev_num_pv, open_vidx)
        next_vid_in_e_2, open_vidx_2, next_vertex_mask_2, open_vertex_mask_2 = MeshSampling.find_next_vertex(
                      edgev_pv_open, remain_edges_pv_open, valid_ev_num_pv_open, e, 2)

        # update close flag for second round open vertex
        open_v_2 = tf.gather(edgev_pv_open, open_vidx_2)
        is_close = tf.cast(tf.equal(open_v_2[:,0], open_v_2[:,-1]), tf.int32)
        is_close = is_close * 2 -1
        open_vidx_2 = tf.expand_dims(tf.gather(open_vidx, open_vidx_2),1)
        new_close_flag = tf.scatter_nd(open_vidx_2, is_close, [vertex_num])
        close_flag += new_close_flag
        loop_vid_in_e_start += tf.maximum(new_close_flag,0)

        # if it is still open, should reach the end, just leave it and set -1 in
        # next_vidx
        tmp = tf.gather(open_vidx, next_vid_in_e_2[:,0:1])
        next_vid_in_e_2 = tf.concat([tmp, next_vid_in_e_2[:,1:2]], 1)
        next_vid_in_e = tf.concat([next_vid_in_e, next_vid_in_e_2], 0)

      # gather the raw vertex in the scene
      next_vertex_idx = tf.gather_nd(remain_edges_pv, next_vid_in_e)
      next_vidx = tf.scatter_nd(next_vid_in_e[:,0:1], next_vertex_idx+1, [vertex_num])-1
      next_vidx = tf.expand_dims(next_vidx, 1)

      # update valid_ev_num_pv
      add_valid = tf.scatter_nd(next_vid_in_e[:,0:1], tf.ones(tf.shape(next_vid_in_e)[0:1], tf.int32), [vertex_num])
      valid_ev_num_pv += add_valid


      # update edgev_per_vertex
      if second_round:
        edgev_per_vertex_reversed = tf.scatter_nd(tf.expand_dims(open_vidx,-1), edgev_pv_open,
                                                  [vertex_num, get_tensor_shape(edgev_pv_open)[1]])
        edgev_per_vertex *= 1-tf.cast(tf.expand_dims(open_vertex_mask,1), tf.int32)
        edgev_per_vertex += edgev_per_vertex_reversed

      # if it reaches the end, loop the cycle
      is_loop_invalid = True
      if is_loop_invalid:
        need_loop_vidx = tf.where(tf.less(valid_ev_num_pv, e+2))
        next_vid_in_e_loop = tf.gather(loop_vid_in_e_start, need_loop_vidx[:,0])
        tmp = tf.SparseTensor(need_loop_vidx, tf.ones(tf.shape(need_loop_vidx)[0:1], tf.int32), tf.cast(tf.shape(loop_vid_in_e_start),tf.int64))
        loop_vid_in_e_start = tf.sparse_add(loop_vid_in_e_start, tmp)
        next_vid_in_e_loop = tf.expand_dims(next_vid_in_e_loop, 1)
        next_vid_in_e_loop = tf.concat([tf.cast(need_loop_vidx,tf.int32), next_vid_in_e_loop], 1)
        next_vertex_idx_loop = tf.gather_nd(edgev_per_vertex, next_vid_in_e_loop)
        next_vidx_loop = tf.scatter_nd(next_vid_in_e_loop[:,0:1], next_vertex_idx_loop+1, [vertex_num])
        next_vidx += tf.expand_dims(next_vidx_loop,1)

      edgev_per_vertex = tf.concat([edgev_per_vertex, next_vidx], 1)

      # update remain_edges_pv
      if second_round:
        next_vertex_mask_2 = tf.scatter_nd(tf.expand_dims(open_vidx,-1), next_vertex_mask_2, tf.shape(next_vertex_mask))
        next_vertex_mask += next_vertex_mask_2
      remain_edges_pv = (remain_edges_pv+2) * tf.cast(1-next_vertex_mask, tf.int32) - 2

      e += 1
      return e, edgev_per_vertex, remain_edges_pv, loop_vid_in_e_start, valid_ev_num_pv, close_flag


    e = tf.constant(1)
    edgev_per_vertex = edges_per_vertex[:,0:2]
    remain_edges_pv = edges_per_vertex[:,2:]
    loop_vid_in_e_start = tf.ones([vertex_num], tf.int32)*0 # assume the path close, so loop start from the second one
    valid_ev_num_pv = tf.ones([vertex_num], tf.int32)*2
    close_flag = tf.zeros([vertex_num], tf.int32)
    cond = lambda e, edgev_per_vertex, remain_edges_pv, loop_vid_in_e_start, valid_ev_num_pv, close_flag: tf.less(e, edge_num)

    e, edgev_per_vertex, remain_edges_pv, loop_vid_in_e_start, valid_ev_num_pv, close_flag = tf.while_loop(cond, sort_one_edge, \
                    [e, edgev_per_vertex, remain_edges_pv, loop_vid_in_e_start, valid_ev_num_pv, close_flag])
    valid_ev_num_pv = tf.expand_dims(valid_ev_num_pv, 1)

    return edgev_per_vertex, valid_ev_num_pv, close_flag

  @staticmethod
  def sort_by_spectral(edges):
    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    pass


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
  def rm_some_labels(raw_datas, dset_metas, splited_vidx):
    unwanted_classes = ['void']
    unwanted_labels = tf.constant([[dset_metas.class2label[c] for c in unwanted_classes]], tf.int32)

    label_category = raw_datas['label_category']
    keep_face_mask = tf.not_equal(label_category, unwanted_labels)
    keep_face_mask = tf.reduce_all(keep_face_mask, 1)
    keep_face_idx = tf.squeeze(tf.cast(tf.where(keep_face_mask), tf.int32),1)

    vidx_per_face = raw_datas['vidx_per_face']
    keep_vidx = tf.gather(vidx_per_face, keep_face_idx)
    keep_vidx = tf.reshape(keep_vidx, [-1])
    keep_vertex_idx ,_ = tf.unique(keep_vidx)

    num_vertex0 = get_shape0(raw_datas['xyz'])
    keep_face_idx, vidx_per_face_new, _ = MeshSampling.down_sampling_face(
                 keep_vertex_idx, num_vertex0, raw_datas['vidx_per_face'], True)

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
  def sampling_mesh( _num_vertex_sp, raw_datas, mesh_summary):
    num_vertex0 = get_shape0(raw_datas['xyz'])
    if not isinstance(num_vertex0, tf.Tensor):
      sampling_rate = 1.0 * _num_vertex_sp / tf.cast(num_vertex0, tf.float32)
      print('\nsampling org_num={}, fixed_num={}, valid_sp_rate={} (after rm void)\n'.format(num_vertex0, _num_vertex_sp, sampling_rate))

    is_down_sampling = tf.less(_num_vertex_sp, num_vertex0)
    sampled_datas = tf.cond(is_down_sampling,
                  lambda: MeshSampling.down_sampling_mesh(_num_vertex_sp, raw_datas.copy(), mesh_summary),
                  lambda: MeshSampling.up_sampling_mesh(_num_vertex_sp, raw_datas.copy()))
    return sampled_datas

  @staticmethod
  def up_sampling_mesh( _num_vertex_sp, raw_datas):
    #MeshSampling.show_datas_shape(raw_datas)
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
  def down_sampling_mesh(_num_vertex_sp, raw_datas, mesh_summary):
    vertex_downsample_method = 'random'
    num_vertex0 = get_shape0(raw_datas['xyz'])
    if vertex_downsample_method == 'prefer_simple':
      vertex_sp_indices = MeshSampling.down_sampling_vertex_presimple(
                                raw_datas['same_normal_mask'], _num_vertex_sp)
    elif vertex_downsample_method == 'random':
      vertex_sp_indices = MeshSampling.down_sampling_vertex_random(
                                num_vertex0, _num_vertex_sp)

    face_sp_indices, vidx_per_face_new, edgev_per_vertex_new, valid_ev_num_pv_new  =\
        MeshSampling.update_face_edgev(
        vertex_sp_indices, num_vertex0, raw_datas['vidx_per_face'],
        raw_datas['edgev_per_vertex'], raw_datas['valid_ev_num_pv'], xyz=raw_datas['xyz'],
        mesh_summary=mesh_summary)
    raw_datas = MeshSampling.gather_datas(raw_datas, vertex_sp_indices,
                                    face_sp_indices, vidx_per_face_new,
                                    edgev_per_vertex_new, valid_ev_num_pv_new)
    return raw_datas

  @staticmethod
  def gather_datas(datas, vertex_sp_indices, face_sp_indices, vidx_per_face_new,
                   edgev_per_vertex_new=None, valid_ev_num_pv_new=None):
    num_vertex0 = get_shape0(datas['xyz'])
    new_datas = {}

    for item in datas:
      if item in ['vidx_per_face', 'edgev_per_vertex', 'valid_ev_num_pv']:
        continue
      is_vertex_0 = tf.equal(tf.shape(datas[item])[0], num_vertex0)
      is_vertex = item in MeshSampling._vertex_eles
      check0 = tf.assert_equal(is_vertex, is_vertex_0)
      with tf.control_dependencies([check0]):
        is_vertex = tf.identity(is_vertex)

      sp_indices = tf.cond(is_vertex,
                           lambda: vertex_sp_indices,
                           lambda: face_sp_indices )
      new_datas[item] = tf.gather(datas[item], sp_indices)

    new_datas['vidx_per_face'] = vidx_per_face_new
    if 'edgev_per_vertex' in datas:
      new_datas['edgev_per_vertex'] = edgev_per_vertex_new
      new_datas['valid_ev_num_pv'] = valid_ev_num_pv_new
    return new_datas

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
  def down_sampling_vertex_random(num_vertex0, _num_vertex_sp):
    vertex_sp_indices = tf.random_shuffle(tf.range(num_vertex0))[0:_num_vertex_sp]
    vertex_sp_indices = tf.contrib.framework.sort(vertex_sp_indices)
    return vertex_sp_indices


  @staticmethod
  def down_sampling_vertex_presimple(same_normal_mask, _num_vertex_sp):
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
  def move_neg_left(edgev_per_vertex0):
    invalid_mask = tf.less(edgev_per_vertex0, 0)
    invalid_idx = tf.cast(tf.where(invalid_mask), tf.int32)
    left_idx0 = invalid_idx - tf.constant([[0,1]])
    vn, en = get_tensor_shape(edgev_per_vertex0)
    min_limit = tf.cast( tf.less(left_idx0[:,1:2], 0), tf.int32 ) * en
    tmp = tf.zeros([get_tensor_shape(left_idx0)[0],1], tf.int32)
    min_limit = tf.concat([tmp, min_limit], 1)
    left_idx0 += min_limit

    left_idx = tf.gather_nd(edgev_per_vertex0, left_idx0) + 1
    move_left = tf.scatter_nd(invalid_idx, left_idx, tf.shape(edgev_per_vertex0))
    edgev_per_vertex1 = edgev_per_vertex0 + move_left
    invalid_num = tf.reduce_sum(tf.cast(tf.less(edgev_per_vertex1,0), tf.int32))
    return edgev_per_vertex1, invalid_num

  @staticmethod
  def move_neg_right(edgev_per_vertex0):
    invalid_mask = tf.less(edgev_per_vertex0, 0)
    invalid_idx = tf.cast(tf.where(invalid_mask), tf.int32)
    vn, en = get_tensor_shape(edgev_per_vertex0)
    right_idx0 = invalid_idx + tf.constant([[0,1]])
    max_limit = tf.cast( tf.less(right_idx0[:,1:2], en), tf.int32 )
    tmp = tf.ones([get_tensor_shape(right_idx0)[0],1], tf.int32)
    max_limit = tf.concat([tmp, max_limit], 1)
    right_idx0 *= max_limit

    right_idx = tf.gather_nd(edgev_per_vertex0, right_idx0) + 1
    move_right = tf.scatter_nd(invalid_idx, right_idx, tf.shape(edgev_per_vertex0))
    edgev_per_vertex1 = edgev_per_vertex0 + move_right
    invalid_num = tf.reduce_sum(tf.cast(tf.less(edgev_per_vertex1,0), tf.int32))
    return edgev_per_vertex1, invalid_num

  @staticmethod
  def replace_neg(edgev_per_vertex0, invalid_num, round_id):
    edgev_per_vertex1, invalid_num = MeshSampling.move_neg_left (edgev_per_vertex0)
    def no_op():
      return edgev_per_vertex1, invalid_num
    def move_right():
      return MeshSampling.move_neg_right(edgev_per_vertex1)
    edgev_per_vertex1, invalid_num = tf.cond(tf.greater(invalid_num,0), move_right, no_op)
    return edgev_per_vertex1, invalid_num, round_id+1

  @staticmethod
  def replace_neg_by_lr(edgev_per_vertex_new1, max_round=2):
      invalid_num = tf.reduce_sum(tf.cast(tf.less(edgev_per_vertex_new1,0), tf.int32))
      round_id = tf.constant(0)
      cond = lambda edgev_per_vertex_new1, invalid_num, round_id: tf.logical_and(tf.greater(invalid_num, 0), tf.less(round_id, max_round))
      edgev_per_vertex_new2, invalid_num2, round_id2 = tf.while_loop(cond,
                MeshSampling.replace_neg,
                [edgev_per_vertex_new1, invalid_num, round_id])
      return edgev_per_vertex_new2


  @staticmethod
  def update_new_edges(edgev_per_vertex, vertex_sp_indices, xyz):
    twounit_edgev = MeshSampling.twounit_edgev(edgev_per_vertex, xyz)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  @staticmethod
  def get_twounit_edgev(edgev_per_vertex0, xyz0, raw_vidx_2_sp_vidx, vertex_sp_indices):
    #  the edgev of edgev: geodesic distance = 2 unit
    edgev_edgev0 = tf.gather(edgev_per_vertex0,  edgev_per_vertex0)
    edgev_edgev = tf.gather(edgev_edgev0, vertex_sp_indices)

    edgev_per_vertex = tf.gather(edgev_per_vertex0, vertex_sp_indices)
    edgev_xyz = tf.gather(xyz0, edgev_per_vertex)
    edgev_edgev_xyz = tf.gather(xyz0, edgev_edgev)
    xyz = tf.expand_dims(tf.gather(xyz0, vertex_sp_indices), 1)
    # get the vector from base vertex to edgev
    V0 = edgev_xyz - xyz
    V0 = tf.expand_dims(V0, 2)
    xyz = tf.expand_dims(xyz, 1)
    # get the vector from base vertex to edgev_edgev
    V1 = edgev_edgev_xyz - xyz

    # (1) At tht other side of edgev: project_l > 1
    V0_normal = tf.norm(V0, axis=-1, keepdims=True)
    V0 = V0 / V0_normal
    project_l = tf.reduce_sum(V0 * V1,-1) / tf.squeeze(V0_normal,3)
    other_side_mask = tf.cast(tf.greater(project_l, 1), tf.float32)
    # (2) valid mask
    edgev_edgev_sampled = tf.gather(raw_vidx_2_sp_vidx, edgev_edgev)
    valid_mask = tf.cast(tf.greater(edgev_edgev_sampled, 0), tf.float32)

    # (3) Distance to the projection line: want the colest one
    vn, evn = get_tensor_shape(edgev_per_vertex)
    tmp = tf.cross(tf.tile(V0, [1,1,evn,1]), V1)
    dis_to_proj_line = tf.abs(tf.reduce_sum(tmp, -1))

    # make all the vertices not at the other side very big
    dis_to_proj_line += (1 - other_side_mask)*10
    dis_to_proj_line += (1 - valid_mask)*100

    min_idx = tf.argmin(dis_to_proj_line, axis=-1,  output_type=tf.int32)
    min_idx = tf.expand_dims(min_idx, -1)
    # gather twounit_edgev satisfy three requiements
    v_idx = tf.tile(tf.reshape(tf.range(vn), [-1,1,1]), [1,evn,1])
    ev_idx = tf.tile(tf.reshape(tf.range(evn),[1,-1,1]), [vn,1,1])
    closest_idx = tf.concat([v_idx, ev_idx, min_idx], -1)
    twounit_edgev = tf.gather_nd(edgev_edgev, closest_idx)

    # (4) transfer to sampled idx
    twounit_edgev_new = tf.gather(raw_vidx_2_sp_vidx, twounit_edgev)

    # (5) Failed to find 2 unit edgev: mainly because all the edgev of a vertex are
    # removed. The rate should be small. Just remove the edgev.
    fail_2unit_mask = tf.less(twounit_edgev_new, 0)
    any_2unit_failed = tf.reduce_any(fail_2unit_mask)

    def rm_invalid_2uedgev():
      fail_2uedge_num = tf.reduce_sum(tf.cast(fail_2unit_mask, tf.float32))
      fail_2uedge_num = tf.Print(fail_2uedge_num, [fail_2uedge_num], message="fail_2uedge_num")
      fail_2unit_rate = fail_2uedge_num / tf.cast(vn, tf.float32)
      # (a) All vertices for a path are deleted
      # (b) Spliting lead to some lost near the boundary
      check_fail = tf.assert_less(fail_2unit_rate, 5e-3)
      with tf.control_dependencies([check_fail]):
        return MeshSampling.replace_neg_by_lr(twounit_edgev_new), fail_2uedge_num
    def no_op():
      return twounit_edgev_new, tf.constant(0, tf.float32)
    twounit_edgev_new, fail_2uedge_num = tf.cond(any_2unit_failed, rm_invalid_2uedgev, no_op)

    # (6)final check
    min_idx = tf.reduce_min(twounit_edgev_new)
    check = tf.assert_greater(min_idx, 0)
    with tf.control_dependencies([check]):
      twounit_edgev_new = tf.identity(twounit_edgev_new)
    #if min_idx.numpy() < 0:
    #  # check min dis
    #  min_dis = tf.reduce_min(dis_to_proj_line, -1)
    #  max_mindis = tf.reduce_max(min_dis)
    #  invalid_mask = tf.greater(min_dis, 100)
    #  invalid_num = tf.reduce_sum(tf.cast(invalid_mask, tf.int32))
    #  invalid_idx = tf.where(invalid_mask)
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #  pass
    return twounit_edgev_new, fail_2uedge_num

  @staticmethod
  def replace_neg_by_2unit_edgev(edgev_per_vertex_new1, twounit_edgev):
    neg_mask = tf.cast(tf.less(edgev_per_vertex_new1, 0), tf.int32)
    edgev_per_vertex_new = edgev_per_vertex_new1 * (1-neg_mask) + twounit_edgev * (neg_mask)
    return edgev_per_vertex_new

  @staticmethod
  def update_valid_ev_num_pv(edgev_per_vertex_new1, valid_ev_num_pv, vertex_sp_indices):
      rmed_edgev_mask0 = tf.less(edgev_per_vertex_new1, 0)
      eshape = get_tensor_shape(edgev_per_vertex_new1)
      tmp = tf.tile(tf.reshape(tf.range(eshape[1]), [1,-1]), [eshape[0],1])
      valid_ev_num_pv_new = tf.gather(valid_ev_num_pv, tf.squeeze(vertex_sp_indices,1))
      valid_mask = tf.less(tmp, valid_ev_num_pv_new)
      rmed_edgev_mask = tf.logical_and(rmed_edgev_mask0, valid_mask)
      rmed_edgev_num = tf.reduce_sum(tf.cast(rmed_edgev_mask, tf.int32), 1)
      valid_ev_num_pv_new = valid_ev_num_pv_new - tf.expand_dims(rmed_edgev_num,1)
      return valid_ev_num_pv_new

  @staticmethod
  def rm_lost_face(vidx_per_face, raw_vidx_2_sp_vidx, rm_cond):
    '''
    rm_cond = 'any': remove the face is any vertex of the 3 is deleted
            Used when rm_some_labels
    rm_cond = 'any': remove the face only when all the 3 vertices are deleted
           Used when coarserning vertex
    '''
    assert rm_cond=='any' or rm_cond=='all'
    vidx_per_face_new = tf.gather(raw_vidx_2_sp_vidx, vidx_per_face)
    if rm_cond=='all':
      vidx_per_face_new = MeshSampling.replace_neg_by_lr(vidx_per_face_new, max_round=1)
    remain_mask = tf.reduce_all(tf.greater(vidx_per_face_new, -1),1)
    face_sp_indices = tf.squeeze(tf.where(remain_mask), 1)
    face_sp_indices = tf.cast(face_sp_indices, tf.int32)

    vidx_per_face_new = tf.gather(vidx_per_face_new, face_sp_indices)
    return face_sp_indices, vidx_per_face_new

  @staticmethod
  def edgev_to_face(edgev, valid_ev_num_pv):
    eshape = get_tensor_shape(edgev)
    v0 = tf.expand_dims(edgev[:,0:-1], 2)
    v1 = tf.expand_dims(edgev[:,1:], 2)
    v2 = tf.tile(tf.reshape(tf.range(eshape[0]), [-1,1,1]), [1,eshape[1]-1,1])
    face = tf.concat([v2, v0, v1], -1)

    tmp = tf.tile( tf.reshape(tf.range(eshape[1]), [1, -1]), [eshape[0],1])
    valid_mask = tf.less(tmp, valid_ev_num_pv-1)
    valid_idx = tf.where(valid_mask)
    vidx_per_face_new = tf.gather_nd(face, valid_idx)
    return vidx_per_face_new

  @staticmethod
  def update_face_edgev(vertex_sp_indices, num_vertex0, vidx_per_face, edgev_per_vertex,
                        valid_ev_num_pv, xyz, mesh_summary):
    face_sp_indices, vidx_per_face_new, raw_vidx_2_sp_vidx = MeshSampling.down_sampling_face(\
                                  vertex_sp_indices, num_vertex0, vidx_per_face, False)
    edgev_per_vertex_new3, valid_ev_num_pv_new = MeshSampling.rich_edges(vertex_sp_indices,\
                            edgev_per_vertex, valid_ev_num_pv, xyz, raw_vidx_2_sp_vidx, mesh_summary)
    return face_sp_indices, vidx_per_face_new, edgev_per_vertex_new3, valid_ev_num_pv_new

  @staticmethod
  def rich_edges(vertex_sp_indices, edgev_per_vertex, valid_ev_num_pv, xyz, raw_vidx_2_sp_vidx, mesh_summary):
    #rich_edge_method = 'remove'
    rich_edge_method = 'twounit_edgev'

    edgev_per_vertex_new0 = tf.gather(edgev_per_vertex, vertex_sp_indices)
    edgev_per_vertex_new1 = tf.gather(raw_vidx_2_sp_vidx, edgev_per_vertex_new0)

    if rich_edge_method == 'twounit_edgev':
      twounit_edgev, mesh_summary['fail_2uedge_num'] = MeshSampling.get_twounit_edgev(
                    edgev_per_vertex, xyz, raw_vidx_2_sp_vidx, vertex_sp_indices)
      edgev_per_vertex_new2 = MeshSampling.replace_neg_by_2unit_edgev(edgev_per_vertex_new1, twounit_edgev)
      valid_ev_num_pv_new = tf.gather( valid_ev_num_pv, vertex_sp_indices )

    elif rich_edge_method == 'remove':
      valid_ev_num_pv_new = MeshSampling.update_valid_ev_num_pv(edgev_per_vertex_new1, valid_ev_num_pv, vertex_sp_indices)
      edgev_per_vertex_new2 = MeshSampling.replace_neg_by_lr(edgev_per_vertex_new1)

    # there may still be some negative, but very few. Just handle as lonely
    # points. Assign self vertex idx to the lonely edges
    lonely_vidx = tf.cast(tf.where(tf.less(edgev_per_vertex_new2, 0)), tf.int32)
    tmp = tf.scatter_nd(lonely_vidx, lonely_vidx[:,0]+1, tf.shape(edgev_per_vertex_new2))
    edgev_per_vertex_new3 = edgev_per_vertex_new2 + tmp
    return edgev_per_vertex_new3, valid_ev_num_pv_new

  @staticmethod
  def down_sampling_face(vertex_sp_indices, num_vertex0, vidx_per_face, is_rm_some_label):
    _num_vertex_sp = get_tensor_shape(vertex_sp_indices)[0]
    vertex_sp_indices = tf.expand_dims(tf.cast(vertex_sp_indices, tf.int32),1)
    # scatter new vertex index
    raw_vidx_2_sp_vidx = tf.scatter_nd(vertex_sp_indices, tf.range(_num_vertex_sp)+1, [num_vertex0])-1
    rm_cond = 'any' if is_rm_some_label else 'all'
    face_sp_indices, vidx_per_face_new = MeshSampling.rm_lost_face(vidx_per_face, raw_vidx_2_sp_vidx, rm_cond=rm_cond)
    return face_sp_indices, vidx_per_face_new, raw_vidx_2_sp_vidx

  @staticmethod
  def gen_mesh_ply_basic(datas, dir_name='', base_name='', ply_dir=None, gen_edgev=False):
    if ply_dir == None:
      ply_dir = '/tmp'
    path =  '{}/{}'.format(ply_dir, dir_name)
    for item in datas:
      if isinstance(datas[item], tf.Tensor):
        datas[item] = datas[item].numpy()


    if gen_edgev:
      ply_fn = '{}/edgev_{}.ply'.format(path, base_name)
      down_sample_rate = 1e-1
      edgev_per_vertex = datas['edgev_per_vertex']
      edgev_vidx_per_face = MeshSampling.edgev_to_face(edgev_per_vertex, datas['valid_ev_num_pv'])

      ply_util.gen_mesh_ply(ply_fn, datas['xyz'], edgev_vidx_per_face,
                          vertex_color=datas['color'])

    ply_fn = '{}/{}.ply'.format(path, base_name)
    ply_util.gen_mesh_ply(ply_fn, datas['xyz'], datas['vidx_per_face'],
                          face_label=datas['label_category'])


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
  read_tfrecord(dataset_name, tfrecord_path)


