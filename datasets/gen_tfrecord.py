import h5py, os, glob,sys
import numpy as np
import tensorflow as tf
#from datasets.block_data_prep_util import Raw_H5f
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
import math
from tfrecord_util import MeshSampling

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)

Vertex_feles = ['xyz','nxnynz', 'color']
Vertex_ieles = ['face_idx_per_vertex', 'fidx_pv_empty_mask', \
                'edges_per_vertex','edges_pv_empty_mask',\
                'same_normal_mask', 'same_category_mask']
Face_ieles = ['label_category', 'label_instance', 'label_material', \
                    'label_raw_category', 'vertex_idx_per_face', 'label_simplity']

MAX_FLOAT_DRIFT = 1e-6
DEBUG = False


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


class Raw_To_Tfrecord():
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
    self.dataset_meta = DatasetsMeta(self.dataset_name)

  def __call__(self, rawfns):
    bsfn0 = os.path.join(self.tfrecord_path, 'block_split_settings.txt')
    if type(self.block_size)!=type(None):
      with open(bsfn0, 'w') as bsf:
        bsf.write('num_point:{}\nblock_size:{}\nblock_stride:{}\nmin_pn_inblock:{}'.format(\
                      self.num_point, self.block_size,\
                      self.block_stride, self.min_pn_inblock))

    block_split_dir = os.path.join(self.tfrecord_path, 'block_split_summary')
    if not os.path.exists(block_split_dir):
      os.makedirs(block_split_dir)


    self.fn = len(rawfns)
    #rawfns = rawfns[5577:]
    rawfns.sort()
    for fi, rawfn in enumerate(rawfns):
      self.fi = fi
      region_name = os.path.splitext(os.path.basename(rawfn))[0]
      scene_name = os.path.basename(os.path.dirname(os.path.dirname(rawfn)))
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

      if self.dataset_name == "MATTERPORT":
        block_num, valid_block_num, num_points_splited = self.transfer_onefile_matterport(rawfn)

      with open(scene_bs_fn, 'w') as bsf:
        bsf.write('intact\n')
        bnstr = '{} \tblock_num:{} \tvalid block num:{}\n\n'.format(region_name, block_num, valid_block_num)
        bsf.write(bnstr)
        vnstr = '\n'+'\n'.join(['{}_{}: {}'.format(region_name, i, num_points_splited[i]) for i in range(valid_block_num)])
        bsf.write(vnstr)
    print('All {} file are converted to tfreord'.format(fi+1))

  def sort_eles(self, all_eles):
    # elements *************************
    tmp = [e in Vertex_feles + Vertex_ieles + Face_ieles for e in all_eles]
    assert sum(tmp) == len(tmp)
    self.eles_sorted = {}
    self.eles_sorted['vertex_f'] = [e for e in Vertex_feles if e in all_eles]
    self.eles_sorted['vertex_i'] = [e for e in Vertex_ieles if e in all_eles]
    self.eles_sorted['face_i'] = [e for e in Face_ieles if e in all_eles]

  def record_metas(self, raw_datas, dls):
    ele_idxs = {}
    for item in self.eles_sorted:
      ele_idxs[item] = {}
      n = 0
      for e in self.eles_sorted[item]:
        shape_i = raw_datas[e].shape
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

  def split_vertex(self, xyz):
    if type(self.block_size) == type(None):
      return [None], 1

    xyz_min = np.min(xyz, 0)
    xyz_max = np.max(xyz, 0)
    xyz_scope = xyz_max - xyz_min

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
    #print('raw scope:\n{} \nblock_num:{}'.format(xyz_scope, block_num))
    #print('splited bot:\n{} splited top:\n{}'.format(bot, top))

    if block_num == 1:
      return [None],  block_num

    if block_num>1:
      for i in range(block_num):
        for j in range(3):
          if top[i,j] > xyz_scope[j]:
            top[i,j] = xyz_scope[j] - MAX_FLOAT_DRIFT
            bot[i,j] = np.maximum(xyz_scope[j] - block_size[j] + MAX_FLOAT_DRIFT, 0)

    bot += xyz_min
    top += xyz_min

    dls_splited = []
    num_points_splited = []
    splited_vidx = []
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
      splited_vidx.append(indices)
    return splited_vidx, block_num


  @staticmethod
  def downsample_face(vertex_idx_per_face, point_sampling_indices, num_point0):
    num_face0 = vertex_idx_per_face.shape[0]
    point_mask = np.zeros(num_point0, np.int8)
    point_mask[point_sampling_indices] = 1
    point_mask = point_mask.astype(np.bool)
    face_vertex_mask = np.take(point_mask, vertex_idx_per_face)
    face_vertex_mask = np.all(face_vertex_mask, 1)
    face_keep_indices = np.where(face_vertex_mask)[0]
    face_del_indices = np.where(np.logical_not(face_vertex_mask))[0]
    del_face_num = face_del_indices.shape[0] # 16653

    del_vertex_rate = 1 - 1.0* point_sampling_indices.shape[0] / num_point0
    del_face_rate = 1.0 * del_face_num / num_face0
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass


  def get_label_MODELNET40(self):
    # get label for MODELNET40
    if self.dataset_name == 'MODELNET40':
      tmp = base_name.split('_')
      if len(tmp) == 2:
        category = tmp[0]
      elif len(tmp) == 3:
        category = '_'.join(tmp[0:2])
      object_label = dataset_meta.class2label[category]
      dls['labels'] = np.array([object_label])

  def transfer_onefile_matterport(self, rawfn):
    from MATTERPORT_util import parse_ply_file, parse_ply_vertex_semantic

    base_name = os.path.splitext(os.path.basename(rawfn))[0]
    base_name1 = os.path.basename(os.path.dirname(os.path.dirname(rawfn)))
    base_name = base_name1 + '_' + base_name

    # ['label_material', 'label_category', 'vertex_idx_per_face', 'color',
    # 'xyz', 'nxnynz', 'label_raw_category', 'label_instance']
    raw_datas = parse_ply_file(rawfn)

    splited_vidx, block_num = self.split_vertex(raw_datas['xyz'])
    valid_block_num = len(splited_vidx)
    num_points_splited = [e.shape[0] if type(e)!=type(None) else raw_datas['xyz'].shape[0]\
                          for e in splited_vidx]

    splited_sampled_datas, raw_vertex_nums = MeshSampling.eager_split_sampling_rawmesh(
                          raw_datas, self.num_point, splited_vidx)


    print('starting {} th file: {}'.format(self.fi, rawfn))
    for bi in range(block_num):
      tmp = '' if block_num==1 else '_'+str(bi)
      tfrecord_fn = os.path.join(self.data_path, base_name)+tmp + '.tfrecord'
      self.transfer_one_block(tfrecord_fn, splited_sampled_datas[bi], raw_vertex_nums[bi])
    print('finish {} th file: {}'.format(self.fi, rawfn))
    return block_num, valid_block_num, num_points_splited

  def transfer_one_block(self, tfrecord_fn, block_sampled_datas, raw_vertex_num):
    from tfrecord_util import bytes_feature

    if not hasattr(self, 'eles_sorted'):
      self.sort_eles(block_sampled_datas.keys())
    #*************************************************************************
    dls = {}
    for item in self.eles_sorted:
      dls[item] = []
      for e in self.eles_sorted[item]:
        data = block_sampled_datas[e]
        dls[item].append( data )
      if len(dls[item]) >  0:
        dls[item] = np.concatenate(dls[item], -1)

    #*************************************************************************
    # convert to expample
    assert dls['vertex_i'].dtype == np.int32
    assert dls['vertex_f'].dtype == np.float32
    assert dls['face_i'].dtype == np.int32
    if not hasattr(self, 'ele_idxs'):
      self.record_metas(block_sampled_datas, dls)
      #print(self.ele_idxs)

    max_category =  np.max(dls['face_i'][:, self.ele_idxs['face_i']['label_category']])
    assert max_category < self.dataset_meta.num_classes, "max_category {} > {}".format(\
                                          max_category, self.dataset_meta.num_classes)

    vertex_f_bin = dls['vertex_f'].tobytes()
    vertex_i_shape_bin = np.array(dls['vertex_i'].shape, np.int32).tobytes()
    vertex_i_bin = dls['vertex_i'].tobytes()
    face_i_bin = dls['face_i'].tobytes()


    features_map = {
      'vertex_f': bytes_feature(vertex_f_bin),
      'vertex_i': bytes_feature(vertex_i_bin),
      'face_i':   bytes_feature(face_i_bin),
      #'valid_num':int64_feature(raw_vertex_num)
    }

    example = tf.train.Example(features=tf.train.Features(feature=features_map))

    #*************************************************************************
    with tf.python_io.TFRecordWriter( tfrecord_fn ) as raw_tfrecord_writer:
      raw_tfrecord_writer.write(example.SerializeToString())

    if self.fi %50 ==0:
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
  raw_fns = glob.glob(rawh5_glob)

  raw_to_tf = Raw_To_Tfrecord(dataset_name, tfrecord_path, num_point, block_size)
  raw_to_tf(raw_fns)

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

def main_matterport():
  dataset_name = 'MATTERPORT'
  dset_path = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans'
  num_point = {'MODELNET40':None, 'MATTERPORT':150000}
  block_size = {'MODELNET40':None, 'MATTERPORT':np.array([5.0, 5.0, 5.0]) }

  #raw_glob = os.path.join(dset_path, '*/*/region_segmentations/*.ply')
  raw_glob = os.path.join(dset_path, '17DRP5sb8fy/*/region_segmentations/*.ply')
  tfrecord_path = '/DS/Matterport3D/MATTERPORT_TF/mesh_tfrecord'

  main_write(dataset_name, raw_glob, tfrecord_path, num_point[dataset_name],\
             block_size[dataset_name])

  #main_merge_tfrecord(dataset_name, tfrecord_path)

  #main_gen_ply(dataset_name, tfrecord_path)

  #get_dataset_summary(dataset_name, tfrecord_path)
  #get_dset_shape_idxs(tfrecord_path)


if __name__ == '__main__':
  main_matterport()

