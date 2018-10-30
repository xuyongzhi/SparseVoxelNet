# Sep 18

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

def get_tensor_shape(t):
  return TfUtil.get_tensor_shape(t)


class MeshSampling():
  _full_edge_dis = 3
  _max_norm_dif_angle = 15.0
  _check_optial = True

  _edgev_num = 14

  _max_nf_perv = 9

  _vertex_eles = ['color', 'xyz', 'nxnynz', 'fidx_per_vertex', 'edgev_per_vertex', 'valid_ev_num_pv',\
                  'edges_per_vertex', 'edges_pv_empty_mask', 'fidx_pv_empty_mask',\
                  'same_normal_mask', 'same_category_mask',\
                  'label_category', 'label_instance', 'label_material']
  _face_eles = ['label_raw_category', 'label_instance', 'label_material', \
                'label_category', 'vidx_per_face', ]

  _fi = 0
  _only_vertex = True

  @staticmethod
  def sess_split_sampling_rawmesh(raw_datas, _num_vertex_sp, splited_vidx,
                                  dset_metas, parse_local_graph_pv, ply_dir):
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
                  dset_metas, parse_local_graph_pv, ply_dir, mesh_summary_)

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
                                   dset_metas, parse_local_graph_pv,
                                   ply_dir=None):
    start = MeshSampling._fi == 0
    MeshSampling._fi += 1
    if start:
      tf.enable_eager_execution()

    raw_vertex_nums = [e.shape[0] if type(e)!=type(None) else raw_datas['xyz'].shape[0]\
                         for e in splited_vidx]
    mesh_summary = {}
    splited_sampled_datas = MeshSampling.main_split_sampling_rawmesh(
                            raw_datas, _num_vertex_sp, splited_vidx, dset_metas,
                            parse_local_graph_pv,
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
                                  dset_metas, parse_local_graph_pv,
                                  ply_dir=None, mesh_summary={}):

    is_show_shapes = False
    IsGenply_Raw = False
    IsGenply_Cleaned = False
    IsGenply_SameMask = False
    IsGenply_Splited = False
    IsGenply_SplitedSampled = False

    if IsGenply_Raw:
      GenPlys.gen_mesh_ply_basic(raw_datas, 'Raw', 'raw', ply_dir)
    t_start = tf.timestamp()
    #***************************************************************************
    # rm some labels
    with tf.variable_scope('rm_some_labels'):
      raw_datas, splited_vidx = MeshSampling.rm_some_labels(raw_datas, dset_metas, splited_vidx)
    if IsGenply_Cleaned:
      GenPlys.gen_mesh_ply_basic(raw_datas, 'Cleaned', 'Cleaned', ply_dir)
    # check not all void vertices
    valid_num_vertex = tf.shape(raw_datas['xyz'])[0]
    check_enough_nonvoid = tf.assert_greater(valid_num_vertex, 1000,
          message="not enough nonvoid vertex: {}. Add to bad file list".format(
                  valid_num_vertex))
    with tf.control_dependencies([check_enough_nonvoid]):
      raw_datas['xyz'] = tf.identity(raw_datas['xyz'])
    #***************************************************************************
    if parse_local_graph_pv:
      #face idx per vetex, edges per vertyex
      num_vertex0 = TfUtil.tshape0(raw_datas['xyz'])
      fidx_per_vertex, fidx_pv_empty_mask, edgev_per_vertex, valid_ev_num_pv, \
      edges_per_vertex, edges_pv_empty_mask, lonely_vertex_idx = \
                      MeshSampling.get_fidx_nbrv_per_vertex(
            raw_datas['vidx_per_face'], num_vertex0, xyz=raw_datas['xyz'],
            norm = raw_datas['nxnynz'], mesh_summary=mesh_summary)
      raw_datas['fidx_per_vertex'] = fidx_per_vertex
      raw_datas['fidx_pv_empty_mask'] = fidx_pv_empty_mask
      raw_datas['edgev_per_vertex'] = edgev_per_vertex
      raw_datas['valid_ev_num_pv'] = valid_ev_num_pv
      raw_datas['edges_per_vertex'] = edges_per_vertex
      raw_datas['edges_pv_empty_mask'] = edges_pv_empty_mask

    if is_show_shapes:
      MeshSampling.show_datas_shape(raw_datas, 'raw datas')

    #***************************************************************************
    parse_same_mask = False
    if parse_same_mask:
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
        GenPlys.gen_mesh_ply_basic(splited_datas[bi], 'Splited' ,'Block_{}'.format(bi), ply_dir)
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
        GenPlys.gen_mesh_ply_basic(splited_sampled_datas[bi], 'SplitedSampled',
                        'Block{}_sampled_{}'.format(bi, _num_vertex_sp), ply_dir, gen_edgev=True)

    if is_show_shapes:
      MeshSampling.show_datas_shape(splited_sampled_datas, 'sampled datas')

    return splited_sampled_datas


  @staticmethod
  def split_vertex(raw_datas, splited_vidx, mesh_summary):
    num_vertex0 = TfUtil.tshape0(raw_datas['xyz'])

    # get splited_fidx
    bn = len(splited_vidx)
    splited_fidx = []
    vidx_per_face_new_ls = []
    edgev_per_vertex_new_ls = []
    valid_ev_num_pv_new_ls = []

    if 'vidx_per_face' not in raw_datas or  raw_datas['vidx_per_face'] is None:
      splited_fidx = vidx_per_face_new_ls = edgev_per_vertex_new_ls =\
        valid_ev_num_pv_new_ls = [None]*bn

    else:
      for bi,block_vidx in enumerate(splited_vidx):
        with tf.variable_scope('spv_dsf_b%d'%(bi)):
          if 'edgev_per_vertex' in raw_datas:
            edgev_per_vertex = raw_datas['edgev_per_vertex']
            valid_ev_num_pv = raw_datas['valid_ev_num_pv']
          else:
            edgev_per_vertex = None
            valid_ev_num_pv = None

          face_sp_indices, vidx_per_face_new, edgev_per_vertex_new, valid_ev_num_pv_new \
                    = MeshSampling.update_face_edgev(\
                    block_vidx, num_vertex0, raw_datas['vidx_per_face'],
                    edgev_per_vertex, valid_ev_num_pv, raw_datas['xyz'], mesh_summary)
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
  def get_fidx_nbrv_per_vertex(vidx_per_face, num_vertex0, xyz=None, norm=None, mesh_summary={}):
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
    edgev_sort_method = 'geodesic_angle'
    if edgev_sort_method == 'geodesic_angle':
      edgev_per_vertex, valid_ev_num_pv = EdgeVPath.sort_edgev_by_angle(edges_per_vertex, xyz, norm, cycle_idx=True, max_evnum_next=MeshSampling._edgev_num, geodis=1)
      #EdgeVPath.main_test_expand_path(edgev_per_vertex, valid_ev_num_pv, xyz, norm)
    elif edgev_sort_method == 'path':
      edgev_per_vertex, valid_ev_num_pv, close_flag = MeshSampling.sort_edge_vertices(edges_per_vertex)

    valid_ev_num_ave = tf.reduce_mean(valid_ev_num_pv)
    valid_ev_num_max = tf.reduce_max(valid_ev_num_pv)
    #close_num = tf.reduce_sum(tf.cast(tf.equal(close_flag, 1), tf.int32))
    mesh_summary['valid_edgev_num_ave'] = valid_ev_num_ave
    mesh_summary['valid_edgev_num_max'] = valid_ev_num_max
    edgev_per_vertex = tf.Print(edgev_per_vertex, [valid_ev_num_ave, valid_ev_num_max], message="ave max valid_ev_num")

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
    '''
    get edgev_per_vertex: edge vertices sorted by path
    '''
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
    num_total = tf.cast( TfUtil.tshape0( same_mask ), tf.float64)
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
    vertex_label =  raw_datas['label_category'].shape[0] == raw_datas['xyz'].shape[0]
    if not vertex_label:
      return MeshSampling.rm_some_face_labels(raw_datas, dset_metas, splited_vidx)
    else:
      return MeshSampling.rm_some_vertex_labels(raw_datas, dset_metas, splited_vidx)

  @staticmethod
  def rm_some_vertex_labels(raw_datas, dset_metas, splited_vidx):
    unwanted_classes = ['void']
    unwanted_labels = tf.constant([[dset_metas.class2label[c] for c in unwanted_classes]], tf.int32)

    label_category = raw_datas['label_category']
    keep_vertex_mask = tf.not_equal(label_category, unwanted_labels)
    keep_vertex_mask = tf.reduce_all(keep_vertex_mask, 1)
    keep_vertex_idx = tf.squeeze(tf.cast(tf.where(keep_vertex_mask), tf.int32),1)

    num_vertex0 = TfUtil.tshape0(raw_datas['xyz'])
    if 'vidx_per_face' in raw_datas:
      keep_face_idx, vidx_per_face_new, _ = MeshSampling.down_sampling_face(
                 keep_vertex_idx, num_vertex0, raw_datas['vidx_per_face'], True)
    else:
      keep_face_idx = vidx_per_face_new = None

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
  def rm_some_face_labels(raw_datas, dset_metas, splited_vidx):
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

    num_vertex0 = TfUtil.tshape0(raw_datas['xyz'])
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
    num_vertex0 = TfUtil.tshape0(raw_datas['xyz'])
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
    num_vertex0 = TfUtil.tshape0(raw_datas['xyz'])
    duplicate_num = _num_vertex_sp - num_vertex0
    with tf.control_dependencies([tf.assert_greater_equal(duplicate_num, 0, message="duplicate_num")]):
      duplicate_num = tf.identity(duplicate_num)

    if 'same_category_mask' in raw_datas:
      raw_datas['same_category_mask']  = tf.cast(raw_datas['same_category_mask'], tf.int32)
      raw_datas['same_normal_mask']  = tf.cast(raw_datas['same_normal_mask'], tf.int32)
    for item in raw_datas:
      is_vertex = item in MeshSampling._vertex_eles
      if is_vertex:
        if TfUtil.tsize(raw_datas[item])>1:
          duplicated = tf.tile(raw_datas[item][-1:,:], [duplicate_num, 1])
        else:
          duplicated = tf.tile(raw_datas[item][-1:], [duplicate_num])
        raw_datas[item] = tf.concat([raw_datas[item], duplicated], 0)
        shape0 = TfUtil.get_tensor_shape(raw_datas[item])
        shape0[0] = _num_vertex_sp
        raw_datas[item].set_shape(shape0)
    if 'same_category_mask' in raw_datas:
      raw_datas['same_category_mask']  = tf.cast(raw_datas['same_category_mask'], tf.int8)
      raw_datas['same_normal_mask']  = tf.cast(raw_datas['same_normal_mask'], tf.int8)
    return raw_datas


  @staticmethod
  def down_sampling_mesh(_num_vertex_sp, raw_datas, mesh_summary):
    #return VertexDecimation.down_sampling_mesh(_num_vertex_sp, raw_datas, mesh_summary)
    return MeshSampling.down_sampling_mesh0(_num_vertex_sp, raw_datas, mesh_summary)

  @staticmethod
  def down_sampling_mesh0(_num_vertex_sp, raw_datas, mesh_summary, down_sp_method='random'):
    num_vertex0 = TfUtil.tshape0(raw_datas['xyz'])
    if down_sp_method == 'prefer_simple':
      vertex_spidx = MeshSampling.down_sampling_vertex_presimple(
                                raw_datas['same_normal_mask'], _num_vertex_sp)
    elif down_sp_method == 'random':
      vertex_spidx = MeshSampling.down_sampling_vertex_random(
                                num_vertex0, _num_vertex_sp)

    if 'vidx_per_face' in raw_datas and raw_datas['vidx_per_face'] is not None:
      if 'edgev_per_vertex' in raw_datas:
        edgev_per_vertex = raw_datas['edgev_per_vertex']
        valid_ev_num_pv = raw_datas['valid_ev_num_pv']
      else:
        edgev_per_vertex = None
        valid_ev_num_pv = None

      face_sp_indices, vidx_per_face_new, edgev_per_vertex_new, valid_ev_num_pv_new =\
          MeshSampling.update_face_edgev(
          vertex_spidx, num_vertex0, raw_datas['vidx_per_face'],
          edgev_per_vertex, valid_ev_num_pv, xyz=raw_datas['xyz'],
          mesh_summary=mesh_summary)
    else:
      face_sp_indices = vidx_per_face_new = edgev_per_vertex_new = valid_ev_num_pv_new = None
    raw_datas = MeshSampling.gather_datas(raw_datas, vertex_spidx,
                                    face_sp_indices, vidx_per_face_new,
                                    edgev_per_vertex_new, valid_ev_num_pv_new)
    return raw_datas

  @staticmethod
  def gather_datas(datas, vertex_spidx, face_sp_indices=None, vidx_per_face_new=None,
                   edgev_per_vertex_new=None, valid_ev_num_pv_new=None):
    num_vertex0 = TfUtil.tshape0(datas['xyz'])
    new_datas = {}

    for item in datas:
      if item in ['vidx_per_face', 'edgev_per_vertex', 'valid_ev_num_pv']:
        # not only order changed, update seperatly
        continue
      is_vertex_0 = tf.equal(tf.shape(datas[item])[0], num_vertex0)
      is_vertex = item in MeshSampling._vertex_eles
      check0 = tf.assert_equal(is_vertex, is_vertex_0)
      with tf.control_dependencies([check0]):
        is_vertex = tf.identity(is_vertex)

      if face_sp_indices is None:
        with tf.control_dependencies([tf.assert_equal(is_vertex, True)]):
          sp_indices = vertex_spidx
      else:
        sp_indices = tf.cond(is_vertex,
                           lambda: vertex_spidx,
                           lambda: face_sp_indices )
      new_datas[item] = tf.gather(datas[item], sp_indices)

    if vidx_per_face_new is not None:
      new_datas['vidx_per_face'] = vidx_per_face_new
    if 'edgev_per_vertex' in datas:
      new_datas['edgev_per_vertex'] = edgev_per_vertex_new
      new_datas['valid_ev_num_pv'] = valid_ev_num_pv_new
    return new_datas

  @staticmethod
  def up_sampling_vertex(same_normal_mask, _num_vertex_sp):
    num_vertex0 = TfUtil.tshape0( same_normal_mask )
    #simple_indices = tf.squeeze(tf.where(tf.greater_equal(
    #                        same_normal_mask, MeshSampling._full_edge_dis)),1)
    duplicate_num = _num_vertex_sp - num_vertex0
    #duplicate_indices = tf.tile( simple_indices[0:1], [duplicate_num] )
    duplicate_indices = tf.ones([duplicate_num], tf.int32) * (num_vertex0 -1)
    vertex_spidx = tf.concat([tf.range(num_vertex0), duplicate_indices], 0)
    return vertex_spidx

  @staticmethod
  def down_sampling_vertex_random(num_vertex0, _num_vertex_sp):
    vertex_spidx = tf.random_shuffle(tf.range(num_vertex0))[0:_num_vertex_sp]
    vertex_spidx = tf.contrib.framework.sort(vertex_spidx)
    return vertex_spidx


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
      vertex_spidx = tf.concat([complex_indices, simple_sp_indices], 0)
      return vertex_spidx

    def rm_all_simple_partof_complex():
      sp_num_from_complex = _num_vertex_sp
      tmp = tf.random_shuffle(tf.range(complex_num))[0:sp_num_from_complex]
      vertex_spidx = tf.gather(complex_indices, tmp)
      return vertex_spidx

    vertex_spidx = tf.cond(simple_is_enough_to_rm,
                                  rm_part_of_simple_only,
                                  rm_all_simple_partof_complex )

    vertex_spidx = tf.contrib.framework.sort(vertex_spidx)

    check_s = tf.assert_equal(tf.shape(vertex_spidx)[0], _num_vertex_sp)
    with tf.control_dependencies([check_s]):
      vertex_spidx = tf.identity(vertex_spidx)
    vertex_spidx.set_shape([_num_vertex_sp])

    if MeshSampling._check_optial:
      # check no duplicate
      tmp0, tmp1, tmp_count = tf.unique_with_counts(vertex_spidx)
      max_count = tf.reduce_max(tmp_count)
      check_no_duplicate = tf.assert_equal(max_count,1)
      with tf.control_dependencies([check_no_duplicate]):
        vertex_spidx = tf.identity(vertex_spidx)

    return vertex_spidx

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
  def replace_neg_by_self(edgev_per_vertex_new2):
    lonely_mask = tf.less(edgev_per_vertex_new2, 0)
    any_lonely = tf.reduce_any(lonely_mask)
    def do_replace_by_self():
      lonely_vidx = tf.cast(tf.where(lonely_mask), tf.int32)
      lonely_num = tf.cast(tf.shape(lonely_vidx)[0], tf.float32)
      vn = tf.cast(get_tensor_shape(edgev_per_vertex_new2)[0], tf.float32)
      lonely_rate = lonely_num / vn
      tmp = tf.scatter_nd(lonely_vidx, lonely_vidx[:,0]+1, tf.shape(edgev_per_vertex_new2))
      return edgev_per_vertex_new2 + tmp
    def no_op():
      return edgev_per_vertex_new2
    edgev_per_vertex_new3 = tf.cond(any_lonely, do_replace_by_self, no_op)
    return edgev_per_vertex_new3


  @staticmethod
  def get_twounit_edgev(edgev_per_vertex0, xyz0, raw_vidx_2_sp_vidx, vertex_spidx,
                        max_fail_2unit_ev_rate, scale=None):
    #  the edgev of edgev: geodesic distance = 2 unit
    edgev_edgev_idx0 = tf.gather(edgev_per_vertex0,  edgev_per_vertex0)
    edgev_edgev = tf.gather(edgev_edgev_idx0, vertex_spidx)

    edgev_per_vertex = tf.gather(edgev_per_vertex0, vertex_spidx)
    edgev_xyz = tf.gather(xyz0, edgev_per_vertex)
    edgev_edgev_xyz = tf.gather(xyz0, edgev_edgev)
    xyz = tf.expand_dims(tf.gather(xyz0, vertex_spidx), 1)
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
    fail_2unit_ev_mask = tf.less(twounit_edgev_new, 0)
    any_2unit_failed = tf.reduce_any(fail_2unit_ev_mask)

    if max_fail_2unit_ev_rate is None:
      max_fail_2unit_ev_rate = 3e-3  # should be 1e-4
    def rm_invalid_2uedgev():
      # all the edgev for a vertex are lost
      fail_2unit_e_mask = tf.reduce_all(fail_2unit_ev_mask, 1)
      fail_2uedge_e_num = tf.reduce_sum(tf.cast(fail_2unit_e_mask, tf.float32))
      fail_2unit_e_rate = fail_2uedge_e_num / tf.cast(vn, tf.float32)
      check_e_fail = tf.assert_less(fail_2unit_e_rate, 1e-4,
            message="fail_2unit_e_rate: all two unit edgev of a vertex are lost, scale {}".format(scale))

      # the detailed edgev lost for each vertex
      fail_2uedge_ev_num = tf.reduce_sum(tf.cast(fail_2unit_ev_mask, tf.float32))
      # normally 5e-5 for downsample in data preprocess
      fail_2unit_ev_rate = fail_2uedge_ev_num / tf.cast(vn, tf.float32) / tf.cast(evn, tf.float32)
      #fail_2unit_ev_rate = tf.Print(fail_2unit_ev_rate, [fail_2unit_ev_rate],
      #                              message="\n\t\tfail_2unit_ev_rate scale {}: {} > ".format(scale, max_fail_2unit_ev_rate))
      # (a) All vertices for a path are deleted
      # (b) Spliting lead to some lost near the boundary
      check_ev_fail = tf.assert_less(fail_2unit_ev_rate, max_fail_2unit_ev_rate, message="fail_2unit_ev_rate scale {}".format(scale))
      with tf.control_dependencies([check_e_fail, check_ev_fail]):
        twounit_edgev_new_2 = MeshSampling.replace_neg_by_lr(twounit_edgev_new)
        return MeshSampling.replace_neg_by_self(twounit_edgev_new_2), fail_2unit_ev_rate
    def no_op():
      return twounit_edgev_new, tf.constant(0, tf.float32)
    twounit_edgev_new, fail_2uedge_rate = tf.cond(any_2unit_failed, rm_invalid_2uedgev, no_op)

    # (6)final check
    min_idx = tf.reduce_min(twounit_edgev_new)
    #if min_idx.numpy() < 0:
    #  # check min dis
    #  invalid_mask = tf.less(twounit_edgev_new,0)
    #  invalid_idx = tf.where(invalid_mask)
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #  pass
    check = tf.assert_greater(min_idx, -1, message="twounit_edgev_new")
    with tf.control_dependencies([check]):
      twounit_edgev_new = tf.identity(twounit_edgev_new)
    return twounit_edgev_new, fail_2uedge_rate

  @staticmethod
  def replace_neg_by_2unit_edgev(edgev_per_vertex_new1, twounit_edgev):
    neg_mask = tf.cast(tf.less(edgev_per_vertex_new1, 0), tf.int32)
    edgev_per_vertex_new = edgev_per_vertex_new1 * (1-neg_mask) + twounit_edgev * (neg_mask)
    return edgev_per_vertex_new

  @staticmethod
  def update_valid_ev_num_pv(edgev_per_vertex_new1, valid_ev_num_pv, vertex_spidx):
      rmed_edgev_mask0 = tf.less(edgev_per_vertex_new1, 0)
      eshape = get_tensor_shape(edgev_per_vertex_new1)
      tmp = tf.tile(tf.reshape(tf.range(eshape[1]), [1,-1]), [eshape[0],1])
      valid_ev_num_pv_new = tf.gather(valid_ev_num_pv, tf.squeeze(vertex_spidx,1))
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
  def update_face_edgev(vertex_spidx, num_vertex0, vidx_per_face, edgev_per_vertex,
                        valid_ev_num_pv, xyz, mesh_summary):
    face_sp_indices, vidx_per_face_new, raw_vidx_2_sp_vidx = MeshSampling.down_sampling_face(\
                                  vertex_spidx, num_vertex0, vidx_per_face, False)
    if edgev_per_vertex is not None:
      edgev_per_vertex_new3, valid_ev_num_pv_new, raw_edgev_spvidx = MeshSampling.rich_edges(vertex_spidx,\
                edgev_per_vertex, xyz, raw_vidx_2_sp_vidx, valid_ev_num_pv, mesh_summary)
    else:
      edgev_per_vertex_new3 = valid_ev_num_pv_new = None
    return face_sp_indices, vidx_per_face_new, edgev_per_vertex_new3, valid_ev_num_pv_new

  @staticmethod
  def get_raw_vidx_2_sp_vidx(vertex_spidx, num_vertex0):
    assert TfUtil.tsize(vertex_spidx) == 1
    _num_vertex_sp = TfUtil.get_tensor_shape(vertex_spidx)[0]
    vertex_spidx = tf.expand_dims(tf.cast(vertex_spidx, tf.int32),1)
    # scatter new vertex index
    raw_vidx_2_sp_vidx = tf.scatter_nd(vertex_spidx, tf.range(_num_vertex_sp)+1, [num_vertex0])-1
    return raw_vidx_2_sp_vidx

  @staticmethod
  def rich_edges(vertex_spidx, edgev_per_vertex, xyz, raw_vidx_2_sp_vidx,
                 valid_ev_num_pv,  mesh_summary={}, max_fail_2unit_ev_rate=None, scale=None):
    assert len(get_tensor_shape(vertex_spidx)) == 1
    assert len(get_tensor_shape(edgev_per_vertex)) == len(get_tensor_shape(xyz)) == 2
    #rich_edge_method = 'remove'
    rich_edge_method = 'twounit_edgev'

    raw_edgev_spvidx = tf.gather(raw_vidx_2_sp_vidx, edgev_per_vertex)
    edgev_per_vertex_new1 = tf.gather(raw_edgev_spvidx, vertex_spidx)

    #raw_edgev_spvidx = tf.gather(edgev_per_vertex, vertex_spidx)
    #edgev_per_vertex_new1 = tf.gather(raw_vidx_2_sp_vidx, raw_edgev_spvidx)

    if rich_edge_method == 'twounit_edgev':
      twounit_edgev, mesh_summary['fail_2uedge_rate'] = MeshSampling.get_twounit_edgev(
                    edgev_per_vertex, xyz, raw_vidx_2_sp_vidx, vertex_spidx, max_fail_2unit_ev_rate, scale)
      edgev_per_vertex_new2 = MeshSampling.replace_neg_by_2unit_edgev(edgev_per_vertex_new1, twounit_edgev)
      if valid_ev_num_pv is None:
        valid_ev_num_pv_new = None
      else:
        valid_ev_num_pv_new = tf.gather( valid_ev_num_pv, vertex_spidx )

    elif rich_edge_method == 'remove':
      if valid_ev_num_pv is None:
        valid_ev_num_pv_new = None
      else:
        valid_ev_num_pv_new = MeshSampling.update_valid_ev_num_pv(edgev_per_vertex_new1, valid_ev_num_pv, vertex_spidx)
      edgev_per_vertex_new2 = MeshSampling.replace_neg_by_lr(edgev_per_vertex_new1)

    # there may still be some negative, but very few. Just handle as lonely
    # points. Assign self vertex idx to the lonely edges
    edgev_per_vertex_new3 = MeshSampling.replace_neg_by_self(edgev_per_vertex_new2)

    return edgev_per_vertex_new3, valid_ev_num_pv_new, raw_edgev_spvidx

  @staticmethod
  def get_raw2sp(edgev_per_vertex, raw_vidx_2_sp_vidx, valid_ev_num_pv, raw_edgev_spvidx,
                 max_bp_fail_rate=5e-4, scale=None):
    ''' [vn]  [vn,12]
    (1) If a vertex is not removed, use itself in raw_vidx_2_sp_vidx
    (2) If a vertex is removed, use any edgev
    (3) If edgev not found, use 2 unit edgev
    '''
    assert  len(get_tensor_shape(raw_vidx_2_sp_vidx)) == 1
    assert len(get_tensor_shape(valid_ev_num_pv)) == len(get_tensor_shape(raw_edgev_spvidx)) == 2

    eshape = get_tensor_shape(raw_edgev_spvidx)
    tmp = tf.tile(tf.reshape(tf.range(eshape[1]), [1,-1]), [eshape[0],1])
    valid_mask = tf.cast(tf.less(tmp, valid_ev_num_pv), tf.int32)
    edgev_bridge1 = (raw_edgev_spvidx+1) * valid_mask -1
    edgev_backp, _ = tf.nn.top_k(edgev_bridge1, 1, sorted=False)
    edgev_backp = tf.squeeze(edgev_backp, 1)

    lost_raw_mask = tf.cast(tf.less(raw_vidx_2_sp_vidx, 0), tf.int32)
    backprop_vidx_0 = raw_vidx_2_sp_vidx + lost_raw_mask * (1+edgev_backp)

    # (3) Some lost vertex cannot find edgev. Use 2 unit edgev
    bp_fail_mask0 = tf.less(backprop_vidx_0, 0)
    bp_fail_num0 = tf.reduce_sum(tf.cast(bp_fail_mask0, tf.float32))
    bp_fail_rate0 = bp_fail_num0 / tf.cast(eshape[0], tf.float32)

    bp_fail_idx = tf.squeeze(tf.where(bp_fail_mask0),1)
    fail_edgev = tf.gather(edgev_per_vertex, bp_fail_idx)
    fail_2u_edgev = tf.gather(edgev_per_vertex, fail_edgev)
    fail_2u_edgev_sp = tf.gather(raw_vidx_2_sp_vidx, fail_2u_edgev)
    fail_2u_edgev_sp = tf.reshape(fail_2u_edgev_sp, [-1, eshape[1]*eshape[1]])
    edgev2u_backp, _ = tf.nn.top_k(fail_2u_edgev_sp, 1)
    edgev2u_backp = tf.squeeze(edgev2u_backp, 1)
    backprop_vidx_2 = tf.scatter_nd(tf.expand_dims(bp_fail_idx,1), edgev2u_backp+1, [eshape[0]])

    backprop_vidx = backprop_vidx_0 + backprop_vidx_2

    bp_fail_mask = tf.less(backprop_vidx, 0)
    bp_fail_num = tf.reduce_sum(tf.cast(bp_fail_mask, tf.float32))
    bp_fail_rate = bp_fail_num / tf.cast(eshape[0], tf.float32)
    #backprop_vidx = tf.Print(backprop_vidx, [bp_fail_rate],
    #                   message="bp_fail_rate scale {}: {} > ".format(scale, max_bp_fail_rate))
    check_bp_fail = tf.assert_less(bp_fail_rate, max_bp_fail_rate,
                  message="too many back prop fail num at scale {}".format(scale))
    with tf.control_dependencies([bp_fail_rate]):
      backprop_vidx = tf.identity(backprop_vidx)
    return backprop_vidx, bp_fail_mask

  @staticmethod
  def down_sampling_face(vertex_spidx, num_vertex0, vidx_per_face, is_rm_some_label):
    assert  TfUtil.tsize(vertex_spidx) == 1
    raw_vidx_2_sp_vidx = MeshSampling.get_raw_vidx_2_sp_vidx(vertex_spidx, num_vertex0)
    rm_cond = 'any' if is_rm_some_label else 'all'
    face_sp_indices, vidx_per_face_new = MeshSampling.rm_lost_face(vidx_per_face, raw_vidx_2_sp_vidx, rm_cond=rm_cond)
    return face_sp_indices, vidx_per_face_new, raw_vidx_2_sp_vidx


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



class EdgeVPath():
  @staticmethod
  def sort_edgev_by_angle(edge_aug_vidx, xyz, norm, xyz_raw=None, cycle_idx=False, max_evnum_next=None, edge_vidx_base=None, geodis=None):
    '''
    edge_aug_vidx: [batch_size, vertex_num, k1, k2]
    xyz: [batch_size, vertex_num, 3]
    cycle_idx=True to enable fill up empty idx by cycle valid idx
    clean_innder: clean the edgev not with largest geodesic distance.
                  Has to be True when expdand edgev.
    max_evnum_next: fix edgev size and speed up
    '''

    with tf.variable_scope('sort_edgev_by_angle'):
      with_batch_dim = True
      if len(get_tensor_shape(edge_aug_vidx)) == 3:
        with_batch_dim = False
        # not include batch size dim
        edge_aug_vidx = tf.expand_dims(edge_aug_vidx, 0)
        xyz = tf.expand_dims(xyz, 0)
        norm = tf.expand_dims(norm, 0)
        if xyz_raw is not None:
          xyz_raw = tf.expand(xyz_raw, 0)
        if edge_vidx_base is not None:
          edge_vidx_base = tf.expand(edge_vidx_base, 0)

      eshape0 = get_tensor_shape(edge_aug_vidx)
      assert len(eshape0) == 4
      batch_size, vn, evn1, evn2 = eshape0

      edge_vidx_next, valid_ev_num_next = EdgeVPath.clean_duplicate_edgev(edge_aug_vidx)
      if edge_vidx_base is not None:
        assert tsize(edge_vidx_base) == 3
        # only used for expand path
        edge_vidx_next, valid_ev_num_next = EdgeVPath.clean_edgebase_in_next(edge_vidx_next, edge_vidx_base, valid_ev_num_next)

      if max_evnum_next is not None:
        # Fix the edge vertex num here has advantage to save some time for later
        # process. But keep in mind, if some long path are cut. It's not a good
        # path any more. Because the edgev are not sorted yet.
        edge_vidx_next = edge_vidx_next[:,:,0:max_evnum_next]

      org_max_valid_evn = get_tensor_shape(edge_vidx_next)[-1]
      cos_angle = EdgeVPath.get_geodesic_angle(edge_vidx_next, xyz, norm, valid_ev_num_next, xyz_raw, geodis=geodis)

      # cycle idx
      if cycle_idx:
        cycle_num = org_max_valid_evn
        # cos_angle: [-3,1]. -5 to make cycled all smaller. Empty is smaller than -20
        cos_angle1 = cos_angle[:,:,0:cycle_num] - 5
        cos_angle = tf.concat([cos_angle, cos_angle1], -1)
        edge_vidx_next = tf.concat([edge_vidx_next, edge_vidx_next[:,:,0:cycle_num]],-1)
      sort_idx = tf.contrib.framework.argsort(cos_angle, axis=-1, direction='DESCENDING')
      if cycle_idx:
        sort_idx = sort_idx[:,:,0:org_max_valid_evn]

      edge_vidx_next_sorted = TfUtil.gather_third_d(edge_vidx_next, sort_idx)

      # if still -1, replace with the first
      edge_vidx_next_sorted = EdgeVPath.replace_neg_by_first_or_last(edge_vidx_next_sorted, 'last_valid', valid_ev_num_next)

      # fix shape
      if max_evnum_next is not None:
        evn_ap = max_evnum_next - get_tensor_shape(edge_vidx_next_sorted)[-1]
        edge_vidx_next_sorted = tf.concat([edge_vidx_next_sorted, tf.tile(edge_vidx_next_sorted[:,:,0:1], [1,1,evn_ap])], -1)

      if not with_batch_dim:
        edge_vidx_next_sorted = tf.squeeze(edge_vidx_next_sorted, 0)
        valid_ev_num_next = tf.squeeze(valid_ev_num_next, 0)
      valid_ev_num_next = tf.expand_dims(valid_ev_num_next, -1)

      # check
      check = tf.assert_greater(tf.reduce_min(edge_vidx_next_sorted),-1)
      with tf.control_dependencies([check]):
        edge_vidx_next_sorted = tf.identity(edge_vidx_next_sorted)


    return edge_vidx_next_sorted, valid_ev_num_next

  @staticmethod
  def set_inner_idx_neg(edge_vidx_base_aug, edge_vidx_base, xyz, xyz_raw=None):
    '''
    Assume all the outside vertices have angle > 90
    '''
    assert tsize(edge_vidx_base) == 3
    assert tsize(edge_vidx_base_aug) == 4
    with tf.variable_scope('set_inner_idx_neg'):
      if xyz_raw is None:
        xyz_raw = xyz
      edge2u_v = TfUtil.gather_second_d(xyz_raw, edge_vidx_base)
      edge2u = edge2u_v - tf.expand_dims(xyz, 2)
      edge2u = tf.expand_dims(edge2u, 3)
      edge2u_aug_v = TfUtil.gather_second_d(xyz_raw, edge_vidx_base_aug)
      edge2u_aug = edge2u_aug_v - tf.expand_dims(edge2u_v, 3)

      # the angle between edge and edge2u
      cosa = tf.reduce_sum(edge2u * edge2u_aug,-1)
      outside_mask = tf.cast(tf.greater(cosa, 0), tf.int32)

      edge_vidx_base_aug_cleaned = (edge_vidx_base_aug+1) * outside_mask - 1

      neg_mask = tf.less(edge_vidx_base_aug_cleaned, 0)
      neg_num = tf.reduce_sum(tf.cast(neg_mask, tf.int32), -1)
      max_neg_num = tf.reduce_max(neg_num)
      min_neg_num = tf.reduce_min(neg_num)
      is_cut_neg = True
      # inner (neg) idx can be cut here to save later time or memory.
      # It can also be left cut later. If the neg rate is nor large, maybe leave
      # it later saves more time

    return edge_vidx_base_aug_cleaned

  @staticmethod
  def cut_neg_by_scatter(edge_vidx):
    neg_idx = tf.where(neg_mask)

  @staticmethod
  def clean_edgebase_in_next(edge_vidx_next, edge_vidx_base, valid_ev_num_next):
    '''
    Sometimes edge_vidx_base can not be filtered by set_inner_idx_neg
    '''
    assert tsize(edge_vidx_next) == tsize(edge_vidx_base) == 3
    mask0 = tf.expand_dims(edge_vidx_next, 3) - tf.expand_dims(edge_vidx_base, 2)
    mask0 = tf.equal(mask0, 0)
    mask1 = tf.cast(tf.reduce_any(mask0, -1), tf.int32)
    invalid_num = tf.reduce_sum(mask1, -1)
    edge_vidx_next_cleaned = edge_vidx_next *(1- mask1) - mask1
    edge_vidx_next_cleaned = tf.contrib.framework.sort(edge_vidx_next_cleaned, -1, direction='DESCENDING')
    valid_ev_num_next = valid_ev_num_next - invalid_num
    return edge_vidx_next_cleaned, valid_ev_num_next

  @staticmethod
  def replace_neg_by_first_or_last(edgev0, place, valid_ev_num=None):
    assert tsize(edgev0)==3
    with tf.variable_scope('replace_neg_by_self'):
      eshape0 = get_tensor_shape(edgev0)
      neg_mask = tf.less(edgev0, 0)
      any_neg = tf.reduce_any(neg_mask)
      def do_replace_by_first():
        neg_vidx = tf.cast(tf.where(neg_mask), tf.int32)
        if place == 'first':
          placement = tf.gather_nd(edgev0, neg_vidx*tf.constant([[1,1,0]]))
        elif place == 'last_valid':
          valid_ev_num_cycled = tf.minimum(valid_ev_num*2, eshape0[-1])
          last_valid_idx = tf.expand_dims(tf.gather_nd( valid_ev_num_cycled-1, neg_vidx[:,0:2]), -1)
          last_valid_idx = tf.concat([neg_vidx[:,0:2], last_valid_idx], -1)
          placement = tf.gather_nd(edgev0, last_valid_idx)
        tmp = tf.scatter_nd(neg_vidx, placement+1, eshape0)
        return edgev0 + tmp
      def no_op():
        return edgev0
      edgev1 = tf.cond(any_neg, do_replace_by_first, no_op)
    return edgev1

  @staticmethod
  def clean_duplicate_edgev(edge_aug_vidx):
    eshape0 = get_tensor_shape(edge_aug_vidx)
    assert len(eshape0) == 4
    batch_size, vn, evn1, evn2 = eshape0

    with tf.variable_scope('clean_duplicate_edgev'):
      # remove duplicated
      edgev_idx0 = tf.reshape(edge_aug_vidx, [batch_size, vn, -1])
      edgev_idx0 = tf.contrib.framework.sort(edgev_idx0, -1, direction='DESCENDING')
      tmp = edgev_idx0[:, :, 0:-1]
      dif_with_pre = tf.not_equal(edgev_idx0[:,:,1:], tmp)
      tmp = tf.constant(True, tf.bool, [batch_size, 1, 1])
      tmp = tf.tile(tmp, [1,vn,1])
      dif_with_pre = tf.concat([tmp, dif_with_pre], -1)
      edgev_idx0 = (1+edgev_idx0) * tf.cast(dif_with_pre, tf.int32) - 1
      edgev_idx0 = tf.contrib.framework.sort(edgev_idx0, -1, direction='DESCENDING')
      valid_ev_num = tf.reduce_sum(tf.cast(tf.greater(edgev_idx0, -1),tf.int32), -1)
      max_evn = tf.reduce_max(valid_ev_num)
      edge_vidx_next = edgev_idx0[:,:,0:max_evn]

      #mean_evn = tf.reduce_mean(valid_ev_num)
      #hist = tf.histogram_fixed_width(valid_ev_num, [1,max_evn+1], nbins=max_evn)
      #hist = tf.cast(hist, tf.float32)
      #hist_rates = hist / tf.reduce_sum(hist)
      #hist_rates = tf.cast(hist_rates * 100, tf.int32)
      #bins = tf.range(1,max_evn+1)
    return edge_vidx_next, valid_ev_num

  @staticmethod
  def get_zero_vec(norm):
    # norm: [batch_size, vn, 3]
    assert tsize(norm) == 4
    with tf.variable_scope('get_zero_vec'):
      x = tf.constant([1,0,0], tf.float32, [1,1,1,3])
      x_tan, costheta_x = EdgeVPath.project_vector(x, norm)
      use_x = tf.cast(tf.less(tf.abs(costheta_x), 0.7), tf.float32)

      y = tf.constant([0,1,0], tf.float32, [1,1,1,3])
      y_tan, costheta_y = EdgeVPath.project_vector(y, norm)

      zero_vec = x_tan * use_x + y_tan *(1-use_x)
      tmp = tf.norm(zero_vec, axis=-1, keepdims=True)
      min_project_norm = tf.reduce_min(tmp)
      zero_vec /= tmp

      nan_mask = tf.is_nan(zero_vec)
      nan_idx = tf.where(nan_mask)
      any_nan = tf.reduce_any(nan_mask)
      check1 = tf.assert_greater(min_project_norm, 0.6)
      check2 = tf.assert_equal(any_nan, False)
      with tf.control_dependencies([check1, check2]):
        zero_vec  = tf.identity(zero_vec)
    return zero_vec


  @staticmethod
  def project_vector(edgev, norm):
    # not normalized
    assert tsize(edgev) == 4
    assert tsize(norm) == 4
    with tf.variable_scope('project_vector'):
      costheta = tf.reduce_sum(edgev * norm, -1, keepdims=True)
      edgev_tangent = edgev - costheta * norm
    return edgev_tangent, costheta

  @staticmethod
  def get_geodesic_angle(edgev_idx, xyz, norm, valid_ev_num=None, xyz_raw=None, geodis=None):
    assert tsize(edgev_idx) == 3
    assert tsize(norm) == 3
    batch_size, vertex_num, evn = get_tensor_shape(edgev_idx)

    with tf.variable_scope('geodesic_angle'):
      empty_mask = tf.cast(tf.less(edgev_idx, 0), tf.float32)
      edgev_idx = EdgeVPath.replace_neg_by_first_or_last(edgev_idx, 'first')
      if xyz_raw is None:
        xyz_raw = xyz
      edgev = TfUtil.gather_second_d(xyz_raw, edgev_idx) - tf.expand_dims(xyz,2)
      # (1) Project edgev to the tangent plane: edgev_tan
      #     Project x to the tangent to get 0 angle vec
      norm = tf.expand_dims(norm,2)
      edgev_tan, _ = EdgeVPath.project_vector(edgev, norm)
      zero_vec = EdgeVPath.get_zero_vec(norm)

      # (2) Get cos of the angel
      ev_tan_norm = tf.norm(edgev_tan, axis=-1, keepdims=True)
      ev_norm = tf.norm(edgev, axis=-1, keepdims=True)

      check_manifold = True
      if check_manifold:
        valid_mask = TfUtil.valid_num_to_mask(valid_ev_num, evn)
        ev_norm_rate = ev_tan_norm / (ev_norm+1e-5)
        # For a manifold, ev_norm_rate shoul close to 1
        min_norm_rate = TfUtil.mask_reduce_min(tf.squeeze(ev_norm_rate,-1), valid_mask)
        # normally 0.98 for geodis=1; 0.97617507 for geodis=2;
        # 0.96850145 for geodis=3
        mean_norm_rate = TfUtil.mask_reduce_mean(tf.squeeze(ev_norm_rate,-1), valid_mask)
        check_manifold = tf.assert_greater(mean_norm_rate, 0.93, message="check manifold failed geodis={}".format(geodis))
        with tf.control_dependencies([check_manifold]):
          ev_norm = tf.identity(ev_norm)

      edgev_tan_normed = edgev_tan / (ev_norm+1e-5)
      cos_angle = tf.reduce_sum(zero_vec * edgev_tan_normed, -1)

      # (3) Judge if the angle is over 180
      # the norm for each edgev
      edgev_norm = tf.cross(tf.tile(zero_vec,[1,1,evn,1]), edgev_tan_normed)
      #nan_idx = tf.where( tf.is_nan(edgev_norm))
      cos_nn = tf.reduce_sum(norm * edgev_norm, -1)
      over_pi = tf.cast(tf.less(cos_nn, 0), tf.float32)
      # (3.1)change sign if over pi. (3.2) minus 2 if over pi.
      sign = 1-over_pi*2
      # [-3,1]
      cos_angle = cos_angle * sign - 2*over_pi

      # (4) set cos angle for empty as -10
      cos_angle = cos_angle * (1-empty_mask) + empty_mask * (-20)

      # check nan
      any_nan = tf.reduce_any(tf.is_nan(cos_angle))
      check = tf.assert_equal(any_nan, False, message='cos angel nan')
      with tf.control_dependencies([check]):
        cos_angle = tf.identity(cos_angle)
    return cos_angle

  @staticmethod
  def expand_path(edge_vidx_base, valid_ev_num_base, edge_vidx_interval,
                  valid_ev_num_interval, xyz, norm, xyz_raw=None, geodis=None, max_evnum_next=None):
    '''
    expand path from edge_vidx_base by edge_vidx_interval
    '''
    #print('\nstart expand_path {}'.format(geodis))

    with_batch_dim = True
    if len(get_tensor_shape(edge_vidx_base)) == 2:
      with_batch_dim = False
      edge_vidx_base = tf.expand_dims(edge_vidx_base, 0)
      valid_ev_num_base = tf.expand_dims(valid_ev_num_base, 0)
      edge_vidx_interval = tf.expand_dims(edge_vidx_interval, 0)
      valid_ev_num_interval = tf.expand_dims(valid_ev_num_interval, 0)
      xyz = tf.expand_dims(xyz, 0)
      norm = tf.expand_dims(norm, 0)
      if xyz_raw:
        xyz_raw = tf.expand_dims(xyz_raw,0)
    assert tsize(edge_vidx_base) == tsize(edge_vidx_interval) == \
      tsize(valid_ev_num_base) == tsize(valid_ev_num_interval) == \
      tsize(xyz) == tsize(norm) ==  3

    #***************************************************************************
    max_evn_base = tf.reduce_max(valid_ev_num_base)
    edge_vidx_base = edge_vidx_base[:,:,0:max_evn_base]

    max_evn_interval = tf.reduce_max(valid_ev_num_interval)
    edge_vidx_interval = edge_vidx_interval[:,:,0:max_evn_interval]

    with tf.variable_scope('expand_path_%s'%(geodis)):
      edge_vidx_base_aug = TfUtil.gather_second_d(edge_vidx_interval, edge_vidx_base)
      edge_vidx_base_aug = EdgeVPath.set_inner_idx_neg(edge_vidx_base_aug, edge_vidx_base, xyz, xyz_raw)
      edge_vidx_next, valid_ev_num_next = EdgeVPath.sort_edgev_by_angle(edge_vidx_base_aug,
              xyz, norm, xyz_raw, cycle_idx=True, max_evnum_next=max_evnum_next, edge_vidx_base=edge_vidx_base, geodis=geodis)

    #***************************************************************************
    is_ply = False
    if is_ply:
      EdgeVPath.edge_vidx_ply(edge_vidx_next[0], xyz_raw[0], valid_ev_num_next[0], geodis)
      if geodis==2:
        EdgeVPath.edge_vidx_ply(edge_vidx_base[0], xyz_raw[0], valid_ev_num_base[0], 1)

    if not with_batch_dim:
      edge_vidx_next = tf.squeeze(edge_vidx_next, 0)
      valid_ev_num_next = tf.squeeze(valid_ev_num_next, 0)
      xyz = tf.squeeze(xyz, 0)

    return edge_vidx_next, valid_ev_num_next

  @staticmethod
  def random_down_sp_idx(org_num, batch_size, sp_rate):
    aim_num = tf.cast(tf.ceil(org_num * sp_rate), tf.int32)
    sp_idx_ls = []
    for i in range(batch_size):
      sp_idx_i = tf.random_shuffle(tf.range(org_num))[0:aim_num]
      sp_idx_ls.append( tf.expand_dims(sp_idx_i, 0) )
    sp_idx = tf.concat(sp_idx_ls, 0)
    return sp_idx

  @staticmethod
  def down_sample_edgevidx(sp_rate, edge_vidx, valid_ev_num, xyz, norm):
    assert tsize(edge_vidx) == 3
    batch_size, org_num, evn = TfUtil.get_tensor_shape(edge_vidx)
    sp_idx = EdgeVPath.random_down_sp_idx(org_num, batch_size, sp_rate)
    #sp_idx = tf.contrib.framework.sort(sp_idx, -1, direction='ASCENDING')

    #sp_idx = tf.reshape(tf.range(180000,180000+10), [1,-1])

    edge_vidx_dsp = TfUtil.gather_second_d(edge_vidx, sp_idx)
    valid_ev_num_dsp = TfUtil.gather_second_d(valid_ev_num, sp_idx)
    norm_dsp = TfUtil.gather_second_d(norm, sp_idx)
    xyz_dsp = TfUtil.gather_second_d(xyz, sp_idx)
    return edge_vidx_dsp, valid_ev_num_dsp, xyz_dsp, norm_dsp, sp_idx

  @staticmethod
  def main_test_expand_path(edge1u_vidx, valid_1uev_num, xyz, norm):
    with_batch_dim = True
    if TfUtil.tsize(edge1u_vidx) == 2:
      with_batch_dim = False
      edge1u_vidx = tf.expand_dims(edge1u_vidx, 0)
      valid_1uev_num = tf.expand_dims(valid_1uev_num, 0)
      xyz = tf.expand_dims(xyz, 0)
      norm = tf.expand_dims(norm, 0)

    xyz_raw = xyz
    edge1u_vidx_raw = edge1u_vidx
    valid_1uev_num_raw = valid_1uev_num
    sp_rate = 0.01

    #-------------------------------
    if sp_rate<1.0:
      edge1u_vidx, valid_1uev_num, xyz, norm, sp_idx_1u = \
        EdgeVPath.down_sample_edgevidx(sp_rate, edge1u_vidx, valid_1uev_num, xyz, norm)

    edge2u_vidx, valid_2uev_num = EdgeVPath.expand_path(edge1u_vidx, valid_1uev_num,
          edge1u_vidx_raw, valid_1uev_num_raw, xyz, norm, xyz_raw, geodis=2, max_evnum_next=28)

    #-------------------------------
    sp_rate = 1.01
    if sp_rate<1.0:
      edge2u_vidx, valid_2uev_num, xyz, norm, sp_idx_2u = \
        EdgeVPath.down_sample_edgevidx(sp_rate, edge2u_vidx, valid_2uev_num, xyz, norm)

    edge3u_vidx, valid_3uev_num = EdgeVPath.expand_path(edge2u_vidx, valid_2uev_num,
                edge1u_vidx_raw, valid_1uev_num_raw, xyz, norm, xyz_raw, geodis=3, max_evnum_next=50)
    #-------------------------------
    if sp_rate<1.0:
      edge3u_vidx, valid_3uev_num, xyz, norm, sp_idx_3u = \
        EdgeVPath.down_sample_edgevidx(sp_rate, edge3u_vidx, valid_3uev_num, xyz, norm)

    edge4u_vidx, valid_4uev_num = EdgeVPath.expand_path(edge3u_vidx, valid_3uev_num,
                edge1u_vidx_raw, valid_1uev_num_raw, xyz, norm, xyz_raw, geodis=4, max_evnum_next=60)
    print('expand path test ok')

  @staticmethod
  def edge_vidx_ply(edgev_idx, xyz, valid_ev_num, geodis):
    assert tsize(edgev_idx) == 2
    print('start gen edgev_idx ply as geodis={}'.format(geodis))
    xyz = xyz.numpy()
    edgev_idx = edgev_idx.numpy()

    is_downsp = False

    if is_downsp:
      vn, evn = TfUtil.get_tensor_shape(edgev_idx)
      np.random.seed(0)
      sample_idx = np.random.choice(vn, 500, False)
      edgev_idx = np.take(edgev_idx, sample_idx, axis=0)

    ply_fn = '/home/z/Desktop/plys/edgev_{}.ply'.format(geodis)
    edges = ply_util.closed_path_to_edges(edgev_idx, valid_ev_num)
    ply_util.gen_mesh_ply(ply_fn, xyz, edges)


class VertexDecimation():
  _sp_w_norm = 0.2
  @staticmethod
  def down_sampling_mesh(_num_vertex_sp, raw_datas, mesh_summary):
    with_batch_dim = True
    if TfUtil.tsize(raw_datas['xyz']) == 2:
      with_batch_dim = False
      for item in raw_datas:
        raw_datas[item] = tf.expand_dims(raw_datas[item], 0)

    down_sp_method = 'prefer_smooth'
    num_vertex0 = TfUtil.get_tensor_shape(raw_datas['xyz'])[1]
    if down_sp_method == 'prefer_smooth':
      smooth_factor = VertexDecimation.get_smooth_perv_raw(raw_datas['xyz'],
          raw_datas['nxnynz'], raw_datas['edgev_per_vertex'], raw_datas['valid_ev_num_pv'])
      vertex_spidx, vertex_rm_idx, sp_smooth_loss = VertexDecimation.smooth_sampling(smooth_factor, _num_vertex_sp)
      #GenPlys.gen_mesh_ply_basic(raw_datas, 'vertex_decimation', 'nw_%d'%(10*VertexDecimation._sp_w_norm),
      #                           vertex_spidx=vertex_spidx, vertex_rm_idx=vertex_rm_idx)

    elif down_sp_method == 'random':
      vertex_spidx = MeshSampling.down_sampling_vertex_random(
                                num_vertex0, _num_vertex_sp)

    raise NotImplementedError
    VertexDecimation.update_face(vertex_spidx, vertex_rm_idx, raw_datas['vidx_per_face'], num_vertex0)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT

    face_sp_indices, vidx_per_face_new, edgev_per_vertex_new, valid_ev_num_pv_new =\
        VertexDecimation.update_face_edgev(
        vertex_spidx, num_vertex0, raw_datas['vidx_per_face'],
        raw_datas['edgev_per_vertex'], raw_datas['valid_ev_num_pv'], xyz=raw_datas['xyz'],
        mesh_summary=mesh_summary)
    raw_datas = MeshSampling.gather_datas(raw_datas, vertex_spidx,
                                    face_sp_indices, vidx_per_face_new,
                                    edgev_per_vertex_new, valid_ev_num_pv_new)


  @staticmethod
  def get_smooth_perv_raw(xyz, norm, edgev_per_vertex, valid_ev_num_pv):
    xyz_smooth = VertexDecimation.get_smooth_perv(xyz, edgev_per_vertex, valid_ev_num_pv)
    norm_smooth = VertexDecimation.get_smooth_perv(norm, edgev_per_vertex, valid_ev_num_pv)
    norm_w = VertexDecimation._sp_w_norm
    smooth_factor = norm_smooth * norm_w + xyz_smooth * (1-norm_w)
    return smooth_factor

  @staticmethod
  def smooth_sampling(smooth_factor, _num_vertex_sp):
    sort_idx = tf.contrib.framework.argsort(smooth_factor, axis=-1, direction='DESCENDING')
    sp_idx = sort_idx[:,0:_num_vertex_sp]
    rm_idx = sort_idx[:,_num_vertex_sp:]

    rm_smoothf = TfUtil.gather_second_d(smooth_factor, rm_idx)
    sp_smooth_loss = tf.reduce_mean(rm_smoothf,-1)
    return sp_idx, rm_idx, sp_smooth_loss

  @staticmethod
  def get_smooth_perv(features, edgev_per_vertex, valid_ev_num_pv):
    '''
    Smaller mean smoother
    '''
    assert TfUtil.tsize(features) == TfUtil.tsize(edgev_per_vertex) == 3
    max_evn = tf.reduce_max(valid_ev_num_pv)
    mean_evn = tf.cast( tf.reduce_mean(valid_ev_num_pv), tf.float32)
    evn = tf.minimum(max_evn, tf.cast(mean_evn*1.6, tf.int32))
    edgev_per_vertex = edgev_per_vertex[:,:,0:evn]

    features_aug = TfUtil.gather_second_d(features, edgev_per_vertex)
    mean_f = tf.reduce_mean(features_aug, -2)
    mean_err = tf.norm( mean_f - features, axis=-1)
    tmp = tf.reduce_mean(mean_err, axis=-1, keepdims=True)
    smooth = mean_err / tmp

    return smooth


  @staticmethod
  def get_raw_vidx_2_sp_vidx(vertex_spidx, num_vertex0):
    assert len(get_tensor_shape(vertex_spidx)) == 2
    batch_size, _num_vertex_sp = TfUtil.get_tensor_shape(vertex_spidx)
    batch_idx = tf.tile(tf.reshape(tf.range(batch_size), [-1,1,1]), [1,_num_vertex_sp,1])
    vertex_spidx = tf.expand_dims(tf.cast(vertex_spidx, tf.int32),-1)
    vertex_spidx = tf.concat([batch_idx, vertex_spidx], -1)

    new_vidx = tf.tile(tf.reshape(tf.range(_num_vertex_sp),[1,-1]), [batch_size,1])
    raw_vidx_2_sp_vidx = tf.scatter_nd(vertex_spidx, new_vidx+1, [batch_size, num_vertex0])-1
    return raw_vidx_2_sp_vidx

  @staticmethod
  def update_edgev(vertex_spidx, vertex_rm_idx, edgev_per_vertex, num_vertex0):
    pass

  @staticmethod
  def update_face(vertex_spidx, vertex_rm_idx, vidx_per_face, num_vertex0):
    raw_vidx_2_sp_vidx = VertexDecimation.get_raw_vidx_2_sp_vidx(vertex_spidx, num_vertex0)

    edgev_rmv = TfUtil.gather_second_d()
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass


  @staticmethod
  def update_face_edgev(vertex_spidx, num_vertex0, vidx_per_face, edgev_per_vertex,
                        valid_ev_num_pv, xyz, mesh_summary):
    edgev_per_vertex_new3, valid_ev_num_pv_new, raw_edgev_spvidx = MeshSampling.rich_edges(vertex_spidx,\
              edgev_per_vertex, xyz, raw_vidx_2_sp_vidx, valid_ev_num_pv, mesh_summary)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    face_sp_indices, vidx_per_face_new, raw_vidx_2_sp_vidx = MeshSampling.down_sampling_face(\
                                  vertex_spidx, num_vertex0, vidx_per_face, False)
    return face_sp_indices, vidx_per_face_new, edgev_per_vertex_new3, valid_ev_num_pv_new

    return raw_datas

  @staticmethod
  def contract_vertex_pairs(vertex_spidx, edgev_per_vertex, xyz, raw_vidx_2_sp_vidx,
                 valid_ev_num_pv,  mesh_summary={}, max_fail_2unit_ev_rate=None, scale=None):
    assert len(get_tensor_shape(vertex_spidx)) == 1
    assert len(get_tensor_shape(edgev_per_vertex)) == len(get_tensor_shape(xyz)) == 2
    pass


class GenPlys():
  @staticmethod
  def gen_mesh_ply_basic(datas, dir_name='', base_name='', ply_dir=None, gen_edgev=False,
                         vertex_spidx=None, vertex_rm_idx=None):
    if ply_dir == None:
      ply_dir = '/tmp/plys'
    path =  '{}/{}'.format(ply_dir, dir_name)
    if base_name=='':
      base_name = '1'
    for item in datas:
      if isinstance(datas[item], tf.Tensor):
        datas[item] = datas[item].numpy()
      if datas[item].ndim == 3:
        datas[item] = datas[item][0]

    if vertex_spidx is not None:
      if isinstance(vertex_spidx, tf.Tensor):
        vertex_spidx = vertex_spidx.numpy()
      if vertex_spidx.ndim == 2:
        vertex_spidx = vertex_spidx[0]

    # **************
    if gen_edgev:
      ply_fn = '{}/edgev_{}.ply'.format(path, base_name)
      down_sample_rate = 1e-1
      if 'edgev_per_vertex' in datas:
        edgev_per_vertex = datas['edgev_per_vertex']
        edgev_vidx_per_face = MeshSampling.edgev_to_face(edgev_per_vertex, datas['valid_ev_num_pv'])

        ply_util.gen_mesh_ply(ply_fn, datas['xyz'], edgev_vidx_per_face,
                            vertex_color=datas['color'])

    # **************
    num_vertex0 = datas['xyz'].shape[0]
    if vertex_spidx is not None:
      sp_xyz = np.take(datas['xyz'], vertex_spidx, axis=0)
      rm_xyz = np.take(datas['xyz'], vertex_rm_idx, axis=0)
      ply_fn = '{}/sp_{}.ply'.format(path, base_name)
      ply_util.create_ply(sp_xyz, ply_fn)
      ply_fn = '{}/rm_{}.ply'.format(path, base_name)
      ply_util.create_ply(rm_xyz, ply_fn)


    # **************
    ply_fn = '{}/{}.ply'.format(path, base_name)
    label_category = datas['label_category']
    if 'vidx_per_face' in datas:
      ply_util.gen_mesh_ply(ply_fn, datas['xyz'], datas['vidx_per_face'],
                          face_label=label_category)
    else:
        ply_util.create_ply(datas['xyz'], ply_fn)

if __name__ == '__main__':
  dataset_name = 'MATTERPORT'
  dset_path = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans'
  tfrecord_path = '/DS/Matterport3D/MATTERPORT_TF/mesh_tfrecord_555'
  tfrecord_path = '/home/z/Research/SparseVoxelNet/data/MATTERPORT_TF/mesh_tfrecord'
  read_tfrecord(dataset_name, tfrecord_path)



