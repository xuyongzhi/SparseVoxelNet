# xyz Sep 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
from models.conv_util import ResConvOps, gather_second_d
from utils.tf_util import TfUtil

DEBUG = False

DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

def ele_in_feature(features, ele, dset_shape_idx):
  ds_idxs = dset_shape_idx['indices']
  for g in ds_idxs:
    if ele in ds_idxs[g]:
      ele_idx = ds_idxs[g][ele]
      ele_data = tf.gather(features[g], ele_idx, axis=-1)
      return ele_data
  raise ValueError, ele+' not found'

class Model(ResConvOps):
  CNN = 'FAN'
  def __init__(self, net_data_configs, data_format, dtype):
    self.dset_shape_idx = net_data_configs['dset_shape_idx']
    self.data_configs = net_data_configs['data_configs']
    self.dset_metas = net_data_configs['dset_metas']
    self.net_flag = net_data_configs['net_flag']
    self.block_paras = BlockParas(net_data_configs['block_configs'])
    super(Model, self).__init__(net_data_configs, data_format, dtype)

    if self.CNN == 'TRIANGLE':
      cnn_class = TriangleCnn
      self.block_fn = self.inception_block_v2
    elif self.CNN == 'FAN':
      cnn_class = FanCnn
      self.block_fn = self.fan_block_v2
    self.mesh_cnn = cnn_class(self.blocks_layers, self.block_fn, self.block_paras)

  def __call__(self, features,  is_training):
    '''
    vertices: [B,N,C]
    edges_per_vertex: [B,N,10*2]
    '''
    if self.CNN == 'TRIANGLE':
      return self.main_triangle_cnn(features, is_training)
    elif self.CNN == 'FAN':
      return self.main_fan_cnn(features, is_training)

  def main_test_pool(self, features, is_training=True):
    self.is_training = is_training
    inputs, xyz = self.parse_inputs(features)
    vertices = inputs['vertices']
    edgev_per_vertex = inputs['edgev_per_vertex']
    valid_ev_num_pv = inputs['valid_ev_num_pv']
    valid_ev_num_pv = inputs['valid_ev_num_pv']
    vidx_per_face = inputs['vidx_per_face']
    valid_num_face = tf.cast(inputs['valid_num_face'], tf.int32)

    vertices = tf.expand_dims(vertices, 2)
    for s in range(3):
      vertices_, edgev_per_vertex_, xyz_, valid_ev_num_pv_, backprop_vidx, \
        r2s_fail_mask = FanCnn.pool_mesh(s,
                          vertices, edgev_per_vertex, xyz, valid_ev_num_pv)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      vertices, edgev_per_vertex, xyz, valid_ev_num_pv =\
        tf.expand_dims(vertices_,2), edgev_per_vertex_, xyz_, valid_ev_num_pv_

  def main_fan_cnn(self, features, is_training):
    self.is_training = is_training
    inputs, xyz = self.parse_inputs(features)
    vertices = inputs['vertices']
    edgev_per_vertex = inputs['edgev_per_vertex']
    valid_ev_num_pv = inputs['valid_ev_num_pv']
    valid_ev_num_pv = inputs['valid_ev_num_pv']
    vidx_per_face = inputs['vidx_per_face']
    valid_num_face = tf.cast(inputs['valid_num_face'], tf.int32)

    #***************************************************************************
    # multi scale vertex feature encoder
    scale_n = self.block_paras.scale_num
    vertices_scales = []
    backprop_vidx_scales = []
    r2s_fail_mask_scales = []
    for scale in range(scale_n):
      with tf.variable_scope('FanCnn_S%d'%(scale)):
        vertices = self.mesh_cnn.update_vertex(scale, is_training, vertices,
              edgev_per_vertex, valid_ev_num_pv)
      if scale < self.block_paras.scale_num-1:
        vertices_scales.append(vertices)
        with tf.variable_scope('MeshPool_S%d'%(scale)):
          vertices, edgev_per_vertex, xyz, valid_ev_num_pv, backprop_vidx, \
            r2s_fail_mask = FanCnn.pool_mesh(scale,
                                vertices, edgev_per_vertex, xyz, valid_ev_num_pv)
        backprop_vidx_scales.append(backprop_vidx)
        r2s_fail_mask_scales.append(r2s_fail_mask)

    #***************************************************************************
    with tf.variable_scope('CatGlobal'):
      vertices = self.add_global(vertices)
    #***************************************************************************
    # multi scale feature back propogation
    for i in range(scale_n-1):
      scale = scale_n -2 - i
      with tf.variable_scope('FanCnn_S%d'%(scale)):
        vertices = self.feature_backprop(scale, vertices, vertices_scales[scale],
                      backprop_vidx_scales[scale], r2s_fail_mask_scales[scale])


    flogits, flabel_weight = self.face_classifier(vertices, vidx_per_face, valid_num_face)
    self.log_model_summary()
    return flogits, flabel_weight

  def feature_backprop(self, scale, cur_vertices, lasts_vertices, backprop_vidx, r2s_fail_mask):
    cur_vertices = gather_second_d(cur_vertices, backprop_vidx)
    r2s_fail_mask = tf.cast(tf.expand_dims(tf.expand_dims(r2s_fail_mask, -1),-1), tf.float32)
    cur_vertices = cur_vertices * (1-r2s_fail_mask)
    vertices = tf.concat([lasts_vertices, cur_vertices], -1)

    blocks_params = self.block_paras.get_block_paras('backprop', scale)
    vertices = self.blocks_layers(vertices, blocks_params, self.block_fn,
                    self.is_training, 'BackProp_%d'%(scale), with_initial_layer=False)
    return vertices

  def add_global(self, vertices):
    blocks_params = self.block_paras.get_block_paras('global', 0)
    if not blocks_params:
      return vertices

    #***************************************************************************
    global_f = tf.reduce_max(vertices, 1, keepdims=True)
    # encoder global feature
    global_f = self.blocks_layers(global_f, blocks_params, self.block_fn,
                                  self.is_training, 'Global',
                                  with_initial_layer=False)
    nv = TfUtil.get_tensor_shape(vertices)[1]
    global_f = tf.tile(global_f, [1, nv, 1, 1])
    vertices = tf.concat([vertices, global_f], -1)
    self.log_tensor_p(vertices, '', 'cat global')

    #***************************************************************************
    # encoder fused vertex feature
    blocks_params = self.block_paras.get_block_paras('global', 1)
    if not blocks_params:
      return vertices
    vertices = self.blocks_layers(vertices, blocks_params, self.block_fn,
                                  self.is_training, 'GlobalFusedV',
                                  with_initial_layer=False)
    return vertices

  def face_classifier(self, vertices, vidx_per_face, valid_num_face):
    dense_filters = self.block_paras.dense_filters + [self.dset_metas.num_classes]
    vlogits = self.dense_block(vertices, dense_filters, self.is_training)
    vlogits = tf.squeeze(vlogits, 2)
    flogits = gather_second_d(vlogits, vidx_per_face)
    flogits = tf.reduce_mean(flogits, 2)
    fn = TfUtil.get_tensor_shape(vidx_per_face)[1]
    valid_face_mask = tf.tile(tf.reshape(tf.range(fn), [1,fn]), [self.batch_size,1])
    flabel_weight = tf.cast(tf.less(valid_face_mask, valid_num_face), tf.float32)
    return flogits, flabel_weight

  def main_triangle_cnn(self, features, is_training):
    self.is_training = is_training
    inputs = self.parse_inputs(features)
    vertices = inputs['vertices']
    fidx_per_vertex = inputs['fidx_per_vertex']
    fidx_pv_empty_mask = inputs['fidx_pv_empty_mask']
    vidx_per_face = inputs['vidx_per_face']
    valid_num_face = inputs['valid_num_face']


    vertices_scales = []
    for scale in range(self.block_paras.scale_num):
      with tf.variable_scope('S%d'%(scale)):
        vertices = self.mesh_cnn.update_vertex(scale, is_training, vertices,
              vidx_per_face, valid_num_face, fidx_per_vertex, fidx_pv_empty_mask)

      vertices_scales.append(vertices)

    #vertices = tf.concat(vertices_scales, -1)
    simplicity_logits = self.simplicity_classifier(vertices)
    simplicity_label = self.simplicity_label(features)
    self.log_model_summary()
    return simplicity_logits, simplicity_label


  def simplicity_label(self, features):
    min_same_norm_mask = 2
    min_same_category_mask = 2
    same_category_mask = self.get_ele(features, 'same_category_mask')
    same_category_mask = tf.greater_equal(same_category_mask, min_same_category_mask)
    same_normal_mask = self.get_ele(features, 'same_normal_mask')
    same_normal_mask = tf.greater_equal(same_normal_mask, min_same_norm_mask)

    simplicity_mask = tf.logical_and(same_normal_mask, same_category_mask)
    simplicity_mask = tf.squeeze(simplicity_mask, -1)
    simplicity_label = tf.cast(simplicity_mask, tf.int32)
    return simplicity_label

  def get_ele(self, features, ele):
    return ele_in_feature(features, ele, self.dset_shape_idx)


  def normalize_xyz(self, xyz):
    norm_xyz_method = self.data_configs['normxyz']
    if norm_xyz_method == 'mean0':
      mean_xyz = tf.reduce_mean(xyz, 1, keepdims=True)
      new_xyz = xyz - mean_xyz
    elif norm_xyz_method == 'min0':
      min_xyz = tf.reduce_min(xyz, 1, keepdims=True)
      new_xyz = xyz - min_xyz
    elif norm_xyz_method == 'max1':
      min_xyz = tf.reduce_min(xyz, 1, keepdims=True)
      new_xyz = xyz - min_xyz
      new_xyz = new_xyz / 5.0
    elif norm_xyz_method == 'raw':
      new_xyz = xyz
      pass
    else:
      raise NotImplementedError
    return new_xyz

  def parse_inputs(self, features):
    inputs = {}
    vertices = []
    for e in self.data_configs['feed_data']:
      ele = self.get_ele(features, e)
      if e=='xyz':
        xyz = ele = self.normalize_xyz(ele)
      vertices.append(ele)
    inputs['vertices'] = vertices = tf.concat(vertices, -1)
    vshape = TfUtil.get_tensor_shape(vertices)
    #self.batch_size = vshape[0]
    self.num_vertex0 = vshape[1]
    self.log_tensor_p(vertices, 'vertices', 'raw_input')

    inputs['vidx_per_face'] = self.get_ele(features, 'vidx_per_face')
    inputs['valid_num_face'] = features['valid_num_face']

    if self.CNN == 'TRIANGLE':
      inputs['fidx_per_vertex'] = self.get_ele(features, 'fidx_per_vertex')
      inputs['fidx_pv_empty_mask'] = self.get_ele(features, 'fidx_pv_empty_mask')

    elif self.CNN == 'FAN':
      edgevnum = self.block_paras.edgevnum
      inputs['edgev_per_vertex'] = self.get_ele(features, 'edgev_per_vertex')[:,:,0:edgevnum]
      inputs['valid_ev_num_pv'] = self.get_ele(features, 'valid_ev_num_pv')

    is_check_data = True
    # check data
    if is_check_data:
      check_ev = tf.assert_greater( tf.reduce_min(inputs['edgev_per_vertex']), -1,
                                   message='found neg edgev_per_vertex')
      with tf.control_dependencies([check_ev]):
        inputs['edgev_per_vertex'] = tf.identity(inputs['edgev_per_vertex'])

    return inputs, xyz

  def simplicity_classifier(self, vertices):
    dense_filters = [32, 16, 2]
    simplicity_logits = self.dense_block(vertices, dense_filters, self.is_training)
    return simplicity_logits


class FanCnn():
  def __init__(self, blocks_layers_fn=None, block_fn=None, block_paras=None,
                      ):
    self.block_fn = block_fn
    self.blocks_layers = blocks_layers_fn
    self.block_paras = block_paras

  def update_vertex(self, scale, is_training, vertices,\
                    edgev_per_vertex, valid_ev_num_pv):
    vertices = tf.expand_dims(vertices, 2)

    blocks_params = self.block_paras.get_block_paras('vertex', scale)
    vertices = self.blocks_layers(vertices, blocks_params, self.block_fn,
                               is_training, 'S%d'%( scale),
                               edgev_per_vertex=edgev_per_vertex)
    return vertices

  @staticmethod
  def pool_mesh(scale, vertices, edgev_per_vertex, xyz, valid_ev_num_pv, pool_method='mean', pool_rate=0.5):
    '''
    max mean identity
    '''
    if pool_method == 'identity':
      pass
    else:
      # (1) replace vertice features by max/mean group features
      if pool_method == 'max':
        pool_fn = tf.reduce_max
      elif pool_method == 'mean':
        pool_fn = tf.reduce_mean
      vertices = gather_second_d(tf.squeeze(vertices,2), edgev_per_vertex)
      vertices = tf.reduce_max(vertices, 2)

    # (2) Downsampling vertices
    vn = TfUtil.get_tensor_shape(vertices)[1]
    new_vn = int(pool_rate * vn)
    vertex_sp_indices = []
    batch_size = TfUtil.get_tensor_shape(vertices)[0]
    for bi in range(batch_size):
      vertex_sp_indices.append( tf.expand_dims(tf.random_shuffle(tf.range(vn))[0:new_vn], 0))
    vertex_sp_indices = tf.concat(vertex_sp_indices, 0)
    vertex_sp_indices = tf.contrib.framework.sort(vertex_sp_indices, -1)
    vertices_new = tf.squeeze(gather_second_d(vertices, tf.expand_dims(vertex_sp_indices,-1)), 2)
    xyz_new = tf.squeeze(gather_second_d(xyz, tf.expand_dims(vertex_sp_indices,-1)), 2)

    # (3) New edges
    from datasets.tfrecord_util import MeshSampling
    edgev_per_vertex_new_ls = []
    valid_ev_num_pv_new_ls = []
    backprop_vidx_ls = []
    r2s_fail_mask_ls = []
    for bi in range(batch_size):
      raw_vidx_2_sp_vidx = MeshSampling.get_raw_vidx_2_sp_vidx(vertex_sp_indices[bi], vn)
      edgev_per_vertex_new, valid_ev_num_pv_new, raw_edgev_spvidx = MeshSampling.rich_edges(
                  vertex_sp_indices[bi], edgev_per_vertex[bi],
                  xyz[bi], raw_vidx_2_sp_vidx, valid_ev_num_pv[bi],
                  max_fail_2unit_ev_rate = [2e-2*10, 3e-2*10, 3e-2*10][scale], scale=scale) # 5e-3
      backprop_vidx, r2s_fail_mask = MeshSampling.get_raw2sp(edgev_per_vertex[bi],
                  raw_vidx_2_sp_vidx, valid_ev_num_pv[bi], raw_edgev_spvidx,
                  max_bp_fail_rate = [9e-4, 9e-4, 5e-3][scale], scale = scale)

      edgev_per_vertex_new_ls.append(tf.expand_dims(edgev_per_vertex_new, 0))
      valid_ev_num_pv_new_ls.append(tf.expand_dims(valid_ev_num_pv_new, 0))
      backprop_vidx_ls.append(tf.expand_dims(backprop_vidx,0))
      r2s_fail_mask_ls.append(tf.expand_dims(r2s_fail_mask,0))

    edgev_per_vertex_new = tf.concat(edgev_per_vertex_new_ls, 0)
    valid_ev_num_pv_new = tf.concat(valid_ev_num_pv_new_ls, 0)
    backprop_vidx = tf.concat(backprop_vidx_ls, 0)
    r2s_fail_mask = tf.concat(r2s_fail_mask_ls, 0)
    return vertices_new, edgev_per_vertex_new, xyz_new, valid_ev_num_pv_new, backprop_vidx, r2s_fail_mask

class TriangleCnn():
  def __init__(self, blocks_layers_fn=None, block_fn=None, block_paras=None,
                      ):
    self.block_fn = block_fn
    self.blocks_layers = blocks_layers_fn
    self.block_paras = block_paras

  def update_vertex(self, scale, is_training, vertices,\
                    vidx_per_face, valid_num_face, fidx_per_vertex, fidx_pv_empty_mask):
    '''
    Inputs:
      vertices: (nv, cv)
      vidx_per_face: (nf, 3)

    Middle:
      face_centroid: (nf, 1, cc)
      edges: (nf, 3, ce)
      faces: (nf, cf)

    Out:
      vertices: (nv, cvo)
    '''
    self.scale = scale
    self.is_training = is_training
    face_centroid, edges = self.vertex_2_edge(vertices, vidx_per_face, valid_num_face)
    edges = self.encoder(edges, 'edge')
    face_centroid = self.encoder(face_centroid, 'centroid')
    faces = self.edge_2_face(edges, face_centroid)
    faces = self.encoder(faces, 'face')
    vertices = self.face_2_vertex(faces, fidx_per_vertex, fidx_pv_empty_mask)
    vertices = self.encoder(vertices, 'vertex')
    return vertices

  def vertex_2_edge(self, vertices, vidx_per_face, valid_num_face):
    vertices_per_face = gather_second_d(vertices, vidx_per_face)
    face_centroid = tf.reduce_mean(vertices_per_face, 2, keepdims=True) # (nf, 1, cv)
    edges = vertices_per_face - face_centroid  # (nf, 3, cv)
    return face_centroid, edges

  def encoder(self, inputs, represent):
    if not self.block_fn:
      return inputs
    assert represent in ['edge', 'centroid', 'face', 'vertex']
    blocks_params = self.block_paras.get_block_paras(represent, self.scale)
    outputs = self.blocks_layers(inputs, blocks_params, self.block_fn,
                               self.is_training, '%s_s%d'%(represent, self.scale) )
    return outputs

  def edge_2_face(self, edges, face_centroid):
    face_local = []
    if 'max' in self.block_paras.e2fl_pool:
      face_local.append( tf.reduce_max (edges, 2) )
    if 'mean' in self.block_paras.e2fl_pool:
      face_local.append( tf.reduce_mean (edges, 2) )
    face_local = tf.concat(face_local, -1)

    face_global = tf.squeeze(face_centroid, 2)

    use_global = self.scale>0 or self.block_paras.use_face_global_scale0
    if use_global:
      faces = tf.concat([face_local, face_global], -1)
    else:
      faces = face_local
    return faces

  def face_2_vertex(self, faces, fidx_per_vertex, fidx_pv_empty_mask):
    vertices_flat = gather_second_d(faces, fidx_per_vertex)
    vertices = []
    if 'max' in self.block_paras.f2v_pool:
      vertices.append( tf.reduce_max(vertices_flat, 2) )
    if 'mean' in self.block_paras.f2v_pool:
      vertices.append( TfUtil.mask_reduce_mean(vertices_flat, 1-fidx_pv_empty_mask, 2) )
    vertices = tf.concat(vertices, axis=-1) # (nv, 2cf)
    return vertices


import numpy as np

class BlockParas():
  def __init__(self, block_configs):
    block_sizes = block_configs['block_sizes']
    filters = block_configs['filters']
    if 'kernels' in block_configs:
      kernels = block_configs['kernels']
    else:
      kernels = None
    self.edgevnum = block_configs['edgevnum']
    self.scale_num = len(filters['vertex'])
    if hasattr(self, 'e2fl_pool'):
      self.e2fl_pool = block_configs['e2fl_pool']
      self.f2v_pool = block_configs['f2v_pool']
      self.use_face_global_scale0 = block_configs['use_face_global_scale0']

    self.dense_filters = block_configs['dense_filters']
    #self.with_globals = [len(fs)>0 for fs in filters['global']]

    all_paras = {}
    for item in filters:
      # Always 1 if not exist
      block_size_1 = item not in block_sizes
      kernel_1 = item not in kernels
      if block_size_1:
        block_sizes[item] = []
      if kernel_1:
        kernels[item] = []

      if block_size_1 or kernel_1:
        scale_num  = len(filters[item])
        for s in range(scale_num):
          if block_size_1:
            block_sizes[item].append([1 for _ in range(len(filters[item][s]))])
          if kernel_1:
            kernels[item].append([1 for _ in range(len(filters[item][s]))])

      all_paras[item] = BlockParas.complete_scales_paras(block_sizes[item],
                                                filters[item], kernels[item])
    self.all_paras = all_paras

  def get_block_paras(self, element, scale):
    return self.all_paras[element][scale]

  @staticmethod
  def complete_scales_paras(block_size, filters, kernels):
    scale_num = len(block_size)
    scales_blocks_paras = []
    for s in range(scale_num):
      if kernels:
        kernels_s = kernels[s]
      else:
        kernels_s = None
      blocks_paras = BlockParas.complete_paras_1scale(block_size[s], filters[s], kernels_s)
      scales_blocks_paras.append(blocks_paras)
    return scales_blocks_paras

  @staticmethod
  def complete_paras_1scale(block_size, filters, kernels):
    if len(block_size) == 0:
      return None
    assert not isinstance(block_size[0], list)
    block_size  = np.array(block_size)
    filters     = np.array(filters)
    block_num = block_size.shape[0]

    blocks_paras = {}
    blocks_paras['block_sizes'] = block_size
    blocks_paras['filters'] = filters
    blocks_paras['kernels'], blocks_paras['strides'], blocks_paras['pad_stride1'] = \
                                BlockParas.get_1_kernel_block_paras(block_num, kernels)
    blocks_paras = BlockParas.split_block_paras(blocks_paras)

    return blocks_paras

  @staticmethod
  def split_block_paras(block_paras):
    '''
    from one dictionary  to one list
    Orginal: one dictionary contianing list of len = block_num
    Result: one list with len=block num, each element of the list is a dictionary for one block
    '''
    block_num = block_paras['block_sizes'].shape[0]
    block_paras_splited = []
    for s in range(block_num):
      block_para_s = {}
      for item in block_paras:
        block_para_s[item] = block_paras[item][s]
      block_paras_splited.append(block_para_s)
    return block_paras_splited


  @staticmethod
  def get_1_kernel_block_paras(block_num, kernels_):
    if kernels_:
      kernels = [kernels_[i] for i in range(block_num)]
    else:
      kernels = [1 for i in range(block_num)]
    strides = [1 for i in range(block_num)]
    paddings = ['v' for i in range(block_num)]
    return kernels, strides, paddings


