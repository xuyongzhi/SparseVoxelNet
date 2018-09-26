# xyz Sep 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
from models.conv_util import ResConvOps, gather_second_d

DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

class Model(ResConvOps):
  def __init__(self, net_data_configs, data_format, dtype):
    self.dset_shape_idx = net_data_configs['dset_shape_idx']
    self.data_configs = net_data_configs['data_configs']
    self.dset_metas = net_data_configs['dset_metas']
    self.net_flag = net_data_configs['net_flag']
    super(Model, self).__init__(net_data_configs, data_format, dtype)

    block_style = 'Regular'
    if block_style == 'Regular':
      self.block_fn = self.building_block_v2
    elif block_style == 'Bottleneck':
      self.block_fn = self.bottleneck_block_v2
    elif block_style == 'Inception':
      self.block_fn = self.inception_block_v2
    elif block_style == 'PointNet':
      self.block_fn = None
    else:
      raise NotImplementedError
    self.mesh_cnn = MeshCnn(self.blocks_layers, self.block_fn)

  def __call__(self, features, labels, is_training):
    '''
    vertices: [B,N,C]
    edges_per_vertex: [B,N,10*2]
    '''
    vertex_datas = features
    face_datas = labels
    self.is_training = is_training
    vertices, fidx_per_vertex, fidx_pv_empty_mask, vidx_per_face, valid_num_face\
                              = self.parse_inputs(vertex_datas, face_datas)

    #
    self.num_vertex0 = vertices.shape[1].value

    vertices_scales = []
    for scale in range(2):
      with tf.variable_scope('S%d'%(scale)):
        vertices = self.mesh_cnn.update_vertex(scale, is_training, vertices,
              vidx_per_face, valid_num_face, fidx_per_vertex, fidx_pv_empty_mask)

      vertices_scales.append(vertices)

    #vertices = tf.concat(vertices_scales, -1)
    simplicity_logits = self.simplicity_classifier(vertices)
    simplicity_label = self.simplicity_label(vertex_datas)
    self.log_model_summary()
    return simplicity_logits, simplicity_label


  def simplicity_label(self, vertex_datas):
    min_same_norm_mask = 2
    min_same_category_mask = 2
    same_category_mask = self.get_ele(vertex_datas, 'same_category_mask')
    same_category_mask = tf.greater_equal(same_category_mask, min_same_category_mask)
    same_normal_mask = self.get_ele(vertex_datas, 'same_normal_mask')
    same_normal_mask = tf.greater_equal(same_normal_mask, min_same_norm_mask)

    simplicity_mask = tf.logical_and(same_normal_mask, same_category_mask)
    simplicity_mask = tf.squeeze(simplicity_mask, -1)
    simplicity_label = tf.cast(simplicity_mask, tf.int32)
    return simplicity_label

  def get_ele(self, vertex_datas, ele):
    ds_idxs = self.dset_shape_idx['indices']
    for g in ds_idxs:
      if ele in ds_idxs[g]:
        ele_idx = ds_idxs[g][ele]
        ele_data = tf.gather(vertex_datas[g], ele_idx, axis=-1)
        return ele_data
    raise ValueError, ele+' not found'

  def parse_inputs(self, vertex_datas, face_datas):

    vertices = [self.get_ele(vertex_datas,e) for e in self.data_configs['feed_data']]
    vertices = tf.concat(vertices, -1)

    fidx_per_vertex = self.get_ele(vertex_datas, 'fidx_per_vertex')
    fidx_pv_empty_mask = self.get_ele(vertex_datas, 'fidx_pv_empty_mask')

    vidx_per_face = self.get_ele(face_datas, 'vidx_per_face')
    valid_num_face = face_datas['valid_num_face']

    self.batch_size = tf.shape(vertices)[0]
    self.log_tensor_p(vertices, 'vertices', 'raw_input')
    return vertices, fidx_per_vertex, fidx_pv_empty_mask, \
          vidx_per_face, valid_num_face

  def simplicity_classifier(self, vertices):
    dense_filters = [24, 2]
    simplicity_logits = self.dense_block(vertices, dense_filters, self.is_training)
    return simplicity_logits


class MeshCnn():
  def __init__(self, blocks_layers_fn=None, block_fn=None):
    self.block_fn = block_fn
    self.blocks_layers = blocks_layers_fn
    self.use_face_global_scale0 = False

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
    blocks_params = BlockParas.block_paras(represent)[self.scale]
    outputs = self.blocks_layers(self.scale, inputs, blocks_params, self.block_fn,
                               self.is_training, '%s_s%d'%(represent, self.scale) )
    return outputs

  def edge_2_face(self, edges, face_centroid):
    face_local0 = tf.reduce_max (edges, 2)
    face_local1 = tf.reduce_mean(edges, 2)
    #face_local = tf.concat([face_local0, face_local1], -1)
    face_local = face_local0

    face_global = tf.squeeze(face_centroid, 2)

    use_global = self.scale>0 or self.use_face_global_scale0
    if use_global:
      faces = tf.concat([face_local, face_global], -1)
    else:
      faces = face_local
    return faces

  def face_2_vertex(self, faces, fidx_per_vertex, fidx_pv_empty_mask):
    vertices_flat = gather_second_d(faces, fidx_per_vertex)
    vertices0 = tf.reduce_max(vertices_flat, 2)
    vertices1 = tf.reduce_mean(vertices_flat, 2)
    vertices = tf.concat([vertices0, vertices1], axis=-1) # (nv, 2cf)
    return vertices


import numpy as np

class BlockParas():
  @staticmethod
  def block_paras(element):
    block_sizes = {}
    filters = {}

    block_sizes['edge'] = [ [1],  [1] ]
    filters['edge']     = [ [32], [32]]

    block_sizes['centroid']=[ [1],  [1] ]
    filters['centroid']   = [ [32], [32]]

    block_sizes['face'] = [ [1, 1],  [1, 1] ]
    filters['face']     = [ [64, 32], [64, 32]]

    block_sizes['vertex']=[ [1],  [1] ]
    filters['vertex']   = [ [64], [64]]


    all_paras = {}
    for item in block_sizes:
      all_paras[item] = BlockParas.complete_scales_paras(block_sizes[item], filters[item])
    return all_paras[element]

  @staticmethod
  def complete_scales_paras(block_size, filters):
    scale_num = len(block_size)
    scales_blocks_paras = []
    for s in range(scale_num):
      blocks_paras = BlockParas.complete_paras_1scale(block_size[s], filters[s])
      scales_blocks_paras.append(blocks_paras)
    return scales_blocks_paras

  @staticmethod
  def complete_paras_1scale(block_size, filters):
    assert not isinstance(block_size[0], list)
    block_size  = np.array(block_size)
    filters     = np.array(filters)
    block_num = block_size.shape[0]

    blocks_paras = {}
    blocks_paras['block_sizes'] = block_size
    blocks_paras['filters'] = filters
    blocks_paras['kernels'], blocks_paras['strides'], blocks_paras['pad_stride1'] = \
                                BlockParas.get_1_kernel_block_paras(block_num)
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
  def get_1_kernel_block_paras(block_num):
    kernels = [1 for i in range(block_num)]
    strides = [1 for i in range(block_num)]
    paddings = ['v' for i in range(block_num)]
    return kernels, strides, paddings


