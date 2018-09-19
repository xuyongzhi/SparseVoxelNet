# xyz Sep 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
from models.conv_util import ResConvOps

DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

class Model(ResConvOps):
  def __init__(self, net_data_configs, data_format, dtype):
    self.dset_shape_idx = net_data_configs['dset_shape_idx']
    self.data_config = net_data_configs['data_config']
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

  def __call__(self, features, is_training):
    '''
    vertices: [B,N,C]
    edges_per_vertex: [B,N,10*2]
    '''
    self.is_training = is_training
    vertices, face_idx_per_vertex, fidx_pv_empty_mask, edges_per_vertex, \
      edges_pv_empty_mask = self.parse_inputs(features)

    #
    self.batch_size = tf.shape(vertices)[0]
    self.num_vertex0 = vertices.shape[1].value

    vertices_scales = []
    for scale in range(2):
      two_edge_vertices = Model.vertexF_2_envF(vertices, edges_per_vertex)
      edges = Model.vertexF_2_edgeF(two_edge_vertices,
                            tf.expand_dims( tf.expand_dims(vertices, 2),3))
      edges = self.edge_encoder(scale, edges)
      faces = self.edgeF_2_faceF(scale, edges)
      vertices = self.faceF_2_vertexF(scale, faces, vertices)
      vertices_scales.append(vertices)

    vertices = tf.concat(vertices_scales, -1)
    simplicity_logits = self.simplicity_classifier(vertices)
    simplicity_label = self.simplicity_label(features)
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
    ds_idxs = self.dset_shape_idx['indices']
    for g in ds_idxs:
      if ele in ds_idxs[g]:
        ele_idx = ds_idxs[g][ele]
        ele_data = tf.gather(features[g], ele_idx, axis=-1)
        return ele_data
    raise ValueError

  def parse_inputs(self, features):

    vertices = [self.get_ele(features,e) for e in self.data_config['feed_data']]
    vertices = tf.concat(vertices, -1)

    face_idx_per_vertex = self.get_ele(features, 'face_idx_per_vertex')
    fidx_pv_empty_mask = self.get_ele(features, 'fidx_pv_empty_mask')
    edges_per_vertex = self.get_ele(features, 'edges_per_vertex')
    edges_pv_empty_mask = self.get_ele(features, 'edges_pv_empty_mask')

    return vertices, face_idx_per_vertex, fidx_pv_empty_mask, edges_per_vertex, \
      edges_pv_empty_mask

  def simplicity_classifier(self, vertices):
    dense_filters = [24, 2]
    simplicity_logits = self.dense_block(vertices, dense_filters, self.is_training)
    return simplicity_logits


  @staticmethod
  def vertexF_2_envF(vertices, edges_per_vertex):
    '''
    vertex feature to two edge (neighbour) vertex feature
    edges_per_vertex: [B,N,10*2]
    two_edge_vertices: [B,N,10,2,C]
    '''
    epv_shape = edges_per_vertex.shape.as_list()
    assert len(epv_shape) == 3
    batch_size = tf.shape(edges_per_vertex)[0]
    num_vertex0 = edges_per_vertex.shape[1].value
    num_edges_perv = edges_per_vertex.shape[2].value
    assert num_edges_perv %2 == 0
    num_faces_perv = num_edges_perv // 2

    edges_per_vertex = tf.reshape(edges_per_vertex,
                          [batch_size, num_vertex0, num_faces_perv, 2])

    # gather neighbor(edge) vertex features
    batch_idx = tf.reshape(tf.range(batch_size), [-1,1,1,1,1])
    batch_idx = tf.tile(batch_idx, [1, num_vertex0, num_faces_perv, 2,1])
    edges_per_vertex = tf.expand_dims(edges_per_vertex, -1)
    edges_per_vertex = tf.concat([batch_idx, edges_per_vertex], -1)
    # the two edge vertices for each vertex
    # [B,N,10,2,C]
    two_edge_vertices = tf.gather_nd(vertices, edges_per_vertex)
    return two_edge_vertices

  @staticmethod
  def vertexF_2_edgeF(two_edge_vertices, base_vertex):
    '''
     two_edge_vertices: [B,N,10,2,C]
     base_vertex:       [B,N,1,1,C]

     edges_012_global:  [B,N,10,3,C]
     edges_01_local:    [B,N,10,2,C]
     edge_2_local:      [B,N,10,1,C]
    '''
    tev_shape = two_edge_vertices.shape.as_list()
    bv_shape = base_vertex.shape.as_list()
    assert len(tev_shape) == 5
    assert tev_shape[3] == 2
    assert len(bv_shape) == 5
    assert bv_shape[2] == bv_shape[2] == 1
    edges = {}
    edges['e01l'] = two_edge_vertices - base_vertex
    edges_01_global = (two_edge_vertices + base_vertex) / 2
    edges['e2l'] = tf.abs(two_edge_vertices[:,:,:,0:1,:] - two_edge_vertices[:,:,:,1:2,:])
    edge_2_global = (two_edge_vertices[:,:,:,0:1,:] + two_edge_vertices[:,:,:,1:2,:])/2

    edges['e012g'] = tf.concat([edges_01_global, edge_2_global], -2)
    return edges

  def edge_encoder(self, scale, edges):
    '''
    '''

    blocks_params = BlockParas.block_paras('edge')[scale]
    for edge_flag in edges:
      if self.IsShowModel:
        self.log('\t\t\t******* %s *******'%(edge_flag))
      edges[edge_flag] = self.blocks_layers(scale, edges[edge_flag],
                  blocks_params, self.block_fn, self.is_training, edge_flag+'_s%d'%(scale))
    e012l = tf.concat( [edges['e01l'], edges['e2l']], -2)
    edges = e012l + edges['e012g']

    if self.IsShowModel:
      self.log('\n')
    self.log_tensor_p(edges, 'add', 'edge encoder out')
    return edges

  def edgeF_2_faceF(self, scale, edges):
    '''
    edges: [B,N,face_num,3,C]
    '''
    faces_maxp = tf.reduce_max(edges, -2)
    faces_meanp = tf.reduce_mean(edges, -2)
    faces = tf.concat([faces_maxp, faces_meanp], -1)
    blocks_params = BlockParas.block_paras('face')[scale]
    faces = self.blocks_layers(scale, faces, blocks_params, self.block_fn,
                         self.is_training, 'face_s%d'%(scale))
    return faces

  def faceF_2_vertexF(self, scale, faces, vertices):
    new_vertices_maxp = tf.reduce_max(faces, -2)
    new_vertices_meanp = tf.reduce_mean(faces, -2)
    new_vertices = new_vertices_maxp
    #new_vertices = tf.concat([new_vertices_maxp, new_vertices_meanp])

    blocks_params = BlockParas.block_paras('vertex')[scale]
    new_vertices = self.blocks_layers(scale, new_vertices, blocks_params,
                            self.block_fn, self.is_training, 'vertex_s%d'%(scale))
    return new_vertices

import numpy as np
class BlockParas():
  @staticmethod
  def block_paras(element):
    block_sizes = {}
    filters = {}

    block_sizes['edge'] = [ [1],  [1] ]
    filters['edge']     = [ [24], [32]]

    block_sizes['face'] = [ [2],  [1] ]
    filters['face']     = [ [32], [48]]

    block_sizes['vertex']=[ [1],  [1] ]
    filters['vertex']   = [ [48], [64]]

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




