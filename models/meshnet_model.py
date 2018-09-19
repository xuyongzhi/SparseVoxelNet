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

  def parse_inputs(self, features):
    ds_idxs = self.dset_shape_idx['indices']
    def get_ele(ele):
      for g in ds_idxs:
        if ele in ds_idxs[g]:
          ele_idx = ds_idxs[g][ele]
          ele_data = tf.gather(features[g], ele_idx, axis=-1)
          return ele_data

    points = [get_ele(e) for e in self.data_config['feed_data']]
    points = tf.concat(points, -1)

    face_idx_per_vertex = get_ele('face_idx_per_vertex')
    fidx_pv_empty_mask = get_ele('fidx_pv_empty_mask')
    edges_per_vertex = get_ele('edges_per_vertex')
    edges_pv_empty_mask = get_ele('edges_pv_empty_mask')

    return points, face_idx_per_vertex, fidx_pv_empty_mask, edges_per_vertex, \
      edges_pv_empty_mask

  def __call__(self, features, is_training):
    '''
    points: [B,N,C]
    edges_per_vertex: [B,N,10*2]
    '''
    self.is_training = is_training
    points, face_idx_per_vertex, fidx_pv_empty_mask, edges_per_vertex, \
      edges_pv_empty_mask = self.parse_inputs(features)

    #
    self.batch_size = tf.shape(points)[0]
    self.num_vertex0 = points.shape[1].value

    for scale in range(3):
      two_edge_vertices = Model.vertexF_2_envF(points, edges_per_vertex)
      edges = Model.vertexF_2_edgeF(two_edge_vertices,
                            tf.expand_dims( tf.expand_dims(points, 2),3))
      self.edge_encoder(scale, edges)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass


  @staticmethod
  def vertexF_2_envF(points, edges_per_vertex):
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
    two_edge_vertices = tf.gather_nd(points, edges_per_vertex)
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
    edges['edges_01_local'] = two_edge_vertices - base_vertex
    edges_01_global = (two_edge_vertices + base_vertex) / 2
    edges['edge_2_local'] = tf.abs(two_edge_vertices[:,:,:,0:1,:] - two_edge_vertices[:,:,:,1:2,:])
    edge_2_global = (two_edge_vertices[:,:,:,0:1,:] + two_edge_vertices[:,:,:,1:2,:])/2

    edges['edges_012_global'] = tf.concat([edges_01_global, edge_2_global], -2)
    return edges

  def edge_encoder(self, scale, edges):
    '''
    '''
    with tf.variable_scope('edge_encoder_%d'%(scale)):
      block_fn = self.block_fn if scale!=0 else self.building_block_v2
      block_params = BlockParas.edge_block_paras()
      block_num = len(block_params)
      for edge_flag in edges:
        with tf.variable_scope(edge_flag):
          for bi in range(block_num):
            with tf.variable_scope('B%d'%(bi)):
              edges[edge_flag] = self.block_layer(scale, edges[edge_flag],
                                block_params[bi], block_fn,
                                self.is_training, '{}_b{}'.format(edge_flag, bi))
    return edges
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass


import numpy as np
class BlockParas():

  @staticmethod
  def edge_block_paras():
    block_size  = np.array([1, 1])
    filters     = np.array([24, 24])
    block_num = block_size.shape[0]

    block_paras = {}
    block_paras['block_sizes'] = block_size
    block_paras['filters'] = filters
    block_paras['kernels'], block_paras['strides'], block_paras['pad_stride1'] = \
                                BlockParas.get_1_kernel_block_paras(block_num)
    block_paras = BlockParas.split_block_paras(block_paras)

    return block_paras

  @staticmethod
  def split_block_paras(block_paras):
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




