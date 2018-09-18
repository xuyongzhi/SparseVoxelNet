# xyz Sep 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta

_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_EPSILON = 1e-4
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

class Model(object):
  def __init__(self, net_flag, dset_metas, net_data_configs, data_format, dtype):
    self.dset_shape_idx = net_data_configs['dset_shape_idx']
    self.data_config = net_data_configs['data_config']

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

  def __call__(self, features, training):
    '''
    points: [B,N,C]
    edges_per_vertex: [B,N,10*2]
    '''
    points, face_idx_per_vertex, fidx_pv_empty_mask, edges_per_vertex, \
      edges_pv_empty_mask = self.parse_inputs(features)

    #
    batch_size = tf.shape(points)[0]
    num_vertex0 = points.shape[1].value

    two_edge_vertices = Model.vertexF_2_envF(points, edges_per_vertex)
    edges_012_global, edges_01_local, edge_2_local = Model.vertexF_2_edgeF(two_edge_vertices,
                           tf.expand_dims( tf.expand_dims(points, 2),3))
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
     edges_012_global:  [B,N,3,C]
     edges_01_local:    [B,N,2,C]
     edge_2_local:      [B,N,1,C]
    '''
    tev_shape = two_edge_vertices.shape.as_list()
    bv_shape = base_vertex.shape.as_list()
    assert len(tev_shape) == 5
    assert tev_shape[3] == 2
    assert len(bv_shape) == 5
    assert bv_shape[2] == bv_shape[2] == 1
    edges_01_local = two_edge_vertices - base_vertex
    edges_01_global = (two_edge_vertices + base_vertex) / 2
    edge_2_local = tf.abs(two_edge_vertices[:,:,:,0:1,:] - two_edge_vertices[:,:,:,1:2,:])
    edge_2_global = (two_edge_vertices[:,:,:,0:1,:] + two_edge_vertices[:,:,:,1:2,:])/2

    edge_012_global = tf.concat([edges_01_global, edge_2_global], -2)

    batch_size = tf.shape(two_edge_vertices)[0]
    es = edge_012_global.shape.as_list()
    edges_012_global = tf.reshape(edge_012_global, [batch_size, es[1]*es[2], es[3], es[4]])
    edges_01_local = tf.reshape(edges_01_local, [batch_size, es[1]*es[2], 2, es[4]])
    edge_2_local = tf.reshape(edge_2_local, [batch_size, es[1]*es[2], 1, es[4]])
    return edges_012_global, edges_01_local, edge_2_local
