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
  CNN = 'POINT'
  def __init__(self, net_data_configs, data_format, dtype):
    self.dset_shape_idx = net_data_configs['dset_shape_idx']
    self.data_configs = net_data_configs['data_configs']
    self.dset_metas = net_data_configs['dset_metas']
    self.net_flag = net_data_configs['net_flag']
    self.block_paras = BlockParas(net_data_configs['block_configs'])
    super(Model, self).__init__(net_data_configs, data_format, dtype)

  def __call__(self, features,  is_training):
    '''
    points: [B,N,C]
    edges_per_vertex: [B,N,10*2]
    '''
    return self.main_pointnet_vsg(features, is_training)

  def main_pointnet_vsg(self, features, is_training):
    self.is_training = is_training
    inputs, xyz = self.parse_inputs(features)
    points = inputs['points']

    scale_n = self.block_paras.scale_num
    for scale in range(scale_n):
        points = self.point_encoder(scale, points)
        points = self.pooling(scale, points)

    for scale in range(scale_n):
        points = self.point_decoder(scale, points)
        points = self.uppooling(scale, points)

    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  def point_encoder(self, scale,  points):
    with tf.variable_scope('PointEncoder_S%d'%(scale)):
      block_paras = self.block_paras[scale]
      new_points = self.blocks_layers(points, block_paras, self.building_block_v2,
                                     self.is_training, 'S%d'%(scale))
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return new_points

  def pooling(self, scale, points):
    with tf.variable_scope('Pool_S%d'%(scale)):
      new_points = points
    return new_points

  def point_decoder(self, scale,  points):
    with tf.variable_scope('PointDecoder_S%d'%(scale)):
      new_points = points
    return new_points

  def pooling(self, scale, points):
    with tf.variable_scope('UpPool_S%d'%(scale)):
      new_points = points
    return new_points


  def main_fan_cnn(self, features, is_training):
    self.is_training = is_training
    inputs, xyz = self.parse_inputs(features)
    points = inputs['points']
    edgev_per_vertex = inputs['edgev_per_vertex']
    valid_ev_num_pv = inputs['valid_ev_num_pv']
    valid_ev_num_pv = inputs['valid_ev_num_pv']
    vidx_per_face = inputs['vidx_per_face']
    valid_num_face = tf.cast(inputs['valid_num_face'], tf.int32)

    #***************************************************************************
    # multi scale vertex feature encoder
    scale_n = self.block_paras.scale_num
    points_scales = []
    backprop_vidx_scales = []
    r2s_fail_mask_scales = []
    for scale in range(scale_n):
      with tf.variable_scope('FanCnn_S%d'%(scale)):
        points = self.mesh_cnn.update_vertex(scale, is_training, points,
              edgev_per_vertex, valid_ev_num_pv)
      if scale < self.block_paras.scale_num-1:
        points_scales.append(points)
        with tf.variable_scope('MeshPool_S%d'%(scale)):
          points, edgev_per_vertex, xyz, valid_ev_num_pv, backprop_vidx, \
            r2s_fail_mask = FanCnn.pool_mesh(scale,
                                points, edgev_per_vertex, xyz, valid_ev_num_pv)
        backprop_vidx_scales.append(backprop_vidx)
        r2s_fail_mask_scales.append(r2s_fail_mask)

    #***************************************************************************
    with tf.variable_scope('CatGlobal'):
      points = self.add_global(points)
    #***************************************************************************
    # multi scale feature back propogation
    for i in range(scale_n-1):
      scale = scale_n -2 - i
      with tf.variable_scope('FanCnn_S%d'%(scale)):
        points = self.feature_backprop(scale, points, points_scales[scale],
                      backprop_vidx_scales[scale], r2s_fail_mask_scales[scale])


    flogits, flabel_weight = self.face_classifier(points, vidx_per_face, valid_num_face)
    self.log_model_summary()
    return flogits, flabel_weight

  def feature_backprop(self, scale, cur_points, lasts_points, backprop_vidx, r2s_fail_mask):
    cur_points = gather_second_d(cur_points, backprop_vidx)
    r2s_fail_mask = tf.cast(tf.expand_dims(tf.expand_dims(r2s_fail_mask, -1),-1), tf.float32)
    cur_points = cur_points * (1-r2s_fail_mask)
    points = tf.concat([lasts_points, cur_points], -1)

    blocks_params = self.block_paras.get_block_paras('backprop', scale)
    points = self.blocks_layers(points, blocks_params, self.block_fn,
                    self.is_training, 'BackProp_%d'%(scale), with_initial_layer=False)
    return points

  def add_global(self, points):
    blocks_params = self.block_paras.get_block_paras('global', 0)
    if not blocks_params:
      return points

    #***************************************************************************
    global_f = tf.reduce_max(points, 1, keepdims=True)
    # encoder global feature
    global_f = self.blocks_layers(global_f, blocks_params, self.block_fn,
                                  self.is_training, 'Global',
                                  with_initial_layer=False)
    nv = TfUtil.get_tensor_shape(points)[1]
    global_f = tf.tile(global_f, [1, nv, 1, 1])
    points = tf.concat([points, global_f], -1)
    self.log_tensor_p(points, '', 'cat global')

    #***************************************************************************
    # encoder fused vertex feature
    blocks_params = self.block_paras.get_block_paras('global', 1)
    if not blocks_params:
      return points
    points = self.blocks_layers(points, blocks_params, self.block_fn,
                                  self.is_training, 'GlobalFusedV',
                                  with_initial_layer=False)
    return points

  def face_classifier(self, points, vidx_per_face, valid_num_face):
    dense_filters = self.block_paras.dense_filters + [self.dset_metas.num_classes]
    vlogits = self.dense_block(points, dense_filters, self.is_training)
    vlogits = tf.squeeze(vlogits, 2)
    flogits = gather_second_d(vlogits, vidx_per_face)
    flogits = tf.reduce_mean(flogits, 2)
    fn = TfUtil.get_tensor_shape(vidx_per_face)[1]
    valid_face_mask = tf.tile(tf.reshape(tf.range(fn), [1,fn]), [self.batch_size,1])
    flabel_weight = tf.cast(tf.less(valid_face_mask, valid_num_face), tf.float32)
    return flogits, flabel_weight


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


  def normalize_color(self, color):
    color = tf.cast(color, tf.float32) / 255.0
    return color

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
    points = []
    for e in self.data_configs['feed_data']:
      ele = self.get_ele(features, e)
      if e=='xyz':
        xyz = ele = self.normalize_xyz(ele)
      if e=='color':
        ele = self.normalize_color(ele)
      points.append(ele)
    inputs['points'] = points = tf.concat(points, -1)
    vshape = TfUtil.get_tensor_shape(points)
    #self.batch_size = vshape[0]
    self.num_vertex0 = vshape[1]
    self.log_tensor_p(points, 'points', 'raw_input')
    return inputs, xyz

  def simplicity_classifier(self, points):
    dense_filters = [32, 16, 2]
    simplicity_logits = self.dense_block(points, dense_filters, self.is_training)
    return simplicity_logits



import numpy as np

class BlockParas():
  def __init__(self, block_configs):
    # include global
    self.scale_num = len(block_configs['filters'])
    self.dense_filters = block_configs['dense_filters']

    self.blocks_paras_scales = []
    for s in range(self.scale_num):
      block_num = len(block_configs['block_sizes'][s])
      assert block_num == len(block_configs['filters'][s])
      blocks_pars = []
      for b in range(block_num):
        bpar = {}
        bpar['block_sizes'] = np.array( block_configs['block_sizes'][s][b] )
        bpar['filters'] = np.array( block_configs['filters'][s][b] )
        bpar['kernels'] = np.ones([1], np.int32)
        bpar['strides'] = np.ones([1], np.int32)
        bpar['pad_stride1'] = 'v'
        blocks_pars.append(bpar)

      self.blocks_paras_scales.append( blocks_pars )

  def __getitem__(self, scale):
    return self.blocks_paras_scales[scale]

