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
    if 'sg_settings' in net_data_configs:
      self.sg_settings = net_data_configs['sg_settings']
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
    points = tf.expand_dims( inputs['points'], 1)

    scale_n = self.block_paras.scale_num
    for scale in range(scale_n):
        points = self.point_encoder(scale, points)
        points = self.sg_pooling(scale, points, inputs['grouped_pindex'][scale])

    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    for scale in range(scale_n):
        points = self.point_decoder(scale, points)
        points = self.interp_uppooling(scale, points)

    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

  def point_encoder(self, scale,  points):
    with tf.variable_scope('PointEncoder_S%d'%(scale)):
      block_paras = self.block_paras[scale]
      new_points = self.blocks_layers(points, block_paras, self.building_block_v2,
                                    self.is_training, 'PE_S%d'%(scale),
                                    with_initial_layer = scale==0)
    return new_points

  def sg_pooling(self, scale, points, grouped_pindex):
    with tf.variable_scope('Pool_S%d'%(scale)):
      new_points = TfUtil.gather_third_d(points, grouped_pindex)
    return new_points

  def point_decoder(self, scale,  points):
    with tf.variable_scope('PointDecoder_S%d'%(scale)):
      new_points = points
    return new_points

  def interp_uppooling(self, scale, points):
    with tf.variable_scope('UpPool_S%d'%(scale)):
      new_points = points
    return new_points


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

    sg_params = self.gather_sg_params(features)
    inputs.update(sg_params)

    vshape = TfUtil.get_tensor_shape(points)
    #self.batch_size = vshape[0]
    self.num_vertex0 = vshape[1]
    self.log_tensor_p(points, 'points', 'raw_input')
    return inputs, xyz

  def gather_sg_params(self, features):
    if not hasattr(self, 'sg_settings'):
      return {}
    sg_scale_num = self.sg_settings['num_sg_scale']
    assert sg_scale_num >= self.block_paras.scale_num + 1, "not enough sg scales"
    sg_params = {}
    for s in range(sg_scale_num):
      for item in ['grouped_pindex', 'grouped_bot_cen_top', 'empty_mask', 'nb_enough_p']:
        if s==0:
          sg_params[item] = []
        else:
          sg_params[item].append(tf.squeeze( features[item+'_%d'%(s)], 1) )
    return sg_params

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
        bpar['block_sizes'] =  block_configs['block_sizes'][s][b]
        bpar['filters'] =  block_configs['filters'][s][b]
        bpar['kernels'] = 1
        bpar['strides'] = 1
        bpar['pad_stride1'] = 'v'
        blocks_pars.append(bpar)

      self.blocks_paras_scales.append( blocks_pars )

  def __getitem__(self, scale):
    return self.blocks_paras_scales[scale]

