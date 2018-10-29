# xyz Sep 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
from models.conv_util import ResConvOps, gather_second_d
from utils.tf_util import TfUtil
import numpy as np

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
    self.model_paras = ModelParams(net_data_configs['block_configs'])
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

    endpoints = []
    scale_n = self.model_paras.scale_num
    for scale in range(scale_n):
        points = self.point_encoder(scale, points)
        endpoints.append(points)
        if scale>0:
          points = self.pooling(scale, points)
        if scale != scale_n-1:
          points = self.sampling_grouping(scale, points, inputs['grouped_pindex'][scale])

    for i in range(scale_n-1):
      scale = scale_n - i -2
      points = self.interp_uppooling(scale, points, inputs['flatting_idx'][scale], endpoints[scale])
      points = self.point_decoder(scale, points)

    dense_filters = self.model_paras.dense_filters + [self.dset_metas.num_classes]
    points = tf.squeeze(points, 2)
    logits = self.dense_block(points, dense_filters, self.is_training)
    label_weights = tf.constant(1.0)
    return logits, label_weights

  def point_encoder(self, scale,  points):
    with tf.variable_scope('PointEncoder_S%d'%(scale)):
      block_paras = self.model_paras('e',scale)
      new_points = self.blocks_layers(points, block_paras, self.building_block_v2,
                                    self.is_training, 'PE_S%d'%(scale),
                                    with_initial_layer = scale==0)
    return new_points

  def pooling(self, scale, points, pool='max'):
    with tf.variable_scope('Pool_S%d'%(scale)):
      points = tf.reduce_max(points, 2)
      points = tf.expand_dims(points, 1)
      self.log_tensor_p(points, pool+' pooling', 'scale%d'%(scale))
      return points

  def sampling_grouping(self, scale, points, grouped_pindex):
    with tf.variable_scope('Pool_S%d'%(scale)):
      new_points = TfUtil.gather_third_d(points, grouped_pindex)
    return new_points

  def point_decoder(self, scale,  points):
    with tf.variable_scope('PointDecoder_S%d'%(scale)):
      block_paras = self.model_paras('d',scale)
      new_points = self.blocks_layers(points, block_paras, self.building_block_v2,
                                    self.is_training, 'PD_S%d'%(scale),
                                    with_initial_layer = False)
    return new_points

  def interp_uppooling(self, scale, points, flatting_idx, last_points):
    with tf.variable_scope('UpPool_S%d'%(scale)):
      points = TfUtil.gather_third_d(points, flatting_idx)
      self.log_tensor_p(points, 'uppooling', 'scale%d'%(scale))
      points = tf.reduce_mean(points, 2, keepdims=True)

      if scale==0:
        last_points = tf.transpose(last_points, [0,2,1,3])
      fnl = TfUtil.get_tensor_shape(last_points)[2]
      points = tf.tile(points, [1,1,fnl,1])
      points = tf.concat([last_points, points], -1)
      self.log_tensor_p(points, 'mean & interperation', 'scale%d'%(scale))

    return points


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
    assert sg_scale_num >= self.model_paras.scale_num -1 , "not enough sg scales"
    sg_params = {}
    for s in range(sg_scale_num):
      for item in ['grouped_pindex', 'grouped_bot_cen_top', 'empty_mask', \
                   'nb_enough_p', 'flatting_idx', 'flat_valid_mask']:
        if s==0:
          sg_params[item] = []
        else:
          item_v = features[item+'_%d'%(s)]
          if item not in ['flatting_idx', 'flat_valid_mask']:
            item_v =  tf.squeeze( item_v, 1)
          sg_params[item].append( item_v )
    return sg_params


class ModelParams():
  def __init__(self, block_configs):
    # include global
    self.scale_num = len(block_configs['filters_e'])
    self.dense_filters = block_configs['dense_filters']

    self.blocks_paras_scales = {}
    for net in ['e', 'd']:
      self.blocks_paras_scales[net] = []
      for s in range(self.scale_num-int(net=='d')):
        block_num = len(block_configs['block_sizes_'+net][s])
        assert block_num == len(block_configs['filters_'+net][s])
        blocks_pars = []
        for b in range(block_num):
          bpar = {}
          bpar['block_sizes'] =  block_configs['block_sizes_'+net][s][b]
          bpar['filters'] =  block_configs['filters_'+net][s][b]
          bpar['kernels'] = 1
          bpar['strides'] = 1
          bpar['pad_stride1'] = 'v'
          blocks_pars.append(bpar)

        self.blocks_paras_scales[net].append( blocks_pars )

  def __call__(self, encoder_or_decoder, scale):
    return self.blocks_paras_scales[encoder_or_decoder][scale]

