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
    xyz_scales = []
    scale_n = self.model_paras.scale_num
    #***************************************************************************
    # Encoder
    with_global_scale = False
    for scale in range(scale_n):
      points, end_points_s = self.point_encoder(scale, points)
      endpoints.append(end_points_s)
      if scale>0:
        # Change voxel to point. Scale 0 is already point, no need pooing
        points = self.pooling(scale, points, xyz)
      if scale != scale_n-1:
        if scale > len(inputs['grouped_pindex'])-1:
          assert scale == scale_n-2 # global scale has to be the last
          grouped_pindex_s = None
          with_global_scale = True
        else:
          grouped_pindex_s = inputs['grouped_pindex'][scale]
        points, xyz = self.sampling_grouping(scale, points, grouped_pindex_s, xyz)

    #***************************************************************************
    # Decoder
    for i in range(scale_n-1):
      scale = scale_n - i -2
      if with_global_scale and scale == scale_n -2:
        flatting_idx = None
      else:
        flatting_idx = inputs['flatting_idx'][scale]
      points = self.interp_uppooling(scale, points, flatting_idx, endpoints[scale])
      points = self.point_decoder(scale, points)

    dense_filters = self.model_paras.dense_filters + [self.dset_metas.num_classes]
    points = tf.squeeze(points, 2)
    logits = self.dense_block(points, dense_filters, self.is_training)
    label_weights = tf.constant(1.0)

    self.log_model_summary()
    return logits, label_weights

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
    assert TfUtil.tsize(points) == 3

    sg_params = self.gather_sg_params(features)
    inputs.update(sg_params)

    vshape = TfUtil.get_tensor_shape(points)
    #self.batch_size = vshape[0]
    self.num_vertex0 = vshape[1]
    self.log_tensor_p(points, 'points', 'raw_input')
    return inputs, xyz

  def point_encoder(self, scale,  points):
    with tf.variable_scope('PointEncoder_S%d'%(scale)):
      block_paras = self.model_paras('e',scale)
      new_points, end_points_s = self.blocks_layers(points, block_paras, self.building_block_v2,
                                    self.is_training, 'PE_S%d'%(scale),
                                    with_initial_layer = scale==0,
                                    end_blocks=self.model_paras.end_blocks[scale])
    return new_points, end_points_s

  def pooling(self, scale, points, xyz, pool='max', use_xyz = True):
    with tf.variable_scope('Pool_S%d'%(scale)):
      points = tf.reduce_max(points, 2)
      points = tf.expand_dims(points, 1)
      use_xyz_str = ''
      if use_xyz and xyz is not None:
        points = tf.concat([points, tf.expand_dims(xyz,1)], -1)
        use_xyz_str = 'use_xyz'
      self.log_tensor_p(points, use_xyz_str, pool+' pooling')
      return points

  def sampling_grouping(self, scale, points, grouped_pindex, xyz):
    with tf.variable_scope('SG_S%d'%(scale)):
      if grouped_pindex is None:
        # global scale
        new_points = points
        new_xyz = None
      else:
        new_points = TfUtil.gather_third_d(points, grouped_pindex)
        new_xyz = TfUtil.gather_second_d(xyz, grouped_pindex)
        new_xyz = tf.reduce_mean(new_xyz, 2)
        new_points = self.normalize_feature(new_points, 2)
        self.log_tensor_p(new_points, '', 'saming grouping norm')
    return new_points, new_xyz

  def point_decoder(self, scale,  points):
    with tf.variable_scope('PointDecoder_S%d'%(scale)):
      block_paras = self.model_paras('d',scale)
      new_points, _ = self.blocks_layers(points, block_paras, self.building_block_v2,
                                    self.is_training, 'PD_S%d'%(scale),
                                    with_initial_layer = False)
    return new_points

  def interp_uppooling(self, scale, points, flatting_idx, last_points):
    with tf.variable_scope('UpPool_S%d'%(scale)):
      if flatting_idx is None:
        voexl_num = TfUtil.get_tensor_shape(last_points)[1]
        points = tf.tile(points, [1,voexl_num,1,1])
      else:
        points = TfUtil.gather_third_d(points, flatting_idx)
        points = tf.reduce_mean(points, 2, keepdims=True)
      self.log_tensor_p(points, 'uppooling', 'scale%d'%(scale))

      if scale==0:
        last_points = tf.transpose(last_points, [0,2,1,3])
      if self.sg_settings['num_sg_scale'] >1:
        fnl = TfUtil.get_tensor_shape(last_points)[2]
        points = tf.tile(points, [1,1,fnl,1])
      else:
        fnl = TfUtil.get_tensor_shape(last_points)[1]
        points = tf.tile(points, [1,fnl,1,1])
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

  def normalize_feature(self, points, dim):
    assert TfUtil.tsize(points) == 4
    mean_p = tf.reduce_mean(points, dim, keepdims=True)
    points = points - mean_p
    return points

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
          sg_params[item].append( item_v )

      if s>0:
        assert TfUtil.tsize(sg_params['flatting_idx'][-1]) == 3
        assert TfUtil.tsize(sg_params['grouped_pindex'][-1]) == 3
        assert TfUtil.tsize(sg_params['grouped_bot_cen_top'][-1]) == 4
    return sg_params


class ModelParams():
  def __init__(self, block_configs):
    # include global
    self.dense_filters = block_configs['dense_filters']
    self.end_blocks = block_configs['end_blocks']

    self.blocks_paras_scales = {}
    scale_nums = {}
    for net in ['e', 'd']:
      scale_nums[net] = scale_num = len(block_configs['filters_'+net])
      self.blocks_paras_scales[net] = []
      for s in range(scale_num):
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

    self.scale_num = scale_nums['e']
    assert scale_nums['e'] == scale_nums['d'] + 1

  def __call__(self, encoder_or_decoder, scale):
    return self.blocks_paras_scales[encoder_or_decoder][scale]

