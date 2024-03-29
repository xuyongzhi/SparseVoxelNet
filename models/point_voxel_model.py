
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys
import tf_util
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))

DEBUG_TMP = False

_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_EPSILON = 1e-4
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
#CASTABLE_TYPES = (tf.float32,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES
#ALLOWED_TYPES = (DEFAULT_DTYPE,)

NoRes_InceptionReduction = True

KERNEL_INI = tf.variance_scaling_initializer()       # res official

def tensor_info(tensor_ls, tensor_name_ls=None, layer_name=None,
                weight_num_bytes_shapes=None, batch_size=None):
  if type(tensor_ls) != list:
    tensor_ls = [tensor_ls]
  if tensor_name_ls == None:
    tensor_name_ls = [''] * len(tensor_ls)
  elif type(tensor_name_ls) != list:
    tensor_name_ls = [tensor_name_ls]
  tensor_sum = ''

  for i in range(len(tensor_ls)):
    if layer_name!=None:
      tensor_sum += '%-20s'%(layer_name)
    tensor_sum += '%-20s: '%(tensor_name_ls[i])
    if tensor_ls[i] == None:
        tensor_sum += 'None'
    else:
      if tensor_ls[i].shape[0].value!=None and batch_size!=None:
        map_size = tensor_ls[i].shape[0].value / batch_size
      else:
        map_size = 1
      shape_i = tensor_ls[i].shape.as_list()
      activation_shape_str = str(shape_i)
      shape_i = shape_i[1:]
      activation_size = np.prod(shape_i)  * tensor_ls[i].dtype.size * map_size
      activation_size_str = '(%0.1fK)'%(activation_size/1024.0)
      tensor_sum += '%-40s'%(str( activation_shape_str + activation_size_str ))

    if weight_num_bytes_shapes!=None:
      weight_num, weight_bytes, weight_shapes = weight_num_bytes_shapes
      weight_str = '  '.join([str(shape) for shape in weight_shapes])
      weight_str += ' (%d %0.1fK)'%(weight_num, weight_bytes/1024.0)
      tensor_sum += '%-30s'%(weight_str)
    if i < len(tensor_ls)-1:
        tensor_sum += '\n'
  return tensor_sum


def unique_nd( inputs, axis=-1, unit=3 ):
    org_inputs = inputs
    org_shape = inputs.shape
    batch_size = org_shape[0].value
    block_num = org_shape[1].value
    point_num = org_shape[2].value
    assert org_shape[3].value == 3

    units = tf.constant( [[9],[3],[1]], tf.float32 )
    inputs = tf.identity( inputs, name='uni_in0' ) # gpu_0/sa_layer4/uni_in0:0
    inputs = tf.reshape( inputs, [batch_size*block_num, point_num,3] )
    first_unique_masks = []
    for i in range(batch_size*block_num):
        inputs_i = tf.reshape( inputs[i], [-1,3], name='uni_inb_%d'%(i) ) # gpu_0/sa_layer4/uni_inb_0:0
        ids = tf.squeeze( tf.matmul( inputs_i, units, name='ids_%d'%(i) ))
        ids_unique, idx_unique = tf.unique( ids, name='idx_unique_%d'%(i) ) # gpu_0/sa_layer4/idx_unique_0:0  gpu_0/sa_layer4/idx_unique_0:1
        is_the_first = idx_unique[1:] - idx_unique[0:idx_unique.shape[0]-1]
        is_the_first = tf.concat( [tf.constant([1],tf.int32),is_the_first],0, name='is_the_first_%d'%(i) ) # gpu_0/sa_layer4/is_the_first_0:0
        first_unique_mask = tf.equal( is_the_first, 1, name='first_unique_mask_%d'%(i) ) # gpu_0/sa_layer4/first_unique_mask_0:0
        first_unique_masks.append( tf.expand_dims(first_unique_mask,0) )
    first_unique_masks = tf.concat( first_unique_masks, 0)
    first_unique_masks = tf.reshape( first_unique_masks, org_shape[0:3], name='first_unique_masks' )
    # set all the replicated items as -9999
    first_unique_masks = tf.expand_dims( first_unique_masks,-1 )
    first_unique_masks = tf.tile( first_unique_masks, [1,1,1,3] )
    output = tf.where( first_unique_masks, org_inputs, tf.ones(org_shape,tf.float32)*(-99), name='uni_out' ) # gpu_0/sa_layer4/uni_out:0
    return output, first_unique_masks

################################################################################
# Convenience functions for building the ResNet model.
################################################################################


def conv3d_fixed_padding(inputs, filters, kernel_size, strides, padding, data_format):
  """Strided 3-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv3d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=padding, use_bias=False,
      kernel_initializer=KERNEL_INI,
      data_format=data_format)


class ResConvOps(object):
  ''' Basic convolution operations '''
  _block_layers_num = 0
  _conv1d_num = 0
  _conv2d_num = 0
  _conv3d_num = 0
  _dense_num = 0
  _inception_block_layer = 0
  IsShowModel = False
  _epoch = 0
  _trainable_num = 0
  _trainable_bytes = 0
  _activation_size = 0
  _padding = {'s':'same', 'v':'valid' }

  def __init__(self, data_net_configs, data_format):
    self.data_format = data_format
    self.data_net_configs= data_net_configs
    self.residual = data_net_configs['residual']
    self.voxel3d = 'V' in data_net_configs['model_flag']
    train_global_step = tf.train.get_or_create_global_step()
    self.batch_norm_decay = data_net_configs['bndecay_fn'](train_global_step)
    tf.summary.scalar('batch_norm_decay', self.batch_norm_decay)
    self.use_bias = data_net_configs['use_bias']
    self.shortcut_method = data_net_configs['shortcut'] #'C' 'PC' 'PZ'
    self.res_scale = 1.0

    model_dir = data_net_configs['model_dir']
    if ResConvOps._epoch==0:
      self.IsShowModel = True
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
      self.model_log_fn = os.path.join(model_dir, 'log_model.txt')
      self.model_log_f = open(self.model_log_fn, 'w')

    ResConvOps._epoch += 1

  def log_model_summary(self):
      dnc = self.data_net_configs

      #*************************************************************************
      # key training parameters
      res = 'R' if self.residual else 'P'
      use_xyz_str = '' if dnc['use_xyz'] else 'Np'
      key_para_names = 'model bs aug feed drop_imo loss_weight lr0_drate_depoch \
bnd optimizer block_config\n'
      key_paras_str = '\n\n{model_name} {bs} {aug} {feed_data_eles} \
{drop_imo} {loss_weight} {lr0}_{lr_decay_rate}_{lr_decay_epochs} {bnd} {optimizer} par \
{block_config}\n'.format(
        model_name=dnc['model_name'],
        bs=dnc['batch_size'],
        feed_data_eles=dnc['feed_data_eles'].replace('nxnynz','n') + use_xyz_str,
        aug=dnc['aug_types'],
        drop_imo=dnc['drop_imo_str'],
        loss_weight=dnc['loss_lw_gama'] if dnc['loss_lw_gama']>0 else 'No',
        lr0=dnc['learning_rate0'],
        lr_decay_rate=dnc['lr_decay_rate'],
        lr_decay_epochs=dnc['lr_decay_epochs'],
        bnd=dnc['batch_norm_decay0'],
        optimizer=dnc['optimizer'][0:3],
        block_config=dnc['block_config_str']
        )

      self.key_paras_str = key_para_names + key_paras_str

      items_to_write = ['model_flag', 'block_config_str',  'aug_types', 'drop_imo', \
        'feed_data', 'xyz_elements', 'num_gpus', 'batch_size',\
        'optimizer',\
        'learning_rate0', 'lr_decay_rate', 'batch_norm_decay0', 'lr_vals', \
        'bndecay_vals', 'use_bias', 'shortcut',\
        'weight_decay', 'resnet_size', 'data_dir', 'label_num_weights']
      for item in items_to_write:
        str_i = '%s:%s\n'%(item, dnc[item])
        self.model_log_f.write(str_i)
        if self.IsShowModel:
          print(str_i)

      #*************************************************************************
      # write network block params
      self.model_log_f.write('\nBlock parameters:\n')
      block_params = dnc['block_params']
      for ele in block_params:
        if ele == 'icp_block_ops': continue
        str_i ='%s:%s\n'%(ele,block_params[ele])
        self.model_log_f.write(str_i)
        if self.IsShowModel:
          print(str_i)
      self.model_log_f.write('\n--------------------------------------------\n')

      #*************************************************************************
      # sampling grouping settings
      self.model_log_f.write(dnc['sg_settings']['sg_str'])
      self.model_log_f.write('\n--------------------------------------------\n')

      self.model_log_f.write('\n--------------------------------------------\n')
      self.model_log_f.write(self.key_paras_str)

      self.model_log_f.flush()

  def train_w_bytes(self, scope=None):
    trainable_variables = tf.trainable_variables(scope)
    weight_num = len(trainable_variables)
    weight_bytes = np.sum([np.prod(v.get_shape().as_list()) * v.dtype.size \
                          for v in trainable_variables])
    if scope!=None: # assume it is a unique scope
      self._trainable_num += weight_num
      self._trainable_bytes += weight_bytes

    weight_shapes = [np.array(v.get_shape().as_list()) \
                          for v in trainable_variables]
    conv_shapes = [shape for shape in weight_shapes if shape.size>=3]
    return weight_num, weight_bytes, conv_shapes

  def add_activation_size(self, activation):
    self._activation_size += np.prod(activation.shape.as_list()[1:]) \
                             * activation.dtype.size

  def batch_norm(self, inputs, training, activation):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    epsilon = _BATCH_NORM_EPSILON
    fused = True
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if self.data_format == 'channels_first' else -1,
        momentum=self.batch_norm_decay , epsilon=epsilon, center=True,
        scale=True, training=training, fused=fused)

    if activation!=None:
      inputs = activation(inputs)
      if self.IsShowModel:  self.log('%30s'%('BN RELU'))
    else:
      if self.IsShowModel:  self.log('%30s'%('BN'))

    return inputs

  def log(self, log_str):
    self.model_log_f.write(log_str+'\n')
    self.model_log_f.flush()
    print(log_str)

  def log_tensor_c(self, inputs, kernels, strides, padding_s1, var_scope,
                   pre_indent=''):
      if not self.IsShowModel: return
      self.add_activation_size(inputs)
      if len(inputs.shape)==4 :
        conv_str = 'conv2d'
      elif len(inputs.shape)==3:
        conv_str = 'conv1d'
      elif len(inputs.shape)==5:
        conv_str = 'conv3d'
      layer_name = pre_indent+'/'.join(var_scope.split('/')[2:])
      self.log( tensor_info(inputs, '%s (%d,%d,%s)'%
                    (conv_str, kernels, strides, padding_s1), layer_name,
                    self.train_w_bytes(var_scope), self.batch_size) )

  def log_tensor_p(self, inputs, pool_op_name, layer_name,
                   kernels=None, strides=None, paddings=None):
      if not self.IsShowModel: return
      self.add_activation_size(inputs)
      pool_str = pool_op_name
      if kernels!=None:
        pool_str += ' (%d,%d,%s)'%(kernels, strides, paddings)
      self.log(tensor_info(inputs, pool_str, layer_name,
                           batch_size=self.batch_size))

  def show_layers_num_summary(self):
    self.log('block layers num:{}\nconv2d num:{}\nconv3d num:{}\nconv1d num:{}\ndense num:{}'.format(
          self._block_layers_num, self._conv2d_num, self._conv3d_num, self._conv1d_num, self._dense_num))
    if self._inception_block_layer > 0:
      self.log('inception block layer:{}\n'.format(self._inception_block_layer))

  def get_feature_shape(self, net):
    if len(net.shape)==4:
      if self.data_format == 'channels_last':
        shape = net.shape.as_list()[1:3]
    elif len(net.shape)==5:
      if self.data_format == 'channels_last':
        shape = net.shape.as_list()[1:4]
    elif len(net.shape)==3:
      if self.data_format == 'channels_last':
        shape = net.shape.as_list()[1]
    else:
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      raise NotImplementedError
    return np.array(shape)

  def get_feature_channels(self, net):
    if self.data_format == 'channels_last':
      return net.shape[-1].value
    elif self.data_format == 'channels_first':
      return net.shape[1].value

  def conv2d3d(self, inputs, filters, kernels, strides, padding_s1):
    """Strided 2-D or 3-D convolution with explicit padding.
      padding_s1:  only used when strides==1
    """
    if len(inputs.shape)==5:
      conv_fn = tf.layers.conv3d
      self._conv3d_num += 1
    elif len(inputs.shape) == 4:
      conv_fn = tf.layers.conv2d
      self._conv2d_num += 1
    else:
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
      raise NotImplementedError

    inputs, padding = self.padding2d3d(inputs, kernels, strides, padding_s1)

    assert self.data_format == 'channels_last'
    outputs = conv_fn(
        inputs=inputs, filters=filters, kernel_size=kernels, strides=strides,
        padding=padding, use_bias=self.use_bias,
        kernel_initializer=KERNEL_INI,
        data_format=self.data_format)
    return outputs

  def conv1d(self, inputs, filters, kernels, strides, padding):
    self._conv1d_num += 1
    outputs = tf.layers.conv1d(
        inputs=inputs, filters=filters, kernel_size=kernels, strides=strides,
        padding=padding, use_bias=self.use_bias,
        kernel_initializer=KERNEL_INI,
        data_format=self.data_format)
    return outputs

  def fixed_padding_2d3d(self, inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                  Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if self.data_format == 'channels_first':
      if len(inputs.shape)==4:
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
      elif len(inputs.shape)==5:
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                      [pad_beg, pad_end], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
      if len(inputs.shape)==4:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
      elif len(inputs.shape)==5:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                              [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs

  def padding_channel(self, inputs, pad_total):
    assert pad_total > 0
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if self.data_format == 'channels_first' and len(inputs.shape)==4:
      padded_inputs = tf.pad(inputs, [[0,0],[pad_beg,pad_end],[0,0],[0,0]])

    elif self.data_format == 'channels_first' and len(inputs.shape)==5:
      padded_inputs = tf.pad(inputs, [[0,0],[pad_beg,pad_end],[0,0],[0,0],[0,0]])

    elif self.data_format == 'channels_last' and len(inputs.shape)==4:
      padded_inputs = tf.pad(inputs, [[0,0],[0,0],[0,0],[pad_beg,pad_end]])

    elif self.data_format == 'channels_last' and len(inputs.shape)==5:
      padded_inputs = tf.pad(inputs, [[0,0],[0,0],[0,0],[0,0],[pad_beg,pad_end]])
    return padded_inputs

  def padding2d3d(self, inputs, kernels, strides, padding_s1):
    '''
    When strides==1, padding = 'v' is used to reduce feature map,
    When strides>1, fixed padding is used to keep reduce rate equal to strides.
    '''
    padding = self._padding[padding_s1]
    if strides > 1:
      assert padding == 'valid'
      inputs = self.fixed_padding_2d3d(inputs, kernels)
    if padding == 'same':
      self.check_kernel_waste(inputs, kernels, True)
    return inputs, padding

  def check_kernel_waste(self, inputs, kernels, warning=False):
    # padding == 'same'
    if kernels==1:
      return False
    in_shape = self.get_feature_shape(inputs)
    pad_waste = kernels-1
    pad_waste_r = pad_waste / in_shape
    is_waste =  (kernels == 2 and (in_shape < 3).any())  or\
                (kernels == 3 and (in_shape < 5).any())  or \
                (kernels >= 4 and (pad_waste_r > 0.3).any())
    info = 'kernel waste:%s, inputs shape:%s kernels:%d'%(is_waste, in_shape, kernels)
    if is_waste:
      if warning:
        assert False, info
    return is_waste

  def pool2d3d(self, inputs, pool, kernels, strides, padding_s1):
    if len(inputs.shape)==5:
      if pool == 'max':
        pool_fn = tf.layers.max_pooling3d
      elif pool == 'ave':
        pool_fn = tf.layers.average_pooling3d
    elif len(inputs.shape) == 4:
      if pool == 'max':
        pool_fn = tf.layers.max_pooling2d
      elif pool == 'ave':
        pool_fn = tf.layers.average_pooling2d

    inputs, padding = self.padding2d3d(inputs, kernels, strides, padding_s1)
    outputs = pool_fn(inputs, kernels, strides, padding, self.data_format)
    return outputs

  def feature_uncompress_block(self, inputs, feature_rate, uncompress_times):
    assert self.data_format == 'channels_last'
    with tf.variable_scope('fu'):
      for i in range(uncompress_times):
        #filters_i = int(self.get_feature_channels(inputs)*2) - 3*self.use_xyz
        filters_i = int(self.get_feature_channels(inputs)*2)
        inputs = self.conv2d3d(inputs, filters_i, 1,1,'s')
        self.log_tensor_c(inputs, 1,1,'s', tf.get_variable_scope().name)
    self.block_num_count += 1
    return inputs


  def building_block_v2(self, inputs, block_params, training, projection_shortcut,
                        half_layer=None, no_ini_bn=False):
    """A single block for ResNet v2, without a bottleneck.

    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block; shape should match inputs.
    """
    filters = block_params['filters']
    kernels = block_params['kernels']
    strides = block_params['strides']
    padding_s1 = block_params['padding_s1']

    shortcut = inputs
    if not no_ini_bn:
      inputs = self.batch_norm(inputs, training, tf.nn.relu)
    if projection_shortcut == 'FirstResUnit':
      # For pointnet, projection shortcut is not needed at the First ResUnit.
      # However, BN and Activation is still required at the First ResUnit for
      # pre-activation.
      shortcut = inputs
      projection_shortcut = None
      if self.IsShowModel:  self.log(
            'shortcut after activation identity for pointnet first res unit')
    if half_layer:
      projection_shortcut = None

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
      shortcut = projection_shortcut(inputs)

    with tf.variable_scope('c0'):
      inputs = self.conv2d3d(inputs, filters, kernels, strides, padding_s1)
      self.log_tensor_c(inputs, kernels, strides, padding_s1,
                        tf.get_variable_scope().name)
    if half_layer: return inputs
    inputs = self.batch_norm(inputs, training, tf.nn.relu)

    with tf.variable_scope('c1'):
      if self.check_kernel_waste(inputs, kernels):
        kernels = 1
      inputs = self.conv2d3d(inputs, filters, kernels, 1, 's')
      self.log_tensor_c(inputs, kernels, 1, 's',
                        tf.get_variable_scope().name)

    if self.residual:
      assert inputs.shape == shortcut.shape
      if self.IsShowModel: self.log('Add shortcut*%0.1f'%(self.res_scale))
      return inputs * self.res_scale + shortcut
    else:
      return inputs

  def bottleneck_block_v2(self, inputs, block_params, training, projection_shortcut,
                          half_layer=None, no_ini_bn=False):
    """A single block for ResNet v2, with a bottleneck.

    Similar to building_block_v2(), except using the "bottleneck" blocks
    described in:
      Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    Adapted to the ordering conventions of:
      Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block; shape should match inputs.
    """
    filters = block_params['filters']
    kernels = block_params['kernels']
    strides = block_params['strides']
    padding_s1 = block_params['padding_s1']

    shortcut = inputs
    if not no_ini_bn:
      inputs = self.batch_norm(inputs, training, tf.nn.relu)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
      shortcut = projection_shortcut(inputs)

    with tf.variable_scope('c0'):
      inputs = self.conv2d3d(inputs, filters//4, 1, 1, 's')
      self.log_tensor_c(inputs, 1, 1, 's', tf.get_variable_scope().name)

    inputs = self.batch_norm(inputs, training, tf.nn.relu)

    with tf.variable_scope('c1'):
      inputs = self.conv2d3d(inputs, filters//4, kernels, strides, padding_s1)
      self.log_tensor_c(inputs, kernels, strides, padding_s1,
                        tf.get_variable_scope().name)

    inputs = self.batch_norm(inputs, training, tf.nn.relu)

    with tf.variable_scope('c2'):
      inputs = self.conv2d3d(inputs, filters, 1, 1, 's')
      self.log_tensor_c(inputs, 1, 1, 's', tf.get_variable_scope().name)

    if self.residual:
      if not inputs.shape == shortcut.shape:
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

      if self.IsShowModel: self.log('Add shortcut*%0.1f'%(self.res_scale))
      return inputs * self.res_scale + shortcut
    else:
      return inputs

  def operation(self, op, net, pre_indent):
    ''' general operation entry

    Args:
      op: ['conv',filters,kernels,strides,padding_s1]
          ['max'/'ave',kernel_size,strides,padding_s1]
    '''
    var_scope = tf.get_variable_scope().name
    if op[0] == 'conv':
      net = self.conv2d3d(net, op[1], op[2], op[3], op[4])
      self.log_tensor_c(net, op[2], op[3], op[4], var_scope, pre_indent)
    elif op[0] == 'max':
      net = self.pool2d3d(net, 'max', op[1], op[2], op[3])
      layer_name = pre_indent+'/'.join(var_scope.split('/')[2:])
      self.log_tensor_p(net, 'max', layer_name)
    else:
      raise NotImplementedError
    return net

  def inception_block_v2(self, inputs, block_params, training,
                         projection_shortcut, half_layer=None, no_ini_bn=False):
    """A single block for ResNet v2, with inception structure


    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    if not no_ini_bn:
      inputs = self.batch_norm(inputs, training, tf.nn.relu)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
      shortcut = projection_shortcut(inputs)

    icp_block_ops = block_params['icp_block_ops'](
      self.get_feature_shape(inputs)[0], self.get_feature_channels(inputs),
      block_params['filters'])
    inputs_branches = []
    for b, ops_branch in enumerate(icp_block_ops):
      for l,op in enumerate(ops_branch):
        with tf.variable_scope('b%d_l%d'%(b,l)):
          pre_indent = '  ' if l==len(ops_branch)-1 else '    '
          if l==0:
            inputs_b = inputs
          else:
            inputs_b = self.batch_norm(inputs_b, training, tf.nn.relu)
          inputs_b = self.operation(op, inputs_b, pre_indent)
      inputs_branches.append(inputs_b)
    self._inception_block_layer += max([len(ops_branch) for ops_branch in icp_block_ops])

    c_axis = -1 if self.data_format == 'channels_last' else 1
    inputs = tf.concat(inputs_branches, c_axis)
    layer_name = '/'.join(tf.get_variable_scope().name.split('/')[2:])
    self.log_tensor_p(inputs, 'inception concat', layer_name)

    inputs = self.conv2d3d(inputs, block_params['filters'], 1, 1, 's')
    self.log_tensor_c(inputs, 1, 1, 's', tf.get_variable_scope().name)

    if self.residual:
      if not inputs.shape == shortcut.shape:
        if NoRes_InceptionReduction:
          return inputs
        else:
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
      else:
        if self.IsShowModel: self.log('Add shortcut*%0.1f'%(self.res_scale))
        return inputs * self.res_scale + shortcut
    else:
      return inputs

  def shortcut_fn(self, inputs, block_params):
    ''' shortcut functions when feature map size decrease or channel increase
    (1) Four methods for reducing feature map size:
      (a) Conv+Padding (b) Pool+Padding  => Kernel > 1, Padding='v'
      (c) Conv+Stride (d) Pool+Stride    => Stride > 1
      Stride>1 is commonly used by 2D, but not used here, because feature map
      is too small. Focus on a,b
    (2) Tow methos for increasing channels:
      (i) conv (j) zero padding
    (3) shortcut: C, MC, AC, MZ, AZ
      Candidate procedure for reducing feature map and increasing channels:
      ['C'] Conv(kernel>1) + Padding('v') #Large model
      ['MC'/'AC'] Max/Ave Pool(Kernle>1) + Padding('v') => 1x1 Conv # Small weight
      ['MZ'/'AZ'] Max/Ave Pool(Kernel>1) + Padding('v') => Channel zero padding # No weight
    '''
    scm = self.shortcut_method
    filters_out_sc = block_params['filters']
    kernels = block_params['kernels']
    strides = block_params['strides']
    padding_s1 = block_params['padding_s1']

    need_projection = self.get_feature_channels(inputs) != filters_out_sc or\
          (padding_s1=='v' and kernels>1) or strides>1
    if not need_projection:
      return inputs

    # In offical resnet, kernel_sc is always 1, because strides>1 reduces map.
    # But kernel_sc>1 here to reduce map, and strides is always 1.
    kernel_sc = kernels if padding_s1=='v' else 1
    if scm == 'C':
      # padding_s1=='v: feature map size have to be reduced => kernels
      with tf.variable_scope('sc_C'):
        shortcut = self.conv2d3d(inputs, filters_out_sc, kernel_sc, strides,
                                 padding_s1)
        self.log_tensor_c(shortcut, kernel_sc, strides,
                        padding_s1, tf.get_variable_scope().name)

    elif scm == 'MC' or scm == 'AC' or scm == 'MZ' or scm == 'AZ':
      with tf.variable_scope('sc_'+scm):
        layer_name = '/'.join(tf.get_variable_scope().name.split('/')[2:])
        if kernel_sc!=1:
          pool = 'max' if scm[0]=='M' else 'ave'
          shortcut = self.pool2d3d(inputs, pool, kernel_sc, strides, padding_s1)
          self.log_tensor_p(shortcut, pool, layer_name, kernel_sc, strides, padding_s1)
        else:
          shortcut = inputs

        channels_dif = filters_out_sc - self.get_feature_channels(shortcut)
        if channels_dif != 0:
          if scm[1] == 'C':
            shortcut = self.conv2d3d(shortcut, filters_out_sc, 1, 1, 's')
            self.log_tensor_c(shortcut, 1, 1, 's', tf.get_variable_scope().name)
          else:
            #assert channels_dif>0
            if channels_dif<0:
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
            shortcut = self.padding_channel(shortcut, channels_dif)
            self.log_tensor_p(shortcut, 'padding', layer_name)

    else:
      raise NotImplementedError, scm
    return shortcut

  def block_layer(self, scale, inputs, block_params, block_fn, is_training, name):
    """Creates one layer of block_size for the ResNet model.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      block_size: The number of block_size contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      is_training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block layer.
    """
    block_size = block_params['block_sizes']
    filters = block_params['filters']

    if block_size==0: return inputs
    half_layer = block_size==0.5
    if half_layer:
      block_size = 1
    # Bottleneck block_size end with 4x the number of filters as they start with
    bottleneck = block_fn == self.bottleneck_block_v2

    def shortcut_projection(inputs):
      return self.shortcut_fn(inputs, block_params)

    # (1) Only the first block per block_layer uses projection_shortcut and strides
    # and padding_s1
    # (2) No need to change map size and channels in first unit if Pointnet
    if self._block_layers_num == 0 and filters==inputs.shape[0].value:
        projection_shortcut_0 = 'FirstResUnit'
    else:
      projection_shortcut_0 = shortcut_projection
      if scale!=0 and self.block_style == 'Inception' and NoRes_InceptionReduction:
        projection_shortcut_0 = None
    if not self.residual:
      projection_shortcut_0 = None
    with tf.variable_scope('L0'):
      no_ini_bn = scale==1
      inputs = block_fn(inputs, block_params, is_training, projection_shortcut_0,
                        half_layer, no_ini_bn=no_ini_bn)

    block_params['strides'] = 1
    block_params['padding_s1'] = 's'
    for j in range(1, block_size):
      with tf.variable_scope('L%d'%(j)):
        inputs = block_fn(inputs, block_params, is_training, None)

    self._block_layers_num += 1
    return tf.identity(inputs, name)


  def feature_back_propagate(self, inputs):
    outputs = tf.layers.conv2d(inputs, 128, 1, 1, 'v')
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return outputs


def mytile(tensor, axis, eval_views):
  org_shape = tensor.shape.as_list()
  tensor = tf.expand_dims(tensor, axis)
  multiples = np.ones(len(tensor.shape))
  multiples[axis] = eval_views
  tensor = tf.tile(tensor, multiples)
  org_shape[0]=-1
  tensor = tf.reshape(tensor, org_shape)
  return tensor


def my_reduce_mean(grouped_center):
  '''
  reduce mean exclusive of 0.
  grouped_center mayu include 0 that should not be included by mean
  grouped_center: (b,n1.n2,3)
  mean_xyz:(b,n1,3)
  '''
  sum_xyz = tf.reduce_sum(grouped_center, -2)
  tmp = tf.reduce_mean(grouped_center,-1)
  tmp = tf.cast(tf.not_equal(tmp, 0),tf.float32)
  valid_num = tf.reduce_sum(tmp, -1)
  valid_num = tf.expand_dims(valid_num, -1)
  mean_xyz = sum_xyz / valid_num
  return mean_xyz


def pc_normalize(points):
  has_normal = points.shape[-1].value == 6
  points_xyz = points[:,0:3]
  if has_normal:
    points_normal = points[:,3:6]
  centroid = tf.reduce_mean(points_xyz, axis=0)
  points_xyz -= centroid
  m = tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.pow(points_xyz, 2),axis=1)))
  points_xyz = points_xyz / m
  if has_normal:
    points_normed = tf.concat([points_xyz, points_normal], -1)
  else:
    points_normed = points_xyz
  return points_normed


class Model(ResConvOps):
  """Base class for building the Resnet Model."""

  def __init__(self, model_flag, resnet_size, block_style,
               block_params, resnet_version=DEFAULT_VERSION,
               data_format=None, dtype=DEFAULT_DTYPE, data_net_configs={}):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      block_style: Regular, Bottleneck, Inception
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block
        layer.
      kernel_size: The kernel size to use for convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer.
        If none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used
        if first_pool_size is None.
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.

    Raises:
      ValueError: if invalid version is selected.
    """
    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    super(Model, self).__init__(data_net_configs, data_format)
    self.model_flag = model_flag
    self.resnet_size = resnet_size
    self.batch_size = data_net_configs['batch_size']//data_net_configs['num_gpus']
    self.num_gpus = data_net_configs['num_gpus']

    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
          'Resnet version should be 1 or 2. See README for citations.')

    self.block_style = block_style
    self.max_filters = {'Regular':1024, 'Bottleneck':512, 'Inception':1024}
    if resnet_version == 1:
        raise NotImplementedError
    else:
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

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.block_params = block_params
    self.dtype = dtype
    self.pre_activation = resnet_version == 2
    self.data_net_configs = data_net_configs
    self.block_num_count = 0

    self._preprocess_configs()

    if self.task == 'segmentation':
      self.end_points = []

  def _preprocess_configs(self):
    if '_' in self.model_flag:
        self.model_flag, num_neighbors = modelf_nein.split('_')
        self.num_neighbors = np.array( [ int(n) for n in num_neighbors ] )
    else:
        self.num_neighbors= None
    for key in ['dataset_name', 'feed_data', 'sg_settings', 'dset_shape_idxs',
                'xyz_elements', 'datasets_meta']:
      setattr(self, key, self.data_net_configs[key])

    tasks = {'MODELNET40': 'recognition', 'MATTERPORT': 'segmentation'}
    self.task = tasks[self.dataset_name]
    self.global_numpoint = self.dset_shape_idxs['shape']['points'][0]
    self.net_num_scale = len(self.data_net_configs['block_params']['filters'])
    self.sg_num_scale = len(self.data_net_configs['sg_settings']['width'])
    assert self.sg_num_scale == self.net_num_scale
    self.num_classes = self.datasets_meta.num_classes

    self.data_idxs = self.dset_shape_idxs['indices']
    assert self.data_idxs['points']['xyz'] == [0,1,2]
    for e in self.feed_data:
      assert e in self.data_idxs['points']
    IsAllInputs = len(self.data_idxs['points']) == len(self.feed_data)
    if IsAllInputs:
      self.feed_data_idxs = 'ALL'
    else:
      self.feed_data_idxs = np.sort([i for e in self.feed_data for i in self.data_idxs['points'][e] ])

    self.use_xyz = self.data_net_configs['use_xyz']
    self.mean_grouping_position = True

    from utils.grouping_sampling_voxelization import BlockGroupSampling
    log_path = self.data_net_configs['data_dir']+'/sg_log'
    self.bsg = BlockGroupSampling(self.sg_settings, log_path)


  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.

    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.

    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.

    Returns:
      A variable scope for the model.
    """

    return tf.variable_scope('resnet_model',
                             custom_getter=self._custom_dtype_getter)


  def sg_in_inputpipeline(self, inputs_dic, sg_num_scale):
    # scale 0 is global scale!
    inputs_dic1 = {}

    items = ['grouped_pindex','grouped_bot_cen_top', 'empty_mask', 'bot_cen_top',
             'vox_index', 'flatting_idx', 'flat_valid_mask']
    for item in items:
      inputs_dic1[item] = []
      for s in range(0, sg_num_scale+1):
        if item+'_%d'%(s) in inputs_dic:
          value = inputs_dic[item+'_%d'%(s)]
          inputs_dic1[item].append(value)

    points = inputs_dic['points']
    ps = [e.value for e in points.shape]
    if len(ps) == 4:
      assert self.batch_size == ps[0]
      self.batch_size = ps[0] * ps[1]
      # include global block dim, fuse to batch dim
      points = tf.reshape(points, [self.batch_size, ps[2], ps[3]])

    return self.pre_pro_inputs(inputs_dic1, points)

  def pre_sampling_grouping(self, inputs_dic):
    # get the indices for grouping and sampling on line

    #t0 = tf.timestamp()
    points = inputs_dic['points']

    dsb = {}
    dsb['grouped_pindex'], dsb['vox_index'], dsb['grouped_bot_cen_top'], \
      dsb['empty_mask'], dsb['bot_cen_top'], nblock_valid, others =\
      self.bsg.grouping_multi_scale(points[..., 0:3])
    return self.pre_pro_inputs(dsb, points)

  def pre_pro_inputs(self, dsb, points):
    assert self.net_num_scale  == len(dsb['grouped_bot_cen_top'])
    dsb['grouped_center'] = []
    for s in range(len(dsb['grouped_bot_cen_top'])):
      dsb['grouped_center'].append([])
      dsb['grouped_center'][s] = dsb['grouped_bot_cen_top'][s][...,3:6]
    del dsb['grouped_bot_cen_top']

    if self.feed_data_idxs!='ALL':
      points = tf.gather(points, self.feed_data_idxs, axis=-1)
    #*************************************************************
    is_show = False
    def show_shapes(state=''):
      print('\n\n{}\n'.format(state))
      for item in dsb:
        item_shapes = []
        if item == 'points':
          item_ls = [dsb[item]]
        else:
          item_ls = dsb[item]
        for s in item_ls:
          if s!=[]:
            item_shapes.append([e.value for e in s.shape])
          else:
            item_shapes.append([])
        print("{}: {}".format(item, item_shapes))
    if is_show: show_shapes('orginal')

    # rm global scale
    for item in dsb:
      if self.net_num_scale == 1 and item in ['grouped_center', 'bot_cen_top']:
        dsb[item][0] = tf.expand_dims(tf.squeeze(dsb[item][0], 1), 2)
      else:
        del dsb[item][0]

      if len(dsb[item]) < self.net_num_scale:
        if len(dsb[item]) == 0:  # one scale only: pointnet
          dsb[item].append([])
        else:
          dsb[item].append(tf.zeros([0]*len(dsb[item][0].shape), dsb[item][0].dtype))

    if is_show: show_shapes('after deleting global scale')
    #*************************************************************
    # merge batch size dim and global block dim
    for item in dsb:
      num_scale = len(dsb[item])
      if item in ['flatting_idx', 'flat_valid_mask']:
        for s in range(num_scale):
          if dsb[item][s]==[]: continue
          shape0 = [e.value for e in dsb[item][s].shape]
          assert len(shape0) == 3
          if shape0[0] != self.batch_size:
            dsb[item][s] = tf.reshape(dsb[item][s], [self.batch_size, -1, shape0[-1]])

      else:
        for s in range(num_scale):
          if dsb[item][s] == []: # one scale only: pointnet
            continue
          shape_i = [e.value for e in dsb[item][s].shape]
          dsb[item][s] = tf.reshape(dsb[item][s], [self.batch_size] + shape_i[2:])
          shape_i1 = [e.value for e in dsb[item][s].shape]

    #*************************************************************
    dsb['points'] = points

    if is_show: show_shapes('after merging batch and global block dims')
    return dsb

  def __call__(self, inputs_dic, is_training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      is_training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    #t00 = tf.timestamp()
    if not self.data_net_configs['precpu_sg']:
      inputs_dic = self.pre_sampling_grouping(inputs_dic)
    else:
      inputs_dic = self.sg_in_inputpipeline(inputs_dic, self.sg_num_scale)

    IsMultiView = len(inputs_dic['points'].shape) == 4
    if not IsMultiView:
      assert len(inputs_dic['points'].shape) == 3
      outputs = self._call(
            inputs_dic['points'],
            inputs_dic['grouped_center'],
            inputs_dic['grouped_pindex'],
            inputs_dic['empty_mask'],
            inputs_dic['bot_cen_top'],
            inputs_dic['vox_index'],
            inputs_dic['flatting_idx'],
            inputs_dic['flat_valid_mask'],
            is_training)
    else:
      raise NotImplementedError
      eval_views = inputs_dic['points'].shape[1].value
      batch_size = inputs_dic['points'].shape[0].value
      if batch_size==None:
        batch_size  = -1
      b_bottom_centers_mm = {}
      sg_bidxmaps = {}
      net_num_scale = len(inputs_dic['b_bottom_centers_mm'])
      for c in range(net_num_scale):
        #b_bottom_centers_mm[c] = inputs_dic['b_bottom_centers_mm'][c][:,v,:,:]

        tmp = inputs_dic['b_bottom_centers_mm'][c]
        if tmp!=[]:
          b_bottom_centers_mm[c] = tf.reshape(tmp, [-1]+tmp.shape[2:4].as_list())
          sg_bidxmaps[c] = mytile(inputs_dic['sg_bidxmaps'][c], 1, eval_views)
        else: # single scale pointnet
          b_bottom_centers_mm[c] = []
          sg_bidxmaps[c] = []

      points = tf.reshape(inputs_dic['points'], [-1]+inputs_dic['points'].shape[2:4].as_list())
      if inputs_dic['bidxmaps_flat'][0]!=[]:
        bidxmaps_flat = mytile(inputs_dic['bidxmaps_flat'], 1, eval_views)
        fmap_neighbor_idis = mytile(inputs_dic['fmap_neighbor_idis'], 1, eval_views)
      else:
        bidxmaps_flat = []
        fmap_neighbor_idis = []

      outputs = self._call(
            points,
            sg_bidxmaps,
            b_bottom_centers_mm,
            bidxmaps_flat,
            fmap_neighbor_idis,
            is_training,
            eval_views=eval_views)
      outputs = tf.reshape(outputs, [batch_size, eval_views, outputs.shape[-1].value])

    if self.IsShowModel:
      self.model_log_f.close()
    #tf.summary.scalar('t_batch', (tf.timestamp() - t00)*1000 )
    return outputs

  def _call(self, inputs, grouped_center_ms, grouped_pindex_ms, empty_mask_ms, \
            bot_cen_top_ms, vox_index_ms, flatting_idx, flat_valid_mask, \
            is_training, eval_views=-1):
    '''
    inputs: (12, 4096, 3)
    grouped_center_ms: [(12, 1, 4096, 3), (12, 1024, 32, 3), (12, 48, 48, 3)]
    grouped_pindex_ms: [(12, 1, 4096), (12, 1024, 32), (12, 48, 48)]
    empty_mask_ms: [(12, 1, 4096), (12, 1024, 32), (12, 48, 48), (12, 1, 48)]
    bot_cen_top_ms: [(12, 1, 9), (12, 1024, 9), (12, 48, 9)]
    vox_index_ms: [[], [], (12, 48, 48, 3), (12, 1, 48, 3)]
    '''
    #***************************************************************************
    # check input shapes
    assert len(inputs.shape) == 3
    for s in range(self.sg_num_scale):
      if grouped_center_ms[s]!=[]:
        assert len(grouped_center_ms[s].shape) == 4
      if grouped_pindex_ms[s]!=[]:
        assert len(grouped_pindex_ms[s].shape) == 3
      if empty_mask_ms[s]!=[]:
        assert len(empty_mask_ms[s].shape) == 3
      if bot_cen_top_ms[s]!=[]:
        assert len(bot_cen_top_ms[s].shape) == 3
      if s==0:
        assert vox_index_ms[s] == [] or vox_index_ms[s].shape[-1].value==0
      else:
        assert len(vox_index_ms[s].shape) == 4
    #***************************************************************************
    tf.add_to_collection('raw_inputs', inputs)
    if self.IsShowModel: self.log('')
    self.log_tensor_p(inputs, 'raw_inputs', '\nOriginal')
    self.is_training = is_training

    with self._model_variable_scope():
      l_points = []                       # size = l_points+1
      l_xyz = inputs[...,0:3]

      #l_xyz, b_bottom_centers_mm = pc_normalize(l_xyz, b_bottom_centers_mm)

      new_points = inputs

      scales_feature = []

      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        raise NotImplementedError
        new_points = tf.transpose(new_points, [0, 2, 1])

      ##########################################################################
      #                       Point encoder
      for k in range(self.net_num_scale):
          new_points, root_point_features = self.res_sa_module(k,
                        new_points, grouped_center_ms[k], grouped_pindex_ms[k],
                        empty_mask_ms[k], bot_cen_top_ms[k], vox_index_ms[k] )

          # Last output of point encoder: Only apply the BN and ReLU for model that does pre_activation in each
          # building/bottleneck block, eg resnet V2.
          if k==self.net_num_scale-1 and self.block_style != 'PointNet' and self.pre_activation:
              new_points = self.batch_norm(new_points, is_training, tf.nn.relu)

          if k == 0:
              l_points.append(self.ave_grouped_features(root_point_features))
          l_points.append(new_points)
          if self.IsShowModel: self.log(
            '*****************************************************************')

      # use reduce_mean to reduce feature shape to 1 if not already
      if self.get_feature_shape(new_points)!=1:
        axis = [2] if self.data_format == 'channels_first' else [1]
        new_points = tf.reduce_mean(new_points, axis, keepdims=True)
        if self.IsShowModel: self.log( tensor_info(new_points, 'reduce_mean', 'final') )

      ##########################################################################
      if self.dataset_name == 'MODELNET40':
        new_points = tf.squeeze(new_points, 1)
        new_points = self.classifier(new_points, is_training,\
                          dense_features = self.block_params['dense_filters']['final_dense_with_dp'])
      if self.dataset_name == 'MATTERPORT':
        new_points = self.segmentation(l_points, flatting_idx, flat_valid_mask, is_training)
        new_points = self.classifier(new_points, is_training,\
                          dense_features = self.block_params['dense_filters']['final_dense_with_dp'])
      ##########################################################################
      if self.IsShowModel:
        self.log( tensor_info(new_points, 'dense', 'final1') +'\n\n' )
        self.show_layers_num_summary()

        total_w_num, total_w_bytes, train_w_shapes = self.train_w_bytes()
        self.log('Total trainable weights: (%d %0.3fM)  Counted (%d %0.3fM)'%(
          total_w_num, total_w_bytes/1e6, self._trainable_num,
          self._trainable_bytes/pow(1024.0,2)))
        self.log('Total activation size:%0.1fM'%\
                 (self._activation_size/pow(1024.0,2)))
        self.log('------------------------------------------------------------\n\n')
        self.log_model_summary()

      return new_points

  def classifier(self, inputs, is_training, dense_features=[]):
    with tf.variable_scope('classifier'):
      #                     Fully connect layers
      out_drop_rate=self.data_net_configs['drop_imo']['output']
      for k,features in enumerate(dense_features):
        inputs = tf.layers.dense(inputs, features, None, True, KERNEL_INI )
        if self.IsShowModel: self.log( tensor_info(inputs, 'dense%d'%(k), 'dense0'))
        inputs = self.batch_norm(inputs, is_training, tf.nn.relu)
        inputs = tf.layers.dropout(inputs, out_drop_rate, training=is_training)

      inputs = tf.layers.dense(inputs, self.num_classes, None, True, KERNEL_INI )
      self._dense_num += len(dense_features) + 1
      inputs = tf.identity(inputs, 'final_dense')
    return inputs


  def segmentation(self, l_points, flatting_idx, flat_valid_mask, is_training):
    if self.IsShowModel:
      self.log('--------------- bottle_last_scale ---------------')
    with tf.variable_scope('/Bottle'):
      end_points = l_points[-1]
      for k,f in enumerate(self.block_params['dense_filters']['bottle_last_scale']):
        end_points = self.conv1d(end_points, f, 1, 1, 'valid')
        self.log_tensor_c(end_points, 1, 1, 'v', tf.get_variable_scope().name)
        end_points = self.batch_norm(end_points, is_training, tf.nn.relu)
      l_points[-1] = end_points

    for s0 in  range(self.net_num_scale):
      s = self.net_num_scale -  s0 - 1
      flatten_filters = self.block_params['flatten_filters'][s]
      if self.IsShowModel:
        self.log('******************* scale %d Feature Propagate *******************'%(s))
      with tf.variable_scope('/Seg%d'%(s)):
        if s == self.net_num_scale-1:
          # the last scale
          assert l_points[s+1].shape[1].value == 1
          if s==0:
            nb = self.sg_settings['npoint_per_block'][s]
          else:
            nb = self.sg_settings['nblock'][s]
          flatten_points = tf.tile(l_points[s+1], [1, nb,1 ])
        else:
          flatten_points = self.flatting(l_points[s+1], flatting_idx[s], flat_valid_mask[s])
        if l_points[s]==None:
          new_points = flatten_points
          self.log_tensor_p(new_points, 'points0 is None', 'use flatten only')
        else:
          new_points = tf.concat([l_points[s], flatten_points], 2)
          self.log_tensor_p(new_points, '%d+%d'%(l_points[s].shape[-1].value, flatten_points.shape[-1].value) ,'concat')

        for k,f in enumerate(flatten_filters):
          with tf.variable_scope('c%d'%(k)):
            new_points = self.conv1d(new_points, f, 1, 1, 'valid')
            self.log_tensor_c(new_points, 1, 1, 'v', tf.get_variable_scope().name)
            new_points = self.batch_norm(new_points, is_training, tf.nn.relu)
        l_points[s] = new_points
    return l_points[0]

  def flatting(self, points_upscale, flatting_idx, flat_valid_mask):
    with tf.variable_scope('flatting'):
      shape0 = [e.value for e in points_upscale.shape]
      shape1 = [e.value for e in flatting_idx.shape]

      batch_idx = tf.reshape(tf.range(self.batch_size), [-1,1,1,1])
      batch_idx = tf.tile(batch_idx, [1,shape1[1], shape1[2],1])
      flatting_idx_auged = tf.concat([batch_idx, tf.expand_dims(flatting_idx,-1)], -1)
      flatten_points = tf.gather_nd(points_upscale, flatting_idx_auged)
      flat_valid_mask = tf.cast(flat_valid_mask, tf.float32)
      flatten_points *= tf.expand_dims(flat_valid_mask, -1)
      flatten_points = tf.reduce_sum(flatten_points, 2)
      valid_flat_num = tf.reduce_sum(flat_valid_mask, 2, keepdims=True)
      flatten_points /= valid_flat_num

      #bi = tf.constant(0, tf.int32)
      #cond = lambda bi, flatten_points: tf.less(bi, self.batch_size)
      #flatten_points = tf.zeros([0, shape1[1], shape0[2]])
      #def body(bi, flatten_points):
      #  flatten_points_i = tf.gather(points_upscale[bi], flatting_idx[bi])
      #  mask_i = tf.cast(tf.expand_dims(flat_valid_mask[bi],2), tf.float32)
      #  flatten_points_i = flatten_points_i * mask_i
      #  valid_flat_num = tf.reduce_sum(mask_i, 1)
      #  flatten_points_i = tf.reduce_sum(flatten_points_i, 1)
      #  flatten_points_i = flatten_points_i / valid_flat_num
      #  flatten_points_i = tf.expand_dims(flatten_points_i,0)
      #  flatten_points = tf.concat([flatten_points, flatten_points_i], 0)
      #  return bi, flatten_points
      #bi, flatten_points = tf.while_loop(cond, body, [bi, flatten_points],
      #                        shape_invariants=[bi.get_shape(), \
      #                              tf.TensorShape([None, shape1[1], shape0[2]])])
      #flatten_points.set_shape([self.batch_size, shape1[1], shape0[2]])
      self.log_tensor_p(flatten_points, 'from {}'.format(shape0), 'flatting')
      return flatten_points

  def ave_grouped_features(self, grouped_features):
    shape0 = [e.value for e in grouped_features.shape ]
    if shape0[1] == 1:
      return tf.squeeze(grouped_features, 1)
    else:
      return None

  @staticmethod
  def grouping_online(points, grouped_pindex):
      shape = tf.shape(grouped_pindex)
      batch_size = shape[0]
      tmp = tf.reshape( tf.range(0,batch_size,1), (batch_size,1,1,1) )
      tmp = tf.tile(tmp, [1,shape[1],shape[2],1])
      grouped_pindex = tf.expand_dims(grouped_pindex, -1)
      grouped_pindex = tf.concat([tmp, grouped_pindex], -1)
      grouped_points = tf.gather_nd(points, grouped_pindex)
      return grouped_points


  def normalize_xyz(self, scale, grouped_center, bot_cen_top, grouped_points):
    shape0 = [e.value for e in  grouped_center.shape]
    shape1 = [e.value for e in  bot_cen_top.shape]
    shape2 = [e.value for e in  grouped_points.shape]

    assert len(shape0) == len(shape2) == 4
    assert len(shape1) == 3
    assert shape0[0] == shape1[0] == shape2[0] == self.batch_size
    if shape0[1] > 0:
      assert shape0[1] == shape1[1] == shape2[1] # block num

    if scale == 0:
      grouped_center = grouped_points[:,:,:,0:3] # actually should be shape

    #  use last scale block center as point position of current scale
    grouped_xyz = grouped_center

    #if self.mean_grouping_position and (not self.voxel3d):
    if self.mean_grouping_position:
      # use mean os centroid, relevant with resolution
      sub_block_mid = tf.reduce_mean(grouped_xyz, 2, keepdims=True)
    else:
      # use current block scale center as centroid, indepent of resolution
      sub_block_mid = tf.expand_dims(bot_cen_top[:,:,3:6], 2)

    #***************************************************************************
    # normalize xyz per sub block to [-1, 1]
    grouped_xyz_sb_normed = grouped_xyz - sub_block_mid
    sub_block_max = tf.reduce_max(tf.abs(grouped_xyz_sb_normed), [2,3], keepdims=True)
    #grouped_xyz_sb_normed /= sub_block_max

    #***************************************************************************
    # normalize xyz per global block to [-1, 1]
    if scale>0:
      global_block_mid = tf.reduce_mean( sub_block_mid,1, keepdims=True )
      grouped_xyz_gb_normed = grouped_xyz - global_block_mid
      global_block_max = tf.reduce_max(tf.abs(grouped_xyz_gb_normed), [1,2,3], keepdims=True)
      #grouped_xyz_gb_normed /= global_block_max

    grouped_xyz_feed = []
    if 'raw' in self.xyz_elements:
        grouped_xyz_feed.append( grouped_xyz )
    if 'sub_mid' in self.xyz_elements:
        grouped_xyz_feed.append( grouped_xyz_sb_normed )
    if 'global_mid' in self.xyz_elements:
        assert scale>0, "s and g is same, use s"
        grouped_xyz_feed.append( grouped_xyz_gb_normed )
    grouped_xyz_feed = tf.concat( grouped_xyz_feed, -1 )

    if scale==0:
        # xyz must be at the first in feed_data_elements !!!!
        tf.add_to_collection('raw_inputs_COLC', grouped_points)
        grouped_points = tf.concat( [grouped_xyz_feed, grouped_points[...,3:]],-1 )

        #if len(indrop_keep_mask.get_shape()) != 0:
        #    if InDropMethod == 'set1st':
        #        # set all the dropped item as the first item
        #        tmp1 = tf.multiply( grouped_points, grouped_indrop_keep_mask )
        #        points_1st = grouped_points[:,:,0:1,:]
        #        points_1st = tf.tile( points_1st, [1,1,grouped_points.shape[2],1] )
        #        indrop_mask_inverse = 1 - grouped_indrop_keep_mask
        #        tmp2 = indrop_mask_inverse * points_1st
        #        grouped_points = tf.add( tmp1, tmp2, name='grouped_points_droped' ) # gpu_0/sa_layer0/grouped_points_droped
        #        #tf.add_to_collection( 'check', grouped_points )
        #    elif InDropMethod == 'set0':
        #        valid_mask = tf.logical_and( valid_mask, tf.equal(grouped_indrop_keep_mask,0), name='valid_mask_droped' )   # gpu_1/sa_layer0/valid_mask_droped

    else:
      if self.use_xyz and (not scale==self.net_num_scale-1):
          grouped_points = tf.concat([grouped_points, grouped_xyz_feed],axis=-1)
          self.log_tensor_p(grouped_points, 'use xyz', 'cas%d'%(scale))
    return grouped_points


  def res_sa_module(self, scale, points, grouped_center,
                      grouped_pindex,  empty_mask, bot_cen_top, vox_index ):
    with tf.variable_scope('scale_%d'%(scale)):
      if scale < self.net_num_scale-1:
        grouped_points = Model.grouping_online(points, grouped_pindex)
      else:
        grouped_points = tf.expand_dims(points, 1)
      grouped_points = self.normalize_xyz(scale, grouped_center,
                                            bot_cen_top, grouped_points)

      if scale == 0:
        grouped_points = self.initial_layer(grouped_points)
      else:
        if self.voxel3d:
          grouped_points = self.voxelization(scale, grouped_points,
                                              vox_index, empty_mask)

      outputs = self.res_sa_model(scale, grouped_points)

      if scale == 0 or (not self.voxel3d):
        # use max pooling to reduce map size
        if scale==0:
          #outputs = self.feature_uncompress_block(outputs, 2, 1)
          root_point_features = outputs
        else:
          root_point_features = None
        assert len(outputs.shape)==4

        outputs = self.batch_norm(outputs, self.is_training, tf.nn.relu)
        outputs = tf.reduce_max(outputs, axis=2 if self.data_format=='channels_last' else 3)
        self.log_tensor_p(outputs, 'max', 'cas%d'%(scale))

      else:
        # already used 3D CNN to reduce map size, just reshape
        root_point_features = None
        if self.voxel3d:
          # self.grouping only spport 2D point cloud
          assert len(outputs.shape)==5
          channels_idxs = np.arange(1,4) + int(self.data_format=='channels_first')
          tmp = np.array( [outputs.shape[j].value for j in channels_idxs] )
          tmp = tmp[0]*tmp[1]*tmp[2]
          # Except the last cascade, the voxel size should be reduced to 1
          if scale != self.net_num_scale-1:
            assert tmp==1, "Network design not match grouping configuration"
          else:
            assert outputs.shape[0].value == self.batch_size # global block
          if self.data_format=='channels_last':
            outputs = tf.reshape(outputs, [self.batch_size,-1,outputs.shape[-1].value])
          else:
            outputs = tf.reshape(outputs, [self.batch_size,outputs.shape[1].value,-1])
          self.log_tensor_p(outputs, 'reshape', 'cas%d'%(scale))

      return outputs, root_point_features

  def initial_layer(self, grouped_inputs, scope_ini='initial'):
    self.log_tensor_p(grouped_inputs, 'grouped_inputs', '\nOriginal')
    with tf.variable_scope(scope_ini):
      grouped_inputs = self.conv2d3d(
          inputs=grouped_inputs, filters=self.block_params['num_filters0'], kernels=1,
          strides=1, padding_s1='s')
      grouped_inputs = tf.identity(grouped_inputs, 'initial_conv')
      self.log_tensor_c(grouped_inputs, 1, 1, 's', tf.get_variable_scope().name)

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.resnet_version == 1:
        grouped_inputs = self.batch_norm(grouped_inputs, training, tf.nn.relu)

    return grouped_inputs

  def res_sa_model(self, scale, inputs):
      for i, block_size in enumerate(self.block_params['block_sizes'][scale]):
        if self.IsShowModel:
          self.log('-------------------scale %d, block %d---------------------'%(scale, i))
        block_params = {}
        block_params['filters'] = self.block_params['filters'][scale][i]

        if scale == 0:
          # point net is special
          block_params['kernels'] = block_params['strides'] = 1
          block_params['padding_s1'] = 's'
          block_params['block_sizes'] = self.block_params['block_sizes'][scale][i]
        else:
          for ele in ['kernels', 'strides', 'padding_s1', 'block_sizes',
                      'icp_block_ops']:
            if ele in self.block_params:
              block_params[ele] = self.block_params[ele][scale][i]

        with tf.variable_scope('B%d'%(i)):
          block_fn = self.block_fn if scale!=0 else self.building_block_v2
          inputs = self.block_layer(scale, inputs, block_params, block_fn,
                    self.is_training, 'block_layer{}'.format(i + 1))
        self.block_num_count += 1
      return inputs


  def voxelization(self, scale, grouped_points, vox_index, empty_mask):
    '''
    '''
    gp_shape = [e.value for e in grouped_points.shape]
    voxind_shape = [e.value for e in vox_index.shape]
    assert len(gp_shape) == len(voxind_shape) == 4
    for i in range(3):
      assert gp_shape[i] ==  voxind_shape[i]
    batch_size = gp_shape[0]
    block_num = gp_shape[1]
    point_num = gp_shape[2]

    vox_size = self.sg_settings['vox_size'][scale+1]
    grouped_vox_size = [batch_size, block_num, vox_size[0], vox_size[1],
                        vox_size[2], gp_shape[3]]

    batch_idx = tf.reshape( tf.range(batch_size),[batch_size,1,1,1] )
    batch_idx = tf.tile( batch_idx, [1,block_num,point_num,1] )
    bn_idx = tf.reshape( tf.range(block_num),[1,block_num,1,1] )
    bn_idx = tf.tile( bn_idx, [batch_size,1,point_num,1] )
    vox_index = tf.concat( [batch_idx, bn_idx, vox_index], -1 )
    voxind_shape = [e.value for e in vox_index.shape]

    grouped_voxels = tf.scatter_nd(vox_index, grouped_points, grouped_vox_size)

    self.log_tensor_p(grouped_voxels, 'voxel', 'cas%d'%(scale))
    grouped_vox_size1 = [batch_size * block_num, vox_size[0], vox_size[1],
                        vox_size[2], gp_shape[3]]
    grouped_voxels = tf.reshape(grouped_voxels, grouped_vox_size1)
    self.log_tensor_p(grouped_voxels, 'reshape', 'cas%d'%(scale))
    return grouped_voxels

