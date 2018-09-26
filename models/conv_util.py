
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
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.path.dirname(BASE_DIR)

DEBUG_TMP = False

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

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


def get_tensor_shape(tensor):
  if isinstance(tensor, tf.Tensor):
    shape = tensor.shape.as_list()
    for i,s in enumerate(shape):
      if s==None:
        shape[i] = tf.shape(tensor)[i]
    return shape
  else:
    return tensor.shape


def gather_second_d(inputs, indices):
  '''
  Gather inputs by indices, indices is for the second dim.
    inputs: (batch_size, n1, ...)
  '''
  idx_shape = get_tensor_shape(indices)
  batch_idx = tf.reshape(tf.range(idx_shape[0]), [-1,1,1,1])
  batch_idx = tf.tile(batch_idx, [1, idx_shape[1], idx_shape[2], 1])
  indices = tf.expand_dims(indices, -1)
  indices = tf.concat([batch_idx, indices], -1)
  outputs = tf.gather_nd(inputs, indices)
  return outputs


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

  def __init__(self, net_data_configs, data_format, dtype):
    self.data_format = data_format
    if data_format == None:
      self.data_format = 'channels_last'
    self.net_data_configs= net_data_configs

    self.residual = net_data_configs['net_configs']['residual']
    train_global_step = tf.train.get_or_create_global_step()
    self.res_scale = 1.0
    self.use_bias = True
    self.block_style = 'Regular'
    self.drop_imo = net_data_configs['net_configs']['drop_imo']

    #self.batch_norm_decay = _BATCH_NORM_DECAY
    self.batch_norm_decay_fn = net_data_configs['net_configs']['bn_decay_fn']

    model_dir = net_data_configs['data_configs']['model_dir']
    if ResConvOps._epoch==0:
      self.IsShowModel = True
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
      self.model_log_fn = os.path.join(model_dir, 'log_model.txt')
      self.model_log_f = open(self.model_log_fn, 'w')

    ResConvOps._epoch += 1

  def log_model_summary(self):
    if not self.IsShowModel:
      return

    self.model_log_f.write('\n--------------------------------------------\n')
    self.show_layers_num_summary()
    self.model_log_f.write('\n\n--------------------------------------------\n')

    dnc = self.net_data_configs
    net_configs = dnc['net_configs']
    data_configs = dnc['data_configs']

    #*************************************************************************
    # key training parameters
    res = 'R' if net_configs['residual'] else 'P'
    key_para_names = 'model bs feed drop_imo lr0 \n'
    key_paras_str = '{net_flag} {bs} {feed_data_eles} \
{drop_imo}  {lr0}'.format(
      net_flag=dnc['net_flag'],
      bs=net_configs['batch_size'],
      feed_data_eles=data_configs['feed_data_eles'].replace('nxnynz','n') ,
      drop_imo=net_configs['drop_imo_str'],
      lr0=net_configs['lr0'],
      )

    self.key_paras_str = key_para_names + key_paras_str
    #*************************************************************************
    config_group_to_write = ['net_configs', 'data_configs']
    for gp in config_group_to_write:
      gpc = self.net_data_configs[gp]
      str_g = '\n%s\n'%(gp)
      self.model_log_f.write(str_g)
      print(str_g)
      for item in gpc:
        str_i = '\t%s:%s\n'%(item, gpc[item])
        self.model_log_f.write(str_i)
        print(str_i)

    #*************************************************************************
    self.model_log_f.write('\n--------------------------------------------\n')

    total_w_num, total_w_bytes, train_w_shapes = self.train_w_bytes()
    self.log('Total trainable weights: (%d %0.3fM)  Counted (%d %0.3fM)'%(
      total_w_num, total_w_bytes/1e6, self._trainable_num,
      self._trainable_bytes/pow(1024.0,2)))
    self.log('Total activation size:%0.1fM'%\
              (self._activation_size/pow(1024.0,2)))
    self.log('------------------------------------------------------------\n\n')

    #*************************************************************************
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
    global_step = tf.train.get_or_create_global_step()
    batch_norm_decay = self.batch_norm_decay_fn(global_step)
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if self.data_format == 'channels_first' else -1,
        momentum=batch_norm_decay , epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)

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

  def log_tensor_c(self, inputs, kernels, strides, pad_stride1, var_scope,
                   pre_indent=''):
      if not self.IsShowModel: return
      self.add_activation_size(inputs)
      if len(inputs.shape)==4 :
        conv_str = 'conv2d'
      elif len(inputs.shape)==3:
        conv_str = 'conv1d'
      elif len(inputs.shape)==5:
        conv_str = 'conv3d'
      layer_name = pre_indent+'/'.join(var_scope.split('/')[1:])
      self.log( tensor_info(inputs, '%s (%d,%d,%s)'%
                    (conv_str, kernels, strides, pad_stride1), layer_name,
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

  def log_dotted_line(self, name):
    if self.IsShowModel:
      self.log('\n---------------------- %s ----------------------'%(name))

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

  def conv1d2d3d(self, inputs, filters, kernels, strides, pad_stride1):
    """Strided 2-D or 3-D convolution with explicit padding.
      pad_stride1:  only used when strides==1
    """
    if len(inputs.shape)==5:
      conv_fn = tf.layers.conv3d
      self._conv3d_num += 1
    elif len(inputs.shape) == 4:
      conv_fn = tf.layers.conv2d
      self._conv2d_num += 1
    elif len(inputs.shape) == 3:
      conv_fn = tf.layers.conv1d
      self._conv1d_num += 1
    else:
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass
      raise NotImplementedError

    inputs, padding = self.padding2d3d(inputs, kernels, strides, pad_stride1)

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

  def padding2d3d(self, inputs, kernels, strides, pad_stride1):
    '''
    When strides==1, padding = 'v' is used to reduce feature map,
    When strides>1, fixed padding is used to keep reduce rate equal to strides.
    '''
    padding = self._padding[pad_stride1]
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

  def pool2d3d(self, inputs, pool, kernels, strides, pad_stride1):
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

    inputs, padding = self.padding2d3d(inputs, kernels, strides, pad_stride1)
    outputs = pool_fn(inputs, kernels, strides, padding, self.data_format)
    return outputs

  def feature_uncompress_block(self, inputs, feature_rate, uncompress_times):
    assert self.data_format == 'channels_last'
    with tf.variable_scope('fu'):
      for i in range(uncompress_times):
        #filters_i = int(self.get_feature_channels(inputs)*2) - 3*self.use_xyz
        filters_i = int(self.get_feature_channels(inputs)*2)
        inputs = self.conv1d2d3d(inputs, filters_i, 1,1,'s')
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
    pad_stride1 = block_params['pad_stride1']

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
      inputs = self.conv1d2d3d(inputs, filters, kernels, strides, pad_stride1)
      self.log_tensor_c(inputs, kernels, strides, pad_stride1,
                        tf.get_variable_scope().name)
    if half_layer: return inputs
    inputs = self.batch_norm(inputs, training, tf.nn.relu)

    with tf.variable_scope('c1'):
      if self.check_kernel_waste(inputs, kernels):
        kernels = 1
      inputs = self.conv1d2d3d(inputs, filters, kernels, 1, 's')
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
    pad_stride1 = block_params['pad_stride1']

    shortcut = inputs
    if not no_ini_bn:
      inputs = self.batch_norm(inputs, training, tf.nn.relu)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
      shortcut = projection_shortcut(inputs)

    with tf.variable_scope('c0'):
      inputs = self.conv1d2d3d(inputs, filters//4, 1, 1, 's')
      self.log_tensor_c(inputs, 1, 1, 's', tf.get_variable_scope().name)

    inputs = self.batch_norm(inputs, training, tf.nn.relu)

    with tf.variable_scope('c1'):
      inputs = self.conv1d2d3d(inputs, filters//4, kernels, strides, pad_stride1)
      self.log_tensor_c(inputs, kernels, strides, pad_stride1,
                        tf.get_variable_scope().name)

    inputs = self.batch_norm(inputs, training, tf.nn.relu)

    with tf.variable_scope('c2'):
      inputs = self.conv1d2d3d(inputs, filters, 1, 1, 's')
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
      op: ['conv',filters,kernels,strides,pad_stride1]
          ['max'/'ave',kernel_size,strides,pad_stride1]
    '''
    var_scope = tf.get_variable_scope().name
    if op[0] == 'conv':
      net = self.conv1d2d3d(net, op[1], op[2], op[3], op[4])
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

    inputs = self.conv1d2d3d(inputs, block_params['filters'], 1, 1, 's')
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
    pad_stride1 = block_params['pad_stride1']

    need_projection = self.get_feature_channels(inputs) != filters_out_sc or\
          (pad_stride1=='v' and kernels>1) or strides>1
    if not need_projection:
      return inputs

    # In offical resnet, kernel_sc is always 1, because strides>1 reduces map.
    # But kernel_sc>1 here to reduce map, and strides is always 1.
    kernel_sc = kernels if pad_stride1=='v' else 1
    if scm == 'C':
      # pad_stride1=='v: feature map size have to be reduced => kernels
      with tf.variable_scope('sc_C'):
        shortcut = self.conv1d2d3d(inputs, filters_out_sc, kernel_sc, strides,
                                 pad_stride1)
        self.log_tensor_c(shortcut, kernel_sc, strides,
                        pad_stride1, tf.get_variable_scope().name)

    elif scm == 'MC' or scm == 'AC' or scm == 'MZ' or scm == 'AZ':
      with tf.variable_scope('sc_'+scm):
        layer_name = '/'.join(tf.get_variable_scope().name.split('/')[2:])
        if kernel_sc!=1:
          pool = 'max' if scm[0]=='M' else 'ave'
          shortcut = self.pool2d3d(inputs, pool, kernel_sc, strides, pad_stride1)
          self.log_tensor_p(shortcut, pool, layer_name, kernel_sc, strides, pad_stride1)
        else:
          shortcut = inputs

        channels_dif = filters_out_sc - self.get_feature_channels(shortcut)
        if channels_dif != 0:
          if scm[1] == 'C':
            shortcut = self.conv1d2d3d(shortcut, filters_out_sc, 1, 1, 's')
            self.log_tensor_c(shortcut, 1, 1, 's', tf.get_variable_scope().name)
          else:
            #assert channels_dif>0
            if channels_dif<0:
              import pdb; pdb.set_trace()  # XXX BREAKPOINT
              pass
            shortcut = self.padding_channel(shortcut, channels_dif)
            self.log_tensor_p(shortcut, 'padding', layer_name)

    else:
      raise NotImplementedError
    return shortcut

  def blocks_layers(self, scale, inputs, blocks_params, block_fn, is_training, scope):
    self.log_tensor_p(inputs, '', scope)
    self.log_dotted_line(scope)
    for bi in range(len(blocks_params)):
      with tf.variable_scope(scope+'_b%d'%(bi)):
        inputs = self.block_layer(scale, inputs, blocks_params[bi], block_fn,
                                is_training, scope+'_b%d'%(bi))
    return inputs

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
    # and pad_stride1
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
    block_params['pad_stride1'] = 's'
    for j in range(1, block_size):
      with tf.variable_scope('L%d'%(j)):
        inputs = block_fn(inputs, block_params, is_training, None)

    self._block_layers_num += 1
    return tf.identity(inputs, name)


  def dense_block(self, inputs, filters, is_training):
    out_drop_rate = self.drop_imo[2]
    n = len(filters)
    for i, f in enumerate(filters):
      inputs = tf.layers.dense(inputs, f)
      if self.IsShowModel: self.log_tensor_p(inputs, '', 'dense_%d'%(i))
      if i!=n-1:
        inputs = self.batch_norm(inputs, is_training, tf.nn.relu)
        if out_drop_rate>0:
          if self.IsShowModel: self.log('dropout {}'.format(out_drop_rate))
          inputs = tf.layers.dropout(inputs, out_drop_rate, training=is_training)
    return inputs


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


