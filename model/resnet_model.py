
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))

DEBUG_TMP = False

_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
#CASTABLE_TYPES = (tf.float32,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES
#ALLOWED_TYPES = (DEFAULT_DTYPE,)

USE_CHARLES = True


def tensor_info(tensor_ls, tensor_name_ls=None, scope=None):
  if type(tensor_ls) != list:
    tensor_ls = [tensor_ls]
  if tensor_name_ls == None:
    tensor_name_ls = [''] * len(tensor_ls)
  elif type(tensor_name_ls) != list:
    tensor_name_ls = [tensor_name_ls]
  tensor_info = ''

  for i in range(len(tensor_ls)):
    if scope!=None:
      tensor_info += '%-20s'%(scope)
    tensor_info += '%-20s: '%(tensor_name_ls[i])
    if tensor_ls[i] == None:
        tensor_info += 'None'
    else:
        tensor_info += str( [s.value for s in tensor_ls[i].shape] )
    if i < len(tensor_ls)-1:
        tensor_info += '\n'
  return tensor_info

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


def fixed_padding(inputs, kernel_size, data_format):
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

  if data_format == 'channels_first':
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
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v1, without a bottleneck.

  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

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

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = self.batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = self.batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = self.batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs




def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v1, with a bottleneck.

  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

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

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = self.batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = self.batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d3d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = self.batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d3d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = self.batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs




class ResConvOps(object):
  ''' Basic convolution operations '''
  _block_layers_num = 0
  _conv2d_num = 0
  _conv3d_num = 0
  IsShowModel = False
  _epoch = 0

  def __init__(self, data_net_configs):
    self.residual = data_net_configs['residual']
    self.voxel3d = 'V' in data_net_configs['model_flag']
    global_step = tf.train.get_or_create_global_step()
    self.batch_norm_decay = data_net_configs['bndecay_fn'](global_step)
    tf.summary.scalar('batch_norm_decay', self.batch_norm_decay)
    self.use_bias = data_net_configs['use_bias']

    model_dir = data_net_configs['model_dir']
    if ResConvOps._epoch==0:
      self.IsShowModel = True
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
      self.model_log_fn = os.path.join(model_dir, 'log_model.txt')
      self.model_log_f = open(self.model_log_fn, 'w')


      dnc = data_net_configs
      res = 'rs' if self.residual else 'pl'
      key_para_names = 'model use_bias bs feed aug drop_imo lr0_drate_depoch \
bnd optimizer filters0\n'
      key_paras_str = '{model_name} {use_bias} {bs} {feed_data_eles}\
{aug} {drop_imo} {lr0}_{lr_decay_rate}_{lr_decay_epochs} {bnd} {optimizer} \
{filters0}\n\n'.format(
        model_name=res+str(dnc['resnet_size'])+dnc['model_flag'],
        use_bias=str(dnc['use_bias']),
        bs=dnc['batch_size'],
        feed_data_eles=dnc['feed_data_eles'],
        aug=dnc['aug_types'],
        drop_imo=dnc['drop_imo_str'],
        lr0=dnc['learning_rate0'],
        lr_decay_rate=dnc['lr_decay_rate'],
        lr_decay_epochs=dnc['lr_decay_epochs'],
        bnd=dnc['batch_norm_decay0'],
        optimizer=dnc['optimizer'],
        filters0=dnc['num_filters0'])

      self.key_paras_str = key_para_names + key_paras_str
      self.model_log_f.write(self.key_paras_str)

      items_to_write = ['model_flag', 'dataset_name', 'aug_types', 'drop_imo', \
        'feed_data', 'xyz_elements', 'points', 'global_step','global_stride',\
        'sub_block_stride_candis', 'sub_block_step_candis','optimizer',\
        'learning_rate0', 'lr_decay_rate', 'batch_norm_decay0', 'lr_vals', \
        'bndecay_vals', 'use_bias', \
        'weight_decay', 'num_filters0','resnet_size', 'block_kernels', \
        'block_strides', 'block_paddings', 'data_dir']
      for item in items_to_write:
        str_i = '%s:%s\n'%(item, data_net_configs[item])
        self.model_log_f.write(str_i)
        if self.IsShowModel:
          print(str_i)
      self.model_log_f.write('\n')
      self.model_log_f.flush()
    ResConvOps._epoch += 1

  def batch_norm(self, inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    if USE_CHARLES:
      scale = False
      epsilon = 0.001
      fused = None
    else:
      scale = True
      epsilon = _BATCH_NORM_EPSILON
      fused = True
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
        momentum=self.batch_norm_decay , epsilon=epsilon, center=True,
        scale=scale, training=training, fused=fused)

  def log(self, log_str):
    self.model_log_f.write(log_str+'\n')
    self.model_log_f.flush()
    print(log_str)

  def show_layers_num_summary(self):
    self.log('block layers num:{}\nconv2d num:{}\nconv3d num:{}\n'.format(
                  self._block_layers_num, self._conv2d_num, self._conv3d_num))

  def conv2d3d_fixed_padding(self, inputs, filters,
                              kernel_size, strides, padding_s1, data_format):
    """Strided 2-D or 3-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if len(inputs.shape)==5:
      conv_fn = tf.layers.conv3d
      self._conv3d_num += 1
    elif len(inputs.shape) == 4:
      conv_fn = tf.layers.conv2d
      self._conv2d_num += 1

    # only used when strides==1
    if padding_s1=='s':
      padding_s1 = 'SAME'
    else:
      assert padding_s1=='v'
      padding_s1 = 'VALID'

    if strides > 1:
      inputs = fixed_padding(inputs, kernel_size, data_format)
      padding = 'VALID'
    else:
      padding = padding_s1

    if USE_CHARLES:
      kernel_initializer = tf.contrib.layers.xavier_initializer()   # charles
    else:
      kernel_initializer = tf.variance_scaling_initializer()       # res official

    inputs = conv_fn(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=self.use_bias,
        kernel_initializer=kernel_initializer,
        data_format=data_format)
    return inputs

  def _building_block_v2(self, inputs, filters, training, projection_shortcut,
                          b_kernel_size, strides, padding_s1, data_format):
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
    shortcut = inputs
    inputs = self.batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    conv_str = 'conv2d' if len(inputs.shape)==4 else 'conv3d'
    if (not self.voxel3d) and len(inputs.shape)==5:
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
      shortcut = projection_shortcut(inputs)

    inputs = self.conv2d3d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=b_kernel_size, strides=strides,
        padding_s1=padding_s1, data_format=data_format)
    if self.IsShowModel:
      self.log( tensor_info(inputs, '%s k,s,p=%d,%d,%s'%
                    (conv_str,b_kernel_size,strides,padding_s1), 'block_v2'))

    inputs = self.batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = self.conv2d3d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=b_kernel_size, strides=1,
        padding_s1='s', data_format=data_format)
    if self.IsShowModel:
      self.log( tensor_info(inputs, '%s k,s,p=%d,%d,%s'%
                (conv_str,b_kernel_size,strides,padding_s1), 'block_v2')+'\n')

    if self.residual:
      assert inputs.shape == shortcut.shape
      return inputs + shortcut
    else:
      return inputs
  def _bottleneck_block_v2(self, inputs, filters, training, projection_shortcut,
                          b_kernel_size, strides, padding_s1, data_format):
    """A single block for ResNet v2, without a bottleneck.

    Similar to _building_block_v2(), except using the "bottleneck" blocks
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
    shortcut = inputs
    inputs = self.batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    conv_str = 'conv2d' if len(inputs.shape)==4 else 'conv3d'

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
      shortcut = projection_shortcut(inputs)

    inputs = self.conv2d3d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1, padding_s1=padding_s1,
        data_format=data_format)
    if self.IsShowModel:
      self.log( tensor_info(inputs, '%s k,s=1,1'%(conv_str), 'bottle_v2'))

    inputs = self.batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = self.conv2d3d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=b_kernel_size, strides=strides,
        padding_s1=padding_s1, data_format=data_format)
    if self.IsShowModel:
      self.log( tensor_info(inputs, '%s k,s=%d,%d'%
                        (conv_str,b_kernel_size,strides), 'bottle_v2'))

    inputs = self.batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = self.conv2d3d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        padding_s1=padding_s1, data_format=data_format)
    if self.IsShowModel: self.log( tensor_info(inputs, '%s k,s=1,1'%(conv_str),
                                        'bottle_v2')+'\n' )

    if self.residual:
      assert inputs.shape == shortcut.shape
      return inputs + shortcut
    else:
      return inputs

  def block_layer(self, inputs, filters, bottleneck, block_fn, blocks, b_kernel_size,
                  strides, padding_s1, training, name, data_format):
    """Creates one layer of blocks for the ResNet model.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      bottleneck: Is the block created a bottleneck block.
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
      # 2d resenet use strides>1 to reduce feature map and create shortcut.
      # Here we use kernel>1 and padding_s1='VALID'
      # Use kernel>1 in shortcut may somewhat impede the identity forward, try
      # optimize later.
      kernel_size_shortcut = b_kernel_size if padding_s1=='v' else 1
      shortcut = self.conv2d3d_fixed_padding(
          inputs=inputs, filters=filters_out,
          kernel_size=kernel_size_shortcut,
          strides=strides, padding_s1=padding_s1, data_format=data_format)
      if self.IsShowModel:
        conv_str = 'conv2d' if len(inputs.shape)==4 else 'conv3d'
        self.log( tensor_info(shortcut, '%s k,s,p=%d,%d,%s'%(conv_str,
              kernel_size_shortcut, strides, padding_s1),'projection_shortcut'))
      return shortcut

    # Only the first block per block_layer uses projection_shortcut and strides
    # and padding_s1
    if (b_kernel_size==1 and strides==1 and inputs.shape[-1].value==filters_out)\
          or (not self.residual):
      projection_shortcut_0 = None
    else:
      projection_shortcut_0 = projection_shortcut
    inputs = block_fn(inputs, filters, training, projection_shortcut_0, b_kernel_size,
                      strides, padding_s1, data_format)

    for _ in range(1, blocks):
      inputs = block_fn(inputs, filters, training, None, b_kernel_size, 1, 's', data_format)

    self._block_layers_num += 1
    return tf.identity(inputs, name)

class Model(ResConvOps):
  """Base class for building the Resnet Model."""

  def __init__(self, model_flag, resnet_size, bottleneck, num_classes, num_filters,
               block_sizes, block_kernels, block_strides, block_paddings,
               final_size, resnet_version=DEFAULT_VERSION, data_format=None,
               dtype=DEFAULT_DTYPE, data_net_configs={}):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      bottleneck: Use regular blocks or bottleneck blocks.
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
      final_size: The expected size of the model after the second pooling.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.

    Raises:
      ValueError: if invalid version is selected.
    """
    super(Model, self).__init__(data_net_configs)
    self.model_flag = model_flag
    self.resnet_size = resnet_size

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
          'Resnet version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    if bottleneck:
      if resnet_version == 1:
        self.block_fn = _bottleneck_block_v1
      else:
        self.block_fn = self._bottleneck_block_v2
    else:
      if resnet_version == 1:
        self.block_fn = _building_block_v1
      else:
        self.block_fn = self._building_block_v2

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.block_sizes = block_sizes
    self.block_kernels = block_kernels
    self.block_strides = block_strides
    self.block_paddings = block_paddings
    self.final_size = final_size
    self.dtype = dtype
    self.pre_activation = resnet_version == 2
    self.data_net_configs = data_net_configs
    self.block_num_count = 0

    self._preprocess_configs()

  def _preprocess_configs(self):
    if '_' in self.model_flag:
        self.model_flag, num_neighbors = modelf_nein.split('_')
        self.num_neighbors = np.array( [ int(n) for n in num_neighbors ] )
    else:
        self.num_neighbors= None
    self.global_numpoint = self.data_net_configs['points'][0]
    #self.cascade_num = int(self.model_flag[0])
    self.cascade_num = len(self.data_net_configs['block_sizes'])
    assert self.cascade_num <= self.data_net_configs['sg_bm_extract_idx'].shape[0]-1
    #self.log('cascade_num:{}'.format(self.cascade_num))
    self.IsOnlineGlobal = self.model_flag[-1] == 'G'
    for key in self.data_net_configs:
      setattr(self, key, self.data_net_configs[key])

    for e in self.feed_data:
      assert e in self.data_idxs
    IsAllInputs = len(self.data_idxs) == len(self.feed_data)
    if IsAllInputs:
      self.feed_data_idxs = 'ALL'
    else:
      self.feed_data_idxs = np.sort([i for e in self.feed_data for i in self.data_idxs[e] ])

    self.use_xyz = True
    self.mean_grouping_position = True


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

  def __call__(self, inputs_dic, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    return self._call(
          inputs_dic['points'],
          inputs_dic['sg_bidxmaps'],
          inputs_dic['b_bottom_centers_mm'],
          inputs_dic['bidxmaps_flat'],
          inputs_dic['fmap_neighbor_idis'],
          training)

  def _call(self, inputs, sg_bidxmaps, b_bottom_centers_mm, bidxmaps_flat,
            fmap_neighbor_idis, is_training):

    if self.IsShowModel: self.log('')
    self.is_training = is_training
    sg_bm_extract_idx = self.data_net_configs['sg_bm_extract_idx']

    if self.feed_data_idxs!='ALL':
      inputs = tf.gather(inputs, self.feed_data_idxs, axis=-1)

    with self._model_variable_scope():
      l_points = []                       # size = l_points+1
      l_points.append( inputs )
      l_xyz = inputs[...,0:3]
      new_points = inputs

      globalb_bottom_center_mm = b_bottom_centers_mm[self.cascade_num-1]
      globalb_bottom_center = tf.multiply( tf.cast( globalb_bottom_center_mm, tf.float32), 0.001, name='globalb_bottom_center' ) # gpu_0/globalb_bottom_center
      self.max_step_stride = tf.multiply( globalb_bottom_center[:,:,3:6] - globalb_bottom_center[:,:,0:3], 2.0, name='max_step_stride') # gpu_0/max_step_stride

      full_cascades = sg_bm_extract_idx.shape[0]-1
      scales_feature = []

      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        new_points = tf.transpose(new_points, [0, 2, 1])

      for k in range(self.cascade_num):
          IsExtraGlobalLayer = False

          if k==self.cascade_num-1 and self.IsOnlineGlobal:
              sg_bidxmap_k = None
              block_bottom_center_mm = globalb_bottom_center_mm
          else:
            block_bottom_center_mm = b_bottom_centers_mm[k]
            sg_bidxmap_k = sg_bidxmaps[k]
          block_bottom_center_mm = tf.cast(block_bottom_center_mm, tf.float32, name='block_bottom_center_mm')

          l_xyz, new_points, root_point_features = self.res_sa_module(k, l_xyz,
                          new_points, sg_bidxmap_k, block_bottom_center_mm, scope='sa_layer'+str(k) )
          if k == 0:
              l_points[0] = root_point_features
          l_points.append(new_points)
          if self.IsShowModel: self.log('------------------\n')

      # ----------------------
      if self.IsShowModel: self.log( tensor_info(new_points, 'end', 'res blocks') )
      inputs = new_points
      axes = [2] if self.data_format == 'channels_first' else [1]
      inputs = tf.reduce_mean(inputs, axes)
      inputs = tf.identity(inputs, 'final_reduce_mean')
      if self.IsShowModel: self.log( tensor_info(inputs, 'reduce_mean', 'final') )
      inputs = tf.layers.dropout(inputs,
                              rate=self.data_net_configs['drop_imo']['output'],
                              training=is_training)

      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      inputs = tf.identity(inputs, 'final_dense')
      if self.IsShowModel:
        self.log( tensor_info(inputs, 'dense', 'final') +'\n\n' )
        self.show_layers_num_summary()
        self.log('------------------------------------------------------------')
        self.model_log_f.close()
      return inputs

  def res_sa_module(self, cascade_id, xyz, points, bidmap, block_bottom_center_mm, scope):
    batch_size = xyz.shape[0].value
    new_xyz, grouped_xyz, inputs, valid_mask = self.grouping(cascade_id, xyz,
                        points, bidmap, block_bottom_center_mm, scope)
    if cascade_id == 0:
      inputs = self.initial_layer(inputs)
    elif self.voxel3d:
      inputs = self.grouped_points_to_voxel_points( cascade_id, inputs,
                          valid_mask, block_bottom_center_mm, grouped_xyz)

    outputs= self.res_sa_model(cascade_id,
                inputs, grouped_xyz, valid_mask, block_bottom_center_mm, scope)

    if cascade_id == 0 or (not self.voxel3d):
      # use max pooling to reduce map size
      if cascade_id == 0:
        root_point_features = outputs
      else:
        root_point_features = None
      assert len(outputs.shape)==4
      outputs = tf.reduce_max(outputs, axis=2 if self.data_format=='channels_last' else 3)
      if self.IsShowModel: self.log( tensor_info(outputs, 'max', 'cas%d'%(cascade_id)) +'\n' )
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
        if cascade_id != self.cascade_num-1:
          assert tmp==1
        else:
          assert outputs.shape[0].value == batch_size # global block
        if self.data_format=='channels_last':
          outputs = tf.reshape(outputs, [batch_size,-1,outputs.shape[-1].value])
        else:
          outputs = tf.reshape(outputs, [batch_size,outputs.shape[1].value,-1])

    return new_xyz, outputs, root_point_features

  def initial_layer(self, inputs):
      inputs = self.conv2d3d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=1,
          strides=1, padding_s1='s', data_format=self.data_format)
      inputs = tf.identity(inputs, 'initial_conv')

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.resnet_version == 1:
        inputs = self.batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)
      if self.IsShowModel:self.log(tensor_info(inputs,'conv2d ks:1,1','initial'))

      return inputs

  def res_sa_model(self, cascade_id, inputs, grouped_xyz,
                   valid_mask, block_bottom_center_mm, scope):
      for i, num_blocks in enumerate(self.block_sizes[cascade_id]):
        if self.IsShowModel:
          self.log('--------------cascade_id %d, block %d----------------'%(cascade_id, i))
        num_filters = self.num_filters * (2**(self.block_num_count-cascade_id))
        num_filters = min(num_filters, 1024)
        inputs = self.block_layer(
            inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
            block_fn=self.block_fn, blocks=num_blocks,
            b_kernel_size=self.block_kernels[cascade_id][i],
            strides=self.block_strides[cascade_id][i],
            padding_s1=self.block_paddings[cascade_id][i],
            training=self.is_training,
            name='block_layer{}'.format(i + 1), data_format=self.data_format)
        self.block_num_count += 1
      return inputs

  def grouping(self, cascade_id, xyz, points, bidmap, block_bottom_center_mm, scope):
    if self.data_format == 'channels_first':
      points = tf.transpose(points, [0, 2, 1])

    batch_size = xyz.get_shape()[0].value
    with tf.variable_scope(scope) as sc:
        assert self.cascade_num == self.flatten_bm_extract_idx.shape[0]-1  # include global here (Note: cascade_num does not include global in block_pre_util )
        assert self.sub_block_step_candis.size == self.cascade_num-1
        #if cascade_id==0:
        #    indrop_keep_mask = tf.get_default_graph().get_tensor_by_name('indrop_keep_mask:0') # indrop_keep_mask:0

        assert len(xyz.shape) == 3
        if not ( len(xyz.shape) == len(points.shape) == len(bidmap.shape) == \
                len(block_bottom_center_mm.shape) == 3 ):
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass

        if bidmap==None:
            grouped_xyz = tf.expand_dims( xyz, 1 )
            grouped_points = tf.expand_dims( points, 1 )
            new_xyz = None
            valid_mask = None
        else:
            batch_idx = tf.reshape( tf.range(batch_size),[batch_size,1,1,1] )
            nsubblock = bidmap.get_shape()[1].value
            npoint_subblock = bidmap.get_shape()[2].value
            batch_idx_ = tf.tile( batch_idx,[1,nsubblock,npoint_subblock,1] )
            bidmap = tf.expand_dims( bidmap,axis=-1, name='bidmap' )
            bidmap_concat = tf.concat( [batch_idx_,bidmap],axis=-1, name='bidmap_concat' )  # gpu_0/sa_layer0/bidmap_concat:0
            # The value for invalid item in bidmap is -17.
            # On GPU, the responding grouped_xyz and grouped_points is 0.
            # NOT WORK on CPU !!!

            # invalid indices comes from merge_blocks_while_fix_bmap
            # set point_indices_f for invalid points as
            # NETCONFIG['redundant_points_in_block'] ( shoud be set < -500)
            valid_mask = tf.greater( bidmap, tf.constant(-500,tf.int32), 'valid_mask' ) # gpu_0/sa_layer0/valid_mask:0

            grouped_xyz = tf.gather_nd(xyz, bidmap_concat, name='grouped_xyz')  # gpu_0/sa_layer0/grouped_xyz:0
            grouped_points = tf.gather_nd(points,bidmap_concat, name='group_points')
            #if cascade_id==0 and  len(indrop_keep_mask.get_shape()) != 0:
            #    grouped_indrop_keep_mask = tf.gather_nd( indrop_keep_mask, bidmap_concat, name='grouped_indrop_keep_mask' )  # gpu_0/sa_layer0/grouped_indrop_keep_mask:0

        # new_xyz is the "voxel center" or "mean position of points in the voxel"
        if self.mean_grouping_position and (not self.voxel3d):
            new_xyz = tf.reduce_mean(grouped_xyz,-2)
        else:
            new_xyz = block_bottom_center_mm[:,:,3:6] * tf.constant( 0.001, tf.float32 )
        # the mid can be mean or block center, decided by configs['mean_grouping_position']
        sub_block_mid = tf.expand_dims( new_xyz,-2, name = 'sub_block_mid' )   # gpu_1/sa_layer0/sub_block_mid
        global_block_mid = tf.reduce_mean( sub_block_mid,1, keepdims=True, name = 'global_block_mid' )
        grouped_xyz_submid = grouped_xyz - sub_block_mid
        grouped_xyz_glomid = grouped_xyz - global_block_mid

        grouped_xyz_feed = []
        if 'raw' in self.xyz_elements:
            grouped_xyz_feed.append( grouped_xyz )
        if 'sub_mid' in self.xyz_elements:
            grouped_xyz_feed.append( grouped_xyz_submid )
        if 'global_mid' in self.xyz_elements:
            grouped_xyz_feed.append( grouped_xyz_glomid )
        grouped_xyz_feed = tf.concat( grouped_xyz_feed, -1 )

        if cascade_id==0:
            # xyz must be at the first in feed_data_elements !!!!
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
          if self.use_xyz and (not cascade_id==self.cascade_num-1):
              grouped_points = tf.concat([grouped_xyz_feed, grouped_points],axis=-1)

        if self.IsShowModel:
          sc = 'grouping %d'%(cascade_id)
          self.log(tensor_info(xyz, 'xyz', sc))
          self.log(tensor_info(new_xyz, 'new_xyz', sc))
          self.log(tensor_info(grouped_xyz, 'grouped_xyz', sc))
          self.log(tensor_info(grouped_points, 'grouped_points', sc))
          self.log('')

        new_points = grouped_points
        if self.data_format == 'channels_first':
          new_points = tf.transpose(new_points, [0, 3, 1, 2])
        return new_xyz, grouped_xyz, new_points, valid_mask

  def grouped_points_to_voxel_points (self, cascade_id, new_points, valid_mask, block_bottom_center_mm, grouped_xyz):
    IS_merge_blocks_while_fix_bmap = True

    if self.data_format == 'channels_first':
      new_points = tf.transpose(new_points, [0, 2, 3, 1])

    block_bottom_center_mm = tf.identity( block_bottom_center_mm,'block_bottom_center_mm' )      # gpu_0/sa_layer3/block_bottom_center_mm:0
    new_points = tf.identity(new_points,name='points_tov') # gpu_0/sa_layer4/points_tov:0
    c500 = tf.constant([500],tf.float32)
    c1000 = tf.constant([1000],tf.float32)
    c1 = tf.constant([1,1,1],tf.float32)
    step_last_org = self.sub_block_step_candis[cascade_id-1] * c1
    step_last = tf.minimum( step_last_org, self.max_step_stride, name='step_last' )    # gpu_0/sa_layer1/step_last:0
    step_last = tf.expand_dims(step_last,1)
    stride_last_org = self.sub_block_stride_candis[cascade_id-1] * c1
    stride_last = tf.minimum( stride_last_org, self.max_step_stride, name='stride_last' )  # gpu_0/sa_layer1/stride_last:0
    stride_last = tf.expand_dims(stride_last,1)

    voxel_bottom_xyz_mm = block_bottom_center_mm[:,:,0:3]
    # NOTE: c1=[1,1,1]*0.5 ONLY when the sh5 step is also the same on three dimensions.
    #                      Otherwise, the stride at each cascade may also be changed.
    min_point_bottom_xyz_mm = voxel_bottom_xyz_mm
    min_point_bottom_xyz_mm = tf.expand_dims( min_point_bottom_xyz_mm, -2, name='min_point_bottom_xyz_mm' ) # gpu_0/sa_layer1/min_point_bottom_xyz_mm:0
    grouped_bottom_xyz_mm = grouped_xyz * c1000 - step_last * c500  # gpu_0/sa_layer1/sub_1:0
        # For ExtraGlobal layer, the step_last may be cropped, thus the point_indices_f is smaller.
    point_indices_f = (grouped_bottom_xyz_mm - min_point_bottom_xyz_mm) / (stride_last*c1000)  # gpu_0/sa_layer3/div:0
    point_indices_f = tf.identity( point_indices_f, name='point_indices_f' )    # gpu_0/sa_layer4/point_indices_f:0

    # invalid indices comes from merge_blocks_while_fix_bmap
    # set point_indices_f for invalid points as
    # NETCONFIG['redundant_points_in_block'] ( shoud be set < -500)
    invalid_mask = tf.equal( valid_mask, False )
    invalid_mask = tf.tile( invalid_mask, [1,1,1,3], name='invalid_mask')  # gpu_0/sa_layer1/valid_mask:0
    point_indices_f = tf.where( invalid_mask, tf.ones(shape=point_indices_f.shape,dtype=tf.float32)*tf.constant( -9999,dtype=tf.float32), point_indices_f )
    point_indices = tf.rint( point_indices_f,'point_indices' )  # gpu_0/sa_layer3/point_indices:0
    point_indices_checkmin = tf.where( invalid_mask, tf.ones(shape=point_indices_f.shape,dtype=tf.float32)*tf.constant(999,dtype=tf.float32), point_indices, name='point_indices_checkmin' )

    # ------------------------------------------------------------------
    # check indice err
    Max_Assert_0 = 1e-4

    point_indices_err = tf.abs( point_indices - point_indices_f, name='point_indices_err' )     # gpu_0/sa_layer3/point_indices_err:0
    point_indices_maxerr = tf.reduce_max( point_indices_err, name='point_indices_maxerr_xyz' ) # gpu_0/sa_layer3/point_indices_maxerr_xyz:0
    check_point_indices = tf.assert_less( point_indices_maxerr, Max_Assert_0, data=[cascade_id, point_indices_maxerr],
                                            message='point indices in voxel check on cascade %d '%(cascade_id), name='check_point_indices' )
    tf.add_to_collection( 'check', check_point_indices )


    # check indice scope:
    # Actually only works when IS_merge_blocks_while_fix_bmap=False
    Max_Assert = 1e-4+5

    batch_size = new_points.shape[0].value
    block_num = new_points.shape[1].value
    point_num = new_points.shape[2].value
    channel_num = new_points.shape[3].value

    if self.dataset_name == 'MODELNET40':
        IsTolerateBug = 2
        IsTolerateBug = 0
    else:
        IsTolerateBug = 1

    if cascade_id==self.cascade_num-1:
        # only in this global cascde, the steps and strides in each dimension
        # can be different
        if self.dataset_name == 'MODELNET40' and self.global_step[0]==3.5:
            self.global_step = np.array( [2.3,2.3,2.3] )
        max_indice_f = ( np.abs(self.global_step) - np.array([1,1,1])*self.sub_block_step_candis[cascade_id-1] ) / (np.array([1,1,1])*self.sub_block_stride_candis[cascade_id-1])
        max_indice_v = np.rint( max_indice_f )
        if self.dataset_name != 'MODELNET40':
            assert np.sum(np.abs(max_indice_f-max_indice_v)) < Max_Assert
        max_indice_v += 1* IsTolerateBug

        voxel_size = max_indice_v.astype(np.int32)+1
        voxel_shape = [batch_size, block_num, voxel_size[0], voxel_size[1], voxel_size[2], channel_num]

        point_indices_checkmin = tf.identity(point_indices_checkmin, 'point_indices_checkmin_A') #
        point_indices_checkmin += (max_indice_v+2*IsTolerateBug) * IS_merge_blocks_while_fix_bmap
        point_indices_checkmin = tf.identity(point_indices_checkmin, 'point_indices_checkmin_B') # gpu_1/sa_layer4/point_indices_checkmin_B:0
        point_indices, first_unique_masks_global = unique_nd( point_indices )

        for i in range(3):
            real_max = tf.reduce_max(point_indices[:,:,:,i])
            check_max_indice = tf.assert_less( real_max - max_indice_v[i], tf.constant(Max_Assert + IS_merge_blocks_while_fix_bmap * max_indice_v[i], dtype=tf.float32 ),
                                              data=[cascade_id, i, real_max, max_indice_v[i]], name='check_max_indice_'+str(i) )
            tf.add_to_collection( 'check', check_max_indice )

    else:
        max_indice_f = ( self.sub_block_step_candis[cascade_id] - self.sub_block_step_candis[cascade_id-1] ) / self.sub_block_stride_candis[cascade_id-1]
        max_indice_v = np.rint( max_indice_f ).astype(np.float32)
        assert abs(max_indice_f-max_indice_v) < Max_Assert + IS_merge_blocks_while_fix_bmap * max_indice_v
        voxel_size = max_indice_v.astype(np.int32)+1
        voxel_shape = [batch_size, block_num, voxel_size, voxel_size, voxel_size, channel_num]

        max_indice_1 = tf.constant(max_indice_v,tf.float32)
        real_max = tf.reduce_max(point_indices)
        check_max_indice = tf.assert_less( real_max - max_indice_1, tf.constant(Max_Assert + IS_merge_blocks_while_fix_bmap * max_indice_v, tf.float32 ),
                                          data=[cascade_id, real_max, max_indice_1], name='check_max_indice' )
        tf.add_to_collection( 'check', check_max_indice )
        point_indices_checkmin += (max_indice_v) * IS_merge_blocks_while_fix_bmap + IsTolerateBug*1


    point_indices_min = tf.reduce_min(point_indices_checkmin, name='point_indices_min') # gpu_0/sa_layer4/point_indices_min:0
    check_min_indice = tf.assert_less( tf.constant(-Max_Assert, tf.float32),
                                      point_indices_min, data=[cascade_id,point_indices_min], name='check_min_indice' )
    tf.add_to_collection( 'check', check_min_indice )
    # ------------------------------------------------------------------
    point_indices = tf.cast( point_indices, tf.int32, name='point_indices' )    # gpu_0/sa_layer1/point_indices_1:0
    batch_idx = tf.reshape( tf.range(batch_size),[batch_size,1,1,1] )
    batch_idx = tf.tile( batch_idx, [1,block_num,point_num,1] )
    bn_idx = tf.reshape( tf.range(block_num),[1,block_num,1,1] )
    bn_idx = tf.tile( bn_idx, [batch_size,1,point_num,1] )
    point_indices = tf.concat( [batch_idx, bn_idx, point_indices], -1, name='point_indices' ) # gpu_0/sa_layer4/point_indices_1:0

    # Note: if point_indices have replicated items, the responding value will be multiplied which will lead to error!
    # For global cascade, the replicated indices can come from replicated aim
    # block of the last gs cascade. This should be solved while generating point_indices for global in this function.
    # For other cascades, the replicated indices can come from replicated points
    #       inside aim block in bidxmap file. This shoule be solved by add np.unique  while merging blocks in bidxmap.
    voxel_points = tf.scatter_nd( point_indices, new_points, shape=voxel_shape, name='voxel_points' )   # gpu_0/sa_layer1/voxel_points:0

    # check voxel: takes long time, only perform for debug
    check_points = tf.gather_nd( voxel_points, point_indices, name='check_points' ) # gpu_0/sa_layer4/check_points:0
    scatter_err = tf.abs( check_points - new_points) # gpu_0/sa_layer1/scatter_err:0
    scatter_err = scatter_err * tf.cast(invalid_mask[:,:,:,0:1], tf.float32)
    scatter_err = tf.identity( scatter_err, name='scatter_err'  )
    scatter_err_max = tf.reduce_max( scatter_err, name = 'scatter_err_max') # gpu_0/sa_layer1/scatter_err_max:0
    points_check = tf.assert_less( scatter_err_max, Max_Assert, data=[cascade_id, scatter_err_max], name='scatter_check' )
    if DEBUG_TMP and not IS_merge_blocks_while_fix_bmap:
        tf.add_to_collection( 'check', points_check )

    # ------------------------------------------------------------------
    new_voxel_shape = tf.concat( [ tf.constant([batch_size*block_num],tf.int32), voxel_shape[2:6] ],0 )
    voxel_points = tf.reshape( voxel_points, shape = new_voxel_shape )
    #if self.aug_types['RotateVox']:
    #    voxel_points = rotate_voxel_randomly( voxel_points, configs )

    if self.data_format == 'channels_first':
      voxel_points = tf.transpose(voxel_points, [0, 4, 1, 2, 3])
    return voxel_points
