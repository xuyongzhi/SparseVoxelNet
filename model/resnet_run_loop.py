# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf
import time

from official.resnet import resnet_model
from official.utils.flags import core as flags_core
from official.utils.export import export
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers
# pylint: enable=g-bad-import-order

# must be False when num_gpus>1
IsCheckNet = True

################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, data_net_configs, num_epochs=1,
                           num_gpus=None, examples_per_epoch=None):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
  #  # Shuffle the records. Note that we shuffle before repeating to ensure
  #  # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  if is_training and num_gpus and examples_per_epoch:
    total_examples = examples_per_epoch*num_epochs
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    dataset.take(total_batches * batch_size)

  if data_net_configs!=None and data_net_configs['precpu_sg']:
    from utils.grouping_sampling_voxelization import BlockGroupSampling
    bsg = BlockGroupSampling(data_net_configs['sg_settings'])
  else:
    bsg = None

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  dset_shapes = data_net_configs['dset_shape_idxs']
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: parse_record_fn(value, is_training, dset_shapes, bsg, is_normalize_pcl=False),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=True if num_gpus>1 else True))
  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return dataset


def get_synth_input_fn(height, width, num_channels, num_classes):
  """Returns an input function that returns a dataset with zeroes.

  This is useful in debugging input pipeline performance, as it removes all
  elements of file reading and image preprocessing.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):  # pylint: disable=unused-argument
    images = tf.zeros((batch_size, height, width, num_channels), tf.float32)
    labels = tf.zeros((batch_size), tf.int32)
    return tf.data.Dataset.from_tensors((images, labels)).repeat()

  return input_fn


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, lr_decay_rates,
    bn_decay_rates, initial_learning_rate, initial_bndecay ):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.

  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  #initial_learning_rate = initial_learning_rate * batch_size / batch_denom
  #initial_bndecay = 1 - (1-initial_bndecay) * batch_size / batch_denom
  initial_learning_rate = initial_learning_rate
  initial_bndecay = initial_bndecay

  batches_per_epoch = num_images / batch_size

  # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  lr_vals = [max(initial_learning_rate * decay, 1e-5) for decay in lr_decay_rates]

  lr_warmup = lr_decay_rates[0] < lr_decay_rates[1]
  bndecay_vals = [min( 1-(1-initial_bndecay) * decay, 0.99) for decay in bn_decay_rates]
  if lr_warmup:
    boundaries_bnd = boundaries[1:len(boundaries)]
  else:
    boundaries_bnd = boundaries

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, lr_vals)

  def bndecay_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries_bnd, bndecay_vals)

  return learning_rate_fn, bndecay_fn, lr_vals, bndecay_vals

def resnet_model_fn(model_flag, features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, resnet_version, loss_scale,
                    loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE, data_net_configs={}):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    resnet_version: Integer representing which version of the ResNet network to
      use. See README for details. Valid values: [1, 2]
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    dtype: the TensorFlow dtype to use for calculations.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  assert data_format == 'channels_last'
  dataset_name = data_net_configs['dataset_name']
  model = model_class(model_flag, resnet_size, data_format, resnet_version=resnet_version,
                      dtype=dtype, data_net_configs=data_net_configs)

  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)

  predictions = {
      'classes': tf.argmax(logits, axis=-1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if IsCheckNet:
    add_check(predictions)

  eval_views = data_net_configs['eval_views']
  if eval_views>1: # eval multi views
    eval_views = logits.shape[1].valu
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    logits = tf.reduce_min(logits, 1)
  else:
    eval_views = 1

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  dset_shape_idxs = data_net_configs['dset_shape_idxs']
  category_idx = dset_shape_idxs['indices']['labels']['label_category'][0]
  labels_shape = labels.shape
  if len(labels_shape) == 4:
    labels = tf.reshape(labels, [-1, labels_shape[2], labels_shape[3]])
  labels = labels[..., category_idx]
  assert len(labels.shape) == len(logits.shape)-1, "network error"
  if data_net_configs['loss_lw_gama'] < 0:
    weights = 1
  else:
    weights = tf.gather( data_net_configs['label_num_weights'], labels)
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels, weights=weights)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  for v in tf.trainable_variables():
    try:
      l2loss = tf.nn.l2_loss(tf.cast(v, tf.float32))
    except:
      print(v)
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
      if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    if data_net_configs['optimizer'] == 'momentum':
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=momentum
      )
    elif data_net_configs['optimizer'] == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
      raise NotImplementedError

    if loss_scale != 1:
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      minimize_op = optimizer.minimize(loss, global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
  else:
    train_op = None

  if not tf.contrib.distribute.has_distribution_strategy():
    if eval_views>1:
      labels_ = tf.tile(tf.expand_dims(labels,1), [1,eval_views])
    else:
      labels_ = labels
    accuracy = tf.metrics.accuracy(labels_, predictions['classes'])
    accuracies = []
    if eval_views>1:
      for i in range(eval_views):
        accuracies.append( tf.metrics.accuracy(labels, predictions['classes'][:,i]) )

    #pred_classes = tf.reduce_mean(predictions['classes'], -1, keepdims=True)
    #accuracy = tf.metrics.accuracy(labels, pred_classes)
  else:
    # Metrics are currently not compatible with distribution strategies during
    # training. This does not affect the overall performance of the model.
    accuracy = (tf.no_op(), tf.constant(0))
  metrics = {'accuracy': accuracy}
  if eval_views>1:
    for i in range(eval_views):
      metrics['accuracy%d'%(i)] = accuracies[i]

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def per_device_batch_size(batch_size, num_gpus):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.

  Note that this should eventually be handled by DistributionStrategies
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.

  Args:
    batch_size: Global batch size to be divided among devices. This should be
      equal to num_gpus times the single-GPU batch_size for multi-gpu training.
    num_gpus: How many GPUs are used with DistributionStrategies.

  Returns:
    Batch size per device.

  Raises:
    ValueError: if batch_size is not divisible by number of devices
  """
  if num_gpus <= 1:
    return batch_size

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. Found {} '
           'GPUs with a batch size of {}; try --batch_size={} instead.'
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)
  return int(batch_size / num_gpus)


def add_check(predictions):
  inputs = tf.get_collection('raw_inputs_COLC')
  predictions['inputs'] = inputs[0]

  block_bottom_center_COLC = tf.get_collection('block_bottom_center_COLC')
  new_xyz_COLCs = tf.get_collection('new_xyz_COLC')
  grouped_xyz_COLCs = tf.get_collection('grouped_xyz_COLC')
  cascade_num = len(new_xyz_COLCs)
  for i in range(cascade_num):
    predictions['new_xyz_%d'%(i)] = new_xyz_COLCs[i]
    predictions['grouped_xyz_%d'%(i)] = grouped_xyz_COLCs[i]
    predictions['block_bottom_center_%d'%(i)] = block_bottom_center_COLC[i]

  voxel_indices_COLC = tf.get_collection('voxel_indices_COLC')
  for i in range( len(voxel_indices_COLC) ):
    predictions['voxel_indices_%d'%(i)] = voxel_indices_COLC[i]

def check_net(classifier, input_fn_eval, dataset_name, data_net_configs):
  k_start = 0
  N = 6
  res_dir = '/tmp/check_net'
  gen_inputs = True
  gen_new_xyz = False
  gen_grouped_xyz = False
  if gen_grouped_xyz:
    gen_box_to_grouped = True
  gen_grouped_xyz_subblock = False
  gen_voxel_indices = False
  max_subblock_num = 10

  if not os.path.exists(res_dir):
    os.makedirs(res_dir)
  from ply_util import create_ply_dset, draw_points_and_voxel_indices,\
                       draw_blocks_by_bot_cen_top
  pred_results = classifier.predict(input_fn=input_fn_eval)
  check_items = []
  if gen_inputs:
    check_items.append('inputs')
  cascade_num = len( data_net_configs['block_params']['filters'])
  for i in range(cascade_num):
    if gen_new_xyz:
      check_items.append('new_xyz_%d'%(i))
    if gen_grouped_xyz or gen_grouped_xyz_subblock:
      check_items.append('grouped_xyz_%d'%(i))

  for j, pred in enumerate(pred_results):
    if j < k_start:
      continue
    # gen block box   *****************************************************
    if 'block_bottom_center_0' in pred:
      for v in range(cascade_num):
        block_bottom_center = pred['block_bottom_center_%d'%(v)]
        ply_fn = '{}/{}_block_{}.ply'.format(res_dir, j, v)
        draw_blocks_by_bot_cen_top(ply_fn, block_bottom_center)

    # gen voxel edges *****************************************************
    if 'voxel_indices_0' in pred:
      for v in range(cascade_num-1):
        grouped_xyz = pred['grouped_xyz_%d'%(v+1)] # (240, 27, 3)
        grouped_voxel_indices = pred['voxel_indices_%d'%(v)] # (240, 27, 3)

        dir_k = res_dir+'/%d_grouped_xyz_%d'%(j,v+1)
        if not os.path.exists(dir_k):
          os.makedirs(dir_k)

        for k in range(min(grouped_xyz.shape[0], max_subblock_num)):
          ply_fn = '{}/edge_{}.ply'.format(dir_k, k)
          draw_points_and_voxel_indices(ply_fn, grouped_xyz[k], grouped_voxel_indices[k])

    ############################################################################
    checks = {}
    for item in check_items:
      if item not in pred:
        continue
      checks[item] = pred[item]
      # gen_grouped_xyz_subblock *******************************************
      if 'grouped' in item and gen_grouped_xyz_subblock:
        data = checks[item]
        for k in range(min(max_subblock_num,data.shape[0])):
          dir_k = res_dir+'/%d_%s'%(j,item)
          if not os.path.exists(dir_k):
            os.makedirs(dir_k)
          ply_fn = '{}/{}.ply'.format(dir_k, k)
          create_ply_dset(dataset_name, checks[item][k], ply_fn, extra='random_same_color')

      # gen_grouped_xyz, gen_new_xyz ****************************************
      ply_fn = '{}/{}_{}.ply'.format(res_dir, j, item)
      if 'grouped_xyz' in item and gen_box_to_grouped:
        create_ply_dset(dataset_name, checks[item], ply_fn,  extra = 'random_same_color')
      else:
        create_ply_dset(dataset_name, checks[item], ply_fn,  extra = 'random_same_color')


    if j==N+k_start-1:
      print('no more')
      break
  print('ply finished')
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  pass

def resnet_main(
    flags_obj, model_function, input_function, dataset_name, data_net_configs):
  """Shared main loop for ResNet Models.

  Args:
    flags_obj: An object containing parsed flags. See define_resnet_flags()
      for details.
    model_function: the function that instantiates the Model and builds the
      ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
      dataset that the estimator can train on. This will be wrapped with
      all the relevant flags for running and passed to estimator.
    dataset_name: the name of the dataset for training and evaluation. This is
      used for logging purpose.
    shape: list of ints representing the shape of the images used for training.
      This is only used if flags_obj.export_dir is passed.
  """

  IsMetricLog = True   # temporally used
  OnlyEval = data_net_configs['only_eval']
  if IsMetricLog:
    metric_log_fn = os.path.join(flags_obj.model_dir, 'log_metric.txt')
    metric_log_f = open(metric_log_fn, 'a')

  from tensorflow.contrib.memory_stats.ops import gen_memory_stats_ops
  max_memory_usage = gen_memory_stats_ops.max_bytes_in_use()

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Create session config based on values of inter_op_parallelism_threads and
  # intra_op_parallelism_threads. Note that we default to having
  # allow_soft_placement = True, which is required for multi-GPU and not
  # harmful for other modes.
  session_config = tf.ConfigProto(
      inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
      intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
      gpu_options = tf.GPUOptions(allow_growth = True),
      allow_soft_placement=True)

  if flags_core.get_num_gpus(flags_obj) == 0:
    distribution = tf.contrib.distribute.OneDeviceStrategy('device:CPU:0')
  elif flags_core.get_num_gpus(flags_obj) == 1:
    distribution = tf.contrib.distribute.OneDeviceStrategy('device:GPU:%d'%(flags_obj.gpu_id))
  else:
    distribution = tf.contrib.distribute.MirroredStrategy(
        num_gpus=flags_core.get_num_gpus(flags_obj)
    )

  run_config = tf.estimator.RunConfig(train_distribute=distribution,
                                      session_config=session_config)

  classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
      params={
          'model_flag': flags_obj.model_flag,
          'resnet_size': flags_obj.resnet_size,
          'data_format': flags_obj.data_format,
          'batch_size': flags_obj.batch_size,
          'resnet_version': int(flags_obj.resnet_version),
          'loss_scale': flags_core.get_loss_scale(flags_obj),
          'dtype': flags_core.get_tf_dtype(flags_obj),
          'data_net_configs': data_net_configs
      })

  run_params = {
      'batch_size': flags_obj.batch_size,
      'dtype': flags_core.get_tf_dtype(flags_obj),
      'resnet_size': flags_obj.resnet_size,
      'resnet_version': flags_obj.resnet_version,
      'synthetic_data': flags_obj.use_synthetic_data,
      'train_epochs': flags_obj.train_epochs,
  }
  benchmark_logger = logger.config_benchmark_logger(flags_obj)
  benchmark_logger.log_run_info('resnet', dataset_name, run_params)

  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      batch_size=flags_obj.batch_size)

  def input_fn_train():
    return input_function(
        is_training=True, data_dir=flags_obj.data_dir,
        batch_size=per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        data_net_configs = data_net_configs,
        num_epochs=flags_obj.epochs_between_evals)
  eval_views = data_net_configs['eval_views']

  def input_fn_eval():
    return input_function(
        is_training=False, data_dir=flags_obj.data_dir,
        batch_size=per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)) // eval_views,
        data_net_configs = data_net_configs,
        num_epochs=1)

  if OnlyEval:
    total_training_cycle = 1
  else:
    total_training_cycle = (flags_obj.train_epochs //
                            flags_obj.epochs_between_evals)

    if IsCheckNet:
      check_net(classifier, input_fn_eval, dataset_name, data_net_configs)
    # train for one step to check max memory usage
    train_res = classifier.train(input_fn=input_fn_train, hooks=train_hooks, steps=10)

    with tf.Session() as sess:
      max_memory_usage_v = sess.run(max_memory_usage)
      tf.logging.info('\n\nmemory usage: %0.3f G\n\n'%(max_memory_usage_v*1.0/1e9))

  for cycle_index in range(total_training_cycle):
    tf.logging.info('\n\n\nStarting a training cycle: %d/%d\n\n',
                    cycle_index, total_training_cycle)
    eval_train_steps = 80
    if not OnlyEval:
      t0 = time.time()
      classifier.train(input_fn=input_fn_train, hooks=train_hooks,
                      max_steps=flags_obj.max_train_steps)
      train_t = (time.time()-t0)/flags_obj.epochs_between_evals


      #Temporally used before metric in training is not supported in distribution
      tf.logging.info('Starting to evaluate train data.')
      t0 = time.time()
      train_eval_results = classifier.evaluate(input_fn=input_fn_train,
                                        steps=eval_train_steps, name='train')
      eval_train_t = time.time() - t0
      if train_eval_results['global_step'] >= flags_obj.max_train_steps:
        print('global_step: {} reaches max stop now'.format(train_eval_results['global_step']))
        break

    else:
      train_t = 0
      eval_train_t = 0
      train_eval_results = {}
      train_eval_results['accuracy'] = 0
      train_eval_results['loss'] = 0

    tf.logging.info('Starting to evaluate.')
    # flags_obj.max_train_steps is generally associated with testing and
    # profiling. As a result it is frequently called with synthetic data, which
    # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
    # eval (which is generally unimportant in those circumstances) to terminate.
    # Note that eval will run for max_train_steps each loop, regardless of the
    # global_step count.
    t0 = time.time()

    eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                        name='test')
    eval_t = time.time() - t0

    benchmark_logger.log_evaluation_result(eval_results)
    if IsMetricLog:
      if eval_views>1:
        eval_acu_str = '{:.3f}-{}_{:.3f}'.format(eval_results['accuracy0'], eval_views, eval_results['accuracy'])
      else:
        eval_acu_str = '{:.3f}'.format(eval_results['accuracy'])
      metric_log_f.write('epoch loss accuracy: {} {:.3f}/{:.3f}--{:.3f}/{}\n'.format(\
          int(eval_results['global_step']/flags_obj.steps_per_epoch), train_eval_results['loss'], eval_results['loss'],\
          train_eval_results['accuracy'], eval_acu_str))
      if eval_views>1:
        accuracies = [eval_results['accuracy%d'%(i)] for i in range(eval_views)]
        accuracies_str = ['%0.3f'%(a) for a in accuracies]
        metric_log_f.write('eval multi view: %s\n'%(accuracies_str))
      metric_log_f.write('train t:{:.3f}sec    eval train t:{:.3f}sec({}steps)    eval t:{:.3f} eval_views:{}\n\n'.format(
            train_t, eval_train_t, eval_train_steps, eval_t, eval_views))
      metric_log_f.flush()

    if model_helpers.past_stop_threshold(
        flags_obj.stop_threshold, eval_results['accuracy']):
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      break

  if flags_obj.export_dir is not None:
    # Exports a saved model for the given classifier.
    input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
        shape, batch_size=flags_obj.batch_size)
    classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn)


def define_resnet_flags(resnet_size_choices=None):
  """Add flags and validators for ResNet."""
  flags_core.define_base()
  flags_core.define_performance(num_parallel_calls=False)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_enum(
      name='resnet_version', short_name='rv', default='2',
      enum_values=['1', '2'],
      help=flags_core.help_wrap(
          'Version of ResNet. (1 or 2) See README.md for details.'))

  #choice_kwargs = dict(
  #    name='resnet_size', short_name='rs', default='50',
  #    help=flags_core.help_wrap('The size of the ResNet model to use.'))

  #if resnet_size_choices is None:
  #  flags.DEFINE_string(**choice_kwargs)
  #else:
  #  flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)

  # The current implementation of ResNet v1 is numerically unstable when run
  # with fp16 and will produce NaN errors soon after training begins.
  msg = ('ResNet version 1 is not currently supported with fp16. '
         'Please use version 2 instead.')
  @flags.multi_flags_validator(['dtype', 'resnet_version'], message=msg)
  def _forbid_v1_fp16(flag_values):  # pylint: disable=unused-variable
    return (flags_core.DTYPE_MAP[flag_values['dtype']][0] != tf.float16 or
            flag_values['resnet_version'] != '1')
