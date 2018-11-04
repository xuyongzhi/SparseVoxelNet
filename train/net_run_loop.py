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
parsing. Code for defining the ResNet layers can be found in meshnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.export import export
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
import numpy as np
from utils.tf_util import TfUtil
# pylint: enable=g-bad-import-order
from models.meshnet_model import DEFAULT_DTYPE
from train import tfmetric

################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, dset_shape_idx, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None, sg_settings=None):
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
    num_gpus: The number of gpus used for training.
    examples_per_epoch: The number of examples in an epoch.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  if is_training and num_gpus and examples_per_epoch:
    total_examples = num_epochs * examples_per_epoch
    # Force the number of batches to be divisible by the number of devices.
    # This prevents some devices from receiving batches while others do not,
    # which can lead to a lockup. This case will soon be handled directly by
    # distribution strategies, at which point this .take() operation will no
    # longer be needed.
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    dataset.take(total_batches * batch_size)

  if sg_settings is None:
    bsg = None
  else:
    from utils.grouping_sampling_voxelization import BlockGroupSampling
    bsg = BlockGroupSampling(sg_settings)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: parse_record_fn(value, is_training, dset_shape_idx, bsg=bsg),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=True))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return dataset


def get_synth_input_fn(height, width, num_channels, num_classes,
                       dtype=tf.float32):
  """Returns an input function that returns a dataset with random data.

  This input_fn returns a data set that iterates over a set of random data and
  bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
  copy is still included. This used to find the upper throughput bound when
  tunning the full input pipeline.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  # pylint: disable=unused-argument
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
    """Returns dataset filled with random data."""
    # Synthetic input should be within [0, 255].
    inputs = tf.truncated_normal(
        [batch_size] + [height, width, num_channels],
        dtype=dtype,
        mean=127,
        stddev=60,
        name='synthetic_inputs')

    labels = tf.random_uniform(
        [batch_size],
        minval=0,
        maxval=num_classes - 1,
        dtype=tf.int32,
        name='synthetic_labels')
    data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
    data = data.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return data

  return input_fn

################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, batches_per_epoch, boundary_epochs, lr_decay_rate,
    base_lr=0.1, warmup=False, base_bnd=0.99, bnd_decay_rate=0.1, net_configs=None):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.
    base_lr: Initial learning rate scaled based on batch_denom.
    warmup: Run a 5 epoch warmup to the initial lr.
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = base_lr * batch_size / batch_denom
  initial_bnd_rate = base_bnd

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  n = len(boundaries)
  lr_vals = [initial_learning_rate * pow(lr_decay_rate, i) for i in range(n+1)]
  lr_vals = [max(l, 6e-5) for l in lr_vals]
  bnd_vals = [(1-initial_bnd_rate) * pow(bnd_decay_rate, i) for i in range(n+1)]
  bnd_vals = [min(1-d, 0.999) for d in bnd_vals]

  def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 5 epoch warm up."""
    lr = tf.train.piecewise_constant(global_step, boundaries, lr_vals)
    if warmup:
      warmup_steps = int(2 * 5)
      warmup_lr = (
          initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
              warmup_steps, tf.float32))
      return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
    return lr

  def bn_decay_fn(global_step):
    bnd = tf.train.piecewise_constant(global_step, boundaries, bnd_vals)
    return bnd

  net_configs['boundaries'] = boundaries
  net_configs['lr_vals'] = lr_vals
  net_configs['bnd_vals'] = bnd_vals
  return learning_rate_fn, bn_decay_fn

def get_cm_metric():
  from tensorflow.python.ops import state_ops
  total_cm = tf.get_default_graph().get_tensor_by_name('mean_iou/total_confusion_matrix:0')
  cm_metric = (total_cm, tf.identity(total_cm))
  return cm_metric

def net_model_fn( features, labels, mode, model_class,
                  net_data_configs,
                  weight_decay, learning_rate_fn, momentum,
                  data_format, loss_scale,
                  loss_filter_fn=None, dtype=DEFAULT_DTYPE,
                  fine_tune=False):
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
    fine_tune: If True only train the dense layers(final layers).

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  labels = tf.cast(labels, tf.int32)
  #from datasets.tfrecord_util import get_ele
  #labels = tf.squeeze(get_ele(features, 'label_category', net_data_configs['dset_shape_idx']), 2)

  model = model_class(net_data_configs=net_data_configs,
                      data_format=data_format, dtype=dtype)

  logits, label_weight = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)

  predictions = {
      'classes': tf.argmax(logits, axis=-1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
      'labels': labels,
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    from models.meshnet_model import ele_in_feature
    dset_shape_idx = net_data_configs['dset_shape_idx']
    xyz = ele_in_feature(features, 'xyz', dset_shape_idx)
    vidx_per_face = ele_in_feature(features, 'vidx_per_face', dset_shape_idx)
    predictions.update({'xyz': xyz, 'vidx_per_face': vidx_per_face,
                       'valid_num_face': features['valid_num_face'] })
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        #export_outputs={
        #    'predict': tf.estimator.export.PredictOutput(predictions) }
        )

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  check_loss1 = tf.assert_equal( tf.is_nan(loss), False, message="\nGot nan loss\n")
  with tf.control_dependencies([check_loss1]):
    loss = tf.identity(loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer_alg = net_data_configs['net_configs']['optimizer']
    if optimizer_alg  == 'momentum':
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=momentum    )
    elif optimizer_alg  == 'adam':
      optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate )

    def _dense_grad_filter(gvs):
      """Only apply gradient updates to the final layer.

      This function is used for fine tuning.

      Args:
        gvs: list of tuples with gradients and variable info
      Returns:
        filtered gradients so that only the dense layer remains
      """
      return [(g, v) for g, v in gvs if 'dense' in v.name]

    if loss_scale != 1:
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      raise NotImplementedError
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      if fine_tune:
        scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      grad_vars = optimizer.compute_gradients(loss)
      if fine_tune:
        grad_vars = _dense_grad_filter(grad_vars)
      minimize_op = optimizer.apply_gradients(grad_vars, global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(labels, predictions['classes'])
  dset_metas = net_data_configs['dset_metas']
  num_classes = dset_metas.num_classes
  mean_iou = tf.metrics.mean_iou(labels, predictions['classes'], num_classes)

  MODEL_DR = net_data_configs['data_configs']['model_dir']
  classes_names = [dset_metas.label2class[i] for i in range(num_classes)]
  confusionMatrixSaveHook = tfmetric.SaverHook(classes_names,
                     'mean_iou/total_confusion_matrix',
                     tf.summary.FileWriterCache.get(MODEL_DR + "/eval"))

  mean_iou = (mean_iou[0], tf.identity(mean_iou[1]))
  cm_metric = get_cm_metric()

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('accuracy', accuracy[1])
  tf.summary.scalar('mean_iou', mean_iou[0])

  metrics = {'accuracy': accuracy, 'mean_iou':mean_iou, 'cm':cm_metric}

  return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op,
          eval_metric_ops=metrics,
          evaluation_hooks=[confusionMatrixSaveHook])

def compute_mean_iou(total_cm, num_classes, name='mean_iou'):
  """Compute the mean intersection-over-union via the confusion matrix."""
  sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
  sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
  cm_diag = tf.to_float(tf.diag_part(total_cm))
  denominator = sum_over_row + sum_over_col - cm_diag

  # The mean is only computed over classes that appear in the
  # label or prediction tensor. If the denominator is 0, we need to
  # ignore the class.
  num_valid_entries = tf.reduce_sum(tf.cast(
      tf.not_equal(denominator, 0), dtype=tf.float32))

  # If the value of the denominator is 0, set it to 1 to avoid
  # zero division.
  denominator = tf.where(
      tf.greater(denominator, 0),
      denominator,
      tf.ones_like(denominator))
  iou = tf.div(cm_diag, denominator)

  for i in range(num_classes):
    tf.identity(iou[i], name='train_iou_class{}'.format(i))
    #tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

  # If the number of valid entries is 0 (no classes) we return 0.
  result = tf.where(
      tf.greater(num_valid_entries, 0),
      tf.reduce_sum(iou, name=name) / num_valid_entries,
      0)
  return result, iou

def net_main_check(
    flags_obj, model_class, input_function, net_data_configs, shape=None):
  tf.enable_eager_execution()
  from datasets.tfrecord_util import get_ele
  import numpy as np

  model_cls = model_class(net_data_configs)
  dset_shape_idx = net_data_configs['dset_shape_idx']

  #with tf.Graph().as_default():
  if True:
    def input_fn_train(num_epochs):
      return input_function(
          is_training=True, data_dir=flags_obj.data_dir,
          batch_size=distribution_utils.per_device_batch_size(
              flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
          num_epochs=num_epochs,
          num_gpus=flags_core.get_num_gpus(flags_obj),
          examples_per_epoch = flags_obj.examples_per_epoch,
          sg_settings = net_data_configs['sg_settings']
          )
    dataset = input_fn_train(1)
    ds_iterator = dataset.make_one_shot_iterator()

    for i in range(1000):
      t0 = time.time()
      features, labels = ds_iterator.get_next()
      t = time.time() - t0
      print('t: {} ms'.format(t*1000))
      #model_cls.main_test_pool(features)

      items = ['xyz', 'color', 'vidx_per_face', 'edgev_per_vertex','valid_ev_num_pv']
      datas = {}
      for item in items:
        d = get_ele(features, item, dset_shape_idx)
        if d is not None:
          datas[item] = d

      sg_params = {}
      items = ['grouped_pindex', 'flatting_idx']
      for s in range(net_data_configs['sg_settings']['num_sg_scale']):
        for item in items:
          if s==0:
            sg_params[item] = []
          else:
            sg_params[item].append(features[item+'_%d'%(s)])
            min_v = np.min(sg_params[item][-1])
            assert min_v >= 0
      pass


def net_main(
    flags_obj, model_function, input_function, net_data_configs, shape=None):
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

  model_helpers.apply_clean(flags.FLAGS)
  is_metriclog = True
  if is_metriclog:
    metric_logfn = os.path.join(flags_obj.model_dir, 'log_metric.txt')
    metric_logf = open(metric_logfn, 'a')

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
      allow_soft_placement=True)

  distribution_strategy = distribution_utils.get_distribution_strategy(
      flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

  run_config = tf.estimator.RunConfig(
      train_distribute=distribution_strategy, session_config=session_config)

  # initialize our model with all but the dense layer from pretrained resnet
  if flags_obj.pretrained_model_checkpoint_path is not None:
    warm_start_settings = tf.estimator.WarmStartSettings(
        flags_obj.pretrained_model_checkpoint_path,
        vars_to_warm_start='^(?!.*dense)')
  else:
    warm_start_settings = None

  classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
      warm_start_from=warm_start_settings, params={
          'data_format': flags_obj.data_format,
          'batch_size': flags_obj.batch_size,
          'loss_scale': flags_core.get_loss_scale(flags_obj),
          'weight_decay': flags_obj.weight_decay,
          'dtype': flags_core.get_tf_dtype(flags_obj),
          'fine_tune': flags_obj.fine_tune,
          'examples_per_epoch': flags_obj.examples_per_epoch,
          'net_data_configs': net_data_configs
      })

  run_params = {
      'batch_size': flags_obj.batch_size,
      'dtype': flags_core.get_tf_dtype(flags_obj),
      'synthetic_data': flags_obj.use_synthetic_data,
      'train_epochs': flags_obj.train_epochs,
  }
  dataset_name = net_data_configs['dataset_name']
  if flags_obj.use_synthetic_data:
    dataset_name = dataset_name + '-synthetic'

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info('meshnet', dataset_name, run_params,
                                test_id=flags_obj.benchmark_test_id)

  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      model_dir=flags_obj.model_dir,
      batch_size=flags_obj.batch_size)

  def input_fn_train(num_epochs):
    return input_function(
        is_training=True, data_dir=flags_obj.data_dir,
        batch_size=distribution_utils.per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=num_epochs,
        num_gpus=flags_core.get_num_gpus(flags_obj),
        examples_per_epoch = flags_obj.examples_per_epoch,
        sg_settings = net_data_configs['sg_settings']
        )

  def input_fn_eval():
    return input_function(
        is_training=False, data_dir=flags_obj.data_dir,
        batch_size=distribution_utils.per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=1,
        sg_settings = net_data_configs['sg_settings']
        )

  if flags_obj.eval_only or flags_obj.pred_ply or not flags_obj.train_epochs:
    # If --eval_only is set, perform a single loop with zero train epochs.
    schedule, n_loops = [0], 1
  else:
    # Compute the number of times to loop while training. All but the last
    # pass will train for `epochs_between_evals` epochs, while the last will
    # train for the number needed to reach `training_epochs`. For instance if
    #   train_epochs = 25 and epochs_between_evals = 10
    # schedule will be set to [10, 10, 5]. That is to say, the loop will:
    #   Train for 10 epochs and then evaluate.
    #   Train for another 10 epochs and then evaluate.
    #   Train for a final 5 epochs (to reach 25 epochs) and then evaluate.
    n_loops = math.ceil(flags_obj.train_epochs / flags_obj.epochs_between_evals)
    schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
    schedule[-1] = flags_obj.train_epochs - sum(schedule[:-1])  # over counting.

    classifier.train(input_fn=lambda: input_fn_train(1) ,hooks=train_hooks, max_steps=10)
    with tf.Session() as sess:
      max_memory_usage_v = sess.run(max_memory_usage)
      tf.logging.info('\n\nmemory usage: %0.3f G\n\n'%(max_memory_usage_v*1.0/1e9))

  best_acc, best_acc_checkpoint = load_saved_best(flags_obj.model_dir)
  for cycle_index, num_train_epochs in enumerate(schedule):
    tf.logging.info('Starting cycle: %d/%d', cycle_index, int(n_loops))

    t0 = time.time()
    train_t = 0
    if num_train_epochs:
      classifier.train(input_fn=lambda: input_fn_train(num_train_epochs),
                       hooks=train_hooks, max_steps=flags_obj.max_train_steps)
      train_t = (time.time()-t0)/num_train_epochs

    tf.logging.info('Starting to evaluate.')

    # flags_obj.max_train_steps is generally associated with testing and
    # profiling. As a result it is frequently called with synthetic data, which
    # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
    # eval (which is generally unimportant in those circumstances) to terminate.
    # Note that eval will run for max_train_steps each loop, regardless of the
    # global_step count.
    only_train = False and (not flags_obj.eval_only) and (not flags_obj.pred_ply)
    if not only_train:
      t0 = time.time()
      eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                        steps=flags_obj.max_train_steps,)
                                        #checkpoint_path=best_acc_checkpoint)
      eval_t = time.time()-t0

      if flags_obj.pred_ply:
        pred_generator = classifier.predict(input_fn=input_fn_eval)
        num_classes = net_data_configs['dset_metas'].num_classes
        gen_pred_ply(eval_results, pred_generator, flags_obj.model_dir, num_classes)

      benchmark_logger.log_evaluation_result(eval_results)

      if model_helpers.past_stop_threshold(
          flags_obj.stop_threshold, eval_results['accuracy']):
        break

      cur_is_best = ''
      if num_train_epochs and  eval_results['accuracy'] > best_acc:
        best_acc = eval_results['accuracy']
        save_cur_model_as_best_acc(flags_obj.model_dir, best_acc)
        cur_is_best = 'best'
      global_step = cur_global_step(flags_obj.model_dir)
      epoch = int( global_step / flags_obj.examples_per_epoch * flags_obj.num_gpus)
      ious_str = get_ious_str( eval_results['cm'], net_data_configs['dset_metas'], eval_results['mean_iou'])
      metric_logf.write('{} train t:{:.1f}  eval t:{:.1f} \teval acc:{:.3f} \tmean_iou:{:.3f} {} {}\n'.format(epoch,
                        train_t, eval_t, eval_results['accuracy'], eval_results['mean_iou'], cur_is_best, ious_str))
      metric_logf.flush()
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      pass

  if flags_obj.export_dir is not None:
    # Exports a saved model for the given classifier.
    input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
        shape, batch_size=flags_obj.batch_size)
    classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn)


def cur_global_step(model_dir):
  import os
  cur_model_path = tf.train.latest_checkpoint(model_dir)
  cur_name = os.path.basename(cur_model_path)
  global_step = int(cur_name.split('-')[1])
  return global_step

def load_saved_best(model_dir):
  bafn = model_dir+'/best_accuracy.txt'
  best_acc_checkpoint = model_dir + '/best_acc'
  if not os.path.exists(bafn) or not os.path.exists(best_acc_checkpoint+'.meta'):
    return 0, None
  with open(bafn, 'r') as bf:
    for line in bf:
      best_acc = float(line.strip())
      return best_acc, best_acc_checkpoint


def save_cur_model_as_best_acc(model_dir, best_acc):
  import glob, os, shutil
  cur_model_path = tf.train.latest_checkpoint(model_dir)
  cur_name = os.path.basename(cur_model_path)
  global_step = int(cur_name.split('-')[1])
  cur_fns = glob.glob(cur_model_path+'*')
  new_fns = [fn.replace(cur_name, 'best_acc') for fn in cur_fns]
  for i in range(3):
    shutil.copyfile(cur_fns[i], new_fns[i])

  with open(model_dir+'/best_accuracy.txt', 'w') as bf:
    bf.write(str(best_acc))


def define_net_flags():
  """Add flags and validators for ResNet."""
  flags_core.define_base()
  flags_core.define_performance(num_parallel_calls=False)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  #flags.DEFINE_enum(
  #    name='resnet_version', short_name='rv', default='1',
  #    enum_values=['1', '2'],
  #    help=flags_core.help_wrap(
  #        'Version of ResNet. (1 or 2) See README.md for details.'))
  flags.DEFINE_bool(
      name='fine_tune', short_name='ft', default=False,
      help=flags_core.help_wrap(
          'If True do not train any parameters except for the final layer.'))
  flags.DEFINE_string(
      name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
      help=flags_core.help_wrap(
          'If not None initialize all the network except the final layer with '
          'these values'))
  flags.DEFINE_boolean(
      name='eval_only', default=False,
      help=flags_core.help_wrap('Skip training and only perform evaluation on '
                                'the latest checkpoint.'))


def get_ious_str(total_cm, dset_metas, mean_iou0):
  mean_iou, ious, label_hist, pred_hist = compute_iou_cm_np(total_cm)
  assert abs(mean_iou - mean_iou0) < 0.01
  classes_names = [dset_metas.label2class[i] for i in range(dset_metas.num_classes)]
  cs_str = '\n%8s'%('classes:')+ ' '.join([cs[0:3] for cs in classes_names])
  def f2intstr(fls, name):
    return '\n%8s'%(name+': ')  + ' '.join(['%3d'%(int(i*1000)) for i in fls])
  ious_str = f2intstr(ious, 'ious')
  label_hist = f2intstr(label_hist, 'lhist')
  pred_hist = f2intstr(pred_hist, 'phist')
  classes_str = cs_str + ious_str + label_hist + pred_hist
  print(classes_str)
  return classes_str

def compute_iou_cm_np(total_cm):
  sum_over_row = np.sum(total_cm,0) # label
  sum_over_col = np.sum(total_cm,1) # prediction
  cm_diag = np.diag(total_cm)
  denominator = sum_over_row + sum_over_col - cm_diag

  num_valid_entries = np.sum(denominator!=0)
  assert num_valid_entries > 0
  denominator += (denominator==0).astype(np.float64)
  ious = cm_diag / denominator
  mean_iou = np.sum(ious) / num_valid_entries

  # histgram
  label_hist = sum_over_row / np.sum(sum_over_row)
  pred_hist = sum_over_col / np.sum(sum_over_col)
  return mean_iou, ious, label_hist, pred_hist


def compute_iou_np(pred, label, num_classes, weighted=False):
  assert pred.shape == label.shape

  I = np.zeros(num_classes)
  U = np.zeros(num_classes)
  for c in range(num_classes):
    pred_c = pred == c
    label_c = label == c
    I[c] = float(np.sum(np.logical_and(label_c, pred_c)))
    U[c] = float(np.sum(np.logical_or(label_c, pred_c)))

  iou_classes = I / (U+1e-5)

  if weighted:
    weights,_bins = np.histogram(label, range(num_classes+1))
    weights = weights*1.0/label.shape[0]
    mean_iou_weighted = np.sum(iou_classes * weights)
    mean_iou = mean_iou_weighted
  else:
    non_zero_num = np.sum(iou_classes!=0) *1.0
    mean_iou = np.sum(iou_classes) / non_zero_num
  return mean_iou, iou_classes

def gen_pred_ply(eval_results, pred_generator, model_dir, num_classes):
  from utils.ply_util import gen_mesh_ply

  k = 0
  for pred in pred_generator:
    k += 1

    valid_num_face = int( pred['valid_num_face'] )
    classes = pred['classes'][0:valid_num_face]
    probabilities = pred['probabilities'][0:valid_num_face,:]
    labels = pred['labels'][0:valid_num_face]
    xyz = pred['xyz']
    vidx_per_face = pred['vidx_per_face'][0:valid_num_face, :]


    # eval
    correct_mask = classes == labels
    cort_idx = np.where(correct_mask)[0]
    cor_num = cort_idx.shape[0]
    err_idx = np.where(np.logical_not(correct_mask))[0]
    acc = 1.0 * cor_num / valid_num_face

    mean_iou, iou_classes = compute_iou_np(classes, labels, num_classes)
    print('acc:{} mean_iou:{}'.format(acc, mean_iou))

    crt_vidx_per_face = np.take(vidx_per_face, cort_idx, 0)
    crt_classes = np.take(classes, cort_idx)
    err_vidx_per_face = np.take(vidx_per_face, err_idx, 0)
    err_classes = np.take(classes, err_idx)

    pred_res_dir = '/%s/plys/b%d_Acc_%d_IOU_%d'%(model_dir, k, int(100*acc), int(100*mean_iou))

    ply_fn = os.path.join(pred_res_dir, 'crt.ply')
    gen_mesh_ply(ply_fn, xyz, crt_vidx_per_face, face_label=crt_classes)

    ply_fn = os.path.join(pred_res_dir, 'err.ply')
    gen_mesh_ply(ply_fn, xyz, err_vidx_per_face, face_label=err_classes)


    ply_fn = os.path.join(pred_res_dir, 'pred.ply')
    gen_mesh_ply(ply_fn, xyz, vidx_per_face, face_label=classes)

    ply_fn = os.path.join(pred_res_dir, 'gt.ply')
    gen_mesh_ply(ply_fn, xyz, vidx_per_face, face_label=labels)

    pass

