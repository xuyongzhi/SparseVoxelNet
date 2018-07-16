import tensorflow as tf

def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  assert os.path.exists(data_dir), ('not exsit: %s'%(data_dir))
  if is_training:
    return glob.glob(os.path.join(data_dir, 'train_*.tfrecord'))
  else:
    return glob.glob(os.path.join(data_dir, 'test_*.tfrecord'))

def input_fn(is_training, data_dir, batch_size, data_net_configs=None, num_epochs=1):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  assert len(filenames)>0, (data_dir)
  #print('\ngot {} tfrecord files\n'.format(len(filenames)))
  dataset = tf.data.TFRecordDataset(filenames)
                                   # compression_type="",
                                   # buffer_size=_SHUFFLE_BUFFER,
                                   #num_parallel_reads=3)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  #dataset = dataset.apply(tf.contrib.data.parallel_interleave(
  #    tf.data.TFRecordDataset, cycle_length=5))

  return resnet_run_loop.process_record_dataset(
      dataset, is_training, batch_size, _SHUFFLE_BUFFER, parse_pl_record, data_net_configs,
      num_epochs
  )

def model(net):
  filters = [64,64,128,128,256,256,521,1024]
  for filter in filters:
    net = tf.layers.conv2d(net, filter, 1, 1)
  return net

def train():
  data_dir = '/home/z/Research/SparseVoxelNet/data/MODELNET40H5F/Merged_tfrecord/4_mgs0.2048_gs2_2d2-rep_fmn1_mvp1-1-1024--pd3-1M'

  with tf.Graph().as_default():
    with tf.device('/gpu:0')
      inputs_op, labels_op = input_fn()
      outputs_op = model(inputs_op)
      loss_op = tf.layers.losses(outputs_op, labels_op)
      optimizer = tf.train.AdamOptimizer(0.001)
      train_op = optimizer.minimize(loss_op)
      logits_op = tf.argmax(outputs_op, -1)
      accuracy_op = tf.reduce_mean(tf.equal(logits_op, labels_op))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for epoch in range(0,51):
      accuracy_v = sess.run([accuracy_op])

      print('epoch {}: accuracy:{}'.format(epoch, accuracy_v))





