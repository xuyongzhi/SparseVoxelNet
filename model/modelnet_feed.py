import tensorflow as tf
import numpy as np
import glob, os
import h5py, time
from dataset_utils import parse_pl_record, get_dataset_summary

DATASET_NAME = 'MODELNET40'
_NUM_TRAIN_FILES = 20
_SHUFFLE_BUFFER = 10000

def input_fn_h5(is_training, data_dir, batch_size, data_net_configs=None, num_epochs=1):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A dataset that can be used for iteration.
  """
  import h5py

  bunch = batch_size // 8
  class generator_h5:
    def __call__(self, fn):
      with h5py.File(fn,'r') as h5f:
        d_size = h5f['data'].shape[0]
        for i in range(0, d_size//bunch):
          start = i*bunch
          end = (i+1)*bunch
          data = h5f['data'][start:end,:,:]
          label = h5f['label'][start:end,:]
          label = label.astype(np.int32)
          yield data, label

  def parse_pl_h5(datas, labels):
    datas = tf.reshape(datas, [-1, datas.shape[-2], datas.shape[-1]])
    shape = datas.shape.as_list()
    shape[-2] = 1024
    datas = tf.random_crop(datas, shape)
    labels = tf.reshape(labels, [-1, labels.shape[-1]])
    return datas, labels

  is_shuffle = True
  data_dir = '/DS/MODELNET/charles/modelnet40_ply_hdf5_2048'
  if is_training:
    fn_glob = data_dir + '/*train*.h5'
  else:
    fn_glob = data_dir + '/*test*.h5'
  filenames = glob.glob(fn_glob)

  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.interleave(lambda fn: tf.data.Dataset.from_generator(
    generator_h5(),
    (tf.float32, tf.int32),
    (tf.TensorShape([bunch,2048,3]), tf.TensorShape([bunch,1])),
    args=(fn,) ),
    cycle_length=5, block_length=32)

  batch_size = batch_size // bunch
  if is_training and is_shuffle:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  dataset = dataset.map( parse_pl_h5, num_parallel_calls=3 )
  if is_training and is_shuffle:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)


  DEBUG = False
  if DEBUG:
    from ply_util import create_ply_dset
    from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
    datasets_meta = DatasetsMeta(DATASET_NAME)

    model_dir = data_net_configs['model_dir']
    ply_dir = os.path.join(model_dir,'ply')
    aug = data_net_configs['aug_types']
    aug_ply_fn = os.path.join(ply_dir, aug)

    next_item = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
      datas, labels = sess.run(next_item)
      for i in range(batch_size):
        category = '_'+datasets_meta.label2class[labels[i][0]]
        create_ply_dset(DATASET_NAME, datas[i], aug_ply_fn+category+str(i)+'.ply')
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

  return dataset


def load_h5(fns):
  t0 = time.time()
  print('\n\nstart loading all modelnet h5')
  datas = []
  labels = []
  for fn in fns:
    with h5py.File(fn,'r') as h5f:
      print(os.path.basename(fn))
      data = h5f['data'][:]
      label = h5f['label'][:]
      label = label.astype(np.int32)
      datas.append(data)
      labels.append(label)
  datas = np.concatenate(datas, 0)
  labels = np.concatenate(labels, 0)
  print('finish loading all modelnet h5: %0.1f sec %d\n\n'%(time.time()-t0, datas.shape[0]))
  return datas, labels

def load_h5_all():
  data_dir = '/DS/MODELNET/charles/modelnet40_ply_hdf5_2048'
  fn_globs = {}
  fn_globs['train'] = data_dir + '/*train*.h5'
  fn_globs['test'] = data_dir + '/*test*.h5'
  datas_h5 = {}
  labels_h5 = {}
  for t in ['train', 'test']:
    filenames = glob.glob(fn_globs[t])
    datas_h5[t], labels_h5[t] = load_h5(filenames)
  return datas_h5, labels_h5

DATAS_H5, LABELS_H5 = load_h5_all()

def input_fn_h5_(is_training, data_dir, batch_size, data_net_configs=None, num_epochs=1):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A dataset that can be used for iteration.
  """

  def preprocess_pl_h5(datas, labels):
    shape = datas.shape.as_list()
    shape[-2] = 1024
    datas = tf.random_crop(datas, shape)
    return datas, labels

  is_shuffle = True
  tot = 'train' if is_training else 'test'
  dataset = tf.data.Dataset.from_tensor_slices((DATAS_H5[tot], LABELS_H5[tot]))
  dataset = dataset.map( preprocess_pl_h5, num_parallel_calls=3 )

  if is_training and is_shuffle:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  if is_training and is_shuffle:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)


  DEBUG = False
  if DEBUG:
    from ply_util import create_ply_dset
    from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
    datasets_meta = DatasetsMeta(DATASET_NAME)

    model_dir = data_net_configs['model_dir']
    ply_dir = os.path.join(model_dir,'ply')
    aug = data_net_configs['aug_types']
    aug_ply_fn = os.path.join(ply_dir, aug)

    next_item = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
      datas, labels = sess.run(next_item)
      import pdb; pdb.set_trace()  # XXX BREAKPOINT
      for i in range(batch_size):
        category = '_'+datasets_meta.label2class[labels[i][0]]
        create_ply_dset(DATASET_NAME, datas[i], aug_ply_fn+category+str(i)+'.ply')
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

  return dataset

