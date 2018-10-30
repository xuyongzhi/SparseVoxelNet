# xyz Oct 2018
import tensorflow as tf

class TfUtil():
  @ staticmethod
  def valid_num_to_mask(valid_num, num0):
    shape0 = TfUtil.get_tensor_shape(valid_num)
    dims = len(shape0)
    aim_shape = [1]*dims + [num0]
    tmp = tf.reshape(tf.range(num0), aim_shape)
    tmp = tf.tile(tmp, shape0+[1])
    valid_mask = tf.less(tmp,tf.expand_dims(valid_num,-1))
    return valid_mask


  @ staticmethod
  def mask_reduce_sum(inputs, valid_mask, dim=None):
    in_shape = TfUtil.get_tensor_shape(inputs)
    mask_shape = TfUtil.get_tensor_shape(valid_mask)
    assert in_shape == mask_shape

    valid_mask = tf.cast(valid_mask, tf.float32)
    vsum = inputs * valid_mask
    vsum = tf.reduce_sum(vsum, dim)
    return vsum

  @ staticmethod
  def mask_reduce_mean(inputs, valid_mask, dim=None):
    in_shape = TfUtil.get_tensor_shape(inputs)
    mask_shape = TfUtil.get_tensor_shape(valid_mask)
    assert len(in_shape) == len(mask_shape)

    valid_mask = tf.cast(valid_mask, tf.float32)
    valid_num = tf.reduce_sum(valid_mask, dim, keepdims=True)
    valid_mask /= valid_num
    mean = inputs * valid_mask
    mean = tf.reduce_sum(mean, dim)
    return mean

  @ staticmethod
  def mask_reduce_min(inputs, valid_mask, dim=None):
    in_shape = TfUtil.get_tensor_shape(inputs)
    mask_shape = TfUtil.get_tensor_shape(valid_mask)
    assert len(in_shape) == len(mask_shape)

    valid_mask = tf.cast(valid_mask, tf.float32)
    max_ = tf.reduce_max(inputs)
    inputs = inputs * valid_mask + max_ * (1-valid_mask)
    vmin = tf.reduce_min(inputs, dim)
    return vmin

  @ staticmethod
  def t_shape(tensor):
    return TfUtil.get_tensor_shape(tensor)

  @ staticmethod
  def get_tensor_shape(tensor):
    if isinstance(tensor, tf.Tensor):
      shape = tensor.shape.as_list()
      for i,s in enumerate(shape):
        if s==None:
          shape[i] = tf.shape(tensor)[i]
      return shape
    else:
      return tensor.shape

  @ staticmethod
  def tshape0(t):
    return TfUtil.get_tensor_shape(t)[0]

  @ staticmethod
  def tsize(tensor):
    return len(TfUtil.get_tensor_shape(tensor))

  @ staticmethod
  def gather_second_d(inputs, indices):
    '''
    Gather inputs by indices, indices is for the second dim.
      inputs: (batch_size, n1, ...)
    '''
    assert TfUtil.tsize(inputs) >= 2
    idx_shape = TfUtil.get_tensor_shape(indices)
    if len(idx_shape)==3:
      batch_idx = tf.reshape(tf.range(idx_shape[0]), [-1,1,1,1])
      batch_idx = tf.tile(batch_idx, [1, idx_shape[1], idx_shape[2], 1])
    elif len(idx_shape)==2:
      batch_idx = tf.reshape(tf.range(idx_shape[0]), [-1,1,1])
      batch_idx = tf.tile(batch_idx, [1, idx_shape[1],  1])
    elif len(idx_shape)==4:
      batch_idx = tf.reshape(tf.range(idx_shape[0]), [-1,1,1,1,1])
      batch_idx = tf.tile(batch_idx, [1, idx_shape[1], idx_shape[2], idx_shape[3],1])
    indices = tf.expand_dims(indices, -1)
    indices = tf.concat([batch_idx, indices], -1)
    outputs = tf.gather_nd(inputs, indices)
    return outputs

  @ staticmethod
  def gather_third_d(inputs, indices):
    assert TfUtil.tsize(inputs) >= 3
    idx_shape = TfUtil.get_tensor_shape(indices)
    if len(idx_shape)==3:
      # gather along evn
      batch_size, vertex_num, evn = idx_shape
      batch_idx = tf.reshape(tf.range(batch_size), [-1,1,1,1])
      batch_idx = tf.tile(batch_idx, [1, vertex_num, evn, 1])
      vn_idx = tf.reshape(tf.range(vertex_num), [1,-1,1,1])
      vn_idx = tf.tile(vn_idx, [batch_size, 1, evn, 1])

    elif len(idx_shape)==4:
      batch_size, vertex_num, n2, n3 = idx_shape
      batch_idx = tf.reshape(tf.range(batch_size), [-1,1,1,1,1])
      batch_idx = tf.tile(batch_idx, [1, vertex_num, n2, n3, 1])
      vn_idx = tf.reshape(tf.range(vertex_num), [1,-1,1,1,1])
      vn_idx = tf.tile(vn_idx, [batch_size, 1, n2, n3, 1])

    indices = tf.expand_dims(indices, -1)
    indices = tf.concat([batch_idx, vn_idx, indices], -1)
    outputs = tf.gather_nd(inputs, indices)
    return outputs


