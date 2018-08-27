# July 2018 xyz
import tensorflow as tf
import glob, os, sys
import numpy as np
from datasets.rawh5_to_tfrecord import parse_pl_record, get_dset_metas
from datasets.all_datasets_meta.datasets_meta import DatasetsMeta
from utils.ply_util import create_ply_dset, draw_blocks_by_bot_cen_top
import time

DEBUG = True
MAX_FLOAT_DRIFT = 1e-5
SMALL_BUGS_IGNORE = True


def get_data_summary_from_tfrecord(filenames, raw_tfrecord_path):
  dset_shapes = {}
  batch_size = 100
  is_training = False

  with tf.Graph().as_default():
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        lambda value: parse_pl_record(value, is_training),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=False))
    iterator = dataset.make_one_shot_iterator().get_next()

    frame_num = 0
    xyz_scopes = None

    def xyz_scopes_str(xs, nf):
      xs_min = np.min(xs, 0)
      xs_mean = np.mean(xs, 0)
      xs_max = np.max(xs, 0)
      xs_str = 'frame num: {}\nscope min: {}\nscope mean:{}\nscope max: {}'.format(
                  nf, xs_min, xs_mean, xs_max)
      return xs_str

    with tf.Session() as sess:
      try:
        while True:
          features, labels = sess.run(iterator)
          points = features['points']
          frame_num += points.shape[0]
          xyz = points[:,:,0:3]
          xyz_min = np.min(xyz, 1)
          xyz_max = np.max(xyz, 1)
          xyz_scope_i = xyz_max - xyz_min
          if type(xyz_scopes) == type(None):
            xyz_scopes = xyz_scope_i
          else:
            xyz_scopes = np.concatenate([xyz_scopes, xyz_scope_i], 0)
          if frame_num // batch_size % 5 ==0:
            print( xyz_scopes_str(xyz_scopes, frame_num) )
      except:
        pass
    assert frame_num > batch_size*2

    xs_str = xyz_scopes_str(xyz_scopes, frame_num)
    print(xs_str)

    data_summary_fn = os.path.join(raw_tfrecord_path, 'xyz_scope.txt')
    with open(data_summary_fn, 'w') as sf:
      sf.write(xs_str)
    print('write: {}'.format(data_summary_fn))


def add_permutation_combination_template(A, B):
  '''
  Utilize tf broadcast: https://stackoverflow.com/questions/43534057/evaluate-all-pair-combinations-of-rows-of-two-tensors-in-tensorflow/43542926
  A:(a,d)
  B:(b,d)
  '''
  A_ = tf.expand_dims(A,0)
  B = tf.expand_dims(B,1)


def permutation_combination_2D(up_bound_2d, low_bound_2d):
  '''
  low_bound_2d: (2)
  up_bound_2d: (2)
  '''
  pairs = []
  for i in range(2):
    zeros = tf.zeros([up_bound_2d[i] - low_bound_2d[i],1], tf.int32)
    pairs_i = tf.reshape(tf.range(low_bound_2d[i], up_bound_2d[i], 1), [-1,1])
    if i==1:
      pairs_i = tf.concat([zeros, pairs_i], 1)
      pairs_i = tf.expand_dims(pairs_i, 0)
    else:
      pairs_i = tf.concat([pairs_i, zeros], 1)
      pairs_i = tf.expand_dims(pairs_i, 1)
    pairs.append(pairs_i)

  pairs_2d = tf.reshape(pairs[0] + pairs[1], [-1,2])
  return pairs_2d


def permutation_combination_3D(up_bound_3d, low_bound_3d=[0,0,0]):
  '''
  low_bound_3d: (3)
  up_bound_3d: (3)
  '''
  pairs_2d = permutation_combination_2D(up_bound_3d[0:2], low_bound_3d[0:2])
  zeros = tf.zeros([tf.shape(pairs_2d)[0],1], tf.int32)
  pairs_2d = tf.concat([pairs_2d, zeros], -1)

  pairs_3th = tf.reshape(tf.range(low_bound_3d[2], up_bound_3d[2], 1), [-1,1])
  zeros = tf.zeros([up_bound_3d[2] - low_bound_3d[2],2], tf.int32)
  pairs_3th = tf.concat([zeros, pairs_3th], -1)

  pairs_2d = tf.expand_dims(pairs_2d, 0)
  pairs_3th = tf.expand_dims(pairs_3th, 1)
  pairs_3d = tf.reshape(pairs_2d + pairs_3th, [-1,3])
  return pairs_3d



class BlockGroupSampling():
  def __init__(self, sg_settings, log_path=None):
    '''
    sg_settings:
      width = [1,1,1]
      stride = [1,1,1]
      nblock = 480
      npoint_per_block = 32
      max_nblock = 1000
    '''
    if log_path!=None:
      self.log_path = log_path + '/'
    self._num_scale = len(sg_settings['width'])
    self._widths =  sg_settings['width']
    self._strides = sg_settings['stride']
    self._nblocks_per_point = sg_settings['nblocks_per_point']
    self._nblocks = sg_settings['nblock']
    #self._padding_rate = [0.5, 0.7, 0.8, 0.8]
    self._npoint_per_blocks = sg_settings['npoint_per_block']
    # cut all the block with too few points
    self._np_perb_min_includes = sg_settings['np_perb_min_include']
    assert self._widths.shape[1] == self._strides.shape[1] == 3
    self._nblock_buf_maxs = sg_settings['max_nblock']
    self._empty_point_index = 'first' # -1(GPU only) or 'first'
    self._vox_sizes = sg_settings['vox_size']

    # record grouping parameters
    self._record_samplings = sg_settings['record']
    self.samplings = []
    if self._record_samplings:
      for i in range(self._num_scale):
        self.samplings.append({})
        self.samplings[i]['nbatches_'] = tf.constant(0, tf.int64)
        self.samplings[i]['t_per_frame_'] = tf.constant(0, tf.float64)
        self.samplings[i]['nb_enoughp_ave_'] = tf.constant(0, tf.int64)
        self.samplings[i]['nb_lessp_ave_'] = tf.constant(0, tf.int64)
        self.samplings[i]['ave_np_perb_'] = tf.constant(0, tf.int64)
        self.samplings[i]['std_np_perb_'] = tf.constant(0, tf.int64)
        self.samplings[i]['nmissing_perb_'] = tf.constant(0, tf.int64)
        self.samplings[i]['nempty_perb_'] = tf.constant(0, tf.int64)
        self.samplings[i]['ngp_out_global_'] = tf.constant(0, tf.int64)
        self.samplings[i]['ngp_out_block_'] = tf.constant(0, tf.int64)
        self.samplings[i]['npoint_grouped_'] = tf.constant(0, tf.int64)

        self.samplings[i]['ngp_valid_rate'] = tf.constant(0, tf.float32)
        #self.samplings[i]['b_perp_toomany_r'] = tf.constant(0, tf.float32)
        #self.samplings[i]['b_perp_toofew_r'] = tf.constant(0, tf.float32)


    # debug flags
    self._shuffle = True  # compulsory
    self._cut_bindex_by_global_scope = True # compulsory

    self._check_optional = True
    self._use_less_points_block_when_not_enough_optial = True and self._check_optional
    # Theoretically, num_block_per_point should be same for all. This is 0.
    self._check_nblock_per_points_same_optial = True and self._check_optional
    self._check_gbinb_optial = True and self._check_optional
    self._check_voxelization_optial = True and self._check_optional

    self.debugs = {}
    self._gen_ply = sg_settings['gen_ply']


  def update_global_bot_cen_top_for_global(self, xyz):
    xyz_max = tf.reduce_max(xyz, 1)
    xyz_min = tf.reduce_min(xyz, 1)
    xyz_mean = (xyz_max + xyz_min) / 2
    self.xyz_scope = xyz_max - xyz_min
    print('raw xyz scope:{}'.format(self.xyz_scope))

    # align to 0.1
    bot = 0.1 * tf.floor(xyz_min / 0.1)
    top0 = 0.1 * tf.ceil(xyz_max / 0.1)

    # align to width and stride
    tmp = ((top0-bot) - self._widths[0]) / self._strides[0]
    tmp = tf.maximum(tf.ceil(tmp),0)
    top = tmp * self._strides[0] + self._widths[0] + bot
    cen = (bot + top) / 2
    global_bot_cen_top = tf.concat([bot, cen, top], -1)

    self.global_bot_cen_top = tf.reshape(global_bot_cen_top,
                                [self.batch_size, 1, 1, 9])
            # batch_size, global_block_num, num_point, 9
    self.global_block_num = self.global_bot_cen_top.shape[1].value
    self.global_bot_bot_bot = tf.tile(self.global_bot_cen_top[...,0:3], [1,1,1,3])

  def update_global_bot_cen_top_for_sub(self, out_bot_cen_top_scale0):
    out_bot_cen_top = tf.squeeze( out_bot_cen_top_scale0, 1 )
    out_bot_cen_top = tf.expand_dims( out_bot_cen_top, 2 )
    self.global_bot_cen_top = out_bot_cen_top
    self.global_block_num = self.global_bot_cen_top.shape[1].value
    self.global_bot_bot_bot = tf.tile(self.global_bot_cen_top[...,0:3], [1,1,1,3])
    #print(self.global_bot_cen_top[...,0:3])

  def pre_sampling(self, xyz):
    if xyz.shape[1].value < 20000:
      return xyz
    xyz = tf.transpose(xyz, [1,0,2])
    xyz = tf.random_shuffle(xyz)
    xyz = tf.transpose(xyz, [1,0,2])
    xyz = xyz[:,0:20000,:]
    return xyz


  def grouping_multi_scale(self, xyz):
    '''
    xyz: (batch_size, num_point, 3)
    '''
    xyz_shape = xyz.shape
    assert len(xyz_shape) == 3
    assert xyz_shape[2] == 3
    #xyz = self.pre_sampling(xyz)
    self.batch_size = xyz_shape[0].value
    self.update_global_bot_cen_top_for_global(xyz)

    num_scale = self._num_scale
    ds_ms = {}
    others_ms = {}

    in_bot_cen_top = tf.tile(xyz, [1,1,3])
    # add dim for global block, global_block_num=1 for input
    in_bot_cen_top = tf.expand_dims(in_bot_cen_top, 1)

    for s in range(num_scale):
        t0 = tf.timestamp()
        # in_bot_cen_top: (batch_size, global_block_num, num_point0, 9)
        assert len(in_bot_cen_top.shape) == 4
        assert in_bot_cen_top.shape[3] == 9
        global_block_num = in_bot_cen_top.shape[1].value

        others = {}
        ds = {}

        ds = {}
        ds['grouped_pindex'], ds['vox_index'], ds['grouped_bot_cen_top'], \
        ds['grouped_empty_mask'], ds['out_bot_cen_top'], ds['nb_enough_p'], others = \
          self.grouping(s, in_bot_cen_top)

        if s == 0:
          in_bot_cen_top = tf.squeeze(ds['grouped_bot_cen_top'], 1)
          self.update_global_bot_cen_top_for_sub(ds['out_bot_cen_top'])
        else:
          in_bot_cen_top = ds['out_bot_cen_top']

        for item in ds:
          if item not in ds_ms:
            ds_ms[item] = []
          ds_ms[item].append(ds[item])

        for j in range(len(others['name'])):
          name = others['name'][j]
          if name not in others_ms:
            others_ms[name] = []
          others_ms[name].append( others['value'][j] )

        #if self._record_samplings:
        #  self.samplings[s]['t_per_frame_'] += tf.timestamp() - t0


    self.scale = num_scale
    if num_scale>1:
      global_vox_index, global_empty_mask = self.gen_global_voxelization(
        ds_ms['out_bot_cen_top'][-1], ds_ms['out_bot_cen_top'][0],
        ds_ms['nb_enough_p'][-1])

      ds_ms['vox_index'].append(global_vox_index)
      ds_ms['grouped_empty_mask'].append(global_empty_mask)

    # **************************************************************************
    # output shape
    # out_bot_cen_top: (bs, gbn, nb, 9)
    # grouped_pindex:  (bs, gbn, nb, np)
    # grouped_empty_mask: (bs, gbn, nb, np)
    # grouped_bot_cen_top: (bs, gbn, nb, np, 9)
    # nb_enough_p:    (bs, gbn)
    # vox_index: (bs, gbn, nb, np, 3)
    # **************************************************************************

    if self._record_samplings:
      write_record = self.write_sg_log()
      with tf.control_dependencies([write_record]):
        ds_ms['grouped_pindex'][0] = tf.identity(ds_ms['grouped_pindex'][0])
        return ds_ms['grouped_pindex'], ds_ms['vox_index'], ds_ms['grouped_bot_cen_top'],\
          ds_ms['grouped_empty_mask'], ds_ms['out_bot_cen_top'], ds_ms['nb_enough_p'], others_ms
    else:
      return ds_ms['grouped_pindex'], ds_ms['vox_index'], ds_ms['grouped_bot_cen_top'],\
          ds_ms['grouped_empty_mask'], ds_ms['out_bot_cen_top'], ds_ms['nb_enough_p'], others_ms


  def gen_global_voxelization(self, last_scale_bot_cen_top, gb_bot_cen_top, nb_enough_p):
    assert len(last_scale_bot_cen_top.shape) == 4

    grouped_bot_cen_top = tf.expand_dims(last_scale_bot_cen_top, 2)
    out_bot_cen_top = tf.squeeze( gb_bot_cen_top, 1 )
    out_bot_cen_top = tf.expand_dims( out_bot_cen_top, 2 )

    global_vox_index = self.voxelization(grouped_bot_cen_top, out_bot_cen_top)

    shape = [e.value for e in global_vox_index.shape][0:-1]
    grouped_empty_mask = tf.reshape(tf.range(shape[-1]), [1,1,1,shape[-1]])
    grouped_empty_mask = tf.tile(grouped_empty_mask, [shape[0], shape[1], shape[2],1])
    nb_enough_p = tf.expand_dims( tf.expand_dims(nb_enough_p, -1), -1)
    global_empty_mask = tf.less(grouped_empty_mask, nb_enough_p)
    return global_vox_index, global_empty_mask


  def grouping(self, scale, bot_cen_top):
    '''
    bot_cen_top: (batch_size, global_block_num, num_point0, 9)
    global_bot_cen_top_ = (batch_size, global_block_num, 1, 9)

    Search by point: for each point in xyz, find all the block d=ids
    '''
    shape0 = [e.value for e in bot_cen_top.shape]
    assert len(shape0) == 4
    self.scale = scale
    assert self.global_block_num == shape0[1]
    self.bsgbn = self.batch_size * self.global_block_num
    self.num_point0 = shape0[2]

    # Align to the gloabl bottom: 1) all block index is positive, so that block
    # index to block id is exclusive. 2) This alignment allows voxelizing when
    # width and stride is different.
    bot_cen_top -= self.global_bot_bot_bot

    bid_pindex = self.get_block_id(bot_cen_top)

    bid_index__pindex_inb, point_index, block_id_unique = self.get_bid_point_index(bid_pindex)

    grouped_pindex, grouped_empty_mask = self.get_grouped_pindex(
                                      bid_index__pindex_inb, point_index)

    grouped_bot_cen_top = self.gather_grouped(grouped_pindex, bot_cen_top)

    bids_sampling, out_bot_cen_top = self.all_bot_cen_tops(block_id_unique)

    if self._check_gbinb_optial:
      gb_valid_mask, nerr_gb_in_ob = self.check_block_inblock(grouped_bot_cen_top,
                                        tf.expand_dims(out_bot_cen_top, -2),
                                        grouped_empty_mask)
      with tf.control_dependencies([tf.assert_equal(nerr_gb_in_ob, 0)]):
        grouped_bot_cen_top = tf.identity(grouped_bot_cen_top)

    vox_index = self.voxelization(grouped_bot_cen_top, out_bot_cen_top)

    # **************************************************************************
    #       Add middle tensors to check between graph model and eager model
    ots = 'OTS_'
    others = {}
    others['value'] = []
    others['name'] = []
    #point_index_O = tf.concat([point_index, tf.zeros(self.npoint_grouped-tf.shape(point_index),tf.int32)],0)
    #bid_index__pindex_inb_O = tf.concat([bid_index__pindex_inb,
    #        tf.zeros([self.npoint_grouped-tf.shape(bid_index__pindex_inb)[0],2],tf.int32)],0)
    #others['value'] += [bid_pindex, bid_index__pindex_inb_O, point_index_O]
    #others['name'] += ['bid_pindex', 'bid_index__pindex_inb', 'point_index']

    #block_id_unique_O = tf.concat([block_id_unique, tf.zeros(1000-tf.shape(block_id_unique),tf.int32)],0)
    #bid_index_sampling_inall_O = tf.concat([bid_index_sampling_inall, tf.zeros(1000-tf.shape(bid_index_sampling_inall),tf.int32)],0)
    if self._gen_ply:
      others['value'] += [bids_sampling]
      others['name'] += ['bids_sampling']

    #if DEBUG:
    #  for name in self.debugs:
    #    others['value'] += [self.debugs[name]]
    #    others['name'] += [name]
    others['name'] = [ots+e for e in others['name']]

    out_bot_cen_top += self.global_bot_bot_bot
    grouped_bot_cen_top += tf.expand_dims(self.global_bot_bot_bot, -2)
    nb_enoughp_per_gb = tf.reshape(self.nb_enoughp_per_gb, [self.batch_size, self.global_block_num])


    # ******************** test **********************
    #if self.scale == 2:
    #  test_voxelization(self._vox_sizes[self.scale], grouped_bot_cen_top, vox_index)

    return grouped_pindex, vox_index, grouped_bot_cen_top, grouped_empty_mask, out_bot_cen_top, nb_enoughp_per_gb, others


  def gather_grouped(self, grouped_pindex, bot_cen_top):
    '''
    grouped_pindex: (batch_size, global_block_num, _nblock, npoint_per_block)
    bot_cen_top:    (batch_size, global_block_num, num_point0, 9)

    grouped_pindex  value is the index along third dimension of bot_cen_top
    '''
    assert len(grouped_pindex.shape) == 4
    assert len(bot_cen_top.shape) == 4
    bs_tmp = tf.reshape(tf.range(0, self.batch_size, 1), [self.batch_size,1,1,1,1])
    bs_tmp = tf.tile(bs_tmp, [1, self.global_block_num, self._nblocks[self.scale],\
                              self._npoint_per_blocks[self.scale], 1])
    gb_tmp = tf.reshape(tf.range(0, self.global_block_num, 1), [1,self.global_block_num,1,1,1])
    gb_tmp = tf.tile(gb_tmp, [self.batch_size, 1, self._nblocks[self.scale],\
                              self._npoint_per_blocks[self.scale], 1])
    grouped_pindex = tf.expand_dims(grouped_pindex, -1)
    grouped_pindex = tf.concat([bs_tmp, gb_tmp, grouped_pindex], -1)
    grouped_bot_cen_top = tf.gather_nd(bot_cen_top, grouped_pindex)
    return grouped_bot_cen_top


  def show_settings(self):
    items = ['_widths', '_strides', '_nblocks_per_point', '_nblocks', '_npoint_per_blocks',
              '_np_perb_min_includes','_nblock_buf_maxs', '_shuffle']
    print('\n\nsettings:\n')
    for item in items:
      #if hasattr(self, item):
      print('{}:{}'.format(item, getattr(self,item)))
    print('\n\n')


  def show_samplings_np_multi_scale(self, samplings_np_ms):
    for s in range(len(samplings_np_ms)):
      samplings_np = samplings_np_ms[s]
      self.show_samplings_np(samplings_np,s)


  def show_samplings_np(self, samplings_np, scale):
    t = samplings_np['nbatches_']
    assert t>0
    def summary_str(item, real, config):
      real = real/t
      return '{}: {}/{}, {:.3f}\n'.format(item, real, config,
                      1.0*tf.cast(real,tf.float32)/tf.cast(config,tf.float32))
    summary = '\n'
    summary += 'scale={} t={}  Real / Config\n\n'.format(scale, t)
    summary += summary_str('nb_enough_p', samplings_np['nb_enoughp_ave_'], self._nblocks[scale])
    summary += 'nblock_lessp(few points):{}, np_perb_min_include:{}\n'.format(
                  samplings_np['nb_lessp_ave_'], self._np_perb_min_includes[scale])
    summary += 'ngp_out_global:{}  '.format(
      samplings_np['ngp_out_global_'])
    summary += 'ngp_out_block:{} npoint_grouped:{}\n'.format(
      samplings_np['ngp_out_block_'],  samplings_np['npoint_grouped_'])
    summary += summary_str('ave np per block', samplings_np['ave_np_perb_'], self._npoint_per_blocks[scale])
    summary += summary_str('nmissing per block', samplings_np['nmissing_perb_'], self._npoint_per_blocks[scale])
    summary += summary_str('nempty per block', samplings_np['nempty_perb_'], self._npoint_per_blocks[scale])
    for name in samplings_np:
      if name not in ['nbatches_', 'nb_enoughp_ave_', 'ave_np_perb_', 'nempty_perb_', 'nmissing_perb_', 'nb_lessp_ave_']:
        summary += '{}: {}\n'.format(name, samplings_np[name]/t)
    summary += '\n'
    print(summary)
    return summary


  def write_sg_log(self):
    the_items = ['scale','nbatches_','','_nblocks', 'nb_enoughp_ave_', '',
              'ngp_valid_rate', 'ngp_out_block_','ngp_out_global_','',
              '_np_perb_min_includes','nb_lessp_ave_', '',
              '_npoint_per_blocks','ave_np_perb_', 'std_np_perb_', 'nempty_perb_','',]
    key_items = ['nb_enoughp_ave_','ave_np_perb_']

    def sum_str(scale, item):
      value = self.samplings[scale][item]
      nbatches_ = self.samplings[scale]['nbatches_']
      if item != 'nbatches_':
        value = value / tf.cast(nbatches_, value.dtype)
      s = tf.as_string(value)
      s = tf.string_join([item, s], ':')
      #if self.batch_size<3 and item in key_items:
      #  s = tf.Print(s, [s], message=' scale {} '.format(scale))
      return s

    def set_str(scale, item):
      value = getattr(self, item)
      value = value[scale]
      s = tf.as_string(value)
      s = tf.string_join([item, s], ':')
      #s = tf.Print(s, [s], message=' scale {} '.format(scale))
      return s

    def item_str(scale, item):
      if item == '':
        return ''
      elif item == 'scale':
        scale_str = tf.as_string(scale)
        scale_str = tf.string_join(
          ['-------------------------------------------------------------\nscale',
                                  scale_str], ':')
        return scale_str
      elif item[0] == '_':
        return set_str(scale, item)
      else:
        return sum_str(scale, item)


    sg_str_ms = []
    for scale in range(self._num_scale):
      for item in the_items:
        istr = item_str(scale, item)
        sg_str_ms.append(istr)

    sg_str = tf.string_join(sg_str_ms, '\n')
    frame_str = tf.as_string(self.samplings[0]['nbatches_'])
    #log_fn = tf.string_join([self.log_path, frame_str, '_sg_log.txt'], '' )
    log_fn = tf.string_join([self.log_path, 'sg_log.txt'], '' )
    write_sg = tf.write_file(log_fn, sg_str)
    return write_sg


  def block_index_to_scope(self, block_index):
    '''
    block_index: (n,3)
    '''
    block_index = tf.cast(block_index, tf.float32)
    width = self._widths[self.scale]
    bottom = block_index * self._strides[self.scale]
    top = bottom + width
    center = (bottom + top)/2
    bot_cen_top = tf.concat([bottom, center, top], -1)
    return bot_cen_top


  def replace_points_outside_block_UNUSED(self, block_index, pinb_mask):
    '''
    Replace the points out of block with the first one in this block.
    The first one must inside the block.
    '''
    def my_sparse_to_dense(dense_shape, sparse_indices0, sparse_value0):
      # dense_shape: [1024,    8,    3]
      # sparse_indices0: (4, 2)
      # sparse_value0: (4, 3)

      # return: sparse_values: (1024, 8, 3)
      sparse_shape0 = tf.shape(sparse_indices0)
      d = dense_shape[-1]

      sparse_indices = tf.expand_dims(sparse_indices0, 1)
      sparse_indices = tf.tile(sparse_indices, [1,d,1])
      tmp1 = tf.reshape(tf.range(d),[1,d,1])
      tmp1 = tf.tile(tmp1, [sparse_shape0[0], 1, 1])
      sparse_indices = tf.concat([sparse_indices, tmp1],2)
      sparse_indices = tf.reshape(sparse_indices, (-1,d))
      sparse_values = tf.reshape(sparse_value0, [-1])
      sparse_values = tf.sparse_to_dense(sparse_indices, dense_shape, sparse_values)
      return sparse_values

    invalid_indices = tf.cast(tf.where(tf.logical_not(pinb_mask)),tf.int32)
    tmp0 = tf.constant([[1,0],[0,0]], tf.int32)
    valid_indices = tf.matmul( invalid_indices, tmp0)

    invalid_values = tf.gather_nd(block_index, invalid_indices)
    valid_values = tf.gather_nd(block_index, valid_indices)
    dense_shape = tf.shape(block_index)

    invalid_values_dense = my_sparse_to_dense(dense_shape, invalid_indices, invalid_values)
    valid_values_dense = my_sparse_to_dense(dense_shape, invalid_indices, valid_values)
    block_index_fixed = block_index - invalid_values_dense + valid_values_dense
    print('scale {}, invalid_indices num:{}'.format(self.scale, tf.shape(invalid_indices)[0]))
    return block_index_fixed


  def cut_bindex_by_global_scope(self, block_index):
    '''
    The blocks at the edge
    '''
    assert len(block_index.shape) == 5
    global_bot_cen_top =  self.global_bot_cen_top - self.global_bot_bot_bot
    # add dim for nblock_perp_3d
    global_bot_cen_top = tf.expand_dims(global_bot_cen_top, -2)
    bindex_min = global_bot_cen_top[...,0:3] / self._strides[self.scale]
    bindex_max = (global_bot_cen_top[...,6:9] - self._widths[self.scale]) / self._strides[self.scale]
    bindex_min = tf.cast(tf.ceil(bindex_min - MAX_FLOAT_DRIFT), tf.int32)
    bindex_max = tf.cast(tf.floor(bindex_max + MAX_FLOAT_DRIFT), tf.int32)

    low_bmask = tf.greater_equal(block_index, bindex_min)
    low_bmask = tf.reduce_all(low_bmask, -1)
    up_bmask = tf.less_equal(block_index, bindex_max)
    up_bmask = tf.reduce_all(up_bmask, -1)
    in_global_mask = tf.logical_and(low_bmask, up_bmask)
    ngp_out_global = tf.reduce_sum(1-tf.cast(in_global_mask, tf.int32))
    self.ngp_out_global = ngp_out_global

    if self.scale==0 and not SMALL_BUGS_IGNORE:
      check = tf.assert_equal(ngp_out_global,0)
      with tf.control_dependencies([check]):
        ngp_out_global = tf.identity(ngp_out_global)
    return in_global_mask, ngp_out_global


  def check_bindex_inblock(self, block_index, bot_cen_top_small):
    bot_cen_top_large = self.block_index_to_scope(block_index)
    bot_cen_top_small = tf.expand_dims(bot_cen_top_small, -2)
    pinb_mask, ngp_out_block  = self.check_block_inblock(bot_cen_top_small, bot_cen_top_large)
    self.ngp_out_block = ngp_out_block
    #print('scale {}, ngp_out_block {}'.format(self.scale, ngp_out_block))
    #max_ngp_out_block_rates = [0.01, 0.3, 0.3, 0.3]
    #ngp_out_block_rate =
    return pinb_mask, ngp_out_block


  def check_block_inblock(self, bot_cen_top_small, bot_cen_top_large, empty_mask=None):
    '''
    empty_mask for small only used to fix nerr_scope
    '''
    shape_large = [e.value for e in bot_cen_top_large.shape]
    shape_small = [e.value for e in bot_cen_top_small.shape]
    assert len(shape_small) == len(shape_large)
    assert shape_small[-1] == shape_large[-1] == 9

    #assert shape_large[0] == shape_small[0]
    #assert shape_large[-1] == shape_small[-1]
    #assert len(shape_small) ==3 or len(shape_small) ==2
    #assert len(shape_large) ==2 or len(shape_large) ==3
    #if len(shape_small) ==2:
    #  bot_cen_top_small = tf.expand_dims(bot_cen_top_small, 1)
    #if len(shape_large) ==2:
    #  bot_cen_top_large = tf.expand_dims(bot_cen_top_large, 1)

    if empty_mask!=None:
      assert len(empty_mask.shape) == len(shape_small)-1

    check_top = tf.reduce_all(tf.less_equal(bot_cen_top_small[...,6:9] - MAX_FLOAT_DRIFT,
                                      bot_cen_top_large[...,6:9]),-1)
    check_bottom = tf.reduce_all(tf.less_equal(bot_cen_top_large[...,0:3] - MAX_FLOAT_DRIFT,
                                               bot_cen_top_small[...,0:3]),-1)
    correct_mask = tf.logical_and(check_bottom, check_top)
    if empty_mask!=None:
      correct_mask = tf.logical_or(empty_mask, correct_mask)
    nerr_scope = tf.reduce_sum(1-tf.cast(correct_mask, tf.int32))
    return correct_mask, nerr_scope


  def get_block_id(self, bot_cen_top):
    '''
    (1) Get the responding block index of each point
    bot_cen_top: (num_point0, 9)
      block_index: (num_point0, nblock_perp_3d, 3)
    block_id: (num_point0, nblock_perp_3d, 1)
    '''
    #***************************************************************************
    # (1.1) lower and upper block index bound
    # align bot to zeor here
    width = self._widths[self.scale]
    if self.scale==0:
      xyz = bot_cen_top[:,:,:,3:6]
      low_b_index = (xyz - width) / self._strides[self.scale]
      up_b_index = (xyz) / self._strides[self.scale]
    else:
      low_b_index = (bot_cen_top[:,:,:,6:9] - width) / self._strides[self.scale]
      up_b_index = (bot_cen_top[:,:,:,0:3]) / self._strides[self.scale]

    # Allow point at te intersection point belong to both blocks at first.
    # If there is some poitns belong to both blocks. nblocks_per_points >
    # self._nblocks_per_point. But still use self._nblocks_per_point, so that
    # only set the points to lower block index.
    low_b_index_fixed = tf.cast(tf.ceil(low_b_index - MAX_FLOAT_DRIFT), tf.int32)    # include

    #***************************************************************************
    #(1.2) the block number for each point should be equal
    # If not, still make it equal to get point index per block.
    # Then rm the grouped points out of block scope.

    # ***** (optional)
    # block index is got by linear ranging from low bound by a fixed offset for
    # all the points. This is based on: num block per point is the same for all!
    # Check: No points will belong to greater nblocks. But few may belong to
    # smaller blocks: err_npoints
    #if self._check_nblock_per_points_same_optial:
    #  up_b_index_fixed = tf.cast(tf.floor(up_b_index + MAX_FLOAT_DRIFT), tf.int32) + 1 # not include

    #  nblocks_per_points = up_b_index_fixed - low_b_index_fixed
    #  ave_nblocks_per_point = tf.reduce_mean(tf.cast(nblocks_per_points, tf.float32), [0,1,2])
    #  nbpp_err_rate = tf.abs(self._nblocks_per_point[self.scale] - ave_nblocks_per_point) /

    #  nb_per_ps_err0 = self._nblocks_per_point[self.scale] - nblocks_per_points
    #  nb_per_ps_err_toofew = tf.reduce_any(tf.greater(nb_per_ps_err0, 0), -1)
    #  nb_per_ps_err_toomany = tf.reduce_any(tf.less(nb_per_ps_err0, 0), -1)
    #  nb_per_ps_err_toofew = tf.cast(nb_per_ps_err_toofew, tf.float32)
    #  nb_per_ps_err_toomany = tf.cast(nb_per_ps_err_toomany, tf.float32)
    #  b_perp_toofew_r = tf.reduce_mean(nb_per_ps_err_toofew)
    #  b_perp_toomany_r = tf.reduce_mean(nb_per_ps_err_toomany)
    #  if self._record_samplings:
    #    self.samplings[self.scale]['b_perp_toomany_r'] += b_perp_toomany_r
    #    self.samplings[self.scale]['b_perp_toofew_r'] += b_perp_toofew_r
    #  #print("scale {}, too many:{}\ntoo few:{}".format(self.scale, b_perp_toomany_r, b_perp_toofew_r))
    #  max_b_perp_toomany_r = 0.8

    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #  check_toofew = tf.assert_less_equal(b_perp_toofew_r, [0.01, 1.0, 1.0, 1.0][self.scale],
    #                  message="nb_per_ps too few {} is too large, at scale {}".format(b_perp_toofew_r, self.scale))
    #  check_toomany = tf.assert_less(b_perp_toomany_r, [max_b_perp_toomany_r, max_b_perp_toomany_r, max_b_perp_toomany_r][self.scale],
    #                  message="nb_per_ps too many {} is tooo large, at scale {}".format(b_perp_toomany_r, self.scale))
    #  with tf.control_dependencies([check_toomany, check_toofew]):
    #    low_b_index_fixed = tf.identity(low_b_index_fixed)
    # *****

    self.nblock_perp_3d = np.prod(self._nblocks_per_point[self.scale])
    self.npoint_grouped_buf = self.nblock_perp_3d * self.num_point0

    pairs_3d0 = permutation_combination_3D(self._nblocks_per_point[self.scale])
    assert self.nblock_perp_3d == pairs_3d0.shape[0].value
    pairs_3d = tf.tile( tf.reshape(pairs_3d0, [1,1,1,self.nblock_perp_3d,3]),
                [self.batch_size, self.global_block_num, self.num_point0, 1, 1])

    # add dim for "num block per point"
    block_index = tf.expand_dims(low_b_index_fixed, -2)
    block_index = tf.tile(block_index, [1,1,1, self.nblock_perp_3d, 1])
    block_index += pairs_3d

    pinb_mask, ngp_out_block = self.check_bindex_inblock(block_index, bot_cen_top)

    #***************************************************************************
    # (1.3) Remove the points belong to blocks out of edge bound
    if self._cut_bindex_by_global_scope:
      in_global_mask, ngp_out_global = self.cut_bindex_by_global_scope(block_index)
    else:
      self.ngp_out_global = 0

    #***************************************************************************
    # (1.4) concat block index and point index
    point_indices = tf.reshape( tf.range(0, self.num_point0, 1), (1,1,-1,1,1))
    point_indices = tf.tile(point_indices, [self.batch_size, \
                            self.global_block_num, 1, self.nblock_perp_3d, 1])
    bindex_pindex = tf.concat([block_index, point_indices], -1)

    # combine two dims to one: "num_point0" and "nblock_perp_3d"
    bindex_pindex = tf.reshape(bindex_pindex, (self.batch_size, \
                            self.global_block_num, self.npoint_grouped_buf, 4))
    pinb_mask = tf.reshape(pinb_mask, (self.batch_size, self.global_block_num, \
                                       self.npoint_grouped_buf,))
    if self._cut_bindex_by_global_scope:
      in_global_mask = tf.reshape(in_global_mask, (self.batch_size,
                                self.global_block_num, self.npoint_grouped_buf))
      ngb_invalid = ngp_out_block + ngp_out_global
      gp_valid_mask = tf.logical_and(pinb_mask, in_global_mask)
      #print('ngp_out_block:{}  ngp_out_global:{}'.format(ngp_out_block, ngp_out_global))
    else:
      ngb_invalid = ngp_out_block
      gp_valid_mask = pinb_mask
      #print('ngp_out_block:{}  ngp_out_global:{}'.format(ngp_out_block, 0))

    #***************************************************************************
    # (1.5) rm grouped points out of block or points in block out of edge bound

    bindex_pindex = self.rm_poutb(gp_valid_mask, bindex_pindex)

    #***************************************************************************
    # (1.6) block index -> block id
    if self._check_optional:
      # Make sure all block index positive. So that the blockid for each block index is
      # exclusive.
      min_bindex = tf.reduce_min(bindex_pindex[...,0:3])
      #if min_bindex < 0:
      #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
      #  pass
      check_bindex_pos = tf.assert_greater_equal( 0, min_bindex,
                                  message="neg block index, scale {}".format(self.scale))
      with tf.control_dependencies([check_bindex_pos]):
        bindex_pindex = tf.identity(bindex_pindex)

    max_block_index = tf.reduce_max(bindex_pindex[...,0:3], [0])
    # use the maximum block size for all the batch and global blocks
    self.block_size = max_block_index  + 1 # ADD ONE!!!

    # (num_point0, nblock_perp_3d, 1)
    block_id = self.block_index_to_block_id(
            bindex_pindex[...,0:3], self.block_size,  self._check_gbinb_optial)
    bid_pindex = tf.concat([block_id, bindex_pindex[...,3:4]], -1)

    #***************************************************************************
    # (1.7) check not too less valid grouped points
    if self._record_samplings:
      self.samplings[self.scale]['ngp_valid_rate'] += self.ngp_valid_rate
    check0 = tf.assert_greater(tf.reduce_min(self.npoint_grouped_all), 0,
                  message="npoints_grouped_min==0")
    check1 = tf.assert_greater(self.ngp_valid_rate, [0.01, 0.1, 0.1, 0.0][self.scale],
                  message="scale {} ngp_valid_rate {}".format(self.scale, self.ngp_valid_rate))
    with tf.control_dependencies([check0, check1]):
      bid_pindex = tf.identity(bid_pindex)

    return bid_pindex


  def rm_poutb(self, gp_valid_mask, bindex_pindex):
    '''
    gp_valid_mask: [batch_size, global_block_num, npoint_grouped_buf]
    bindex_pindex: [batch_size, global_block_num, npoint_grouped_buf, 4]
    bindex_pindex_valid: [npoint_grouped_all, 4]
    '''
    shape0 = [e.value for e in  bindex_pindex.shape]
    num_grouped_point0 = shape0[0] * shape0[1]

    bindex_pindex_ls = []
    for bi in range(self.batch_size):
      for gi in range(self.global_block_num):
        pindex_inb = tf.squeeze(tf.where(gp_valid_mask[bi,gi]),1)
        pindex_inb = tf.cast(pindex_inb, tf.int32)
        bindex_pindex_ls.append(tf.gather(bindex_pindex[bi,gi], pindex_inb, 0))

    self.npoints_grouped = tf.concat([tf.expand_dims(tf.shape(e)[0],0) for e in bindex_pindex_ls],0)
    bindex_pindex_valid = tf.concat(bindex_pindex_ls, 0)
    self.npoint_grouped_all = tf.shape(bindex_pindex_valid)[0]
    self.npoint_grouped_ave = self.npoint_grouped_all / (self.bsgbn)
    self.ngp_valid_rate = tf.cast( self.npoint_grouped_all, tf.float32) / \
                      tf.cast( tf.reduce_prod(gp_valid_mask.shape), tf.float32)
    return bindex_pindex_valid


  def block_index_to_block_id(self, block_index, block_size, check=False):
    '''
    block_index: (batch_size, global_num_block, npg,3)
    block_size:(3)

    block_id: (batch_size, global_num_block, npg,1)
    '''
    shape0 = block_index.shape
    shape1 = block_size.shape
    assert len(shape1) == 1 and shape1[0].value == 3
    assert len(shape0) == 2
    batch_size = shape0[0].value

    block_id0 = block_index[...,2] * block_size[1] * block_size[0] + \
                block_index[...,1] * block_size[0] + block_index[...,0]

    # Differeniate block ids between different global blocks and frames in a batch
    start_bids = self.get_start_bids(self.npoints_grouped)
    if self._check_gbinb_optial:
      check_start_bid = tf.assert_less(block_id0, self.start_bid_step)
      with tf.control_dependencies([check_start_bid]):
        start_bids = tf.identity(start_bids)
    block_id = block_id0 + start_bids
    #if self.scale==1:
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #  pass


    if check:
      block_index_C = self.block_id_to_block_index(block_id - start_bids, block_size)
      check_bi = tf.assert_equal(block_index, block_index_C)
      with tf.control_dependencies([check_bi]):
        block_id = tf.identity(block_id)

    block_id = tf.expand_dims(block_id, -1)
    return block_id


  def get_start_bids(self, np_all):
    '''
    np_all: self.npoints_grouped or
    '''
    #assert np_all.shape[0].value == self.bsgbn
    self.start_bid_step = tf.reduce_prod( self.block_size )+10
    self.start_bids_unique = tf.range(0,self.bsgbn,1) * self.start_bid_step
    start_bids = []
    for i in range(self.bsgbn):
      start_bids.append( tf.ones([np_all[i]], tf.int32) * \
                        self.start_bids_unique[i])
    cur_start_bids = tf.concat(start_bids, 0)
    #if self.scale==1:
    #  import pdb; pdb.set_trace()  # XXX BREAKPOINT
    #  pass
    return cur_start_bids


  def block_id_to_block_index(self, block_id, block_size):
    '''
    block_id shape is flexible
    '''
    shape0 = block_id.shape

    batch_size = self.batch_size
    tmp0 = block_size[0] * block_size[1]
    block_index_2 = tf.expand_dims(tf.floordiv(block_id, tmp0),-1)
    tmp1 = tf.floormod(block_id, tmp0)
    block_index_1 = tf.expand_dims(tf.floordiv(tmp1, block_size[0]),-1)
    block_index_0 = tf.expand_dims(tf.floormod(tmp1, block_size[0]),-1)
    block_index = tf.concat([block_index_0, block_index_1, block_index_2], -1)
    return block_index



  def split_bids_and_get_nblocks(self, block_id_unique, valid_bid_mask):
    #***************************************************************************
    # get self.nblocks_per_gb for each global block
    tmp0 = tf.cast(block_id_unique,tf.float32) / tf.cast(self.start_bid_step, tf.float32)
    # get the indices of global block for each item in block_id_unique
    gb_indices = tf.cast(tf.floor(tmp0), tf.int32)
    gb_indices_unique, gbi_index, nblocks_per_gb =\
                                                tf.unique_with_counts(gb_indices)
    check_gbn0 = tf.assert_equal(tf.shape(nblocks_per_gb)[0], self.bsgbn,
                            message="at least one global block has bot valid block")
    nblock_max = tf.reduce_max(nblocks_per_gb)
    check_gbn1 = tf.assert_greater_equal(self._nblock_buf_maxs[self.scale], nblock_max,
                            message="scale {} _nblock_buf_maxs:{} < {}".format\
                            (self.scale, self._nblock_buf_maxs[self.scale], nblock_max))

    self.nblocks_per_gb = nblocks_per_gb
    self.nblock_max = nblock_max
    self.nblock_min = tf.reduce_min(nblocks_per_gb)
    with tf.control_dependencies([check_gbn0, check_gbn1]):
      gb_indices = tf.identity(gb_indices)

    #***************************************************************************
    # get self.nb_enough_p for each global block
    # invalid means the block contains too less points
    valid_gb_indices = (gb_indices + 1) * tf.cast(valid_bid_mask, tf.int32)-1
    valid_gb_indices = tf.concat([tf.constant([-1],tf.int32), valid_gb_indices], 0)
    valid_gb_indices_unique, valid_gbi_index, valid_nblocks_per_gb =\
                                        tf.unique_with_counts(valid_gb_indices)
    self.nb_enoughp_per_gb = valid_nblocks_per_gb[1:]
    valid_gb_num = tf.shape(self.nb_enoughp_per_gb)[0]
    check_valid_gbn = tf.assert_equal(valid_gb_num, self.bsgbn,
            message="not all global block has at least one valid block with enough points")
    with tf.control_dependencies([check_valid_gbn]):
      self.nb_enoughp_per_gb = tf.identity(self.nb_enoughp_per_gb)
      self.nb_enoughp_ave = tf.reduce_mean(self.nb_enoughp_per_gb)
      self.nb_enoughp_all = tf.reduce_sum(self.nb_enoughp_per_gb)
      self.nb_lessp_all = valid_nblocks_per_gb[0]-1
      self.nb_lessp_ave = tf.reduce_mean(self.nb_lessp_all)

    #***************************************************************************
    # sampling valid bid index
    bid_inds = []
    last_num = 0
    for i in range(self.bsgbn):
      # shuffle before cut the first _nblock points
      bid_ind = tf.range(self.nb_enoughp_per_gb[i])
      apd_num = self._nblocks[self.scale] - self.nb_enoughp_per_gb[i]
      # if the valid nblock is not enough, append the first one
      f_bn_toofew = lambda: tf.concat([bid_ind, tf.tile(bid_ind[0:1], [apd_num]) ],0)
      if self._shuffle:
        f_bn_toomany = lambda: tf.random_shuffle(bid_ind)[0:self._nblocks[self.scale]]
      else:
        f_bn_toomany = lambda: bid_ind[0:self._nblocks[self.scale]]
      bid_ind_fixed = tf.case([(tf.greater(apd_num,0), f_bn_toofew)], default=f_bn_toomany)

      # merge batch size and global block dims
      bid_ind_fixed += last_num
      last_num += self.nblocks_per_gb[i]
      bid_inds.append(bid_ind_fixed)

    valid_bid_index_all_sampled = tf.concat(bid_inds, 0)
    return valid_bid_index_all_sampled


  def get_bid_point_index(self, bid_pindex):
    '''
    Get "block id index: and "point index within block".
    bid_pindex: (num_point0 * nblock_perp_3d, 2)
                                [block_id, point_index]
    bid_index__pindex_inb:  (num_point0 * nblock_perp_3d, 2)
                                [bid_index, pindex_inb]
    '''
    #(2.1) sort by block id, to put all the points belong to same block together
    # shuffle before sort for randomly sampling later
    if self._shuffle:
      bid_pindex = tf.random_shuffle(bid_pindex)
    sort_indices = tf.contrib.framework.argsort(bid_pindex[:,0],
                                                axis = 0)
    bid_pindex = tf.gather(bid_pindex, sort_indices, axis=0)
    block_id = bid_pindex[:,0]
    grouped_point_index = bid_pindex[:,1]

    #(2.2) block id -> block id index
    # get valid block num
    block_id_unique, blockid_index, npoint_per_block =\
                                                tf.unique_with_counts(block_id)
    # get blockid_index for blocks with fewer points than self._np_perb_min_include
    valid_bid_mask = tf.greater_equal(npoint_per_block, self._np_perb_min_includes[self.scale])

    self.valid_bid_index_all_sampled = self.split_bids_and_get_nblocks(block_id_unique, valid_bid_mask)

    #(2.3) Get point index per block
    #      Based on: all points belong to same block is together
    tmp0 = tf.cumsum(npoint_per_block)[0:-1]
    tmp0 = tf.concat([tf.constant([0],tf.int32), tmp0],0)
    tmp1 = tf.gather(tmp0, blockid_index)
    tmp2 = tf.range(self.npoint_grouped_all)
    #tmp2 = tf.range(tf.shape(blockid_index)[0])
    point_index_per_block = tf.expand_dims( tmp2 - tmp1,  1)
    blockid_index = tf.expand_dims(blockid_index, 1)

    bid_index__pindex_inb = tf.concat(
                                    [blockid_index, point_index_per_block], -1)
    # (2.4) sampling fixed number of points for each block
    remain_mask = tf.less(tf.squeeze(point_index_per_block,1), self._npoint_per_blocks[self.scale])
    remain_index = tf.squeeze(tf.where(remain_mask),1)
    bid_index__pindex_inb = tf.gather(bid_index__pindex_inb, remain_index)
    grouped_point_index = tf.gather(grouped_point_index, remain_index)

    # (2.5) record sampling parameters
    if self._record_samplings:
      self.record_samplings(npoint_per_block)

    if not SMALL_BUGS_IGNORE:
      check_gpn = tf.assert_less_equal(tf.shape(grouped_point_index)[0],
                            tf.reduce_sum( self._npoint_per_blocks[self.scale] *\
                                          self.nb_enoughp_per_gb),
                            message="sampled grouped point num too many")
      with tf.control_dependencies([check_gpn]):
        grouped_point_index = tf.identity(grouped_point_index)
    return bid_index__pindex_inb, grouped_point_index, block_id_unique

  def record_samplings(self, npoint_per_block):
    self.samplings[self.scale]['ngp_out_global_'] += tf.cast(self.ngp_out_global,tf.int64) / self.bsgbn
    self.samplings[self.scale]['ngp_out_block_'] += tf.cast(self.ngp_out_block,tf.int64) / self.bsgbn
    self.samplings[self.scale]['npoint_grouped_'] += tf.cast(self.npoint_grouped_ave,tf.int64)

    self.samplings[self.scale]['nbatches_'] += 1
    self.samplings[self.scale]['nb_enoughp_ave_'] += tf.cast(self.nb_enoughp_ave, tf.int64)
    self.samplings[self.scale]['nb_lessp_ave_'] += tf.cast(self.nb_lessp_ave, tf.int64)

    tmp = npoint_per_block - self._npoint_per_blocks[self.scale]
    missing_mask = tf.cast(tf.greater(tmp,0), tf.int32)
    tmp_missing = missing_mask * tmp
    less_mask = tf.cast(tf.less(tmp,0), tf.int32)
    tmp_less = less_mask * tmp


    ave_np_perb, variance = tf.nn.moments(npoint_per_block,0)
    std_np_perb = tf.sqrt(tf.cast(variance,tf.float32))
    self.samplings[self.scale]['ave_np_perb_'] += tf.cast(ave_np_perb, tf.int64)
    self.samplings[self.scale]['std_np_perb_'] +=  tf.cast(std_np_perb, tf.int64)
    nblock_has_missing = tf.reduce_sum(missing_mask)
    nblock_less = tf.reduce_sum(less_mask)
    nmissing_perb = tf.cond( tf.greater(nblock_has_missing,0),
            lambda: tf.reduce_sum(tmp_missing) / (nblock_has_missing + tf.cast(tf.equal(nblock_has_missing, 0),tf.int32)) \
                            * tf.cast(tf.greater(nblock_has_missing, 0),tf.int32),
            lambda: 0)
    self.samplings[self.scale]['nmissing_perb_'] += tf.cast(nmissing_perb, tf.int64)
    tmp = tf.cast(tf.greater(nblock_less, 0), tf.int64) * tf.cast(tf.reduce_sum(tmp_less) / \
                (nblock_less + tf.cast(tf.equal(nblock_less,0), tf.int32)), tf.int64)
    self.samplings[self.scale]['nempty_perb_'] += tmp


  def get_grouped_pindex(self, bid_index__pindex_inb, point_index):
    #(3.1) gather grouped point index
    # gen point index: (real nblock, self._npoint_per_block, 1)
    bid_index__pindex_inb_max = tf.reduce_max(bid_index__pindex_inb, 0)
    #bid_index__pindex_inb = tf.Print(bid_index__pindex_inb, [bid_index__pindex_inb_max], message="ind")
    grouped_pindex0 = tf.sparse_to_dense(bid_index__pindex_inb,
        (self._nblock_buf_maxs[self.scale] * self.bsgbn,
                           self._npoint_per_blocks[self.scale]),
                              point_index, default_value=-1)

    #(3.3) sampling fixed number of blocks when too many blocks are provided
    grouped_pindex = tf.gather(grouped_pindex0, self.valid_bid_index_all_sampled)

    #(3.4) replace -1 by the first pindex of each block
    grouped_empty_mask = tf.less(grouped_pindex,0)
    if self._empty_point_index != -1:
      # replace -1 with first one of each block
      first_pindices0 = grouped_pindex[:,0:1] + 1
      first_pindices1 = tf.tile(first_pindices0, [1,grouped_pindex.shape[1]])
      first_pindices2 = first_pindices1 * tf.cast(grouped_empty_mask, tf.int32)
      grouped_pindex = grouped_pindex + first_pindices2

    #(3.5) reshape batch size and global block dims
    aim_shape = [self.batch_size, self.global_block_num,
                 self._nblocks[self.scale], self._npoint_per_blocks[self.scale]]
    grouped_pindex = tf.reshape(grouped_pindex, aim_shape)
    grouped_empty_mask = tf.reshape(grouped_empty_mask, aim_shape)
    return grouped_pindex, grouped_empty_mask

  def rm_start_bids(self, bids_with_start):
    tmp = tf.cast(bids_with_start, tf.float32) / tf.cast(self.start_bid_step, tf.float32)
    tmp = tf.cast(tf.floor(tmp), tf.int32) * self.start_bid_step
    bids_without_start = bids_with_start - tmp
    return bids_without_start

  def all_bot_cen_tops(self, block_id_unique):
    '''
    block_id_unique: (self.nblock,)
    '''
    bids_sampling0 = tf.gather(block_id_unique, self.valid_bid_index_all_sampled)
    bids_sampling = self.rm_start_bids(bids_sampling0)
    bindex_sampling = self.block_id_to_block_index(bids_sampling, self.block_size)
    bot_cen_top = self.block_index_to_scope(bindex_sampling)

    aim_shape = [self.batch_size, self.global_block_num,
                 self._nblocks[self.scale], 9]
    bot_cen_top = tf.reshape(bot_cen_top, aim_shape)

    if self._check_optional:
      global_scope = self.global_bot_cen_top - self.global_bot_bot_bot
      correct_mask, nerr_scope = self.check_block_inblock( bot_cen_top, global_scope)
      check = tf.assert_equal(nerr_scope, 0,
                message="sub block is out of global block, scale {}".format(self.scale))
      with tf.control_dependencies([check]):
        bids_sampling = tf.identity(bids_sampling)

    return bids_sampling, bot_cen_top



  def voxelization(self, grouped_bot_cen_top, out_bot_cen_top):
    '''
    grouped_bot_cen_top: (batch_size, global_block_num, self._nblocks[self.scale], num_point, 9)
    out_bot_cen_top: (batch_size, global_block_num, self._nblocks[self.scale], 9)
    '''
    shape0 = [e.value for e in grouped_bot_cen_top.shape]
    shape1 = [e.value for e in out_bot_cen_top.shape]
    assert len(shape0) == 5
    assert len(shape1) == 4
    assert shape0[0] == shape1[0] == self.batch_size
    assert shape0[1] == shape1[1] == self.global_block_num
    assert shape0[2] == shape1[2]
    assert shape0[-1] == shape1[-1] == 9

    if self.scale <= 1:
      return tf.zeros([self.batch_size]+[0]*4, tf.int32)
    out_bot_cen_top = tf.expand_dims(out_bot_cen_top, -2)
    grouped_bot_aligned = grouped_bot_cen_top[...,0:3] - out_bot_cen_top[...,0:3]
    vox_index0 = grouped_bot_aligned / self._strides[self.scale-1]
    vox_index1 = tf.round(vox_index0)
    vox_index = tf.cast(vox_index1, tf.int32)

    self.global_bot_cen_top

    if self._check_voxelization_optial:
      vox_size = self._vox_sizes[self.scale]
      vox_index_align_err = tf.abs(vox_index0 - vox_index1)
      max_vox_index_align_err = tf.reduce_max(vox_index_align_err)
      check_align = tf.assert_less(max_vox_index_align_err, 5e-5,
                                  message="scale {} points are not aligned".format(self.scale))

      check_low_bound = tf.assert_greater_equal(vox_index, 0,
                                                message="vox index < 0")
      check_up_bound = tf.assert_less(vox_index, vox_size,
                      message="scale {} vox index > size-1".format(self.scale))
      with tf.control_dependencies([check_align, check_low_bound, check_up_bound]):
        vox_index = tf.identity(vox_index)
    return vox_index


  def grouped_mean(self, grouped_xyz, empty_mask):
    valid_mask = 1-empty_mask
    valid_num = tf.expand_dims(tf.cast( tf.reduce_sum(valid_mask, 1), tf.float32),1)
    valid_mask = tf.expand_dims(tf.cast(valid_mask, tf.float32),2)
    tmp_sum = tf.reduce_sum(grouped_xyz * valid_mask, 1)
    grouped_mean = tmp_sum / valid_num
    return grouped_mean


def main_input_pipeline(DATASET_NAME, filenames, sg_settings, dset_metas, batch_size, cycles=1, num_epoch=1):
  dataset_meta = DatasetsMeta(DATASET_NAME)
  num_classes = dataset_meta.num_classes
  log_path = os.path.dirname(os.path.dirname(filenames[0]))
  log_path = os.path.join(log_path, 'sg_log')
  if not os.path.exists(log_path):
    os.makedirs(log_path)


  bsg = BlockGroupSampling(sg_settings, log_path)
  bsg.show_settings()
  with tf.Graph().as_default():
   with tf.device('/device:GPU:0'):
    dataset = tf.data.TFRecordDataset(filenames,
                                        compression_type="",
                                        buffer_size=1024*100,
                                        num_parallel_reads=1)

    #dataset.shuffle(buffer_size = 10000)

    is_training = False

    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        lambda value: parse_pl_record(value, is_training, dset_metas, bsg),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=True))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    get_next = dataset.make_one_shot_iterator().get_next()
    features_next, label_next = get_next

    points_next = features_next['points']
    grouped_pindex_next = []
    vox_index_next = []
    grouped_xyz_next = []
    empty_mask_next = []
    bot_cen_top_next = []
    others_next = []

    num_scale = bsg._num_scale
    for s in range(num_scale):
      grouped_pindex_next.append(features_next['grouped_pindex_%d'%(s)])
      vox_index_next.append(features_next['vox_index_%d'%(s)])
      grouped_xyz_next.append(features_next['grouped_bot_cen_top_%d'%(s)])
      empty_mask_next.append(features_next['empty_mask_%d'%(s)])
      bot_cen_top_next.append(features_next['bot_cen_top_%d'%(s)])

      others_next.append({})
      for name0 in features_next:
        if 'OTS_' in name0:
          name = name0.split('OTS_')[1]
          others_next[s][name] = features_next[name0]
    #samplings_next = bsg.samplings

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
      for i in xrange(cycles):
        #points_i = sess.run(points_next)
        #grouped_pindex = sess.run(grouped_pindex_next)
        #empty_masks = sess.run(empty_mask_next)
        #bot_cen_tops = sess.run(bot_cen_top_next)
        #others = sess.run(others_next)

        t0 = time.time()
        points, grouped_pindex, vox_index, grouped_xyzs, empty_masks, bot_cen_tops, others = \
          sess.run([points_next, grouped_pindex_next, vox_index_next, grouped_xyz_next, empty_mask_next, bot_cen_top_next, others_next] )
        t_batch = time.time()-t0
        xyzs = points[:,:,0:3]
        print("\n\n{} t_batch:{} t_frame:{}\n".format(i, t_batch*1000, t_batch/batch_size*1000))


        #for frame in range(batch_size):
        #  for s in range(num_scale):
        #    others[s]['empty_mask'] = empty_masks[s]
        #    if bsg._gen_ply:
        #      valid_flag = ''
        #      if not bsg._shuffle:
        #        valid_flag += '_NoShuffle'
        #      bids_sampling = others[s]['bids_sampling_%d'%(s)][frame,0]
        #      gen_plys(DATASET_NAME, frame, points[frame], grouped_xyzs[s][frame], bot_cen_tops[s][frame],\
        #                bids_sampling, valid_flag, 'S%d'%(s))

  return xyzs, grouped_xyzs, others, bsg._shuffle


def main_gpu(DATASET_NAME, filenames, sg_settings, dset_metas, batch_size, cycles=1, num_epoch=10):
  tg0 = time.time()
  dataset_meta = DatasetsMeta(DATASET_NAME)
  num_classes = dataset_meta.num_classes
  log_path = os.path.dirname(os.path.dirname(filenames[0]))
  log_path = os.path.join(log_path, 'sg_log')
  if not os.path.exists(log_path):
    os.makedirs(log_path)

  with tf.Graph().as_default():
   with tf.device('/device:GPU:0'):
    dataset = tf.data.TFRecordDataset(filenames)

    #dataset.shuffle(buffer_size = 10000)
    dataset = dataset.repeat(num_epoch)
    is_training = False

    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        lambda value: parse_pl_record(value, is_training, dset_metas),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=True))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    get_next = dataset.make_one_shot_iterator().get_next()
    features_next, label_next = get_next

    points_next = features_next['points']
    tg1 = time.time()
    print('before pre sg:{}'.format(tg1-tg0))
    dsb_next = pre_sampling_grouping(points_next, sg_settings, log_path)
    tg2 = time.time()
    print('pre sg:{}'.format(tg2-tg1))

    init = tf.global_variables_initializer()
    tf.get_default_graph().finalize()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    #config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
      for i in xrange(cycles):
        t0 = time.time()
        dsb = sess.run(dsb_next)
        t_batch = time.time()-t0
        print("t_batch:{} t_frame:{}".format( t_batch * 1000, t_batch/batch_size*1000 ))

        print('group OK %d'%(i))


def pre_sampling_grouping(points, sg_settings, log_path):
  # get the indices for grouping and sampling on line
  #t0 = tf.timestamp()
  t0 = time.time()
  with tf.device('/device:CPU:0'):
    bsg = BlockGroupSampling(sg_settings, log_path)
    t1 = time.time()
    print('create class bsg:{}'.format(t1-t0))
    dsbb = {}

    dsb = {}
    dsb['grouped_pindex'], dsb['vox_index'], dsb['grouped_xyz'], dsb['empty_mask'],\
      dsb['bot_cen_top'], nb_enough_p, others =\
      bsg.grouping_multi_scale(points[...,0:3])

    #*************************************************************
    is_show_inputs = True
    if is_show_inputs:
      print('\n\n\n')

    #*************************************************************
    # get inputs for each global block
    grouped_pindex_global = tf.squeeze(dsb['grouped_pindex'][0], 2)
    shape0 = [e.value for e in grouped_pindex_global.shape]
    grouped_pindex_global = tf.expand_dims(grouped_pindex_global, -1)
    tmp0 = tf.reshape(tf.range(0, batch_size, 1), [-1, 1, 1, 1])
    tmp0 = tf.tile(tmp0, [1]+shape0[1:]+[1])
    tmp1 = tf.concat([tmp0, grouped_pindex_global], -1)
    dsb['points'] = tf.gather_nd(points, tmp1)

    #sg_t_batch = (tf.timestamp() - t0)*1000
    #tf.summary.scalar('sg_t_batch', sg_t_batch)
    #sg_t_frame = sg_t_batch / tf.cast(batch_size,tf.float64)
    #tf.summary.scalar('sg_t_frame', sg_t_frame)

    return dsb


def main_eager(DATASET_NAME, filenames, sg_settings, dset_metas, batch_size, cycles=1):
  tf.enable_eager_execution()
  dataset_meta = DatasetsMeta(DATASET_NAME)
  num_classes = dataset_meta.num_classes

  log_path = os.path.dirname(os.path.dirname(filenames[0]))
  log_path = os.path.join(log_path, 'sg_log')
  if not os.path.exists(log_path):
    os.makedirs(log_path)

  dataset = tf.data.TFRecordDataset(filenames,
                                      compression_type="",
                                      buffer_size=1024*100,
                                      num_parallel_reads=1)

  batch_size = batch_size
  is_training = False

  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(
    tf.contrib.data.map_and_batch(
      lambda value: parse_pl_record(value, is_training, dset_metas),
      batch_size=batch_size,
      num_parallel_batches=1,
      drop_remainder=False))
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  get_next = dataset.make_one_shot_iterator().get_next()

  bsg = BlockGroupSampling(sg_settings, log_path)
  bsg.show_settings()

  for n in range(cycles):
    features_next, label_next = get_next
    points_next = features_next['points'][:,:,0:3]

    grouped_pindex, vox_index, grouped_bot_cen_top, empty_mask, out_bot_cen_top, nb_enoughp_ave, others = \
      bsg.grouping_multi_scale(points_next[:,:,0:3])

    samplings_i = bsg.samplings
    samplings_np_ms = []
    for s in range(len(samplings_i)):
      samplings_np_ms.append({})
      for name in samplings_i[s]:
        samplings_np_ms[s][name] = samplings_i[s][name].numpy()
    bsg.show_samplings_np_multi_scale(samplings_np_ms)

  points = points_next.numpy()
  num_scale = len(grouped_pindex)
  for s in range(num_scale):
    grouped_bot_cen_top[s] = grouped_bot_cen_top[s].numpy()
    out_bot_cen_top[s] = out_bot_cen_top[s].numpy()
    for item in others:
      others[item][s] = others[item][s].numpy()

  # gen ply
  if bsg._gen_ply:
    valid_flag = ''
    if not bsg._shuffle:
      valid_flag += '_NoShuffle'
    for f in range(batch_size):
      for s in range(num_scale):
        global_block_num = grouped_bot_cen_top[s].shape[1]
        for g in range(global_block_num):
          gen_plys(DATASET_NAME, f, points[f], grouped_bot_cen_top[s][f,g],
                  out_bot_cen_top[s][f,g], valid_flag, '_ES%sGb%d'%(s,g))

  return points, grouped_bot_cen_top, others, bsg._shuffle


def gen_plys(DATASET_NAME, frame, points, grouped_bot_cen_top,
             out_bot_cen_top, valid_flag='', main_flag=''):
  path = '/tmp/%d_plys'%(frame) + main_flag
  grouped_xyz = grouped_bot_cen_top[...,3:6]

  xyz_shape = grouped_xyz.shape
  out_shape = out_bot_cen_top.shape
  assert len(xyz_shape) == 3
  assert len(out_shape) == 2


  if type(points)!=type(None):
    ply_fn = '%s/points.ply'%(path)
    create_ply_dset(DATASET_NAME, points, ply_fn,
                    extra='random_same_color')

  ply_fn = '%s/grouped_points_%s.ply'%(path, valid_flag)
  create_ply_dset(DATASET_NAME, grouped_xyz, ply_fn,
                  extra='random_same_color')

  ply_fn = '%s/block_center_%s.ply'%(path, valid_flag)
  create_ply_dset(DATASET_NAME, out_bot_cen_top[...,3:6], ply_fn,
                  extra='random_same_color')

  ply_fn = '%s/blocks%s.ply'%(path, valid_flag)
  draw_blocks_by_bot_cen_top(ply_fn, out_bot_cen_top, random_crop=0)

  tmp = np.random.randint(0, min(grouped_xyz.shape[0], 10))
  tmp = np.arange(0, min(grouped_xyz.shape[0],10))
  for j in tmp:
    ply_fn = '%s/blocks%s/%d_blocks.ply'%(path, valid_flag, j)
    draw_blocks_by_bot_cen_top(ply_fn, out_bot_cen_top[j:j+1], random_crop=0)
    ply_fn = '%s/blocks%s/%d_points.ply'%(path, valid_flag, j)
    create_ply_dset(DATASET_NAME, grouped_xyz[j], ply_fn,
                  extra='random_same_color')



def main(filenames, dset_metas):
  from utils.sg_settings import get_sg_settings
  sg_settings = get_sg_settings('32768_1024_64')

  batch_size = 12
  if len(sys.argv) > 1:
    main_flag = sys.argv[1]
    if len(sys.argv) > 2:
      batch_size = int(sys.argv[2])
      print('batch_size {}'.format(batch_size))
  else:
    main_flag = 'g'
    #main_flag = 'G'
    #main_flag = 'eg'
    #main_flag = 'e'
  print(main_flag)

  file_num = 12311
  num_epoch = 1
  cycles = (file_num // batch_size) * num_epoch
  cycles = 200

  if 'e' in main_flag:
    xyzs_E, grouped_xyzs_E, others_E, shuffle_E = \
      main_eager(DATASET_NAME, filenames, sg_settings, dset_metas, batch_size, cycles)
  if 'g' in main_flag:
    xyzs, grouped_xyzs, others, shuffle = \
      main_input_pipeline(DATASET_NAME, filenames, sg_settings, dset_metas, batch_size, cycles, num_epoch)
  if 'G' in main_flag:
    main_gpu(DATASET_NAME, filenames, sg_settings, dset_metas, batch_size, cycles, num_epoch)

  if main_flag=='eg' and shuffle==False and shuffle_E==False:
    assert xyzs.shape[0] == xyzs_E.shape[0], "Make batch_size=batch_size in main "
    batch_size = xyzs.shape[0]
    for b in xrange(batch_size):
      assert (xyzs_E[b] == xyzs[b]).all(), 'time %d xyz different'%(b)
      print('time %d xyzs of g and e is same'%(b))

      num_scale = len(grouped_xyzs)
      for s in xrange(num_scale):
        for name in others_E[s]:
          check = (others_E[s][name][b] == others[s][name][b]).all()
          if not check:
            print('time %d scale %d others- %s different'%(b, s, name))
            err = others_E[s][name][b] != others[s][name][b]
            nerr = np.sum(err)
            print(others[s][name][b])
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            pass
          else:
            print('time %d scale %d others - %s of g and e is same'%(b, s, name))

        diff = grouped_xyzs_E[s][b] - grouped_xyzs[s][b]
        diff = np.sum(diff, -1)
        nerr = np.sum(diff != 0)
        if nerr!=0:
          print("time %d, scale %d, grouped_xyz nerr=%d"%(b, s, nerr))
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
        else:
          print('time %d, sclae %s, grouped_xyzs of g and e is same'%(b, s))


if __name__ == '__main__':
  import random

  DATASET_NAME = 'MODELNET40'
  DATASET_NAME = 'MATTERPORT'
  raw_tfrecord_path = '/home/z/Research/SparseVoxelNet/data/{}_H5TF/raw_tfrecord'.format(DATASET_NAME)
  data_path = os.path.join(raw_tfrecord_path, 'data')
  #data_path = os.path.join(raw_tfrecord_path, 'merged_data')
  tmp = '*'
  #tmp = 'Vvot9Ly1tCj_region24*'
  filenames = glob.glob(os.path.join(data_path, tmp+'.tfrecord'))
  #random.shuffle(filenames)
  assert len(filenames) >= 1, data_path

  dset_metas = get_dset_metas(raw_tfrecord_path)

  main(filenames, dset_metas)
  #get_data_summary_from_tfrecord(filenames, raw_tfrecord_path)


