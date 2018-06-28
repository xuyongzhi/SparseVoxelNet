# xyz Dec 2017
# Do 3d point cloud  sample and group by block index
import tensorflow as tf
import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+'/../utils')
from block_data_prep_util import GlobalSubBaseBLOCK
import geometric_util as geo_util
import geometric_tf_util as geo_tf_util
import tf_util
import numpy as np

DEBUG_TMP = True


# IS_merge_blocks_while_fix_bmap should be set exactly based on the bidxmap
# configuration. This is origibally set in NETCONFIG. But the configuration is
# not obtained here from bxmap automatically. Should be set manually.
IS_merge_blocks_while_fix_bmap = 1
IsTolerateBug = True
InDropMethod = 'set0'

'''
Checking list:
    new_xyz
'''

def shape_str(tensor_ls):
    shape_str = ''
    for i in range(len(tensor_ls)):
        if tensor_ls[i] == None:
            shape_str += '\t None'
        else:
            shape_str += '\t' + str( [s.value for s in tensor_ls[i].shape] )
        if i < len(tensor_ls)-1:
            shape_str += '\n'
    return shape_str


def get_flatten_bidxmap_concat( flatten_bidxmaps, flatten_bm_extract_idx, cascade_id ):
        '''
            flatten_bidxmaps: (2, 26368, 2)
            flatten_bm_extract_idx:
                array([[    0,     0],
                       [25600,     2],
                       [26112,     2],
                       [26368,     2]], dtype=int32)
        '''
        batch_size = flatten_bidxmaps.get_shape()[0].value
        start = flatten_bm_extract_idx[cascade_id]
        end = flatten_bm_extract_idx[cascade_id+1]
        flatten_bidxmap_i = flatten_bidxmaps[ :,start[0]:end[0],: ]

        batch_idx = tf.reshape( tf.range(batch_size),[batch_size,1,1] )
        flatten_bidxmap_i_shape1 = flatten_bidxmap_i.get_shape()[1].value
        batch_idx = tf.tile( batch_idx,[1,flatten_bidxmap_i_shape1,1] )
        flatten_bidxmap_i_concat = tf.concat( [batch_idx,flatten_bidxmap_i],axis=-1,name="flatten_bidxmap%d_concat"%(cascade_id) )
        return flatten_bidxmap_i_concat

def pointnet_sa_module(cascade_id, xyz, points, bidmap, mlp_configs, block_bottom_center_mm, configs, sgf_config_pls,
                       is_training, bn_decay,scope,bn=True, tnet_spec=None, use_xyz=True, IsShowModel=False):
    '''
    Input cascade_id==0:
        xyz is grouped_points: (batch_size,nsubblock0,npoint_subblock0,6)
        points: None
        bidmap: None
    Input cascade_id==1:
        xyz: (batch_size,nsubblock0,3)
        points: (batch_size,nsubblock0,channel)
        bidmap: (batch_size,nsubblock1,npoint_subblock1)
    Medium cascade_id==1:
        grouped_xyz: (batch_size,nsubblock1,npoint_subblock1,3)
        new_xyz: (batch_size,nsubblock1,3)
        group_points: (batch_size,nsubblock1,npoint_subblock1,channel)

    output cascade_id==1:
        new_xyz: (batch_size,nsubblock1,3)
        new_points: (batch_size,nsubblock1,channel)
    '''
    block_bottom_center_mm = tf.cast(block_bottom_center_mm, tf.float32, name='block_bottom_center_mm') # gpu_0/sa_layer3/block_bottom_center_mm:0
    batch_size = xyz.get_shape()[0].value
    with tf.variable_scope(scope) as sc:
        cascade_num = configs['flatten_bm_extract_idx'].shape[0]-1  # include global here (Note: cascade_num does not include global in block_pre_util )
        assert configs['sub_block_step_candis'].size == cascade_num-1
        if cascade_id==0:
            indrop_keep_mask = tf.get_default_graph().get_tensor_by_name('indrop_keep_mask:0') # indrop_keep_mask:0

        assert len(xyz.shape) == 3

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
            if cascade_id==0 and  len(indrop_keep_mask.get_shape()) != 0:
                grouped_indrop_keep_mask = tf.gather_nd( indrop_keep_mask, bidmap_concat, name='grouped_indrop_keep_mask' )  # gpu_0/sa_layer0/grouped_indrop_keep_mask:0

        # new_xyz is the "voxel center" or "mean position of points in the voxel"
        if configs['mean_grouping_position'] and (not mlp_configs['block_learning']=='3DCNN'):
            new_xyz = tf.reduce_mean(grouped_xyz,-2)
        else:
            new_xyz = block_bottom_center_mm[:,:,3:6] * tf.constant( 0.001, tf.float32 )
        # the mid can be mean or block center, decided by configs['mean_grouping_position']
        sub_block_mid = tf.expand_dims( new_xyz,-2, name = 'sub_block_mid' )   # gpu_1/sa_layer0/sub_block_mid
        global_block_mid = tf.reduce_mean( sub_block_mid,1, keepdims=True, name = 'global_block_mid' )
        grouped_xyz_submid = grouped_xyz - sub_block_mid
        grouped_xyz_glomid = grouped_xyz - global_block_mid

        grouped_xyz_feed = []
        if 'raw' in configs['xyz_elements']:
            grouped_xyz_feed.append( grouped_xyz )
        if 'sub_mid' in configs['xyz_elements']:
            grouped_xyz_feed.append( grouped_xyz_submid )
        if 'global_mid' in configs['xyz_elements']:
            grouped_xyz_feed.append( grouped_xyz_glomid )
        grouped_xyz_feed = tf.concat( grouped_xyz_feed, -1 )

        if cascade_id==0:
            # xyz must be at the first in feed_data_elements !!!!
            grouped_points = tf.concat( [grouped_xyz_feed, grouped_points[...,3:]],-1 )

            if len(indrop_keep_mask.get_shape()) != 0:
                if InDropMethod == 'set1st':
                    # set all the dropped item as the first item
                    tmp1 = tf.multiply( grouped_points, grouped_indrop_keep_mask )
                    points_1st = grouped_points[:,:,0:1,:]
                    points_1st = tf.tile( points_1st, [1,1,grouped_points.shape[2],1] )
                    indrop_mask_inverse = 1 - grouped_indrop_keep_mask
                    tmp2 = indrop_mask_inverse * points_1st
                    grouped_points = tf.add( tmp1, tmp2, name='grouped_points_droped' ) # gpu_0/sa_layer0/grouped_points_droped
                    #tf.add_to_collection( 'check', grouped_points )
                elif InDropMethod == 'set0':
                    valid_mask = tf.logical_and( valid_mask, tf.equal(grouped_indrop_keep_mask,0), name='valid_mask_droped' )   # gpu_1/sa_layer0/valid_mask_droped

        elif use_xyz:
            grouped_points = tf.concat([grouped_xyz_feed, grouped_points],axis=-1)

        tf.add_to_collection( 'grouped_xyz', grouped_xyz )
        tf.add_to_collection( 'grouped_xyz_submid', grouped_xyz_submid )
        tf.add_to_collection( 'grouped_xyz_glomid', grouped_xyz_glomid )

        if cascade_id>0 and use_xyz and (not cascade_id==cascade_num-1):
            grouped_points = tf.concat([grouped_xyz_feed, grouped_points],axis=-1)

        nsample = grouped_points.get_shape()[2].value  # the conv kernel size

        if IsShowModel:
            print('\n\npointnet_sa_module cascade_id:%d\n xyz:%s\n grouped_xyz:%s\n new_xyz:%s\n grouped_points:%s\n nsample:%d'%(
                    cascade_id, shape_str([xyz]), shape_str([grouped_xyz]), shape_str([new_xyz]), shape_str([grouped_points]), nsample))

        new_points = grouped_points
        if valid_mask!=None:
            new_points = new_points * tf.cast(valid_mask[:,:,:,0:1], tf.float32)

        if 'growth_rate'in mlp_configs['point_encoder'][cascade_id]:
            new_points = tf_util.dense_net( new_points, mlp_configs['point_encoder'][cascade_id], bn, is_training, bn_decay,\
                                           scope = 'dense_cascade_%d_point_encoder'%(cascade_id) , is_show_model = IsShowModel )
        else:
            for i, num_out_channel in enumerate(mlp_configs['point_encoder'][cascade_id]):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay)
                if configs['Cnn_keep_prob']<1:
                    if ( not configs['only_last_layer_ineach_cascade'] ) or i == len(mlp_configs['point_encoder'][cascade_id])-1:
                        new_points = tf_util.dropout(new_points, keep_prob=configs['Cnn_keep_prob'], is_training=is_training, scope='dropout', name='cnn_dp%d'%(i))
                if IsShowModel:
                    print('point encoder1 %d, new_points:%s'%(i, shape_str([new_points])))


        if cascade_id == 0:
            root_point_features = new_points
            #if InDropMethod == 'set0':
            #    if len(indrop_keep_mask.get_shape()) != 0:
            #            new_points = tf.identity(new_points,'points_before_droped') # gpu_0/sa_layer0/points_before_droped:0
            #            new_points = tf.multiply( new_points, grouped_indrop_keep_mask, name='droped_points' )   # gpu_0/sa_layer0/droped_points:0
        else:
            root_point_features = None

        pooling = mlp_configs['block_learning']
        if pooling == '3DCNN' and ( cascade_id == 0):
            pooling = 'max'

        #if pooling=='avg':
        #    new_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        #elif pooling=='weighted_avg':
        #    with tf.variable_scope('weighted_avg1'):
        #        dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
        #        exp_dists = tf.exp(-dists * 5)
        #        weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
        #        new_points *= weights # (batch_size, npoint, nsample, mlps_0[-1])
        #        new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        if pooling=='max':
            # Even the grouped_points and grouped_xyz are 0 for invalid points, the
            # vaule after mlp will not be. It has to be set as 0 forcely before
            # pooling.
            if valid_mask!=None:
                new_points = new_points * tf.cast(valid_mask[:,:,:,0:1], tf.float32)
            new_points = tf.identity( new_points, 'points_before_max' )             # gpu_0/sa_layer0/points_before_max
            new_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='points_after_max')
        #elif pooling=='min':
        #    new_points = tf_util.max_pool2d(-1*new_points, [1,nsample], stride=[1,1], padding='VALID', scope='minpool1')
        #elif pooling=='max_and_avg':
        #    avg_points = tf_util.max_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='maxpool1')
        #    max_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        #    new_points = tf.concat([avg_points, max_points], axis=-1)
        elif pooling == '3DCNN':
            new_points = grouped_points_to_voxel_points( cascade_id, new_points, valid_mask, block_bottom_center_mm, configs,  grouped_xyz, IsShowVoxelModel=IsShowModel )
            if IsShowModel:
                print('voxel points:%s'%(shape_str([new_points])))
            for i, num_out_channel in enumerate( mlp_configs['voxel_channels'][cascade_id] ):
                kernel_i = [mlp_configs['voxel_kernels'][cascade_id][i]]*3
                stride_i = [mlp_configs['voxel_strides'][cascade_id][i]]*3
                if new_points.shape[1]%2 == 0:
                    padding_i = np.array([[0,0],[1,0],[1,0],[1,0],[0,0]]) * mlp_configs['voxel_paddings'][cascade_id][i]
                else:
                    padding_i = np.array([[0,0],[1,1],[1,1],[1,1],[0,0]]) * mlp_configs['voxel_paddings'][cascade_id][i]
                new_points = tf.pad( new_points, padding_i, "CONSTANT" )

                if type(num_out_channel) == int:
                    new_points = tf_util.conv3d(new_points,
                                                num_out_channel,
                                                kernel_i,
                                                scope = '3dconv_%d'%(i),
                                                stride = stride_i,
                                                padding = 'VALID',
                                                bn=bn,
                                                is_training = is_training,
                                                bn_decay = bn_decay,
                                                name = 'points_3dcnn_%d'%(i) )
                    if IsShowModel:
                        print('block learning by 3dcnn %d, new_points:%s'%(i, shape_str([new_points])))
                elif num_out_channel == 'max':
                    new_points = tf_util.max_pool3d( new_points,
                                                    kernel_i,
                                                    scope = '3dmax_%d'%(i),
                                                    stride = stride_i,
                                                    padding = 'VALID')
                    if IsShowModel:
                        print('block learning max pooling %d, new_points:%s'%(i, shape_str([new_points])))
                elif num_out_channel == 'avg':
                    new_points = tf_util.avg_pool3d( new_points,
                                                    kernel_i,
                                                    scope = '3dmax_%d'%(i),
                                                    stride = stride_i,
                                                    padding = 'VALID')
                    if IsShowModel:
                        print('block learning avg pooling %d, new_points:%s'%(i, shape_str([new_points])))
                # gpu_0/sa_layer1/3dconv_0/points_3dcnn_0:0
                if configs['Cnn_keep_prob']<1:
                    if ( not configs['only_last_layer_ineach_cascade'] ) or i == len(mlp_configs['voxel_channels'][cascade_id])-1:
                        new_points = tf_util.dropout(new_points, keep_prob=configs['Cnn_keep_prob'], is_training=is_training, scope='dropout', name='3dcnn_dp%d'%(i))
                # gpu_0/sa_layer4/3dconv_0/points_3dcnn_0:0
            new_points = tf.squeeze( new_points, [1,2,3] )
            new_points = tf.reshape( new_points, [batch_size, -1, 1, new_points.shape[-1].value] )

        if IsShowModel:
            print('after %s, new_points:%s'%( pooling, shape_str([new_points])))


        if 'growth_rate'in mlp_configs['block_encoder'][cascade_id]:
            new_points = tf_util.dense_net( new_points, mlp_configs['block_encoder'][cascade_id], bn, is_training, bn_decay, scope = 'dense_cascade_%d_block_encoder'%(cascade_id) , is_show_model = IsShowModel )
        else:
            for i, num_out_channel in enumerate(mlp_configs['block_encoder'][cascade_id]):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay)
                if configs['Cnn_keep_prob']<1:
                    if ( not configs['only_last_layer_ineach_cascade'] ) or i == len(mlp_configs['block_encoder'][cascade_id])-1:
                        new_points = tf_util.dropout(new_points, keep_prob=configs['Cnn_keep_prob'], is_training=is_training, scope='dropout', name='cnn_dp%d'%(i))
                if IsShowModel:
                    print('block encoder %d, new_points:%s'%(i, shape_str([new_points])))
        # (2, 512, 1, 64)
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlps_1[-1])

        if IsShowModel:
            print('pointnet_sa_module return\n new_xyz: %s\n new_points:%s\n\n'%(shape_str([new_xyz]),shape_str([new_points])))
            #import pdb;pdb.set_trace()
        # (2, 512, 64)
        return new_xyz, new_points, root_point_features

def grouped_points_to_voxel_points (cascade_id, new_points, valid_mask, block_bottom_center_mm, configs, grouped_xyz, IsShowVoxelModel=False):
    cascade_num = configs['sub_block_step_candis'].size+1
    block_bottom_center_mm = tf.identity( block_bottom_center_mm,'block_bottom_center_mm' )      # gpu_0/sa_layer3/block_bottom_center_mm:0
    new_points = tf.identity(new_points,name='points_tov') # gpu_0/sa_layer4/points_tov:0
    c500 = tf.constant([500],tf.float32)
    c1000 = tf.constant([1000],tf.float32)
    c1 = tf.constant([1,1,1],tf.float32)
    step_last_org = configs['sub_block_step_candis'][cascade_id-1] * c1
    step_last = tf.minimum( step_last_org, configs['max_step_stride'], name='step_last' )    # gpu_0/sa_layer1/step_last:0
    step_last = tf.expand_dims(step_last,1)
    stride_last_org = configs['sub_block_stride_candis'][cascade_id-1] * c1
    stride_last = tf.minimum( stride_last_org, configs['max_step_stride'], name='stride_last' )  # gpu_0/sa_layer1/stride_last:0
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

    if configs['dataset_name'] == 'MODELNET40':
        IsTolerateBug = 2
        IsTolerateBug = 0
    else:
        IsTolerateBug = 1

    if cascade_id==cascade_num-1:
        # only in this global cascde, the steps and strides in each dimension
        # can be different
        if configs['dataset_name'] == 'MODELNET40' and configs['global_step'][0]==3.5:
            configs['global_step'] = np.array( [2.3,2.3,2.3] )
        max_indice_f = ( np.abs(configs['global_step']) - np.array([1,1,1])*configs['sub_block_step_candis'][cascade_id-1] ) / (np.array([1,1,1])*configs['sub_block_stride_candis'][cascade_id-1])
        max_indice_v = np.rint( max_indice_f )
        if configs['dataset_name'] != 'MODELNET40':
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
        if IsShowVoxelModel:
            print( 'cascade:%d (global) \tvoxel size:%s'%(cascade_id, voxel_size) )

    else:
        max_indice_f = ( configs['sub_block_step_candis'][cascade_id] - configs['sub_block_step_candis'][cascade_id-1] ) / configs['sub_block_stride_candis'][cascade_id-1]
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
        if IsShowVoxelModel:
            print( 'cascade:%d \tvoxel size:%s'%(cascade_id, voxel_size) )


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
    if configs['aug_types']['RotateVox']:
        voxel_points = rotate_voxel_randomly( voxel_points, configs )
    return voxel_points


def rotate_voxel_randomly( voxel_points, configs ):
    voxel_shape = np.array( voxel_points.shape[1:4].as_list() )
    grid = np.indices( voxel_shape )
    grid = np.transpose( grid,(1,2,3,0) )

    version = 'tf'
    #---------------------------------------------------------------------------
    if version == 'numpy':
        rz_angle = np.pi * 0.5
        R = np.rint( geo_util.Rz( rz_angle ) ).astype(np.int32)
        grid_r = np.matmul( grid, R )
        # The rotation center is not voxel center, but the bottom. An offsetis required.
        offset_mask = np.sum(R,0) == -1
        offset = offset_mask * (voxel_shape-1)
        grid_ro = grid_r + offset

    #---------------------------------------------------------------------------
    if version == 'tf':
        #rz_angle = tf.random_uniform( shape=(), minval=-3, maxval=4, dtype=tf.int32 )
        #rz_angle = tf.cast(rz_angle,tf.float32) * tf.constant(np.pi * 0.5)
        RotateVoxXYZChoices = configs['aug_types']['RotateVoxXYZChoices']
        axis = ['x','y','z']
        rxyz = tf.Variable([0,0,0], dtype=tf.float32, trainable=False)
        for i in range(3):
            if len ( RotateVoxXYZChoices[i] )>0:
                r_angle_i = tf.random_crop( RotateVoxXYZChoices[i], size=[1], name='r_angle_%d'%(i) )[0]
                rxyz = tf.scatter_update( rxyz, i, r_angle_i )      # gpu_0/sa_layer1/rxyz
        rxyz = tf.identity(rxyz,'rxyz')
        R = geo_tf_util.tf_EulerRotate( rxyz, order='xyz' )
        R = tf.rint( R )
        R = tf.cast( R, tf.int32, name='R' )    # gpu_0/sa_layer1/R:0
        grid = tf.Variable( grid, dtype=tf.int32, name='grid', trainable=False )
        grid_ = tf.reshape(grid, (-1,3) )
        grid_r = tf.matmul( grid_, R )
        grid_r = tf.reshape(grid_r, grid.shape, name='grid_r')
        # The rotation center is not voxel center, but the bottom. An offsetis required.
        offset_mask = tf.equal( tf.reduce_sum( R, axis=0 ), -1 )
        offset_mask = tf.cast( offset_mask, tf.int32, 'offset_mask' )
        offset = tf.multiply( offset_mask, (voxel_shape-1), name='offset' )
        grid_ro = tf.add( grid_r, offset, name='grid_ro')       # gpu_0/sa_layer1/grid_ro:0

    #---------------------------------------------------------------------------
    #voxel_points = tf.identity( voxel_points,'voxel_points_before_r' )                    # gpu_0/sa_layer1/voxel_points_before_r:0
    voxel_points = tf.transpose( voxel_points, [1,2,3,4,0] )
    voxel_points = tf.gather_nd( voxel_points, grid_ro )
    voxel_points = tf.transpose( voxel_points, [4,0,1,2,3], name='voxel_points_rotated' ) # gpu_0/sa_layer1/voxel_points_rotated:0

    tf.add_to_collection('check', voxel_points)
    return voxel_points


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


def pointnet_fp_module( cascade_id, num_neighbors, points1, points2, flatten_bidxmap, fbmap_neighbor_idis, mlps_e1, mlps_fp, is_training, bn_decay, scope, configs, bn=True, IsShowModel=False):
    '''
    in Qi's code, 3 larger balls are weighted back-propogated to one point
    Here, I only back-propogate one

    Input:
        points1 (cascade_id=2): (2, 256, 256)
        points2 (cascade_id=3): (2, 64, 512)
        flatten_bidxmap: (B,num_point,self.flatbxmap_max_nearest_num,2)
                    [:,:,:,0]: aim_b_index
                    [:,:,:,1]: point_index_in_aimb  (useless when cascade_id>0)
        fbmap_neighbor_idis: (B,num_point,self.flatbxmap_max_nearest_num,1)
                    [:,:,:,2]: index_distance
        mlps_fp: [256,256]
    Output:
        new_points1: (2, 256, 256)
    '''
    if IsShowModel:
        print('\n\npointnet_fp_module %s\n points1: %s\n points2: %s\n flatten_bidxmap: %s\n'%( scope, shape_str([points1]), shape_str([points2]), shape_str([flatten_bidxmap]) ))
    with tf.variable_scope(scope) as sc:
        assert len(flatten_bidxmap.get_shape()) == 4
        if cascade_id == 0:
            # points1 is grouped point features
            assert len(points1.get_shape()) == 4
        else:
            # points1 is flat point features
            assert len(points1.get_shape()) == 3
            assert flatten_bidxmap.shape[1] == points1.shape[1]
        batch_size = points2.get_shape()[0].value
        batch_idx0 = tf.reshape( tf.range(batch_size),[batch_size,1,1,1] )
        point1_num = flatten_bidxmap.get_shape()[1].value


        if cascade_id == 0:
            # convert grouped points1 to flat point features
            #flatten_bidxmap_aimbidx1 = flatten_bidxmap[:,:,0,0:2]  # (2, 256, 1)
            #flatten_bidxmap_aimbidx_concat1 = tf.concat( [batch_idx, flatten_bidxmap_aimbidx1], axis=-1 )
            #points1 = tf.gather_nd(points1, flatten_bidxmap_aimbidx_concat1 )

            num_neighbor0 = num_neighbors[0]
            disw_theta0 = -1.5 # the abs smaller, more smooth
            assert num_neighbor0 <= flatten_bidxmap.shape[2].value
            batch_idx = tf.tile( batch_idx0,[1, point1_num, num_neighbor0 ,1] ) # (2, 256, 1)
            flatten_bidxmap_concat1 = tf.concat( [batch_idx, flatten_bidxmap[:,:,0:num_neighbor0,0:2]], axis=-1 )  # [...,[batch_idx,aimb_idx,point_idx_in_aimb] ]
            points1_nei = tf.gather_nd( points1, flatten_bidxmap_concat1 )
            if num_neighbor0 > 1:
                dis_weight = tf.nn.softmax( fbmap_neighbor_idis[:,:,0:num_neighbor0,:] * disw_theta0, axis=2 )
                points1_nei = tf.multiply( points1_nei, dis_weight )
            points1 = tf.reduce_sum( points1_nei, axis=2 )

        #flatten_bidxmap_aimbidx = flatten_bidxmap[:,:,0,0:1]  # (2, 256, 1)
        #flatten_bidxmap_aimbidx_concat = tf.concat( [batch_idx, flatten_bidxmap_aimbidx],axis=-1 ) # (2, 256, 2)
        #mapped_points2 = tf.gather_nd(points2, flatten_bidxmap_aimbidx_concat) # (2, 256, 512)


        # use the inverse distance weighted sum of 3 neighboured point features
        if cascade_id == 0:
            num_neighbor = num_neighbors[1]
        else:
            num_neighbor = num_neighbors[2]
        assert num_neighbor <= flatten_bidxmap.shape[2].value
        neighbor_method = 'A'
        #-----------------------------------
        if neighbor_method=='A':
            batch_idx = tf.tile( batch_idx0,[1, point1_num, 1 ,1] ) # (2, 256, 1)
            # from distance to weight
            if num_neighbor>1:
                disw_theta = -0.5 # the abs smaller, more smooth
                dis_weight = tf.nn.softmax( fbmap_neighbor_idis * disw_theta, axis=2 )
            for i in range(num_neighbor):
                flatten_bidxmap_aimbidx_concat_i = tf.concat( [batch_idx, flatten_bidxmap[:,:,i:(i+1),0:1]],axis=-1 ) # (2, 256, 2)
                mapped_points2_nei_i = tf.gather_nd(points2, flatten_bidxmap_aimbidx_concat_i) # (2, 256, 512)
                if num_neighbor>1:
                    mapped_points2_nei_i = tf.multiply( mapped_points2_nei_i, dis_weight[:,:,i:(i+1),:] )
                    if i==0:
                        mapped_points2 = mapped_points2_nei_i
                    else:
                        mapped_points2 += mapped_points2_nei_i
                else:
                    mapped_points2 = mapped_points2_nei_i
            mapped_points2 = tf.squeeze( mapped_points2,2 )
        #-----------------------------------
        if neighbor_method=='B':
            batch_idx = tf.tile( batch_idx0,[1, point1_num, num_neighbor ,1] ) # (2, 256, 1)
            flatten_bidxmap_aimbidx_concat = tf.concat( [batch_idx, flatten_bidxmap[:,:,0:num_neighbor,0:1]],axis=-1 ) # (2, 256, 2)
            mapped_points2_nei = tf.gather_nd(points2, flatten_bidxmap_aimbidx_concat) # (2, 256, 512)
            if num_neighbor>1:
                disw_theta = -0.7 # the abs smaller, more smooth
                dis_weight = tf.nn.softmax( fbmap_neighbor_idis * disw_theta, axis=2 )
                mapped_points2_nei = tf.multiply( mapped_points2_nei, dis_weight )
            mapped_points2 = tf.reduce_sum( mapped_points2_nei,2 )
        #-----------------------------------

        new_points1 = points1
        new_points1 = tf.expand_dims(new_points1,1)
        if 'growth_rate'in mlps_e1:
            new_points1 = tf_util.dense_net( new_points0, mlps_e1, bn, is_training, bn_decay, scope = 'dense_cascade_%d_fp'%(cascade_id) , is_show_model = IsShowModel )
        else:
            for i, num_out_channel in enumerate(mlps_e1):
                new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_encoder1_%d'%(i), bn_decay=bn_decay)
                if configs['Cnn_keep_prob']<1:
                    if ( not configs['only_last_layer_ineach_cascade'] ) or i == len(mlps_e1)-1:
                        new_points1 = tf_util.dropout(new_points1, keep_prob=configs['Cnn_keep_prob'], is_training=is_training, scope='dropout', name='cnn_dp%d'%(i))
                if IsShowModel: print('new_points1:%s'%(shape_str([new_points1])))

        mapped_points2 = tf.expand_dims(mapped_points2,1)
        new_points1 = tf.concat(values=[new_points1,mapped_points2],axis=-1)
        if IsShowModel: print('after concat new_points1:%s'%(shape_str([new_points1])))

        if 'growth_rate'in mlps_fp:
            new_points1 = tf_util.dense_net( new_points1, mlps_fp, bn, is_training, bn_decay, scope = 'dense_cascade_%d_fp'%(cascade_id) , is_show_model = IsShowModel )
        else:
            for i, num_out_channel in enumerate(mlps_fp):
                new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay)
                if configs['Cnn_keep_prob']<1:
                    if ( not configs['only_last_layer_ineach_cascade'] ) or i == len(mlps_fp)-1:
                        new_points1 = tf_util.dropout(new_points1, keep_prob=configs['Cnn_keep_prob'], is_training=is_training, scope='dropout', name='cnn_dp%d'%(i))
                if IsShowModel: print('new_points1:%s'%(shape_str([new_points1])))
        new_points1 = tf.squeeze(new_points1,[1]) # (2, 256, 256)
        if IsShowModel: print('new_points1:%s'%(shape_str([new_points1])));
        #if IsShowModel:  import pdb; pdb.set_trace()
    return new_points1


