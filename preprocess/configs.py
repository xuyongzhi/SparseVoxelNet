# xyz
# Jan 2017

from __future__ import print_function
import numpy as np

#-------------------------------------------------------------------------------
NETCONFIG = {}
NETCONFIG['max_global_sample_rate'] = 3   # sample_res_num / org_num  This is very large for global block with few points which should be abandoned.
NETCONFIG['merge_blocks_while_fix_bmap'] = True
NETCONFIG['redundant_points_in_block'] = -777  # 'replicate' or a negative number to be asigned in bidxmap  (<-500)
#-------------------------------------------------------------------------------
# gsbb config
#_gsbb_config = '3E1'
#_gsbb_config = '3D1_benz'
_gsbb_config = '2M2pp'
#_gsbb_config = '2S2'
print('\n gsbb_config:%s \n-----------------------------------------------------'%(_gsbb_config))

def get_gsbb_config( gsbb_config = _gsbb_config ):
    '''
    global_step: When this <0, it's set to whole scene scope. But the limit is -global_step.
    global_stride: When global_step <0, should also <0
    '''

    flatbxmap_max_dis = 4
    padding = 0.3
    cascade_num = int(_gsbb_config[0])
    min_valid_point = np.ones( (cascade_num+1), dtype=np.float32 ) # include
    #---------------------------------------------------------------------------
    #                       KITTI
    if gsbb_config == 'benz_d3':
         ## generating the KITTI dataset benchmark
        global_stride = np.array([-100,-100,-100]).astype(np.float)
        global_step = np.array([-100,-100,-100]).astype(np.float)
        max_global_num_point = 32768
        global_num_point = 32768
        flatbxmap_max_nearest_num = [-10, -10, -10]  # do not generate flatbxmap
        NETCONFIG['max_global_sample_rate'] = 5
        NETCONFIG['merge_blocks_while_fix_bmap'] = True


        sub_block_stride_candis = np.array([0.2, 0.4, 0.8]).astype(np.float)
        sub_block_step_candis   = np.array([0.4, 0.8, 1.6]).astype(np.float)
        nsubblock_candis        =np.array([11800, 4800, 1800]).astype(np.int32)
        npoint_subblock_candis = np.array([16,  8, 8]).astype(np.int32)

    elif gsbb_config == '3D1_benz':
        global_stride = np.array([ 5, 5,-100]).astype(np.float)
        global_step = np.array([ 10, 10,-100]).astype(np.float)
        max_global_num_point = 4000
        global_num_point = 4000
        flatbxmap_max_nearest_num = [-10, -10, -10]  # do not generate flatbxmap
        NETCONFIG['max_global_sample_rate'] = 10
        NETCONFIG['merge_blocks_while_fix_bmap'] = True


        sub_block_stride_candis = np.array([0.2, 0.3, 0.4]).astype(np.float)
        sub_block_step_candis   = np.array([0.4, 1.2, 2.4]).astype(np.float)
        nsubblock_candis        =np.array([1600, 1800, 800, 1]).astype(np.int32)
        npoint_subblock_candis = np.array([16,  16, 24, 32]).astype(np.int32)
        min_valid_point = np.array([1,3,3,3])


    #---------------------------------------------------------------------------
    #                           SCANNET
    elif gsbb_config == '2S1':  # ***
        global_stride = np.array([2.4,2.4,2.4]).astype(np.float)
        global_step = np.array([4.6,4.6,4.6]).astype(np.float)
        global_num_point = 10000 * 10
        flatbxmap_max_nearest_num = [1,4]

        sub_block_stride_candis = np.array([0.1,0.4]).astype(np.float)
        sub_block_step_candis   = np.array([0.1,0.6]).astype(np.float)
        nsubblock_candis =       np.array([4800, 480, 1]).astype(np.int32)
        npoint_subblock_candis = np.array([48, 56,  480]).astype(np.int32)
        min_valid_point =       np.array( [ 1,   3,   2 ] )
        #                                      5  11
    elif gsbb_config == '2S2':  # ***A
        global_stride = np.array([2.4,2.4,2.4]).astype(np.float)
        global_step = np.array([4.6,4.6,4.6]).astype(np.float)
        global_num_point = 10000 * 2
        flatbxmap_max_nearest_num = [1,4]

        sub_block_stride_candis = np.array([0.1,0.4]).astype(np.float)
        sub_block_step_candis   = np.array([0.1,0.6]).astype(np.float)
        nsubblock_candis =       np.array([3200, 400, 1]).astype(np.int32)
        npoint_subblock_candis = np.array([36, 48,  480]).astype(np.int32)
        min_valid_point =       np.array( [ 1,   3,   2 ] )
    elif gsbb_config == '3S1':  # ***
        global_stride = np.array([-4.8,-4.8,-4.8]).astype(np.float)
        global_step = np.array([-7,-7,-7]).astype(np.float)
        global_num_point = 10000 * 10
        flatbxmap_max_nearest_num = [1,4,4]

        sub_block_stride_candis = np.array([0.1,0.4,1.6]).astype(np.float)
        sub_block_step_candis   = np.array([0.2,0.6,2.2]).astype(np.float)
        nsubblock_candis =       np.array([8000, 480, 32, 1]).astype(np.int32)
        npoint_subblock_candis = np.array([128, 56,  56,  48]).astype(np.int32)
        min_valid_point =       np.array( [ 1, 3,   3,   2 ] )
        #                                      5   5   4

    elif gsbb_config == '4A1':  # ***
        global_stride = np.array([-3.6,-3.6,-1.8]).astype(np.float)
        global_step = np.array([-6.3,-6.3,-4.5]).astype(np.float)
        global_num_point = 10000 * 9
        flatbxmap_max_nearest_num = [1,4,4,4]

        sub_block_stride_candis = np.array([0.1,0.2,0.6,1.8]).astype(np.float)
        sub_block_step_candis   = np.array([0.1,0.3,0.9,2.7]).astype(np.float)
        nsubblock_candis =       np.array([6400, 2400, 320, 32]).astype(np.int32)
        npoint_subblock_candis = np.array([32, 16,  32,  48]).astype(np.int32)

    elif gsbb_config == '4A2':  # ***
        global_stride = np.array([-3.6,-3.6,-1.8]).astype(np.float)
        global_step = np.array([-6.3,-6.3,-4.5]).astype(np.float)
        global_num_point = 10000 * 9
        flatbxmap_max_nearest_num = [1,4,4,4]

        sub_block_stride_candis = np.array([0.1,0.2,0.6,1.8]).astype(np.float)
        sub_block_step_candis   = np.array([0.1,0.3,0.9,2.7]).astype(np.float)
        nsubblock_candis =       np.array([6400, 2400, 320, 48]).astype(np.int32)
        npoint_subblock_candis = np.array([32, 27,  64,  64]).astype(np.int32)

    elif gsbb_config == '4B1':  # ***
        global_stride = np.array([-2.4,-2.4,-2.4]).astype(np.float)
        global_step = np.array([-3.4,-3.4,-3.4]).astype(np.float)
        global_num_point = 10000 * 3
        flatbxmap_max_nearest_num = [1,4,4,4]

        sub_block_stride_candis = np.array([0.1,0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis   = np.array([0.1,0.4,1.0,2.2]).astype(np.float)
        nsubblock_candis =       np.array([2048, 1024, 128, 24]).astype(np.int32)
        npoint_subblock_candis = np.array([48, 32,  48,  27]).astype(np.int32)
        #                                       4   4   3  2

    #---------------------------------------------------------------------------
    #                           MODELNET
    elif gsbb_config == '4M1':
        global_stride = np.array([2.4,2.4,2.4]).astype(np.float)
        global_step = np.array([2.3,2.3,2.3]).astype(np.float)
        global_num_point = 10000
        flatbxmap_max_nearest_num = [1,4,4,4]

        sub_block_stride_candis = np.array([0.05,0.1,0.3,0.6]).astype(np.float)
        sub_block_step_candis   = np.array([0.05,0.2,0.5,1.1]).astype(np.float)
        #                                         4    4  3    3
        nsubblock_candis =       np.array([2560, 1024, 80, 16, 1]).astype(np.int32)
        npoint_subblock_candis = np.array([24, 32,  48,  27, 27]).astype(np.int32)
        min_valid_point =       np.array( [ 1, 2,   2,   2,  2 ] )

    elif gsbb_config == '3M1':
        NETCONFIG['merge_blocks_while_fix_bmap'] = False
        NETCONFIG['max_global_sample_rate'] = 1
        global_stride = np.array([2.0,2.0,2.0]).astype(np.float)
        global_step = np.array([2.2,2.2,2.2]).astype(np.float)
        global_num_point = 4096
        flatbxmap_max_nearest_num = [1,4,4,4]

        sub_block_stride_candis = np.array([0.05,0.1,0.4]).astype(np.float)
        sub_block_step_candis   = np.array([0.1,0.2,0.6]).astype(np.float)
        #                                       3    5  5
        nsubblock_candis =       np.array([3200, 1024, 48, 1]).astype(np.int32)
        npoint_subblock_candis = np.array([18,  24, 56, 56]).astype(np.int32)
        min_valid_point =       np.array( [ 1, 2,   2,   2,  2 ] )

    elif gsbb_config == '2M2pp':
        NETCONFIG['max_global_sample_rate'] = 1
        global_stride = np.array([2,2,2]).astype(np.float)
        global_step = np.array([2,2,2]).astype(np.float)
        global_num_point = 4096
        flatbxmap_max_nearest_num = [1,4]

        sub_block_stride_candis = np.array([0.1,0.2]).astype(np.float)
        sub_block_step_candis   = np.array([0.2,0.4]).astype(np.float)
        nsubblock_candis =       np.array([1024, 240, 1]).astype(np.int32)
        npoint_subblock_candis = np.array([64, 27, 256]).astype(np.int32)
        min_valid_point =       np.array( [ 1,  1,  1 ] )
        #                                      3  9

    #---------------------------------------------------------------------------
    #                             ETH
    elif gsbb_config == '3E1':
        global_stride = np.array([3.6, 3.6, 3.6]).astype(np.float)
        global_step = np.array([6.6, 6.6, 6.6]).astype(np.float)
        global_num_point = 60000
        flatbxmap_max_nearest_num = [1,4,4,4]

        sub_block_stride_candis = np.array([0.2,0.4,1.2]).astype(np.float)
        sub_block_step_candis   = np.array([0.2,0.6,1.8]).astype(np.float)
        nsubblock_candis =       np.array([ 480, 640, 32]).astype(np.int32)
        npoint_subblock_candis = np.array([ 320,  48,  24]).astype(np.int32)

    elif gsbb_config == '4E1':
        NETCONFIG['max_global_sample_rate'] = 200
        global_stride = np.array([4.8,4.8,4.8]).astype(np.float)
        global_step = np.array([7.8,7.8,7.8]).astype(np.float)
        global_num_point = 80000
        flatbxmap_max_nearest_num = [1,4,4,4]

        sub_block_stride_candis = np.array([0.2,0.4,0.8,1.6]).astype(np.float)
        sub_block_step_candis   = np.array([0.2,0.6,1.4,3.0]).astype(np.float)
        nsubblock_candis =       np.array([ 2048, 640, 256, 128, 1]).astype(np.int32)
        npoint_subblock_candis = np.array([ 72,  16, 24,  24, 64]).astype(np.int32)
        min_valid_point = np.array( [ 1, 2, 2, 2, 2 ] )
    else:
        assert False,"gsbb config flag not recognized: %s"%(gsbb_config)

    gsbb_config_dic = {}
    gsbb_config_dic['max_global_num_point'] = global_num_point
    gsbb_config_dic['global_stride'] = global_stride
    gsbb_config_dic['global_step'] = global_step
    gsbb_config_dic['global_num_point'] = global_num_point
    gsbb_config_dic['sub_block_stride_candis'] = sub_block_stride_candis
    gsbb_config_dic['sub_block_step_candis'] = sub_block_step_candis
    gsbb_config_dic['nsubblock_candis'] = nsubblock_candis
    gsbb_config_dic['npoint_subblock_candis'] = npoint_subblock_candis
    gsbb_config_dic['gsbb_config'] = gsbb_config
    gsbb_config_dic['flatbxmap_max_nearest_num'] = flatbxmap_max_nearest_num
    gsbb_config_dic['flatbxmap_max_dis'] = flatbxmap_max_dis
    gsbb_config_dic['padding'] = padding
    gsbb_config_dic['min_valid_point'] = min_valid_point

    return  gsbb_config_dic

#-------------------------------------------------------------------------------

