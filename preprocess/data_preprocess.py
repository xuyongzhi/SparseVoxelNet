# xyz
from __future__ import print_function
import pdb, traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+'/all_datasets_meta')
from block_data_prep_util import Raw_H5f, Sort_RawH5f,Sorted_H5f,Normed_H5f,show_h5f_summary_info,MergeNormed_H5f,get_stride_step_name
from block_data_prep_util import GlobalSubBaseBLOCK,get_mean_sg_sample_rate,get_mean_flatten_sample_rate,check_h5fs_intact
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
import pickle
import json
from  datasets_meta import DatasetsMeta
import geometric_util as geo_util

TMPDEBUG = False
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')

DATASETS = ['MATTERPORT', 'SCANNET', 'ETH', 'MODELNET40','KITTI']
for ds in DATASETS:
    sys.path.append('%s/%s_util'%(BASE_DIR,ds))

#DATASET = 'SCANNET'
#DATASET = 'ETH'
DATASET = 'MODELNET40'
#DATASET = 'KITTI'
DS_Meta = DatasetsMeta( DATASET )

ORG_DATA_DIR = os.path.join(DATA_DIR, DATASET+'__H5F' )
MERGED_DATA_DIR = os.path.join(DATA_DIR, DATASET+'H5F' )

#CLASS_NAMES = DS_Meta.label_names

def WriteSortH5f_FromRawH5f(rawh5_file_ls,block_step_xyz,sorted_path, RotateBeforeSort, IsShowInfoFinished):
    Sort_RawH5f(rawh5_file_ls,block_step_xyz,sorted_path,RotateBeforeSort, IsShowInfoFinished)
    return rawh5_file_ls

def GenPyramidSortedFlie(file_id, fn, data_aug_configs):

    with h5py.File(fn,'r') as f:
        sorted_h5f = Sorted_H5f(f,fn)
        Always_CreateNew_plh5 = False
        Always_CreateNew_bmh5 = False
        Always_CreateNew_bxmh5 = False
        if TMPDEBUG:
            Always_CreateNew_bmh5 = False
            Always_CreateNew_plh5 = True
            Always_CreateNew_bxmh5 = False

        sorted_h5f.file_saveas_pyramid_feed(
          file_id,
          IsShowSummaryFinished=True,
          Always_CreateNew_plh5 = Always_CreateNew_plh5,
          Always_CreateNew_bmh5 = Always_CreateNew_bmh5,
          Always_CreateNew_bxmh5=Always_CreateNew_bxmh5,
          IsGenPly=False,
          data_aug_configs = data_aug_configs )
    return fn

def split_fn_ls( nonvoid_plfn_ls, bxmh5_fn_ls, tfrecordfn_ls, merged_n=2 ):
    nf = len(nonvoid_plfn_ls)
    merged_n = min( merged_n, nf )
    group_n = int( nf/merged_n )
    allfn_ls = [ [], [], [] ]
    all_group_name_ls = []
    for i in range( 0, nf, group_n ):
        end = min( nf, i+group_n )
        allfn_ls[0].append( nonvoid_plfn_ls[i:end] )
        allfn_ls[1].append( bxmh5_fn_ls[i:end] )
        allfn_ls[2].append( tfrecordfn_ls[i:end] )
        all_group_name_ls.append( '%d_%d'%(i, end) )
    return allfn_ls, all_group_name_ls

def split_fn_ls_benchmark( plsph5_folder, bxmh5_folder, nonvoid_plfn_ls, bxmh5_fn_ls, tfrecordfn_ls, void_f_n ):
    folder_names = {}
    folder_names['sph5'] = plsph5_folder
    folder_names['bxmh5'] = folder_names['tfrecord'] = bxmh5_folder

    real_fn_ls = {}
    real_fn_ls['sph5'] = nonvoid_plfn_ls
    real_fn_ls['bxmh5'] = bxmh5_fn_ls
    real_fn_ls['tfrecord'] = tfrecordfn_ls

    eles = ['sph5', 'bxmh5', 'tfrecord']
    dirs = {}
    for e in eles:
      dirs[e] = '%s/ORG_%s/%s'%( ORG_DATA_DIR, e, folder_names[e] )

    # get the desiged base file names of train and test from benchmark
    if DATASET == 'SCANNET_util':
        train_basefn_ls = list(np.loadtxt('./SCANNET_util/scannet_trainval.txt','string'))
        test_basefn_ls = list(np.loadtxt('./SCANNET_util/scannet_test.txt','string'))
    elif DATASET == 'MODELNET40':
        train_basefn_ls = list(np.loadtxt('./MODELNET_util/modelnet40_train.txt','string'))
        test_basefn_ls = list(np.loadtxt('./MODELNET_util/modelnet40_test.txt','string'))
    elif DATASET == 'KITTI':
        train_basefn_ls = list(np.loadtxt('./KITTI_util/train.txt','string'))
        test_basefn_ls = list(np.loadtxt('./KITTI_util/test.txt','string'))


    train_fn_ls = {}
    test_fn_ls = {}
    print('checking all files exsit')
    for e in eles:
      # get desiged file name list
      train_fn_ls[e] = [os.path.join(dirs[e], fn + '.' + e) for fn in train_basefn_ls]
      test_fn_ls[e] = [os.path.join(dirs[e], fn + '.' + e) for fn in test_basefn_ls]
      # check all file exist
      train_fn_ls[e] = [fn for fn in train_fn_ls[e] if fn in real_fn_ls[e]]
      test_fn_ls[e] = [fn for fn in test_fn_ls[e] if fn in real_fn_ls[e]]

      assert len(train_fn_ls[e]) == len(train_basefn_ls)
      assert len(test_fn_ls[e]) == len(test_basefn_ls)

    assert len(train_fn_ls['sph5']) ==  len(train_fn_ls['bxmh5'])
    assert len(train_fn_ls['sph5']) ==  len(train_fn_ls['tfrecord'])
    assert len(test_fn_ls['sph5']) ==  len(test_fn_ls['bxmh5'])
    assert len(test_fn_ls['sph5']) ==  len(test_fn_ls['tfrecord'])

    train_num = {}
    train_num['SCANNET'] = 1201
    train_num['MODELNET40'] = 9843
    test_num = {}
    test_num['SCANNET'] = 312
    test_num['MODELNET40'] = 2468
    train_num['KITTI'] = 7
    test_num['KITTI'] = 8

    is_shuffle = True
    from random import shuffle
    for e in eles:
      assert len(train_fn_ls[e]) + len(test_fn_ls[e]) + void_f_n == train_num[DATASET] + test_num[DATASET]
      if void_f_n==0:
        assert len(train_fn_ls[e]) ==  train_num[DATASET]
        assert len(test_fn_ls[e]) == test_num[DATASET]

      if is_shuffle:
        shuffle(train_fn_ls[e])
        shuffle(test_fn_ls[e])
      else:
        train_fn_ls[e].sort()
        test_fn_ls[e].sort()

    print('all files existing checked PK, start grouping')
    all_lses = {}
    for e in eles:
      all_lses[e]  =[]
    all_group_name_ls = []

    # split test ls
    group_ns = {}
    group_ns['SCANNET'] = 105
    group_ns['MODELNET40'] = 823
    group_ns['KITTI'] = 5
    group_n = group_ns[DATASET]

    for e in eles:
      for k in range( 0, len(test_fn_ls[e]), group_n ):
        end  = min( k+group_n, len(test_fn_ls[e]) )
        all_lses[e] += [test_fn_ls[e][k:end]]
        if e=='tfrecord':
          if is_shuffle:
            fn_0 = str(k)
            fn_1 = str(end)
            all_group_name_ls += ['test_'+fn_0+'_to_'+fn_1]
          else:
            fn_0 = os.path.splitext( os.path.basename(test_fn_ls[e][k]) )[0]
            fn_1 = os.path.splitext( os.path.basename(test_fn_ls[e][end-1]) )[0]
            all_group_name_ls += ['test_'+fn_0+'_to_'+fn_1+'-'+str(end-k)]

    # split train ls
    group_ns = {}
    group_ns['SCANNET'] = 301
    group_ns['MODELNET40'] = 1969
    group_ns['KITTI'] = 5
    group_n = group_ns[DATASET]
    for e in eles:
      for k in range( 0, len(train_fn_ls[e]), group_n ):
        end  = min( k+group_n, len(train_fn_ls[e]) )
        all_lses[e] += [train_fn_ls[e][k:end]]
        if e=='tfrecord':
          if is_shuffle:
            fn_0 = str(k)
            fn_1 = str(end)
            all_group_name_ls += ['train_'+fn_0+'_to_'+fn_1]
          else:
            fn_0 = os.path.splitext( os.path.basename(test_fn_ls[e][k]) )[0]
            fn_1 = os.path.splitext( os.path.basename(test_fn_ls[e][end-1]) )[0]
            all_group_name_ls += ['train_'+fn_0+'_to_'+fn_1+'-'+str(end-k)]

    return [all_lses['sph5'], all_lses['bxmh5'], all_lses['tfrecord']], all_group_name_ls


def WriteRawH5f( fn, rawh5f_dir ):
    if DATASET == 'SCANNET':
        return WriteRawH5f_SCANNET( fn, rawh5f_dir )
    elif DATASET == 'ETH':
        return WriteRawH5f_ETH( fn, rawh5f_dir )
    elif DATASET == 'MODELNET40':
        return WriteRawH5f_MODELNET40( fn, rawh5f_dir )

def WriteRawH5f_SCANNET( fn, rawh5f_dir ):
    # save as rh5
    import SCANNET_util
    fn_base = os.path.basename( fn )
    rawh5f_fn = os.path.join(rawh5f_dir, fn_base+'.rh5')
    if Raw_H5f.check_rh5_intact( rawh5f_fn )[0]:
        print('rh5 intact: %s'%(rawh5f_fn))
        return fn
    print('start write rh5: %s'%(rawh5f_fn))

    scene_points, instance_labels, semantic_labels, mesh_labels = SCANNET_util.parse_raw_SCANNET( fn )
    num_points = scene_points.shape[0]
    with h5py.File(rawh5f_fn,'w') as h5f:
        raw_h5f = Raw_H5f(h5f,rawh5f_fn,'SCANNET')
        raw_h5f.set_num_default_row(num_points)
        raw_h5f.append_to_dset('xyz', scene_points[:,0:3])
        raw_h5f.append_to_dset('color', scene_points[:,3:6])
        raw_h5f.append_to_dset('label_category', semantic_labels)
        raw_h5f.append_to_dset('label_instance', instance_labels)
        raw_h5f.append_to_dset('label_mesh', mesh_labels)
        raw_h5f.rh5_create_done()
    return fn

def WriteRawH5f_ETH( fn_txt, rawh5f_dir ):
    import ETH_util
    fn_base = os.path.basename( fn_txt )
    fn_base = os.path.splitext( fn_base )[0]
    if fn_base[-3:] == 'txt':
        fn_base = os.path.splitext( fn_base )[0]
    rawh5f_fn = os.path.join(rawh5f_dir, fn_base+'.rh5')
    if Raw_H5f.check_rh5_intact( rawh5f_fn )[0]:
        print('rh5 intact: %s'%(rawh5f_fn))
        return fn_txt
    print('start write rh5: %s'%(rawh5f_fn))

    fn_labels =  os.path.splitext( fn_txt )[0] + '.labels'

    #xyz, intensity, rgb, labels = ETH_util.parse_raw_ETH( fn_txt )
    num_points = 1e7

    with h5py.File( rawh5f_fn, 'w' ) as h5f:
        raw_h5f = Raw_H5f(h5f,rawh5f_fn,'ETH')
        raw_h5f.set_num_default_row(num_points)
        with open( fn_txt, 'r' ) as txtf:
            # {x, y, z, intensity, r, g, b}
            n_read = 0
            buf_size = int(1e7)
            while True:
                lines = txtf.readlines( buf_size )
                if len(lines) == 0:
                    break
                lines = [ np.fromstring( line.strip(), dtype=np.float32, sep=' ' ).reshape(1,-1) for line in lines ]
                buf = np.concatenate( lines,0 )
                raw_h5f.append_to_dset('xyz', buf[:,0:3])
                raw_h5f.append_to_dset('color', buf[:,3:6])
                raw_h5f.append_to_dset('intensity', buf[:,6:7])
                n_read += buf.shape[0]
                print( 'data read: %d line \t%s'%(n_read, fn_base) )

        if os.path.exists( fn_labels ):
            with open( fn_labels,'r' ) as labelsf:
                buf_size = int(1e7)
                n_read_l = 0
                while True:
                    lines = labelsf.readlines( buf_size )
                    if len(lines) == 0:
                        break
                    lines = [ np.fromstring( line.strip(), dtype=np.int32, sep=' ' ).reshape(1,-1) for line in lines ]
                    buf = np.concatenate( lines,0 )
                    raw_h5f.append_to_dset( 'label_category', buf )
                    n_read_l += buf.shape[0]
                    print( 'label read: %d line \t%s'%(n_read_l, fn_base) )
                assert n_read == n_read_l

        raw_h5f.rh5_create_done()
    if IsLablesExist:
        f_labels.close()
    print('finish : %s'%(rawh5f_fn))
    return rawh5f_fn

def WriteRawH5f_MODELNET40( txt_path, rawh5f_dir ):
    tmp = txt_path.split('/')
    rawh5f_fn = os.path.join( rawh5f_dir, tmp[-2], os.path.splitext(tmp[-1])[0] + '.rh5' )
    if not os.path.exists( os.path.dirname(rawh5f_fn) ):
        os.makedirs( os.path.dirname(rawh5f_fn) )

    if Raw_H5f.check_rh5_intact( rawh5f_fn )[0]:
        print('rh5 intact: %s'%(rawh5f_fn))
        return rawh5f_fn
    print('start write rh5: %s'%(rawh5f_fn))

    data = np.loadtxt( txt_path, delimiter=',' ).astype(np.float32)
    num_points = data.shape[0]
    print(num_points)
    with h5py.File(rawh5f_fn,'w') as h5f:
        raw_h5f = Raw_H5f(h5f,rawh5f_fn,'MODELNET40')
        raw_h5f.set_num_default_row(num_points)
        raw_h5f.append_to_dset('xyz', data[:,0:3])
        if data.shape[1]==6:
            raw_h5f.append_to_dset('nxnynz', data[:,3:6])
        raw_h5f.rh5_create_done()
    return txt_path


def get_modelnet_fnls( root ):
    modelnet10 = False
    datapaths = {}
    for split in ['test','train']:
        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet10_train.txt'))]
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_train.txt'))]
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(root, 'modelnet40_test.txt'))]
        assert(split=='train' or split=='test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        datapaths[split] = [os.path.join(root, shape_names[i], shape_ids[split][i])+'.txt' for i in range(len(shape_ids[split]))]
    datapath = datapaths['test'] + datapaths['train']
    return datapath

class H5Prepare():
    '''

    '''
    BasicDataDir = os.path.join( ORG_DATA_DIR,'BasicData' )

    def __init__(self):
        self.rawh5f_dir =  self.BasicDataDir+'/rawh5'

    def ParseRaw(self, MultiProcess):
        raw_path = '/DS/' + DATASET

        rawh5f_dir = self.rawh5f_dir
        if not os.path.exists(rawh5f_dir):
            os.makedirs(rawh5f_dir)

        if DATASET == 'SCANNET':
            glob_fn = raw_path+'/scene*'
            fn_ls =  glob.glob( glob_fn )
        elif DATASET == 'ETH':
            glob_fn = raw_path+'/*.txt'
            fn_ls =  glob.glob( glob_fn )
        elif DATASET == 'MODELNET40':
            root_path = raw_path+'/charles/modelnet40_normal_resampled'
            fn_ls = get_modelnet_fnls( root_path )

        fn_ls.sort()
        if len(fn_ls) == 0:
            print('no file matches %s'%( glob_fn ))

        if TMPDEBUG:
            fn_ls = fn_ls[0:1]

        if MultiProcess < 2:
            for fn in fn_ls:
                WriteRawH5f( fn, rawh5f_dir )
        else:
            pool = mp.Pool(MultiProcess)
            for fn in fn_ls:
                results = pool.apply_async( WriteRawH5f, ( fn, rawh5f_dir))
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(fn_ls)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"ParseRaw failed. only %d files successed"%(len(success_fns))
            print("\n\nParseRaw:all %d files successed\n******************************\n"%(len(success_fns)))



    def SortRaw(self, block_step_xyz, MultiProcess=0 , RxyzBeforeSort=None ):
        t0 = time.time()
        if DATASET == 'MODELNET40':
            rawh5_file_ls = glob.glob( os.path.join( self.rawh5f_dir,'*/*.rh5' ) )
        else:
            rawh5_file_ls = glob.glob( os.path.join( self.rawh5f_dir,'*.rh5' ) )
        rawh5_file_ls.sort()
        sorted_path = self.BasicDataDir + '/'+get_stride_step_name(block_step_xyz,block_step_xyz)
        if type(RxyzBeforeSort)!=type(None) and np.sum(RxyzBeforeSort==0)!=0:
            RotateBeforeSort = geo_util.EulerRotate( RxyzBeforeSort, 'xyz' )
            rdgr = RxyzBeforeSort * 180/np.pi
            RotateBeforeSort_str = '-R_%d_%d_%d'%( rdgr[0], rdgr[1], rdgr[2] )
            sorted_path += RotateBeforeSort_str
        else:
            RotateBeforeSort = None
        IsShowInfoFinished = True

        IsMultiProcess = MultiProcess>1
        if not IsMultiProcess:
            WriteSortH5f_FromRawH5f(rawh5_file_ls,block_step_xyz,sorted_path, RotateBeforeSort, IsShowInfoFinished)
        else:
            pool = mp.Pool(MultiProcess)
            for rawh5f_fn in rawh5_file_ls:
                results = pool.apply_async(WriteSortH5f_FromRawH5f,([rawh5f_fn],block_step_xyz,sorted_path, RotateBeforeSort, IsShowInfoFinished))
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(rawh5_file_ls)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"SortRaw failed. only %d files successed"%(len(success_fns))
            print("\n\nSortRaw:all %d files successed\n******************************\n"%(len(success_fns)))
        print('sort raw t= %f'%(time.time()-t0))

    def GenPyramid(self, base_stride, base_step, data_aug_configs, MultiProcess=0):
        sh5f_dir = self.BasicDataDir+'/%s'%(get_stride_step_name(base_stride,base_step))
        file_list = glob.glob( os.path.join( sh5f_dir, '*.sh5' ) )
        file_list.sort()
        N = len(file_list)
        if TMPDEBUG:
            choice = range(0,N,N//12)
            file_list = [ file_list[c] for c in choice ]
            #file_list = file_list[0:2]   # L
            #file_list = file_list[750:len(file_list)] # R
            #sh5f_dir = sh5f_dir+'_parts'
            #file_list = glob.glob( os.path.join( sh5f_dir, 'untermaederbrunnen_station3_xyz_intensity_rgb--0_0_n100_10_10_100.sh5' ) )
            #file_list = glob.glob( os.path.join( sh5f_dir, 'untermaederbrunnen_station3_xyz_*.sh5' ) )

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for k,fn in enumerate( file_list ):
            if not IsMultiProcess:
                GenPyramidSortedFlie(k, fn, data_aug_configs)
                print( 'Finish %d / %d files'%( k+1, len(file_list) ))
            else:
                results = pool.apply_async(GenPyramidSortedFlie,(k, fn,data_aug_configs))
        if IsMultiProcess:
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(file_list)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"Norm failed. only %d files successed"%(len(success_fns))
            print("\n\n GenPyramid: all %d files successed\n******************************\n"%(len(success_fns)))


    def MergeNormed(self, data_aug_configs):
        split_method = 'benchmark'
        #split_method = 'order'

        if DATASET == 'SCANNET':
            plsph5_folder = 'ORG_sph5/240000_mgs3_gs2d4_4d6'
            bxmh5_folder = 'ORG_bxmh5/240000_mgs3_gs2d4_4d6_fmn14_mvp2-4800_480_1-48_56_480-0d1_0d6-0d1_0d4-pd3-mbf-neg-2S1'

        if DATASET == 'MODELNET40':
            plsph5_folder = '4096_mgs1_gs2_2-mbf-neg'
            bxmh5_folder = '4096_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'

        if DATASET == 'KITTI':
            plsph5_folder = 'BasicData/ORG_sph5/4000_mgs10_gs5_10-mbf-neg'
            bxmh5_folder = 'BasicData/ORG_bxmh5/4000_mgs10_gs5_10-mbf-neg_fmn-10-10-10_mvp2-1600_1800_800_1-16_16_24_32-0d4_1d2_2d4-0d2_0d3_0d4-pd3-3D1_benz'

        folder_names = [ plsph5_folder, bxmh5_folder, bxmh5_folder]
        formats = ['sph5','bxmh5','tfrecord']
        pl_base_fn_ls = []
        pl_region_h5f_path = ORG_DATA_DIR + '/ORG_sph5/' + folder_names[0]
        plfn_ls = glob.glob( pl_region_h5f_path + '/*.' +  formats[0] )
        plfn_ls.sort()
        if len(plfn_ls) == 0:
            print('no file mathces %s'%(pl_region_h5f_path + '/*.' +  formats[0] ))
        print('%d files found for %s'%(len(plfn_ls), pl_region_h5f_path + '/*.' +  formats[0] ))

        nonvoid_plfn_ls = []
        bxmh5_fn_ls = []
        tfrecordfn_ls = []
        void_f_n = 0
        for pl_fn in plfn_ls:
            is_intact, ck_str = Normed_H5f.check_sph5_intact( pl_fn )
            region_name = os.path.splitext(os.path.basename( pl_fn ))[0]
            if not is_intact:
                print(' ! ! ! Abort merging %s not intact: %s'%(formats[0], pl_fn))
                if split_method=='benchmark':
                  assert False
                else:
                  continue
            if ck_str == 'void file':
                print('void file: %s'%(pl_fn))
                void_f_n += 1
                continue
            bxmh5_fn = ORG_DATA_DIR + '/ORG_bxmh5/' + folder_names[1] + '/' + region_name + '.' + formats[1]
            tfrecord_fn = ORG_DATA_DIR + '/ORG_tfrecord/' + folder_names[1] + '/' + region_name + '.tfrecord'
            if not (os.path.exists( bxmh5_fn ) and os.path.exists(tfrecord_fn)):
                print(' ! ! ! Abort merging %s not exist: %s'%(formats[0], bxmh5_fn))
                if split_method=='benchmark':
                  assert False
                else:
                  continue
            if not DATASET == 'KITTI':    ## benz_m, in my case, there is no bidxmaps_flat
                with h5py.File( pl_fn, 'r' ) as plh5f, h5py.File( bxmh5_fn, 'r' ) as bxmh5f:
                    if not plh5f['data'].shape[0] == bxmh5f['bidxmaps_flat'].shape[0]:
                        print('Abort merging %s \n  data shape (%d) != bidxmaps_flat shape (%d): %s'%( pl_region_h5f_path, plh5f['data'].shape[0], bxmh5f['bidxmaps_flat'].shape[0], pl_fn) )
                        assert False
                    else:
                        #print('shape match check ok: %s'%(region_name))
                        pass
            nonvoid_plfn_ls.append( pl_fn )
            bxmh5_fn_ls.append( bxmh5_fn )
            tfrecordfn_ls.append(tfrecord_fn)
        if len( nonvoid_plfn_ls )  == 0:
            print(  "no file, skip merging" )
            return

        if split_method == 'benchmark':
          allfn_ls, all_group_name_ls = split_fn_ls_benchmark( plsph5_folder, bxmh5_folder, nonvoid_plfn_ls, bxmh5_fn_ls, tfrecordfn_ls, void_f_n )
        else:
          allfn_ls, all_group_name_ls = split_fn_ls( nonvoid_plfn_ls, bxmh5_fn_ls, tfrecordfn_ls, merged_n=5 )

        for k in range( len(allfn_ls[0]) ):
            merged_file_names = ['','','']

            for j in range(3):
                merged_path = MERGED_DATA_DIR + '/Merged_' + formats[j] + '/' + folder_names[j][3:len(folder_names[j])] + '/'
                merged_file_names[j] = merged_path + all_group_name_ls[k] + '.' + formats[j]
                if not os.path.exists(merged_path):
                    os.makedirs(merged_path)
                if j<2:
                  status = MergeNormed_H5f(allfn_ls[j][k], merged_file_names[j], IsShowSummaryFinished=True)
                else:
                  if status != 'already_intact':
                    from dataset_utils import merge_tfrecord
                    merge_tfrecord(allfn_ls[j][k], merged_file_names[j])
                  else:
                    print('bxmh5 already intact, skip tfrecord as well')

            # check after merged
            if not DATASET == 'KITTI':    ## benz_m, in my case, there is no bidxmaps_flat
                with h5py.File( merged_file_names[0], 'r' ) as plh5f, h5py.File( merged_file_names[1], 'r' ) as bxmh5f:
                    if not plh5f['data'].shape[0] == bxmh5f['bidxmaps_flat'].shape[0]:
                        print('! ! ! shape check failed:  data shape (%d) != bidxmaps_flat shape (%d): \n\t%s \n\t%s'%( plh5f['data'].shape[0], bxmh5f['bidxmaps_flat'].shape[0], merged_file_names[0],merged_file_names[1]) )
                    else:
                        print( 'After merging, shape match check ok: %s'%(os.path.basename( merged_file_names[0] )) )
                        pass

def GenObj_rh5():
    xyz_cut_rate= [0,0,0.9]
    xyz_cut_rate= None

    path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/BasicData/rawh5'
    fn_ls = glob.glob( path+'/scene0002*.rh5' )

    path = '/home/z/Research/dynamic_pointnet/data/ETH__H5F/BasicData/rawh5'
    fn_ls = glob.glob( path+'/marketplacefeldkirch_station7_intensity_rgb.rh5' )
    fn_ls = glob.glob( path+'/StGallenCathedral_station6_rgb_intensity-reduced.rh5' )
    fn_ls = glob.glob( path+'/untermaederbrunnen_station3_xyz_intensity_rgb.rh5' )

    for fn in fn_ls:
        if not Raw_H5f.check_rh5_intact( fn )[0]:
            print('rh5 not intact, abort gen obj')
            return
        with h5py.File( fn,'r' ) as h5f:
            rawh5f = Raw_H5f(h5f,fn)
            rawh5f.generate_objfile(IsLabelColor=False,xyz_cut_rate=xyz_cut_rate)

def GenObj_sh5():
    path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/BasicData/stride_0d1_step_0d1'
    path = '/home/z/Research/dynamic_pointnet/data/MODELNET40__H5F/BasicData/stride_0d05_step_0d05'
    path = '/home/z/Research/dynamic_pointnet/data/ETH__H5F/BasicData/stride_0d2_step_0d2_parts'
    fn_ls = glob.glob( path+'/untermaederbrunnen_station3_xyz_intensity_rgb--0_0_n100_10_10_100.sh5' )
    for fn in fn_ls:
        with h5py.File( fn,'r' ) as h5f:
            sh5f = Sorted_H5f(h5f,fn)
            sh5f.gen_file_obj(IsLabelColor=False)


def GenObj_sph5():
    path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/ORG_sph5/30000_gs-2d4_-3d4'
    path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/ORG_sph5/30000_gs-2d4_-3d4-dec5'
    path = '/home/z/Research/dynamic_pointnet/data/MODELNET40__H5F/BasicData/stride_0d05_step_0d05'
    path = '/home/z/Research/dynamic_pointnet/data/ETH__H5F/ORG_sph5/80000_gs4d8_7d8'
    fn_ls = glob.glob( path+'/untermaederbrunnen_station3_xyz_intensity_rgb.sph5' )
    for fn in fn_ls:
        with h5py.File(fn,'r') as h5f:
            normedh5f = Normed_H5f(h5f,fn)
            normedh5f.gen_gt_pred_obj_examples()

def main( ):
    t0 = time.time()
    MultiProcess = 0
    h5prep = H5Prepare()

    #h5prep.ParseRaw( MultiProcess )
    if DATASET == 'SCANNET':
        base_step_stride = [0.1,0.1,0.1]
    elif DATASET == 'ETH':
        base_step_stride = [0.2,0.2,0.2]
    elif DATASET == 'MODELNET40':
        base_step_stride = [0.05,0.05,0.05]
    elif DATASET  == 'KITTI':
        base_step_stride = [0.2, 0.2, 0.4]
    #RxyzBeforeSort = np.array([0,0,45])*np.pi/180
    RxyzBeforeSort = None
    # h5prep.SortRaw( base_step_stride, MultiProcess, RxyzBeforeSort )

    data_aug_configs = {}
    # data_aug_configs['delete_unlabelled'] = True
    # data_aug_configs['delete_easy_categories_num'] = 3

    #h5prep.GenPyramid(base_step_stride, base_step_stride, data_aug_configs,  MultiProcess)
    h5prep.MergeNormed( data_aug_configs )
    print('T = %f sec'%(time.time()-t0))

if __name__ == '__main__':
    main()
    #GenObj_rh5()
    #GenObj_sph5()
    #GenObj_sh5()
