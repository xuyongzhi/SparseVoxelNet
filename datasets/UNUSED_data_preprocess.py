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
import h5py, math
import glob
import time
import multiprocessing as mp
import itertools
import pickle
import json
from  datasets_meta import DatasetsMeta
import geometric_util as geo_util
import random


TMPDEBUG = True
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')

DATASETS = ['MATTERPORT', 'SCANNET', 'ETH', 'MODELNET40','KITTI']
for ds in DATASETS:
    sys.path.append('%s/%s_util'%(BASE_DIR,ds))

#DATASET = 'SCANNET'
#DATASET = 'ETH'
DATASET = 'MODELNET40'
#DATASET = 'KITTI'
DATASET = 'MATTERPORT'

DS_Meta = DatasetsMeta( DATASET )

H5TF_DATA_DIR = os.path.join(DATA_DIR, DATASET+'_H5TF' )

#CLASS_NAMES = DS_Meta.label_names

def WriteRawh5_to_Tfrecord(rawh5_file_ls, rawtfrecord_path, IsShowInfoFinished):
    from datasets.rawh5_to_tfrecord import RawH5_To_Tfrecord
    raw_to_tf = RawH5_To_Tfrecord(DATASET, rawtfrecord_path)
    raw_to_tf(rawh5_file_ls)

def split_fn_ls( tfrecordfn_ls, merged_n):
    nf = len(tfrecordfn_ls)
    assert merged_n < nf
    group_n = int(math.ceil( 1.0*nf/merged_n ))
    merged_fnls = []
    group_name_ls = []
    for i in range( 0, nf, group_n ):
        end = min( nf, i+group_n )
        merged_fnls.append( tfrecordfn_ls[i:end] )
        group_name_ls.append( '%d_%d'%(i, end) )
    return merged_fnls, group_name_ls

def split_fn_ls_benchmark_UNUSED( plsph5_folder, bxmh5_folder, nonvoid_plfn_ls, bxmh5_fn_ls, tfrecordfn_ls, void_f_n ):
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
      dirs[e] = '%s/ORG_%s/%s'%( H5TF_DATA_DIR, e, folder_names[e] )

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

    from random import shuffle
    for e in eles:
      assert len(train_fn_ls[e]) + len(test_fn_ls[e]) + void_f_n == train_num[DATASET] + test_num[DATASET]
      if void_f_n==0:
        assert len(train_fn_ls[e]) ==  train_num[DATASET]
        assert len(test_fn_ls[e]) == test_num[DATASET]

    shuffle(train_fn_ls['tfrecord'])
    shuffle(test_fn_ls['tfrecord'])

    for e in ['sph5', 'bxmh5']:
        train_fn_ls[e] = []
        test_fn_ls[e] = []
    for tfrecord_fn in train_fn_ls['tfrecord']:
        base_name = os.path.basename(tfrecord_fn)
        base_name = os.path.splitext(base_name)[0]
        for e in ['sph5', 'bxmh5']:
            train_fn_ls[e].append( os.path.join(dirs[e], base_name+'.'+e))
    for tfrecord_fn in test_fn_ls['tfrecord']:
        base_name = os.path.basename(tfrecord_fn)
        base_name = os.path.splitext(base_name)[0]
        for e in ['sph5', 'bxmh5']:
            test_fn_ls[e].append( os.path.join(dirs[e], base_name+'.'+e) )

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
            fn_0 = str(k)
            fn_1 = str(end)
            all_group_name_ls += ['test_'+fn_0+'_to_'+fn_1]

    # split train ls
    group_ns = {}
    group_ns['SCANNET'] = 301
    group_ns['MODELNET40'] = 985
    group_ns['KITTI'] = 5
    group_n = group_ns[DATASET]
    for e in eles:
      for k in range( 0, len(train_fn_ls[e]), group_n ):
        end  = min( k+group_n, len(train_fn_ls[e]) )
        all_lses[e] += [train_fn_ls[e][k:end]]
        if e=='tfrecord':
            fn_0 = str(k)
            fn_1 = str(end)
            all_group_name_ls += ['train_'+fn_0+'_to_'+fn_1]

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

    def __init__(self):
        self.rawh5f_dir =  H5TF_DATA_DIR+'/rawh5'

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

        #if TMPDEBUG:
        #    fn_ls = fn_ls[0:1]

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


    def RawToTfrecord(self, MultiProcess=0):

      t0 = time.time()
      if DATASET == 'MODELNET40' or 'MATTERPORT':
          rawh5_file_ls = glob.glob( os.path.join( self.rawh5f_dir,'*/*.rh5' ) )
      else:
          rawh5_file_ls = glob.glob( os.path.join( self.rawh5f_dir,'*.rh5' ) )
      rawh5_file_ls.sort()
      rawtfrecord_path = H5TF_DATA_DIR + '/raw_tfrecord'
      IsShowInfoFinished = True

      IsMultiProcess = MultiProcess>1
      if not IsMultiProcess:
          WriteRawh5_to_Tfrecord(rawh5_file_ls, rawtfrecord_path, IsShowInfoFinished)
      else:
          pool = mp.Pool(MultiProcess)
          for rawh5f_fn in rawh5_file_ls:
              results = pool.apply_async(WriteRawh5_to_Tfrecord,([rawh5f_fn],rawtfrecord_path, IsShowInfoFinished))
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

    def SortRaw(self, block_step_xyz, MultiProcess=0 , RxyzBeforeSort=None ):
        t0 = time.time()
        if DATASET == 'MODELNET40':
            rawh5_file_ls = glob.glob( os.path.join( self.rawh5f_dir,'*/*.rh5' ) )
        else:
            rawh5_file_ls = glob.glob( os.path.join( self.rawh5f_dir,'*.rh5' ) )
        rawh5_file_ls.sort()
        sorted_path = H5TF_DATA_DIR + '/'+get_stride_step_name(block_step_xyz,block_step_xyz)
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
        sh5f_dir = H5TF_DATA_DIR+'/%s'%(get_stride_step_name(base_stride,base_step))
        file_list = glob.glob( os.path.join( sh5f_dir, '*.sh5' ) )
        file_list.sort()
        assert len(file_list)>0, sh5f_dir
        N = len(file_list)
        if TMPDEBUG:
            choice = range(0,N,N//200)
            file_list = [ file_list[c] for c in choice ]
            #file_list = file_list[0:320]   # L
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


    def MergeTfrecord(self):
        from dataset_utils import merge_tfrecord

        tfrecord_path = '/home/z/Research/SparseVoxelNet/data/{}_H5TF/raw_tfrecord'.format(DATASET)
        rawdata_dir = tfrecord_path+'/data'
        merged_dir = tfrecord_path+'/merged_data'
        if not os.path.exists(merged_dir):
          os.makedirs(merged_dir)

        train_fn_ls = DS_Meta.get_train_test_file_list(rawdata_dir, True)
        random.shuffle(train_fn_ls)
        test_fn_ls = DS_Meta.get_train_test_file_list(rawdata_dir, False)
        random.shuffle(test_fn_ls)

        grouped_train_fnls, train_groupe_names = split_fn_ls(train_fn_ls, 10)
        train_groupe_names = ['train_'+e for e in train_groupe_names]
        grouped_test_fnls, test_groupe_names = split_fn_ls(test_fn_ls, 5)
        test_groupe_names = ['test_'+e for e in test_groupe_names]

        grouped_fnls = grouped_train_fnls + grouped_test_fnls
        group_names = train_groupe_names + test_groupe_names
        merged_fnls = [os.path.join(merged_dir, e+'.tfrecord') for e in group_names]

        for k in range(len(grouped_fnls)):
          merge_tfrecord(grouped_fnls[k], merged_fnls[k])


def GenObj_rh5():
    xyz_cut_rate= [0,0,0.9]
    xyz_cut_rate= None

    path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/BasicData/rawh5'
    fn_ls = glob.glob( path+'/scene0002*.rh5' )

    path = '/home/z/Research/dynamic_pointnet/data/ETH__H5F/BasicData/rawh5'
    fn_ls = glob.glob( path+'/marketplacefeldkirch_station7_intensity_rgb.rh5' )
    fn_ls = glob.glob( path+'/StGallenCathedral_station6_rgb_intensity-reduced.rh5' )
    fn_ls = glob.glob( path+'/untermaederbrunnen_station3_xyz_intensity_rgb.rh5' )

    path = '/home/z/Research/SparseVoxelNet/data/MATTERPORT_H5TF/rawh5/17DRP5sb8fy'
    fn_ls = glob.glob( path+'/region0.rh5' )

    for fn in fn_ls:
        if not Raw_H5f.check_rh5_intact( fn )[0]:
            print('rh5 not intact, abort gen obj')
            return
        with h5py.File( fn,'r' ) as h5f:
            rawh5f = Raw_H5f(h5f,fn)
            rawh5f.generate_objfile(IsLabelColor=True,xyz_cut_rate=xyz_cut_rate)


def main( ):
    t0 = time.time()
    MultiProcess = 0
    h5prep = H5Prepare()

    ##h5prep.ParseRaw( MultiProcess )
    #h5prep.RawToTfrecord(MultiProcess)
    h5prep.MergeTfrecord()
    print('T = %f sec'%(time.time()-t0))

if __name__ == '__main__':
    #main()
    GenObj_rh5()
    #GenObj_sph5()
    #GenObj_sh5()
