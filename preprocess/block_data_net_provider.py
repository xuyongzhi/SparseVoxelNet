# xyz Dec 2017

from __future__ import print_function
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))
sys.path.append(os.path.join(BASE_DIR,'all_datasets_meta'))
#from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
from block_data_prep_util import Normed_H5f,Sorted_H5f,GlobalSubBaseBLOCK
from ply_util import create_ply
import geometric_util as geo_util
from datasets_meta import DatasetsMeta
import aug_data
import tensorflow as tf

tf.enable_eager_execution()

DATA_DIR = os.path.join(ROOT_DIR,'data')
def DATASET_DIR(dataset_name):
    return os.path.join(DATA_DIR, dataset_name+'H5F')

DEBUGTMP = True
ERRTMP = True
#-------------------------------------------------------------------------------
# provider for training and testing
#------------------------------------------------------------------------------
class Net_Provider():
    '''
    (1) provide data for training
    (2) load file list to list of Norm_H5f[]
    dataset_name: 'stanford_indoor3d' 'scannet'
    all_filename_globs:  stride_1_step_2_test_small_4096_normed/*.sph5
    eval_fnglob_or_rate: file name str glob or file number rate. 'scan1*.sph5' 0.2
    num_point_block: if the block point number is not this, do randomly sample
    feed_data_elements: sub list of ['xyz_1norm','xyz_midnorm','nxnynz','color_1norm','intensity_1norm']
    feed_label_elements: sub list of ['label_category','label_instance','label_material']
    '''
    # input normalized h5f files
    # normed_h5f['data']: [blocks*block_num_point*num_channel],like [1000*4096*9]
    # one batch would contain sevel(batch_size) blocks,this will be set out side
    # provider with train_start_idx and test_start_idx


    def __init__(self, net_configs, dataset_name,all_filename_globs,eval_fnglob_or_rate, bxmh5_folder_name,\
                 only_evaluate,feed_data_elements, feed_label_elements, num_point_block = None, \
                 train_num_block_rate=1,eval_num_block_rate=1 ):
        t_init0 = time.time()
        self.net_configs = net_configs
        self.all_filename_globs = all_filename_globs
        self.bxmh5_folder_name = bxmh5_folder_name
        self.dataset_name = dataset_name
        self.feed_data_elements = feed_data_elements
        self.feed_label_elements = feed_label_elements
        self.num_point_block = num_point_block
        all_file_list = self.get_all_file_name_list(dataset_name,all_filename_globs)
        train_file_list,eval_file_list = self.split_train_eval_file_list\
                            (all_file_list,eval_fnglob_or_rate)
        if only_evaluate:
            open_type = 'a' # need to write pred labels
        else:
            open_type = 'r'
        train_file_list, train_bxmh5_fls = self.get_bxmh5_fn_ls( train_file_list )
        eval_file_list, eval_bxmh5_fls = self.get_bxmh5_fn_ls( eval_file_list )
        self.train_file_N = train_file_N = len(train_file_list)
        self.eval_file_N = eval_file_N = len(eval_file_list)
        self.g_file_N = train_file_N + eval_file_N
        self.sph5_file_list =  sph5_file_list = train_file_list + eval_file_list
        self.bxmh5_fn_ls = train_bxmh5_fls + eval_bxmh5_fls
        if len(sph5_file_list) > 10:
            print('WARING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\ntoo many (%d) files can lead to long read time'%(len(sph5_file_list)))
        #-----------------------------------------------------------------------
        # open each file as a Normed_H5f class instance
        self.norm_h5f_L = []
        # self.g_block_idxs: within the whole train/test dataset  (several files)
        #     record the start/end row idx  of each file to help search data from all files
        #     [ [start_global_row_idxs,end_global__idxs] ]
        #     [[  0,  38], [ 38,  90],[ 90, 150],...[259, 303],[303, 361],[361, 387]]
        #   self.train_num_blocks: 303
        #   self.eval_num_blocks: 84
        #   self.eval_global_start_idx: 303
        self.g_block_idxs = np.zeros((self.g_file_N,2),np.int32)
        self.eval_global_start_idx = None

        file_N = len(sph5_file_list)
        print('Totally %d files. \nReading the block number in each file.'%(file_N))
        t_start_global_blockid = time.time()
        for i,fn in enumerate(sph5_file_list):
            assert(os.path.exists(fn))

            h5f = h5py.File(fn,open_type)
            norm_h5f = Normed_H5f(h5f,fn)
            self.norm_h5f_L.append( norm_h5f )
            file_block_N = self.get_block_n(norm_h5f)
            self.g_block_idxs[i,1] = self.g_block_idxs[i,0] + file_block_N
            if i<self.g_file_N-1:
                self.g_block_idxs[i+1,0] = self.g_block_idxs[i,1]

        t_end_global_blockid = time.time()
        print('global block id reading finished, t=%f ms'%(1000.0*(t_end_global_blockid-t_start_global_blockid)))

        if train_file_N < file_N:
            self.eval_global_start_idx = self.g_block_idxs[train_file_N,0]
        else:
            self.eval_global_start_idx = self.g_block_idxs[-1,1]
        if train_file_N > 0:
            self.train_num_blocks = self.g_block_idxs[train_file_N-1,1] # = self.eval_global_start_idx
        else: self.train_num_blocks = 0
        self.eval_num_blocks = self.g_block_idxs[-1,1] - self.train_num_blocks
        self.num_classes = self.norm_h5f_L[0].num_classes
        self.label_ele_idxs_isph5f = self.norm_h5f_L[0].label_ele_idxs
        self.label_eles_isph5f = self.norm_h5f_L[0].label_set_elements

        self.update_sample_loss_weight()
        t_end_update_loss_weight = time.time()
        print('update_sample_loss_weight t: %f ms'%(1000*(t_end_update_loss_weight-t_end_global_blockid)))
        self.update_train_eval_shuffled_idx()

        #-----------------------------------------------------------------------
        # use only part of the data to test code:
        if train_num_block_rate!=1 or eval_num_block_rate!=1:
            self.get_data_label_shape()
            print('whole train data shape: %s'%(str(self.whole_train_data_shape)))
            print('whole eval data shape: %s'%(str(self.whole_eval_data_shape)))
            # train: use the front part
            self.train_num_blocks = int( self.train_num_blocks * train_num_block_rate )
            if not only_evaluate:
                self.train_num_blocks = max(self.train_num_blocks,2)
            new_eval_num_blocks = int( max(2,self.eval_num_blocks * eval_num_block_rate) )
            # eval:use the back part, so train_file_list and eval_file_list can be
            # the same
            self.eval_global_start_idx += self.eval_num_blocks - new_eval_num_blocks
            self.eval_num_blocks = new_eval_num_blocks

        self.gsbb_load = GlobalSubBaseBLOCK(  )
        self.gsbb_load.load_para_from_bxmh5( self.bxmh5_fn_ls[0] )
        self.global_num_point = self.gsbb_load.global_num_point
        self.gsbb_config = self.gsbb_load.gsbb_config
        self.get_data_label_shape_info()
        t_get_data_laebl_shape = time.time()
        print('get_data_label_shape t: %f ms'%(1000*(t_get_data_laebl_shape - t_end_update_loss_weight)))
        self.update_data_summary()

        print('Net_Provider init t: %f ms\n\n'%(1000*(time.time()-t_init0)))

    def get_bxmh5_fn_ls( self, plsph5_fn_ls ):
        bxmh5_fn_ls = []
        plsph5_fn_ls_new  = []
        for plsph5_fn in plsph5_fn_ls:
            bxmh5_fn = self.get_bxmh5_fn_1( plsph5_fn )
            if os.path.exists( bxmh5_fn ):
                # check shapes match with each other
                with h5py.File( bxmh5_fn, 'r' ) as bxmh5f:
                    with h5py.File( plsph5_fn, 'r' ) as plsph5f:
                        if bxmh5f['bidxmaps_flat'].shape[0] == plsph5f['data'].shape[0]:
                            bxmh5_fn_ls.append( bxmh5_fn )
                            plsph5_fn_ls_new.append( plsph5_fn )
                        else:
                            print('bxmh5(%d) and plsph5(%d) shapes do not match for %s'%( bxmh5f['bidxmaps_flat'].shape[0], plsph5f['data'].shape[0],plsph5_fn ))
                            assert False
            else:
                print( 'not exist: %s'%(bxmh5_fn) )
        return plsph5_fn_ls_new, bxmh5_fn_ls

    #def get_bxmh5_fn( self, plsph5_fn ):
    #    house_name = os.path.splitext( os.path.basename(plsph5_fn) )[0]
    #    pl_config_dir_name = os.path.dirname(plsph5_fn)
    #    pl_config_name = os.path.basename(pl_config_dir_name)
    #    each_house_dirname = os.path.dirname( pl_config_dir_name )
    #    bxmh5_dirname = os.path.join( each_house_dirname, self.bxmh5_folder_name )
    #    bxmh5_fn = os.path.join( bxmh5_dirname, house_name + '.bxmh5' )
    #    return bxmh5_fn

    def get_bxmh5_fn_1( self, plsph5_fn ):
        #fn1 = os.path.dirname(os.path.dirname( os.path.dirname(plsph5_fn) ))
        #tmp = plsph5_fn.split( os.sep )
        #tmp[-2] = self.bxmh5_folder_name
        ##tmp[-3] = self.bxmh5_folder_name
        #base_fn = os.path.splitext('/'+os.path.join( *tmp ))[0]
        #bxmh5_fn = base_fn+'.bxmh5'

        tmp = plsph5_fn.split('/')[-3:-1]
        sph5_folder = tmp[0]+'/'+tmp[1]+'/'
        for i,s in enumerate(self.all_filename_globs):
            if sph5_folder == s:
                fgi = i
        path = os.path.dirname(os.path.dirname( os.path.dirname(plsph5_fn) )) +'/'+ self.bxmh5_folder_name[fgi] + '/'
        base_name = os.path.splitext( os.path.basename( plsph5_fn ) )[0]
        bxmh5_fn = path + base_name + '.bxmh5'
        return bxmh5_fn

    def get_data_label_shape_info(self):
        self.feed_data_ele_idxs,self.feed_label_ele_idxs = self.norm_h5f_L[0].get_feed_ele_ids(self.feed_data_elements, self.feed_label_elements)
        self.data_num_eles = len([idx for e in self.feed_data_ele_idxs for idx in self.feed_data_ele_idxs[e] ])
        self.label_num_eles = len([self.feed_label_ele_idxs[e] for e in self.feed_label_ele_idxs])
        if 'nxnynz' in self.feed_label_elements:
            self.feed_label_ele_idxs['nxnynz'] = np.arange(self.label_num_eles,self.label_num_eles+3)
            self.label_num_eles += 3
        self.cascade_num = self.gsbb_load.cascade_num
        self.whole_train_data_shape = np.array([ self.train_num_blocks, self.gsbb_load.global_num_point, self.data_num_eles])
        self.whole_eval_data_shape = np.copy(self.whole_train_data_shape)
        self.whole_eval_data_shape[0] = self.eval_num_blocks
        self.whole_train_label_shape = np.copy(self.whole_train_data_shape)
        self.whole_train_label_shape[-1] = self.label_num_eles
        print('\ntrain data shape',self.whole_train_data_shape)
        print('train label shape',self.whole_train_label_shape)
        print('eval data shape',self.whole_eval_data_shape)

        #self.get_data_label_shape_byread()

        block_sample = self.whole_train_data_shape[1:-1]
        if block_sample.size==1:
            block_sample= (block_sample[0],)
        elif block_sample.size==2:
            block_sample = (block_sample[0],block_sample[1])
        self.block_sample = block_sample

        self.sg_bidxmaps_shape = self.gsbb_load.get_sg_bidxmaps_fixed_shape()
        self.flatten_bidxmaps_shape = self.gsbb_load.get_flatten_bidxmaps_shape()
        self.sg_bidxmaps_extract_idx = self.gsbb_load.sg_bidxmaps_extract_idx
        self.flatten_bidxmaps_extract_idx = self.gsbb_load.flatten_bidxmaps_extract_idx
        print('sg_bidxmaps_shape',self.sg_bidxmaps_shape)
        print('flatten_bidxmaps_shape',self.flatten_bidxmaps_shape)
        print('sg_bidxmaps_extract_idx\n',self.sg_bidxmaps_extract_idx)
        print('flatten_bidxmaps_extract_idx\n',self.flatten_bidxmaps_extract_idx)

    def get_data_label_shape_byread(self):
        t0 = time.time()
        data_batches,label_batches,sample_weights,sg_bidxmaps,flatten_bidxmaps, fmap_neighbor_idises = self.get_train_batch(0,min(self.train_num_blocks,32))
        #data_batches,label_batches,_ = self.get_train_batch(0,1)
        self.whole_train_data_shape = np.array(data_batches.shape)
        self.whole_train_data_shape[0] = self.train_num_blocks
        self.data_num_eles = self.whole_train_data_shape[-1]
        self.whole_eval_data_shape = np.array(data_batches.shape)
        self.whole_eval_data_shape[0] = self.eval_num_blocks
        self.whole_train_label_shape = np.array(label_batches.shape)
        self.whole_train_label_shape[0] = self.train_num_blocks
        self.label_num_eles = label_batches.shape[-1]
        print('\ntrain data shape',self.whole_train_data_shape)
        print('train label shape',self.whole_train_label_shape)
        print('eval data shape',self.whole_eval_data_shape)
        print('read %d global block t: %f ms\n'%( data_batches.shape[0], 1000*(time.time()-t0)))

    def check_bidxmap(self):
        datas,labels,sample_weights,sg_bidxmaps,flatten_bidxmaps, fmap_neighbor_idises = self.get_train_batch(0,min(self.train_num_blocks,32))


    def get_block_n(self,norm_h5f):
        file_block_N = norm_h5f.data_set.shape[0]
            #file_block_N = GlobalSubBaseBLOCK.get_block_n_of_new_stride_step_(norm_h5f.h5f,norm_h5f.file_name,'global')
        return file_block_N

    def update_data_summary(self):
        self.data_summary_str = '%s \nfeed_data_elements:%s \nfeed_label_elements:%s \n'%(self.dataset_name,self.feed_data_elements,self.feed_label_elements)
        self.data_summary_str += 'train data shape: %s \ntest data shape: %s \n'%(
                         str(self.whole_train_data_shape),str(self.whole_eval_data_shape))
       # self.data_summary_str += 'train labels histogram: %s \n'%( np.array_str(np.transpose(self.train_labels_hist_1norm) ))
       # self.data_summary_str += 'test labels histogram: %s \n'%( np.array_str(np.transpose(self.test_labels_hist_1norm) ))
        self.data_summary_str += 'label histogram:'+','.join('%5.2f'%(lh) for lh in self.labels_hist_1norm[:,0].tolist()) + '\n'
        self.data_summary_str += 'label   weights:'+','.join('%5.2f'%(lw) for lw in self.labels_weights[:,0].tolist()) + '\n'
        self.data_summary_str += 'class      name:'+','.join( '%5s'%(DatasetsMeta.g_label2class[self.dataset_name][label][0:5])  for label in range(len(DatasetsMeta.g_label2class[self.dataset_name])) )
        #print(self.data_summary_str)

    def get_all_file_name_list(self,dataset_name,all_filename_globs):
        all_file_list = []
        fn_globs = []
        file_format = '.sph5'

        for all_filename_globs in all_filename_globs:
            fn_glob = os.path.join(DATASET_DIR(dataset_name),all_filename_globs+'*'+file_format)
            fn_candis  = glob.glob( fn_glob )
            for fn in fn_candis:
                IsIntact,_ = Normed_H5f.check_sph5_intact( fn )
                if IsIntact:
                    all_file_list.append(fn)

            fn_globs.append(fn_glob)
        assert len(all_file_list)!=0,"no file in %s"%(fn_globs)
        #assert len(all_file_list)!=1,"only one file, should > 1 to seperate as train and test"
        return all_file_list

    def split_train_eval_file_list(self,all_file_list,eval_fnglob_or_rate=None):
        if eval_fnglob_or_rate == None:
            if self.dataset_name=='stanford_indoor3d':
                eval_fnglob_or_rate = 'Area_6'
            if self.dataset_name=='scannet':
                eval_fnglob_or_rate = 0.2
        if type(eval_fnglob_or_rate)==str:
            # split by name
            train_file_list = []
            eval_file_list = []
            for fn in all_file_list:
                if fn.find(eval_fnglob_or_rate) > 0:
                    eval_file_list.append(fn)
                else:
                    train_file_list.append(fn)
        elif type(eval_fnglob_or_rate) == float:
            if eval_fnglob_or_rate<0:
                inverse = True
                eval_fnglob_or_rate=-eval_fnglob_or_rate
            else:
                inverse = False
            all_file_list.sort()
            # split by number
            n = len(all_file_list)
            m = int(n*(1-eval_fnglob_or_rate))
            if not inverse:
                train_file_list = all_file_list[0:m]
                eval_file_list = all_file_list[m:n]
            else:
                eval_file_list = all_file_list[0:m]
                train_file_list = all_file_list[m:n]

        if len(eval_file_list) == 0 and eval_fnglob_or_rate<0:
            eval_file_list = [train_file_list[0]]

        log_str = '\ntrain file list (n=%d) = \n%s\n\n'%(len(train_file_list),train_file_list[-2:])
        log_str += 'eval file list (n=%d) = \n%s\n\n'%(len(eval_file_list),eval_file_list[-2:])
        print( log_str )
        return train_file_list, eval_file_list


    def test_tmp(self):
        s = 0
        e = 1
        train_data,train_label = self.get_train_batch(s,e)
        eval_data,eval_label = self.get_eval_batch(s,e)
        print('train:\n',train_data[0,0,:])
        print('eval:\n',eval_data[0,0,:])
        print('err=\n',train_data[0,0,:]-eval_data[0,0,:])


    def __exit__(self):
        print('exit Net_Provider')
        for norm_h5f in self.norm_h5f:
            norm_h5f.h5f.close()

    def global_idx_to_local(self,g_start_idx,g_end_idx):
        # not include g_end_idx (not 100% sure)
        # include end_file_idx (not 100% sure)
        assert(g_start_idx>=0 and g_start_idx<=self.g_block_idxs[-1,1])
        assert(g_end_idx>=0 and g_end_idx<=self.g_block_idxs[-1,1])
        for i in range(self.g_file_N):
            if g_start_idx >= self.g_block_idxs[i,0] and g_start_idx < self.g_block_idxs[i,1]:
                start_file_idx = i
                local_start_idx = g_start_idx - self.g_block_idxs[i,0]
                for j in range(i,self.g_file_N):
                    if g_end_idx > self.g_block_idxs[j,0] and g_end_idx <= self.g_block_idxs[j,1]:
                        end_file_idx = j
                        local_end_idx = g_end_idx - self.g_block_idxs[j,0]

        return start_file_idx,end_file_idx,local_start_idx,local_end_idx

    def set_pred_label_batch(self,pred_label,g_start_idx,g_end_idx):
        start_file_idx,end_file_idx,local_start_idx,local_end_idx = \
            self.global_idx_to_local(g_start_idx,g_end_idx)
        pred_start_idx = 0
        for f_idx in range(start_file_idx,end_file_idx+1):
            if f_idx == start_file_idx:
                start = local_start_idx
            else:
                start = 0
            if f_idx == end_file_idx:
                end = local_end_idx
            else:
                end = self.get_block_n( self.norm_h5f_L[f_idx] )
            n = end-start
            self.norm_h5f_L[f_idx].set_dset_value('pred_label',\
                pred_label[pred_start_idx:pred_start_idx+n,:],start,end)
            pred_start_idx += n
            self.norm_h5f_L[f_idx].h5f.flush()


    @staticmethod
    def get_indices_in_voxel( sg_bidxmaps, sg_bidxmaps_extract_idx, sub_block_step_candis, sub_block_stride_candis ):
        '''
        Convert from block center xyz_mm to point indices inside voxel.
        This should be performed in training model, just test it here in advance.
        '''
        IsForceSetMergedBlockIndice = True  # Forcely set merged block indice from -1 to 0
        def ForceSetMergedBlockIndices(points_indices_f, MAX_INDICE):
            # Because of merging blocks from nearby position, some point indices
            # are -1. Forcely set as zero.
            min_indice = np.min( points_indices_f )
            if not min_indice > -1e-4-1:
                print( "min_indice=%f"%(min_indice) )
                if not ERRTMP:
                    import pdb; pdb.set_trace()  # XXX BREAKPOINT
                pass
            max_indice = np.max( points_indices_f )
            if not max_indice < MAX_INDICE+1e-4+1:
                print( "max_indice=%s, MAX_INDICE=%s"%(max_indice, MAX_INDICE) )
                if not ERRTMP:
                    import pdb; pdb.set_trace()  # XXX BREAKPOINT
                pass

            points_indices_f0 = points_indices_f+0
            if not ERRTMP:
                mask_min = points_indices_f == -1.0
                points_indices_f[mask_min] += 1
                mask_max = points_indices_f == MAX_INDICE+1
                points_indices_f[mask_max] -= 1
            else:
                mask_min = points_indices_f < 0
                points_indices_f[mask_min] = 0
                mask_max = points_indices_f > MAX_INDICE
                points_indices_f[mask_max] = MAX_INDICE

            return points_indices_f


        aimb_bottom_mm_ls = []
        strides_cascades_mm = sub_block_stride_candis * 1000
        steps_cascades_mm =  sub_block_step_candis * 1000
        points_indices_in_voxel_all = []
        cascade_num = strides_cascades_mm.shape[0]
        for cascade_id in range( cascade_num ):
            sg_bidxmap_acxm =  GlobalSubBaseBLOCK.extract_sg_bidxmaps_( sg_bidxmaps, sg_bidxmaps_extract_idx, cascade_id, flag='both' )
            sg_bidxmap = sg_bidxmap_acxm[...,0:sg_bidxmap_acxm.shape[-1]-6]
            aimb_center_mm = sg_bidxmap_acxm[...,-3:]
            aimb_bottom_mm = sg_bidxmap_acxm[...,-6:-3]

            # check
            #step_mm_check = aimb_center_mm - aimb_bottom_mm - steps_cascades_mm[cascade_id]*0.5
            #if np.sum(np.abs(step_mm_check)) > 1e-10 :
            #    import pdb; pdb.set_trace()  # XXX BREAKPOINT
            #    pass

            # Note: transform both point and block position from center to min firstly
            aimb_bottom_mm_ls.append( aimb_bottom_mm )

            aimb_bottom_min_mm = np.min( aimb_bottom_mm[0], axis=0 )
            aimb_bottom_max_mm = np.max( aimb_bottom_mm[0], axis=0 )

            print( '\ncascade_id', cascade_id )
            print( 'aimb_bottom_min_mm',aimb_bottom_min_mm )
            print( 'aimb_bottom_max_mm',aimb_bottom_max_mm )
            if cascade_id>0:
                # get the grouped xyz
                points_indices_in_voxel_ls = []
                for batch in range(sg_bidxmap.shape[0]):
                    grouped_points_xyz_mm = np.take( aimb_bottom_mm_ls[cascade_id-1][batch], sg_bidxmap[batch], axis=0 )
                    points_indices_ls = []
                    MAX_INDICE_f =  ( steps_cascades_mm[cascade_id] - steps_cascades_mm[cascade_id-1] ) / strides_cascades_mm[cascade_id-1]
                    MAX_INDICE = np.ceil( MAX_INDICE_f )
                    if not MAX_INDICE - MAX_INDICE_f == 0:
                        import pdb; pdb.set_trace()  # XXX BREAKPOINT
                        pass

                    for aimb in range(grouped_points_xyz_mm.shape[0]):
                        # points_invoxel_xyzs_mm is positions of all the points inside a voxels
                        # point_stride_mm is the stride between these points
                        points_invoxel_xyzs_mm = grouped_points_xyz_mm[aimb,:]
                        voxelbmin_xyz_mm = aimb_bottom_mm[batch,aimb]
                        points_indices_f = (points_invoxel_xyzs_mm - voxelbmin_xyz_mm) * 1.0 / strides_cascades_mm[cascade_id-1] # [1050, 1650, 1850] - [ 850., 1450., 1650.] / 100
                        if IsForceSetMergedBlockIndice:
                            points_indices_f = ForceSetMergedBlockIndices( points_indices_f, MAX_INDICE )
                        points_indices = np.rint( points_indices_f ).astype( np.int32 )
                        points_indices_ls.append(np.expand_dims(points_indices,axis=0))

                        #if batch==0 and aimb==0 and cascade_id>=1:
                        #    print( "voxelbmin_xyz_mm: ", voxelbmin_xyz_mm )
                        #    print( "points_invoxel_xyzs_mm:", points_invoxel_xyzs_mm[0] )
                        #    print( "points_indices_f:", points_indices_f[0] )


                        # Check indices err
                        points_indices_err = np.max(np.abs(points_indices - points_indices_f), axis=0)
                        points_indices_errmax = np.max( points_indices_err )
                        if points_indices_errmax > 1e-5:
                            print ( "cascade %d points_indices_errmax: %s"%(cascade_id, points_indices_err) )
                            import pdb; pdb.set_trace()  # XXX BREAKPOINT
                            pass

                        # Check max indice
                        assert points_indices.min() >= 0
                        if not (points_indices.max(axis=0) <= MAX_INDICE).all():
                            print( "Failed: %s < %s"%( points_indices.max(axis=0), MAX_INDICE ) )
                            import pdb; pdb.set_trace()  # XXX BREAKPOINT
                            pass

                    points_indices_batch = np.concatenate( points_indices_ls, 0 )
                    points_indices_in_voxel_ls.append( np.expand_dims(points_indices_batch,0) )
                points_indices_in_voxe = np.concatenate( points_indices_in_voxel_ls, 0 )
                points_indices_in_voxel_all.append( points_indices_in_voxe )


    def get_global_batch(self, g_start_idx, g_end_idx, aug_types):
        # not include g_end_idx (not 100% sure)
        start_file_idx,end_file_idx,local_start_idx,local_end_idx = \
            self.global_idx_to_local(g_start_idx,g_end_idx)
        #t0 = time.time()

        if 'nxnynz' in self.feed_label_elements and 'nxnynz' not in self.feed_data_elements:
            self.feed_data_elements += ['nxnynz']

        data_ls = []
        label_ls = []
        sg_bidxmaps_ls = []
        flatten_bidxmaps_ls = []
        fmap_neighbor_idis_ls = []
        fid_start_end = []
        xyz_mid_ls = []
        for f_idx in range(start_file_idx,end_file_idx+1):
            if f_idx == start_file_idx:
                start = local_start_idx
            else:
                start = 0
            if f_idx == end_file_idx:
                end = local_end_idx
            else:
                end = self.get_block_n(self.norm_h5f_L[f_idx])
            fid_start_end.append( np.expand_dims(np.array([f_idx, start, end]),0) )

            feed_data_ele_idxs,_ = self.norm_h5f_L[0].get_feed_ele_ids(self.feed_data_elements, self.feed_label_elements)
            data_i = self.norm_h5f_L[f_idx].get_data_byeles(start,end, self.feed_data_elements)
            label_i = self.norm_h5f_L[f_idx].get_label_byeles(start,end, self.feed_label_elements)
            sg_bidxmaps, flatten_bidxmaps, fmap_neighbor_idises = Normed_H5f.get_bidxmaps( self.bxmh5_fn_ls[f_idx],start,end )

            assert data_i.ndim == label_i.ndim
            if self.dataset_name == 'MODELNET40':
                assert label_i.shape[-2] == 1
            else:
                assert (data_i.shape[0:-1] == label_i.shape[0:-1])

            data_ls.append(data_i)
            label_ls.append(label_i)
            sg_bidxmaps_ls.append(sg_bidxmaps)
            flatten_bidxmaps_ls.append(flatten_bidxmaps)
            fmap_neighbor_idis_ls.append(fmap_neighbor_idises )

        data_batches = np.concatenate(data_ls,0)
        label_batches = np.concatenate(label_ls,0)
        sg_bidxmaps = np.concatenate( sg_bidxmaps_ls,axis=0 )
        flatten_bidxmaps = np.concatenate( flatten_bidxmaps_ls,axis=0 )
        fmap_neighbor_idises = np.concatenate( fmap_neighbor_idis_ls,0 )

        if 'nxnynz' in self.feed_label_elements:
            nxnynz_batches = data_batches[:,:,feed_data_ele_idxs['nxnynz']]
            data_batches = np.delete( data_batches, feed_data_ele_idxs['nxnynz'], axis=-1 )

            label_batches = np.tile( label_batches, [1,nxnynz_batches.shape[1],1] )
            label_batches = np.concatenate( [label_batches, nxnynz_batches], axis=-1 )

        num_label_eles = len(self.feed_label_elements)

        # for each label, there is a weight. For all weight, when the point is
        # at edge, the weight is set to 0
        if self.net_configs['loss_weight'] != 'E': # Equal
            sample_weights = []
            for k in range(num_label_eles):
                if k == self.feed_label_ele_idxs['label_category'][0]:
                    sample_weights_k = np.take(self.labels_weights[:,k],label_batches[...,k])
                else:
                    sample_weights_k = np.ones_like(label_batches[...,k])
                sample_weights.append( np.expand_dims(sample_weights_k,axis=-1) )
            sample_weights = np.concatenate(sample_weights,axis=-1).astype( np.float32 )

        if self.net_configs['loss_weight'] == 'E': # Equal
            sample_weights = np.ones_like(label_batches)
        elif self.net_configs['loss_weight'] == 'N': # Number
            sample_weights = sample_weights
        elif self.net_configs['loss_weight'] == 'C': # Center
            assert NotImplementedError
        elif self.net_configs['loss_weight'] == 'CN':
            assert NotImplementedError
        else:
            assert False

        fid_start_end = np.concatenate( fid_start_end,0 )

        is_include_normal = 'nxnynz' in self.feed_data_elements and 'nxnynz' not in self.feed_label_elements
        data_batches = aug_data.aug_batch( data_batches, is_include_normal, aug_types, self.dataset_name )

        #Net_Provider.get_indices_in_voxel( sg_bidxmaps, self.sg_bidxmaps_extract_idx, self.gsbb_load.sub_block_step_candis, self.gsbb_load.sub_block_stride_candis )

     #   print('\nin global')
     #   print('file_start = ',start_file_idx)
     #   print('file_end = ',end_file_idx)
     #   print('local_start = ',local_start_idx)
     #   print('local end = ',local_end_idx)
     #   #print('data = \n',data_batches[0,:])
        #t_block = (time.time()-t0)/(g_end_idx-g_start_idx)
        #print('get_global_batch t_block:%f ms'%(1000.0*t_block))
       # print(sg_bidxmaps.shape)
       # print(flatten_bidxmaps.shape)

        return data_batches, label_batches, sample_weights, sg_bidxmaps, flatten_bidxmaps, fmap_neighbor_idises, fid_start_end


    def get_fn_from_fid(self,fid):
        return self.sph5_file_list[ fid ]

    def get_center_mask(self,f_idx, xyz_midnorm, edge_rate=0.1):
        # true for center, false for edge
        # edge_rate: distance to center of the block_step
        block_step = self.norm_h5f_L[f_idx].h5f.attrs['block_step']
        center_rate = np.abs(xyz_midnorm / block_step) # -0.5 ~ 0.5
        # NOTE: Always true along z direction
        center_mask = (center_rate[...,0] < (0.5-edge_rate)) * ( center_rate[...,1] < (0.5-edge_rate) )
        #print('center n rate= %f'%(np.sum(center_mask).astype(float)/xyz_midnorm.shape[0]/xyz_midnorm.shape[1]))
        return center_mask

    def get_shuffled_global_batch(self,g_shuffled_idx_ls,aug_types ):
        data_batches = []
        label_batches = []
        sample_weights = []
        sg_bidxmaps_ls = []
        flatten_bidxmaps_ls = []
        fmap_neighbor_idis_ls = []
        fid_start_end_ls = []
        for idx in g_shuffled_idx_ls:
            data_i,label_i,smw_i,sg_bidxmaps_i,flatten_bidxmaps_i, fmap_neighbor_idis_i,fid_start_end_i = self.get_global_batch(idx,idx+1,aug_types=aug_types)
            sg_bidxmaps_ls.append(sg_bidxmaps_i)
            flatten_bidxmaps_ls.append(flatten_bidxmaps_i)
            fmap_neighbor_idis_ls.append( fmap_neighbor_idis_i )
            data_batches.append(data_i)
            label_batches.append(label_i)
            sample_weights.append(smw_i)
            fid_start_end_ls.append(fid_start_end_i)
        if len(data_batches)>0:
            data_batches = np.concatenate(data_batches,axis=0)
            label_batches = np.concatenate(label_batches,axis=0)
            sample_weights = np.concatenate(sample_weights,axis=0)
            sg_bidxmaps = np.concatenate(sg_bidxmaps_ls,0)
            flatten_bidxmaps = np.concatenate(flatten_bidxmaps_ls,0)
            fmap_neighbor_idises = np.concatenate( fmap_neighbor_idis_ls,0 )
            fid_start_end = np.concatenate(fid_start_end_ls,0)
        else:
            data_batches=label_batches=sample_weights=sg_bidxmaps=flatten_bidxmaps=fmap_neighbor_idises=fid_start_end=np.array([])

        return data_batches,label_batches,sample_weights,sg_bidxmaps,flatten_bidxmaps, fmap_neighbor_idises,fid_start_end

    def update_train_eval_shuffled_idx(self):
        self.train_shuffled_idx = np.arange(self.train_num_blocks)
        np.random.shuffle(self.train_shuffled_idx)
        self.eval_shuffled_idx = np.arange(self.eval_num_blocks)
        np.random.shuffle(self.eval_shuffled_idx)

    def update_eval_shuffled_idx(self, eval_num_blocks_limit=None):
        if eval_num_blocks_limit!=None:
            tmp = min(self.eval_num_blocks,eval_num_blocks_limit)
        else:
            tmp = self.eval_num_blocks
        self.eval_shuffled_idx = np.arange(tmp)
        np.random.shuffle(self.eval_shuffled_idx)

    def update_train_shuffled_idx(self, train_num_blocks_limit=None):
        if train_num_blocks_limit!=None:
            tmp = min(self.train_num_blocks, train_num_blocks_limit)
        else:
            tmp = self.train_num_blocks
        self.train_shuffled_idx = np.arange( tmp )
        np.random.shuffle(self.train_shuffled_idx)

    def get_train_batch(self,train_start_block_idx,train_end_block_idx,IsShuffleIdx, aug_types, fids=None ):
        assert(train_start_block_idx>=0 and train_start_block_idx<=self.train_num_blocks)
        assert(train_end_block_idx>=0 and train_end_block_idx<=self.train_num_blocks),"train_end_block_idx = %d,  train_num_blocks=%d"%(train_end_block_idx,self.train_num_blocks)
        # all train files are before eval files
        if IsShuffleIdx:
            g_shuffled_block_idx = self.train_shuffled_idx[range(train_start_block_idx,train_end_block_idx)]

            # Since we cannot read same file in multi threads, use fids as file
            # filter. Note: the returned data is partial, the first shape is not
            # batchsize.

            if fids!=None:
                g_shuffled_block_idx = self.clean_g_shuffled_block_idx( g_shuffled_block_idx, fids )

            return self.get_shuffled_global_batch(g_shuffled_block_idx,aug_types)
        else:
            assert fids==None,"current multi processing is deisgned for shuffled mode"
            return self.get_global_batch(train_start_block_idx,train_end_block_idx,aug_types=aug_types)

    def clean_g_shuffled_block_idx(self, g_shuffled_block_idx, fids):
        g_shuffled_block_idx_cleaned = []
        for batch_idx in g_shuffled_block_idx:
            start_file_idx,end_file_idx,local_start_idx,local_end_idx = \
                self.global_idx_to_local(batch_idx, batch_idx+1)
            if start_file_idx in fids:
                g_shuffled_block_idx_cleaned.append( batch_idx )
        return g_shuffled_block_idx_cleaned

    def get_eval_batch(self,eval_start_block_idx,eval_end_block_idx,IsShuffleIdx, aug_types, fids=None):
        assert(eval_start_block_idx>=0 and eval_start_block_idx<=self.eval_num_blocks),"eval_start_block_idx = %d,  eval_num_blocks=%d"%(eval_start_block_idx,self.eval_num_blocks)
        assert(eval_end_block_idx>=0 and eval_end_block_idx<=self.eval_num_blocks),"eval_end_block_idx = %d,  eval_num_blocks=%d"%(eval_end_block_idx,self.eval_num_blocks)

        if IsShuffleIdx:
            g_shuffled_block_idx = self.eval_shuffled_idx[range(eval_start_block_idx,eval_end_block_idx)] + self.eval_global_start_idx
            if fids!=None:
                g_shuffled_block_idx = self.clean_g_shuffled_block_idx( g_shuffled_block_idx, fids )
            return self.get_shuffled_global_batch(g_shuffled_block_idx,aug_types)
        else:
            assert fids==None,"current multi processing is deisgned for shuffled mode"
            eval_start_block_idx  += self.eval_global_start_idx
            eval_end_block_idx  += self.eval_global_start_idx
            return self.get_global_batch(eval_start_block_idx,eval_end_block_idx,aug_types=aug_types)

    def batch_dataset(self, split, eval_start_block_idx,eval_end_block_idx,IsShuffleIdx, aug_types, fids=None):
        if split == 'train':
          get_batch_fn = self.get_eval_batch
        elif split == 'eval':
          get_batch_fn = self.get_train_batch
        data_batches, label_batches, sample_weights, sg_bidxmaps, flatten_bidxmaps, fmap_neighbor_idises, fid_start_end = \
                get_batch_fn(eval_start_block_idx,eval_end_block_idx,IsShuffleIdx, aug_types, fids)

        dataset = tf.data.Dataset.from_tensor_slices((  {'data': data_batches},
                                                        {'label': label_batches},
                                                        {'smp_weights': sample_weights},
                                                        {'sg_bidxmaps': sg_bidxmaps},
                                                        {'flatten_bidxmaps': flatten_bidxmaps},
                                                        {'fmap_neighbor_idises': fmap_neighbor_idises} ))
        i = 0
        for batch in dataset:
          print(i)
          i+=1
          print(batch)
          import pdb; pdb.set_trace()  # XXX BREAKPOINT
          pass
        return dataset

    def gen_gt_pred_objs(self,visu_fn_glob='The glob for file to be visualized',obj_dump_dir=None):
        for k,norm_h5f in enumerate(self.norm_h5f_L):
            if norm_h5f.file_name.find(visu_fn_glob) > 0:
                norm_h5f.gen_gt_pred_obj( obj_dump_dir )

    def sample(self,data_batches,label_batches,num_point_block):
        NUM_POINT_IN = data_batches.shape[1]
        if num_point_block == None:
            num_point_block = NUM_POINT_IN
        if NUM_POINT_IN != num_point_block:
            sample_choice = GLOBAL_PARA.sample(NUM_POINT_IN,num_point_block,'random')
            data_batches = data_batches[:,sample_choice,...]
            label_batches = label_batches[:,sample_choice]
        return data_batches,label_batches

    def update_sample_loss_weight(self):
        # amount is larger, the loss weight is smaller
        # get all the labels
        train_labels_hist_1norm = []
        test_labels_hist_1norm = []
        labels_hist_1norm = []
        labels_weights = []

        label_name = 'label_category'
        #for label_name in self.norm_h5f_L[0].label_set_elements:
        if label_name in self.feed_label_elements:
            label_hist = np.zeros(self.num_classes).astype(np.int64)
            train_label_hist = np.zeros(self.num_classes).astype(np.int64)
            test_label_hist = np.zeros(self.num_classes).astype(np.int64)
            for k,norm_h5f in enumerate(self.norm_h5f_L):
                if label_name + '_hist' in norm_h5f.labels_set.attrs:
                    # segmentation datset
                    label_hist_k = norm_h5f.labels_set.attrs[label_name+'_hist']
                else:
                    # classification dataset
                    label_hist_k,_ = np.histogram(norm_h5f.labels_set[:,:,0], range(self.num_classes+1))
                ## delete unlabeled label hist
                #if label_hist_k.size != self.num_classes:
                #    label_hist_k = np.delete( label_hist_k, Normed_H5f.g_unlabelled_categories[self.dataset_name] )
                label_hist += label_hist_k.astype(np.int64)
                if k < self.train_file_N:
                    train_label_hist += label_hist_k.astype(np.int64)
                else:
                    test_label_hist += label_hist_k.astype(np.int64)
            train_labels_hist_1norm.append( np.expand_dims( train_label_hist / np.sum(train_label_hist).astype(float),axis=-1) )
            test_labels_hist_1norm.append(   np.expand_dims(test_label_hist / np.sum(test_label_hist).astype(float),axis=-1) )
            cur_labels_hist_1norm =  np.expand_dims(label_hist / np.sum(label_hist).astype(float),axis=-1)
            labels_hist_1norm.append(  cur_labels_hist_1norm )
            labels_weights.append(  1/np.log(1.015+cur_labels_hist_1norm) )
        self.train_labels_hist_1norm = np.concatenate( train_labels_hist_1norm,axis=-1 )
        self.test_labels_hist_1norm = np.concatenate( test_labels_hist_1norm,axis=-1 )
        self.labels_hist_1norm = np.concatenate( labels_hist_1norm,axis=-1 )
        self.labels_weights = np.concatenate( labels_weights,axis=1 )
        self.labels_weights /= self.labels_weights.min()


    def write_file_accuracies(self,obj_dump_dir=None):
        Write_all_file_accuracies(self.sph5_file_list,obj_dump_dir)


def main_NormedH5f():
    '''
    error global blocks in 17DRP5sb8fy region1:  1, 9,11,12,15,16
    '''
    t0 = time.time()
    dataset_name = 'MODELNET40'

    all_fn_globs = ['Merged_sph5/4096_mgs1_gs2_2d2_nmbf/']
    bxmh5_folder_name = ['Merged_bxmh5/4096_mgs1_gs2_2d2_fmn1444_mvp1-3200_1024_48_1-18_24_56_56-0d1_0d2_0d6-0d0_0d1_0d4-pd3-neg-3M1']
    eval_fnglob_or_rate = 'test'

    #all_fn_globs = ['ORG_sph5/90000_gs-4_-6d3/']
    #bxmh5_folder_name = 'ORG_bxmh5/90000_gs-4_-6d3_fmn1111-6400_2400_320_32-32_16_32_48-0d1_0d3_0d9_2d7-0d1_0d2_0d6_1d8-pd3-4C0'
    #eval_fnglob_or_rate = 'scene0000_01'

    #all_fn_globs = ['Org_sph5/9000_gs-4_-6d3/']
    #bxmh5_folder_name = 'Org_bxmh5/9000_gs-4_-6d3_fmn1-320_32-320_48-0d9_2d7-0d6_1d8-pd3-TMP'
    #eval_fnglob_or_rate = 'scene0000_00'

    #eval_fnglob_or_rate = 'region0'

    #all_filename_globs = ['all_merged_nf5']
    #eval_fnglob_or_rate = '17DRP5sb8fy'
    num_point_block = None

    only_evaluate = False
    Feed_Data_Elements = ['xyz','xyz_1norm_file','xyz_midnorm_block']
    Feed_Data_Elements = ['xyz']
    Feed_Label_Elements = ['label_category']
    #feed_label_elements = ['label_category','label_instance']
    net_configs = {}
    net_configs['loss_weight'] = 'E'
    net_provider = Net_Provider(
                                net_configs=net_configs,
                                dataset_name=dataset_name,
                                all_filename_globs=all_fn_globs,
                                eval_fnglob_or_rate=eval_fnglob_or_rate,
                                bxmh5_folder_name = bxmh5_folder_name,
                                only_evaluate = False,
                                feed_data_elements=Feed_Data_Elements,
                                feed_label_elements=Feed_Label_Elements)
    t1 = time.time()
    print(net_provider.data_summary_str)
    print('init time:',t1-t0)


    #ply_flag = 'region'
    ply_flag = 'global_block'
    #ply_flag = 'sub_block'
    #ply_flag = 'none'
    steps = { 'region':net_provider.eval_num_blocks, 'global_block':2, 'sub_block':1,'none':8 }
    IsShuffleIdx = False
    for bk in  range(0,net_provider.eval_num_blocks,steps[ply_flag]):
        #end = min(bk+s,net_provider.eval_num_blocks)
        #end = net_provider.eval_num_blocks
        end = min( bk+steps[ply_flag], net_provider.eval_num_blocks )
        t0 = time.time()

        dataset = net_provider.batch_dataset( 'eval', bk, end, False, aug_types='' )
        import pdb; pdb.set_trace()  # XXX BREAKPOINT

        cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps, \
            cur_fmap_neighbor_idis, fid_start_end  = net_provider.get_eval_batch(bk, end, False, aug_types='')
        print('read each block t=%f ms IsShuffleIdx=%s'%( (time.time()-t0)/(end-bk)*1000, IsShuffleIdx ) )
        xyz_idxs = net_provider.feed_data_ele_idxs['xyz']

        if 'color_1norm' not in net_provider.feed_data_ele_idxs:
          return

        color_idxs = net_provider.feed_data_ele_idxs['color_1norm']
        xyz = cur_data[...,xyz_idxs]
        color0 = cur_data[...,color_idxs] * 255
        color = color0.astype(np.uint8)
        xyz_color = np.concatenate( [xyz,color],axis=-1 )

        if ply_flag == 'region':
            ply_fn = DATASET_DIR(dataset_name) + '/PlyFile_17DRP5sb8fy/' + str(eval_fnglob_or_rate)+ '.ply'
            create_ply( xyz_color,ply_fn )
            break
        if ply_flag == 'global_block':
            ply_fn = DATASET_DIR(dataset_name) + '/PlyFile_17DRP5sb8fy/globalblocks/'+ str(eval_fnglob_or_rate) +'/global_b' + str(bk) + '.ply'
            for i in range(xyz_color.shape[0]):
                for j in range(xyz_color.shape[1]):
                    if cur_smp_weights[i,j,0 ] == 0:
                        xyz_color[i,j,3:6] = 0
            create_ply(xyz_color,ply_fn)
            if bk>5:
                break
        if ply_flag == 'sub_block':
            assert xyz_color.ndim == 4
            for sub_k in range(0,xyz_color.shape[1]):
                ply_fn = DATASET_DIR(dataset_name) + '/PlyFile_17DRP5sb8fy/'+ str(eval_fnglob_or_rate) +'/global0_subblocks/sub' + str(sub_k) + '.ply'
                create_ply(xyz_color[0,sub_k,...],ply_fn)
            break

    print(cur_label.shape)
    print(cur_smp_weights.shape)
    print(cur_data.shape)
    print(cur_data[0,0:3,:])
    print(cur_label[0,0:3,:])
    print(cur_smp_weights[0,0:3,:])


if __name__=='__main__':
    main_NormedH5f()
    #check_bxmap_pl_shape_match()
    #CheckModelNet()
