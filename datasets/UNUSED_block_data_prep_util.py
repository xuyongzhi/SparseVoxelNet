#xyz Sep 2017
'''
Data preparation for datsets: stanford_indoor, scannet, ETH_semantic3D
Core idea: store all the information in hdf5 file itself

# The workflow to use this tool:
Raw_H5f -> Sorted_H5f -> merge block to get new block size -> randomnly select n points
    -> Normed_H5f -> Net_Provider

## Raw_H5f store the raw data of dataset, which contains several datasets: xyz, label, color.... Each dataset
    stores the whole data for one dtype data.
    (.rh5)
## Sorted_H5f contains lots of lots of dataset. Each dataset stores all types of data within a spacial block.
    The point number of each block/dataset can be fix or not.
    (.sh5) Use class Sort_RawH5f to generate sorted file with unfixed point num in each block, and a small stride / step size.
    Then merge .sh5 file with small stride/step size to get larger size block.
    (.rsh5) Randomly sampling .sh5 file to get Sorted_H5f file with fixed point number in each block.
## Normed_H5f includes 4 datasets: data, label, raw_xyz, pred_logit
    (.sph5) This file is directly used to feed data for deep learning models.
    .sph5 file is generated by Sorted_H5f.file_normalize_to_NormedH5F()
## For all three files, show_h5f_summary_info() can use to show the info summary.
## scannet_block_sample.py is the basic usage for these classes.
'''

from __future__ import print_function
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))
#from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import math
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
import ply_util
#from global_para import GLOBAL_PARA
sys.path.append(BASE_DIR+'/MATTERPORT_util')
sys.path.append(BASE_DIR+'/KITTI_util')
from MATTERPORT_util import get_cat40_from_rawcat
sys.path.append(BASE_DIR+'/all_datasets_meta')
from datasets_meta import DatasetsMeta
import csv,pickle
from configs import get_gsbb_config, NETCONFIG
import magic

'''                         Def key words list
Search with "name:" to find the definition.

    rootb_split_idxmap
    bxmh5
    flatten_bidxmap
    sg_bidxmap
    baseb_exact_flat_num
    global_step
'''
'''    Important functions
    get_blockids_of_dif_stride_step
    get_bidxmap
    get_all_bidxmaps
    gsbb naming: get_pyramid_flag
    file_saveas_pyramid_feed
'''
'''     step, stride Configuration
    (1) set_whole_scene_stride_step: limit stride, step of every cascade by whole scene scope. By calling update_align_scope_by_stridetoalign_
    (2) IsLimitStrideStepCascades_Inbxmap : Always limit step and stride larger than last cascade in bxmh5
'''

SHOW_ONLY_ERR = False
DEBUGTMP = False
ENABLECHECK = False
START_T = time.time()

g_h5_num_row_1M = 5*1000
ROOT_DIR = os.path.dirname(BASE_DIR)
UPER_DIR = os.path.dirname(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')

DATA_SOURCE_NAME_LIST = ['ETH','STANFORD_INDOOR3D','SCANNET','MATTERPORT','KITTI', 'MODELNET40']
FLOAT_BIAS = 1e-8

def isin_sorted( a,v ):
    i = np.searchsorted(a,v)
    if i>=a.size: return False
    r = a[i] == v
    return r

def get_stride_step_name(block_stride,block_step):
    if not block_step[0] == block_step[1]:
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass
    assert block_stride[0] == block_stride[1]
    #assert (block_step[0] == block_step[2] and block_stride[0] == block_stride[2]) or (block_step[2]<0 and block_stride[2]<0)

    def get_str(v):
        return str(v).replace('.','d')
        #assert (v*100) % 1 < 1e-8, "v=%s"%(str(v))
        #if v%1!=0:
        #    if (v*10)%1 < 1e-8: return '%dd%d'%(v,v%1*10)
        #    else: return '%dd%d%d'%(v,v%1*10, v*10%1*10)
        #else: return str(int(v))

    if block_stride[2] == -1:
        return 'stride-%s-step-%s'%(get_str(block_stride[0]),get_str(block_step[0]))
    else:
        return 'stride_%s_step_%s'%(get_str(block_stride[0]),get_str(block_step[0]))
def rm_file_name_midpart(fn,rm_part):
    base_name = os.path.basename(fn)
    parts = base_name.split(rm_part)
    if len(parts)>1:
        new_bn = parts[0] + parts[1]
    else:
        new_bn = parts[0]
    new_fn = os.path.join(os.path.dirname(fn),new_bn)
    return new_fn


def copy_h5f_attrs(h5f_attrs):
    attrs = {}
    for e in h5f_attrs:
        attrs[e] = h5f_attrs[e]
    return attrs
def get_mean_sg_sample_rate(sum_sg_bidxmap_sample_num):
    global_block_num = sum_sg_bidxmap_sample_num[0,4]
    subblock_num = sum_sg_bidxmap_sample_num[:,-1]
    mean_sg_bidxmap_sample_num = np.copy(sum_sg_bidxmap_sample_num)
    for i in range(sum_sg_bidxmap_sample_num.shape[0]):
        mean_sg_bidxmap_sample_num[i,0:5] /= mean_sg_bidxmap_sample_num[i,4]
        mean_sg_bidxmap_sample_num[i,5:8] /= mean_sg_bidxmap_sample_num[i,7]
    return mean_sg_bidxmap_sample_num,global_block_num,subblock_num
def get_mean_flatten_sample_rate(sum_flatten_bmap_sample_num):
    global_block_num = sum_flatten_bmap_sample_num[0,2]
    mean_flatten_bmap_sample_num = np.copy(sum_flatten_bmap_sample_num)
    for i in range(sum_flatten_bmap_sample_num.shape[0]):
        mean_flatten_bmap_sample_num[i,0:3] /= mean_flatten_bmap_sample_num[i,2]
    return mean_flatten_bmap_sample_num,global_block_num

def get_attrs_str(attrs):
    attrs_str = ''
    for a in attrs:
        elenames = ''
        if type(attrs[a])==str:
            a_str = attrs[a]
        else:
            a_val = attrs[a]
            if a == "sum_sg_bidxmap_sample_num":
                a_val,global_block_num,subblock_num = get_mean_sg_sample_rate(a_val)
                elenames = str(GlobalSubBaseBLOCK.get_sg_bidxmap_sample_num_elename()) + '\n' + 'global_block_num: %d'%(global_block_num) + '\tsubblock_num: %s'%(subblock_num)  + '\n'
            if a == "sum_flatten_bmap_sample_num":
                a_val,global_block_num = get_mean_flatten_sample_rate(a_val)
                elenames = str(GlobalSubBaseBLOCK.get_flatten_bidxmaps_sample_num_elename()) +'\n' + 'global_block_num: %d'%(global_block_num) + '\n'

            a_str = np.array2string(a_val,precision=2,separator=',',suppress_small=True)
        attrs_str += ( a+':\n'+elenames+a_str+'\n' )
    return attrs_str

def show_h5f_summary_info(h5f):
    root_attrs = [attr for attr in h5f.attrs]
    summary_str = ''
    summary_str += '--------------------------------------------------------------------------\n'
    summary_str += 'The root_attr: %s'%(root_attrs) + '\n'
    summary_str += get_attrs_str(h5f.attrs) + '\n'

    summary_str += '\n--------------------------------------------------------------------------\n'
    summary_str += 'The elements in h5f\n'
    def show_dset(dset_name,id):
        dset_str = ''
        if id>10: return dset_str
        dset = h5f[dset_name]
        dset_str += '# dataset %d: %s shape=%s\n'%(id,dset_name,dset.shape)
        if id>6: return dset_str
        dset_str += get_attrs_str(dset.attrs) + '\n'
        if len(dset.shape)==2:
            dset_str += str( dset[0:min(10,dset.shape[0]),:]) + '\n'
        if len(dset.shape)==3:
           dset_str += str( dset[0:min(2,dset.shape[0]),:] ) + '\n'
        elif len(dset.shape)==4:
            var = dset[0:min(1,dset.shape[0]),0,0:min(2,dset.shape[2]),:]
            dset_str += np.array2string(var,formatter={'float_kind':lambda var:"%0.2f"%var}) + '\n'
        dset_str += '\n'
        return dset_str

    def show_root_ele(ele_name,id):
        root_ele_str = ''
        ele = h5f[ele_name]
        if type(ele) == h5py._hl.group.Group:
            root_ele_str += 'The group: %s'%(ele_name) + '\n'
            root_ele_str += get_attrs_str(ele.attrs) + '\n'
            for dset_name in ele:
                root_ele_str += show_dset(ele_name+'/'+dset_name,id)
        else:
            root_ele_str += show_dset(ele_name,id)
        return root_ele_str

    k = -1
    for k, ele_name in enumerate(h5f):
        if ele_name == 'xyz':
            summary_str += show_dset(ele_name,k)
            continue
        summary_str += show_root_ele(ele_name,k)
    summary_str += '%d datasets totally'%(k+1)+'\n'
    print( summary_str )
    return summary_str

def get_sample_choice(org_N,sample_N,random_sampl_pro=None):
    '''
    all replace with random_choice laer
    '''
    sample_method='random'
    if sample_method == 'random':
        if org_N == sample_N:
            sample_choice = np.arange(sample_N)
        elif org_N > sample_N:
            sample_choice = np.random.choice(org_N,sample_N,replace=False,p=random_sampl_pro)
        else:
            #sample_choice = np.arange(org_N)
            new_samp = np.random.choice(org_N,sample_N-org_N)
            sample_choice = np.concatenate( (np.arange(org_N),new_samp) )
        reduced_num = org_N - sample_N
        #str = '%d -> %d  %d%%'%(org_N,sample_N,100.0*sample_N/org_N)
        #print(str)
    return sample_choice,reduced_num
def random_choice(org_vector,sample_N,random_sampl_pro=None, keeporder=True, only_tile_last_one=False):
    assert org_vector.ndim == 1
    org_N = org_vector.size
    if org_N == sample_N:
        sampled_vector = org_vector
    elif org_N > sample_N:
        sampled_vector = np.random.choice(org_vector,sample_N,replace=False,p=random_sampl_pro)
        if keeporder:
            sampled_vector = np.sort(sampled_vector)
    else:
        if only_tile_last_one:
            new_vector = np.array( [ org_vector[-1] ]*(sample_N-org_N) ).astype(org_vector.dtype)
        else:
            new_vector = np.random.choice(org_vector,sample_N-org_N,replace=True)
        sampled_vector = np.concatenate( [org_vector,new_vector] )
    #str = '%d -> %d  %d%%'%(org_N,sample_N,100.0*sample_N/org_N)
    #print(str)
    return sampled_vector

def index_in_sorted(sorted_vector,values):
    if values.ndim==0:
        values = np.array([values])
    assert values.ndim<=1 and sorted_vector.ndim==1
    #values_valid = values[np.isin(values,sorted_vector)]
    indexs = np.searchsorted(sorted_vector,values)
    indexs_valid = []
    for j,index in enumerate(indexs):
        if index<sorted_vector.size and  sorted_vector[index] == values[j]:
            indexs_valid.append( index )
    indexs_valid = np.array(indexs_valid)
    assert indexs_valid.size <= values.size
    #assert indexs.size==0 or np.max(indexs) < sorted_vector.size, 'err in index_in_sorted'
    return indexs_valid

def check_h5fs_intact(file_name):
    if not os.path.exists(file_name):
        return False,"file not exist: %s"%(file_name)
    f_format = os.path.splitext(file_name)[-1]
    if f_format == '.rh5':
        return Raw_H5f.check_rh5_intact(file_name)
    elif f_format == '.sh5' or f_format == '.rsh5':
        return Sorted_H5f.check_sh5_intact(file_name)
    elif f_format == '.sph5' or f_format == '.prh5':
        return Normed_H5f.check_sph5_intact(file_name)
    elif f_format == '.bmh5':
        return GlobalSubBaseBLOCK.check_bmh5_intact(file_name)
    else:
        return False, "file format not recognized %s"%(f_format)

def float_exact_division( A, B ):
    C = A / B
    r = np.isclose( C, np.rint(C) )
    R = r.all()
    return R

def my_fix(orgvar):
    # why do not use np.fix() directly: np.fix(2.999999) = 2.0
    assert orgvar.ndim == 1
    rint_var = np.rint(orgvar)
    zero_gap = rint_var - orgvar
    fix_var = np.copy(orgvar).astype(np.int64)
    for i in range(orgvar.size):
        if np.isclose(zero_gap[i],0):
            fix_var[i] = rint_var[i].astype(np.int64)
        else:
            fix_var[i] = np.fix(orgvar[i]).astype(np.int64)
    return fix_var

def my_ceil(orgvar):
    # why do not use np.ceil: np.ceil(12.0000000000000001)=13
    assert orgvar.ndim == 1
    rint_var = np.rint(orgvar)
    zero_gap = rint_var - orgvar
    ceil_var = np.copy(orgvar).astype(np.int64)
    for i in range(orgvar.size):
        if np.isclose(zero_gap[i],0):
            ceil_var[i] = rint_var[i].astype(np.int64)
        else:
            try:
                ceil_var[i] = np.ceil(orgvar[i]).astype(np.int64)
            except:
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
                pass
    return ceil_var



class Raw_H5f():
    '''
    * raw data:unsorted points,all the time in one dataset
    * Each data type as a hdf5 dataset: xyz, intensity, label, color
    * class "Sorted_H5f" will sort data to blocks based on this class
    '''
    file_flag = 'RAW_H5F'
    h5_num_row_1M = 50*1000
    dtypes = { 'xyz':np.float32, 'nxnynz':np.float32, 'intensity':np.int32, \
              'color':np.uint8,'label_category':np.uint32,'label_instance':np.int32,\
              'label_material':np.int32, 'label_mesh':np.int32, 'label_raw_category':np.int32 }
    num_channels = {'xyz':3,'nxnynz':3,'intensity':1,'color':3,'label_category':1,\
                    'label_instance':1,'label_material':1,'label_mesh':1, 'label_raw_category':1}
    def __init__(self,raw_h5_f,file_name,datasource_name=None):
        self.h5f = raw_h5_f
        if datasource_name == None:
            assert 'datasource_name' in self.h5f.attrs
        else:
            self.h5f.attrs['datasource_name'] = datasource_name
        assert self.h5f.attrs['datasource_name'] in DATA_SOURCE_NAME_LIST
        self.datasource_name = self.h5f.attrs['datasource_name']
        self.dataset_meta = DatasetsMeta(self.datasource_name)
        self.get_summary_info()
        self.file_name = file_name
        self.num_default_row = 0

    def show_h5f_summary_info(self):
        print('\n\nsummary of file: ',self.file_name)
        return show_h5f_summary_info(self.h5f)

    def set_num_default_row(self,N):
        self.num_default_row = N

    def get_dataset(self,data_name):
        if data_name in self.h5f:
            return self.h5f[data_name]
        assert(data_name in self.dtypes)
        nc = self.num_channels[data_name]
        dset = self.h5f.create_dataset(data_name,shape=(self.num_default_row,nc),\
                                    maxshape=(None,nc),dtype=self.dtypes[data_name],\
                                    chunks = (self.h5_num_row_1M,nc),\
                                    compression = "gzip")
        dset.attrs['valid_num'] = 0
        setattr(self,data_name+'_dset',dset)
        if 'element_names' not in self.h5f.attrs:
            self.h5f.attrs['element_names'] = [data_name]
        else:
            self.h5f.attrs['element_names'] = [data_name]+[e for e in self.h5f.attrs['element_names']]
        return dset
    def get_total_num_channels_name_list(self):
        total_num_channels = 0
        data_name_list = [str(dn) for dn in self.h5f]
        for dn in data_name_list:
            total_num_channels += self.num_channels[dn]

        return total_num_channels,data_name_list

    def append_to_dset(self,dset_name,new_data):
       self.add_to_dset(dset_name,new_data,None,None)

    def get_all_dsets(self,start_idx,end_idx):
        out_dset_order = ['xyz','color','label','intensity']
        data_list = []
        for dset_name in out_dset_order:
            if dset_name in self.h5f:
                data_k = self.h5f[dset_name][start_idx:end_idx,:]
                data_list.append(data_k)
        data = np.concatenate(data_list,1)
        return data

    def add_to_dset(self,dset_name,new_data,start,end):
        dset = self.get_dataset(dset_name)
        assert dset.ndim == new_data.ndim
        valid_n  = dset.attrs['valid_num']
        if start == None:
            start = valid_n
            end = start + new_data.shape[0]
        if dset.shape[0] < end:
            dset.resize((end,)+dset.shape[1:])
        if valid_n < end:
            dset.attrs['valid_num'] = end
        if new_data.ndim==1 and dset.ndim==2 and dset.shape[1]==1:
            new_data = np.expand_dims(new_data,1)
        dset[start:end,:] = new_data

    def rm_invalid(self):
        for dset_name in self.h5f:
            dset = self.h5f[dset_name]
            if 'valid_num' in dset.attrs:
                valid_num = dset.attrs['valid_num']
                if valid_num < dset.shape[0]:
                    dset.resize( (valid_num,)+dset.shape[1:] )

    def get_summary_info(self):
        for dset_name in self.h5f:
            setattr(self,dset_name+'_dset',self.h5f[dset_name])
        if 'xyz' in self.h5f:
            self.total_row_N = self.xyz_dset.shape[0]
            self.xyz_max = self.xyz_dset.attrs['max']
            self.xyz_min = self.xyz_dset.attrs['min']
            self.xyz_scope = self.xyz_max - self.xyz_min


    def generate_objfile(self,obj_file_name=None,IsLabelColor=False,xyz_cut_rate=None):
        if obj_file_name==None:
            base_fn = os.path.basename(self.file_name)
            base_fn = os.path.splitext(base_fn)[0]
            folder_path = os.path.dirname(self.file_name)
            obj_folder = os.path.join(folder_path,'obj/'+base_fn)
            print('obj_folder:',obj_folder)
            obj_file_name_nocolor = os.path.join(obj_folder,base_fn+'_xyz.obj')
            if IsLabelColor:
              base_fn = base_fn + '_TrueLabel'
            obj_file_name = os.path.join(obj_folder,base_fn+'.obj')
            if not os.path.exists(obj_folder):
                os.makedirs(obj_folder)
            print('automatic obj file name: %s'%(obj_file_name))


        with open(obj_file_name,'w') as out_obj_file:
          with open(obj_file_name_nocolor,'w') as xyz_obj_file:
            xyz_dset = self.xyz_dset
            if 'color' in self.h5f:
                color_dset = self.color_dset
            else:
                if 'label_category' in self.h5f:
                    IsLabelColor = True
            if IsLabelColor:
                label_category_dset = self.label_category_dset

            if xyz_cut_rate != None:
                # when rate < 0.5: cut small
                # when rate >0.5: cut big
                xyz_max = np.array([ np.max(xyz_dset[:,i]) for i in range(3) ])
                xyz_min = np.array([ np.min(xyz_dset[:,i]) for i in range(3) ])
                xyz_scope = xyz_max - xyz_min
                xyz_thres = xyz_scope * xyz_cut_rate + xyz_min
                print('xyz_thres = ',str(xyz_thres))
            cut_num = 0

            row_step = self.h5_num_row_1M * 10
            row_N = xyz_dset.shape[0]
            for k in range(0,row_N,row_step):
                end = min(k+row_step,row_N)
                xyz_buf_k = xyz_dset[k:end,:]

                if 'color' in self.h5f:
                    color_buf_k = color_dset[k:end,:]
                    buf_k = np.hstack((xyz_buf_k,color_buf_k))
                else:
                    buf_k = xyz_buf_k
                if IsLabelColor:
                    label_k = label_category_dset[k:end,0]
                for j in range(0,buf_k.shape[0]):
                    is_cut_this_point = False
                    if xyz_cut_rate!=None:
                        # cut by position
                        for xyz_j in range(3):
                            if (xyz_cut_rate[xyz_j] >0.5 and buf_k[j,xyz_j] > xyz_thres[xyz_j]) or \
                                (xyz_cut_rate[xyz_j]<=0.5 and buf_k[j,xyz_j] < xyz_thres[xyz_j]):
                                is_cut_this_point =  True
                    if is_cut_this_point:
                        cut_num += 1
                        continue

                    if not IsLabelColor:
                        str_j = 'v   ' + '\t'.join( ['%0.5f'%(d) for d in  buf_k[j,0:3]]) + '  \t'\
                        + '\t'.join( ['%d'%(d) for d in  buf_k[j,3:6]]) + '\n'
                    else:
                        label = label_k[j]
                        label_color = self.dataset_meta.label2color[label]
                        str_j = 'v   ' + '\t'.join( ['%0.5f'%(d) for d in  buf_k[j,0:3]]) + '  \t'\
                        + '\t'.join( ['%d'%(d) for d in  label_color ]) + '\n'
                    nocolor_str_j = 'v   ' + '\t'.join( ['%0.5f'%(d) for d in  buf_k[j,0:3]]) + '  \n'
                    out_obj_file.write(str_j)
                    xyz_obj_file.write(nocolor_str_j)

            print('gen raw obj: %s'%(obj_file_name,))

    def rh5_create_done(self):
        self.rm_invalid()
        self.add_geometric_scope()

        self.write_raw_summary()
        #self.show_h5f_summary_info()

    def write_raw_summary(self):
        summary_fn = os.path.splitext( self.file_name )[0]+'.txt'
        with open(summary_fn,'w') as summary_f:
            summary_f.write( self.show_h5f_summary_info() )

    def add_geometric_scope(self,line_num_limit=None):
        ''' calculate the geometric scope of raw h5 data, and add the result to attrs of dset'''
        #begin = time.time()
        max_xyz = -np.ones((3))*1e10
        min_xyz = np.ones((3))*1e10

        xyz_dset = self.xyz_dset
        row_step = self.h5_num_row_1M
        print('File: %s   %d lines'\
              %(os.path.basename(self.file_name),xyz_dset.shape[0]) )
        #print('read row step = %d'%(row_step))

        for k in range(0,xyz_dset.shape[0],row_step):
            end = min(k+row_step,xyz_dset.shape[0])
            xyz_buf = xyz_dset[k:end,:]
            xyz_buf_max = xyz_buf.max(axis=0)
            xyz_buf_min = xyz_buf.min(axis=0)
            max_xyz = np.maximum(max_xyz,xyz_buf_max)
            min_xyz = np.minimum(min_xyz,xyz_buf_min)

            if line_num_limit!=None and k > line_num_limit:
                print('break at k = ',line_num_limit)
                break
        xyz_dset.attrs['max'] = max_xyz
        xyz_dset.attrs['min'] = min_xyz
        self.h5f.attrs['xyz_max'] = max_xyz
        self.h5f.attrs['xyz_min'] = min_xyz
        max_str = '  '.join([ str(e) for e in max_xyz ])
        min_str = '  '.join([ str(e) for e in min_xyz ])
        print('max_str=%s\tmin_str=%s'%(max_str,min_str) )
        #print('T=',time.time()-begin)

    @staticmethod
    def check_rh5_intact( file_name ):
        f_format = os.path.splitext(file_name)[-1]
        assert f_format == '.rh5'
        if not os.path.exists(file_name):
            return False, "%s not exist"%(file_name)
        #if os.path.getsize( file_name ) / 1000.0 < 100:
        #    return False,"file too small < 20 K"
        file_type = magic.from_file(file_name)
        if "Hierarchical Data Format" not in file_type:
            return False,"File signature err"
        with h5py.File(file_name,'r') as h5f:
            attrs_to_check = ['xyz_max','xyz_min']
            for attrs in attrs_to_check:
                if attrs not in h5f.attrs:
                    return False, "%s not in %s"%(attrs,file_name)
        return True,""



def Write_all_file_accuracies(normed_h5f_file_list=None,out_path=None,pre_out_fn=''):
    if normed_h5f_file_list == None:
        normed_h5f_file_list = glob.glob( GLOBAL_PARA.stanford_indoor3d_globalnormedh5_stride_0d5_step_1_4096 +
                            '/Area_2_office_1*' )
    if out_path == None: out_path = os.path.join(GLOBAL_PARA.stanford_indoor3d_globalnormedh5_stride_0d5_step_1_4096,
                                    'pred_accuracy')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    all_acc_fn = os.path.join(out_path,pre_out_fn+'accuracies.txt')
    all_ave_acc_fn = os.path.join(out_path,pre_out_fn+'average_accuracies.txt')
    class_TP = class_FN = class_FP = np.zeros(shape=(len(Normed_H5f.g_class2label)))
    total_num = 0
    average_class_accu_ls = []
    with open(all_acc_fn,'w') as all_acc_f,open(all_ave_acc_fn,'w') as all_ave_acc_f:
        for i,fn in enumerate(normed_h5f_file_list):
            h5f = h5py.File(fn,'r')
            norm_h5f = Normed_H5f(h5f,fn)
            class_TP_i,class_FN_i,class_FP_i,total_num_i,acc_str_i,ave_acc_str_i = norm_h5f.Get_file_accuracies(
                IsWrite=False, out_path = out_path)
            class_TP = class_TP_i + class_TP
            class_FN = class_FN_i + class_FN
            class_FP = class_FP_i + class_FP
            total_num = total_num_i +  total_num

            if acc_str_i != '':
                all_acc_f.write('File: '+os.path.basename(fn)+'\n')
                all_acc_f.write(acc_str_i+'\n')
                all_ave_acc_f.write(ave_acc_str_i+'\t: '+os.path.basename(fn)+'\n')

        acc_str,ave_acc_str = Normed_H5f.cal_accuracy(class_TP,class_FN,class_FP,total_num)
        ave_str = 'Throughout All %d files.\n'%(i+1) +  acc_str
        all_acc_f.write('\n'+ave_str)
        all_ave_acc_f.write('\n'+ave_str)
    print('accuracy file: '+all_acc_fn)
    print('average accuracy file: '+all_ave_acc_fn)
    return ave_str,out_path,class_TP,class_FN,class_FP,total_num

def Write_Area_accuracies():
    ave_str_areas = ''
    class_TP = class_FN = class_FP = np.zeros(shape=(len(Normed_H5f.g_class2label)))
    total_num = 0
    for i in range(6):
        glob_i = 'Area_%d'%(i+1)
        normed_h5f_file_list = glob.glob( os.path.join(GLOBAL_PARA.stanford_indoor3d_globalnormedh5_stride_0d5_step_1_4096,
                                glob_i+'*') )
        ave_str,out_path,class_TP_i,class_FN_i,class_FP_i,total_num_i = Write_all_file_accuracies(normed_h5f_file_list,pre_out_fn=glob_i+'_')
        class_TP = class_TP_i + class_TP
        class_FN = class_FN_i + class_FN
        class_FP = class_FP_i + class_FP
        total_num = total_num_i + total_num

        ave_str_areas += '\nArea%d\n'%i
        ave_str_areas += ave_str
    acc_str,ave_acc_str = Normed_H5f.cal_accuracy(class_TP,class_FN,class_FP,total_num)
    all_area_str = '\nThrough %d areas.\n'%(i+1)+acc_str
    with open(os.path.join(out_path,'areas_accuracies.txt'),'w' ) as area_acc_f:
        area_acc_f.write(ave_str_areas)
        area_acc_f.write(all_area_str)



#-------------------------------------------------------------------------------
# Test above codes
#-------------------------------------------------------------------------------


def main(file_list):

    outdoor_prep = MAIN_DATA_PREP()
    actions = ['merge','sample_merged','obj_sampled_merged','norm_sampled_merged']
    actions = ['merge','sample_merged','norm_sampled_merged']
    outdoor_prep.main(file_list,actions,sample_num=4096,sample_method='random',\
                      stride=[8,8,-1],step=[8,8,-1])

    #outdoor_prep.Do_sort_to_blocks()
    #Do_extract_part_area()
    #outdoor_prep.test_sub_block_ks()
    #outdoor_prep.DO_add_geometric_scope_file()
    #outdoor_prep.DO_gen_rawETH_to_h5()

def show_h5f_file():
    fn = '/home/y/Research/dynamic_pointnet/data/Matterport3D_H5F/v1/scans/17DRP5sb8fy/stride_0d1_step_0d1/region2.sh5'
    fn = '/home/y/DS/Matterport3D/Matterport3D_H5F/v1/scans/17DRP5sb8fy/stride_0d1_step_0d1_pyramid-1_2-512_128_64_16-0d2_0d4_0d8_16/region2.prh5'
    with h5py.File(fn,'r') as h5f:
        show_h5f_summary_info(h5f)

if __name__ == '__main__':
    START_T = time.time()
    Do_extract_part_area()
    T = time.time() - START_T
    print('exit main, T = ',T)