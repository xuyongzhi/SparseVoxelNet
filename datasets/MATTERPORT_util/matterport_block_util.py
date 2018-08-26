#xyz Dec 2017
from __future__ import print_function
import pdb, traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from datasets.block_data_prep_util import Raw_H5f, check_h5fs_intact
#Sort_RawH5f,Sorted_H5f,Normed_H5f,show_h5f_summary_info,MergeNormed_H5f,get_stride_step_name
#from block_data_prep_util import GlobalSubBaseBLOCK,get_mean_sg_sample_rate,get_mean_flatten_sample_rate,
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
import zipfile,gzip
from plyfile import PlyData, PlyElement

TMPDEBUG = False
SHOWONLYERR = True
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(ROOT_DIR,'data')

Material_Index = 0
Instance_Index = 1
Category_Index = 2

def zip_extract(ply_item_name,zipf,house_dir_extracted):
    '''
    extract file if not already
    '''
    #zipfile_name = '%s/%s/%s.%s'%(house_name,groupe_name,file_name,file_format)
    file_path = house_dir_extracted + '/' + ply_item_name
    if not os.path.exists(file_path):
        print('extracting %s...'%(file_path))
        file_path_extracted  = zipf.extract(ply_item_name,house_dir_extracted)
        print('file extracting finished: %s'%(file_path_extracted) )
        assert file_path == file_path_extracted
    else:
        print('file file already extracted: %s'%(file_path))
    return file_path

def parse_ply_file(ply_fo):
    '''
    element vertex 1522546
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    property float tx
    property float ty
    property uchar red
    property uchar green
    property uchar blue

    element face 3016249
    property list uchar int vertex_indices
    property int material_id
    property int segment_id
    property int category_id
    '''
    plydata = PlyData.read(ply_fo)
    num_ele = len(plydata.elements)
    num_vertex = plydata['vertex'].count
    num_face = plydata['face'].count
    data_vertex = plydata['vertex'].data
    data_face = plydata['face'].data

    ## face
    face_vertex_indices = data_face['vertex_indices']
    face_vertex_indices = np.concatenate(face_vertex_indices,axis=0)
    face_vertex_indices = np.reshape(face_vertex_indices,[-1,3])

    face_eles = ['vertex_indices','material_id','segment_id','category_id']
    datas_face = {}
    for e in face_eles:
        datas_face[e] = np.expand_dims(data_face[e],axis=-1)
    face_semantic = np.concatenate([datas_face['material_id'], datas_face['segment_id'], datas_face['category_id']],axis=1)

    #min_cat_id = np.min(data_face['category_id'])
    #neg_id_index = np.where(data_face['category_id']<0)[0]
    #print(len(neg_id_index))

    ## vertex
    vertex_eles = ['x','y','z','nx','ny','nz','tx','ty','red','green','blue']
    datas_vertex = {}
    for e in vertex_eles:
        datas_vertex[e] = np.expand_dims(data_vertex[e],axis=-1)
    vertex_xyz = np.concatenate([datas_vertex['x'],datas_vertex['y'],datas_vertex['z']],axis=1)
    vertex_nxnynz = np.concatenate([datas_vertex['nx'],datas_vertex['ny'],datas_vertex['nz']],axis=1)
    vertex_rgb = np.concatenate([datas_vertex['red'],datas_vertex['green'],datas_vertex['blue']],axis=1)

    vertex_semantic,vertex_indices_multi_semantic,face_indices_multi_semantic = get_vertex_label_from_face(face_vertex_indices,face_semantic,num_vertex)

    IsDeleteNonPosId = False
    if IsDeleteNonPosId:
      # get vertex and face  with category_id==-1 or ==0
      vertex_nonpos_indices = np.where(vertex_semantic[:, Category_Index]<1)[0]
      face_nonpos_indices = np.where(face_semantic[:,Category_Index]<1)[0]

      vertex_del_indices = np.concatenate([vertex_indices_multi_semantic, vertex_nonpos_indices], 0)
      face_del_indices = np.concatenate([face_indices_multi_semantic, face_nonpos_indices], 0)
    else:
      vertex_del_indices = vertex_indices_multi_semantic
      face_del_indices = face_indices_multi_semantic

    # Del VexMultiSem
    vertex_xyz = np.delete(vertex_xyz, vertex_del_indices, axis=0)
    vertex_nxnynz = np.delete(vertex_nxnynz, vertex_del_indices, axis=0)
    vertex_rgb = np.delete(vertex_rgb, vertex_del_indices, axis=0)
    vertex_semantic = np.delete(vertex_semantic, vertex_del_indices,axis=0)
    face_vertex_indices = np.delete(face_vertex_indices, face_del_indices, axis=0)
    face_semantic = np.delete(face_semantic, face_del_indices, axis=0)

    return vertex_xyz,vertex_nxnynz,vertex_rgb,vertex_semantic,face_vertex_indices,face_semantic

def parse_house_file(house_fo):
    for i,line in enumerate( house_fo ):
        if i<1:
            print(line)
        break
def get_vertex_label_from_face(face_vertex_indices, face_semantic, num_vertex):
    '''
    face_vertex_indices: the vertex indices in each face
    vertex_face_indices: the face indices in each vertex
    '''
    vertex_face_indices = -np.ones(shape=[num_vertex,30])
    face_num_per_vertex = np.zeros(shape=[num_vertex]).astype(np.int8)
    vertex_semantic = np.zeros(shape=[num_vertex,3]) # only record the first one
    vertex_semantic_num = np.zeros(shape=[num_vertex])
    vertex_indices_multi_semantic = set()
    face_indices_multi_semantic = set()
    for i in range(face_vertex_indices.shape[0]):
        for vertex_index in face_vertex_indices[i]:
            face_num_per_vertex[vertex_index] += 1
            vertex_face_indices[vertex_index,face_num_per_vertex[vertex_index]-1] = i

            if vertex_semantic_num[vertex_index] == 0:
                vertex_semantic_num[vertex_index] += 1
                vertex_semantic[vertex_index] = face_semantic[i]
            else:
                # (1) Only 60% vertexs have unique labels for all three semntics
                # (2) There are 96% vertexs have unique labels for the first two:  category_id and segment_id
                # (3) only need category_id to be same
                IsSameSemantic = (vertex_semantic[vertex_index][Category_Index]==face_semantic[i][Category_Index]).all()
                if not IsSameSemantic:
                    vertex_semantic_num[vertex_index] += 1
                    vertex_indices_multi_semantic.add(vertex_index)
                    face_indices_multi_semantic.add(i)
    vertex_indices_multi_semantic = np.array(list(vertex_indices_multi_semantic))
    face_indices_multi_semantic = np.array(list(face_indices_multi_semantic))
    print('vertex rate with multiple semantic: %f'%(1.0*vertex_indices_multi_semantic.shape[0]/num_vertex))

    vertex_semantic = vertex_semantic.astype(np.int32)

   # vertex_semantic_num_max = np.max(vertex_semantic_num)
   # vertex_semantic_num_min = np.min(vertex_semantic_num)
   # vertex_semantic_num_mean = np.mean(vertex_semantic_num)
   # vertex_semantic_num_one = np.sum(vertex_semantic_num==1)
   # print(vertex_semantic_num_max)
   # print(vertex_semantic_num_mean)
   # print(vertex_semantic_num_min)
   # print(1.0*vertex_semantic_num_one/num_vertex)

    return vertex_semantic,vertex_indices_multi_semantic,face_indices_multi_semantic

def WriteRawH5f_Region_Ply(ply_item_name,rs_zf,house_name,scans_h5f_dir,house_dir_extracted):
    from MATTERPORT_util import get_cat40_from_rawcat
    #file_name = 'region'+str(k_region)
    region_ply_fn = zip_extract(ply_item_name,rs_zf,house_dir_extracted)
    s = ply_item_name.index('region_segmentations/region')+len('region_segmentations/region')
    e = ply_item_name.index('.ply')
    k_region = int( ply_item_name[ s:e ] )
    rawh5f_dir = scans_h5f_dir+'/rawh5/'+house_name
    if not os.path.exists(rawh5f_dir):
      os.makedirs(rawh5f_dir)
    rawh5f_fn = rawh5f_dir + '/region' + str(k_region)+'.rh5'
    IsIntact,_  = check_h5fs_intact(rawh5f_fn)

    if  IsIntact:
        print('file intact: %s'%(region_ply_fn))
    else:
        with open(region_ply_fn,'r') as ply_fo, h5py.File(rawh5f_fn,'w') as h5f:
            vertex_xyz, vertex_nxnynz, vertex_rgb, vertex_semantic, face_vertex_indices, face_semantic = \
                            parse_ply_file(ply_fo)
            label_category40 = get_cat40_from_rawcat(vertex_semantic[:,2])
            label_category40 = np.expand_dims(label_category40, 1)

            raw_h5f = Raw_H5f(h5f, rawh5f_fn,'MATTERPORT')
            raw_h5f.set_num_default_row(vertex_xyz.shape[0])
            raw_h5f.append_to_dset('xyz',vertex_xyz)
            raw_h5f.append_to_dset('nxnynz',vertex_nxnynz)
            raw_h5f.append_to_dset('color',vertex_rgb)
            raw_h5f.append_to_dset('label_material', vertex_semantic[:,Material_Index:Material_Index+1]) # material_id
            raw_h5f.append_to_dset('label_instance', vertex_semantic[:,Instance_Index:Instance_Index+1]) # segment_id
            raw_h5f.append_to_dset('label_raw_category', vertex_semantic[:,Category_Index:Category_Index+1]) # category_id
            raw_h5f.append_to_dset('label_category', label_category40) # category_id
            raw_h5f.rh5_create_done()
            raw_h5f.show_h5f_summary_info()

    return region_ply_fn

def WriteSortH5f_FromRawH5f(rawh5_file_ls,block_step_xyz,sorted_path,IsShowInfoFinished):
    Sort_RawH5f(rawh5_file_ls,block_step_xyz,sorted_path,IsShowInfoFinished)
    return rawh5_file_ls

def MergeSampleNorm_FromSortedH5f( base_sorted_h5fname,new_stride,new_step,new_sorted_path,more_actions_config ):
    with h5py.File(base_sorted_h5fname,'r') as f:
        sorted_h5f = Sorted_H5f(f,base_sorted_h5fname)
        sorted_h5f.merge_to_new_step(new_stride,new_step,new_sorted_path,more_actions_config)
    return base_sorted_h5fname
def SampleFile(base_sorted_h5fname,numpoint_block,IsGenNorm):
    with h5py.File(base_sorted_h5fname,'r') as f:
        sorted_h5f = Sorted_H5f(f,base_sorted_h5fname)
        sorted_h5f.file_random_sampling(numpoint_block,gen_norm=IsGenNorm)
    return base_sorted_h5fname

def NormSortedSampledFlie(fn):
    with h5py.File(fn,'r') as f:
        sorted_h5f = Sorted_H5f(f,fn)
        sorted_h5f.file_normalization(True)
    return fn

def GenPyramidSortedFlie(fn):
    with h5py.File(fn,'r') as f:
        sorted_h5f = Sorted_H5f(f,fn)
        Always_CreateNew_plh5 = False
        Always_CreateNew_bmh5 = False
        Always_CreateNew_bxmh5 = False
        if TMPDEBUG:
            Always_CreateNew_bmh5 = False
            Always_CreateNew_plh5 = False
            Always_CreateNew_bxmh5 = False

        sorted_h5f.file_saveas_pyramid_feed( IsShowSummaryFinished=False, Always_CreateNew_plh5 = Always_CreateNew_plh5, Always_CreateNew_bmh5 = Always_CreateNew_bmh5, Always_CreateNew_bxmh5=Always_CreateNew_bxmh5,
                                            IsGenPly = False and TMPDEBUG)
    return fn

class Matterport3D_Prepare():
    '''
    Read each region as a h5f.
    Vertex and face are stored in seperate h5f.
    In vertex h5f, the corresponding face ids are listed as "face indices".
    In face h5f, the corresponding vertex ids are also listed as "vertex indices"
    While downsampling of vertex set, all the faces related with the deleted point are deleted. Thus, the downsampling rate should be very low.
        If want high downsamplig rate, it should start with possion reconstuction mesh of low depth.
    The semantic labels of each point are achieved from faces.
    '''

    matterport3D_root_dir = '/DS/Matterport3D/Matterport3D_WHOLE'
    matterport3D_extracted_dir = '/DS/Matterport3D/Matterport3D_WHOLE_extracted'
    matterport3D_h5f_dir = DATA_DIR + '/MATTERPORT_H5TF'

    def __init__(self):
        self.scans_name = scans_name = '/v1/scans'
        self.scans_dir = self.matterport3D_root_dir+scans_name
        #self.scans_h5f_dir = self.matterport3D_h5f_dir+scans_name
        self.scans_h5f_dir = self.matterport3D_h5f_dir

        #self.house_dir = self.scans_dir+'/%s'%(house_name)
        #self.region_segmentations_zip_fn = self.house_dir+'/region_segmentations.zip'

        #self.house_h5f_dir = self.scans_h5f_dir+'/%s'%(house_name)
        #self.house_rawh5f_dir = self.house_h5f_dir+'/rawh5f'
        #if not os.path.exists(self.house_rawh5f_dir):
        #    os.makedirs(self.house_rawh5f_dir)
        #self.house_dir_extracted = self.matterport3D_extracted_dir+scans_name+'/%s'%(house_name)
        #if not os.path.exists(self.house_dir_extracted):
        #    os.makedirs(self.house_dir_extracted)

    def Parse_houses_regions(self,house_names_ls,MultiProcess=0):
        for house_name in house_names_ls:
            self.Parse_house_regions(house_name,MultiProcess)

    def move_path(self, house_name ):
        org_house_dir = self.scans_h5f_dir+'/%s'%(house_name)
        dir_ls = ['stride_0d1_step_0d1','stride_0d1_step_0d1-bidxmap-1d6_2-512_256_64-128_12_6-0d2_0d6_1d2','stride_0d1_step_0d1_pyramid-1d6_2-512_256_64-128_12_6-0d2_0d6_1d2']
        for dn in dir_ls:
            org_rawh5f_path = org_house_dir + '/'+ dn
            new_rawh5f_path = self.scans_h5f_dir+'/'+ dn +'/'+house_name
            import shutil
            if os.path.exists(org_rawh5f_path):
                shutil.move( org_rawh5f_path, new_rawh5f_path )
                print('move %s to %s'%(org_rawh5f_path, new_rawh5f_path))
            if os.path.exists( org_house_dir ):
                shutil.rmtree( org_house_dir )

    def Parse_house_regions(self,house_name,MultiProcess=0):
        t0 = time.time()
        house_dir = self.scans_dir+'/%s'%(house_name)
        house_dir_extracted = self.matterport3D_extracted_dir + self.scans_name+'/%s'%(house_name)
        region_segmentations_zip_fn = house_dir+'/region_segmentations.zip'
        rs_zf = zipfile.ZipFile(region_segmentations_zip_fn,'r')

        namelist_ply = [ name for name in rs_zf.namelist()  if 'ply' in name]
        num_region = len(namelist_ply)

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for ply_item_name in namelist_ply:
            if not IsMultiProcess:
                WriteRawH5f_Region_Ply(ply_item_name,rs_zf, house_name, self.scans_h5f_dir, house_dir_extracted)
            else:
                results = pool.apply_async(WriteRawH5f_Region_Ply,(ply_item_name,rs_zf, house_name, self.scans_h5f_dir, house_dir_extracted))
                s = ply_item_name.index('region_segmentations/region')+len('region_segmentations/region')
                e = ply_item_name.index('.ply')
                k_region = int( ply_item_name[ s:e ] )
                print('apply_async %d'%(k_region))
        if IsMultiProcess:
            pool.close()
            pool.join()

            success_fns = []
            success_N = num_region
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"Parse_house_regions failed. only %d files successed"%(len(success_fns))
            print("\n\n Parse_house_regions:all %d files successed\n******************************\n"%(len(success_fns)))

        print('Parse house time: %f'%(time.time()-t0))

    def SortRaw(self,house_names_ls,block_step_xyz,MultiProcess=0):
        t0 = time.time()
        rawh5_file_ls = []
        house_names_ls.sort()
        for house_name in house_names_ls:
            house_rawh5f_dir = self.scans_h5f_dir+'/rawh5f/%s'%(house_name)
            rawh5_file_ls += glob.glob( os.path.join(house_rawh5f_dir,'*.rh5') )
        #rawh5_file_ls = glob.glob(self.house_h5f_dir+'/rawh5f/*.rh5')
        #block_step_xyz = [0.5,0.5,0.5]
        sorted_path = self.scans_h5f_dir + '/'+get_stride_step_name(block_step_xyz,block_step_xyz) + '/' + house_name
        IsShowInfoFinished = True

        IsMultiProcess = MultiProcess>1
        if not IsMultiProcess:
            WriteSortH5f_FromRawH5f(rawh5_file_ls,block_step_xyz,sorted_path,IsShowInfoFinished)
        else:
            pool = mp.Pool(MultiProcess)
            for rawh5f_fn in rawh5_file_ls:
                results = pool.apply_async(WriteSortH5f_FromRawH5f,([rawh5f_fn],block_step_xyz,sorted_path,IsShowInfoFinished))
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

    def MergeSampleNorm(self,base_step_stride,new_stride,new_step,numpoint_block,MultiProcess=0):
        '''
         1 merge to new block step/stride size
             obj_merged: generate obj for merged
         2 randomly sampling to fix point number in each block
             obj_sampled_merged
         3 normalizing sampled block
        '''
        base_stride = base_step = base_step_stride
        base_path = self.house_h5f_dir+'/'+get_stride_step_name(base_stride,base_step)
        new_stride = new_stride
        new_step = new_step
        new_sorted_path = self.house_h5f_dir+'/'+get_stride_step_name(new_stride,new_step)


        base_file_list = glob.glob( os.path.join(base_path,'*.sh5') )
        print('%d sh5 files in %s'%(len(base_file_list),base_path))

        more_actions_config = {}
        more_actions_config['actions'] = []
        more_actions_config['actions'] = ['sample_merged']
        #more_actions_config['actions'] = ['sample_merged','norm_sampled_merged']
        more_actions_config['sample_num'] = numpoint_block

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for fn in base_file_list:
            if not IsMultiProcess:
                MergeSampleNorm_FromSortedH5f( fn,new_stride,new_step,new_sorted_path,more_actions_config )
            else:
                results = pool.apply_async(MergeSampleNorm_FromSortedH5f,( fn,new_stride,new_step,new_sorted_path,more_actions_config ))
        if IsMultiProcess:
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(base_file_list)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"MergeSampleNorm failed. only %d files successed"%(len(success_fns))
            print("\n\n MergeSampleNorm:all %d files successed\n******************************\n"%(len(success_fns)))

    def Sample(self,base_stride,base_step,numpoint_block,MultiProcess=0):
        base_path = self.house_h5f_dir+'/'+get_stride_step_name(base_stride,base_step)

        base_file_list = glob.glob( os.path.join(base_path,'*.sh5') )
        print('%d sh5 files in %s'%(len(base_file_list),base_path))

        IsGenNorm = False

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for fn in base_file_list:
            if not IsMultiProcess:
                SampleFile(fn,numpoint_block,IsGenNorm)
            else:
                results = pool.apply_async( SampleFile,(fn,numpoint_block,IsGenNorm) )
        if IsMultiProcess:
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(base_file_list)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"Sample failed. only %d files successed"%(len(success_fns))
            print("\n\n SampleFile:all %d files successed\n******************************\n"%(len(success_fns)))

    def get_mean_pr_sample_num(self,base_stride,base_step):
        base_sorted_path = self.house_h5f_dir+'/'+get_stride_step_name(base_stride,base_step)
        py_normed_path = base_sorted_path +'_pyramid-'+GlobalSubBaseBLOCK.get_pyramid_flag()
        file_list = glob.glob( os.path.join(py_normed_path,'*.prh5') )
        sum_sg_bidxmap_sample_num = np.zeros((GlobalSubBaseBLOCK.cascade_num,6)).astype(np.uint64)
        sum_flatten_bmap_sample_num = np.zeros((GlobalSubBaseBLOCK.cascade_num,3)).astype(np.uint64)
        for fn in file_list:
            with h5py.File(fn,'r') as f:
                #print('sum_sg_bidxmap_sample_num')
                #print(f['bidxmaps'].attrs['sum_sg_bidxmap_sample_num'].astype(np.uint64))
                sum_sg_bidxmap_sample_num += f['bidxmaps'].attrs['sum_sg_bidxmap_sample_num'].astype(np.uint64)
                sum_flatten_bmap_sample_num += f['bidxmaps'].attrs['sum_flatten_bmap_sample_num'].astype(np.uint64)
        mean_sg_sample_num = get_mean_sg_sample_rate(sum_sg_bidxmap_sample_num)
        mean_flatten_sample_num = get_mean_flatten_sample_rate(sum_flatten_bmap_sample_num)
        print('mean_sg_sample_num:')
        print(mean_sg_sample_num)
        print('\nmean_flatten_sample_num')
        print(mean_flatten_sample_num)

    def GenPyramid(self,house_names_ls, base_stride, base_step, MultiProcess=0):
        file_list = []
        #house_names_ls.sort()
        for house_name in house_names_ls:
            house_sh5f_dir = self.scans_h5f_dir+'/%s/%s'%(get_stride_step_name(base_stride,base_step), house_name)
            file_list += glob.glob( os.path.join(house_sh5f_dir, '*.sh5') )
            if TMPDEBUG:
                file_list = glob.glob( os.path.join(house_sh5f_dir, '*region0.sh5') )

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for n,fn in enumerate(file_list):
            if not IsMultiProcess:
                GenPyramidSortedFlie(fn)
                print('Finish %d / %d Files'%(n,len(file_list)))
            else:
                results = pool.apply_async(GenPyramidSortedFlie,(fn,))
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

    def Norm(self,base_stride,base_step,numpoint_block,MultiProcess=0):
      #  base_stride = [2,2,-1]
      #  base_step = [4,4,-1]
      #  numpoint_block = 8192
        base_sorted_sampled_path = self.house_h5f_dir+'/'+get_stride_step_name(base_stride,base_step)+'_'+str(numpoint_block)
        file_list = glob.glob( os.path.join(base_sorted_sampled_path,'*.rsh5') )
        #file_list.sort()

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for fn in file_list:
            if not IsMultiProcess:
                NormSortedSampledFlie(fn)
            else:
                results = pool.apply_async(NormSortedSampledFlie,(fn,))
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
            print("\n\n Norm:all %d files successed\n******************************\n"%(len(success_fns)))

    def MergeNormed(self, flag, house_name=None, house_group_num=5 ):
        plnh5_folder_name = 'stride_0d1_step_0d1_pl_nh5-1d6_2'
        bxmh5_folder_name = 'stride_0d1_step_0d1_bxmh5-12800_1d6_2_fmn4-480_80_24-80_20_10-0d2_0d6_1d2-0d2_0d6_1d2-3A1'
        nh5_folder_names = [ plnh5_folder_name, bxmh5_folder_name]
        formats = ['.nh5','.bxmh5']
        pl_base_fn_ls = []

        if flag == 'region':
            pl_region_h5f_path = self.scans_h5f_dir + '/' + nh5_folder_names[0] + '/' + house_name
            plfn_ls = glob.glob( pl_region_h5f_path + '/*' +  formats[0] )
            plfn_ls.sort()
            nonvoid_plfn_ls = []
            bxmh5_fn_ls = []
            for pl_fn in plfn_ls:
                is_intact, ck_str = Normed_H5f.check_nh5_intact( pl_fn )
                if not is_intact:
                    print(' ! ! ! Abort merging %s not intact: %s'%(house_name+formats[0], pl_fn))
                    continue
                if ck_str == 'void file':
                    if not SHOWONLYERR: print('void file: %s'%(pl_fn))
                    continue
                region_name = os.path.splitext(os.path.basename( pl_fn ))[0]
                bxmh5_fn = self.scans_h5f_dir + '/' + nh5_folder_names[1] + '/' + house_name + '/' + region_name + formats[1]
                if not os.path.exists( bxmh5_fn ):
                    print(' ! ! ! Abort merging %s not intact: %s'%(house_name+formats[0], bxmh5_fn))
                    return
                with h5py.File( pl_fn, 'r' ) as plh5f, h5py.File( bxmh5_fn, 'r' ) as bxmh5f:
                    if not plh5f['data'].shape[0] == bxmh5f['bidxmaps_flat'].shape[0]:
                        print('Abort merging %s \n  data shape (%d) != bidxmaps_flat shape (%d): %s'%( pl_region_h5f_path, plh5f['data'].shape[0], bxmh5f['bidxmaps_flat'].shape[0], pl_fn) )
                        return
                    else:
                        #print('shape match check ok: %s'%(region_name))
                        pass
                nonvoid_plfn_ls.append( pl_fn )
                bxmh5_fn_ls.append( bxmh5_fn )
            if len( nonvoid_plfn_ls )  == 0:
                print(  "no file, skip %s"%( house_name ) )
                return
            fn_ls = [ nonvoid_plfn_ls, bxmh5_fn_ls ]
            merged_file_names = ['','']

            for j in range(2):
                merged_path = os.path.dirname( self.scans_h5f_dir ) + '/each_house/' + nh5_folder_names[j] + '/'
                merged_file_names[j] = merged_path + house_name+formats[j]
                if not os.path.exists(merged_path):
                    os.makedirs(merged_path)
                MergeNormed_H5f( fn_ls[j], merged_file_names[j], IsShowSummaryFinished=True)
            # check after merged
            with h5py.File( merged_file_names[0], 'r' ) as plh5f, h5py.File( merged_file_names[1], 'r' ) as bxmh5f:
                if not plh5f['data'].shape[0] == bxmh5f['bidxmaps_flat'].shape[0]:
                    print('! ! ! shape check failed:  data shape (%d) != bidxmaps_flat shape (%d): \n\t%s \n\t%s'%( plh5f['data'].shape[0], bxmh5f['bidxmaps_flat'].shape[0], merged_file_names[0],merged_file_names[1]) )
                else:
                    print( 'After merging, shape match check ok: %s'%(os.path.basename( merged_file_names[0] )) )
                    pass

        elif flag == 'house':
            assert house_name == None
            house_h5f_path = os.path.dirname( self.scans_h5f_dir ) + '/each_house/' + nh5_folder_names[0]
            pl_fn_ls = glob.glob( house_h5f_path + '/*' + formats[0] )
            pl_fn_ls.sort()
            for pl_fn in pl_fn_ls:
                is_intact, ck_str = Normed_H5f.check_nh5_intact( pl_fn )
                assert is_intact, "not intact: %s"%(pl_fn)
            print('all %d houses intact'%(len(pl_fn_ls)))

            for k in range( 0, len(pl_fn_ls), house_group_num ):
                end = min( (k+house_group_num),len(pl_fn_ls) )
                pl_fn_ls_k = pl_fn_ls[k : end ]
                bxmh5_fn_ls_k = []
                for pl_fn in pl_fn_ls_k:
                    house_name =  os.path.splitext(os.path.basename( pl_fn ))[0]
                    bxmh5_fn =  os.path.dirname( self.scans_h5f_dir ) + '/each_house/' + nh5_folder_names[1] + '/' + house_name + formats[1]
                    if not os.path.exists(bxmh5_fn):
                        print('Abort merging %d-%d, not exist: %s'%(k, end, bxmh5_fn))
                        return
                    with h5py.File( pl_fn, 'r' ) as plh5f, h5py.File( bxmh5_fn, 'r' ) as bxmh5f:
                        if not plh5f['data'].shape[0] == bxmh5f['bidxmaps_flat'].shape[0]:
                            print('Abort merging %d-%d, data shape (%d) != bidxmaps_flat shape (%d): %s'%( k, end, plh5f['data'].shape[0], bxmh5f['bidxmaps_flat'].shape[0], pl_fn) )
                            return
                        else:
                            print('shape match check ok: %s'%(house_name))
                    bxmh5_fn_ls_k.append( bxmh5_fn )
                fn_ls_k = [ pl_fn_ls_k, bxmh5_fn_ls_k ]

                merged_fns = ['','']
                for j in range(2):
                    house_name_ls_k = [ os.path.splitext(os.path.basename( fn ))[0] for fn in fn_ls_k[j] ]
                    merged_house_name = ''
                    for i,hn in enumerate( house_name_ls_k ):
                        merged_house_name += hn[0:3]
                        if i != len(house_name_ls_k)-1:
                            merged_house_name += '_'
                    if j==1:
                        merged_fn_pl = os.path.dirname( self.scans_h5f_dir ) + '/merged_house/' + nh5_folder_names[0] + '/' + merged_house_name + formats[0]
                        if not os.path.exists( merged_fn_pl ):
                            print('abort because not exist: %s'%(merged_fn_pl))
                            continue
                    merged_path = os.path.dirname( self.scans_h5f_dir ) + '/merged_house/' + nh5_folder_names[j]
                    merged_fns[j] = merged_path + '/'+ merged_house_name + formats[j]
                    if not os.path.exists(merged_path):
                        os.makedirs(merged_path)
                    MergeNormed_H5f( fn_ls_k[j], merged_fns[j], IsShowSummaryFinished=True)
                # check after merged
                with h5py.File( merged_fns[0], 'r' ) as plh5f, h5py.File( merged_fns[1], 'r' ) as bxmh5f:
                    if not plh5f['data'].shape[0] == bxmh5f['bidxmaps_flat'].shape[0]:
                        print('! ! ! shape check failed:  data shape (%d) != bidxmaps_flat shape (%d): \n\t%s \n\t%s'%( plh5f['data'].shape[0], bxmh5f['bidxmaps_flat'].shape[0], merged_fns[0],merged_fns[1]) )
                    else:
                        #print( 'After merging, shape match check ok: %s'%(os.path.basename(fn_ls[0][i])) )
                        pass


    def GenObj_RawH5f(self,house_name):
        house_h5f_dir = self.scans_h5f_dir+'/rawh5'+'/%s'%(house_name)
        file_name = house_h5f_dir+'/region3.rh5'
        xyz_cut_rate= [0,0,0.9]
        xyz_cut_rate= [0,0,0]
        with h5py.File(file_name,'r') as h5f:
            rawh5f = Raw_H5f(h5f,file_name)
            rawh5f.generate_objfile(IsLabelColor=False,xyz_cut_rate=xyz_cut_rate)
            rawh5f.generate_objfile(IsLabelColor=True,xyz_cut_rate=xyz_cut_rate)

    def GenObj_SortedH5f(self):
        stride = step = [0.1,0.1,0.1]
        sorted_path = self.house_h5f_dir+'/'+get_stride_step_name(stride,step)
        file_name = sorted_path + '/region7.sh5'
        with h5py.File(file_name,'r') as h5f:
            sortedh5f = Sorted_H5f(h5f,file_name)
            sortedh5f.gen_file_obj(IsLabelColor=False)


    def GenObj_NormedH5f(self):
        file_name = '/home/z/Research/dynamic_pointnet/data/Matterport3D_H5F/v1/scans/stride_0d1_step_0d1_pl_nh5_0d5_1/17DRP5sb8fy/region0.nh5'
        with h5py.File(file_name,'r') as h5f:
            normedh5f = Normed_H5f(h5f,file_name)
            normedh5f.gen_gt_pred_obj_examples()
            #normedh5f.gen_gt_pred_obj_examples(['void'])

    def ShowSummary(self):
        file_name = self.house_rawh5f_dir+'/region1.rh5'
        step = stride = [0.1,0.1,0.1]
        file_name = self.house_h5f_dir+'/'+get_stride_step_name(step,stride) + '/region2.sh5'
        #file_name = self.house_rawh5f_dir + '/region2.rh5'
        #file_name = self.house_h5f_dir+'/'+get_stride_step_name(step,stride) +'_pyramid-'+GlobalSubBaseBLOCK.get_pyramid_flag() + '/region2.prh5'
        file_name = '/DS/Matterport3D/Matterport3D_H5F/v1/scans/17DRP5sb8fy/stride_0d1_step_0d1_pyramid-1_2-512_256_64-128_12_6-0d2_0d6_1d1/region1.prh5'
        #file_name = '/DS/Matterport3D/Matterport3D_H5F/all_merged_nf5/17DRP5sb8fy_stride_0d1_step_0d1_pyramid-1_2-512_256_64-128_12_6-0d2_0d6_1d1.prh5'
        IsIntact,check_str = check_h5fs_intact(file_name)
        if IsIntact:
            with h5py.File(file_name,'r') as h5f:
                show_h5f_summary_info(h5f)
        else:
            print("file not intact: %s \n\t %s"%(file_name,check_str))

    def ShowBidxmap(self,house_name):
        house_h5f_dir = self.scans_h5f_dir+'/%s'%(house_name)
        house_bmh5_dir = house_h5f_dir+'/stride_0d1_step_0d1-bidxmap-1d6_2-512_256_64-128_12_6-0d2_0d6_1d2'
        bmh5_fn = house_bmh5_dir + '/region0.bmh5'
        gsbb_load = GlobalSubBaseBLOCK(bmh5_fn=bmh5_fn)
        gsbb_load.show_all()


def parse_house(house_names_ls, operations):
    MultiProcess = 0
    matterport3d_prepare = Matterport3D_Prepare()


    if 'ParseRaw' in operations:
        matterport3d_prepare.Parse_houses_regions( house_names_ls,  MultiProcess)

    base_step_stride = [0.1,0.1,0.1]
    if 'SortRaw' in operations:
        matterport3d_prepare.SortRaw(house_names_ls, base_step_stride, MultiProcess)

    if 'GenPyramid' in operations:
        matterport3d_prepare.GenPyramid(house_names_ls, base_step_stride, base_step_stride, MultiProcess)
    if 'pr_sample_rate' in operations:
        matterport3d_prepare.get_mean_pr_sample_num(base_step_stride,base_step_stride)

    new_stride = [1,1,-1]
    new_step = [2,2,-1]
    numpoint_block = 8192

    new_stride=new_step=base_step_stride
    numpoint_block = 8
    if 'MergeSampleNorm' in operations:
        matterport3d_prepare.MergeSampleNorm(base_step_stride,new_stride,new_step,numpoint_block,MultiProcess)
    if 'Sample' in operations:
        new_stride=new_step=base_step_stride
        matterport3d_prepare.Sample(new_stride,new_step,numpoint_block,MultiProcess)
    if 'Norm' in operations:
        matterport3d_prepare.Norm(new_stride,new_step,numpoint_block,MultiProcess)
    if 'MergeNormed_region' in operations:
        for house_name in house_names_ls:
            matterport3d_prepare.MergeNormed('region', house_name=house_name)
    if 'MergeNormed_house' in operations:
        matterport3d_prepare.MergeNormed('house')
    if 'GenObj_RawH5f' in operations:
        for house_name in house_names_ls:
            matterport3d_prepare.GenObj_RawH5f(house_name)
    if 'GenObj_SortedH5f' in operations:
        matterport3d_prepare.GenObj_SortedH5f()
    if 'GenObj_NormedH5f' in operations:
        matterport3d_prepare.GenObj_NormedH5f()

def parse_house_ls():
    house_names = ['17DRP5sb8fy']
    #house_names = ['17DRP5sb8fy','1pXnuDYAj8r','2azQ1b91cZZ','2t7WUuJeko7']
    #house_names += ['5q7pvUzZiYa', '759xd9YjKW5','8194nk5LbLH','8WUmhLawc2A','ac26ZMwG7aT','B6ByNegPMKs']

    #scans_name_abs = Matterport3D_Prepare.matterport3D_h5f_dir + '/v1/scans/rawh5f'
    #all_house_names = os.listdir(scans_name_abs)
    #house_names = all_house_names
    house_names.sort()

    operations = ['ParseRaw','SortRaw','GenPyramid','MergeSampleNorm','Sample','Norm','MergeNormed']
    operations  = ['ParseRaw']
    #operations  = ['GenObj_RawH5f']
    #operations  = ['pr_sample_rate']


    group_n = 5
    for i in range(0,len(house_names),group_n):
        #if i>50: continue
        house_names_i = house_names[i:min(i+group_n,len(house_names))]
        print('\nstart parsing houses %s  %d-%d/%d\n'%(house_names_i,i,i+len(house_names_i),len(house_names)))
        parse_house(house_names_i, operations)
        if operations==['MergeNormed_house']: break

def main_parse_house():
    house_names_ls = ['17DRP5sb8fy']
    #house_names_ls = ['8194nk5LbLH']
    MultiProcess = 0
    matterport3d_prepare = Matterport3D_Prepare()
    matterport3d_prepare.Parse_houses_regions( house_names_ls,  MultiProcess)

def main_GenObj_RawH5f():
    house_name = '17DRP5sb8fy'
    matterport3d_prepare = Matterport3D_Prepare()
    matterport3d_prepare.GenObj_RawH5f(house_name)

def show_summary():
    matterport3d_prepare = Matterport3D_Prepare()
    matterport3d_prepare.ShowSummary()


if __name__ == '__main__':
  main_parse_house()
  #main_GenObj_RawH5f()
  #show_summary()

