#xyz Dec 2017
from __future__ import print_function
import pdb, traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from datasets.block_data_prep_util import Raw_H5f, check_h5fs_intact
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
    noneed_extract = False
    if os.path.exists(file_path):
      try:
        with open(file_path, 'r') as ply_fo:
          plydata = PlyData.read(ply_fo)
        noneed_extract = True
      except:
        print('\nfile extracted but not intact: %s\n'%(file_path))

    if not noneed_extract:
        print('extracting %s...'%(file_path))
        file_path_extracted  = zipf.extract(ply_item_name,house_dir_extracted)
        print('file extracting finished: %s'%(file_path_extracted) )
        assert file_path == file_path_extracted
    else:
        print('file already extracted: %s'%(file_path))
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
        self.scans_h5f_dir = self.matterport3D_h5f_dir


    def Parse_houses_regions(self,house_names_ls,MultiProcess=0):
        for house_name in house_names_ls:
            self.Parse_house_regions(house_name,MultiProcess)


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



    def GenObj_RawH5f(self,house_name):
        house_h5f_dir = self.scans_h5f_dir+'/rawh5'+'/%s'%(house_name)
        file_name = house_h5f_dir+'/region0.rh5'
        xyz_cut_rate= [0,0,0.9]
        xyz_cut_rate= [0,0,0]
        with h5py.File(file_name,'r') as h5f:
            rawh5f = Raw_H5f(h5f,file_name)
            rawh5f.generate_objfile(IsLabelColor=False,xyz_cut_rate=xyz_cut_rate)
            rawh5f.generate_objfile(IsLabelColor=True,xyz_cut_rate=xyz_cut_rate)


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



def main_parse_house():
    house_names_ls = os.listdir('/DS/Matterport3D/Matterport3D_WHOLE/v1/scans')
    #house_names_ls = ['s8pcmisQ38h', 'Vt2qJdWjCF2', '8WUmhLawc2A', 'dhjEzFoUFzH',\
    #                  'b8cTxDM8gDG', 'ARNzJeq3xxb',  'EDJbREhghzL',\
    #                  'VLzqgDo317F', 'r47D5H71a5s', 'TbHJrupSAjP', 'gxdoqLR6rwA',\
    #                  'jtcxE69GiFV', 'cV4RVeZvu5T', 'VVfe2KiqLaN', '7y3sRwLe3Va',\
    #                  'ac26ZMwG7aT', '5ZKStnWn8Zo', 'S9hNv5qa7GM', 'Vvot9Ly1tCj',\
    #                  'sKLMLpTHeUy', 'VFuaQ6m2Qom', 'uNb9QFRL6hY', 'q9vSo1VnCiC',\
    #                  'e9zR4mvMWw7', '8194nk5LbLH', '759xd9YjKW5', '2t7WUuJeko7',\
    #                  '1pXnuDYAj8r', '82sE5b5pLXE', 'PX4nDJXEHrG', 'oLBMNvg9in8', 'ULsKaCPVFJR', 'Uxmj2M2itWa', 'mJXqzFtmKg4', 'JF19kD82Mey', 'vyrNrziPKCB', 'aayBHfsNo7d', 'D7G3Y4RVNrH', 'gYvKGZ5eRqb',\
    #                  'jh4fc5c5qoQ', 'pa4otMbVnkk', '17DRP5sb8fy', 'qoiz87JEwZ2', 'ur6pFq6Qu1A', 'pLe4wQe7qrG', 'fzynW3qQPVF', '5q7pvUzZiYa', 'E9uDoFAP3SH', 'p5wJjkQkbXX', '1LXtFkjw3qL', 'rPc6DW4iMge',\
    #                  'UwV83HsGsw3', 'V2XKFyX4ASd', 'r1Q1Z4BcV1o', 'gTV8FGcVJC9', 'RPmz2sHmrrY', 'kEZ7cmS4wCh', 'EU6Fwq7SyZv', '29hnd4uzFmX', 'rqfALeAoiTq', 'PuKPg4mmafe', 'JmbYfDe2QKZ', 'SN83YJsR3w2',\
    #                  'B6ByNegPMKs', 'i5noydFURQK', '2azQ1b91cZZ', 'GdvgFV5R1Z5', 'QUCTc6BB5sX', 'gZ6f7yhEvPG', 'JeFG25nYj2p', '2n8kARJN3HM', 'Pm6F8kyY3z2', 'pRbA3pwrgk9', 'sT4fr6TAbpF', '5LpN3gDmAk7',\
    #                  'HxpKQynjfin', 'D7N2EKCX4Sj']
    house_names_ls.sort()
    #house_names_ls = ['17DRP5sb8fy']
    #house_names_ls = ['7y3sRwLe3Va']
    MultiProcess = 0
    matterport3d_prepare = Matterport3D_Prepare()
    matterport3d_prepare.Parse_houses_regions( house_names_ls,  MultiProcess)

def main_GenObj_RawH5f():
    house_name = '17DRP5sb8fy'
    #house_name = 'EDJbREhghzL'
    matterport3d_prepare = Matterport3D_Prepare()
    matterport3d_prepare.GenObj_RawH5f(house_name)

def show_summary():
    matterport3d_prepare = Matterport3D_Prepare()
    matterport3d_prepare.ShowSummary()


if __name__ == '__main__':
  main_parse_house()
  #main_GenObj_RawH5f()
  #show_summary()

