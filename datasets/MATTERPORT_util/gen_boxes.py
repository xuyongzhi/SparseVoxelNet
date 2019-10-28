#xyz Oct 2019
from __future__ import print_function
import pdb, traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import numpy as np
import glob
import time
import multiprocessing as mp
import itertools
import zipfile,gzip
from plyfile import PlyData, PlyElement
#import open3d
from MATTERPORT_util import MATTERPORT_Meta
label2color =  MATTERPORT_Meta['label2color']

Material_Index = 0
Instance_Index = 1
Category_Index = 2

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

    # do not collect vertex semantic
    visualize(vertex_xyz, face_vertex_indices, face_semantic)
    return vertex_xyz,vertex_nxnynz,vertex_rgb,face_vertex_indices,face_semantic

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

def visualize(vertex_xyz, face_vertex_indices, face_semantic):
    # visualize by instance
    instance_label = face_semantic[:,Instance_Index]
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    pass

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

def read_ply():
    region_ply_fn = '/DS/Matterport3D/Matterport3D_WHOLE_extracted/v1/scans/17DRP5sb8fy/17DRP5sb8fy/region_segmentations/region0.ply'
    with open(region_ply_fn,'r') as ply_fo:
            vertex_xyz, vertex_nxnynz, vertex_rgb, vertex_semantic, face_vertex_indices, face_semantic = \
                            parse_ply_file(ply_fo)


if __name__ == '__main__':
  read_ply()



